#!/usr/bin/env python3
"""The box-side instrument for a latency run: GPU power, load and swap I/O over
an explicit UTC window, read from that box's own Netdata and written to a
receipt.

WHY THIS EXISTS AS A TOOL RATHER THAN A DASHBOARD READING.  Principle 15 wants
two instruments and this is the one no in-process profiler can supply: whether
the box was actually loaded.  On the 2026-09-03 campaign that reading was taken
by hand, out of a chat client, and pasted into a document -- which is how #5
ended up reporting a *bracket* instead of a number, because nobody had recorded
what the box was doing in the seven minutes before its process existed.  A
number a script wrote, with the query beside it, is re-runnable; a number a
person pasted is not.

WHY POWER AND NOT ``gpu_utilization``.  On GB10 ``gpu_utilization`` means "a
kernel is resident", not "the SMs are working": it reads ~96% for a stalled
kernel and a saturated one alike, and ``utilization.memory`` returns a hard 0.
Power against the ~140 W envelope is the reading that separates them, and the
envelope fraction is also the only estimate of remaining headroom -- wall clock
cannot give one.  ``gpu_utilization`` is collected here anyway, and reported,
precisely so a reader can see it saying nothing.

WHY SWAP *I/O* AND NOT SWAP IN USE.  ``mem.swapio`` is pages moving; ``mem.swap``
is pages resident.  A box that swapped hard hours ago still shows GiB in use with
nothing moving, and that residue is not contention -- while a box with a modest
resident figure and pages streaming is thrashing.  #83's armB was the second
case.  Both are recorded; neither is dropped.

usage:
  box_power_window.py --label idle-before --window -1800:0 --out x.json
  box_power_window.py --host gx10-6b77 --label sparklina-idle --window -1800:0
  box_power_window.py --label armA-decode --window 2026-09-04T06:10:00Z:2026-09-04T06:12:30Z
"""
from __future__ import annotations

import argparse
import calendar
import json
import statistics
import time
import urllib.parse
import urllib.request

#: The GB10 board envelope the repo reads power against.  Not a measured
#: maximum: the standing figure this project's principle 15 clause names, kept
#: as one constant so every fraction in every receipt divides by the same thing.
ENVELOPE_W = 140.0

#: context -> the dimensions worth carrying.  ``gpu_utilization`` is here to be
#: seen being useless, not to be believed.
SERIES = (
    ("nvidia_smi.gpu_power_draw", ("power_draw",)),
    ("nvidia_smi.gpu_utilization", ("gpu",)),
    ("system.load", ("load1",)),
    ("system.cpu", ("user", "system", "iowait")),
    ("mem.swapio", ("in", "out")),
    ("mem.available", ("avail",)),
)


def _parse_instant(text: str, now: float) -> float:
    """``-1800`` (seconds before now), ``0`` (now), or an ISO-8601 UTC stamp."""
    text = text.strip()
    try:
        return now + float(text)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return float(calendar.timegm(time.strptime(text, fmt)))
        except ValueError:
            continue
    raise ValueError(f"cannot read {text!r} as a UTC instant or an offset in seconds")


def split_window(text: str, now: float) -> tuple[float, float]:
    """``AFTER:BEFORE`` -> the two instants, when AFTER itself contains colons.

    ``--window`` accepts an ISO-8601 UTC stamp on either side, and an ISO-8601
    stamp contains two colons of its own.  Splitting on the FIRST colon --
    which is what this did -- turns
    ``2026-09-04T09:16:12Z:2026-09-04T09:17:26Z`` into ``2026-09-04T09`` and
    refuses the tool's own documented usage line.  Splitting on the last one
    fails the same way from the other end.

    So the separator is found rather than assumed: every colon is tried as the
    split, and the one where BOTH halves parse is the separator.  On
    ``-1800:0`` there is one colon and one answer; on a pair of stamps only the
    middle colon leaves two readable halves, because ``2026-09-04T09`` and
    ``26Z`` are not instants.  An ambiguous string is refused rather than
    guessed at -- an A/B whose box-side window is silently the wrong minute is
    worse than one that stops.

    THIS WAS NOT A COSMETIC BUG.  ``window_gemv_latency_ab.sh`` calls this tool
    with each arm's own ``decode_start:prefill_end`` marks and runs it under
    ``subprocess.run(..., check=False)``, so the ValueError went nowhere: the
    2026-09-04 campaign wrote four latency receipts and four traces and **no**
    ``power-arm*.json`` at all.  Principle 15 wants two instruments; the box-
    side one was absent from every timed window in the run, and nothing said so.
    """

    positions = [i for i, ch in enumerate(text) if ch == ":"]
    if not positions:
        raise SystemExit(
            f"--window {text!r} has no ':' -- it is AFTER:BEFORE, e.g. -1800:0 "
            f"or 2026-09-04T06:10:00Z:2026-09-04T06:12:30Z")
    found = []
    for i in positions:
        try:
            found.append((_parse_instant(text[:i], now), _parse_instant(text[i + 1:], now)))
        except ValueError:
            continue
    if len(found) == 1:
        return found[0]
    if not found:
        raise SystemExit(
            f"--window {text!r} is not AFTER:BEFORE with readable instants; each "
            f"side is an offset in seconds (-1800, 0) or an ISO-8601 UTC stamp "
            f"(2026-09-04T06:10:00Z)")
    raise SystemExit(
        f"--window {text!r} splits into a readable pair {len(found)} ways; "
        f"write both sides as ISO-8601 UTC stamps so the separator is unambiguous")


def _fetch(host: str, context: str, dims, after: int, before: int, points: int) -> dict:
    q = urllib.parse.urlencode({
        "contexts": context, "after": after, "before": before, "points": points,
        "group_by": "dimension", "format": "json2",
        "time_group": "average", "dimensions": "|".join(dims),
    })
    url = f"http://{host}:19999/api/v2/data?{q}"
    with urllib.request.urlopen(url, timeout=60) as fh:
        return {"url": url, "doc": json.loads(fh.read().decode())}


def _stats(doc: dict, dims) -> dict:
    """Per-dimension min/median/mean/max over the returned points.

    An empty window is reported as ``None`` and a ``points`` count of 0, never
    as a zero: "the GPU drew 0 W" and "Netdata has no samples here" are
    different facts, and only one of them is evidence.
    """
    labels = doc["result"]["labels"]
    rows = doc["result"]["data"]
    out = {}
    for d in dims:
        if d not in labels:
            out[d] = {"points": 0, "note": "dimension not present in this context"}
            continue
        i = labels.index(d)
        vals = [r[i][0] for r in rows if r[i][0] is not None]
        if not vals:
            out[d] = {"points": 0, "note": "no samples in this window"}
            continue
        out[d] = {"points": len(vals),
                  "min": round(min(vals), 3), "max": round(max(vals), 3),
                  "mean": round(statistics.fmean(vals), 3),
                  "median": round(statistics.median(vals), 3)}
    return out


def collect(host: str, after: int, before: int, points: int) -> dict:
    series = {}
    for context, dims in SERIES:
        try:
            got = _fetch(host, context, dims, after, before, points)
        except Exception as exc:  # noqa: BLE001 -- a missing series is recorded, not fatal
            series[context] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        doc = got["doc"]
        entry = {"query": got["url"], "stats": _stats(doc, dims),
                 "update_every_s": doc.get("view", {}).get("update_every")}
        if context == "nvidia_smi.gpu_power_draw":
            labels = doc["result"]["labels"]
            i = labels.index("power_draw") if "power_draw" in labels else None
            entry["samples"] = ([[int(r[0]), r[i][0]] for r in doc["result"]["data"]]
                                if i is not None else [])
        series[context] = entry
    return series


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1",
                    help="the box whose Netdata to read (127.0.0.1, sparky, gx10-6b77)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--window", required=True,
                    help="AFTER:BEFORE, each an ISO-8601 UTC stamp or an offset "
                         "in seconds from now (e.g. -1800:0)")
    ap.add_argument("--points", type=int, default=0,
                    help="points to return; 0 asks for one per 10 s of window")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    now = time.time()
    after_f, before_f = split_window(args.window, now)
    after, before = int(after_f), int(before_f)
    if before <= after:
        raise SystemExit(f"window BEFORE ({before}) must be after AFTER ({after})")
    points = args.points or max(4, min(600, (before - after) // 10))

    receipt = {
        "schema": "tessera.box_power_window/1",
        "label": args.label, "host": args.host,
        "window_utc": [time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(after)),
                       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(before))],
        "window_unix": [after, before],
        "window_s": before - after,
        "points_requested": points,
        "envelope_w": ENVELOPE_W,
        "taken_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "series": collect(args.host, after, before, points),
    }
    power = receipt["series"].get("nvidia_smi.gpu_power_draw", {}).get("stats", {}) \
                             .get("power_draw", {})
    if power.get("points"):
        receipt["gpu_power_w"] = {k: power[k] for k in ("min", "median", "mean", "max")}
        receipt["envelope_fraction"] = {
            k: round(power[k] / ENVELOPE_W, 4) for k in ("median", "max")}
    else:
        receipt["gpu_power_w"] = None

    print(f"== {args.label} on {args.host}  {receipt['window_utc'][0]} .. "
          f"{receipt['window_utc'][1]}  ({receipt['window_s']} s)")
    for context, entry in receipt["series"].items():
        if "error" in entry:
            print(f"   {context:34s} ERROR {entry['error']}")
            continue
        for dim, st in entry["stats"].items():
            if not st.get("points"):
                print(f"   {context}.{dim:12s} -- {st.get('note')}")
                continue
            print(f"   {context}.{dim:12s} min {st['min']:>9.2f}  med {st['median']:>9.2f}"
                  f"  mean {st['mean']:>9.2f}  max {st['max']:>9.2f}  (n={st['points']})")
    if receipt["gpu_power_w"]:
        f = receipt["envelope_fraction"]
        print(f"   -> GPU power median {receipt['gpu_power_w']['median']} W "
              f"= {100*f['median']:.1f}% of the {ENVELOPE_W:.0f} W envelope; "
              f"peak {100*f['max']:.1f}%")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(receipt, fh, indent=1, sort_keys=True)
        print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
