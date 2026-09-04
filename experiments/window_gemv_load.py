#!/usr/bin/env python3
"""Drive a served Tessera arm and read the latency the ENGINE measured.

Two loads, because the window GEMV serves one of the two regimes and the
materialised path serves the other:

* **decode**: a short prompt and many output tokens at concurrency 1, so
  almost every forward is ``M = 1`` -- the shape ``fp8_gemv.streamed_apply``
  routes to the kernel.
* **prefill**: a long prompt and one output token, so the cost is the
  many-row forward the GEMV refuses by name and the materialised path serves.

The numbers reported are ``vllm:time_to_first_token_seconds`` and
``vllm:request_time_per_output_token_seconds`` -- vLLM's own histograms, differenced
across each load, so they describe the requests this script drove and nothing
else.  Client-side wall clock is recorded beside them as a cross-check, never
as the headline: a client number folds in HTTP and this box's scheduler.

The torch profile is taken over a SECOND, shorter decode load, because a
profiled forward is not a timed forward -- profiling a run and quoting its
latency would be quoting the profiler.  The trace answers a different
question: which kernels launched.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request


def _utc() -> str:
    """One spelling of "now, in UTC", so every mark in a receipt is the same
    clock the Netdata window will be cut on."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


#: Whole-second marks are the right resolution for a Netdata window (its finest
#: tier is 10 s) and the WRONG one for cutting a chrome trace: the profiled
#: sub-loads run for a few seconds, so a one-second rounding is a large
#: fraction of the window being cut.  Both are recorded; the trace cut reads
#: this one.
_MARKS_UNIX: dict = {}


def _mark(name: str, marks: dict) -> str:
    """Stamp ``name`` in both clocks and return the UTC string."""
    _MARKS_UNIX[name] = time.time()
    marks[name] = _utc()
    return marks[name]


def load_now() -> dict:
    """The host's run-queue load, right now.

    Recorded at both ends of every timed window because a latency number taken
    on a contended box is noise, and a contended number that SAYS SO is still
    useful while one that does not is worse than nothing.  This box runs many
    agents' jobs at once and the GPU lock does not serialise the CPU-bound ones,
    so "I held the lock" is not evidence the box was quiet.  ``nproc`` travels
    with the reading so load is interpretable as a ratio rather than a bare
    number.
    """
    one, five, fifteen = os.getloadavg()
    out = {"load1": one, "load5": five, "load15": fifteen,
           "ncpu": os.cpu_count(),
           "load1_per_cpu": round(one / (os.cpu_count() or 1), 3)}
    out.update(_meminfo())
    out.update(_swap_io())
    return out


def _meminfo() -> dict:
    """Available memory and swap in use, beside the run queue.

    LOAD AVERAGE IS NOT A SUFFICIENT CONTENTION LABEL ON THIS BOX, and the
    #83 campaign proved it the expensive way: the armB/eager arm peaked at
    load1/cpu 2.49 and ran EIGHT TIMES slower than the armA/eager arm, which
    peaked HIGHER at 3.413.  The difference was not the run queue at all --
    armB ran while 13 of 15 GB of swap were in use and the box was thrashing.
    A reader handed only the load numbers would have concluded the slower arm
    was the less contended one and read an 8x memory-pressure artifact as a
    lane result.  GB10 shares one physical pool between GPU and host, so a
    serve is exposed to host memory pressure directly.
    """
    out: dict = {}
    try:
        info = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                info[k] = float(v.split()[0]) / (1024 * 1024)  # kB -> GiB
        out["mem_available_gib"] = round(info.get("MemAvailable", 0.0), 1)
        out["mem_total_gib"] = round(info.get("MemTotal", 0.0), 1)
        swap_total = info.get("SwapTotal", 0.0)
        out["swap_used_gib"] = round(swap_total - info.get("SwapFree", 0.0), 1)
        out["swap_total_gib"] = round(swap_total, 1)
    except OSError:
        pass
    return out


#: ``/proc/vmstat`` counts swap traffic in PAGES, and the page size is a
#: property of the kernel this runs on rather than a constant worth assuming --
#: GB10 boxes are configured with a 64 KiB page in some images and 4 KiB in
#: others, and a 16x error in a contention figure is the kind that gets read as
#: "the box was fine".
_PAGE_BYTES = os.sysconf("SC_PAGE_SIZE")


def _swap_io() -> dict:
    """Cumulative pages swapped in and out, since boot.

    SWAP IN USE IS NOT SWAP ACTIVITY, and the difference decides whether a
    receipt's contention label is describing this run or last night's.  A box
    that thrashed hours ago still reports GiB resident in swap with nothing
    moving; a box with a modest resident figure and pages streaming is
    thrashing right now.  ``_meminfo``'s ``swap_used_gib`` reads the first and
    is what the ``contended`` verdict is built on -- deliberately conservative,
    and NOT relaxed here.  These counters are the second reading, differenced
    across each timed window by ``_swap_delta``, so a receipt can say "2 GiB
    resident, nothing moved" or "2 GiB resident and 400 MiB paged during the
    window" and a reader can tell those apart instead of guessing which one
    the label meant.
    """
    out: dict = {}
    try:
        with open("/proc/vmstat") as fh:
            for line in fh:
                k, _, v = line.partition(" ")
                if k in ("pswpin", "pswpout"):
                    out[k] = int(v)
    except OSError:
        pass
    return out


def _swap_delta(start: dict, end: dict) -> dict:
    """Pages (and MiB) swapped between two ``load_now()`` readings."""
    out: dict = {}
    for key, name in (("pswpin", "swap_in"), ("pswpout", "swap_out")):
        if key in start and key in end:
            pages = max(0, end[key] - start[key])
            out[name + "_pages"] = pages
            out[name + "_mib"] = round(pages * _PAGE_BYTES / (1024 * 1024), 2)
    if out:
        out["moved"] = bool(out.get("swap_in_pages", 0) or out.get("swap_out_pages", 0))
    return out


#: vLLM 0.28 publishes per-output-token latency as
#: ``vllm:request_time_per_output_token_seconds``; the name this script first
#: read, ``vllm:time_per_output_token_seconds``, does not exist in this release
#: and returned None for every window.  Recorded here as a named constant so the
#: stem is stated once.  ``vllm:inter_token_latency_seconds`` is read beside it:
#: same mean, but counted per token rather than per request, so the two together
#: say whether a difference is in the per-request or the per-token accounting.
TPOT_STEM = "vllm:request_time_per_output_token_seconds"

_HIST = re.compile(r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})? (?P<value>\S+)$')


def _get(url: str, timeout: float = 600.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        return fh.read().decode()


def _post(url: str, payload: dict | None, timeout: float = 600.0) -> dict | None:
    data = b"" if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        body = fh.read().decode()
    return json.loads(body) if body.strip() else None


def scrape(url: str) -> dict:
    """``metric name -> value`` for the ``_sum``/``_count`` series we read."""
    out: dict[str, float] = {}
    for line in _get(url + "/metrics").splitlines():
        if line.startswith("#"):
            continue
        m = _HIST.match(line.strip())
        if not m:
            continue
        name = m.group("name")
        if name.endswith("_sum") or name.endswith("_count"):
            try:
                out[name] = out.get(name, 0.0) + float(m.group("value"))
            except ValueError:
                pass
    return out


def moved(before: dict, after: dict) -> dict:
    """Every vLLM series that CHANGED over the window, with its delta.

    Insurance, and cheap: if a histogram is renamed between vLLM releases the
    two stems read below silently return ``None`` and the receipt would say
    nothing about a serve that cost a lock slot to take.  Recording what
    actually moved makes the number recoverable from the receipt instead of
    from a second serve.
    """
    out = {}
    for k, v in after.items():
        if not k.startswith("vllm:"):
            continue
        d = v - before.get(k, 0.0)
        if d:
            out[k] = d
    return out


def delta(before: dict, after: dict, stem: str):
    """Mean of one histogram over the window, or None when it did not move."""
    s = after.get(stem + "_sum", 0.0) - before.get(stem + "_sum", 0.0)
    n = after.get(stem + "_count", 0.0) - before.get(stem + "_count", 0.0)
    if n <= 0:
        return None
    return {"mean_s": s / n, "sum_s": s, "count": n}


def completion(prompt_tokens: int, max_tokens: int, req: int) -> dict:
    """One request's body.  Token ids, so "512 tokens" is 512 tokens and not a
    tokenizer's opinion of a string.

    EVERY REQUEST GETS ITS OWN PROMPT, and that is load-bearing rather than
    tidy.  The serve runs with vLLM's default ``enable_prefix_caching=True``,
    so a repeated prompt is a prefix-cache HIT: the many-row forward never
    runs and a "prefill" window would time a cache lookup instead of the
    materialised path this A/B exists to compare.  Perturbing the ids per
    request is the fix that keeps the serve's configuration identical to the
    KL serve's -- turning prefix caching off would make the latency arm a
    different serve from the one whose KL is reported.
    """
    ids = [(i * 7919 + 104729 * (req + 1)) % 100000 + 1000 for i in range(prompt_tokens)]
    return {"model": "kl-target", "prompt": ids, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": False}


#: Monotonic across the whole run, so no two requests anywhere in it -- warmup,
#: timed windows, profiled load -- share a prompt.
_REQ = 0


def drive(url: str, n: int, prompt_tokens: int, max_tokens: int) -> dict:
    global _REQ
    t0 = time.time()
    for _ in range(n):
        _REQ += 1
        _post(url + "/v1/completions", completion(prompt_tokens, max_tokens, _REQ))
    wall = time.time() - t0
    return {"requests": n, "prompt_tokens": prompt_tokens, "max_tokens": max_tokens,
            "wall_s": wall, "wall_s_per_request": wall / n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--regime", required=True)
    ap.add_argument("--decode-requests", type=int, default=12)
    ap.add_argument("--decode-out-tokens", type=int, default=128)
    ap.add_argument("--prefill-requests", type=int, default=12)
    ap.add_argument("--prefill-prompt-tokens", type=int, default=512)
    ap.add_argument("--warmup-requests", type=int, default=4)
    # The PROFILED sub-loads, which are not the timed ones.  Longer than the
    # 2x24 the first campaign used: the trace is cut to these marks in both
    # arms, and a window of a second or two is not many multiples of the
    # clock's resolution nor many steady-state decode steps.
    ap.add_argument("--profile-decode-requests", type=int, default=3)
    ap.add_argument("--profile-decode-out-tokens", type=int, default=64)
    ap.add_argument("--profile-prefill-requests", type=int, default=2)
    args = ap.parse_args()

    started = _utc()
    # Warm up: the first forwards of a compiled serve pay capture, and the
    # first of any serve pays allocator growth.  Neither belongs in the number.
    drive(args.url, args.warmup_requests, 32, 32)
    drive(args.url, 2, args.prefill_prompt_tokens, 1)

    windows = {}
    marks = {}
    _mark("decode_start", marks)
    loads = {"decode_start": load_now()}
    b = scrape(args.url)
    windows["decode"] = drive(args.url, args.decode_requests, 32, args.decode_out_tokens)
    a = scrape(args.url)
    windows["decode"]["ttft"] = delta(b, a, "vllm:time_to_first_token_seconds")
    windows["decode"]["tpot"] = delta(b, a, TPOT_STEM)
    windows["decode"]["itl"] = delta(b, a, "vllm:inter_token_latency_seconds")
    windows["decode"]["series_moved"] = moved(b, a)
    _mark("decode_end", marks)
    loads["decode_end"] = load_now()

    marks["prefill_start"] = marks["decode_end"]
    _MARKS_UNIX["prefill_start"] = _MARKS_UNIX["decode_end"]
    loads["prefill_start"] = load_now()
    b = scrape(args.url)
    windows["prefill"] = drive(args.url, args.prefill_requests,
                               args.prefill_prompt_tokens, 1)
    a = scrape(args.url)
    windows["prefill"]["ttft"] = delta(b, a, "vllm:time_to_first_token_seconds")
    windows["prefill"]["tpot"] = delta(b, a, TPOT_STEM)
    windows["prefill"]["itl"] = delta(b, a, "vllm:inter_token_latency_seconds")
    windows["prefill"]["series_moved"] = moved(b, a)
    _mark("prefill_end", marks)
    loads["prefill_end"] = load_now()

    # The profile, over its own short load.  A profiled forward is not a timed
    # forward, so nothing above is taken from this window.
    #
    # THE PROFILED SUB-LOADS ARE MARKED, and that is what lets two arms'
    # traces be cut the same way.  ``window_gemv_trace_summary.py --phases``
    # identifies a decode bin by the presence of a window-GEMV launch, which
    # works only in the arm that HAS the lane: the fallback arm has no such
    # marker, so bins identified that way would be identified by two different
    # rules in the two arms and the comparison would not be one.  A trace's
    # absolute clock is ``baseTimeNanoseconds + ts``, so these marks cut both
    # arms by the same wall-clock rule.
    profiled = None
    try:
        _post(args.url + "/start_profile", None)
        _mark("profile_decode_start", marks)
        prof_decode = drive(args.url, args.profile_decode_requests, 32,
                            args.profile_decode_out_tokens)
        _mark("profile_decode_end", marks)
        marks["profile_prefill_start"] = marks["profile_decode_end"]
        _MARKS_UNIX["profile_prefill_start"] = _MARKS_UNIX["profile_decode_end"]
        prof_prefill = drive(args.url, args.profile_prefill_requests,
                             args.prefill_prompt_tokens, 1)
        _mark("profile_prefill_end", marks)
        profiled = {"decode": prof_decode, "prefill": prof_prefill}
        _post(args.url + "/stop_profile", None, timeout=900.0)
        # The trace is flushed asynchronously; give the writer a moment.
        time.sleep(20)
    except Exception as exc:  # noqa: BLE001 -- a missing profile is reported, not fatal
        profiled = {"error": f"{type(exc).__name__}: {exc}"}

    receipt = {
        # /2 adds ``marks_unix``, ``swap_io`` and the profiled sub-load marks.
        # A reader of a /1 receipt (the 2026-09-03 campaign's four) must not
        # assume those fields: those runs have no swap-activity reading and
        # their traces cannot be cut by wall clock.
        "schema": "tessera.window_gemv.served_latency/2",
        "arm": args.arm, "serve_mode": args.mode, "forward": args.regime,
        "started_utc": started,
        "finished_utc": _utc(),
        "marks_utc": marks,
        "marks_unix": dict(_MARKS_UNIX),
        "host_load": loads,
        # A verdict the receipt carries itself, so a reader does not have to
        # decide from four raw numbers whether the box was quiet.  The threshold
        # is one runnable process per core: above that the run queue is
        # oversubscribed and a host-driven latency number is contended by
        # definition.  It is a LABEL, never a filter -- the numbers are reported
        # either way.
        # Contended if EITHER the run queue is oversubscribed or the box is
        # swapping.  Both legs are needed: see _meminfo -- the arm that was 8x
        # slower had the LOWER load average, and only the swap reading caught it.
        "contended": (max(v["load1_per_cpu"] for v in loads.values()) > 1.0
                      or max(v.get("swap_used_gib", 0.0) for v in loads.values()) > 1.0),
        "max_load1_per_cpu": max(v["load1_per_cpu"] for v in loads.values()),
        "max_swap_used_gib": max(v.get("swap_used_gib", 0.0) for v in loads.values()),
        "min_mem_available_gib": min(v.get("mem_available_gib", 0.0)
                                     for v in loads.values()),
        # Pages actually swapped during each timed window, beside the resident
        # figure the ``contended`` label reads.  See ``_swap_io``: the label
        # stays conservative and these say whether it is describing this run.
        "swap_io": {
            "decode": _swap_delta(loads["decode_start"], loads["decode_end"]),
            "prefill": _swap_delta(loads["prefill_start"], loads["prefill_end"]),
            "page_bytes": _PAGE_BYTES,
        },
        "windows": windows,
        "profiled_load": profiled,
    }
    with open(args.out, "w") as fh:
        json.dump(receipt, fh, indent=1, sort_keys=True)
    for name, w in windows.items():
        ttft = w.get("ttft") or {}
        tpot = w.get("tpot") or {}
        itl = w.get("itl") or {}
        print(f"{name}: wall {w['wall_s_per_request']*1000:.2f} ms/req  "
              f"TTFT {1000*ttft.get('mean_s', float('nan')):.2f} ms  "
              f"TPOT {1000*tpot.get('mean_s', float('nan')):.3f} ms "
              f"(n={tpot.get('count')})  "
              f"ITL {1000*itl.get('mean_s', float('nan')):.3f} ms "
              f"(n={itl.get('count')})")
    lo = max(v["load1_per_cpu"] for v in loads.values())
    sw = max(v.get("swap_used_gib", 0.0) for v in loads.values())
    av = min(v.get("mem_available_gib", 0.0) for v in loads.values())
    quiet = lo <= 1.0 and sw <= 1.0
    print(f"host load1/cpu peaked at {lo:.2f}, swap in use peaked at {sw:.1f} GiB, "
          f"available memory bottomed at {av:.1f} GiB "
          f"({'box was quiet' if quiet else 'CONTENDED -- report these as contended'})")
    for name in ("decode", "prefill"):
        io = receipt["swap_io"][name]
        if io:
            print(f"  {name} window: swapped in {io.get('swap_in_mib', 0)} MiB, "
                  f"out {io.get('swap_out_mib', 0)} MiB "
                  f"({'PAGES MOVED' if io.get('moved') else 'nothing moved'})")
    print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
