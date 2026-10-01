"""Batched units x best-form window Viterbi on the census's own unit path: identity, then time.

The T-8 routed census encodes one expert projection at a time, and under LDLQ
each window-Viterbi call is one ``ldl_block`` (32) columns wide.  Two levers
were measured before and each one alone does nothing for that path:

* batching units (``export.encode_linears_planes``, tessera#385) joins the calls
  along the column axis, but the front-form ``_layout`` holds a column's two
  ``2^L`` fronts in the L2 budget (L2/6, 4 MiB on GB10), which at L=14 is 32
  columns, so a joined call is cut back into 32-column batches: 1.01x
  (``docs/measurements/tessera385-batched-encoder-after-profile-2026-09-13.md``);
* the best-form step (``TESSERA_WINDOW_BEST_FORM=1``) carries ``2^(L-R)`` class
  minima instead of the front, so the same budget holds ``2^R`` times as many
  columns, but at a 32-column call it has none to hold: 0.387x wall
  (``docs/measurements/window-best-form-r7-2026-09-19.md``, best@w32).

Together the joined call is wide and the best-form layout keeps it wide
(``_layout`` caps at ``chunk`` = 512 columns = 16 units x 32).  Both levers are
machine knobs that return identical states (their tests pin it), so the first
thing measured here is that every unit's blob is byte-identical to today's
per-unit front-form blob, on real GLM-5.3 expert bytes with the producer
authority's Hessians; only then are the legs timed.

Legs (each encodes the same units, ``verify`` on, parse + pack per unit as the
exporter does):
  seq_front   today's path: one unit per call, front form
  bat_front   ``--batch`` units per call, front form          (the 1.01x control)
  seq_best    one unit per call, best form                    (the width-held control)
  bat_best    ``--batch`` units per call, best form            (the candidate)

Each leg records its UTC epoch window (for the Netdata power series), its wall
time and NVML power at 10 Hz.  Optional: a counted pass (Viterbi calls, widths,
the synced seconds inside the fused Viterbi) and a CUDA-only torch.profiler
capture of one seq_front unit and one bat_best batch, aggregated from the
kineto events directly (no FunctionEvent table, no chrome trace above a bound):
kernels by device time, device-busy fraction, idle gaps between device ops, and
the host runtime calls that synchronise.

usage (inside the image; ab_batched_best_form.sh runs it there):
  python3 experiments/t8_census/ab_batched_best_form.py OUT --hessian H.json \\
      --producer-authority A.py --rungs 1280,1408 [--batch 16] [--batches 3]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.export import (ActivationSource, encode_linear_planes,        # noqa: E402
                            encode_linears_planes, served_recipe)
from tessera.export_serving import grid_for, load_producer_authority      # noqa: E402
from tessera.fused import pack_fused                                      # noqa: E402
from tessera.structure import STRUCTURE_ROUTED_MOE                        # noqa: E402
from tessera.unit_artifact import parse_unit_artifact                     # noqa: E402

from profile_unit_encode import SRC, Power, read_tensor                   # noqa: E402

BEST_ENV = "TESSERA_WINDOW_BEST_FORM"


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def set_best(on: bool):
    if on:
        os.environ[BEST_ENV] = "1"
    else:
        os.environ.pop(BEST_ENV, None)


def finish(name, exported):
    """The exporter's per-unit tail: parse the manifest off the bytes, pack the container."""
    parse_unit_artifact(exported.blob, device="cuda")
    blob = pack_fused([(name.split(".")[-2], exported.rows, exported.blob)])
    torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
    return hashlib.sha256(bytes(exported.blob)).hexdigest(), hashlib.sha256(blob).hexdigest()


def encode_seq(names, sources, grid, q256, recipe, activation):
    out = {}
    for n in names:
        weight = sources[n].to("cuda", torch.float32).contiguous()
        extra = activation.for_unit(n, weight.shape[1], "cuda", scale_plane=recipe.scale_plane)
        exported, _unit, _f = encode_linear_planes(
            weight, grid=grid, q256=q256, body=recipe.body, name=n, verify=True, **extra)
        extra.clear()
        out[n] = finish(n, exported)
    return out


def encode_bat(names, sources, grid, q256, recipe, activation, batch):
    out = {}
    for i in range(0, len(names), batch):
        group = names[i:i + batch]
        weights = [sources[n].to("cuda", torch.float32).contiguous() for n in group]
        per_unit = [activation.for_unit(n, w.shape[1], "cuda", scale_plane=recipe.scale_plane)
                    for n, w in zip(group, weights)]
        results = encode_linears_planes(
            weights, grid=grid, q256=q256, names=list(group), per_unit=per_unit,
            body=recipe.body, verify=True)
        per_unit.clear()
        for n, (exported, _unit, _f) in zip(group, results):
            out[n] = finish(n, exported)
        del weights, results
    return out


def run_leg(leg, names, sources, grid, q256, recipe, activation, batch):
    set_best(leg.endswith("_best"))
    torch.cuda.synchronize()
    power = Power()
    start = utc()
    power.start()
    t0 = time.perf_counter()
    if leg.startswith("seq_"):
        blobs = encode_seq(names, sources, grid, q256, recipe, activation)
    else:
        blobs = encode_bat(names, sources, grid, q256, recipe, activation, batch)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    pw = power.stop()
    end = utc()
    set_best(False)
    return {"leg": leg, "units": len(names), "start_utc": start, "end_utc": end,
            "wall_s": round(wall, 3), "s_per_unit": round(wall / len(names), 4),
            "power": pw}, blobs


class Count:
    """Viterbi calls by (columns, rate), the synced seconds inside the fused call, and the layouts."""

    def __init__(self):
        from tessera import encode as enc
        from tessera import window_viterbi as wv
        self.enc, self.wv = enc, wv
        self.calls: dict[str, int] = {}
        self.fused_s = 0.0

    def __enter__(self):
        enc, wv = self.enc, self.wv
        self._real = (enc.viterbi_window, wv.viterbi_window_fused)
        real_window, real_fused = self._real

        def window(targets, vectors, window_bits, rate, *a, **kw):
            key = f"cols={targets.shape[1]} rows={targets.shape[0]} L={window_bits} R={rate}"
            self.calls[key] = self.calls.get(key, 0) + 1
            return real_window(targets, vectors, window_bits, rate, *a, **kw)

        def fused(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = real_fused(*a, **kw)
            torch.cuda.synchronize()
            self.fused_s += time.perf_counter() - t0
            return r

        enc.viterbi_window, wv.viterbi_window_fused = window, fused
        return self

    def __exit__(self, *exc):
        self.enc.viterbi_window, self.wv.viterbi_window_fused = self._real
        return False

    def record(self):
        """The calls, plus the column layout each spelling gives each call shape.

        Plans are cached after the warm, so ``_layout`` is evaluated here for
        the observed shapes rather than intercepted (the same function the
        plan calls, at chunk 512, the ``viterbi_window`` default)."""
        lay = {}
        for key in self.calls:
            f = dict(kv.split("=") for kv in key.split())
            cols, L, R = int(f["cols"]), int(f["L"]), int(f["R"])
            dev = torch.device("cuda", torch.cuda.current_device())
            front = self.wv._layout(dev, 1 << L, cols, 512)
            best = self.wv._layout(dev, 1 << L, cols, 512, 1 << (L - R))
            lay[key] = {"front_width": front[1], "front_batches": len(front[2]),
                        "best_width": best[1], "best_batches": len(best[2])}
        return {"calls": self.calls, "fused_synced_s": round(self.fused_s, 3), "layouts": lay}


def cuda_profile(fn, out: Path, label: str, trace_bound: int = 200_000):
    """One CUDA-only kineto capture of ``fn()``, aggregated from the raw events."""
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    events = prof.profiler.kineto_results.events()

    def ns(e, what):
        f = getattr(e, f"{what}_ns", None)
        if f is not None:
            return int(f())
        return int(getattr(e, f"{what}_us")() * 1000)

    dev, host = {}, {}
    intervals = []
    for e in events:
        name = e.name()
        dur = ns(e, "duration")
        if "CUDA" in str(e.device_type()):
            s = ns(e, "start")
            intervals.append((s, s + dur))
            d = dev.setdefault(name, [0, 0])
            d[0] += 1
            d[1] += dur
        else:
            h = host.setdefault(name, [0, 0])
            h[0] += 1
            h[1] += dur
    intervals.sort()
    busy = 0
    gaps = []
    cur_s, cur_e = None, None
    for s, e in intervals:
        if cur_e is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            busy += cur_e - cur_s
            gaps.append(s - cur_e)
            cur_s, cur_e = s, e
    if cur_e is not None:
        busy += cur_e - cur_s
    span = (intervals[-1][1] - intervals[0][0]) if intervals else 0
    gaps.sort()
    dev_total = sum(v[1] for v in dev.values())
    rec = {
        "label": label, "wall_s": round(wall, 3), "device_ops": len(intervals),
        "device_span_s": round(span / 1e9, 4), "device_busy_s": round(busy / 1e9, 4),
        "device_busy_frac_of_wall": round(busy / 1e9 / wall, 4) if wall else None,
        "device_busy_frac_of_span": round(busy / span, 4) if span else None,
        "gaps": {"count": len(gaps), "sum_s": round(sum(gaps) / 1e9, 4),
                 "p50_us": round(gaps[len(gaps) // 2] / 1e3, 3) if gaps else None,
                 "p90_us": round(gaps[int(len(gaps) * 0.9)] / 1e3, 3) if gaps else None,
                 "p99_us": round(gaps[int(len(gaps) * 0.99)] / 1e3, 3) if gaps else None},
        "kernels_by_device_time": [
            {"name": k[:160], "calls": v[0], "device_ms": round(v[1] / 1e6, 3),
             "mean_us": round(v[1] / v[0] / 1e3, 3),
             "frac_of_device": round(v[1] / dev_total, 4) if dev_total else None}
            for k, v in sorted(dev.items(), key=lambda kv: -kv[1][1])[:25]],
        "host_runtime_calls": [
            {"name": k[:120], "calls": v[0], "ms": round(v[1] / 1e6, 3)}
            for k, v in sorted(host.items(), key=lambda kv: -kv[1][1])[:25]],
    }
    if len(events) <= trace_bound:
        trace = out / f"trace-{label}.json"
        prof.export_chrome_trace(str(trace))
        rec["trace"] = str(trace)
    else:
        rec["trace"] = f"not written: {len(events)} events > bound {trace_bound}"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--grid", default="E4M3")
    ap.add_argument("--rungs", default="1280,1408", help="q256 rungs; the first gets the full ABBA")
    ap.add_argument("--hessian", type=Path, required=True)
    ap.add_argument("--producer-authority", type=Path, required=True)
    ap.add_argument("--projection", default="gate_proj")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--batches", type=int, default=3, help="timed batches per leg")
    ap.add_argument("--first-expert", type=int, default=112)
    ap.add_argument("--control-units", type=int, default=16,
                    help="units for the two controls (seq_best, bat_front)")
    ap.add_argument("--no-profile", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("a GPU measurement with no GPU")
    args.out.mkdir(parents=True, exist_ok=True)

    from tessera.export import DEFAULT_LDLQ_BLOCK, DEFAULT_LDLQ_SIGMA
    _authority, canonical_capture = load_producer_authority(args.producer_authority)
    activation = ActivationSource.from_capture(
        args.hessian, canonical_capture=canonical_capture, ldlq_sigma=DEFAULT_LDLQ_SIGMA,
        ldlq_block=DEFAULT_LDLQ_BLOCK, refit_reach_floor=False)
    grid = grid_for(args.grid)
    rungs = [int(r) for r in args.rungs.split(",")]

    b = args.batch
    n_timed = b * args.batches
    experts = list(range(args.first_expert, args.first_expert + n_timed + b))
    names = [f"model.language_model.layers.{args.layer}.mlp.experts.{e}.{args.projection}.weight"
             for e in experts]
    warm, timed = names[:b], names[b:]
    t = time.perf_counter()
    sources = {n: read_tensor(args.src, n) for n in names}
    rec = {
        "schema": "tessera.t8_census.ab_batched_best_form.v1",
        "tree": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                               cwd=ROOT).stdout.strip() or None,
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
        "host": os.uname().nodename, "src": str(args.src), "layer": args.layer,
        "projection": args.projection, "shape": list(sources[timed[0]].shape),
        "grid": args.grid, "batch": b, "timed_units": n_timed, "hessian": str(args.hessian),
        "ldlq_block": DEFAULT_LDLQ_BLOCK, "source_read_s": round(time.perf_counter() - t, 3),
        "rungs": {},
    }
    outp = args.out / "ab_batched_best_form.json"

    def save():
        outp.write_text(json.dumps(rec, indent=1))

    for ri, q in enumerate(rungs):
        recipe = served_recipe(grid, q, STRUCTURE_ROUTED_MOE)
        r = rec["rungs"][str(q)] = {
            "recipe": {"body": recipe.body.name, "plane": recipe.scale_plane.name,
                       "window_bits": recipe.window_bits, "span": recipe.span},
            "warm": {}, "legs": [], "identity": {}}
        # Warm every configuration: each width compiles and captures its own plan.
        for leg in ("seq_front", "bat_front", "seq_best", "bat_best"):
            units = warm[:1] if leg.startswith("seq_") else warm
            t0 = time.perf_counter()
            run_leg(leg, units, sources, grid, q, recipe, activation, b)
            r["warm"][leg] = round(time.perf_counter() - t0, 3)
        save()
        # ABBA on the claim pair (today's path, the candidate); the two controls
        # run once, in the middle, on one batch's worth of units.
        if ri == 0:
            order = ["seq_front", "bat_best", "bat_front", "seq_best", "bat_best", "seq_front"]
        else:
            order = ["seq_front", "bat_best", "bat_best", "seq_front"]
        reference = None
        for leg in order:
            units = timed[:args.control_units] if leg in ("seq_best", "bat_front") else timed
            info, blobs = run_leg(leg, units, sources, grid, q, recipe, activation, b)
            if reference is None:
                if leg != "seq_front":
                    raise SystemExit("the first timed leg must be today's path")
                reference = blobs
                r["reference_blobs"] = {n.split(".experts.")[1]: v[0] for n, v in blobs.items()}
            diff = [n for n in blobs if blobs[n] != reference[n]]
            info["identical_units"] = len(blobs) - len(diff)
            info["differing_units"] = [n.split(".experts.")[1] for n in diff]
            r["legs"].append(info)
            print(f"[{q}] {leg:>9}: {info['s_per_unit']:.3f} s/unit over {info['units']}, "
                  f"{info['power'].get('mean_w')} W, identical {info['identical_units']}/{info['units']}",
                  flush=True)
            save()
        by = {}
        for info in r["legs"]:
            by.setdefault(info["leg"], []).append(info)
        r["summary"] = {}
        base = sum(i["s_per_unit"] for i in by["seq_front"]) / len(by["seq_front"])
        for leg, infos in by.items():
            spu = sum(i["s_per_unit"] for i in infos) / len(infos)
            w = [i["power"].get("mean_w") for i in infos if i["power"].get("mean_w")]
            mw = sum(w) / len(w) if w else None
            r["summary"][leg] = {
                "s_per_unit": round(spu, 4), "speedup_vs_seq_front": round(base / spu, 3),
                "mean_w": round(mw, 2) if mw else None,
                "units_per_kJ": round(1000.0 / (spu * mw), 3) if mw else None,
                "all_identical": all(not i["differing_units"] for i in infos)}
        r["identity"] = {"all_legs_identical": all(v["all_identical"] for v in r["summary"].values())}
        print(json.dumps({q: r["summary"]}, indent=1), flush=True)
        save()

        if ri == 0:
            # Counted: one unit of today's path, one batch of the candidate.
            with Count() as c:
                set_best(False)
                encode_seq(timed[:1], sources, grid, q, recipe, activation)
            r["counted_seq_front"] = c.record()
            with Count() as c:
                set_best(True)
                encode_bat(timed[:b], sources, grid, q, recipe, activation, b)
                set_best(False)
            r["counted_bat_best"] = c.record()
            save()
            if not args.no_profile:
                set_best(False)
                r["profile_seq_front_unit"] = cuda_profile(
                    lambda: encode_seq(timed[:1], sources, grid, q, recipe, activation),
                    args.out, f"seq_front-q{q}")
                save()
                set_best(True)
                r["profile_bat_best_batch"] = cuda_profile(
                    lambda: encode_bat(timed[:b], sources, grid, q, recipe, activation, b),
                    args.out, f"bat_best-q{q}")
                set_best(False)
                save()
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
