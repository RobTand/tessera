"""The best-form step's tile, screened, so its default becomes a measurement.

``_tile_best`` sizes a program by lanes rather than by ``FAN``-wide output --
about 256 class-column elements -- but its rule caps ``bl`` at 128 and then
tests ``bl * bc < 256``, so ``bc`` can only ever reach 2.  Both production
shapes therefore land on exactly ``128x2`` at 4 warps.  That point was chosen
against the column counts ``_layout`` admitted BEFORE the class-width resident
set let it admit ``FAN`` times as many, which is the shape the candidate
actually runs in now.  Whether 128x2 is still the right point in the wider
column shape is a measurement nobody has taken.

This takes it, and it is a TUNING screen: the incumbent tile is an arm, not
the baseline, and the front form is carried only to keep the reported ratios
on the same axis as the A/B they extend.

What it does not move, on purpose:

  outer chunk       512, the production chunk.  ``chunk`` is the OUTER loop:
                    moving it moves the epilogue's ``min``, its ``sse``
                    accumulation and its traceback call count as well, which
                    is three changes in a screen that names one.
  width             each arm runs the width its resident set earns -- 192 at
                    R3, 64 at R4.  The tile is what varies; the layout is not.
  scan unroll       left derived from ``(fan, bl, bc, warps)``, so each tile
                    is screened as the code would configure it rather than
                    under another tile's unroll.

Every tile writes identical bytes.  ``_step_best`` masks both axes
(``li < low``, ``ci < m``), so an oversized tile wastes lanes and changes
nothing else, and the identity gate runs before any clock is read: a screen
that cannot show identical states and an identical ``sse`` for every
configuration has measured something other than this tile.

Two modes, never in one run:

  time        the arms interleaved in ABCD blocks, each block long enough for
              a 1 Hz sampler, energy integrated over each block's own bracket.
  pbprofile   one call per arm under one in-process torch profiler, exactly
              one config per action so no trace overwrites another's path.

Run::

    python experiments/window_viterbi_best_tile_screen.py --mode time \\
        --configs R3 --out .../tile-r3.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera import window_viterbi as wv  # noqa: E402
from tessera.encode import viterbi_window  # noqa: E402
from window_viterbi_best_form_ab import (CONFIGS, ENVELOPE_W, Power,  # noqa: E402
                                         _inputs, _Spy)

#: ``(name, BL, BC, WARPS)``.  ``front`` is the reference spelling and ignores
#: the tile; ``t128x2w4`` is the incumbent ``_tile_best`` returns for both
#: production shapes, carried as an arm so the screen can say whether the
#: default was already right rather than only what beats it.
#:
#: The list is deliberately short and explicit.  A wide autotune over this
#: axis would take longer than the encode it is meant to speed up, and would
#: pick per box: what is wanted is one defensible point, or the finding that
#: the incumbent is it.
ARMS = (
    ("front", None),
    ("t128x2w4", (128, 2, 4)),        # the incumbent
    ("t128x4w4", (128, 4, 4)),
    ("t128x4w8", (128, 4, 8)),
    ("t128x8w4", (128, 8, 4)),
    ("t256x2w4", (256, 2, 4)),
    ("t256x4w8", (256, 4, 8)),
    ("t64x4w2", (64, 4, 2)),
    ("t64x8w4", (64, 8, 4)),
)


def _select(tile):
    """One environment write per arm; the plan cache is keyed on both."""
    os.environ[wv._BEST_FORM_ENV] = "0" if tile is None else "1"
    os.environ[wv._GRAPH_ENV] = "1"
    if tile is None:
        os.environ.pop(wv._BEST_TILE_ENV, None)
    else:
        os.environ[wv._BEST_TILE_ENV] = ",".join(str(x) for x in tile)


def _call(tile, targets, vectors, window_bits, rate, weights):
    _select(tile)
    return viterbi_window(targets, vectors, window_bits, rate,
                          weights=weights, impl="fused")


def _registers(tile, targets, vectors, window_bits, rate, weights):
    """Regs off the launched kernel, per tile.

    A tile that wins by spilling is a tile that will lose on the next shape,
    so the spill count travels with every number here.
    """
    ks = list(wv._kernels())
    slot = 0 if tile is None else 5                  # _step / _step_best
    spy = _Spy(ks[slot])
    held = list(ks)
    held[slot] = spy
    saved = wv._CACHE.get("k")
    wv._CACHE["k"] = tuple(held)
    wv.window_plan_cache_clear()
    try:
        _call(tile, targets, vectors, window_bits, rate, weights)
    finally:
        wv._CACHE["k"] = saved
        wv.window_plan_cache_clear()
    if not spy.compiled:
        return None
    ck = spy.compiled[-1]
    return dict(n_regs=getattr(ck, "n_regs", None),
                n_spills=getattr(ck, "n_spills", None),
                shared=getattr(getattr(ck, "metadata", None), "shared", None))


def _grid(tile, window_bits, rate, cols, dev):
    """What the arm actually launches: tile, grid and the width it runs at."""
    import triton

    size = 1 << window_bits
    resident = size if tile is None else (size >> rate)
    _, width, descs = wv._layout(dev, size, cols, 512, resident)
    low = size >> rate
    if tile is None:
        bl, bc, warps = wv._tile(1 << rate, low, width)
    else:
        bl, bc, warps = tile
    scan = wv._resolve_scan_unroll(1 << rate, bl, bc, warps)
    return dict(width=width, batches=len(descs), low=low,
                bl=bl, bc=bc, warps=warps, scan_unroll=scan,
                grid=[triton.cdiv(low, bl), triton.cdiv(width, bc)],
                programs=triton.cdiv(low, bl) * triton.cdiv(width, bc),
                elements_per_program=bl * bc,
                elements_per_thread=round(bl * bc / (32.0 * warps), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("time", "pbprofile"), default="time")
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--min-block-s", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="restrict to these arm names; the profile phase runs "
                         "front, the incumbent and the winner, not all nine")
    a = ap.parse_args()

    arms = [x for x in ARMS if not a.arms or x[0] in a.arms]
    if len(arms) < 2:
        raise SystemExit("a screen needs at least two arms")
    dev = "cuda"
    records = []
    want = [c for c in CONFIGS if not a.configs or c[0] in a.configs]
    for name, L, R, arity, rows, cols in want:
        targets, vectors, weights = _inputs(L, R, arity, rows, cols, dev)
        rec = dict(config=name, window_bits=L, rate=R, arity=arity, rows=rows,
                   cols=cols, weighted=True, chunk=512,
                   arms=[x[0] for x in arms],
                   launch={arm: _grid(tile, L, R, cols, dev)
                           for arm, tile in arms})

        # -- identity, before any clock -------------------------------------
        wv.window_plan_cache_clear()
        ref_states, ref_sse = viterbi_window(targets, vectors, L, R,
                                             weights=weights, impl="reference")
        rec["identity"] = {}
        for arm, tile in arms:
            s, e = _call(tile, targets, vectors, L, R, weights)
            rec["identity"][arm] = dict(
                states_equal=bool(torch.equal(s, ref_states)),
                sse=e.hex(), sse_equal=bool(e == ref_sse))
            del s
        del ref_states
        torch.cuda.empty_cache()
        if not all(v["states_equal"] and v["sse_equal"]
                   for v in rec["identity"].values()):
            rec["verdict"] = "a tile does not return the reference's answer"
            records.append(rec)
            print(json.dumps(rec), flush=True)
            continue

        if a.mode == "pbprofile":
            from torch.profiler import ProfilerActivity, profile, record_function
            out = os.environ.get("PRISMABUILD_PROFILE_TORCH_OUT")
            if not out:
                raise SystemExit(
                    "pbprofile mode needs PRISMABUILD_PROFILE_TORCH_OUT; run "
                    "this under pbrun --profile torch")
            if len(want) != 1:
                raise SystemExit(
                    "pbprofile takes exactly one --configs entry: the fleet "
                    f"names one path and {len(want)} configs would overwrite "
                    "each other in it")
            for arm, tile in arms:                        # capture, untraced
                _call(tile, targets, vectors, L, R, weights)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU,
                                     ProfilerActivity.CUDA]) as prof:
                for arm, tile in arms:
                    # Every candidate arm runs _step_best, so the kernel name
                    # cannot separate them; the marker does.
                    with record_function(f"arm:{arm}"):
                        _call(tile, targets, vectors, L, R, weights)
                        torch.cuda.synchronize()
            keep = Path(a.out).with_name(f"{name}.tile.chrome-trace.json.gz")
            keep.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(keep))
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(keep, out)
            rec["pb_profile"] = dict(
                trace=out, retained=str(keep), exists=Path(out).exists(),
                bytes=keep.stat().st_size,
                sha256=hashlib.sha256(keep.read_bytes()).hexdigest())
            rec["kernels"] = {
                ev.key[:56]: dict(us=round(ev.self_device_time_total, 1),
                                  calls=ev.count)
                for ev in sorted(prof.key_averages(),
                                 key=lambda e: -e.self_device_time_total)[:12]
                if ev.self_device_time_total > 0}
            records.append(rec)
            print(json.dumps(rec), flush=True)
            continue

        rec["registers"] = {arm: _registers(tile, targets, vectors, L, R, weights)
                            for arm, tile in arms}

        # -- timing ---------------------------------------------------------
        # One clear, then every arm's plan is built and NOTHING clears again:
        # clearing per repeat would time a graph capture and call it a step.
        wv.window_plan_cache_clear()
        single = {}
        for arm, tile in arms:
            _call(tile, targets, vectors, L, R, weights)      # capture
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _call(tile, targets, vectors, L, R, weights)
            torch.cuda.synchronize()
            single[arm] = time.perf_counter() - t0
        inner = {arm: max(1, int(a.min_block_s / single[arm]) + 1)
                 for arm, _ in arms}
        rec["inner_repeats"] = inner
        rec["single_call_s"] = {k: round(v, 5) for k, v in single.items()}

        power = Power().start()
        time.sleep(2.0)                 # lead in, so block 1 is bracketed
        blocks = {arm: [] for arm, _ in arms}
        try:
            for _ in range(a.blocks):
                for arm, tile in arms:
                    torch.cuda.synchronize()
                    w0, t0 = time.time(), time.perf_counter()
                    for _ in range(inner[arm]):
                        _call(tile, targets, vectors, L, R, weights)
                    torch.cuda.synchronize()
                    dt, w1 = time.perf_counter() - t0, time.time()
                    blocks[arm].append(dict(seconds=dt, wall_start=w0,
                                            wall_end=w1, calls=inner[arm]))
        finally:
            time.sleep(2.0)             # lead out, so the last block is too
            power.stop()

        steps = rows // arity
        work_per_call = steps * cols * (1 << L)
        rec["work_per_call"] = work_per_call
        rec["timing"] = {}
        for arm, _ in arms:
            bs = blocks[arm]
            for b in bs:
                j, meta = power.energy(b["wall_start"], b["wall_end"])
                b["joules"] = round(j, 2) if j else None
                b["power"] = meta
                b["seconds_per_call"] = round(b["seconds"] / b["calls"], 5)
            secs = [b["seconds"] for b in bs]
            paid = [b for b in bs if b["joules"]]
            total_j = sum(b["joules"] for b in paid) if paid else None
            total_work = work_per_call * sum(b["calls"] for b in paid)
            energy_seconds = sum(b["seconds"] for b in paid)
            rec["timing"][arm] = dict(
                blocks=[{k: (round(v, 5) if isinstance(v, float) else v)
                         for k, v in b.items()} for b in bs],
                seconds_per_call_min=round(min(secs) / bs[0]["calls"], 5),
                seconds_per_call_median=round(statistics.median(secs) / bs[0]["calls"], 5),
                total_calls=sum(b["calls"] for b in bs),
                total_seconds=round(sum(secs), 4),
                energy_blocks=len(paid), energy_seconds=round(energy_seconds, 4),
                total_joules=round(total_j, 1) if total_j else None,
                power_w_mean=(round(total_j / energy_seconds, 2)
                              if total_j else None),
                power_envelope_frac=(round(total_j / energy_seconds / ENVELOPE_W, 3)
                                     if total_j else None),
                work_per_joule=(round(total_work / total_j, 1) if total_j else None))

        # Two axes, because they answer different questions: against the front
        # form is the number the A/B reports, and against the incumbent tile is
        # the only number that says whether re-tiling was worth anything.
        base = rec["timing"].get("front")
        inc = rec["timing"].get("t128x2w4")
        rec["vs"] = {}
        for arm, _ in arms:
            t = rec["timing"][arm]
            row = {}
            for label, ref in (("front", base), ("incumbent", inc)):
                if ref is None or arm == label:
                    continue
                row[f"speedup_vs_{label}"] = round(
                    ref["seconds_per_call_median"] / t["seconds_per_call_median"], 4)
                if t["work_per_joule"] and ref["work_per_joule"]:
                    row[f"work_per_joule_vs_{label}"] = round(
                        t["work_per_joule"] / ref["work_per_joule"], 4)
            rec["vs"][arm] = row
        records.append(rec)
        del targets, vectors, weights
        torch.cuda.empty_cache()
        print(json.dumps(rec), flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
