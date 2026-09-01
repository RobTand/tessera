"""Price a whole rate band from ONE encode per Linear.

The completion axis is embedded: `build_forest` guarantees "truncating
completion bits lands on an ancestor", so a single deep encode at rung R can be
read back at every level c <= cap-R, yielding the band [R/arity, cap/arity] bpp
without a second Viterbi.  Measured 2026-09-01: the truncation itself costs
1.03x-1.28x, while the *rung* choice costs up to 2.15x at equal bpp -- so this
is a candidate generator, not a shipping path.  The selected point is re-encoded
natively; that is the house spine (surrogates generate, real measurement
selects) applied to the encoder.

The bias is measured, never assumed.  Truncation makes the probe systematically
OVERSTATE cost at the shallow end, and unevenly across levels.  Every run
re-measures that penalty on a sample of its own units, on this model's real
shapes, and emits it beside the estimates.  A surrogate whose bias is recorded
is usable; one whose bias is asserted from another tensor is not.

Emits per-unit SSE per rate point.  It does NOT emit a cost: weighting SSE by
h_trace is the cost model's job, and weight-space error alone is a known-
misleading axis here (it inverted once NVFP4 was priced W4A4 as it serves).
"""
import argparse, json, os, time
import torch
from safetensors import safe_open
from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid, completion_capacity
from tessera.encode import encode_unit
from tessera.decode import reconstruct_unit
from tessera.manifest import RotationState
from tessera.trellis import ConvCode


def sse(a, b):
    return float(((a.float() - b.float()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--rung", type=int, default=4,
                    help="body rate per code; covers [rung/arity, cap/arity] bpp")
    ap.add_argument("--calibrate-every", type=int, default=32,
                    help="every Nth unit also gets native encodes, to measure the bias")
    ap.add_argument("--result", required=True)
    a = ap.parse_args()

    grid = tuple_grid(E2M1_GRID, 2)
    cap, arity = grid.rate_cap, grid.arity
    depth = completion_capacity(a.rung, cap)
    forests = {a.rung: build_forest(a.rung, grid=grid)}
    cc = ConvCode(memory=6)

    index = json.load(open(f"{a.source}/model.safetensors.index.json"))["weight_map"]
    # Shards are 1-based, as the source names them.  Resolve against the index
    # rather than reconstructing the filename: a shard that silently matches no
    # tensors would produce a valid, empty, wrong receipt.
    shard_file = f"model-{a.shard:05d}-of-{len(set(index.values())):05d}.safetensors"
    names = sorted(n for n, s in index.items() if s == shard_file)
    if not names:
        raise SystemExit(f"shard {a.shard} -> {shard_file} matches no tensor in the index")

    units, calib, t0 = {}, {}, time.time()
    with safe_open(f"{a.source}/{shard_file}", framework="pt") as f:
        for i, name in enumerate(names):
            w = f.get_tensor(name)
            if w.ndim != 2 or w.shape[0] % arity:
                continue                      # not a Linear this grid can span
            w = w.cuda()
            rates = (a.rung,) * w.shape[1]
            deep = encode_unit(w, forests, rates, code=cc,
                               rotation=RotationState.NONE, completion=depth)
            units[name] = {
                "params": int(w.numel()),
                "shape": list(w.shape),
                "sse": {f"{(a.rung + c) / arity:.2f}":
                        sse(w, reconstruct_unit(deep, forests, code=cc, completion=c))
                        for c in range(depth + 1)},
            }
            if i % a.calibrate_every == 0:
                native = {}
                for c in range(depth + 1):
                    nat = encode_unit(w, forests, rates, code=cc,
                                      rotation=RotationState.NONE, completion=c)
                    native[f"{(a.rung + c) / arity:.2f}"] = sse(
                        w, reconstruct_unit(nat, forests, code=cc))
                calib[name] = native
            del w

    # The bias, as this run measured it on its own weights.
    penalty = {}
    for name, native in calib.items():
        for bpp, s in native.items():
            penalty.setdefault(bpp, []).append(
                (units[name]["sse"][bpp] / s) ** 0.5 if s > 0 else 1.0)
    out = {
        "shard": a.shard, "shard_file": shard_file, "grid": grid.name,
        "rung": a.rung, "arity": arity, "rate_cap": cap,
        "band_bpp": [a.rung / arity, cap / arity],
        "units": units,
        "truncation_penalty_rms": {b: round(sum(v) / len(v), 4)
                                   for b, v in sorted(penalty.items())},
        "calibrated_units": len(calib),
    }
    os.makedirs(os.path.dirname(a.result), exist_ok=True)
    json.dump(out, open(a.result, "w"))
    # Host-dependent facts go to stdout, never into the receipt.
    print(f"shard {a.shard}: {len(units)} units, {len(calib)} calibrated, "
          f"{time.time() - t0:.1f}s")
    print("penalty:", out["truncation_penalty_rms"])


if __name__ == "__main__":
    main()
