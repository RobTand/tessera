"""Is a two-run window rung's error the chord of its one-run neighbours? (tessera#750)

A Tessera-8 rung between two whole rates mixes the two column rates over the
unit's columns, and the exporter places the upper-rate columns by
``grammar.bresenham_rate_schedule``: evenly by column index, not by importance.
The window code codes each column as its own stream.  So if the columns are
exchangeable, a unit's squared error is linear in its count of upper-rate
columns: the chord between the two whole-rate rungs.  A fractional rung then
buys no accuracy per byte that an allocator mixing whole rates ACROSS units
does not buy at least as well, because the allocator also chooses WHICH units
go up.

This encodes real GLM-5.3-Flash routed experts at whole and fractional rungs
with the exporter's own encoder (``export.encode_linear_planes``, the recipe
default) and decodes them the way the E4M3 route does (``decode.materialize_fp8``).
Per (layer, expert, projection, rung) it records the weight MSE and, for
gate/up, the output MSE on the layer's recorded expert inputs.  Then, per pair,
the chord residual of every interior rung, and the error of a byte-matched
across-unit mix of the two whole rates (units raised in order of their gain).

    rung_chord.py --out DIR [--layers 5,20,42] [--experts 0,1] [--rungs ...]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# torch and tessera are imported where they are used, so --reanalyse runs
# on a host without torch.

SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830/act"
RUNGS = [768, 800, 832, 864, 896, 928, 960, 992, 1024, 1088, 1152, 1216, 1280]


def encode(w: "torch.Tensor", q256: int, name: str, grid: str = "e4m3") -> "torch.Tensor":
    """Return canonical effective weights without a per-weight BF16 conversion."""
    import torch
    from tessera.alphabet import BF16_GRID, E4M3_GRID
    from tessera.decode import materialize_bf16, materialize_fp8
    from tessera.export import encode_linear_planes

    payload = BF16_GRID if grid == "bf16" else E4M3_GRID
    _exported, unit, forests = encode_linear_planes(w, grid=payload, q256=q256, name=name,
                                                    verify=True)
    if grid == "bf16":
        values, row_scale = materialize_bf16(unit, forests, None)
        return values.float() * row_scale[:, None]
    tile, row_scale = materialize_fp8(unit, forests, None)
    # materialize_fp8 returns the E4M3FN BYTES (uint8): view them as FP8
    # before widening, or the byte codes 0..255 are read as values.
    return tile.view(torch.float8_e4m3fn).float() * row_scale.float().reshape(-1, 1)


def pairs_of(rungs):
    whole = sorted(r for r in rungs if r % 256 == 0)
    return [(lo, lo + 256) for lo in whole if lo + 256 in whole]


def _total(row, metric):
    """The unit's summed squared error, so units of different sizes add: the
    weight metric over rows x cols weights; the output metric per token, over
    the unit's output rows (``y_mse`` is a mean over tokens x rows)."""
    rows, cols = row["shape"]
    value = row.get(metric)
    if value is None:
        return None
    return value * rows * cols if metric == "w_mse" else value * rows


def analyse(rows, rungs, metric):
    """Per pair: chord residuals of the interior rungs, and the within-unit
    fractional rung against a byte-matched across-unit mix of the whole rates.

    Units differ in size (down_proj is rows x 2048, gate/up rows x 4096), so
    errors are summed squared errors (``_total``) and bytes are weights: moving
    a unit from rate r to r + 1 costs one bit per weight.  At fraction f the
    within-unit rung spends f bits per weight on every unit; the across-unit mix
    spends the same f * sum(size) bits raising whole units in order of gain per
    bit, the last one fractionally -- the continuous-knapsack optimum, which a
    whole-bit allocator over many units approaches."""
    units = sorted({(r["layer"], r["expert"], r["proj"]) for r in rows if r.get(metric) is not None})
    table = {(r["layer"], r["expert"], r["proj"], r["q256"]): r for r in rows
             if r.get(metric) is not None}
    out = {}
    for lo, hi in pairs_of(rungs):
        inner = [q for q in rungs if lo < q < hi]
        per_q = {}
        for q in inner:
            f = (q - lo) / 256.0
            resid, within, gains, size_total = [], 0.0, [], 0
            for u in units:
                rs = [table.get(u + (x,)) for x in (lo, hi, q)]
                if None in rs:
                    continue
                d_lo, d_hi, d_q = (_total(r, metric) for r in rs)
                size = rs[0]["shape"][0] * rs[0]["shape"][1]
                chord = (1 - f) * d_lo + f * d_hi
                resid.append((d_q - chord) / max(d_lo - d_hi, 1e-30))
                within += d_q
                gains.append(((d_lo - d_hi) / size, d_lo, d_hi, size))
                size_total += size
            if not gains:
                continue
            gains.sort(key=lambda g: g[0], reverse=True)
            budget = f * size_total
            across = 0.0
            for per_bit, d_lo, d_hi, size in gains:
                take = min(1.0, max(0.0, budget / size)) if per_bit > 0 else 0.0
                budget -= take * size
                across += d_lo - take * (d_lo - d_hi)
            resid.sort()
            n = len(resid)
            per_q[q] = {"units": n, "chord_resid_median": resid[n // 2],
                        "chord_resid_min": resid[0], "chord_resid_max": resid[-1],
                        "within_unit_total": within, "across_unit_total": across,
                        "within_over_across": within / across if across else None}
        out[f"{lo}-{hi}"] = per_q
    return out


def report(result):
    for metric, per_pair in result["analysis"].items():
        for pair, per_q in per_pair.items():
            for q, s in per_q.items():
                print(f"{metric} {pair} q{q}: chord resid median {s['chord_resid_median']:+.4f} "
                      f"[{s['chord_resid_min']:+.4f}, {s['chord_resid_max']:+.4f}] "
                      f"within/across {s['within_over_across']:.4f} (n={s['units']})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reanalyse", action="store_true",
                    help="recompute the analysis of --out/rung_chord.json's rows; encode nothing")
    ap.add_argument("--layers", default="5,20,42")
    ap.add_argument("--experts", default="0,1")
    ap.add_argument("--projs", default="gate_proj,up_proj,down_proj")
    ap.add_argument("--rungs", default=",".join(map(str, RUNGS)))
    ap.add_argument("--rows", type=int, default=0, help="encode only the first ROWS rows (0 = all)")
    ap.add_argument("--grid", choices=("e4m3", "bf16"), default="e4m3",
                    help="the payload grid: Tessera-8 (E4M3) or Tessera-16 (BF16, folded lane)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rungs = [int(r) for r in args.rungs.split(",")]
    if args.reanalyse:
        result = json.loads((out_dir / "rung_chord.json").read_text())
        result["analysis"] = {m: analyse(result["rows"], result["meta"]["rungs"], m)
                              for m in ("w_mse", "y_mse")}
        (out_dir / "rung_chord.json").write_text(json.dumps(result, indent=1))
        report(result)
        return 0
    import torch
    from safetensors import safe_open
    from tessera.alphabet import BF16_GRID, E4M3_GRID
    from tessera.export import wire_recipe

    torch.manual_seed(0)
    mapping = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    payload = BF16_GRID if args.grid == "bf16" else E4M3_GRID
    meta = {"grid": args.grid, "recipe": {str(q): repr(wire_recipe(payload, q)) for q in rungs[:1]},
            "tessera_head": os.environ.get("TESSERA_HEAD"), "image": os.environ.get("ORACLE_IMAGE"),
            "host": os.environ.get("HOST_NAME"), "pb_action": os.environ.get("PB_ACTION_KEY"),
            "src": SRC, "act": ACT, "rungs": rungs}
    rows = []
    t0 = time.time()
    for layer in (int(x) for x in args.layers.split(",")):
        blob = torch.load(f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                          map_location="cpu", weights_only=False)
        x = blob["inputs"].float().cuda()
        for expert in (int(x) for x in args.experts.split(",")):
            for proj in args.projs.split(","):
                name = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                with safe_open(f"{SRC}/{mapping[name]}", framework="pt") as handle:
                    w = handle.get_tensor(name).cuda().float()
                if args.rows:
                    w = w[:args.rows].contiguous()
                ref = x @ w.T if proj != "down_proj" else None
                for q in rungs:
                    t = time.time()
                    w_hat = encode(w, q, name, args.grid)
                    d = w_hat - w
                    row = {"layer": layer, "expert": expert, "proj": proj, "q256": q,
                           "shape": list(w.shape), "w_mse": float(d.pow(2).mean()),
                           "w_rel": float(d.norm() / w.norm()),
                           "y_mse": (float((x @ w_hat.T - ref).pow(2).mean())
                                     if ref is not None else None),
                           "encode_s": time.time() - t}
                    if row["w_rel"] > 0.5:
                        raise SystemExit(f"{name} q{q}: w_rel {row['w_rel']:.3f} -- the decode "
                                         "does not reconstruct the weight; refusing to analyse it")
                    rows.append(row)
                    print(f"L{layer} e{expert} {proj:<9} q{q:<5} w_rel {row['w_rel']:.5f} "
                          f"y_mse {row['y_mse'] if row['y_mse'] is None else round(row['y_mse'], 8)} "
                          f"({row['encode_s']:.1f}s, {time.time() - t0:.0f}s total)", flush=True)
                    del w_hat, d
                del w, ref
                torch.cuda.empty_cache()
        del x
    result = {"meta": meta, "rows": rows,
              "analysis": {m: analyse(rows, rungs, m) for m in ("w_mse", "y_mse")}}
    (out_dir / "rung_chord.json").write_text(json.dumps(result, indent=1))
    report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
