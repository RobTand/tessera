#!/usr/bin/env python
"""LDLQ and the h-weighted refit on a Tessera wire, in weight space.

The two encoder-side levers of ``tessera-ldlq-window-served-2026-09-02``, swept
on a handful of dense Qwen3-0.6B Linears before either is put in front of a
served KL.  ``--grid``/``--q256`` choose the wire: ``E4M3 1024`` is the FP8
route's window body over the CHANNEL plane, where both levers were measured
first; ``E2M1x2 896`` is the 4-bit route's TCQ cap wire over the LUT plane,
where the same two levers had to be implemented rather than bypassed
(``tessera-ldlq-lut-plane-served-2026-09-02``).  The arms follow the wire: the
reach floor is a CHANNEL mechanism and is not offered on a block plane.

* **LDLQ** -- cross-column error feedback over input-feature blocks, the only
  coupling the trellis leaves on the table (``compensate.py``).  ``--sigmas``
  sweeps the Hessian regulariser.
* **the refit metric** -- the row-scale least squares run under the diagonal
  ``h^alpha`` or under the full Hessian's exact quadratic instead of the plain
  squared error.

Every arm is the shipping wire for that grid and rung (``wire_recipe``),
materialised the way the stock twin materialises it, so the number here is a
number about the bytes that would be served.

The score is **out-space on held-out rows**: ``||X_ev (W - What)^T|| / ||X_ev
W^T||`` with ``X_ev`` the activations of ``capture_h_full.py``'s eval slice,
which is disjoint from both the Hessian's fit slice and the KL corpus.  Plain
weight error and the diagonal-h-weighted error are reported beside it because
the two levers move them in opposite directions and a receipt that quotes one
of them alone is choosing its answer.

    PYTHONPATH=src TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache \
      python experiments/ldlq_window_sweep.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.alphabet import SERIALISABLE_GRIDS              # noqa: E402
from tessera.compensate import block_ldl, regularize_hessian  # noqa: E402
from tessera.export import (                                 # noqa: E402
    DEFAULT_CODE, encode_linear_planes, wire_recipe)
from tessera.manifest import ScalePlaneKind                  # noqa: E402
from tessera.stock import materialize_stock, stock_dequant   # noqa: E402


def grid_by_name(name: str):
    for g in SERIALISABLE_GRIDS.values():
        if g.name == name:
            return g
    raise SystemExit(f"unknown grid {name!r}; one of "
                     f"{[g.name for g in SERIALISABLE_GRIDS.values()]}")


def rel(num: torch.Tensor, den: torch.Tensor) -> float:
    return float(num.norm() / den.norm())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/rob/models/Qwen3-0.6B")
    ap.add_argument("--h", default="/home/rob/tessera-runs/ldlq/h_full_qwen06b.pt")
    ap.add_argument("--acts", default="/home/rob/tessera-runs/ldlq/x_eval_qwen06b.pt")
    ap.add_argument("--grid", default="E4M3", help="E4M3 (FP8 route) or E2M1x2 (4-bit route)")
    ap.add_argument("--q256", type=int, default=1024)
    ap.add_argument("--block", type=int, nargs="+", default=[128])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--units", nargs="*", default=None)
    ap.add_argument("--pair", type=float, nargs=2, default=None, metavar=("SIGMA", "BLOCK"),
                    help="carry the refit arms on THIS (sigma, block) rather than each unit's own "
                         "best.  A per-unit best gives every unit a differently named arm, which is "
                         "not a thing a geomean can be taken over -- and an exporter has to pick one "
                         "setting for the whole checkpoint anyway.")
    ap.add_argument("--out", default="experiments/results/tessera_ldlq_window_sweep.json")
    a = ap.parse_args()

    grid = grid_by_name(a.grid)
    recipe = wire_recipe(grid, a.q256)
    channel = recipe.scale_plane is ScalePlaneKind.CHANNEL
    payload = torch.load(a.h, map_location="cpu", weights_only=False)
    acts = torch.load(a.acts, map_location="cpu", weights_only=False)
    Hall, prov = payload["H"], payload["provenance"]
    units = a.units or sorted(acts["x"])
    dev = "cuda"
    out = {"args": vars(a), "provenance": prov, "units": {}}
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"wire: {grid.name} q256={a.q256} -> body {recipe.body.name} plane "
        f"{recipe.scale_plane.name} span {recipe.span} L={recipe.window_bits}")
    log(f"H from {prov['source']}  fit {prov['fit_tokens']} tok "
        f"(sha {prov['fit_ids_sha256'][:12]})  eval {prov['eval_tokens']} tok "
        f"(sha {prov['eval_ids_sha256'][:12]})")

    with safe_open(f"{a.model}/model.safetensors", framework="pt") as f:
        for name in units:
            if name not in Hall:
                raise SystemExit(f"no Hessian for {name}; the keys must match the encoder's units")
            W = f.get_tensor(name + ".weight").to(dev, torch.float32).contiguous()
            H = Hall[name].to(dev, torch.float32)
            if H.shape[0] != W.shape[1]:
                raise SystemExit(f"{name}: H is {tuple(H.shape)} for {W.shape[1]} inputs")
            X = acts["x"][name].to(dev, torch.float32)
            Y = X @ W.T
            h = H.diagonal().clone()
            hn = h / h.mean()
            den_w = W.norm()
            den_h = float(((W * W).sum(0) * hn).sum())
            # The refit is provably monotone in ``E H E^T`` -- the FIT metric.
            # Reporting only ``out`` (the held-out eval rows) cannot tell a
            # generalisation gap from a broken accept guard, so carry the
            # quantity the guard claims to lower, next to the one that decides.
            den_hf = float(((W @ H) * W).sum())
            res = {}
            seen_bytes: dict = {}
            log(f"\n== {name} {tuple(W.shape)}  eval rows {X.shape[0]}")
            log(f"    {'arm':<44} {'out':>9} {'plain':>9} {'hwt':>9} {'hfit':>9} "
                f"{'clip%':>7} {'s':>6}")

            def score(arm, What, secs):
                E = What - W
                r = {
                    "out": rel(X @ E.T, Y),
                    "plain": float(E.norm() / den_w),
                    "hweighted": math.sqrt(float(((E * E).sum(0) * hn).sum()) / den_h),
                    "hfit": math.sqrt(float(((E @ H) * E).sum()) / den_hf),
                    "secs": secs,
                }
                res[arm] = r
                log(f"    {arm:<44} {r['out']:9.5f} {r['plain']:9.5f} "
                    f"{r['hweighted']:9.5f} {r['hfit']:9.5f} {'':>7} {secs:6.1f}")
                return r

            def run(arm, **kw):
                t0 = time.time()
                _, unit, forests = encode_linear_planes(
                    W, grid=grid, q256=a.q256, name=name, verify=False, **kw)
                secs = time.time() - t0
                st = materialize_stock(unit, forests, DEFAULT_CODE)
                What = stock_dequant(st).to(dev).float()
                # A lever that encodes to the same bytes as an arm without it
                # is a silent no-op -- exactly what a named arm hides.  Say so,
                # loudly, next to the number.
                key = hash(What.cpu().numpy().tobytes())
                if key in seen_bytes:
                    log(f"    !! IDENTICAL BYTES: {arm!r} == {seen_bytes[key]!r} "
                        f"-- that lever did nothing on this unit")
                else:
                    seen_bytes[key] = arm
                return score(arm, What, secs)

            run("baseline (no LDLQ, plain refit)")

            factors = {}
            for sigma in a.sigmas:
                Hr = regularize_hessian(H, sigma_reg=sigma)
                for blk in a.block:
                    L = block_ldl(Hr, blk)
                    factors[(sigma, blk)] = L
                    run(f"LDLQ sigma={sigma} block={blk}", ldl=L, ldl_block=blk)

            # One (sigma, block) carries the refit arms: the pair named on the
            # command line, or -- for a first pass over an unknown grid -- this
            # unit's own best.
            if a.pair is not None:
                best = (a.pair[0], int(a.pair[1]))
                if best not in factors:
                    Hr = regularize_hessian(H, sigma_reg=best[0])
                    factors[best] = block_ldl(Hr, best[1])
            else:
                best = min(
                    factors, key=lambda k: res[f"LDLQ sigma={k[0]} block={k[1]}"]["out"])
            L = factors[best]
            log(f"    -- best LDLQ: sigma={best[0]} block={best[1]}")
            res["_best_ldlq"] = {"sigma": best[0], "block": best[1]}

            if channel:
                run(f"LDLQ {best[0]}/{best[1]} + reach floor",
                    ldl=L, ldl_block=best[1], refit_reach_floor=True)
            for alpha in a.alphas:
                m = hn.pow(alpha)
                run(f"refit h^{alpha} only", refit_metric=m)
                run(f"LDLQ {best[0]}/{best[1]} + refit h^{alpha}",
                    ldl=L, ldl_block=best[1], refit_metric=m)
            run("refit full-H only", refit_metric=H)
            run(f"LDLQ {best[0]}/{best[1]} + refit full-H",
                ldl=L, ldl_block=best[1], refit_metric=H)
            if channel:
                run(f"LDLQ {best[0]}/{best[1]} + refit full-H + reach floor",
                    ldl=L, ldl_block=best[1], refit_metric=H, refit_reach_floor=True)

            out["units"][name] = res
            Path(a.out).write_text(json.dumps(out, indent=1))
            del W, H, X, Y, factors
            torch.cuda.empty_cache()

    # Geomean of the out-space score per arm, over the units every arm ran on.
    arms = set.intersection(*[{k for k in v if not k.startswith("_")}
                              for v in out["units"].values()])
    log("\n== geomean out-space (held-out rows), all units")
    rows = sorted(
        ((arm, math.exp(sum(math.log(out["units"][u][arm]["out"]) for u in out["units"])
                        / len(out["units"]))) for arm in arms), key=lambda r: r[1])
    base = dict(rows)["baseline (no LDLQ, plain refit)"]
    for arm, g in rows:
        log(f"    {arm:<44} {g:9.5f}  {g / base:6.4f}x")
    out["geomean_out"] = dict(rows)
    Path(a.out).write_text(json.dumps(out, indent=1))
    Path(a.out).with_suffix(".log").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
