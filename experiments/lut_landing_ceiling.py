#!/usr/bin/env python
"""How much of the LUT refit's step does the sixteen-entry landing give back?

Issue #50, and the cheap decisive form of it: the refit's landed error is the
step it reaches minus what the table takes back, so **run the encode with the
table removed and read the same geomean**.  That number is the most any table
fit could return -- it is not a number a table fit reaches -- and if it is
small there is nothing to build, for the same reason `#4` closed.

Three landings per refit objective, everything else held:

* ``table`` -- the wire.  Sixteen E4M3 entries fit by ``_fit_lut`` under the
  separable model ``sum_b A_b (c_b - s*_b)^2``, each block assigned
  nearest-in-linear.  Four bits per block.
* ``grid``  -- every in-range E4M3 value, nearest-in-linear.  Eight bits per
  block, which the LUT index does not have.  It removes the sixteen-entry
  budget and keeps the values the wire can name, so ``grid`` against ``none``
  separates "the table is too small" from "the grid is too coarse".
* ``none``  -- the continuous per-block optimum.  Not a plane at all.

and two objectives: the served ``h^1.0`` default and issue `#35`'s promoted
full-Hessian Gauss-Seidel arm, plus its Jacobi predecessor.

**Only ``table`` arms are serialisable.**  The other two build a unit whose
scale plane is not what the encode held, so every arm here is scored on the
reconstruction the encoder hands back through ``lut_landing``'s sink -- and
each ``table`` arm is scored BOTH ways, so the run proves the sink equals
``stock_dequant`` of the same unit before any ceiling number is believed.

The served default runs first and again last, in one process, as `#35`'s
receipt did: an arm-to-arm gap below the control's own spread is not a result.

    PYTHONPATH=src TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache \
      python experiments/lut_landing_ceiling.py --out .../qwen_lut_landing.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.alphabet import SERIALISABLE_GRIDS               # noqa: E402
from tessera.compensate import block_ldl, regularize_hessian  # noqa: E402
from tessera.encode import lut_landing, refit_diagnostics     # noqa: E402
from tessera.export import (                                  # noqa: E402
    DEFAULT_CODE, encode_linear_planes, wire_recipe)
from tessera.manifest import ScalePlaneKind                   # noqa: E402
from tessera.stock import materialize_stock, stock_dequant    # noqa: E402


def grid_by_name(name: str):
    for g in SERIALISABLE_GRIDS.values():
        if g.name == name:
            return g
    raise SystemExit(f"unknown grid {name!r}")


def rel(num: torch.Tensor, den: torch.Tensor) -> float:
    return float(num.norm() / den.norm())


def geo(units, arm, field):
    return math.exp(sum(math.log(units[u][arm][field]) for u in units) / len(units))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/rob/models/Qwen3-0.6B")
    ap.add_argument("--h", default="/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt")
    ap.add_argument("--acts", default="/mnt/shared/tessera-runs/ldlq/x_eval_qwen06b.pt")
    ap.add_argument("--grid", default="E2M1x2")
    ap.add_argument("--q256", type=int, default=896)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--units", nargs="*", default=None)
    ap.add_argument("--out", default="experiments/results/tessera_lut_landing_ceiling.json")
    a = ap.parse_args()

    grid = grid_by_name(a.grid)
    recipe = wire_recipe(grid, a.q256)
    if recipe.scale_plane is not ScalePlaneKind.LUT:
        raise SystemExit(f"{a.grid} q256={a.q256} is a {recipe.scale_plane.name} plane; "
                         "the landing this measures is the LUT plane's")
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
    log(f"H from {prov['source']}  fit {prov['fit_tokens']} tok  "
        f"eval {prov['eval_tokens']} tok (sha {prov['eval_ids_sha256'][:12]})")

    with safe_open(f"{a.model}/model.safetensors", framework="pt") as f:
        for name in units:
            if name not in Hall:
                raise SystemExit(f"no Hessian for {name}")
            W = f.get_tensor(name + ".weight").to(dev, torch.float32).contiguous()
            H = Hall[name].to(dev, torch.float32)
            X = acts["x"][name].to(dev, torch.float32)
            Y = X @ W.T
            h = H.diagonal().clone()
            hn = h / h.mean()
            den_w, den_h = W.norm(), float(((W * W).sum(0) * hn).sum())
            den_hf = float(((W @ H) * W).sum())
            # Built exactly as ``ldlq_window_sweep.py`` builds them, so this
            # run's control arm is the SAME encode as #35's control arm and the
            # two receipts join on it rather than sitting side by side.
            L = block_ldl(regularize_hessian(H, sigma_reg=a.sigma), a.block)
            hmetric = hn.pow(a.alpha)
            res: dict = {}
            seen: dict = {}
            log(f"\n== {name} {tuple(W.shape)}  eval rows {X.shape[0]}")
            log(f"    {'arm':<48} {'out':>9} {'plain':>9} {'hwt':>9} {'hfit':>9} {'s':>6}")

            def run(arm, landing, dup_ok=False, **kw):
                t0 = time.time()
                with lut_landing(landing) as sink, refit_diagnostics() as diag:
                    _, unit, forests = encode_linear_planes(
                        W, grid=grid, q256=a.q256, name=name, verify=False,
                        ldl=L, ldl_block=a.block, **kw)
                secs = time.time() - t0
                What = sink["work_reconstruction"].to(dev).float()
                r = {"landing": landing, "serialisable": bool(sink["serialisable"])}
                # A ``table`` arm is the wire, so it is scored the way the wire
                # is scored TOO, and the two must agree.  That identity is what
                # licenses reading the ceiling arms off the sink at all.
                if landing == "table":
                    wire = stock_dequant(
                        materialize_stock(unit, forests, DEFAULT_CODE)).to(dev).float()
                    r["sink_vs_wire_max_abs"] = float((wire - What).abs().max())
                    r["sink_vs_wire_rel"] = float((wire - What).norm() / W.norm())
                    r["sink_vs_wire_bit_identical"] = bool(torch.equal(wire, What))
                E = What - W
                r.update({
                    "out": rel(X @ E.T, Y),
                    "plain": float(E.norm() / den_w),
                    "hweighted": math.sqrt(float(((E * E).sum(0) * hn).sum()) / den_h),
                    "hfit": math.sqrt(float(((E @ H) * E).sum()) / den_hf),
                    "secs": secs,
                })
                key = hashlib.sha256(What.cpu().numpy().tobytes()).hexdigest()
                if key in seen and not dup_ok:
                    log(f"    !! IDENTICAL RECONSTRUCTION: {arm!r} == {seen[key]!r} "
                        f"-- that lever did nothing on this unit")
                seen.setdefault(key, arm)
                r["sha256"] = key
                r["refit"] = [dict(d) for d in diag]
                res[arm] = r
                log(f"    {arm:<48} {r['out']:9.5f} {r['plain']:9.5f} "
                    f"{r['hweighted']:9.5f} {r['hfit']:9.5f} {secs:6.1f}"
                    + ("" if landing == "table" else "   [NOT A WIRE]"))
                return r

            ctl = f"LDLQ {a.sigma}/{a.block} + refit h^{a.alpha}"
            run(f"drift control FIRST [{ctl}]", "table", dup_ok=True, refit_metric=hmetric)
            for landing in ("grid", "none"):
                run(f"{ctl} | landing={landing}", landing, refit_metric=hmetric)
            gs = f"LDLQ {a.sigma}/{a.block} + refit full-H (Gauss-Seidel)"
            run(gs, "table", refit_metric=H, refit_gauss_seidel=True)
            for landing in ("grid", "none"):
                run(f"{gs} | landing={landing}", landing,
                    refit_metric=H, refit_gauss_seidel=True)
            jac = f"LDLQ {a.sigma}/{a.block} + refit full-H"
            run(jac, "table", refit_metric=H)
            run(f"{jac} | landing=none", "none", refit_metric=H)
            last = run(f"drift control LAST [{ctl}]", "table", dup_ok=True,
                       refit_metric=hmetric)
            first = res[f"drift control FIRST [{ctl}]"]
            same = first["sha256"] == last["sha256"]
            log(f"    -- drift control: reconstruction "
                f"{'IDENTICAL' if same else 'DIFFERS'}  "
                f"out {last['out'] / first['out'] - 1:+.4%}  "
                f"hfit {last['hfit'] / first['hfit'] - 1:+.4%}")
            out["units"][name] = res
            del W, H, X, Y, L
            torch.cuda.empty_cache()

    names = list(out["units"])
    arms = list(out["units"][names[0]])
    log("\n== geomeans over %d units" % len(names))
    log(f"    {'arm':<48} {'out':>9} {'hfit':>9} {'out/ctl':>9} {'hfit/ctl':>9}")
    ctl_arm = next(x for x in arms if x.startswith("drift control FIRST"))
    gc, hc = geo(out["units"], ctl_arm, "out"), geo(out["units"], ctl_arm, "hfit")
    for arm in arms:
        g, hh = geo(out["units"], arm, "out"), geo(out["units"], arm, "hfit")
        log(f"    {arm:<48} {g:9.5f} {hh:9.5f} {g / gc:9.4f} {hh / hc:9.4f}")
    out["geomeans"] = {arm: {f: geo(out["units"], arm, f) for f in ("out", "hfit")}
                       for arm in arms}
    out["log"] = lines
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
