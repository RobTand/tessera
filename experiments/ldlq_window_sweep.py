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
import hashlib
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
from tessera.encode import refit_diagnostics                  # noqa: E402
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
    ap.add_argument("--gauss-seidel", action="store_true",
                    help="add the Gauss-Seidel sweep of the metric refit's block scales "
                         "beside the Jacobi step it replaces (issue #35).  LUT plane and "
                         "a full-H metric only -- under a diagonal metric the blocks "
                         "decouple and the encoder refuses the flag rather than name an "
                         "arm that changed nothing.")
    ap.add_argument("--coupled-landing", action="store_true",
                    help="add issue #50's arms: the full-H refit with its landing made "
                         "cross-block aware (refit_coupled_landing), under both sweep orders, "
                         "on every pass and on the trailing refit only")
    ap.add_argument("--drift-control", action="store_true",
                    help="run the served default (`LDLQ <pair> + refit h^<first alpha>`) as "
                         "the FIRST arm and again as the LAST, under distinct names.  One "
                         "process, same weights, same H: the two runs are the same encode, "
                         "so any difference between them is drift and bounds what a "
                         "difference between two ARMS is allowed to mean.  Needs --pair, "
                         "because a per-unit best is not known before the sweep runs.")
    ap.add_argument("--out", default="experiments/results/tessera_ldlq_window_sweep.json")
    a = ap.parse_args()
    if a.drift_control and a.pair is None:
        raise SystemExit("--drift-control needs --pair: the control arm has to be the same "
                         "named arm on every unit, and a per-unit best is not one")

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

            def run(arm, dup_ok=False, **kw):
                t0 = time.time()
                with refit_diagnostics() as diag:
                    _, unit, forests = encode_linear_planes(
                        W, grid=grid, q256=a.q256, name=name, verify=False, **kw)
                secs = time.time() - t0
                st = materialize_stock(unit, forests, DEFAULT_CODE)
                What = stock_dequant(st).to(dev).float()
                # A lever that encodes to the same bytes as an arm without it
                # is a silent no-op -- exactly what a named arm hides.  Say so,
                # loudly, next to the number.  sha256, not hash(): the digest
                # is written to the JSON and read across processes, and
                # PYTHONHASHSEED is randomised per run.
                key = hashlib.sha256(
                    What.cpu().numpy().tobytes()).hexdigest()
                if key in seen_bytes and not dup_ok:
                    log(f"    !! IDENTICAL BYTES: {arm!r} == {seen_bytes[key]!r} "
                        f"-- that lever did nothing on this unit")
                seen_bytes.setdefault(key, arm)
                r = score(arm, What, secs)
                r["sha256"] = key
                # Where the refit's own arithmetic went, pass by pass.  The
                # landed number is the only one the encode acts on, but it is
                # the STEP and the LANDING added together, and #35 is a
                # question about the step alone.
                r["refit"] = [dict(d) for d in diag]
                return r

            # The drift control runs FIRST, before any other arm has touched
            # the box, and again LAST after every one of them has.  Both are
            # the same encode of the same weights in the same process, so the
            # pair bounds every other difference in the table: an arm-to-arm
            # gap smaller than the control's own spread is not a result.
            control = None
            if a.drift_control:
                pair = (a.pair[0], int(a.pair[1]))
                Lc = block_ldl(regularize_hessian(H, sigma_reg=pair[0]), pair[1])
                mc = hn.pow(a.alphas[0])
                control = (f"LDLQ {pair[0]}/{pair[1]} + refit h^{a.alphas[0]}",
                           dict(ldl=Lc, ldl_block=pair[1], refit_metric=mc))
                run(f"drift control FIRST [{control[0]}]", dup_ok=True, **control[1])

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
                # Under --drift-control this arm IS the control arm, run a
                # third time.  Its bytes matching the control's is the
                # encoder being deterministic, not a lever doing nothing, so
                # it must not raise the no-op warning that means the latter.
                run(f"LDLQ {best[0]}/{best[1]} + refit h^{alpha}",
                    dup_ok=(control is not None and
                            control[0] == f"LDLQ {best[0]}/{best[1]} + refit h^{alpha}"),
                    ldl=L, ldl_block=best[1], refit_metric=m)
            run("refit full-H only", refit_metric=H)
            run(f"LDLQ {best[0]}/{best[1]} + refit full-H",
                ldl=L, ldl_block=best[1], refit_metric=H)
            if a.gauss_seidel:
                # The ONE difference from the arm above: the sweep order of
                # the metric refit's block scales.  Same LDLQ factor, same
                # block, same metric, same everything else -- two treatments
                # in one arm would make the comparison say nothing.
                run(f"LDLQ {best[0]}/{best[1]} + refit full-H (Gauss-Seidel)",
                    ldl=L, ldl_block=best[1], refit_metric=H,
                    refit_gauss_seidel=True)
                run("refit full-H only (Gauss-Seidel)",
                    refit_metric=H, refit_gauss_seidel=True)
            if a.coupled_landing:
                # Issue #50: the ONE difference from the two full-H arms above
                # is the landing -- each block re-assigned to the table entry
                # minimising the full quadratic given its neighbours, instead
                # of nearest to its own continuous target.  Under both sweep
                # orders, so the sweep and the landing stay separate treatments.
                run(f"LDLQ {best[0]}/{best[1]} + refit full-H (Jacobi, coupled landing)",
                    ldl=L, ldl_block=best[1], refit_metric=H,
                    refit_coupled_landing=True)
                run(f"LDLQ {best[0]}/{best[1]} + refit full-H (Gauss-Seidel, coupled landing)",
                    ldl=L, ldl_block=best[1], refit_metric=H,
                    refit_gauss_seidel=True, refit_coupled_landing=True)
                # The same landing on the LAST refit only: the inner passes
                # are the plain full-H arms' passes, so the trellis sees the
                # planes it always saw and only the shipped plane is re-landed.
                run(f"LDLQ {best[0]}/{best[1]} + refit full-H (Jacobi, trailing coupled landing)",
                    ldl=L, ldl_block=best[1], refit_metric=H,
                    refit_coupled_landing="trailing")
                run(f"LDLQ {best[0]}/{best[1]} + refit full-H (Gauss-Seidel, trailing coupled landing)",
                    ldl=L, ldl_block=best[1], refit_metric=H,
                    refit_gauss_seidel=True, refit_coupled_landing="trailing")
            if channel:
                run(f"LDLQ {best[0]}/{best[1]} + refit full-H + reach floor",
                    ldl=L, ldl_block=best[1], refit_metric=H, refit_reach_floor=True)

            if control is not None:
                last = run(f"drift control LAST [{control[0]}]", dup_ok=True, **control[1])
                first = res[f"drift control FIRST [{control[0]}]"]
                same = last["sha256"] == first["sha256"]
                log(f"    -- drift control: bytes {'IDENTICAL' if same else 'DIFFER'}"
                    f"  out {first['out']:.6f} -> {last['out']:.6f}"
                    f"  ({last['out'] / first['out'] - 1.0:+.4%})"
                    f"  hfit {first['hfit']:.6f} -> {last['hfit']:.6f}"
                    f"  ({last['hfit'] / first['hfit'] - 1.0:+.4%})")
                res["_drift"] = {
                    "bytes_identical": same,
                    "out_first": first["out"], "out_last": last["out"],
                    "hfit_first": first["hfit"], "hfit_last": last["hfit"],
                }

            out["units"][name] = res
            Path(a.out).write_text(json.dumps(out, indent=1))
            del W, H, X, Y, factors
            torch.cuda.empty_cache()

    # Geomean of the out-space score per arm, over the units every arm ran on.
    arms = set.intersection(*[{k for k in v if not k.startswith("_")}
                              for v in out["units"].values()])

    def geo(arm, field):
        return math.exp(sum(math.log(out["units"][u][arm][field]) for u in out["units"])
                        / len(out["units"]))

    log("\n== geomean, all units -- out-space (held-out rows) and hfit (the fit "
        "rows' own quadratic, the one the refit's guard is monotone in)")
    rows = sorted(((arm, geo(arm, "out")) for arm in arms), key=lambda r: r[1])
    base = dict(rows)["baseline (no LDLQ, plain refit)"]
    log(f"    {'arm':<50} {'out':>9} {'vs base':>9} {'hfit':>9}")
    for arm, g in rows:
        log(f"    {arm:<50} {g:9.5f} {g / base:8.4f}x {geo(arm, 'hfit'):9.5f}")
    out["geomean_out"] = dict(rows)
    out["geomean_hfit"] = {arm: geo(arm, "hfit") for arm in arms}

    # The promotion rule of issue #35, evaluated in the run rather than by
    # hand afterwards: Gauss-Seidel is promoted only if it beats the SERVED
    # h^1.0 default on the out geomean by more than the 1.38% the two refit
    # objectives already span, and stays monotone on hfit.  The margin is
    # written here so that it cannot be chosen once the numbers are visible.
    MARGIN = 0.0138
    gs = next((x for x in arms if "Gauss-Seidel" in x and x.startswith("LDLQ")), None)
    ja = next((x for x in arms if x.endswith("refit full-H") and x.startswith("LDLQ")), None)
    ctl = next((x for x in arms if x.startswith("drift control FIRST")), None) or \
        next((x for x in arms if "refit h^" in x and x.startswith("LDLQ")), None)
    if gs and ctl:
        need = geo(ctl, "out") / (1.0 + MARGIN)
        got = geo(gs, "out")
        # (i) the guard reading: every refit arm at or below the baseline on
        # hfit.  (ii) the arm reading: GS at or below Jacobi on hfit, per
        # unit.  Both are reported; they answer different questions and a
        # single word "monotone" would hide which one held.
        guard = {u: out["units"][u][gs]["hfit"] <= out["units"][u]["baseline (no LDLQ, plain refit)"]["hfit"]
                 for u in out["units"]}
        versus = ({u: out["units"][u][gs]["hfit"] <= out["units"][u][ja]["hfit"]
                   for u in out["units"]} if ja else {})
        verdict = {
            "control_arm": ctl, "gs_arm": gs, "jacobi_arm": ja,
            "margin_required": MARGIN,
            "control_out_geomean": geo(ctl, "out"),
            "gs_out_geomean": got,
            "gs_over_control": got / geo(ctl, "out"),
            "threshold_out_geomean": need,
            "beats_margin": got < need,
            "hfit_below_baseline_per_unit": guard,
            "hfit_gs_below_jacobi_per_unit": versus,
            "promoted": bool(got < need and all(guard.values())),
        }
        out["verdict_issue_35"] = verdict
        log(f"\n== issue #35 promotion rule (written before the numbers)")
        log(f"    control  {ctl!r}  out geomean {geo(ctl, 'out'):.5f}")
        log(f"    GS       {gs!r}  out geomean {got:.5f}  "
            f"({got / geo(ctl, 'out') - 1.0:+.4%} vs control)")
        if ja:
            log(f"    Jacobi   {ja!r}  out geomean {geo(ja, 'out'):.5f}")
        log(f"    needs    < {need:.5f} (control / 1.0138)")
        log(f"    hfit <= baseline on every unit: {all(guard.values())}  {guard}")
        if versus:
            log(f"    hfit GS <= Jacobi on every unit: {all(versus.values())}  {versus}")
        log(f"    VERDICT: {'PROMOTE to a serve' if verdict['promoted'] else 'MEASURED AND REJECTED'}")

    Path(a.out).write_text(json.dumps(out, indent=1))
    Path(a.out).with_suffix(".log").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
