#!/usr/bin/env python
"""The BF16 reach term, measured through the encode an export actually runs (issue #48).

``BF16_RECIPE`` used to leave ``window_sigma`` unset, which ties the window
table's spread to the row scale and pins the body's **reach** -- how many
row-RMS the largest table entry can express -- to one value at every rung.
Issue #18's ``--stage reach`` found that value optimal at R=4 on four dense
Qwen Linears and 19% off at R=8, and could only find it by pinning
``window_sigma`` outside the recipe.  ``export._window_sigma_for`` now carries
the term, and this is the measurement of the thing that was built rather than
of a proxy for it.

**Which objective the reach is fitted to, decided before the run.**  The first
pass of this harness ran a *weights-only* encode and read ``wt``, and the
receipt it produced recorded a disagreement it could not settle: at R=7 and
R=8 the wide arm beat the law on the Hessian-weighted ``h`` while losing on
``wt``.  A default fitted to ``wt`` would be choosing the reach for every
future BF16 export by a metric no export optimises -- PrismaQuant writes those
bytes with a Hessian, and LDLQ (sigma=1, block=32) plus the exact full-H
row-scale refit are what ``ActivationSource``'s defaults turn on.  So:

* ``--production CAPTURE`` runs every arm through ``ActivationSource`` at its
  shipping defaults, which is the path an export driver takes.  No hand-built
  kwargs: if the shipped recipe changes, this harness changes with it.
* ``--eval-x ACTS`` adds ``out``, the output-space error on the capture's
  **held-out** rows -- wikitext-2 train, a slice disjoint from the 16k tokens
  H was fit on (``capture_h_full.py``).  **``out`` is the deciding column.**
  It is the quantity the production objective is a surrogate for, measured
  where the refit could not have fitted it.  ``wt`` and ``h`` are kept in the
  table for continuity with the weights-only receipt and decide nothing.
* ``h`` is read from the **capture's own** H diagonal, not from
  ``bf16_route_weight_space.DENSE_H``.  Those are two different captures: on
  these four units the older ``h_diag.pt`` differs from this H's diagonal by
  2% to 54% relative.  Scoring an h-aware encode against a Hessian it was not
  fit on measures the gap between two captures as well as the arm.

**The reach grid, and what counts as a disagreement.**  Two brackets a factor
of ``sqrt(2)`` apart cannot separate ``sqrt(R/R0)`` from ``sqrt((R + c)/(R0 +
c))``, which the receipt flagged as unpinned.  Each rung is swept on a
quarter-octave grid, ``2^(k/4)`` for ``k`` in ``-steps..+steps`` around the
law, and the optimum is reported **as a reach number**: the discrete minimum
refined by a parabola through its two neighbours in ``(log reach, log
metric)``.  An argmin at the edge of the grid is flagged rather than reported,
because a parabola through an edge extrapolates.

The materiality threshold is fixed here, before the run, at **1.06x in
reach**: that is ``sqrt(4.5 / 4)``, the error the recipe already accepts by
rounding a rung's rate to whole bits (``export._reach_rate_for``).  A law
whose reach is off by less than its own rounding tolerance is not off.  If the
``out``-optimal reach at R=7 or R=8 sits further than 1.06x from the law, the
law is not the production optimum and the recipe must not default to it.

**The claims this registers before it measures anything.**

1.  The built path at the reference rung writes **the same bytes as the pinned
    wire**.  ``window_sigma = channel_sigma = 1.0`` is what ``encode_unit``
    resolves an unset spread to on a CHANNEL plane, so "explicit 1.0" and
    "unset" must be the same file, not merely the same error.  If they are
    not, the term is not a re-parameterisation and everything below is
    measuring two changes at once.
2.  R=4 is swept like every other rung.  The amplitude was calibrated on the
    weights-only optimum at R=4; if the production optimum there is a
    different reach, the amplitude recalibrates and
    ``BF16_REACH_REFERENCE_RATE`` may not be 4.  That is a bigger change than
    the exponent, because the byte-identity in claim 1 depends on the
    reference reach being the pinned one.
3.  The law predicts rungs it was never fitted to.  Under the weights-only
    encode it did, on ``wt``; whether it does under the production encode on
    ``out`` is the question this run exists to answer, and either answer is
    the finding.

**The controls.**  Every arm of a unit runs in one process in a fixed order;
the built arm is run first at each rung and repeated last, and the repeat is
asserted byte- *and* tensor-identical -- the encoder is deterministic, so this
is not a noise estimate but a check that no arm leaked state (the LDL factor
is built once per unit and handed to every arm, so a mutating encode is
exactly the failure this catches).  Every arm within a (unit, rung) writes the
same number of bytes, which is what makes the ratios readable at all; the bpp
column is printed so that is visible and not trusted.

**Note on the recorded R<=3 rows in the weights-only run.**  Those arms were
run *before* the floor existed: ``arms_for`` reads its ``law`` value from
``wire_recipe``, and the recipe is now floored at the reference rung, so a
re-run at R<=3 gives ``built (recipe) == pinned 1.0``.  The floor was added
**on** that result -- it is the finding, not a precondition of it.  To
reproduce the unfloored sub-reference arms, name them with ``--extra-sigmas``
(R=1 0.5, R=2 0.7071, R=3 0.8660).

Weight space and output space on four dense Linears of one small model.  **No
serve.**  Principle 3: the next gate is a served A/B at matched bytes in one
vLLM session on the BF16 lane, and nothing here is promotable without it.

Run::

    P=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
    E="TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache PYTHONPATH=src:experiments"
    env $E $P experiments/bf16_reach_recipe.py --out OUT/reach_production.json \
        --production /mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt \
        --eval-x /mnt/shared/tessera-runs/bf16/refs/x_eval_dense4.pt \
        --rungs 1024 1280 1536 1792 2048
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import BF16_GRID                              # noqa: E402
from tessera.export import (                                        # noqa: E402
    BF16_CHANNEL_SIGMA,
    BF16_WINDOW_SIGMA,
    ActivationSource,
    encode_linear_planes,
    wire_recipe,
)
from tessera.manifest import ScalePlaneKind                         # noqa: E402
from tessera.unit_artifact import read_unit_artifact                # noqa: E402

from bf16_l_sigma_sweep import (                                    # noqa: E402
    DENSE_UNITS,
    Bench,
    check_repeat_tensor,
    reach_stats,
    score,
    sha,
    tensor_sha,
)
from bf16_route_weight_space import DENSE_H, DENSE_SRC, geomean, open_all  # noqa: E402

#: The reach error the recipe already accepts by rounding a rung's rate to
#: whole bits (``export._reach_rate_for``), and therefore the threshold at
#: which a disagreement between the law and the measured optimum is material.
#: Fixed here rather than chosen once the numbers are in.
MATERIAL = math.sqrt(4.5 / 4.0)

#: The metrics, in the order the table prints them.  ``out`` decides; the
#: other two are continuity with the weights-only receipt.
AXES = ("wt", "h", "out")
DECIDING = "out"


def arms_for(q256: int, extra=(), steps: int = 2, gauge_twin: bool = False):
    """``(label, window_sigma, channel_sigma)`` for one rung, built arm first.

    ``None`` means "let the recipe decide" -- the built path, and the only arm
    that measures the change rather than a hand-set spread.  Around it, a
    quarter-octave grid: ``2^(k/4)`` for ``k`` in ``-steps..+steps``, which
    resolves the optimum to about 9% in reach instead of the 41% a
    ``sqrt(2)`` bracket resolves it to.  An arm that coincides with one
    already listed is dropped rather than run twice under two names.
    """
    law = wire_recipe(BF16_GRID, q256).window_sigma
    arms = [("built (recipe)", None, BF16_CHANNEL_SIGMA)]
    seen = {round(law, 9)}

    def add(label, sigma, csigma=BF16_CHANNEL_SIGMA):
        if round(float(sigma), 9) in seen:
            return
        seen.add(round(float(sigma), 9))
        arms.append((label, float(sigma), float(csigma)))

    # The old wire, spelled explicitly.  Kept at every rung including the
    # reference, where it is the *point*: there the recipe resolves to this
    # value and the two arms must be the same file, not merely the same error.
    add(f"pinned 1.0 (old wire) s={BF16_WINDOW_SIGMA:.4g}", BF16_WINDOW_SIGMA)
    for k in range(-steps, steps + 1):
        if k == 0:
            continue
        add(f"law*2^({k}/4) s={law * 2.0 ** (k / 4.0):.4g}", law * 2.0 ** (k / 4.0))
    # The sweep's own parameterisation of the same reach: a non-dyadic gauge
    # shift of the built arm.  Its tolerance is already established (0.12% at
    # worst across R=1..8), so it is off by default and available when a run
    # is reproducing #18 rather than deciding a default.
    if gauge_twin and round(law, 9) != round(BF16_WINDOW_SIGMA, 9):
        arms.append((f"gauge twin (1.0, {1.0 / law:.4g})", 1.0, 1.0 / law))
    for sigma in extra:
        add(f"extra s={float(sigma):.4g}", float(sigma))
    return arms


def optimum_reach(points, axis: str) -> dict:
    """The reach that minimises ``axis``, refined off the grid.

    ``points`` is ``[(reach, metrics), ...]`` on one rung of one unit.  The
    grid is geometric, so the parabola is fitted in ``(log reach, log
    metric)`` through the discrete minimum and its two neighbours; a minimum
    at either end of the grid is reported as an **edge** and not refined,
    because a parabola through an edge extrapolates rather than interpolates.
    """
    pts = sorted((r, m[axis]) for r, m in points if axis in m and r > 0)
    if len(pts) < 3:
        return {"reach": None, "edge": True, "why": "fewer than three arms"}
    i = min(range(len(pts)), key=lambda j: pts[j][1])
    if i in (0, len(pts) - 1):
        return {"reach": pts[i][0], "edge": True,
                "why": "argmin at the low edge" if i == 0 else "argmin at the high edge",
                "grid": [p[0] for p in pts]}
    (xa, ya), (xb, yb), (xc, yc) = (
        (math.log(pts[j][0]), math.log(pts[j][1])) for j in (i - 1, i, i + 1))
    # Vertex of the parabola through three points, in log-log.
    d1, d2 = (ya - yb) / (xa - xb), (yb - yc) / (xb - xc)
    curv = (d1 - d2) / (xa - xc)
    if curv <= 0:
        return {"reach": pts[i][0], "edge": True, "why": "not convex at the minimum",
                "grid": [p[0] for p in pts]}
    return {"reach": math.exp(0.5 * (xa + xb - d1 / curv)), "edge": False,
            "grid_reach": pts[i][0], "grid": [p[0] for p in pts]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/shared/tessera-runs/bf16/qsweep/reach_recipe.json")
    ap.add_argument("--units", nargs="*", default=DENSE_UNITS)
    ap.add_argument("--rungs", nargs="*", type=int, default=[1024, 1280, 1536, 1792, 2048])
    ap.add_argument("--extra-sigmas", nargs="*", type=float, default=[])
    ap.add_argument("--grid-steps", type=int, default=2,
                    help="quarter-octave arms each side of the law")
    ap.add_argument("--gauge-twin", action="store_true",
                    help="add #18's parameterisation of the same reach")
    ap.add_argument("--production", default=None, metavar="CAPTURE",
                    help="a capture_h_full.py payload; every arm is then encoded "
                         "through ActivationSource at its shipping defaults -- "
                         "the path an export takes")
    ap.add_argument("--eval-x", default=None, metavar="ACTS",
                    help="the capture's held-out activations; adds the deciding "
                         "'out' column")
    ap.add_argument("--weights-only", action="store_true",
                    help="the matched control for --production: the capture is "
                         "loaded and every metric is scored against it, but the "
                         "encode is given no Hessian. Two treatments are not a "
                         "control -- the production and weights-only arms must "
                         "differ in the encode alone, or the objective flip "
                         "cannot be attributed to it")
    a = ap.parse_args()

    b = Bench(a.out)
    src = open_all(DENSE_SRC)
    act = None
    if a.production:
        act = ActivationSource.from_capture(a.production)
        hessians = dict(act.hessians)
        hsrc = (f"{a.production} (the capture the encode is fit on)"
                if not a.weights_only else
                f"{a.production} (scoring only -- the encode is given no Hessian)")
        if a.weights_only:
            act = None
    else:
        hessians = torch.load(DENSE_H)
        hsrc = f"{DENSE_H} (a DIFFERENT capture from any production encode)"
    xs, x_prov = {}, None
    if a.eval_x:
        payload = torch.load(a.eval_x, map_location="cpu", weights_only=False)
        xs, x_prov = payload["x"], payload.get("provenance")

    b.doc = {"args": vars(a), "units": {},
             "hessian_source": hsrc,
             "eval_x_source": a.eval_x,
             "eval_x_provenance": x_prov,
             "activation_aware": None if act is None else act.config_block(),
             "material_threshold_reach": MATERIAL,
             "deciding_axis": DECIDING}
    b.log(f"BF16 reach recipe: window_sigma per rung, channel_sigma pinned at "
          f"{BF16_CHANNEL_SIGMA} (a gauge).  The built arm names no spread.")
    b.log(f"  h from {hsrc}")
    if act is not None:
        b.log(f"  production encode: LDLQ sigma={act.ldlq_sigma} block={act.ldlq_block}, "
              f"refit objective {dict(act.refit_objective)}, "
              f"reach floor {act.refit_reach_floor}")
    else:
        b.log("  weights-only encode: no LDLQ, no metric-aware refit.  This is NOT "
              "the encode an export runs."
              + ("  Scored against the production capture: this is the matched "
                 "control, differing from the production run in the encode alone."
                 if a.production else ""))
    b.log(f"  deciding axis '{DECIDING}'"
          + ("" if a.eval_x else " -- NOT AVAILABLE, no --eval-x given")
          + f"; material at {MATERIAL:.4f}x in reach")

    for name in a.units:
        w = src[name + ".weight"].get_tensor(name + ".weight").cuda().float().contiguous()
        h = hessians[name].cuda().float()
        h = h.diagonal() if h.ndim == 2 else h
        kw: dict = {}
        if act is not None:
            kw = act.for_unit(name + ".weight", w.shape[1], device="cuda",
                              scale_plane=ScalePlaneKind.CHANNEL)
        x = y = ny = None
        if name in xs:
            x = xs[name].cuda().float()
            y = x @ w.T
            ny = float(y.norm())
        elif a.eval_x:
            b.log(f"  !! {name}: no held-out activations in {a.eval_x}; "
                  f"this unit cannot be decided")
        res: dict = {"rows": w.shape[0], "cols": w.shape[1], "has_out": x is not None}
        for q in a.rungs:
            recipe = wire_recipe(BF16_GRID, q)
            b.log(f"\n== {name} {tuple(w.shape)}  R={q / 256:g}  recipe "
                  f"L={recipe.window_bits} window_sigma={recipe.window_sigma:.6f}")
            b.header(("bpp", "wt", "h", "out", "reach_rms", "over"))
            arms = arms_for(q, a.extra_sigmas, a.grid_steps, a.gauge_twin)
            arms.append((arms[0][0] + " [repeat]", arms[0][1], arms[0][2]))
            for label, sigma, csigma in arms:
                key = f"R{q} {label}"
                table_sigma = recipe.window_sigma if sigma is None else sigma
                st = reach_stats(w, BF16_GRID, recipe.window_bits, table_sigma, csigma)
                started = time.time()
                try:
                    named = {} if sigma is None else {"window_sigma": float(sigma)}
                    exported, _u, _f = encode_linear_planes(
                        w, grid=BF16_GRID, q256=q, name=name,
                        channel_sigma=float(csigma), verify=True, **named, **kw)
                    hat = read_unit_artifact(exported.blob, device=w.device)
                except Exception as exc:                          # noqa: BLE001
                    torch.cuda.empty_cache()
                    b.log(f"    {key:<34} !! FAILED: {type(exc).__name__}: {exc}")
                    continue
                r = {"bpp": float(exported.bpp), "sha": sha(exported.blob),
                     "tsha": tensor_sha(hat), "secs": time.time() - started,
                     "window_sigma": table_sigma, "channel_sigma": csigma,
                     "reach_rms": st["reach_row_rms"], "over": st["rows_over_reach"],
                     **st, **score(w, hat, h=h, x=x, y=y, ny=ny)}
                res[key] = r
                b.row(key, r, ("bpp", "wt", "h", "out", "reach_rms", "over"))
                del hat
                torch.cuda.empty_cache()
            first, last = res.get(f"R{q} {arms[0][0]}"), res.get(f"R{q} {arms[-1][0]}")
            if first and last:
                res[f"R{q}_control"] = check_repeat_tensor(
                    b, first, last, f"R{q} {arms[0][0]}")
            if not first:
                continue
            bpps = {round(v["bpp"], 6) for k, v in res.items()
                    if isinstance(v, dict) and k.startswith(f"R{q} ")}
            res[f"R{q}_bytes"] = {"distinct_bpp": sorted(bpps),
                                  "identical": len(bpps) == 1}
            b.log(f"    bytes: {len(bpps)} distinct bpp across the rung's arms "
                  f"-> {'IDENTICAL' if len(bpps) == 1 else '!! DIFFER'}")
            res[f"R{q}_ratios"] = {
                k: {axis: v[axis] / first[axis] for axis in AXES if axis in v}
                for k, v in res.items()
                if isinstance(v, dict) and k.startswith(f"R{q} ") and "wt" in v}
            # The optimum as a number, on the arms that share the gauge: a
            # gauge-twin arm names the same reach with a different pair and
            # would appear twice on the reach axis.
            pts = [(v["reach_rms"], v) for k, v in res.items()
                   if isinstance(v, dict) and k.startswith(f"R{q} ")
                   and not k.endswith("[repeat]")
                   and abs(v["channel_sigma"] - BF16_CHANNEL_SIGMA) < 1e-12]
            law_reach = first["reach_rms"]
            opt = {axis: optimum_reach(pts, axis) for axis in AXES
                   if any(axis in v for _, v in pts)}
            for o in opt.values():
                if o.get("reach"):
                    o["vs_law"] = o["reach"] / law_reach
                    o["material"] = (not o["edge"]) and (
                        max(o["vs_law"], 1.0 / o["vs_law"]) > MATERIAL)
            res[f"R{q}_optimum"] = {"law_reach": law_reach, **opt}
            for axis, o in opt.items():
                mark = ("EDGE" if o["edge"] else
                        ("MATERIAL" if o.get("material") else "within tolerance"))
                b.log(f"    optimum[{axis}]: reach "
                      f"{(o['reach'] or float('nan')):.3f} vs law {law_reach:.3f} "
                      f"= {o.get('vs_law', float('nan')):.4f}x  {mark}")
        b.doc["units"][name] = res
        b.save()
        del w, x, y
        torch.cuda.empty_cache()

    units = b.doc["units"]
    summary = {}
    for q in a.rungs:
        labels = {k for u in units.values() for k in u
                  if isinstance(u[k], dict) and k.startswith(f"R{q} ")}
        summary[f"R{q}"] = {"arms": {}, "optimum": {}}
        for label in sorted(labels):
            for axis in AXES:
                vals = [u[f"R{q}_ratios"][label][axis] for u in units.values()
                        if f"R{q}_ratios" in u and label in u[f"R{q}_ratios"]
                        and axis in u[f"R{q}_ratios"][label]]
                if not vals:
                    continue
                row = summary[f"R{q}"]["arms"].setdefault(label, {})
                row[axis] = geomean(vals)
                row[f"{axis}_worse_than_built"] = sum(v > 1.0 for v in vals)
                row[f"{axis}_units"] = len(vals)
        for axis in AXES:
            got = [u for u in units.values()
                   if f"R{q}_optimum" in u and axis in u[f"R{q}_optimum"]
                   and u[f"R{q}_optimum"][axis].get("reach")]
            if not got:
                continue
            rs = [u[f"R{q}_optimum"][axis] for u in got]
            law_reach = got[0][f"R{q}_optimum"]["law_reach"]
            g = geomean([o["reach"] for o in rs])
            summary[f"R{q}"]["optimum"][axis] = {
                "reach_geomean": g, "law_reach": law_reach, "vs_law": g / law_reach,
                "units": len(rs), "edges": sum(1 for o in rs if o["edge"]),
                "per_unit": [o["reach"] for o in rs],
                "material": (max(g / law_reach, law_reach / g) > MATERIAL
                             and not any(o["edge"] for o in rs)),
            }
    b.doc["summary_vs_built"] = summary
    b.log("\n== geomean over the units, each arm against the built arm "
          "(>1 means the built recipe wins)")
    for rung, blk in summary.items():
        b.log(f"  {rung}")
        for label, v in blk["arms"].items():
            b.log(f"    {label:<34} " + "  ".join(
                f"{ax} {v[ax]:.4f} ({v[f'{ax}_worse_than_built']}/{v[f'{ax}_units']})"
                for ax in AXES if ax in v))
        for axis, o in blk["optimum"].items():
            b.log(f"    optimum[{axis}] geomean reach {o['reach_geomean']:.3f} "
                  f"vs law {o['law_reach']:.3f} = {o['vs_law']:.4f}x, "
                  f"{o['edges']} edge of {o['units']} -> "
                  f"{'MATERIAL' if o['material'] else 'within tolerance'}")
    b.save()
    b.log(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
