#!/usr/bin/env python
"""E4M3 ``channel_sigma``: the cliff, the gauge under it, and the coordinate (issue #36).

Issue #36 found, with ``bf16_l_sigma_sweep.py --stage gauge``, that on the
shipped E4M3 wire (q1024, L=14, CHANNEL) halving or quartering
``channel_sigma`` is free to five digits while doubling costs 5-19% and
quadrupling 70-98%: a one-sided cliff, with the default at its edge and all of
its margin below.  It also found that ``scale_channel._default_sigma``'s
dyadic ladder cannot see the axis, because every dyadic step below the default
decodes to (nearly) the same tensor.  Both observations are correct and both
are read here as symptoms of one thing: **``channel_sigma`` is not the
coordinate the loss depends on.**

**The mechanism, stated before it is measured.**  On the E4M3 recipe
``window_sigma`` is unset, so the window table is built at ``sigma =
channel_sigma`` (``encode_unit``'s CHANNEL branch).  The table's entries are
the equal-mass quantiles of a Gaussian at that spread, snapped to E4M3; its
extreme entry, the body's **reach**, is ``snap(z_max(L) * sigma)`` capped at
the format's peak 448, with ``z_max(14) = 4.05``.  Rows are scaled to
``channel_sigma`` grid units of RMS.  So the reach **in row-RMS units** is

    r(sigma) = min(z_max(L) * sigma, 448) / sigma = min(z_max(L), 448 / sigma)

and every other property of the encode is scale-free, because E4M3 is
log-spaced: a value's relative snap error is the same in every normal binade,
and the grid is closed under x2 inside the normal range.  Two consequences,
each a prediction this harness checks:

* **Below the ceiling** ``sigma_hi = 448 / z_max(L)`` (110.6 at L=14) a
  dyadic shift of ``sigma`` is an exact gauge -- the same codes, the same
  reconstruction -- until the table's smallest entries fall through E4M3's
  subnormal floor (``2^-9``), which costs nothing measurable because those
  entries are ~1e-4 of a row's RMS.  A non-dyadic shift changes which
  mantissa phase each quantile snaps to (second order).  **The flat side of
  the cliff is not a flat loss; it is a degenerate coordinate.**
* **Above the ceiling** the table's top pins at 448 and ``r`` falls as
  ``448 / sigma``: x1.5 is ``r = 3.17``, x2 is ``r = 2.38``, x4 is ``r =
  1.19``.  That is the same reach axis issue #18 swept on BF16 by pinning the
  table and moving the rows, traversed here in the direction the format
  permits -- and only that direction.  **The steep side of the cliff is the
  low-reach wall of the reach bowl.**

So the coordinate is ``r``, the reach in row-RMS, and on this wire it is
reachable only two ways: ``sigma`` above the ceiling (downward in ``r``, with
the table's top entries collapsing onto 448 as a side effect), or a pinned
table with ``channel_sigma`` lowered (upward or downward in ``r``, table
untouched).  The ladder searches neither: it walks ``sigma`` on the gauge
orbit and is choosing between equals.

Stages::

    P=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
    E="TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache PYTHONPATH=src:experiments"
    OUT=/mnt/shared/tessera-runs/e4m3/reach_cliff

    # 1. The issue's curve at quarter-octave resolution, both walls in view.
    #    window_sigma tracks channel_sigma, as the wire ships.
    env $E $P experiments/e4m3_reach_cliff.py --stage tracked --out $OUT/tracked.json

    # 2. The reach axis: the shipped table pinned, rows moved.  Weights-only
    #    (the issue's instrument) and the production encode (what ships).
    env $E $P experiments/e4m3_reach_cliff.py --stage reach --out $OUT/reach_wo.json
    env $E $P experiments/e4m3_reach_cliff.py --stage reach --out $OUT/reach_prod.json \\
        --production /mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt \\
        --eval-x /mnt/shared/tessera-runs/bf16/refs/x_eval_dense4.pt

    # 3. The overlay: every tracked arm above the ceiling, re-run with the
    #    table pinned at the same r.  If r is the coordinate the two agree.
    env $E $P experiments/e4m3_reach_cliff.py --stage overlay --out $OUT/overlay.json

**The controls.**  Every arm of a stage runs in one process in a fixed order;
the shipped arm (``window_sigma = channel_sigma = default_channel_sigma``,
spelled explicitly, which ``encode_unit`` resolves an unset spread to) runs
first and is repeated last, asserted byte- and tensor-identical.  Every arm
inside a (unit, rung) writes the same number of bytes -- a CHANNEL plane is
one fp16 per row and the table is ``2^L`` bytes whatever its spread -- and
the harness checks that rather than assuming it.  ``wt`` is the plain
relative Frobenius error, ``h`` the diagonal-H-weighted one, ``out`` the
output-space error on the capture's **held-out** rows (production stage
only); ``out`` decides where it exists, as in ``bf16_reach_recipe.py``.

Weight space and output space on four dense Qwen3-0.6B Linears.  No serve.
Principle 3: nothing here promotes anything; the E4M3 wire's served receipt
is ``docs/measurements/tessera-dense-reach-fix-2026-09-02.md`` and a change
to its reach would need the same instrument.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import E4M3_GRID, GAUSSIAN_SOURCE                # noqa: E402
from tessera.encode import grid_vector_table, window_table            # noqa: E402
from tessera.export import (                                          # noqa: E402
    E4M3_WINDOW_BITS,
    ActivationSource,
    encode_linear_planes,
)
from tessera.manifest import ScalePlaneKind                           # noqa: E402
from tessera.scale_channel import default_channel_sigma               # noqa: E402
from tessera.unit_artifact import read_unit_artifact                  # noqa: E402

from bf16_l_sigma_sweep import (                                      # noqa: E402
    DENSE_UNITS,
    Bench,
    check_repeat_tensor,
    score,
    sha,
    tensor_sha,
)
from bf16_route_weight_space import (                                 # noqa: E402
    DENSE_H, DENSE_SRC, GLM_ACT, GLM_SRC, geomean, open_all)

#: The format's peak.  ``0x7E`` is 448; ``0x7F`` is NaN in E4M3FN and carries
#: its neighbour's value on the grid, which is why the table can pin there.
E4M3_PEAK = max(abs(v) for v in E4M3_GRID.values)
#: The smallest positive E4M3 value, ``2^-9``: below half of it a table entry
#: snaps to zero.
E4M3_MIN_SUBNORMAL = min(abs(v) for v in E4M3_GRID.values if v != 0.0)

AXES = ("wt", "h", "out")


def z_max(window_bits: int) -> float:
    """The largest |quantile| the table models at width ``L``, in sigma.

    ``GAUSSIAN_SOURCE(2^L, 1.0)`` is the equal-mass sample the table is built
    from, so this is exactly its extreme entry, not a formula for it.
    """
    return max(abs(z) for z in GAUSSIAN_SOURCE(1 << window_bits, 1.0))


def ceiling_sigma(window_bits: int) -> float:
    """The spread past which the table's top entry pins at the peak: ``448 / z_max(L)``."""
    return E4M3_PEAK / z_max(window_bits)


def table_stats(w: torch.Tensor, window_bits: int, table_sigma: float, channel_sigma: float) -> dict:
    """What the table looks like and how far it reaches, in this unit's rows."""
    codes = window_table(E4M3_GRID, window_bits, sigma=table_sigma, seed=0, half=16,
                         device=w.device)
    vals = grid_vector_table(E4M3_GRID, w.device)[codes.long()].abs().flatten()
    reach = float(vals.max())
    rms = w.float().pow(2).mean(dim=1).sqrt()
    amax = w.float().abs().amax(dim=1)
    over = amax * channel_sigma > reach * rms
    nonzero = vals[vals > 0]
    return {
        "reach_grid_units": reach,
        "reach_rms": reach / channel_sigma,
        "over": float(over.float().mean()),
        "saturated": float((vals >= E4M3_PEAK).float().mean()),
        "zero": float((vals == 0).float().mean()),
        "subnormal": float((vals < 2.0 ** -6).float().mean()),
        "distinct": int(torch.unique(codes).numel()),
        "min_nonzero_rms": (float(nonzero.min()) / channel_sigma) if nonzero.numel() else None,
    }


def optimum_reach(points, axis: str) -> dict:
    """The reach minimising ``axis``, refined off a geometric grid.

    Copied from ``bf16_reach_recipe.py`` (issue #48, branch
    ``claude/ts-48-bf16-reach``) so this harness runs on ``master``: a
    parabola through the discrete minimum and its neighbours in ``(log reach,
    log metric)``; a minimum at either end of the grid is reported as an edge
    rather than extrapolated.
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
    d1, d2 = (ya - yb) / (xa - xb), (yb - yc) / (xb - xc)
    curv = (d1 - d2) / (xa - xc)
    if curv <= 0:
        return {"reach": pts[i][0], "edge": True, "why": "not convex at the minimum",
                "grid": [p[0] for p in pts]}
    return {"reach": math.exp(0.5 * (xa + xb - d1 / curv)), "edge": False,
            "grid_reach": pts[i][0], "grid": [p[0] for p in pts]}


class Runner:
    """One unit's encodes, scored the same way whatever the stage."""

    def __init__(self, b: Bench, a, name: str, w, h, x, kw: dict):
        self.b, self.a, self.name = b, a, name
        self.w, self.h, self.kw = w, h, kw
        self.x = self.y = self.ny = None
        if x is not None:
            self.x = x
            self.y = self.x @ self.w.T
            self.ny = float(self.y.norm())

    def arm(self, key: str, q256: int, window_sigma: float, channel_sigma: float) -> "dict | None":
        st = table_stats(self.w, self.a.window_bits, window_sigma, channel_sigma)
        started = time.time()
        try:
            exported, _u, _f = encode_linear_planes(
                self.w, grid=E4M3_GRID, q256=q256, name=self.name,
                window_bits=self.a.window_bits, window_sigma=float(window_sigma),
                channel_sigma=float(channel_sigma), verify=True, **self.kw)
            hat = read_unit_artifact(exported.blob, device=self.w.device)
        except Exception as exc:                                  # noqa: BLE001
            torch.cuda.empty_cache()
            self.b.log(f"    {key:<36} !! FAILED: {type(exc).__name__}: {exc}")
            return None
        r = {"bpp": float(exported.bpp), "sha": sha(exported.blob),
             "tsha": tensor_sha(hat), "secs": time.time() - started,
             "window_sigma": float(window_sigma), "channel_sigma": float(channel_sigma),
             **st, **score(self.w, hat, h=self.h, x=self.x, y=self.y, ny=self.ny)}
        del hat
        torch.cuda.empty_cache()
        return r

    def close(self) -> None:
        del self.w, self.x, self.y
        torch.cuda.empty_cache()


COLS = ("bpp", "wt", "h", "out", "reach_rms", "over", "saturated")


def bytes_identical(res: dict, prefix: str) -> dict:
    bpps = {round(v["bpp"], 6) for k, v in res.items()
            if isinstance(v, dict) and k.startswith(prefix) and "bpp" in v}
    return {"distinct_bpp": sorted(bpps), "identical": len(bpps) == 1}


def load_inputs(a):
    """The units as ``(name, w, h, x, encode kwargs)`` in order: dense Qwen3-0.6B
    by default; ``--glm-layers`` swaps in GLM-5.3-Flash expert-0 Linears with
    the last ``--glm-eval-rows`` captured input rows as ``x`` (a screen on
    capture rows, not a held-out split; there is no production H for GLM
    here, so the GLM arms are weights-only)."""
    act = None
    if a.glm_layers:
        if a.production or a.eval_x:
            raise SystemExit("--glm-layers is weights-only: no --production / --eval-x")
        index = json.load(open(f"{GLM_SRC}/model.safetensors.index.json"))["weight_map"]
        units = []
        for layer in a.glm_layers:
            blob = torch.load(f"{GLM_ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                              map_location="cpu", weights_only=False)
            xa = blob["inputs"].float()
            x = xa[xa.shape[0] - a.glm_eval_rows:].contiguous().cuda()
            del xa, blob
            h = (x * x).sum(dim=0)
            for proj in a.glm_projs:
                name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
                with safe_open(f"{GLM_SRC}/{index[name]}", framework="pt") as f:
                    w = f.get_tensor(name).contiguous().cuda().float()
                units.append((f"L{layer}.{proj}", w, h, x, {}))
        hsrc = f"{GLM_ACT} (x^T x of the last {a.glm_eval_rows} captured rows; GLM arms are weights-only)"
        return units, hsrc, {"source": GLM_ACT, "rows": a.glm_eval_rows,
                             "held_out": False, "note": "capture rows, a screen"}, None
    src = open_all(DENSE_SRC)
    if a.production:
        act = ActivationSource.from_capture(a.production)
        hessians = dict(act.hessians)
        hsrc = f"{a.production} (the capture the encode is fit on)"
    else:
        hessians = torch.load(DENSE_H)
        hsrc = f"{DENSE_H} (weights-only encode; h scored against the stock census diagonal)"
    xs, x_prov = {}, None
    if a.eval_x:
        payload = torch.load(a.eval_x, map_location="cpu", weights_only=False)
        xs, x_prov = payload["x"], payload.get("provenance")
    units = []
    for name in a.units:
        w = src[name + ".weight"].get_tensor(name + ".weight").cuda().float().contiguous()
        h = hessians[name].cuda().float()
        h = h.diagonal() if h.ndim == 2 else h
        kw = {}
        if act is not None:
            kw = act.for_unit(name + ".weight", w.shape[1], device="cuda",
                              scale_plane=ScalePlaneKind.CHANNEL)
        x = xs[name].cuda().float() if name in xs else None
        units.append((name, w, h, x, kw))
    return units, hsrc, x_prov, act


def preamble(b: Bench, a, hsrc, x_prov, act) -> None:
    sigma0 = default_channel_sigma(E4M3_GRID)
    zc = z_max(a.window_bits)
    b.doc.update({
        "args": vars(a), "units": {},
        "sigma0": sigma0, "z_max": zc, "ceiling_sigma": ceiling_sigma(a.window_bits),
        "shipped_reach_rms": table_stats(torch.zeros(1, 1, device="cuda"), a.window_bits,
                                         sigma0, sigma0)["reach_rms"],
        "hessian_source": hsrc, "eval_x_source": a.eval_x, "eval_x_provenance": x_prov,
        "activation_aware": None if act is None else act.config_block(),
    })
    b.log(f"E4M3 L={a.window_bits}: default_channel_sigma = {sigma0:.4f}; z_max = {zc:.4f}; "
          f"ceiling 448/z_max = {b.doc['ceiling_sigma']:.4f} "
          f"(default is {b.doc['ceiling_sigma'] / sigma0:.4f}x = "
          f"{math.log2(b.doc['ceiling_sigma'] / sigma0):.3f} binades below it); "
          f"shipped reach {b.doc['shipped_reach_rms']:.4f} row-RMS")
    b.log(f"  h from {hsrc}")
    if act is not None:
        b.log(f"  production encode: LDLQ sigma={act.ldlq_sigma} block={act.ldlq_block}, "
              f"refit objective {dict(act.refit_objective)}, reach floor {act.refit_reach_floor}")
    else:
        b.log("  weights-only encode: no LDLQ, no metric-aware refit -- the issue's "
              "instrument, NOT the encode an export runs")


# ---------------------------------------------------------------- tracked


def stage_tracked(a) -> None:
    """The issue's curve, finely: ``window_sigma = channel_sigma = sigma0 * 2^(k/4)``."""
    b = Bench(a.out)
    units, hsrc, x_prov, act = load_inputs(a)
    preamble(b, a, hsrc, x_prov, act)
    sigma0 = b.doc["sigma0"]
    ks = list(range(a.k_lo, a.k_hi + 1))
    for name, w, h, x, kw in units:
        u = Runner(b, a, name, w, h, x, kw)
        res: dict = {"rows": u.w.shape[0], "cols": u.w.shape[1]}
        q = a.rung
        b.log(f"\n== {name} {tuple(u.w.shape)}  R={q / 256:g}  tracked: window_sigma = channel_sigma")
        b.header(COLS)
        arms = [(0, "shipped")] + [(k, f"2^({k}/4)") for k in ks if k != 0] + [(0, "shipped [repeat]")]
        for k, label in arms:
            s = sigma0 * 2.0 ** (k / 4.0)
            key = f"R{q} k={k:+d} {label} s={s:.4g}"
            r = u.arm(key, q, s, s)
            if r is None:
                continue
            r["k"] = k
            res[key] = r
            b.row(key, r, COLS)
        first = res.get(f"R{q} k=+0 shipped s={sigma0:.4g}")
        last = res.get(f"R{q} k=+0 shipped [repeat] s={sigma0:.4g}")
        if first and last:
            res["control"] = check_repeat_tensor(b, first, last, "shipped")
        res["bytes"] = bytes_identical(res, f"R{q} ")
        b.log(f"    bytes: {res['bytes']}")
        if first:
            res["gauge"] = {
                "tensor_identical_to_shipped": sorted(
                    k for k, v in res.items()
                    if isinstance(v, dict) and v.get("tsha") == first["tsha"]),
                "ratios": {k: {ax: v[ax] / first[ax] for ax in AXES if ax in v and ax in first}
                           for k, v in res.items()
                           if isinstance(v, dict) and "k" in v},
            }
            b.log(f"    tensor-identical to shipped: {len(res['gauge']['tensor_identical_to_shipped'])} "
                  f"of {len(arms)} arms")
        b.doc["units"][name] = res
        b.save()
        u.close()
    summarise(b, "k")
    b.save()
    b.log(f"\nwrote {a.out}")


# ------------------------------------------------------------------ reach


def reach_arms(a, sigma0: float):
    """``(label, window_sigma, channel_sigma)``: the table pinned at sigma0, rows moved."""
    arms = [("shipped (r=4.08)", sigma0, sigma0)]
    for k in range(a.r_lo, a.r_hi + 1):
        if k == 0:
            continue
        cs = sigma0 * 2.0 ** (-k / 4.0)
        arms.append((f"r*2^({k:+d}/4)", sigma0, cs))
    for r in a.extra_reach:
        # An explicit reach in row-RMS: the shipped table's top is 384, so
        # channel_sigma = 384 / r puts the rows exactly there.
        arms.append((f"r={r:g}", sigma0, 384.0 / float(r)))
    return arms


def stage_reach(a) -> None:
    b = Bench(a.out)
    units, hsrc, x_prov, act = load_inputs(a)
    preamble(b, a, hsrc, x_prov, act)
    sigma0 = b.doc["sigma0"]
    for name, w, h, x, kw in units:
        u = Runner(b, a, name, w, h, x, kw)
        res: dict = {"rows": u.w.shape[0], "cols": u.w.shape[1], "has_out": u.x is not None}
        for q in a.rungs:
            b.log(f"\n== {name} {tuple(u.w.shape)}  R={q / 256:g}  table pinned at sigma0={sigma0:.4f}")
            b.header(COLS)
            arms = reach_arms(a, sigma0)
            arms.append((arms[0][0] + " [repeat]", arms[0][1], arms[0][2]))
            for label, ws, cs in arms:
                key = f"R{q} {label} cs={cs:.4g}"
                r = u.arm(key, q, ws, cs)
                if r is None:
                    continue
                res[key] = r
                b.row(key, r, COLS)
            first = res.get(f"R{q} {arms[0][0]} cs={sigma0:.4g}")
            last = res.get(f"R{q} {arms[-1][0]} cs={sigma0:.4g}")
            if first and last:
                res[f"R{q}_control"] = check_repeat_tensor(b, first, last, f"R{q} shipped")
            res[f"R{q}_bytes"] = bytes_identical(res, f"R{q} ")
            b.log(f"    bytes: {res[f'R{q}_bytes']}")
            if not first:
                continue
            res[f"R{q}_ratios"] = {
                k: {ax: v[ax] / first[ax] for ax in AXES if ax in v and ax in first}
                for k, v in res.items()
                if isinstance(v, dict) and k.startswith(f"R{q} ") and "wt" in v}
            pts = [(v["reach_rms"], v) for k, v in res.items()
                   if isinstance(v, dict) and k.startswith(f"R{q} ") and not k.endswith(f"[repeat] cs={sigma0:.4g}")]
            opt = {ax: optimum_reach(pts, ax) for ax in AXES if any(ax in v for _, v in pts)}
            for ax, o in opt.items():
                if o.get("reach"):
                    o["vs_shipped"] = o["reach"] / first["reach_rms"]
                b.log(f"    optimum[{ax}]: reach {(o['reach'] or float('nan')):.3f} vs shipped "
                      f"{first['reach_rms']:.3f} = {o.get('vs_shipped', float('nan')):.4f}x"
                      f"{'  EDGE: ' + o['why'] if o['edge'] else ''}")
            res[f"R{q}_optimum"] = {"shipped_reach": first["reach_rms"], **opt}
        b.doc["units"][name] = res
        b.save()
        u.close()
    summarise(b, "reach")
    # The rung law: does the optimum reach move as sqrt(R / 4)?
    law = {}
    for q in a.rungs:
        R = q / 256.0
        for ax in AXES:
            got = [u[f"R{q}_optimum"][ax] for u in b.doc["units"].values()
                   if f"R{q}_optimum" in u and ax in u[f"R{q}_optimum"]
                   and u[f"R{q}_optimum"][ax].get("reach")]
            if not got:
                continue
            g = geomean([o["reach"] for o in got])
            law.setdefault(f"R{q}", {})[ax] = {
                "reach_geomean": g, "per_unit": [o["reach"] for o in got],
                "edges": sum(1 for o in got if o["edge"]),
                "sqrt_law_from_shipped": b.doc["shipped_reach_rms"] * math.sqrt(max(R, 4) / 4),
            }
    b.doc["rung_law"] = law
    b.log("\n== optimum reach per rung (geomean over units) against sqrt(R/4) from the shipped 4.08")
    for rung, blk in law.items():
        for ax, v in blk.items():
            b.log(f"  {rung} [{ax}] optimum {v['reach_geomean']:.3f}  sqrt-law {v['sqrt_law_from_shipped']:.3f}  "
                  f"ratio {v['reach_geomean'] / v['sqrt_law_from_shipped']:.4f}  edges {v['edges']}  "
                  f"per-unit {[round(x, 2) for x in v['per_unit']]}")
    b.save()
    b.log(f"\nwrote {a.out}")


# ---------------------------------------------------------------- overlay


def stage_overlay(a) -> None:
    """Each tracked arm above the ceiling, beside the pinned-table arm at the same ``r``.

    Tracked at ``sigma = sigma0 * m`` (``m > ceiling/sigma0``) the reach is
    ``448 / sigma`` row-RMS; the pinned table's top is 384, so ``channel_sigma
    = 384 * sigma / 448`` puts the rows at the same ``r`` with the shipped
    table.  If ``r`` is the coordinate the two arms agree up to the second-order
    effects the tracked arm carries (a re-snapped table, its top entries
    collapsed onto 448).
    """
    b = Bench(a.out)
    units, hsrc, x_prov, act = load_inputs(a)
    preamble(b, a, hsrc, x_prov, act)
    sigma0 = b.doc["sigma0"]
    top0 = table_stats(torch.zeros(1, 1, device="cuda"), a.window_bits, sigma0, sigma0)["reach_grid_units"]
    q = a.rung
    for name, w, h, x, kw in units:
        u = Runner(b, a, name, w, h, x, kw)
        res: dict = {"rows": u.w.shape[0], "cols": u.w.shape[1]}
        b.log(f"\n== {name} {tuple(u.w.shape)}  R={q / 256:g}  overlay: tracked vs pinned at equal r")
        b.header(COLS)
        pairs = []
        first = u.arm(f"R{q} shipped", q, sigma0, sigma0)
        if first:
            res[f"R{q} shipped"] = first
            b.row(f"R{q} shipped", first, COLS)
        for m in a.multipliers:
            s = sigma0 * m
            tracked = u.arm(f"R{q} tracked x{m:g}", q, s, s)
            if tracked is None:
                continue
            res[f"R{q} tracked x{m:g}"] = tracked
            b.row(f"R{q} tracked x{m:g}", tracked, COLS)
            cs = top0 / tracked["reach_rms"]
            pinned = u.arm(f"R{q} pinned r={tracked['reach_rms']:.3f}", q, sigma0, cs)
            if pinned is None:
                continue
            res[f"R{q} pinned r={tracked['reach_rms']:.3f}"] = pinned
            b.row(f"R{q} pinned r={tracked['reach_rms']:.3f}", pinned, COLS)
            pairs.append({"multiplier": m, "reach_rms": tracked["reach_rms"],
                          "tracked_over": tracked["over"], "pinned_over": pinned["over"],
                          **{f"{ax}_tracked_over_pinned": tracked[ax] / pinned[ax]
                             for ax in AXES if ax in tracked and ax in pinned}})
            b.log(f"    x{m:g}: r={tracked['reach_rms']:.3f}  tracked/pinned  "
                  + "  ".join(f"{ax} {pairs[-1][f'{ax}_tracked_over_pinned']:.4f}"
                              for ax in AXES if f"{ax}_tracked_over_pinned" in pairs[-1]))
        last = u.arm(f"R{q} shipped [repeat]", q, sigma0, sigma0)
        if first and last:
            res[f"R{q} shipped [repeat]"] = last
            res["control"] = check_repeat_tensor(b, first, last, "shipped")
        res["pairs"] = pairs
        res["bytes"] = bytes_identical(res, f"R{q} ")
        b.doc["units"][name] = res
        b.save()
        u.close()
    b.doc["summary"] = {}
    ms = sorted({p["multiplier"] for u in b.doc["units"].values() for p in u.get("pairs", [])})
    b.log("\n== tracked / pinned at equal r, geomean over units (1.0000 = r is the whole story)")
    for m in ms:
        row = {}
        for ax in AXES:
            vals = [p[f"{ax}_tracked_over_pinned"] for u in b.doc["units"].values()
                    for p in u.get("pairs", []) if p["multiplier"] == m and f"{ax}_tracked_over_pinned" in p]
            if vals:
                row[ax] = geomean(vals)
                row[f"{ax}_n"] = len(vals)
        b.doc["summary"][f"x{m:g}"] = row
        b.log(f"  x{m:g}: " + "  ".join(f"{ax} {row[ax]:.4f} (n={row[f'{ax}_n']})" for ax in AXES if ax in row))
    b.save()
    b.log(f"\nwrote {a.out}")


def summarise(b: Bench, axis_name: str) -> None:
    arms: "dict[str, list]" = {}
    for res in b.doc["units"].values():
        for arm, r in res.items():
            if isinstance(r, dict) and "bpp" in r and "[repeat]" not in arm:
                arms.setdefault(arm.split(" cs=")[0].split(" s=")[0], []).append(r)
    n = len(b.doc["units"])
    b.doc["summary"] = {}
    b.log(f"\n== geomean over {n} unit(s), swept on {axis_name}")
    b.log("    " + f"{'arm':<36}" + "".join(f"{k:>10}" for k in ("reach", "over", "wt", "h", "out")))
    for arm, rs in arms.items():
        if len(rs) != n:
            continue
        row = {"reach_rms": geomean([r["reach_rms"] for r in rs]),
               "over": sum(r["over"] for r in rs) / n}
        for k in AXES:
            if all(k in r for r in rs):
                row[k] = geomean([r[k] for r in rs])
        b.doc["summary"][arm] = row
        b.log("    " + f"{arm:<36}" + f"{row['reach_rms']:10.3f}{row['over']:10.3f}"
              + "".join(f"{row.get(k, float('nan')):10.5f}" for k in AXES))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["tracked", "reach", "overlay"])
    ap.add_argument("--units", nargs="+", default=DENSE_UNITS)
    ap.add_argument("--rung", type=int, default=1024)
    ap.add_argument("--rungs", type=int, nargs="+", default=[1024, 1280, 1536])
    ap.add_argument("--window-bits", type=int, default=E4M3_WINDOW_BITS)
    # tracked: sigma0 * 2^(k/4) for k in [k_lo, k_hi]
    ap.add_argument("--k-lo", type=int, default=-40)
    ap.add_argument("--k-hi", type=int, default=12)
    # reach: channel_sigma = sigma0 * 2^(-k/4), i.e. r = 4.08 * 2^(k/4)
    ap.add_argument("--r-lo", type=int, default=-6)
    ap.add_argument("--r-hi", type=int, default=8)
    ap.add_argument("--extra-reach", type=float, nargs="*", default=[])
    ap.add_argument("--multipliers", type=float, nargs="+",
                    default=[1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0])
    ap.add_argument("--glm-layers", type=int, nargs="*", default=[],
                    help="GLM-5.3-Flash expert-0 layers instead of the dense units (weights-only)")
    ap.add_argument("--glm-projs", nargs="+", default=["down_proj", "gate_proj"])
    ap.add_argument("--glm-eval-rows", type=int, default=8192)
    ap.add_argument("--production", default=None, metavar="CAPTURE")
    ap.add_argument("--eval-x", default=None, metavar="ACTS")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    {"tracked": stage_tracked, "reach": stage_reach, "overlay": stage_overlay}[a.stage](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
