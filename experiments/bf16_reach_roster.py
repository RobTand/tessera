#!/usr/bin/env python
"""The BF16 reach, searched on a roster and decomposed per row (issues #18, #54).

Two issues ask one question from two ends.  #18: the BF16 recipe's spread was
stated, not searched, and the evidence is a handful of units.  #54: under the
encode an export runs, the reach optimum is **per unit** -- 0.5x to 1.5x of
the ``sqrt(R/4)`` rate law on four dense Qwen Linears -- and the law regresses
``down_proj`` 2-23% on ``out``.  Four units cannot say whether that spread is
a property of the unit, of its role, or of something *inside* the unit that a
per-unit number only averages.  This harness answers on a roster, and keeps
the per-row error of every arm so the last question can be asked of the same
encodes.

**What "reach" is, and why a per-row reach is already on the wire.**  With a
CHANNEL plane every row is scaled to ``channel_sigma`` grid units of RMS and
the window table is ``2^L`` Gaussian quantiles at spread ``window_sigma``, so
``window_sigma / channel_sigma`` is how many row-RMS the table's last entry
reaches.  Encoding a row at scale ``s`` against the table at spread ``sigma``
is the same trellis problem as encoding it at scale ``s * sigma`` against the
table at spread 1 (the table is ``sigma`` times the standard quantiles, up to
its bf16 snap -- #18 measured that orbit at 0.02%).  So the reach is a *row
scale rule*, and the CHANNEL plane already carries one free fp16 word per row:
a per-row reach costs no wire.  Rows the reach start clamps
(``initial_channel_scale``: ``amax * channel_sigma > reach * rms``) have
``s * sigma = amax / q_max`` whatever ``sigma`` is -- the spread is a gauge on
them -- and only the unclamped rows move when the spread moves.  That is the
mechanism the per-row decomposition reads.

**What is measured.**  Every arm of a (unit, rung) is encoded through
``ActivationSource`` at its shipping defaults (LDLQ sigma=1 block=32 plus the
exact full-H row-scale refit) -- the path ``export_checkpoint`` takes -- and
scored on **held-out** rows: ``out`` is ``|X_eval E^T| / |X_eval W^T|`` and is
computed as ``tr(E H_eval E^T)``, which is the same number exactly and needs
one ``[cols, cols]`` matrix per unit instead of an activation dump.  ``H_eval``
comes from the same capture's eval slice (``capture_h_full.py --eval-h-out``),
disjoint from the fit rows the encode's Hessian was accumulated on; where the
activation dump exists (four units) the two are cross-checked.  Per row,
``out`` is ``e_r H_eval e_r^T`` and ``wt`` is ``|e_r|^2``, and both are stored
for every arm so the per-row optimum is read from the encodes that were
actually run, not modelled.  ``out`` decides; ``wt`` and the fit-diagonal ``h``
are continuity with the earlier receipts.

**The reach term is ts-48's, not a second one.**  The arms are a
quarter-octave grid around the rule ``window_sigma(R) = sqrt(max(R, 4) / 4)``
(branch ``claude/ts-48-bf16-reach``, ``export._window_sigma_for``).  When that
branch is on the tree ``wire_recipe`` names the value and it is read from
there; on a tree without it the same formula is applied and the log says so.
Either way the ``law`` arm is the same spread, and the ``pinned`` arm is the
old wire (``window_sigma = channel_sigma = 1.0``).

**Distribution.**  The search assumes nothing about the rows: every arm is an
encode and every number is measured.  The *table* assumes a Gaussian source
(its entries are Gaussian quantiles), and the rows are not one -- their
kurtosis is recorded per row so that gap is visible next to the optimum.

**Controls.**  One process per run, fixed arm order, the law arm first and
repeated last and asserted byte- and tensor-identical (the encoder is
deterministic, so this pins that no arm leaked state -- the LDL factor is
built once per unit and shared).  Every arm of a (unit, rung) writes the same
number of bytes, asserted.  ``--weights-only`` is the matched control: same
capture loaded, same scorer, the encode given no Hessian.

Encode-side screen.  **No serve.**  Principle 3: nothing here promotes.

Run (dense roster, on sparklina)::

    P=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
    E="TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache PYTHONPATH=src:experiments"
    env $E $P experiments/bf16_reach_roster.py --out OUT/roster_a.json --rows-dir OUT/rows \\
        --production /mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt \\
        --eval-h /mnt/shared/tessera-runs/bf16/refs/h_eval_qwen06b.pt \\
        --layers 0 4 8 12 --rungs 1024 1792 2048

    # GLM experts: H fit on the capture's first rows, scored on its last 1024
    env $E $P experiments/bf16_reach_roster.py --source glm --out OUT/roster_glm.json \\
        --rows-dir OUT/rows --layers 5 20 42 --projs gate_proj up_proj --rungs 1024 2048
"""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import BF16_GRID                              # noqa: E402
from tessera.export import (                                        # noqa: E402
    BF16_CHANNEL_SIGMA,
    ActivationSource,
    encode_linear_planes,
    wire_recipe,
)
from tessera.manifest import ScalePlaneKind                         # noqa: E402
from tessera.unit_artifact import read_unit_artifact                # noqa: E402

from bf16_l_sigma_sweep import (                                    # noqa: E402
    Bench,
    check_repeat_tensor,
    reach_stats,
    sha,
    tensor_sha,
)
from bf16_route_weight_space import (                               # noqa: E402
    DENSE_SRC,
    GLM_ACT,
    GLM_SRC,
    geomean,
    open_all,
)

#: The seven roles of a dense Qwen3 layer, in the order the table prints them.
DENSE_ROLES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
               "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")

#: Every fourth layer plus the last: eight depths of the 28, times seven roles.
DENSE_LAYERS = (0, 4, 8, 12, 16, 20, 24, 27)

#: The reach error the rate law already accepts by rounding a rung's rate to
#: whole bits (ts-48 ``export._reach_rate_for``): the threshold at which a
#: disagreement between the law and a measured optimum is material.
MATERIAL = math.sqrt(4.5 / 4.0)

AXES = ("wt", "h", "out")
DECIDING = "out"

#: ts-48's rule, spelled once here for a tree that does not carry it.
REACH_REFERENCE_RATE = 4


def law_sigma(q256: int) -> "tuple[float, str]":
    """ts-48's ``window_sigma(R)``: from ``wire_recipe`` when the tree has it."""
    named = wire_recipe(BF16_GRID, q256).window_sigma
    if named is not None:
        return float(named), "wire_recipe (ts-48 term on this tree)"
    exact = int(q256) * BF16_GRID.arity / 256.0
    rate = max(1, min(BF16_GRID.payload_bits, int(math.floor(exact + 0.5))))
    rate = max(REACH_REFERENCE_RATE, rate)
    return (float(BF16_CHANNEL_SIGMA) * math.sqrt(rate / REACH_REFERENCE_RATE),
            "sqrt(max(R,4)/4) -- ts-48's _window_sigma_for applied by hand; "
            "this tree's wire_recipe leaves window_sigma unset")


def arms_for(q256: int, steps: int) -> "list[tuple[str, float]]":
    """``(label, window_sigma)``: the law first, the old wire, then the grid."""
    law, _ = law_sigma(q256)
    arms = [(f"law s={law:.4g}", law)]
    seen = {round(law, 9)}

    def add(label, sigma):
        if round(float(sigma), 9) in seen:
            return
        seen.add(round(float(sigma), 9))
        arms.append((label, float(sigma)))

    add(f"pinned 1.0 (old wire)", 1.0)
    for k in range(-steps, steps + 1):
        if k:
            add(f"law*2^({k}/4) s={law * 2.0 ** (k / 4.0):.4g}", law * 2.0 ** (k / 4.0))
    return arms


def optimum_reach(points, key) -> dict:
    """The reach minimising ``key``, refined by a log-log parabola through the
    grid minimum and its neighbours; an argmin on the grid's edge is reported
    as an edge and not refined (a parabola through an edge extrapolates).
    Same construction as ts-48's ``bf16_reach_recipe.optimum_reach``."""
    pts = sorted((r, m) for r, m in points if r > 0 and m > 0)
    if len(pts) < 3:
        return {"reach": None, "edge": True, "why": "fewer than three arms"}
    i = min(range(len(pts)), key=lambda j: pts[j][1])
    if i in (0, len(pts) - 1):
        return {"reach": pts[i][0], "edge": True,
                "why": "argmin at the low edge" if i == 0 else "argmin at the high edge"}
    (xa, ya), (xb, yb), (xc, yc) = (
        (math.log(pts[j][0]), math.log(pts[j][1])) for j in (i - 1, i, i + 1))
    d1, d2 = (ya - yb) / (xa - xb), (yb - yc) / (xb - xc)
    curv = (d1 - d2) / (xa - xc)
    if curv <= 0:
        return {"reach": pts[i][0], "edge": True, "why": "not convex at the minimum"}
    return {"reach": math.exp(0.5 * (xa + xb - d1 / curv)), "edge": False,
            "grid_reach": pts[i][0]}


def row_moments(w: torch.Tensor) -> dict:
    """Per-row RMS, amax, ``z = amax / rms`` and excess-free kurtosis
    (a Gaussian row reads 3)."""
    w = w.float()
    rms = w.pow(2).mean(dim=1).sqrt()
    mu = w.mean(dim=1, keepdim=True)
    c = w - mu
    m2 = c.pow(2).mean(dim=1)
    kurt = c.pow(4).mean(dim=1) / m2.pow(2).clamp_min(1e-30)
    amax = w.abs().amax(dim=1)
    return {"rms": rms.cpu(), "amax": amax.cpu(), "z": (amax / rms.clamp_min(1e-30)).cpu(),
            "kurtosis": kurt.cpu()}


def per_row_scores(w, hat, h_eval, h_fit_diag) -> dict:
    """Per-row SSE on each axis, fp32, plus the denominators.

    ``out`` per row is ``e_r H_eval e_r^T``; summed over rows it is
    ``|X_eval E^T|^2`` exactly, so the unit's ``out`` is
    ``sqrt(sum / sum_r w_r H_eval w_r^T)``.
    """
    e = (hat - w).float()
    wt = e.pow(2).sum(dim=1)
    out = ((e @ h_eval) * e).sum(dim=1)
    hd = (e.pow(2) * h_fit_diag.reshape(1, -1)).sum(dim=1)
    return {"wt": wt.cpu(), "out": out.cpu(), "h": hd.cpu()}


def denominators(w, h_eval, h_fit_diag) -> dict:
    w = w.float()
    return {"wt": float(w.pow(2).sum()),
            "out": float(((w @ h_eval) * w).sum()),
            "h": float((w.pow(2) * h_fit_diag.reshape(1, -1)).sum())}


def dense_roster(layers, roles) -> "list[str]":
    return [f"model.layers.{l}.{r}" for l in layers for r in roles]


def load_dense(a):
    """The dense roster: weights from the checkpoint, H from the production
    capture, H_eval from the held-out capture.  Yields per unit."""
    src = open_all(DENSE_SRC)
    act = ActivationSource.from_capture(a.production)
    fit_h = dict(act.hessians)
    if a.weights_only:
        act = None
    payload = torch.load(a.eval_h, map_location="cpu", weights_only=False)
    ev_h, ev_prov = payload["H"], payload["provenance"]
    fit_prov = ActivationSource.from_capture(a.production).provenance
    for f in ("text_sha256", "fit_ids_sha256", "eval_ids_sha256"):
        if ev_prov.get(f) != fit_prov.get(f):
            raise SystemExit(f"held-out H {a.eval_h} and fit H {a.production} disagree on "
                             f"{f}: {ev_prov.get(f)} vs {fit_prov.get(f)} -- not one split")
    xs = {}
    if a.eval_x:
        xp = torch.load(a.eval_x, map_location="cpu", weights_only=False)
        if xp["provenance"].get("eval_ids_sha256") != ev_prov.get("eval_ids_sha256"):
            raise SystemExit("--eval-x is not the same held-out slice as --eval-h")
        xs = xp["x"]
    meta = {"hessian_source": a.production, "eval_h_source": a.eval_h,
            "eval_h_provenance": ev_prov, "eval_x_source": a.eval_x,
            "activation_aware": None if act is None else act.config_block()}
    units = a.units or dense_roster(a.layers or DENSE_LAYERS, a.roles or DENSE_ROLES)

    def gen():
        for name in units:
            w = src[name + ".weight"].get_tensor(name + ".weight").cuda().float().contiguous()
            H = fit_h[name].cuda().float()
            kw = ({} if act is None else
                  act.for_unit(name + ".weight", w.shape[1], device="cuda",
                               scale_plane=ScalePlaneKind.CHANNEL))
            he = ev_h[name].cuda().float()
            x = xs[name].cuda().float() if name in xs else None
            yield name, w, H.diagonal().contiguous(), he, kw, x
            del w, H, he, kw, x
            torch.cuda.empty_cache()
    return meta, gen


def load_glm(a):
    """GLM-5.3 experts (expert 0 of the named layers): H fit on the capture's
    first rows, H_eval on its last ``--eval-rows`` -- the split the six-expert
    receipt scored on, now with the fit half used to *encode* too."""
    import json
    index = json.load(open(f"{GLM_SRC}/model.safetensors.index.json"))["weight_map"]
    meta = {"hessian_source": f"{GLM_ACT} (fit rows = all but the last {a.eval_rows})",
            "eval_h_source": f"{GLM_ACT} (last {a.eval_rows} rows, held out)",
            "activation_aware": None}
    pairs = [(l, p) for l in (a.layers or (5, 20, 42)) for p in (a.projs or ("gate_proj", "up_proj"))]

    def gen():
        for layer, proj in pairs:
            blob = torch.load(f"{GLM_ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                              map_location="cpu", weights_only=False)
            xa = blob["inputs"].float()
            n_fit = xa.shape[0] - a.eval_rows
            x_fit, x_ev = xa[:n_fit].contiguous().cuda(), xa[n_fit:].contiguous().cuda()
            del xa, blob
            H = (x_fit.T @ x_fit) / n_fit
            he = (x_ev.T @ x_ev)                       # a sum; the ratio cancels it
            fit_sha = hashlib.sha256(x_fit.cpu().numpy().tobytes()).hexdigest()
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}"
            with safe_open(f"{GLM_SRC}/{index[name + '.weight']}", framework="pt") as f:
                w = f.get_tensor(name + ".weight").contiguous().cuda().float()
            act = None
            if not a.weights_only:
                act = ActivationSource(
                    hessians={name: H},
                    provenance={"source": f"{GLM_ACT} expert-input capture, layer {layer}",
                                "text_sha256": fit_sha, "fit_tokens": int(n_fit),
                                "fit_ids_sha256": fit_sha,
                                "eval_rows": int(a.eval_rows),
                                "note": "identity fields are the sha of the fit rows' "
                                        "fp32 bytes; this capture has no token ids"})
                if meta["activation_aware"] is None:
                    meta["activation_aware"] = act.config_block()
            kw = ({} if act is None else
                  act.for_unit(name + ".weight", w.shape[1], device="cuda",
                               scale_plane=ScalePlaneKind.CHANNEL))
            yield f"L{layer}.{proj}", w, H.diagonal().contiguous(), he, kw, x_ev
            del w, H, he, kw, x_fit, x_ev
            torch.cuda.empty_cache()
    return meta, gen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("dense", "glm"), default="dense")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows-dir", required=True,
                    help="per-(unit, rung) .pt files with per-row SSE per arm")
    ap.add_argument("--units", nargs="*", default=None)
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--roles", nargs="*", default=None)
    ap.add_argument("--projs", nargs="*", default=None)
    ap.add_argument("--rungs", nargs="*", type=int, default=[1024, 1792, 2048])
    ap.add_argument("--grid-steps", type=int, default=4)
    ap.add_argument("--production", default="/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt")
    ap.add_argument("--eval-h", default="/mnt/shared/tessera-runs/bf16/refs/h_eval_qwen06b.pt")
    ap.add_argument("--eval-x", default=None,
                    help="an activation dump for a cross-check of the H_eval score")
    ap.add_argument("--eval-rows", type=int, default=1024, help="glm: held-out rows")
    ap.add_argument("--weights-only", action="store_true",
                    help="the matched control: same capture, same scorer, no Hessian "
                         "in the encode")
    a = ap.parse_args()

    b = Bench(a.out)
    rows_dir = Path(a.rows_dir)
    rows_dir.mkdir(parents=True, exist_ok=True)
    meta, gen = (load_dense if a.source == "dense" else load_glm)(a)
    b.doc = {"args": vars(a), "units": {}, **meta,
             "material_threshold_reach": MATERIAL, "deciding_axis": DECIDING,
             "law_source": law_sigma(a.rungs[0])[1]}
    b.log(f"BF16 reach on a roster, per-row.  law: {b.doc['law_source']}")
    b.log(f"  encode: {'WEIGHTS-ONLY (control)' if a.weights_only else 'production (ActivationSource defaults)'}"
          f"  {meta.get('activation_aware')}")
    b.log(f"  out = tr(E H_eval E^T) on held-out rows: {meta['eval_h_source']}")

    for name, w, h_fit_diag, h_eval, kw, x in gen():
        y = ny = None
        if x is not None:
            y = x @ w.T
            ny = float(y.norm())
        den = denominators(w, h_eval, h_fit_diag)
        mom = row_moments(w)
        res: dict = {"rows": w.shape[0], "cols": w.shape[1], "has_x": x is not None,
                     "row_kurtosis_median": float(mom["kurtosis"].median()),
                     "row_z_median": float(mom["z"].median()),
                     "row_z_max": float(mom["z"].max())}
        for q in a.rungs:
            recipe = wire_recipe(BF16_GRID, q)
            law, _ = law_sigma(q)
            arms = arms_for(q, a.grid_steps)
            arms.append((arms[0][0] + " [repeat]", arms[0][1]))
            b.log(f"\n== {name} {tuple(w.shape)}  R={q / 256:g}  L={recipe.window_bits}  "
                  f"law window_sigma={law:.6f}  rows z>4.05: "
                  f"{float((mom['z'] > 4.05).float().mean()):.3f}")
            b.header(("bpp", "wt", "h", "out", "out_x", "reach_rms", "over"))
            store = {"unit": name, "q256": q, "labels": [], "sigmas": [], "reach_rms": [],
                     "sse": {ax: [] for ax in AXES}, "denominators": den, **mom}
            for label, sigma in arms:
                key = f"R{q} {label}"
                st = reach_stats(w, BF16_GRID, recipe.window_bits, sigma, BF16_CHANNEL_SIGMA)
                started = time.time()
                try:
                    exported, _u, _f = encode_linear_planes(
                        w, grid=BF16_GRID, q256=q, name=name, window_sigma=float(sigma),
                        channel_sigma=float(BF16_CHANNEL_SIGMA), verify=True, **kw)
                    hat = read_unit_artifact(exported.blob, device=w.device)
                except Exception as exc:                          # noqa: BLE001
                    torch.cuda.empty_cache()
                    b.log(f"    {key:<34} !! FAILED: {type(exc).__name__}: {exc}")
                    continue
                pr = per_row_scores(w, hat, h_eval, h_fit_diag)
                r = {"bpp": float(exported.bpp), "sha": sha(exported.blob),
                     "tsha": tensor_sha(hat), "secs": time.time() - started,
                     "window_sigma": float(sigma), "channel_sigma": BF16_CHANNEL_SIGMA,
                     "reach_rms": st["reach_row_rms"], "over": st["rows_over_reach"],
                     **{ax: math.sqrt(float(pr[ax].sum()) / den[ax]) for ax in AXES}}
                if x is not None:
                    r["out_x"] = float((x @ hat.T - y).norm() / ny)
                res[key] = r
                b.row(key, r, ("bpp", "wt", "h", "out", "out_x", "reach_rms", "over"))
                if not label.endswith("[repeat]"):
                    store["labels"].append(label)
                    store["sigmas"].append(float(sigma))
                    store["reach_rms"].append(st["reach_row_rms"])
                    for ax in AXES:
                        store["sse"][ax].append(pr[ax])
                del hat
                torch.cuda.empty_cache()
            first, last = res.get(f"R{q} {arms[0][0]}"), res.get(f"R{q} {arms[-1][0]}")
            if first and last:
                res[f"R{q}_control"] = check_repeat_tensor(b, first, last, f"R{q} {arms[0][0]}")
            if not first:
                continue
            bpps = {round(v["bpp"], 6) for k, v in res.items()
                    if isinstance(v, dict) and k.startswith(f"R{q} ")}
            res[f"R{q}_bytes"] = {"distinct_bpp": sorted(bpps), "identical": len(bpps) == 1}
            b.log(f"    bytes: {len(bpps)} distinct bpp -> "
                  f"{'IDENTICAL' if len(bpps) == 1 else '!! DIFFER'}")
            if x is not None:
                gap = max(abs(v["out_x"] - v["out"]) / v["out"] for k, v in res.items()
                          if isinstance(v, dict) and k.startswith(f"R{q} ") and "out_x" in v)
                res[f"R{q}_out_crosscheck"] = {"max_rel_gap": gap}
                b.log(f"    out via H_eval vs via X_eval: max relative gap {gap:.2e}")
            pinned = next((v for k, v in res.items() if isinstance(v, dict)
                           and k.startswith(f"R{q} pinned")), None)
            res[f"R{q}_ratios_vs_law"] = {
                k: {ax: v[ax] / first[ax] for ax in AXES}
                for k, v in res.items() if isinstance(v, dict) and k.startswith(f"R{q} ")
                and "wt" in v}
            pts = [(v["reach_rms"], v[DECIDING]) for k, v in res.items()
                   if isinstance(v, dict) and k.startswith(f"R{q} ") and "wt" in v
                   and not k.endswith("[repeat]")]
            opt = optimum_reach(pts, DECIDING)
            if opt.get("reach"):
                opt["vs_law"] = opt["reach"] / first["reach_rms"]
                opt["material"] = (not opt["edge"]) and (
                    max(opt["vs_law"], 1.0 / opt["vs_law"]) > MATERIAL)
            res[f"R{q}_optimum_out"] = {"law_reach": first["reach_rms"], **opt}
            if pinned:
                res[f"R{q}_law_vs_pinned_out"] = first["out"] / pinned["out"]
            b.log(f"    optimum[out]: reach {(opt.get('reach') or float('nan')):.3f} vs law "
                  f"{first['reach_rms']:.3f} = {opt.get('vs_law', float('nan')):.4f}x  "
                  f"{'EDGE' if opt['edge'] else ('MATERIAL' if opt.get('material') else 'within tolerance')}"
                  + (f";  law/pinned out {first['out'] / pinned['out']:.4f}" if pinned else ""))
            for ax in AXES:
                store["sse"][ax] = torch.stack(store["sse"][ax])      # [arms, rows]
            torch.save(store, rows_dir / f"{name}.q{q}.pt")
        b.doc["units"][name] = res
        b.save()

    units = b.doc["units"]
    summary = {}
    for q in a.rungs:
        blk = {"arms": {}, "law_vs_pinned_out": {}, "optimum_out": {}}
        labels = {k for u in units.values() for k in u if isinstance(u[k], dict)
                  and k.startswith(f"R{q} ")}
        for label in sorted(labels):
            for ax in AXES:
                vals = [u[f"R{q}_ratios_vs_law"][label][ax] for u in units.values()
                        if f"R{q}_ratios_vs_law" in u and label in u[f"R{q}_ratios_vs_law"]]
                if vals:
                    row = blk["arms"].setdefault(label, {})
                    row[ax] = geomean(vals)
                    row[f"{ax}_worse_than_law"] = sum(v > 1.0 for v in vals)
                    row[f"{ax}_units"] = len(vals)
        lp = [u[f"R{q}_law_vs_pinned_out"] for u in units.values() if f"R{q}_law_vs_pinned_out" in u]
        if lp:
            blk["law_vs_pinned_out"] = {"geomean": geomean(lp), "units": len(lp),
                                        "law_worse_on": sum(v > 1.0 for v in lp),
                                        "worst": max(lp), "best": min(lp)}
        opts = [u[f"R{q}_optimum_out"] for u in units.values() if f"R{q}_optimum_out" in u
                and u[f"R{q}_optimum_out"].get("reach")]
        if opts:
            vs = [o["vs_law"] for o in opts]
            blk["optimum_out"] = {"geomean_vs_law": geomean(vs), "min": min(vs), "max": max(vs),
                                  "units": len(opts), "edges": sum(o["edge"] for o in opts),
                                  "material": sum(bool(o.get("material")) for o in opts)}
        summary[f"R{q}"] = blk
    b.doc["summary"] = summary
    b.log("\n== geomean over the roster, each arm against the LAW arm (>1: the law wins)")
    for rung, blk in summary.items():
        b.log(f"  {rung}")
        for label, v in blk["arms"].items():
            b.log(f"    {label:<34} " + "  ".join(
                f"{ax} {v[ax]:.4f} ({v[f'{ax}_worse_than_law']}/{v[f'{ax}_units']})"
                for ax in AXES if ax in v))
        if blk["law_vs_pinned_out"]:
            lp = blk["law_vs_pinned_out"]
            b.log(f"    law/pinned out: geomean {lp['geomean']:.4f}, law worse on "
                  f"{lp['law_worse_on']}/{lp['units']}, range {lp['best']:.4f}..{lp['worst']:.4f}")
        if blk["optimum_out"]:
            o = blk["optimum_out"]
            b.log(f"    optimum[out] vs law: geomean {o['geomean_vs_law']:.4f}x, range "
                  f"{o['min']:.3f}..{o['max']:.3f}, {o['edges']} edge, {o['material']} material of {o['units']}")
    b.save()
    b.log(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
