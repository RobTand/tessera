"""Issue #89: why a non-dyadic window-table sigma costs one dense unit 1.36x h error.

The measured fact (#89, from ``reach_e4m3_{spread,ratio}.json``): on E4M3 /
window body / CHANNEL plane / L=14 / seed 0 / R2048, ``layers.2.mlp.down_proj``
of Qwen3-0.6B reads h = 1.000 at table sigma 47.09 and 94.18 (dyadic multiples
of the shipped default) and h = 1.36x at 70.64 (0.75x), at *identical* realised
reach in row-RMS.  ``wt`` moves 0.3% over the same arms.

**The hypothesis this registers before it measures anything.**  Under a CHANNEL
plane the reach-aware start (``initial_channel_scale``) puts every over-reach
row's largest weight *exactly on* the table's extreme value ``reach``.  The
reach entry occupies 2 of 2^14 table states, so at R=8 the trellis can land on
it only from a narrow set of predecessor states; the ordinary best is the next
distinct table value below.  The residual on that one weight is then the
**E4M3 relative ULP at the reach** -- ``ulp(reach) / reach``, a sawtooth in
``log2(reach)`` with period 1.  Those amax entries sit in the Hessian-dominant
(massive-activation) columns, so they carry the h-weighted error and almost
none of the unweighted one.  A dyadic sigma shift moves ``reach`` by x2 and
leaves its mantissa alone; a non-dyadic one moves the mantissa.

Predicted h, relative to the default's reach of 384 (ulp fraction 32/384):

    reach   256    288    320    352    384    416    448
    ulp/r  .1250  .1111  .1000  .0909  .0833  .0769  .0714
    h/h0   1.500  1.333  1.200  1.091  1.000  0.923  0.857

which is a **step function** of sigma: flat inside a reach band, jumping at the
band edge.  Slope inside a band falsifies "the reach alone" and says the rest
of the snapped table matters.

Stages::

    P=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
    E="TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache PYTHONPATH=src:experiments"

    env $E $P experiments/ts89_dyadic_reach.py --stage repro     --out OUT/repro.json
    env $E $P experiments/ts89_dyadic_reach.py --stage decompose --out OUT/decompose.json
    env $E $P experiments/ts89_dyadic_reach.py --stage ladder    --out OUT/ladder.json
    env $E $P experiments/ts89_dyadic_reach.py --stage general   --out OUT/general.json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import E4M3_GRID  # noqa: E402
from tessera.encode import grid_vector_table, window_table  # noqa: E402
from tessera.export import encode_linear_planes  # noqa: E402
from tessera.scale_channel import default_channel_sigma  # noqa: E402
from tessera.unit_artifact import read_unit_artifact  # noqa: E402

from bf16_route_weight_space import DENSE_H, DENSE_SRC, open_all  # noqa: E402

UNIT = "model.layers.2.mlp.down_proj"
#: The eight the #80/#89 sweep ran, in its order.
ALL_UNITS = [
    "model.layers.2.mlp.down_proj",
    "model.layers.2.self_attn.q_proj",
    "model.layers.2.self_attn.k_proj",
    "model.layers.14.mlp.gate_proj",
    "model.layers.2.mlp.up_proj",
    "model.layers.14.mlp.down_proj",
    "model.layers.27.self_attn.o_proj",
    "model.layers.14.self_attn.v_proj",
]
WINDOW_BITS = 14


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()[:16]


def e4m3_ulp(v: float) -> float:
    """The E4M3 spacing above ``v``; at the grid's peak, the spacing below it.

    Two different quantities are in play and they part company exactly at a
    binade floor: the gap *above* a value and the gap *below* it.  A target
    pinned on the reach that the trellis cannot emit falls to the next value
    **below**, so ``reach_gap_rel`` is the physical one; this is reported
    beside it because the two predictors disagree at ``reach = 256`` and that
    disagreement is the ladder's sharpest test.
    """
    vals = sorted({abs(x) for x in E4M3_GRID.values})
    for a, b in zip(vals, vals[1:]):
        if abs(a - abs(v)) < 1e-9:
            return b - a
    if abs(vals[-1] - abs(v)) < 1e-9:
        return vals[-1] - vals[-2]
    raise ValueError(f"{v} is not an E4M3 value")


def table_facts(sigma: float, device="cpu") -> dict:
    """Everything about the snapped table that does not need a GPU."""
    codes = window_table(E4M3_GRID, WINDOW_BITS, sigma=sigma, seed=0, half=16)
    val = grid_vector_table(E4M3_GRID)[codes.long()].abs().squeeze(-1)
    uniq, cnt = torch.unique(val, return_counts=True)
    order = uniq.argsort(descending=True)
    top = uniq[order][:6].tolist()
    topn = cnt[order][:6].tolist()
    reach = float(top[0])
    below = float(top[1])
    return {
        "sigma": sigma,
        "reach": reach,
        "reach_over_sigma": reach / sigma,
        "n_at_reach": int(topn[0]),
        "next_below_reach": below,
        "reach_gap_rel": (reach - below) / reach,
        "ulp_at_reach_rel": e4m3_ulp(reach) / reach,
        "n_distinct": int(uniq.numel()),
        "top_values": top,
        "top_counts": topn,
    }


def score(w, hat, h) -> dict:
    e = hat - w
    return {
        "wt": float(e.norm() / w.norm()),
        "h": float(math.sqrt(float(((e * e).sum(0) * h).sum()
                                   / ((w * w).sum(0) * h).sum()))),
    }


def encode_arm(w, q256, name, *, window_sigma, channel_sigma, want_unit=False,
               scale_refit=None, refit_metric=None, refit_reach_floor=False):
    t0 = time.time()
    exported, unit, _f = encode_linear_planes(
        w, grid=E4M3_GRID, q256=q256, name=name, window_bits=WINDOW_BITS,
        window_sigma=window_sigma, channel_sigma=channel_sigma, verify=True,
        **({} if scale_refit is None else {"scale_refit": scale_refit}),
        refit_metric=refit_metric, refit_reach_floor=refit_reach_floor,
    )
    hat = read_unit_artifact(exported.blob, device=w.device)
    out = {"bpp": float(exported.bpp), "sha": sha(exported.blob),
           "secs": time.time() - t0}
    return (hat, out, unit) if want_unit else (hat, out)


class Bench:
    def __init__(self, out_path: str):
        self.out = Path(out_path)
        self.lines: list[str] = []
        self.doc: dict = {}

    def log(self, s: str) -> None:
        print(s, flush=True)
        self.lines.append(s)
        self.out.with_suffix(".log").write_text("\n".join(self.lines) + "\n")

    def save(self) -> None:
        self.out.write_text(json.dumps(self.doc, indent=1))


def load(name):
    src = open_all(DENSE_SRC)
    w = src[name + ".weight"].get_tensor(name + ".weight").cuda().float().contiguous()
    H = torch.load(DENSE_H)
    return w, H[name].cuda().float()


# ------------------------------------------------------------------ repro


def stage_repro(a) -> None:
    """The issue's own arms, re-run on this HEAD, one unit, R2048 (and R1024)."""
    b = Bench(a.out)
    base = default_channel_sigma(E4M3_GRID)
    w, h = load(UNIT)
    b.doc = {"stage": "repro", "unit": UNIT, "base_channel_sigma": base,
             "shape": list(w.shape), "arms": {}}
    # (label, channel multiplier, table ratio) -- the spread sweep then the
    # ratio sweep, default first and again last, exactly as #80's harness ran.
    arms = ([("m=1", 1.0, 1.0)]
            + [(f"m={m:g}", m, 1.0) for m in (0.5, 0.75, 1.25, 1.5, 2.0)]
            + [(f"rho={r:g}", 1.0, r) for r in (0.5, 0.75, 1.25, 1.5, 2.0)]
            + [("m=1 [repeat]", 1.0, 1.0)])
    for q in a.rungs:
        b.log(f"\n== {UNIT} {tuple(w.shape)}  R={q}  (R{q} = {q // 256} b/wt)")
        b.log(f"    {'arm':<16}{'tsigma':>10}{'reach':>8}{'ulp/r':>8}"
              f"{'wt':>10}{'h':>10}{'h ratio':>9}{'s':>6}")
        ref = None
        for label, m, ratio in arms:
            cs = m * base
            ws = None if ratio == 1.0 else ratio * cs
            tsig = cs if ws is None else ws
            facts = table_facts(tsig)
            hat, meta = encode_arm(w, q, UNIT, window_sigma=ws, channel_sigma=cs)
            r = {**meta, "channel_sigma": cs, "window_sigma": ws,
                 "table_sigma": tsig, **facts, **score(w, hat, h)}
            if ref is None:
                ref = r["h"]
            r["h_ratio"] = r["h"] / ref
            b.doc["arms"][f"R{q} {label}"] = r
            b.log(f"    {label:<16}{tsig:10.3f}{facts['reach']:8.0f}"
                  f"{facts['ulp_at_reach_rel']:8.4f}{r['wt']:10.6f}{r['h']:10.6f}"
                  f"{r['h_ratio']:9.3f}{r['secs']:6.0f}")
            del hat
            torch.cuda.empty_cache()
            b.save()
    b.log(f"\nwrote {a.out}")


# -------------------------------------------------------------- decompose


def stage_decompose(a) -> None:
    """Where the h error lives: amax entries of over-rows, or everything else."""
    b = Bench(a.out)
    base = default_channel_sigma(E4M3_GRID)
    w, h = load(UNIT)
    gv = grid_vector_table(E4M3_GRID, w.device).squeeze(-1)
    rms = w.pow(2).mean(dim=1).sqrt()
    amax = w.abs().amax(dim=1)
    z = amax / rms.clamp_min(1e-30)
    amax_col = w.abs().argmax(dim=1)
    rows = w.shape[0]
    b.doc = {"stage": "decompose", "unit": UNIT, "base_channel_sigma": base,
             "shape": list(w.shape), "arms": {}}
    for q in a.rungs:
        for label, m in (("m=1", 1.0), ("m=0.75", 0.75), ("m=0.5", 0.5)):
            cs = m * base
            facts = table_facts(cs)
            reach = facts["reach"]
            hat, meta, unit = encode_arm(w, q, UNIT, window_sigma=None,
                                         channel_sigma=cs, want_unit=True)
            e = hat - w
            hw = h.view(1, -1)
            energy = (e * e) * hw                                  # [rows, cols]
            total = float(energy.sum())
            denom = float(((w * w) * hw).sum())
            over = (amax * cs > reach * rms)
            ridx = torch.arange(rows, device=w.device)
            amask = torch.zeros_like(e, dtype=torch.bool)
            amask[ridx, amax_col] = True
            omask = over.view(-1, 1).expand_as(e)
            parts = {
                "over_amax": amask & omask,
                "over_other": (~amask) & omask,
                "under_amax": amask & (~omask),
                "under_other": (~amask) & (~omask),
            }
            # The row scale, recovered from the wire: every reconstructed
            # weight is ``grid_value(code) * scale[row]``, so the ratio is the
            # scale wherever the code is not zero.  Median over the row rather
            # than one element, so a single zero code cannot decide it.
            nz = gv[unit.codes] != 0
            ratio = torch.where(nz, hat / torch.where(nz, gv[unit.codes],
                                                      torch.ones_like(hat)),
                                torch.full_like(hat, float("nan")))
            row_scale = torch.nanmedian(ratio, dim=1).values
            tgt = w[ridx, amax_col] / row_scale                    # grid units
            emit = gv[unit.codes][ridx, amax_col]
            rec = {**meta, "channel_sigma": cs, "table_sigma": cs, **facts,
                   **score(w, hat, h),
                   "rows": rows, "rows_over": int(over.sum()),
                   "h_energy_total": total, "h_energy_denom": denom,
                   "share": {k: float(energy[v].sum()) / total for k, v in parts.items()},
                   "energy": {k: float(energy[v].sum()) for k, v in parts.items()},
                   }
            ov = over
            rec["amax_over"] = {
                "n": int(ov.sum()),
                "target_grid_units_mean": float(tgt[ov].abs().mean()),
                "target_over_reach_mean": float((tgt[ov].abs() / reach).mean()),
                "emit_over_reach_mean": float((emit[ov].abs() / reach).mean()),
                "frac_emit_is_reach": float((emit[ov].abs() >= reach - 1e-6).float().mean()),
                "frac_emit_ge_next_below": float(
                    (emit[ov].abs() >= facts["next_below_reach"] - 1e-6).float().mean()),
                "rel_err_mean": float(
                    ((emit[ov].abs() - tgt[ov].abs()).abs() / tgt[ov].abs().clamp_min(1e-30)).mean()),
            }
            # Emission histogram over the top table values.
            vals, counts = torch.unique(emit[ov].abs(), return_counts=True)
            o = vals.argsort(descending=True)
            rec["amax_over"]["emit_hist"] = [
                [float(vals[i]), int(counts[i])] for i in o[:10].tolist()]
            # Which columns carry the h energy.
            col = (energy.sum(0) / total)
            topc = col.argsort(descending=True)[:12]
            rec["top_columns"] = [[int(c), float(col[c]), float(h[c])] for c in topc.tolist()]
            b.doc["arms"][f"R{q} {label}"] = rec
            b.log(f"\n-- R{q} {label} sigma={cs:.3f} reach={reach:.0f} "
                  f"ulp/r={facts['ulp_at_reach_rel']:.4f} wt={rec['wt']:.6f} h={rec['h']:.6f}")
            b.log(f"   rows over reach {rec['rows_over']}/{rows}; h-energy share "
                  + ", ".join(f"{k} {v:.4f}" for k, v in rec["share"].items()))
            b.log(f"   amax of over-rows: emit/reach mean "
                  f"{rec['amax_over']['emit_over_reach_mean']:.4f}, "
                  f"hits reach {rec['amax_over']['frac_emit_is_reach']:.3f}, "
                  f"rel err {rec['amax_over']['rel_err_mean']:.4f}")
            b.log(f"   emit hist (value, n): {rec['amax_over']['emit_hist']}")
            del hat, e, energy
            torch.cuda.empty_cache()
            b.save()
    b.log(f"\nwrote {a.out}")


# ----------------------------------------------------------------- ladder


def stage_ladder(a) -> None:
    """A fine sigma ladder across one binade: step function, or slope?"""
    b = Bench(a.out)
    base = default_channel_sigma(E4M3_GRID)
    w, h = load(UNIT)
    sigmas = [float(s) for s in a.sigmas] if a.sigmas else (
        # 2-3 inside each reach band from 256 to 448, plus the two dyadic
        # gauge arms the issue did not run (0.375x and 0.1875x base).
        [63.0, 65.0, 67.0,                 # reach 256
         68.5, 70.63529888131201, 74.5,    # reach 288 (0.75x base)
         77.0, 79.5, 83.0,                 # reach 320
         85.5, 88.0, 90.0,                 # reach 352
         92.5, 94.18039850841602, 99.0,    # reach 384 (the default)
         100.5, 103.0, 106.5,              # reach 416
         109.0, 110.5, 111.5,              # reach 448, still below the clamp
         35.31764944065600,                # 0.375x base: the dyadic gauge of 0.75x
         17.65882472032800]                # 0.1875x base: and again
    )
    b.doc = {"stage": "ladder", "unit": UNIT, "base_channel_sigma": base,
             "rung": a.rungs[0], "arms": {}}
    q = a.rungs[0]
    b.log(f"\n== {UNIT} {tuple(w.shape)}  R{q}  ladder over table sigma "
          f"(window_sigma = channel_sigma = sigma)")
    b.log(f"    {'sigma':>10}{'reach':>8}{'ulp/r':>8}{'predict':>9}"
          f"{'wt':>10}{'h':>10}{'measured':>9}{'s':>6}")
    ref = None
    order = [base] + [s for s in sigmas if s != base] + [base]
    for i, s in enumerate(order):
        facts = table_facts(s)
        hat, meta = encode_arm(w, q, UNIT, window_sigma=None, channel_sigma=s)
        r = {**meta, "channel_sigma": s, "table_sigma": s, **facts, **score(w, hat, h)}
        if ref is None:
            ref = r["h"]
            ref_ulp = facts["ulp_at_reach_rel"]
        r["h_ratio"] = r["h"] / ref
        r["predicted_ratio"] = facts["ulp_at_reach_rel"] / ref_ulp
        key = f"sigma={s:.6g}" + (" [repeat]" if i == len(order) - 1 else "")
        b.doc["arms"][key] = r
        b.log(f"    {s:10.4f}{facts['reach']:8.0f}{facts['ulp_at_reach_rel']:8.4f}"
              f"{r['predicted_ratio']:9.3f}{r['wt']:10.6f}{r['h']:10.6f}"
              f"{r['h_ratio']:9.3f}{r['secs']:6.0f}")
        del hat
        torch.cuda.empty_cache()
        b.save()
    b.log(f"\nwrote {a.out}")


# ---------------------------------------------------------------- general


def stage_general(a) -> None:
    """The candidate sigma against the default, on every dense unit, both rungs."""
    b = Bench(a.out)
    base = default_channel_sigma(E4M3_GRID)
    cands = [base] + [float(s) for s in (a.sigmas or [])]
    H = torch.load(DENSE_H)
    src = open_all(DENSE_SRC)
    b.doc = {"stage": "general", "base_channel_sigma": base,
             "candidates": cands, "units": {}}
    for name in ALL_UNITS:
        w = src[name + ".weight"].get_tensor(name + ".weight").cuda().float().contiguous()
        h = H[name].cuda().float()
        res: dict = {"shape": list(w.shape)}
        for q in a.rungs:
            b.log(f"\n== {name} {tuple(w.shape)} R{q}")
            b.log(f"    {'sigma':>10}{'reach':>8}{'ulp/r':>8}{'bpp':>9}"
                  f"{'wt':>10}{'h':>10}{'wt rat':>8}{'h rat':>8}")
            ref = None
            for s in cands:
                facts = table_facts(s)
                hat, meta = encode_arm(w, q, name, window_sigma=None, channel_sigma=s)
                r = {**meta, "channel_sigma": s, **facts, **score(w, hat, h)}
                if ref is None:
                    ref = r
                r["h_ratio"] = r["h"] / ref["h"]
                r["wt_ratio"] = r["wt"] / ref["wt"]
                res[f"R{q} sigma={s:.6g}"] = r
                b.log(f"    {s:10.4f}{facts['reach']:8.0f}"
                      f"{facts['ulp_at_reach_rel']:8.4f}{r['bpp']:9.4f}"
                      f"{r['wt']:10.6f}{r['h']:10.6f}{r['wt_ratio']:8.3f}"
                      f"{r['h_ratio']:8.3f}")
                del hat
                torch.cuda.empty_cache()
        b.doc["units"][name] = res
        b.save()
        del w
        torch.cuda.empty_cache()
    b.log(f"\nwrote {a.out}")


def stage_refit(a) -> None:
    """Is the 1.367x in the TABLE, or in the scale refit that follows it?

    The reduced model says it is not in the table.  Re-running the first
    Viterbi pass alone on this unit's top-64 Hessian columns -- exact, on the
    CPU, at both sigmas, with production's ``trellis_weighting="scale"`` branch
    weight -- gives an h ratio of **1.0369**, not 1.367
    (``ts89_table_surgery.py``).  Those 64 columns carry 99.93% of this unit's
    h, so the reduction is not throwing the metric away; what it throws away is
    ``scale_refit=4``.

    The table cannot be carrying it on its own, either.  Careful about *why*:
    the snapped reach is a **step function** of sigma (the ladder measures the
    steps), so ``reach = 4.0773 * sigma`` is false in general.  What is exactly
    true is ``reach(2^k * sigma) = 2^k * reach(sigma)`` on a floating-point
    grid away from the floor and the peak -- and this particular pair happens
    to be exact for the same reason: ``reach(0.75 * s0) = 288 = 0.75 * 384``,
    because 288 is itself an E4M3 value.  So on *these two arms* the reach-aware
    row scales are exactly proportional and the two normalised tables differ
    only by where the E4M3 snap lands: relative snapped-vs-ideal energy
    6.976e-4 at the default against 7.031e-4 at ``m=0.75``, a predicted ratio
    of **1.004** against a measured 1.367.

    So this stage walks ``scale_refit`` from 0 to 4 at both sigmas and reports
    the ratio at each.  Flat at ~1.0 across the walk would refute the refit
    story; a ratio that is ~1.0 at 0 refits and 1.367 at 4 locates the
    mechanism in the alternation, not in the alphabet -- which is a different
    claim from the one the issue's title makes, and changes what a fix is.

    **Registered before the arms ran.**

    * ``refit=0`` reads ~1.04.  The reduced model is faithful and the refit is
      the amplifier.  Sigma is then not the mechanism; the refit's response to
      a 0.8% table change is.
    * ``refit=0`` reads ~1.37.  Something else in production carries it, and
      the next place to look is the post-trellis release
      (``encode.py`` ``release_index``/``release_code``), which overrides codes
      after the Viterbi and is where an amax effect would hide.  ``n_released``
      is recorded on every arm for exactly this.

    Then the 2x2 at the deepest refit, both sigmas, since a mechanism is only
    useful if it names a knob.  Neither arm is a wire change; both are
    parameters ``encode_linear_planes`` already takes.

    * ``refit_metric=h``.  The whole of #89's table was measured with an
      **h-blind refit and then scored on h**: ``encode_arm`` never passed a
      metric.  The refit's per-row least squares is dominated by the 3066
      columns h does not care about, so it lands the row scale for the bulk and
      the six columns that ARE h take whatever reach that scale leaves them.  A
      0.8% table change nudging that scale is then a large h change on six
      columns.  If the metric collapses the ratio toward 1.04, that is the
      mechanism.  (An h-aware refit scored on h is teaching to the test in
      weight space: a mechanism probe, never a ship claim.)
    * ``refit_metric=H`` (``--fullh``, added after the diagonal arm read
      0.9853).  ``DEFAULT_REFIT_OBJECTIVE["channel"] == "hessian"``, so the
      shipping CHANNEL plane already refits under a metric -- but when export
      supplies an ``ActivationSource`` that metric is the **full** ``[cols,
      cols]`` Hessian, not this diagonal.  "Production already avoids the
      h-blind refit" is a claim about an arm nobody had run, and a diagonal
      cannot stand in for it: the full form solves ``B/A`` with
      ``A = u H u^T``, which couples the columns the diagonal treats as
      independent.  The full H comes from a different capture than the scoring
      diagonal (``ldlq/h_full_qwen06b.pt`` vs ``bf16/refs/h_diag.pt``); the two
      agree on this unit's top-six columns exactly and on the median diagonal
      to 7%, so the arm is honest about which columns matter while remaining
      an out-of-sample metric for the score.
    * ``refit_reach_floor=True``.  ``floor`` is ``None`` by default, so the
      refit may re-inflate an over-row's scale past ``amax / reach`` and
      re-clip the weight the reach-aware start was protecting.  ``rows_at_clip``
      says whether it does, and whether it does so differentially.

    If either arm removes the sensitivity, the answer to "defect or cost" is
    that neither is about sigma: ``default_channel_sigma`` stays and the
    proposal is a refit change, with served evidence still owed.  If neither
    does, and the walk shows the ratio growing with refit count while each
    arm's own error falls monotonically -- which the alternation is documented
    to do under exactly this configuration (``trellis_weighting="scale"``, no
    ``ldl``, ``metric=None``) -- then this is basin sensitivity in a non-convex
    coordinate descent, no smooth objective in the residue exists, and
    "re-derive the constant" is the wrong shape of fix.

    Three diagnostics were added after the walk read 1.0367 -> 1.3667, to
    turn "the refit is the amplifier" from a shape into a mechanism.

    * ``rows_moved``: how many rows' recovered scale the refit actually moved,
      against the same unit's ``scale_refit=0`` scales.  "The refit is a null
      lever at ``m=0.75``" has two very different readings, and this separates
      them: the refit ran and its per-row least squares landed back where it
      started (the start is already the optimum there), or the refit proposed
      moves and the accept-only-if-lower test rejected them row by row.
    * ``top_h``: the h-weighted squared error of the six columns that carry
      99.9% of this unit's h, per arm.  The claim under test is that the 32%
      is bought on those columns and nowhere else; six numbers per arm say so
      or refute it.
    * ``--units all``: the same walk at refits {0, max} on all eight units of
      the #80/#89 sweep.  #89 says the effect is this unit's alone (the other
      seven move 0.1-5.7%), and the mechanism as stated predicts *why*: the
      refit is h-blind, so it can only be a large h lever where h is
      concentrated on a few columns.  A unit with flat h should show the refit
      buying the same h at both sigmas.  That is a prediction on units the
      mechanism was not fitted to, which is the only kind of check available
      without a serve.
    """
    base = default_channel_sigma(E4M3_GRID)
    units = a.units if a.units != ["all"] else ALL_UNITS
    b = Bench(a.out)
    b.doc = {"stage": "refit", "units": units, "base_channel_sigma": base,
             "arms": {}}
    for uname in units:
        w, h = load(uname)
        gv = grid_vector_table(E4M3_GRID, w.device).squeeze(-1).float()
        # Which columns h actually cares about, and how concentrated it is.
        order = h.argsort(descending=True)
        top_cols = order[:6]
        conc = {"h_top6_share": float(h[top_cols].sum() / h.sum()),
                "h_max_over_median": float(h.max() / h.median()),
                "h_top1_share": float(h[order[0]] / h.sum())}
        b.doc["arms"][f"{uname}/h_concentration"] = conc

        def arm(cs, nref, kw, rs0=None):
            facts = table_facts(cs)
            hat, meta, unit = encode_arm(
                w, q, uname, window_sigma=None, channel_sigma=cs,
                want_unit=True, scale_refit=nref, **kw)
            # Recover the effective row scale from the artifact the reader
            # sees: hat = row_scale * table_value, so the ratio is constant
            # down a row wherever the table value is nonzero.
            val = gv[unit.codes]
            rs = torch.where(val.abs() > 0, hat / val,
                             torch.full_like(hat, float("nan"))).nanmedian(dim=1).values
            # A row is AT THE CLIP when its loudest weight needs more than the
            # table's outermost entry at the scale the refit landed on.
            clipped = int((w.abs().amax(dim=1) > rs * facts["reach"] * (1 + 1e-6)).sum())
            e = hat - w
            colsse = (e * e).sum(0)
            denom = float(((w * w).sum(0) * h).sum())
            r = {**meta, "channel_sigma": cs, "table_sigma": cs, "refits": nref,
                 **facts, **score(w, hat, h), "rows_at_clip": clipped,
                 "rows": int(w.shape[0]),
                 # Requested vs realised, per #84: the requested reach is what
                 # sigma asks for; the realised one is what the row scale the
                 # encoder LANDED on can actually emit, in that row's own rms
                 # units.  They part company exactly on the rows the
                 # reach-aware start (and then the refit) moved.
                 "reach_rms_requested": facts["reach"] / cs,
                 "reach_rms": float(
                     (facts["reach"] * rs
                      / w.pow(2).mean(dim=1).sqrt()).median()),
                 "top_h_cols": top_cols.tolist(),
                 "top_h_energy": [float(h[c] * colsse[c] / denom) for c in top_cols],
                 "top_h_share": float((h[top_cols] * colsse[top_cols]).sum()
                                      / (h * colsse).sum()),
                 "n_released": int(unit.release_index.numel())}
            if rs0 is not None:
                moved = (rs - rs0).abs() > 1e-3 * rs0.abs()
                r["rows_moved"] = int(moved.sum())
                r["rs_max_rel_move"] = float(
                    ((rs - rs0).abs() / rs0.abs()).max())
            out_rs = rs.clone()
            del hat, unit, val, rs, e, colsse
            torch.cuda.empty_cache()
            return r, out_rs

        for q in a.rungs:
            b.log(f"\n== {uname} {tuple(w.shape)}  R={q}  ({q // 256} b/wt)"
                  f"   h top-6 share {conc['h_top6_share']:.4f}"
                  f"  h_max/med {conc['h_max_over_median']:.3g}")
            b.log(f"    {'refit':>6}{'m':>7}{'sigma':>10}{'reach':>8}"
                  f"{'reach_rms':>10}{'wt':>10}{'h':>10}{'h ratio':>9}"
                  f"{'rows@clip':>11}{'moved':>7}{'top6%':>7}{'s':>7}")
            rs_base = {}
            for nref in a.refits:
                per_sigma = {}
                for m in (1.0, 0.75):
                    r, rs = arm(m * base, nref, {}, rs_base.get(m))
                    if nref == min(a.refits):
                        rs_base[m] = rs
                    per_sigma[m] = r
                    b.doc["arms"][f"{uname}/R{q}/refit{nref}/m{m:g}"] = r
                ratio = per_sigma[0.75]["h"] / per_sigma[1.0]["h"]
                for m in (1.0, 0.75):
                    r = per_sigma[m]
                    shown = ratio if m == 0.75 else 1.0
                    b.log(f"    {nref:>6}{m:>7.2f}{r['channel_sigma']:>10.4f}"
                          f"{r['reach']:>8.1f}{r['reach_rms']:>10.4f}"
                          f"{r['wt']:>10.5f}{r['h']:>10.5f}"
                          f"{shown:>9.4f}{r['rows_at_clip']:>11d}"
                          f"{r.get('rows_moved', 0):>7d}"
                          f"{100 * r['top_h_share']:>7.2f}{r['secs']:>7.0f}")
                b.doc["arms"][f"{uname}/R{q}/refit{nref}/ratio"] = ratio
                b.save()

            if a.skip_2x2:
                continue
            # ---- the 2x2: does an h-aware refit, or a reach floor, remove it?
            deep = max(a.refits)
            b.log(f"\n    at scale_refit={deep}, the two refit knobs, both sigmas")
            b.log(f"    {'refit arm':<22}{'m':>6}{'wt':>10}{'h':>10}"
                  f"{'h ratio':>9}{'rows@clip':>11}{'moved':>7}{'top6%':>7}{'s':>7}")
            arms = [("metric=h", {"refit_metric": h}),
                    ("reach_floor", {"refit_reach_floor": True}),
                    ("metric=h+floor", {"refit_metric": h,
                                        "refit_reach_floor": True})]
            if a.fullh:
                # Production's CHANNEL default is objective "hessian", and when
                # export supplies an ActivationSource that is the FULL H, not
                # this diagonal.  A diagonal-h arm is a different objective, so
                # "production already avoids this" cannot rest on it.
                Hfull = torch.load(a.fullh, map_location="cpu",
                                   weights_only=False)["H"][uname]
                arms.insert(1, ("metric=H(full)",
                                {"refit_metric": Hfull.cuda().float()}))
            for arm_name, kw in arms:
                per_sigma = {}
                for m in (1.0, 0.75):
                    r, _rs = arm(m * base, deep, kw, rs_base.get(m))
                    r["refit_arm"] = arm_name
                    per_sigma[m] = r
                    b.doc["arms"][f"{uname}/R{q}/{arm_name}/m{m:g}"] = r
                ratio = per_sigma[0.75]["h"] / per_sigma[1.0]["h"]
                for m in (1.0, 0.75):
                    r = per_sigma[m]
                    b.log(f"    {arm_name:<22}{m:>6.2f}{r['wt']:>10.5f}"
                          f"{r['h']:>10.5f}"
                          f"{(ratio if m == 0.75 else 1.0):>9.4f}"
                          f"{r['rows_at_clip']:>11d}{r.get('rows_moved', 0):>7d}"
                          f"{100 * r['top_h_share']:>7.2f}{r['secs']:>7.0f}")
                b.doc["arms"][f"{uname}/R{q}/{arm_name}/ratio"] = ratio
                b.save()
        del w, h, gv
        torch.cuda.empty_cache()
    b.save()
    b.log(f"\nwrote {a.out}")


STAGES = {"repro": stage_repro, "decompose": stage_decompose,
          "ladder": stage_ladder, "general": stage_general,
          "refit": stage_refit}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--out", required=True)
    p.add_argument("--rungs", type=int, nargs="+", default=[2048])
    p.add_argument("--sigmas", type=float, nargs="*", default=None)
    p.add_argument("--refits", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--units", nargs="+", default=[UNIT],
                   help="unit names, or the single word all")
    p.add_argument("--fullh", default=None,
                   help="path to a full-Hessian capture; adds a metric=H(full) "
                        "arm to the 2x2, which is production's CHANNEL objective")
    p.add_argument("--skip-2x2", action="store_true",
                   help="walk only; for the multi-unit generality run")
    a = p.parse_args()
    torch.manual_seed(0)
    STAGES[a.stage](a)


if __name__ == "__main__":
    main()
