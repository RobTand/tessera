"""Three follow-ups to `tessera8_targets.py`, on the same rows and the same legs.

Rob, 2026-09-01, after the Tessera-8 targets doc: the EXL3 ladder stopped at
K=4 so every sub-4-bit Tessera and Gridbook rung was compared to an
*extrapolation*; Gridbook was priced with `make_nvfp4_cb_qdq` called bare, i.e.
*not* as PrismaQuant's production render calls it; and no Tessera arm carried
LDLQ, the one lever the doc names as remaining.  This script closes all three
on the identical protocol:

  * six GLM-5.3 routed-expert projections (L5/20/42 gate/up, 2048x4096),
  * the 2026-09-01 pread capture, the LAST 1024 rows held out,
  * three legs -- out (A16), W?A4 (served NVFP4 activations), W?A8 (served
    per-token FP8) -- computed by the same expressions `tessera8_targets.rec`
    uses, and checked: the EXL3 K=4 arm this script re-scores must reproduce
    0.06787 / 0.10989 / 0.07200.

**A. EXL3 K=2 and K=3.**  `exl3_reference_quantise.py --rates 2 3` under the
same protocol (H from the first 7168 rows, the library's own LDLQ, eval on the
last 1024), reconstructions at `L{layer}_{proj}_K{2,3}.pt`, bpw K + 0.011723
(the exact `bpw_bytes_exact` the reference records: K trellis bits plus the
suh/svh sign vectors).  With them the EXL3 ladder brackets every rung measured
here, so no ratio below is an extrapolation.

**B. Gridbook as it ships.**  PrismaQuant's production render hands the CB
encoder an imatrix: `col_weights = E[x^2]` per input column, built for packed
experts at `prismaquant/moe_imatrix.py:330`

    col_weights[gu_name] = (X.pow(2).mean(dim=0).reshape(1, 1, -1).cpu())

so this script passes `x_fit.pow(2).mean(dim=0)` -- the FIRST 7168 capture
rows, the same rows the Hessian comes from, never the held-out 1024.  Every
rung is run twice, bare and weighted, and the bare arm is required to
reproduce `tessera8_targets.json` to four digits so the imatrix delta is not
confounded with an environment difference (`PRISMAQUANT_CB_ENCODE_TIER` is
resolved per call).  LDLQ: `run-pipeline.sh:209` sets
`PRISMAQUANT_CB_LDLQ:=0`, so `nvfp4_cb_footprint._ldlq_for_format` returns
False for every rung on the production path -- the LDLQ arm here is research,
run through `ldlq_reassign_cb_fields_gated` with its do-no-harm gate, whose
verdict is recorded per tensor.

**C. LDLQ on the Tessera arms.**  H from the first 7168 rows in fp64,
`regularize_hessian(H, count, sigma_reg=sigma)` at sigma in {1.0, 3.0}
(`tessera_ldlq_generalisation.py` measured 0.025 harmful and 1-3 worth
1.06-1.10x out-of-document on the raw basis; EXL3 damps at 0.025 but runs on a
Hadamard-rotated basis, so the two regularisers are not comparable and the
asymmetry is stated, not netted).

  C.1 Tessera-4, the default wire (E2M1x2, q256=896, span 2, LUT plane, refit
  4, scale-weighted trellis).  `compensated_targets` over 32-column blocks
  with the production `encode_unit` inside each block.  The block encode is
  *not* slice-equal to the wire: `_pack_scales_lut` fits its 16-entry table
  over whatever matrix it is handed, so a 32-column slice gets a table fit to
  4096 halves where the unit's own is fit to 524288.  The reported arm is
  therefore the one `compensate.py`'s docstring prescribes -- re-encode the
  compensated target ONCE with the whole-unit production encoder -- and the
  stitched reconstruction is carried alongside as a diagnostic, with the gap
  between them reported.  A second inner encoder (S6b plane, which *is*
  slice-exact at block 32 because group and half are within-row) isolates how
  much the optimistic inner encoder costs.  The stitched reconstruction is
  **not** a number at the default wire and is not offered as one: 128 slices
  means 128 independent 16-entry tables and 128 globals, ~2 KB over 8.4 M
  weights (+0.002 bpp) but 128x the table adaptivity.  It is a lead for a wire
  change, labelled as such.

  C.2 Tessera-8 per-channel (`tessera8_targets.per_channel_tcq`: no block
  plane, LM+midpoint E4M3 anchors, Ungerboeck code, least-squares row scale,
  s=64) at R=4 and R=5.  Here the slice encode IS exact: there is no plane,
  `viterbi_columns` treats columns independently, and the only cross-column
  state is the row scale and the (s/s.max())^2 branch weights.  Both are
  frozen from an uncompensated pre-pass, so a 32-column slice encode equals
  the corresponding span of the whole encode bit for bit -- asserted by running
  the same schedule with L = I and requiring an exact match.  The assertion is
  structural, so it runs on the first tensor only; a second tensor would re-pay
  80 s to re-derive the same fact.  The row scale is NOT refit after LDLQ in the
  reported arm: a refit re-optimises weight-space SSE against W and would partly
  undo the H-weighted feedback.

Ratios against EXL3 are taken per tensor at the arm's own bpp, with EXL3
interpolated log-linearly in bpp between its adjacent rungs (the convention of
`docs/measurements/tessera8-targets-2026-09-01.md` §2), then reported as the
mean with the per-tensor min-max.
"""
import argparse, json, math, sys, time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import E2M1_GRID, PayloadGrid, build_forest, tuple_grid
from tessera.compensate import block_ldl, compensated_targets, regularize_hessian
from tessera.decode import reconstruct_unit
from tessera.encode import encode_unit, viterbi_columns
from tessera.export import _plan_for
from tessera.manifest import ScalePlaneKind

from tessera8_bounds import gaussian_sample
from tessera8_targets import (ACT, EXL3, LARSEN6, SRC, UNG6, midpoint_codebook,
                              per_channel_tcq, snap_unique)

from prismaquant.fp8_dynamic import fp8_dynamic_activation_qdq_vllm
from prismaquant.nvfp4_activation_contract import (
    nvfp4_activation_qdq_served, select_mse_grid_input_global_scale)
from prismaquant.nvfp4_cb_formats import (ldlq_reassign_cb_fields_gated,
                                          make_nvfp4_cb_qdq,
                                          nvfp4_cb_fields, nvfp4_cb_reconstruct)

# bpw_bytes_exact the reference records for every K: K trellis bits plus the
# suh/svh fp16 sign vectors over a [2048, 4096] tensor.
EXL3_BPW_OVERHEAD = 0.011722564697266
# The K=4 arm's three legs, from experiments/results/tessera8_targets.json.
# This script must reproduce them or its legs are not the harness's legs.
EXL3_K4_CHECK = {"out": 0.06787, "a4": 0.10989, "a8": 0.07200}
# tessera8_targets.json means, for the reproduction arms.  Compared at the
# precision that document publishes (4 decimals): the encoders are deterministic
# within a process -- the per-channel arm is asserted bit-identical to
# ``per_channel_tcq`` below -- but not bit-identical ACROSS processes.  Measured
# here: R=5 reproduces to 0.0 on all six tensors, R=4 drifts up to 5.4e-5 in the
# output leg (1.3e-7 in weight space, so a handful of ties in the alternating
# LS/Viterbi refit fall the other way and the activation leg amplifies them).
# The drift per arm is recorded, so a real regression cannot hide inside it.
REPRO = {
    "Gridbook FP8-CB K32": 0.08712, "Gridbook FP8-CB K40": 0.04602,
    "Gridbook FP8-CB K48": 0.02397, "Gridbook FP4-CB K24 (v2 two-tier)": 0.14246,
    "Gridbook FP4-CB K20 (v2 two-tier)": 0.19744,
    "Gridbook FP4-CB K16 (v2 two-tier)": 0.27833,
    "Tessera-4 q256=896 (span2 LUT refit4 scale-wt)": 0.07983,
    "Tessera-8 R=4 per-channel (s=64)": 0.07800,
    "Tessera-8 R=5 per-channel (s=64)": 0.04101,
}

WIRE = dict(completion=0, span=2, scale_plane=ScalePlaneKind.LUT,
            scale_refit=4, trellis_weighting="scale")
S6B_WIRE = dict(completion=0, span=1, scale_plane=ScalePlaneKind.S6B,
                scale_refit=4, trellis_weighting="none")


# --------------------------------------------------------------------------
def pc_fit(w, R, code, book, sigma_units, refits, weighted):
    """``tessera8_targets.per_channel_tcq``, returning the state LDLQ freezes.

    Byte-for-byte the same loop; it hands back ``(hat, s, values, forest,
    weights)`` so the compensated pass can run the identical trellis on a
    column slice.  Checked against ``per_channel_tcq`` itself.
    """
    rows, cols = w.shape
    rms = w.pow(2).mean(dim=1, keepdim=True).sqrt()
    s = rms / sigma_units
    values = book
    grid = PayloadGrid(f"pc{R}", tuple(float(v) for v in values.tolist()))
    forest = build_forest(R, grid=grid)
    hat, weights = None, None
    for it in range(refits + 1):
        targets = (w / s).contiguous()
        weights = None
        if weighted:
            weights = (s / s.max()).pow(2).expand(rows, cols).contiguous()
        anchors, _, _ = viterbi_columns(targets, forest, code, completion=0, weights=weights)
        q = values.to(w.device)[anchors]
        if it < refits:
            num = (w * q).sum(dim=1, keepdim=True)
            den = (q * q).sum(dim=1, keepdim=True)
            s = torch.where(den > 0, num / den, s)
        hat = q * s
    return hat, s, values.to(w.device), forest, weights


def exl3_interp(bpp, rungs):
    """log-linear interpolation of an EXL3 leg at ``bpp``.

    ``rungs`` is ``{bpw: err}``.  log(err) is linear in bpw between the two
    bracketing rungs; outside the ladder the nearest pair is extended, and the
    caller is told so.
    """
    ks = sorted(rungs)
    lo = max([k for k in ks if k <= bpp], default=None)
    hi = min([k for k in ks if k >= bpp], default=None)
    extrapolated = lo is None or hi is None
    if lo is None:
        lo, hi = ks[0], ks[1]
    elif hi is None:
        lo, hi = ks[-2], ks[-1]
    if lo == hi:
        return rungs[lo], False
    t = (bpp - lo) / (hi - lo)
    return math.exp((1 - t) * math.log(rungs[lo]) + t * math.log(rungs[hi])), extrapolated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--exl3", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 3.0])
    ap.add_argument("--ldlq-block", type=int, default=32)
    ap.add_argument("--q256", type=int, default=896)
    ap.add_argument("--rates", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--sigma-units", type=float, default=64.0)
    ap.add_argument("--refits", type=int, default=2)
    ap.add_argument("--fp4cb", type=int, nargs="+", default=[24, 20, 16])
    ap.add_argument("--fp8cb", type=int, nargs="+", default=[32, 40, 48])
    ap.add_argument("--cb-ldlq", type=int, default=1, help="run the research CB-LDLQ arm")
    ap.add_argument("--s6b-inner", type=int, default=1,
                    help="C.1 control: the slice-exact S6b inner encoder, sigma[0] only")
    ap.add_argument("--out", default="experiments/results/tessera_vs_exl3_followups.json")
    ap.add_argument("--summarize-only", action="store_true",
                    help="re-derive the summary, the EXL3 ratios and both checks "
                         "from an existing --out, encoding nothing. The per-tensor "
                         "arms are the measurement; this rebuilds what is computed "
                         "from them, so a reporting fix costs no GPU time")
    a = ap.parse_args()
    logf = open(a.out.replace(".json", ".log"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    log(f"\n##### tessera_vs_exl3_followups {time.strftime('%Y-%m-%d %H:%M:%S')} args={vars(a)}")
    if a.summarize_only:
        out = json.load(open(a.out))
        out["arms"] = {k: v for k, v in out["arms"].items()
                       if not k.endswith("[imported from tessera8_targets.json]")}
        summarize(out, a, log)
        json.dump(out, open(a.out, "w"), indent=1)
        log(f"\nrewrote {a.out} (summary only)")
        return 0 if out["exl3_k4_leg_check"]["pass"] else sys.exit(2)
    torch.manual_seed(0)
    grid4 = tuple_grid(E2M1_GRID, 2)
    rates4, forests4 = _plan_for(grid4, a.q256, 4096)
    z = gaussian_sample(1 << 16, "cuda")
    books = {R: snap_unique(midpoint_codebook(z, R) * a.sigma_units) for R in a.rates}
    cb4 = {k: make_nvfp4_cb_qdq(k, "fp4", "product", scale_coding="two_tier") for k in a.fp4cb}
    cb8 = {k: make_nvfp4_cb_qdq(k, "fp8", "product") for k in a.fp8cb}
    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    out = {"tensors": [], "legs": [], "arms": {}, "args": vars(a),
           "exl3_bpw_overhead": EXL3_BPW_OVERHEAD, "notes": {}, "cb_ldlq_gate": {}}

    for layer in a.layers:
        blob = torch.load(f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                          map_location="cpu", weights_only=False)
        xa = blob["inputs"].float()
        n_fit = xa.shape[0] - a.eval_rows
        x_fit = xa[:n_fit].contiguous().cuda()
        x_ev = xa[n_fit:].contiguous().cuda()
        g = select_mse_grid_input_global_scale([x_fit])
        xq4 = nvfp4_activation_qdq_served(x_ev, g).float()
        xq8 = fp8_dynamic_activation_qdq_vllm(x_ev).dequant.float()
        # The production imatrix convention, prismaquant/moe_imatrix.py:330.
        col_weights = x_fit.pow(2).mean(dim=0).float().contiguous()
        # H as exl3_reference_quantise.build_H accumulates it: fp64, cast to fp32.
        H0 = (x_fit.double().T @ x_fit.double()).float()
        torch.cuda.empty_cache()

        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name).contiguous().cuda().float()
            R_, C = w.shape
            y = x_ev @ w.T
            ny, nw = y.norm(), w.norm()
            legs = {"a4": float((xq4 @ w.T - y).norm() / ny),
                    "a8": float((xq8 @ w.T - y).norm() / ny)}
            tag = f"L{layer}.{proj}"
            out["tensors"].append(tag); out["legs"].append(legs)
            log(f"\n== {tag} {tuple(w.shape)}  H {n_fit} rows  held-out {x_ev.shape[0]} rows  "
                f"act legs: A4 {legs['a4']:.5f}  A8 {legs['a8']:.5f}")
            log(f"    {'arm':<66} {'bpp':>6} {'wt':>8} {'out':>8} {'W?A4':>8} {'W?A8':>8} {'s':>6}")
            t_arm = [time.time()]

            def rec(arm, hat, bpp, **extra):
                r = {"tensor": tag, "bpp": bpp, "wt": float((hat - w).norm() / nw),
                     "out": float((x_ev @ hat.T - y).norm() / ny),
                     "a4": float((xq4 @ hat.T - y).norm() / ny),
                     "a8": float((xq8 @ hat.T - y).norm() / ny)}
                r.update(extra)
                out["arms"].setdefault(arm, []).append(r)
                now = time.time(); dt = now - t_arm[0]; t_arm[0] = now
                log(f"    {arm:<66} {bpp:6.3f} {r['wt']:8.5f} {r['out']:8.5f} "
                    f"{r['a4']:8.5f} {r['a8']:8.5f} {dt:6.1f}")
                return r

            # ---- A: the EXL3 ladder, K=2..8 -----------------------------
            for K in a.exl3:
                p = Path(f"{EXL3}/L{layer}_{proj}_K{K}.pt")
                if not p.exists():
                    log(f"    !! missing {p}")
                    continue
                rec(f"EXL3 K={K}", torch.load(p, map_location="cuda").float(),
                    K + EXL3_BPW_OVERHEAD)

            # ---- B: Gridbook bare (reproduction) and as production renders
            for k in a.fp8cb:
                rec(f"Gridbook FP8-CB K{k}", cb8[k](w), k / 8 + 32 / C)
                rec(f"Gridbook FP8-CB K{k} +imatrix", cb8[k](w, col_weights), k / 8 + 32 / C)
            for k in a.fp4cb:
                rec(f"Gridbook FP4-CB K{k} (v2 two-tier)", cb4[k](w), k / 8 + 0.28125)
                rec(f"Gridbook FP4-CB K{k} (v2 two-tier) +imatrix",
                    cb4[k](w, col_weights), k / 8 + 0.28125)
            if a.cb_ldlq:
                for grid_name, ks, coding, bpp_of in (
                        ("fp8", a.fp8cb, None, lambda k: k / 8 + 32 / C),
                        ("fp4", a.fp4cb, "two_tier", lambda k: k / 8 + 0.28125)):
                    for k in ks:
                        kw = {} if coding is None else {"scale_coding": coding}
                        try:
                            fields = nvfp4_cb_fields(w, k, grid=grid_name, mode="product",
                                                     col_weights=col_weights, **kw)
                            fields2, gate_info = ldlq_reassign_cb_fields_gated(
                                w, fields, col_weights, x_fit,
                                grid=grid_name, mode="product", k=k)
                        except Exception as exc:      # noqa: BLE001 - reported, not hidden
                            log(f"    !! CB-LDLQ {grid_name} K{k}: {type(exc).__name__}: {exc}")
                            break
                        label = ("Gridbook FP8-CB" if grid_name == "fp8"
                                 else "Gridbook FP4-CB") + f" K{k}"
                        label += " (v2 two-tier)" if coding else ""
                        rec(label + " +imatrix +LDLQ(gated)",
                            nvfp4_cb_reconstruct(fields2, k, grid=grid_name,
                                                 mode="product").float(), bpp_of(k))
                        out["cb_ldlq_gate"].setdefault(label, []).append(
                            {"tensor": tag, **{kk: vv for kk, vv in gate_info.items()
                                               if isinstance(vv, (str, bool, int, float))}})

            # ---- C.1: Tessera-4 default wire, uncompensated and with LDLQ
            unit = encode_unit(w, forests4, rates4, LARSEN6, **WIRE)
            base4 = reconstruct_unit(unit, forests4, LARSEN6)
            rec("Tessera-4 q256=896 (span2 LUT refit4 scale-wt)", base4, a.q256 / 256 + 0.5)
            del unit

            def wire_slice(target, start, stop):
                u = encode_unit(target.contiguous(), forests4,
                                tuple(rates4[start:stop]), LARSEN6, **WIRE)
                return reconstruct_unit(u, forests4, LARSEN6)

            def s6b_slice(target, start, stop):
                u = encode_unit(target.contiguous(), forests4,
                                tuple(rates4[start:stop]), LARSEN6, **S6B_WIRE)
                return reconstruct_unit(u, forests4, LARSEN6)

            for sig in a.sigmas:
                Hreg = regularize_hessian(H0, count=n_fit, sigma_reg=sig)
                L = block_ldl(Hreg, a.ldlq_block)
                inners = [("wire", wire_slice)]
                if a.s6b_inner and sig == a.sigmas[0]:
                    inners.append(("S6b", s6b_slice))
                for iname, fn in inners:
                    target, stitched = compensated_targets(w, L, fn, block=a.ldlq_block)
                    whole = reconstruct_unit(
                        encode_unit(target, forests4, rates4, LARSEN6, **WIRE),
                        forests4, LARSEN6)
                    gap = float((whole - stitched).norm() / nw)
                    suffix = "" if iname == "wire" else f", {iname} inner"
                    rec(f"Tessera-4 + LDLQ s={sig} (whole re-encode{suffix})", whole,
                        a.q256 / 256 + 0.5, stitched_vs_whole_rel=gap,
                        ldlq_block=a.ldlq_block, inner=iname)
                    rec(f"Tessera-4 + LDLQ s={sig} (STITCHED, diagnostic{suffix})",
                        stitched, a.q256 / 256 + 0.5, ldlq_block=a.ldlq_block, inner=iname)
                    del target, stitched, whole
                del Hreg, L
                torch.cuda.empty_cache()

            # ---- C.2: Tessera-8 per-channel, uncompensated and with LDLQ
            for R in a.rates:
                book = books[R]
                ref, _ = per_channel_tcq(w, R, UNG6, book, a.sigma_units, a.refits, True, log)
                hat0, s_row, values, forest, wgt = pc_fit(
                    w, R, UNG6, book, a.sigma_units, a.refits, True)
                same = float((hat0 - ref).abs().max())
                assert same == 0.0, f"pc_fit diverged from per_channel_tcq by {same}"
                rec(f"Tessera-8 R={R} per-channel (s=64)", hat0, R + 32 / C)

                def pc_slice(target, start, stop):
                    t = (target / s_row).contiguous()
                    ww = wgt[:, start:stop].contiguous()
                    anchors, _, _ = viterbi_columns(t, forest, UNG6, completion=0, weights=ww)
                    return values[anchors] * s_row

                # Slice-exactness: with L = I no compensation flows, so the
                # stitched encode must equal the frozen-scale whole encode bit
                # for bit.  If it does not, the block schedule is not encoding
                # what the whole encoder would.
                # Structural, so it is asserted on the first tensor only; a
                # second tensor would re-pay 80 s to re-derive the same fact.
                if layer == a.layers[0] and proj == a.projs[0]:
                    eye = block_ldl(torch.eye(C, device=w.device), a.ldlq_block)
                    _, ident = compensated_targets(w, eye, pc_slice, block=a.ldlq_block)
                    drift = float((ident - hat0).abs().max())
                    out["notes"][f"{tag} R={R} slice_exactness_max_abs"] = drift
                    assert drift == 0.0, f"slice encode is not the whole encode ({drift})"
                    del eye, ident

                for sig in a.sigmas:
                    Hreg = regularize_hessian(H0, count=n_fit, sigma_reg=sig)
                    L = block_ldl(Hreg, a.ldlq_block)
                    _, hat = compensated_targets(w, L, pc_slice, block=a.ldlq_block)
                    rec(f"Tessera-8 R={R} per-channel + LDLQ s={sig} (frozen row scale)",
                        hat, R + 32 / C, ldlq_block=a.ldlq_block)
                    del Hreg, L, hat
                    torch.cuda.empty_cache()
                del ref, hat0, s_row, values, forest, wgt
                torch.cuda.empty_cache()

            json.dump(out, open(a.out, "w"), indent=1)
            del w, y, base4
            torch.cuda.empty_cache()
        del x_ev, xq4, xq8, H0, col_weights, x_fit
        torch.cuda.empty_cache()

    summarize(out, a, log)
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"\nwrote {a.out}")
    if not out["exl3_k4_leg_check"]["pass"]:
        sys.exit(2)


# --------------------------------------------------------------------------
# Arms measured by tessera8_targets.py on the SAME six tensors, the same rows
# and (checked below, per tensor, bit for bit) the same legs, whose bpp the
# K=4..8 ladder could only extrapolate to.  Imported rather than re-encoded:
# re-running them would produce the identical bytes.
IMPORT_SOURCE = "experiments/results/tessera8_targets.json"
IMPORT_ARMS = ("Tessera-4 q256=768 (span2 LUT refit4 scale-wt)",
               "Tessera-4 q256=640 (span2 LUT refit4 scale-wt)")
# The arm both files must agree on, per tensor, for the import to be legal.
IMPORT_WITNESS = "Tessera-4 q256=896 (span2 LUT refit4 scale-wt)"


def import_arms(out, log):
    """Pull IMPORT_ARMS across, but only after proving the two runs' legs agree."""
    try:
        src = json.load(open(IMPORT_SOURCE))
    except OSError as exc:                       # noqa: BLE001 - reported
        log(f"\nimport skipped: {exc}")
        return
    order = {t: i for i, t in enumerate(src["tensors"])}
    mine = out["tensors"]
    if not all(t in order for t in mine):
        log("\nimport skipped: tensor sets differ")
        return
    a_rows = out["arms"].get(IMPORT_WITNESS)
    b_rows = src["arms"].get(IMPORT_WITNESS)
    if not a_rows or not b_rows:
        log("\nimport skipped: witness arm absent")
        return
    worst = 0.0
    for r in a_rows:
        b = b_rows[order[r["tensor"]]]
        worst = max(worst, max(abs(r[k] - b[k]) for k in ("wt", "out", "a4", "a8")))
    out["import_witness_max_abs"] = worst
    if worst != 0.0:
        log(f"\nimport REFUSED: witness arm differs by {worst:g}; the two runs "
            "are not scoring the same thing")
        return
    for arm in IMPORT_ARMS:
        rows = src["arms"].get(arm)
        if not rows:
            continue
        out["arms"][arm + " [imported from tessera8_targets.json]"] = [
            dict(rows[order[t]], tensor=t) for t in mine]
    log(f"\nimported {len(IMPORT_ARMS)} arms from {IMPORT_SOURCE} "
        f"(witness arm identical to {worst:g})")


def summarize(out, a, log):
    arms = out["arms"]
    n = len(out["tensors"])
    mean = lambda v: sum(v) / len(v)
    import_arms(out, log)
    out["summary"] = {arm: {k: mean([r[k] for r in rs]) for k in ("bpp", "wt", "out", "a4", "a8")}
                      for arm, rs in arms.items()}
    legs = {k: mean([l[k] for l in out["legs"]]) for k in ("a4", "a8")}
    out["summary_legs"] = legs

    # The EXL3 ladder keyed BY TENSOR: an arm that is missing on one tensor
    # must not shift every later row onto another tensor's ladder.
    ladder = {leg: {} for leg in ("out", "a4", "a8", "wt")}
    for K in a.exl3:
        for r in arms.get(f"EXL3 K={K}", []):
            for leg in ladder:
                ladder[leg].setdefault(r["tensor"], {})[r["bpp"]] = r[leg]
    incomplete = {arm: len(rs) for arm, rs in arms.items() if len(rs) != n}
    out["incomplete_arms"] = incomplete
    if incomplete:
        log(f"\n!! arms without one row per tensor: {incomplete}")

    ratios = {}
    for arm, rs in arms.items():
        if arm.startswith("EXL3"):
            continue
        entry = {}
        for leg in ("out", "a4", "a8", "wt"):
            vals, extrap = [], False
            for r in rs:
                ref, ex = exl3_interp(r["bpp"], ladder[leg][r["tensor"]])
                extrap |= ex
                vals.append(r[leg] / ref)
            entry[leg] = {"mean": mean(vals), "min": min(vals), "max": max(vals),
                          "extrapolated": extrap, "n": len(vals)}
        ratios[arm] = entry
    out["ratio_vs_exl3_interpolated"] = ratios

    k4 = out["summary"].get("EXL3 K=4", {})
    ok = all(abs(round(k4.get(k, float("nan")), 5) - v) < 5e-6
             for k, v in EXL3_K4_CHECK.items())
    out["exl3_k4_leg_check"] = {"got": {k: k4.get(k) for k in EXL3_K4_CHECK},
                                "want": EXL3_K4_CHECK, "pass": ok}
    repro = {}
    for arm, want in REPRO.items():
        got = out["summary"].get(arm, {}).get("out")
        if got is not None:
            repro[arm] = {"got": got, "want": want, "drift": got - want,
                          "pass": round(got, 4) == round(want, 4)}
    out["reproduction_check"] = repro

    log(f"\n== mean over {n} tensors   act legs: A4 {legs['a4']:.5f}  A8 {legs['a8']:.5f}")
    log(f"    {'arm':<70} {'bpp':>6} {'wt':>8} {'out':>8} {'W?A4':>8} {'W?A8':>8}")
    for arm in arms:
        r = out["summary"][arm]
        log(f"    {arm:<70} {r['bpp']:6.3f} {r['wt']:8.5f} {r['out']:8.5f} "
            f"{r['a4']:8.5f} {r['a8']:8.5f}")

    log("\n| arm | bpp | out | vs EXL3 out | W?A4 | vs EXL3 A4 | W?A8 | vs EXL3 A8 |")
    log("|---|---:|---:|---:|---:|---:|---:|---:|")
    for arm in arms:
        r = out["summary"][arm]
        if arm.startswith("EXL3"):
            log(f"| {arm} | {r['bpp']:.3f} | {r['out']:.5f} | - | {r['a4']:.5f} | - "
                f"| {r['a8']:.5f} | - |")
            continue
        q = ratios[arm]
        fmt = lambda d: (f"{d['mean']:.3f}x ({d['min']:.3f}-{d['max']:.3f})"
                         + ("*" if d["extrapolated"] else ""))
        log(f"| {arm} | {r['bpp']:.3f} | {r['out']:.5f} | {fmt(q['out'])} | {r['a4']:.5f} "
            f"| {fmt(q['a4'])} | {r['a8']:.5f} | {fmt(q['a8'])} |")
    log("\n(* = the arm's bpp is outside the EXL3 ladder and the nearest pair was extended.)")

    log(f"\nEXL3 K=4 leg check (must match tessera8_targets.json): "
        + ", ".join(f"{k} got {k4.get(k, float('nan')):.5f} want {v:.5f}"
                    for k, v in EXL3_K4_CHECK.items())
        + f"  -> {'PASS' if ok else 'FAIL'}")
    for arm, d in repro.items():
        log(f"reproduction {arm:<52} got {d['got']:.5f} want {d['want']:.5f} "
            f"drift {d['drift']:+.1e} -> {'PASS' if d['pass'] else 'FAIL'}")
    log("\nslice-exactness (C.2, L = I must reproduce the whole encode): "
        + ("; ".join(f"{k}={v:g}" for k, v in out["notes"].items()) or "not run"))
    if out["cb_ldlq_gate"]:
        log("\nCB LDLQ gate verdicts (research arm; production sets "
            "PRISMAQUANT_CB_LDLQ=0, run-pipeline.sh:209):")
        for arm, rows in out["cb_ldlq_gate"].items():
            log(f"  {arm}: " + "; ".join(
                f"{r['tensor']}={r.get('gate')}"
                f"(holdout {r.get('holdout_ratio', float('nan')):.3f})" for r in rows))
        log("  the holdout ratio is ldlq/raw output MSE on a content-keyed random "
            "half of the SAME first-7168 rows the Hessian was fit on "
            "(_ldlq_holdout_split, nvfp4_cb_formats.py:2635); the +LDLQ(gated) rows "
            "above are scored on the last 1024 rows, a disjoint later window.")


if __name__ == "__main__":
    main()
