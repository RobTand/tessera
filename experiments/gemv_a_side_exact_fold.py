#!/usr/bin/env python3
"""The fold ITSELF, propagated to the logits -- no noise model (#110).

``gemv_a_side_propagation.py`` answers "what is a 1.6e-3 per-Linear
perturbation worth as a KL" by injecting *independent Gaussian noise* of that
relative size.  That is a MODEL of the term, and it is the weak leg of the
#110 receipt: the fold is not independent noise.  It is a deterministic
rounding of ``a_q * a_scale`` to bf16, and within one token ``a_scale`` is one
number, so the relative error is a function of the E4M3 code's three mantissa
bits alone -- eight distinct values, each shared by every element carrying
that mantissa.  An error that correlated can add coherently through a
196-Linear stack where independent noise adds in quadrature, and a Gaussian
screen cannot see the difference.

This script removes the model.  It runs the served model twice, once in each
arm's ARITHMETIC, and takes the KL between the two logit sets:

    arm A (the lane before this branch's fix, "folded")
        x_bf16 -> per-token E4M3 (a_q, a_scale)
        -> bf16(a_q * a_scale) -> fp32 GEMM against the decoded fp8 tile
        -> bf16 out
    arm B (``torch._scaled_mm``, and the lane AFTER the fix)
        x_bf16 -> the SAME (a_q, a_scale)
        -> fp32 GEMM of the codes -> * a_scale * w_scale in the epilogue
        -> bf16 out

Both arms share the quantiser, the weights and every other op, so the only
thing that varies is where ``a_scale`` enters -- a matched pair by
construction.  The fixed lane and arm B are the same expression here: the
1.6e-07 that separates them as served is fp32 reduction ORDER, which a CPU
emulation cannot and need not reproduce (it is measured on the real kernel by
``gemv_a_side_precision.py`` and moves no bf16 output word).

WHAT THIS IS AND IS NOT.  It is still a SCREEN -- CPU, fp32 residual stream,
HF rather than vLLM, RTN weights rather than the Tessera wire's (see
``--weight-bits``) -- so it does not close #110; only the served re-run does.
What it settles is the thing the Gaussian screen could not: whether the fold,
priced as the deterministic correlated rounding it actually is, is worth
0.012 nats on the served position set or three hundredths of that.

THE PERTURBATION LANDS WHERE THE SERVE PUTS IT.  ``--regime decode``
reproduces what ``kl_tool --regime decode`` actually does: one prefill per
chunk fills the KV cache (and in the serve BOTH arms take
``_materialised_path`` for it, so that cache is bit-identical across arms),
then each scored prefix is ONE M = 1 forward off that cache.  So the fold
touches only the scored token's own path through the 28 layers -- never the
cached keys and values of the tokens before it.  ``--regime prefill`` applies
it to every row of one 512-row forward instead; that over-applies it, and the
difference between the two is measured rather than argued.

The scoring is deliberately the SERVED metric's shape, not a convenient one:

* the served decode regime scores prefix lengths 1, 17, 33, ... 497 -- 32 per
  chunk -- so ``--stride 16 --chunks 8`` reproduces its 256 positions exactly;
* the served headline is ``kl_tool``'s lumped DPI LOWER bound over the
  top-``--topk`` intersection, not a full-vocab KL, so this reports that
  estimator (imported from ``kl_estimator`` -- the same code) beside the
  full-vocab number.  Reporting only the full-vocab KL against a served lower
  bound would compare two different metrics.

CPU-ONLY, and deliberately few threads: sparky was running another
worktree's latency A/B while this was written, and that measurement has a
quiet-box gate.  Nothing here touches a GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

FP8_MAX = 448.0
KL_TOOL_DIR = "/home/rob/dq-runs"


def per_token_e4m3(x: torch.Tensor):
    """The route's quantiser, as ``_reference_fp8_quant`` pins it.

    Returns the CODES as fp32 (their exact values -- an E4M3 byte is exact in
    both fp32 and bf16) and the per-token fp32 scale.  Both arms are handed
    these same two tensors; that is what makes the pair matched.
    """
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / FP8_MAX
    q = (xf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.float(), scale


def row_e4m3(w: torch.Tensor):
    """Per-output-row E4M3 weights: the pair the TESSERA_FP8 route serves.

    The Tessera window wire decodes to exactly this shape -- E4M3 codes plus a
    per-channel scale -- so the weight-side ARITHMETIC here is the served one.
    Which codes differ: these are RTN's, the served ones are the window
    encoder's.  ``--weight-bits`` exists to reach a comparable operating point
    without decoding the wire; the substitution is named in the receipt.
    """
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX
    q = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.float(), scale


class ArmLinear(torch.nn.Module):
    """One served Linear, in whichever arm's arithmetic ``state`` names."""

    def __init__(self, wq, w_scale, bias, state):
        super().__init__()
        self.wq = wq
        self.w_scale = w_scale
        self.bias = bias
        self.state = state
        self.fold_rel = []

    def forward(self, x):
        arm = self.state["arm"]
        if arm == "clean":
            y = x.float() @ (self.wq * self.w_scale).t()
            return y if self.bias is None else y + self.bias

        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).to(torch.bfloat16)
        if not self.state["act_quant"]:
            # ONE variable removed: the route's per-token E4M3 activation
            # quantiser.  Both arms lose it together, so the pair still differs
            # only by the perturbation.  This is what the propagation screen's
            # BF16-teacher run implicitly assumed -- that a term of this size
            # can be priced on a trajectory that is NOT carrying W8A8's own
            # activation rounding -- and it is the assumption this switch
            # tests, at fixed weights, fixed positions and fixed estimator.
            a_q, a_scale = x2.float(), torch.ones(x2.shape[0], 1)
        else:
            a_q, a_scale = per_token_e4m3(x2)
        if arm == "folded":
            # THE LANE BEFORE THE FIX: a_scale folded into the bf16 operand.
            exact = a_q * a_scale
            xa = exact.to(torch.bfloat16).float()
            self.fold_rel.append(float((xa - exact).norm() / exact.norm()))
            y = xa @ (self.wq * self.w_scale).t()
        elif arm == "gaussian":
            # THE CONTROL, and the ONLY thing that varies from ``folded``:
            # independent Gaussian noise of the SAME relative rms, standing in
            # for the fold's deterministic rounding.  This is the model
            # ``gemv_a_side_propagation.py`` uses; running it here, at the same
            # injection point, in the same harness, on the same weights and the
            # same quantiser, isolates "correlated vs independent" as the one
            # variable.  Everything else -- operating point, position set,
            # estimator, model -- is pinned.
            exact = a_q * a_scale
            rel = self.state["rel"]
            if rel is None:
                rel = float((exact.to(torch.bfloat16).float() - exact).norm() / exact.norm())
            self.fold_rel.append(rel)
            g = self.state["gen"]
            xa = exact + torch.randn(exact.shape, generator=g) * (rel * exact.abs())
            y = xa @ (self.wq * self.w_scale).t()
        elif arm == "scaled_mm":
            # torch._scaled_mm, and the lane after the fix: codes multiplied
            # exactly, both scales applied in the fp32 epilogue.
            y = (a_q @ self.wq.t()) * a_scale * self.w_scale.t()
        else:
            raise ValueError(arm)
        y = y.to(torch.bfloat16).float().reshape(*shape[:-1], -1)
        return y if self.bias is None else y + self.bias


def full_vocab_kl(p_logits: torch.Tensor, q_logits: torch.Tensor):
    lp = torch.log_softmax(p_logits.double(), dim=-1)
    lq = torch.log_softmax(q_logits.double(), dim=-1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    top1 = (p_logits.argmax(-1) == q_logits.argmax(-1)).double().mean()
    return kl, float(top1)


def served_estimator(p_logits, q_logits, top_k, floor):
    """``kl_tool compare``'s own number, on this pair.

    The served headline is a lumped DPI lower bound over the intersection of
    the two dumps' top-K lists, NOT a full-vocab KL.  Running the real
    estimator (``kl_estimator.lumped_kl``) over top-K dumps taken from these
    logits is what makes the screen's number and the served number the same
    metric.
    """
    sys.path.insert(0, KL_TOOL_DIR)
    from kl_estimator import lumped_kl  # noqa: E402

    lp = torch.log_softmax(p_logits.double(), dim=-1)
    lq = torch.log_softmax(q_logits.double(), dim=-1)
    out = []
    for i in range(lp.shape[0]):
        pv, pi = lp[i].topk(top_k)
        qv, qi = lq[i].topk(top_k)
        tmap = {int(t): float(v) for t, v in zip(pi.tolist(), pv.tolist())}
        smap = {int(t): float(v) for t, v in zip(qi.tolist(), qv.tolist())}
        shared = [t for t in tmap if t in smap]
        p_on_s = [math.exp(tmap[t]) for t in shared]
        q_on_s = [math.exp(smap[t]) for t in shared]
        st = lumped_kl(p_on_s, q_on_s, student_prob_floor=floor, n_tail_tokens=None)
        st["coverage"] = float(sum(p_on_s))
        out.append(st)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/rob/models/Qwen3-0.6B")
    ap.add_argument("--corpus", default="/mnt/shared/tessera-kl/corpus_qwen_n8_s512.json")
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--stride", type=int, default=16,
                    help="the served decode set: prefix lengths 1, 1+s, ... (16 = the "
                         "serve's KV block size).  0 scores every position.")
    ap.add_argument("--weight-bits", type=int, default=0,
                    help="RTN the weights to this many bits per output row BEFORE the "
                         "fp8 pair, to reach an operating point comparable to the served "
                         "arms (which sit 0.436 nats from the teacher).  0 = fp8 only.")
    ap.add_argument("--arm-a", choices=("folded", "gaussian"), default="folded",
                    help="'folded' is the lane's actual arithmetic; 'gaussian' replaces its "
                         "deterministic rounding with independent noise of the SAME measured "
                         "relative rms -- the matched control for the noise MODEL the "
                         "propagation screen uses, with nothing else changed")
    ap.add_argument("--rel", type=float, default=None,
                    help="override the gaussian arm's relative rms (default: the rms the "
                         "fold itself measures on this trajectory).  Needed with "
                         "--act-quant off, where there is no fold left to measure.")
    ap.add_argument("--regime", choices=("decode", "prefill"), default="decode",
                    help="'decode' is the served shape: a clean prefill fills the KV cache, "
                         "then each scored position is one M=1 forward whose arithmetic is "
                         "the arm's.  'prefill' perturbs every row of one 512-row forward, "
                         "which is NOT what the serve does.")
    ap.add_argument("--act-quant", choices=("on", "off"), default="on",
                    help="'on' is the served route (per-token E4M3 activations, W8A8); "
                         "'off' removes the quantiser from BOTH arms, so the perturbation "
                         "is priced on a trajectory that is not already carrying the "
                         "route's own activation rounding")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--student-prob-floor", type=float, default=3.72e-44)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()

    with open(a.corpus) as fh:
        corpus = json.load(fh)
    chunks = corpus["chunks"] if isinstance(corpus, dict) and "chunks" in corpus else corpus
    ids = [c["ids"] if isinstance(c, dict) else c for c in chunks[:a.chunks]]

    keep = torch.arange(0, 511, a.stride) if a.stride else None
    n_pos = (len(keep) if keep is not None else 511) * len(ids)
    print(f"{len(ids)} chunks x {len(keep) if keep is not None else 511} scored positions "
          f"= {n_pos} (the served decode set is 8 x 32 = 256)")

    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and "lm_head" not in n]
    print(f"{len(targets)} Linears carry the arm (the served checkpoint's 112 fused "
          f"modules are these, unfused; a_scale is per TOKEN so fusing does not move it)")

    state = {"arm": "clean", "gen": torch.Generator().manual_seed(a.seed),
             "act_quant": a.act_quant == "on", "rel": a.rel}
    with torch.no_grad():
        for name, mod in targets:
            w = mod.weight.data
            if a.weight_bits:
                lvl = float(2 ** (a.weight_bits - 1))
                sc = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / lvl
                w = (w / sc).round().clamp(-lvl, lvl - 1) * sc
            wq, w_scale = row_e4m3(w)
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            setattr(parent, name.rsplit(".", 1)[1],
                    ArmLinear(wq, w_scale, mod.bias.data if mod.bias is not None else None, state))

    def run(arm, seq):
        state["arm"] = arm
        x = torch.tensor([seq[:512]], dtype=torch.long)
        with torch.no_grad():
            return model(x).logits[0, :-1].float()

    def run_decode(seq, positions):
        """The served regime: one clean prefill, then one M=1 forward per position.

        The cache is built in arm B's arithmetic because that is what the
        serve's warm-up prefill runs -- ``_materialised_path``, in BOTH arms.
        Each scored prefix of length L then re-forwards ONLY token L-1, off a
        cache cropped to L-1, exactly as the serve's decode request does.
        """
        from transformers import DynamicCache
        x = torch.tensor([seq[:512]], dtype=torch.long)
        state["arm"] = "scaled_mm"
        with torch.no_grad():
            pre = model(x, use_cache=True)
        kv = [(lyr.keys.clone(), lyr.values.clone()) for lyr in pre.past_key_values.layers]
        out = {}
        for arm in (a.arm_a, "scaled_mm"):
            state["arm"] = arm
            got = []
            for L in positions:                      # L = prefix length, scores token L
                cache = DynamicCache()
                with torch.no_grad():
                    for li, (k, v) in enumerate(kv):
                        cache.update(k[:, :, :L - 1].clone(), v[:, :, :L - 1].clone(), li)
                    step = model(x[:, L - 1:L], past_key_values=cache, use_cache=True)
                got.append(step.logits[0, -1].float())
            out[arm] = torch.stack(got)
        return out[a.arm_a], out["scaled_mm"]

    # Prefix lengths the served decode regime scores: 1, 1+stride, ...  A prefix
    # of length L is scored on prefill logits row L-1, which is what `keep` picks.
    positions = [int(k) + 1 for k in (keep if keep is not None else torch.arange(511))]

    rows = []
    for i, seq in enumerate(ids):
        if a.regime == "decode":
            lg_a, lg_b = run_decode(seq, positions)
        else:
            lg_a = run(a.arm_a, seq)
            lg_b = run("scaled_mm", seq)
            if keep is not None:
                lg_a, lg_b = lg_a[keep], lg_b[keep]
        kl, top1 = full_vocab_kl(lg_a, lg_b)
        est = served_estimator(lg_a, lg_b, a.topk, a.student_prob_floor)
        lower = sum(e["kl_lower"] for e in est) / len(est)
        upper = sum(e["kl_lower"] + e["gap_upper"] for e in est) / len(est)
        cov = min(e["coverage"] for e in est)
        rows.append({"chunk": i, "positions": int(lg_a.shape[0]),
                     "kl_full_mean": float(kl.mean()), "kl_full_median": float(kl.median()),
                     "top1_agree": top1, "kl_lower_mean": lower, "kl_upper_mean": upper,
                     "topk_coverage_min": cov})
        print(f"chunk {i}: n={int(lg_a.shape[0])}  full-vocab KL {float(kl.mean()):.6f}  "
              f"served-shape KL >= {lower:.6f}  top-1 {top1:.2%}  cov_min {cov:.3f}")

    # The harness must read exactly zero when nothing varies.
    zero = run("scaled_mm", ids[0])
    zero2 = run("scaled_mm", ids[0])
    ctrl_kl, ctrl_top1 = full_vocab_kl(zero, zero2)

    fold_rel = [r for m in model.modules() if isinstance(m, ArmLinear) for r in m.fold_rel]
    agg = {"model": a.model, "corpus": a.corpus, "stride": a.stride, "arm_a": a.arm_a,
           "act_quant": a.act_quant, "regime": a.regime,
           "per_element_rel_rms": (sum(fold_rel) / len(fold_rel)) if fold_rel else None,
           "weight_bits": a.weight_bits, "topk": a.topk, "chunks": rows,
           "positions_total": sum(r["positions"] for r in rows),
           "kl_full_mean": sum(r["kl_full_mean"] for r in rows) / len(rows),
           "kl_lower_mean": sum(r["kl_lower_mean"] for r in rows) / len(rows),
           "kl_upper_mean": sum(r["kl_upper_mean"] for r in rows) / len(rows),
           "top1_agree": sum(r["top1_agree"] for r in rows) / len(rows),
           "control_same_arm_kl": float(ctrl_kl.mean()),
           "control_same_arm_top1": ctrl_top1}
    print(f"\nALL {agg['positions_total']} positions: full-vocab KL {agg['kl_full_mean']:.6f}  "
          f"served-shape KL >= {agg['kl_lower_mean']:.6f} (<= {agg['kl_upper_mean']:.6f})  "
          f"top-1 {agg['top1_agree']:.2%}")
    print(f"control (same arm twice): KL {agg['control_same_arm_kl']:.3e}  "
          f"top-1 {agg['control_same_arm_top1']:.2%}")
    if agg["per_element_rel_rms"] is not None:
        print(f"arm A = {a.arm_a}, per-element relative rms {agg['per_element_rel_rms']:.4e} "
              f"(both arms of the folded/gaussian pair match on this by construction)")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(agg, fh, indent=2)
        print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
