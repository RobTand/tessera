#!/usr/bin/env python3
"""How far a 1.6e-3 per-Linear perturbation travels to the logits (#110).

SUPERSEDED AS AN ANSWER, KEPT AS A CONTROL.  This script prices the fold with
a noise MODEL -- independent Gaussians of the measured relative size -- on a
trajectory that carries NO per-token FP8 activation quantiser, because it
perturbs the BF16 teacher (or a weight-degraded copy of it).  Both of those
choices matter, and the second one dominates: put the route's per-token E4M3
quantiser back and the SAME Gaussian, same rms, same regime, same 256
positions, goes from KL >= 0.000113 at 99.22% top-1 to the 0.0xx band -- one
to two orders of magnitude, measured by
`gemv_a_side_exact_fold.py --act-quant on|off`.  So the "factor of forty"
this script contributed to the #110 receipt was an artefact of pricing the
term off the operating point, not evidence of a second term.  Read it as the
control it now is; read `gemv_a_side_exact_fold.py` for the number.

What it does: run the model over the corpus chunks twice, once with
independent Gaussian noise of `--rel` relative rms injected at every served
Linear (at the INPUT by default -- the fold's own injection point -- with
`--where output` as a coarser cross-check), and report full-vocab KL and
top-1 agreement.  `--stride 16` restricts scoring to the served decode
regime's position set; `--degrade-bits` reaches a quantised operating point on
the WEIGHT side only, which is the axis that turned out not to be the one that
mattered.

`gemv_a_side_precision.py` prices the arithmetic this stands in for: folding
the per-token FP8 scale into the GEMV's bf16 operand perturbs every activation
element by ~1.6e-3 relative rms -- one bf16 rounding -- against a MEASURED
fp32 reduction error of 1.7e-07.

THIS IS A SCREEN, NOT A SERVED RESULT.  The lane's own served number is the
decode-regime A/B (`experiments/decode_regime_campaign.sh`).

CPU-only.  It runs locally rather than through the PrismaBuild pool: a CPU-only
submission from a box-local checkout is pinned to that box, and on 2026-09-04
sparky offered 10 cpu tokens but zero free mem_gb -- every one of its 48 was
held by one long GPU action -- so `pbrun --demand cpu=2[,mem_gb=0]` queued and
never placed (twice, 120 s and 150 s).  The earlier note here, that sparky's
offer carried no cpu capacity at all, was true when it was written and is not
the reason now.
"""

from __future__ import annotations

import argparse
import json
import os

import torch


def kl_and_top1(clean: torch.Tensor, noisy: torch.Tensor, keep=None):
    """Full-vocab KL(clean || noisy) per position, and top-1 agreement.

    ``keep`` restricts the scored positions.  The served decode regime scores a
    STRIDED subset (prefix lengths 1, 17, 33, ...), not every position, and that
    subset is deliberately weighted towards short prefixes where the next token
    is barely determined -- so the position set is part of the metric, not a
    sampling convenience.
    """
    if keep is not None:
        clean, noisy = clean[keep], noisy[keep]
    lp_c = torch.log_softmax(clean.double(), dim=-1)
    lp_n = torch.log_softmax(noisy.double(), dim=-1)
    kl = (lp_c.exp() * (lp_c - lp_n)).sum(-1)
    top1 = (clean.argmax(-1) == noisy.argmax(-1)).double().mean()
    return float(kl.mean()), float(kl.median()), float(top1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/rob/models/Qwen3-0.6B")
    ap.add_argument("--corpus", default="/mnt/shared/tessera-kl/corpus_qwen_n8_s512.json")
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--rel", type=float, default=1.612e-3,
                    help="relative rms perturbation per Linear output (the measured fold)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--where", choices=("input", "output"), default="input",
                    help="the fold rounds the Linear's INPUT (every activation element), so "
                         "that is the faithful injection point; 'output' is the coarser proxy")
    ap.add_argument("--degrade-bits", type=int, default=0,
                    help="RTN the target weights to this many bits per output row first, so "
                         "the sensitivity is measured at a QUANTISED operating point rather "
                         "than on the BF16 teacher (0 = no degradation)")
    ap.add_argument("--stride", type=int, default=0,
                    help="score only the served decode regime's positions: prefix lengths "
                         "1, 1+stride, 1+2*stride, ... (stride 16 = the serve's KV block "
                         "size, the set `decode_regime_kl.sh` scores).  0 = every position, "
                         "which is what a prefill dump scores.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()

    with open(a.corpus) as fh:
        corpus = json.load(fh)
    chunks = corpus["chunks"] if isinstance(corpus, dict) and "chunks" in corpus else corpus
    ids = [c["ids"] if isinstance(c, dict) else c for c in chunks[:a.chunks]]

    # A prefix of length L is scored on logits row L-1 (row i predicts token i+1).
    keep = (torch.arange(0, 511, a.stride) if a.stride else None)
    if keep is not None:
        print(f"scoring {len(keep)} strided positions per chunk (prefix lengths "
              f"{1}, {1 + a.stride}, ... {1 + a.stride * (len(keep) - 1)}), "
              f"the served decode regime's set")

    # Every Linear the Tessera FP8 route serves: the decoder blocks' projections.
    # lm_head is excluded -- it is not on the route (`vllm-lm-head-embed-legality`).
    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and "lm_head" not in n]
    print(f"{len(targets)} Linears carry the perturbation")

    teacher_gap = None
    if a.degrade_bits:
        # The served arms are a quantised model, and a degraded model need not
        # amplify a perturbation the way the teacher does.  Per-output-row RTN
        # is not Tessera's encoder; it is a way to reach a comparable operating
        # point (its own KL from the teacher is reported) so the SENSITIVITY can
        # be compared at one.
        lvl = float(2 ** (a.degrade_bits - 1))
        clean_ref = []
        for seq in ids:
            x = torch.tensor([seq[:512]], dtype=torch.long)
            with torch.no_grad():
                clean_ref.append(model(x).logits[0, :-1].float())
        with torch.no_grad():
            for n, m in targets:
                w = m.weight.data
                sc = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / lvl
                m.weight.data = (w / sc).round().clamp(-lvl, lvl - 1) * sc
        gaps = []
        for seq, ref in zip(ids, clean_ref):
            x = torch.tensor([seq[:512]], dtype=torch.long)
            with torch.no_grad():
                gaps.append(kl_and_top1(ref, model(x).logits[0, :-1].float(), keep)[0])
        teacher_gap = sum(gaps) / len(gaps)
        print(f"degraded to {a.degrade_bits} bits per row: KL from the BF16 teacher "
              f"{teacher_gap:.6f}")

    gen = torch.Generator().manual_seed(a.seed)
    noise_on = {"v": False}

    def perturb(t):
        rms = t.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return t + torch.randn(t.shape, generator=gen) * (a.rel * rms)

    def post_hook(_mod, _inp, out):
        return out if not noise_on["v"] else perturb(out)

    def pre_hook(_mod, inp):
        return inp if not noise_on["v"] else (perturb(inp[0]),) + tuple(inp[1:])

    for _n, m in targets:
        if a.where == "output":
            m.register_forward_hook(post_hook)
        else:
            m.register_forward_pre_hook(pre_hook)

    rows = []
    for i, seq in enumerate(ids):
        x = torch.tensor([seq[:512]], dtype=torch.long)
        noise_on["v"] = False
        with torch.no_grad():
            clean = model(x).logits[0, :-1].float()
        noise_on["v"] = True
        with torch.no_grad():
            noisy = model(x).logits[0, :-1].float()
        mean, med, top1 = kl_and_top1(clean, noisy, keep)
        n_scored = int(clean.shape[0]) if keep is None else int(len(keep))
        rows.append({"chunk": i, "positions": n_scored,
                     "kl_mean": mean, "kl_median": med, "top1_agree": top1})
        print(f"chunk {i}: positions={n_scored}  KL mean={mean:.6f} "
              f"median={med:.6f}  top-1 {top1:.2%}")

    agg = {"rel": a.rel, "model": a.model, "where": a.where, "stride": a.stride,
           "chunks": rows,
           "degrade_bits": a.degrade_bits, "degraded_teacher_kl": teacher_gap,
           "kl_mean": sum(r["kl_mean"] for r in rows) / len(rows),
           "top1_agree": sum(r["top1_agree"] for r in rows) / len(rows)}
    print(f"ALL: KL mean {agg['kl_mean']:.6f}  top-1 {agg['top1_agree']:.2%}  "
          f"at rel={a.rel:.3e} per Linear {a.where}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(agg, fh, indent=2)
        print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
