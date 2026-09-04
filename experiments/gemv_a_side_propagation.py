#!/usr/bin/env python3
"""How far a 1.6e-3 per-Linear output perturbation travels to the logits (#110).

`gemv_a_side_precision.py` prices the arithmetic: folding the per-token FP8
scale into the GEMV's bf16 operand perturbs every Linear's output by ~1.6e-3
relative rms, against an fp32 accumulation floor ~800x below that.  What that
is worth *as a KL* is a property of the model, not of the kernel, so this
script measures it on the model directly: the BF16 teacher, forward twice,
once with independent Gaussian noise of the measured relative size injected at
the output of every Linear the Tessera route serves.

THIS IS A SCREEN, NOT A SERVED RESULT.  It uses a noise MODEL of the term, on
the BF16 model, on prefill positions; the lane's own served number is the
decode-regime A/B (`experiments/decode_regime_campaign.sh`).  What it can
settle is magnitude: whether a perturbation of this size plausibly accounts
for a mutual KL of 0.012, or whether the served disagreement needs a second
term.  Both readings are useful and the receipt must say which one it got.

CPU by default -- 0.6B over a couple of chunks -- because the pool cannot
place a CPU-only action from a box-local checkout.
"""
from __future__ import annotations

import argparse
import json
import os

import torch


def kl_and_top1(clean: torch.Tensor, noisy: torch.Tensor):
    """Full-vocab KL(clean || noisy) per position, and top-1 agreement."""
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
    ap.add_argument("--degrade-bits", type=int, default=0,
                    help="RTN the target weights to this many bits per output row first, so "
                         "the sensitivity is measured at a QUANTISED operating point rather "
                         "than on the BF16 teacher (0 = no degradation)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()

    with open(a.corpus) as fh:
        corpus = json.load(fh)
    chunks = corpus["chunks"] if isinstance(corpus, dict) and "chunks" in corpus else corpus
    ids = [c["ids"] if isinstance(c, dict) else c for c in chunks[:a.chunks]]

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
                gaps.append(kl_and_top1(ref, model(x).logits[0, :-1].float())[0])
        teacher_gap = sum(gaps) / len(gaps)
        print(f"degraded to {a.degrade_bits} bits per row: KL from the BF16 teacher "
              f"{teacher_gap:.6f}")

    gen = torch.Generator().manual_seed(a.seed)
    noise_on = {"v": False}

    def hook(_mod, _inp, out):
        if not noise_on["v"]:
            return out
        rms = out.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return out + torch.randn(out.shape, generator=gen) * (a.rel * rms)

    for _n, m in targets:
        m.register_forward_hook(hook)

    rows = []
    for i, seq in enumerate(ids):
        x = torch.tensor([seq[:512]], dtype=torch.long)
        noise_on["v"] = False
        with torch.no_grad():
            clean = model(x).logits[0, :-1].float()
        noise_on["v"] = True
        with torch.no_grad():
            noisy = model(x).logits[0, :-1].float()
        mean, med, top1 = kl_and_top1(clean, noisy)
        rows.append({"chunk": i, "positions": int(clean.shape[0]),
                     "kl_mean": mean, "kl_median": med, "top1_agree": top1})
        print(f"chunk {i}: positions={clean.shape[0]}  KL mean={mean:.6f} "
              f"median={med:.6f}  top-1 {top1:.2%}")

    agg = {"rel": a.rel, "model": a.model, "chunks": rows,
           "degrade_bits": a.degrade_bits, "degraded_teacher_kl": teacher_gap,
           "kl_mean": sum(r["kl_mean"] for r in rows) / len(rows),
           "top1_agree": sum(r["top1_agree"] for r in rows) / len(rows)}
    print(f"ALL: KL mean {agg['kl_mean']:.6f}  top-1 {agg['top1_agree']:.2%}  "
          f"at rel={a.rel:.3e} per Linear")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(agg, fh, indent=2)
        print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
