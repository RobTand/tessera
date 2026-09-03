#!/usr/bin/env python
"""Full input Hessians ``H = E[x x^T]`` per Linear, on text disjoint from the KL corpus.

LDLQ needs the whole Hessian, not the diagonal ``h_diag.pt`` carries, and it
needs it from text the served comparison does not also grade on.  The KL corpus
(``/mnt/shared/tessera-kl/corpus_qwen_n8_s512.json``) is wikitext-2 **test**;
this captures on wikitext-2-raw-v1 **train**, two disjoint slices:

* ``--fit-tokens`` build ``H`` (the LDL factor and the refit metric), and
* ``--eval-tokens`` from a later offset are saved as raw activations for the
  units named by ``--eval-units``, so a weight-space sweep can be scored on
  rows nothing was fit on.  The Gridbook LDLQ measurement that regressed every
  rung had a hold-out half of which was its own fit rows; that is the mistake
  this split exists to not repeat.

fp32 accumulation throughout: ``x^T x`` over 16k tokens in bf16 loses the small
off-diagonals LDLQ reads.

    PYTHONPATH=src TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache \
      python experiments/capture_h_full.py
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def wikitext_train_text(min_chars: int) -> "tuple[str, str]":
    """The local wikitext-2-raw-v1 TRAIN split as one string, plus its source id."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    parts, total = [], 0
    for row in ds:
        line = row["text"]
        parts.append(line)
        total += len(line)
        if total >= min_chars:
            break
    return "".join(parts), "wikitext-2-raw-v1/train (local datasets cache)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/rob/models/Qwen3-0.6B")
    ap.add_argument("--out", default="/home/rob/tessera-runs/ldlq/h_full_qwen06b.pt")
    ap.add_argument("--acts-out", default="/home/rob/tessera-runs/ldlq/x_eval_qwen06b.pt")
    ap.add_argument("--eval-h-out", default=None,
                    help="also accumulate the HELD-OUT second moment "
                         "``x_eval^T x_eval / n`` for EVERY Linear and write it "
                         "here. ``tr(E H_eval E^T)`` is ``|X_eval E^T|^2`` exactly, "
                         "so an out-space score over the whole roster needs this "
                         "and not 196 activation dumps; per-row it is "
                         "``e_r H_eval e_r^T``, which is what a per-row census reads")
    ap.add_argument("--fit-tokens", type=int, default=16384)
    ap.add_argument("--eval-tokens", type=int, default=8192)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--eval-units", nargs="*", default=[
        "model.layers.2.mlp.down_proj",
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.self_attn.k_proj",
        "model.layers.13.mlp.down_proj",
        "model.layers.14.mlp.gate_proj",
        "model.layers.27.self_attn.o_proj",
    ])
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    need = a.fit_tokens + a.eval_tokens
    text, source = wikitext_train_text(min_chars=need * 12)
    ids = tok(text, return_tensors="pt").input_ids[0]
    if ids.numel() < need:
        raise SystemExit(f"only {ids.numel()} tokens of {source}; need {need}")
    fit = ids[: a.fit_tokens].reshape(-1, a.seqlen)
    ev = ids[a.fit_tokens : a.fit_tokens + a.eval_tokens].reshape(-1, a.seqlen)
    text_sha = hashlib.sha256(text.encode()).hexdigest()
    fit_sha = hashlib.sha256(fit.numpy().tobytes()).hexdigest()
    ev_sha = hashlib.sha256(ev.numpy().tobytes()).hexdigest()
    print(f"source {source}\n  chars {len(text)} sha {text_sha[:16]}")
    print(f"  fit {fit.numel()} tokens sha {fit_sha[:16]}   "
          f"eval {ev.numel()} tokens sha {ev_sha[:16]}")

    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16).cuda().eval()
    targets = {
        n: m for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and "proj" in n and "layers." in n
    }
    print(f"  {len(targets)} Linears")

    acc: "dict[str, torch.Tensor]" = {}
    counts: "dict[str, int]" = {}
    acc_ev: "dict[str, torch.Tensor]" = {}
    counts_ev: "dict[str, int]" = {}
    keep: "dict[str, list]" = {u: [] for u in a.eval_units}
    mode = {"phase": "fit"}

    def hook(name):
        def fn(mod, inp, out):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            if mode["phase"] == "fit":
                if name not in acc:
                    acc[name] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
                acc[name].addmm_(x.T, x)
                counts[name] = counts.get(name, 0) + x.shape[0]
            else:
                if name in keep:
                    keep[name].append(x.to(torch.float16).cpu())
                if a.eval_h_out:
                    if name not in acc_ev:
                        acc_ev[name] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
                    acc_ev[name].addmm_(x.T, x)
                    counts_ev[name] = counts_ev.get(name, 0) + x.shape[0]
        return fn

    handles = [m.register_forward_hook(hook(n)) for n, m in targets.items()]
    with torch.no_grad():
        for phase, rows in (("fit", fit), ("eval", ev)):
            mode["phase"] = phase
            for row in rows:
                model(row[None].cuda())
    for h in handles:
        h.remove()

    payload = {
        "H": {n: (acc[n] / counts[n]).cpu() for n in acc},
        "counts": counts,
        "provenance": {
            "source": source, "text_sha256": text_sha, "text_chars": len(text),
            "model": a.model, "seqlen": a.seqlen,
            "fit_tokens": int(fit.numel()), "fit_ids_sha256": fit_sha,
            "eval_tokens": int(ev.numel()), "eval_ids_sha256": ev_sha,
            "kl_corpus_split": "wikitext-2 TEST -- disjoint from this TRAIN capture",
            "dtype": "fp32 accumulation of x^T x, divided by the token count",
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, a.out)
    print(f"wrote {a.out}  ({len(payload['H'])} Hessians)")
    if any(keep.values()):
        torch.save({"x": {n: torch.cat(v) for n, v in keep.items() if v},
                    "provenance": payload["provenance"]}, a.acts_out)
        print(f"wrote {a.acts_out}  ({sum(1 for v in keep.values() if v)} units)")
    if a.eval_h_out:
        # Same provenance block: the identity fields name the FIT rows this
        # matrix is disjoint from, and ``eval_ids_sha256`` names its own rows.
        torch.save({"H": {n: (acc_ev[n] / counts_ev[n]).cpu() for n in acc_ev},
                    "counts": counts_ev,
                    "provenance": {**payload["provenance"],
                                   "role": "HELD-OUT second moment x_eval^T x_eval / n; "
                                           "scores, never fits"}},
                   a.eval_h_out)
        print(f"wrote {a.eval_h_out}  ({len(acc_ev)} held-out Hessians)")
    ratios = sorted(float(H.diagonal().max() / H.diagonal().median())
                    for H in payload["H"].values())
    print(f"  diag max/median: median {ratios[len(ratios) // 2]:.0f}  max {ratios[-1]:.0f}")


if __name__ == "__main__":
    main()
