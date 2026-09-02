#!/usr/bin/env python3
"""Dump per-layer hidden states and per-Linear inputs/outputs of one prefill.

usage: layer_dump.py <checkpoint> <out.npz> [--chunk 0]
Runs the Qwen contract's chunk through vllm.LLM in-process (eager), hooks every
decoder layer and every quantized Linear (qkv_proj, o_proj, gate_up_proj,
down_proj), and saves the first (prefill) call's tensors as float32.
"""
import argparse, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("out")
ap.add_argument("--chunk", type=int, default=0)
ap.add_argument("--contract", default="/mnt/shared/tessera-kl/corpus_qwen_n8_s512.json")
ap.add_argument("--gpu-memory-utilization", type=float, default=0.4)
args = ap.parse_args()
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
from vllm import LLM, SamplingParams

contract = json.load(open(args.contract))
ids = list(contract["chunks"][args.chunk])

llm = LLM(model=args.model, enforce_eager=True, max_model_len=1024,
          gpu_memory_utilization=args.gpu_memory_utilization, seed=0)

captured = {}
def install(model):
    def keep(name, which):
        def hook(mod, inp, out):
            if name + ":" + which in captured:
                return
            t = out
            if isinstance(t, tuple):
                t = t[0] if which == "out" else t
            if which == "in":
                t = inp[0]
            if isinstance(t, tuple):
                for i, u in enumerate(t):
                    if torch.is_tensor(u):
                        captured[f"{name}:{which}{i}"] = u.detach().float().cpu().numpy()
            elif torch.is_tensor(t):
                captured[f"{name}:{which}"] = t.detach().float().cpu().numpy()
        return hook
    n = 0
    for name, mod in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
            mod.register_forward_hook(keep(name, "in")); mod.register_forward_hook(keep(name, "out")); n += 1
        elif name.startswith("model.layers.") and name.count(".") == 2:
            mod.register_forward_hook(keep(name, "out")); n += 1
        elif leaf in ("embed_tokens", "norm", "lm_head"):
            mod.register_forward_hook(keep(name, "out")); n += 1
    return n

print("hooked modules:", llm.apply_model(install)[0])
llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0.0))
np.savez(args.out, **captured)
print(f"saved {len(captured)} tensors -> {args.out}")
for k in list(captured)[:6]:
    print(" ", k, captured[k].shape)
