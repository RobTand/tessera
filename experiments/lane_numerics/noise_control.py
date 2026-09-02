#!/usr/bin/env python3
"""Sensitivity control: multiplicative noise of relative std eps at every quantized
Linear output of the STOCK model; final hidden states per chunk per eps -> npz.
usage: noise_control.py <checkpoint> <out.npz> [eps ...]"""
import sys, os, json, numpy as np
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch
from vllm import LLM, SamplingParams
ck, out = sys.argv[1], sys.argv[2]
epss = [float(e) for e in sys.argv[3:]] or [0.0, 1e-3, 2.3e-3, 5e-3]
contract = json.load(open("/mnt/shared/tessera-kl/corpus_qwen_n8_s512.json"))
chunks = [list(c) for c in contract["chunks"]]
llm = LLM(model=ck, enforce_eager=True, max_model_len=1024, gpu_memory_utilization=0.4, seed=0)
STATE = {"eps": 0.0, "gen": None, "final": None}
def install(model):
    def noise(mod, inp, outp):
        if STATE["eps"] == 0.0: return None
        y = outp[0] if isinstance(outp, tuple) else outp
        n = torch.randn(y.shape, device=y.device, dtype=torch.float32, generator=STATE["gen"])
        y2 = (y.float() * (1.0 + STATE["eps"] * n)).to(y.dtype)
        return (y2,) + tuple(outp[1:]) if isinstance(outp, tuple) else y2
    def keep(mod, inp, outp):
        if STATE["final"] is None:
            t = outp[0] if isinstance(outp, tuple) else outp
            STATE["final"] = t.detach().float().cpu().numpy()
    k = 0
    for name, mod in model.named_modules():
        if name.split(".")[-1] in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
            mod.register_forward_hook(noise); k += 1
        elif name == "model.norm":
            mod.register_forward_hook(keep)
    STATE["gen"] = torch.Generator(device="cuda"); STATE["gen"].manual_seed(1234)
    return k
print("hooked:", llm.apply_model(install)[0])
res = {}
for eps in epss:
    for ci, ids in enumerate(chunks):
        def setup(model, eps=eps):
            STATE["eps"] = eps; STATE["final"] = None; return 0
        llm.apply_model(setup)
        llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
        res[f"eps{eps:g}:chunk{ci}"] = llm.apply_model(lambda m: STATE["final"])[0]
    print(f"eps={eps:g} done")
np.savez(out, **res); print("->", out, len(res))
