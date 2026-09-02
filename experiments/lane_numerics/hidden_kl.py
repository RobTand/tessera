#!/usr/bin/env python3
"""Exact full-vocab KL between forwards from their final hidden states.
usage: hidden_kl.py noise_stock.npz [layers_stock.npz layers_gridbook.npz]"""
import sys, glob, re, numpy as np, torch
from safetensors import safe_open
CK = "/home/rob/tessera-runs/stock/qwen3-0.6b-tessera-k2-q896-nvfp4"
E = None
for fn in glob.glob(f"{CK}/*.safetensors"):
    with safe_open(fn, "pt") as f:
        for k in f.keys():
            if k == "lm_head.weight" or (E is None and k == "model.embed_tokens.weight"):
                E = f.get_tensor(k)
E = E.cuda().float()
def logp(h):  # h: [T, H] float32 numpy -> log-softmax [T, V] on GPU (chunked)
    h = torch.from_numpy(h).cuda()
    return torch.log_softmax(h @ E.t(), dim=-1)
def kl(lp_p, lp_q):  # mean over positions of KL(p||q), positions 1.. (teacher-style: predict next token at every position)
    return float((lp_p.exp() * (lp_p - lp_q)).sum(-1).mean()), float((lp_p.exp() * (lp_p - lp_q)).sum(-1).max())
def top1(lp_p, lp_q): return float((lp_p.argmax(-1) == lp_q.argmax(-1)).float().mean())
N = np.load(sys.argv[1])
epss = sorted({float(re.match(r"eps([0-9.e-]+):", k).group(1)) for k in N.files})
chunks = sorted({int(re.search(r"chunk(\d+)", k).group(1)) for k in N.files})
ref = {c: logp(N[f"eps0:chunk{c}"]) for c in chunks}
print(f"{'eps':>8s} {'KL(ref||noisy) mean':>20s} {'max-pos':>9s} {'top-1 agree':>12s}")
for e in epss:
    if e == 0: continue
    ks, ms, ts = [], [], []
    for c in chunks:
        q = logp(N[f"eps{e:g}:chunk{c}"]); m, mx = kl(ref[c], q); ks.append(m); ms.append(mx); ts.append(top1(ref[c], q))
    print(f"{e:8.4g} {np.mean(ks):20.4f} {max(ms):9.3f} {100*np.mean(ts):11.2f}%")
if len(sys.argv) > 3:
    A = np.load(sys.argv[2]); B = np.load(sys.argv[3])
    a = logp(A["model.norm:out"]); b = logp(B["model.norm:out"])
    m, mx = kl(a, b); print(f"\nchunk 0 exact KL(stock||gridbook) = {m:.4f}  max-pos {mx:.3f}  top-1 agree {100*top1(a,b):.2f}%")
    m2, _ = kl(a, ref[0]); print(f"chunk 0 exact KL(stock layer-dump || stock noise-run eps=0) = {m2:.6f}  (same forward twice)")
