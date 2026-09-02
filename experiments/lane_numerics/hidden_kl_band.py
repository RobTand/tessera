#!/usr/bin/env python3
"""KL-vs-BF16 band under bf16-level noise.

usage: hidden_kl_band.py noise_teacher.npz noise_stock.npz [layers_stock.npz layers_gridbook.npz]

noise_teacher.npz holds the BF16 model's final-norm output (eps=0 only);
noise_stock.npz holds the stock W4A4 arm under output noise eps in {0, ...}.
Prints exact full-vocab KL(teacher || stock_eps) per eps: the width of that
band is what "reproduces the stock arm within the kernel's floor" means on
the KL-vs-BF16 axis.  Each side is projected through its own lm_head.
"""
import os, sys, re, glob, numpy as np, torch
DEV = "cpu" if os.environ.get("BAND_CPU") == "1" or not torch.cuda.is_available() else "cuda"
from safetensors import safe_open
TEACHER = "/home/rob/models/Qwen3-0.6B"
STUDENT = "/home/rob/tessera-runs/stock/qwen3-0.6b-tessera-k2-q896-nvfp4"

def head(ck):
    E = None
    for fn in glob.glob(f"{ck}/*.safetensors"):
        with safe_open(fn, "pt") as f:
            for k in f.keys():
                if k == "lm_head.weight" or (E is None and k == "model.embed_tokens.weight"):
                    E = f.get_tensor(k)
    return E.to(DEV).float()

def logp(h, E):
    return torch.log_softmax(torch.from_numpy(h).to(DEV) @ E.t(), dim=-1)

def kl(p, q):
    per = (p.exp() * (p - q)).sum(-1)
    return float(per.mean()), float(per.max())

def top1(p, q):
    return float((p.argmax(-1) == q.argmax(-1)).float().mean())

Et, Es = head(TEACHER), head(STUDENT)
print(f"teacher/student head identical: {bool(torch.equal(Et, Es))}")
T = np.load(sys.argv[1]); S = np.load(sys.argv[2])
chunks = sorted({int(re.search(r"chunk(\d+)", k).group(1)) for k in T.files if k.startswith("eps0:")})
ref = {c: logp(T[f"eps0:chunk{c}"], Et) for c in chunks}
epss = sorted({float(re.match(r"eps([0-9.e-]+):", k).group(1)) for k in S.files})
print(f"{'eps':>8s} {'KL(BF16||stock_eps)':>20s} {'per-chunk std':>13s} {'max-pos':>9s} {'top-1':>8s}")
base, deltas = None, []
for e in epss:
    ks, ms, ts = [], [], []
    for c in chunks:
        q = logp(S[f"eps{e:g}:chunk{c}"], Es); m, mx = kl(ref[c], q)
        ks.append(m); ms.append(mx); ts.append(top1(ref[c], q))
    print(f"{e:8.4g} {np.mean(ks):20.4f} {np.std(ks):13.4f} {max(ms):9.3f} {100*np.mean(ts):7.2f}%"
          + ("   per-chunk " + " ".join(f"{k:.4f}" for k in ks) if os.environ.get("BAND_PER_CHUNK") == "1" else ""))
    if e == 0: base = ks
    elif base is not None: deltas += [k - b for k, b in zip(ks, base)]
if deltas:
    sd = float(np.std(deltas, ddof=1)); n = len(base)
    print(f"per-chunk delta vs eps=0 over {len(deltas)} (eps,chunk) pairs: std {sd:.4f}; "
          f"implied 1-sigma of the {n}-chunk mean under bf16-level perturbation: {sd/np.sqrt(n):.4f}")
if len(sys.argv) > 4:
    A = np.load(sys.argv[3]); B = np.load(sys.argv[4])
    a = logp(A["model.norm:out"], Es); b = logp(B["model.norm:out"], Es)
    for name, x in (("stock layer-dump", a), ("gridbook layer-dump", b)):
        m, mx = kl(ref[0], x); print(f"chunk 0 exact KL(BF16 || {name}) = {m:.4f}  max-pos {mx:.3f}  top-1 {100*top1(ref[0], x):.2f}%")
