#!/usr/bin/env python3
"""Where do two arms' forwards part?  usage: layer_compare.py a.npz b.npz"""
import sys, re
import numpy as np
A = np.load(sys.argv[1]); B = np.load(sys.argv[2])
keys = [k for k in A.files if k in B.files]
def rel(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    d = np.linalg.norm(a - b); n = np.linalg.norm(a)
    rowrel = np.linalg.norm(a - b, axis=-1) / (np.linalg.norm(a, axis=-1) + 1e-30)
    return d / (n + 1e-30), rowrel.max(), float(np.median(rowrel))
def layer_no(k):
    m = re.search(r"layers\.(\d+)", k); return int(m.group(1)) if m else -1
rows = []
for k in keys:
    if A[k].shape != B[k].shape: print("shape mismatch", k, A[k].shape, B[k].shape); continue
    r, rmax, rmed = rel(A[k], B[k]); rows.append((layer_no(k), k, r, rmax, rmed))
rows.sort(key=lambda t: (t[0], t[1]))
print(f"{'tensor':44s} {'rel_fro':>10s} {'row_max':>9s} {'row_med':>9s}")
for L, k, r, rmax, rmed in rows:
    print(f"{k:44s} {r:10.3e} {rmax:9.3e} {rmed:9.3e}")
