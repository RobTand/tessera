"""Rate-distortion curve of the k-tuple trellis vs scalar NVFP4.

A k-tuple's squared error is additive over its positions, and the subset is
(sum of value-order ranks) mod 4, so the per-subset best tuple is a 4-state DP
over k positions with 16 branches each -- no 16^k alphabet is ever built.
"""
import torch, sys, os, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tessera.alphabet import value_order
from tessera.encode import _pack_scales, e2m1_value_table
from tessera.trellis import ConvCode

dev='cuda'; torch.manual_seed(0)
rows, cols = 1024, 1024
W = torch.randn(rows, cols, device=dev)*0.02
_,_,eff = _pack_scales(W.float(),32,16)
scale = torch.repeat_interleave(eff,16).reshape(W.shape)
v = e2m1_value_table(dev); T = W.float()/scale
order = value_order(); rank = torch.zeros(16, dtype=torch.long, device=dev)
for i,c in enumerate(order): rank[c]=i
def rel(R): return (torch.linalg.norm(W-R)/torch.linalg.norm(W)).item()

cc = ConvCode(memory=6); S = cc.states
prev = torch.zeros(2,S,dtype=torch.long,device=dev); sof = torch.zeros(2,S,dtype=torch.long,device=dev)
tn = torch.zeros(2,S,dtype=torch.long,device=dev); ts = torch.zeros(2,S,dtype=torch.long,device=dev)
fill=[0]*S
for s in range(S):
    for b in (0,1):
        nx,sb = cc.step(s,b); tn[b,s]=nx; ts[b,s]=sb
        prev[fill[nx],nx]=s; sof[fill[nx],nx]=sb; fill[nx]+=1

def run(k):
    """rate = (4k-1)/k payload bits per position."""
    steps = rows//k
    Tk = T[:steps*k].reshape(steps, k, cols)
    cost = torch.full((cols,S), float('inf'), device=dev); cost[:,0]=0.0
    choice = torch.zeros(steps, cols, S, dtype=torch.bool, device=dev)
    keep = torch.zeros(steps, cols, 4, k, dtype=torch.uint8, device=dev)
    for i in range(steps):
        # 4-state DP over the k positions, state = running rank-sum mod 4
        dp = torch.full((cols,4), float('inf'), device=dev); dp[:,0]=0.0
        bp = torch.zeros(k, cols, 4, dtype=torch.uint8, device=dev)
        for j in range(k):
            e = (Tk[i,j].unsqueeze(1)-v.unsqueeze(0))**2          # [cols,16]
            nd = torch.full((cols,4), float('inf'), device=dev)
            nb = torch.zeros(cols,4, dtype=torch.uint8, device=dev)
            for c in range(16):
                tgt = (torch.arange(4,device=dev)+rank[c]) % 4     # from-state -> to-state
                cand = dp + e[:,c:c+1]                             # [cols,4] indexed by from
                idx = tgt.argsort()
                cand_r = cand[:, idx]                              # reorder to to-state
                better = cand_r < nd
                nd = torch.where(better, cand_r, nd)
                nb = torch.where(better, torch.full_like(nb,c), nb)
            dp, bp[j] = nd, nb
        # backtrack the DP per subset
        for sres in range(4):
            st = torch.full((cols,), sres, dtype=torch.long, device=dev)
            for j in range(k-1,-1,-1):
                c = bp[j].gather(1, st.unsqueeze(1)).squeeze(1)
                keep[i,:,sres,j] = c
                st = (st - rank[c.long()]) % 4
        best = dp
        br = torch.stack([cost[:,prev[m]] + best[:,sof[m]] for m in (0,1)])
        cost, taken = br.min(dim=0); choice[i]=taken.bool()
    end = cost.argmin(1); state = end; col = torch.arange(cols, device=dev)
    out = torch.zeros(steps, k, cols, dtype=torch.long, device=dev)
    for i in range(steps-1,-1,-1):
        side = choice[i][col,state].long(); sb = sof[side,state]
        out[i] = keep[i][col,sb].long().T
        state = prev[side,state]
    rec = torch.zeros_like(T); rec[:steps*k] = out.reshape(steps*k, cols)
    return rel(v[rec.long()]*scale)

nv = rel(v[((T.unsqueeze(-1)-v)**2).argmin(-1)]*scale)
print(f"{'k':>3}{'payload':>9}{'total bpp':>11}{'rel_err':>10}{'scalar@same':>13}{'gain dB':>9}{'== scalar bpp':>15}")
print(f"{'-':>3}{4.0:>9.3f}{4.5:>11.3f}{nv:>10.5f}{nv:>13.5f}{0.0:>9.2f}{4.5:>15.3f}")
for k in (1,2,4,8,16):
    r = (4*k-1)/k
    e = run(k)
    scalar = nv * 2**(4.0-r)
    gain = 20*math.log10(scalar/e)
    print(f"{k:>3}{r:>9.3f}{r+0.5:>11.3f}{e:>10.5f}{scalar:>13.5f}{gain:>9.2f}{4.0-math.log2(e/nv)+0.5:>15.3f}")
