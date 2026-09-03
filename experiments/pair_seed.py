"""Rate-7 trellis over position PAIRS: 3.5 bits/position, free placement.

|A_R| = 2^(R+1) caps a scalar trellis at R=3 over a 16-code grid.  But a pair
of positions has 256 joint codes = 2^8, so a rate-7 trellis over pairs is
exactly the same construction one level up: 2^(7+1) alphabet, 7 bits spent,
one bit of redundancy -> 3.5 bits/position with coding gain intact.
"""
import torch, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tessera.alphabet import build_forest, value_order, E2M1_VALUES
from tessera.encode import encode_unit, _pack_scales, e2m1_value_table
from tessera.decode import decode_codes, dequantize
from tessera.trellis import ConvCode

import os
torch.manual_seed(int(os.environ.get("SEED","0")))
rows, cols, dev = int(os.environ.get('ROWS','512')), int(os.environ.get('COLS','1024')), 'cuda'
W = torch.randn(rows, cols, device=dev)*0.02
_,_,eff = _pack_scales(W.float(),32,16)
scale = torch.repeat_interleave(eff,16).reshape(W.shape)
v = e2m1_value_table(dev); T = W.float()/scale
def rel(R): return (torch.linalg.norm(W-R)/torch.linalg.norm(W)).item()

# --- pair alphabet: 256 joint codes, subsets by (rank1+rank2) mod 4 ---
order = value_order(); rank = {c:i for i,c in enumerate(order)}
pair_c1 = torch.tensor([c1 for c1 in range(16) for c2 in range(16)], device=dev)
pair_c2 = torch.tensor([c2 for c1 in range(16) for c2 in range(16)], device=dev)
pv1, pv2 = v[pair_c1], v[pair_c2]
sub_id = torch.tensor([(rank[c1]+rank[c2]) % 4 for c1 in range(16) for c2 in range(16)], device=dev)
subsets = [torch.nonzero(sub_id==s).squeeze(1) for s in range(4)]
P = min(len(s) for s in subsets); subsets = torch.stack([s[:P] for s in subsets])
print(f"pair alphabet 256, 4 subsets x {P}; rate = 1 + log2({P}) = {1+P.bit_length()-1} bits/pair"
      f" = {(1+P.bit_length()-1)/2} bits/position")

cc = ConvCode(memory=6); S = cc.states
tn = torch.zeros(2,S,dtype=torch.long,device=dev); ts = torch.zeros(2,S,dtype=torch.long,device=dev)
for s in range(S):
    for b in (0,1):
        nx,sb = cc.step(s,b); tn[b,s]=nx; ts[b,s]=sb
prev = torch.zeros(2,S,dtype=torch.long,device=dev); sof = torch.zeros(2,S,dtype=torch.long,device=dev)
fill=[0]*S
for s in range(S):
    for b in (0,1):
        nx,sb = cc.step(s,b); prev[fill[nx],nx]=s; sof[fill[nx],nx]=sb; fill[nx]+=1

# Viterbi over pairs, down each column
t1 = T[0::2].contiguous(); t2 = T[1::2].contiguous()   # [rows/2, cols]
steps = t1.shape[0]
cost = torch.full((cols,S), float('inf'), device=dev); cost[:,0]=0.0
choice = torch.zeros(steps, cols, S, dtype=torch.bool, device=dev)
picked = torch.zeros(steps, cols, 4, dtype=torch.long, device=dev)
for i in range(steps):
    e = (t1[i].unsqueeze(1)-pv1.unsqueeze(0))**2 + (t2[i].unsqueeze(1)-pv2.unsqueeze(0))**2
    bysub = e[:, subsets.reshape(-1)].reshape(cols,4,P)
    best, pt = bysub.min(dim=2); picked[i]=pt
    br = torch.stack([cost[:,prev[k]] + best[:,sof[k]] for k in (0,1)])
    cost, taken = br.min(dim=0); choice[i]=taken.bool()
end = cost.argmin(1); state = end
col = torch.arange(cols, device=dev)
oc1 = torch.zeros(steps, cols, dtype=torch.long, device=dev); oc2 = torch.zeros_like(oc1)
for i in range(steps-1,-1,-1):
    side = choice[i][col,state].long(); sb = sof[side,state]
    pt = picked[i][col,sb]; idx = subsets[sb,pt]
    oc1[i] = pair_c1[idx]; oc2[i] = pair_c2[idx]
    state = prev[side,state]
rec = torch.zeros_like(T); rec[0::2] = v[oc1]; rec[1::2] = v[oc2]
pair_err = rel(rec*scale)

# --- baselines at matched bpp ---
f = build_forest(3)
u = encode_unit(W, f, (3,)*cols, cc, released_positions=int(0.125*rows*cols))
scalar_rel = rel(dequantize(decode_codes(u,f,cc), scale))
u0 = encode_unit(W, f, (3,)*cols, cc)
print(f"\n{'arm':<44}{'bpp':>7}{'rel_err':>10}")
print(f"{'Tessera R=3, no release':<44}{3.500:>7.3f}{rel(dequantize(decode_codes(u0,f,cc),scale)):>10.5f}")
print(f"{'Tessera R=3 + 12.5% release (S9 order)':<44}{4.000:>7.3f}{scalar_rel:>10.5f}")
print(f"{'PAIR trellis rate-7 (3.5 payload bits)':<44}{4.000:>7.3f}{pair_err:>10.5f}")
tv = ((T.unsqueeze(-1)-v)**2).argmin(-1)
print(f"{'NVFP4 RTN (scalar 4-bit)':<44}{4.500:>7.3f}{rel(v[tv]*scale):>10.5f}")
