"""Rotation on FULL tensors.  The earlier sweep sliced [:512,:2048] and cut the
outliers off: layers.0.linear_attn.out_proj is kurtosis 152 whole and 7.8 sliced.
A screen that removes the phenomenon under test measures nothing."""
import sys, glob, torch
sys.path.insert(0, "/home/rob/tessera/src")
from safetensors import safe_open
from tessera.alphabet import build_forest
from tessera.encode import encode_unit
from tessera.decode import reconstruct_unit
from tessera.trellis import ConvCode
from tessera.manifest import RotationState
from tessera.grammar import bresenham_rate_schedule, root_from_q256

dev="cuda"; CC=ConvCode(memory=6); F={r:build_forest(r) for r in (1,2,3)}
files = sorted(glob.glob("/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/model-*.safetensors"))
WANT = {
  "model.language_model.layers.0.linear_attn.out_proj.weight",
  "model.language_model.layers.3.self_attn.o_proj.weight",
  "model.language_model.layers.1.linear_attn.out_proj.weight",
  "model.language_model.layers.0.mlp.gate_proj.weight",
}
POINTS=[("3.5 bpp",640,0.0),("4.0 bpp",768,0.25),("4.25 bpp",768,0.50)]

def run(W,q256,frac,rot,diag=False):
    rates=bresenham_rate_schedule(root_from_q256(q256),W.shape[1])
    u=encode_unit(W,F,rates,CC,rotation=rot,with_diagonals=diag,
                  released_positions=int(frac*W.numel()))
    return ((reconstruct_unit(u,F,CC)-W).norm()/W.norm()).item()

print(f"{'tensor':<42}{'shape':>14}{'kurt':>9}  " + "".join(f"{p[0]:>20}" for p in POINTS))
seen=set()
for path in files[:6]:
    with safe_open(path,"pt") as f:
        for k in f.keys():
            if k not in WANT or k in seen: continue
            seen.add(k)
            W=f.get_tensor(k).to(dev).float().contiguous()
            if W.shape[1]%256: W=W[:, :W.shape[1]//256*256].contiguous()
            x=W-W.mean(); kurt=((x**4).mean()/(x.var()**2)).item()
            cells=[]
            for _,q,fr in POINTS:
                a=run(W,q,fr,RotationState.NONE)
                b=run(W,q,fr,RotationState.R_IN_ONLY)
                cells.append(f"{a:.4f}->{100*(a-b)/a:+6.2f}%")
            print(f"{k.replace('model.language_model.',''):<42}{str(tuple(W.shape)):>14}{kurt:9.2f}  "
                  + "".join(f"{c:>20}" for c in cells), flush=True)
