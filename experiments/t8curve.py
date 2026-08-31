"""One tensor, one scale scheme, both families, and both scalar baselines."""
import sys, glob, torch; sys.path.insert(0,"/home/rob/tessera/src")
from safetensors import safe_open
from tessera.alphabet import build_forest, E4M3_GRID, E2M1_GRID
from tessera.encode import encode_unit, _pack_scales
from tessera.decode import reconstruct_unit
from tessera.trellis import ConvCode
from tessera.manifest import RotationState
dev="cuda"; CC=ConvCode(memory=6)
E8 = {R: build_forest(R, grid=E4M3_GRID) for R in range(1, 8)}
E2 = {R: build_forest(R, grid=E2M1_GRID) for R in (1, 2, 3)}
files = sorted(glob.glob("/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/model-*.safetensors"))
W=None
for path in files[:4]:
    with safe_open(path,"pt") as f:
        for k in f.keys():
            if k.endswith("layers.0.mlp.gate_proj.weight"): W=f.get_tensor(k)[:2048,:2048].to(dev).float().contiguous()
def rel(a): return ((a-W).norm()/W.norm()).item()
rows=[]
for peak, name, dt, lim in ((6.0,"NVFP4 scalar (E2M1)",None,6.0),(448.0,"FP8 scalar (E4M3)",torch.float8_e4m3fn,448.0)):
    _b,_r,eff = _pack_scales(W,32,16,peak=peak)
    sc = torch.repeat_interleave(eff,16).reshape(W.shape)
    if dt is None:
        from tessera.encode import grid_value_table
        tbl = grid_value_table(E2M1_GRID, dev)
        q = (W/sc); idx = (q.unsqueeze(-1)-tbl).abs().argmin(-1); out = tbl[idx]*sc
    else:
        out = (W/sc).clamp(-lim,lim).to(dt).float()*sc
    rows.append((4.5 if dt is None else 8.5, name, rel(out)))
for grid, fam, forests in ((E2M1_GRID,"TESSERA-4",E2),(E4M3_GRID,"TESSERA-8",E8)):
    for R in sorted(forests):
        for c in range(0, grid.rate_cap - R + 1):
            u = encode_unit(W, forests, (R,)*W.shape[1], CC, rotation=RotationState.NONE,
                            with_diagonals=False, completion=c, released_positions=0)
            rows.append((R+c+0.5, f"{fam} R={R} c={c}", rel(reconstruct_unit(u, forests, CC, completion=c).float())))
best={}
for bpp, name, e in rows:
    fam = name.split(' R=')[0]
    key = (round(bpp,3), fam)
    if key not in best or e < best[key][1]: best[key] = (name, e)
print(f"{'bpp':>7}  {'best arm':<26}{'rel_err':>10}   vs NVFP4  vs FP8")
nv = min(e for (b,f),(n,e) in best.items() if 'NVFP4' in f)
f8 = min(e for (b,f),(n,e) in best.items() if 'FP8' in f)
for (bpp, fam) in sorted(best):
    name, e = best[(bpp, fam)]
    print(f"{bpp:7.4f}  {name:<26}{e:10.5f}   {e/nv:7.2f}x {e/f8:7.2f}x")
