"""The 4.5-6.5 bpp band is dominated by NVFP4 at 4.5.  E2M1 caps at 4 payload
bits and E4M3's 256 codes are spread over 2^-9..448, so neither grid is matched
to a Gaussian there.  Both constraints exist only so the STOCK lane can
materialise into a hardware format.  The kernel lane reads an arbitrary value
LUT, so ask what a grid matched to the source is worth."""
import sys, glob, torch; sys.path.insert(0,"/home/rob/tessera/src")
from safetensors import safe_open
from tessera.alphabet import build_forest, PayloadGrid, E4M3_GRID, E2M1_GRID, GAUSSIAN_SOURCE
from tessera.encode import encode_unit, _pack_scales
from tessera.decode import reconstruct_unit
from tessera.trellis import ConvCode
from tessera.manifest import RotationState
dev="cuda"; CC=ConvCode(memory=6)

def lloyd_max(size, sigma, iterations=80):
    """Optimal scalar levels for a Gaussian: the grid E2M1/E4M3 are not."""
    s = torch.tensor(GAUSSIAN_SOURCE(1 << 15, sigma), dtype=torch.float64)
    lo, hi = s.min().item(), s.max().item()
    levels = torch.linspace(lo, hi, size, dtype=torch.float64)
    for _ in range(iterations):
        idx = (s[:, None] - levels[None, :]).abs().argmin(1)
        for j in range(size):
            m = idx == j
            if m.any(): levels[j] = s[m].mean()
    return tuple(float(v) for v in levels.sort().values)

files = sorted(glob.glob("/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/model-*.safetensors"))
W=None
for path in files[:4]:
    with safe_open(path,"pt") as f:
        for k in f.keys():
            if k.endswith("layers.0.mlp.gate_proj.weight"): W=f.get_tensor(k)[:2048,:2048].to(dev).float().contiguous()
def rel(a): return ((a-W).norm()/W.norm()).item()

print(f"{'bpp':>6}  {'arm':<34}{'rel_err':>10}  vs NVFP4@4.5")
NV = 0.09869
for bits in (5, 6, 7):
    size = 1 << bits
    values = lloyd_max(size, 6.0 / 6.0)      # peak-6 units, same as E2M1
    grid = PayloadGrid(f"LM{size}", values, tuple(range(size)))
    forests = {R: build_forest(R, grid=grid) for R in range(1, grid.rate_cap + 1)}
    best = None
    for R in sorted(forests):
        for c in range(0, grid.rate_cap - R + 1):
            if R + c != grid.rate_cap: continue
            u = encode_unit(W, forests, (R,)*W.shape[1], CC, rotation=RotationState.NONE,
                            with_diagonals=False, completion=c, released_positions=0)
            e = rel(reconstruct_unit(u, forests, CC, completion=c).float())
            if best is None or e < best[0]: best = (e, R, c)
    e, R, c = best
    bpp = grid.rate_cap + 0.5
    print(f"{bpp:6.2f}  {f'free {size}-code grid, R={R} c={c}':<34}{e:10.5f}  {e/NV:7.2f}x")
print(f"{4.50:6.2f}  {'NVFP4 scalar (E2M1)':<34}{NV:10.5f}  {1.0:7.2f}x")
print(f"{5.50:6.2f}  {'TESSERA-8 R=5 (E4M3 grid)':<34}{0.14610:10.5f}  {0.14610/NV:7.2f}x")
print(f"{6.50:6.2f}  {'TESSERA-8 R=6 (E4M3 grid)':<34}{0.06456:10.5f}  {0.06456/NV:7.2f}x")
