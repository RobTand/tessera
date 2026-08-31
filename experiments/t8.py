"""TESSERA-8 against scalar FP8, matched scale scheme.

Both arms carry the identical S6b scale plane (E8M0/32 + 4-bit refine/16, 0.5
bpp) scaled to E4M3's peak, so the only difference measured is the payload:
p trellis bits against 8 scalar bits."""
import sys, glob, itertools, torch; sys.path.insert(0,"/home/rob/tessera/src")
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
W = None
for path in files[:4]:
    with safe_open(path, "pt") as f:
        for k in f.keys():
            if k.endswith("layers.0.mlp.gate_proj.weight"):
                W = f.get_tensor(k)[:2048, :2048].to(dev).float().contiguous()
if W is None: raise SystemExit("tensor not found")
print(f"tensor {tuple(W.shape)}  sigma={W.std():.5f}")

def rel(a): return ((a - W).norm() / W.norm()).item()

# --- scalar FP8 baseline, same scale plane ---
_b, _r, eff = _pack_scales(W, 32, 16, peak=448.0)
scale = torch.repeat_interleave(eff, 16).reshape(W.shape)
fp8 = (W / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale
print(f"\nscalar E4M3 (FP8) + S6b scale plane: 8.5000 bpp   rel={rel(fp8):.5f}")
nv = None
_b2, _r2, eff2 = _pack_scales(W, 32, 16, peak=6.0)

print(f"\n{'R':>3}{'c':>3}{'payload':>9}{'bpp':>8}{'rel_err':>10}   vs FP8")
best = {}
for R in range(1, 8):
    for c in range(0, 8 - R):
        rates = (R,) * W.shape[1]
        u = encode_unit(W, E8, rates, CC, rotation=RotationState.NONE,
                        with_diagonals=False, completion=c, released_positions=0)
        got = reconstruct_unit(u, E8, CC, completion=c)
        e = rel(got.float()); p = R + c; bpp = p + 0.5
        if p not in best or e < best[p][0]: best[p] = (e, R, c)
r8 = rel(fp8)
for p in sorted(best):
    e, R, c = best[p]
    print(f"{R:>3}{c:>3}{p:>9}{p+0.5:8.4f}{e:10.5f}   {e/r8:6.2f}x err at {(p+0.5)/8.5:.3f}x bytes")
print("\n(best (R,c) split shown per payload width p = R + c)")
