"""Does a free 32-code grid reach 4.0 bpp without a k-tuple trellis?

A free grid has 32 codes, so cap = 4 and R=4 is legal -- which the E2M1 grid
forbids (|A_R| = 2^(R+1) needs 32 anchors and E2M1 has 16).  A mixed R=3/R=4
column schedule averages 3.5 payload bits = 4.0 bpp, the same rate the pair
trellis reaches, using only machinery that already ships.

Same tensor and slice as release-vs-tuple-trellis.md Finding 4, so the pair
numbers there are directly comparable.  Also runs the free-16 SCALAR arm that
doc flags as unrun: it separates "Lloyd-Max beats E2M1" from "the trellis's
redundancy bit".
"""
import sys, glob, torch; sys.path.insert(0, "/home/rob/tessera/src")
from safetensors import safe_open
from tessera.alphabet import build_forest, PayloadGrid, GAUSSIAN_SOURCE
from tessera.encode import encode_unit, _pack_scales, e2m1_value_table, grid_value_table
from tessera.decode import reconstruct_unit
from tessera.trellis import ConvCode
from tessera.manifest import RotationState

dev = "cuda"; CC = ConvCode(memory=6)

def lloyd_max(size, sigma, iterations=80):
    s = torch.tensor(GAUSSIAN_SOURCE(1 << 15, sigma), dtype=torch.float64)
    levels = torch.linspace(s.min().item(), s.max().item(), size, dtype=torch.float64)
    for _ in range(iterations):
        idx = (s[:, None] - levels[None, :]).abs().argmin(1)
        for j in range(size):
            m = idx == j
            if m.any(): levels[j] = s[m].mean()
    return tuple(float(v) for v in levels.sort().values)

W = None
for path in sorted(glob.glob("/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/model-*.safetensors"))[:4]:
    with safe_open(path, "pt") as f:
        for k in f.keys():
            if k.endswith("layers.0.mlp.gate_proj.weight"):
                W = f.get_tensor(k)[:2048, :5120].to(dev).float().contiguous()
rows, cols = W.shape
def rel(a): return ((a - W).norm() / W.norm()).item()

def scalar(values):
    """Round each position to the nearest grid level -- no trellis at all.

    The normalisation must use THIS grid's peak, which is what ``encode_unit``
    does (``peak=max|grid.values|``).  Holding peak at E2M1's 6.0 while the
    grid tops out near 3 clips a fifth of the mass and measures the mismatch
    rather than the grid -- it read 0.389 for free-16 before this was fixed.
    """
    v = torch.tensor(values, device=dev, dtype=torch.float32)
    _, _, eff = _pack_scales(W, 32, 16, peak=max(abs(x) for x in values))
    sc = torch.repeat_interleave(eff, 16).reshape(W.shape)
    t = W / sc
    return rel(v[((t.unsqueeze(-1) - v) ** 2).argmin(-1)] * sc)

def trellis(grid, rates, c=0):
    forests = {R: build_forest(R, grid=grid) for R in sorted(set(rates))}
    u = encode_unit(W, forests, rates, CC, rotation=RotationState.NONE,
                    with_diagonals=False, completion=c)
    return rel(reconstruct_unit(u, forests, CC, completion=c).float())

lm16, lm32 = lloyd_max(16, 1.0), lloyd_max(32, 1.0)
G16 = PayloadGrid("LM16", lm16, tuple(range(16)))
G32 = PayloadGrid("LM32", lm32, tuple(range(32)))
E2M1 = e2m1_value_table(dev).tolist()

alt = tuple(3 + (i & 1) for i in range(cols))   # 3,4,3,4,... -> 3.5 payload bits
rows_out = [
    ("NVFP4 RTN: E2M1 scalar",              4.500, scalar(E2M1)),
    ("free-16 scalar (Lloyd-Max, no trellis)", 4.500, scalar(lm16)),
    ("free-32 trellis R=4",                 4.500, trellis(G32, (4,) * cols)),
    ("E2M1 trellis R=3 (k=1)",              3.500, trellis(build_forest(3).grid, (3,) * cols)),
    ("free-32 scalar (Lloyd-Max, no trellis)", 5.500, scalar(lm32)),
    ("free-32 trellis R=3",                 3.500, trellis(G32, (3,) * cols)),
    ("free-32 trellis, mixed R=3/R=4",      4.000, trellis(G32, alt)),
]
NV = rows_out[0][2]
print(f"{'arm':<42}{'bpp':>7}{'rel_err':>10}{'vs NVFP4':>10}")
for name, bpp, e in rows_out:
    print(f"{name:<42}{bpp:>7.3f}{e:>10.5f}{e/NV:>9.2f}x")
print(f"{'pair trellis k=2 (Finding 4)':<42}{4.000:>7.3f}{0.10941:>10.5f}{0.10941/NV:>9.2f}x")
print(f"{'pair trellis k=4 (Finding 4)':<42}{4.250:>7.3f}{0.10098:>10.5f}{0.10098/NV:>9.2f}x")
