"""The k-tuple trellis as a derived payload grid.

Acceptance: E2M1 k=2 at R=7 must reproduce pair.py's 0.10941 on the Finding-4
slice.  pair.py hand-rolled the pair alphabet, the coset partition and the
Viterbi; this runs the same construction through the shipping encoder with
nothing but ``tuple_grid(E2M1_GRID, 2)``.  A mismatch is a bug, not a variant.
"""
import sys, glob, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from safetensors import safe_open
from tessera.alphabet import (
    build_forest, tuple_grid, lloyd_max_grid, E2M1_GRID, PayloadGrid,
)
from tessera.encode import encode_unit
from tessera.decode import reconstruct_unit
from tessera.trellis import ConvCode
from tessera.manifest import RotationState

dev = "cuda"; CC = ConvCode(memory=6)
for path in sorted(glob.glob("/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/model-*.safetensors"))[:4]:
    with safe_open(path, "pt") as f:
        for k in f.keys():
            if k.endswith("layers.0.mlp.gate_proj.weight"):
                W = f.get_tensor(k)[:2048, :5120].to(dev).float().contiguous()
rows, cols = W.shape
def rel(a): return ((a - W).norm() / W.norm()).item()

def arm(grid, R):
    F = {R: build_forest(R, grid=grid)}
    u = encode_unit(W, F, (R,) * cols, CC, rotation=RotationState.NONE,
                    with_diagonals=False, completion=0)
    return rel(reconstruct_unit(u, F, CC, completion=0).float())

NV = 0.09869
print(f"{'arm':<40}{'bpp':>7}{'rel_err':>10}{'vs NVFP4':>10}")
def show(name, grid, R):
    e = arm(grid, R)
    bpp = R / grid.arity + 0.5
    print(f"{name:<40}{bpp:>7.3f}{e:>10.5f}{e/NV:>9.2f}x")
    return e

def restride(g, rule):
    return PayloadGrid(g.name, g.values, None, g.arity, g.keys, rule)

lm16 = lloyd_max_grid(16)
pair = show("E2M1 k=2 R=7 coset  (pair trellis)", tuple_grid(E2M1_GRID, 2), 7)
print(f"{'  ^ pair.py reference':<40}{4.000:>7.3f}{0.10941:>10.5f}"
      f"{0.10941/NV:>9.2f}x   delta {abs(pair - 0.10941) / 0.10941 * 100:.2f}%")
show("E2M1 k=2 R=7 stride", restride(tuple_grid(E2M1_GRID, 2), "stride"), 7)
print()
print("--- k=2 BELOW its cap is dominated: k=1 spends a redundancy bit per")
print("    POSITION (two per pair), k=2 only one per pair ---")
for R in (5, 6):
    show(f"E2M1 k=2 R={R} stride", restride(tuple_grid(E2M1_GRID, 2), "stride"), R)
show("E2M1 k=1 R=2  (beats k=2 R=5, fewer bytes)", E2M1_GRID, 2)
show("E2M1 k=1 R=3  (beats k=2 R=6, same bytes)", E2M1_GRID, 3)

print()
print("--- the ladder is (base size, k) at R=cap, where k buys rates the")
print("    scalar grammar cannot express at all ---")
rungs = []
for size in (8, 16, 32):
    base = lloyd_max_grid(size) if size != 16 else lm16
    for k in (1, 2):
        grid = tuple_grid(base, k) if k > 1 else base
        R = grid.rate_cap
        rungs.append((R / k + 0.5, f"free-{size} k={k} R={R}", grid, R))
for bpp, name, grid, R in sorted(rungs):
    show(name, grid, R)
show("E2M1 k=1 R=3 (NVFP4-materialisable)", E2M1_GRID, 3)
show("E2M1 k=2 R=7 (NVFP4-materialisable base)", tuple_grid(E2M1_GRID, 2), 7)
print(f"{'NVFP4 RTN (E2M1 scalar)':<40}{4.500:>7.3f}{NV:>10.5f}{1.0:>9.2f}x")
