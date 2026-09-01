"""Is a 4.0 bpp k-tuple body as cheap to decode as the 3.5 bpp scalar one?

Same shape, same comparator and same author as kernel-lane.md, so the three
arms differ only in what they store.  Block shapes are swept rather than
reasoned about: the last time a coalescing argument picked them it was wrong
by 2.7x in the direction it predicted was better.
"""
import sys, time, torch; sys.path.insert(0, "/home/rob/tessera/src")
from tessera.alphabet import build_forest, tuple_grid, lloyd_max_grid, E2M1_GRID
from tessera.encode import encode_unit
from tessera.decode import decode_codes_mixed, materialize_nvfp4, reconstruct_unit
from tessera.trellis import ConvCode
from tessera.manifest import RotationState
from tessera.wire import nvfp4_scale_bytes
from tessera.kernel import (
    build_tuple_value_lut, build_value_lut, pack_kernel_planes, pack_nvfp4_column_major,
    nvfp4_gemv_sliced, tessera_gemv_tuple, tessera_gemv_wide,
)

dev = "cuda"; CC = ConvCode(memory=6)
ROWS, COLS = 17408, 5120
PEAK = 246.0
torch.manual_seed(0)
W = (torch.randn(ROWS, COLS, device=dev) * 0.02).contiguous()
x = torch.randn(COLS, device=dev)

def bench(fn, iters=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6

# One grid for every arm.  The comparator is swept exactly as wide as the
# kernel it is judging -- an under-tuned baseline is the cheapest way to
# manufacture a speedup, and kernel-lane.md's NVFP4 number came from this grid.
def sweep(make):
    best = None
    for lanes in (32, 64, 128, 256, 512):
        for split_k in (16, 32, 64, 128, 256):
            try:
                fn = make(lanes, split_k)
                us = bench(fn, 20)
            except Exception:
                continue
            if best is None or us < best[0]:
                best = (us, lanes, split_k)
    return best

rows_out = []

# --- scalar body, 3.5 bpp -------------------------------------------------
F1 = {3: build_forest(3)}
u1 = encode_unit(W, F1, (3,) * COLS, CC, rotation=RotationState.NONE,
                 with_diagonals=False, released_positions=0)
codes = decode_codes_mixed(u1, F1, CC)
_p, e4m3, gs = materialize_nvfp4(codes, u1.scale_base, u1.scale_refine, u1.group, u1.half)
scales = e4m3.reshape(ROWS, COLS // 16).t().contiguous()
sel1, pt1 = pack_kernel_planes(u1.body_bits)
lut1 = build_value_lut(F1[3], CC)
us, la, sk = sweep(lambda l, s: (lambda: tessera_gemv_wide(
    x, sel1, pt1, lut1, scales, gs, ROWS, COLS, lanes=l, split_k=s)))
rows_out.append(("Tessera scalar k=1 R=3", 3.5, sel1.numel() + pt1.numel(), us, la, sk))

# --- tuple body, 4.0 bpp --------------------------------------------------
for label, base in (("E2M1", E2M1_GRID), ("free-16", lloyd_max_grid(16))):
    g = tuple_grid(base, 2); R = g.rate_cap
    F2 = {R: build_forest(R, grid=g)}
    u2 = encode_unit(W, F2, (R,) * COLS, CC, rotation=RotationState.NONE,
                     with_diagonals=False, completion=0)
    sel2, pt2 = pack_kernel_planes(u2.body_bits, rate=R, memory=6)
    lut2 = build_tuple_value_lut(F2[R], CC, dev)
    e2, gs2 = nvfp4_scale_bytes(u2.scale_base, u2.scale_refine, u2.group, u2.half)
    sc2 = e2.reshape(ROWS, COLS // 16).t().contiguous()
    ref = reconstruct_unit(u2, F2, CC, completion=0).float()
    got = tessera_gemv_tuple(x, sel2, pt2, lut2, sc2, gs2, ROWS, COLS,
                             rate=R, arity=2, lanes=32, split_k=8)
    err = ((got - ref @ x).norm() / (ref @ x).norm()).item()
    us, la, sk = sweep(lambda l, s: (lambda: tessera_gemv_tuple(
        x, sel2, pt2, lut2, sc2, gs2, ROWS, COLS, rate=R, arity=2,
        lanes=l, split_k=s)))
    rows_out.append((f"Tessera tuple k=2 R=7 ({label})", 4.0,
                     sel2.numel() + pt2.numel(), us, la, sk))
    print(f"  [{label}] kernel-vs-reference GEMV rel {err:.2e}")

# --- NVFP4 comparator, 4.5 bpp -------------------------------------------
packed = pack_nvfp4_column_major(codes)
us, la, sk = sweep(lambda l, s: (lambda: nvfp4_gemv_sliced(
    x, packed, scales, gs, ROWS, COLS, block_n=l, split_k=s)))
rows_out.append(("NVFP4 comparator (matched)", 4.5, packed.numel(), us, la, sk))

scale_bytes = scales.numel()
print(f"\n{'kernel':<34}{'bpp':>5}{'µs':>8}{'GB/s':>8}{'% peak':>8}"
      f"{'body MiB':>10}   block")
for name, bpp, body, us, la, sk in rows_out:
    total = body + scale_bytes
    gbs = total / us * 1e6 / 1e9
    print(f"{name:<34}{bpp:>5.1f}{us:>8.0f}{gbs:>8.0f}{gbs/PEAK*100:>7.1f}%"
          f"{body/2**20:>10.1f}   lanes={la} splitk={sk}")
