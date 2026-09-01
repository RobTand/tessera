"""Prefill: does the decode-in-mainloop lose, as S13's prior says it should?

S13's negative prior is a DECOMPOSED benchmark on gridbook 0.9.1 -- decode
236.9 us per 4096^2 tile against MMA 50.6 us at M=1 rising to 889.4 at M=8192,
giving a crossover at M ~ 2100-4096.  That prior is scoped "on this decoder
generation", and Tessera's decoder is a different object: `ConvCode.step` is a
shift register, so a tile decodes from a local span with a six-row halo in N and
NO dependence along K -- and K is the reduction axis.

Same 4096^2 tile, so the numbers are directly comparable.  Every arm is swept
over one identical block grid: an under-tuned baseline is the cheapest way to
manufacture a speedup, and this project has already caught itself doing it once.
"""
import sys, torch, triton, triton.language as tl; sys.path.insert(0, "/home/rob/tessera/src")
from tessera.alphabet import build_forest
from tessera.encode import encode_unit
from tessera.decode import reconstruct_unit, decode_codes, materialize_nvfp4
from tessera.trellis import ConvCode
from tessera.manifest import RotationState
from tessera.kernel import build_value_lut, tessera_gemm, pack_kernel_planes

dev = "cuda"; CC = ConvCode(memory=6); R = 3
N = K = 4096
torch.manual_seed(0)


@triton.jit
def _nvfp4_gemm(x_ptr, w_ptr, s_ptr, lut_ptr, out_ptr, gs, M, rows, cols,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """The comparator: packed E2M1 nibbles + an E4M3 block scale per 16.

    Structurally identical to `_gemm_kernel` -- same tiling, same dot, same
    scale decode -- so the only difference measured is what it costs to turn
    stored bytes into weights.  NVFP4 reads half a byte per weight and indexes a
    16-entry table; Tessera reads two planes and indexes a 512-entry one.
    """
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    lm = offs_m < M; ln = offs_n < rows
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, cols, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        lk = offs_k < cols
        k = offs_k[:, None]; n = offs_n[None, :]
        live = lk[:, None] & ln[None, :]
        byte = tl.load(w_ptr + (k // 2) * rows + n, mask=live, other=0).to(tl.int32)
        nib = tl.where(k % 2 == 0, byte >> 4, byte & 0xF)
        val = tl.load(lut_ptr + nib, mask=live, other=0.0)
        sb = tl.load(s_ptr + (k // 16) * rows + n, mask=live, other=0).to(tl.int32)
        sc = tl.exp2(((sb >> 3) & 0xF).to(tl.float32) - 7.0) * (
            1.0 + (sb & 0x7).to(tl.float32) / 8.0)
        xt = tl.load(x_ptr + offs_m[:, None] * cols + offs_k[None, :],
                     mask=lm[:, None] & lk[None, :], other=0.0)
        acc += tl.dot(xt, val * sc, allow_tf32=False)
    tl.store(out_ptr + offs_m[:, None] * rows + offs_n[None, :], acc * gs,
             mask=lm[:, None] & ln[None, :])


def nvfp4_gemm(x, w, s, lut, gs, rows, cols, bm=64, bn=64, bk=64):
    out = torch.empty((x.shape[0], rows), dtype=torch.float32, device=x.device)
    _nvfp4_gemm[(triton.cdiv(x.shape[0], bm), triton.cdiv(rows, bn))](
        x, w, s, lut, out, float(gs), x.shape[0], rows, cols,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
    return out


def bench(fn, iters=20):
    for _ in range(4): fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(iters): fn()
    ev1.record(); torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) * 1e3 / iters


W = (torch.randn(N, K, device=dev) * 0.02).float()
F = {R: build_forest(R)}
u = encode_unit(W, F, (R,) * K, CC, rotation=RotationState.NONE, with_diagonals=False)
ref = reconstruct_unit(u, F, CC).float()
sel, pt = pack_kernel_planes(u.body_bits, R, 6)
lut = build_value_lut(F[R], CC, dev)
codes = decode_codes(u, F[R], CC)
_pack, e4m3, gs = materialize_nvfp4(codes, u.scale_base, u.scale_refine, u.group, u.half)
scales = e4m3.reshape(N, K // 16).t().contiguous()

# The comparator holds the SAME weights, so a quality difference cannot leak in.
from tessera.encode import e2m1_value_table
e2lut = e2m1_value_table(dev).float()
nib = codes.to(torch.int32)
packed = ((nib[0::2, :] << 4) | nib[1::2, :]).to(torch.uint8).contiguous()
Wb = W.to(torch.bfloat16)

GRID = [(bm, bn, bk) for bm in (16, 32, 64, 128) for bn in (32, 64, 128)
        for bk in (32, 64, 128)]

def sweep(make, M):
    best = None
    for bm, bn, bk in GRID:
        if bm > max(16, M) * 2: continue
        try:
            fn = make(bm, bn, bk); us = bench(fn)
        except Exception:
            continue
        if best is None or us < best[0]: best = (us, bm, bn, bk)
    return best

print(f"tile {N}x{K}, fp32 accumulate, all arms swept over one identical grid\n")
print(f"{'M':>6}{'tessera 3.5bpp':>17}{'nvfp4 4.5bpp':>15}{'bf16 torch':>13}"
      f"{'t/nv':>8}{'decode/FLOP':>13}")
x1 = torch.randn(1, K, device=dev)
y = tessera_gemm(x1, sel, pt, lut, scales, gs, N, K)
assert torch.allclose(y, x1 @ ref.t(), rtol=2e-5, atol=2e-4), "gemm disagrees with reference"
for M in (1, 8, 32, 128, 512, 2048, 8192):
    x = torch.randn(M, K, device=dev)
    t = sweep(lambda a, b, c: (lambda: tessera_gemm(
        x, sel, pt, lut, scales, gs, N, K, block_m=a, block_n=b, block_k=c)), M)
    v = sweep(lambda a, b, c: (lambda: nvfp4_gemm(
        x, packed, scales, e2lut, gs, N, K, bm=a, bn=b, bk=c)), M)
    xb = x.to(torch.bfloat16)
    bf = bench(lambda: xb @ Wb.t())
    flops = 2.0 * M * N * K
    print(f"{M:>6}{t[0]:>12.1f} us{v[0]:>10.1f} us{bf:>9.1f} us"
          f"{t[0]/v[0]:>8.2f}{(t[0]-v[0])*1e-6*flops/flops if False else (t[0]-bf)/max(bf,1e-9):>13.2f}")
