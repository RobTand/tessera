"""Prefill: does the decode-in-mainloop lose, as S13's prior says it should?

S13's negative prior is a DECOMPOSED benchmark on gridbook 0.9.1 -- decode
236.9 us per 4096^2 tile against MMA 50.6 us at M=1 rising to 889.4 at M=8192,
giving a crossover at M ~ 2100-4096.  That prior is scoped "on this decoder
generation", and Tessera's decoder is a different object: `ConvCode.step` is a
shift register, so a tile decodes from a local span with a six-row halo in N and
NO dependence along K -- and K is the reduction axis.

Same 4096^2 tile, so the numbers are directly comparable.  The two *Triton*
arms are swept over one identical block grid -- an under-tuned baseline is the
cheapest way to manufacture a speedup, and this project has already caught
itself doing it once.  The bf16 arm is cuBLAS and is not swept, and it is bf16
in / bf16 out where the Triton arms accumulate and return fp32; read it as a
reference point, not a matched arm.  Both swept arms now carry a value check on
their winning config -- the nvfp4 arm did not, and spent its whole life reading
the transpose of its own weights.
"""
import sys, torch, triton, triton.language as tl
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
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
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                BF16: tl.constexpr, GROUP_M: tl.constexpr):
    """The comparator: packed E2M1 nibbles + an E4M3 block scale per 16.

    Structurally identical to `_gemm_kernel` -- same tiling, same dot, same
    scale decode -- so the only difference measured is what it costs to turn
    stored bytes into weights.  NVFP4 reads half a byte per weight and indexes a
    16-entry table; Tessera reads two planes and indexes a 512-entry one.
    """
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M); grid_n = tl.cdiv(rows, BLOCK_N)
    width = GROUP_M * grid_n
    group = pid // width
    first_m = group * GROUP_M
    span = tl.minimum(grid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % width) % span); pid_n = (pid % width) // span
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
        # The comparator gets EXACTLY the treatment the kernel under test gets.
        # Giving only one arm tensor cores is the same class of error as giving
        # only one arm a scale plane, and that one already cost a retraction.
        w = val * sc
        if BF16:
            acc += tl.dot(xt.to(tl.bfloat16), w.to(tl.bfloat16), out_dtype=tl.float32)
        else:
            acc += tl.dot(xt, w, allow_tf32=False)
    tl.store(out_ptr + offs_m[:, None] * rows + offs_n[None, :], acc * gs,
             mask=lm[:, None] & ln[None, :])


def nvfp4_gemm(x, w, s, lut, gs, rows, cols, bm=64, bn=64, bk=64,
               bf16=True, gm=8, warps=4, stages=3):
    out = torch.empty((x.shape[0], rows), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(x.shape[0], bm) * triton.cdiv(rows, bn),)
    _nvfp4_gemm[grid](
        x, w, s, lut, out, float(gs), x.shape[0], rows, cols,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, BF16=bf16, GROUP_M=gm,
        num_warps=warps, num_stages=stages)
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
# The kernel indexes `w_ptr + (k // 2) * rows + n`, i.e. the two nibbles in a
# byte are consecutive in **k**.  `codes` is [N, K], so it must be transposed
# before pairing -- without this the comparator silently reads the transpose of
# its own weights (byte count, shape and addressing are all still valid, so it
# never faults and no shape surfaces it; only a value check does).
nib = codes.t().contiguous().to(torch.int32)          # [K, N]
packed = ((nib[0::2, :] << 4) | nib[1::2, :]).to(torch.uint8).contiguous()
Wb = W.to(torch.bfloat16)

# One grid, both arms.  Now includes num_warps/num_stages, which were never
# passed before -- every launch took Triton's default 4 warps / 3 stages, so no
# amount of block-size sweeping could have found the real optimum.
GRID = [(bm, bn, bk, w, st)
        for bm in (32, 64, 128) for bn in (64, 128, 256) for bk in (32, 64, 128)
        for w in (4, 8) for st in (3, 4)]

def sweep(make, M, verify=None):
    best = None
    for bm, bn, bk, w, st in GRID:
        if bm > max(16, M) * 2: continue
        try:
            fn = make(bm, bn, bk, w, st); us = bench(fn)
        except Exception:
            continue
        if best is None or us < best[0]: best = (us, bm, bn, bk, w, st)
    if verify is not None and best is not None and not verify(*best[1:]):
        raise SystemExit(f"fastest config {best[1:]} does not reproduce the reference")
    return best

import socket
print(f"tile {N}x{K} on {socket.gethostname()}; the two Triton arms are swept "
      f"over one identical grid, bf16 is unswept cuBLAS\n")
print(f"{'M':>6}{'tessera 3.5bpp':>17}{'nvfp4 4.5bpp':>15}{'bf16 torch':>13}"
      f"{'t/nv':>8}{'t/bf16':>13}")
x1 = torch.randn(1, K, device=dev)
y = tessera_gemm(x1, sel, pt, lut, scales, gs, N, K)
assert torch.allclose(y, x1 @ ref.t(), rtol=2e-5, atol=2e-4), "gemm disagrees with reference"
# M values come from argv so the sweep can be split across Sparky and
# Sparklina.  Both boxes are GB10 sm_121; run one M on BOTH as a cross-check,
# because "we measured it on the other box" is only sound if the boxes agree.
_MS = [int(a) for a in sys.argv[1:]] or [1, 8, 32, 128, 512, 2048, 4096]
for M in _MS:
    x = torch.randn(M, K, device=dev)
    t = sweep(lambda a, b, c, w, st: (lambda: tessera_gemm(
        x, sel, pt, lut, scales, gs, N, K, block_m=a, block_n=b, block_k=c,
        bf16=True, num_warps=w, num_stages=st)), M,
        verify=lambda a, b, c, w, st: torch.allclose(
            tessera_gemm(x, sel, pt, lut, scales, gs, N, K, block_m=a, block_n=b,
                         block_k=c, bf16=True, num_warps=w, num_stages=st),
            x @ ref.t(), rtol=3e-2, atol=3e-2))
    v = sweep(lambda a, b, c, w, st: (lambda: nvfp4_gemm(
        x, packed, scales, e2lut, gs, N, K, bm=a, bn=b, bk=c,
        bf16=True, warps=w, stages=st)), M,
        verify=lambda a, b, c, w, st: torch.allclose(
            nvfp4_gemm(x, packed, scales, e2lut, gs, N, K, bm=a, bn=b, bk=c,
                       bf16=True, warps=w, stages=st),
            x @ ref.t(), rtol=3e-2, atol=3e-2))
    xb = x.to(torch.bfloat16)
    bf = bench(lambda: xb @ Wb.t())
    print(f"{M:>6}{t[0]:>12.1f} us{v[0]:>10.1f} us{bf:>9.1f} us"
          f"{t[0]/v[0]:>8.2f}{t[0]/max(bf, 1e-9):>13.2f}")
