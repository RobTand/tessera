"""Toolchain gate: does dot_scaled(e2m1, ue4m3) lower to native FP4 MMA on sm121?

Run once per (triton package, scale dtype) combination from the driver below; a
compiler assertion kills the process, so each combination must be its own
process.

Usage: python3 a4_probe_mma.py --triton triton|tokenspeed_triton \
           --scale-dtype e4m3|u8 --out /tmp/probe.json [--iters 50]
"""
import argparse
import importlib
import json

import torch

E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triton", default="triton")
    ap.add_argument("--scale-dtype", default="u8", choices=["e4m3", "u8"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--ptx-head", type=int, default=0)
    args = ap.parse_args()

    triton = importlib.import_module(args.triton)
    tl = importlib.import_module(args.triton + ".language")

    result = {
        "triton_module": args.triton,
        "triton": triton.__version__,
        "triton_file": triton.__file__,
        "scale_dtype": args.scale_dtype,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }

    if args.scale_dtype == "e4m3":
        scale_torch_dtype = torch.float8_e4m3fn
    else:
        scale_torch_dtype = torch.uint8
    bitcast_scales = args.scale_dtype == "u8"

    @triton.jit
    def probe_gemm(a_ptr, sa_ptr, b_ptr, sb_ptr, out_ptr,
                   M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   BITCAST: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kp = tl.arange(0, BK // 2)
            kq = tl.arange(0, BK // 16)
            a = tl.load(a_ptr + offs_m[:, None] * (K // 2) + (k0 // 2 + kp)[None, :])
            b = tl.load(b_ptr + (k0 // 2 + kp)[:, None] * N + offs_n[None, :])
            sa = tl.load(sa_ptr + offs_m[:, None] * (K // 16) + (k0 // 16 + kq)[None, :])
            sb = tl.load(sb_ptr + offs_n[:, None] * (K // 16) + (k0 // 16 + kq)[None, :])
            if BITCAST:
                sa = sa.to(tl.float8e4nv, bitcast=True)
                sb = sb.to(tl.float8e4nv, bitcast=True)
            acc = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1", acc)
        tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    def gemm(a, sa, b, sb, M, N, K, BM, BN=64, BK=128, stages=3):
        out = torch.empty((M, N), dtype=torch.float32, device=a.device)
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        probe_gemm[grid](a, sa, b, sb, out, M=M, N=N, K=K, BM=BM, BN=BN, BK=BK,
                         BITCAST=bitcast_scales, num_warps=4, num_stages=stages)
        return out

    def random_packed(m, k, gen):
        return torch.randint(0, 256, (m, k // 2), dtype=torch.int64, generator=gen,
                             device="cuda").to(torch.uint8)

    def random_scale(n, k, gen):
        raw = torch.randint(1, 40, (n, k // 16), dtype=torch.int64, generator=gen,
                            device="cuda").to(torch.uint8)
        if args.scale_dtype == "e4m3":
            return raw.view(scale_torch_dtype)
        return raw

    table = torch.tensor(
        [(-1 if i >> 3 else 1) * E2M1_VALUES[i & 7] for i in range(16)],
        dtype=torch.float32, device="cuda")

    def unpack_nibbles(packed):
        ah = packed.contiguous().view(torch.uint8)
        out = torch.empty(ah.shape[0], ah.shape[1] * 2, dtype=torch.long, device="cuda")
        out[:, 0::2] = (ah & 0x0F).long()
        out[:, 1::2] = (ah >> 4).long()
        return out

    def reference(a, sa, b, sb, K):
        av = table[unpack_nibbles(a)].double()
        sa_f = sa.view(torch.uint8).to(torch.float64).repeat_interleave(16, dim=1)
        bv = table[unpack_nibbles(b)].double()
        sb_f = sb.view(torch.uint8).to(torch.float64).repeat_interleave(16, dim=1)
        return ((av * sa_f) @ (bv * sb_f.t())).cpu().float()

    gen = torch.Generator(device="cuda").manual_seed(0)
    M, N, K = 32, 64, 256
    a = random_packed(M, K, gen)
    b = random_packed(K, N, gen)
    sa = random_scale(M, K, gen)
    sb = random_scale(N, K, gen)
    out = gemm(a, sa, b, sb, M, N, K, BM=32)
    ref = reference(a, sa, b, sb, K)
    diff = (out.double().cpu() - ref).abs()
    result["small_case"] = {
        "max_abs": float(diff.max()),
        "max_ref": float(ref.abs().max()),
        "rel": float(diff.max() / max(float(ref.abs().max()), 1e-12)),
    }

    try:
        kern = list(probe_gemm.device_caches[torch.cuda.current_device()][0].values())[-1]
    except Exception:  # noqa: BLE001
        kern = list(probe_gemm.cache[torch.cuda.current_device()].values())[-1]
    ptx = kern.asm["ptx"]
    result["ptx"] = {
        "tokens": [t for t in ("mxf4nvf4", "kind::mxf4", "mma.sync", "tcgen05",
                               "fma.rn.f32", "cvt.rn.satfinite.e2m1x2", "wgmma")
                   if t in ptx],
        "bytes": len(ptx),
    }
    if args.ptx_head:
        result["ptx_head"] = ptx[:args.ptx_head]

    timings = {}
    for (M2, BM) in ((16, 16), (16, 64), (128, 64)):
        N2, K2 = 2048, 4096
        a2 = random_packed(M2, K2, gen)
        b2 = random_packed(K2, N2, gen)
        sa2 = random_scale(M2, K2, gen)
        sb2 = random_scale(N2, K2, gen)
        fn = lambda: gemm(a2, sa2, b2, sb2, M2, N2, K2, BM)
        fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        timings[f"triton_M{M2}_BM{BM}"] = start.elapsed_time(end) / args.iters

        try:
            import vllm._custom_ops  # noqa: F401
            gs = torch.tensor([448.0 * 6.0 / 6.0], dtype=torch.float32, device="cuda")
            xa = torch.randn(M2, K2, dtype=torch.bfloat16, device="cuda") * 0.5
            xb = torch.randn(N2, K2, dtype=torch.bfloat16, device="cuda") * 0.5
            pa, sca = torch.ops._C.scaled_fp4_quant(xa, gs, True)
            pb, scb = torch.ops._C.scaled_fp4_quant(xb, gs, True)
            base = lambda: torch._scaled_mm(pa, pb.t(), scale_a=sca, scale_b=scb,
                                            out_dtype=torch.bfloat16)
            base()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                base()
            end.record()
            torch.cuda.synchronize()
            timings[f"torch_scaled_mm_M{M2}"] = start.elapsed_time(end) / args.iters
        except Exception as exc:  # noqa: BLE001
            timings[f"torch_scaled_mm_M{M2}"] = f"{type(exc).__name__}: {exc}"
    result["timings_ms"] = timings
    print("PROBE_RESULT " + json.dumps(result))
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main()
