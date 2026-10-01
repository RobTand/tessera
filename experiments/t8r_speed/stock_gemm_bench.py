"""Stock-vLLM GEMMs that run off the Blackwell libraries in the GLM-5.3 serve.

The A8SE-SH rank-0 trace (L8192, 2048-token chunks) has three stock GEMMs on
Ampere-generation kernels, about 18 ms per chunk together:

* MLA k up-projection, ``torch.bmm(q_nope, W_UK_T, out=ql.transpose(0, 1))``,
  (N, B, P) x (N, P, L) with N = 32 heads per rank, P = 256, L = 512:
  ``cutlass_80_wmma_tensorop_bf16_s161616gemm``, 19.4 TFLOPS, 9.73 ms/chunk;
* MLA v up-projection, ``torch.bmm(x, W_UV, out=o.transpose(0, 1))``,
  (N, B, L) x (N, L, V), V = 256: ``cutlass_80_tensorop_bf16_s16816gemm``,
  30.9 TFLOPS, 6.12 ms/chunk;
* the sparse indexer's fp32 head gate, ``torch.mm(h.float(), W_fp32)``,
  (B, 4096) x (4096, 32): ``cutlass_80_simt_sgemm``, 2.3 TFLOPS, 2.56 ms/chunk.

For each, this times the stock call (in the serve's layouts, from
``mla_attention.py`` and ``glm5next/common/attention.py`` of image 5be13705)
against call forms that change no arithmetic contract the model depends on,
and records the kernel each form dispatches, whether its output is bitwise the
stock output, the largest difference, and GPU power and SM clock over a
sustained loop.  Nothing here edits vLLM; it says which form a plugin override
should use and what it would save.

Usage: stock_gemm_bench.py --out DIR [--tokens 512,2048]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sustained_peaks import Sampler  # noqa: E402

N, P, R, L, V = 32, 256, 64, 512, 256      # heads per rank (TP 2), nope, rope, kv_lora, v
HIDDEN, IDX_HEADS = 4096, 32
LOOP_S = float(os.environ.get("LOOP_S", "1.5"))


def ev_ms(call, reps=20, trials=5):
    out = []
    for _ in range(trials):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            call()
        b.record(); b.synchronize()
        out.append(a.elapsed_time(b) / reps)
    return statistics.median(out)


def kernels(call):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    names = {}
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            k = e.name[:140]
            names[k] = names.get(k, 0.0) + e.device_time_total
    return [{"kernel": k, "us": round(v, 2)} for k, v in sorted(names.items(), key=lambda kv: -kv[1])]


def sustained(call, sampler):
    torch.cuda.synchronize()
    t0 = time.time(); n = 0
    while time.time() - t0 < LOOP_S:
        call(); n += 1
        if n % 16 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    t1 = time.time()
    return sampler.window(t0 + (t1 - t0) / 2, t1)


def compare(ref, got):
    ref32, got32 = ref.float(), got.float()
    d = (ref32 - got32).abs()
    scale = ref32.abs().max().item() or 1.0
    return {"bitwise": bool(torch.equal(ref, got)), "max_abs": d.max().item(),
            "max_abs_over_max_ref": d.max().item() / scale,
            "mismatch_frac": (d > 0).float().mean().item()}


def run_case(group, B, flops, forms, sampler, results):
    ref_out = None
    for name, call, result in forms:
        try:
            call(); torch.cuda.synchronize()
            out = result().clone()
            ms = ev_ms(call)
            rec = {"group": group, "tokens": B, "form": name, "ms": ms,
                   "tflops": flops / (ms * 1e-3) / 1e12, "kernels": kernels(call),
                   "sampler": sustained(call, sampler)}
            if ref_out is None:
                ref_out = out
                rec["vs_stock"] = {"bitwise": True, "max_abs": 0.0}
            else:
                rec["vs_stock"] = compare(ref_out, out)
        except Exception as e:  # noqa: BLE001  a form the build lacks is a result, not a failure
            rec = {"group": group, "tokens": B, "form": name, "error": repr(e)[:300]}
        print(json.dumps(rec), flush=True)
        results.append(rec)


def mla_k_up(B, sampler, results, w_layout):
    g = torch.Generator(device="cuda").manual_seed(1)
    q = torch.randn(B, N, P + R, device="cuda", dtype=torch.bfloat16, generator=g)
    q_nope = q[..., :P].transpose(0, 1)                         # (N, B, P), the serve's view
    w_lnp = torch.randn(L, N, P, device="cuda", dtype=torch.bfloat16, generator=g) * 0.05
    W = w_lnp.permute(1, 2, 0)                                  # (N, P, L): W_UK.permute(1, 2, 0)
    if w_layout == "contig":
        W = W.contiguous()
    out = torch.empty(B, N, L, device="cuda", dtype=torch.bfloat16)
    tmp = torch.empty(N, B, L, device="cuda", dtype=torch.bfloat16)
    qc = torch.empty(N, B, P, device="cuda", dtype=torch.bfloat16)
    flops = 2.0 * N * B * P * L

    def stock():
        torch.bmm(q_nope, W, out=out.transpose(0, 1))

    def contig_out():
        torch.bmm(q_nope, W, out=tmp)
        out.copy_(tmp.transpose(0, 1))

    def contig_in():
        qc.copy_(q_nope)
        torch.bmm(qc, W, out=out.transpose(0, 1))

    def contig_both():
        qc.copy_(q_nope)
        torch.bmm(qc, W, out=tmp)
        out.copy_(tmp.transpose(0, 1))

    def per_head():
        for n in range(N):
            torch.mm(q_nope[n], W[n], out=tmp[n])
        out.copy_(tmp.transpose(0, 1))

    def einsum():
        out.copy_(torch.einsum("bnp,npl->bnl", q[..., :P], W))

    forms = [("stock", stock, lambda: out), ("contig_out", contig_out, lambda: out),
             ("contig_in", contig_in, lambda: out), ("contig_both", contig_both, lambda: out),
             ("per_head_mm", per_head, lambda: out), ("einsum", einsum, lambda: out)]
    run_case(f"mla_k_up/W_{w_layout}", B, flops, forms, sampler, results)


def mla_v_up(B, sampler, results, w_layout):
    g = torch.Generator(device="cuda").manual_seed(2)
    x_bnl = torch.randn(B, N, L, device="cuda", dtype=torch.bfloat16, generator=g)
    x = x_bnl.transpose(0, 1)                                   # (N, B, L), the serve's view
    w_lnv = torch.randn(L, N, V, device="cuda", dtype=torch.bfloat16, generator=g) * 0.05
    W = w_lnv.transpose(0, 1)                                   # (N, L, V): W_UV.transpose(0, 1)
    if w_layout == "contig":
        W = W.contiguous()
    out = torch.empty(B, N * V, device="cuda", dtype=torch.bfloat16)
    tmp = torch.empty(N, B, V, device="cuda", dtype=torch.bfloat16)
    xc = torch.empty(N, B, L, device="cuda", dtype=torch.bfloat16)
    flops = 2.0 * N * B * L * V

    def stock():
        torch.bmm(x, W, out=out.view(B, N, V).transpose(0, 1))

    def contig_out():
        torch.bmm(x, W, out=tmp)
        out.view(B, N, V).copy_(tmp.transpose(0, 1))

    def contig_in():
        xc.copy_(x)
        torch.bmm(xc, W, out=out.view(B, N, V).transpose(0, 1))

    def contig_both():
        xc.copy_(x)
        torch.bmm(xc, W, out=tmp)
        out.view(B, N, V).copy_(tmp.transpose(0, 1))

    def per_head():
        for n in range(N):
            torch.mm(x[n], W[n], out=tmp[n])
        out.view(B, N, V).copy_(tmp.transpose(0, 1))

    def einsum():
        out.view(B, N, V).copy_(torch.einsum("bnl,nlv->bnv", x_bnl, W))

    forms = [("stock", stock, lambda: out), ("contig_out", contig_out, lambda: out),
             ("contig_in", contig_in, lambda: out), ("contig_both", contig_both, lambda: out),
             ("per_head_mm", per_head, lambda: out), ("einsum", einsum, lambda: out)]
    run_case(f"mla_v_up/W_{w_layout}", B, flops, forms, sampler, results)


def indexer_gate(B, sampler, results):
    g = torch.Generator(device="cuda").manual_seed(3)
    h = torch.randn(B, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g)
    w = (torch.randn(IDX_HEADS, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02)
    w32 = w.t().contiguous().float()                            # self._wp_fp32
    wt = w.t()                                                  # (HIDDEN, IDX_HEADS) bf16 view
    res = {}
    flops = 2.0 * B * HIDDEN * IDX_HEADS

    def stock():
        res["o"] = torch.mm(h.float(), w32)

    def bf16_in_fp32_out():
        res["o"] = torch.mm(h, wt, out_dtype=torch.float32)

    def tf32():
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            res["o"] = torch.mm(h.float(), w32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev

    def fp32_no_cast_copy():
        # the same fp32 GEMM with the activation cast hoisted out: what the
        # SIMT GEMM alone costs
        res["o"] = torch.mm(h32, w32)
    h32 = h.float()

    forms = [("stock", stock, lambda: res["o"]), ("bf16_in_fp32_out", bf16_in_fp32_out, lambda: res["o"]),
             ("tf32", tf32, lambda: res["o"]), ("fp32_gemm_only", fp32_no_cast_copy, lambda: res["o"])]
    run_case("indexer_gate_fp32", B, flops, forms, sampler, results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default="512,2048")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    sampler = Sampler(); sampler.start()
    meta = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "cuda": torch.version.cuda, "image": os.environ.get("PB_CONTAINER_IMAGE") or os.environ.get("ORACLE_IMAGE"),
            "pb_action": os.environ.get("PB_ACTION_KEY"), "start_unix": time.time(),
            "loop_s": LOOP_S}
    results = []
    for B in [int(t) for t in a.tokens.split(",")]:
        for lay in ("perm", "contig"):
            mla_k_up(B, sampler, results, lay)
            mla_v_up(B, sampler, results, lay)
        indexer_gate(B, sampler, results)
    sampler.stop_evt.set()
    meta["end_unix"] = time.time(); meta["sampler_errors"] = sampler.errors
    with open(os.path.join(a.out, "stock_gemm_bench.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1)
    print("wrote", os.path.join(a.out, "stock_gemm_bench.json"))


if __name__ == "__main__":
    main()
