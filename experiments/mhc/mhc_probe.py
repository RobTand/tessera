#!/usr/bin/env python3
"""The before-measurements for the mHC and elementwise overrides (GLM-5.3 prefill).

Three parts, each on the pinned serving image's own vLLM code:

``onorm``  The KDA output norm (``FusedRMSNormGated``, ``activation="sigmoid"``)
           on its two paths: ``forward_native`` (the eager decomposition a
           GLM-5.3 serve runs today, because the model is not compiled and
           ``custom_ops`` resolves to ``none``) and ``forward_cuda`` (the Triton
           ``rms_norm_gated`` kernel that ``+fused_rms_norm_gated`` selects).
           Both are held against an fp64 reference, and against each other in
           bf16 ulps of the row max.  Per-call time comes from CUDA-graph
           replay over enough input copies to exceed L2, so each call reads
           DRAM, as the served chunk does.

``mhc``    Stock ``mhc_fused_post_pre_tilelang`` at every token count from 1 to
           8192, with the real layer-1 ``hc_attn``/``hc_ffn`` projections,
           scales, bases and norm weights of the served checkpoint.  Records
           each kernel's device time (``torch.profiler``), its byte model and
           its fraction of the DRAM roofline, and the split-invariance the
           sequence-parallel override depends on: one call over T tokens
           against the same tokens in two and four calls.

``l2``     L2 capacity (from the device properties) and a read-rate curve over
           buffer sizes 1 to 256 MB, which locates the L2 knee and the DRAM
           plateau on this box.

``kdaconv`` The KDA prefill conv once per q/k/v slice (``TESSERA_GLM53_KDA_CONV_SPLIT``)
           against the stock merged conv on the image's Triton
           ``causal_conv1d_fn``: bitwise q/k/v and conv state on varlen batches
           (both conv-state layouts, with and without spec columns, sequences
           shorter than the conv width), then the per-layer time of the conv
           plus the dense copies FlashKDA's input contract forces, on both paths.

``mhcsplit`` The ``mhc`` split invariance with the pre-norm GEMM's split-k forced
           to one value for the full call and its chunks (the full call's split,
           the chunk's split, and 1), to test whether the token-count-dependent
           split is the only cause of the mismatch; with timing, the T/2-token call at
           the full batch's split against its own (the price of an exact SP rank).

``kdaptx`` The KDA prefill conv's per-element arithmetic, as read off the served
           Triton ``_causal_conv1d_fwd_kernel`` SASS on sm_121 (a 4-FFMA chain
           from +0 over the taps oldest to newest, fp32 weights, then
           ``div.full(acc, 1 + ex2.approx(0 - acc times log2e))`` and a bf16
           round), rebuilt in CUDA with explicit-rounding PTX, against the
           stock conv's q/k/v output and entire conv state bit for bit on
           varlen batches, signed zeros and large magnitudes. Arithmetic and
           physical-tail state mutants compare against the unmutated candidate,
           so a shared reference error cannot manufacture a mutation witness.
           v2 requires output/state witnesses for FMA, division and state
           indexing/writes. The ex2-ftz control instead must change the raw
           exponential while preserving downstream denominator/FP32/BF16 bits,
           bound to the current source and compiled artifacts. Any missing or
           malformed evidence fails. This convolution screen does not cover
           recurrent output/final state.

``kdaex2`` A standalone 29-finite-accumulator observation of both exponentials,
           denominators and SiLU outputs. It qualifies the ex2 equivalence
           control, not the convolution or a serving implementation by itself.

``kdafwd`` FlashKDA's two kernels (``_flash_kda_fwd_prepare`` and
           ``_flash_kda_fwd_recurrence``) at the served shape: per-kernel
           device time at ``--kda-tokens`` for ``--kda-heads`` local heads and
           selected state dtypes. ``H=1`` is an aggregate launch diagnostic;
           the prepare still launches one CTA per tile, so its total device
           time is not a single-CTA latency. Eager kernel profiles and resident
           graph power/timing windows are reported separately; no candidate.

``--ncu`` runs only the NCU-gated mHC calls (T 1024 and 2048) between
``cudaProfilerStart``/``Stop`` for ``mhc_probe.sh ORACLE_NCU=1``; with
``--ncu-part kda`` it runs the ``kdafwd`` calls instead (set
``ORACLE_NCU_KERNELS`` to the FlashKDA kernel regex).

``--numerics-only`` keeps only what does not depend on having the box to
itself: the ``onorm`` ulp comparisons and the ``mhc`` split invariance.  It
skips every timing path (graph replay, profiler device times, power windows,
the ``l2`` part) and writes ``mhc_probe_numerics.json``, so a row that shares
its GPU can gate the overrides' numerics without producing a timing number.

Run inside the serving image through ``experiments/mhc/mhc_probe.sh``.
"""
from __future__ import annotations

import argparse
import hashlib
import runpy
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import socket
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import routed_pair_oracle as rpo  # noqa: E402  (power sampler, Netdata window, ulp stats)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kda"))
import conv_gate as _kda_gate
import conv_progress as _kda_progress
from conv_gate import (KDA_P, KDA_HEADS, KDA_WIDTH, KDA_CONV_PTX_SRC, KDA_PTX_CASES,
                       KDA_PTX_MODES, KDA_GATE_CONTRACT, KDA_PTX_CFLAGS,
                       kdaex2_gate_errors, kdaptx_case_gate_errors, kdaptx_gate_errors,
                       harness_source_identity, harness_source_identity_matches,
                       require_harness_source_identity)
from conv_progress import kda_commit


def _harness_source(expected=None):
    root = Path(__file__).resolve().parents[1]
    origins = {"mhc/mhc_probe.py": [__file__],
               "kda/conv_gate.py": [_kda_gate.__file__, _kda_gate.kdaptx_gate_errors.__code__.co_filename],
               "kda/conv_progress.py": [_kda_progress.__file__, _kda_progress.kda_commit.__code__.co_filename]}
    if expected is None:
        return harness_source_identity(root, origins=origins)
    require_harness_source_identity(expected, root, origins=origins)


#: GB10 LPDDR5x peak: 8533 MT/s x 256-bit bus.  The practical plateau is
#: measured by the ``l2`` part on the same box and reported beside it.
PEAK_DRAM_GBS = 273.0
HIDDEN = 4096
HC = 4
MIX = HC * (HC + 2)  # 24
RMS_EPS = 1e-5       # config rms_norm_eps (input_layernorm.variance_epsilon too)
HC_EPS = 1e-6        # config hc_eps (hc_pre_eps and hc_sinkhorn_eps)
POST_MULT = 2.0      # Glm5NextConfig.mhc_post_mult_value default; not in config.json
SINKHORN = 20        # config hc_sinkhorn_iters
ONORM_EPS = 1e-5     # FusedRMSNormGated default; the call site passes none
HEAD_DIM = 128


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_tensors(model_dir: Path, names: list[str]) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for name in names:
        with safe_open(str(model_dir / index[name]), framework="pt", device="cpu") as f:
            out[name] = f.get_tensor(name)
    return out


def graph_ms(calls, reps: int, warmup: int = 3) -> float:
    """Mean ms per call: one CUDA graph over ``calls`` (each on its own input
    copy), replayed ``reps`` times.  Falls back to eager launch timing when the
    graph cannot be captured, and says so."""
    for _ in range(warmup):
        for c in calls:
            c()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    mode = "graph"
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for c in calls:
                c()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(graph):
            for c in calls:
                c()
        run = graph.replay
    except Exception as exc:  # noqa: BLE001
        mode = f"eager ({type(exc).__name__}: {str(exc)[:120]})"
        torch.cuda.synchronize()

        def run():
            for c in calls:
                c()
    run()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        run()
    e1.record()
    torch.cuda.synchronize()
    graph_ms.last_mode = mode
    return e0.elapsed_time(e1) / (reps * len(calls))


graph_ms.last_mode = "graph"


def copies_for(bytes_per_call: int, target: int = 256 << 20, cap: int = 64) -> int:
    return max(2, min(cap, math.ceil(target / max(bytes_per_call, 1))))


def kernel_device_us(fn, iters: int, trace_path: Path | None = None) -> dict[str, dict]:
    """Per-kernel device time from CUPTI (``key_averages``), by kernel name."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    if trace_path is None:
        context = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
    else:
        from profile_torch import prismabuild_torch_profile

        trace_path.parent.mkdir(parents=True, exist_ok=True)
        context = prismabuild_torch_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)
    previous = os.environ.get("PRISMABUILD_PROFILE_TORCH_OUT")
    if trace_path is not None:
        os.environ["PRISMABUILD_PROFILE_TORCH_OUT"] = str(trace_path)
    try:
        with context as prof:
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
    finally:
        if trace_path is not None:
            if previous is None:
                os.environ.pop("PRISMABUILD_PROFILE_TORCH_OUT", None)
            else:
                os.environ["PRISMABUILD_PROFILE_TORCH_OUT"] = previous
    out = {}
    for row in rpo.kernel_table(prof, limit=200):
        out[row["name"]] = {"calls": row["count"], "mean_us": row["self_device_us_per_call"]}
    return out


def part_onorm(args, model_dir: Path, sampler) -> dict:
    from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

    w = load_tensors(model_dir, ["model.language_model.layers.1.self_attn.o_norm.weight"])
    weight = next(iter(w.values())).cuda()
    ns = SimpleNamespace(weight=weight, bias=None, eps=ONORM_EPS, activation="sigmoid")
    ns.forward_cuda = lambda *a, **k: FusedRMSNormGated.forward_cuda(ns, *a, **k)
    native = lambda x, g: FusedRMSNormGated.forward_native(ns, x, g)  # noqa: E731
    cuda = lambda x, g: FusedRMSNormGated.forward_cuda(ns, x, g)  # noqa: E731
    out = {"weight": "layers.1.self_attn.o_norm.weight", "eps": ONORM_EPS, "cells": []}
    gen = torch.Generator(device="cuda").manual_seed(0)
    for heads in args.onorm_heads:
        for t in args.tokens:
            # x: the KDA core output (1, T, H, D), per-head magnitudes spread over
            # four binades; g: gate logits (T, H, D).  Shapes as the call site.
            scale = torch.exp2(torch.randint(-2, 3, (1, 1, heads, 1), device="cuda", generator=gen).float())
            x = (torch.randn(1, t, heads, HEAD_DIM, device="cuda", generator=gen) * scale).bfloat16()
            g = (torch.randn(t, heads, HEAD_DIM, device="cuda", generator=gen) * 2.0).bfloat16()
            # forward_cuda writes y into x in place (layer_norm_gated_fwd: y = x when
            # out_dtype is None), so every CUDA call gets its own copy of x, and x
            # stays the input the reference is computed from.
            yn = native(x, g)
            yc, yc2 = cuda(x.clone(), g), cuda(x.clone(), g)
            x64, g64, w64 = x.double(), g.double(), weight.double()
            ref = x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + ONORM_EPS) * w64 * torch.sigmoid(g64)
            ref = ref.reshape(-1, HEAD_DIM)
            yn2, yc_2 = yn.reshape(-1, HEAD_DIM), yc.reshape(-1, HEAD_DIM)
            cell = {"tokens": t, "heads": heads,
                    "native_vs_fp64": rpo.bf16_ulp_stats(yn2, ref),
                    "cuda_vs_fp64": rpo.bf16_ulp_stats(yc_2, ref),
                    "cuda_vs_native": rpo.bf16_ulp_stats(yc_2, yn2.double()),
                    "cuda_deterministic": bool(torch.equal(yc, yc2))}
            if args.numerics_only:
                out["cells"].append(cell)
                log("onorm", heads, t, "numerics only",
                    f"cuda-vs-native {cell['cuda_vs_native']['max_diff_in_bf16_ulps_of_row_max']:.2f} ulp",
                    f"native {cell['native_vs_fp64']['max_diff_in_bf16_ulps_of_row_max']:.2f}",
                    f"cuda {cell['cuda_vs_fp64']['max_diff_in_bf16_ulps_of_row_max']:.2f}")
                continue
            nbytes = 3 * t * heads * HEAD_DIM * 2
            k = copies_for(nbytes)
            xs = [x.clone() for _ in range(k)]
            gs = [g.clone() for _ in range(k)]
            for name, fn in (("native", native), ("cuda", cuda)):
                calls = [(lambda xi=xi, gi=gi, fn=fn: fn(xi, gi)) for xi, gi in zip(xs, gs)]
                t0 = time.time()
                ms = graph_ms(calls, args.reps)
                t1 = time.time()
                cell[f"{name}_ms"] = ms
                cell[f"{name}_timing_mode"] = graph_ms.last_mode
                cell[f"{name}_power"] = sampler.window(t0, t1)
            cell["floor_bytes"] = nbytes
            cell["copies"] = k
            cell["cuda_gbs"] = nbytes / cell["cuda_ms"] / 1e6
            cell["cuda_fraction_of_peak"] = cell["cuda_gbs"] / PEAK_DRAM_GBS
            cell["native_over_cuda"] = cell["native_ms"] / cell["cuda_ms"]
            out["cells"].append(cell)
            log("onorm", heads, t, f"native {cell['native_ms']:.4f} ms", f"cuda {cell['cuda_ms']:.4f} ms",
                f"x{cell['native_over_cuda']:.1f}",
                f"cuda-vs-native {cell['cuda_vs_native']['max_diff_in_bf16_ulps_of_row_max']:.2f} ulp",
                f"native {cell['native_vs_fp64']['max_diff_in_bf16_ulps_of_row_max']:.2f}",
                f"cuda {cell['cuda_vs_fp64']['max_diff_in_bf16_ulps_of_row_max']:.2f}")
            del xs, gs
    return out


def mhc_params(model_dir: Path, which: str) -> dict:
    p = f"model.language_model.layers.1.hc_{which}_"
    norm = ("model.language_model.layers.1.input_layernorm.weight" if which == "attn"
            else "model.language_model.layers.1.post_attention_layernorm.weight")
    t = load_tensors(model_dir, [p + "fn", p + "scale", p + "base", norm])
    return {"fn": t[p + "fn"].float().cuda().contiguous(), "scale": t[p + "scale"].float().cuda(),
            "base": t[p + "base"].float().cuda(), "norm": t[norm].bfloat16().cuda()}


def mhc_call(prm):
    from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_tilelang

    def call(x, residual, post, comb):
        return mhc_fused_post_pre_tilelang(x, residual, post, comb, prm["fn"], prm["scale"], prm["base"],
                                           RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN, 1, 1,
                                           norm_weight=prm["norm"], norm_eps=RMS_EPS)
    return call


def mhc_inputs(t: int, gen, call_prev):
    """Residual, x, and the previous layer's post/comb mixes as the model produces
    them: one stock call on random streams supplies realistic mixes."""
    residual = torch.randn(t, HC, HIDDEN, device="cuda", generator=gen).bfloat16()
    x = torch.randn(t, HIDDEN, device="cuda", generator=gen).bfloat16()
    post0 = torch.rand(t, HC, 1, device="cuda", generator=gen) * POST_MULT
    comb0 = torch.full((t, HC, HC), 1.0 / HC, device="cuda")
    res1, post1, comb1, _ = call_prev(x, residual, post0, comb0)
    x2 = torch.randn(t, HIDDEN, device="cuda", generator=gen).bfloat16()
    return x2, res1.contiguous(), post1.contiguous(), comb1.contiguous()


def mhc_bytes(t: int) -> dict[str, int]:
    from vllm.model_executor.kernels.mhc.tilelang_kernels import compute_num_split

    splits = compute_num_split(64, HC * HIDDEN, math.ceil(t / 64))
    stream = t * HC * HIDDEN * 2
    act = t * HIDDEN * 2
    mixes = t * (HC + HC * HC) * 4
    gemm_out = splits * t * (MIX + 1) * 4
    return {"split_k": splits,
            "post": stream + act + mixes + stream,
            "gemm": stream + MIX * HC * HIDDEN * 4 + gemm_out,
            "pre": gemm_out + stream + act + mixes,
            "floor": stream + act + mixes + stream + act + mixes}


def classify_mhc_kernel(name: str) -> str | None:
    if "mhc_post" in name:
        return "post"
    if "hc_prenorm_gemm" in name:
        return "gemm"
    if "mhc_pre_big_fuse" in name:
        return "pre"
    if "mhc_fused" in name:
        return "fused_small"
    return None


def _value_compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = (a.double() - b.double()).abs()
    return {"equal": bool(torch.equal(a, b)), "elements_differing": int((d > 0).sum()),
            "max_abs": float(d.max()), "max_abs_over_max_ref": float(d.max() / b.double().abs().max().clamp(min=1e-300))}


def part_mhc(args, model_dir: Path, sampler) -> dict:
    out = {"constants": {"rms_eps": RMS_EPS, "hc_eps": HC_EPS, "post_mult": POST_MULT, "sinkhorn": SINKHORN},
           "cells": [], "split_invariance": []}
    for which in ("attn", "ffn"):
        prm = mhc_params(model_dir, which)
        call = mhc_call(prm)
        gen = torch.Generator(device="cuda").manual_seed(1)
        for t in args.tokens:
            x, res, post, comb = mhc_inputs(t, gen, call)
            by = mhc_bytes(t)
            if not args.numerics_only:  # timing: only on a box this row has to itself
                k = copies_for(by["floor"])
                ins = [(x.clone(), res.clone(), post.clone(), comb.clone()) for _ in range(k)]
                calls = [(lambda a=a: call(*a)) for a in ins]
                t0 = time.time()
                ms = graph_ms(calls, args.reps)
                t1 = time.time()
                kern = kernel_device_us(lambda: [c() for c in calls], 2)
                per = {}
                for name, v in kern.items():
                    role = classify_mhc_kernel(name)
                    if role:
                        per[role] = {"kernel": name[:120], "mean_us": v["mean_us"], "calls": v["calls"]}
                        if role in by:
                            gbs = by[role] / (v["mean_us"] * 1e3)
                            per[role].update(bytes=by[role], gbs=gbs, fraction_of_peak=gbs / PEAK_DRAM_GBS)
                cell = {"which": which, "tokens": t, "ms_per_call": ms, "timing_mode": graph_ms.last_mode,
                        "copies": k, "bytes": by, "kernels": per, "power": sampler.window(t0, t1),
                        "floor_ms_at_peak": by["floor"] / PEAK_DRAM_GBS / 1e6,
                        "per_token_us": ms * 1e3 / t}
                out["cells"].append(cell)
                log("mhc", which, t, f"{ms:.4f} ms/call", f"{cell['per_token_us']:.3f} us/token",
                    " ".join(f"{r}={v['mean_us']:.1f}us" + (f"({v.get('fraction_of_peak', 0):.0%})" if "gbs" in v else "")
                             for r, v in per.items()))
                del ins, calls
            # Split invariance: T tokens in one call against the same tokens in 2 and 4 calls.
            if t >= 128 and t in args.split_tokens:
                full = call(x, res, post, comb)
                again = call(x, res, post, comb)
                rec = {"which": which, "tokens": t,
                       "deterministic": all(torch.equal(a, b) for a, b in zip(full, again))}
                for parts in (2, 4):
                    chunks = [call(*(v.chunk(parts, 0)[i].contiguous() for v in (x, res, post, comb)))
                              for i in range(parts)]
                    joined = [torch.cat([c[j] for c in chunks], 0) for j in range(4)]
                    rec[f"parts_{parts}"] = {
                        "split_k": [mhc_bytes(t // parts)["split_k"], by["split_k"]],
                        "residual_cur": compare(joined[0], full[0]),
                        "post_mix": compare(joined[1], full[1]),
                        "comb_mix": compare(joined[2], full[2]),
                        "layer_input": compare(joined[3], full[3]),
                        "layer_input_ulps": rpo.bf16_ulp_stats(joined[3].reshape(-1, HIDDEN),
                                                               full[3].reshape(-1, HIDDEN).double())}
                out["split_invariance"].append(rec)
                log("split", which, t, json.dumps({k2: {kk: vv.get("equal", vv) if isinstance(vv, dict) else vv
                                                        for kk, vv in v2.items()} if isinstance(v2, dict) else v2
                                                   for k2, v2 in rec.items()})[:400])
    return out


def part_mhcsplit(args, model_dir: Path) -> dict:
    """Forced split count: is the pre-norm GEMM's token-count-dependent split-k the
    only reason ``mhc`` split invariance fails?

    Stock ``_hc_prenorm_gemm_outputs`` re-imports ``compute_num_split`` on every
    call, so patching the module attribute fixes the split for one call.  Each
    (T, parts) case runs the full call and the chunked calls at one forced split
    S, for S in {the full call's stock split, the chunk's stock split, 1}; every
    call's actual split is recorded from the patched function, so a forced value
    that never reached the GEMM shows as a mismatch, not a pass.  If the outputs
    are bitwise equal at every forced S, the split is the whole cause, and an SP
    rank that passes the full batch's split would be exact.  Off --numerics-only it
    also times that rank's T/2-token call at the full batch's split against its own."""
    import vllm.model_executor.kernels.mhc.tilelang_kernels as tk
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    stock = tk.compute_num_split
    seen: list[int] = []
    out = {"deep_gemm_supported": bool(is_deep_gemm_supported()),
           "sm_count": torch.cuda.get_device_properties(0).multi_processor_count, "cases": []}

    def run(call, ins, force):
        seen.clear()

        def forced(block_k, k, grid_size):
            s = stock(block_k, k, grid_size) if force is None else force
            seen.append(s)
            return s
        tk.compute_num_split = forced
        try:
            return call(*ins), list(seen)
        finally:
            tk.compute_num_split = stock

    try:
        for which in ("attn", "ffn"):
            prm = mhc_params(model_dir, which)
            call = mhc_call(prm)
            gen = torch.Generator(device="cuda").manual_seed(1)
            for t in args.split_tokens:
                x, res, post, comb = mhc_inputs(t, gen, call)
                for parts in (2, 4):
                    s_full = stock(64, HC * HIDDEN, math.ceil(t / 64))
                    s_chunk = stock(64, HC * HIDDEN, math.ceil(t // parts / 64))
                    for label, force in (("stock", None), ("full_split", s_full), ("chunk_split", s_chunk), ("one", 1)):
                        rec = {"which": which, "tokens": t, "parts": parts, "force": label, "forced_split": force}
                        try:
                            full, sf = run(call, (x, res, post, comb), force)
                            again, _ = run(call, (x, res, post, comb), force)
                            chunks, sc = [], []
                            for i in range(parts):
                                o, s = run(call, tuple(v.chunk(parts, 0)[i].contiguous() for v in (x, res, post, comb)),
                                           force)
                                chunks.append(o)
                                sc += s
                        except Exception as exc:  # noqa: BLE001  (a split the GEMM refuses is a result)
                            rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
                            out["cases"].append(rec)
                            log("mhcsplit", which, t, parts, label, rec["error"])
                            continue
                        joined = [torch.cat([c[j] for c in chunks], 0) for j in range(4)]
                        rec.update({
                            "splits_seen_full": sf, "splits_seen_chunks": sc,
                            "deterministic": all(torch.equal(a, b) for a, b in zip(full, again)),
                            "residual_cur": compare(joined[0], full[0]), "post_mix": compare(joined[1], full[1]),
                            "comb_mix": compare(joined[2], full[2]), "layer_input": compare(joined[3], full[3]),
                            "layer_input_ulps": rpo.bf16_ulp_stats(joined[3].reshape(-1, HIDDEN),
                                                                   full[3].reshape(-1, HIDDEN).double())})
                        rec["bitwise"] = all(rec[k]["equal"] for k in ("residual_cur", "post_mix", "comb_mix", "layer_input"))
                        out["cases"].append(rec)
                        log("mhcsplit", which, t, parts, label, f"splits full {sf} chunks {sc}",
                            f"bitwise {rec['bitwise']}", f"post max_abs {rec['post_mix']['max_abs']:.3g}")
                if not args.numerics_only:  # timing: only on a box this row has to itself
                    # The price of exactness on an SP rank: its T/2-token call at the full batch's split
                    # against the same call at its own stock split.  The patch is live through capture.
                    half = tuple(v[: t // 2].contiguous() for v in (x, res, post, comb))
                    s_full = stock(64, HC * HIDDEN, math.ceil(t / 64))
                    k = copies_for(mhc_bytes(t // 2)["floor"])
                    cell = {"which": which, "tokens": t, "half_tokens": t // 2, "s_full": s_full,
                            "s_half_stock": stock(64, HC * HIDDEN, math.ceil(t // 2 / 64)), "copies": k}
                    for label, force in (("half_stock", None), ("half_at_s_full", s_full)):
                        ins = [tuple(v.clone() for v in half) for _ in range(k)]
                        calls = [(lambda a=a: call(*a)) for a in ins]

                        def forced(block_k, kk, grid_size, force=force):
                            return stock(block_k, kk, grid_size) if force is None else force
                        tk.compute_num_split = forced
                        try:
                            cell[label] = {"ms_per_call": graph_ms(calls, args.reps), "mode": graph_ms.last_mode}
                        finally:
                            tk.compute_num_split = stock
                        del ins, calls
                    cell["exact_cost_ms_per_call"] = cell["half_at_s_full"]["ms_per_call"] - cell["half_stock"]["ms_per_call"]
                    out.setdefault("timing", []).append(cell)
                    log("mhcsplit-time", which, t, json.dumps({kk: cell[kk] for kk in ("s_full", "s_half_stock",
                                                                                     "half_stock", "half_at_s_full")}))
                del x, res, post, comb
    finally:
        tk.compute_num_split = stock
    return out


#: TP2 local KDA projection (64 heads x 128 / 2) and the short-conv width
#: (``linear_attn_config.short_conv_kernel_size``).


def kda_conv_case(lens, has, p: int, state_len: int, layout: str, gen) -> dict:
    """One varlen prefill batch for the KDA conv: merged q|k|v input, the serve's fp32 merged
    weight, a bf16 conv-state cache in ``layout`` (``SD`` stores (state_len, dim) and the layer
    transposes it, ``DS`` stores (dim, state_len)), and the precomputed conv metadata."""
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    t, n = sum(lens), len(lens) + 2
    qkv = torch.randn(t, 3 * p, device="cuda", generator=gen).bfloat16()
    weight = torch.randn(3 * p, KDA_WIDTH, device="cuda", generator=gen) * 0.5
    if layout == "SD":
        store = torch.randn(n, state_len, 3 * p, device="cuda", generator=gen).bfloat16()
    else:
        store = torch.randn(n, 3 * p, state_len, device="cuda", generator=gen).bfloat16()
    qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32)
    nums, batch_ptr, offs = compute_causal_conv1d_metadata(qsl, device=torch.device("cuda"))
    return {"qkv": qkv, "weight": weight, "store": store, "layout": layout,
            "idx": torch.tensor(list(range(n))[::-1][: len(lens)], dtype=torch.int32, device="cuda"),
            "has": torch.tensor(has, dtype=torch.bool, device="cuda"), "qsl": qsl.cuda(),
            "md": SimpleNamespace(nums_dict=nums, batch_ptr=batch_ptr, token_chunk_offset_ptr=offs)}


def kda_state_view(c: dict, store: torch.Tensor) -> torch.Tensor:
    return store.transpose(-1, -2) if c["layout"] == "SD" else store


def kda_stock(conv_fn, c: dict, store: torch.Tensor, p: int):
    """The stock layer: one merged conv over q|k|v, then a split into strided views."""
    out = conv_fn(c["qkv"].transpose(0, 1), c["weight"], None, activation="silu",
                  conv_states=kda_state_view(c, store), has_initial_state=c["has"], cache_indices=c["idx"],
                  query_start_loc=c["qsl"], metadata=c["md"]).transpose(0, 1)
    return out.split(p, dim=-1)


def kda_split(gp, conv_fn, c: dict, store: torch.Tensor, p: int):
    return gp.conv_split(conv_fn, c["qkv"], c["weight"], None, kda_state_view(c, store), c["has"], c["idx"],
                         c["qsl"], c["md"], p)


def kda_flashkda_inputs(q, k, v):
    """What ``_flashkda_prefill`` hands FlashKDA: ``_rearr`` then ``.contiguous()``."""
    return [x.reshape(1, -1, KDA_HEADS, HEAD_DIM).contiguous() for x in (q, k, v)]


def part_kdaconv(args, sampler) -> dict:
    """The KDA prefill conv per q/k/v slice (``TESSERA_GLM53_KDA_CONV_SPLIT``) against the stock
    merged conv: bitwise q/k/v and conv state on varlen batches, then the per-layer time of
    conv plus FlashKDA's dense-input copies on both paths."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

    from tessera.serving import glm53_prefill as gp

    p = KDA_P
    out = {"p": p, "width": KDA_WIDTH, "heads": KDA_HEADS, "numerics": [], "cells": []}
    gen = torch.Generator(device="cuda").manual_seed(3)
    cases = [((5, 2, 9, 700), (True, False, True, False)), ((2, 1, 3), (False, True, True)),
             ((2048,), (True,)), ((512, 512, 512, 512), (False, True, False, True)), ((1,), (True,))]
    for layout in ("SD", "DS"):
        for state_len in (KDA_WIDTH - 1, KDA_WIDTH - 1 + 3):
            for lens, has in cases:
                c = kda_conv_case(lens, has, p, state_len, layout, gen)
                sa, sb = c["store"].clone(), c["store"].clone()
                ref = kda_stock(causal_conv1d_fn, c, sa, p)
                got = kda_split(gp, causal_conv1d_fn, c, sb, p)
                torch.cuda.synchronize()
                row = {"layout": layout, "state_len": state_len, "lens": list(lens), "has": list(has),
                       "q": compare(got[0], ref[0]), "k": compare(got[1], ref[1]), "v": compare(got[2], ref[2]),
                       "state": compare(sb, sa), "state_changed": not torch.equal(sa, c["store"]),
                       "split_dense": [bool(x.is_contiguous()) for x in got],
                       "split_rearr_dense": [bool(x.reshape(1, -1, KDA_HEADS, HEAD_DIM).is_contiguous()) for x in got],
                       "stock_rearr_dense": [bool(x.reshape(1, -1, KDA_HEADS, HEAD_DIM).is_contiguous()) for x in ref]}
                row["bitwise"] = all(row[r]["equal"] for r in ("q", "k", "v", "state"))
                out["numerics"].append(row)
                log("kdaconv", layout, state_len, lens, "bitwise" if row["bitwise"] else "DIFFERS",
                    row["split_dense"], row["stock_rearr_dense"])
    out["bitwise_all"] = all(r["bitwise"] for r in out["numerics"])
    if args.numerics_only:
        return out
    for t in args.kda_tokens:
        c = kda_conv_case((t,), (True,), p, KDA_WIDTH - 1, "SD", gen)
        io = 3 * p * t * 2  # one pass over the merged q|k|v activations, bf16
        model = {"stock": 4 * io, "split": 2 * io}  # conv read+write, plus the copies' read+write
        k = copies_for(model["split"])
        ins = [(c["qkv"].clone(), c["store"].clone()) for _ in range(k)]
        arms = {
            "stock": lambda a: kda_flashkda_inputs(*kda_stock(causal_conv1d_fn, dict(c, qkv=a[0]), a[1], p)),
            "split": lambda a: kda_flashkda_inputs(*kda_split(gp, causal_conv1d_fn, dict(c, qkv=a[0]), a[1], p)),
        }
        cell = {"tokens": t, "copies": k, "bytes_model": model}
        for name, fn in arms.items():
            calls = [(lambda a=a, fn=fn: fn(a)) for a in ins]
            t0 = time.time()
            ms = graph_ms(calls, args.reps)
            t1 = time.time()
            kern = kernel_device_us(lambda calls=calls: [cl() for cl in calls], 2)
            gbs = model[name] / (ms * 1e6)
            cell[name] = {"ms_per_layer": ms, "timing_mode": graph_ms.last_mode, "gbs": gbs,
                          "fraction_of_peak": gbs / PEAK_DRAM_GBS,
                          "floor_ms_at_peak": model[name] / PEAK_DRAM_GBS / 1e6,
                          "ms_per_step_34_layers": 34 * ms, "power": sampler.window(t0, t1),
                          "kernels": {n[:120]: v for n, v in kern.items()}}
            log("kdaconv", t, name, f"{ms * 1e3:.1f} us/layer", f"{gbs:.1f} GB/s", graph_ms.last_mode)
        cell["saved_ms_per_step_34_layers"] = 34 * (cell["stock"]["ms_per_layer"] - cell["split"]["ms_per_layer"])
        out["cells"].append(cell)
        del ins
    return out











def bits_compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Bit-pattern equality (``torch.equal`` treats +0 and -0 as equal), plus the value view."""
    word = torch.int32 if a.dtype == torch.float32 else torch.int16
    ai, bi = a.contiguous().view(word), b.contiguous().view(word)
    return {"bit_equal": bool(torch.equal(ai, bi)), "bits_differing": int((ai != bi).sum()),
            "value": _value_compare(a, b)}


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Keep value diagnostics, but make the advertised equality gate bitwise."""
    bits = bits_compare(a, b)
    return {**bits["value"], "equal": bits["bit_equal"], "bits_differing": bits["bits_differing"]}


def load_kda_ptx():
    """One JIT build owns the convolution and its arithmetic observation control."""
    from torch.utils.cpp_extension import load_inline

    return load_inline(name="tessera_kda_conv_ptx", cpp_sources="torch::Tensor conv_ref(torch::Tensor x, "
                      "torch::Tensor w, torch::Tensor st, int64_t state_len, torch::Tensor qsl, torch::Tensor idx, "
                      "torch::Tensor has, int64_t mode); std::vector<torch::Tensor> ex2_control(torch::Tensor accs);",
                      cuda_sources=KDA_CONV_PTX_SRC, functions=["conv_ref", "ex2_control"],
                      extra_cuda_cflags=KDA_PTX_CFLAGS, verbose=False)


def part_kdaex2(args) -> dict:
    """A minimal FTZ intermediate witness; this alone does not admit the conv screen."""
    ext = load_kda_ptx()
    # Values cross ex2's -126 normal/subnormal boundary and the input's
    # zero/subnormal boundary. Extremal finite accs also exercise overflow in z.
    tiny = torch.finfo(torch.float32).tiny
    c = 1.44269502162933349609375
    values = [0.0, -0.0, 2.0**-149, -2.0**-149, 2.0**-127, -2.0**-127,
              tiny, -tiny, 1.0, -1.0, 32.0, -32.0, 64.0, -64.0,
              torch.finfo(torch.float32).max, -torch.finfo(torch.float32).max]
    values += [v / c for v in (125.5, 126.0, 126.25, 127.0, 128.0, 140.0, 149.0, 150.0)]
    values += [-v / c for v in (125.5, 126.0, 127.0, 128.0, 140.0)]
    accs = torch.tensor(values, dtype=torch.float32, device="cuda")
    raw, bf16 = ext.ex2_control(accs)
    torch.cuda.synchronize()
    bits = raw.contiguous().view(torch.int32)
    input_subnormal = (raw[:, 1].abs() < tiny) & (raw[:, 1] != 0)
    changed = bits[:, 2] != bits[:, 3]
    output_subnormal = (raw[:, 2] > 0) & (raw[:, 2] < tiny)
    out = {"schema": "tessera.kda_ex2_control.v1", "finite_accumulator_count": len(values),
           "all_accumulators_finite": bool(torch.isfinite(accs).all()),
           "columns": ["acc", "z", "ex2", "ex2_ftz",
           "den", "den_ftz", "silu_fp32", "silu_fp32_ftz"],
           "raw_fp32_bits": bits.cpu().tolist(),
           "raw_bf16_bits": bf16.contiguous().view(torch.int16).cpu().tolist(),
           "ex2": bits_compare(raw[:, 2], raw[:, 3]),
           "denominator": bits_compare(raw[:, 4], raw[:, 5]),
           "output_fp32": bits_compare(raw[:, 6], raw[:, 7]),
           "output_bf16": bits_compare(bf16[:, 0], bf16[:, 1]),
           "input_subnormal_count": int(input_subnormal.sum()),
           "input_subnormal_ex2_is_one_both": bool(((bits[input_subnormal, 2] == 0x3f800000) &
                                                      (bits[input_subnormal, 3] == 0x3f800000)).all()),
           "changed_ex2_is_subnormal_flushed_to_positive_zero": bool((output_subnormal[changed] &
                                                                       (bits[changed, 3] == 0)).all()),
           "ptx_source_sha256": hashlib.sha256(KDA_CONV_PTX_SRC.encode()).hexdigest(),
           "cuda_flags": list(KDA_PTX_CFLAGS), "compiled_module": ext.__file__,
           "compiled_module_sha256": hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest()}
    # Raw equality is the observable here. An inf-inf value subtraction is
    # undefined and would add NaN summary values to otherwise exact records.
    for key in ("ex2", "denominator", "output_fp32", "output_bf16"):
        out[key].pop("value")
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None and os.environ.get("CUDA_HOME"):
        candidate = Path(os.environ["CUDA_HOME"]) / "bin/cuobjdump"
        if candidate.is_file():
            cuobjdump = str(candidate)
    if cuobjdump:
        sass = subprocess.run([cuobjdump, "--dump-sass", ext.__file__], capture_output=True,
                              text=True, timeout=30, check=True).stdout
        path = Path(args.out) / "kda_ptx_control.sass"
        path.write_text(sass)
        out["sass"] = {"path": str(path), "sha256": hashlib.sha256(sass.encode()).hexdigest()}
    else:
        out["sass"] = {"error": "cuobjdump unavailable"}
    out["errors"] = kdaex2_gate_errors(out)
    out["control_passed"] = not out["errors"]
    log("kdaex2", {k: out[k] for k in ("control_passed", "ex2", "denominator", "input_subnormal_count")})
    return out


def part_kdaptx(args) -> dict:
    """The conv's served per-element arithmetic, rebuilt with explicit-rounding PTX, against the
    stock Triton conv bit for bit (the exactness gate for fusing the conv into FlashKDA's loads)."""
    from vllm.model_executor.layers.mamba.ops import causal_conv1d as stock_module

    causal_conv1d_fn = stock_module.causal_conv1d_fn
    ext = load_kda_ptx()
    p = KDA_P
    gen = torch.Generator(device="cuda").manual_seed(11)
    out = {"p": p, "width": KDA_WIDTH, "modes": KDA_PTX_MODES, "cases": [],
           "gate_contract": KDA_GATE_CONTRACT, "ex2_equivalence": part_kdaex2(args),
           "coverage": {"compared": ["q", "k", "v", "conv_state"],
                        "not_compared": ["recurrent_output", "final_recurrent_state"],
                        "reason": "convolution reference only; no fused recurrent kernel exists"},
           "source_bindings": {
               "stock_conv_file": stock_module.__file__,
               "stock_conv_sha256": hashlib.sha256(Path(stock_module.__file__).read_bytes()).hexdigest(),
               "ptx_source_sha256": hashlib.sha256(KDA_CONV_PTX_SRC.encode()).hexdigest()}}
    for layout in ("SD", "DS"):
        for state_len in (KDA_WIDTH - 1, KDA_WIDTH - 1 + 3):
            for name, lens, has, scale, zero_frac in KDA_PTX_CASES:
                c = kda_conv_case(lens, has, p, state_len, layout, gen)
                if scale != 1.0:
                    c["qkv"] = (c["qkv"].float() * scale).bfloat16()
                if zero_frac:
                    u = torch.rand(c["qkv"].shape, device="cuda", generator=gen)
                    c["qkv"] = torch.where(u < zero_frac / 2, torch.zeros_like(c["qkv"]), c["qkv"])
                    c["qkv"] = torch.where((u >= zero_frac / 2) & (u < zero_frac),
                                           torch.full_like(c["qkv"], -0.0), c["qkv"])
                ref_state = c["store"].clone()
                ref = causal_conv1d_fn(c["qkv"].transpose(0, 1), c["weight"], None, activation="silu",
                                       conv_states=kda_state_view(c, ref_state), has_initial_state=c["has"],
                                       cache_indices=c["idx"], query_start_loc=c["qsl"],
                                       metadata=c["md"]).transpose(0, 1)
                row = {"layout": layout, "state_len": state_len, "case": name, "lens": list(lens),
                       "has": list(has), "cache_indices": c["idx"].tolist(),
                       "input_strides": list(c["qkv"].stride()),
                       "state_strides": list(kda_state_view(c, c["store"]).stride()),
                       "ref_dense": bool(ref.is_contiguous())}
                for mode, label in KDA_PTX_MODES.items():
                    got_state = c["store"].clone()
                    st = kda_state_view(c, got_state)
                    got = ext.conv_ref(c["qkv"], c["weight"], st, state_len, c["qsl"], c["idx"], c["has"], mode)
                    torch.cuda.synchronize()
                    row[label] = bits_compare(got, ref)
                    row[label]["qkv"] = {key: bits_compare(a, b) for key, a, b in
                                         zip(("q", "k", "v"), got.split(p, -1), ref.split(p, -1))}
                    row[label]["conv_state"] = bits_compare(got_state, ref_state)
                    if mode == 0:
                        candidate, candidate_state = got, got_state
                    else:
                        row[label]["vs_candidate"] = bits_compare(got, candidate)
                        row[label]["state_vs_candidate"] = bits_compare(got_state, candidate_state)
                out["cases"].append(row)
                log("kdaptx", layout, state_len, name,
                    {lab: row[lab]["bits_differing"] for lab in KDA_PTX_MODES.values()})
    out["served_output_bit_equal_all"] = all(r["served_sass"]["bit_equal"] for r in out["cases"])
    out["served_conv_state_bit_equal_all"] = all(r["served_sass"]["conv_state"]["bit_equal"] for r in out["cases"])
    out["served_bit_equal_all"] = out["served_output_bit_equal_all"] and out["served_conv_state_bit_equal_all"]
    out["mutants_seen"] = {lab: any(not r[lab]["vs_candidate"]["bit_equal"] or
                                    not r[lab]["state_vs_candidate"]["bit_equal"] for r in out["cases"])
                           for lab in list(KDA_PTX_MODES.values())[1:]}
    return out


def kdafwd_call(t: int, h: int, state_dtype, gen):
    """One served-shape FlashKDA prefill call (varlen, one sequence, state in and out)."""
    import vllm._flashkda_C  # noqa: F401

    ops = torch.ops._flashkda_C
    q, k, v, g = (torch.randn(1, t, h, HEAD_DIM, device="cuda", generator=gen).bfloat16() for _ in range(4))
    beta = torch.randn(1, t, h, device="cuda", generator=gen).bfloat16()
    a_log = torch.randn(h, device="cuda", generator=gen) * 0.1
    dt_bias = torch.randn(h, HEAD_DIM, device="cuda", generator=gen) * 0.1
    init = (torch.randn(1, h, HEAD_DIM, HEAD_DIM, device="cuda", generator=gen) * 0.01).to(state_dtype)
    final = torch.empty_like(init)
    cu = torch.tensor([0, t], dtype=torch.int32, device="cuda")
    ws = torch.empty(int(ops.get_workspace_size(t, h, 1)), dtype=torch.uint8, device="cuda")
    o = torch.empty(1, t, h, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    call = lambda: ops.fwd(q, k, v, g, beta, HEAD_DIM ** -0.5, o, ws, a_log, dt_bias, -5.0, init, final, cu,
                           None, None)
    call.workspace_bytes = ws.numel()
    call.resident_bytes = sum(a.numel() * a.element_size() for a in
                              (q, k, v, g, beta, a_log, dt_bias, init, final, cu, ws, o))
    return call


def kda_steady(graph, copies: int, seconds: float) -> dict:
    """Completed stock graph calls; the event interval includes host replay gaps."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    t0, begin = time.time(), time.perf_counter()
    replays = 0
    while time.perf_counter() - begin < seconds:
        for _ in range(32):
            graph.replay()
        torch.cuda.synchronize()
        replays += 32
    end.record()
    torch.cuda.synchronize()
    t1 = time.time()
    elapsed_ms = start.elapsed_time(end)
    return {"interval_unix": [t0, t1], "duration_s": t1 - t0, "graph_replays": replays,
            "completed_calls": replays * copies, "cuda_interval_ms": elapsed_ms,
            "graph_ms_per_call": elapsed_ms / (replays * copies)}


def kda_raw_netdata(label: str, host: str, t0: float, t1: float) -> dict:
    """Reuse the box instrument's fetcher and retain every raw returned group."""
    import box_power_window as bpw

    contexts = [(c, dims) for c, dims in bpw.SERIES if c != "nvidia_smi.gpu_utilization"]
    out = {"box": label, "request_host": host, "phase_interval_unix": [t0, t1],
           "query_window_unix": [math.floor(t0), math.ceil(t1)], "series": {},
           "energy_status": "unqualified; full raw groups retained, coverage agreement required"}
    for context, dims in contexts:
        try:
            got = bpw._fetch(host, context, dims, math.floor(t0), math.ceil(t1), 0)
            raw = got.get("raw_doc", got["doc"])
            out["series"][context] = {"query": got["url"], "raw_response": raw,
                                      "returned_view": raw.get("view")}
        except Exception as exc:
            out["series"][context] = {"error": f"{type(exc).__name__}: {exc}"}
    return out




def part_kdafwd(args, sampler) -> dict:
    """FlashKDA prepare and recurrence device time at the served shape, per local head count."""
    import vllm._flashkda_C as stock_module

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    gen = torch.Generator(device="cuda").manual_seed(13)
    props = torch.cuda.get_device_properties(0)
    out_dir = Path(args.out)
    hosts = dict(v.split("=", 1) for v in args.kda_netdata_hosts)
    if set(hosts) != {"sparky", "sparklina"}:
        raise ValueError("stock timing requires both named Sparks' Netdata endpoints")
    out = {"schema": "tessera.kda_stock_screen.v1", "sms": props.multi_processor_count, "cells": [],
           "scope": "stock synthetic resident kernel screen; no candidate or serving comparison",
           "source_bindings": {"stock_module": stock_module.__file__,
                               "stock_module_sha256": hashlib.sha256(Path(stock_module.__file__).read_bytes()).hexdigest(),
                               "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               "harness_source_identity": _harness_source()},
           "primary_state_dtype": "float32 (pinned model kda_state_dtype default)",
           "energy_status": "unqualified until fast-sampler/Netdata coverage agreement", "startup": []}
    cells = []
    for t in args.kda_tokens:
        for h in args.kda_heads:
            for dtype in args.kda_state_dtypes:
                t0 = time.time()
                call = kdafwd_call(t, h, getattr(torch, dtype), gen)
                copies = copies_for(call.resident_bytes)
                calls = [call] + [kdafwd_call(t, h, getattr(torch, dtype), gen) for _ in range(copies - 1)]
                for _ in range(3):
                    for fn in calls:
                        fn()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for fn in calls:
                        fn()
                graph.replay()
                torch.cuda.synchronize()
                out["startup"].append({"tokens": t, "heads": h, "state_dtype": dtype,
                                       "interval_unix": [t0, time.time()], "copies": copies})
                cells.append((t, h, dtype, calls, graph))
    primary_trace = os.environ.get("PRISMABUILD_PROFILE_TORCH_OUT")
    for round_index in range(args.kda_rounds):
        ordered = cells if round_index % 2 == 0 else list(reversed(cells))
        for t, h, dtype, calls, graph in ordered:
            copies, tiles = len(calls), (t + 15) // 16
            label = f"r{round_index}-t{t}-h{h}-{dtype}"
            trace = Path(primary_trace) if not out["cells"] and primary_trace else out_dir / f"{label}.trace.json.gz"
            t0 = time.time()
            kern = kernel_device_us(lambda: [fn() for fn in calls], args.reps, trace)
            profile_end = time.time()
            cell = {"tokens": t, "heads": h, "state_dtype": dtype, "round": round_index, "tiles": tiles,
                    "role": "native-default-state" if dtype == "float32" else "alternate-state-diagnostic",
                    "copies": copies, "resident_bytes": sum(fn.resident_bytes for fn in calls),
                    "workspace_bytes_per_copy": calls[0].workspace_bytes,
                    "prepare_grid_from_pinned_source": [tiles + 1, h, 1],
                    "workspace_tile_upper_bound": tiles + 1,
                    "h1_scope": "aggregate prepare launch diagnostic; not a single-CTA measurement" if h == 1 else None,
                    "profile": {"interval_unix": [t0, profile_end], "trace": str(trace),
                                "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest()},
                    "kernels": kern}
            for role, key in (("prepare", "_flash_kda_fwd_prepare"), ("recurrence", "_flash_kda_fwd_recurrence")):
                us = [v["mean_us"] for n, v in kern.items() if key in n]
                if len(us) != 1:
                    raise RuntimeError(f"stock profile has ambiguous or absent {role} kernel: {us}")
                cell[f"{role}_us"] = us[0]
            cell["recurrence_us_per_tile"] = cell["recurrence_us"] / tiles
            cell["settle"] = kda_steady(graph, copies, args.kda_warm_s)
            cell["steady"] = kda_steady(graph, copies, args.kda_steady_s)
            t0, t1 = cell["steady"]["interval_unix"]
            cell["power"] = sampler.window(t0, t1)
            cell["power"]["raw_samples"] = [[stamp, watts] for stamp, watts in sampler.samples if t0 <= stamp <= t1]
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {label: pool.submit(kda_raw_netdata, label, host, t0, t1) for label, host in hosts.items()}
                cell["netdata_both_boxes"] = {label: f.result() for label, f in futures.items()}
            out["cells"].append(cell)
            _harness_source(out["source_bindings"]["harness_source_identity"])
            kda_commit(out, out_dir / "kdafwd_partial.json", len(out["cells"]))
            log("kdafwd", label, cell["prepare_us"], cell["recurrence_us"], cell["steady"]["graph_ms_per_call"],
                cell["power"].get("mean_w"), "W; energy unqualified")
    return out


def part_l2(args, sampler) -> dict:
    props = torch.cuda.get_device_properties(0)
    out = {"l2_cache_bytes": getattr(props, "L2_cache_size", None), "sms": props.multi_processor_count,
           "curve": []}
    for mb in args.l2_sizes:
        n = (mb << 20) // 2
        buf = torch.randn(n, device="cuda").bfloat16()
        acc = torch.empty((), device="cuda", dtype=torch.float32)
        reads = max(8, min(400, (2048 << 20) // (mb << 20)))
        calls = [lambda: torch.sum(buf, dim=0, dtype=torch.float32, out=acc)] * reads
        t0 = time.time()
        ms = graph_ms(calls, max(3, args.reps // 4))
        t1 = time.time()
        gbs = (mb << 20) / (ms * 1e6)
        out["curve"].append({"mb": mb, "ms_per_read": ms, "gbs": gbs, "timing_mode": graph_ms.last_mode,
                             "power": sampler.window(t0, t1)})
        log("l2", mb, "MB", f"{gbs:.1f} GB/s")
        del buf
    return out


def part_ncu(args, model_dir: Path) -> dict:
    prm = mhc_params(model_dir, "attn")
    call = mhc_call(prm)
    gen = torch.Generator(device="cuda").manual_seed(2)
    ins = {t: mhc_inputs(t, gen, call) for t in args.ncu_tokens}
    for t, a in ins.items():
        call(*a)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for t, a in ins.items():
        call(*a)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    return {"ncu_tokens": args.ncu_tokens}


def part_ncu_kda(args) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(13)
    calls = [kdafwd_call(t, h, torch.float32, gen) for t in args.kda_tokens for h in args.kda_heads]
    for call in calls:
        call()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for call in calls:
        call()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    return {"kda_tokens": args.kda_tokens, "kda_heads": args.kda_heads, "state_dtype": "float32"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported")
    ap.add_argument("--parts", default="l2,onorm,mhc")
    ap.add_argument("--tokens", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--split-tokens", type=int, nargs="+", default=[512, 1024, 2048, 8192])
    ap.add_argument("--onorm-heads", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--l2-sizes", type=int, nargs="+",
                    default=[1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 128, 256])
    ap.add_argument("--ncu-tokens", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--kda-tokens", type=int, nargs="+", default=[2048, 8192])
    ap.add_argument("--kda-heads", type=int, nargs="+", default=[32, 1],
                    help="kdafwd local heads (32 is TP2; 1 is an aggregate fusion-ceiling diagnostic)")
    ap.add_argument("--kda-state-dtypes", choices=("float32", "bfloat16"), nargs="+", default=["float32", "bfloat16"])
    ap.add_argument("--kda-steady-s", type=float, default=20.0)
    ap.add_argument("--kda-warm-s", type=float, default=3.0)
    ap.add_argument("--kda-rounds", type=int, default=2)
    ap.add_argument("--kda-netdata-hosts", nargs="+", default=["sparky=sparky", "sparklina=sparklina"])
    ap.add_argument("--ncu-part", choices=("mhc", "kda"), default="mhc")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--ncu", action="store_true")
    ap.add_argument("--numerics-only", action="store_true",
                    help="ulp and split-invariance checks only; no timing (a shared-GPU row may run it)")
    ap.add_argument("--stub", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if min(args.kda_steady_s, args.kda_warm_s, args.kda_rounds, args.reps) <= 0:
        ap.error("KDA durations, rounds and profile repetitions must be positive")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model)
    import vllm

    meta = {"host": os.environ.get("HOST_NAME", socket.gethostname()), "image": os.environ.get("ORACLE_IMAGE"),
            "tessera_head": os.environ.get("TESSERA_HEAD"), "tessera_state": os.environ.get("TESSERA_STATE"),
            "pb_action": os.environ.get("PB_ACTION_KEY"), "torch": torch.__version__, "vllm": vllm.__version__,
            "device": torch.cuda.get_device_name(0), "model": str(model_dir), "argv": sys.argv,
            "peak_dram_gbs": PEAK_DRAM_GBS, "utc_start": time.time()}
    log("meta", json.dumps(meta))
    meta["harness_source_identity"] = _harness_source()
    if args.ncu:
        res = {"meta": meta, "ncu": part_ncu_kda(args) if args.ncu_part == "kda" else part_ncu(args, model_dir)}
        _harness_source(meta["harness_source_identity"])
        (out_dir / "mhc_probe_ncu.json").write_text(json.dumps(res, indent=1) + "\n")
        _harness_source(meta["harness_source_identity"])
        return 0
    sampler = rpo.PowerSampler()
    sampler.start()
    res = {"meta": meta}
    name = "mhc_probe_numerics.json" if args.numerics_only else "mhc_probe.json"
    meta["numerics_only"] = args.numerics_only
    failures = []
    for part in args.parts.split(","):
        if part == "l2" and args.numerics_only:
            raise SystemExit("--numerics-only has no l2 part (the L2 curve is a timing)")
        if part == "l2":
            res["l2"] = part_l2(args, sampler)
        elif part == "onorm":
            res["onorm"] = part_onorm(args, model_dir, sampler)
        elif part == "mhc":
            res["mhc"] = part_mhc(args, model_dir, sampler)
        elif part == "kdaconv":
            res["kdaconv"] = part_kdaconv(args, sampler)
        elif part == "mhcsplit":
            res["mhcsplit"] = part_mhcsplit(args, model_dir)
        elif part == "kdaptx":
            res["kdaptx"] = part_kdaptx(args)
        elif part == "kdaex2":
            res["kdaex2"] = part_kdaex2(args)
            if not res["kdaex2"]["control_passed"]:
                failures = ["ex2 intermediate control failed"]
        elif part == "kdafwd":
            if args.numerics_only:
                raise SystemExit("--numerics-only has no kdafwd part (it is a timing)")
            res["kdafwd"] = part_kdafwd(args, sampler)
        else:
            raise SystemExit(f"unknown part {part}")
        if part == "kdaptx":
            failures = kdaptx_gate_errors(res["kdaptx"])
            res["kdaptx"]["gate"] = {"passed": not failures, "errors": failures}
        _harness_source(meta["harness_source_identity"])
        (out_dir / name).write_text(json.dumps(res, indent=1) + "\n")
        if failures:
            break
    sampler.stop_flag = True
    meta["utc_end"] = time.time()
    meta["power_sampler"] = sampler.source
    if "kdafwd" in res:
        res["netdata"] = {"note": "full unfiltered per-window responses for both boxes are in kdafwd.cells",
                          "energy_status": "unqualified; no whole-run or cropped-series energy claim"}
    else:
        try:
            res["netdata"] = rpo.netdata_window(meta["utc_start"], meta["utc_end"])
        except Exception as exc:  # noqa: BLE001
            res["netdata"] = {"error": f"{type(exc).__name__}: {exc}"}
    _harness_source(meta["harness_source_identity"])
    if "kdafwd" in res:
        kda_commit(res, out_dir / name, len(res["kdafwd"]["cells"]), "publish")
    else:
        (out_dir / name).write_text(json.dumps(res, indent=1) + "\n")
    log("done", out_dir / name)
    _harness_source(meta["harness_source_identity"])
    if failures:
        log("kdaptx gate FAILED", "; ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
