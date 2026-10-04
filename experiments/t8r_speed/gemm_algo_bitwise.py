"""Which cuBLASLt algorithms reproduce the served BF16 projection GEMMs bit for bit, and how fast are they.

The A8SESHMN prefill chunk (2048 tokens, TP2 rank shapes) spends ~171 ms in
BF16 projection GEMMs (the A8S artifact's ignored Linears, served by vLLM's
``UnquantizedLinearMethod``, so ``torch.nn.functional.linear``). A faster
algorithm is a bitwise lever only if every output element is accumulated in
the same order. That is checked here, not assumed (tessera#806).

Per shape (``out[M, N] = x[M, K] @ w[N, K]^T``, bf16 in and out, fp32 compute):

1. At ``M0`` (2048), random-normal input: ``F.linear`` gives the reference,
   twice (determinism), and its kernel names. Candidates are cuBLASLt's
   heuristics (up to 64, 64 MiB workspace) plus an exhaustive sweep (every
   algorithm id x tile x stages x CTA swizzle, with split-K 1, no reduction
   scheme and custom option 0, kept when ``cublasLtMatmulAlgoCheck`` accepts
   it). Each runs twice; it is bitwise when both outputs equal the reference
   (int16 view). Then it is timed: 10 calls between CUDA events after 3 warm-up
   calls. A non-bitwise candidate records the fraction of elements that differ:
   this is what a changed accumulation order looks like.
2. Survivors are the bitwise candidates faster than the reference (up to 8).
   Each is checked again, at every ``--m``, on every input distribution, against
   ``F.linear`` at that M and distribution:
   - ``normal``: x ~ N(0, 1), the step-1 draw;
   - ``normal2``: an independent draw, x ~ N(0, 3^2);
   - ``adversarial``: sign * 2^e * (1 + u), e uniform in [-12, 8], mixed signs,
     finite products and sums;
   - ``real``: recorded GLM-5.3 activations, where a capture exists for this K
     (``CAPTURE``). Shapes without one say so (``real: null``).
   A survivor counts at an M only if it is bitwise on every distribution there.
3. Per M, the fastest counting survivor gets an interleaved A/B against the
   reference (reference, pick, pick, reference; 3 rounds) and board power over
   2 s of each. Timing in a shared gap is a screen [S]; bitwise is the result.
4. Row tiles (``--row-tiles``): whether ``F.linear`` on ``tm`` rows stores the
   same bits as those rows of the ``M0`` call, on every distribution, and the
   same for the three fastest survivors run at ``tm``. A producer split into
   row tiles (so its all-reduce can start on the first one) is bitwise only
   where this holds.

The descriptors are ``pinned_gemm.cpp`` beside this file; a serving lever must
run that same file (its sha256 is in the output) for this evidence to carry. Writes ``<out>/gemm_algo_bitwise.json`` after every shape and
``<out>/pinned_gemm_table.json``: the lever's table (schema
``tessera.glm53_pinned_gemm.v1``), keyed to this process's cuBLASLt version,
torch version, CUDA runtime and device.

Weights are random (w ~ N(0, 0.02^2)): the accumulation order does not depend
on the values. The reference counts as the serve only when this process picks
the served kernel (``matches_served_kernel``).

Qualification is closed by three predicates, not one: the reference must match
the served kernel AND repeat bit-deterministically, every counted survivor must
be bitwise on every distribution, and the sample statistic must be the
conventional median of the raw ABBA observations. A reference mismatch is
retained as a diagnostic row and refuses the serving table for that shape.

``--synthetic-only`` is an explicit opt-in screen posture: it never stats or
opens the recorded-capture paths, reports ``real_source: not_measured`` and
stamps the run as a screen. Real-input qualification remains a separate step
against the existing pinned intake.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

from gemm_sweep_decisions import (abba_summary, admit_pick, non_bitwise_fraction_summary,
                                  reference_qualified, resolve_real_input)

#: name, N, K, calls per chunk (A8SESHMN rank 0), served ms per chunk (r0), served kernel, capture
SHAPES = [
    ("kda_in_proj", 12576, 4096, 34, 90.08, "nvjet_sm121_tst_mma_128x208x64_2_32x104x64_tmaAB_bz_TNNN",
     "kda_hidden"),
    ("kda_o_proj", 4096, 4096, 34, 27.66, "nvjet_sm121_tst_mma_192x144x64_2_48x72x64_tmaAB_bz_TNNN", None),
    ("mla_o_proj", 4096, 8192, 11, 20.18, "nvjet_sm121_tst_mma_192x144x64_2_48x72x64_tmaAB_bz_TNNN", None),
    ("qa_kva_and_shared_gate_up", 2048, 4096, 44, 18.58,
     "nvjet_sm121_tst_mma_128x176x64_2_32x88x64_tmaAB_bz_TNNN", "post_attention_hidden"),
    ("shared_down", 4096, 1024, 40, 8.80, "cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128",
     "shared_down_input"),
    ("q_b", 8192, 1536, 11, 6.13, "cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256", None),
]
#: Recorded GLM-5.3 BF16 activations (pread probe, 8192 rows each, fp32 of bf16 values).
CAPTURE = "/mnt/shared/dq-runs/glm53-bf16-pread-capture-1469b9b-20260901/act"
CAPTURES = {
    # The fused KDA in_proj_qkvbfg_a reads the attention's hidden_states (vLLM
    # glm5next/common/kda.py:458), the same tensor its f_a_proj slice reads.
    "kda_hidden": ("model__language_model__layers__0__self_attn__forget_gate__f_a_proj.pt", None),
    # A normed K=4096 hidden from another site: the MLA q_a/kv_a input itself is
    # not captured, so this is the post-attention norm the shared gate reads.
    "post_attention_hidden": ("model__language_model__layers__10__mlp__shared_experts__gate_proj.pt", None),
    # Rank 0's row-parallel slice of the shared down input.
    "shared_down_input": ("model__language_model__layers__10__mlp__shared_experts__down_proj.pt", 1024),
}
PINNED_GEMM_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pinned_gemm.cpp")
M0 = 2048
WS_BYTES = 64 << 20
CONFIG_FIELDS = ["algo_id", "tile_id", "splitk_num", "reduction_scheme", "cta_swizzling", "custom_option",
                 "stages_id", "inner_shape_id", "cluster_shape_id"]
DISTRIBUTIONS = ("normal", "normal2", "adversarial", "real")


def load_extension():
    """``pinned_gemm.cpp`` (beside this file), JIT-built against the image's cuBLASLt."""
    from torch.utils.cpp_extension import load
    return load(name="tessera_pinned_gemm_sweep", sources=[PINNED_GEMM_SOURCE],
                extra_ldflags=["-lcublasLt"], with_cuda=True, verbose=False)


def cublaslt_version():
    """``cublasLtGetVersion()`` of the library this process maps."""
    import ctypes
    try:
        lib = ctypes.CDLL("libcublasLt.so.13")
        lib.cublasLtGetVersion.restype = ctypes.c_size_t
        return int(lib.cublasLtGetVersion())
    except Exception as exc:  # noqa: BLE001
        return repr(exc)


def kernel_names(call):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    names = []
    for ev in prof.events():
        if ev.device_type is not None and str(ev.device_type).endswith("CUDA") and ev.name not in names:
            names.append(ev.name[:140])
    return names


def time_calls(call, reps=10, warm=3):
    for _ in range(warm):
        call()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        call()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / reps


def power_of(call, seconds, sampler):
    if sampler is None:
        return None
    try:
        return sampler.sample_during(call, seconds)
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def inputs(capture, rows, k, dev, seed, allow_capture=True):
    """Per distribution, an [rows, k] bf16 tensor (``real`` None without a capture)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    out = {"normal": torch.randn(rows, k, generator=g, device=dev).to(torch.bfloat16),
           "normal2": (torch.randn(rows, k, generator=g, device=dev) * 3).to(torch.bfloat16)}
    e = torch.randint(-12, 9, (rows, k), generator=g, device=dev).float()
    sign = torch.randint(0, 2, (rows, k), generator=g, device=dev).float() * 2 - 1
    u = torch.rand(rows, k, generator=g, device=dev)
    out["adversarial"] = (sign * torch.exp2(e) * (1 + u)).to(torch.bfloat16)
    out["real"], out["real_source"] = None, None
    # The path is not even built when the run may not open it: a synthetic-only
    # screen never names, stats or opens a mutable capture.
    capture_desc = None
    if capture is not None and allow_capture:
        name, cols = CAPTURES[capture]
        capture_desc = (name, cols, os.path.join(CAPTURE, name))
    res = resolve_real_input(allow_capture, capture_desc, rows, k, os.path.exists,
                             lambda p: torch.load(p, map_location="cpu",
                                                  weights_only=False)["inputs"])
    out["real"], out["real_source"] = res["real"], res["real_source"]
    if out["real"] is not None:
        out["real"] = out["real"].contiguous().to(torch.bfloat16).to(dev)
    return out


def bitwise(ext, t, x, w, ref, out, ws):
    out.fill_(0)
    ext.run(t, x, w, out, ws)
    first = out.clone()
    ext.run(t, x, w, out, ws)
    torch.cuda.synchronize()
    a, b, r = first.view(torch.int16), out.view(torch.int16), ref.view(torch.int16)
    return bool(torch.equal(a, r) and torch.equal(b, r)), float((a != r).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-exhaustive", type=int, default=1500)
    ap.add_argument("--shapes", default="all")
    ap.add_argument("--m", default="2048,2049,512", help="the M values checked; the first is the sweep's")
    ap.add_argument("--survivors", type=int, default=8)
    ap.add_argument("--row-tiles", type=lambda v: [int(x) for x in v.split(",") if x], default=[128, 256, 512, 1024],
                    help="tile row counts for the row-tile check (empty to skip)")
    ap.add_argument("--synthetic-only", action="store_true",
                    help="never stat or open the recorded-capture paths; the real distribution is "
                         "reported as not_measured and this run is a screen, not qualification")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    allow_capture = not args.synthetic_only
    ms_list = [int(v) for v in args.m.split(",")]
    m0 = ms_list[0]
    t0 = time.time()
    ext = load_extension()
    sampler = None
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from bench_t8r import PowerSampler
        sampler = PowerSampler()
    except Exception as exc:  # noqa: BLE001
        print("power sampler unavailable:", repr(exc), flush=True)
    dev = torch.device("cuda")
    meta = dict(host=os.environ.get("HOST_NAME"), image=os.environ.get("ORACLE_IMAGE"),
                pb_action=os.environ.get("PB_ACTION_KEY"), tessera_head=os.environ.get("TESSERA_HEAD"),
                torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                capability=list(torch.cuda.get_device_capability()), cublaslt_version=cublaslt_version(),
                build_s=round(time.time() - t0, 1), m=ms_list,
                pinned_gemm_cpp_sha256=hashlib.sha256(open(PINNED_GEMM_SOURCE, "rb").read()).hexdigest(), workspace_bytes=WS_BYTES,
                power_source=getattr(sampler, "source", None),
                synthetic_only=allow_capture is False,
                real_input="not_measured" if not allow_capture else "recorded_capture_or_missing",
                capture_root=None if not allow_capture else CAPTURE,
                started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    ws = torch.empty(WS_BYTES, dtype=torch.uint8, device=dev)
    results, entries = [], []
    wanted = None if args.shapes == "all" else set(args.shapes.split(","))
    path = os.path.join(args.out, "gemm_algo_bitwise.json")
    table_path = os.path.join(args.out, "pinned_gemm_table.json")

    def rounded_saving(pick):
        return round(pick["saving_ms"], 4) if pick and pick["saving_ms"] is not None else None

    def dump(done):
        meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta["complete"] = done
        tmp = path + ".part"
        with open(tmp, "w") as fh:
            json.dump(dict(meta=meta, shapes=results), fh, indent=1)
        os.replace(tmp, path)
        table = dict(schema="tessera.glm53_pinned_gemm.v1", cublaslt_version=meta["cublaslt_version"],
                     torch=meta["torch"], cuda=meta["cuda"], device_name=meta["device"],
                     capability=meta["capability"], image=meta["image"], source_action=meta["pb_action"],
                     pinned_gemm_cpp_sha256=meta["pinned_gemm_cpp_sha256"],
                     complete=done, entries=entries)
        with open(table_path + ".part", "w") as fh:
            json.dump(table, fh, indent=1)
        os.replace(table_path + ".part", table_path)

    for name, n, k, calls, served_ms, served_kernel, capture in SHAPES:
        if wanted and name not in wanted:
            continue
        ts = time.time()
        torch.manual_seed(n * 7 + k)
        w = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
        xs = inputs(capture, max(ms_list), k, dev, seed=n + 3 * k, allow_capture=allow_capture)
        refs = {}
        for m in ms_list:
            for dist in DISTRIBUTIONS:
                if xs[dist] is not None:
                    refs[(m, dist)] = torch.nn.functional.linear(xs[dist][:m], w)
        x0, ref0 = xs["normal"][:m0], refs[(m0, "normal")]
        ref_call = lambda: torch.nn.functional.linear(x0, w)  # noqa: E731
        rec = dict(shape=name, n=n, k=k, calls_per_chunk=calls, served_ms_per_chunk_r0=served_ms,
                   served_kernel=served_kernel, gflop_m0=2 * m0 * n * k / 1e9,
                   real_source=xs["real_source"],
                   reference=dict(kernels=kernel_names(ref_call), ms=time_calls(ref_call),
                                  deterministic=bool(torch.equal(
                                      ref0.view(torch.int16),
                                      torch.nn.functional.linear(x0, w).view(torch.int16)))))
        rec["reference"]["matches_served_kernel"] = any(served_kernel in kn for kn in rec["reference"]["kernels"])
        # Reference identity is a qualification precondition, not a note.  A
        # candidate may only be counted against a reference that is the kernel the
        # served shape actually dispatched AND that repeats deterministically; a
        # mismatch stays diagnostic and the table stays closed for this shape.
        rec["reference_qualified"] = reference_qualified(rec["reference"]["matches_served_kernel"],
                                                         rec["reference"]["deterministic"])
        if not rec["reference_qualified"]:
            rec["reference_refusal"] = admit_pick(rec["reference"]["matches_served_kernel"],
                                                  rec["reference"]["deterministic"], None)[1]
        cands, seen = [], set()
        for source, algos in (("heuristic", ext.heuristics(m0, n, k, WS_BYTES, 64)),
                              ("exhaustive", ext.exhaustive(m0, n, k, WS_BYTES, args.max_exhaustive))):
            for rank, t in enumerate(algos):
                key = bytes(t[:8].numpy().tobytes())
                if key in seen:
                    continue
                seen.add(key)
                cands.append((source, rank, t))
        rows = []
        out = torch.empty(m0, n, device=dev, dtype=torch.bfloat16)
        for source, rank, t in cands:
            row = dict(source=source, rank=rank, config=dict(zip(CONFIG_FIELDS, ext.describe(t))),
                       blob=[int(v) for v in t[:8].tolist()], workspace=int(t[8]), waves=int(t[9]) / 1000.0)
            try:
                row["bitwise"], frac = bitwise(ext, t, x0, w, ref0, out, ws)
                if not row["bitwise"]:
                    row["frac_elems_differ"] = frac
                once = time_calls(lambda: ext.run(t, x0, w, out, ws), reps=1, warm=0)
                if once > 3 * rec["reference"]["ms"]:
                    row["ms"], row["timed_once"] = once, True  # far slower: not worth 13 calls
                else:
                    row["ms"] = time_calls(lambda: ext.run(t, x0, w, out, ws))
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            row["_t"] = t
            rows.append(row)
        ok = [r for r in rows if "ms" in r]
        bitwise_rows = sorted((r for r in ok if r["bitwise"]), key=lambda r: r["ms"])
        anyfast = sorted(ok, key=lambda r: r["ms"])
        first = [r for r in ok if r["source"] == "heuristic" and r["rank"] == 0]
        for r in first + bitwise_rows[:5] + anyfast[:3]:
            if "kernels" not in r:
                r["kernels"] = kernel_names(lambda t=r["_t"]: ext.run(t, x0, w, out, ws))
        survivors = [r for r in bitwise_rows if r["ms"] < rec["reference"]["ms"]][:args.survivors]
        # A partial record now: a row that times out in the per-M stage still reports the sweep.
        rec["survivors_at_m0"] = len(survivors)
        rec["bitwise_count"] = len(bitwise_rows)
        rec["top_bitwise_partial"] = [{k: v for k, v in r.items() if k != "_t"} for r in bitwise_rows[:10]]
        results.append(rec)
        dump(False)
        results.pop()
        # Every survivor at every M on every distribution, against F.linear there.
        per_m = {}
        for m in ms_list:
            ref_m_call = lambda m=m: torch.nn.functional.linear(xs["normal"][:m], w)  # noqa: E731
            mrec = dict(reference_ms=time_calls(ref_m_call), reference_kernels=kernel_names(ref_m_call),
                        candidates=[])
            out_m = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
            for r in survivors:
                t = r["_t"]
                c = dict(blob=r["blob"], config=r["config"], workspace_needed=int(ext.check(t, m, n, k)))
                if c["workspace_needed"] < 0 or c["workspace_needed"] > WS_BYTES:
                    c["valid"] = False
                    mrec["candidates"].append(c)
                    continue
                c["valid"], c["bitwise"] = True, {}
                try:
                    for dist in DISTRIBUTIONS:
                        if xs[dist] is None:
                            c["bitwise"][dist] = None
                            continue
                        same, frac = bitwise(ext, t, xs[dist][:m], w, refs[(m, dist)], out_m, ws)
                        c["bitwise"][dist] = same
                        if not same:
                            c.setdefault("frac_elems_differ", {})[dist] = frac
                    c["counts"] = all(v is not False for v in c["bitwise"].values()) and \
                        sum(v is True for v in c["bitwise"].values()) >= 3
                    xm = xs["normal"][:m]
                    c["ms"] = time_calls(lambda: ext.run(t, xm, w, out_m, ws))
                except Exception as exc:  # noqa: BLE001
                    c["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    c["counts"] = False
                mrec["candidates"].append(c)
            counting = sorted((c for c in mrec["candidates"] if c.get("counts") and c["ms"] < mrec["reference_ms"]),
                              key=lambda c: c["ms"])
            if counting:
                best = counting[0]
                bt = next(r["_t"] for r in survivors if r["blob"] == best["blob"])
                xm = xs["normal"][:m]
                ab = {"reference": [], "pick": []}
                for _ in range(3):
                    for arm in ("reference", "pick", "pick", "reference"):
                        call = ref_m_call if arm == "reference" else (lambda: ext.run(bt, xm, w, out_m, ws))
                        ab[arm].append(time_calls(call))
                best["interleaved_ms"] = ab
                best["kernels"] = kernel_names(lambda: ext.run(bt, xm, w, out_m, ws))
                best["power"] = {"reference": power_of(ref_m_call, 2.0, sampler),
                                 "pick": power_of(lambda: ext.run(bt, xm, w, out_m, ws), 2.0, sampler)}
                ref_med, pick_med, saving_ms = abba_summary(ab["reference"], ab["pick"])
                mrec["pick"] = dict(blob=best["blob"], config=best["config"], ms=pick_med,
                                    reference_ms=ref_med, saving_ms=saving_ms,
                                    observations=dict(reference=list(ab["reference"]),
                                                      pick=list(ab["pick"])))
                admitted, refusal = admit_pick(rec["reference"]["matches_served_kernel"],
                                               rec["reference"]["deterministic"], saving_ms)
                if admitted:
                    entries.append(dict(
                        m=m, n=n, k=k, dtype="bf16", layout="linear_x_wT", shape=name,
                        algo_blob=best["blob"], config=best["config"], workspace=best["workspace_needed"],
                        evidence=dict(bitwise=best["bitwise"], real_source=xs["real_source"],
                                      reference_kernels=mrec["reference_kernels"], kernels=best["kernels"],
                                      reference_ms=ref_med, ms=pick_med,
                                      reference_matches_served_kernel=rec["reference"]["matches_served_kernel"],
                                      reference_deterministic=rec["reference"]["deterministic"],
                                      synthetic_only=not allow_capture,
                                      label="[S] timing screen")))
                elif refusal is not None:
                    # Fail closed: a faster pick under an unqualified reference is
                    # a diagnostic row, never a serving table entry.
                    mrec["pick"]["refused"] = refusal
            per_m[str(m)] = mrec
            del out_m
        # Row tiles: is out[s:s+tm] of a tm-row call the same bits as rows s:s+tm of the M0 call?
        # Stock first (F.linear at tm against F.linear at M0); then the fastest survivors, run at
        # tm on the row slice, against the same stock M0 rows. A producer tiled by rows (so its
        # all-reduce can start on the first tile) is bitwise only where this holds.
        row_tiles = {}
        for tm in args.row_tiles:
            if tm >= m0 or m0 % tm:
                continue
            trec = dict(stock={}, stock_kernels=kernel_names(
                lambda tm=tm: torch.nn.functional.linear(xs["normal"][:tm], w)), survivors=[])
            for dist in DISTRIBUTIONS:
                if xs[dist] is None:
                    trec["stock"][dist] = None
                    continue
                full = refs[(m0, dist)].view(torch.int16)
                diff = 0
                for s0 in range(0, m0, tm):
                    part = torch.nn.functional.linear(xs[dist][s0:s0 + tm], w)
                    diff += int((part.view(torch.int16) != full[s0:s0 + tm]).sum())
                trec["stock"][dist] = diff == 0
                if diff:
                    trec.setdefault("stock_frac_elems_differ", {})[dist] = diff / full.numel()
            out_t = torch.empty(tm, n, device=dev, dtype=torch.bfloat16)
            for r in survivors[:3]:
                t = r["_t"]
                c = dict(blob=r["blob"], config=r["config"], workspace_needed=int(ext.check(t, tm, n, k)), rows={})
                if 0 <= c["workspace_needed"] <= WS_BYTES:
                    for dist in DISTRIBUTIONS:
                        if xs[dist] is None:
                            c["rows"][dist] = None
                            continue
                        full = refs[(m0, dist)].view(torch.int16)
                        diff = 0
                        for s0 in range(0, m0, tm):
                            ext.run(t, xs[dist][s0:s0 + tm], w, out_t, ws)
                            diff += int((out_t.view(torch.int16) != full[s0:s0 + tm]).sum())
                        c["rows"][dist] = diff == 0
                trec["survivors"].append(c)
            del out_t
            row_tiles[str(tm)] = trec
        rec["row_tiles"] = row_tiles
        for r in rows:
            r.pop("_t", None)
        rec["candidates"] = len(rows)
        rec["ran"] = len(ok)
        rec["bitwise_count"] = len(bitwise_rows)
        rec["errors"] = sum(1 for r in rows if "error" in r)
        diffs = [r["frac_elems_differ"] for r in ok if not r["bitwise"]]
        rec["non_bitwise_frac_elems_differ"] = non_bitwise_fraction_summary(diffs)
        rec["heuristic_rank0"] = first[0] if first else None
        rec["survivors_at_m0"] = len(survivors)
        rec["per_m"] = per_m
        rec["top_bitwise"] = bitwise_rows[:10]
        rec["top_any"] = anyfast[:10]
        rec["all_rows_summary"] = [dict(source=r["source"], rank=r["rank"], bitwise=r.get("bitwise"),
                                        frac_elems_differ=r.get("frac_elems_differ"), ms=r.get("ms"),
                                        config=r["config"], error=r.get("error"))
                                   for r in rows]
        rec["seconds"] = round(time.time() - ts, 1)
        results.append(rec)
        dump(False)
        print(json.dumps(dict(shape=name, ref_ms=round(rec["reference"]["ms"], 4),
                              ref_kernels=rec["reference"]["kernels"][:2],
                              ref_matches_served=rec["reference"]["matches_served_kernel"],
                              rank0_bitwise=first[0].get("bitwise") if first else None,
                              candidates=len(rows), bitwise=len(bitwise_rows), survivors=len(survivors),
                              non_bitwise_frac_min=rec["non_bitwise_frac_elems_differ"]["min"],
                              real=xs["real_source"],
                              picks={m: rounded_saving(per_m[m].get("pick"))
                                     for m in per_m},
                              stock_row_tiles={tm: v["stock"] for tm, v in row_tiles.items()},
                              seconds=rec["seconds"])), flush=True)
        del w, xs, refs, out
        torch.cuda.empty_cache()
    dump(True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
