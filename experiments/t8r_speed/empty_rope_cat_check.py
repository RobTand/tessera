"""GPU check: skipping the zero-width RoPE query cat leaves sparse MLA bitwise equal.

Calls the stock SM120 sparse-MLA kernel exactly as vLLM's
``FlashInferMLASparseSM120Impl._run_mqa_kernel`` does (image 5be13705), at the
GLM-5.3 TP2 shapes: 32 heads per rank, 512-wide latent query, fp8_ds_mla
656-byte cache rows in 64-token pages, top-k width 2176, ``arbitrary_fp32``
scales. Three query forms per case:

- ``cat``: ``torch.cat((q_nope, q_pe_empty), dim=-1)``, the stock form, run
  twice as a determinism control;
- ``direct``: ``tessera.serving.glm53_empty_rope.query_without_empty_rope``
  applied to the same tuple, which must return ``q_nope`` itself.

Order per case: cat, direct, cat, direct (forward then reverse). Every output is
compared bitwise (int16 view) against the first cat output.

Cases: prefill chunk 1 (positions 0-2047, short rows padded with -1), prefill
chunk 4 (positions 6144-8191, 2048 valid slots each), and decode batches of 1,
2 and 4 tokens at position 8191 (the captured decode graph sizes).

A torch.profiler screen then times each form at the chunk-4 shape. It runs on a
non-measurement row, so it is a screen.

Writes ``<out>/empty_rope_cat_check.json``; exits 1 unless every case is bitwise
equal and the helper returned ``q_nope`` itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time

import torch

N_HEADS, LATENT, ROPE_PAD, ROW_BYTES, PAGE = 32, 512, 64, 656, 64
TOPK, WIDTH, CONTEXT = 2048, 2176, 8192
SCALE = 256 ** -0.5


def build_cache(dev, log):
    """fp8_ds_mla cache for CONTEXT tokens, written by vLLM's own cache writer."""
    pages = CONTEXT // PAGE
    cache = torch.zeros((pages, PAGE, ROW_BYTES), dtype=torch.uint8, device=dev)
    gen = torch.Generator(device=dev).manual_seed(1)
    latent = torch.randn((CONTEXT, LATENT), generator=gen, device=dev, dtype=torch.float32)
    latent = (latent * 0.5).to(torch.bfloat16)
    rope = torch.zeros((CONTEXT, ROPE_PAD), dtype=torch.bfloat16, device=dev)
    slots = torch.arange(CONTEXT, device=dev, dtype=torch.int64)
    try:
        from vllm import _custom_ops as ops
        ops.concat_and_cache_mla(latent, rope, cache, slots, "fp8_ds_mla",
                                 torch.ones(1, device=dev, dtype=torch.float32))
        torch.cuda.synchronize()
        log["cache_writer"] = "vllm._custom_ops.concat_and_cache_mla(fp8_ds_mla)"
    except Exception as exc:  # a legal synthetic cache still tests the query identity
        log["cache_writer"] = f"synthetic (writer raised {type(exc).__name__}: {exc})"
        body = torch.randint(0, 0x7F, (pages, PAGE, LATENT), generator=gen, device=dev,
                             dtype=torch.int32).to(torch.uint8)
        sign = torch.randint(0, 2, (pages, PAGE, LATENT), generator=gen, device=dev,
                             dtype=torch.int32).to(torch.uint8) << 7
        cache[..., :LATENT] = body | sign
        scales = torch.rand((pages, PAGE, 4), generator=gen, device=dev) * 0.01 + 1e-3
        cache[..., LATENT:LATENT + 16] = scales.view(torch.uint8).view(pages, PAGE, 16)
    log["cache_sha256"] = hashlib.sha256(cache.cpu().numpy().tobytes()).hexdigest()
    return cache


def build_indices(positions, dev, seed):
    """Slot ids per query token: up to TOPK distinct causal slots, -1 padded to WIDTH."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    n = positions.numel()
    keys = torch.rand((n, CONTEXT), generator=gen, device=dev)
    cols = torch.arange(CONTEXT, device=dev).view(1, -1)
    keys = keys.masked_fill(cols > positions.view(-1, 1), -1.0)
    values, idx = keys.topk(TOPK, dim=1)
    idx = idx.to(torch.int32).masked_fill(values < 0, -1)
    out = torch.full((n, WIDTH), -1, dtype=torch.int32, device=dev)
    out[:, :TOPK] = idx
    return out.contiguous()


def run_kernel(q, cache, indices, workspace, mla):
    n = q.shape[0]
    output = q.new_empty((n, N_HEADS, LATENT), dtype=q.dtype)
    out = mla(
        query=q.unsqueeze(1),
        kv_cache=cache.view(torch.uint8).unsqueeze(1),
        workspace_buffer=workspace,
        qk_nope_head_dim=256,
        kv_lora_rank=LATENT,
        qk_rope_head_dim=0,
        block_tables=indices.unsqueeze(1),
        seq_lens=None,
        max_seq_len=WIDTH,
        out=output.unsqueeze(1),
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        sparse_mla_top_k=WIDTH,
        kv_scale_format="arbitrary_fp32",
    )
    return out.squeeze(1)


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()


def check_case(name, positions, cache, workspace, mla, helper, dev, seed):
    n = positions.numel()
    gen = torch.Generator(device=dev).manual_seed(seed)
    # vLLM allocates mqa_ql_nope with new_empty((B, N, L)) and fills it via
    # bmm(out=transpose(0, 1)): a contiguous [B, N, L] tensor.
    nope = torch.empty((n, N_HEADS, LATENT), dtype=torch.bfloat16, device=dev)
    nope.transpose(0, 1).copy_(
        torch.randn((N_HEADS, n, LATENT), generator=gen, device=dev).to(torch.bfloat16))
    pe = nope.new_empty((n, N_HEADS, 0))
    q = (nope, pe)
    indices = build_indices(positions, dev, seed + 1)
    direct_q = helper(q)
    same_object = direct_q is nope
    outs = []
    for form in ("cat", "direct", "cat", "direct"):
        qq = torch.cat(q, dim=-1) if form == "cat" else helper(q)
        outs.append((form, run_kernel(qq, cache, indices, workspace, mla)))
    torch.cuda.synchronize()
    ref = outs[0][1]
    rows = []
    for i, (form, out) in enumerate(outs):
        equal = torch.equal(out.view(torch.int16), ref.view(torch.int16))
        diff = (out.float() - ref.float()).abs()
        diff = diff[torch.isfinite(diff)]
        rows.append(dict(order=i, form=form, bitwise_equal_to_first_cat=bool(equal),
                         sha256=digest(out),
                         max_abs_diff=float(diff.max()) if diff.numel() else None))
    finite = torch.isfinite(ref.float()).float().mean().item()
    valid = (indices >= 0).sum(dim=1)
    return dict(case=name, tokens=n, positions=[int(positions.min()), int(positions.max())],
                valid_slots_min=int(valid.min()), valid_slots_max=int(valid.max()),
                helper_returned_q_nope=bool(same_object),
                q_nope_data_ptr_mod_512=int(nope.data_ptr() % 512),
                cat_equals_q_nope_bitwise=bool(torch.equal(
                    torch.cat(q, dim=-1).view(torch.int16), nope.view(torch.int16))),
                finite_fraction=finite, runs=rows,
                bitwise=all(r["bitwise_equal_to_first_cat"] for r in rows) and same_object)


def screen(cache, workspace, mla, helper, dev, reps, trace_root=None):
    from torch.profiler import ProfilerActivity, profile
    positions = torch.arange(6144, 8192, device=dev)
    nope = torch.randn((2048, N_HEADS, LATENT), device=dev).to(torch.bfloat16)
    q = (nope, nope.new_empty((2048, N_HEADS, 0)))
    indices = build_indices(positions, dev, 77)
    result = {}
    for _ in range(3):  # warm both forms
        run_kernel(torch.cat(q, dim=-1), cache, indices, workspace, mla)
        run_kernel(helper(q), cache, indices, workspace, mla)
    torch.cuda.synchronize()
    for order, form in enumerate(("cat", "direct", "cat", "direct")):
        window_start = time.time()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(reps):
                qq = torch.cat(q, dim=-1) if form == "cat" else helper(q)
                run_kernel(qq, cache, indices, workspace, mla)
            end.record()
            torch.cuda.synchronize()
        window_end = time.time()
        trace = os.path.join(trace_root, f"empty-rope-{order}-{form}.trace.json") \
            if trace_root is not None else None
        if trace is not None:
            prof.export_chrome_trace(trace)
        kernels = {}
        for ev in prof.key_averages():
            t = getattr(ev, "self_device_time_total", None) or getattr(ev, "self_cuda_time_total", 0)
            if t and ev.count:
                kernels[ev.key[:120]] = dict(count=ev.count, us_per_call=t / ev.count)
        result.setdefault(form, []).append(dict(
            event_ms_per_iter=start.elapsed_time(end) / reps, kernels=kernels,
            trace=trace, window_unix=[window_start, window_end]))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import flashinfer
    import vllm
    from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_with_kv_cache_mla as mla
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import _get_workspace_buffer
    from tessera.serving.glm53_empty_rope import query_without_empty_rope as helper

    dev = torch.device("cuda")
    log = dict(host=socket.gethostname(), started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               torch=torch.__version__, vllm=vllm.__version__, flashinfer=flashinfer.__version__,
               device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
               image=os.environ.get("ORACLE_IMAGE"), tessera_head=os.environ.get("TESSERA_HEAD"),
               tessera_state=os.environ.get("TESSERA_STATE"), pb_action=os.environ.get("PB_ACTION_KEY"))
    cache = build_cache(dev, log)
    workspace = _get_workspace_buffer(dev)
    cases = [
        ("prefill_chunk1", torch.arange(0, 2048, device=dev)),
        ("prefill_chunk4", torch.arange(6144, 8192, device=dev)),
        ("decode_b1", torch.full((1,), 8191, device=dev)),
        ("decode_b2", torch.full((2,), 8191, device=dev)),
        ("decode_b4", torch.full((4,), 8191, device=dev)),
    ]
    results = []
    for i, (name, positions) in enumerate(cases):
        try:
            results.append(check_case(name, positions, cache, workspace, mla, helper, dev, 10 * i + 3))
        except Exception as exc:
            results.append(dict(case=name, bitwise=False, error=f"{type(exc).__name__}: {exc}"))
        print(json.dumps({k: v for k, v in results[-1].items() if k != "runs"}), flush=True)
    try:
        log["screen"] = screen(cache, workspace, mla, helper, dev, args.reps, args.out)
        log["screen_label"] = "[S] torch.profiler, non-measurement row"
    except Exception as exc:
        log["screen_error"] = f"{type(exc).__name__}: {exc}"
    log["cases"] = results
    log["all_bitwise"] = all(r.get("bitwise") for r in results)
    log["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(args.out, "empty_rope_cat_check.json"), "w") as fh:
        json.dump(log, fh, indent=1)
    print("ALL_BITWISE" if log["all_bitwise"] else "NOT_BITWISE", flush=True)
    if "screen" in log:
        for form, rows in log["screen"].items():
            print(form, [round(r["event_ms_per_iter"], 4) for r in rows], flush=True)
    return 0 if log["all_bitwise"] else 1


if __name__ == "__main__":
    sys.exit(main())
