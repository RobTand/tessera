"""Dense E4M3 modules through the served path, against their floors (tessera#750 WP2).

Each module is encoded the way the exporter writes it (``export.encode_linear_planes``
per role, ``fused.pack_fused``), parsed from its blob and prepared by
``native_window.prepare_dense_native_module`` -- the load path a serve runs -- so
the lane the module takes is the lane a serve would take, and it is recorded.
The timed call is ``module.apply`` on the route's own per-token FP8 activation
(``native_fp8_quant``), and separately the quantizer plus ``apply``.

Modules (TP2 per-rank shapes of GLM-5.3-Flash):

- ``kda_in``: the KDA input module ``in_proj_qkvbfg_a``: q, k, v at 4096 rows,
  b 32, f_a 64, g_a 64 (12448 rows) over K = 4096.
- ``o_proj``: KDA ``o_proj``, one 4096-row role over K = 4096 (the N % 128
  control: a shape the N-tail change leaves on the same launch).
- ``lm_head``: the vocab-parallel LM head, one 77440-row role (154880 / 2)
  over K = 4096, read from the checkpoint's ``lm_head.weight`` (untied).  Its M
  is the number of positions sampled in a step (vLLM prunes the hidden states
  to them before the head), so a prefill of 8192 tokens is not one of its
  shapes; time it at decode batch sizes.
- One role each, for a K-split sweep of one geometry per launch: KDA
  ``q_proj`` (4096 x 4096), ``b_proj`` (32 x 4096), ``f_a_proj`` (64 x 4096),
  ``f_b_proj`` (4096 x 128); MLA ``q_a_proj`` (1536 x 4096),
  ``kv_a_proj_with_mqa`` (512 x 4096), ``q_b_proj`` (8192 x 1536), read from
  ``--mla-layer``.
- ``idx_wq_b``: the DSA indexer's ``wq_b`` (4096 x 1536, replicated), from the
  same ``--mla-layer``.
- The MLPs a prefill chunk runs on the dense lane (tessera#750 package 3):
  ``mlp_gate_up``, the dense MLP's gate and up at 6144 rows each over K = 4096
  (layers 0-2; read from layer 2); ``mlp_down``, its down over K = 6144 (read
  from layer 0); ``shared_gate_up``, the shared expert's gate and up at 1024
  rows each over K = 4096; ``shared_down``, its down over K = 1024 (both read
  from layer 7, the first MoE layer a short checkpoint holds).  Each reads its
  layer from :data:`MODULE_SOURCES`, overridable with ``--source-layers``.
- ``--served PREFIX[,...]`` with ``--artifact DIR``: a module's SERVED bytes,
  the exported checkpoint's own ``PREFIX.wire_bytes`` under the scheme its
  ``config.json`` declares, cut as TP rank 0 of ``--served-tp`` loads it
  (``plan_shard``: an input cut for ``down_proj``/``o_proj``, an output cut
  otherwise).  Group ``served:PREFIX``; no source weight, so its numerics are
  fused against Triton only; timed at ``--served-ms`` on ``--served-lanes``.

Weights: ``--model DIR`` encodes the real GLM-5.3 bytes of ``--layer L``'s
tensors (the TP2 rank-0 shard: the leading rows of a column-parallel role, the
leading columns of the row-parallel ``o_proj``); without it, seeded Gaussian
weights.  Before timing, each module group records a numerics check: the fused
lane against the Triton lane on the same prepared wire (both decode the same
bytes, so the difference is accumulation order), each lane against the source
``F.linear`` in fp32, and fused determinism.

References at each module's whole shape: bf16 ``F.linear`` (the BF16 source
passthrough a serve runs today) and ``torch._scaled_mm`` FP8 row-wise.

Floors: bytes (the wire at q256 bits per 256 weights plus the fp32 row scales,
the activation and the bf16 output) over the measured read rate, and FLOPs
over the E4M3 ``mma.sync`` peak; the floor is the larger.  ``copy_frac`` is
the byte floor over the time (the WP2 M = 1 criterion: 0.90 or more);
``roof_frac`` is the floor over the time (the M >= 512 criterion: within 20%).

Each cell is timed as a CUDA-graph replay in a forward pass over the cell list
and again in a reverse pass; the cell's time is the mean of the two medians.
A replayed module stays in the 24 MB L2 when it fits, so the default (``--l2
warm``) times decode from L2.  ``--l2 cold`` (or ``warm,cold``) also times
``apply_cold``: a read of four L2s before each replay evicts the module, as a
served forward finds it after the rest of the model has passed through.
The forward pass also records torch.profiler device time per kernel and, at
``--power-ms``, NVML board power over a back-to-back loop, with unix times for
the Netdata series.

Usage: bench_dense_module.py --out DIR [--modules kda_in,o_proj] [--q256 1024,1088]
       [--ms 1,2,4,8,16,64,512,2048,8192] [--refs] [--model DIR --layer L]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import zlib

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_t8r import ENVELOPE_W, PowerSampler, kernel_profile, time_events  # noqa: E402

READ_GBPS = 232.2            # GB10 measured read rate (docs/measurements/2026-09-30-fp8-prefill-roofline.md)
MMA_E4M3_TFLOPS = 246.5      # sm_121 mma.sync E4M3 peak (same note)
MODULES = {
    "kda_in": ([("q_proj", 4096), ("k_proj", 4096), ("v_proj", 4096), ("b_proj", 32),
                ("f_a_proj", 64), ("g_a_proj", 64)], 4096),
    "o_proj": ([("o_proj", 4096)], 4096),
    # One role each, so a K-split sweep times one geometry per launch.
    "q_proj": ([("q_proj", 4096)], 4096),
    "b_proj": ([("b_proj", 32)], 4096),
    "f_a_proj": ([("f_a_proj", 64)], 4096),
    "f_b_proj": ([("f_b_proj", 4096)], 128),
    # MLA (a full-attention layer, ``--mla-layer``).
    "q_a_proj": ([("q_a_proj", 1536)], 4096),
    "kv_a_proj_with_mqa": ([("kv_a_proj_with_mqa", 512)], 4096),
    "q_b_proj": ([("q_b_proj", 8192)], 1536),
    # vLLM's merged MLA input (``fused_qkv_a_proj``, replicated): two roles, one
    # module, so the E4M3 libraries take it in one launch (tessera#750 WP2).
    "fused_qkv_a": ([("q_a_proj", 1536), ("kv_a_proj_with_mqa", 512)], 4096),
    # The DSA indexer's query projection: a ``ReplicatedLinear`` (every rank
    # holds all 4096 rows) that vLLM offers the quant config.  Its sibling
    # ``wk_weights_proj`` is built with ``quant_config=None`` and is not.
    "idx_wq_b": ([("indexer.wq_b", 4096)], 1536),
    # The LM head (``ParallelLMHead``): the vocab split across TP2.
    "lm_head": ([("lm_head", 77440)], 4096),
    # The MLPs (tessera#750 package 3): a dense MLP (``intermediate_size``
    # 12288) and the shared expert (``moe_intermediate_size`` 2048), TP2 per
    # rank.  Gate and up are column-parallel, down row-parallel.
    "mlp_gate_up": ([("gate_proj", 6144), ("up_proj", 6144)], 4096),
    "mlp_down": ([("down_proj", 4096)], 6144),
    "shared_gate_up": ([("gate_proj", 1024), ("up_proj", 1024)], 4096),
    "shared_down": ([("down_proj", 4096)], 1024),
}
MLA_MODULES = {"q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "fused_qkv_a", "idx_wq_b"}
SOURCE_PREFIX = "model.language_model.layers.{layer}.self_attn.{name}.weight"
#: Roles whose source tensor is not under a layer's ``self_attn``.
SOURCE_KEYS = {"lm_head": "lm_head.weight"}
#: Modules read from outside ``self_attn``: (key template, default layer).  The
#: defaults are the GLM-5.3 layers whose module the A8S re-solve keeps on the
#: dense E4M3 lane at q256 1024 (layer 2's gate_up, layer 0's down), and for
#: the shared expert layer 7, the first MoE layer an 8-layer checkpoint holds
#: (its down is served on that lane at layers 32 and 36: same geometry).
MODULE_SOURCES = {
    "mlp_gate_up": ("model.language_model.layers.{layer}.mlp.{name}.weight", 2),
    "mlp_down": ("model.language_model.layers.{layer}.mlp.{name}.weight", 0),
    "shared_gate_up": ("model.language_model.layers.{layer}.mlp.shared_experts.{name}.weight", 7),
    "shared_down": ("model.language_model.layers.{layer}.mlp.shared_experts.{name}.weight", 7),
}


def source_weight(model, layer, name, rows, cols, template=None):
    """The TP2 rank-0 shard of one real source tensor, bf16 on the GPU."""
    from safetensors import safe_open

    key = SOURCE_KEYS.get(name) or (template or SOURCE_PREFIX).format(layer=layer, name=name)
    index = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(model, index[key]), framework="pt", device="cuda") as fh:
        w = fh.get_tensor(key)
    if w.shape[0] < rows or w.shape[1] < cols:
        raise ValueError(f"{key} is {tuple(w.shape)}; the shard needs [{rows}, {cols}]")
    return w[:rows, :cols].to(torch.bfloat16).contiguous()


def graph_time(call, warmup, iters):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    samples = time_events(graph.replay, warmup, iters)
    del graph
    return samples


def cold_scratch():
    """A read-only buffer of four L2s (at least 64 MB): reading it before a
    timed call leaves the call's weights out of L2, as a served forward
    finds them after the rest of the model has passed through.  Read, not
    written, so the timed call writes back no dirty lines of it."""
    l2 = int(getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0) or (24 << 20))
    return torch.ones(max(4 * l2, 64 << 20) // 4, dtype=torch.float32, device="cuda")


def graph_time_cold(call, warmup, iters, scratch):
    """``graph_time`` with L2 cleared before each replay (``cold_scratch``);
    the events bracket the replay alone."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(warmup):
        scratch.sum()
        graph.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        scratch.sum()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); graph.replay(); b.record(); b.synchronize()
        out.append(float(a.elapsed_time(b)))
    del graph
    return out


def kernel_profile_cold(call, scratch, reps=3):
    """``kernel_profile`` with L2 cleared before each call; the clearing
    read's own kernels (profiled alone first) are left out."""
    alone = kernel_profile(lambda: scratch.sum(), reps=1)["top"]
    prof = kernel_profile(lambda: (scratch.sum(), call()), reps=reps)
    top = {k: v for k, v in prof["top"].items() if k not in alone}
    return {"kernel_us_per_call": sum(v["us_per_call"] for v in top.values()),
            "launches_per_call": prof["launches_per_call"] - len(alone), "top": top}


def encode_module(roles, cols, q256, seed, source=None):
    """The served module: encode each role on the E4M3 grid, pack, parse, prepare.
    ``source(name, rows)`` gives a role's real weight; otherwise seeded Gaussian."""
    from tessera import export, fused
    from tessera.alphabet import E4M3_GRID
    from tessera.serving.native_window import prepare_dense_native_module
    from tessera.serving.scheme import (TESSERA_FP8, parse_compact_blob_for_scheme,
                                        validate_tessera_scheme)
    from tessera.serving.sharding import plan_shard

    torch.manual_seed(seed)
    blobs = []
    weights = []
    t0 = time.time()
    for name, rows in roles:
        w = (source(name, rows).float() if source is not None
             else torch.randn(rows, cols, device="cuda") * 0.02).contiguous()
        exported, _unit, _forests = export.encode_linear_planes(w, grid=E4M3_GRID, q256=q256, name=name,
                                                                verify=False)
        blobs.append((name, rows, exported.blob))
        weights.append(w.bfloat16())
        del w
    blob = fused.pack_fused(blobs)
    blob_sha256 = hashlib.sha256(bytes(blob)).hexdigest()
    rows = sum(r for _, r in roles)
    scheme = {"family": TESSERA_FP8, "grid": "E4M3", "body": "WINDOW", "plane": "CHANNEL", "q256": q256,
              "rows": rows, "columns": cols, "wire_bytes": len(blob), "roles": [[n, r] for n, r in roles]}
    declared = validate_tessera_scheme(scheme, "bench")
    plan = plan_shard("bench", roles=[(n, r) for n, r in roles], columns=cols,
                      out_partitions=[r for _, r in roles], in_size=cols, tp_rank=0, tp_size=1,
                      input_size=cols, output_size=rows)

    def prepare(fused_lane):
        prev = os.environ.get("TESSERA_DENSE_FUSED")
        os.environ["TESSERA_DENSE_FUSED"] = "1" if fused_lane else "0"
        try:
            compact = parse_compact_blob_for_scheme(blob, scheme, "bench", device="cuda")
            return prepare_dense_native_module(compact, plan, family=declared["family"], device="cuda")
        finally:
            if prev is None:
                os.environ.pop("TESSERA_DENSE_FUSED", None)
            else:
                os.environ["TESSERA_DENSE_FUSED"] = prev
    return prepare, len(blob), time.time() - t0, torch.cat(weights), blob_sha256


def _sha256(t):
    """sha256 of a CUDA tensor's bytes (contiguous, viewed as uint8)."""
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def bitwise(lanes, cols, ms, name, q256):
    """Each lane's output hash at each M, on an input seeded by (module, q256, M)
    alone -- not by the group generator, which the cells consume in pass order
    -- so two arms (two source snapshots) hash the same activation and can be
    compared bitwise.  Each lane runs twice; ``repeat_equal`` is its
    determinism.  The input's own hashes travel with it: a mismatch with
    unequal input hashes is the quantiser, not the GEMM."""
    from tessera.serving.native_ops import native_fp8_quant

    out = {}
    for m in ms:
        g = torch.Generator(device="cuda").manual_seed(zlib.crc32(f"hash:{name}:{q256}:{m}".encode()))
        x = (torch.randn(m, cols, device="cuda", generator=g) * 0.5).bfloat16()
        xq, a = native_fp8_quant(x)
        a = a.reshape(-1).contiguous().float()
        rec = {"x_sha256": _sha256(x), "xq_sha256": _sha256(xq), "a_sha256": _sha256(a)}
        for k, mod in lanes.items():
            y1 = mod.apply(xq, a)
            y2 = mod.apply(xq, a)
            torch.cuda.synchronize()
            rec[f"{k}_sha256"] = _sha256(y1)
            rec[f"{k}_repeat_equal"] = bool(torch.equal(y1, y2))
            del y1, y2
        out[str(m)] = rec
        print(json.dumps({"bitwise_m": m, "module": name, "q256": q256,
                          **{k: v for k, v in rec.items() if not k.startswith(("x_", "a_"))}}), flush=True)
    return out


def numerics(lanes, w_src, cols, ms, seed):
    """Fused vs Triton on one prepared wire, each vs the source ``F.linear`` (fp32),
    and fused determinism, at each M."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    from tessera.serving.native_ops import native_fp8_quant

    out = {}
    for m in ms:
        x = (torch.randn(m, cols, device="cuda", generator=g) * 0.5).bfloat16()
        xq, a = native_fp8_quant(x)
        a = a.reshape(-1).contiguous().float()
        ref = torch.nn.functional.linear(x.float(), w_src.float()) if w_src is not None else None
        got = {k: v.apply(xq, a).float() for k, v in lanes.items()}
        rec = {}
        for k, y in got.items():
            if ref is not None:
                rec[f"{k}_vs_source_rel_fro"] = float((y - ref).norm() / ref.norm())
        if "fused" in got and "triton" in got:
            d = got["fused"] - got["triton"]
            rec["fused_vs_triton_rel_fro"] = float(d.norm() / got["triton"].norm())
            rec["fused_vs_triton_max_rel"] = float(d.abs().max() / got["triton"].abs().max())
        if "fused" in lanes:
            rec["fused_deterministic"] = bool(torch.equal(lanes["fused"].apply(xq, a).float(), got["fused"]))
        out[str(m)] = rec
        print(json.dumps({"numerics_m": m, **{k: (round(v, 8) if isinstance(v, float) else v)
                                               for k, v in rec.items()}}), flush=True)
    torch.cuda.synchronize()
    return out


def served_scheme(artifact, prefix):
    """The scheme ``config.json`` declares for one module of an exported checkpoint."""
    cfg = json.load(open(os.path.join(artifact, "config.json")))
    qc = cfg.get("quantization_config") or cfg.get("text_config", {}).get("quantization_config") or {}
    for group in (qc.get("config_groups") or {}).values():
        if prefix in (group.get("targets") or []):
            return dict(group["scheme"])
    raise SystemExit(f"{artifact}: no config group targets {prefix}")


def served_geometry(artifact, prefix, tp):
    """(scheme, roles, columns, cut, out_partitions, in_size) of a served module
    at TP rank 0 of ``tp``: down_proj and o_proj are row-parallel (an input
    cut), every other dense module column-parallel (an output cut)."""
    scheme = served_scheme(artifact, prefix)
    roles = [(str(n), int(r)) for n, r in scheme["roles"]]
    columns = int(scheme["columns"])
    cut = "input" if prefix.endswith(("down_proj", "o_proj")) else "output"
    out_parts = [r // tp for _, r in roles] if cut == "output" else [r for _, r in roles]
    in_size = columns // tp if cut == "input" else columns
    return scheme, roles, columns, cut, out_parts, in_size


def served_module(artifact, prefix, tp):
    """The served module: the checkpoint's own container bytes, parsed and
    prepared as TP rank 0 of ``tp`` loads them.  Returns ``(prepare,
    blob_bytes, blob_sha256, cut)``."""
    from safetensors import safe_open
    from tessera.serving.native_window import prepare_dense_native_module
    from tessera.serving.scheme import parse_compact_blob_for_scheme, validate_tessera_scheme
    from tessera.serving.sharding import plan_shard

    scheme, roles, columns, cut, out_parts, in_size = served_geometry(artifact, prefix, tp)
    declared = validate_tessera_scheme(scheme, prefix)
    plan = plan_shard(prefix, roles=roles, columns=columns, out_partitions=out_parts, in_size=in_size,
                      tp_rank=0, tp_size=tp, input_size=columns, output_size=sum(r for _, r in roles))
    key = f"{prefix}.wire_bytes"
    index = json.load(open(os.path.join(artifact, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(artifact, index[key]), framework="pt") as fh:
        blob = fh.get_tensor(key).numpy().tobytes()

    def prepare(fused_lane):
        prev = os.environ.get("TESSERA_DENSE_FUSED")
        os.environ["TESSERA_DENSE_FUSED"] = "1" if fused_lane else "0"
        try:
            compact = parse_compact_blob_for_scheme(blob, scheme, prefix, device="cuda")
            return prepare_dense_native_module(compact, plan, family=declared["family"], device="cuda")
        finally:
            if prev is None:
                os.environ.pop("TESSERA_DENSE_FUSED", None)
            else:
                os.environ["TESSERA_DENSE_FUSED"] = prev
    return prepare, len(blob), hashlib.sha256(blob).hexdigest(), cut


def floors(wire_bytes, m, rows, cols, out_bytes=2, a_bytes=1):
    bytes_ = wire_bytes + m * cols * a_bytes + m * rows * out_bytes
    byte_ms = bytes_ / (READ_GBPS * 1e9) * 1e3
    flop_ms = 2.0 * m * rows * cols / (MMA_E4M3_TFLOPS * 1e12) * 1e3
    return {"bytes": bytes_, "byte_ms": byte_ms, "flop_ms": flop_ms, "floor_ms": max(byte_ms, flop_ms)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--modules", default="kda_in,o_proj")
    ap.add_argument("--q256", default="1024,1088")
    ap.add_argument("--ms", default="1,2,4,8,16,64,512,2048,8192")
    ap.add_argument("--refs", action="store_true")
    ap.add_argument("--lanes", default="fused,triton", help="which lanes of each module to time")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--power-ms", default="1,512,8192")
    ap.add_argument("--power-s", type=float, default=0.5)
    ap.add_argument("--ncu", action="store_true",
                    help="one apply per (module, lane, M) between cudaProfilerStart/Stop; no timing")
    ap.add_argument("--model", default=None, help="source checkpoint: encode its real bytes")
    ap.add_argument("--layer", type=int, default=1, help="the KDA layer read from --model")
    ap.add_argument("--mla-layer", type=int, default=3, help="the MLA layer the MLA modules read")
    ap.add_argument("--numerics-ms", default="1,64,2048")
    ap.add_argument("--hash-ms", default="512,2048,8192",
                    help="M values at which each lane's output is hashed on a (module, q256, M)-seeded "
                         "input, for a bitwise comparison across arms")
    ap.add_argument("--artifact", default=None, help="an exported checkpoint, for --served")
    ap.add_argument("--served", default="", help="module prefixes of --artifact to time on their served bytes")
    ap.add_argument("--served-tp", type=int, default=2, help="the TP size --served modules are cut for (rank 0)")
    ap.add_argument("--served-ms", default="512,2048,8192", help="the M values --served groups are timed at")
    ap.add_argument("--served-lanes", default="fused", help="the lanes --served groups are timed on")
    ap.add_argument("--source-layers", default="",
                    help="override MODULE_SOURCES' layers: module=layer[,module=layer...]")
    ap.add_argument("--l2", default="warm",
                    help="warm, cold or warm,cold: cold also times apply with L2 cleared before "
                         "each replay (call apply_cold), the state a served forward finds a module in")
    ap.add_argument("--k-splits", default="",
                    help="measurement only: also time the fused lane at these fixed K splits "
                         "(lane fused@S<s>; the launch's own model picks S on lane fused)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    source_layers = {k: layer for k, (_t, layer) in MODULE_SOURCES.items()}
    for item in (v for v in args.source_layers.split(",") if v):
        k, _, layer = item.partition("=")
        if k not in MODULE_SOURCES or not layer.isdigit():
            raise SystemExit(f"--source-layers takes module=layer over {sorted(MODULE_SOURCES)}, not {item!r}")
        source_layers[k] = int(layer)
    from tessera import routed_fused as rf
    from tessera.serving.native_ops import native_fp8_quant, require_native_fp8_quant
    require_native_fp8_quant("bench_dense_module")

    dev = torch.device("cuda")
    ms = [int(v) for v in args.ms.split(",")]
    power_ms = {int(v) for v in args.power_ms.split(",") if v}
    l2_modes = {v for v in args.l2.split(",") if v}
    if not l2_modes <= {"warm", "cold"} or not l2_modes:
        raise SystemExit(f"--l2 takes warm, cold or both, not {args.l2!r}")
    scratch = cold_scratch() if "cold" in l2_modes else None
    power = PowerSampler()
    meta = {"device": torch.cuda.get_device_name(), "sms": torch.cuda.get_device_properties(dev).multi_processor_count,
            "library": rf.library_for("e4m3"), "read_gbps": READ_GBPS, "mma_tflops": MMA_E4M3_TFLOPS,
            "kernel_sha": os.environ.get("KERNEL_SHA"), "tessera_head": os.environ.get("TESSERA_HEAD"),
            "image": os.environ.get("ORACLE_IMAGE"), "host": os.environ.get("HOST_NAME"),
            "pb_action": os.environ.get("PB_ACTION_KEY"), "power_source": power.source,
            "envelope_w": ENVELOPE_W, "start_unix": time.time(), "torch": torch.__version__,
            "dense_row_quantum": getattr(rf, "DENSE_ROW_QUANTUM", None),
            "weights": ({"model": args.model, "layer": args.layer, "mla_layer": args.mla_layer,
                         "source_layers": source_layers, "shard": "TP2 rank 0"} if args.model
                        else "seeded Gaussian"),
            "statistic": "mean of the forward and reverse passes' medians (graph replay); spread = |F - R| / mean",
            "l2": sorted(l2_modes),
            "cold_scratch_bytes": (scratch.numel() * 4 if scratch is not None else 0)}
    groups = []          # (key, builder) -> builder() returns (head, make) with make(m) -> (meta, call)
    for name in (v for v in args.modules.split(",") if v):
        roles, cols = MODULES[name]
        rows = sum(r for _, r in roles)
        for q in (int(v) for v in args.q256.split(",")):
            groups.append((f"{name}:q{q}", ("module", name, roles, cols, rows, q)))
        if args.refs:
            groups.append((f"{name}:bf16_linear", ("bf16", name, roles, cols, rows, None)))
            groups.append((f"{name}:scaled_mm", ("fp8", name, roles, cols, rows, None)))
    for prefix in (v for v in args.served.split(",") if v):
        if not args.artifact:
            raise SystemExit("--served needs --artifact")
        scheme, _roles, _columns, _cut, out_parts, in_size = served_geometry(args.artifact, prefix, args.served_tp)
        rank_roles = [(n, o) for (n, _r), o in zip(_roles, out_parts)]
        groups.append((f"served:{prefix}", ("served", prefix, rank_roles, in_size, sum(out_parts),
                                            int(scheme["q256"]))))
    cells = {}
    path = os.path.join(args.out, "bench_dense_module.json")

    def save():
        json.dump({"meta": meta, "groups": cells}, open(path, "w"), indent=1)

    def ref_calls(call):
        calls = {"apply": call} if "warm" in l2_modes else {}
        if "cold" in l2_modes:
            calls["apply_cold"] = call
        return calls

    def build(spec):
        kind, name, roles, cols, rows, q = spec
        g = torch.Generator(device=dev).manual_seed(zlib.crc32(f"{name}:{kind}:{q}".encode()))
        if kind in ("module", "served"):
            served = None
            if kind == "served":
                prepare, blob_bytes, blob_sha, cut = served_module(args.artifact, name, args.served_tp)
                enc_s, w_src, layer = 0.0, None, None
                served = {"artifact": args.artifact, "tp_size": args.served_tp, "tp_rank": 0, "cut": cut}
            else:
                template = MODULE_SOURCES[name][0] if name in MODULE_SOURCES else None
                layer = (source_layers[name] if name in MODULE_SOURCES
                         else args.mla_layer if name in MLA_MODULES else args.layer)
                source = ((lambda n, r: source_weight(args.model, layer, n, r, cols, template)) if args.model
                          else None)
                prepare, blob_bytes, enc_s, w_src, blob_sha = encode_module(
                    roles, cols, q, zlib.crc32(f"{name}:{q}".encode()), source)
            lanes = {}
            for lane in args.lanes.split(","):
                mod = prepare(lane == "fused")
                lanes[lane] = mod
            head = {"kind": kind, "module": name, "q256": q, "rows": rows, "cols": cols, "roles": roles,
                    "blob_bytes": blob_bytes, "blob_sha256": blob_sha, "encode_s": enc_s,
                    "source_layer": layer if args.model else None, "served": served,
                    "lanes": {k: {"lane": v.lane, "reason": v.lane_reason, "launch_pair": list(v.launch_pair)}
                              for k, v in lanes.items()}}
            head["numerics"] = numerics(lanes, w_src, cols,
                                        [int(v) for v in args.numerics_ms.split(",") if v],
                                        zlib.crc32(f"num:{name}:{q}".encode()))
            head["bitwise"] = bitwise(lanes, cols, [int(v) for v in args.hash_ms.split(",") if v], name, q)
            del w_src
            wire = sum(r * cols * q // 256 // 8 + 4 * r for _, r in roles)
            head["wire_bytes"] = wire

            def make(m, lane):
                base, _, forced = lane.partition("@S")
                mod = lanes[base]
                x = (torch.randn(m, cols, device=dev, generator=g) * 0.5).bfloat16()
                xq, a = native_fp8_quant(x)
                a = a.reshape(-1).contiguous().float()
                holder = {}

                def split(fn):
                    if not forced:
                        return fn

                    def at_split():
                        # the split is read at capture; replay runs what was captured
                        model = rf.dense_k_split
                        # capped where the library caps it (``dense_split_max``)
                        rf.dense_k_split = lambda m_, rows_, cols_, sms_, **kw: min(
                            int(forced), rf.dense_split_max(cols_))
                        try:
                            fn()
                        finally:
                            rf.dense_k_split = model
                    return at_split

                @split
                def apply():
                    holder["out"] = mod.apply(xq, a)

                @split
                def quant_apply():
                    q8, s8 = native_fp8_quant(x)
                    holder["out"] = mod.apply(q8, s8.reshape(-1))
                calls = {"apply": apply, "quant_apply": quant_apply}
                if "cold" in l2_modes:
                    calls["apply_cold"] = apply
                if "warm" not in l2_modes:
                    calls.pop("apply")
                    calls.pop("quant_apply")
                return {"floor": floors(wire, m, rows, cols)}, calls, holder
            extra = [f"fused@S{v}" for v in args.k_splits.split(",") if v] if "fused" in lanes else []
            if served is not None:
                return (head, make, [k for k in args.served_lanes.split(",") if k in lanes],
                        [int(v) for v in args.served_ms.split(",") if v])
            return head, make, list(lanes) + extra
        if kind == "bf16":
            w = (torch.randn(rows, cols, device=dev, generator=g) * 0.02).bfloat16()
            head = {"kind": kind, "module": name, "rows": rows, "cols": cols, "wire_bytes": rows * cols * 2}

            def make(m, _lane):
                x = (torch.randn(m, cols, device=dev, generator=g) * 0.5).bfloat16()
                holder = {}

                def call():
                    holder["out"] = torch.nn.functional.linear(x, w)
                return {"floor": floors(rows * cols * 2, m, rows, cols, a_bytes=2)}, ref_calls(call), holder
            return head, make, ["bf16"]
        w8 = (torch.randn(rows, cols, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
        sw = torch.rand(1, rows, device=dev, generator=g) * 1e-2 + 1e-3
        head = {"kind": kind, "module": name, "rows": rows, "cols": cols, "wire_bytes": rows * cols + 4 * rows}

        def make(m, _lane):
            x = (torch.randn(m, cols, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
            sa = torch.rand(m, 1, device=dev, generator=g) * 0.1 + 0.01
            holder = {}

            def call():
                holder["out"] = torch._scaled_mm(x, w8.t(), scale_a=sa, scale_b=sw, out_dtype=torch.bfloat16)
            return {"floor": floors(rows * cols + 4 * rows, m, rows, cols)}, ref_calls(call), holder
        return head, make, ["scaled_mm"]

    built = {}
    for pas in (("F",) if args.ncu else ("F", "R")):
        order = groups if pas == "F" else list(reversed(groups))
        for key, spec in order:
            if key not in built:
                built[key] = build(spec)
                cells[key] = dict(built[key][0], cells={})
            head, make, lanes = built[key][:3]
            gms = built[key][3] if len(built[key]) > 3 else ms
            seq = [(m, lane) for lane in lanes for m in gms]
            for m, lane in (seq if pas == "F" else list(reversed(seq))):
                ckey = f"{lane}:{m}"
                cell = cells[key]["cells"].setdefault(ckey, {})
                try:
                    cmeta, calls, holder = make(m, lane)
                    if args.ncu:
                        # ncu --profile-from-start off: exactly one profiled apply per cell.
                        for _ in range(3):
                            calls["apply"]()
                        torch.cuda.synchronize()
                        torch.cuda.cudart().cudaProfilerStart()
                        calls["apply"]()
                        torch.cuda.synchronize()
                        torch.cuda.cudart().cudaProfilerStop()
                        cell["ncu"] = True
                        print(json.dumps({"g": key, "cell": ckey, "ncu": True}), flush=True)
                        del calls, holder
                        continue
                    for cname, call in calls.items():
                        cold = cname.endswith("_cold")
                        samples = (graph_time_cold(call, args.warmup, args.iters, scratch) if cold
                                   else graph_time(call, args.warmup, args.iters))
                        rec = cell.setdefault(cname, {})
                        rec[pas] = {"median_ms": statistics.median(samples), "min_ms": min(samples),
                                    "unix": time.time()}
                        if pas == "F":
                            rec["floor"] = cmeta["floor"]
                            if cname == "apply":
                                rec["profile"] = kernel_profile(call, reps=3)
                            elif cold:
                                rec["profile"] = kernel_profile_cold(call, scratch, reps=3)
                                if m in power_ms:
                                    rec["power"] = power.sample_during(call, args.power_s)
                        else:
                            f, r = rec.get("F", {}).get("median_ms"), rec["R"]["median_ms"]
                            if f:
                                rec["ms"] = 0.5 * (f + r)
                                rec["spread"] = abs(f - r) / rec["ms"]
                                fl = rec["floor"]
                                rec["copy_frac"] = fl["byte_ms"] / rec["ms"]
                                rec["roof_frac"] = fl["floor_ms"] / rec["ms"]
                                print(json.dumps({"g": key, "cell": ckey, "call": cname,
                                                  "ms": round(rec["ms"], 4), "spread": round(rec["spread"], 4),
                                                  "copy": round(rec["copy_frac"], 3),
                                                  "roof": round(rec["roof_frac"], 3)}), flush=True)
                    del calls, holder
                except Exception as exc:  # noqa: BLE001
                    cell.setdefault("error", {})[pas] = repr(exc)[:500]
                    print(json.dumps({"g": key, "cell": ckey, "pass": pas, "error": repr(exc)[:300]}), flush=True)
            save()
    meta["end_unix"] = time.time()
    save()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
