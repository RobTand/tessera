"""Finite native T-4 serving checks and a matched-budget weight-space screen.

GPU entry: --mode correctness|timing [--q256 896] [--ms 1 16 2048 4096].
Before that invocation, run THE SAME arguments plus --dry-run on PB x86.
CPU independent fixture: --mode research-fixture. It is not serving evidence.
Real weight screen: --mode quality-screen --source-spec SPEC --data-manifest MANIFEST.
Prepare the bounded model read set on PB x86: --mode prepare-inputs --out DIRECTORY.

Production builders create/load/finalize/apply the actual wire. Their retained
owners supply diagnostic mode 0/1/2 calls; they are never constructed in place
of the production route. The existing fused_e2m1_check byte-weight, activation,
one-hot and comparison helpers are the numerical oracle. Random checks remain
conditional FP32 diagnostics, not a resolution of issue 1007/PR 1008's missing
native MMA contract. A successful screen is not PACT, G3, or end-to-end quality.

D30 containment uses the existing managed_window.Envelope subprocess owner,
with a sampled local MemAvailable callback, owned TERM then KILL, and a finite
deadline. This is sampled protection, not a guarantee against transient OOM.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import struct
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
PURE_Q256 = tuple(range(128, 1025, 128))
ALL_M = (1, 16, 2048, 4096)
SCHEMA = "tessera.t4_fused_qualify.v3"
ROLE_NAMES = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
ROLE_SEEDS = {"gate": 11, "up": 37, "down": 71}
DENSE_SHAPES = (("index_weights", 32, 4096), ("index_wk", 128, 4096),
                ("mla_qa", 1536, 4096))


def head_stamp():
    if os.environ.get("TESSERA_HEAD"):
        return os.environ["TESSERA_HEAD"]
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def mem_available_gib():
    values = Path("/proc/meminfo").read_text().splitlines()
    value = next((line.split() for line in values if line.startswith("MemAvailable:")), None)
    if value is None or len(value) != 3 or value[2] != "kB":
        raise RuntimeError("MemAvailable safety observation unavailable")
    result = int(value[1]) / 1048576
    if not math.isfinite(result) or result < 0:
        raise RuntimeError("MemAvailable safety observation invalid")
    return result


def check_mem_guard(floor=2.0):
    available = mem_available_gib()
    if available is None or not isinstance(available, (int, float)) or not math.isfinite(available):
        raise RuntimeError("MemAvailable safety observation unavailable")
    if available < floor:
        raise RuntimeError(f"MemAvailable {available:.3f} GiB below {floor:.3f} GiB")
    return {"available_gib": available, "floor_gib": floor}


def recipe_kwargs(recipe):
    return {"body": recipe.body, "span": recipe.span, "scale_plane": recipe.scale_plane,
            "window_bits": recipe.window_bits, "window_seed": recipe.window_seed,
            "window_sigma": recipe.window_sigma, "channel_sigma": recipe.channel_sigma}


def resolve_window_kwargs(q256, structure="dense"):
    from tessera.alphabet import E2M1_GRID, tuple_grid
    from tessera.export import served_recipe
    from tessera.manifest import BodyKind, ScalePlaneKind

    grid = tuple_grid(E2M1_GRID, 2)
    recipe = served_recipe(grid, q256, structure)
    if (recipe.body != BodyKind.WINDOW or recipe.span != 1
            or recipe.scale_plane != ScalePlaneKind.LUT or recipe.window_bits != 14):
        raise ValueError("integrated served_recipe does not supply WINDOW span 1 LUT L14")
    return dict(grid=grid, q256=q256, **recipe_kwargs(recipe)), "served-recipe"


def encode_bytes(weight, q256, kind="window", structure="dense"):
    from tessera.alphabet import E2M1_GRID, tuple_grid
    from tessera.export import TCQ_RECIPE, encode_linear
    from tessera.manifest import BodyKind, ScalePlaneKind

    if kind == "window":
        kwargs, source = resolve_window_kwargs(q256, structure)
    elif kind == "research-window":
        kwargs = dict(grid=tuple_grid(E2M1_GRID, 2), q256=q256, body=BodyKind.WINDOW,
                      span=1, scale_plane=ScalePlaneKind.LUT, window_bits=14)
        source = "explicit-research-fixture-not-serving"
    elif kind == "tcq":
        kwargs = dict(grid=tuple_grid(E2M1_GRID, 2), q256=q256, **recipe_kwargs(TCQ_RECIPE))
        source = "explicit-TCQ-span2-screen-not-serving"
    else:
        raise ValueError(f"unknown encoding {kind}")
    return bytes(encode_linear(weight, **kwargs).blob), source


def byte_receipt(blob, rows, cols):
    from tessera.unit_artifact import parse_unit_metadata

    metadata = parse_unit_metadata(blob, device="cpu")
    return {"wire_bytes": len(blob), "wire_sha256": hashlib.sha256(blob).hexdigest(),
            "rows": rows, "cols": cols, "bits_per_256_weights": len(blob) * 8 * 256 / (rows * cols),
            "code_table_bytes": metadata.manifest.window_bits and 16384,
            "bytes_include": "container, manifest, tuple table, scale planes and packed body"}


def synthetic(rows, cols, seed, device):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(rows, cols, generator=generator) * .02).to(device)


def routing(m, experts, top_k, seed, device):
    import torch

    generator = torch.Generator().manual_seed(seed)
    # Balanced round robin before shuffle, including repeated experts in a token
    # if top_k exceeds the fixture expert count; partial/empty tiles occur at M1.
    ids = (torch.arange(m * top_k) % experts)[torch.randperm(m * top_k, generator=generator)]
    weights = torch.rand(m * top_k, generator=generator).float() + .25
    return ids.reshape(m, top_k).to(device), weights.reshape(m, top_k).to(device)


def dense_scheme(blob, rows, cols, q256, name="weight"):
    return dict(family="TESSERA_NVFP4", structure="dense", grid="E2M1x2", body="WINDOW",
                plane="LUT", span=1, q256=q256, rows=rows, columns=cols,
                wire_bytes=len(blob), roles=[[name, rows]])


def routed_scheme(frames, hidden, inter, experts, q256):
    return dict(family="TESSERA_NVFP4", structure="routed_moe", grid="E2M1x2",
                body="WINDOW", plane="LUT", span=1, experts=experts,
                groups={"w13": dict(rows=2 * inter, columns=hidden, q256=q256,
                                    wire_stride=max(len(b) for r in ("gate", "up") for b in frames[r]),
                                    roles=[["gate_proj", inter], ["up_proj", inter]]),
                        "w2": dict(rows=hidden, columns=inter, q256=q256,
                                   wire_stride=max(map(len, frames["down"])),
                                   roles=[["down_proj", hidden]])})


def dense_frame(blob, rows):
    from tessera.fused_frame import pack_fused
    return pack_fused([("weight", rows, blob)])


def load_dense_route(blob, rows, cols, q256, *, tp_size=1, tp_rank=0, axis="row"):
    import torch
    from tessera.serving.nvfp4_route import build_tessera_nvfp4_method

    frame = dense_frame(blob, rows)
    scheme = dense_scheme(frame, rows, cols, q256)
    layer = torch.nn.Module()  # genuine registered parameters, no method/module mocks
    layer.tp_rank, layer.tp_size = tp_rank, tp_size
    method = build_tessera_nvfp4_method(scheme, "qualification.dense", "resident")
    local_rows = rows // tp_size if axis == "row" else rows
    local_cols = cols // tp_size if axis == "column" else cols
    method.create_weights(layer, local_cols, [local_rows], cols, rows, torch.bfloat16)
    layer.wire_bytes.data.copy_(torch.frombuffer(bytearray(frame), dtype=torch.uint8))
    layer.trellis_input_global_scale.data.fill_(100.0)
    layer.to("cuda")
    method.process_weights_after_loading(layer)
    if not getattr(layer, "tessera_a4_roles", None):
        raise ValueError("production dense route did not retain native WINDOW roles")
    return layer, method


def make_moe_config(hidden, inter, experts, top_k, tp_size, tp_rank):
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig

    parallel = FusedMoEParallelConfig(tp_size=tp_size, pcp_size=1, dp_size=1, ep_size=1,
        tp_rank=tp_rank, pcp_rank=0, dp_rank=0, ep_rank=0, sp_size=1, use_ep=False,
        all2all_backend="allgather_reducescatter", enable_eplb=False)
    return FusedMoEConfig(num_experts=experts, experts_per_token=top_k, hidden_dim=hidden,
        intermediate_size=inter, num_local_experts=experts, num_logical_experts=experts,
        activation=MoEActivation.SILU, device=torch.device("cuda"), routing_method="topk",
        moe_parallel_config=parallel, in_dtype=torch.bfloat16,
        intermediate_size_per_partition=inter // tp_size)


def load_routed_route(frames, hidden, inter, experts, top_k, q256, tp_size, tp_rank):
    import torch
    from tessera.serving.nvfp4_moe_route import build_tessera_nvfp4_moe_method
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    layer = torch.nn.Module()
    layer.moe_config = make_moe_config(hidden, inter, experts, top_k, tp_size, tp_rank)
    layer.activation = MoEActivation.SILU
    layer.expert_map = None
    layer.global_num_experts = experts
    layer.apply_router_weight_on_input = False
    layer.swiglu_limit = None
    layer.swiglu_alpha = layer.swiglu_beta = None
    method = build_tessera_nvfp4_moe_method(
        routed_scheme(frames, hidden, inter, experts, q256), "qualification.experts", "resident", layer)
    method.create_weights(layer, experts, hidden, inter // tp_size, torch.bfloat16,
                          global_num_experts=experts)
    layer.to("cuda")
    for part, shard in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
        wire_parameter = layer.w2_wire if part == "down" else layer.w13_wire
        scale_parameter = (layer.w2_input_global_scale if part == "down"
                           else layer.w13_input_global_scale)
        for e, frame in enumerate(frames[part]):
            wire_parameter.weight_loader(wire_parameter,
                torch.frombuffer(bytearray(frame), dtype=torch.uint8), "weight", shard, e)
            # Distinct calibrated scales exercise the existing reduction, not
            # an arbitrary replacement static scale in the owner constructor.
            scale_parameter.weight_loader(scale_parameter,
                torch.tensor([100.0 + e + ROLE_SEEDS[part]], dtype=torch.float32),
                "input_global_scale", shard, e)
    method.process_weights_after_loading(layer)
    owner = getattr(layer, "tessera_routed_fused", None)
    if owner is None or owner.family != "e2m1":
        raise ValueError("production routed intake did not retain the native E2M1 owner")
    return layer, method, owner


class ReferenceWeights:
    """Decode one selected expert at a time, not the complete expert pool."""
    def __init__(self, blobs, cut, axis):
        self.blobs, self.cut, self.axis = blobs, cut, axis

    def __len__(self):
        return len(self.blobs)

    def __getitem__(self, expert):
        from experiments.t4_code.fused_e2m1_check import ref_weight
        return ref_weight(self.blobs[expert], self.cut, "cuda", self.axis)


def decode_references(blobs, cut=None, axis="rows"):
    return ReferenceWeights(blobs, cut, axis)


def quantized_reference(x, gs):
    from tessera.kernel_a4 import a4_quantize_activation
    from experiments.t4_code.fused_e2m1_check import a_deq

    codes, scale = a4_quantize_activation(x.contiguous(), gs)
    return codes, scale.view(__import__("torch").uint8).contiguous(), a_deq(codes, scale)


def conditional_gamma(operations):
    # Full ULP, NOT RN's half ULP: conditional arithmetic model only.
    eps = 2.0 ** -23
    if operations * eps >= 1:
        raise ValueError("conditional FP32 gamma domain exceeded")
    return operations * eps / (1 - operations * eps)


def contraction_reference(a, weights, ids, ratios, *, exact, router_weights=None):
    import torch

    n = weights[0].shape[0]
    reference = torch.empty(a.shape[0], n, dtype=torch.float64, device=a.device)
    bounds = torch.empty_like(reference)
    k = int(a.shape[1])
    for e in range(len(weights)):
        selected = (ids == e).nonzero().reshape(-1)
        if not selected.numel():
            continue
        weight = weights[e]
        # Tile the independent FP64 reductions; never keep the whole model or
        # a [routes,N,K] products tensor. All requested outputs are checked.
        for start in range(0, n, 128):
            w = weight[start:start + 128]
            operands = a[selected]
            acc = operands @ w.T
            mag = operands.abs() @ w.abs().T
            ratio = ratios[e]
            if exact:
                value = acc.float() * ratio
                if router_weights is not None:
                    value = value * router_weights[selected, None]
                reference[selected, start:start + 128] = value.double()
                bounds[selected, start:start + 128] = 0
            else:
                factor = ratio.double()
                if router_weights is not None:
                    factor = factor * router_weights[selected, None].double()
                reference[selected, start:start + 128] = acc * factor
                # Magnitude's FP64 positive sum is inflated outward before the
                # conditional fp32 contraction + epilogue envelope.
                mag = torch.nextafter(mag / (1 - k * 2.0 ** -52),
                    torch.full_like(mag, float("inf")))
                bounds[selected, start:start + 128] = mag * factor.abs() * conditional_gamma(2 * k + 3)
    return reference, bounds


def compare_projection(got, reference, bound, exact):
    from experiments.t4_code.fused_e2m1_check import compare

    result = compare(got, reference, bound, exact)
    result.update(exact=exact, bound_kind="exact-one-hot" if exact else "conditional-FP32-full-ULP-gamma",
                  max_bound=float(bound.max()), arithmetic_qualified=False,
                  limitation="native MMA local error/order/subnormals unresolved in issue 1007/PR 1008")
    return result


def fixed_token_sum(route_rows, m, top_k):
    import torch

    value = torch.zeros(m, route_rows.shape[1], dtype=torch.float32, device=route_rows.device)
    rows = route_rows.reshape(m, top_k, -1)
    for j in range(top_k):
        value = value + rows[:, j].float()
    return value.to(torch.bfloat16)


def capture_callable(fn):
    import torch

    eager = fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    graph.replay()
    torch.cuda.synchronize()
    equal = bool(torch.equal(eager.view(torch.int16), output.view(torch.int16)))
    return eager, graph, output, equal


def time_callable(fn, warmup, iters):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    # Acquisition order is authoritative. Sorting is only the median helper's
    # internal operation and never rewrites the raw event stream.
    return dict(samples_ms=samples, median_ms=statistics.median(samples), min_ms=min(samples))


def parse_resource_usage(raw):
    """cuobjdump --dump-resource-usage: retain every matching native template."""
    records = []
    name = None
    for line in raw.splitlines():
        found = re.search(r"Function\s*:\s*(\S+)", line)
        if found:
            name = found.group(1)
        fields = dict((k, int(v)) for k, v in re.findall(
            r"\b(REG|STACK|SHARED|LOCAL|CONSTANT\[\d+\])\s*:\s*(\d+)", line))
        if name and "REG" in fields:
            records.append(dict(symbol=name, **fields))
    if not records:
        raise ValueError("cuobjdump supplied no register resource records")
    return records


def native_resources():
    from tessera.routed_fused_e2m1 import _ext

    library = _ext()
    executable = shutil.which("cuobjdump")
    if executable is None:
        raise ValueError("D41 register extraction requires cuobjdump in the serving image")
    command = [executable, "--dump-resource-usage", library.__file__]
    raw = subprocess.check_output(command, text=True, timeout=60)
    functions = parse_resource_usage(raw)
    demangler = shutil.which("c++filt")
    if demangler is None:
        raise ValueError("D41 template resource extraction requires c++filt")
    names = subprocess.check_output([demangler], input="\n".join(r["symbol"] for r in functions),
                                    text=True, timeout=30).splitlines()
    for record, name in zip(functions, names):
        record["demangled"] = name
        match = re.search(r"routed_fused_fp4_kernel<(\d+), (true|false), (true|false), (\d+), (true|false)>", name)
        if match:
            mode, dense, split, rate, mixed = match.groups()
            record.update(mode=int(mode), dense=dense == "true", split=split == "true",
                          rate=int(rate), mixed=mixed == "true")
    return dict(library=str(library.__file__), command=command, raw=raw, functions=functions)


def measured_registers(resources, q, kind):
    modes = [0, 2] if kind in ("chain", "router_input") else ([2] if kind == "dense" else [int(kind[-1])])
    result = []
    for mode in modes:
        matches = [r for r in resources["functions"] if r.get("rate") == q // 128
                   and r.get("mode") == mode and r.get("dense") == (kind == "dense")
                   and r.get("mixed") is False and r.get("split") is False]
        if not matches:
            raise ValueError(f"D41 register resources missing for q{q} mode{mode} {kind}")
        result.extend(matches)
    return result


def geometry(role, mode):
    import torch
    from tessera.routed_fused_e2m1 import _ext, smem_bytes

    words = role.words
    need = smem_bytes(mode, role.slot_words)
    available = int(_ext().max_dynamic_smem_bytes(torch.cuda.current_device()))
    return dict(dynamic_smem_bytes=need, max_dynamic_smem_bytes=available, smem_fit=need <= available,
                slot_words=int(role.slot_words), tile_words=int(role.tile_words),
                words_alignment_bytes=words.data_ptr() % 16,
                words_stride_alignment_bytes=words.stride(0) * words.element_size() % 16,
                decode_width_cols=64, window_bits=14, word_layout="legacy")


def bundle_geometry(owner, part, mode):
    from types import SimpleNamespace

    role = SimpleNamespace(words=getattr(owner, "words_" + part),
        slot_words=owner.slot_words_down if part == "down" else owner.slot_words_gate_up,
        tile_words=owner.tile_words_down if part == "down" else owner.tile_words_gate_up)
    return geometry(role, mode)


def storage_bytes(named):
    from tessera.serving.residency import resident_storage_bytes

    return resident_storage_bytes(named)


def source_stamp(args):
    paths = [ROOT / "src/tessera/serving/nvfp4_route.py",
             ROOT / "src/tessera/serving/nvfp4_moe_route.py",
             ROOT / "src/tessera/serving/csrc/routed_fused_window.cu"]
    return dict(head=head_stamp(), files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in paths}, pb_action=os.environ.get("PRISMABUILD_ACTION_KEY"),
                image=os.environ.get("ORACLE_IMAGE"), host=os.environ.get("HOST_NAME"), args=vars(args))


def row_key(q, m, kind, shape, pattern, rank):
    return f"q{q}/M{m}/{kind}/{shape}/{pattern}/{rank}"


def rank_cases(args):
    return [(1, 0)] + ([(2, 0), (2, 1)] if args.tp_cuts else [])


def dense_axes(args, shape, size):
    # A replicated fixture is explicit. Its layer still carries its actual
    # world size. Output height never selects a sharding policy.
    del shape
    return list(args.dense_shard_axes) if size > 1 else ["replicated"]


def validate_rank_local_geometry(args):
    from tessera.serving.scheme import e2m1_shape_reason
    checked = []
    for size, rank in rank_cases(args):
        if args.part in ("all", "routed"):
            local_inter = args.inter // size
            for part, rows, columns in (("gate_proj", local_inter, args.hidden),
                                       ("up_proj", local_inter, args.hidden),
                                       ("down_proj", args.hidden, local_inter)):
                reason = e2m1_shape_reason(rows, columns, structure="routed_moe", projection=part)
                if reason:
                    raise ValueError(f"TP{size} rank{rank} {part}: {reason}")
                checked.append(dict(projection=part, rows=rows, columns=columns, tp_size=size, tp_rank=rank))
        if args.part in ("all", "dense"):
            for shape in args.dense_shapes:
                tag, rows, columns = shape
                for axis in dense_axes(args, shape, size):
                    if (axis == "row" and rows % size) or (axis == "column" and columns % size):
                        raise ValueError(f"{tag}: requested TP cut has a fractional extent")
                    local_rows = rows // size if axis == "row" else rows
                    local_columns = columns // size if axis == "column" else columns
                    reason = e2m1_shape_reason(local_rows, local_columns, structure="dense")
                    if reason:
                        raise ValueError(f"TP{size} rank{rank} {tag} {axis}: {reason}")
                    checked.append(dict(projection=tag, rows=local_rows, columns=local_columns,
                                        tp_size=size, tp_rank=rank, axis=axis))
    return checked


def requested_keys(args):
    patterns = ["random"] if args.mode == "timing" else ["onehot", "random"]
    result = []
    for q in args.q256:
        for size, rank in rank_cases(args):
            for m in args.ms:
                for pattern in patterns:
                    if args.part in ("all", "routed"):
                        for kind in ("mode0", "mode1", "mode2", "chain", "router_input"):
                            result.append(row_key(q, m, kind, f"{args.hidden}x{args.inter}", pattern,
                                                  f"TP{size}r{rank}"))
                    if args.part in ("all", "dense"):
                        for tag, rows, cols in args.dense_shapes:
                            # TP fixtures explicitly use row cuts (and separately
                            # column cuts when --dense-column-cuts is selected).
                            for axis in dense_axes(args, (tag, rows, cols), size):
                                result.append(row_key(q, m, "dense", f"{tag}:{rows}x{cols}:{axis}", pattern,
                                                      f"TP{size}r{rank}"))
    return result


def finish_cell(key, fn, numeric, metadata, args):
    eager, graph, replayed, equal = capture_callable(fn)
    check = numeric(eager)
    row = dict(key=key, **metadata, eager_ok=check["ok"], graph_equal=equal, numeric=check,
               output_sha256=hashlib.sha256(eager.contiguous().view(__import__("torch").uint8)
                                            .cpu().numpy().tobytes()).hexdigest())
    if args.mode == "timing":
        row["timing"] = {"eager": time_callable(fn, args.warmup, args.iters),
                         "graph": time_callable(graph.replay, args.warmup, args.iters)}
    return row


def routed_cells(args, q, blobs, frames, tp_size, rank):
    import torch
    from experiments.t4_code.fused_e2m1_check import bf16_ulp, onehot_x

    layer, method, owner = load_routed_route(frames, args.hidden, args.inter, args.experts,
                                           args.top_k, q, tp_size, rank)
    inter = args.inter // tp_size
    cut = (rank * inter, (rank + 1) * inter) if tp_size > 1 else None
    weights = {p: decode_references(blobs[p], cut, "cols" if p == "down" else "rows")
               for p in ROLE_NAMES}
    resident = storage_bytes(method.resident_tensors(layer))
    base = dict(q256=q, rate_class=f"R{q // 128}",
                source_kind=("synthetic-distinct-experts" if args.mode != "timing"
                             else "synthetic-one-wire-per-projection-distinct-resident-slots"),
                hidden=args.hidden, inter=inter, experts=args.experts, top_k=args.top_k,
                total_serialized_bytes=sum(map(len, (b for part in frames.values() for b in part))),
                projection_serialized_bytes={p: sum(map(len, b)) for p, b in frames.items()},
                resident_bytes=resident, serving_owner=True,
                serving_intake="nvfp4_moe_route create_weights/wire+scale loaders/finalize/apply",
                tp_size=tp_size, tp_rank=rank,
                geometry={"gate_up": bundle_geometry(owner, "gate", 0),
                          "down": bundle_geometry(owner, "down", 2)})
    if args.mode == "timing":
        base["source_weight_seeds"] = {p: args.seed + ROLE_SEEDS[p] for p in ROLE_NAMES}
        base["source_weight_sha256"] = {p: tensor_digest(synthetic(*((args.hidden, args.inter) if p == "down"
                                           else (args.inter, args.hidden)), args.seed + ROLE_SEEDS[p], "cpu"))
                                       for p in ROLE_NAMES}
    patterns = ["random"] if args.mode == "timing" else ["onehot", "random"]
    for m in args.ms:
        ids, rw = routing(m, args.experts, args.top_k, args.seed + m, "cuda")
        flat_ids = ids.reshape(-1)
        routes = m * args.top_k
        for pattern in patterns:
            exact = pattern == "onehot"
            if exact:
                x = onehot_x(m, args.hidden, args.seed + m, "cuda")
                xd = onehot_x(routes, inter, args.seed + m + 1, "cuda")
            else:
                x = synthetic(m, args.hidden, args.seed + m, "cuda").to(torch.bfloat16)
                xd = synthetic(routes, inter, args.seed + m + 1, "cuda").to(torch.bfloat16)
            _, _, a = quantized_reference(x, owner.gs13)
            ar = a.repeat_interleave(args.top_k, dim=0)
            rg, bg = contraction_reference(ar, weights["gate"], flat_ids, owner.ratio_gate, exact=exact)
            ru, bu = contraction_reference(ar, weights["up"], flat_ids, owner.ratio_up, exact=exact)
            ref = torch.cat((rg, ru), dim=1).reshape(m, args.top_k, -1)
            bound = torch.cat((bg, bu), dim=1).reshape_as(ref)
            key = lambda kind: row_key(q, m, kind, f"{args.hidden}x{args.inter}", pattern,
                                        f"TP{tp_size}r{rank}")
            metadata = dict(base, m=m, pattern=pattern,
                seed=args.seed + m, input_sha256=tensor_digest(x), routing_sha256=tensor_digest(ids),
                routing_weights_sha256=tensor_digest(rw))
            fn1 = lambda: owner.gate_up(x, ids, rw)
            yield finish_cell(key("mode1"), fn1,
                lambda got: compare_projection(got, ref, bound, exact), dict(metadata, mode=1), args)
            # Reuse the already validated rounded gate/up for the nonlinear
            # stage, as the existing oracle does; native mode0 rounds them
            # BEFORE SiLU. This isolates the nonlinear stage from contraction.
            gu = fn1().reshape(routes, 2 * inter).float()
            routing_tables = owner._routing(ids, rw)
            xq, sfa, _ = quantized_reference(x, owner.gs13)
            mode0_out = torch.empty(routes, inter, dtype=torch.bfloat16, device="cuda")
            def fn0():
                owner._launch(0, xq, sfa, routing_tables, a_row_mode=0, mul_weight=False,
                              limit=float("inf"), out=mode0_out, counter=0)
                return mode0_out
            act = ((gu[:, :inter] / (1 + torch.exp(-gu[:, :inter]))) * gu[:, inter:]).to(torch.bfloat16)
            want0 = act[routing_tables.flat_sorted.long()]
            def compare_nonlinear(got):
                error = (got.double() - want0.double()).abs()
                allowance = bf16_ulp(want0.double())
                finite = bool(torch.isfinite(got).all())
                return dict(ok=finite and bool((error <= allowance).all()), mismatch=int((error > allowance).sum()),
                            max_err=float(error.max()), max_bound=float(allowance.max()),
                            bound_kind="one-BF16-ULP nonlinear rounding diagnostic", arithmetic_qualified=False)
            yield finish_cell(key("mode0"), fn0, compare_nonlinear, dict(metadata, mode=0), args)
            _, _, ad = quantized_reference(xd, owner.gs2)
            rd, bd = contraction_reference(ad, weights["down"], flat_ids, owner.ratio_down,
                                           exact=exact, router_weights=rw.reshape(-1))
            # Check down before combine via the owner's mode2 entry, then the
            # public down_routes fixed-order output against independently combined rows.
            def down_numeric(got, expected=rd, allowance=bd):
                rounded = expected.float().to(torch.bfloat16)
                want = fixed_token_sum(rounded, m, args.top_k)
                # Exact one-hot requires all route rounds and the final fixed
                # FP32 combine. Random allowance propagates per-route BF16
                # rounding plus the conditional contraction before that combine.
                if exact:
                    return compare_projection(got, want.double(), torch.zeros_like(want.double()), True)
                propagated = (allowance + bf16_ulp(expected)).reshape(m, args.top_k, -1).sum(1)
                propagated += conditional_gamma(args.top_k) * rounded.double().abs().reshape(m, args.top_k, -1).sum(1)
                return compare_projection(got, want.double(), propagated, False)
            yield finish_cell(key("mode2"), lambda: owner.down_routes(xd, ids, rw), down_numeric,
                              dict(metadata, mode=2, input_sha256=tensor_digest(xd)), args)
            # A complete route apply plus a separately executed mode0 -> mode2
            # owner chain. Both also receive a reference check of down's exact
            # executed quantized mode0 activation, not a finite-output check.
            act0 = fn0().clone()
            aq, sf2, ac = quantized_reference(act0, owner.gs2)
            sorted_ids = flat_ids[routing_tables.flat_sorted.long()]
            cr, cb = contraction_reference(ac, weights["down"], sorted_ids, owner.ratio_down,
                                            exact=False, router_weights=routing_tables.rw_sorted)
            unsorted_ref, unsorted_bound = torch.empty_like(cr), torch.empty_like(cb)
            unsorted_ref[routing_tables.flat_sorted.long()] = cr
            unsorted_bound[routing_tables.flat_sorted.long()] = cb
            route_rows = torch.empty(routes, args.hidden, dtype=torch.bfloat16, device="cuda")
            def fnchain_stage():
                fn0()
                qact, sact = owner._quantized(mode0_out, owner.gs2)
                owner._launch(2, qact, sact, routing_tables, a_row_mode=1, mul_weight=True,
                              limit=float("inf"), out=route_rows, counter=1)
                return fixed_token_sum(route_rows, m, args.top_k)
            staged = fnchain_stage().clone()
            def chain_numeric(got):
                rounded = unsorted_ref.float().to(torch.bfloat16)
                want = fixed_token_sum(rounded, m, args.top_k)
                tolerance = (unsorted_bound + bf16_ulp(unsorted_ref)).reshape(m, args.top_k, -1).sum(1)
                tolerance += conditional_gamma(args.top_k) * rounded.double().abs().reshape(m, args.top_k, -1).sum(1)
                result = compare_projection(got, want.double(), tolerance, False)
                result["route_vs_staged_bitwise"] = bool(torch.equal(got.view(torch.int16), staged.view(torch.int16)))
                result["ok"] = result["ok"] and result["route_vs_staged_bitwise"]
                return result
            apply = lambda: method.apply(layer, x, rw, ids, None, None)
            yield finish_cell(key("chain"), apply, chain_numeric, dict(metadata, mode="0+2+combine"), args)
            # The production top-one router-input contract must apply weights
            # before quantization rather than after down. Compare its real
            # route apply to the same owner forward on explicitly scaled BF16 X.
            top_ids, top_rw = ids[:, :1].contiguous(), rw[:, :1].contiguous()
            weighted_x = x * top_rw.to(x.dtype)
            # Owner's input placement uses X internally; call with original X
            # for both paths and check against an unweighted-input explicit
            # top-one chain at routing weights one.
            placed = owner(weighted_x, top_ids, torch.ones_like(top_rw))
            layer.apply_router_weight_on_input = True
            def router_input():
                return method.apply(layer, x, top_rw, top_ids, None, None)
            def check_router(got):
                equal = bool(torch.equal(got.view(torch.int16), placed.view(torch.int16)))
                return dict(ok=equal, mismatch=int((got != placed).sum()),
                            bound_kind="bitwise router-weight input placement", arithmetic_qualified=False)
            yield finish_cell(key("router_input"), router_input, check_router,
                dict(metadata, mode="router-input", top_k=1,
                     routing_sha256=tensor_digest(top_ids), routing_weights_sha256=tensor_digest(top_rw)), args)
            layer.apply_router_weight_on_input = False


def tensor_digest(tensor):
    import torch

    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def dense_cells(args, q, blob, shape, tp_size, rank, axis):
    import torch
    from experiments.t4_code.fused_e2m1_check import onehot_x, ref_weight

    tag, rows, cols = shape
    layer, method = load_dense_route(blob, rows, cols, q, tp_size=tp_size, tp_rank=rank, axis=axis)
    role = layer.tessera_a4_roles[0]
    cut = None
    if tp_size > 1 and axis != "replicated":
        extent = cols if axis == "column" else rows
        cut = (rank * (extent // tp_size), (rank + 1) * (extent // tp_size))
    weight = ref_weight(blob, cut, "cuda", "cols" if axis == "column" else "rows")
    accounting = byte_receipt(blob, rows, cols)
    accounting["unit_wire_bytes"] = accounting["wire_bytes"]
    accounting["wire_bytes"] = len(dense_frame(blob, rows))
    accounting["framing_bytes"] = accounting["wire_bytes"] - accounting["unit_wire_bytes"]
    accounting["bits_per_256_weights"] = accounting["wire_bytes"] * 8 * 256 / (rows * cols)
    base = dict(accounting, q256=q, rate_class=f"R{q // 128}", projection="dense",
        shape_tag=tag, tp_size=tp_size, tp_rank=rank, cut_axis=axis, cut=cut,
        serving_owner=True, serving_intake="nvfp4_route create_weights/load/finalize/apply",
        resident_bytes=storage_bytes(method.resident_tensors(layer)), geometry=geometry(role, 2),
        source_weight_seeds={"weight": args.seed + rows},
        source_weight_sha256={"weight": tensor_digest(synthetic(rows, cols, args.seed + rows, "cpu"))})
    patterns = ["random"] if args.mode == "timing" else ["onehot", "random"]
    for m in args.ms:
        for pattern in patterns:
            exact = pattern == "onehot"
            x = (onehot_x(m, role.cols, args.seed + m, "cuda") if exact
                 else synthetic(m, role.cols, args.seed + m, "cuda").to(torch.bfloat16))
            _, _, a = quantized_reference(x, role.gs)
            ids = torch.zeros(m, dtype=torch.int64, device="cuda")
            reference, bound = contraction_reference(a, [weight], ids, role.ratio, exact=exact)
            yield finish_cell(row_key(q, m, "dense", f"{tag}:{rows}x{cols}:{axis}", pattern,
                                      f"TP{tp_size}r{rank}"), lambda: method.apply(layer, x),
                lambda got: compare_projection(got, reference, bound, exact),
                dict(base, m=m, pattern=pattern, seed=args.seed + m, input_sha256=tensor_digest(x)), args)


def compare_t8_cell(cell, baseline):
    """A current comparability fact, not a recorded-identity seal."""
    fields = ["m", "wire_bytes", "tp_size", "tp_rank", "seed", "input_sha256",
              "source_weight_seeds", "source_weight_sha256", "weight_bytes_scope"]
    if cell.get("kind") == "dense":
        fields += ["rows", "cols", "cut_axis", "shape_tag"]
    else:
        fields += ["hidden", "inter", "experts", "top_k", "routing_sha256", "routing_weights_sha256"]
    if any(cell.get(k) is None or cell.get(k) != baseline.get(k) for k in fields):
        raise ValueError("T8 comparison requires equal actual serialized bytes, geometry, seeds and routing")
    if baseline.get("format") != "T8" or baseline.get("kind") != cell.get("kind"):
        raise ValueError("T8 comparison has a different format or component")
    if any(baseline.get(field) is not True for field in ("serving_owner", "eager_ok", "graph_equal")):
        raise ValueError("T8 comparison requires actual successful serving and graph outcomes")
    result = {}
    for execution in ("eager", "graph"):
        numerator = cell["timing"][execution]["median_ms"]
        denominator = baseline["timing"][execution]["median_ms"]
        samples = baseline["timing"][execution].get("samples_ms")
        if not samples or any(not math.isfinite(x) or x < 0 for x in samples):
            raise ValueError("T8 comparison has no valid ordered raw timing samples")
        if statistics.median(samples) != denominator:
            raise ValueError("T8 timing median disagrees with its actual raw samples")
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError("T8 comparison has invalid raw timing")
        result[execution] = dict(ratio=numerator / denominator, threshold=1.5,
                                pass_kill=numerator / denominator <= 1.5)
    return result


def prepare_comparison_cell(row):
    kind = row["kind"]
    if kind == "dense":
        row["weight_bytes_scope"] = ["weight"]
        row["case_id"] = f"dense:{row['shape_tag']}:{row['rows']}x{row['cols']}"
    else:
        group = "gate_up" if kind in ("mode0", "mode1") else ("down" if kind == "mode2" else "chain")
        parts = ["gate", "up"] if group == "gate_up" else (["down"] if group == "down" else list(ROLE_NAMES))
        row["case_id"] = "routed:" + group
        row["weight_bytes_scope"] = parts
        row["wire_bytes"] = sum(row["projection_serialized_bytes"][p] for p in parts)
    row["serialized_scope"] = "Full source containers before rank-local TP cuts"


def load_t8_comparison(path):
    baseline = json.loads(Path(path).read_text())
    cells = baseline.get("cells")
    if not isinstance(cells, list):
        raise ValueError("T8 comparison input has no cell list")
    if cells:
        return baseline
    plans = baseline.get("byte_plan")
    if not isinstance(plans, list) or not plans:
        raise ValueError("T8 comparison has neither measured cells nor a classified byte plan")
    for plan in plans:
        if plan.get("exact_match") is not False:
            raise ValueError("An exact-match T8 plan needs actual measured cells")
        status = plan.get("status")
        if status == "unattainable_scalar_floor":
            if plan.get("t8_plane_lower_bound", 0) <= plan.get("target_bytes", 0):
                raise ValueError("The T8 scalar floor does not exceed its actual target budget")
        elif status == "no_exact_match_found":
            if not plan.get("measured"):
                raise ValueError("The unresolved T8 byte search has no actual observations")
        else:
            raise ValueError("An empty T8 cell list needs classified unmatched byte plans")
    return baseline


def attach_t8_comparison(row, baseline):
    plans = [p for p in baseline.get("byte_plan", []) if p.get("case_id") == row["case_id"]
             and p.get("t4_q256") == row["q256"]]
    if len(plans) != 1:
        row["comparison_error"] = "The T8 producer has no unique actual byte plan for this cell"
        return
    plan = plans[0]
    if plan.get("actual_t4_serialized_bytes") != row["wire_bytes"]:
        row["comparison_error"] = "The T8 byte plan has a different actual T4 budget"
        return
    if not plan.get("exact_match"):
        row["competitive_status"] = plan["status"]
        row["comparability_evidence"] = plan
        return
    matches = [b for b in baseline.get("cells", []) if b.get("case_id") == row["case_id"]
        and b.get("kind") == row["kind"] and b.get("m") == row["m"]
        and b.get("tp_size") == row["tp_size"] and b.get("tp_rank") == row["tp_rank"]
        and b.get("target_t4_q256") == row["q256"] and b.get("cut_axis") == row.get("cut_axis")]
    if len(matches) != 1:
        row["comparison_error"] = "The exact byte plan has no unique measured T8 row"
        return
    try:
        row["t8_comparison"] = compare_t8_cell(row, matches[0])
        row["competitive_status"] = "measured"
    except ValueError as exc:
        row["comparison_error"] = str(exc)


def run_gpu(args):
    import torch
    from tessera.fused_frame import pack_fused

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        raise ValueError("native T4 serving requires the actual sm_121a GPU")
    check_mem_guard()
    report = dict(schema=SCHEMA, mode=args.mode, **source_stamp(args), cells=[], skips=[],
                  requested_population=requested_keys(args), resources=native_resources())
    patterns = ["random"] if args.mode == "timing" else ["onehot", "random"]
    baseline = load_t8_comparison(args.compare_json) if args.compare_json else None
    for q in args.q256:
        blobs, frames = {}, {}
        if args.part in ("routed", "all"):
            for part in ROLE_NAMES:
                rows, cols = ((args.hidden, args.inter) if part == "down" else (args.inter, args.hidden))
                seeds = [0] if args.mode == "timing" else range(args.experts)
                templates = [encode_bytes(synthetic(rows, cols, args.seed + ROLE_SEEDS[part] + 1000 * e,
                                                      "cuda"), q, structure="routed_moe")[0]
                             for e in seeds]
                blobs[part] = templates * args.experts if args.mode == "timing" else templates
                frames[part] = [pack_fused([(ROLE_NAMES[part], rows, b)]) for b in blobs[part]]
        for size, rank in rank_cases(args):
            if args.part in ("all", "routed"):
                try:
                    for row in routed_cells(args, q, blobs, frames, size, rank):
                        row["kind"] = row["key"].split("/")[2]
                        row["format"] = "T4"
                        row["register_resources"] = measured_registers(report["resources"], q, row["kind"])
                        prepare_comparison_cell(row)
                        if baseline:
                            attach_t8_comparison(row, baseline)
                        report["cells"].append(row)
                        save_report(args.out, report)
                except Exception as exc:
                    report["skips"].append(dict(q256=q, tp_size=size, tp_rank=rank,
                        reason=f"routed failed: {type(exc).__name__}: {exc}"))
            if args.part in ("all", "dense"):
                for shape in args.dense_shapes:
                    tag, rows, cols = shape
                    try:
                        blob, _ = encode_bytes(synthetic(rows, cols, args.seed + rows, "cuda"), q)
                        for axis in dense_axes(args, shape, size):
                            for row in dense_cells(args, q, blob, shape, size, rank, axis):
                                row["format"], row["kind"] = "T4", "dense"
                                row["register_resources"] = measured_registers(report["resources"], q, "dense")
                                prepare_comparison_cell(row)
                                if baseline:
                                    attach_t8_comparison(row, baseline)
                                report["cells"].append(row)
                                save_report(args.out, report)
                    except Exception as exc:
                        report["skips"].append(dict(q256=q, shape=shape, tp_size=size, tp_rank=rank,
                            reason=f"dense geometry failed: {type(exc).__name__}: {exc}"))
        torch.cuda.empty_cache()
    report["population"] = dict(requested=len(report["requested_population"]),
        observed=len(report["cells"]), skips=len(report["skips"]))
    report["competitive_coverage"] = {"requested": bool(baseline),
        "measured": sum(r.get("competitive_status") == "measured" for r in report["cells"]),
        "unattainable": [r["key"] for r in report["cells"] if r.get("competitive_status") == "unattainable_scalar_floor"],
        "unresolved": [r["key"] for r in report["cells"] if r.get("competitive_status") == "no_exact_match_found"],
        "errors": [r["key"] for r in report["cells"] if r.get("comparison_error")]}
    return report


def mode_correctness(args):
    return run_gpu(args)


def mode_timing(args):
    return run_gpu(args)


def qualify_serving(*, q256=(896,), ms=(1, 16), hidden=512, inter=1280, experts=2,
                    top_k=2, dense_shapes=(("partial", 64, 512), ("index", 128, 512)),
                    tp_cuts=True, out="/tmp/t4-serving-qualification.json"):
    """The one finite numerical entry point for pytest/manual serving wrappers."""
    args = build_parser().parse_args(["--mode", "correctness", "--out", out])
    args.q256, args.ms = list(q256), list(ms)
    args.hidden, args.inter, args.experts, args.top_k = hidden, inter, experts, top_k
    args.dense_shapes, args.tp_cuts = list(dense_shapes), tp_cuts
    report = mode_correctness(args)
    report["complete"], report["failures"] = validate_report(report, args)
    save_report(out, report)
    return report


def mode_dry_run(args):
    import torch
    from tessera.unit_artifact import parse_unit_artifact, read_unit_artifact
    from tessera.serving.scheme import (parse_compact_blob_for_scheme, validate_tessera_scheme,
                                        validate_tessera_moe_scheme)
    from tessera.serving import nvfp4_route, nvfp4_moe_route
    from tessera import routed_fused_e2m1
    from tessera.fused_frame import pack_fused

    torch.set_num_threads(1)
    checked_geometry = validate_rank_local_geometry(args)
    cells = []
    for q in args.q256:
        # Only a SMALL fixture is encoded/read on CPU; requested production
        # geometry and TP cuts are parsed separately and never CUDA repacked.
        blob, source = encode_bytes(synthetic(32, 256, args.seed, "cpu"), q)
        unit = parse_unit_artifact(blob, "cpu").unit
        decoded = read_unit_artifact(blob, "cpu")
        frame = dense_frame(blob, 32)
        parse_compact_blob_for_scheme(frame, dense_scheme(frame, 32, 256, q),
                                      "cpu.small.dense", device="cpu")
        f = {p: [pack_fused([(ROLE_NAMES[p], 32, blob)])] for p in ROLE_NAMES}
        # Actual small bytes are read above; admission of production shapes
        # uses their declared geometry with placeholder byte COUNTS only,
        # never executes or claims a load/forward on those declarations.
        production = routed_scheme(f, args.hidden, args.inter, args.experts, q)
        validate_tessera_moe_scheme(production, "cpu.production.geometry")
        for tag, rows, cols in args.dense_shapes:
            validate_tessera_scheme(dense_scheme(blob, rows, cols, q), "cpu." + tag)
        cells.append(dict(byte_receipt(blob, 32, 256), q256=q, recipe_source=source,
            decoded_shape=list(decoded.shape), rates=list(map(int, unit.rates)),
            cuda_repack="not-called", serving_owner=False,
            parsed_routing=[dict(m=m, ids_shape=list(routing(m, args.experts, args.top_k,
                                                args.seed + m, "cpu")[0].shape)) for m in args.ms]))
    t8_read = None
    if args.mode == "t8-baseline":
        from experiments.t4_code.t4_t8_baseline import encode_t8, t8_bounds
        from tessera.serving import fp8_route, moe_route
        low, _high, _step = t8_bounds()
        t8_blob = encode_t8(synthetic(32, 256, args.seed, "cpu"), low, "dense")
        t8_frame = dense_frame(t8_blob, 32)
        t8_scheme = dense_scheme(t8_frame, 32, 256, low)
        t8_scheme.update(family="TESSERA_FP8", grid="E4M3", plane="CHANNEL")
        parse_compact_blob_for_scheme(t8_frame, t8_scheme, "cpu.small.t8", device="cpu")
        t8_read = {"wire_bytes": len(t8_frame), "q256": low,
                   "routes_imported": [fp8_route.__name__, moe_route.__name__], "cuda_repack": "not-called"}
    reads = []
    if args.data_manifest:
        public_sdk = Path(os.environ.get("PRISMABUILD_READER_HELPER_ROOT", "/mnt/shared/prismabuild-fleet/repo"))
        sys.path.insert(0, str(public_sdk / "src"))
        from prismabuild.client import read_data_manifest
        read_data_manifest(args.data_manifest)
    if args.source_spec:
        for spec in json.loads(Path(args.source_spec).read_text())["units"]:
            weight = read_weight(spec, small=True)
            reads.append(dict(spec=spec, shape=list(weight.shape), sha256=tensor_digest(weight)))
    if args.compare_json:
        load_t8_comparison(args.compare_json)
    return dict(schema=SCHEMA, mode="dry-run", target_mode=args.mode,
                **source_stamp(args), cells=cells, skips=[], input_reads=reads,
                requested_population=[f"q{q}" for q in args.q256],
                population=dict(requested=len(args.q256), observed=len(cells), skips=0),
                imported_routes=[nvfp4_route.__name__, nvfp4_moe_route.__name__, routed_fused_e2m1.__name__],
                t8_small_read=t8_read,
                gpu_exercised=False, cuda_only_repack="not-called",
                rank_local_geometry=checked_geometry,
                serving_population_pending=requested_keys(args))


def mode_research_fixture(args):
    import torch
    from tessera.unit_artifact import parse_unit_artifact, read_unit_artifact

    rows = []
    for q in args.q256:
        b, source = encode_bytes(synthetic(32, 256, args.seed, "cpu"), q, "research-window")
        parsed = parse_unit_artifact(b, "cpu")
        got = read_unit_artifact(b, "cpu")
        rows.append(dict(byte_receipt(b, 32, 256), q256=q, recipe_source=source,
                         decoded_shape=list(got.shape), rates=list(map(int, parsed.unit.rates))))
    return dict(schema=SCHEMA, mode="research-fixture", **source_stamp(args), cells=rows, skips=[],
                population=dict(requested=len(args.q256), observed=len(rows), skips=0),
                serving_owner=False, serving_evidence=False)


def safetensor_header(path):
    with open(path, "rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = stream.read(length)
    if len(header) != length:
        raise ValueError("short safetensors header")
    return json.loads(header), 8 + length


def default_source_keys():
    prefix = "model.language_model.layers."
    return [prefix + "3.self_attn.q_a_proj.weight"] + [
        prefix + "3.mlp.experts.0." + ROLE_NAMES[p] + ".weight" for p in ROLE_NAMES]


def prepare_inputs(args):
    """Bounded input range inventory; never loads/downloads the model."""
    root = Path(args.model_root)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    entries, units, headers = [], [], {}
    for key in args.source_keys or default_source_keys():
        path = root / index[key]
        if path not in headers:
            header, offset = safetensor_header(path)
            headers[path] = (header, offset)
            with path.open("rb") as fh:
                raw = fh.read(offset)
            entries.append(dict(path=str(path), offset=0, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        header, offset = headers[path]
        info = header[key]
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            observed_shape = list(handle.get_slice(key).get_shape())
        if observed_shape != info["shape"]:
            raise ValueError("safetensor slice shape disagrees with the byte range")
        start, end = info["data_offsets"]
        if len(info["shape"]) != 2:
            raise ValueError("quality read set requires real two-dimensional weight units")
        # Reading/hashing only these exact tensor ranges belongs to PB CPU,
        # not a control-seat analysis or a whole model materialization.
        with path.open("rb") as fh:
            fh.seek(offset + start)
            digest = hashlib.sha256()
            left = end - start
            while left:
                block = fh.read(min(left, 1 << 20))
                if not block:
                    raise ValueError("short model tensor range")
                digest.update(block)
                left -= len(block)
        entries.append(dict(path=str(path), offset=offset + start, bytes=end - start, sha256=digest.hexdigest()))
        units.append(dict(path=str(path), tensor=key, shape=info["shape"], dtype=info["dtype"],
                          offset=offset + start, bytes=end - start,
                          source_sha256=digest.hexdigest(), slice=None,
                          kind="dense" if ".experts." not in key else key.split(".")[-2]))
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=True)
    spec = directory / "source-spec.json"
    manifest = directory / "data-manifest.json"
    spec.write_text(json.dumps(dict(units=units), indent=1) + "\n")
    raw_spec = spec.read_bytes()
    entries.append(dict(path=str(spec), offset=0, bytes=len(raw_spec),
                        sha256=hashlib.sha256(raw_spec).hexdigest()))
    manifest.write_text(json.dumps(dict(schema="prismaquant.prismabuild.data_manifest.v1",
        produced_by={"entry_point": "experiments/t4_code/t4_fused_qualify.py", "source_head": head_stamp()},
        annotations={"objective": "bounded GLM full units for a raw weight-space screen"},
        mount_prefix=os.path.commonpath([str(root.resolve()), str(directory.resolve())]),
        entries=entries, entry_count=len(entries),
        total_bytes=sum(e["bytes"] for e in entries)), indent=1) + "\n")
    public_sdk = Path(os.environ.get("PRISMABUILD_READER_HELPER_ROOT", "/mnt/shared/prismabuild-fleet/repo"))
    sys.path.insert(0, str(public_sdk / "src"))
    from prismabuild.client import read_data_manifest
    read_data_manifest(manifest)
    return dict(schema=SCHEMA, mode="prepare-inputs", **source_stamp(args),
                source_spec=str(spec), data_manifest=str(manifest), units=units,
                full_model_materialized=False)


def read_weight(spec, *, small=False, staged=None):
    import torch
    from safetensors import safe_open

    if staged is not None and not small:
        raw = staged.read(spec["path"], spec["offset"])
        if len(raw) != spec["bytes"]:
            raise ValueError("staged tensor length differs from actual selected unit")
        dtype = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}[spec["dtype"]]
        weight = torch.frombuffer(raw, dtype=dtype).reshape(spec["shape"])
        if spec.get("slice"):
            (r0, r1), (c0, c1) = spec["slice"]
            weight = weight[r0:r1, c0:c1]
    else:
        # Crucially get_slice, NOT get_tensor followed by slicing. A dry run
        # consumes two rows and sixteen columns from EACH declared input.
        with safe_open(spec["path"], framework="pt", device="cpu") as fh:
            handle = fh.get_slice(spec["tensor"])
            if list(handle.get_shape()) != list(spec["shape"]):
                raise ValueError("actual source weight shape differs from selected input")
            if small:
                weight = handle[:2, :16]
            elif spec.get("slice"):
                (r0, r1), (c0, c1) = spec["slice"]
                weight = handle[r0:r1, c0:c1]
            else:
                weight = handle[:, :]
    return weight.float().contiguous()


def plane_bytes(q, rows, cols, body):
    from tessera.calculator import terminal_rate
    from tessera.manifest import BodyKind

    rate = terminal_rate(q * 2, rows, cols, with_scale_base=False, with_scale_refine=True,
                         with_diagonals=False, completion=0, cap=8 if body == "window" else 7,
                         arity=2, span=1 if body == "window" else 2,
                         window_bits=14 if body == "window" else 0,
                         code_bytes=1, with_forest=body == "tcq")
    value = rate * rows * cols / 8
    if value.denominator != 1:
        raise ValueError("wire accountant returned fractional serialized bytes")
    return int(value)


def budget_candidates(budget, rows, cols, base_bytes, base_q):
    """Cheap arithmetic search; encode only exact/nearest candidates, not a campaign."""
    overhead = base_bytes - plane_bytes(base_q, rows, cols, "tcq")
    prices = []
    for q in range(128, 897):
        try:
            price = overhead + plane_bytes(q, rows, cols, "tcq")
        except ValueError:
            continue
        if price <= budget:
            prices.append((q, price))
    exact = sorted((q for q, price in prices if price == budget), reverse=True)
    nearest = sorted(prices, key=lambda p: (p[1], p[0]), reverse=True)
    return list(dict.fromkeys(exact + [q for q, _ in nearest[:3]]))[:4]


def weight_error(weight, decoded):
    import torch

    difference = decoded.double() - weight.double()
    numerator = float((difference * difference).sum())
    denominator = float((weight.double() * weight.double()).sum())
    if denominator <= 0 or not all(math.isfinite(v) for v in (numerator, denominator)):
        raise ValueError("weight-space SSE requires finite nonzero source energy")
    return dict(squared_error=numerator, source_squared_norm=denominator,
                relative_sse=numerator / denominator,
                relative_l2=math.sqrt(numerator / denominator))


def mode_quality(args):
    import torch
    from tessera.unit_artifact import read_unit_artifact

    if not args.source_spec:
        raise ValueError("quality screen needs the bounded real GLM source specification")
    if args.quality_device == "cuda" and not args.data_manifest:
        raise ValueError("GPU quality input ranges must be declared in the PB data manifest")
    specs = json.loads(Path(args.source_spec).read_text())["units"]
    if not specs or not any(s["kind"] == "dense" for s in specs) or not all(
        any(s["kind"] == ROLE_NAMES[p] for s in specs) for p in ROLE_NAMES):
        raise ValueError("quality screen requires real dense and gate/up/down units")
    cells = []
    with contextlib.ExitStack() as stack:
        staged = None
        if args.data_manifest and not args.dry_run:
            sys.path.insert(0, str(ROOT / "experiments/t8r_speed"))
            from pb_staged_store import StagedInputs
            staged = StagedInputs(args.data_manifest)
            stack.callback(staged.close)
        for spec in specs:
            weight = read_weight(spec, staged=staged).to(args.quality_device)
            rows, cols = weight.shape
            base_tcq, _ = encode_bytes(weight, 128, "tcq")
            cache = {128: base_tcq}
            for q in args.q256:
                blob, source = encode_bytes(weight, q)
                budget = len(blob)
                candidates = budget_candidates(budget, rows, cols, len(base_tcq), 128)
                measured = []
                for qt in candidates:
                    if qt not in cache:
                        cache[qt] = encode_bytes(weight, qt, "tcq")[0]
                    tb = cache[qt]
                    # Verify the REAL bytes, including manifest/header/table,
                    # not an arithmetic price mistaken for actual serialization.
                    measured.append(dict(q256=qt, wire_bytes=len(tb),
                        sha256=hashlib.sha256(tb).hexdigest(),
                        error=weight_error(weight, read_unit_artifact(tb, args.quality_device))))
                feasible = [r for r in measured if r["wire_bytes"] <= budget]
                if not feasible:
                    selected = None
                else:
                    selected = max(feasible, key=lambda r: (r["wire_bytes"], r["q256"]))
                exact = selected is not None and selected["wire_bytes"] == budget
                cells.append(dict(tensor=spec["tensor"], provenance=spec, shape=[rows, cols],
                    q256=q, budget_bytes=budget, window=dict(wire_bytes=budget, recipe_source=source,
                        error=weight_error(weight, read_unit_artifact(blob, args.quality_device))),
                    tcq_span2=selected, searched_candidate_q256=candidates, measured_candidates=measured,
                    exact_byte_match=exact,
                    unmatched_slack_bytes=None if selected is None else budget - selected["wire_bytes"],
                    complete=exact, screen_only=True,
                    interpretation="common upper-byte budget; unequal artifacts are NOT matched",
                    claim="raw weight-space screen only, not PACT/G3/end-to-end quality"))
                save_report(args.out, dict(schema=SCHEMA, mode="quality-screen", cells=cells))
            del cache, weight
    return dict(schema=SCHEMA, mode="quality-screen", **source_stamp(args), cells=cells,
                requested_units=[s["tensor"] for s in specs], population=dict(units=len(specs),
                requested_cells=len(specs) * len(args.q256), observed=len(cells), skips=0), skips=[])


def validate_report(report, args):
    failures = []
    cells = report.get("cells", [])
    if report["mode"] == "t8-baseline":
        from experiments.t4_code.t4_t8_baseline import validate_baseline
        return validate_baseline(report, args)
    if not cells:
        failures.append("no requested cells were observed")
    if report.get("skips"):
        failures.append("mandatory cells were skipped or failed")
    if report["mode"] in ("correctness", "timing"):
        expected = set(report.get("requested_population", requested_keys(args)))
        observed = [c.get("key") for c in cells]
        if len(observed) != len(set(observed)) or set(observed) != expected:
            failures.append("requested population differs from observed cells")
        for c in cells:
            if c.get("serving_owner") is not True or c.get("eager_ok") is not True:
                failures.append(f"mandatory serving or eager check failed: {c.get('key')}")
            if c.get("graph_equal") is not True:
                failures.append(f"mandatory graph replay missing or mismatched: {c.get('key')}")
            if not c.get("register_resources"):
                failures.append(f"D41 matched template register resources missing: {c.get('key')}")
            if c.get("numeric", {}).get("ok") is not True:
                failures.append(f"mandatory independent numeric comparison failed: {c.get('key')}")
            if report["mode"] == "timing" and not c.get("timing", {}).get("graph", {}).get("samples_ms"):
                failures.append(f"mandatory graph timing missing: {c.get('key')}")
            if c.get("comparison_error"):
                failures.append(f"T8 paired comparison failed: {c.get('key')}: {c['comparison_error']}")
            if c.get("competitive_status") == "measured" and not all(
                    v.get("pass_kill") for v in c["t8_comparison"].values()):
                failures.append(f"equal-byte T8 kill comparison failed: {c.get('key')}")
        if not report.get("resources", {}).get("functions"):
            failures.append("D41 actual register extraction missing")
    elif report["mode"] == "quality-screen":
        for c in cells:
            if c.get("complete") is not True:
                failures.append(f"equal-byte screen incomplete: {c.get('tensor')} q{c.get('q256')}; slack disclosed")
        if len(cells) != report.get("population", {}).get("requested_cells"):
            failures.append("quality population incomplete")
    elif report["mode"] in ("dry-run", "research-fixture"):
        if len(cells) != len(args.q256):
            failures.append("CPU byte-read population incomplete")
    return not failures, failures


def save_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + "\n")


def parse_shape(value):
    parts = value.split(":")
    geometry = parts[-1].split("x")
    if len(geometry) != 2:
        raise argparse.ArgumentTypeError("dense shape is name:ROWSxCOLS")
    rows, cols = map(int, geometry)
    return (parts[0] if len(parts) == 2 else value, rows, cols)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True,
        choices=("dry-run", "correctness", "timing", "t8-baseline", "quality-screen", "research-fixture", "prepare-inputs"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--q256", nargs="+", type=int, default=list(PURE_Q256))
    ap.add_argument("--ms", nargs="+", type=int, default=list(ALL_M))
    ap.add_argument("--part", choices=("all", "dense", "routed"), default="all")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--inter", type=int, default=1280)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--dense-shapes", type=parse_shape, nargs="+", default=list(DENSE_SHAPES))
    ap.add_argument("--tp-cuts", action="store_true")
    ap.add_argument("--dense-shard-axes", nargs="+", choices=("row", "column", "replicated"), default=["row"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--compare-json")
    ap.add_argument("--plan-only", action="store_true", help="Write actual T4/T8 byte feasibility before timing.")
    ap.add_argument("--source-spec")
    ap.add_argument("--data-manifest")
    ap.add_argument("--quality-device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--model-root", default="/mnt/shared/models/GLM-5.3-Flash-BF16")
    ap.add_argument("--source-keys", nargs="+")
    ap.add_argument("--start-memory-gib", type=float, default=9.0)
    ap.add_argument("--seconds", type=int, default=1700)
    ap.add_argument("--guarded-child", action="store_true", help=argparse.SUPPRESS)
    return ap


def validate_args(args):
    if not args.q256 or any(q not in PURE_Q256 for q in args.q256):
        raise ValueError("serving qualification requires pure paired R1..8 q128..1024")
    if len(args.q256) != len(set(args.q256)) or len(args.ms) != len(set(args.ms)):
        raise ValueError("duplicate requested cells")
    if not args.ms or any(m <= 0 for m in args.ms):
        raise ValueError("M must be positive")
    if min(args.experts, args.top_k, args.iters) <= 0 or args.warmup < 0:
        raise ValueError("invalid routing or event population")
    if args.hidden < 256 or args.hidden % 256 or args.inter < 256 or args.inter % 128:
        raise ValueError("routed geometry requires H multiple 256, I multiple 128 and K at least 256")
    if args.tp_cuts and (args.inter % 256 or args.inter // 2 < 256):
        raise ValueError("TP2 route requires each local intermediate to be multiple 128 and at least 256")
    for tag, rows, cols in args.dense_shapes:
        if rows <= 0 or cols <= 0:
            raise ValueError(f"invalid dense shape {tag}")
    if args.compare_json and args.mode != "timing":
        raise ValueError("T8 timing rows compare with T4 timing, not a different source fixture")
    if args.plan_only and args.mode != "t8-baseline":
        raise ValueError("A byte-only plan belongs to the T8 baseline producer")
    if not 1 <= args.seconds <= 1700:
        raise ValueError("finite payload must leave cleanup within the 1800 second quantum")


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.mode == "prepare-inputs":
        report = prepare_inputs(args)
        save_report(Path(args.out) / "preparation.json", report)
        return 0
    check_mem_guard()
    report = None
    try:
        if args.dry_run or args.mode == "dry-run":
            report = mode_dry_run(args)
        elif args.mode == "correctness":
            report = mode_correctness(args)
        elif args.mode == "timing":
            report = mode_timing(args)
        elif args.mode == "t8-baseline":
            from experiments.t4_code.t4_t8_baseline import produce
            report = produce(args)
        elif args.mode == "quality-screen":
            report = mode_quality(args)
        else:
            report = mode_research_fixture(args)
        complete, failures = validate_report(report, args)
        report.update(complete=complete, failures=failures)
        save_report(args.out, report)
        print(json.dumps(dict(mode=report["mode"], complete=complete,
                              population=report.get("population"), failures=failures)))
        return 0 if complete else 1
    except Exception as exc:
        if report is None:
            report = dict(schema=SCHEMA, mode=args.mode, cells=[], complete=False)
        report.update(complete=False, fatal_error=f"{type(exc).__name__}: {exc}")
        save_report(args.out, report)
        raise


def guarded_cli():
    args = build_parser().parse_args()
    if args.guarded_child:
        return main()
    # Existing owner supplies the sole process group lifecycle, timeout and
    # TERM/KILL cleanup. No new dispatcher, polling client, or process scanner.
    sys.path.insert(0, str(ROOT / "experiments/graph_attest_702"))
    from managed_window import Envelope

    floor = 2.0 if args.dry_run or args.mode in ("dry-run", "research-fixture", "prepare-inputs") else args.start_memory_gib
    start = check_mem_guard(floor)
    envelope = Envelope(time.time() + args.seconds + 20, cleanup_seconds=20)
    samples = []
    last = 0.0
    def tick():
        nonlocal last
        now = time.monotonic()
        if now - last < 1:
            return
        last = now
        samples.append(dict(unix=time.time(), **check_mem_guard()))
    result = None
    try:
        result = envelope.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--guarded-child"],
                              check=False, tick=tick, stdout=sys.stdout, limit=args.seconds)
        return result.returncode
    finally:
        path = (Path(args.out) / "memory-guard.json" if args.mode == "prepare-inputs"
                else Path(args.out).with_suffix(".memory.json"))
        save_report(path, dict(start=start, samples=samples, terminations=envelope.terminations,
                              sample_seconds=1, abort_below_gib=2, term_grace_seconds=10,
                              returncode=None if result is None else result.returncode))


if __name__ == "__main__":
    raise SystemExit(guarded_cli())
