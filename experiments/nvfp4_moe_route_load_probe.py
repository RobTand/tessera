#!/usr/bin/env python3
"""Load and execute a Tessera NVFP4 (E2M1x2, W4A4) routed-MoE stack on the pinned runtime.

WHAT THIS IS EVIDENCE FOR, AND WHAT IT IS NOT.  It is the LOAD-AND-EXECUTE
contract of ``tessera.serving.nvfp4_moe_route`` (tessera#492), measured rather
than asserted: vLLM's real ``RoutedExperts`` is constructed with a Tessera
checkpoint's ``quantization_config`` on the NVFP4 family, its real
``load_weights`` is handed per-expert E2M1x2 wires AND the static A-side
``input_global_scale`` beside each one, the route's own loader decodes them
into the modelopt parameter set, the runtime's own NVFP4 oracle picks the
kernel for a config carrying GLM-5.3-Flash's ``swiglu_limit`` (the clamp that
rules ``--moe-backend triton`` out), and that kernel multiplies them.

Three references, each answering one question:

* ``vs_dequantised_reference`` -- torch over the DEQUANTISED expert tiles the
  kernel holds, activations in fp32.  Disagreement here is the whole W4A4
  activation contract plus the plumbing.
* ``vs_w4a4_emulated`` -- the same, with the A side quantised in torch the
  way ``scaled_fp4_quant`` quantises it (group-16 e4m3 block scale under the
  ONE per-layer global the FlashInfer backends collapse the per-expert scales
  to, E2M1 round-to-nearest).  The residue against THIS is what is left once
  the arithmetic both sides agreed to run is accounted for; without it the
  first number has nothing to be read against.
* ``vs_bf16_source`` -- the quantisation error, reported as such.

THE CLAMP LEG.  GLM's ``swiglu_limit=10.0`` is why the stub cannot serve on the
triton backend; a kernel picked for it must APPLY it.  The second positive leg
drives activations 40x hotter (scales recalibrated at that amplitude) so the
clamp bites, and reports the kernel against the clamped and the unclamped
reference.  A kernel nearer the unclamped one is not applying the clamp.

It is NOT a served census and NOT a KL: no model is loaded, no engine is
started, and the weights are random.  A ``routed_moe`` cell for TESSERA_E2M1_K2
in ``runtime_contract.json`` needs a served artifact and is not earned here.

THE NEGATIVE LEGS MATTER AS MUCH AS THE POSITIVE ONE.  A route that decodes
correctly but accepts bytes it should refuse is a wrong tensor waiting for a
different checkpoint: one payload byte flipped, a group's ``wire_stride``
understated and overstated, an expert count the sidecar does not declare, a
rung the reader does not read, one expert projection with no A-side scale,
one expert missing half of w13, and a stock modelopt tensor handed to a stack
declared as wires.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

EXPERTS, HIDDEN, INTER, TOPK = 4, 512, 256, 2
Q256 = 896                      # the one E2M1_K2 rung the reader publishes
SWIGLU_LIMIT = 10.0             # GLM-5.3-Flash text_config.swiglu_limit
CLAMP_X_SCALE = 40.0            # the hot leg: gate values well past the clamp
FP4_CAPACITY = 448.0 * 6.0      # e4m3 max x e2m1 max: modelopt's amax denominator
LAYER = "model.layers.0.mlp.experts"
EXPORT_LAYER = "model.language_model.layers.1.mlp.experts"
PROJECTIONS = (("gate_proj", INTER, HIDDEN, "w13"),
               ("up_proj", INTER, HIDDEN, "w13"),
               ("down_proj", HIDDEN, INTER, "w2"))


def _encode(weight, name, q256):
    """One E2M1x2 unit on ``weight``'s device: the container and the stock tile."""
    from tessera.alphabet import E2M1_GRID, tuple_grid
    from tessera.export import DEFAULT_CODE, encode_linear_planes
    from tessera.fused import pack_fused
    from tessera.stock import materialize_stock

    exported, unit, forests = encode_linear_planes(
        weight.contiguous(), grid=tuple_grid(E2M1_GRID, 2), q256=q256, name=name, verify=False)
    blob = pack_fused([(name, weight.shape[0], exported.blob)])
    return blob, materialize_stock(unit, forests, DEFAULT_CODE)


def calibration_input(device, x_scale):
    """The activations the static scales are read from.  A different seed
    from the probe input: a static scale is a checkpoint fact, not a fit."""
    generator = torch.Generator(device=device).manual_seed(7)
    return (torch.randn(96, HIDDEN, generator=generator, device=device, dtype=torch.float32)
            * x_scale)


def static_scales(stock, device, x_scale):
    """``input_global_scale`` per (expert, projection): capacity / amax, modelopt's
    ``1 / input_scale``.  Gate and up read the same activation; down reads the
    intermediate the DEQUANTISED gate/up produce on it (clamped as the kernel
    clamps), so the scale is the one a real calibration would write."""
    from tessera.stock import stock_dequant

    x = calibration_input(device, x_scale)
    scales = {}
    for expert in range(EXPERTS):
        amax = float(x.abs().amax())
        scales[(expert, "gate_proj")] = FP4_CAPACITY / amax
        scales[(expert, "up_proj")] = FP4_CAPACITY / amax
        w1 = stock_dequant(stock[(expert, "gate_proj")]).to(device, torch.float32)
        w3 = stock_dequant(stock[(expert, "up_proj")]).to(device, torch.float32)
        h = _swiglu(x @ w1.t(), x @ w3.t(), SWIGLU_LIMIT)
        scales[(expert, "down_proj")] = FP4_CAPACITY / float(h.abs().amax())
    return scales


def _swiglu(gate, up, limit):
    """vLLM's ``SiluAndMulWithClamp`` (alpha 1, beta 0); ``limit=None`` is plain SiLU."""
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return torch.nn.functional.silu(gate) * up


def build_stack(device, seed=0, q256=Q256, x_scale=1.0):
    """E experts of (gate, up, down): the wires and the scales under the names
    a checkpoint carries, the scheme, the BF16 source, the stock tiles."""
    from tessera.serving.scheme import TESSERA_NVFP4

    generator = torch.Generator(device="cpu").manual_seed(seed)
    wires, source, stock = {}, {}, {}
    strides = {"w13": 0, "w2": 0}
    for expert in range(EXPERTS):
        for name, rows, cols, group in PROJECTIONS:
            weight = (torch.randn(rows, cols, generator=generator) * 0.02).to(device, torch.float32)
            blob, tensors = _encode(weight, name, q256)
            wires[f"{expert}.{name}.wire"] = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
            source[(expert, name)] = weight
            stock[(expert, name)] = tensors
            strides[group] = max(strides[group], len(blob))
    scales = static_scales(stock, device, x_scale)
    for (expert, name), value in scales.items():
        wires[f"{expert}.{name}.input_global_scale"] = torch.tensor([value], dtype=torch.float32)
    scheme = {
        "family": TESSERA_NVFP4, "structure": "routed_moe", "grid": "E2M1x2", "body": "TCQ",
        "plane": "LUT", "experts": EXPERTS,
        "groups": {
            "w13": {"rows": 2 * INTER, "columns": HIDDEN, "q256": q256,
                    "wire_stride": strides["w13"],
                    "roles": [["gate_proj", INTER], ["up_proj", INTER]]},
            "w2": {"rows": HIDDEN, "columns": INTER, "q256": q256,
                   "wire_stride": strides["w2"], "roles": [["down_proj", HIDDEN]]}},
    }
    return wires, scheme, source, stock, scales


def quantization_config(scheme, layer_name=LAYER):
    return {"quant_method": "tessera", "format": "tessera",
            "config_groups": {"tessera_experts": {"format": "TESSERA", "targets": [layer_name],
                                                  "scheme": scheme}},
            "ignore": []}


def build_stack_from_export(device, workdir, seed=0, q256=Q256, x_scale=1.0):
    """The same stack, with the wires, the scales and the scheme WRITTEN BY THE
    EXPORTER (``--input-scales`` beside ``--plan-json``).  Everything is pinned
    but the producer.  ``stock`` is read back out of the exported container, so
    the tile is compared against what the artifact holds."""
    import importlib.util

    from safetensors import safe_open
    from safetensors.torch import save_file
    from tessera.export import DEFAULT_CODE
    from tessera.fused import parse_fused
    from tessera.stock import materialize_stock
    from tessera.unit_artifact import parse_unit_artifact

    workdir = Path(workdir)
    src, out = workdir / "src", workdir / "out"
    src.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source, tensors, cpu_stock = {}, {}, {}
    for expert in range(EXPERTS):
        for name, rows, cols, _group in PROJECTIONS:
            weight = (torch.randn(rows, cols, generator=generator) * 0.02)
            tensors[f"{EXPORT_LAYER}.{expert}.{name}.weight"] = weight
            source[(expert, name)] = weight.to(device, torch.float32)
    tensors[f"{EXPORT_LAYER.rsplit('.', 1)[0]}.gate.weight"] = torch.zeros(EXPERTS, HIDDEN)
    save_file({k: v.contiguous() for k, v in tensors.items()},
              str(src / "model.safetensors"), metadata={"format": "pt"})
    (src / "config.json").write_text(json.dumps({
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {"hidden_size": HIDDEN, "moe_intermediate_size": INTER,
                        "num_hidden_layers": 2, "n_routed_experts": EXPERTS,
                        "swiglu_limit": SWIGLU_LIMIT}}))
    plan = workdir / "plan.json"
    plan.write_text(json.dumps({EXPORT_LAYER: {"grid": "E2M1x2", "q256": q256}}))
    # The A-side scales the exporter is handed: read off the SAME stock tiles
    # the A side reads its own from, so the pair differs only in the producer.
    # (The tiles are the encoder's, which is deterministic for a given input
    # and device; the B side re-derives them from the artifact below and the
    # comparison in positive_leg is against those.)
    for expert in range(EXPERTS):
        for name, _rows, _cols, _group in PROJECTIONS:
            cpu_stock[(expert, name)] = _encode(source[(expert, name)], name, q256)[1]
    scales = static_scales(cpu_stock, device, x_scale)
    save_file({f"{EXPORT_LAYER}.{expert}.{name}.input_global_scale":
               torch.tensor([value], dtype=torch.float32)
               for (expert, name), value in scales.items()},
              str(workdir / "input_scales.safetensors"), metadata={"format": "pt"})

    spec = importlib.util.spec_from_file_location(
        "export_tessera_serving", Path(__file__).resolve().parent / "export_tessera_serving.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    argv = sys.argv
    sys.argv = ["export", str(src), str(out), "--grid", "E2M1x2", "--q256", str(q256),
                "--plan-json", str(plan), "--device", str(device),
                "--input-scales", str(workdir / "input_scales.safetensors"),
                # No routed_moe cell for TESSERA_E2M1_K2 exists yet: the exporter
                # refuses without this and stamps the override with it.
                "--allow-unserveable"]
    try:
        exporter.main()
    finally:
        sys.argv = argv

    written = json.loads((out / "config.json").read_text())["quantization_config"]
    groups = [g for g in written["config_groups"].values() if g["targets"] == [EXPORT_LAYER]]
    if len(groups) != 1:
        raise RuntimeError(f"the exporter declared {len(groups)} group(s) for {EXPORT_LAYER}")
    scheme = groups[0]["scheme"]
    manifest = json.loads((out / "tessera_serving_manifest.json").read_text())

    wires, stock, read_scales = {}, {}, {}
    with safe_open(str(out / "model.safetensors"), framework="pt") as handle:
        for expert in range(EXPERTS):
            for name, _rows, _cols, _group in PROJECTIONS:
                blob = handle.get_tensor(f"{EXPORT_LAYER}.{expert}.{name}.wire")
                wires[f"{expert}.{name}.wire"] = blob.clone()
                scale = handle.get_tensor(f"{EXPORT_LAYER}.{expert}.{name}.input_global_scale")
                wires[f"{expert}.{name}.input_global_scale"] = scale.clone()
                read_scales[(expert, name)] = float(scale.reshape(-1)[0])
                member, = parse_fused(bytes(blob.tolist()))
                parsed = parse_unit_artifact(member.blob, device=str(device))
                stock[(expert, name)] = materialize_stock(parsed.unit, parsed.forests, DEFAULT_CODE)
    # The exporter writes each scale as a float32 tensor, so compare at float32:
    # the Python floats in ``scales`` carry bits the wire cannot.
    given = {key: float(torch.tensor(value, dtype=torch.float32)) for key, value in scales.items()}
    if read_scales != given:
        raise RuntimeError("the exporter wrote different input_global_scale values than it was given")
    return wires, scheme, source, stock, read_scales, manifest


def vllm_config():
    from vllm.config import VllmConfig, set_current_vllm_config
    return set_current_vllm_config(VllmConfig())


def init_parallel():
    from vllm.distributed import (ensure_model_parallel_initialized,
                                  init_distributed_environment)
    init_distributed_environment(
        world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:52733",
        local_rank=0, backend="gloo")
    ensure_model_parallel_initialized(1, 1)
    from vllm.v1.worker.workspace import init_workspace_manager
    init_workspace_manager(torch.device("cuda"))


def build_layer(scheme, device, layer_name=LAYER, swiglu_limit=SWIGLU_LIMIT):
    from vllm.model_executor.layers.fused_moe import RoutedExperts, RoutingMethodType
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig, FusedMoEParallelConfig, MoEActivation)
    from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
    from tessera.serving.config import TesseraConfig

    parallel = FusedMoEParallelConfig(
        tp_size=1, tp_rank=0, pcp_size=1, pcp_rank=0, dp_size=1, dp_rank=0,
        ep_size=1, ep_rank=0, sp_size=1, use_ep=False,
        all2all_backend="naive", enable_eplb=False)
    moe = FusedMoEConfig(
        num_experts=EXPERTS, experts_per_token=TOPK, hidden_dim=HIDDEN,
        intermediate_size=INTER, num_local_experts=EXPERTS, num_logical_experts=EXPERTS,
        activation=MoEActivation.SILU, device=device,
        routing_method=RoutingMethodType.TopK, moe_parallel_config=parallel,
        in_dtype=torch.bfloat16, intermediate_size_per_partition=INTER, moe_backend="auto",
        swiglu_limit=swiglu_limit)
    manager = ExpertMapManager(
        max_num_batched_tokens=64, top_k=TOPK, global_num_experts=EXPERTS,
        num_redundant_experts=0, num_expert_group=None, moe_parallel_config=parallel,
        placement_strategy="linear", enable_eplb=False)
    config = TesseraConfig.from_config(quantization_config(scheme, layer_name))
    layer = RoutedExperts(layer_name=layer_name, params_dtype=torch.bfloat16, moe_config=moe,
                          quant_config=config, expert_map_manager=manager,
                          swiglu_limit=swiglu_limit)
    return layer.to(device)


def load(layer, wires):
    """Through the runtime's OWN loader, under the names a checkpoint carries."""
    return sorted(layer.load_weights([(k, v) for k, v in sorted(wires.items())]))


def moved_reference_tiles(stock):
    """What the loader holds: gate and up moved onto ONE global per expert
    (``stock.share_global``, the LUT-plane move the route makes), down as is.
    The tile bytes and the multiplier the kernel is handed are read from here."""
    from tessera.stock import share_global

    tiles = {}
    for expert in range(EXPERTS):
        moved, divisor = share_global({"gate_proj": stock[(expert, "gate_proj")],
                                       "up_proj": stock[(expert, "up_proj")]})
        tiles[(expert, "gate_proj")] = moved["gate_proj"]
        tiles[(expert, "up_proj")] = moved["up_proj"]
        tiles[(expert, "down_proj")] = stock[(expert, "down_proj")]
        tiles[(expert, "w13_global")] = 1.0 / float(divisor)
        tiles[(expert, "w2_global")] = 1.0 / float(
            stock[(expert, "down_proj")]["weight_global_scale"].reshape(-1)[0])
    return tiles


def tile_check(layer, tiles):
    """Before finalize: the SERVED scale plane IS materialize_stock's tile
    scale, expert by expert, and the global handed on is the multiplier.

    Since the materialising expert-reader fell back to zero-size stock anchors
    (tessera#506-era route retirement), the decoded weights live on the native
    A4 stacks' scale planes, not on the stock names: the loader's served LUT
    expanded at its nibble indices reproduces the per-element E4M3 scale plane
    the stock tile carries, so that is the byte-for-byte identity checked
    here.  (``weight_packed`` nibble bytes are verified separately by the
    probe's A/B legs: w13 into ``a_side_after_finalize``, plus the digests the
    negative legs cover.)
    """
    from tessera.stock import NVFP4_KEYS, stock_dequant

    result = {"identical": True, "globals_match": True}
    stacks = {"gate_proj": layer.tessera_a4_gate_stack,
              "up_proj": layer.tessera_a4_up_stack,
              "down_proj": layer.tessera_a4_down_stack}
    for name, stack in stacks.items():
        result[f"stack_{name}_present"] = stack is not None
    for expert in range(EXPERTS):
        for role, stack in stacks.items():
            if stack is None:
                result["identical"] = False
                continue
            # The served plane, [rows, groups] float8 elementwise.
            rows, cols = stack.rows, stack.cols
            packed = stack.nibbles[expert].view(torch.uint8).cpu().to(torch.int64)
            groups = cols // 16
            index = torch.empty((groups, rows), dtype=torch.int64)
            index[:, 0::2] = (packed >> 4).reshape(groups, rows // 2)
            index[:, 1::2] = (packed & 0xF).reshape(groups, rows // 2)
            served = stack.lut_bytes[expert].view(torch.uint8).cpu()[index].t().contiguous()
            want = tiles[(expert, role)]["weight_scale"].view(torch.uint8).cpu()
            result["identical"] &= bool(torch.equal(served.view(torch.uint8),
                                                    want.view(torch.uint8)))
        g13 = layer.tessera_a4_gate_stack.globals[expert].item() \
            if layer.tessera_a4_gate_stack is not None else None
        g2 = layer.tessera_a4_down_stack.globals[expert].item() \
            if layer.tessera_a4_down_stack is not None else None
        if g13 is None or g2 is None:
            result["globals_match"] = False
            continue
        result["globals_match"] &= (
            abs(g13 - tiles[(expert, "w13_global")]) <= 1e-6 * abs(g13)
            and abs(g2 - tiles[(expert, "w2_global")]) <= 1e-6 * abs(g2))
    return result


def dequantised(tiles, device):
    from tessera.stock import stock_dequant

    return {key: stock_dequant(value).to(device, torch.float32)
            for key, value in tiles.items() if isinstance(value, dict)}


def moe_reference(weights, x, topk_weights, topk_ids, device, limit, quantize=None):
    """The MoE in torch.  ``quantize`` is the A-side model: identity, or the
    kernel's fp4 quant-dequant under the per-group global it was handed."""
    out = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(EXPERTS):
        mask = (topk_ids == expert)
        if not mask.any():
            continue
        rows, slots = mask.nonzero(as_tuple=True)
        xs = x[rows].float()
        if quantize is not None:
            xs = quantize(xs, "w13")
        w1, w3, w2 = (weights[(expert, "gate_proj")], weights[(expert, "up_proj")],
                      weights[(expert, "down_proj")])
        h = _swiglu(xs @ w1.t(), xs @ w3.t(), limit)
        if quantize is not None:
            h = quantize(h, "w2")
        y = h @ w2.t()
        out.index_add_(0, rows, y * topk_weights[rows, slots].unsqueeze(1).float())
    return out


E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def fp4_quant_dequant(t, global_scale):
    """``scaled_fp4_quant`` in torch: per 16 values a block scale
    ``e4m3(amax/6 * gs)``, the values ``e2m1(x * gs / sf)`` saturated at 6,
    dequantised ``q * sf / gs``.  Round-to-nearest on the E2M1 grid with ties
    up; the hardware's tie rule differs on a measure-zero set."""
    levels = E2M1_LEVELS.to(t.device)
    mid = (levels[:-1] + levels[1:]) / 2
    m, n = t.shape
    blocks = t.float().view(m, n // 16, 16)
    vmax = blocks.abs().amax(dim=-1, keepdim=True)
    sf = (vmax / 6.0 * global_scale).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    scale = torch.where(sf > 0, global_scale / sf, torch.zeros_like(sf))
    scaled = blocks * scale
    q = levels[torch.bucketize(scaled.abs(), mid, right=True)] * torch.sign(scaled)
    return (q * sf / global_scale).view(m, n)


def _err(got, want):
    got, want = got.float(), want.float()
    denom = want.norm().item() or 1.0
    return {"rel_l2": (got - want).norm().item() / denom,
            "max_abs": (got - want).abs().max().item(),
            "want_absmax": want.abs().max().item()}


def positive_leg(device, tokens, q256, build=None, layer_name=LAYER, x_scale=1.0):
    """Load and execute one stack.  ``build`` says where its bytes came from."""
    from tessera.serving.telemetry import read_route

    build = build or (lambda: build_stack(device, q256=q256, x_scale=x_scale))
    built = build()
    wires, scheme, source, stock, scales = built[:5]
    manifest = built[5] if len(built) > 5 else None
    tiles = moved_reference_tiles(stock)
    layer = build_layer(scheme, device, layer_name)
    method = layer.quant_method
    record = {"method": type(method).__name__,
              "backend": str(getattr(getattr(method, "nvfp4_backend", None), "value", None)),
              "experts_cls": getattr(getattr(method, "experts_cls", None), "__name__", None),
              "x_scale": x_scale, "swiglu_limit": SWIGLU_LIMIT}
    record["params_before_load"] = sorted(n for n, _ in layer.named_parameters())
    loaded = load(layer, wires)
    record["loaded_param_names"] = sorted(set(loaded))
    record["load_calls"] = len(loaded)
    # BEFORE finalize: the tile is the stock pair and the globals are multipliers.
    # (The FlashInfer backends reorder [w1,w3] -> [w3,w1] and swizzle the block
    # scales at finalize, so this is the last point the bytes are comparable.)
    record["tile_is_materialize_stock_byte_for_byte"] = tile_check(layer, tiles)
    record["input_global_scale_loaded"] = {
        "w13": layer.w13_input_global_scale.data.tolist(),
        "w2": layer.w2_input_global_scale.data.tolist()}
    method.process_weights_after_loading(layer)
    record["params_after_load"] = sorted(n for n, _ in layer.named_parameters())
    for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale",
                 "w13_weight_scale_2", "w2_weight_scale_2", "w13_input_scale", "w2_input_scale"):
        tensor = getattr(layer, name)
        record[f"{name}_shape"] = list(tensor.shape)
        record[f"{name}_dtype"] = str(tensor.dtype)
    # What the kernel was handed for the A side: the FlashInfer backends collapse
    # the per-expert static scales to ONE per group (the max input_scale, i.e.
    # the smallest capacity/amax).  Read it from the runtime, never re-derived.
    a13 = layer.w13_input_scale.detach().float().reshape(-1)
    a2 = layer.w2_input_scale.detach().float().reshape(-1)
    record["a_side_after_finalize"] = {
        "w13_input_scale_unique": sorted(set(a13.tolist())),
        "w2_input_scale_unique": sorted(set(a2.tolist())),
        "w13_input_global_scale_given": sorted({v for (e, n), v in scales.items() if n != "down_proj"}),
        "w2_input_global_scale_given": sorted({v for (e, n), v in scales.items() if n == "down_proj"}),
        "collapsed_to_one_per_group": bool(a13.unique().numel() == 1 and a2.unique().numel() == 1)}
    a1_gscale = 1.0 / float(a13.max())
    a2_gscale = 1.0 / float(a2.max())
    record["a_side_after_finalize"]["a1_gscale"] = a1_gscale
    record["a_side_after_finalize"]["a2_gscale"] = a2_gscale
    record["tessera"] = {"decoder": getattr(layer, "tessera_decoder", None),
                         "backend": getattr(layer, "tessera_backend", None),
                         "activation_contract": getattr(layer, "tessera_activation_contract", None)}

    generator = torch.Generator(device=device).manual_seed(11)
    x = (torch.randn(tokens, HIDDEN, generator=generator, device=device, dtype=torch.float32)
         * x_scale).to(torch.bfloat16)
    logits = torch.randn(tokens, EXPERTS, generator=generator, device=device, dtype=torch.float32)
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), TOPK, dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(torch.float32)
    ids = ids.to(torch.int32)

    got = method.apply(layer, x, weights, ids, None, None)
    torch.cuda.synchronize()
    record["output_shape"] = list(got.shape)
    record["output_dtype"] = str(got.dtype)
    deq = dequantised(tiles, device)

    def quantize(t, group):
        return fp4_quant_dequant(t, a1_gscale if group == "w13" else a2_gscale)

    record["vs_w4a4_emulated"] = _err(
        got, moe_reference(deq, x, weights, ids, device, SWIGLU_LIMIT, quantize))
    record["vs_dequantised_reference"] = _err(
        got, moe_reference(deq, x, weights, ids, device, SWIGLU_LIMIT))
    record["vs_dequantised_unclamped"] = _err(
        got, moe_reference(deq, x, weights, ids, device, None))
    src = {key: value for key, value in source.items()}
    record["vs_bf16_source"] = _err(
        got, moe_reference(src, x, weights, ids, device, SWIGLU_LIMIT))
    record["clamp_active_fraction"] = float(
        (torch.cat([(x.float() @ deq[(e, "gate_proj")].t()).abs().reshape(-1)
                    for e in range(EXPERTS)]) > SWIGLU_LIMIT).float().mean())
    record["resident_bytes"] = {
        name: getattr(layer, name).numel() * getattr(layer, name).element_size()
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale",
                     "w13_weight_scale_2", "w2_weight_scale_2", "w13_input_scale",
                     "w2_input_scale")}
    record["wire_bytes_on_disk"] = int(sum(
        v.numel() for k, v in wires.items() if k.endswith(".wire")))
    record["route_record"] = read_route(layer)
    record["scheme"] = scheme
    if manifest is not None:
        stack = manifest["modules"][layer_name]
        record["exported_manifest"] = {
            "family": stack["family"], "attested_by": stack["attested_by"],
            "resident_bytes_resident_mode": stack["resident_bytes_resident_mode"],
            "serving_gate": manifest["serving_gate"]}
    return record


def negative_leg(name, mutate, expect, device, base):
    """A refusal is a result -- but only when it is the refusal that was asked for.

    ``expect`` is a substring (or a tuple of alternatives) of the message the
    gate under test produces.  A leg that raises something else is a HARNESS
    fault reported as one (``matched: false``), never counted as the route
    refusing."""
    wires = {k: v.clone() for k, v in base[0].items()}
    scheme = copy.deepcopy(base[1])
    mutate(wires, scheme)
    expects = expect if isinstance(expect, tuple) else (expect,)
    row = {"leg": name, "expected_substring": list(expects)}
    try:
        layer = build_layer(scheme, device)
        load(layer, wires)
        layer.quant_method.process_weights_after_loading(layer)
        row.update({"refused": False, "matched": False})
    except Exception as exc:  # noqa: BLE001 -- the refusal IS the measurement
        message = str(exc)
        row.update({"refused": True, "error_type": type(exc).__name__,
                    "matched": any(e in message for e in expects), "message": message[:400]})
    return row


def _flip_a_payload_byte(wires, scheme):
    key = "1.up_proj.wire"
    wires[key][len(wires[key]) // 2] ^= 0x01


def _understate_the_stride(wires, scheme):
    scheme["groups"]["w13"]["wire_stride"] -= 1


def _overstate_the_stride(wires, scheme):
    scheme["groups"]["w13"]["wire_stride"] += 4096


def _wrong_expert_count(wires, scheme):
    scheme["experts"] = EXPERTS + 1


def _wrong_rung(wires, scheme):
    scheme["groups"]["w2"]["q256"] = 768


def _drop_an_input_scale(wires, scheme):
    del wires["1.up_proj.input_global_scale"]


def _drop_half_an_expert(wires, scheme):
    del wires["2.up_proj.wire"]


def _hand_a_stock_tensor(wires, scheme):
    wires["0.gate_proj.weight"] = torch.zeros(INTER, HIDDEN // 2, dtype=torch.uint8)


NEGATIVE_LEGS = (
    ("payload_byte_flipped", _flip_a_payload_byte, ("digest", "checksum")),
    ("stride_understated", _understate_the_stride, "does not fit the group"),
    ("stride_overstated", _overstate_the_stride, ("is not what its lengths imply", "wire_stride")),
    ("expert_count_wrong", _wrong_expert_count, "the sidecar declares"),
    ("rung_wrong", _wrong_rung, "outside the rungs this build's decoder reads"),
    ("input_scale_missing", _drop_an_input_scale, "carry no input_global_scale"),
    ("half_an_expert", _drop_half_an_expert, ("one half of w13", "wire length", "empty wire")),
    ("stock_tensor_for_a_wire_stack", _hand_a_stock_tensor, "carries a stock tensor"),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tokens", type=int, default=17)
    ap.add_argument("--q256", type=int, default=Q256)
    ap.add_argument("--export-workdir", type=Path, default=None,
                    help="where the exported leg writes its checkpoint; a temporary directory "
                         "under TMPDIR when unset. NEVER /tmp.")
    ap.add_argument("--no-exported-leg", action="store_true",
                    help="skip the leg whose wires the exporter wrote (the pair's B side)")
    ap.add_argument("--no-clamp-leg", action="store_true",
                    help="skip the hot-activation leg that makes the SwiGLU clamp bite")
    args = ap.parse_args()

    os.environ.setdefault("TESSERA_SERVE_MODE", "resident")
    device = torch.device("cuda")
    context = vllm_config()
    context.__enter__()
    init_parallel()
    out = {
        "probe": "nvfp4_moe_route_load_probe",
        "vllm": __import__("vllm").__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "runtime_image": os.environ.get("TESSERA_RUNTIME_IMAGE"),
        "runtime_image_digest": os.environ.get("TESSERA_RUNTIME_IMAGE_DIGEST"),
        "serve_mode": os.environ["TESSERA_SERVE_MODE"],
        "dimensions": {"experts": EXPERTS, "hidden": HIDDEN, "intermediate": INTER,
                       "topk": TOPK, "q256": args.q256, "tokens": args.tokens,
                       "swiglu_limit": SWIGLU_LIMIT},
    }
    started = time.time()
    try:
        out["positive"] = positive_leg(device, args.tokens, args.q256)
    except Exception:  # noqa: BLE001 -- a failed probe is a recorded probe
        out["positive"] = {"raised": traceback.format_exc()[-4000:]}
    if not args.no_clamp_leg:
        try:
            out["positive_clamped"] = positive_leg(device, args.tokens, args.q256,
                                                   x_scale=CLAMP_X_SCALE)
        except Exception:  # noqa: BLE001
            out["positive_clamped"] = {"raised": traceback.format_exc()[-4000:]}
    if not args.no_exported_leg:
        workdir = args.export_workdir
        try:
            if workdir is None:
                import tempfile
                workdir = Path(tempfile.mkdtemp(prefix="nvfp4_moe_export_leg_",
                                                dir=os.environ.get("TMPDIR", "/home/rob/tmp")))
            workdir.mkdir(parents=True, exist_ok=True)
            out["positive_exported"] = positive_leg(
                device, args.tokens, args.q256,
                build=lambda: build_stack_from_export(device, workdir, q256=args.q256),
                layer_name=EXPORT_LAYER)
            out["positive_exported"]["export_workdir"] = str(workdir)
        except Exception:  # noqa: BLE001
            out["positive_exported"] = {"raised": traceback.format_exc()[-4000:]}
    try:
        base = build_stack(device, q256=args.q256)
        out["negative"] = [negative_leg(name, mutate, expect, device, base)
                           for name, mutate, expect in NEGATIVE_LEGS]
    except Exception:  # noqa: BLE001
        out["negative"] = [{"raised": traceback.format_exc()[-4000:]}]
    out["seconds"] = round(time.time() - started, 1)
    text = json.dumps(out, indent=1, sort_keys=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)

    def tile_ok(record):
        tile = record.get("tile_is_materialize_stock_byte_for_byte") or {}
        return bool(tile.get("identical")) and bool(tile.get("globals_match"))

    ok = tile_ok(out.get("positive", {}))
    ok = ok and all(row.get("matched") for row in out["negative"])
    for leg in ("positive_exported", "positive_clamped"):
        if leg in out:
            ok = ok and tile_ok(out[leg])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
