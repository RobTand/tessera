#!/usr/bin/env python3
"""Load and execute a Tessera routed-MoE stack on the pinned runtime.

WHAT THIS IS EVIDENCE FOR, AND WHAT IT IS NOT.  It is the LOAD-AND-EXECUTE
contract of ``tessera.serving.moe_route``, measured rather than asserted:
vLLM's real ``RoutedExperts`` is constructed with a Tessera checkpoint's
``quantization_config``, its real ``load_weights`` is handed per-expert wire
tensors under the names a checkpoint would carry, the route's own
``process_weights_after_loading`` decodes them, and the runtime's own
fused-MoE kernel multiplies them.  The output is compared against a torch
reference over the DEQUANTISED expert weights -- so a disagreement is the
kernel or the plumbing, not the quantiser -- and, separately, against the BF16
source, which is the quantisation error and is reported as such.

It is NOT a served census and NOT a KL: no model is loaded, no engine is
started, and the weights are random.  A ``routed_moe`` cell in
``runtime_contract.json`` needs a served artifact and is not earned here.

THE NEGATIVE LEGS MATTER AS MUCH AS THE POSITIVE ONE.  A route that decodes
correctly but accepts bytes it should refuse is a wrong tensor waiting for a
different checkpoint, so the probe also flips one payload byte (the reader
must refuse), understates a group's ``wire_stride`` (the layout must refuse),
and hands a stack an expert count the sidecar does not declare.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

EXPERTS, HIDDEN, INTER, TOPK = 4, 512, 256, 2
Q256 = 1024
LAYER = "model.layers.0.mlp.experts"
#: The same stack under the naming a REAL routed-MoE checkpoint carries, which
#: is what the exporter walks: ``BODY_LAYER`` finds the decoder under a
#: sub-model, and the construction gate resolves ``...mlp.experts`` through the
#: censused GLM mapper.  The exported leg uses this name so that the checkpoint
#: it writes is one the exporter would write for the real model, not a shape
#: the gate happens not to cover.
EXPORT_LAYER = "model.language_model.layers.1.mlp.experts"


def _encode(weight, name, q256):
    from tessera.alphabet import E4M3_GRID
    from tessera.export import DEFAULT_CODE, encode_linear_planes
    from tessera.fused import pack_fused
    from tessera.stock import materialize_stock

    exported, unit, forests = encode_linear_planes(
        weight.contiguous(), grid=E4M3_GRID, q256=q256, name=name, verify=False)
    blob = pack_fused([(name, weight.shape[0], exported.blob)])
    return blob, materialize_stock(unit, forests, DEFAULT_CODE)


def build_stack(device, seed=0, q256=Q256):
    """E experts of (gate, up, down): the wires, the scheme, the BF16 source."""
    from tessera.serving.scheme import TESSERA_FP8

    generator = torch.Generator(device="cpu").manual_seed(seed)
    wires, source, stock = {}, {}, {}
    strides = {"w13": 0, "w2": 0}
    for expert in range(EXPERTS):
        for name, rows, cols, group in (("gate_proj", INTER, HIDDEN, "w13"),
                                        ("up_proj", INTER, HIDDEN, "w13"),
                                        ("down_proj", HIDDEN, INTER, "w2")):
            weight = (torch.randn(rows, cols, generator=generator) * 0.02).to(device, torch.float32)
            blob, tensors = _encode(weight, name, q256)
            wires[f"{expert}.{name}.wire"] = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
            source[(expert, name)] = weight
            stock[(expert, name)] = tensors
            strides[group] = max(strides[group], len(blob))
    scheme = {
        "family": TESSERA_FP8, "structure": "routed_moe", "grid": "E4M3", "body": "WINDOW",
        "plane": "CHANNEL", "experts": EXPERTS,
        "groups": {
            "w13": {"rows": 2 * INTER, "columns": HIDDEN, "q256": q256,
                    "wire_stride": strides["w13"],
                    "roles": [["gate_proj", INTER], ["up_proj", INTER]]},
            "w2": {"rows": HIDDEN, "columns": INTER, "q256": q256,
                   "wire_stride": strides["w2"], "roles": [["down_proj", HIDDEN]]}},
    }
    return wires, scheme, source, stock


def quantization_config(scheme, layer_name=LAYER):
    return {"quant_method": "tessera", "format": "tessera",
            "config_groups": {"tessera_experts": {"format": "TESSERA", "targets": [layer_name],
                                                  "scheme": scheme}},
            "ignore": []}


def build_stack_from_export(device, workdir, seed=0, q256=Q256):
    """The same stack, with the wires and the scheme WRITTEN BY THE EXPORTER.

    THE MATCHED PAIR FOR ``build_stack``, and the seam it closes.  Everything
    is pinned: the same expert count, shapes, rung, seed and weight values,
    the same encoder call underneath, the same load-and-execute legs
    downstream.  The ONE thing that varies is who produced the bytes and the
    sidecar -- this script, or ``experiments/export_tessera_serving.py`` run
    end to end over a checkpoint on disk.  Until 2026-09-04 only the first
    existed, so every routed-MoE receipt was about a stack no exporter could
    have written; a route that decodes wires nobody can produce serves nothing.

    Returns ``build_stack``'s tuple, with ``stock`` derived from the EXPORTED
    container rather than kept from the encode -- the reference the tile is
    compared against is then read out of the artifact, which is the comparison
    that matters here.
    """
    import importlib.util
    import subprocess

    from safetensors.torch import save_file
    from tessera.fused import parse_fused
    from tessera.export import DEFAULT_CODE
    from tessera.stock import materialize_stock
    from tessera.unit_artifact import parse_unit_artifact

    workdir = Path(workdir)
    src, out = workdir / "src", workdir / "out"
    src.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source, tensors = {}, {}
    for expert in range(EXPERTS):
        for name, rows, cols in (("gate_proj", INTER, HIDDEN), ("up_proj", INTER, HIDDEN),
                                 ("down_proj", HIDDEN, INTER)):
            weight = (torch.randn(rows, cols, generator=generator) * 0.02)
            tensors[f"{EXPORT_LAYER}.{expert}.{name}.weight"] = weight
            source[(expert, name)] = weight.to(device, torch.float32)
    # The router beside them: not a projection, passed through, and the reason
    # the checkpoint is a routed-MoE layer rather than a bag of tensors.
    tensors[f"{EXPORT_LAYER.rsplit('.', 1)[0]}.gate.weight"] = torch.zeros(EXPERTS, HIDDEN)
    save_file({k: v.contiguous() for k, v in tensors.items()},
              str(src / "model.safetensors"), metadata={"format": "pt"})
    (src / "config.json").write_text(json.dumps({
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {"hidden_size": HIDDEN, "moe_intermediate_size": INTER,
                        "num_hidden_layers": 2, "n_routed_experts": EXPERTS}}))
    plan = workdir / "plan.json"
    plan.write_text(json.dumps({EXPORT_LAYER: {"grid": "E4M3", "q256": q256}}))

    spec = importlib.util.spec_from_file_location(
        "export_tessera_serving", Path(__file__).resolve().parent / "export_tessera_serving.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    argv = sys.argv
    sys.argv = ["export", str(src), str(out), "--grid", "E4M3", "--q256", str(q256),
                "--plan-json", str(plan), "--device", str(device)]
    try:
        exporter.main()
    finally:
        sys.argv = argv

    written = json.loads((out / "config.json").read_text())["quantization_config"]
    groups = [g for g in written["config_groups"].values() if g["targets"] == [EXPORT_LAYER]]
    if len(groups) != 1:
        raise RuntimeError(f"the exporter declared {len(groups)} group(s) for {EXPORT_LAYER}")
    scheme = groups[0]["scheme"]

    from safetensors import safe_open
    wires, stock = {}, {}
    with safe_open(str(out / "model.safetensors"), framework="pt") as handle:
        for expert in range(EXPERTS):
            for name in ("gate_proj", "up_proj", "down_proj"):
                blob = handle.get_tensor(f"{EXPORT_LAYER}.{expert}.{name}.wire")
                wires[f"{expert}.{name}.wire"] = blob.clone()
                member, = parse_fused(bytes(blob.tolist()))
                parsed = parse_unit_artifact(member.blob, device=str(device))
                stock[(expert, name)] = materialize_stock(parsed.unit, parsed.forests, DEFAULT_CODE)
    return wires, scheme, source, stock


def vllm_config():
    """The engine's config, which ``initialize_model_parallel`` and every
    ``CustomOp`` read from a context rather than an argument."""
    from vllm.config import VllmConfig, set_current_vllm_config
    return set_current_vllm_config(VllmConfig())


def init_parallel():
    from vllm.distributed import (ensure_model_parallel_initialized,
                                  init_distributed_environment)
    init_distributed_environment(
        world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:52731",
        local_rank=0, backend="gloo")
    ensure_model_parallel_initialized(1, 1)
    # The modular kernel allocates its intermediates from the worker's
    # workspace; ``GPUModelRunner.__init__`` normally creates it.
    from vllm.v1.worker.workspace import init_workspace_manager
    init_workspace_manager(torch.device("cuda"))


def build_layer(scheme, device, layer_name=LAYER):
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
        in_dtype=torch.bfloat16, intermediate_size_per_partition=INTER, moe_backend="auto")
    manager = ExpertMapManager(
        max_num_batched_tokens=64, top_k=TOPK, global_num_experts=EXPERTS,
        num_redundant_experts=0, num_expert_group=None, moe_parallel_config=parallel,
        placement_strategy="linear", enable_eplb=False)
    config = TesseraConfig.from_config(quantization_config(scheme, layer_name))
    layer = RoutedExperts(layer_name=layer_name, params_dtype=torch.bfloat16, moe_config=moe,
                          quant_config=config, expert_map_manager=manager)
    return layer.to(device)


def load(layer, wires):
    """Through the runtime's OWN loader, under the names a checkpoint carries."""
    loaded = sorted(layer.load_weights([(k, v) for k, v in sorted(wires.items())]))
    return loaded


def reference(stock, x, topk_weights, topk_ids, device):
    """The same MoE in torch over the DEQUANTISED expert weights."""
    from tessera.stock import stock_dequant

    out = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(EXPERTS):
        w1 = stock_dequant(stock[(expert, "gate_proj")]).to(device, torch.float32)
        w3 = stock_dequant(stock[(expert, "up_proj")]).to(device, torch.float32)
        w2 = stock_dequant(stock[(expert, "down_proj")]).to(device, torch.float32)
        mask = (topk_ids == expert)
        if not mask.any():
            continue
        rows, slots = mask.nonzero(as_tuple=True)
        xs = x[rows].float()
        h = torch.nn.functional.silu(xs @ w1.t()) * (xs @ w3.t())
        y = h @ w2.t()
        out.index_add_(0, rows, y * topk_weights[rows, slots].unsqueeze(1).float())
    return out


def w8a8_reference(stock, x, topk_weights, topk_ids, device):
    """The same MoE with the A side quantised the way the kernel quantises it.

    THE CONTROL THAT MAKES THE OTHER NUMBER READABLE.  ``reference`` keeps the
    activations in fp32, so its disagreement with the kernel is the whole W8A8
    contract -- per-token E4M3 on x, and again on the intermediate -- and not a
    plumbing fault.  This leg emulates that contract in torch, so the residue
    against IT is what is left over once the arithmetic both sides agreed to
    run is accounted for.  Without it, "rel_l2 0.045" has nothing to be
    compared against and is not evidence of anything.
    """
    from tessera.stock import stock_dequant

    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    def per_token(t):
        scale = (t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / fp8_max)
        return (t / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).float() * scale

    out = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(EXPERTS):
        mask = (topk_ids == expert)
        if not mask.any():
            continue
        rows, slots = mask.nonzero(as_tuple=True)
        w1 = stock_dequant(stock[(expert, "gate_proj")]).to(device, torch.float32)
        w3 = stock_dequant(stock[(expert, "up_proj")]).to(device, torch.float32)
        w2 = stock_dequant(stock[(expert, "down_proj")]).to(device, torch.float32)
        xs = per_token(x[rows].float())
        h = torch.nn.functional.silu(xs @ w1.t()) * (xs @ w3.t())
        y = per_token(h) @ w2.t()
        out.index_add_(0, rows, y * topk_weights[rows, slots].unsqueeze(1).float())
    return out


def source_reference(source, x, topk_weights, topk_ids, device):
    out = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(EXPERTS):
        mask = (topk_ids == expert)
        if not mask.any():
            continue
        rows, slots = mask.nonzero(as_tuple=True)
        xs = x[rows].float()
        w1 = source[(expert, "gate_proj")].to(device, torch.float32)
        w3 = source[(expert, "up_proj")].to(device, torch.float32)
        w2 = source[(expert, "down_proj")].to(device, torch.float32)
        h = torch.nn.functional.silu(xs @ w1.t()) * (xs @ w3.t())
        out.index_add_(0, rows, (h @ w2.t()) * topk_weights[rows, slots].unsqueeze(1).float())
    return out


def _err(got, want):
    got, want = got.float(), want.float()
    denom = want.norm().item() or 1.0
    return {"rel_l2": (got - want).norm().item() / denom,
            "max_abs": (got - want).abs().max().item(),
            "want_absmax": want.abs().max().item()}


def positive_leg(device, tokens, q256, build=None, layer_name=LAYER):
    """Load and execute one stack.  ``build`` says where its bytes came from."""
    from tessera.serving.telemetry import read_route

    build = build or (lambda: build_stack(device, q256=q256))
    wires, scheme, source, stock = build()
    layer = build_layer(scheme, device, layer_name)
    method = layer.quant_method
    record = {"method": type(method).__name__,
              "backend": str(getattr(getattr(method, "fp8_backend", None), "value", None)),
              "experts_cls": getattr(getattr(method, "experts_cls", None), "__name__", None)}
    record["params_before_load"] = sorted(n for n, _ in layer.named_parameters())
    loaded = load(layer, wires)
    record["loaded_param_names"] = sorted(set(loaded))
    record["load_calls"] = len(loaded)
    method.process_weights_after_loading(layer)
    record["params_after_load"] = sorted(n for n, _ in layer.named_parameters())
    record["w13_weight_shape"] = list(layer.w13_weight.shape)
    record["w2_weight_shape"] = list(layer.w2_weight.shape)
    record["w13_weight_dtype"] = str(layer.w13_weight.dtype)
    record["w13_weight_scale_shape"] = list(layer.w13_weight_scale.shape)
    record["w2_weight_scale_shape"] = list(layer.w2_weight_scale.shape)

    # The decoded tile IS materialize_stock's, expert by expert.
    identical = True
    for expert in range(EXPERTS):
        w13 = layer.w13_weight[expert].view(torch.uint8)
        if w13.shape[0] != 2 * INTER:
            identical = False
            break
        identical &= bool(torch.equal(
            w13[:INTER].cpu(), stock[(expert, "gate_proj")]["weight"].view(torch.uint8).cpu()))
        identical &= bool(torch.equal(
            w13[INTER:].cpu(), stock[(expert, "up_proj")]["weight"].view(torch.uint8).cpu()))
        identical &= bool(torch.equal(
            layer.w2_weight[expert].view(torch.uint8).cpu(),
            stock[(expert, "down_proj")]["weight"].view(torch.uint8).cpu()))
    record["tile_is_materialize_stock_byte_for_byte"] = bool(identical)

    generator = torch.Generator(device=device).manual_seed(11)
    x = torch.randn(tokens, HIDDEN, generator=generator, device=device, dtype=torch.bfloat16)
    logits = torch.randn(tokens, EXPERTS, generator=generator, device=device, dtype=torch.float32)
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), TOPK, dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(torch.float32)
    ids = ids.to(torch.int32)

    got = method.apply(layer, x, weights, ids, None, None)
    torch.cuda.synchronize()
    record["output_shape"] = list(got.shape)
    record["vs_w8a8_emulated"] = _err(got, w8a8_reference(stock, x, weights, ids, device))
    record["vs_dequantised_reference"] = _err(got, reference(stock, x, weights, ids, device))
    record["vs_bf16_source"] = _err(got, source_reference(source, x, weights, ids, device))
    record["resident_bytes"] = {
        "w13_weight": layer.w13_weight.numel() * layer.w13_weight.element_size(),
        "w2_weight": layer.w2_weight.numel() * layer.w2_weight.element_size(),
        "w13_weight_scale": layer.w13_weight_scale.numel() * 4,
        "w2_weight_scale": layer.w2_weight_scale.numel() * 4,
    }
    record["wire_bytes_on_disk"] = int(sum(v.numel() for v in wires.values()))
    record["route_record"] = read_route(layer)
    record["scheme"] = scheme
    return record


def negative_leg(name, mutate, expect, device, q256):
    """A refusal is a result -- but only when it is the refusal that was asked for.

    ``expect`` is a substring of the message the gate under test produces.  A
    leg that raises something else is a HARNESS fault reported as one
    (``matched: false``), never counted as the route refusing: "an exception
    happened" and "the check fired" are different facts, and a probe that
    conflates them passes on a broken harness.
    """
    wires, scheme, _source, _stock = build_stack(device, q256=q256)
    mutate(wires, scheme)
    row = {"leg": name, "expected_substring": expect}
    try:
        layer = build_layer(scheme, device)
        load(layer, wires)
        layer.quant_method.process_weights_after_loading(layer)
        row.update({"refused": False, "matched": False})
    except Exception as exc:  # noqa: BLE001 -- the refusal IS the measurement
        message = str(exc)
        row.update({"refused": True, "error_type": type(exc).__name__,
                    "matched": expect in message, "message": message[:400]})
    return row


def _flip_a_payload_byte(wires, scheme):
    key = "1.up_proj.wire"
    wires[key] = wires[key].clone()
    wires[key][len(wires[key]) // 2] ^= 0x01


def _understate_the_stride(wires, scheme):
    scheme["groups"]["w13"]["wire_stride"] -= 1


def _overstate_the_stride(wires, scheme):
    scheme["groups"]["w13"]["wire_stride"] += 4096


def _wrong_expert_count(wires, scheme):
    scheme["experts"] = EXPERTS + 1


def _wrong_rung(wires, scheme):
    scheme["groups"]["w2"]["q256"] = 768


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
    args = ap.parse_args()

    os.environ.setdefault("TESSERA_SERVE_MODE", "resident")
    device = torch.device("cuda")
    context = vllm_config()
    context.__enter__()
    init_parallel()
    out = {
        "probe": "moe_route_load_probe",
        "vllm": __import__("vllm").__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "serve_mode": os.environ["TESSERA_SERVE_MODE"],
        "dimensions": {"experts": EXPERTS, "hidden": HIDDEN, "intermediate": INTER,
                       "topk": TOPK, "q256": args.q256, "tokens": args.tokens},
    }
    started = time.time()
    try:
        out["positive"] = positive_leg(device, args.tokens, args.q256)
    except Exception:  # noqa: BLE001 -- a failed probe is a recorded probe
        out["positive"] = {"raised": traceback.format_exc()[-4000:]}
    # THE B SIDE OF THE PAIR: the same stack, from bytes the EXPORTER wrote.
    # Everything but the producer is pinned -- shapes, rung, seed, weights,
    # encoder, and every leg below -- so a difference between the two records
    # is a difference between a synthesised sidecar and a written one, which is
    # the only thing this leg is here to measure.
    if not args.no_exported_leg:
        workdir = args.export_workdir
        try:
            if workdir is None:
                import tempfile
                workdir = Path(tempfile.mkdtemp(prefix="moe_export_leg_",
                                                dir=os.environ.get("TMPDIR", "/home/rob/tmp")))
            workdir.mkdir(parents=True, exist_ok=True)
            out["positive_exported"] = positive_leg(
                device, args.tokens, args.q256,
                build=lambda: build_stack_from_export(device, workdir, q256=args.q256),
                layer_name=EXPORT_LAYER)
            out["positive_exported"]["export_workdir"] = str(workdir)
        except Exception:  # noqa: BLE001 -- a failed leg is a recorded leg
            out["positive_exported"] = {"raised": traceback.format_exc()[-4000:]}
    out["negative"] = [
        negative_leg(name, mutate, expect, device, args.q256)
        for name, mutate, expect in (
            ("payload_byte_flipped", _flip_a_payload_byte, "digest"),
            ("stride_understated", _understate_the_stride, "does not fit the group"),
            ("stride_overstated", _overstate_the_stride, "is not what its lengths imply"),
            ("expert_count_wrong", _wrong_expert_count, "the sidecar declares"),
            ("rung_wrong", _wrong_rung, "the sidecar scheme declares"))]
    out["seconds"] = round(time.time() - started, 1)
    text = json.dumps(out, indent=1, sort_keys=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    ok = bool(out.get("positive", {}).get("tile_is_materialize_stock_byte_for_byte"))
    ok = ok and all(row.get("matched") for row in out["negative"])
    if "positive_exported" in out:
        ok = ok and bool(out["positive_exported"].get("tile_is_materialize_stock_byte_for_byte"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
