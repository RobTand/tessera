#!/usr/bin/env python3
"""Cheap protocol probe: the REAL vLLM base class, before and after load.

The engine's MoE runner asks the method object's own modular protocol
(``is_monolithic``, ``topk_indices_dtype``, ``mk_can_overlap_shared_experts``,
``get_fused_moe_quant_config``) at construction, and again from ``_forward_impl``
during ``profile_run`` -- which is where attempt mixed514-a5 died, with its
weights loaded and the engine still warming up:

    moe_runner -> quant_method.is_monolithic
    base: return self.experts_cls.is_monolithic()
    native window route: experts_cls is None

This runs in the pinned image with the real ``build_tessera_moe_method`` and
the real ``FusedMoEMethodBase``, so a Python protocol bug is caught in a minute
instead of after a 12 GiB two-host load.

  docker run --rm --gpus all --user 1000:1000 -e HOME=/tmp \
    -e PYTHONPATH=/work/src:/work/tests -v <worktree>:/work:ro --entrypoint bash \
    <image> -lc "python3 /work/tests/native_moe_protocol_probe.py"
"""

import importlib.util
import json
import pathlib
import sys
import types

import torch

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "native_window_moe_stock_crosscheck",
    HERE / "native_window_moe_stock_crosscheck.py")
xc = importlib.util.module_from_spec(_spec)
sys.modules["native_window_moe_stock_crosscheck"] = xc
_spec.loader.exec_module(xc)

from tessera.serving import moe_route  # noqa: E402

QUESTIONS = ("is_monolithic", "topk_indices_dtype", "mk_can_overlap_shared_experts")


def answers(method, layer):
    """Exactly what the runner reads, with the real base class underneath."""
    asked = {name: getattr(method, name) for name in QUESTIONS}
    asked["get_fused_moe_quant_config"] = method.get_fused_moe_quant_config(layer)
    return asked


def expect(label, asked, report):
    """Native routes are modular: False / None / False / no stock quant config."""
    ok = (asked["is_monolithic"] is False
          and asked["topk_indices_dtype"] is None
          and asked["mk_can_overlap_shared_experts"] is False
          and asked["get_fused_moe_quant_config"] is None)
    report["arms"].append({
        "arm": label,
        "answers": {k: (None if v is None else str(v)) for k, v in asked.items()},
        "ok": bool(ok)})
    print(f"{'ok  ' if ok else 'FAIL'} {label}: is_monolithic={asked['is_monolithic']} "
          f"topk_indices_dtype={asked['topk_indices_dtype']} "
          f"mk_can_overlap_shared_experts={asked['mk_can_overlap_shared_experts']} "
          f"quant_config_is_none={asked['get_fused_moe_quant_config'] is None}",
          flush=True)
    return ok


def main():
    report = {"schema": "tessera.native_moe_protocol_probe.v1",
              "device": torch.cuda.get_device_name(), "arms": []}
    ok = True

    # --- the ordinary native FP8 route (no research selection) -------------
    w13_blobs, w2_blobs, scheme, _reference = xc._stack()
    layer = xc._native_layer(tp_rank=0, tp_size=2)
    method = moe_route.build_tessera_moe_method(scheme, "m", "resident", layer)
    ok &= expect("fp8_native_pre_load", answers(method, layer), report)
    method.create_weights(layer, xc.EXPERTS, xc.HIDDEN, xc.INTER // 2, torch.bfloat16)
    xc._load_all(method, layer, w13_blobs, w2_blobs)
    method.process_weights_after_loading(layer)
    ok &= expect("fp8_native_post_load", answers(method, layer), report)

    # --- the research-selected folded BF16 route --------------------------
    b13, b2, bscheme, _expected = xc.bf16_wires_native_data()
    layer = xc._native_layer(tp_rank=0, tp_size=2)
    from vllm.config import set_current_vllm_config
    with set_current_vllm_config(types.SimpleNamespace(
            model_config=types.SimpleNamespace(enforce_eager=True))):
        method = moe_route.build_tessera_moe_method(
            bscheme, "m", "resident", layer,
            research_selected=moe_route.ResearchSelectedMoeConfig(
                max_experts_per_chunk=2, expected_tensor_parallel_size=2))
        ok &= expect("bf16_research_pre_load", answers(method, layer), report)
        method.create_weights(layer, 2, xc.HIDDEN, xc.INTER // 2, torch.bfloat16)
        xc._load_all(method, layer, b13, [pair[0] for pair in b2])
        method.process_weights_after_loading(layer)
        ok &= expect("bf16_research_post_load", answers(method, layer), report)

    report["passed"] = bool(ok)
    print(json.dumps(report))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
