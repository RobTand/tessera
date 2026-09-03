#!/usr/bin/env python3
"""Ask the pinned serving build what it does with an NVFP4 MoE on a clamped config.

WHY THIS EXISTS.  ``docs/tessera-serving-and-moe-contract.md`` §9.1 records that
``--moe-backend flashinfer_b12x`` raises on GLM-5.3-Flash's config (which sets
``swiglu_limit: 10.0``), and that **which** of the remaining backends is backed
on sm121 is "not measured".  That gap is what stops an NVFP4 MoE contract cell
being written: principle 14 forbids naming a backend in ``requires_serve_flags``
that nobody has seen the runtime resolve.  Reading the oracle's source answers
the question only as far as a reading goes; this script makes the runtime answer
it.

It runs INSIDE the pinned container with the GPU visible and does two things:

* **device leg** -- calls every NVFP4 expert class's own argument-free
  ``_supports_current_device()`` plus the ``has_flashinfer_*`` availability
  probes.  No config is involved, so nothing here can be an artefact of a
  fabricated config.
* **oracle leg** -- builds a ``FusedMoEConfig`` carrying GLM-5.3-Flash's MoE
  dimensions (288 routed experts, top-8, hidden 4096, moe_intermediate 2048,
  gated SiLU, DeepSeekV3 routing, bf16, no parallelism) and calls
  ``select_nvfp4_moe_backend`` with the W4A4 keys every NVFP4 MoE method passes
  (``kNvfp4Static`` x ``kNvfp4Dynamic``), clamped and unclamped, auto and
  explicit.

SCOPE, because this is a claim about another runtime.  The oracle leg's config
is *constructed*, not lifted off a live serve: it is the oracle's own selection
for those field values, not a served MoE.  A serve would still have to survive
weight loading, ``process_weights_after_loading``, and the "shape-specific
fallbacks may still occur at runtime" the oracle's own docstring warns about.
The activation is asserted gated, because a non-gated pick silently changes the
answer (it did, on the first run of this probe: ``SILU_NO_MUL`` made every
FlashInfer backend report "does not support ... activation" and the auto path
fall through to VLLM_CUTLASS).

Run it through ``experiments/nvfp4_moe_oracle_probe.sh``, which takes the serve
lock -- this creates a CUDA context, and a serve starting beside it would see
less memory than it asked for.
"""
from __future__ import annotations

import json
import traceback

import torch

import vllm
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    backend_to_kernel_cls,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform

# GLM-5.3-Flash text_config (identical in GLM-5.3-Flash-4layer).
NUM_EXPERTS = 288
TOPK = 8
HIDDEN = 4096
MOE_INTERMEDIATE = 2048
SWIGLU_LIMIT = 10.0


def activation() -> MoEActivation:
    """GLM's MoE MLP is gated SwiGLU (``hidden_act: silu`` + SiluAndMulWithClamp)."""
    act = MoEActivation.SILU
    assert act.is_gated, "a non-gated activation would change every backend's answer"
    return act


def make_config(swiglu_limit: float | None, moe_backend: str) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=NUM_EXPERTS,
        experts_per_token=TOPK,
        hidden_dim=HIDDEN,
        intermediate_size=MOE_INTERMEDIATE,
        num_local_experts=NUM_EXPERTS,
        num_logical_experts=NUM_EXPERTS,
        activation=activation(),
        device=torch.device("cuda"),
        routing_method=RoutingMethodType.DeepSeekV3,  # GLM's topk_method: noaux_tc
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        moe_backend=moe_backend,
        swiglu_limit=swiglu_limit,
    )


CASES = [
    # name, swiglu_limit, moe_backend
    ("clamped_explicit_b12x", SWIGLU_LIMIT, "flashinfer_b12x"),
    ("clamped_auto", SWIGLU_LIMIT, "auto"),
    ("clamped_explicit_flashinfer_cutlass", SWIGLU_LIMIT, "flashinfer_cutlass"),
    ("unclamped_auto", None, "auto"),
    ("unclamped_explicit_b12x", None, "flashinfer_b12x"),
]

FLASHINFER_PROBES = [
    "has_flashinfer",
    "has_flashinfer_trtllm_fused_moe",
    "has_flashinfer_cutlass_fused_moe",
    "has_flashinfer_cutedsl_moe_nvfp4",
    "has_flashinfer_cutedsl_grouped_gemm_nt_masked",
    "has_flashinfer_b12x_moe",
]


def main() -> int:
    out: dict = {
        "schema": "tessera.nvfp4_moe_oracle_probe/1",
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "device_capability": str(current_platform.get_device_capability()),
        "is_device_capability_family_100": current_platform.is_device_capability_family(100),
        "is_device_capability_family_120": current_platform.is_device_capability_family(120),
        "model_dims": {
            "source": "GLM-5.3-Flash text_config",
            "num_experts": NUM_EXPERTS,
            "experts_per_token": TOPK,
            "hidden_dim": HIDDEN,
            "moe_intermediate_size": MOE_INTERMEDIATE,
            "swiglu_limit": SWIGLU_LIMIT,
        },
        "activation": str(activation()),
        "routing_method": str(RoutingMethodType.DeepSeekV3),
    }

    device: dict = {}
    for backend in NvFp4MoeBackend:
        try:
            classes = backend_to_kernel_cls(backend)
        except Exception as exc:  # noqa: BLE001
            device[backend.value] = {"import": f"ERROR {type(exc).__name__}: {exc}"}
            continue
        entry = {}
        for cls in classes:
            try:
                entry[cls.__name__] = bool(cls._supports_current_device())
            except Exception as exc:  # noqa: BLE001
                entry[cls.__name__] = f"ERROR {type(exc).__name__}: {exc}"
        device[backend.value] = entry
    out["supports_current_device"] = device

    probes: dict = {}
    try:
        import vllm.utils.flashinfer as fi
    except Exception as exc:  # noqa: BLE001
        probes["import"] = f"ERROR {type(exc).__name__}: {exc}"
    else:
        for name in FLASHINFER_PROBES:
            fn = getattr(fi, name, None)
            if fn is None:
                probes[name] = "ABSENT"
                continue
            try:
                probes[name] = bool(fn())
            except Exception as exc:  # noqa: BLE001
                probes[name] = f"ERROR {type(exc).__name__}: {exc}"
    out["availability_probes"] = probes

    cases: dict = {}
    for name, limit, backend in CASES:
        try:
            config = make_config(limit, backend)
        except Exception as exc:  # noqa: BLE001
            cases[name] = {"config_error": f"{type(exc).__name__}: {exc}"}
            continue
        try:
            resolved, cls = select_nvfp4_moe_backend(config, kNvfp4Static, kNvfp4Dynamic)
        except Exception as exc:  # noqa: BLE001
            cases[name] = {
                "raised": type(exc).__name__,
                "message": str(exc),
                "where": traceback.format_exc().strip().splitlines()[-3:],
            }
        else:
            cases[name] = {"selected": resolved.value, "kernel": cls.__name__}
    out["cases"] = cases

    # Two emissions on purpose: the indented block is for a reader watching the
    # container, and the single tagged line is what the wrapper extracts into
    # the result file.  A multi-line tagged block silently truncates to "{" when
    # a sed grabs only the tagged line, which is how the first run of this probe
    # wrote a 2-byte result.
    print(json.dumps(out, indent=1))
    print("TESSERA_ORACLE_JSON " + json.dumps(out, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
