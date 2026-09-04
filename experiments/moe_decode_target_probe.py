#!/usr/bin/env python3
"""Ask the pinned build which tile a Tessera routed-MoE route may decode to.

WHY THIS EXISTS.  ``docs/measurements/tessera-moe-wire-loader-2026-09-03.md``
settled the LOAD side of #5's expert route -- a wire parameter needs its own
``weight_loader``, established before anything registered one.  This probe is
the same question one step later, and it decides the route rather than
decorating it: **what may the decode land in?**

Tessera's dense ``TESSERA_FP8`` route decodes a window body over the CHANNEL
scale plane to a *per-channel* FP8 pair -- E4M3 bytes plus one fp32 scale per
output row -- and serves it W8A8 through ``torch._scaled_mm``
(``scheme.ROUTES[TESSERA_FP8]``).  An expert route that could not keep that
granularity would have to fold 2048 row scales into one per expert, and the
scale plane is not decoration: deleting it costs 0.77-0.84x
(``docs/measurements/tessera-window-body-2026-09-02.md`` and the
scale-plane-buys-column-structure note).  So whether the runtime's fused-MoE
kernels accept a per-channel weight scale is a fact about the *quality* of the
route, not only its plumbing -- and it is a fact about another runtime, so it
is read off that runtime's own table (AGENTS.md 6, principle 14).

WHAT IT ASKS.  ``select_fp8_moe_backend`` chooses a backend by asking every
candidate kernel class ``is_supported_config(config, weight_key,
activation_key, activation_format) -> (supported, reason)``.  That predicate IS
the machine-readable table: it is the same call the runtime makes to decide,
and its ``reason`` string is the runtime's own words for a refusal.  So this
probe asks it directly, for three weight/activation pairs, over every backend
the build lists:

* ``(kFp8StaticChannelSym, kFp8DynamicTokenSym)`` -- what Tessera's FP8 route
  decodes to and executes today, per Linear.  The question.
* ``(kFp8StaticTensorSym, kFp8DynamicTensorSym)`` -- the stock per-tensor arm
  ``Fp8MoEMethod`` selects for a non-block checkpoint.  A control: if this
  reports unsupported too, the harness is wrong, not the route.
* ``(kFp8Static128BlockSym, kFp8Dynamic128Sym)`` -- the stock block arm, the
  second control, and the fallback granularity if per-channel is refused.

and records the parameters ``Fp8MoEMethod.create_weights`` registers, which is
the tile shape a route's ``process_weights_after_loading`` must end up holding.

SCOPE.  This is this build's answer for these keys, on the machine it runs on.
``is_supported_config`` may consult the device, so the run records
``torch.cuda.is_available()`` and the device name: an answer taken without a
GPU is a claim about the table, not about sm_121, and the receipt says which
one it has.  It is not a claim that any route exists -- there is still no
``ROUTES`` entry and no ``routed_moe`` in ``scheme.STRUCTURES``.
"""
from __future__ import annotations

import dataclasses
import json
import sys

import torch

EXPERTS, HIDDEN, INTERMEDIATE, TOPK = 8, 4096, 2048, 8
LAYER = "model.language_model.layers.1.mlp.experts"


def _keys():
    """The three (weight, activation) pairs, from the runtime's own constants."""
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym, kFp8DynamicTensorSym, kFp8DynamicTokenSym,
        kFp8Static128BlockSym, kFp8StaticChannelSym, kFp8StaticTensorSym)

    return {
        "per_channel_w_per_token_a": (kFp8StaticChannelSym, kFp8DynamicTokenSym),
        "per_tensor_w_per_tensor_a": (kFp8StaticTensorSym, kFp8DynamicTensorSym),
        "block128_w_block128_a": (kFp8Static128BlockSym, kFp8Dynamic128Sym),
    }


def _config(moe_backend: str = "auto"):
    """A ``FusedMoEConfig`` at this model's real MoE dimensions."""
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig, FusedMoEParallelConfig, MoEActivation)
    from vllm.model_executor.layers.fused_moe import RoutingMethodType

    parallel = FusedMoEParallelConfig(
        tp_size=1, tp_rank=0, pcp_size=1, pcp_rank=0, dp_size=1, dp_rank=0,
        ep_size=1, ep_rank=0, sp_size=1, use_ep=False,
        all2all_backend="naive", enable_eplb=False)
    return FusedMoEConfig(
        num_experts=EXPERTS, experts_per_token=TOPK, hidden_dim=HIDDEN,
        intermediate_size=INTERMEDIATE, num_local_experts=EXPERTS,
        num_logical_experts=EXPERTS, activation=MoEActivation.SILU,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=parallel, in_dtype=torch.bfloat16,
        intermediate_size_per_partition=INTERMEDIATE, moe_backend=moe_backend)


def vocabulary_leg() -> dict:
    """The granularity vocabulary the runtime publishes, as values."""
    from vllm.model_executor.layers.fused_moe.routed_experts import (
        FusedMoeWeightScaleSupported)
    from vllm.model_executor.layers.quantization.utils import quant_utils

    keys = _keys()
    return {
        "weight_scale_granularities_the_runtime_names": [
            m.value for m in FusedMoeWeightScaleSupported],
        "fp8_quant_keys_published": sorted(
            n for n in dir(quant_utils) if n.startswith("kFp8")),
        "pairs_asked": {name: {"weight_key": str(w), "activation_key": str(a)}
                        for name, (w, a) in keys.items()},
    }


def backend_leg() -> dict:
    """Every fp8 MoE backend this build lists, and its kernel classes."""
    import vllm.model_executor.layers.fused_moe.oracle.fp8 as mod

    backends = [b.value for b in mod.Fp8MoeBackend]
    available = [b.value for b in getattr(mod, "AVAILABLE_BACKENDS", mod.Fp8MoeBackend)]
    classes = {}
    for backend in mod.Fp8MoeBackend:
        try:
            classes[backend.value] = [c.__name__ for c in mod.backend_to_kernel_cls(backend)]
        except Exception as exc:  # noqa: BLE001 -- an unanswered row is a recorded one
            classes[backend.value] = f"{type(exc).__name__}: {exc}"
    return {"module": mod.__name__, "declared": backends, "available": available,
            "kernel_classes": classes}


def supported_leg() -> dict:
    """``is_supported_config`` for each (backend kernel class, pair, format).

    The runtime's own predicate, called the way ``select_fp8_moe_backend``
    calls it -- unbound, with the class as the first argument, which is how
    that function invokes it.
    """
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    import vllm.model_executor.layers.fused_moe.oracle.fp8 as mod

    keys = _keys()
    rows = []
    for backend in mod.Fp8MoeBackend:
        try:
            kernel_classes = list(mod.backend_to_kernel_cls(backend))
        except Exception as exc:  # noqa: BLE001
            rows.append({"backend": backend.value, "kernel_class": None,
                         "raised": f"{type(exc).__name__}: {exc}"})
            continue
        for k_cls in kernel_classes:
            for fmt_name in ("Standard", "BatchedExperts"):
                fmt = getattr(mk.FusedMoEActivationFormat, fmt_name)
                for pair, (weight_key, activation_key) in keys.items():
                    row = {"backend": backend.value, "kernel_class": k_cls.__name__,
                           "activation_format": fmt_name, "pair": pair}
                    try:
                        supported, reason = k_cls.is_supported_config(
                            k_cls, _config(), weight_key, activation_key, fmt)
                        row.update({"supported": bool(supported),
                                    "reason": None if reason is None else str(reason)})
                    except Exception as exc:  # noqa: BLE001
                        row.update({"supported": None,
                                    "raised": f"{type(exc).__name__}: {exc}"})
                    rows.append(row)
    return {"rows": rows}


class _Layer(torch.nn.Module):
    """A stand-in carrying only what ``create_weights`` reads off the layer."""

    def __init__(self, moe_config):
        super().__init__()
        self.moe_config = moe_config
        self.orig_dtype = torch.bfloat16
        self.layer_name = LAYER


def params_leg() -> dict:
    """What ``Fp8MoEMethod.create_weights`` registers: the tile to decode to.

    Executed, not read: the method is built with ``object.__new__`` and only
    the attributes ``create_weights`` consults are set, so the shapes recorded
    are the ones the real code computes.  The per-tensor arm is the one
    ``Fp8MoEMethod`` selects for a non-block fp8 checkpoint.

    ``loader_is_the_caller_s`` is recorded because it is a fact about the
    runtime's CONVENTION and not about its choice: ``create_weights`` takes one
    ``weight_loader`` argument and puts that same object on every parameter it
    registers.  Whether it is set is therefore not evidence of anything -- this
    probe passed a lambda, so of course it is there.  What the field pins is
    that the loader on an expert parameter is whatever the caller handed in,
    which is why a Tessera expert route can give each wire parameter its own
    loader (§6 item 2) without fighting the runtime for the slot.
    """
    from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

    out = {}
    for arm, block_size in (("per_tensor", None), ("block128", [128, 128])):
        method = object.__new__(Fp8MoEMethod)
        method.moe = _config()
        method.weight_block_size = block_size
        method.block_quant = block_size is not None
        method.weight_scale_name = "weight_scale_inv" if block_size else "weight_scale"
        method.quant_config = type("Cfg", (), {
            "is_checkpoint_fp8_serialized": True,
            "activation_scheme": "dynamic",
            "weight_block_size": block_size})()
        layer = _Layer(method.moe)
        loader = lambda *a, **k: None  # noqa: E731 -- identity is the point
        record = {"registered": {}, "raised": None}
        try:
            Fp8MoEMethod.create_weights(
                method, layer, num_experts=EXPERTS, hidden_size=HIDDEN,
                intermediate_size_per_partition=INTERMEDIATE,
                params_dtype=torch.float8_e4m3fn, weight_loader=loader)
            for name, param in layer.named_parameters(recurse=False):
                record["registered"][name] = {
                    "shape": list(param.shape), "dtype": str(param.dtype),
                    # NOT a runtime choice: see the docstring.  True here means
                    # "create_weights propagated the lambda this probe passed".
                    "loader_is_the_caller_s": getattr(param, "weight_loader", None) is loader}
        except Exception as exc:  # noqa: BLE001
            record["raised"] = f"{type(exc).__name__}: {exc}"
        out[arm] = record
    return out


def _called(value, *args):
    """A value, or what calling it returned, or why it could not be called."""
    if not callable(value):
        return None if value is None else str(value)
    try:
        return str(value(*args))
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def stock_keys_leg() -> dict:
    """Which weight keys the STOCK method ever asks for -- read off its own source.

    ``supported_leg`` answers "would a kernel take a per-channel weight scale".
    That is necessary and not sufficient: a kernel that would take one is never
    handed one unless some method asks.  ``Fp8MoEMethod.__init__`` picks its
    ``weight_key`` from a fixed pair, and this leg records which
    ``QuantKey`` constants appear in that function's body, quoting the runtime's
    own code rather than describing it.  If ``kFp8StaticChannelSym`` is absent,
    a Tessera expert route cannot inherit the stock selection: it has to call
    ``select_fp8_moe_backend`` itself with the channel key (§6 item 3).
    """
    import inspect
    import re

    from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

    out = {}
    for fn in ("__init__", "get_fused_moe_quant_config"):
        member = getattr(Fp8MoEMethod, fn, None)
        if member is None:
            out[fn] = {"present": False}
            continue
        try:
            source = inspect.getsource(member)
        except (OSError, TypeError) as exc:  # noqa: PERF203
            out[fn] = {"present": True, "source_unavailable": str(exc)}
            continue
        out[fn] = {"present": True,
                   "quant_keys_named": sorted(set(re.findall(r"\bkFp8\w+", source))),
                   "source_lines": len(source.splitlines())}
    return out


def quant_config_leg() -> dict:
    """Can the runtime's own fp8 MoE quant config CARRY a per-channel scale?

    Answered without a device, and it is the half of the per-channel question
    that does not need one: ``fp8_w8a8_moe_quant_config`` takes
    ``per_out_ch_quant`` and ``per_act_token_quant`` as arguments, so building
    one with per-row weight scales and reading back what it reports is a fact
    about the config object.  Whether a KERNEL will then take it is
    ``supported_leg``'s question, and that one does need the device.
    """
    from vllm.model_executor.layers.fused_moe.config import (
        fp8_w8a8_moe_quant_config)

    experts, hidden, inter = EXPERTS, HIDDEN, INTERMEDIATE
    arms = {
        "per_out_channel_w_per_token_a": dict(
            w1_scale=torch.ones(experts, 2 * inter, 1, dtype=torch.float32),
            w2_scale=torch.ones(experts, hidden, 1, dtype=torch.float32),
            per_out_ch_quant=True, per_act_token_quant=True),
        "per_tensor_w_per_tensor_a": dict(
            w1_scale=torch.ones(experts, 2, dtype=torch.float32),
            w2_scale=torch.ones(experts, dtype=torch.float32),
            per_out_ch_quant=False, per_act_token_quant=False),
    }
    out = {}
    for arm, kwargs in arms.items():
        record = {"asked": {k: (list(v.shape) if isinstance(v, torch.Tensor) else v)
                            for k, v in kwargs.items()}}
        try:
            config = fp8_w8a8_moe_quant_config(**kwargs)
            name = getattr(config, "config_name", None)
            record["reports"] = {
                # ``config_name`` is a method on this build, not a property, so
                # stringifying the attribute dumps the whole dataclass -- call
                # it, and record what it says it could not answer if it raises.
                # ``config_name`` takes the activation dtype -- the name the
                # runtime gives this configuration is a function of it.
                "config_name": _called(name, torch.bfloat16),
                "quant_dtype": str(getattr(config, "quant_dtype", None)),
                "use_fp8_w8a8": bool(config.use_fp8_w8a8),
                "per_out_ch_quant": bool(config.per_out_ch_quant),
                "per_act_token_quant": bool(config.per_act_token_quant),
                "is_per_tensor": bool(config.is_per_tensor),
                "is_per_act_token": bool(config.is_per_act_token),
                "is_block_quantized": bool(config.is_block_quantized),
                "w1_scale_shape": list(config.w1_scale.shape),
                "w2_scale_shape": list(config.w2_scale.shape),
            }
        except Exception as exc:  # noqa: BLE001 -- a refusal IS the answer
            record["raised"] = f"{type(exc).__name__}: {exc}"
        out[arm] = record
    return out


def interface_leg() -> dict:
    """What a routed-MoE quant method must supply on this build.

    ``__abstractmethods__`` is the runtime's own declaration; the rest is
    execution -- a base method that raises when called is one a subclass must
    write, and calling it is how that is established rather than asserted.
    """
    import inspect

    from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase

    # A CONCRETE subclass, because the base is an ABC: filling only the two
    # abstract methods is what lets the rest be called, and what is then
    # called is the base's own body.
    concrete = type("_Concrete", (FusedMoEMethodBase,), {
        "create_weights": lambda *a, **k: None,
        "get_fused_moe_quant_config": lambda *a, **k: None,
    })
    must_write = {}
    for name in ("apply", "create_weights", "process_weights_after_loading",
                 "get_fused_moe_quant_config", "apply_monolithic"):
        function = getattr(FusedMoEMethodBase, name, None)
        if function is None:
            must_write[name] = "absent on the base"
            continue
        entry = {"signature": str(inspect.signature(function)),
                 "overridden_on_the_base": name not in FusedMoEMethodBase.__abstractmethods__}
        try:
            function(object.__new__(concrete), *([None] * (
                len(inspect.signature(function).parameters) - 1)))
            entry["called_bare"] = "returned"
        except NotImplementedError:
            entry["called_bare"] = "NotImplementedError -- a subclass must write it"
        except Exception as exc:  # noqa: BLE001
            entry["called_bare"] = f"{type(exc).__name__}: {exc}"
        must_write[name] = entry
    return {"abstractmethods": sorted(FusedMoEMethodBase.__abstractmethods__),
            "methods": must_write}


def main() -> int:
    import vllm

    record = {
        "vllm": getattr(vllm, "__version__", "unknown"),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "dimensions": {"experts": EXPERTS, "hidden": HIDDEN,
                       "intermediate": INTERMEDIATE, "topk": TOPK},
    }
    for name, leg in (("vocabulary", vocabulary_leg), ("backends", backend_leg),
                      ("interface", interface_leg), ("params", params_leg),
                      ("quant_config", quant_config_leg), ("stock_keys", stock_keys_leg),
                      ("supported", supported_leg)):
        try:
            record[name] = leg()
        except Exception as exc:  # noqa: BLE001 -- an unanswered leg is a recorded one
            record[name] = {"raised": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(record, indent=2, default=str))
    print("TESSERA_MOE_DECODE_JSON " + json.dumps(record, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
