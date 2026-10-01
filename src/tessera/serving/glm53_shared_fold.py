"""Fold the MoE shared-expert add into the fused routed window's token sum.

Each vLLM MoE layer ends with ``result = shared_output + fused_output``
(``MoERunner.forward``, ``moe_runner.py:786``). That is a bf16 ``[T, H]`` add.
On a 2048-token GLM-5.3 prefill chunk at TP2 on GB10 it takes 195 us, once per
MoE layer, 42 times a chunk (A8SESHMN served trace, 2026-10-01). The routed half
comes from Tessera's fused routed window (``tessera.routed_fused``), whose last
kernel, ``token_sum``, already writes the routed output once. With this lever
on, ``token_sum`` also adds the shared output as it stores, and the runner skips
its own add.

The fold is bitwise. The stock arithmetic is:

- ``token_sum`` stores ``fused = bf16_rn(sum_j f32(routed_j))``;
- ATen's add then stores ``bf16_rn(f32(shared) + 1.0f * f32(fused))``, and its
  bf16 store on ``__CUDA_ARCH__ >= 800`` is ``__float2bfloat16``.

The folded kernel computes ``bf16_rn(f32(shared) + f32(bf16_rn(acc)))`` with the
same intrinsic: the same two roundings, in the same order.
``experiments/t8r_speed/shared_fold_check.py`` checks this on the GPU.

The change is opt-in: set ``TESSERA_GLM53_FOLD_SHARED_ADD=1``.
``TesseraConfig.get_quant_method`` installs it.

Install conditions. The install declines, and the serve keeps the stock path,
if any of these fails:

- the source sha256 of ``runner/moe_runner.py`` and of ``runner/shared_experts.py``
  is one this module was inspected against;
- the serve runs with compilation mode ``NONE``. A traced runner forward would
  not see this module's hand-off at run time.

How the fold works:

1. The routed method's ``_apply_native`` calls its adapter through
   :func:`native_call`. That function passes the shared output to the adapter
   only when every condition below holds:
   - the adapter supports the fold;
   - the runner was checked on its first forward;
   - the shared experts ran on the current stream before the routed call
     (``NO_OVERLAP``);
   - the routed input isn't padded;
   - the shared output is a contiguous bf16 ``[T, H]`` tensor.
2. When it folds, it records the fold.
3. The rebound ``MoERunner._maybe_apply_routed_scale_to_output`` runs just
   before the add. It consumes that record and returns ``(None, fused)``, so the
   runner takes its no-shared branch. A record that doesn't match the tensors
   the runner holds raises, so a folded output is never added twice.

Each process logs exactly one install line: ``installed``, ``declined`` (with
the reason) or ``off``. It also logs one line for the first folded call and one
for the first kept call (with the reason).
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import logging
import os
import threading
from pathlib import Path
from threading import RLock
from typing import Any

import torch

from .flags import latched_bool

__all__ = ["FLAG", "install_for_current_config", "native_call", "runner_decline_reason"]

_log = logging.getLogger(__name__)

FLAG = "TESSERA_GLM53_FOLD_SHARED_ADD"

_RUNNER_MODULE = "vllm.model_executor.layers.fused_moe.runner.moe_runner"
_SHARED_MODULE = "vllm.model_executor.layers.fused_moe.runner.shared_experts"

#: sha256 of each stock source as inspected: image
#: localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a
#: (vLLM 0.30.1rc1.dev336+gaf5b4857e.d20260929). ``MoERunner.forward`` calls
#: ``_maybe_apply_routed_scale_to_output`` once, just before
#: ``result = shared_output + fused_output``. ``SharedExperts`` keeps its
#: output in ``_output[_output_idx]`` until the runner pops it.
_INSPECTED_SHA256 = {
    _RUNNER_MODULE: frozenset({
        "2c6e1fffba1f4fd2e4c5d3981c4d26f7c0de802fc9c35865748b113c30235da5",
    }),
    _SHARED_MODULE: frozenset({
        "59cd2cf13c1e0d6301566f5b415807bb674b6786336d394c01c1473b4740e314",
    }),
}

_SIGNATURE = ("self", "shared_output", "fused_output")

_MARK = "_tessera_fold_shared_add"
#: Set on a runner's ``SharedExperts`` on the runner's first forward: True when
#: the runner admits the fold, else the reason it doesn't.
_RUNNER_MARK = "_tessera_shared_fold"

_LEVER = "tessera.glm53_shared_fold: shared add fold"

_LOCK = RLock()
#: The one install line this process logged ("installed", "declined" or "off").
_REPORTED: list[str] = []
#: Which first-call lines ("folded", "kept") this process has logged.
_FIRST: set[str] = set()
#: Install state: ``installed`` and the stock ``NO_OVERLAP`` order once installed.
_STATE: dict[str, Any] = {}
#: The fold the routed call made and the runner hasn't consumed yet.
_PENDING = threading.local()


def _report(line: str) -> None:
    if not _REPORTED:
        _REPORTED.append(line)
        _log.warning(line)


def _first(kind: str, line: str) -> None:
    if kind not in _FIRST:
        _FIRST.add(kind)
        _log.warning("%s: %s", _LEVER, line)


def _identity(t: Any) -> tuple | None:
    if not isinstance(t, torch.Tensor):
        return None
    return (t.data_ptr(), tuple(t.shape), t.dtype, t.device)


# -- the runner side ---------------------------------------------------------

def runner_decline_reason(runner: Any) -> str | None:
    """None when ``runner``'s combine is the plain add the fold replaces; else why not.

    Read once per runner, on its first forward, from the runner's own
    configuration.
    """
    if getattr(runner, "_shared_experts", None) is None:
        return "the runner has no shared experts"
    scale = getattr(runner, "routed_scaling_factor", None)
    if scale != 1.0:
        return f"routed_scaling_factor is {scale!r}, not 1.0"
    if getattr(runner, "_fused_output_is_reduced", True):
        return "the routed kernel reduces its own output, so the shared output is reduced first"
    if getattr(runner, "routed_input_transform", None) is not None \
            or getattr(runner, "routed_output_transform", None) is not None:
        return "a routed input or output transform is set"
    quant = getattr(runner, "_quant_method", None)
    if getattr(quant, "has_unpadded_output", True):
        return "the quant method may return unpadded output"
    if getattr(runner, "do_naive_dispatch_combine", True):
        return "naive dispatch/combine is on"
    config = getattr(runner, "moe_config", None)
    if getattr(config, "pcp_size", 1) != 1:
        return f"prefill context parallel size is {getattr(config, 'pcp_size', None)!r}"
    if getattr(runner._shared_experts, "enable_dbo", True):
        return "dual-batch overlap is on"
    return None


def _wrap(original: Any) -> Any:
    @functools.wraps(original)
    def _maybe_apply_routed_scale_to_output(self, shared_output, fused_output):
        shared_experts = getattr(self, "_shared_experts", None)
        if shared_experts is not None and getattr(shared_experts, _RUNNER_MARK, None) is None:
            reason = runner_decline_reason(self)
            setattr(shared_experts, _RUNNER_MARK, True if reason is None else reason)
        pending = getattr(_PENDING, "fold", None)
        if pending is None:
            return original(self, shared_output, fused_output)
        _PENDING.fold = None
        if (_identity(shared_output), _identity(fused_output)) != pending:
            raise RuntimeError(
                f"{_LEVER}: the routed call folded shared {pending[0]} into {pending[1]}, but "
                f"the runner holds shared {_identity(shared_output)} and routed "
                f"{_identity(fused_output)}; refusing to add the shared output twice")
        return None, fused_output

    setattr(_maybe_apply_routed_scale_to_output, _MARK, True)
    _maybe_apply_routed_scale_to_output.__wrapped_stock__ = original
    return _maybe_apply_routed_scale_to_output


# -- the routed side ---------------------------------------------------------

def _shared_for_fold(adapter: Any, x: Any, shared_experts: Any,
                     shared_experts_input: Any) -> tuple[Any, str | None]:
    """``(shared, None)`` to fold, ``(None, reason)`` to keep, ``(None, None)`` to stay quiet."""
    if not _STATE.get("installed"):
        return None, None
    if shared_experts is None:
        return None, "no shared experts"
    mark = getattr(shared_experts, _RUNNER_MARK, None)
    if mark is None:
        return None, None  # the runner's first forward: checked after this call
    if mark is not True:
        return None, str(mark)
    if not getattr(adapter, "supports_shared_fold", False):
        return None, f"the routed adapter {type(adapter).__name__} has no shared fold"
    if torch.compiler.is_compiling():
        return None, "the forward is being traced"
    order = shared_experts._determine_shared_experts_order(shared_experts_input)
    if order != _STATE["no_overlap"]:
        return None, f"shared experts order is {getattr(order, 'name', order)}, not NO_OVERLAP"
    if not isinstance(x, torch.Tensor) or x.dim() != 2 or x.shape[0] == 0:
        return None, "the routed input is not a non-empty [T, H] tensor"
    if not isinstance(shared_experts_input, torch.Tensor) \
            or shared_experts_input.shape[-1] != x.shape[-1]:
        return None, "the routed input is padded or transformed"
    slots = getattr(shared_experts, "_output", None)
    shared = slots[shared_experts._output_idx] if isinstance(slots, list) else None
    if not isinstance(shared, torch.Tensor):
        return None, "the shared output isn't computed before the routed call"
    tokens, hidden = x.shape
    if shared.dtype != torch.bfloat16 or tuple(shared.shape) != (tokens, hidden) \
            or not shared.is_contiguous() or shared.device != x.device \
            or shared.data_ptr() % 16:
        return None, (f"the shared output is {shared.dtype} {tuple(shared.shape)} "
                      f"(contiguous {shared.is_contiguous()}), not 16-byte aligned contiguous "
                      f"bf16 {(tokens, hidden)} on {x.device}")
    return shared, None


def native_call(adapter: Any, x: Any, expert_ids: Any, routing_weights: Any, *,
                shared_experts: Any = None, shared_experts_input: Any = None, **kwargs: Any) -> Any:
    """Call the routed adapter, folding the shared output in when the lever admits it.

    With the fold declined (or the lever off), this is exactly
    ``adapter(x, expert_ids, routing_weights, **kwargs)``.
    """
    shared, reason = _shared_for_fold(adapter, x, shared_experts, shared_experts_input)
    if shared is None:
        if reason is not None:
            _first("kept", f"first kept call {tuple(x.shape)}: shared add kept ({reason})")
        return adapter(x, expert_ids, routing_weights, **kwargs)
    if getattr(_PENDING, "fold", None) is not None:
        raise RuntimeError(f"{_LEVER}: a folded shared add from an earlier routed call was never "
                           "consumed by its runner")
    out = adapter(x, expert_ids, routing_weights, shared=shared, **kwargs)
    _PENDING.fold = (_identity(shared), _identity(out))
    _first("folded", f"first folded call {tuple(x.shape)}: shared add folded into token_sum")
    return out


# -- install -----------------------------------------------------------------

def _source_digest(module: Any) -> str | None:
    path = getattr(module, "__file__", None)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None


def _compilation_reason() -> str | None:
    try:
        from vllm import config as vllm_config
        getter = getattr(vllm_config, "get_current_vllm_config_or_none", None)
        config = getter() if getter is not None else vllm_config.get_current_vllm_config()
    except Exception as exc:  # noqa: BLE001 -- any failure to read it is a decline
        return f"the vLLM config can't be read ({type(exc).__name__}: {exc})"
    mode = getattr(getattr(config, "compilation_config", None), "mode", None)
    name = getattr(mode, "name", None)
    if name != "NONE":
        return (f"compilation mode is {name or mode!r}, not NONE; a traced runner forward "
                "would not see the routed call's hand-off")
    return None


def _decline_reason(runner_module: Any, shared_module: Any) -> str | None:
    for name, module in ((_RUNNER_MODULE, runner_module), (_SHARED_MODULE, shared_module)):
        digest = _source_digest(module)
        if digest is None:
            return f"{name} has no source file to inspect"
        if digest not in _INSPECTED_SHA256[name]:
            return f"{name} sha256 {digest[:12]} is not an inspected source"
    runner = getattr(runner_module, "MoERunner", None)
    if runner is None:
        return f"{_RUNNER_MODULE} has no MoERunner"
    params = tuple(inspect.signature(runner._maybe_apply_routed_scale_to_output).parameters)
    if params != _SIGNATURE:
        return (f"MoERunner._maybe_apply_routed_scale_to_output has parameters {params}, "
                f"expected {_SIGNATURE}")
    order = getattr(shared_module, "SharedExpertsOrder", None)
    shared = getattr(shared_module, "SharedExperts", None)
    if getattr(order, "NO_OVERLAP", None) is None or shared is None \
            or not hasattr(shared, "_determine_shared_experts_order") \
            or not hasattr(shared, "_output_idx"):
        return f"{_SHARED_MODULE} lacks SharedExpertsOrder.NO_OVERLAP or the SharedExperts slots"
    return _compilation_reason()


def install_for_current_config() -> bool:
    """Rebind the runner's pre-add hook when the flag is on; True if installed.

    Idempotent. Logs exactly one line per process, at the first call:
    ``installed (...)``, ``declined, stock ...: reason`` or ``off (...)``.
    """
    if not latched_bool(FLAG, meaning="folding the MoE shared-expert add into token_sum"):
        _report(f"{_LEVER} off ({FLAG} unset or 0)")
        return False
    with _LOCK:
        if _REPORTED and not _STATE.get("installed"):
            return False  # declined at the first call; the decision stands for the process
        try:
            runner_module = importlib.import_module(_RUNNER_MODULE)
            shared_module = importlib.import_module(_SHARED_MODULE)
        except ImportError as exc:
            reason = f"the stock MoE runner is not importable ({exc})"
        else:
            runner = getattr(runner_module, "MoERunner", None)
            if runner is not None and getattr(runner._maybe_apply_routed_scale_to_output, _MARK,
                                              False) and _STATE.get("installed"):
                return True
            reason = _decline_reason(runner_module, shared_module)
        if reason is not None:
            _report(f"{_LEVER} declined, stock MoERunner._maybe_apply_routed_scale_to_output: "
                    f"{reason}")
            return False
        runner = runner_module.MoERunner
        stock = runner._maybe_apply_routed_scale_to_output
        stock = getattr(stock, "__wrapped_stock__", stock)
        runner._maybe_apply_routed_scale_to_output = _wrap(stock)
        _STATE.update(installed=True, no_overlap=shared_module.SharedExpertsOrder.NO_OVERLAP)
        image = os.environ.get("TESSERA_CENSUS_RUNTIME_IMAGE", "").rpartition("@sha256:")[2][:12]
        _report(f"{_LEVER} installed (stock source sha256 {_source_digest(runner_module)[:12]} "
                f"and {_source_digest(shared_module)[:12]}, image sha {image or 'unstated'}): "
                "MoERunner skips the shared add when token_sum already made it")
        return True
