"""GLM-5.3 (Glm5Next) under CUDA graphs: the graph serve computes what the eager serve computes.

tessera#702 measured three places where a CUDA-graph serve of GLM-5.3 on the
release image leaves eager's arithmetic, each confirmed by a controlled pair
(``docs/measurements/2026-09-30-glm-nightly-cells-and-graph-equivalence.md``,
``docs/measurements/2026-10-04-glm-graph-equals-eager.md``):

1. **Operator resolution.**  With ``enforce_eager=False`` vLLM defaults to
   compilation mode ``VLLM_COMPILE``, which resolves ``custom_ops`` to
   ``["none"]`` and the two RMS-norm IR ops to ``native``; eager resolves
   ``["all"]`` and ``["vllm_c", "native"]``.  Glm5Next is not torch-compiled,
   so the mode buys nothing and the operators are the whole difference (sbG4:
   default mode with ``custom_ops ['all']`` and eager's IR priority returns
   every response of the ``mode NONE`` arm byte for byte).
   :func:`pin_eager_operators` sets exactly those two fields on a Tessera
   Glm5Next graph serve, from ``MODELS_CONFIG_MAP`` (vLLM's per-architecture
   ``verify_and_update_config`` hook, which runs before vLLM resolves either
   default).  It only fills what the serve left unset; a serve that named
   other operators keeps them and is refused in the worker, where
   ``graph_equivalence.op_implementation_gap`` -- the one home of that rule --
   reads the resolved config.

2. **The indexer's frozen branch.**  A FULL capture builds its attention
   metadata at ``max_seq_len = max_model_len`` (``model_states/
   mamba_hybrid.py``, "so the graph is valid at any replay").  The GLM
   indexer takes a host-side branch on that value
   (``models/glm5next/common/sparse_indexer.py``,
   ``_fill_short_decode_causal_indices``): at ``max_seq_len <= index_topk`` it
   fills causal indices and skips logits and top-k.  A replay re-runs no
   Python, so every graph holds the top-k side and replays it at contexts
   where eager takes the fill.  :func:`install_branch_capture` captures one
   set of FULL graphs per class of ``max_seq_len``
   (``graph_equivalence.branch_capture_bounds``: one class per side of
   ``index_topk``, each captured at its own upper end) and replays the class
   that holds the step's own ``max_seq_len``, read from the very argument
   every metadata builder receives.

3. **The padded replay.**  vLLM's default capture sizes (``[1, 2, 4, 8,
   ...]``) replay a decode batch in the smallest graph that holds it; at
   ``max_model_len`` 8448 that moved the padded batches b5 and b7 off eager,
   where capturing every size kept all 48 choices.
   :func:`pin_unpadded_capture_sizes` captures every uniform-decode count where
   the serve named no sizes, from the same hook as cause 1.

:func:`eager_gaps` judges causes 1 and 3 on every Tessera Glm5Next graph serve,
and refuses one that still has a gap; :func:`branch_plan` decides cause 2.

The class boundaries are every host-side choice on ``max_seq_len`` inside the
captured decode step, read from the inspected interface:

- the indexer's ``max_seq_len <= index_topk`` (``topk_tokens`` is the HF
  text config's ``index_topk`` in the target and in the MTP draft);
- ``persistent_topk`` (vLLM ``csrc/libtorch_stable/topk.cu``), the top-k the
  SM12x family resolves (``cooperative`` excludes it): its sampled path needs
  more than :data:`PERSISTENT_TOPK_SAMPLED_MIN_ROWS` rows, and its radix path
  ``max_seq_len >`` :data:`PERSISTENT_TOPK_RADIX_THRESHOLD`.  The install
  refuses a serve that could reach either, so neither splits a class here.

Both changes are no-ops for an eager serve and for a graph serve whose
``max_model_len`` is at most ``index_topk`` (one class, captured where vLLM
captures).  A Tessera Glm5Next graph serve that needs the second and cannot
have it -- an uninspected interface, or a structure outside what was read --
is refused by name: it would serve another computation than the eager one
its cells were measured on.
"""
from __future__ import annotations

import logging
import threading
from collections import Counter
from types import SimpleNamespace
from typing import Any

from .glm53_prefill import is_glm5next
from .graph_equivalence import (EAGER_IR_OPS, branch_bound, branch_capture_bounds,
                                graph_mode, num_draft_tokens, op_implementation_gap,
                                padded_families)
from .stock_interface import InspectedInterface as _Interface
from .stock_interface import import_modules, match_modules

_log = logging.getLogger(__name__)

#: ``quantization_config.quant_method`` of a Tessera checkpoint (``serving.QUANT_METHOD``).
_QUANT_METHOD = "tessera"

#: The modules the per-class capture rebinds, reads or relies on, in ``_Interface.digests`` order.
GRAPH_MODULES = (
    "vllm.v1.worker.gpu.cudagraph_utils",
    "vllm.v1.worker.gpu.model_states.mamba_hybrid",
    "vllm.v1.worker.gpu.model_runner",
    "vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils",
    "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator",
    "vllm.models.glm5next.common.sparse_indexer",
    "vllm.models.glm5next.nvidia.sparse_indexer",
    "vllm.model_executor.layers.indexer_topk",
)

#: sha256 of each module in GRAPH_MODULES order, read inside the release image
#: ``localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705...`` (vLLM
#: 0.30.1rc1.dev336+gaf5b4857e); refresh by re-reading the image whenever one moves.
_INTERFACES = (
    _Interface("nightly-20260929", (
        "cc090e6749029baaa5055135215acc112ce034a32fc1e886f0f02054ad488b58",
        "1ddd7dc921d74789e77d3319dc167820ad7fc47ed51d8d9a35ba3938f5c6a079",
        "217e4b87c0bd22c4e98640775d7b4e3349000f79d37f9f7d3e569768786c3a7f",
        "42ee43d95882608576c7a98e8c64865eb4053838d0c647661469d00dc71aa0b1",
        "c40e801436bc78a77624e1d2c688db823a21b73ffc65f75d18ad78ffeef18118",
        "a3ab1edda8490b8e21c1c240e07e8c8fcd0bb34246a9ed1f64acfe067d15067c",
        "549f94234e44000c0b9995745328fd262b7ff69e62c6089ad049f65a7573914d",
        "977faa30133c5e0f997ceaae22adfad1ffbee35f48f64ab47b066f779fea8de7",
    )),
)

#: ``persistent_topk`` takes its sampled path only above this many rows
#: (``num_rows > 64``, vLLM af5b4857e ``csrc/libtorch_stable/topk.cu``).
PERSISTENT_TOPK_SAMPLED_MIN_ROWS = 64
#: ``persistent_topk``'s radix path engages above this ``max_seq_len``
#: (``RADIX_THRESHOLD``, ``csrc/libtorch_stable/persistent_topk.cuh``).
PERSISTENT_TOPK_RADIX_THRESHOLD = 32768

#: Replays per (manager class, bound), and FULL graphs captured per (manager class, bound):
#: host-side counts a serve's evidence reads to show each class was captured and replayed.
REPLAYS: Counter = Counter()
CAPTURED: dict[str, dict[int, int]] = {}

_LOCK = threading.Lock()
_STATE = SimpleNamespace(installed=False, bounds=None, capture_bound=None, bound_applied=False,
                         step_max_seq_len=None)


# ------------------------------------------------------------------ cause 1: operators


def _is_tessera(config: Any) -> bool:
    return getattr(getattr(config, "model_config", None), "quantization", None) == _QUANT_METHOD


def _graphs_requested(config: Any) -> bool:
    return not getattr(getattr(config, "model_config", None), "enforce_eager", False)


def eager_ir_priority(config: Any) -> dict[str, list[str]]:
    """The IR-op priority vLLM resolves without codegen, which the eager serve runs.

    Asked of vLLM itself, through the path an eager serve resolves by:
    ``KernelConfig.set_platform_defaults`` on a fresh kernel config, for this
    serve's backend at compilation mode NONE.  A platform default that moves
    moves this with it.
    """
    from vllm.config.compilation import CompilationMode
    from vllm.config.kernel import KernelConfig

    no_codegen = SimpleNamespace(
        model_config=None,
        compilation_config=SimpleNamespace(backend=config.compilation_config.backend,
                                           mode=CompilationMode.NONE))
    kernel = KernelConfig()
    kernel.set_platform_defaults(no_codegen)
    return {op: list(getattr(kernel.ir_op_priority, op)) for op in EAGER_IR_OPS}


def pin_eager_operators(config: Any, eager_priority=eager_ir_priority) -> list[str]:
    """Fill the operator fields a Tessera Glm5Next graph serve left unset with eager's.

    Returns what it set, for the log. Must run before vLLM resolves either
    default (``VllmConfig.__post_init__``: ``custom_ops`` gains ``"none"`` or
    ``"all"`` only when the serve named neither, and ``set_platform_defaults``
    only appends to a priority list the serve set). A field the serve set
    stays as the serve set it; the worker refuses it if eager would not run it.
    """
    if not (_is_tessera(config) and is_glm5next(config) and _graphs_requested(config)):
        return []
    pinned = []
    ops = config.compilation_config.custom_ops
    if "all" not in ops and "none" not in ops:
        ops.append("all")
        pinned.append("custom_ops += all")
    priority = config.kernel_config.ir_op_priority
    eager = None
    for op in EAGER_IR_OPS:
        if not getattr(priority, op, None):
            eager = eager if eager is not None else eager_priority(config)
            setattr(priority, op, list(eager[op]))
            pinned.append(f"ir_op_priority.{op} = {eager[op]}")
    if pinned:
        _log.warning("tessera.glm53_graphs: graph serve runs eager's operators: %s",
                     "; ".join(pinned))
    return pinned


def pin_unpadded_capture_sizes(config: Any) -> list[int]:
    """Capture every uniform-decode count where a Tessera Glm5Next graph serve named no sizes.

    vLLM's default sizes (``[1, 2, 4, 8, ...]``) replay a decode batch in the
    smallest captured graph that holds it, and a padded replay is another
    computation (``graph_equivalence``, the padded replay): at
    ``max_model_len`` 8448 the default sizes moved b5 and b7 off eager
    (tessera#702).  With k draft tokens a decode batch of n requests is
    ``n * (1 + k)`` tokens, so these sizes leave no decode count padded.  Runs
    where vLLM still honours an unset field (``_set_cudagraph_sizes`` fills it
    after this hook); sizes a serve set are its own, and the worker judges them.
    """
    if not (_is_tessera(config) and is_glm5next(config) and _graphs_requested(config)):
        return []
    compilation = config.compilation_config
    if compilation.cudagraph_capture_sizes is not None:
        return []
    query = 1 + num_draft_tokens(config)
    sizes = [n * query for n in range(1, config.scheduler_config.max_num_seqs + 1)]
    compilation.cudagraph_capture_sizes = sizes
    _log.warning("tessera.glm53_graphs: graph serve captures every decode count, sizes %s", sizes)
    return sizes


def install_operator_pin() -> bool:
    """Register :func:`pin_eager_operators` as the Glm5Next ``verify_and_update_config`` hook.

    Called from the plugin's ``register()``, which vLLM runs in
    ``EngineArgs.__post_init__``, before any ``VllmConfig`` is built.  Chains
    to whatever entry the architecture already had.
    """
    try:
        from vllm.model_executor.models.config import MODELS_CONFIG_MAP, VerifyAndUpdateConfig
    except Exception:  # noqa: BLE001 - no vLLM, or a vLLM without the hook: nothing to pin
        return False
    from .glm53_prefill import GLM5NEXT_ARCHITECTURES

    for arch in sorted(GLM5NEXT_ARCHITECTURES):
        previous = MODELS_CONFIG_MAP.get(arch)
        if getattr(previous, "_tessera_operator_pin", False):
            continue

        class _Pin(VerifyAndUpdateConfig):
            _tessera_operator_pin = True
            _previous = previous

            @classmethod
            def verify_and_update_config(cls, vllm_config) -> None:
                if cls._previous is not None:
                    cls._previous.verify_and_update_config(vllm_config)
                pin_eager_operators(vllm_config)
                pin_unpadded_capture_sizes(vllm_config)

        MODELS_CONFIG_MAP[arch] = _Pin
    return True


# ------------------------------------------------------------------ cause 2: the frozen branch


def _text_config(model_config: Any) -> Any:
    return getattr(model_config, "hf_text_config", None) or getattr(model_config, "hf_config", None)


def index_topk_thresholds(config: Any) -> set[int]:
    """Every ``index_topk`` a captured decode step's indexer branches on: the target's, and the draft's."""
    found = set()
    configs = [getattr(config, "model_config", None)]
    speculative = getattr(config, "speculative_config", None)
    if speculative is not None:
        configs.append(getattr(speculative, "draft_model_config", None))
    for model_config in configs:
        topk = getattr(_text_config(model_config), "index_topk", None)
        if topk is not None:
            found.add(int(topk))
    return found


def _full_graphs(config: Any) -> bool:
    graph = graph_mode(config.compilation_config)
    return getattr(graph, "name", None) in ("FULL_DECODE_ONLY", "FULL_AND_PIECEWISE", "FULL")


def eager_gaps(config: Any, breakable: bool) -> list[str]:
    """Why this graph serve would compute other arithmetic than its eager serve, one reason each.

    Judged on every Tessera Glm5Next graph serve, whatever its ``max_model_len``:
    the operators vLLM resolved (``op_implementation_gap``, cause 1), a decode
    batch replayed in a larger captured graph (``padded_families``' decode
    families), and breakable piecewise graphs, which run the mixed-batch family
    nothing here inspected.  Without breakable graphs vLLM runs no piecewise
    graph of this uncompiled model, so the mixed family cannot replay.
    """
    gaps = []
    ops = op_implementation_gap(config)
    if ops:
        gaps.append(f"{ops} (the eager serve runs ['all'] and ['vllm_c', 'native'])")
    graph = graph_mode(config.compilation_config)
    name = getattr(graph, "name", None)
    if breakable and name in ("PIECEWISE", "FULL_AND_PIECEWISE"):
        gaps.append(f"breakable piecewise CUDA graphs under {name} run mixed batches through "
                    "graphs nothing here inspected (serve VLLM_USE_BREAKABLE_CUDAGRAPH=0)")
    padded = sorted({n for family, counts in padded_families(config, graph) if family != "mixed"
                     for n in counts})
    if padded:
        gaps.append(f"decode batches of {padded} tokens replay a larger captured graph, which is "
                    "another computation (leave cudagraph_capture_sizes unset, or list every "
                    "decode count)")
    return gaps


def _breakable_enabled() -> bool:
    from vllm.v1.worker.gpu import cudagraph_utils
    return bool(cudagraph_utils.is_breakable_cudagraph_enabled())


def branch_plan(config: Any) -> tuple[tuple[int, ...] | None, list[str]]:
    """The capture bounds this serve needs, and why it cannot have them.

    ``(None, [])``: one class suffices (not a Tessera Glm5Next FULL-graph serve,
    or ``max_model_len`` at or below every threshold), so stock capture is
    already eager's.  ``(bounds, [])``: install.  ``(bounds, reasons)``:
    refuse, naming each reason.
    """
    if not (_is_tessera(config) and is_glm5next(config) and _graphs_requested(config)
            and _full_graphs(config)):
        return None, []
    max_model_len = int(config.model_config.max_model_len)
    thresholds = index_topk_thresholds(config)
    if not thresholds:
        return None, ["no index_topk in the model config: the indexer's branch cannot be located"]
    bounds = branch_capture_bounds(thresholds, max_model_len)
    if len(bounds) == 1:
        return None, []
    reasons = []
    par = config.parallel_config
    for name in ("pipeline_parallel_size", "data_parallel_size",
                 "prefill_context_parallel_size", "decode_context_parallel_size"):
        if getattr(par, name, 1) not in (None, 1):
            reasons.append(f"{name} {getattr(par, name)} (inspected at 1)")
    if getattr(config, "lora_config", None) is not None:
        reasons.append("LoRA on (inspected off)")
    draft = num_draft_tokens(config)
    spec = getattr(config, "speculative_config", None)
    if draft > 1:
        reasons.append(f"{draft} speculative tokens: draft decode graphs build their own metadata "
                       "(inspected at k <= 1)")
    if spec is not None and getattr(spec, "method", None) != "mtp":
        reasons.append(f"speculative method {spec.method!r} (inspected: mtp)")
    sizes = sorted(set(config.compilation_config.cudagraph_capture_sizes or ()))
    rows = min(config.scheduler_config.max_num_seqs * (1 + draft), sizes[-1]) if sizes else 0
    if rows > PERSISTENT_TOPK_SAMPLED_MIN_ROWS:
        reasons.append(f"{rows} decode rows per graph reach persistent_topk's sampled path "
                       f"(> {PERSISTENT_TOPK_SAMPLED_MIN_ROWS}), a branch not split here")
    if max_model_len > PERSISTENT_TOPK_RADIX_THRESHOLD:
        reasons.append(f"max_model_len {max_model_len} reaches persistent_topk's radix path "
                       f"(> {PERSISTENT_TOPK_RADIX_THRESHOLD}), a branch not split here")
    return bounds, reasons


def _record_build(original):
    """``build_attn_metadata`` as each metadata builder receives it.

    Inside a class capture, a capture build gets the class's bound as its
    ``max_seq_len``; every other build records the ``max_seq_len`` it was
    given, which is the value the indexer reads on that step.
    """
    def build_attn_metadata(*args, **kwargs):
        if "max_seq_len" not in kwargs:
            raise RuntimeError("tessera.glm53_graphs: build_attn_metadata called without "
                               "max_seq_len by keyword; the inspected interface passes it so")
        bound = _STATE.capture_bound
        if bound is not None and kwargs.get("for_cudagraph_capture"):
            kwargs["max_seq_len"] = min(int(kwargs["max_seq_len"]), bound)
            _STATE.bound_applied = True
        elif bound is None:
            _STATE.step_max_seq_len = int(kwargs["max_seq_len"])
        return original(*args, **kwargs)

    build_attn_metadata._tessera_original = original
    return build_attn_metadata


def _manager_name(manager: Any) -> str:
    return type(manager).__name__


def _patch_managers(cudagraph_utils: Any, managers: tuple[type, ...]) -> None:
    base = cudagraph_utils.CudaGraphManager
    capture, run_fullgraph, release = base.capture, base.run_fullgraph, base.release_graphs
    extrapolate = cudagraph_utils._extrapolate_full_graph_memory

    def per_class_capture(self, *args, **kwargs):
        if not isinstance(self, managers):
            return capture(self, *args, **kwargs)
        bounds = _STATE.bounds
        by_bound = {}
        for bound in bounds:
            self.graphs = {}
            self._graphs_captured = False
            _STATE.capture_bound, _STATE.bound_applied = bound, False
            try:
                capture(self, *args, **kwargs)
            finally:
                _STATE.capture_bound = None
            if self.graphs and bound < bounds[-1] and not _STATE.bound_applied:
                raise RuntimeError(
                    f"tessera.glm53_graphs: {_manager_name(self)} captured the bound-{bound} "
                    "class without building its metadata through the inspected "
                    "build_attn_metadata; its graphs would freeze the long-context branch")
            by_bound[bound] = self.graphs
        self._tessera_graphs_by_bound = by_bound
        self.graphs = by_bound[bounds[-1]]
        with _LOCK:
            CAPTURED[_manager_name(self)] = {b: len(g) for b, g in by_bound.items()}
        _log.warning("tessera.glm53_graphs: %s captured FULL graphs per max_seq_len class %s",
                     _manager_name(self), CAPTURED[_manager_name(self)])

    def per_class_run_fullgraph(self, desc):
        by_bound = getattr(self, "_tessera_graphs_by_bound", None)
        if by_bound is not None:
            seq = _STATE.step_max_seq_len
            if seq is None:
                raise RuntimeError("tessera.glm53_graphs: a FULL replay with no step max_seq_len "
                                   "recorded; the class it needs is unknown")
            bound = branch_bound(tuple(by_bound), seq)
            self.graphs = by_bound[bound]
            with _LOCK:
                REPLAYS[(_manager_name(self), bound)] += 1
        return run_fullgraph(self, desc)

    def per_class_release(self):
        release(self)
        self._tessera_graphs_by_bound = None

    def per_class_extrapolate(mem_samples, total_graphs):
        # The profiler counts FULL descriptors; each is captured once per class.
        return extrapolate(mem_samples, total_graphs * len(_STATE.bounds))

    model_manager = cudagraph_utils.ModelCudaGraphManager
    model_capture = model_manager.capture

    def checked_model_capture(self, model, *args, **kwargs):
        _indexer_topk_check(model)
        return model_capture(self, model, *args, **kwargs)

    model_manager.capture = checked_model_capture
    base.capture = per_class_capture
    base.run_fullgraph = per_class_run_fullgraph
    base.release_graphs = per_class_release
    cudagraph_utils._extrapolate_full_graph_memory = per_class_extrapolate


def install_branch_capture(config: Any, breakable=_breakable_enabled) -> bool:
    """Capture one class of FULL graphs per side of each ``max_seq_len`` threshold, or refuse.

    Called from ``TesseraConfig.get_quant_method`` during model construction,
    before the runner captures.  Returns True when installed (once per process).
    """
    if not (_is_tessera(config) and is_glm5next(config) and _graphs_requested(config)):
        return False
    bounds, reasons = branch_plan(config)
    # The operators and the padded replay do not depend on the frozen branch: judged on every
    # graph serve, one class or many.
    reasons += eager_gaps(config, breakable=breakable())
    if bounds is None and not reasons:
        return False
    with _LOCK:
        if _STATE.installed:
            if _STATE.bounds != bounds:
                raise RuntimeError(f"tessera.glm53_graphs: installed for bounds {_STATE.bounds}, "
                                   f"asked again for {bounds}")
            return True
        if not reasons:
            modules, why = import_modules(GRAPH_MODULES)
            if modules is None:
                reasons.append(why)
            else:
                interface, why = match_modules(modules, GRAPH_MODULES, _INTERFACES)
                if interface is None:
                    reasons.append(why)
        if reasons:
            raise RuntimeError(
                "tessera.glm53_graphs: this Tessera GLM-5.3 CUDA-graph serve would not compute "
                "what its eager serve computes, and Tessera refuses it: "
                + "; ".join(reasons)
                + ". Serve --enforce-eager, or max_model_len <= index_topk "
                f"({min(index_topk_thresholds(config) or {0})}), on an inspected interface.")
        cudagraph_utils, mamba_hybrid = modules[0], modules[1]
        speculator_graphs = modules[3]
        mamba_hybrid.build_attn_metadata = _record_build(mamba_hybrid.build_attn_metadata)
        _STATE.bounds = bounds
        _patch_managers(cudagraph_utils, (cudagraph_utils.ModelCudaGraphManager,
                                          speculator_graphs.SpeculatorCudaGraphManager))
        _STATE.installed = True
    _log.warning("tessera.glm53_graphs: FULL graphs captured per max_seq_len class, bounds %s "
                 "(index_topk %s, max_model_len %s)", list(bounds),
                 sorted(index_topk_thresholds(config)), config.model_config.max_model_len)
    return True


def install_for_current_config() -> bool:
    """Called from ``TesseraConfig.get_quant_method`` during model construction."""
    try:
        from vllm import config as vllm_config
    except Exception:  # noqa: BLE001 - no vLLM, nothing to install
        return False
    getter = getattr(vllm_config, "get_current_vllm_config_or_none", None)
    current = getter() if getter is not None else None
    if current is None:
        return False
    return install_branch_capture(current)


def _indexer_topk_check(model: Any) -> None:
    """Every constructed indexer's ``topk_tokens`` is a threshold the bounds split at."""
    bounds = set(_STATE.bounds or ())
    seen = {int(m.topk_tokens) for m in model.modules()
            if type(m).__name__ == "SparseAttnIndexerKpool" and hasattr(m, "topk_tokens")}
    stray = sorted(t for t in seen if t not in bounds and t < max(bounds))
    if stray:
        raise RuntimeError(f"tessera.glm53_graphs: indexers branch at {stray}, which the capture "
                           f"bounds {sorted(bounds)} do not split")


__all__ = ["CAPTURED", "GRAPH_MODULES", "PERSISTENT_TOPK_RADIX_THRESHOLD",
           "PERSISTENT_TOPK_SAMPLED_MIN_ROWS", "REPLAYS", "branch_plan", "eager_ir_priority",
           "index_topk_thresholds", "install_branch_capture", "install_for_current_config",
           "install_operator_pin", "pin_eager_operators", "pin_unpadded_capture_sizes",
           "eager_gaps"]
