"""The vLLM quantization config a Tessera checkpoint selects.

A checkpoint whose ``quantization_config.quant_method`` is ``"tessera"`` is
routed here by vLLM's own dispatch, because ``register()`` registered this
class under that name.  There is no enable flag: the bytes decide.  What this
class does is read the sidecar (``config_groups``: one Tessera scheme per vLLM
module, plus ``ignore``), refuse anything no route serves at CONFIG PARSE time,
and hand each module to its route.

WHAT IT REFUSES, AND WHY LOUDLY.  Three refusals are deliberate and none of
them degrade to BF16:

* a ``LinearBase`` the checkpoint neither declares nor ignores.  A silent
  ``UnquantizedLinearMethod`` there would serve a BF16 tensor for a module the
  producer believed it had quantized -- a typo in one target name would be a
  silently different artifact, and the KL would look merely disappointing.
* a routed-MoE experts layer.  vLLM's own MoE layer takes
  ``UnquantizedFusedMoEMethod`` when a config returns ``None``, so ``None``
  here would serve uninitialised or BF16 expert memory rather than say no.
  The expert route is designed (below) and not built.
* tensor parallelism in a build with no unit slicer.  Not because the artifact
  is per-rank -- it is deliberately TP-agnostic, one whole unit per role, and
  the loader cuts it at load (``sharding``) -- but because the cut needs
  ``tessera.layout.slice_unit``.  Where the slicer IS present, as it is here,
  ``tp_size > 1`` is no longer refused wholesale: the route refuses the one
  AXIS its decoders cannot start from, at ``create_weights``, off the
  ``sharding.ROUTE_TP_AXES`` table.  A refusal, never "every rank holds the
  whole weight".

  A world size above one is ATTEMPTED and not ATTESTED.  The packaged
  ``runtime_contract.json`` still says ``max_world_size: 1`` for every family,
  because no multi-rank serve has been run; what it publishes above that is
  ``loader_axes``, which is what the loader DOES.  Two different questions, two
  machine-readable fields, and a producer may gate on either.

WHERE THE MoE ROUTE PLUGS IN.  ``get_quant_method``'s ``RoutedExperts`` branch
is the seam.  A Tessera MoE method would: parse one ``tessera.fused`` container
per expert group, decode the per-expert wires into the STOCK packed expert
layouts vLLM's fused-MoE kernels read (NVFP4: the packed w13/w2 triple, which
needs ``--moe-backend flashinfer_b12x`` on GB10; FP8: the per-channel
compressed-tensors MoE path), and dispatch through vLLM's own fused-MoE
kernels exactly as the dense routes dispatch through ``torch._scaled_mm``.  The
scheme already carries ``structure`` for it (``scheme.STRUCTURES``); adding the
route is a new value there, a route module, and -- only once a served census
and KL exist -- a ``routed_moe`` cell in ``runtime_contract.json``.  No cell
today, because no measurement.

MoE AND PARALLELISM.  Expert parallelism needs no slicing at all: its
granularity is one whole expert unit per rank, which is the case
``_shard_unit_for_rank`` already serves (identity).  Tensor parallelism INSIDE
an expert is the same row/column cut as a dense Linear's, on the same seam, so
the expert route inherits it rather than restating it.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

from .compile_identity import declare_compile_identity
from .lane import TESSERA_MODE_ENV, build_tessera_method, serve_mode
from .scheme import is_tessera_scheme, validate_tessera_scheme
from .sharding import require_a_cutter

__all__ = ["TesseraConfig", "QUANT_METHOD"]

QUANT_METHOD = "tessera"


def _moe_layer_classes() -> tuple:
    """vLLM's routed-experts layer classes, tolerant of version renames.

    0.28 calls it ``RoutedExperts``; older builds called it ``FusedMoE``.  A
    missing name is not an error -- the tuple simply gets shorter, and the
    name-based check below still catches it.
    """
    classes = []
    try:
        from vllm.model_executor.layers import fused_moe as _moe
    except Exception:  # noqa: BLE001 -- a build without the MoE package
        return ()
    for name in ("RoutedExperts", "FusedMoE"):
        obj = getattr(_moe, name, None)
        if isinstance(obj, type):
            classes.append(obj)
    return tuple(classes)


_MOE_CLASS_NAMES = frozenset({"RoutedExperts", "FusedMoE", "SharedFusedMoE"})


class TesseraConfig(QuantizationConfig):
    """``quant_method: "tessera"`` -- the two dense Tessera routes."""

    def __init__(self, config_groups: Mapping[str, Any], ignore: tuple[str, ...],
                 full_config: Mapping[str, Any]):
        super().__init__()
        self._full_config = dict(full_config)
        self.ignore = tuple(str(i) for i in ignore)
        self.target_scheme: dict[str, dict] = {}
        for name, group in config_groups.items():
            scheme = group.get("scheme")
            targets = [str(t) for t in group.get("targets", ())]
            if not is_tessera_scheme(scheme):
                raise ValueError(
                    f"config group {name!r} is not a Tessera scheme: this plugin reads "
                    "'tessera' checkpoints only, and a group it cannot read is a checkpoint "
                    "written for another runtime, not a group to skip")
            if not targets:
                raise ValueError(f"config group {name!r} names no target")
            for target in targets:
                validate_tessera_scheme(scheme, target)
                if target in self.target_scheme:
                    raise ValueError(f"target {target!r} is declared by two config groups")
                self.target_scheme[target] = dict(scheme)
        self._check_overlap()
        # Resolve the residency HERE, at config parse, so an unset or misspelt
        # mode is one clear message before any weight is touched.
        self._mode = serve_mode()
        self._declared = False

    def _check_overlap(self) -> None:
        overlap = sorted(set(self.ignore) & set(self.target_scheme))
        if overlap:
            raise ValueError(
                f"targets {overlap} are both declared and ignored; one of the two is a mistake")

    # -- checkpoint names are not module names ----------------------------
    def apply_vllm_mapper(self, hf_to_vllm_mapper) -> None:
        """Translate the checkpoint's module names into the ones vLLM builds.

        A quantization config is written in the CHECKPOINT's namespace; vLLM
        dispatches on the namespace of the module tree it built, and for a model
        whose class declares an ``hf_to_vllm_mapper`` the two differ.  vLLM hands
        every quant config the mapper for exactly this reason -- ``model_loader/
        utils.py:277-279`` calls ``quant_config.apply_vllm_mapper(
        hf_to_vllm_mapper.get_unstacked_mapper())`` for a model class that is not
        ``SupportsQuant``, and ``models/interfaces.py:1160`` does it for one that
        is.  ``QuantizationConfig.apply_vllm_mapper`` is a no-op
        (``base_config.py:229-241``), and inheriting it was this plugin's bug:
        every declared target stayed in checkpoint space, so on any mapped
        architecture NOTHING matched and every Linear was refused at load.

        It never showed, because every Tessera artifact served so far is
        Qwen3-0.6B, whose class declares no mapper and whose module path is its
        checkpoint path.  On ``Glm5NextForConditionalGeneration`` the mapper is
        ``{"model.language_model." -> "language_model.model.", "model.visual." ->
        "visual.", "lm_head." -> "language_model.lm_head."}``, so not one target
        would have matched.

        The target-shape rules are compressed-tensors' own
        (``compressed_tensors.py:122-153``): a dotted path is translated, a bare
        module class name or a ``re:`` pattern is left alone.  Where this differs
        from compressed-tensors is that it REFUSES instead of dropping: a
        declared wire whose module the runtime maps away, or two checkpoint
        modules colliding onto one vLLM module, is a checkpoint that cannot be
        served correctly, and silence there is how a wire ends up loaded and
        never executed.
        """
        def mapped(target: str) -> str:
            if "." not in target or target.startswith("re:"):
                return target          # a module class name or a regex, not a path
            out = hf_to_vllm_mapper.apply_list([target])
            if not out:
                raise ValueError(
                    f"the model's hf_to_vllm_mapper drops {target!r}, so the runtime builds no "
                    "module for a name this checkpoint declares. Either the checkpoint was "
                    "written for a different architecture or the wire is dead weight; both are "
                    "refusals, not warnings.")
            return out[0]

        declared: dict[str, dict] = {}
        origin: dict[str, str] = {}
        for target, scheme in self.target_scheme.items():
            name = mapped(target)
            if name in declared:
                raise ValueError(
                    f"{origin[name]!r} and {target!r} both map to the module {name!r}: one vLLM "
                    "module cannot take two Tessera wires")
            declared[name] = scheme
            origin[name] = target
        self.target_scheme = declared
        # ``dict.fromkeys`` keeps the order and drops the duplicates two
        # checkpoint names can become; unlike a declared target, a doubly-named
        # ignore is harmless.
        self.ignore = tuple(dict.fromkeys(mapped(i) for i in self.ignore))
        self._check_overlap()

    # -- vLLM's QuantizationConfig contract -------------------------------
    @classmethod
    def get_name(cls) -> str:
        return QUANT_METHOD

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # Both routes quantize the activation themselves and ask
        # ``torch._scaled_mm`` for a bf16 output.
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        """The floor the ARITHMETIC needs, not the floor that is attested.

        89 is where per-token dynamic FP8 and ``torch._scaled_mm`` W8A8 exist;
        the NVFP4 route additionally needs ``scaled_fp4_quant`` and the FP4
        mainloop, and fails closed at load where they are absent
        (``native_ops.require_native_fp4_quant``).  The only cells this
        package's ``runtime_contract.json`` publishes are sm_121: a serve on
        anything else runs, and is unattested.
        """
        return 89

    @staticmethod
    def get_config_filenames() -> list[str]:
        # The whole declaration lives in config.json's quantization_config.
        return []

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TesseraConfig":
        method = str(config.get("quant_method", ""))
        if method != QUANT_METHOD:
            raise ValueError(
                f"quantization_config.quant_method is {method!r}, not {QUANT_METHOD!r}")
        groups = config.get("config_groups")
        if not isinstance(groups, Mapping) or not groups:
            raise ValueError(
                "a Tessera checkpoint declares its wires in quantization_config.config_groups")
        return cls(groups, tuple(config.get("ignore", ())), config)

    # -- dispatch ----------------------------------------------------------
    def _require_a_cutter(self, prefix: str) -> None:
        """Refuse a TP group this BUILD cannot cut a unit for -- and only that.

        The ARTIFACT is TP-agnostic and stays so: a unit is encoded once, whole,
        and a rank takes its slice at load (``sharding``).  The cutter --
        ``tessera.layout.slice_unit``, which records the trellis state a sliced
        row starts from -- has landed, so this gate is no longer "one rank
        only"; it is the whole-file question, asked once at method construction
        rather than at the first forward of every module, and it is kept rather
        than deleted because a build WITHOUT the slicer must still refuse rather
        than serve rank 0's whole unit to every rank.

        The narrower question -- which AXIS this route's decoders can start from
        -- is not answerable here: the axis is derived from the sizes vLLM asks
        for, which arrive at ``create_weights``.  Each route asks it there
        (``sharding.require_axis_supported``), off the same
        ``sharding.ROUTE_TP_AXES`` table the packaged contract publishes.

        WHAT IS NOT CLAIMED.  A world size above one is something this build
        ATTEMPTS, not something it has served: ``runtime_contract.json`` still
        says ``max_world_size: 1`` for every family and will until a multi-rank
        serve has been run and measured.  That is the same status a rung with no
        ``lane_eligibility`` cell has -- unattested, which is honest, and not a
        refusal.
        """
        try:
            from vllm.distributed import get_tensor_model_parallel_world_size

            world = int(get_tensor_model_parallel_world_size())
        except Exception:  # noqa: BLE001 -- no parallel state: a bare test build
            return
        require_a_cutter(prefix, world)

    def _declare_once(self) -> None:
        if self._declared:
            return
        # The mode selects a different compiled forward over the same files, so
        # it must be in vLLM's compile-cache key before any hash is computed.
        declare_compile_identity(serve_mode=self._mode)
        self._declared = True

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

        moe_classes = _moe_layer_classes()
        if (moe_classes and isinstance(layer, moe_classes)) or \
                type(layer).__name__ in _MOE_CLASS_NAMES:
            if prefix in self.ignore:
                # The checkpoint DECLARED these experts BF16 (the exporter names
                # a passed-through expert stack in ``ignore``), so vLLM's own
                # unquantized MoE method is the right answer and saying so is
                # not a silent fallback.
                return None
            raise ValueError(
                f"tessera target {prefix!r}: routed-MoE experts are not served by this plugin "
                "yet, and vLLM would silently fall back to UnquantizedFusedMoEMethod if this "
                "returned None, so the refusal is here. The expert route will decode per-expert "
                "wires into the stock packed expert layouts and run vLLM's own fused-MoE "
                "kernels (NVFP4 W4A4 needs --moe-backend flashinfer_b12x on GB10; per-channel "
                "FP8 W8A8 is the compressed-tensors MoE path); it carries no lane_eligibility "
                "cell because no served measurement covers it. Export the experts to a format "
                "the pinned runtime serves, or wait for the expert route.")
        if isinstance(layer, LinearBase):
            scheme = self.target_scheme.get(prefix)
            if scheme is not None:
                self._require_a_cutter(prefix)
                self._declare_once()
                return build_tessera_method(scheme, prefix, self._mode)
            if prefix in self.ignore:
                return UnquantizedLinearMethod()
            raise ValueError(
                f"tessera checkpoint declares no wire for Linear {prefix!r} and does not ignore "
                "it. Serving it BF16 anyway would make one mistyped target name an artifact that "
                "looks merely disappointing instead of one that refuses: every Linear a Tessera "
                "checkpoint contains is either declared in config_groups or named in "
                "quantization_config.ignore.")
        # Embeddings, the LM head and attention: vLLM's own unquantized methods.
        # ``ParallelLMHead`` reaches this branch and takes
        # ``UnquantizedEmbeddingMethod``; a Tessera lm_head is not a route.
        return None

    def get_cache_scale(self, name: str):  # pragma: no cover - vLLM optional hook
        return None

    def __repr__(self) -> str:
        return (f"TesseraConfig(modules={len(self.target_scheme)}, ignore={len(self.ignore)}, "
                f"{TESSERA_MODE_ENV}={self._mode!r})")
