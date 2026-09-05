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
* a routed-MoE experts layer the checkpoint neither declares nor ignores, and
  one it declares with a DENSE scheme.  vLLM's own MoE layer takes
  ``UnquantizedFusedMoEMethod`` when a config returns ``None``, so ``None``
  here would serve uninitialised or BF16 expert memory rather than say no.
  A stack declared ``structure: routed_moe`` is served by ``moe_route``.
* tensor parallelism in a build with no unit slicer, or against an artifact
  whose own config does not say its bytes can be cut.  Not because the artifact
  is per-rank -- it is deliberately TP-agnostic, one whole unit per role, and
  the loader cuts it at load (``sharding``) -- but because the cut needs
  ``tessera.layout.slice_unit`` in the build AND a wire that can express a
  shard.  The second is read off the checkpoint (``tp_agnostic``, derived by
  the exporter from the schema minor it wrote at) rather than assumed of every
  artifact, and a checkpoint that declares nothing is refused above one rank
  (tessera#328).  Where the slicer IS present, as it is here,
  ``tp_size > 1`` is no longer refused wholesale: the route refuses the one
  AXIS its decoders cannot start from, at ``create_weights``, off the
  ``sharding.ROUTE_TP_AXES`` table.  A refusal, never "every rank holds the
  whole weight".

  A world size above one is ATTEMPTED and not ATTESTED.  The packaged
  ``runtime_contract.json`` still says ``max_world_size: 1`` for every family,
  because no multi-rank serve has been run; what it publishes above that is
  ``loader_axes``, which is what the loader DOES.  Two different questions, two
  machine-readable fields, and a producer may gate on either.

WHERE THE MoE ROUTE PLUGS IN.  ``get_quant_method``'s MoE branch is the seam
(derived from vLLM's MoE layer module, never a hand-kept name list, so a
version rename cannot slip it into the silent unquantized fallback).  A stack
whose scheme declares ``structure: routed_moe`` goes to ``moe_route``, which
decodes one container per expert PROJECTION into the stock per-channel FP8
expert parameters and dispatches through vLLM's own fused-MoE kernel -- the
same ``select_fp8_moe_backend`` / ``convert_to_fp8_moe_kernel_format`` /
``make_fp8_moe_kernel`` path ``CompressedTensorsW8A8Fp8MoEMethod`` takes at
``strategy: channel``.  Which families have an expert route is
``scheme.MOE_BUILDERS``, and it says why the other two do not: the NVFP4 arm
resolves only under a ``swiglu_limit`` clamp on this build
(``docs/measurements/nvfp4-moe-oracle-2026-09-02.md``), and a BF16 expert
stack is the passthrough ``ignore`` already gives.

``runtime_contract.json`` v16 publishes two ``routed_moe`` cells, and only
two: ``TESSERA_E4M3_K1`` at rung 1024 on ``sm_121``, resident/eager, decode
and batch, attested on the exact ``eugr/spark-vllm`` image the complete
LFM2.5-8B-A1B artifact was censused and KL-compared on
(``docs/measurements/tessera-lfm-campaign-2026-09-04.md`` sections 7-10).
Every other rung, image, the compiled and streamed forms and any expert or
tensor parallelism are unattested for this structure.  That is the
``loader_axes`` precedent: what the loader DOES is a different published
fact from what has been served, and the cell names exactly the second.

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
from .scheme import (STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE, is_tessera_scheme,
                     validate_tessera_scheme)
from .sharding import require_a_cutter, require_a_cuttable_artifact

__all__ = ["TesseraConfig", "QUANT_METHOD"]

QUANT_METHOD = "tessera"


#: Names vLLM has used for its routed-experts layer.  A SEED, not the gate:
#: each is imported tolerantly below (a missing name is not an error), and
#: the namespace scan beside it covers what no seed can -- the next rename.
#: Kept because ``isinstance`` against the actual objects is the one check
#: that survives a move: a class defined outside the MoE module but
#: re-exported from it has another ``__module__`` and only the import finds
#: it.
_MOE_SEED_NAMES = ("RoutedExperts", "FusedMoE", "SharedFusedMoE")


def _moe_layer_classes() -> tuple:
    """vLLM's routed-experts layer classes, derived from the module that owns them.

    The union of the seed names above (tolerant imports -- whatever a build
    renamed or moved, the object itself is what ``isinstance`` needs) and
    every ``torch.nn.Module`` subclass the MoE module's namespace carries:
    0.28 calls it ``RoutedExperts``, older builds called it ``FusedMoE``, and
    whatever a later build adds is picked up without an edit here.  Two
    exclusions, both load-bearing: ``torch.nn.Module`` itself (everything is
    an instance of it) and ``LinearBase`` subclasses (a linear layer the
    module merely imports belongs to the Linear branch, and matching it here
    would refuse every declared Linear as an expert).
    """
    try:
        from vllm.model_executor.layers import fused_moe as _moe
    except Exception:  # noqa: BLE001 -- a build without the MoE package
        return ()
    try:
        from vllm.model_executor.layers.linear import LinearBase
    except Exception:  # noqa: BLE001 -- unreachable where the gate runs
        LinearBase = None  # type: ignore[assignment]
    classes: list = []
    for name in _MOE_SEED_NAMES:
        obj = getattr(_moe, name, None)
        if isinstance(obj, type) and obj not in classes:
            classes.append(obj)
    for name in dir(_moe):
        obj = getattr(_moe, name, None)
        if not isinstance(obj, type) or obj is torch.nn.Module or obj in classes:
            continue
        if not issubclass(obj, torch.nn.Module):
            continue
        if LinearBase is not None and issubclass(obj, LinearBase):
            continue
        classes.append(obj)
    return tuple(classes)


def _moe_class_names() -> frozenset:
    """The names of the imported MoE layer classes, generated, not maintained.

    Catches a subclass defined in another module that carries one of these
    names (a version rename, a downstream subclass): the set follows the
    imports above, so it cannot go stale beside them the way a hand-written
    frozenset did.
    """
    return frozenset(c.__name__ for c in _moe_layer_classes())


def _looks_like_moe_layer(layer: torch.nn.Module) -> bool:
    """Fail-closed backstop for a renamed or moved MoE layer.

    Neither the module scan nor the generated name set can know a class name
    vLLM has not shipped yet; without this, such a layer is not a LinearBase
    and control reaches ``return None`` -- vLLM's silent unquantized MoE
    fallback, the outcome this branch exists to prevent.  A layer whose type
    name says MoE or expert refuses instead.  Narrow on purpose: it fires on
    layers (``torch.nn.Module`` instances) only, so the LM head, embeddings
    and attention keep vLLM's own default, and an explicit ``ignore`` still
    wins at the call site (a declared-BF16 expert stack is an answer, not a
    fallback).
    """
    return isinstance(layer, torch.nn.Module) and (
        "moe" in type(layer).__name__.lower() or "expert" in type(layer).__name__.lower())


class TesseraConfig(QuantizationConfig):
    """``quant_method: "tessera"`` -- the dense routes and the expert route."""

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
        # The two quantized routes quantize the activation themselves and ask
        # ``torch._scaled_mm`` for a bf16 output; the BF16 route consumes this
        # dtype directly.  Either way bf16 in, bf16 out.
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
        """Refuse a TP group this BUILD, or these BYTES, cannot cut a unit for.

        Two whole-file questions, asked once at method construction rather than
        at the first forward of every module, and neither is the per-axis one:

        * **This build.**  ``sharding.require_a_cutter``.  The cutter --
          ``tessera.layout.slice_unit``, which records the trellis state a
          sliced row starts from -- has landed, so this is no longer "one rank
          only"; it is kept because a build WITHOUT the slicer must still
          refuse rather than serve rank 0's whole unit to every rank.
        * **These bytes.**  ``sharding.require_a_cuttable_artifact``, reading
          the checkpoint's own declaration out of the config vLLM handed this
          class (``self._full_config``, which until tessera#328 was retained
          and never read).  The ARTIFACT is TP-agnostic and stays so -- a unit
          is encoded once, whole, and a rank takes its slice at load
          (``sharding``) -- but *that is a property of the wire it was written
          at*, not a standing promise, so it is read off the artifact instead
          of assumed of every artifact.  A checkpoint that declares neither
          ``tp_agnostic`` nor ``schema_minor`` is refused above one rank: it
          predates the declaration and does not get to claim it.

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
        refusal.  What the contract publishes about tensor parallelism is
        tessera#330's question and not this gate's.
        """
        try:
            from vllm.distributed import get_tensor_model_parallel_world_size

            world = int(get_tensor_model_parallel_world_size())
        except Exception:  # noqa: BLE001 -- no parallel state: a bare test build
            return
        require_a_cutter(prefix, world)
        require_a_cuttable_artifact(prefix, world, self._full_config)

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
                type(layer).__name__ in _moe_class_names() or \
                _looks_like_moe_layer(layer):
            if prefix in self.ignore:
                # The checkpoint DECLARED these experts BF16 (the exporter names
                # a passed-through expert stack in ``ignore``), so vLLM's own
                # unquantized MoE method is the right answer and saying so is
                # not a silent fallback.
                return None
            scheme = self.target_scheme.get(prefix)
            if scheme is not None and scheme.get("structure") == STRUCTURE_ROUTED_MOE:
                self._declare_once()
                from .moe_route import build_tessera_moe_method

                return build_tessera_moe_method(scheme, prefix, self._mode, layer)
            if scheme is not None:
                raise ValueError(
                    f"tessera target {prefix!r}: this is a routed-MoE expert stack but its "
                    f"scheme declares structure "
                    f"{scheme.get('structure', STRUCTURE_DENSE)!r}. A dense scheme names one "
                    "blob per module and the expert route reads one per expert per group; "
                    "serving one through the other would read the wrong tensor rank.")
            raise ValueError(
                f"tessera checkpoint declares no wire for the routed-MoE expert stack "
                f"{prefix!r} and does not ignore it. Returning None here would hand vLLM "
                "UnquantizedFusedMoEMethod -- uninitialised or BF16 expert memory served in "
                "silence -- so the refusal is here. Declare the stack with a routed_moe "
                "scheme, or name it in quantization_config.ignore to pass the experts "
                "through at their source precision.")
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
