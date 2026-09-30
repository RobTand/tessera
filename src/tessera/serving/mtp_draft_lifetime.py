"""Avoid vocabulary storage that stock GLM MTP loading discards before use.

The target must NOT be exposed to draft loading or post-processing. Parameter-
free placeholders survive only until stock unconditional MTP target sharing.
Context-local constructor dispatch leaves unrelated threads/loads unchanged.

Only inspected vLLM interfaces are supported, each identified by the source
digests of every module the patch touches (``_INTERFACES``). A serve whose
sources match none of them declines: it loads the stock way and logs that the
saving is absent (tessera#749). A recognized interface that misbehaves (a
changed signature, an unsupported target layout, sharing that leaves a
placeholder behind) still fails closed.
"""
from __future__ import annotations

# vLLM is optional and absent from the device-less development interpreter.
# pyright: reportMissingImports=false

from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import hashlib
import importlib
import inspect
import logging
from pathlib import Path
from threading import RLock
from typing import Any

import torch

_log = logging.getLogger(__name__)

# The modules the patch reads or rebinds, after the GLM MTP module, in the
# digest order of ``_Interface.digests``.
_COMMON_MODULES = (
    "vllm.model_executor.models.deepseek_mtp",
    "vllm.v1.spec_decode.llm_base_proposer",
    "vllm.v1.worker.gpu.spec_decode.mtp.speculator",
    "vllm.v1.worker.gpu.spec_decode.eagle.utils",
    "vllm.model_executor.layers.vocab_parallel_embedding",
    "vllm.model_executor.model_loader.base_loader",
    "vllm.model_executor.model_loader.utils",
)


@dataclass(frozen=True)
class _Interface:
    """One inspected construction/load/share interface.

    ``draft_heads``: stock draft construction builds a per-layer
    ``shared_head.head`` ParallelLMHead that target sharing then discards. The
    nightly builds ``SharedHead(defer_lm_head=True)`` (no head storage), so only
    the draft ``embed_tokens`` is left to intercept there.

    ``draft_load_rename``: the checkpoint-prefix rule the draft class applies
    inside ``load_weights`` without declaring an ``hf_to_vllm_mapper``, so vLLM
    never hands it to the quant config (``TesseraConfig._module_lookup`` adopts
    it). None where the class declares its mapper (fd4a15126's image carries
    ``Glm5NextMTP.hf_to_vllm_mapper``).
    """
    name: str
    glm_module: str
    digests: tuple[str, ...]  # the GLM MTP module, then _COMMON_MODULES
    draft_heads: bool
    draft_load_rename: tuple[str, str] | None = None


_INTERFACES = (
    # Source copies inspected for tessera#645: the GLM image's
    # mtp-census/vllm-src, and fd4a15126's V2 mtp/speculator.py and
    # eagle/utils.py.
    _Interface(
        "fd4a15126", "vllm.models.glm5next.nvidia.mtp",
        ("45124573a928ecd76e6bd4f595aec3c3d98d81a71fc7a41fb32564ace560b7d1",
         "e9270724c39a0152dc0a66b94622ebddd384c592534cbbf38d2f43c0ba1592d0",
         "86fdf5cc84b35ff568ba8d0ab067ce827c1b32f3c5f5e49b9bcdd8f89accfb81",
         "1fcffbf5e5a85e4c901bd71c65a73da814e7273627276dbdfebf367720a0bc1a",
         "65d882b8fb476eb0dd6161247346cd7b0392b22832abf36fbb11e795979f9e28",
         "a187d541a61455d6967b98ff3b9837a752e4f5bd5ab9c91de4aeca881f612c3a",
         "f17471235e9e349dee6a2b3a795979187c51cc69f26d238e47d8e37c0766b8d4",
         "6cd928c158703de94223b056d25fb58b38db49befb2f3c7aee1b5e97daca1e86"),
        draft_heads=True),
    # tessera#749: sha256 of each module inside
    # localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705... (U4_STACK
    # nightly-20260929, vLLM 0.30.1rc1.dev336+gaf5b4857e).
    _Interface(
        "nightly-20260929", "vllm.models.glm5next.common.mtp",
        ("b0a8f47402cd61d8459339f79e5a52448927eae52bc2d9602fea3c0469c085b3",
         "aae80e30b2dbfd1df41b83f51aaba4722e9f2dc3b4479d2a38ce1e32926fac11",
         "5793b956de30b63ff56a81b730af30bab6ebb6cf45914b2638ed58d67c9bceff",
         "926d6c9ffdb368c971647911a60493820d056ce236fcae9d94a42fa2e006255d",
         "a6d4830323e4c134b78d580e57036decee382c08e59ba17a2c5383e7af1f1802",
         "01576027a0262d2135800aab0ddb2fe2701fae2177c33f8ad2426240b6e04073",
         "5291c330c4ed5634a9ea1d535364c26162b60251aa2af03fd409b88c34d836e2",
         "e6e3477d926deb12fd1b50d4e6a7d649f97a5c2b6524cc2e4a604c80bb653f20"),
        draft_heads=False,
        # common/mtp.py load_weights: ``if name.startswith("model.language_model."):
        # name = name.replace("model.language_model.", "model.", 1)``; the class
        # declares no hf_to_vllm_mapper.
        draft_load_rename=("model.language_model.", "model.")),
)


@dataclass
class _DraftLoad:
    embedding: Any
    head: Any
    vocab_size: int
    hidden_size: int
    first_layer: int
    layer_count: int
    quant_config: Any
    draft_heads: bool = True
    constructed: list[str] = field(default_factory=list)

    @property
    def heads(self) -> set[str]:
        return {f"model.layers.{i}.head" for i in
                range(self.first_layer, self.first_layer + self.layer_count)}

    @property
    def intercepted(self) -> set[str]:
        return {"model.embed_tokens"} | (self.heads if self.draft_heads else set())

    def shared_weight(self, name: str) -> bool:
        # Exact source grammar owned by Glm5NextMTP._rewrite_spec_layer_name.
        # No suffix/substring filter: shared_head.norm and all decoder weights stay.
        for i in range(self.first_layer, self.first_layer + self.layer_count):
            for root in ("model.", "model.language_model."):
                if name in (f"{root}layers.{i}.embed_tokens.weight",
                            f"{root}layers.{i}.shared_head.head.weight"):
                    return True
        return False


_LOAD: ContextVar[_DraftLoad | None] = ContextVar("tessera_mtp_draft_load", default=None)
_INSTALL_LOCK = RLock()


class _UnallocatedVocabulary(torch.nn.Module):
    """No parameter, quant method or forward; stock sharing must remove it."""


def require_source_digest(module: Any, expected: str) -> None:
    path = getattr(module, "__file__", None)
    try:
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
    except OSError as exc:
        raise RuntimeError(f"Tessera MTP source identity unreadable: {path}") from exc
    if actual != expected:
        raise RuntimeError(
            f"Tessera MTP unsupported source identity for {module.__name__}: "
            f"expected {expected}, got {actual}; vocabulary allocation avoidance "
            "requires the inspected construction/load/share interface")


def _require_supported_sources(interface: _Interface, *modules: Any) -> None:
    for module, digest in zip(modules, interface.digests, strict=True):
        require_source_digest(module, digest)


_IMPORT_ERRORS: dict[str, str] = {}


def _import(name: str) -> Any:
    """The module, or None when it is absent or fails to import.

    A candidate interface's module failing at import is a non-match, never a
    reason to fail a serve the stock loader would run.
    """
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - any import failure is a non-match
        _IMPORT_ERRORS[name] = f"{type(exc).__name__}: {exc}"
        return None
    _IMPORT_ERRORS.pop(name, None)
    return module


# One slot: the module objects a resolution saw, and its result. get_quant_method
# runs once per layer; a declined serve must not re-hash sources per call.
# Keyed by module identity (the slot holds references, so ids cannot recycle).
_RESOLVED: list[Any] = [None, None]


def _supported_interface() -> tuple[_Interface, tuple[Any, ...]] | None:
    """The first inspected interface whose every source digest matches.

    None when the running vLLM matches none: the serve then loads the draft the
    stock way, without the saving, and the reasons are logged once.
    """
    names = (*(interface.glm_module for interface in _INTERFACES), *_COMMON_MODULES)
    seen = tuple(_import(name) for name in names)
    key, result = _RESOLVED
    if key is not None and all(a is b for a, b in zip(key, seen, strict=True)):
        return result
    result = _resolve()
    _RESOLVED[:] = [seen, result]
    return result


def _resolve() -> tuple[_Interface, tuple[Any, ...]] | None:
    common = tuple(_import(name) for name in _COMMON_MODULES)
    reasons = []
    for interface in _INTERFACES:
        glm = _import(interface.glm_module)
        missing = [name for name, module in zip((interface.glm_module, *_COMMON_MODULES),
                                                (glm, *common)) if module is None]
        if missing:
            reasons.append(f"{interface.name}: not importable: "
                           + ", ".join(f"{name} ({_IMPORT_ERRORS.get(name, 'absent')})"
                                       for name in missing))
            continue
        modules = (glm, *common)
        if getattr(glm, "_tessera_mtp_lifetime", False):
            return interface, modules
        try:
            _require_supported_sources(interface, *modules)
        except RuntimeError as exc:
            reasons.append(f"{interface.name}: {exc}")
            continue
        return interface, modules
    _log.warning(
        "Tessera MTP draft vocabulary saving absent: the running vLLM matches "
        "no inspected construction/load/share interface, so the draft loads "
        "the stock way (tessera#749). %s", "; ".join(reasons))
    return None


def _eligible(config: Any) -> bool:
    speculative = getattr(config, "speculative_config", None)
    draft = getattr(speculative, "draft_model_config", None)
    hf = getattr(draft, "hf_config", None)
    return (getattr(speculative, "method", None) == "mtp"
            and getattr(hf, "architectures", None) == ["Glm5NextMTPModel"]
            and type(getattr(config, "quant_config", None)).__module__
            == "tessera.serving.config")


def _target_load(config: Any, target: Any, draft_heads: bool = True) -> _DraftLoad:
    from vllm.distributed import (
        get_pp_group,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        UnquantizedEmbeddingMethod,
        VocabParallelEmbedding,
    )

    if get_pp_group().world_size != 1 or getattr(config, "lora_config", None) is not None:
        raise RuntimeError("Tessera MTP vocabulary sharing requires PP=1 without LoRA")
    language = target.get_language_model() if hasattr(target, "get_language_model") else target
    embed: Any = getattr(getattr(language, "model", None), "embed_tokens", None)
    head: Any = getattr(language, "lm_head", None)
    draft = config.speculative_config.draft_model_config
    hf = draft.hf_config
    world, rank = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
    if world not in (1, 2):
        raise RuntimeError("Tessera MTP vocabulary sharing requires supported TP=1 or TP=2")
    # Only the vocabularies the interface intercepts must fit the placeholder
    # contract; a head stock never constructs in the draft is not ours to check.
    checked = ((embed, VocabParallelEmbedding),) + (((head, ParallelLMHead),) if draft_heads else ())
    for module, cls in checked:
        if type(module) is not cls or type(getattr(module, "quant_method", None)) is not UnquantizedEmbeddingMethod:
            raise RuntimeError("Tessera MTP vocabulary sharing requires plain unquantized vocab modules")
        weight = getattr(module, "weight", None)
        padding = module.padding_size
        padded = -(-hf.vocab_size // padding) * padding
        if (padded % world or module.num_embeddings != hf.vocab_size
                or module.org_vocab_size != hf.vocab_size
                or module.embedding_dim != hf.hidden_size
                or module.tp_size != world or module.tp_rank != rank
                or not isinstance(weight, torch.Tensor)
                or tuple(weight.shape) != (padded // world, hf.hidden_size)):
            raise RuntimeError("Tessera MTP incompatible target vocabulary shape/TP layout")
        if (weight.layout != torch.strided or not weight.is_contiguous()
                or weight.dtype != draft.dtype or weight.device.type not in ("cpu", "cuda")
                or getattr(module, "bias", None) is not None):
            raise RuntimeError("Tessera MTP incompatible target vocabulary dtype/device/layout")
    if draft_heads and embed.weight.device != head.weight.device:
        raise RuntimeError("Tessera MTP target vocabulary devices disagree")
    if hf.num_nextn_predict_layers <= 0 or hf.num_hidden_layers < 0:
        raise RuntimeError("Tessera MTP invalid draft layer range")
    return _DraftLoad(embed, head, hf.vocab_size, hf.hidden_size,
                      hf.num_hidden_layers, hf.num_nextn_predict_layers, config.quant_config,
                      draft_heads)


def _constructor(original: Any, *, head: bool) -> Any:
    signature = inspect.signature(original)

    class ConstructorMeta(type(original)):
        def __instancecheck__(cls, obj):
            return isinstance(obj, original)
        def __subclasscheck__(cls, sub):
            return issubclass(sub, original)

    class Constructor(original, metaclass=ConstructorMeta):
        _tessera_mtp_constructor = True

        def __new__(cls, *args, **kwargs):
            state = _LOAD.get()
            if state is None:
                return original(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            prefix = values.get("prefix")
            expected = state.heads if head else {"model.embed_tokens"}
            if prefix not in expected:
                # Other vocab constructions remain real; verified source and
                # exact counts below keep a changed GLM construction fail-closed.
                return original(*args, **kwargs)
            if (values.get("num_embeddings") != state.vocab_size
                    or values.get("embedding_dim") != state.hidden_size
                    or (head and values.get("quant_config") is not state.quant_config)
                    or prefix in state.constructed):
                raise RuntimeError("Tessera MTP unsupported vocabulary constructor shape/interface")
            state.constructed.append(prefix)
            return _UnallocatedVocabulary()

    Constructor.__name__ = original.__name__
    Constructor.__qualname__ = original.__qualname__
    Constructor.__module__ = original.__module__
    Constructor.__signature__ = signature
    return Constructor


def _verify(draft: Any, state: _DraftLoad) -> None:
    inner = getattr(draft, "model", None)
    layers = getattr(inner, "layers", {})
    heads_shared = (getattr(draft, "lm_head", None) is state.head
                    and all(getattr(getattr(layer, "shared_head", None), "head", None) is state.head
                            for layer in layers.values()))
    if (set(state.constructed) != state.intercepted
            or getattr(inner, "embed_tokens", None) is not state.embedding
            or (state.draft_heads and not heads_shared)
            or len(layers) != state.layer_count
            or any(isinstance(module, _UnallocatedVocabulary) for module in draft.modules())):
        raise RuntimeError("Tessera MTP stock target sharing did not replace every vocabulary placeholder")


def _wrap_load(original: Any, *, returns_draft: bool, draft_heads: bool):
    @wraps(original)
    def load(self, target_model, *args, **kwargs):
        state = (_target_load(self.vllm_config, target_model, draft_heads)
                 if _eligible(self.vllm_config) else None)
        # Nested unrelated loads explicitly suspend the outer interception.
        token = _LOAD.set(state)
        try:
            result = original(self, target_model, *args, **kwargs)
            if state is not None:
                _verify(result if returns_draft else self.model, state)
            return result
        finally:
            _LOAD.reset(token)
    return load


def draft_load_rename() -> tuple[str, str] | None:
    """The recognized GLM MTP draft's in-code checkpoint rename, if it has one.

    None when the running vLLM matches no inspected interface (the draft lookup
    then refuses as it always did), or when the class declares its own mapper.
    """
    with _INSTALL_LOCK:
        match = _supported_interface()
    return None if match is None else match[0].draft_load_rename


def install_for_current_config() -> None:
    """Called during target construction, before either stock drafter loads."""
    from vllm import config as vllm_config

    getter = getattr(vllm_config, "get_current_vllm_config_or_none", None)
    current = getter() if getter is not None else None
    if not _eligible(current):
        return
    with _INSTALL_LOCK:
        match = _supported_interface()
        if match is None:
            return
        interface, modules = match
        _install(interface, *modules)


def _install(interface: _Interface, glm: Any, deepseek: Any, v1: Any, v2: Any, eagle: Any,
             vocab: Any, base_loader: Any, loader_utils: Any) -> None:
    if getattr(glm, "_tessera_mtp_lifetime", False):
        return
    _require_supported_sources(interface, glm, deepseek, v1, v2, eagle, vocab, base_loader,
                               loader_utils)
    for method, names in (
        (v1.SpecDecodeBaseProposer.load_model, ("self", "target_model")),
        (v2.MTPSpeculator.load_draft_model, ("self", "target_model", "target_attn_layer_names")),
        (glm.Glm5NextMTP.load_weights, ("self", "weights")),
    ):
        if tuple(inspect.signature(method).parameters) != names:
            raise RuntimeError("Tessera MTP unsupported draft load signature")
    original_weights = glm.Glm5NextMTP.load_weights

    @wraps(original_weights)
    def load_weights(self, weights):
        state = _LOAD.get()
        if state is None:
            return original_weights(self, weights)
        return original_weights(self, ((name, weight) for name, weight in weights
                                       if not state.shared_weight(name)))

    glm.VocabParallelEmbedding = _constructor(glm.VocabParallelEmbedding, head=False)
    if interface.draft_heads:
        deepseek.ParallelLMHead = _constructor(deepseek.ParallelLMHead, head=True)
    glm.Glm5NextMTP.load_weights = load_weights
    v1.SpecDecodeBaseProposer.load_model = _wrap_load(
        v1.SpecDecodeBaseProposer.load_model, returns_draft=False,
        draft_heads=interface.draft_heads)
    v2.MTPSpeculator.load_draft_model = _wrap_load(
        v2.MTPSpeculator.load_draft_model, returns_draft=True,
        draft_heads=interface.draft_heads)
    glm._tessera_mtp_lifetime = True
    _log.info("Tessera MTP draft vocabulary saving installed for interface %s "
              "(intercepts: embed_tokens%s)", interface.name,
              ", per-layer shared_head.head" if interface.draft_heads else "")
