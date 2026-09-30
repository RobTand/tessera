"""Avoid vocabulary storage that stock GLM MTP loading discards before use.

The target must NOT be exposed to draft loading or post-processing. Parameter-
free placeholders survive only until stock unconditional MTP target sharing.
Context-local constructor dispatch leaves unrelated threads/loads unchanged.
Only the inspected fd4a15126 interface is supported; this is not image evidence.
"""
from __future__ import annotations

# vLLM is optional and absent from the device-less development interpreter.
# pyright: reportMissingImports=false

from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import hashlib
import inspect
from pathlib import Path
from threading import RLock
from typing import Any

import torch

# Source copies inspected for tessera#645: the GLM image's mtp-census/vllm-src,
# and fd4a15126's V2 mtp/speculator.py and eagle/utils.py. No image pin changes.
_SOURCE_DIGESTS = (
    "45124573a928ecd76e6bd4f595aec3c3d98d81a71fc7a41fb32564ace560b7d1",
    "e9270724c39a0152dc0a66b94622ebddd384c592534cbbf38d2f43c0ba1592d0",
    "86fdf5cc84b35ff568ba8d0ab067ce827c1b32f3c5f5e49b9bcdd8f89accfb81",
    "1fcffbf5e5a85e4c901bd71c65a73da814e7273627276dbdfebf367720a0bc1a",
    "65d882b8fb476eb0dd6161247346cd7b0392b22832abf36fbb11e795979f9e28",
    "a187d541a61455d6967b98ff3b9837a752e4f5bd5ab9c91de4aeca881f612c3a",
    "f17471235e9e349dee6a2b3a795979187c51cc69f26d238e47d8e37c0766b8d4",
    "6cd928c158703de94223b056d25fb58b38db49befb2f3c7aee1b5e97daca1e86",
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
    constructed: list[str] = field(default_factory=list)

    @property
    def heads(self) -> set[str]:
        return {f"model.layers.{i}.head" for i in
                range(self.first_layer, self.first_layer + self.layer_count)}

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


def _require_supported_sources(*modules: Any) -> None:
    for module, digest in zip(modules, _SOURCE_DIGESTS, strict=True):
        require_source_digest(module, digest)


def _eligible(config: Any) -> bool:
    speculative = getattr(config, "speculative_config", None)
    draft = getattr(speculative, "draft_model_config", None)
    hf = getattr(draft, "hf_config", None)
    return (getattr(speculative, "method", None) == "mtp"
            and getattr(hf, "architectures", None) == ["Glm5NextMTPModel"]
            and type(getattr(config, "quant_config", None)).__module__
            == "tessera.serving.config")


def _target_load(config: Any, target: Any) -> _DraftLoad:
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
    for module, cls in ((embed, VocabParallelEmbedding), (head, ParallelLMHead)):
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
    if embed.weight.device != head.weight.device:
        raise RuntimeError("Tessera MTP target vocabulary devices disagree")
    if hf.num_nextn_predict_layers <= 0 or hf.num_hidden_layers < 0:
        raise RuntimeError("Tessera MTP invalid draft layer range")
    return _DraftLoad(embed, head, hf.vocab_size, hf.hidden_size,
                      hf.num_hidden_layers, hf.num_nextn_predict_layers, config.quant_config)


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
    expected = state.heads | {"model.embed_tokens"}
    if (set(state.constructed) != expected
            or getattr(inner, "embed_tokens", None) is not state.embedding
            or getattr(draft, "lm_head", None) is not state.head
            or len(layers) != state.layer_count
            or any(getattr(getattr(layer, "shared_head", None), "head", None) is not state.head
                   for layer in layers.values())
            or any(isinstance(module, _UnallocatedVocabulary) for module in draft.modules())):
        raise RuntimeError("Tessera MTP stock target sharing did not replace every vocabulary placeholder")


def _wrap_load(original: Any, *, returns_draft: bool):
    @wraps(original)
    def load(self, target_model, *args, **kwargs):
        state = _target_load(self.vllm_config, target_model) if _eligible(self.vllm_config) else None
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


def install_for_current_config() -> None:
    """Called during target construction, before either stock drafter loads."""
    from vllm import config as vllm_config

    getter = getattr(vllm_config, "get_current_vllm_config_or_none", None)
    current = getter() if getter is not None else None
    if not _eligible(current):
        return
    from vllm.model_executor.layers import vocab_parallel_embedding as vocab
    from vllm.model_executor.model_loader import base_loader
    from vllm.model_executor.model_loader import utils as loader_utils
    from vllm.model_executor.models import deepseek_mtp as deepseek
    from vllm.models.glm5next.nvidia import mtp as glm
    from vllm.v1.spec_decode import llm_base_proposer as v1
    from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle
    from vllm.v1.worker.gpu.spec_decode.mtp import speculator as v2

    with _INSTALL_LOCK:
        _install(glm, deepseek, v1, v2, eagle, vocab, base_loader, loader_utils)


def _install(glm: Any, deepseek: Any, v1: Any, v2: Any, eagle: Any,
             vocab: Any, base_loader: Any, loader_utils: Any) -> None:
    if getattr(glm, "_tessera_mtp_lifetime", False):
        return
    _require_supported_sources(glm, deepseek, v1, v2, eagle, vocab, base_loader, loader_utils)
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
    deepseek.ParallelLMHead = _constructor(deepseek.ParallelLMHead, head=True)
    glm.Glm5NextMTP.load_weights = load_weights
    v1.SpecDecodeBaseProposer.load_model = _wrap_load(v1.SpecDecodeBaseProposer.load_model,
                                                    returns_draft=False)
    v2.MTPSpeculator.load_draft_model = _wrap_load(v2.MTPSpeculator.load_draft_model,
                                                 returns_draft=True)
    glm._tessera_mtp_lifetime = True
