"""Replay the runtime's explicit mapper API without depending on vLLM imports."""

from __future__ import annotations

import re


_GLM_MTP_DRAFT_ARCHITECTURE = "Glm5NextMTPModel"
_GLM_LAYER_PREFIX = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def glm5next_mtp_module_prefix(declared: str, *, architecture: str,
                               first_layer: int, count: int) -> str | None:
    """The stock GLM MTP class's extra decoder-block segment, in module space.

    ``Glm5NextMTP.hf_to_vllm_mapper`` strips the checkpoint's
    ``model.language_model.`` prefix to ``model.``. Its separately constructed
    ``Glm5NextMultiTokenPredictorLayer`` then owns ``mtp_block`` below each
    speculative layer. The mapper alone does not distinguish that model class
    from another class with the same prefix rule, so the caller must supply the
    actual draft architecture and layer range from vLLM's own configuration.
    This function changes no checkpoint name or cached-unit identity.
    """
    if architecture != _GLM_MTP_DRAFT_ARCHITECTURE:
        return None
    if (type(first_layer) is not int or type(count) is not int
            or first_layer < 0 or count <= 0):
        return None
    match = _GLM_LAYER_PREFIX.fullmatch(declared)
    if match is None:
        return None
    index = int(match.group(1))
    if not first_layer <= index < first_layer + count:
        return None
    remainder = match.group(2)
    if remainder.startswith("mtp_block."):
        return None
    return f"model.layers.{index}.mtp_block.{remainder}"


def module_name_mapper(mapper):
    """Use the same name-only view the runtime hands quantization configs.

    New runtimes expose get_rename_mapper; earlier wrappers use
    get_unstacked_mapper, and plain tables expose neither. An exception from
    an existing method is a broken mapper and propagates rather than falling
    back to a different translation.
    """
    for name in ("get_rename_mapper", "get_unstacked_mapper"):
        unwrap = getattr(mapper, name, None)
        if unwrap is not None:
            return unwrap()
    return mapper
