"""Read only the checkpoint shards the GLM MTP draft's own layers live in (tessera#777).

Stock vLLM loads the MTP draft through ``DefaultModelLoader.get_all_weights``,
whose ``_prepare_weights`` globs every ``*.safetensors`` shard in the model
folder. The draft therefore reads the whole checkpoint (163.42 GiB, 418 s on
the served GLM-5.3 artifact over NFS) to keep the 5.52 GiB of it in four
shards. ``Glm5NextMTP.load_weights`` skips every name for which
``get_spec_layer_idx_from_weight_name`` returns None (after its own
``model.language_model.`` rename) before it does anything else with the
tensor, so a shard that holds no spec-layer tensor contributes nothing to the
draft. This module narrows the shard list ``_prepare_weights`` returns, for
the draft's load only, to the shards that ``model.safetensors.index.json``
maps a spec-layer tensor to. The target's load is untouched.

``allow_patterns_overrides`` cannot express this: ``_prepare_weights`` stops
at the first pattern that matches any file, so a list of shard names would
load only the first shard.

Supported only on inspected interfaces. The GLM MTP module (whose
``load_weights`` makes the argument above) is identified by the draft-lifetime
interface this one is keyed to (``mtp_draft_lifetime._INTERFACES``); this
module adds the sha256 of ``default_loader`` and of the GLM ``model`` module,
where the spec-layer rule lives. A serve whose sources match none of them
declines: the draft reads the whole checkpoint as stock, one warning is
logged, and nothing fails (tessera#749). D32: a source-digest mismatch is a
run-identity seal, not the refusal it was -- default dev mode stamps one
``[DEV-MODE]`` line per module and the narrowing still installs (the
whole-checkpoint read is the certified-mode fallback);
``PRISMAQUANT_DEV_MODE=0`` keeps the verbatim decline. An unreadable source
and a rebound loader signature still fail closed in both modes. The narrowing also declines, with a
warning, for a load the inspection did not cover: an EP weight filter, the
mm-encoder-only filter, a draft that sets its own ``allow_patterns_overrides``
or ``secondary_weights``, a model path that is not a local folder, no index
file, or an index with no spec-layer tensor.

After a complete draft load the wrapper compares the number of spec-layer
tensors the narrowed stream yielded with the number the index lists, and
raises on a difference. That check runs only once the stream is exhausted: it
is not a guard before reading, and an exception during the load propagates as
it would without this module.
"""
from __future__ import annotations

# vLLM is optional and absent from the device-less development interpreter.
# pyright: reportMissingImports=false

from dataclasses import dataclass
from functools import wraps
import hashlib
import importlib
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

from ..dev_mode import seal_check

_log = logging.getLogger(__name__)

# The modules this patch reads or rebinds, in the digest order of
# ``_ShardInterface.digests``.
_MODULES = (
    "vllm.model_executor.model_loader.default_loader",
    "vllm.models.glm5next.common.model",
)

_GET_ALL_WEIGHTS = ("self", "model_config", "model")
_PREPARE_WEIGHTS = ("self", "model_name_or_path", "subfolder", "revision",
                    "fall_back_to_pt", "allow_patterns_overrides")


@dataclass(frozen=True)
class _ShardInterface:
    """One inspected loader interface, keyed to a draft-lifetime interface name."""
    name: str
    digests: tuple[str, ...]  # _MODULES order


_INTERFACES = (
    # tessera#777: sha256 of each module inside
    # localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705... (U4_STACK
    # nightly-20260929, vLLM 0.30.1rc1.dev336+gaf5b4857e). common/mtp.py there
    # (b0a8f474..., pinned by mtp_draft_lifetime) skips non-spec names at the
    # top of the load_weights loop, before any side effect.
    _ShardInterface(
        "nightly-20260929",
        ("d17ab54969489de2338bea88190b4af78c8a1f66e866796f612ff554eea3cca9",
         "9ad4952e048bdec4991327e2877fae8d1385743a371bf69a2de90610d4d3b68e")),
)

# Idempotency flag on the loader class, separate from the lifetime install's
# flag, so a loader mismatch declines only the narrowing.
_INSTALLED = "_tessera_mtp_draft_shards"
# The plan for the draft load in progress, on the loader instance.
_PLAN = "_tessera_mtp_draft_shard_plan"


@dataclass(frozen=True)
class ShardPlan:
    shards: frozenset[str]   # shard file names, as the index names them
    spec_tensors: int        # index entries that are spec-layer tensors
    total_shards: int        # shard files the index names


def renamed(name: str, rename: tuple[str, str] | None) -> str:
    """The name ``load_weights`` tests, after the draft class's own rename."""
    if rename is not None and name.startswith(rename[0]):
        return name.replace(rename[0], rename[1], 1)
    return name


def plan_draft_shards(index_path: str | os.PathLike[str], config: Any,
                      spec_layer: Callable[[Any, str], int | None],
                      rename: tuple[str, str] | None) -> ShardPlan | None:
    """The shards holding the draft's spec-layer tensors, from the index alone.

    None when the index is unreadable or lists no spec-layer tensor.
    """
    try:
        weight_map = json.loads(Path(index_path).read_text())["weight_map"]
        items = list(weight_map.items())
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    shards: set[str] = set()
    count = 0
    for name, shard in items:
        if spec_layer(config, renamed(name, rename)) is not None:
            shards.add(shard)
            count += 1
    if not shards:
        return None
    return ShardPlan(frozenset(shards), count, len({shard for _, shard in items}))


def _import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - any import failure is a non-match
        _log.warning("Tessera MTP draft reads the whole checkpoint: %s is not importable "
                     "(%s: %s) (tessera#777)", name, type(exc).__name__, exc)
        return None


def install(interface_name: str, glm: Any, rename: tuple[str, str] | None) -> None:
    """Narrow the GLM MTP draft's shard list on a recognized loader interface.

    ``interface_name`` and ``rename`` come from the recognized draft-lifetime
    interface; ``glm`` is its GLM MTP module. Declines with one warning when
    the loader sources match no inspected interface.
    """
    match = next((i for i in _INTERFACES if i.name == interface_name), None)
    if match is None:
        _log.warning("Tessera MTP draft reads the whole checkpoint: no inspected loader "
                     "interface for draft interface %s (tessera#777)", interface_name)
        return
    modules = tuple(_import(name) for name in _MODULES)
    if any(module is None for module in modules):
        return
    loader, model = modules
    cls = loader.DefaultModelLoader
    if getattr(cls, _INSTALLED, False):
        return
    try:
        for module, digest in zip(modules, match.digests, strict=True):
            # D32: the inspected loader source digest is a run-identity seal
            # (source-source, not bytes against their own digest). Dev mode
            # stamps one [DEV-MODE] line per module and the narrowing still
            # installs -- the whole-checkpoint fallback is the certified-mode
            # decline. The loader signature checks below still fail closed in
            # both modes, and an unreadable source refuses in both.
            path = getattr(module, "__file__", None)
            try:
                actual = hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
            except OSError as exc:
                raise RuntimeError(f"Tessera MTP source identity unreadable: {path}") from exc
            seal_check("MTP loader source identity", digest, actual,
                       where="the draft shard narrowing",
                       refusal=RuntimeError(
                           f"Tessera MTP unsupported source identity for {module.__name__}: "
                           f"expected {digest}, got {actual}; the draft shard narrowing "
                           "requires the inspected loader interface"))
    except RuntimeError as exc:
        _log.warning("Tessera MTP draft reads the whole checkpoint: %s (tessera#777)", exc)
        return
    # Recognized sources with another signature means something else rebound
    # them: fail closed, as the lifetime install does.
    if (tuple(inspect.signature(cls.get_all_weights).parameters) != _GET_ALL_WEIGHTS
            or tuple(inspect.signature(cls._prepare_weights).parameters) != _PREPARE_WEIGHTS):
        raise RuntimeError("Tessera MTP draft shards: unsupported loader signature")
    _install(cls, glm.Glm5NextMTP, model.get_spec_layer_idx_from_weight_name,
             loader.SAFE_WEIGHTS_INDEX_NAME, rename)
    _log.info("Tessera MTP draft shard narrowing installed for interface %s", match.name)


def _install(cls: Any, draft_cls: type, spec_layer: Callable[[Any, str], int | None],
             index_name: str, rename: tuple[str, str] | None) -> None:
    original_all = cls.get_all_weights
    original_prepare = cls._prepare_weights

    def draft_plan(loader: Any, model_config: Any, model: Any) -> ShardPlan | None:
        if not isinstance(model, draft_cls):
            return None
        folder = getattr(model_config, "model", None)
        reasons = []
        if getattr(model, "allow_patterns_overrides", None) is not None:
            reasons.append("the draft sets allow_patterns_overrides")
        if getattr(model, "secondary_weights", ()):
            reasons.append("the draft declares secondary_weights")
        if getattr(loader, "local_expert_ids", None) is not None:
            reasons.append("an EP weight filter is active")
        if getattr(loader, "_encoder_only_lm_prefixes", None) is not None:
            reasons.append("the mm-encoder-only filter is active")
        if not (isinstance(folder, str) and os.path.isdir(folder)):
            reasons.append(f"the model path {folder!r} is not a local folder")
        elif not os.path.isfile(os.path.join(folder, index_name)):
            reasons.append(f"no {index_name} in {folder}")
        plan = None
        if not reasons:
            plan = plan_draft_shards(os.path.join(folder, index_name), model.config,
                                     spec_layer, rename)
            if plan is None:
                reasons.append(f"{index_name} lists no spec-layer tensor")
        if reasons:
            _log.warning("Tessera MTP draft reads the whole checkpoint: %s (tessera#777)",
                         "; ".join(reasons))
        return plan

    @wraps(original_prepare)
    def _prepare_weights(self, model_name_or_path, subfolder, revision, fall_back_to_pt,
                         allow_patterns_overrides):
        folder, files, use_safetensors, index_file = original_prepare(
            self, model_name_or_path, subfolder, revision, fall_back_to_pt,
            allow_patterns_overrides)
        plan = getattr(self, _PLAN, None)
        if plan is None:
            return folder, files, use_safetensors, index_file
        if not use_safetensors or index_file != index_name or subfolder is not None:
            _log.warning("Tessera MTP draft reads the whole checkpoint: the loader resolved "
                         "safetensors=%s, index %s, subfolder %s (tessera#777)",
                         use_safetensors, index_file, subfolder)
            setattr(self, _PLAN, None)
            return folder, files, use_safetensors, index_file
        found = {os.path.basename(f) for f in files}
        missing = plan.shards - found
        if missing:
            raise RuntimeError(
                f"Tessera MTP draft shards: {index_name} maps spec-layer tensors to "
                f"shards the loader did not find: {sorted(missing)}")
        narrowed = [f for f in files if os.path.basename(f) in plan.shards]
        size = lambda fs: sum(os.path.getsize(f) for f in fs)  # noqa: E731
        _log.info("Tessera MTP draft reads %d of %d checkpoint shards (%.2f of %.2f GiB) "
                  "for its %d spec-layer tensors (tessera#777)", len(narrowed), len(files),
                  size(narrowed) / 2**30, size(files) / 2**30, plan.spec_tensors)
        return folder, narrowed, use_safetensors, index_file

    @wraps(original_all)
    def get_all_weights(self, model_config, model):
        plan = draft_plan(self, model_config, model)
        if plan is None:
            yield from original_all(self, model_config, model)
            return
        setattr(self, _PLAN, plan)
        seen = 0
        try:
            for name, tensor in original_all(self, model_config, model):
                if spec_layer(model.config, renamed(name, rename)) is not None:
                    seen += 1
                yield name, tensor
            narrowed = getattr(self, _PLAN, None) is not None
        finally:
            setattr(self, _PLAN, None)
        # Only after a complete load, and only when the narrowing applied.
        if narrowed and seen != plan.spec_tensors:
            raise RuntimeError(
                f"Tessera MTP draft shards: the narrowed load yielded {seen} spec-layer "
                f"tensors, the index lists {plan.spec_tensors}")
        _log.info("Tessera MTP draft load yielded %d spec-layer tensors (index: %d)",
                  seen, plan.spec_tensors)

    cls._prepare_weights = _prepare_weights
    cls.get_all_weights = get_all_weights
    setattr(cls, _INSTALLED, True)
