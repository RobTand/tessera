"""The serving-owned half of the task endpoint runtime witness (tessera#1056).

The D50 task adapter needs the runtime facts a serve OBSERVED: the listener
endpoint and served alias, the launch attempt and the complete rank set,
the loaded artifact bytes each rank observed, and the server tokenizer
facts. :mod:`tessera.endpoint_observer` reads what an outside process can
read (an HTTP reply, file bytes). This module reads what only the serving
process can read: the rank identity the distributed group established,
the model path the worker loaded, the wire bytes each loaded module holds,
and the tokenizer the engine initialized.

BOUNDARIES. The producer lives here, inside ``tessera.serving``: it may
import torch and vLLM, and it runs inside the worker (see
``tools/tessera_route_census.py`` for the collective pattern). The consumer
-- the D50 task adapter and ``tools/verify_endpoint_witness.py`` -- reads
JSON only and never imports this module. :mod:`tessera.endpoint_witness`
owns the join and its refusals; this module owns the runtime reads.

WHAT EACH READ ESTABLISHES. :func:`observe_worker_identity` returns the
rank and world size from vLLM's own world group (``torch.distributed``
fallback), never a defaulted zero: an uninitialized group refuses, because
a defaulted rank is indistinguishable from a genuine rank 0 (tessera#509).
:func:`observe_worker_model` returns the model path and served names from
the worker's own ``model_config``, and the tokenizer path and vocabulary
length from the engine's initialized tokenizer when one exists. The served
alias the listener reports must equal one of those served names, or the
listener is not this serve's. :func:`observe_loaded_wires` walks the
loaded model, reads each Tessera module's resident wire state, and hashes
it: the bytes the rank serves, not the files beside them.

LIFETIME. All three reads carry the same ``observed_unix`` stamp and the
caller's ``lifetime_id``. The join refuses observations from different
lifetimes. The probe takes all three while the listener still answers, so
a stopped or replaced listener refuses instead of joining stale bytes to
a live alias.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable, Mapping

__all__ = [
    "observe_loaded_wires",
    "observe_worker_identity",
    "observe_worker_model",
]


def _stamp(clock: Callable[[], float] | None) -> float:
    import time

    observed = clock() if clock is not None else time.time()
    if not isinstance(observed, (int, float)) or not observed > 0:
        raise ValueError("observation clock returned no positive time")
    return observed


def _text(value: str, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} is not a non-empty string")
    return value


def observe_worker_identity(worker: Any, *, lifetime_id: str,
                            clock: Callable[[], float] | None = None) -> dict:
    """The rank and world size this worker serves, from its own group.

    Runs inside the worker. Reads vLLM's world group first and
    ``torch.distributed`` second; refuses when neither is initialized,
    because a defaulted rank 0 is indistinguishable from a genuine one.
    """
    _text(lifetime_id, "worker lifetime_id")
    rank = world_size = source = None
    try:
        from vllm.distributed.parallel_state import get_world_group

        group = get_world_group()
        rank, world_size = int(group.rank), int(group.world_size)
        source = "vllm.world_group"
    except Exception:  # noqa: BLE001 -- the fallback below is the same question
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                rank, world_size = int(dist.get_rank()), int(dist.get_world_size())
                source = "torch.distributed"
        except Exception:  # noqa: BLE001 -- uninitialized reads as unavailable
            pass
    if rank is None or world_size is None:
        raise ValueError("the worker's distributed group is not initialized; "
                         "rank identity is unavailable, never defaulted")
    if rank < 0 or world_size <= 0 or rank >= world_size:
        raise ValueError(f"the worker reports rank {rank} of {world_size}, "
                         "which is not a rank of its world")
    local_rank = getattr(worker, "local_rank", None)
    if local_rank is None:
        try:
            from vllm.distributed.parallel_state import get_world_group

            local_rank = int(getattr(get_world_group(), "local_rank", 0))
        except Exception:  # noqa: BLE001 -- the attribute may not exist
            local_rank = 0
    return {"rank": rank, "local_rank": int(local_rank), "world_size": world_size,
            "rank_source": source, "lifetime_id": lifetime_id,
            "observed_unix": _stamp(clock)}


def observe_worker_model(worker: Any, *, lifetime_id: str,
                         clock: Callable[[], float] | None = None,
                         tokenizer: Any = None) -> dict:
    """The model path, served names, tokenizer path and vocabulary this worker serves.

    Runs inside the worker. ``model`` and ``served_model_name`` come from
    the worker's own ``model_config`` -- the table the engine itself
    dispatches on -- never from an argument the probe was given. The
    tokenizer facts come from the engine's initialized tokenizer when the
    caller hands it over (``LLM.get_tokenizer()`` on the driver reads the
    same object the serve tokenizes with); without one the observation
    refuses, because client metadata is not server state.
    """
    _text(lifetime_id, "model lifetime_id")
    config = getattr(worker, "model_config", None)
    if config is None:
        config = getattr(getattr(worker, "vllm_config", None), "model_config", None)
    if config is None:
        raise ValueError("the worker carries no model_config to observe")
    model = getattr(config, "model", None)
    _text(model, "worker model_config.model")
    served = getattr(config, "served_model_name", None)
    if served is None:
        served_names = [model]
    elif isinstance(served, str):
        served_names = [served] if served else [model]
    else:
        served_names = [name for name in served if isinstance(name, str) and name]
        if not served_names:
            served_names = [model]
    tokenizer_path = getattr(config, "tokenizer", None) or model
    if tokenizer is None:
        raise ValueError("the server tokenizer was not handed over; "
                         "client metadata never establishes server state")
    try:
        vocab_size = len(tokenizer)
    except TypeError as exc:
        raise ValueError("the server tokenizer states no vocabulary length") from exc
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("the server tokenizer states no vocabulary length")
    return {"model": model, "served_names": served_names,
            "tokenizer_path": tokenizer_path, "vocab_size": vocab_size,
            "vocab_source": "server-tokenizer",
            "lifetime_id": lifetime_id, "observed_unix": _stamp(clock)}


def _wire_state(module: Any) -> bytes | None:
    """The loaded wire bytes a Tessera module serves, or None when it holds none.

    Reads the module's own ``resident_tensors`` declaration -- the same
    references the residency observer charges -- and hashes every declared
    tensor's bytes. A module whose method declares nothing contributes
    nothing, and the caller refuses the gap by name instead of hashing an
    absence.
    """
    method = getattr(module, "quant_method", None)
    resident = getattr(method, "resident_tensors", None)
    if not callable(resident):
        return None
    try:
        declared = list(resident(module))
    except Exception:  # noqa: BLE001 -- an unreadable declaration refuses below
        return None
    if not declared:
        return None
    digest = hashlib.sha256()
    for name, tensor in sorted(declared, key=lambda item: str(item[0])):
        try:
            raw = tensor.detach().cpu().contiguous().numpy().tobytes()
        except Exception:  # noqa: BLE001 -- an unreadable tensor refuses below
            return None
        digest.update(str(name).encode("utf-8") + b"\0")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.digest()


def _is_tessera_module(module: Any) -> bool:
    """Whether the loader prepared this module as a Tessera module."""
    if getattr(module, "tessera_native", None) is not None:
        return True
    if getattr(module, "tessera_a4_roles", None) is not None:
        return True
    if getattr(module, "tessera_routed_fused", None) is not None:
        return True
    method = getattr(module, "quant_method", None)
    return type(method).__module__.startswith("tessera.serving.")


def observe_loaded_wires(model: Any, *, lifetime_id: str,
                         clock: Callable[[], float] | None = None) -> dict:
    """The wire digests each loaded Tessera module of this worker serves.

    Runs inside the worker. Walks ``named_modules()``, reads each Tessera
    module's resident wire bytes with its layer TP coordinates, and hashes
    them. The coordinates come from the layer's own shard plan -- the same
    table the loader cut the module by -- never from the process rank, so a
    one-rank layer inside a four-rank process reports its own identity
    (tessera#303). Refuses a model with no loaded Tessera module: an empty
    roster is not evidence about an empty serve, it is evidence about
    nothing.
    """
    _text(lifetime_id, "wires lifetime_id")
    named = getattr(model, "named_modules", None)
    if not callable(named):
        raise ValueError("the worker model exposes no named_modules to observe")
    wires: dict[str, str] = {}
    coords: dict[str, list[int]] = {}
    for name, module in named():
        prefix = getattr(module, "prefix", "") or name
        if not _is_tessera_module(module):
            continue
        state = _wire_state(module)
        if state is None:
            raise ValueError(f"loaded module {prefix!r} holds no readable wire state")
        plan = getattr(module, "tessera_shard_plan", None)
        rank = getattr(plan, "tp_rank", getattr(module, "tp_rank", None))
        size = getattr(plan, "tp_size", getattr(module, "tp_size", None))
        if type(rank) is not int or type(size) is not int or size < 1 or not 0 <= rank < size:
            raise ValueError(f"loaded module {prefix!r} carries no layer TP coordinates")
        wires[prefix] = hashlib.sha256(state).hexdigest()
        coords[prefix] = [rank, size]
    if not wires:
        raise ValueError("the worker model holds no loaded Tessera module")
    total = sum(len(name) + 32 for name in wires)
    return {"wires": wires, "coords": coords, "modules": len(wires), "bytes": total,
            "lifetime_id": lifetime_id, "observed_unix": _stamp(clock)}
