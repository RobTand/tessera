"""CPU owner of routed expert class coordinates and schedule validation.

Maps index storage slots; source and plan identities always use global ids.
Class profiles follow the schema's w13 (gate, up), w2 (down) projection order.
"""
from __future__ import annotations

from collections.abc import Mapping


def _group_arities():
    # Lazy because scheme delegates metadata validation back to this module.
    from tessera.serving.scheme import MOE_GROUP_ROLES

    return MOE_GROUP_ROLES


def _integer(value, name, target, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{target}: {name} must be an integer >= {minimum}, got {value!r}")
    return value


def inverse_expert_ids(expert_ids) -> list[int]:
    """Validate a strict storage-to-global bijection and invert it once."""
    if not isinstance(expert_ids, (list, tuple)) or not expert_ids:
        raise ValueError("expert_ids must be a nonempty integer bijection")
    inverse = [-1] * len(expert_ids)
    for storage, original in enumerate(expert_ids):
        _integer(original, f"expert_ids[{storage}]", "expert_ids")
        if original >= len(inverse) or inverse[original] != -1:
            raise ValueError("expert_ids must be a bijection over [0, experts): duplicate or out-of-range id")
        inverse[original] = storage
    return inverse


def storage_expert_ids(inverse, expert_ids):
    """Map router IDs ``[T, top_k]`` to storage IDs through the one device inverse.

    ``index_select`` rejects negative IDs rather than wrapping them like
    advanced indexing.  Positions, and so routing weights, are unchanged.
    """
    return inverse.index_select(0, expert_ids.reshape(-1)).reshape_as(expert_ids)


def _profile(q256, target):
    arities = _group_arities()
    if not isinstance(q256, Mapping) or set(q256) != set(arities):
        raise ValueError(f"{target}: class q256 must declare exactly w13 and w2")
    result = {}
    for group, arity in arities.items():
        row = q256[group]
        if not isinstance(row, (list, tuple)) or len(row) != arity:
            raise ValueError(f"{target}: class q256 {group} must have {arity} projection rung(s)")
        result[group] = [_integer(r, f"q256.{group}[{j}]", target, minimum=1)
                         for j, r in enumerate(row)]
    return result


def _profile_key(profile):
    return tuple(r for group in _group_arities() for r in profile[group])


def _matrices(group_q256, target):
    arities = _group_arities()
    if not isinstance(group_q256, Mapping) or set(group_q256) != set(arities):
        raise ValueError(f"{target}: group rung matrices must declare exactly w13 and w2")
    if any(not isinstance(group_q256[group], (list, tuple)) for group in arities):
        raise ValueError(f"{target}: group rung matrices must be expert-major lists")
    experts = len(group_q256[next(iter(arities))])
    if experts < 1 or any(len(group_q256[group]) != experts for group in arities):
        raise ValueError(f"{target}: group rung matrices must cover the same nonempty expert population")
    return [_profile({group: group_q256[group][e] for group in arities}, target)
            for e in range(experts)]


def normalize_expert_classes(expert_classes, experts: int, *, target="expert classes") -> list[dict]:
    """Validate ordered nonempty contiguous classes covering the population."""
    _integer(experts, "experts", target, minimum=1)
    if not isinstance(expert_classes, (list, tuple)) or not expert_classes:
        raise ValueError(f"{target}: expert_classes is required and must be nonempty")
    result, covered, previous = [], 0, None
    for index, descriptor in enumerate(expert_classes):
        where = f"{target} expert_classes[{index}]"
        if not isinstance(descriptor, Mapping) or set(descriptor) != {"start", "end", "q256"}:
            raise ValueError(f"{where}: descriptor requires start, end and q256 only")
        start = _integer(descriptor["start"], "start", where)
        end = _integer(descriptor["end"], "end", where)
        if start != covered or not start < end <= experts:
            raise ValueError(f"{where}: classes must be nonempty contiguous partitions without holes or overlaps")
        profile = _profile(descriptor["q256"], where)
        key = _profile_key(profile)
        if previous is not None and key <= previous:
            raise ValueError(f"{where}: class profiles must be strictly ordered; equal profiles form one class")
        result.append({"start": start, "end": end, "q256": profile})
        covered, previous = end, key
    if covered != experts:
        raise ValueError(f"{target}: expert_classes must cover [0, experts) exactly")
    return result


def normalize_expert_metadata(expert_ids, expert_classes, group_q256, *, target="expert metadata") -> dict:
    """Validate the map, classes and STORAGE-ordered rung matrices together."""
    profiles = _matrices(group_q256, target)
    if expert_ids is None:
        raise ValueError(f"{target}: expert_ids is required")
    inverse_expert_ids(expert_ids)
    if len(expert_ids) != len(profiles):
        raise ValueError(f"{target}: expert_ids must cover the declared experts")
    classes = normalize_expert_classes(expert_classes, len(profiles), target=target)
    for descriptor in classes:
        start, end = descriptor["start"], descriptor["end"]
        if any(profiles[e] != descriptor["q256"] for e in range(start, end)):
            raise ValueError(f"{target}: class q256 disagrees with storage-ordered group rungs")
        ids = expert_ids[start:end]
        if list(ids) != sorted(ids):
            raise ValueError(f"{target}: original expert ids must be ordered within each class")
    return {"expert_ids": list(expert_ids), "expert_classes": classes}


def build_expert_metadata(group_q256, *, target="expert metadata") -> dict:
    """Sort ORIGINAL-ordered profiles by complete profile then global id."""
    profiles = _matrices(group_q256, target)
    ids = sorted(range(len(profiles)), key=lambda e: (_profile_key(profiles[e]), e))
    classes = []
    for storage, original in enumerate(ids):
        profile = profiles[original]
        if classes and classes[-1]["q256"] == profile:
            classes[-1]["end"] = storage + 1
        else:
            classes.append({"start": storage, "end": storage + 1, "q256": profile})
    return {"expert_ids": ids, "expert_classes": classes}


def validate_gate_up_schedule(gate_q: int, up_q: int, columns: int, *, target="expert schedule") -> None:
    """The native fused entry shares one gate/up schedule and tile stride.

    No allowed-rate menu is introduced: the existing grammar supplies every
    one- or two-run schedule. Down has its own independent schedule.
    """
    from tessera.grammar import bresenham_rate_schedule, root_from_q256

    _integer(columns, "columns", target, minimum=1)
    gate = bresenham_rate_schedule(root_from_q256(gate_q), columns, cap=None)
    up = bresenham_rate_schedule(root_from_q256(up_q), columns, cap=None)
    if gate != up:
        raise ValueError(f"{target}: unservable_gate_up_schedule: gate and up must share schedule and tile stride")
