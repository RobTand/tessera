"""Tensor parallelism: which slice of a unit a rank serves, and who cuts it.

THE ARTIFACT IS TP-AGNOSTIC.  A Tessera checkpoint holds one whole unit per
role, encoded once, and says nothing about how many ranks will read it.  That
is deliberate and it is the only arrangement that survives contact with
deployment: the same bytes serve TP=1 and TP=8, an operator re-shards by
changing a serve flag, and the exporter never learns the topology.  The
alternative -- encoding per rank -- would make the artifact a function of the
machine it was built for, and Tessera units are not concatenations of
independent rows, so a checkpoint cut for 4 ranks could not be re-cut for 8.

WHERE THE CUT HAPPENS.  Every rank loads the WHOLE blob (``wire_bytes`` is a
``BasevLLMParameter``, which vLLM copies rather than splits -- a wire has no
element axis to slice), parses it, and then takes its own shard at the UNIT
level through ``_shard_unit_for_rank``.  Unit-level slicing is not free: the
span-2 trellis carries state along a row, so a column-sliced unit must record
the state its first surviving column starts from.  That is the INITIAL_STATE
plane, and it is why the cut belongs to ``tessera.layout.slice_unit`` -- a
wire-format operation with its own exactness proof -- and not to a serving
route reaching into planes it does not own.

TODAY.  ``tp_size == 1`` is what this plugin serves, and at ``tp_size == 1``
the seam returns the whole unit unchanged -- the same object, so the caller's
parsed view of it stays valid and the served arithmetic is bit-identical to a
build with no TP support at all.  At ``tp_size > 1`` it refuses by name.  It
refuses rather than falling back to "every rank holds the whole weight",
because that would serve correct logits at N times the intended memory and
look merely disappointing.

WHICH AXIS.  vLLM's parallel Linears split one of the two axes and tell the
method by the sizes they ask for, so the axis is DERIVED from the shapes
rather than sniffed off a class name:

* ``ColumnParallelLinear`` / ``MergedColumnParallelLinear`` / ``QKVParallelLinear``
  split the OUTPUT: ``sum(output_partition_sizes) * tp == rows``.  Each fused
  role is split independently -- q, k and v each give every rank its own rows --
  which is why the seam is applied per role, in stacking order.
* ``RowParallelLinear`` splits the INPUT: ``input_size_per_partition * tp ==
  columns``.  One role, cut along columns.

MoE, when it lands: expert parallelism assigns WHOLE units to ranks (no cut at
all, the granularity is one expert), and tensor parallelism inside an expert is
the same row/column cut as here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = [
    "AXIS_ROWS",
    "AXIS_COLUMNS",
    "ShardPlan",
    "plan_shard",
    "can_shard",
    "shard_granularity",
    "tp_rank_and_size",
]

#: The output axis: ColumnParallel and its merged/QKV forms give a rank its own
#: rows.  The SPELLING is ``tessera.layout``'s, not this module's -- these
#: strings are passed straight to ``layout.can_shard(unit, tp, axis)``, and a
#: near-miss ("rows" for "row") would be a GrammarError at load on the one
#: configuration nobody serves yet.  One vocabulary for one concept.
AXIS_ROWS = "row"
#: The input axis: RowParallel gives a rank its own columns.
AXIS_COLUMNS = "column"


@dataclass(frozen=True)
class ShardPlan:
    """What one rank serves of one module.

    ``axis`` is None exactly when there is nothing to cut (``tp_size == 1``),
    and then ``shard_rows``/``shard_columns`` are the module's own.
    """

    prefix: str
    rows: int
    columns: int
    shard_rows: int
    shard_columns: int
    tp_rank: int
    tp_size: int
    axis: Optional[str] = None

    @property
    def is_whole(self) -> bool:
        return self.tp_size == 1 and self.axis is None


def tp_rank_and_size() -> Tuple[int, int]:
    """This process's tensor-parallel coordinates, ``(1, 0)``-safe off vLLM.

    Read from vLLM rather than from an environment variable: the TP group is a
    fact of the engine's own parallel state, and a serve that disagreed with it
    would shard against a group it is not in.
    """
    try:
        from vllm.distributed import (get_tensor_model_parallel_rank,
                                      get_tensor_model_parallel_world_size)
    except Exception:                       # pragma: no cover - no vLLM (producer side)
        return 0, 1
    try:
        return int(get_tensor_model_parallel_rank()), int(get_tensor_model_parallel_world_size())
    except Exception:
        # Called outside an initialised parallel state (unit tests, the
        # producer): one rank is the honest answer, not a guess at a topology.
        return 0, 1


def shard_granularity(unit) -> Optional[Tuple[int, int]]:
    """``(row_gran, col_gran)`` from the wire, or None when the cutter is absent.

    Delegated to ``tessera.layout`` because the granularity is a property of
    the body's own packing (the trellis's step, the plane's superblock), not
    something a serving route may assume.  None means "the branch that can cut
    a unit is not merged yet" -- callers refuse, they do not guess.

    ``layout.shard_granularity`` accepts an ``EncodedUnit``, a ``ParsedUnit``
    or a bare ``Manifest``, so a loader may ask with the parse it already
    holds; a ``ParsedUnit`` is the better argument because the function reads
    the superblock and arity off it instead of defaulting them.
    """
    try:
        from tessera.layout import shard_granularity as _granularity
    except Exception:
        return None
    row_gran, col_gran = _granularity(unit)
    return int(row_gran), int(col_gran)


def can_shard(unit, tp: int, axis: str) -> Optional[bool]:
    """``layout.can_shard``, or None when the cutter is absent.

    Asked BEFORE a cut rather than inferred from the granularity, because the
    granularity is necessary and not sufficient: a column cut of a unit that
    carries a RELEASE plane is confined to whole 256-column superblocks, and
    ``can_shard`` is where that lives.  None means "cannot answer", which
    callers treat as "refuse", never as "yes".
    """
    try:
        from tessera.layout import can_shard as _can_shard
    except Exception:
        return None
    return bool(_can_shard(unit, int(tp), axis))


def plan_shard(prefix: str, *, rows: int, columns: int, out_size: int, in_size: int,
               tp_rank: Optional[int] = None, tp_size: Optional[int] = None) -> ShardPlan:
    """Decide, from the sizes vLLM asks for, which axis this rank is a slice of.

    Refuses -- with both shapes in the message -- when the layer's request is
    neither the whole module nor a clean split of one axis of it.  That is the
    case where a checkpoint and a serve disagree about a module's geometry, and
    silently serving the wrong rows is the failure worth being loud about.
    """
    if tp_rank is None or tp_size is None:
        derived_rank, derived_size = tp_rank_and_size()
        tp_rank = derived_rank if tp_rank is None else tp_rank
        tp_size = derived_size if tp_size is None else tp_size
    rows, columns, out_size, in_size = int(rows), int(columns), int(out_size), int(in_size)
    tp_rank, tp_size = int(tp_rank), int(tp_size)
    if tp_size < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError(f"{prefix}: nonsensical TP coordinates rank {tp_rank} of {tp_size}")
    if out_size == rows and in_size == columns:
        # Whole module.  True at tp_size == 1, and also on a replicated Linear
        # inside a TP group -- which is a whole unit per rank, not a cut.
        return ShardPlan(prefix, rows, columns, rows, columns, tp_rank, tp_size, None)
    if tp_size == 1:
        raise ValueError(
            f"{prefix}: tessera module is {rows}x{columns} but the layer wants "
            f"{out_size}x{in_size} on one rank; a wire is bound to its shape")
    if in_size == columns and out_size * tp_size == rows:
        axis, shard_rows, shard_columns = AXIS_ROWS, out_size, columns
    elif out_size == rows and in_size * tp_size == columns:
        axis, shard_rows, shard_columns = AXIS_COLUMNS, rows, in_size
    else:
        raise ValueError(
            f"{prefix}: tessera module is {rows}x{columns}; rank {tp_rank} of {tp_size} wants "
            f"{out_size}x{in_size}, which is neither the whole module nor a 1/{tp_size} split "
            "of exactly one of its axes")
    return ShardPlan(prefix, rows, columns, shard_rows, shard_columns, tp_rank, tp_size, axis)


def _bounds(extent: int, tp_rank: int, tp_size: int) -> Tuple[int, int]:
    """This rank's half-open span of ``extent``, in equal parts.

    Equal parts only: an uneven split is refused upstream by
    ``check_shard_granularity`` rather than rounded here, because a rank that
    quietly held a different number of rows than its peers would produce a
    wrong all-gather and no error.
    """
    if extent % tp_size:
        raise ValueError(
            f"{extent} does not divide into {tp_size} equal shards; this should have been "
            "refused by check_shard_granularity before any cut was attempted")
    per = extent // tp_size
    return tp_rank * per, (tp_rank + 1) * per


def _shard_unit_for_rank(unit, tp_rank: int, tp_size: int, axis: Optional[str]):
    """THE SEAM.  The unit this rank serves, cut from the whole one.

    ``tp_size == 1`` returns the SAME OBJECT -- identity, not a copy -- so a
    caller holding a parsed view of it may keep that view, and so a TP-capable
    build serves byte-identical bytes to one without this function.

    Above one rank this calls ``tessera.layout.slice_unit``, which returns a
    STANDALONE unit: it decodes through the same ``tessera.decode`` entry
    points to exactly ``decode(parent)[r0:r1, c0:c1]``, bit for bit, with no
    re-encoding.  The caller must re-derive its parsed view of what comes back
    -- a shard's planes are its own, and a stale parse describes the parent's.

    Only this rank's shard is cut, so the cost is O(1) in the TP degree.
    """
    if tp_size == 1:
        return unit
    try:
        from tessera.layout import slice_unit
    except Exception as exc:
        raise NotImplementedError(
            f"tensor parallelism needs tessera.layout.slice_unit, which is not in this build "
            f"({exc}): rank {tp_rank} of {tp_size} asked for a {axis} slice of a unit this "
            "plugin can only serve whole.  Serve with tensor_parallel_size=1, or install a "
            "Tessera carrying the unit slicer.") from exc
    rows, columns = unit.body_bits.shape[0], unit.body_bits.shape[1]
    if axis == AXIS_ROWS:
        r0, r1 = _bounds(rows, tp_rank, tp_size)
        return slice_unit(unit, rows=(r0, r1), cols=(0, columns))
    c0, c1 = _bounds(columns, tp_rank, tp_size)
    return slice_unit(unit, rows=(0, rows), cols=(c0, c1))


def check_shard_granularity(plan: ShardPlan, unit) -> None:
    """Refuse a split the wire cannot express, naming the granularity.

    A unit is not a matrix of independent rows: the body packs along both axes
    and the trellis carries state along a row, so a cut is exact only on the
    packing's own boundaries.  ``tessera.layout.shard_granularity`` is the
    authority on those; when it is absent the caller has already refused at the
    seam, so this is a no-op rather than a guess.

    ``layout.can_shard`` is asked FIRST and is the binding answer: granularity
    is necessary, not sufficient.  A column cut of a unit carrying a RELEASE
    plane is confined to whole 256-column superblocks, and only ``can_shard``
    knows that.  The granularity is then used to say the useful thing in the
    refusal -- a number the operator can act on.
    """
    if plan.axis is None:
        return
    granularity = shard_granularity(unit)
    if granularity is None:
        return
    row_gran, col_gran = granularity
    if can_shard(unit, plan.tp_size, plan.axis) is False:
        extent = plan.rows if plan.axis == AXIS_ROWS else plan.columns
        gran = row_gran if plan.axis == AXIS_ROWS else col_gran
        raise ValueError(
            f"{plan.prefix}: this unit cannot be cut {plan.tp_size} ways on the {plan.axis} "
            f"axis ({extent} {plan.axis}s, granularity {gran}).  A column cut of a unit with a "
            "RELEASE plane or a mixed rate schedule is confined to whole 256-column "
            "superblocks; serve with a tensor_parallel_size that divides it, or with 1.")
    if plan.axis == AXIS_ROWS and plan.shard_rows % row_gran:
        raise ValueError(
            f"{plan.prefix}: {plan.rows} rows over {plan.tp_size} ranks is {plan.shard_rows} per "
            f"rank, not a multiple of this unit's row granularity {row_gran}")
    if plan.axis == AXIS_COLUMNS and plan.shard_columns % col_gran:
        raise ValueError(
            f"{plan.prefix}: {plan.columns} columns over {plan.tp_size} ranks is "
            f"{plan.shard_columns} per rank, not a multiple of this unit's column granularity "
            f"{col_gran}")


def shard_parsed_roles(parsed_roles, plan: ShardPlan):
    """``[(name, parsed)]`` for the whole module -> this rank's roles.

    Fused roles are cut INDEPENDENTLY on the row axis, which is what vLLM does
    with q/k/v and gate/up: rank *r* holds its own slice of each.  On the column
    axis there is one role and the whole of it is cut.
    """
    if plan.is_whole or plan.axis is None:
        return parsed_roles
    out = []
    for name, parsed in parsed_roles:
        check_shard_granularity(plan, parsed.unit)
        sharded = _shard_unit_for_rank(parsed.unit, plan.tp_rank, plan.tp_size, plan.axis)
        if sharded is parsed.unit:
            out.append((name, parsed))
            continue
        raise NotImplementedError(                       # pragma: no cover - until slice_unit lands
            f"{plan.prefix}: the unit slicer returned a new unit; re-parsing a sliced unit into a "
            "role view is the other half of that branch and is not in this build")
    return out
