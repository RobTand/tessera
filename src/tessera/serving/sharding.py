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

TODAY.  ``tp_size == 1`` is what this plugin has SERVED, and at ``tp_size == 1``
the seam returns the whole unit unchanged -- the same object, so the caller's
parsed view of it stays valid and the served arithmetic is bit-identical to a
build with no TP support at all.  At ``tp_size > 1`` the seam CUTS: it asks
``tessera.layout.can_shard`` first (refusing with the granularity the operator
can act on), calls ``tessera.layout.slice_unit``, and re-derives the parsed view
by writing the shard and reading it back -- a shard is a whole artifact, so the
route downstream sees a unit whose manifest describes the rows it actually
holds.  It still never falls back to "every rank holds the whole weight": that
would serve correct logits at N times the intended memory and look merely
disappointing.

TWO GATES, AND THEY ANSWER DIFFERENT QUESTIONS.  ``require_a_cutter`` refuses a
TP group in a build with no ``tessera.layout.slice_unit`` -- the whole-file
answer, asked once per module at method construction.  ``require_axis_supported``
refuses the ONE axis a route's decoders cannot start, read off ``ROUTE_TP_AXES``
and asked at ``create_weights`` where ``plan_shard`` has just named the axis.  A
ROW shard (``r0 > 0``) carries an INITIAL_STATE plane, and only the window body
threads a start state through its pad (``lane_planes.pack_window_planes``) --
which the E4M3 and BF16 families ship and the span-2 TCQ body does not, so the
span-2 packer refuses such a unit by name (``pack_unit_for_kernel``).  So the NVFP4 route serves column cuts
(RowParallel) at any TP and refuses row cuts -- on every rank, including rank 0,
whose shard would in fact pack, because a group whose ranks disagree about
whether a module exists hangs on its first collective rather than failing.

NONE OF THIS IS ATTESTED.  ``runtime_contract.json``'s
``tensor_parallel.units[].max_world_size`` is still 1 and stays 1 until a
multi-rank serve has been run: what is published above it is
``loader_axes``, which is ``ROUTE_TP_AXES`` -- a statement about what this
build's loader DOES, checked against this module so the two cannot drift.  A
two-rank serve is therefore something this build will attempt and nothing has
measured, and that distinction is machine-readable rather than a footnote.

WHICH AXIS.  vLLM's parallel Linears split one of the two axes and tell the
method by the sizes they ask for, so the axis is DERIVED from the shapes
rather than sniffed off a class name:

* ``ColumnParallelLinear`` / ``MergedColumnParallelLinear`` / ``QKVParallelLinear``
  split the OUTPUT: ``sum(output_partition_sizes) * tp == rows``.  Each fused
  role is split independently -- q, k and v each give every rank its own rows --
  which is why the seam is applied per role, in stacking order.
* ``RowParallelLinear`` splits the INPUT: ``input_size_per_partition * tp ==
  columns``.  One role, cut along columns.

MoE: expert parallelism assigns WHOLE units to ranks (no cut at all, the
granularity is one expert), and tensor parallelism inside an expert is the same
row/column cut as here -- ``w13`` on rows, ``w2`` on columns -- so a routed
expert reaches this module through the same two functions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .scheme import TESSERA_BF16, TESSERA_FP8, TESSERA_NVFP4

__all__ = [
    "AXIS_ROWS",
    "AXIS_COLUMNS",
    "AXES",
    "TP_SHARDED",
    "TP_REFUSED",
    "TP_STATUSES",
    "ROUTE_TP_AXES",
    "axis_status",
    "have_unit_slicer",
    "require_a_cutter",
    "require_axis_supported",
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
#: Both, in the order the contract's ``loader_axes`` writes them.
AXES = (AXIS_ROWS, AXIS_COLUMNS)

#: The loader cuts this axis and the route serves the shard.
TP_SHARDED = "sharded"
#: The loader refuses this axis by name, on every rank.
TP_REFUSED = "refused"
TP_STATUSES = (TP_SHARDED, TP_REFUSED)

#: WHAT THE LOADER DOES WITH EACH AXIS, PER ROUTE.  This table is the one
#: source: the routes gate on it at ``create_weights`` and
#: ``runtime_contract.json``'s ``tensor_parallel.units[].loader_axes`` is
#: checked against it (``contract.validate_serving_contract``), so the document
#: and the code cannot drift the way two spec files once did about one runtime.
#:
#: It is a machine-readable STATUS, and it is a statement about this build's
#: DECODERS, not an attestation: ``max_world_size`` in the same block is the
#: attestation, and it is still 1 because no multi-rank serve has been run.
#: A producer that wants "will this even load" reads this; a producer that
#: wants "has this been measured" reads ``max_world_size``.
#:
#: The row axis's answer is a property of the BODY, not of the tile: the window
#: body's L-bit pad IS ``state_{-1}``, so a row shard costs the window decoders
#: an argument, while the span-2 TCQ body's ``SELECT_PAD`` feeds a window whose
#: bit order ``build_span2_luts`` reverses and threading the state through that
#: reversal is unwritten.  It is keyed by ROUTE because family, body and route
#: are one-to-one today (``export_tessera_serving.check_recipe`` enforces it);
#: a third family with a different body brings its own row.
ROUTE_TP_AXES: dict[str, dict[str, str]] = {
    TESSERA_NVFP4: {AXIS_ROWS: TP_REFUSED, AXIS_COLUMNS: TP_SHARDED},
    TESSERA_FP8: {AXIS_ROWS: TP_SHARDED, AXIS_COLUMNS: TP_SHARDED},
    # The third family, and the docstring above said what to do with one: it
    # brings its own row.  BF16 gets FP8's answer for FP8's reason and not by
    # analogy -- the axis answer is a property of the BODY, both routes ship
    # the window body over the CHANNEL plane, and both threaded the start
    # state the same way before this table existed (``bf16_route`` and
    # ``fp8_route`` each pass ``initial_state=getattr(unit, "initial_state",
    # None)`` into the same decoder).  What differs between them is the tile
    # the decode lands in, which the shard never touches.
    TESSERA_BF16: {AXIS_ROWS: TP_SHARDED, AXIS_COLUMNS: TP_SHARDED},
}

#: Why a refused axis is refused.  PROSE, and deliberately not a gate input
#: (principle 14): ``ROUTE_TP_AXES`` is what a gate reads, this is what a
#: person reads.  The contract may carry its own wording.
ROUTE_TP_AXIS_REASONS: dict[str, dict[str, str]] = {
    TESSERA_NVFP4: {
        AXIS_ROWS: (
            "a row shard begins mid-column, so it carries an INITIAL_STATE plane, and the "
            "span-2 TCQ decoders this route packs for (tessera.lane_planes."
            "pack_unit_for_kernel) supply state_{-1} = 0 themselves"),
    },
}


def axis_status(family: str, axis: str) -> str:
    """``TP_SHARDED`` or ``TP_REFUSED`` for one (route, axis).

    Raises on an unknown route or axis rather than defaulting: a family this
    table has never heard of is a route this module cannot answer for, and
    guessing ``sharded`` there is exactly the silent-wrong-rows outcome the
    whole seam exists to prevent.
    """
    axes = ROUTE_TP_AXES.get(str(family))
    if axes is None:
        raise ValueError(
            f"no tensor-parallel answer is published for the route {family!r}; "
            f"tessera.serving.sharding.ROUTE_TP_AXES covers {sorted(ROUTE_TP_AXES)}")
    if axis not in axes:
        raise ValueError(f"{axis!r} is not a shard axis; the axes are {list(AXES)}")
    return axes[axis]


def have_unit_slicer() -> bool:
    """Is ``tessera.layout.slice_unit`` in this build?

    The TP gate is keyed on the cutter's PRESENCE rather than deleted outright:
    a build without it must still refuse rather than serve rank 0's whole unit
    to every rank, which is correct logits at N times the intended memory and
    looks merely disappointing.
    """
    try:
        from tessera.layout import slice_unit  # noqa: F401
    except Exception:  # noqa: BLE001 -- an older Tessera, or a partial install
        return False
    return True


def require_a_cutter(prefix: str, world: int) -> None:
    """Refuse a TP group in a build that cannot cut a unit.  ``world<=1`` passes."""
    if int(world) <= 1 or have_unit_slicer():
        return
    raise ValueError(
        f"tessera target {prefix!r}: this build cannot serve tensor_parallel_size={int(world)}. A "
        "unit's rows are bit-packed against a shared rate schedule and its trellis carries state "
        "along a row, so a shard is not a byte range -- the loader cuts the unit itself, through "
        "tessera.layout.slice_unit, which is not in this build. The checkpoint is not the problem "
        "and does not need re-exporting -- one artifact serves any world size, because the cut "
        "happens at load. Serve with -tp 1, or install a Tessera carrying the unit slicer.")


def require_axis_supported(family: str, plan: "ShardPlan") -> None:
    """Refuse, at ``create_weights``, an axis this route's decoders cannot start.

    Called with the plan and NOT with the rank, on purpose.  A row cut of a
    span-2 unit is packable on rank 0 -- its shard starts at row 0, so it
    carries no INITIAL_STATE plane -- and refusing only where it bites would
    mean rank 0 building a layer while its peers raised.  A TP group whose
    ranks disagree about whether a module exists does not fail: it hangs on the
    first collective, which is a worse bug than the one being reported.  So the
    refusal is symmetric across the group, and it arrives before any byte is
    loaded rather than from inside a packer.
    """
    if plan.axis is None or plan.tp_size <= 1:
        return                       # nothing is cut: replicated, or one rank
    if axis_status(family, plan.axis) == TP_SHARDED:
        return
    reason = ROUTE_TP_AXIS_REASONS.get(family, {}).get(plan.axis, "this route does not cut it")
    served = [axis for axis in AXES if axis_status(family, axis) == TP_SHARDED]
    layer = ("column-parallel (ColumnParallelLinear, MergedColumnParallelLinear, "
             "QKVParallelLinear)" if plan.axis == AXIS_ROWS else
             "row-parallel (RowParallelLinear: o_proj, down_proj)")
    raise ValueError(
        f"tessera target {plan.prefix!r}: {family} does not serve a {plan.axis} cut, and this is a "
        f"{layer} module at tensor_parallel_size={plan.tp_size}. It is refused on every rank, "
        f"including rank 0 whose shard would pack, because a group whose ranks disagree about a "
        f"module hangs on its first collective instead of failing. The reason is the body, not the "
        f"tile: {reason}. This route serves {served} cuts. Serve this model with "
        f"tensor_parallel_size=1, or export this Linear to a Tessera route that cuts the "
        f"{plan.axis} axis (TESSERA_FP8, the E4M3 window wire) -- the checkpoint does not need "
        "re-exporting for the TP degree itself, only for the route.")


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


def _unit_extent(unit) -> Tuple[int, int]:
    """``(rows, columns)`` in WEIGHT space, from whatever the caller holds.

    A ``ParsedUnit`` says so in its manifest geometry, which is the only place
    the arity is recorded: a k-tuple grid packs ``arity`` weight rows per
    trellis step, so ``body_bits.shape[0]`` is a STEP count and reading it as a
    row count would halve the extent and cut every rank's shard in the wrong
    place.  A bare ``EncodedUnit`` carries no arity, and arity 1 is then the
    same assumption ``layout.slice_unit`` itself makes for one.
    """
    from tessera.unit_artifact import ParsedUnit

    if isinstance(unit, ParsedUnit):
        geometry = unit.manifest.geometry
        return int(geometry.rows), int(geometry.columns)
    steps, columns = unit.body_bits.shape
    return int(steps), int(columns)


def _shard_unit_for_rank(unit, tp_rank: int, tp_size: int, axis: Optional[str]):
    """THE SEAM.  The unit this rank serves, cut from the whole one.

    ``tp_size == 1`` returns the SAME OBJECT -- identity, not a copy -- so a
    caller holding a parsed view of it may keep that view, and so a TP-capable
    build serves byte-identical bytes to one without this function.

    Above one rank this asks ``tessera.layout.can_shard`` and then calls
    ``tessera.layout.slice_unit``, which returns a STANDALONE unit: it decodes
    through the same ``tessera.decode`` entry points to exactly
    ``decode(parent)[r0:r1, c0:c1]``, bit for bit, with no re-encoding.  The
    caller must re-derive its parsed view of what comes back -- a shard's planes
    are its own, and a stale parse describes the parent's.

    ``can_shard`` is asked rather than inferred, and it is asked BEFORE the cut
    so the refusal can name the granularity: ``slice_unit`` would refuse too,
    but with an offset the operator cannot map back to a ``tensor_parallel_size``.

    Only this rank's shard is cut, so the cost is O(1) in the TP degree.
    """
    if tp_size == 1:
        return unit
    if axis is None:
        raise ValueError(
            f"rank {tp_rank} of {tp_size} asked for a cut with no axis; plan_shard reports "
            "axis=None only for a module served whole (replicated), which is not cut")
    try:
        from tessera.layout import slice_unit
    except Exception as exc:
        raise NotImplementedError(
            f"tensor parallelism needs tessera.layout.slice_unit, which is not in this build "
            f"({exc}): rank {tp_rank} of {tp_size} asked for a {axis} slice of a unit this "
            "plugin can only serve whole.  Serve with tensor_parallel_size=1, or install a "
            "Tessera carrying the unit slicer.") from exc
    rows, columns = _unit_extent(unit)
    if can_shard(unit, tp_size, axis) is not True:
        granularity = shard_granularity(unit)
        row_gran, col_gran = granularity if granularity is not None else (None, None)
        extent = rows if axis == AXIS_ROWS else columns
        gran = row_gran if axis == AXIS_ROWS else col_gran
        raise ValueError(
            f"this unit cannot be cut {tp_size} ways on the {axis} axis ({extent} {axis}s, "
            f"granularity {gran}).  A row cut lands on a trellis super-symbol and a column cut "
            "of a unit carrying a RELEASE plane or a mixed rate schedule is confined to whole "
            "256-column superblocks; serve with a tensor_parallel_size that divides it, or "
            "with 1.")
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


def _reparse_shard(parsed, sharded, label: str):
    """A shard is a whole artifact: write it, and read it back.

    The cheap alternative -- swapping the sliced unit into the parent's
    ``ParsedUnit`` -- would leave a manifest describing the PARENT: the wrong
    rows, the wrong columns, no shard record, and a parent digest naming a unit
    this rank does not hold.  Everything downstream that asks a parse for a
    shape would then get the whole module's.  Serialising is the same round trip
    ``tests/test_slice_unit.py::test_shard_round_trips_through_bytes`` proves
    exact, and it costs one write and one parse per role, once, at load.
    """
    from tessera.trellis import ConvCode
    from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact

    manifest = parsed.manifest
    _m, _region, blob = build_unit_artifact(
        sharded, label, parsed.forests, int(manifest.branch.root_q256),
        parsed.code or ConvCode(),
        superblock=int(manifest.geometry.superblock_columns),
        container=manifest.branch.container)
    return parse_unit_artifact(blob, device=parsed.unit.body_bits.device)


def shard_parsed_roles(parsed_roles, plan: ShardPlan):
    """``[(name, parsed)]`` for the whole module -> this rank's roles.

    Fused roles are cut INDEPENDENTLY on the row axis, which is what vLLM does
    with q/k/v and gate/up: rank *r* holds its own slice of each.  On the column
    axis there is one role and the whole of it is cut.

    What comes back is a list of ``ParsedUnit``s again -- re-derived from the
    shard's own bytes -- because that is what every route's ``prepare_*_module``
    consume, and because a shard's planes are its own.
    """
    if plan.is_whole or plan.axis is None:
        return parsed_roles
    out = []
    for name, parsed in parsed_roles:
        # Asked with the PARSE, not with ``parsed.unit``: the superblock and the
        # arity live on the parse, and the defaults (256, 1) are wrong for a
        # k-tuple grid -- so a bare unit would be measured against the wrong
        # granularity and a legal cut refused (or an illegal one allowed).
        check_shard_granularity(plan, parsed)
        sharded = _shard_unit_for_rank(parsed, plan.tp_rank, plan.tp_size, plan.axis)
        if sharded is parsed:                       # pragma: no cover - tp_size == 1 is is_whole
            out.append((name, parsed))
            continue
        out.append((name, _reparse_shard(
            parsed, sharded, f"{plan.prefix}.{name}.rank{plan.tp_rank}of{plan.tp_size}")))
    return out
