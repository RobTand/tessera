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

WHICH AXIS, AND WHICH ROWS.  vLLM's parallel Linears split one of the two axes
and tell the method by the sizes they ask for, so the axis is DERIVED from the
shapes rather than sniffed off a class name:

* ``ColumnParallelLinear`` / ``MergedColumnParallelLinear`` / ``QKVParallelLinear``
  split the OUTPUT.  Each fused role is split INDEPENDENTLY -- q, k and v each
  give every rank its own rows -- and ``output_partition_sizes`` is that
  per-role answer, one entry per member, in the container's stacking order.
  So the question "which rows does this rank hold" is asked ONCE PER ROLE,
  against the role's own rows, and never of the container as a whole.
* ``RowParallelLinear`` splits the INPUT: ``input_size_per_partition * tp ==
  columns``.  One role, cut along columns, and there the split is even.

A ROLE IS NOT ALWAYS CUT ``tp`` WAYS.  Under grouped-query attention with
``num_kv_heads < tp`` vLLM REPLICATES the KV heads: every rank gets a whole
head, so ``num_kv_heads`` distinct shards are shared by ``tp`` ranks and two
ranks legitimately hold the SAME k/v rows.  The number of distinct shards is
therefore a property of the role and not of the world::

    shards_i   = role_rows_i // out_partition_i      # exact, or the shapes disagree
    replicas_i = tp_size // shards_i                 # ranks per shard, 1 when even
    index_i    = tp_rank // replicas_i               # which shard THIS rank holds
    rows_i     = (index_i * out_partition_i, (index_i + 1) * out_partition_i)

That is vLLM's own arithmetic, read back off the two numbers it hands the
loader rather than re-derived from head geometry it never passes: ``linear.py``
``QKVParallelLinear.weight_loader`` computes ``shard_rank = self.tp_rank //
self.num_kv_head_replicas`` and ``start_idx = shard_rank * shard_size``, and
``num_kv_head_replicas = tp_size // total_num_kv_heads`` is exactly
``tp_size // shards_i`` when ``out_partition_i`` is one head.  No branch on a
member's NAME, no ``num_heads``/``head_size`` in this module's signature: the
sizes carry the answer, and summing them away was the whole defect (#32).

``shards_i`` is also what ``tessera.layout.can_shard`` must be asked -- it
answers "into how many EQUAL shards", and asking it ``tp_size`` for a
replicated role would test a finer cut than the one being made.

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
    "RoleShard",
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
class RoleShard:
    """Which part of ONE role this rank holds, on the plan's axis.

    ``lo``/``hi`` are half-open and are the role's OWN coordinates -- a fused
    container is framing and each member is its own unit (``fused.py``), so the
    range that gets cut is the member's, not an offset into the stack.

    ``shards`` is how many DISTINCT ranges the role's extent is divided into.
    It equals ``tp_size`` for an ordinary split and is SMALLER when vLLM
    replicates the role across ranks -- ``tp_size // shards`` ranks then hold
    each range, and two of them holding identical rows is the intended
    arrangement, not a collision.  It is the number ``layout.can_shard`` has to
    be asked, because that function answers "into how many EQUAL shards".
    """

    name: str
    extent: int
    lo: int
    hi: int
    shards: int

    @property
    def is_whole(self) -> bool:
        """This rank holds the role entire -- replicated, or not cut at all."""
        return self.lo == 0 and self.hi == self.extent


@dataclass(frozen=True)
class ShardPlan:
    """What one rank serves of one module.

    ``axis`` is None exactly when there is nothing to cut (``tp_size == 1``, or
    a replicated Linear inside a group), and then ``shard_rows``/
    ``shard_columns`` are the module's own.

    ``shard_rows`` is ``sum(output_partition_sizes)`` -- the height of the tile
    THIS RANK computes.  Under KV replication that is not ``rows // tp_size``,
    and the routes hand it to ``scale_b.view(1, N)``, so it is carried rather
    than re-derived.

    ``roles`` is the per-member answer, in the container's stacking order: one
    ``RoleShard`` each, carrying the range to cut and the shard count to ask
    ``can_shard`` with.  It is the whole reason this is a plan and not a pair
    of integers -- q, k and v are cut differently under GQA.
    """

    prefix: str
    rows: int
    columns: int
    shard_rows: int
    shard_columns: int
    tp_rank: int
    tp_size: int
    axis: Optional[str] = None
    roles: Tuple[RoleShard, ...] = ()

    @property
    def is_whole(self) -> bool:
        return self.tp_size == 1 and self.axis is None

    def role(self, name: str) -> RoleShard:
        """The plan for one member, BY NAME.

        By name and not by position: ``parse_tessera_blob_for_scheme`` has
        already refused a container whose members differ from the declared
        roles, so the names are the reliable key, and a positional lookup here
        would be a second ordering to keep in step.
        """
        for role in self.roles:
            if role.name == name:
                return role
        raise ValueError(
            f"{self.prefix}: no shard plan for the role {name!r}; the plan covers "
            f"{[r.name for r in self.roles]}, which is what the scheme declared")


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


def can_shard(unit, shards: int, axis: str) -> Optional[bool]:
    """``layout.can_shard``, or None when the cutter is absent.

    Asked BEFORE a cut rather than inferred from the granularity, because the
    granularity is necessary and not sufficient: a column cut of a unit that
    carries a RELEASE plane is confined to whole 256-column superblocks, and
    ``can_shard`` is where that lives.  None means "cannot answer", which
    callers treat as "refuse", never as "yes".

    The argument is ``shards`` and not ``tp_size``: ``layout.can_shard``
    answers "into how many EQUAL shards", and a role vLLM replicates across the
    group is cut into fewer of them than there are ranks (``RoleShard.shards``).
    Passing the world size there would test a finer cut than the one being made
    and refuse legal ones.
    """
    try:
        from tessera.layout import can_shard as _can_shard
    except Exception:
        return None
    return bool(_can_shard(unit, int(shards), axis))


def plan_shard(prefix: str, *, roles, columns: int, out_partitions, in_size: int,
               tp_rank: Optional[int] = None, tp_size: Optional[int] = None) -> ShardPlan:
    """Decide, from the sizes vLLM asks for, which slice of each role this rank is.

    ``roles`` is the checkpoint's ``[(name, rows)]`` in stacking order --
    ``validate_tessera_scheme``'s own field -- and ``out_partitions`` is vLLM's
    ``output_partition_sizes``, the per-member output size THIS RANK computes.
    Both are LISTS on purpose: a fused container's members are cut
    independently, and summing them to two scalars is what made a
    replicated-KV QKV module unplannable (#32).  ``columns`` and ``in_size``
    stay scalars because the input axis is one width, evenly split.

    THE TWO LISTS ARE PAIRED BY POSITION, and that is a dependency worth
    stating: vLLM passes no names with ``output_partition_sizes``, so the
    correspondence rests on the container's stacking order being the layer's.
    It is not assumed blind -- ``parse_tessera_blob_for_scheme`` refuses a
    container whose members are not exactly the declared roles in order, that
    order is the row order the exporter packed (``fused.pack_fused``), and it
    is the order the SERVED tp=1 tile is stacked in, so a mis-ordering would
    already be wrong at one rank.  The per-role checks below are the only
    cross-check available here; they catch a mis-ordering whenever the roles
    differ in size, and cannot when they do not.

    THE TOTALS ARE NEVER THE ANSWER ON THEIR OWN.  Two lists that sum alike
    can name different boundaries -- ``[4, 8, 4]`` and ``[8, 4, 4]`` are both
    16 rows -- and a plan built on the sums would hand back the whole module
    while the layer read one role's rows as another's.  So the lengths and,
    wherever the output is whole, the per-member extents are compared BEFORE
    any branch: the whole-module shortcut is reached only by a request that
    agrees with the checkpoint member by member.

    Refuses -- with the module's shape, the role and the relation that failed --
    when the request is neither the whole module nor a split this wire can
    express.  A checkpoint and a serve disagreeing about a module's geometry is
    the failure worth being loud about; head replication is NOT that failure,
    and does not get that message.
    """
    if tp_rank is None or tp_size is None:
        derived_rank, derived_size = tp_rank_and_size()
        tp_rank = derived_rank if tp_rank is None else tp_rank
        tp_size = derived_size if tp_size is None else tp_size
    roles = tuple((str(name), int(rows)) for name, rows in roles)
    out_partitions = tuple(int(size) for size in out_partitions)
    if not roles:
        raise ValueError(f"{prefix}: a module has at least one role; the scheme declared none")
    columns, in_size = int(columns), int(in_size)
    tp_rank, tp_size = int(tp_rank), int(tp_size)
    if tp_size < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError(f"{prefix}: nonsensical TP coordinates rank {tp_rank} of {tp_size}")
    rows = sum(role_rows for _name, role_rows in roles)
    out_size = sum(out_partitions)
    shape = (f"{prefix}: tessera module is {rows}x{columns}; rank {tp_rank} of {tp_size} wants "
             f"{out_size}x{in_size}")

    # THE TWO LISTS ARE ONE LIST, and that is asked ONCE, before any branch.
    # It used to be asked only where the output is cut, so the two ways a
    # module can be served WHOLE -- one rank, and a replicated Linear inside a
    # group -- reached their shortcut on the totals alone.
    if len(out_partitions) != len(roles):
        raise ValueError(
            f"{shape}. The checkpoint declares {len(roles)} roles "
            f"{[name for name, _ in roles]} but the layer asks for {len(out_partitions)} output "
            f"partitions {list(out_partitions)}; a fused container's members and a layer's "
            f"output partitions are the same list in the same order, so the two disagree about "
            f"how this module is fused -- which is a checkpoint/serve mismatch, not a shard.")
    if out_size == rows:
        # EQUAL TOTALS ARE NOT EQUAL BOUNDARIES.  A whole output is every
        # role's rows, member by member: a container that stacks q/k/v as
        # [4, 8, 4] and a layer that reads [8, 4, 4] agree on 16 and on
        # nothing else, and the load succeeds serving four q rows and the
        # first four k rows as this layer's q.  Nothing downstream sees it --
        # the decoder returns the role arrangement it was handed, and the
        # sidecar agrees with the wire it was written beside -- so the
        # boundaries are compared HERE, where both lists are in hand.
        disagree = [(name, role_rows, out)
                    for (name, role_rows), out in zip(roles, out_partitions)
                    if out != role_rows]
        if disagree:
            name, role_rows, out = disagree[0]
            raise ValueError(
                f"{shape}, which is every output row of the module -- but not at the "
                f"checkpoint's boundaries. Its role {name!r} is {role_rows} rows on the wire and "
                f"this rank asks for {out} of them; the declared roles stack as "
                f"{[(n, r) for n, r in roles]} and the layer's output partitions are "
                f"{list(out_partitions)}. Equal totals are not equal boundaries: a container "
                f"decoded at the wire's boundaries and consumed at the layer's serves one role's "
                f"rows as another's, which no decoder can detect. The checkpoint and the serve "
                f"disagree about how this module is fused.")

    if out_size == rows and in_size == columns:
        # Whole module.  True at tp_size == 1, and also on a replicated Linear
        # inside a TP group -- which is a whole unit per rank, not a cut.
        return ShardPlan(prefix, rows, columns, rows, columns, tp_rank, tp_size, None,
                         tuple(RoleShard(name, role_rows, 0, role_rows, 1)
                               for name, role_rows in roles))
    if tp_size == 1:
        raise ValueError(
            f"{prefix}: tessera module is {rows}x{columns} but the layer wants "
            f"{out_size}x{in_size} on one rank; a wire is bound to its shape")
    if out_size == rows:
        # The output is whole, so the cut is on the input: RowParallelLinear.
        # One width over the ranks, evenly -- there is no per-role structure on
        # this axis, because every role reads every input feature.
        if in_size * tp_size != columns:
            raise ValueError(
                f"{shape}, which holds every output row but is not a 1/{tp_size} split of its "
                f"{columns} input columns; a row-parallel Linear gives each rank the same "
                f"{columns}/{tp_size} of them, and every role reads every input feature so "
                f"there is no per-member range on this axis.")
        c0 = tp_rank * in_size
        return ShardPlan(prefix, rows, columns, rows, in_size, tp_rank, tp_size, AXIS_COLUMNS,
                         tuple(RoleShard(name, columns, c0, c0 + in_size, tp_size)
                               for name, _role_rows in roles))
    if in_size != columns:
        raise ValueError(
            f"{shape}, which cuts BOTH axes. A vLLM parallel Linear splits exactly one: the "
            f"output (ColumnParallel and its merged/QKV forms) or the input (RowParallel).")
    placed = []
    for (name, role_rows), out in zip(roles, out_partitions):
        if out <= 0 or role_rows % out:
            raise ValueError(
                f"{shape}. Its role {name!r} is {role_rows} rows and this rank asks for {out} of "
                f"them, which does not divide {role_rows}: vLLM gives every rank the SAME size "
                f"for a member, so a member's rows are a whole number of equal shards. The "
                f"checkpoint and the serve disagree about this module's geometry.")
        shards = role_rows // out
        if shards > tp_size:
            raise ValueError(
                f"{shape}. Its role {name!r} is {role_rows} rows in shards of {out}, which is "
                f"{shards} shards for only {tp_size} ranks: this rank asks for less than "
                f"1/{tp_size} of the role, so some of its rows would be served by nobody. The "
                f"checkpoint and the serve disagree about this module's geometry.")
        if tp_size % shards:
            raise ValueError(
                f"{shape}. Its role {name!r} is {role_rows} rows in shards of {out}, which is "
                f"{shards} shards for {tp_size} ranks -- and {shards} does not divide {tp_size}, "
                f"so there is no whole number of ranks per shard. A role held by fewer shards "
                f"than there are ranks is REPLICATED (this is how vLLM serves KV heads under "
                f"GQA), and replication needs {tp_size}/{shards} to be an integer.")
        index = tp_rank // (tp_size // shards)
        placed.append(RoleShard(name, role_rows, index * out, index * out + out, shards))
    return ShardPlan(prefix, rows, columns, out_size, columns, tp_rank, tp_size, AXIS_ROWS,
                     tuple(placed))


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


def _shard_unit_for_rank(unit, plan: ShardPlan, role: RoleShard):
    """THE SEAM.  The unit this rank serves of ONE role, cut from the whole one.

    The range is the PLAN's, not this function's: ``plan_shard`` read it off
    the sizes vLLM asked for, and a seam that re-derived it by splitting the
    extent ``tp_size`` ways would be the even-split assumption again -- correct
    for q, wrong for a replicated k (#32).  So there is no arithmetic here, only
    a cut.

    Returns the SAME OBJECT -- identity, not a copy -- when this rank holds the
    role entire: ``tp_size == 1``, a replicated Linear, or a role vLLM
    replicates across the group.  A caller holding a parsed view of it may then
    keep that view, and a TP-capable build serves byte-identical bytes to one
    without this function.

    Above that this asks ``tessera.layout.can_shard`` and then calls
    ``tessera.layout.slice_unit``, which returns a STANDALONE unit: it decodes
    through the same ``tessera.decode`` entry points to exactly
    ``decode(parent)[r0:r1, c0:c1]``, bit for bit, with no re-encoding.  The
    caller must re-derive its parsed view of what comes back -- a shard's planes
    are its own, and a stale parse describes the parent's.

    ``can_shard`` is asked with ``role.shards`` and not with ``plan.tp_size``:
    it answers "into how many EQUAL shards", and a replicated role is cut into
    fewer of them than there are ranks.  It is asked rather than inferred, and
    BEFORE the cut so the refusal can name the granularity: ``slice_unit``
    would refuse too, but with an offset the operator cannot map back to a
    ``tensor_parallel_size``.

    Only this rank's shard is cut, so the cost is O(1) in the TP degree.
    """
    if plan.tp_size == 1:
        return unit
    if plan.axis is None:
        raise ValueError(
            f"rank {plan.tp_rank} of {plan.tp_size} asked for a cut with no axis; plan_shard "
            "reports axis=None only for a module served whole (replicated), which is not cut")
    if role.is_whole:
        return unit
    try:
        from tessera.layout import slice_unit
    except Exception as exc:
        raise NotImplementedError(
            f"tensor parallelism needs tessera.layout.slice_unit, which is not in this build "
            f"({exc}): rank {plan.tp_rank} of {plan.tp_size} asked for a {plan.axis} slice of a "
            "unit this plugin can only serve whole.  Serve with tensor_parallel_size=1, or "
            "install a Tessera carrying the unit slicer.") from exc
    rows, columns = _unit_extent(unit)
    extent = rows if plan.axis == AXIS_ROWS else columns
    if extent != role.extent:
        raise ValueError(
            f"{plan.prefix}: role {role.name!r} is {extent} {plan.axis}s on the wire but the plan "
            f"was made for {role.extent}; the parsed container and the declared scheme disagree")
    if can_shard(unit, role.shards, plan.axis) is not True:
        granularity = shard_granularity(unit)
        row_gran, col_gran = granularity if granularity is not None else (None, None)
        gran = row_gran if plan.axis == AXIS_ROWS else col_gran
        raise ValueError(
            f"this unit cannot be cut {role.shards} ways on the {plan.axis} axis ({extent} "
            f"{plan.axis}s, granularity {gran}).  A row cut lands on a trellis super-symbol and a "
            "column cut of a unit carrying a RELEASE plane or a mixed rate schedule is confined "
            "to whole 256-column superblocks; serve with a tensor_parallel_size that divides it, "
            "or with 1.")
    if plan.axis == AXIS_ROWS:
        return slice_unit(unit, rows=(role.lo, role.hi), cols=(0, columns))
    return slice_unit(unit, rows=(0, rows), cols=(role.lo, role.hi))


def check_shard_granularity(plan: ShardPlan, role: RoleShard, unit) -> None:
    """Refuse a split the wire cannot express, naming the granularity.

    A unit is not a matrix of independent rows: the body packs along both axes
    and the trellis carries state along a row, so a cut is exact only on the
    packing's own boundaries.  ``tessera.layout.shard_granularity`` is the
    authority on those; when it is absent the caller has already refused at the
    seam, so this is a no-op rather than a guess.

    Asked PER ROLE, against the role's own range.  The container's total is the
    wrong number: q, k and v are cut independently and to different widths
    under GQA, so a module-level ``shard_rows`` would measure a width no role
    actually holds.  Both ends of the range are checked -- a legal width
    starting on an illegal offset is the replicated case's own failure mode.

    ``layout.can_shard`` is asked FIRST, with the role's shard count, and is the
    binding answer: granularity is necessary, not sufficient.  A column cut of a
    unit carrying a RELEASE plane is confined to whole 256-column superblocks,
    and only ``can_shard`` knows that.  The granularity is then used to say the
    useful thing in the refusal -- a number the operator can act on.
    """
    if plan.axis is None or role.is_whole:
        return
    granularity = shard_granularity(unit)
    if granularity is None:
        return
    row_gran, col_gran = granularity
    gran = row_gran if plan.axis == AXIS_ROWS else col_gran
    if can_shard(unit, role.shards, plan.axis) is False:
        raise ValueError(
            f"{plan.prefix}: role {role.name!r} cannot be cut {role.shards} ways on the "
            f"{plan.axis} axis ({role.extent} {plan.axis}s, granularity {gran}).  A column cut of "
            "a unit with a RELEASE plane or a mixed rate schedule is confined to whole "
            "256-column superblocks; serve with a tensor_parallel_size that divides it, or "
            "with 1.")
    width = role.hi - role.lo
    if width % gran or role.lo % gran:
        raise ValueError(
            f"{plan.prefix}: role {role.name!r} is {role.extent} {plan.axis}s over "
            f"{plan.tp_size} ranks, giving this rank [{role.lo}, {role.hi}) -- {width} "
            f"{plan.axis}s at offset {role.lo}, which is not a multiple of this unit's "
            f"{plan.axis} granularity {gran}")


def _reparse_shard(parsed, sharded, label: str):
    """A shard is a whole artifact: write it, and read it back.

    The cheap alternative -- swapping the sliced unit into the parent's
    ``ParsedUnit`` -- would leave a manifest describing the PARENT: the wrong
    rows, the wrong columns, no shard record, and a parent digest naming a unit
    this rank does not hold.  Everything downstream that asks a parse for a
    shape would then get the whole module's.  Serialising is the same round trip
    ``tests/test_slice_unit.py::test_shard_round_trips_through_bytes`` proves
    exact, and it costs one write and one parse per role, once, at load.

    THE PARENT'S ENCODER IS THE SHARD'S ENCODER.  ``fixture_id`` is passed
    EXPLICITLY, including when the parent carries none: cutting restricts
    planes the parent's encoder already produced and re-encodes nothing, so the
    identity a shard names must be the one those bytes were written under.
    ``build_unit_artifact``'s default asks ``encoder_identity`` for THIS
    build's digest, which made every shard of an older artifact claim an
    encoder that never saw it -- and made a load compute that digest, over a
    cold fixture set, to attest bytes it had no part in (tessera#236).  The
    same holds in the other direction: an untagged parent's ``None`` is
    forwarded rather than promoted to a current identity it never carried.
    """
    from tessera.trellis import ConvCode
    from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact

    manifest = parsed.manifest
    _m, _region, blob = build_unit_artifact(
        sharded, label, parsed.forests, int(manifest.branch.root_q256),
        parsed.code or ConvCode(),
        superblock=int(manifest.geometry.superblock_columns),
        container=manifest.branch.container,
        fixture_id=manifest.encoder_fixture_id)
    return parse_unit_artifact(blob, device=parsed.unit.body_bits.device)


def shard_parsed_roles(parsed_roles, plan: ShardPlan):
    """``[(name, parsed)]`` for the whole module -> this rank's roles.

    Fused roles are cut INDEPENDENTLY on the row axis, which is what vLLM does
    with q/k/v and gate/up: rank *r* holds its own slice of each -- and under
    GQA with ``num_kv_heads < tp`` its slice of k and v is one another rank
    holds too.  The plan already says which range that is, per role, so this
    function looks it up by name and cuts; it computes no offsets of its own.
    On the column axis there is one range and every role gets it.

    What comes back is a list of ``ParsedUnit``s again -- re-derived from the
    shard's own bytes -- because that is what every route's ``prepare_*_module``
    consume, and because a shard's planes are its own.  A role this rank holds
    entire is passed through as THE SAME OBJECT, so a replicated k costs no
    round trip.
    """
    if plan.is_whole or plan.axis is None:
        return parsed_roles
    out = []
    for name, parsed in parsed_roles:
        role = plan.role(name)
        # Asked with the PARSE, not with ``parsed.unit``: the superblock and the
        # arity live on the parse, and the defaults (256, 1) are wrong for a
        # k-tuple grid -- so a bare unit would be measured against the wrong
        # granularity and a legal cut refused (or an illegal one allowed).
        check_shard_granularity(plan, role, parsed)
        sharded = _shard_unit_for_rank(parsed, plan, role)
        if sharded is parsed:                       # this rank holds the role entire
            out.append((name, parsed))
            continue
        out.append((name, _reparse_shard(
            parsed, sharded, f"{plan.prefix}.{name}.rank{plan.tp_rank}of{plan.tp_size}")))
    return out
