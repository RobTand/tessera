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
and tell the method so at ``create_weights``, with the TILE this rank computes
(``input_size_per_partition``, ``output_partition_sizes``) AND the module's
GLOBAL shape (``input_size``, ``output_size``), and every ``LinearBase`` sets
its own ``tp_rank``/``tp_size`` before that call.  The axis is DERIVED from
those numbers rather than sniffed off a class name, and it is the global shape
against the tile, times the layer's world, that names it:

* ``ReplicatedLinear`` -- and any ``LinearBase`` at ``tp_size == 1`` -- asks for
  the WHOLE: ``output_size == sum(output_partition_sizes)`` and ``input_size ==
  input_size_per_partition``.  Nothing is cut; the wire must be the module.
* ``ColumnParallelLinear`` / ``MergedColumnParallelLinear`` / ``QKVParallelLinear``
  split the OUTPUT: ``output_size == sum(output_partition_sizes) * tp_size``
  and ``input_size == input_size_per_partition``.  Each fused role is split
  INDEPENDENTLY -- q, k and v each give every rank its own rows -- and
  ``output_partition_sizes`` is that per-role answer, one entry per member, in
  the container's stacking order.  So the question "which rows does this rank
  hold" is asked ONCE PER ROLE, against the role's own rows, and never of the
  container as a whole.
* ``RowParallelLinear`` splits the INPUT: ``input_size == input_size_per_partition
  * tp_size`` and ``output_size == sum(output_partition_sizes)``.  One width,
  cut along columns, and there the split is even.

A layer whose numbers satisfy none of the three has stated a geometry this
plugin does not serve, and is refused with both shapes.

THE TILE ALONE CANNOT SAY WHETHER A WIRE IS WHOLE.  A ``[64, 64]`` wire under a
TP=4 ``ColumnParallelLinear`` over ``[256, 64]`` agrees with that rank's
``[64, 64]`` tile on every local number, and so does a ``[64, 64]`` wire under a
``ReplicatedLinear`` over ``[64, 64]``.  The first is a quarter of a module --
rows ``[64, 256)`` on no rank -- and the second is the module.  A plan that read
only the tile handed both back as a replicated whole (tessera#303); the
layer's ``output_size`` is the fact that tells them apart, so the plan is fed
it and refuses the undersized wire by the role's name, on every rank.

A ROLE IS NOT ALWAYS CUT ``tp`` WAYS -- AND THE LAYER SAYS WHEN.  Under
grouped-query attention with ``num_kv_heads < tp`` vLLM REPLICATES the KV
heads: every rank gets a whole head, so ``num_kv_heads`` distinct shards are
shared by ``tp`` ranks and two ranks legitimately hold the SAME k/v rows.  That
replication is a STATEMENT of ``QKVParallelLinear`` -- ``num_kv_head_replicas
= tp_size // total_num_kv_heads`` in its ``__init__``, and its ``weight_loader``
reads ``shard_rank = self.tp_rank // self.num_kv_head_replicas`` -- and it is
read off the layer (``layer_replicas``), never inferred from the numbers.  It
cannot be: an MQA k under TP=8 (64 rows on the wire, 64 asked for) is
numerically the undersized ``[64, 1024]`` wire of a plain ``[512, 1024]``
ColumnParallel Linear, and only the declaration separates the whole head from
the tile.  Nor is the padded aggregate enough on its own -- ``QKVParallelLinear``
pads ``output_size`` to ``num_kv_heads * head * tp`` per KV member, so the
aggregate is compared with the tile times the world, and each member's
complete extent is then the layer's own arithmetic::

    replicas_i = 1, or QKVParallelLinear.num_kv_head_replicas for k and v
    shards_i   = tp_size // replicas_i               # distinct ranges of the role
    whole_i    = out_partition_i * shards_i          # == role_rows_i, or refused by name
    index_i    = tp_rank // replicas_i               # which range THIS rank holds
    rows_i     = (index_i * out_partition_i, (index_i + 1) * out_partition_i)

No branch on a member's NAME, no ``num_heads``/``head_size`` in this module's
signature: the layer's declared shape and replication carry the answer.
Summing the sizes away was the first defect here (#32); reading the tile
without the layer's global shape and contract was the second (#303).

``shards_i`` is also what ``tessera.layout.can_shard`` must be asked -- it
answers "into how many EQUAL shards", and asking it ``tp_size`` for a
replicated role would test a finer cut than the one being made.

THE TP COORDINATES ARE THE LAYER'S.  ``LinearBase.__init__`` sets ``tp_rank``/
``tp_size`` -- ``(0, 1)`` under ``disable_tp``, the caller's own pair for an
effective group (``DCPGroupColumnParallelLinear``), the process's otherwise --
and reconciles every parameter to them "which correctly accounts for
disable_tp" (its own comment).  This module reads exactly those two
attributes (``layer_tp_coordinates``) and nothing from ``vllm.distributed``: a
process-global answer is wrong for a one-rank layer inside a group, and the
dense methods are handed only ``LinearBase`` instances
(``TesseraConfig.get_quant_method``), so a layer without the pair is refused by
name rather than defaulted.

MoE: expert parallelism assigns WHOLE units to ranks (no cut at all, the
granularity is one expert), and tensor parallelism inside an expert is the same
row/column cut as here -- ``w13`` on rows, ``w2`` on columns -- so a routed
expert reaches this module through the same two functions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

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
    "artifact_tp_agnostic",
    "require_a_cuttable_artifact",
    "require_axis_supported",
    "RoleShard",
    "ShardPlan",
    "plan_shard",
    "plan_shard_for_layer",
    "layer_tp_coordinates",
    "layer_replicas",
    "can_shard",
    "shard_granularity",
    "unsliceable_reason",
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


def artifact_tp_agnostic(config: "Optional[dict]") -> Optional[bool]:
    """Does this artifact's own config say its bytes can be cut at load?

    ``True``/``False`` is the artifact's statement; ``None`` means it makes
    none, which is not the same answer and is never read as ``True``.

    TWO SPELLINGS, ONE RULE.  ``tp_agnostic`` is the declaration the exporter
    stamps (``export._write_config``) and is taken verbatim when present --
    the artifact's own word about its own bytes.  ``schema_minor`` is the fact
    it was derived from, and a config that records the minor without the
    declaration is answered from it through ``layout.tp_agnostic_at_minor``,
    the one home of the rule -- so an artifact written by a producer that
    stamps only the wire minor still resolves correctly instead of being
    refused for a spelling.  A config carrying neither gets ``None``: it
    predates the declaration, and an artifact that does not declare a property
    does not get to claim it (tessera#328).

    ``tp_size`` is deliberately NOT read.  Parts on disk carry ``tp_size: 1``
    from an exporter whose comment asserted the artifact was TP-*specific*;
    that constant said nothing about whether the bytes could be cut, and
    reading it as though it did would turn a stale stamp into a permission.
    """
    if not isinstance(config, Mapping):
        return None
    declared = config.get("tp_agnostic")
    if isinstance(declared, bool):
        return declared
    minor = config.get("schema_minor")
    if isinstance(minor, int) and not isinstance(minor, bool):
        try:
            from tessera.layout import tp_agnostic_at_minor
        except Exception:  # noqa: BLE001 -- an older Tessera, or a partial install
            return None
        return bool(tp_agnostic_at_minor(minor))
    return None


def require_a_cuttable_artifact(prefix: str, world: int, config: "Optional[dict]") -> None:
    """Refuse a TP group against BYTES that cannot be cut.  ``world<=1`` passes.

    The twin of :func:`require_a_cutter`, and the other half of the same
    question: that one asks whether this BUILD carries ``layout.slice_unit``,
    this one asks whether the ARTIFACT admits a cut at all.  Both are
    whole-file questions asked once at method construction; the per-axis one
    (``require_axis_supported``) waits for ``create_weights``, where the axis
    exists.

    FAIL CLOSED, AND SAY SO BY NAME.  A checkpoint written before the
    declaration existed carries neither ``tp_agnostic`` nor ``schema_minor``
    (it carries ``tp_size: 1``, which is not an answer -- see
    :func:`artifact_tp_agnostic`), so it is refused above one rank rather than
    served on the strength of a field nothing ever read.  That is the "a
    loader cannot quietly use it at the wrong degree" the old exporter comment
    promised and never delivered (tessera#328); the remedy is a re-export with
    a current exporter, which stamps the declaration, and the bytes themselves
    do not change.

    A world size above one remains ATTEMPTED and not ATTESTED whatever this
    gate says: ``runtime_contract.json`` publishes ``max_world_size: 1`` for
    every family until a multi-rank serve has been run.
    """
    if int(world) <= 1:
        return
    agnostic = artifact_tp_agnostic(config)
    if agnostic is True:
        return
    if agnostic is False:
        raise ValueError(
            f"tessera target {prefix!r}: this artifact declares tp_agnostic=false, so its own "
            f"bytes say no rank can cut a shard out of them, and this is "
            f"tensor_parallel_size={int(world)}. A cut needs the shard record and the "
            "INITIAL_STATE plane the container gained when slicing became expressible "
            "(tessera.layout.SLICEABLE_SCHEMA_MINOR); wire written below that minor cannot "
            "carry a window of a unit at all. Serve with tensor_parallel_size=1, or re-export "
            "the checkpoint with a current exporter -- the TP degree is not an encoding choice "
            "and never was.")
    raise ValueError(
        f"tessera target {prefix!r}: this checkpoint's quantization_config declares neither "
        f"'tp_agnostic' nor 'schema_minor', so it does not say whether its bytes can be cut, "
        f"and this is tensor_parallel_size={int(world)}. It was written before the exporter "
        "stamped that declaration (such a config carries 'tp_size: 1', which is a constant no "
        "loader ever read and not a statement about slicing), and an artifact that does not "
        "declare a property does not get to claim it. Serve with tensor_parallel_size=1, or "
        "re-export with a current exporter; the wire does not change, only what the config "
        "says about it.")


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


def _check_tp_coordinates(prefix: str, tp_rank: int, tp_size: int) -> Tuple[int, int]:
    tp_rank, tp_size = int(tp_rank), int(tp_size)
    if tp_size < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError(f"{prefix}: nonsensical TP coordinates rank {tp_rank} of {tp_size}")
    return tp_rank, tp_size


def layer_tp_coordinates(prefix: str, layer) -> Tuple[int, int]:
    """The LAYER's ``(tp_rank, tp_size)``, read off it -- never the process's.

    Every vLLM ``LinearBase`` sets both in its ``__init__``, before it calls
    ``create_weights``: ``(0, 1)`` when built with ``disable_tp``, the pair its
    caller passed when it belongs to an effective group
    (``DCPGroupColumnParallelLinear``), and the process's parallel state
    otherwise -- and ``update_param_tp_status`` then reconciles every parameter
    to THESE, not to the global rank, "which correctly accounts for disable_tp"
    (vLLM's own comment).  So the coordinates a plan must be made for are the
    layer's, and ``vllm.distributed`` would answer a different question for a
    one-rank layer inside a four-rank process (tessera#303).

    A layer without the pair is refused BY NAME, not defaulted to one rank:
    ``TesseraConfig.get_quant_method`` hands the dense methods only
    ``LinearBase`` instances, every one of which carries the pair, so an object
    without them is not a vLLM Linear and its TP identity is not a thing this
    module may guess.
    """
    tp_rank = getattr(layer, "tp_rank", None)
    tp_size = getattr(layer, "tp_size", None)
    if tp_rank is None or tp_size is None:
        raise ValueError(
            f"{prefix}: the layer handed to create_weights is a {type(layer).__name__} carrying "
            f"tp_rank={tp_rank!r}, tp_size={tp_size!r}. Every vLLM LinearBase sets both before "
            f"create_weights (0 of 1 under disable_tp, the group's otherwise) and the shard plan "
            f"is made for the LAYER's coordinates, not the process's; a layer without them has no "
            f"tensor-parallel identity to plan for, and one rank is not assumed.")
    return _check_tp_coordinates(prefix, tp_rank, tp_size)


#: The attribute ``QKVParallelLinear`` publishes its KV replication under, and
#: its ``weight_loader`` reads (``shard_rank = self.tp_rank //
#: self.num_kv_head_replicas`` for the k and v shards).  One name, vLLM's.
KV_REPLICAS_ATTRIBUTE = "num_kv_head_replicas"


def layer_replicas(prefix: str, layer, members) -> Tuple[int, ...]:
    """How many ranks hold each member's shard, as the LAYER declares it.

    Returns one integer per member, in stacking order.  ``1`` everywhere for a
    Linear that declares nothing -- Column/Row/Replicated and the merged
    gate/up form cut every member ``tp_size`` ways or not at all -- and
    ``(1, R, R)`` for a ``QKVParallelLinear``, whose ``num_kv_head_replicas`` is
    vLLM's statement that k and v are held by ``R`` ranks each while q is cut
    evenly (``linear.py``: ``shard_rank = tp_rank if id == "q" else tp_rank //
    num_kv_head_replicas``).

    READ, NOT INFERRED.  The old plan derived replication from the numbers --
    "fewer distinct shards than ranks, by divisibility" -- which accepted a
    wire the size of one rank's tile as a replicated whole under any Linear
    (tessera#303).  Replication is a fact about the layer and only the layer
    states it; a Linear without the attribute has none.

    The attribute is ``QKVParallelLinear``'s and describes exactly three
    members.  A layer carrying it over any other member count -- one role, or
    the indexer variant's four (``index_k`` is cut ``tp`` ways, not replicated)
    -- is a statement this function does not know how to map onto the scheme's
    roles, and it is refused by name rather than guessed at.
    """
    members = tuple(str(name) for name in members)
    declared = getattr(layer, KV_REPLICAS_ATTRIBUTE, None)
    if declared is None:
        return (1,) * len(members)
    replicas = int(declared)
    if replicas < 1:
        raise ValueError(
            f"{prefix}: the layer declares {KV_REPLICAS_ATTRIBUTE}={declared!r}, which is not a "
            f"count of ranks per KV shard; vLLM's QKVParallelLinear computes it as "
            f"tp_size // total_num_kv_heads, at least 1.")
    if len(members) != 3:
        raise ValueError(
            f"{prefix}: the layer declares {KV_REPLICAS_ATTRIBUTE}={replicas}, which is "
            f"QKVParallelLinear's statement about three members (q cut evenly, k and v held by "
            f"{replicas} ranks each), but the scheme declares {len(members)} role"
            f"{'' if len(members) == 1 else 's'} {list(members)}. Which of them the replication "
            f"covers is not something this plan may guess, so the module is refused rather than "
            f"served with a replication it cannot place.")
    return (1, replicas, replicas)


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


def unsliceable_reason(unit) -> Optional[str]:
    """``layout.unsliceable_reason``, or None when the cutter is absent.

    WHY the wire refuses this unit outright, in the words of the module that
    decided it. ``can_shard`` can refuse the unit outright (rotation, a
    straddling scale block), the requested cut's geometry, or its granularity.
    This reports only the first, without re-deriving the rule. ``None`` means
    "no whole-unit obstruction, or this build has no cutter"; both are only
    ever read after ``can_shard`` has
    already said no, and the no-cutter case is refused above the callers.
    """
    try:
        from tessera.layout import unsliceable_reason as _reason
    except Exception:
        return None
    return _reason(unit)


def shard_cut_reason(unit, shards: int, axis: str) -> Optional[str]:
    """The cutter's requested-window obstruction, if this build exposes it."""
    try:
        from tessera.layout import shard_cut_reason as _reason
    except Exception:
        return None
    return _reason(unit, int(shards), axis)


def _cannot_cut(unit, plan: "ShardPlan", role: "RoleShard", extent: int) -> str:
    """The refusal for a role ``can_shard`` said no to.  ONE HOME for the text.

    A granularity that does not divide the split may be fixed by another TP
    degree, so its message names that number (tessera#235). Rotation and a
    straddling scale block refuse every cut, including the identity. RELEASE
    can instead refuse a requested column window: changing a row split's TP
    degree never changes its retained width. Both geometry cases need the
    cutter's reason, without an impossible divisor remedy.

    Until tessera#304 made ``can_shard`` correctly answer ``False`` for a
    rotated unit, this branch never saw one -- the seam fell through to
    ``slice_unit``, which raised the true reason.  Afterwards it fired for the
    rotated population too and told the operator to "serve with a
    tensor_parallel_size that divides it" a unit whose granularity was 1 and
    whose extent 2 divided perfectly (tessera#329).  The reason is now asked of
    the code that owns the question rather than re-derived at the raise site,
    which is the same move #304 made for the predicate.
    """
    reason = unsliceable_reason(unit)
    if reason is not None:
        return (
            f"{plan.prefix}: role {role.name!r} cannot be cut on the {plan.axis} axis at all "
            f"({extent} {plan.axis}s) -- {reason}. That is a property of this unit and not of "
            f"the split, so it refuses every cut including the identity one, and NO "
            f"tensor_parallel_size above 1 can serve it: there is no divisor to offer. Serve "
            f"this model with tensor_parallel_size=1, or export this Linear without the "
            f"structure named above.")
    reason = shard_cut_reason(unit, role.shards, plan.axis)
    if reason is not None:
        return (
            f"{plan.prefix}: role {role.name!r} cannot be cut {role.shards} ways on the "
            f"{plan.axis} axis ({extent} {plan.axis}s) -- {reason}. Serve with "
            "tensor_parallel_size=1, or export this Linear with geometry that "
            "supports the requested cut.")
    granularity = shard_granularity(unit)
    row_gran, col_gran = granularity if granularity is not None else (None, None)
    gran = row_gran if plan.axis == AXIS_ROWS else col_gran
    return (
        f"{plan.prefix}: role {role.name!r} cannot be cut {role.shards} ways on the "
        f"{plan.axis} axis ({extent} {plan.axis}s, granularity {gran}).  A row cut lands on a "
        "trellis super-symbol and a column cut of a unit carrying a RELEASE plane or a mixed "
        "rate schedule is confined to whole 256-column superblocks; serve with a "
        "tensor_parallel_size that divides it, or with 1.")


def _require_whole_output_boundaries(shape: str, roles, out_partitions) -> None:
    """A whole output is every role's rows, MEMBER BY MEMBER.

    EQUAL TOTALS ARE NOT EQUAL BOUNDARIES.  A container that stacks q/k/v as
    [4, 8, 4] and a layer that reads [8, 4, 4] agree on 16 and on nothing
    else, and the load would succeed serving four q rows and the first four k
    rows as this layer's q.  Nothing downstream sees it -- the decoder returns
    the role arrangement it was handed, and the sidecar agrees with the wire
    it was written beside -- so the boundaries are compared HERE, where both
    lists are in hand, on every path that serves the output whole (#234).
    """
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


def _require_no_replication(prefix: str, roles, replicas, served: str) -> None:
    """Replication is a statement about a CUT output; anywhere else it
    contradicts the shape the layer gave, and the contradiction is refused."""
    for (name, _rows), factor in zip(roles, replicas):
        if factor != 1:
            raise ValueError(
                f"{prefix}: the layer declares {factor} replicas of its role {name!r} "
                f"({KV_REPLICAS_ATTRIBUTE}) but its shape says the module is {served}; "
                f"replication describes how a CUT output is shared across ranks, so the layer "
                f"contradicts itself and the module is refused rather than planned on one half "
                f"of the statement.")


def plan_shard(prefix: str, *, roles, columns: int, out_partitions, in_size: int,
               tp_rank: int, tp_size: int, input_size: int, output_size: int,
               replicas=None) -> ShardPlan:
    """Decide, from what the LAYER declares, which slice of each role this rank is.

    ``roles`` is the checkpoint's ``[(name, rows)]`` in stacking order --
    ``validate_tessera_scheme``'s own field -- and ``columns`` its width; that
    is the WIRE.  Everything else is the LAYER: ``out_partitions`` is vLLM's
    ``output_partition_sizes`` and ``in_size`` its ``input_size_per_partition``
    (the tile THIS RANK computes); ``input_size``/``output_size`` are the
    module's GLOBAL shape as the layer states it; ``tp_rank``/``tp_size`` are
    the layer's own coordinates (``layer_tp_coordinates``); ``replicas`` is one
    integer per member, how many ranks hold each member's shard
    (``layer_replicas``, ``None`` meaning one everywhere).  The routes call
    ``plan_shard_for_layer``, which reads the last two off the layer.

    ``roles``/``out_partitions``/``replicas`` are LISTS on purpose: a fused
    container's members are cut independently, and summing them to two scalars
    is what made a replicated-KV QKV module unplannable (#32).  ``columns`` and
    ``in_size`` stay scalars because the input axis is one width, evenly split.

    THE GLOBAL SHAPE AGAINST THE TILE NAMES THE AXIS.  With ``tile =
    (sum(out_partitions), in_size)`` there are exactly three vLLM Linear
    relations -- ``(input_size, output_size)`` equal to the tile (served
    WHOLE: ``ReplicatedLinear``, or any Linear at one rank), the output times
    ``tp_size`` (an OUTPUT cut: ColumnParallel and its merged/QKV forms), or the
    input times ``tp_size`` (an INPUT cut: RowParallel) -- and a layer that
    satisfies none is refused with both shapes.  The wire is then held to the
    layer's statement: served whole, it must be the module; cut on the input,
    its width must be ``input_size``; cut on the output, each role's rows must
    be the member's COMPLETE extent under the layer's replication,
    ``out_i * (tp_size // replicas_i)``.  A wire the size of one rank's tile
    agrees with the tile on every local number, and reading only the tile
    handed it back as a replicated whole module on every rank (tessera#303);
    the refusal names the role, the wire's rows, and the extent the layer's
    contract requires.

    THE LISTS ARE PAIRED BY POSITION, and that is a dependency worth stating:
    vLLM passes no names with ``output_partition_sizes``, so the correspondence
    rests on the container's stacking order being the layer's.  It is not
    assumed blind -- ``parse_tessera_blob_for_scheme`` refuses a container
    whose members are not exactly the declared roles in order, that order is
    the row order the exporter packed (``fused.pack_fused``), and it is the
    order the SERVED tp=1 tile is stacked in, so a mis-ordering would already
    be wrong at one rank.  The per-role checks below are the only cross-check
    available here; they catch a mis-ordering whenever the roles differ in
    size, and cannot when they do not.

    THE TOTALS ARE NEVER THE ANSWER ON THEIR OWN.  Two lists that sum alike
    can name different boundaries -- ``[4, 8, 4]`` and ``[8, 4, 4]`` are both
    16 rows -- so the lengths, and wherever the output is served whole the
    per-member extents, are compared before a plan is returned (#234).

    Refuses -- with the module's shape, the role and the relation that failed --
    when the request is neither the whole module nor a split this wire can
    express.  A checkpoint and a serve disagreeing about a module's geometry is
    the failure worth being loud about; head replication the layer DECLARES is
    NOT that failure, and does not get that message.
    """
    roles = tuple((str(name), int(rows)) for name, rows in roles)
    out_partitions = tuple(int(size) for size in out_partitions)
    if not roles:
        raise ValueError(f"{prefix}: a module has at least one role; the scheme declared none")
    columns, in_size = int(columns), int(in_size)
    input_size, output_size = int(input_size), int(output_size)
    tp_rank, tp_size = _check_tp_coordinates(prefix, tp_rank, tp_size)
    rows = sum(role_rows for _name, role_rows in roles)
    out_size = sum(out_partitions)
    shape = (f"{prefix}: tessera module is {rows}x{columns}; rank {tp_rank} of {tp_size} wants "
             f"{out_size}x{in_size} of a layer declared {output_size}x{input_size}")

    # THE LISTS ARE ONE LIST, and that is asked ONCE, before any branch.
    if len(out_partitions) != len(roles):
        raise ValueError(
            f"{shape}. The checkpoint declares {len(roles)} roles "
            f"{[name for name, _ in roles]} but the layer asks for {len(out_partitions)} output "
            f"partitions {list(out_partitions)}; a fused container's members and a layer's "
            f"output partitions are the same list in the same order, so the two disagree about "
            f"how this module is fused -- which is a checkpoint/serve mismatch, not a shard.")
    replicas = (1,) * len(roles) if replicas is None else tuple(int(r) for r in replicas)
    if len(replicas) != len(roles):
        raise ValueError(
            f"{shape}. The layer declares {len(replicas)} replica counts {list(replicas)} for "
            f"{len(roles)} roles {[name for name, _ in roles]}; replication is stated once per "
            f"member, in stacking order.")
    for (name, _role_rows), factor in zip(roles, replicas):
        if factor < 1 or tp_size % factor:
            raise ValueError(
                f"{shape}. The layer declares {factor} replicas of its role {name!r} over "
                f"{tp_size} ranks, which is not a whole number of ranks per shard: replication "
                f"means tp_size / replicas distinct shards, each held by exactly `replicas` "
                f"ranks (vLLM: num_kv_head_replicas = tp_size // total_num_kv_heads, and its "
                f"own divide() refuses a remainder). The layer is malformed.")

    layer_whole = output_size == out_size and input_size == in_size
    if layer_whole:
        # ReplicatedLinear inside a group, or any Linear at one rank: the layer
        # says the tile IS the module, so the wire must be exactly the module.
        # This is the ONLY way a whole wire at tp_size > 1 is a plan: by the
        # layer's declaration, never by the wire happening to match the tile.
        if out_size != rows or in_size != columns:
            if tp_size == 1:
                raise ValueError(
                    f"{prefix}: tessera module is {rows}x{columns} but the layer wants "
                    f"{out_size}x{in_size} on one rank; a wire is bound to its shape")
            raise ValueError(
                f"{shape}, which is a module served whole on every rank of {tp_size} "
                f"(ReplicatedLinear: the layer's global shape is the tile it asks for). The "
                f"wire is {rows}x{columns} and the layer {output_size}x{input_size}; a wire is "
                f"bound to its shape, and a replicated module is that shape on every rank.")
        _require_whole_output_boundaries(shape, roles, out_partitions)
        _require_no_replication(prefix, roles, replicas, f"served whole on every rank of {tp_size}")
        return ShardPlan(prefix, rows, columns, rows, columns, tp_rank, tp_size, None,
                         tuple(RoleShard(name, role_rows, 0, role_rows, 1)
                               for name, role_rows in roles))
    if tp_size == 1:
        raise ValueError(
            f"{prefix}: tessera module is {rows}x{columns}; the layer wants {out_size}x{in_size} "
            f"on one rank and declares itself {output_size}x{input_size}, which is not the tile. "
            f"On one rank a layer asks for its whole shape, and a wire is bound to its shape.")

    output_cut = output_size == out_size * tp_size and input_size == in_size
    input_cut = input_size == in_size * tp_size and output_size == out_size
    if not output_cut and not input_cut:
        raise ValueError(
            f"{shape}: the layer's global shape is neither the tile (ReplicatedLinear, served "
            f"whole), nor the tile's output times {tp_size} over the whole input "
            f"({out_size * tp_size}x{in_size}: ColumnParallel and its merged/QKV forms), nor "
            f"the whole output over the tile's input times {tp_size} ({out_size}x"
            f"{in_size * tp_size}: RowParallel). A vLLM parallel Linear splits exactly one axis, "
            f"and this layer's declaration matches none of the three, so there is no cut to "
            f"plan.")

    if input_cut:
        # RowParallelLinear: the output is whole, the input is one width over
        # the ranks, evenly -- there is no per-role structure on this axis,
        # because every role reads every input feature.
        if input_size != columns:
            raise ValueError(
                f"{shape}, which cuts the input {tp_size} ways: a row-parallel Linear over "
                f"{input_size} input features gives each rank {in_size} of them. The wire has "
                f"{columns} columns, not {input_size}"
                + (f" -- exactly a rank's tile. A whole wire the width of one rank's share is "
                   f"not a replicated module: this layer declared its input {input_size} wide, "
                   f"and columns [{columns}, {input_size}) exist on no rank."
                   if columns == in_size else
                   f"; the checkpoint and the serve disagree about this module's input width."))
        _require_whole_output_boundaries(shape, roles, out_partitions)
        _require_no_replication(prefix, roles, replicas, f"cut on its input {tp_size} ways")
        c0 = tp_rank * in_size
        return ShardPlan(prefix, rows, columns, rows, in_size, tp_rank, tp_size, AXIS_COLUMNS,
                         tuple(RoleShard(name, columns, c0, c0 + in_size, tp_size)
                               for name, _role_rows in roles))

    # ColumnParallelLinear and its merged/QKV forms: the input is whole and
    # each member's rows are cut into tp_size // replicas_i distinct shards.
    if input_size != columns:
        raise ValueError(
            f"{shape}, which cuts the output {tp_size} ways over the whole input -- and the "
            f"layer's input is {input_size} features while the wire has {columns} columns. "
            f"The checkpoint and the serve disagree about this module's input width.")
    placed = []
    for (name, role_rows), out, factor in zip(roles, out_partitions, replicas):
        if out <= 0:
            raise ValueError(
                f"{shape}. Its role {name!r} asks for {out} output rows on this rank; vLLM "
                f"gives every rank a positive share of every member.")
        shards = tp_size // factor
        whole = out * shards
        contract = (f"{shards} shards of {out}"
                    + (f", each held by {factor} ranks as the layer declares"
                       if factor != 1 else ""))
        if whole != role_rows:
            if role_rows < whole:
                raise ValueError(
                    f"{shape}. Its role {name!r} is {role_rows} rows on the wire, but this "
                    f"layer's contract makes its complete extent {whole} rows ({contract}): "
                    f"the wire holds {role_rows} of {whole}"
                    + (f" -- exactly a rank's tile. A whole wire the size of one rank's share "
                       f"is not a replicated module: the layer declared no replication of this "
                       f"role, and rows [{role_rows}, {whole}) exist on no rank."
                       if role_rows == out and factor == 1 else
                       f", and rows [{role_rows}, {whole}) exist on no rank.")
                    + " The checkpoint and the serve disagree about this module's geometry.")
            raise ValueError(
                f"{shape}. Its role {name!r} is {role_rows} rows on the wire, but this "
                f"layer's contract covers only {whole} of them ({contract}): rows [{whole}, "
                f"{role_rows}) would be served by nobody, and an all-gather would return a "
                f"tile with rows no rank computed. The checkpoint and the serve disagree about "
                f"this module's geometry.")
        index = tp_rank // factor
        placed.append(RoleShard(name, role_rows, index * out, index * out + out, shards))
    return ShardPlan(prefix, rows, columns, out_size, columns, tp_rank, tp_size, AXIS_ROWS,
                     tuple(placed))


def plan_shard_for_layer(prefix: str, layer, *, roles, columns: int, input_size_per_partition,
                         output_partition_sizes, input_size, output_size) -> ShardPlan:
    """``plan_shard`` fed from the LAYER: what a route calls at ``create_weights``.

    Takes exactly what vLLM passes to ``LinearMethodBase.create_weights`` and
    reads the two facts it does not pass off the layer object -- its own TP
    coordinates (``layer_tp_coordinates``) and its declared KV replication
    (``layer_replicas``).  Nothing is read from the process's parallel state,
    and nothing about the layer's geometry is inferred from the tile alone
    (tessera#303).
    """
    tp_rank, tp_size = layer_tp_coordinates(prefix, layer)
    replicas = layer_replicas(prefix, layer, (name for name, _rows in roles))
    return plan_shard(prefix, roles=roles, columns=columns,
                      out_partitions=output_partition_sizes,
                      in_size=input_size_per_partition, tp_rank=tp_rank, tp_size=tp_size,
                      input_size=input_size, output_size=output_size, replicas=replicas)


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
    BEFORE the cut so a granularity-bound refusal can name a
    ``tensor_parallel_size``: ``slice_unit`` would refuse too, but with an
    offset the operator cannot map back to one.  Where the unit refuses EVERY
    cut instead, the cutter's sentence is the useful one and ``_cannot_cut``
    raises that -- asking ``unsliceable_reason`` rather than inventing a
    granularity story the unit's problem has nothing to do with (tessera#329).

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
        raise ValueError(_cannot_cut(unit, plan, role, extent))
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
    and only ``can_shard`` knows that.  Its refusal is ``_cannot_cut``'s, the
    same one the seam raises, so the two cannot say different things about one
    unit -- and so this one cannot offer a divisor for a unit no divisor cuts
    either (tessera#329).  The granularity is still what the OFFSET check below
    reports, which is the number an operator can act on there.
    """
    if plan.axis is None or role.is_whole:
        return
    granularity = shard_granularity(unit)
    if granularity is None:
        return
    row_gran, col_gran = granularity
    gran = row_gran if plan.axis == AXIS_ROWS else col_gran
    if can_shard(unit, role.shards, plan.axis) is False:
        raise ValueError(_cannot_cut(unit, plan, role, role.extent))
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
