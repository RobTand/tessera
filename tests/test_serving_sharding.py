"""The TP seam: which slice a rank serves, and that TP=1 is bit-identically today's.

The plugin has served ``tensor_parallel_size=1``.  These tests are about what
happens at the boundary anyway, because the loaders are the part of a serving
route that is expensive to retrofit: they decide, at ``create_weights`` time,
what shape the layer believes it has, and a wrong answer there is a serve that
runs and returns wrong logits.

Three properties matter and are asserted directly, against REAL units:

1. **TP=1 is identity.**  ``_shard_unit_for_rank`` returns the SAME OBJECT, so
   a caller's parsed view of a unit stays valid and the arithmetic is the
   arithmetic of a build with no TP support at all.
2. **TP>1 cuts, and the cut is the parent's own rows.**  The cut belongs to
   ``tessera.layout.slice_unit``, and the check here is the one that matters:
   the shard *decodes* to exactly the parent's slice.  A seam that returned
   some unit of the right shape would pass every structural check and serve
   wrong logits.
3. **A cut the wire cannot express is refused with the granularity.**  A mixed
   rate schedule confines a column cut to whole 256-column superblocks, and the
   refusal has to carry ``256`` -- a number the operator can turn into a
   ``tensor_parallel_size``.

These tests were written for the pre-slicer world, where the seam refused every
``tp_size > 1`` and a bare ``object()`` was a sufficient stand-in for a unit
because nothing ever looked inside one.  ``tessera.layout``'s slicer has landed
(``docs/design/tensor-parallel.md``), so they are driven by real encoded units
now: against the seam as it is, a sentinel would only prove that it still does
not look.
"""
from __future__ import annotations

import pathlib

import pytest

from tessera.serving.sharding import (AXIS_COLUMNS, AXIS_ROWS, RoleShard,
                                      ShardPlan, _shard_unit_for_rank,
                                      _unit_extent, check_shard_granularity,
                                      layer_replicas, layer_tp_coordinates,
                                      plan_shard, plan_shard_for_layer,
                                      shard_granularity, shard_parsed_roles)

torch = pytest.importorskip("torch")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the Tessera encoder is a CUDA path")

#: 64x512 is the smallest shape that still exercises arity 2 -- the E2M1x2 cap
#: unit is 32 trellis steps for 64 weight rows, which is exactly the extent bug
#: ``_unit_extent`` exists to avoid.
UNIT_ROWS, UNIT_COLS = 64, 512


def _make_unit(label, q256, grid_name):
    from tessera.alphabet import E2M1_GRID, E4M3_GRID, tuple_grid
    from tessera.export import encode_linear_planes
    from tessera.unit_artifact import parse_unit_artifact

    grid = E4M3_GRID if grid_name == "E4M3" else tuple_grid(E2M1_GRID, 2)
    torch.manual_seed(11)
    weight = (torch.randn(UNIT_ROWS, UNIT_COLS, device="cuda") * 0.02).contiguous()
    exported, _unit, _forests = encode_linear_planes(
        weight, grid=grid, q256=q256, name=label, verify=False)
    return parse_unit_artifact(exported.blob, device="cuda")


@pytest.fixture(scope="module")
def units():
    """One parsed unit per case, encoded once (the encoder is the slow part)."""
    if not torch.cuda.is_available():
        pytest.skip("the Tessera encoder is a CUDA path")
    return {
        # window body / CHANNEL plane, uniform rates: cuts freely on both axes.
        "e4m3": _make_unit("e4m3", 1024, "E4M3"),
        # the same wire below the cap, where the Bresenham schedule is mixed and
        # the quota is exact only on a whole superblock.
        "e4m3-mixed": _make_unit("e4m3-mixed", 1000, "E4M3"),
        # the E2M1x2 cap wire: TCQ span 2 over an arity-2 grid, LUT plane.
        "e2m1x2": _make_unit("e2m1x2", 896, "E2M1x2"),
    }


def _decode(parsed):
    from tessera.decode import reconstruct_unit

    return reconstruct_unit(parsed.unit, parsed.forests, parsed.code)


def _decode_unit(unit, parsed):
    from tessera.decode import reconstruct_unit

    return reconstruct_unit(unit, parsed.forests, parsed.code)


def _without_tessera_layout(monkeypatch):
    """Make ``import tessera.layout`` fail, as an older Tessera would."""
    import builtins

    real_import = builtins.__import__

    def no_layout(name, *args, **kwargs):
        if name == "tessera.layout":
            raise ImportError("no tessera.layout in this build")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_layout)


def _plan(rows, columns, *, out, in_, rank, tp, input_size=None, output_size=None):
    """A dense one-role plan, the shape most of these tests are about.

    ``input_size``/``output_size`` are the LAYER's global shape -- what a vLLM
    Linear passes beside the tile -- and default to the wire's, which is the
    well-formed case: a module whose checkpoint is the module.
    """
    return plan_shard("m", roles=[("weight", rows)], columns=columns,
                      out_partitions=[out], in_size=in_, tp_rank=rank, tp_size=tp,
                      input_size=columns if input_size is None else input_size,
                      output_size=rows if output_size is None else output_size)


def _seam_plan(rows, columns, axis, *, rank, tp, shards):
    """A one-role plan cut ``shards`` ways on ``axis``, for driving the seam.

    Built through ``plan_shard`` rather than by hand so the seam is always
    exercised against a plan the planner would actually produce.
    """
    if axis == AXIS_ROWS:
        return _plan(rows, columns, out=rows // shards, in_=columns, rank=rank, tp=tp)
    return _plan(rows, columns, out=rows, in_=columns // shards, rank=rank, tp=tp)


def test_one_rank_is_the_whole_module():
    plan = plan_shard("m", roles=[("weight", 1024)], columns=2048, out_partitions=[1024],
                      in_size=2048, tp_rank=0, tp_size=1, input_size=2048, output_size=1024)
    assert plan.is_whole and plan.axis is None
    assert (plan.shard_rows, plan.shard_columns) == (1024, 2048)


def test_an_output_split_is_the_row_axis():
    """ColumnParallel / MergedColumnParallel / QKVParallel: rows over ranks."""
    plan = plan_shard("m", roles=[("weight", 1024)], columns=2048, out_partitions=[256],
                      in_size=2048, tp_rank=1, tp_size=4, input_size=2048, output_size=1024)
    assert plan.axis == AXIS_ROWS
    assert (plan.shard_rows, plan.shard_columns) == (256, 2048)
    assert (plan.role("weight").lo, plan.role("weight").hi) == (256, 512)
    assert plan.role("weight").shards == 4


def test_an_input_split_is_the_column_axis():
    """RowParallel: the input width over ranks, every row present."""
    plan = plan_shard("m", roles=[("weight", 1024)], columns=2048, out_partitions=[1024],
                      in_size=512, tp_rank=3, tp_size=4, input_size=2048, output_size=1024)
    assert plan.axis == AXIS_COLUMNS
    assert (plan.shard_rows, plan.shard_columns) == (1024, 512)
    assert (plan.role("weight").lo, plan.role("weight").hi) == (1536, 2048)


def test_a_replicated_linear_inside_a_tp_group_is_a_whole_unit():
    """Whole shapes at tp_size > 1 are a replicated layer, not a defect --
    when the LAYER says the whole shape is its global shape."""
    plan = plan_shard("m", roles=[("weight", 1024)], columns=2048, out_partitions=[1024],
                      in_size=2048, tp_rank=2, tp_size=4, input_size=2048, output_size=1024)
    assert plan.axis is None
    assert (plan.shard_rows, plan.shard_columns) == (1024, 2048)


# --- the layer's GLOBAL shape and its replication contract (tessera#303) --------

def test_an_undersized_whole_wire_is_a_tile_not_a_replicated_module():
    """FAILS BEFORE -- tessera#303 at the plan level.

    A ColumnParallel layer at TP=4 over ``[256, 64]`` asks each rank for a
    ``[64, 64]`` tile; the wire is a whole ``[64, 64]``.  Tile and wire agree
    on every local number, so a plan reading only the tile handed back a
    replicated whole module on all four ranks -- rows ``[0, 64)`` everywhere,
    rows ``[64, 256)`` nowhere.  The layer's own ``output_size = 256`` is the
    fact that distinguishes the two, and it is refused on EVERY rank, by name.
    """
    for rank in range(4):
        with pytest.raises(ValueError) as e:
            _plan(64, 64, out=64, in_=64, rank=rank, tp=4, input_size=64, output_size=256)
        msg = str(e.value)
        assert "'weight'" in msg and "64 rows on the wire" in msg
        assert "256" in msg and "4 shards of 64" in msg
        assert f"rank {rank} of 4" in msg
        assert "a rank's tile" in msg, "the refusal says what the wire actually is"
    # The control: the wire IS the module, and rank r owns [64r, 64r + 64).
    for rank in range(4):
        plan = _plan(256, 64, out=64, in_=64, rank=rank, tp=4, input_size=64, output_size=256)
        assert plan.axis == AXIS_ROWS
        assert (plan.role("weight").lo, plan.role("weight").hi, plan.role("weight").shards) == \
            (64 * rank, 64 * rank + 64, 4)


def test_an_undersized_whole_wire_on_the_input_axis_is_refused_too():
    """FAILS BEFORE.  RowParallel at TP=4 over ``[64, 256]``: a ``[64, 64]``
    wire is one rank's quarter of the input, not a replicated module."""
    for rank in range(4):
        with pytest.raises(ValueError) as e:
            _plan(64, 64, out=64, in_=64, rank=rank, tp=4, input_size=256, output_size=64)
        msg = str(e.value)
        assert "64 columns" in msg and "256" in msg and f"rank {rank} of 4" in msg
    plan = _plan(64, 256, out=64, in_=64, rank=3, tp=4, input_size=256, output_size=64)
    assert plan.axis == AXIS_COLUMNS
    assert (plan.role("weight").lo, plan.role("weight").hi) == (192, 256)


def test_a_replicated_layer_whose_wire_is_not_its_shape_is_refused():
    """A ReplicatedLinear inside a group asks for its whole shape; a wire of
    another shape is a checkpoint/serve disagreement at every rank."""
    with pytest.raises(ValueError) as e:
        _plan(64, 64, out=256, in_=64, rank=1, tp=4, input_size=64, output_size=256)
    msg = str(e.value)
    assert "served whole" in msg and "64x64" in msg and "256x64" in msg


def test_a_global_shape_that_is_no_vllm_linear_relation_is_refused_with_both_shapes():
    """FAILS BEFORE.  ``(input_size, output_size)`` against the tile is one of
    three relations -- whole, output cut, input cut -- or nothing this plan
    serves.  Before, the global shape was never looked at."""
    with pytest.raises(ValueError) as e:
        _plan(64, 64, out=64, in_=64, rank=1, tp=4, input_size=128, output_size=128)
    msg = str(e.value)
    assert "128x128" in msg and "64x64" in msg and "rank 1 of 4" in msg
    assert "ReplicatedLinear" in msg and "ColumnParallel" in msg and "RowParallel" in msg


def test_the_same_numbers_are_a_whole_kv_head_with_replication_declared_and_a_tile_without():
    """FAILS BEFORE, and it is the clause of tessera#303 that rules out a
    comparison against the padded aggregate alone.

    MQA at TP=8: k is 64 rows on the wire, 64 asked for, and vLLM's
    ``QKVParallelLinear`` declares ``num_kv_head_replicas = 8``.  The same
    three numbers on a Linear declaring no replication are a ``[512, 1024]``
    module with a ``[64, 1024]`` wire.  Only the declared replication tells
    the two apart; the plan may not infer it from the numbers.
    """
    roles = [("q_proj", 8 * 64), ("k_proj", 64), ("v_proj", 64)]
    plan = plan_shard("qkv", roles=roles, columns=1024, out_partitions=[64, 64, 64],
                      in_size=1024, tp_rank=3, tp_size=8, input_size=1024, output_size=1536,
                      replicas=(1, 8, 8))
    assert plan.role("k_proj").is_whole and plan.role("k_proj").shards == 1
    with pytest.raises(ValueError) as e:
        plan_shard("linear", roles=[("weight", 64)], columns=1024, out_partitions=[64],
                   in_size=1024, tp_rank=3, tp_size=8, input_size=1024, output_size=512)
    msg = str(e.value)
    assert "'weight'" in msg and "64 rows on the wire" in msg and "8 shards of 64" in msg


def test_replication_that_does_not_divide_the_world_is_refused_by_name():
    """A layer declaring 4 replicas of k over 6 ranks has no whole number of
    ranks per shard; vLLM's own ``divide`` would have raised in ``__init__``,
    so seeing it is a malformed layer, refused with the relation that failed."""
    roles = [("q_proj", 12 * 64), ("k_proj", 4 * 64), ("v_proj", 4 * 64)]
    with pytest.raises(ValueError) as e:
        plan_shard("qkv", roles=roles, columns=1024, out_partitions=[128, 64, 64],
                   in_size=1024, tp_rank=0, tp_size=6, input_size=1024, output_size=1536,
                   replicas=(1, 4, 4))
    msg = str(e.value)
    assert "'k_proj'" in msg and "4 replicas" in msg and "6 ranks" in msg


def test_replicas_are_one_per_role_or_refused():
    roles = [("q_proj", 512), ("k_proj", 64), ("v_proj", 64)]
    for bad in ((1, 8), (1, 8, 8, 1), (1, 0, 8), (1, 8, 16)):
        with pytest.raises(ValueError, match="replica"):
            plan_shard("qkv", roles=roles, columns=1024, out_partitions=[64, 64, 64],
                       in_size=1024, tp_rank=0, tp_size=8, input_size=1024, output_size=1536,
                       replicas=bad)


def test_replication_declared_on_a_whole_or_input_cut_module_is_refused():
    """Replication is a statement about a cut OUTPUT.  On a module the layer
    serves whole, or cuts on the input, it contradicts the shape."""
    with pytest.raises(ValueError, match="replica"):
        plan_shard("m", roles=[("a", 8), ("b", 8), ("c", 8)], columns=64,
                   out_partitions=[8, 8, 8], in_size=64, tp_rank=0, tp_size=4,
                   input_size=64, output_size=24, replicas=(1, 2, 2))
    with pytest.raises(ValueError, match="replica"):
        plan_shard("m", roles=[("a", 8), ("b", 8), ("c", 8)], columns=64,
                   out_partitions=[8, 8, 8], in_size=16, tp_rank=0, tp_size=4,
                   input_size=64, output_size=24, replicas=(1, 2, 2))


class _LinearStandIn:
    """What ``plan_shard_for_layer`` reads off a vLLM ``LinearBase``."""

    def __init__(self, tp_rank, tp_size, num_kv_head_replicas=None):
        self.tp_rank, self.tp_size = tp_rank, tp_size
        if num_kv_head_replicas is not None:
            self.num_kv_head_replicas = num_kv_head_replicas


def test_tp_coordinates_are_the_layers_own_and_a_layer_without_them_is_refused():
    """FAILS BEFORE.  The coordinates were read from ``vllm.distributed`` --
    the process's group -- which is wrong for ``disable_tp`` (a one-rank layer
    inside a group) and for any layer vLLM builds with an effective group
    (``DCPGroupColumnParallelLinear`` passes its own ``tp_rank``/``tp_size``).
    ``LinearBase`` sets both before ``create_weights``; a layer without them
    is not a vLLM Linear and gets a refusal naming what it is, not a default."""
    assert layer_tp_coordinates("m", _LinearStandIn(2, 4)) == (2, 4)
    assert layer_tp_coordinates("m", _LinearStandIn(0, 1)) == (0, 1)
    with pytest.raises(ValueError) as e:
        layer_tp_coordinates("model.layers.0.q", object())
    msg = str(e.value)
    assert "tp_rank" in msg and "tp_size" in msg and "LinearBase" in msg and "object" in msg
    with pytest.raises(ValueError, match="TP coordinates"):
        layer_tp_coordinates("m", _LinearStandIn(4, 4))


def test_layer_replicas_is_vllms_statement_read_verbatim_or_ones():
    """``num_kv_head_replicas`` is ``QKVParallelLinear``'s: ``(1, R, R)`` over
    q/k/v, and no other member count is a statement this reads."""
    members = ("q_proj", "k_proj", "v_proj")
    assert layer_replicas("m", _LinearStandIn(0, 8, 8), members) == (1, 8, 8)
    assert layer_replicas("m", _LinearStandIn(0, 16, 1), members) == (1, 1, 1)
    assert layer_replicas("m", _LinearStandIn(0, 4), members) == (1, 1, 1)
    assert layer_replicas("m", _LinearStandIn(0, 4), ("weight",)) == (1,)
    with pytest.raises(ValueError) as e:
        layer_replicas("m", _LinearStandIn(0, 4, 2), ("weight",))
    msg = str(e.value)
    assert "num_kv_head_replicas" in msg and "QKVParallelLinear" in msg and "1 role" in msg
    with pytest.raises(ValueError, match="num_kv_head_replicas"):
        layer_replicas("m", _LinearStandIn(0, 4, 2), ("q", "k", "v", "index_k"))
    with pytest.raises(ValueError, match="num_kv_head_replicas"):
        layer_replicas("m", _LinearStandIn(0, 4, 0), members)


def test_plan_shard_for_layer_reads_the_layer_and_plans_gqa_from_its_declaration():
    """The routes' entry point: GQA 16q/8kv over 16 ranks, replicas 2, on the
    layer; and the same call with an undersized single-role wire refuses."""
    roles = [("q_proj", 16 * 64), ("k_proj", 8 * 64), ("v_proj", 8 * 64)]
    plan = plan_shard_for_layer("qkv", _LinearStandIn(5, 16, 2), roles=roles, columns=2048,
                                input_size_per_partition=2048,
                                output_partition_sizes=[64, 64, 64],
                                input_size=2048, output_size=3072)
    assert (plan.tp_rank, plan.tp_size, plan.axis) == (5, 16, AXIS_ROWS)
    assert (plan.role("q_proj").lo, plan.role("q_proj").shards) == (320, 16)
    assert (plan.role("k_proj").lo, plan.role("k_proj").hi, plan.role("k_proj").shards) == \
        (128, 192, 8)
    with pytest.raises(ValueError, match="a rank's tile"):
        plan_shard_for_layer("linear", _LinearStandIn(1, 4), roles=[("weight", 64)],
                             columns=64, input_size_per_partition=64,
                             output_partition_sizes=[64], input_size=64, output_size=256)


def test_equal_totals_with_disagreeing_role_boundaries_are_refused_by_name():
    """A fused module the two sides stack DIFFERENTLY is not a whole module.

    Wire roles q/k/v of ``[4, 8, 4]`` and a layer asking for ``[8, 4, 4]``
    total 16 either way, so a planner that compares only the totals hands back
    a whole-module plan and the load succeeds.  The wire is decoded as four q
    rows then eight k rows while the consumer reads the first eight as q, and
    nothing downstream can catch it: the decoder returns the role arrangement
    it was given, and a self-consistent wire agrees with its own sidecar.  So
    the refusal has to be here, and it has to name the member that failed.

    Both cases the shortcut covered are checked -- ``tp_size == 1`` and a
    module replicated inside a TP group -- plus the same disagreement under a
    row-parallel (input) cut, where the output is whole too.
    """
    roles = [("q_proj", 4), ("k_proj", 8), ("v_proj", 4)]
    for rank, tp, in_size in ((0, 1, 64), (2, 4, 64), (1, 2, 32)):
        with pytest.raises(ValueError) as e:
            plan_shard("qkv", roles=roles, columns=64, out_partitions=[8, 4, 4],
                       in_size=in_size, tp_rank=rank, tp_size=tp,
                       input_size=64, output_size=16)
        msg = str(e.value)
        assert "'q_proj'" in msg, "the refusal names the member that failed"
        assert "4 rows" in msg and "asks for 8" in msg
        assert "Equal totals are not equal boundaries" in msg


def test_a_fused_container_and_a_layer_that_stack_differently_are_refused():
    """The list lengths are the same rule as the boundaries, asked once.

    A three-role container against a layer offering one output partition is
    the same checkpoint/serve disagreement about how a module is fused, and it
    has to be refused for a whole module too -- not only for a cut, which is
    the one branch that used to ask.
    """
    roles = [("q_proj", 512), ("k_proj", 256), ("v_proj", 256)]
    with pytest.raises(ValueError) as e:
        plan_shard("qkv", roles=roles, columns=2048, out_partitions=[1024],
                   in_size=2048, tp_rank=0, tp_size=1, input_size=2048, output_size=1024)
    msg = str(e.value)
    assert "3 roles" in msg and "1 output partition" in msg
    assert "how this module is fused" in msg


def test_replicated_kv_heads_under_gqa_are_a_plan_not_a_refusal():
    """GQA with ``num_kv_heads < tp``: two ranks hold the SAME k/v rows.

    vLLM replicates KV heads when there are fewer of them than ranks
    (``QKVParallelLinear.__init__``: ``num_kv_head_replicas = tp //
    total_num_kv_heads``), and its own loader then reads
    ``start_idx = (tp_rank // num_kv_head_replicas) * shard_size``.  So the
    per-rank sizes sum to MORE than the container's rows and no even split of
    it exists -- which is ordinary, not a checkpoint disagreement.  The
    declaration arrives as ``replicas``, read off the layer (tessera#303); the
    layer's ``output_size`` is the PADDED aggregate ``3 * 64 * 16``.

    16 query heads, 8 KV heads, head_size 64, over 16 ranks: q is cut 16 ways,
    k and v 8 ways with two ranks per shard.
    """
    roles = [("q_proj", 16 * 64), ("k_proj", 8 * 64), ("v_proj", 8 * 64)]
    for rank in (0, 1, 2, 15):
        plan = plan_shard("model.layers.0.self_attn.qkv_proj", roles=roles, columns=2048,
                          out_partitions=[64, 64, 64], in_size=2048,
                          tp_rank=rank, tp_size=16, input_size=2048, output_size=3072,
                          replicas=(1, 2, 2))
        assert plan.axis == AXIS_ROWS
        # The tile this rank computes: three heads, not rows/16.
        assert (plan.shard_rows, plan.shard_columns) == (192, 2048)
        q, k, v = plan.role("q_proj"), plan.role("k_proj"), plan.role("v_proj")
        assert (q.lo, q.hi, q.shards) == (64 * rank, 64 * rank + 64, 16)
        for member in (k, v):
            assert (member.lo, member.hi) == (64 * (rank // 2), 64 * (rank // 2) + 64)
            assert member.shards == 8, "8 KV heads is 8 shards, whatever the world size"
    # Ranks 0 and 1 hold IDENTICAL k rows, and different q rows.
    a, b = (plan_shard("qkv", roles=roles, columns=2048, out_partitions=[64, 64, 64],
                       in_size=2048, tp_rank=r, tp_size=16, input_size=2048,
                       output_size=3072, replicas=(1, 2, 2)) for r in (0, 1))
    assert (a.role("k_proj").lo, a.role("k_proj").hi) == (b.role("k_proj").lo, b.role("k_proj").hi)
    assert a.role("q_proj").lo != b.role("q_proj").lo


def test_a_single_kv_head_is_replicated_whole_to_every_rank():
    """MQA: one KV head, so every rank holds the k role entire and uncut."""
    roles = [("q_proj", 8 * 64), ("k_proj", 64), ("v_proj", 64)]
    plan = plan_shard("qkv", roles=roles, columns=1024, out_partitions=[64, 64, 64],
                      in_size=1024, tp_rank=3, tp_size=8, input_size=1024, output_size=1536,
                      replicas=(1, 8, 8))
    assert plan.axis == AXIS_ROWS
    assert plan.role("k_proj").is_whole and plan.role("k_proj").shards == 1
    assert not plan.role("q_proj").is_whole


def test_a_kv_role_the_wire_holds_short_of_its_declared_replication_is_refused_by_name():
    """GQA declared, wire undersized: replicas 2 over 16 ranks make k's complete
    extent ``64 * 8 = 512``; a wire with 256 k rows is half a module, and the
    declared replication does not excuse it."""
    roles = [("q_proj", 16 * 64), ("k_proj", 4 * 64), ("v_proj", 8 * 64)]
    with pytest.raises(ValueError) as e:
        plan_shard("qkv", roles=roles, columns=2048, out_partitions=[64, 64, 64],
                   in_size=2048, tp_rank=0, tp_size=16, input_size=2048, output_size=3072,
                   replicas=(1, 2, 2))
    msg = str(e.value)
    assert "'k_proj'" in msg and "256 rows on the wire" in msg and "8 shards of 64" in msg
    assert "a rank's tile" not in msg, "half a module is not one rank's tile; say what it is"


def test_a_role_the_wire_holds_beyond_the_layers_extent_is_refused_because_rows_would_go_unserved():
    """The other direction: the wire has MORE rows of a role than the layer's
    contract covers.  A rank asking for ``out`` with no replication declared
    covers ``out * tp_size`` rows; a wire above that has rows no rank holds,
    and an all-gather would return a tile with rows nobody computed.

    16 q heads and 8 KV heads over 4 ranks, with output partitions computed for
    a world of 8 -- the shape of handing a tp=8 layer's sizes to a tp=4 group.
    Before tessera#303 this was refused on divisibility ("8 shards for only 4
    ranks"); it is now refused on the layer's own contract, which is the fact.
    """
    roles = [("q_proj", 16 * 64), ("k_proj", 8 * 64), ("v_proj", 8 * 64)]
    with pytest.raises(ValueError) as e:
        plan_shard("model.layers.0.self_attn.qkv_proj", roles=roles, columns=2048,
                   out_partitions=[256, 64, 64], in_size=2048, tp_rank=0, tp_size=4,
                   input_size=2048, output_size=1536)
    msg = str(e.value)
    assert "'k_proj'" in msg, "the refusal names the member that failed, not the container"
    assert "512 rows on the wire" in msg and "4 shards of 64" in msg
    assert "served by nobody" in msg, "the refusal says what would go wrong, not just that it did"


def test_a_shape_that_is_neither_whole_nor_a_clean_split_is_refused_with_both_shapes():
    with pytest.raises(ValueError) as e:
        plan_shard("model.layers.0.mlp.down_proj", roles=[("weight", 1024)], columns=2048,
                   out_partitions=[300], in_size=2048, tp_rank=0, tp_size=4,
                   input_size=2048, output_size=1200)
    msg = str(e.value)
    assert "model.layers.0.mlp.down_proj" in msg
    assert "1024x2048" in msg and "300x2048" in msg


def test_a_wrong_shape_on_one_rank_still_says_a_wire_is_bound_to_its_shape():
    """The TP=1 message is the one this replaced; it must not have got vaguer."""
    with pytest.raises(ValueError, match="a wire is bound to its shape"):
        plan_shard("m", roles=[("weight", 1024)], columns=2048, out_partitions=[999],
                   in_size=2048, tp_rank=0, tp_size=1, input_size=2048, output_size=999)


def test_nonsensical_coordinates_are_refused():
    for rank, size in ((4, 4), (-1, 2), (0, 0)):
        with pytest.raises(ValueError, match="TP coordinates"):
            plan_shard("m", roles=[("weight", 8)], columns=8, out_partitions=[4],
                       tp_rank=rank, tp_size=size, in_size=8, input_size=8, output_size=16)


def test_the_seam_returns_the_same_object_on_one_rank():
    """One rank is identity, and identity does not look inside the unit.

    A sentinel is the right argument HERE and only here: the whole point is
    that the seam returns before it can inspect anything.
    """
    unit = object()
    plan = _plan(1024, 2048, out=1024, in_=2048, rank=0, tp=1)
    assert _shard_unit_for_rank(unit, plan, plan.role("weight")) is unit


@needs_cuda
@pytest.mark.parametrize("label", ["e4m3", "e2m1x2"])
@pytest.mark.parametrize("axis", [AXIS_ROWS, AXIS_COLUMNS])
def test_the_seam_cuts_a_real_unit_into_the_parents_own_rows(units, label, axis):
    """tp 2, both axes, both shipping wires: the shard IS the parent's slice."""
    parsed = units[label]
    whole = _decode(parsed)
    rows, columns = _unit_extent(parsed)
    assert (rows, columns) == (UNIT_ROWS, UNIT_COLS), \
        "arity 2 halves the step count, not the row count"
    extent = rows if axis == AXIS_ROWS else columns
    half = extent // 2
    for rank in (0, 1):
        plan = _seam_plan(rows, columns, axis, rank=rank, tp=2, shards=2)
        shard = _shard_unit_for_rank(parsed, plan, plan.role("weight"))
        assert shard is not parsed
        got = _decode_unit(shard, parsed)
        want = (whole[rank * half:(rank + 1) * half, :] if axis == AXIS_ROWS
                else whole[:, rank * half:(rank + 1) * half])
        assert torch.equal(got, want), f"{label} {axis} rank {rank}"


@needs_cuda
def test_the_seam_cuts_a_replicated_role_to_the_same_rows_on_two_ranks(units):
    """The #32 case, decoded: 4 ranks over 2 KV shards, and the pairs match.

    This is the property the plan-level test cannot prove.  A shard of the right
    SHAPE on every rank would pass every structural check and serve wrong
    logits; what has to hold is that ranks 0 and 1 decode to the parent's rows
    ``[0:32)`` -- the same rows, bit for bit -- and ranks 2 and 3 to ``[32:64)``.

    Driven through ``shard_parsed_roles``, which is the routes' own entry point,
    so the re-parse is in the loop too.
    """
    parsed = units["e4m3"]
    whole = _decode(parsed)
    rows, columns = _unit_extent(parsed)
    head = rows // 2                       # two KV "heads" of 32 rows, four ranks
    got = []
    for rank in range(4):
        # Two ranks per head is the LAYER's statement (``num_kv_head_replicas``),
        # and its aggregate is padded to ``head * tp``.
        plan = plan_shard("model.layers.0.self_attn.qkv_proj",
                          roles=[("k_proj", rows)], columns=columns,
                          out_partitions=[head], in_size=columns, tp_rank=rank, tp_size=4,
                          input_size=columns, output_size=head * 4, replicas=(2,))
        role = plan.role("k_proj")
        assert role.shards == 2, "two shards for four ranks: this is the replicated case"
        assert (role.lo, role.hi) == (head * (rank // 2), head * (rank // 2) + head)
        out = shard_parsed_roles([("k_proj", parsed)], plan)
        assert len(out) == 1 and out[0][0] == "k_proj"
        got.append(_decode(out[0][1]))
    for rank, shard in enumerate(got):
        want = whole[head * (rank // 2):head * (rank // 2) + head, :]
        assert torch.equal(shard, want), f"rank {rank} decoded the wrong rows"
    assert torch.equal(got[0], got[1]) and torch.equal(got[2], got[3])
    assert not torch.equal(got[0], got[2])


@needs_cuda
def test_the_seam_hands_back_a_role_this_rank_holds_entire(units):
    """One KV head over four ranks: no cut, and no round trip either.

    ``is_whole`` is not a shortcut bolted on -- a range covering the extent is
    not a cut, and returning the parse untouched is what keeps a replicated
    role free.  Identity is the assertion, because a copy here would be a
    silent per-rank cost on every MQA module.
    """
    parsed = units["e4m3"]
    rows, columns = _unit_extent(parsed)
    plan = plan_shard("qkv", roles=[("q_proj", rows), ("k_proj", rows)], columns=columns,
                      out_partitions=[rows // 4, rows], in_size=columns,
                      tp_rank=2, tp_size=4, input_size=columns,
                      output_size=(rows // 4 + rows) * 4, replicas=(1, 4))
    assert plan.role("k_proj").is_whole and plan.role("q_proj").shards == 4
    out = dict(shard_parsed_roles([("q_proj", parsed), ("k_proj", parsed)], plan))
    assert out["k_proj"] is parsed
    assert torch.equal(_decode(out["q_proj"]),
                       _decode(parsed)[rows // 2:rows // 2 + rows // 4, :])


@needs_cuda
def test_the_seam_refuses_a_cut_the_wire_cannot_express_and_names_the_granularity(units):
    """A mixed rate schedule confines a column cut to whole superblocks.

    512 columns over four ranks is 128 each, and the quota ``sum(rates) == root
    * columns`` is exact only on a 256-column superblock -- so the answer is a
    refusal carrying the number 256, not a silently wrong quota.
    """
    parsed = units["e4m3-mixed"]
    assert shard_granularity(parsed) == (1, 256)
    four = _seam_plan(UNIT_ROWS, UNIT_COLS, AXIS_COLUMNS, rank=1, tp=4, shards=4)
    with pytest.raises(ValueError) as e:
        _shard_unit_for_rank(parsed, four, four.role("weight"))
    msg = str(e.value)
    assert "256" in msg and "column" in msg
    assert "tensor_parallel_size" in msg      # the operator's way out, in the message
    # ... and two ranks, which the granularity does divide, is not refused.
    two = _seam_plan(UNIT_ROWS, UNIT_COLS, AXIS_COLUMNS, rank=1, tp=2, shards=2)
    assert _shard_unit_for_rank(parsed, two, two.role("weight")) is not parsed


@needs_cuda
def test_the_seam_names_the_slicer_when_the_build_cannot_cut(monkeypatch, units):
    """An older Tessera has no ``layout.slice_unit``; the refusal says so.

    This is the branch this whole file used to be about.  It is still real --
    the plugin is installed against whatever ``tessera`` the image carries --
    but it is the exception now, so it is provoked rather than assumed.
    """
    _without_tessera_layout(monkeypatch)
    plan = _seam_plan(UNIT_ROWS, UNIT_COLS, AXIS_ROWS, rank=1, tp=4, shards=4)
    with pytest.raises(NotImplementedError) as e:
        _shard_unit_for_rank(units["e4m3"], plan, plan.role("weight"))
    msg = str(e.value)
    assert "tessera.layout.slice_unit" in msg
    assert "tensor_parallel_size=1" in msg


@needs_cuda
def test_the_seam_refuses_a_cut_with_no_axis(units):
    """``axis=None`` above one rank is a caller bug, not a whole-module shortcut."""
    plan = _seam_plan(UNIT_ROWS, UNIT_COLS, AXIS_ROWS, rank=1, tp=2, shards=2)
    plan = ShardPlan(plan.prefix, plan.rows, plan.columns, plan.shard_rows,
                     plan.shard_columns, plan.tp_rank, plan.tp_size, None, plan.roles)
    with pytest.raises(ValueError, match="no axis"):
        _shard_unit_for_rank(units["e4m3"], plan, plan.role("weight"))


def test_sharding_roles_is_identity_on_one_rank():
    """The path a route took before TP: the same list, same objects."""
    roles = [("q_proj", object()), ("k_proj", object()), ("v_proj", object())]
    plan = plan_shard("qkv", roles=[("q_proj", 512), ("k_proj", 256), ("v_proj", 256)],
                      columns=2048, out_partitions=[512, 256, 256], in_size=2048,
                      tp_rank=0, tp_size=1, input_size=2048, output_size=1024)
    assert shard_parsed_roles(roles, plan) is roles


@needs_cuda
@pytest.mark.parametrize("axis", [AXIS_ROWS, AXIS_COLUMNS])
def test_sharding_roles_cuts_and_re_derives_the_parse(units, axis):
    """The route's own path at tp 2: a list of parses in, this rank's out.

    What comes back must be a ``ParsedUnit`` whose MANIFEST describes the shard.
    The routes read shapes off the parse, and a manifest still naming the
    parent's 64 rows is exactly the kind of stale view that serves without
    complaining.
    """
    from tessera.unit_artifact import ParsedUnit

    parsed = units["e4m3" if axis == AXIS_ROWS else "e2m1x2"]
    whole = _decode(parsed)
    rows, columns = _unit_extent(parsed)
    if axis == AXIS_ROWS:
        plan = plan_shard("mlp.gate_up", roles=[("weight", rows)], columns=columns,
                          out_partitions=[rows // 2], in_size=columns, tp_rank=1, tp_size=2,
                          input_size=columns, output_size=rows)
    else:
        plan = plan_shard("mlp.down_proj", roles=[("weight", rows)], columns=columns,
                          out_partitions=[rows], in_size=columns // 2, tp_rank=1, tp_size=2,
                          input_size=columns, output_size=rows)
    assert plan.axis == axis
    out = shard_parsed_roles([("weight", parsed)], plan)
    assert len(out) == 1 and out[0][0] == "weight"
    shard = out[0][1]
    assert isinstance(shard, ParsedUnit)
    geometry = shard.manifest.geometry
    assert (geometry.rows, geometry.columns) == (plan.shard_rows, plan.shard_columns)
    record = shard.manifest.shard
    assert record is not None
    assert record.parent_digest == parsed.manifest.manifest_digest()
    want = whole[rows // 2:, :] if axis == AXIS_ROWS else whole[:, columns // 2:]
    assert torch.equal(_decode(shard), want)


#: Artifacts written by master ``da2b371`` -- a different encoder from this
#: tree's, which is what makes them the right parent for a provenance test.
LEGACY = pathlib.Path(__file__).parent / "data" / "legacy"


def _legacy(name):
    from tessera.unit_artifact import parse_unit_artifact

    return parse_unit_artifact((LEGACY / f"{name}.tessera").read_bytes(), device="cpu")


def test_a_shard_keeps_its_parent_encoder_identity_and_never_asks_the_current_one(monkeypatch):
    """Cutting restricts planes.  It does not encode, so it may not re-attest.

    ``_reparse_shard`` writes the shard and reads it back, and it wrote it at
    ``build_unit_artifact``'s default ``fixture_id`` -- which asks
    ``encoder_identity.stamped_fixture_id`` for the digest of what THIS
    encoder does.  The shard then named an encoder that never produced its
    bytes, which is false for every artifact any other encoder wrote, and it
    made a load compute that identity (a cold fixture set) to attest bytes it
    had no part in.

    The parent is a committed minor-6 artifact, so its stamp really is another
    encoder's.  The current stamp is made to RAISE rather than to differ: a
    reparse that asks the question at all is the defect, not only one that
    gets a different answer back.

    Both directions are checked, because both are a false claim: a tagged
    parent keeps its own id, and an untagged one is not promoted to a current
    one it never carried.
    """
    from tessera import encoder_identity
    from tessera.layout import slice_unit
    from tessera.serving.sharding import _reparse_shard
    from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact

    parsed = _legacy("e4m3-1024-window-channel-256c")
    manifest = parsed.manifest
    assert manifest.encoder_fixture_id is not None, "the parent must carry an identity"

    def refuse():
        raise AssertionError(
            "a shard reparse asked the current encoder to attest bytes it did not produce")

    monkeypatch.setattr(encoder_identity, "stamped_fixture_id", refuse)
    rows = manifest.geometry.rows
    shard = _reparse_shard(parsed, slice_unit(parsed, rows=(rows // 2, rows)), "rank1")
    assert shard.manifest.encoder_fixture_id == manifest.encoder_fixture_id
    assert shard.manifest.shard is not None
    assert shard.manifest.shard.row_offset == rows // 2

    # ``fixture_id=None`` writes no field and asks nothing, so an untagged
    # parent can be built with the stamp still refusing.
    _m, _region, blob = build_unit_artifact(
        parsed.unit, "untagged", parsed.forests, int(manifest.branch.root_q256),
        parsed.code, superblock=int(manifest.geometry.superblock_columns),
        container=manifest.branch.container, fixture_id=None)
    plain = parse_unit_artifact(blob, device="cpu")
    assert plain.manifest.encoder_fixture_id is None
    untagged = _reparse_shard(plain, slice_unit(plain, rows=(rows // 2, rows)), "rank1")
    assert untagged.manifest.encoder_fixture_id is None


@needs_cuda
def test_granularity_is_read_off_the_wire_not_guessed(units):
    """``tessera.layout`` derives these from the body's own packing: arity x
    span under TCQ, one under the window body; the scale block along columns,
    raised to the superblock by a mixed schedule.  Pinned here so a change to
    either has to be a deliberate one."""
    assert shard_granularity(units["e4m3"]) == (1, 1)          # window, CHANNEL plane
    assert shard_granularity(units["e2m1x2"]) == (4, 16)       # arity 2 x span 2, LUT half 16
    assert shard_granularity(units["e4m3-mixed"]) == (1, 256)  # mixed rates -> superblock


def test_granularity_is_none_and_the_check_a_no_op_without_the_cutter(monkeypatch):
    """No ``tessera.layout``: the answer is None, and the check is a no-op.

    The refusal for a build that cannot cut belongs at the seam, in one place.
    A granularity check that invented a number would refuse the wrong splits.
    """
    _without_tessera_layout(monkeypatch)
    sentinel = object()
    assert shard_granularity(sentinel) is None
    plan = plan_shard("m", roles=[("weight", 1024)], columns=2048, out_partitions=[256],
                      in_size=2048, tp_rank=0, tp_size=4, input_size=2048, output_size=1024)
    check_shard_granularity(plan, plan.role("weight"), sentinel)     # no raise


@needs_cuda
def test_granularity_refusal_names_the_granularity(monkeypatch, units):
    """When the cutter is there, an indivisible split says what it needed."""
    import tessera.serving.sharding as sharding

    parsed = units["e4m3"]
    monkeypatch.setattr(sharding, "shard_granularity", lambda unit: (24, 40))
    plan = _seam_plan(UNIT_ROWS, UNIT_COLS, AXIS_ROWS, rank=0, tp=4, shards=4)
    with pytest.raises(ValueError) as e:
        sharding.check_shard_granularity(plan, plan.role("weight"), parsed)
    assert "24" in str(e.value) and "16" in str(e.value)

    plan = _seam_plan(UNIT_ROWS, 512, AXIS_COLUMNS, rank=0, tp=4, shards=4)
    with pytest.raises(ValueError) as e:
        sharding.check_shard_granularity(plan, plan.role("weight"), parsed)
    assert "40" in str(e.value) and "128" in str(e.value)


def test_a_prepared_role_carries_an_initial_state_and_refuses_to_decode_it():
    """The representation carries the sliced-unit plane; the span-2 decoder
    refuses it.  A ROW shard of a TCQ unit is the one cut this route cannot
    serve, and it is refused rather than decoded against a pinned zero start."""
    torch = pytest.importorskip("torch")
    from tessera.serving import ops

    plane = torch.zeros(4, dtype=torch.uint8)
    role = ops._PreparedRole("q", 0, (plane,) * 7, dict(rows=1, cols=16, rate=1, arity=2,
                                                        memory=1, half=0))
    assert role.initial_state is None
    assert role.tensors() == role.planes

    state = torch.zeros(16, dtype=torch.int32)
    role = ops._PreparedRole("q", 0, (plane,) * 7, dict(rows=1, cols=16, rate=1, arity=2,
                                                        memory=1, half=0), state)
    assert role.initial_state is state
    assert role.tensors()[-1] is state          # fingerprinted with the rest

    with pytest.raises(NotImplementedError, match="INITIAL_STATE"):
        ops.PreparedTesseraModule._require_no_initial_state(_FakeModule([role]))


def test_what_an_artifact_says_about_slicing_is_read_off_it_not_assumed():
    """``artifact_tp_agnostic`` -- the config half of the TP gate (tessera#328).

    Three answers, and ``None`` is one of them: a config that does not say is
    not a config that says yes.  The derived spelling and the fact it derives
    from resolve through the SAME rule (``layout.tp_agnostic_at_minor``), which
    is why the minor below is written as an offset from the cutter's own
    constant rather than as a literal 3 -- pinning the number here would make
    this file a second home for the rule.
    """
    from tessera.layout import SLICEABLE_SCHEMA_MINOR
    from tessera.serving.sharding import artifact_tp_agnostic

    assert artifact_tp_agnostic({"tp_agnostic": True}) is True
    assert artifact_tp_agnostic({"tp_agnostic": False}) is False
    assert artifact_tp_agnostic({"schema_minor": SLICEABLE_SCHEMA_MINOR}) is True
    assert artifact_tp_agnostic({"schema_minor": SLICEABLE_SCHEMA_MINOR - 1}) is False
    # The declaration wins over the fact: it is the artifact's own word.
    assert artifact_tp_agnostic(
        {"tp_agnostic": False, "schema_minor": SLICEABLE_SCHEMA_MINOR}) is False
    # No answer, and the stale stamp is not one either.
    assert artifact_tp_agnostic({}) is None
    assert artifact_tp_agnostic({"tp_size": 1}) is None
    assert artifact_tp_agnostic(None) is None


def test_the_artifact_gate_passes_at_one_rank_whatever_the_checkpoint_says():
    """``world <= 1`` cuts nothing, so there is nothing to refuse.

    The gate must not turn every legacy checkpoint into a load failure on the
    only world size this plugin has ever served.
    """
    from tessera.serving.sharding import require_a_cuttable_artifact

    for config in ({}, {"tp_agnostic": False}, {"tp_size": 1}, None):
        require_a_cuttable_artifact("m", 1, config)          # no raise
        require_a_cuttable_artifact("m", 0, config)          # nor a bare test build
    with pytest.raises(ValueError, match="tensor_parallel_size=2"):
        require_a_cuttable_artifact("m", 2, {"tp_agnostic": False})


class _FakeModule:
    """Just enough of PreparedTesseraModule for the refusal, whose only input is the roles."""

    def __init__(self, roles):
        setattr(self, "_PreparedTesseraModule__roles", tuple(roles))


def test_the_window_carries_a_start_state_through_the_pad_not_a_refusal():
    """The window body needs NO decode change to serve a shard.

    ``pack_window_planes`` prepends ``window_bits`` pad bits per column and the
    pad IS ``state_{-1}``: the read at ``(t+1)*R`` for ``t = 0`` yields
    ``(init << R | bits_0) mod 2^L``.  So a start state is THREADED into the
    packing, and ``decode()`` is unchanged -- refusing it here would have
    refused the one lane the TP design says is already done.
    """
    torch = pytest.importorskip("torch")
    from tessera.serving.window import prepare_window

    torch.manual_seed(0)
    steps, cols, L = 8, 4, 6
    body = torch.randint(0, 2, (steps, cols), dtype=torch.uint8)
    table = torch.arange(1 << L, dtype=torch.int64) % 16
    w = prepare_window(body, [1] * cols, L, table, "cpu")
    assert w.initial_state is None, "a whole unit's pad is the pinned zero"
    assert w.decode().shape == (steps, cols)


def test_a_start_state_is_refused_only_when_the_packer_cannot_take_it(monkeypatch):
    """The refusal names the INSTALLED packer, not the concept.

    Packing a shard against a zero pad would decode to plausible wrong weights
    in silence, so an older ``lane_planes`` must fail closed -- but the message
    has to say that it is the build that is old, not that shards are unservable.
    """
    torch = pytest.importorskip("torch")
    import tessera.lane_planes as lane_planes
    from tessera.serving.window import prepare_window

    real = lane_planes.pack_window_planes

    def old_signature(body_bits, rates, window_bits):
        return real(body_bits, rates, window_bits)

    monkeypatch.setattr(lane_planes, "pack_window_planes", old_signature)
    steps, cols, L = 8, 4, 6
    body = torch.randint(0, 2, (steps, cols), dtype=torch.uint8)
    table = torch.arange(1 << L, dtype=torch.int64) % 16
    with pytest.raises(NotImplementedError, match="takes no initial_state"):
        prepare_window(body, [1] * cols, L, table, "cpu",
                       initial_state=torch.zeros(cols, dtype=torch.int32))


def test_the_seam_speaks_tessera_layouts_axis_vocabulary():
    """``AXIS_*`` are passed straight to ``layout.can_shard``; a near-miss is a bug."""
    from tessera.serving.sharding import AXIS_COLUMNS, AXIS_ROWS

    assert (AXIS_ROWS, AXIS_COLUMNS) == ("row", "column")


def test_an_even_split_is_equal_parts_and_an_uneven_one_is_refused():
    """What ``_bounds`` used to assert, now asserted of the planner.

    The even split is no longer a separate function: a role's range comes off
    the sizes vLLM asked for, and equal parts are what those sizes describe
    when nothing is replicated.  An extent that does not divide is refused by
    the planner -- ``_bounds`` could only say the arithmetic failed.  Since
    tessera#303 the refusal is phrased against the LAYER: a layer declaring
    1023 output rows whose two ranks each ask for 511 states no vLLM Linear
    relation (vLLM's own ``divide`` would have refused it first), and a layer
    declaring 1022 over a 1023-row wire leaves one row served by nobody -- and
    that one names the role.
    """
    for rank, want in ((0, (0, 256)), (3, (768, 1024))):
        plan = _plan(1024, 2048, out=256, in_=2048, rank=rank, tp=4)
        role = plan.role("weight")
        assert (role.lo, role.hi) == want and role.shards == 4
    with pytest.raises(ValueError) as excinfo:
        _plan(1023, 2048, out=511, in_=2048, rank=0, tp=2)
    msg = str(excinfo.value)
    assert "1023x2048" in msg and "511x2048" in msg and "none of the three" in msg
    with pytest.raises(ValueError) as excinfo:
        _plan(1023, 2048, out=511, in_=2048, rank=0, tp=2, output_size=1022)
    msg = str(excinfo.value)
    assert "'weight'" in msg and "1023 rows on the wire" in msg and "2 shards of 511" in msg
    assert "rows [1022, 1023) would be served by nobody" in msg


def test_the_pad_really_is_state_minus_one():
    """The pad-threading CLAIM, measured.

    ``window._pack`` threads a shard's start state into ``pack_window_planes``'
    L-bit pad on the argument that the pad *is* ``state_{-1}``.  That is an
    arithmetic claim about the wire, and it is checked here rather than
    asserted: a whole unit's decode from row ``t0`` down must equal the decode
    of the unit's rows ``t0:`` packed with the parent's state at the cut.  The
    identity table makes the decoded value the raw L-bit window, so a threading
    error shows up directly instead of through an alphabet, and the zero-pad
    control fails, so the check cannot pass for the wrong reason.

    This used to skip on a build whose ``lane_planes`` predated the
    ``initial_state`` parameter.  The slicer has landed, so a build without it
    is a REGRESSION and is failed rather than skipped past.
    """
    torch = pytest.importorskip("torch")
    import inspect
    import tessera.lane_planes as lane_planes
    from tessera.serving.window import prepare_window

    assert "initial_state" in inspect.signature(lane_planes.pack_window_planes).parameters, (
        "lane_planes.pack_window_planes has lost its initial_state parameter; the TP seam "
        "cuts row shards that only the pad can start")

    L, cols, steps = 6, 5, 24
    torch.manual_seed(7)
    body = torch.randint(0, 2, (steps, cols), dtype=torch.uint8)
    table = torch.arange(1 << L, dtype=torch.int64)     # identity: decode == window
    rates = [1] * cols
    full = prepare_window(body, rates, L, table, "cpu").decode()

    for t0 in (1, 3, 6, 7, 12, 17):
        init = torch.zeros(cols, dtype=torch.int64)
        for c in range(cols):
            v = 0
            for i in range(t0 - L, t0):
                v = (v << 1) | (int(body[i, c]) if i >= 0 else 0)
            init[c] = v
        shard = prepare_window(body[t0:], rates, L, table, "cpu", initial_state=init).decode()
        assert torch.equal(shard, full[t0:]), f"threading wrong at t0={t0}"

    zeroed = prepare_window(body[7:], rates, L, table, "cpu").decode()
    assert not torch.equal(zeroed, full[7:]), "control: a zero pad must NOT match"


@needs_cuda
def test_the_fp8_route_serves_a_row_shard_as_the_parents_own_rows(units):
    """The composition test: seam -> FP8 route -> tile, against the parent tile.

    The seam and the route are each proven above and in
    ``tests/test_serving_fp8_route.py``, but the interesting failure lives
    between them.  ``prepare_tessera_fp8_module`` threads ``initial_state``
    into ``prepare_window``; if ``tessera.decode.materialize_fp8`` did NOT read
    the same field, the route's own ``torch.equal`` self-check would *refuse* a
    legitimate row shard, and the FP8 family would silently be column-cuts-only
    -- exactly the restriction the NVFP4 lane has and this one claims not to.

    So the assertion is not "it loaded": it is that the bytes the route hands
    the fused kernel, and the per-row scales beside them, are the parent tile's
    own rows for BOTH ranks of a tp=2 cut.
    """
    from tessera.decode import materialize_fp8
    from tessera.serving.fp8_route import prepare_tessera_fp8_module

    parent = units["e4m3"]
    ref_bytes, ref_scale = materialize_fp8(parent.unit, parent.forests, parent.code)
    ref_bytes = ref_bytes.to("cuda")
    ref_scale = ref_scale.to("cuda", torch.float32).reshape(-1)
    half = UNIT_ROWS // 2

    for rank in (0, 1):
        plan = _plan(UNIT_ROWS, UNIT_COLS, out=half, in_=UNIT_COLS, rank=rank, tp=2)
        assert plan.axis == AXIS_ROWS
        roles = shard_parsed_roles([("weight", parent)], plan)
        module = prepare_tessera_fp8_module(roles, device="cuda")

        got = module.decode()
        want = ref_bytes[rank * half:(rank + 1) * half]
        assert got.shape == want.shape, f"rank {rank} served the wrong shape"
        assert torch.equal(got, want), (
            f"rank {rank} of 2 served {int((got != want).sum())} of {want.numel()} "
            "bytes that are not the parent's -- the window pad is not being threaded "
            "through the FP8 route")
        assert torch.equal(module.row_scale(),
                           ref_scale[rank * half:(rank + 1) * half]), (
            f"rank {rank} of 2 got a row scale that is not the parent's")
