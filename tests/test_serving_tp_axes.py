"""Which shard axis each route serves, decided at ``create_weights``.

The seam that cuts a unit is built and proven elsewhere
(``tests/test_slice_unit.py``, ``tests/test_serving_sharding.py``): a shard
decodes to exactly the parent's slice, on both axes, on both shipping wires.
What this file is about is the half a *serve* needs and a slicer cannot supply
-- **who is allowed to ask for a cut**, and how loudly the answer arrives.

Two facts drive every test here:

1. **The row axis is the BODY's answer, not the tile's.**  A row shard begins
   mid-column, so it carries an INITIAL_STATE plane.  The window body's L-bit
   pad *is* ``state_{-1}``, so the E4M3/FP8 route threads it and cuts rows; the
   span-2 TCQ decoders the NVFP4 route packs for supply ``state_{-1} = 0``
   themselves (``lane_planes.pack_unit_for_kernel``, ``csrc/window_gemv.cu``),
   so that route cuts columns only.
2. **The refusal is symmetric across the group.**  Rank 0's row shard starts at
   row 0 and carries no INITIAL_STATE plane, so it would in fact pack.
   Refusing only where it bites would leave rank 0 building a layer while its
   peers raised, and a TP group whose ranks disagree about whether a module
   exists does not fail -- it hangs on the first collective.  So the NVFP4 row
   cut is refused on *every* rank, including rank 0.

Before this file, both of those were true of the packer and of nothing else:
``create_weights`` planned a row cut happily and the refusal arrived from
inside ``pack_unit_for_kernel``, after the blob was loaded, naming a row offset
rather than a ``tensor_parallel_size``.

CPU-only by construction.  ``create_weights`` registers parameters and computes
a shard plan; it decodes nothing, so the whole gate is testable while a GPU
measurement is in flight.
"""
from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from tessera.serving import lane as serving_lane                     # noqa: E402
from tessera.serving.lane import TESSERA_MODE_ENV, build_tessera_method  # noqa: E402
from tessera.serving.scheme import TESSERA_BF16, TESSERA_FP8, TESSERA_NVFP4        # noqa: E402
from tessera.serving.sharding import (                               # noqa: E402
    AXIS_COLUMNS, AXIS_ROWS, ROUTE_TP_AXES, ShardPlan, TP_REFUSED, TP_SHARDED,
    axis_status, require_axis_supported)

ROWS, COLUMNS = 256, 1024


@pytest.fixture(autouse=True)
def _fresh_env(monkeypatch):
    serving_lane.reset_for_tests()
    monkeypatch.delenv(TESSERA_MODE_ENV, raising=False)
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    yield
    serving_lane.reset_for_tests()


def _stub_vllm(monkeypatch, *, tp_rank=0, tp_size=1):
    """vLLM's linear/parameter surface, plus a parallel state that answers.

    The parallel state is stubbed so the PROCESS-GLOBAL coordinates exist and
    can be made to disagree with a layer's own: since tessera#303 the plan
    reads ``layer.tp_rank``/``layer.tp_size`` -- what every vLLM ``LinearBase``
    sets before ``create_weights``, ``(0, 1)`` under ``disable_tp`` -- and a
    test below pins that the layer's answer wins over this one.
    """
    class _LinearMethodBase:
        pass

    def _param(data, **_kw):
        return torch.nn.Parameter(data, requires_grad=False)

    linear = types.ModuleType("vllm.model_executor.layers.linear")
    linear.LinearMethodBase = _LinearMethodBase
    parameter = types.ModuleType("vllm.model_executor.parameter")
    parameter.BasevLLMParameter = _param
    parameter.ModelWeightParameter = _param
    distributed = types.ModuleType("vllm.distributed")
    distributed.get_tensor_model_parallel_rank = lambda: tp_rank
    distributed.get_tensor_model_parallel_world_size = lambda: tp_size
    for name, mod in (("vllm", types.ModuleType("vllm")),
                      ("vllm.model_executor", types.ModuleType("vllm.model_executor")),
                      ("vllm.model_executor.layers",
                       types.ModuleType("vllm.model_executor.layers")),
                      ("vllm.model_executor.layers.linear", linear),
                      ("vllm.model_executor.parameter", parameter),
                      ("vllm.distributed", distributed)):
        monkeypatch.setitem(sys.modules, name, mod)


class _Layer(torch.nn.Module):
    """A vLLM ``LinearBase`` stand-in: the layer's OWN TP coordinates.

    Every ``LinearBase`` sets ``tp_rank``/``tp_size`` in its ``__init__`` before
    it calls ``create_weights`` -- ``(0, 1)`` under ``disable_tp``, the group's
    otherwise -- and this is the surface the plan reads them from.
    """

    def __init__(self, tp_rank=0, tp_size=1):
        super().__init__()
        self.tp_rank, self.tp_size = tp_rank, tp_size


class _QKVLayer(_Layer):
    """``QKVParallelLinear``'s statement of KV replication, on the layer."""

    def __init__(self, tp_rank, tp_size, num_kv_head_replicas):
        super().__init__(tp_rank, tp_size)
        self.num_kv_head_replicas = num_kv_head_replicas


class _BareModule(torch.nn.Module):
    """Not a vLLM Linear at all: no TP identity to read."""


def _nvfp4_scheme(roles=None, columns=COLUMNS):
    roles = roles if roles is not None else [["weight", ROWS]]
    return {"family": TESSERA_NVFP4, "grid": "E2M1x2", "body": "TCQ", "plane": "LUT",
            "q256": 896, "rows": sum(r for _, r in roles), "columns": columns,
            "wire_bytes": 4096, "roles": roles}


def _fp8_scheme(roles=None, columns=COLUMNS):
    roles = roles if roles is not None else [["weight", ROWS]]
    return {"family": TESSERA_FP8, "grid": "E4M3", "body": "WINDOW", "plane": "CHANNEL",
            "q256": 1024, "rows": sum(r for _, r in roles), "columns": columns,
            "wire_bytes": 4096, "roles": roles}


def _bf16_scheme(roles=None, columns=COLUMNS):
    roles = roles if roles is not None else [["weight", ROWS]]
    return {"family": TESSERA_BF16, "grid": "BF16", "body": "WINDOW", "plane": "CHANNEL",
            "q256": 1792, "rows": sum(r for _, r in roles), "columns": columns,
            "wire_bytes": 4096, "roles": roles}


def _create(monkeypatch, scheme, *, axis, tp_size=2, tp_rank=0):
    """Drive ``create_weights`` as the named parallel Linear would.

    ``ColumnParallelLinear`` and its merged/QKV forms split the OUTPUT, so each
    rank asks for ``rows/tp`` outputs over the whole input; ``RowParallelLinear``
    splits the INPUT.  The axis is derived from exactly these numbers and the
    layer's own ``input_size``/``output_size`` -- the plugin never sniffs a
    class name -- so this is the whole difference between the two cases.
    """
    _stub_vllm(monkeypatch, tp_rank=tp_rank, tp_size=tp_size)
    method = build_tessera_method(scheme, "model.layers.0.test")
    layer = _Layer(tp_rank, tp_size)
    rows, columns = scheme["rows"], scheme["columns"]
    if axis == AXIS_ROWS:
        partitions = [r // tp_size for _, r in scheme["roles"]]
        in_size = columns
    else:
        partitions = [r for _, r in scheme["roles"]]
        in_size = columns // tp_size
    method.create_weights(layer, input_size_per_partition=in_size,
                          output_partition_sizes=partitions,
                          input_size=columns, output_size=rows,
                          params_dtype=torch.bfloat16)
    return layer


# --- the gate itself ---------------------------------------------------------

def test_the_two_routes_publish_different_axes():
    """A pin on the table both the routes and the contract read."""
    assert ROUTE_TP_AXES[TESSERA_FP8] == {AXIS_ROWS: TP_SHARDED, AXIS_COLUMNS: TP_SHARDED}
    assert ROUTE_TP_AXES[TESSERA_NVFP4] == {AXIS_ROWS: TP_REFUSED, AXIS_COLUMNS: TP_SHARDED}
    assert ROUTE_TP_AXES[TESSERA_BF16] == {AXIS_ROWS: TP_SHARDED, AXIS_COLUMNS: TP_SHARDED}
    assert axis_status(TESSERA_NVFP4, AXIS_ROWS) == TP_REFUSED
    # BF16 has FP8's answer because it has FP8's body, not because it is new.
    assert ROUTE_TP_AXES[TESSERA_BF16] == ROUTE_TP_AXES[TESSERA_FP8]


def test_an_unknown_route_or_axis_is_a_refusal_not_a_default():
    """Guessing ``sharded`` for a route this table never heard of is exactly
    the silent-wrong-rows outcome the seam exists to prevent."""
    # Not a family this project has plans for: ``TESSERA_BF16`` stood here
    # until it became a route, which is the point -- the refusal has to be for
    # names the table does not carry, not for names it does not carry YET.
    with pytest.raises(ValueError, match="ROUTE_TP_AXES covers"):
        axis_status("TESSERA_NOT_A_FAMILY", AXIS_ROWS)
    with pytest.raises(ValueError, match="is not a shard axis"):
        axis_status(TESSERA_FP8, "rows")


def test_a_module_served_whole_is_never_gated():
    """``axis is None`` is a replicated Linear or a one-rank world: nothing is
    cut, so no route's axis answer applies to it."""
    whole = ShardPlan("t", ROWS, COLUMNS, ROWS, COLUMNS, 0, 4, None)
    require_axis_supported(TESSERA_NVFP4, whole)          # does not raise


# --- what create_weights does with each axis ---------------------------------

def test_the_nvfp4_route_refuses_a_row_cut_by_name(monkeypatch):
    """FAILS BEFORE: ``create_weights`` planned this cut and returned, and the
    refusal arrived later from ``pack_unit_for_kernel`` -- after the blob was
    loaded, naming a row offset rather than a ``tensor_parallel_size``."""
    with pytest.raises(ValueError) as excinfo:
        _create(monkeypatch, _nvfp4_scheme(), axis=AXIS_ROWS, tp_size=2, tp_rank=0)
    message = str(excinfo.value)
    assert "TESSERA_NVFP4 does not serve a row cut" in message
    assert "tensor_parallel_size=2" in message
    assert "ColumnParallelLinear" in message and "QKVParallelLinear" in message
    assert "INITIAL_STATE" in message
    assert "TESSERA_FP8" in message          # the route that does cut rows


def test_the_nvfp4_row_refusal_is_symmetric_across_the_group(monkeypatch):
    """FAILS BEFORE, and it is the half that matters.

    Rank 0's row shard starts at row 0, carries no INITIAL_STATE plane and
    would pack.  If only the other ranks refused, rank 0 would build the layer
    and the group would hang on its first collective instead of failing.
    """
    for rank in (0, 1, 3):
        with pytest.raises(ValueError, match="does not serve a row cut"):
            _create(monkeypatch, _nvfp4_scheme(), axis=AXIS_ROWS, tp_size=4, tp_rank=rank)


def test_the_nvfp4_route_takes_a_column_cut(monkeypatch):
    """REGRESSION PIN (passes before and after): a row-parallel Linear
    (``o_proj``, ``down_proj``) cuts the input axis, which needs no start
    state, so the span-2 route serves it at any TP."""
    layer = _create(monkeypatch, _nvfp4_scheme(), axis=AXIS_COLUMNS, tp_size=2)
    assert layer.tessera_shard_plan.axis == AXIS_COLUMNS
    assert layer.tessera_rows == ROWS and layer.tessera_columns == COLUMNS // 2
    assert layer.tessera_groups == COLUMNS // 2 // 16


@pytest.mark.parametrize("axis, want", [
    (AXIS_ROWS, (ROWS // 2, COLUMNS)),
    (AXIS_COLUMNS, (ROWS, COLUMNS // 2)),
])
def test_the_fp8_route_takes_both_axes(monkeypatch, axis, want):
    """REGRESSION PIN (passes before and after): the window body's pad IS
    ``state_{-1}``, so this route is the cheapest thing in the format to
    shard -- and it must stay reachable now that the gate exists."""
    layer = _create(monkeypatch, _fp8_scheme(), axis=axis, tp_size=2)
    assert layer.tessera_shard_plan.axis == axis
    assert (layer.tessera_rows, layer.tessera_columns) == want


def test_a_fused_nvfp4_module_is_refused_on_the_axis_vllm_would_split(monkeypatch):
    """FAILS BEFORE.  q/k/v is the row-cut case in practice: vLLM gives every
    rank its own rows of each role, which is exactly the cut this route cannot
    start."""
    scheme = _nvfp4_scheme(roles=[["q_proj", 128], ["k_proj", 64], ["v_proj", 64]])
    with pytest.raises(ValueError, match="does not serve a row cut"):
        _create(monkeypatch, scheme, axis=AXIS_ROWS, tp_size=2)


def test_one_rank_is_untouched_on_both_routes(monkeypatch):
    """REGRESSION PIN.  At ``tp_size == 1`` the plan is the whole module and no
    axis answer is consulted, so a build with this gate serves exactly what a
    build without it served."""
    for scheme in (_nvfp4_scheme(), _fp8_scheme()):
        layer = _create(monkeypatch, scheme, axis=AXIS_ROWS, tp_size=1)
        assert layer.tessera_shard_plan.is_whole
        assert (layer.tessera_rows, layer.tessera_columns) == (ROWS, COLUMNS)


# --- the LAYER's contract, not the tile's shape (tessera#303) ------------------
#
# ``create_weights`` is handed four numbers and the layer: the tile this rank
# computes (``input_size_per_partition``, ``output_partition_sizes``), the
# module's GLOBAL shape (``input_size``, ``output_size``) and, on the layer
# itself, its own ``tp_rank``/``tp_size`` and -- on ``QKVParallelLinear`` --
# ``num_kv_head_replicas``.  Before #303 the plan read the tile and a
# process-global rank and threw the rest away, so a wire the size of one rank's
# TILE was indistinguishable from a REPLICATED whole module: both are "the
# layer asks for exactly what the wire has".  The tests here feed the numbers
# a real vLLM Linear supplies and hold the plan to what the layer declared.

_BUILDERS = {
    TESSERA_FP8: _fp8_scheme,
    TESSERA_BF16: _bf16_scheme,
    TESSERA_NVFP4: _nvfp4_scheme,
}


def _create_for_layer(monkeypatch, scheme, layer, *, in_size, partitions, input_size,
                      output_size, prefix="model.layers.0.self_attn.qkv_proj"):
    """``create_weights`` with EXACTLY what a vLLM Linear passes, plus the layer.

    The process-global parallel state is stubbed to AGREE with the layer, so
    the pins below pass before the fix for the right reason and the refusals
    fail before it for the right reason -- the shape, not the coordinates.
    """
    _stub_vllm(monkeypatch, tp_rank=getattr(layer, "tp_rank", 0),
               tp_size=getattr(layer, "tp_size", 1))
    method = build_tessera_method(scheme, prefix)
    method.create_weights(layer, input_size_per_partition=in_size,
                          output_partition_sizes=list(partitions),
                          input_size=input_size, output_size=output_size,
                          params_dtype=torch.bfloat16)
    return layer


@pytest.mark.parametrize("family", [TESSERA_FP8, TESSERA_BF16])
def test_an_undersized_whole_wire_for_a_column_parallel_layer_is_refused_on_every_rank(
        monkeypatch, family):
    """FAILS BEFORE -- the tessera#303 reproduction, on the real ``create_weights``.

    A ``ColumnParallelLinear`` at TP=4 over a ``[256, 64]`` weight asks every
    rank for a ``[64, 64]`` tile, and says so: ``output_size=256``,
    ``input_size=64``, ``layer.tp_size=4``.  The checkpoint declares a whole
    ``[64, 64]`` wire.  Before the fix the plan compared the wire with the TILE
    only, found them equal, and registered a replicated whole module on all
    four ranks -- every rank serving rows ``[0, 64)`` of a 256-row layer
    (observed: ``0 None 0 64``, ``1 None 0 64``, ...).  The wire is a quarter of
    the module and no rank holds the other three quarters.

    It is refused on EVERY rank, by the role's name, with the wire's rows and
    the extent the layer's contract requires -- and before a loadable
    parameter exists, so a weight loader never gets to fill it.
    """
    scheme = _BUILDERS[family](roles=[["weight", 64]], columns=64)
    for rank in range(4):
        layer = _Layer(rank, 4)
        with pytest.raises(ValueError) as excinfo:
            _create_for_layer(monkeypatch, scheme, layer, in_size=64, partitions=[64],
                              input_size=64, output_size=256, prefix="linear")
        message = str(excinfo.value)
        assert "'weight'" in message, "the refusal names the role"
        assert "64 rows on the wire" in message
        assert "256" in message and "4 shards of 64" in message
        assert f"rank {rank} of 4" in message
        assert "wire_bytes" not in dict(layer.named_parameters()), \
            "nothing loadable is registered on a refused layer"
        assert not hasattr(layer, "tessera_shard_plan")


@pytest.mark.parametrize("family", [TESSERA_FP8, TESSERA_BF16])
def test_a_correctly_shaped_column_parallel_layer_gives_each_rank_its_own_rows(
        monkeypatch, family):
    """REGRESSION PIN (passes before and after): the CONTROL for the case above.

    The same layer over a wire that IS ``[256, 64]``: rank *r* owns rows
    ``[64r, 64r + 64)``, four distinct shards, and the tile is 64x64.
    """
    scheme = _BUILDERS[family](roles=[["weight", 256]], columns=64)
    for rank in range(4):
        layer = _create_for_layer(monkeypatch, scheme, _Layer(rank, 4), in_size=64,
                                  partitions=[64], input_size=64, output_size=256,
                                  prefix="linear")
        plan = layer.tessera_shard_plan
        assert plan.axis == AXIS_ROWS and (plan.tp_rank, plan.tp_size) == (rank, 4)
        role = plan.role("weight")
        assert (role.lo, role.hi, role.shards) == (64 * rank, 64 * rank + 64, 4)
        assert (layer.tessera_rows, layer.tessera_columns) == (64, 64)


def test_an_undersized_whole_wire_for_a_row_parallel_layer_is_refused_on_every_rank(monkeypatch):
    """FAILS BEFORE.  The input-axis twin of the case above.

    A ``RowParallelLinear`` at TP=4 over ``[64, 256]`` asks each rank for a
    ``[64, 64]`` tile (``input_size=256``); a whole ``[64, 64]`` wire is one
    rank's quarter of the input, not a replicated module.
    """
    scheme = _fp8_scheme(roles=[["weight", 64]], columns=64)
    for rank in range(4):
        with pytest.raises(ValueError) as excinfo:
            _create_for_layer(monkeypatch, scheme, _Layer(rank, 4), in_size=64,
                              partitions=[64], input_size=256, output_size=64, prefix="linear")
        message = str(excinfo.value)
        assert "64 columns" in message and "256" in message
        assert f"rank {rank} of 4" in message
    # ... and the control: a [64, 256] wire under the same layer is a column cut.
    control = _fp8_scheme(roles=[["weight", 64]], columns=256)
    layer = _create_for_layer(monkeypatch, control, _Layer(3, 4), in_size=64,
                              partitions=[64], input_size=256, output_size=64, prefix="linear")
    assert layer.tessera_shard_plan.axis == AXIS_COLUMNS
    assert (layer.tessera_shard_plan.role("weight").lo,
            layer.tessera_shard_plan.role("weight").hi) == (192, 256)


def test_a_replicated_linear_inside_a_group_is_served_whole_on_every_rank(monkeypatch):
    """REGRESSION PIN.  ``ReplicatedLinear`` at TP=4 asks for the whole shape
    and SAYS it is the whole shape (``output_size == sum(output_partition_sizes)``,
    ``input_size == input_size_per_partition``) -- so a whole wire is exactly
    right, on every rank, and nothing is cut."""
    scheme = _fp8_scheme(roles=[["weight", ROWS]])
    for rank in range(4):
        layer = _create_for_layer(monkeypatch, scheme, _Layer(rank, 4), in_size=COLUMNS,
                                  partitions=[ROWS], input_size=COLUMNS, output_size=ROWS,
                                  prefix="model.layers.0.replicated")
        plan = layer.tessera_shard_plan
        assert plan.axis is None and (plan.tp_rank, plan.tp_size) == (rank, 4)
        assert plan.role("weight").is_whole
        assert (layer.tessera_rows, layer.tessera_columns) == (ROWS, COLUMNS)


def test_a_disable_tp_linear_is_one_rank_whatever_the_process_says(monkeypatch):
    """FAILS BEFORE.  ``disable_tp=True`` gives a ``LinearBase`` ``tp_rank,
    tp_size = 0, 1`` inside a four-rank process; vLLM reconciles its parameters
    to the LAYER's coordinates for exactly this reason.  The plan used to read
    ``vllm.distributed`` instead and built a four-rank plan for a one-rank layer.
    """
    _stub_vllm(monkeypatch, tp_rank=2, tp_size=4)          # the process is a group
    method = build_tessera_method(_fp8_scheme(), "model.layers.0.no_tp")
    layer = _Layer(0, 1)                                    # the layer is not in it
    method.create_weights(layer, input_size_per_partition=COLUMNS,
                          output_partition_sizes=[ROWS], input_size=COLUMNS,
                          output_size=ROWS, params_dtype=torch.bfloat16)
    plan = layer.tessera_shard_plan
    assert plan.is_whole, "the layer's own coordinates, not the process's"
    assert (plan.tp_rank, plan.tp_size) == (0, 1)


def test_gqa_kv_replication_is_read_off_the_layer_and_each_rank_owns_its_heads(monkeypatch):
    """REGRESSION PIN.  16 query heads, 8 KV heads, head 64, TP=16: vLLM
    replicates KV heads two ranks per head.  Its ``QKVParallelLinear`` asks
    ``[64, 64, 64]`` per rank and publishes the PADDED aggregate
    ``output_size = 3072`` (``[1024, 1024, 1024]``, k and v padded to
    ``num_kv_heads * head * tp``) with ``num_kv_head_replicas = 2``.  The wire
    holds ``[1024, 512, 512]`` -- less than the aggregate -- and that is the
    legitimate case: k's complete extent is ``64 * (16 / 2) = 512``.
    """
    roles = [["q_proj", 16 * 64], ["k_proj", 8 * 64], ["v_proj", 8 * 64]]
    scheme = _fp8_scheme(roles=roles, columns=2048)
    for rank in (0, 1, 5, 15):
        layer = _create_for_layer(monkeypatch, scheme, _QKVLayer(rank, 16, 2), in_size=2048,
                                  partitions=[64, 64, 64], input_size=2048, output_size=3072)
        plan = layer.tessera_shard_plan
        assert plan.axis == AXIS_ROWS and plan.shard_rows == 192
        q, k, v = plan.role("q_proj"), plan.role("k_proj"), plan.role("v_proj")
        assert (q.lo, q.hi, q.shards) == (64 * rank, 64 * rank + 64, 16)
        for member in (k, v):
            assert (member.lo, member.hi, member.shards) == \
                (64 * (rank // 2), 64 * (rank // 2) + 64, 8)


def test_mqa_replicates_the_single_kv_head_whole_and_the_same_numbers_without_it_are_refused(
        monkeypatch):
    """FAILS BEFORE, and it is the acceptance clause of tessera#303.

    MQA: 8 query heads, ONE KV head, head 64, TP=8.  ``QKVParallelLinear`` asks
    ``[64, 64, 64]`` per rank with ``num_kv_head_replicas = 8``; the wire's k
    is 64 rows and every rank holds it entire.  That k -- 64 rows on the wire,
    64 asked for, tp 8 -- is NUMERICALLY IDENTICAL to a plain
    ``ColumnParallelLinear`` over a ``[512, 1024]`` weight handed an undersized
    ``[64, 1024]`` wire.  The difference is the layer's declared replication,
    and nothing else: with it the plan is a whole k, without it the same wire
    is one rank's tile and is refused by name.
    """
    qkv = _fp8_scheme(roles=[["q_proj", 8 * 64], ["k_proj", 64], ["v_proj", 64]], columns=1024)
    for rank in (0, 3, 7):
        layer = _create_for_layer(monkeypatch, qkv, _QKVLayer(rank, 8, 8), in_size=1024,
                                  partitions=[64, 64, 64], input_size=1024, output_size=1536)
        plan = layer.tessera_shard_plan
        assert plan.axis == AXIS_ROWS
        assert plan.role("k_proj").is_whole and plan.role("k_proj").shards == 1
        assert plan.role("v_proj").is_whole
        assert (plan.role("q_proj").lo, plan.role("q_proj").hi) == (64 * rank, 64 * rank + 64)
    # The same k numbers on an ordinary Linear that declares no replication.
    plain = _fp8_scheme(roles=[["weight", 64]], columns=1024)
    with pytest.raises(ValueError) as excinfo:
        _create_for_layer(monkeypatch, plain, _Layer(3, 8), in_size=1024, partitions=[64],
                          input_size=1024, output_size=512, prefix="linear")
    message = str(excinfo.value)
    assert "'weight'" in message and "64 rows on the wire" in message
    assert "512" in message and "8 shards of 64" in message


def test_kv_replication_declared_on_a_layer_with_other_members_is_refused_by_name(monkeypatch):
    """FAILS BEFORE.  ``num_kv_head_replicas`` is ``QKVParallelLinear``'s
    statement about q, k and v -- three members, replication on the last two.
    A layer carrying it over one role (or four: the indexer variant adds an
    ``index_k`` partition cut ``tp`` ways) is not a statement this plan knows
    how to read, and guessing which members it covers is the defect's shape.
    Before the fix the attribute was never read, so a wire of 128 rows over
    ``[64]`` at TP=4 was accepted as "two shards for four ranks" on divisibility
    alone -- with nothing from the runtime saying the role was replicated.
    """
    scheme = _fp8_scheme(roles=[["weight", 128]], columns=1024)
    with pytest.raises(ValueError) as excinfo:
        _create_for_layer(monkeypatch, scheme, _QKVLayer(0, 4, 2), in_size=1024,
                          partitions=[64], input_size=1024, output_size=256, prefix="linear")
    message = str(excinfo.value)
    assert "num_kv_head_replicas" in message and "QKVParallelLinear" in message
    assert "1 role" in message


def test_a_layer_without_tp_coordinates_is_refused_not_defaulted(monkeypatch):
    """FAILS BEFORE.  Every vLLM ``LinearBase`` carries ``tp_rank``/``tp_size``
    and the dense methods are handed nothing else (``TesseraConfig
    .get_quant_method`` checks ``isinstance(layer, LinearBase)``), so a layer
    without them is not a vLLM Linear and its TP identity is not a thing to
    guess.  The plan used to fall back to ``vllm.distributed`` -- a
    process-global answer for a layer-local question."""
    with pytest.raises(ValueError) as excinfo:
        _create_for_layer(monkeypatch, _fp8_scheme(), _BareModule(), in_size=COLUMNS,
                          partitions=[ROWS], input_size=COLUMNS, output_size=ROWS,
                          prefix="linear")
    message = str(excinfo.value)
    assert "tp_rank" in message and "tp_size" in message and "LinearBase" in message
    assert "_BareModule" in message, "the refusal names what it was handed"


def test_a_layer_whose_global_shape_is_no_vllm_linear_contract_is_refused_with_both(monkeypatch):
    """FAILS BEFORE.  ``(input_size, output_size)`` against the tile times
    ``tp_size`` is one of exactly three relations -- whole, output cut, input
    cut -- and a layer that satisfies none has told the plan something it
    cannot serve.  Before, these numbers were never looked at."""
    scheme = _fp8_scheme(roles=[["weight", 64]], columns=64)
    with pytest.raises(ValueError) as excinfo:
        _create_for_layer(monkeypatch, scheme, _Layer(1, 4), in_size=64, partitions=[64],
                          input_size=128, output_size=128, prefix="linear")
    message = str(excinfo.value)
    assert "128x128" in message and "64x64" in message and "rank 1 of 4" in message
    assert "ReplicatedLinear" in message and "ColumnParallel" in message \
        and "RowParallel" in message
