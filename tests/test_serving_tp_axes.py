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
from tessera.serving.scheme import TESSERA_FP8, TESSERA_NVFP4        # noqa: E402
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

    The TP coordinates are read from ``vllm.distributed`` by
    ``sharding.tp_rank_and_size`` -- from the engine's own parallel state, not
    from an environment variable -- so a test that wants a two-rank world says
    so there.
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
    pass


def _nvfp4_scheme(roles=None):
    roles = roles if roles is not None else [["weight", ROWS]]
    return {"family": TESSERA_NVFP4, "grid": "E2M1x2", "body": "TCQ", "plane": "LUT",
            "q256": 896, "rows": sum(r for _, r in roles), "columns": COLUMNS,
            "wire_bytes": 4096, "roles": roles}


def _fp8_scheme(roles=None):
    roles = roles if roles is not None else [["weight", ROWS]]
    return {"family": TESSERA_FP8, "grid": "E4M3", "body": "WINDOW", "plane": "CHANNEL",
            "q256": 1024, "rows": sum(r for _, r in roles), "columns": COLUMNS,
            "wire_bytes": 4096, "roles": roles}


def _create(monkeypatch, scheme, *, axis, tp_size=2, tp_rank=0):
    """Drive ``create_weights`` as the named parallel Linear would.

    ``ColumnParallelLinear`` and its merged/QKV forms split the OUTPUT, so each
    rank asks for ``rows/tp`` outputs over the whole input; ``RowParallelLinear``
    splits the INPUT.  The axis is derived from exactly these two numbers -- the
    plugin never sniffs a class name -- so this is the whole difference between
    the two cases.
    """
    _stub_vllm(monkeypatch, tp_rank=tp_rank, tp_size=tp_size)
    method = build_tessera_method(scheme, "model.layers.0.test")
    layer = _Layer()
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
    assert axis_status(TESSERA_NVFP4, AXIS_ROWS) == TP_REFUSED


def test_an_unknown_route_or_axis_is_a_refusal_not_a_default():
    """Guessing ``sharded`` for a route this table never heard of is exactly
    the silent-wrong-rows outcome the seam exists to prevent."""
    with pytest.raises(ValueError, match="ROUTE_TP_AXES covers"):
        axis_status("TESSERA_BF16", AXIS_ROWS)
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
