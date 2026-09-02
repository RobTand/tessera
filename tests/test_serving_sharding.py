"""The TP seam: which slice a rank serves, and that TP=1 is bit-identically today's.

The plugin serves ``tensor_parallel_size=1``.  These tests are about what
happens at the boundary anyway, because the loaders are the part of a serving
route that is expensive to retrofit: they decide, at ``create_weights`` time,
what shape the layer believes it has, and a wrong answer there is a serve that
runs and returns wrong logits.

Two properties matter and are asserted directly:

1. **TP=1 is identity.**  ``_shard_unit_for_rank`` returns the SAME OBJECT, so
   a caller's parsed view of a unit stays valid and the arithmetic is the
   arithmetic of a build with no TP support at all.
2. **TP>1 refuses by name.**  The cut belongs to ``tessera.layout.slice_unit``
   (a column slice needs the trellis state its first surviving column inherits
   -- the INITIAL_STATE plane), and until that lands the honest answer is a
   refusal that names it, never "every rank holds the whole weight".
"""
from __future__ import annotations

import pytest

from tessera.serving.sharding import (AXIS_COLUMNS, AXIS_ROWS, ShardPlan,
                                      _shard_unit_for_rank, check_shard_granularity,
                                      plan_shard, shard_granularity, shard_parsed_roles,
                                      tp_rank_and_size)


def test_one_rank_is_the_whole_module():
    plan = plan_shard("m", rows=1024, columns=2048, out_size=1024, in_size=2048,
                      tp_rank=0, tp_size=1)
    assert plan.is_whole and plan.axis is None
    assert (plan.shard_rows, plan.shard_columns) == (1024, 2048)


def test_an_output_split_is_the_row_axis():
    """ColumnParallel / MergedColumnParallel / QKVParallel: rows over ranks."""
    plan = plan_shard("m", rows=1024, columns=2048, out_size=256, in_size=2048,
                      tp_rank=1, tp_size=4)
    assert plan.axis == AXIS_ROWS
    assert (plan.shard_rows, plan.shard_columns) == (256, 2048)


def test_an_input_split_is_the_column_axis():
    """RowParallel: the input width over ranks, every row present."""
    plan = plan_shard("m", rows=1024, columns=2048, out_size=1024, in_size=512,
                      tp_rank=3, tp_size=4)
    assert plan.axis == AXIS_COLUMNS
    assert (plan.shard_rows, plan.shard_columns) == (1024, 512)


def test_a_replicated_linear_inside_a_tp_group_is_a_whole_unit():
    """Whole shapes at tp_size > 1 are a replicated layer, not a defect."""
    plan = plan_shard("m", rows=1024, columns=2048, out_size=1024, in_size=2048,
                      tp_rank=2, tp_size=4)
    assert plan.axis is None
    assert (plan.shard_rows, plan.shard_columns) == (1024, 2048)


def test_a_shape_that_is_neither_whole_nor_a_clean_split_is_refused_with_both_shapes():
    with pytest.raises(ValueError) as e:
        plan_shard("model.layers.0.mlp.down_proj", rows=1024, columns=2048,
                   out_size=300, in_size=2048, tp_rank=0, tp_size=4)
    msg = str(e.value)
    assert "model.layers.0.mlp.down_proj" in msg
    assert "1024x2048" in msg and "300x2048" in msg


def test_a_wrong_shape_on_one_rank_still_says_a_wire_is_bound_to_its_shape():
    """The TP=1 message is the one this replaced; it must not have got vaguer."""
    with pytest.raises(ValueError, match="a wire is bound to its shape"):
        plan_shard("m", rows=1024, columns=2048, out_size=999, in_size=2048,
                   tp_rank=0, tp_size=1)


def test_nonsensical_coordinates_are_refused():
    for rank, size in ((4, 4), (-1, 2), (0, 0)):
        with pytest.raises(ValueError, match="TP coordinates"):
            plan_shard("m", rows=8, columns=8, out_size=4, in_size=8,
                       tp_rank=rank, tp_size=size)


def test_the_seam_returns_the_same_object_on_one_rank():
    unit = object()
    assert _shard_unit_for_rank(unit, 0, 1, None) is unit
    assert _shard_unit_for_rank(unit, 0, 1, AXIS_ROWS) is unit


@pytest.mark.parametrize("axis", [AXIS_ROWS, AXIS_COLUMNS])
def test_the_seam_refuses_more_than_one_rank_and_names_the_slicer(axis):
    with pytest.raises(NotImplementedError) as e:
        _shard_unit_for_rank(object(), 1, 4, axis)
    msg = str(e.value)
    assert "tessera.layout.slice_unit" in msg
    assert "tensor_parallel_size=1" in msg      # the operator's way out, in the message


def test_sharding_roles_is_identity_on_one_rank():
    """The path a route actually takes today: the same list, same objects."""
    roles = [("q_proj", object()), ("k_proj", object()), ("v_proj", object())]
    plan = plan_shard("qkv", rows=1024, columns=2048, out_size=1024, in_size=2048,
                      tp_rank=0, tp_size=1)
    assert shard_parsed_roles(roles, plan) is roles


def test_sharding_roles_refuses_a_real_split():
    class _Parsed:
        unit = object()
    roles = [("q_proj", _Parsed())]
    plan = plan_shard("qkv", rows=1024, columns=2048, out_size=256, in_size=2048,
                      tp_rank=1, tp_size=4)
    with pytest.raises(NotImplementedError, match="slice_unit"):
        shard_parsed_roles(roles, plan)


def test_granularity_is_absent_until_the_slicer_lands_and_is_not_guessed():
    """No ``tessera.layout``: the answer is None, and the check is a no-op.

    The refusal for a build that cannot cut belongs at the seam, in one place.
    A granularity check that invented a number would refuse the wrong splits.
    """
    assert shard_granularity(object()) is None
    plan = plan_shard("m", rows=1024, columns=2048, out_size=256, in_size=2048,
                      tp_rank=0, tp_size=4)
    check_shard_granularity(plan, object())     # no raise


def test_granularity_refusal_names_the_granularity(monkeypatch):
    """When the cutter is there, an indivisible split says what it needed."""
    import tessera.serving.sharding as sharding
    monkeypatch.setattr(sharding, "shard_granularity", lambda unit: (192, 16))
    plan = ShardPlan("m", rows=1024, columns=2048, shard_rows=256, shard_columns=2048,
                     tp_rank=0, tp_size=4, axis=AXIS_ROWS)
    with pytest.raises(ValueError) as e:
        sharding.check_shard_granularity(plan, object())
    assert "192" in str(e.value) and "256" in str(e.value)

    plan = ShardPlan("m", rows=1024, columns=2048, shard_rows=1024, shard_columns=520,
                     tp_rank=0, tp_size=4, axis=AXIS_COLUMNS)
    with pytest.raises(ValueError) as e:
        sharding.check_shard_granularity(plan, object())
    assert "16" in str(e.value) and "520" in str(e.value)


def test_tp_coordinates_off_vllm_are_one_rank_not_a_guess():
    assert tp_rank_and_size() == (0, 1)


def test_a_prepared_role_carries_an_initial_state_and_refuses_to_decode_it():
    """The representation carries the sliced-unit plane; no decoder consumes it."""
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


def test_bounds_are_equal_parts_and_refuse_an_uneven_split():
    from tessera.serving.sharding import _bounds

    assert _bounds(1024, 0, 4) == (0, 256)
    assert _bounds(1024, 3, 4) == (768, 1024)
    with pytest.raises(ValueError, match="equal shards"):
        _bounds(1023, 0, 2)


def test_the_pad_really_is_state_minus_one_once_the_slicer_lands():
    """The pad-threading CLAIM, measured -- skipped until the packer can take a state.

    ``window._pack`` threads a shard's start state into ``pack_window_planes``'
    L-bit pad on the argument that the pad *is* ``state_{-1}``.  That is an
    arithmetic claim about the wire, and it is checked here rather than
    asserted: a whole unit's decode from row ``t0`` down must equal the decode
    of the unit's rows ``t0:`` packed with the parent's state at the cut.  The
    identity table makes the decoded value the raw L-bit window, so a
    threading error shows up directly instead of through an alphabet, and the
    zero-pad control fails, so the check cannot pass for the wrong reason.

    This skips on a build whose ``lane_planes`` predates the ``initial_state``
    parameter (i.e. before ``tessera.layout``'s slicer merges).  It was run
    against the merged overlay on 2026-09-02 and passed at six cut points --
    see docs/measurements/tessera-serving-plugin-2026-09-02.md section 6.
    """
    torch = pytest.importorskip("torch")
    import inspect
    import tessera.lane_planes as lane_planes
    from tessera.serving.window import prepare_window

    if "initial_state" not in inspect.signature(lane_planes.pack_window_planes).parameters:
        pytest.skip("installed lane_planes predates the initial_state parameter")

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
