"""The span-2 decoders start a ROW SHARD from its own register (tessera#492).

A unit cut below row 0 (``tessera.layout.slice_unit``) carries the trellis
register each column is in at the cut -- the INITIAL_STATE plane.  The window
body always threaded it into its pad; the span-2 TCQ body's decoders read step
0's history out of the select plane's ``SELECT_PAD``, which was pinned to zero,
so ``pack_unit_for_kernel`` refused a shard and the NVFP4 route refused every
row cut at ``create_weights``.  ``lane_planes._thread_start_state`` now writes
the register into the pad in the stream order the decoders' window tables
read, and this file is the proof, in two halves:

* **CPU, against the decoder's own tables.**  No encoder, no kernel: a random
  select stream and a random start state are packed, the kernel's window read
  is emulated bit for bit on the packed plane, and the super-label
  ``build_span2_luts`` returns for that window must be the one
  ``decode._replay_tables`` returns for the state ``decode._conv_state_stream``
  derives from the same stream and the same start -- at every step the start
  can still reach.  A wrong bit order fails here before any GPU is asked.

* **CUDA, on a real unit through the serving seam.**  A real E2M1x2 q256=896
  unit is cut on rows at tp 2 and 4 through ``sharding.shard_parsed_roles``
  (the route's own path) and prepared by ``ops.prepare_tessera_module``, whose
  decode must equal the whole unit's decode sliced, ``torch.equal`` on both
  planes -- on the native decoder, which at load is also held to
  ``materialize_stock`` (``_require_reference_agreement``), and on the
  pure-torch fallback.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tessera.decode import _conv_state_stream, _replay_tables      # noqa: E402
from tessera.errors import GrammarError                             # noqa: E402
from tessera.lane_planes import (                                   # noqa: E402
    SELECT_PAD, _thread_start_state, build_span2_luts, pack_kernel_planes)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the Tessera encoder and the native decoder are CUDA paths")


# --- the CPU half: the pad is read as the state ----------------------------------

def _unpack_columns(plane: torch.Tensor, positions: int, cols: int) -> torch.Tensor:
    """The inverse of ``lane_planes._pack_columns`` at one bit per position:
    ``[positions, cols]`` bits, column-major, MSB-first, from the packed bytes."""
    per_col = -(-positions // 8)
    bytes_ = plane[: per_col * cols].reshape(cols, per_col).to(torch.int64)
    shifts = torch.arange(7, -1, -1, dtype=torch.int64)
    bits = ((bytes_.unsqueeze(-1) >> shifts) & 1).reshape(cols, per_col * 8)
    return bits[:, :positions].t().contiguous()


def _kernel_window(select_bits: torch.Tensor, memory: int, step: int) -> torch.Tensor:
    """What ``tessera_span2_body_kernel`` reads for pair ``step``: the
    ``memory`` pad-or-select positions above it, oldest first, then its own
    bit -- ``(1 << (memory + 1))`` values, one per column.  ``select_bits`` is
    the padded plane, ``[SELECT_PAD + pairs, cols]``."""
    start = SELECT_PAD + step - memory
    window = torch.zeros(select_bits.shape[1], dtype=torch.int64)
    for offset in range(memory + 1):
        window = (window << 1) | select_bits[start + offset].to(torch.int64)
    return window


@pytest.mark.parametrize("memory", [3, 4, 6, 8])       # every published default pair
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_pad_reads_as_the_start_state_at_every_step_it_reaches(memory, seed):
    """FAILS BEFORE (the pad was always zero): for every step ``t`` the
    decoder's window at ``t`` maps -- through the same tables the CUDA kernel
    indexes -- to the state the reference replay is in at ``t`` when started
    from the shard's register."""
    from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
    from tessera.trellis import ConvCode

    grid = tuple_grid(E2M1_GRID, 2)
    forest = build_forest(grid.rate_cap, grid=grid)
    code = ConvCode(memory=memory)
    pairs, cols = 16, 5
    g = torch.Generator().manual_seed(seed)
    select = torch.randint(0, 2, (pairs, cols), generator=g)
    state0 = torch.randint(0, 1 << memory, (cols,), generator=g)
    padded = torch.zeros(SELECT_PAD + pairs, cols, dtype=torch.int32)
    padded[SELECT_PAD:] = select.to(torch.int32)
    _thread_start_state(padded, memory, state0)
    # The reference: the register before each step, started from state0.
    states = _conv_state_stream(select.to(torch.int64), memory, state0.to(torch.int64))
    _subsets, _next, table_sub = _replay_tables(forest, code, "cpu")
    label_lut, _subset_lut = build_span2_luts(forest, code, "cpu")
    for step in range(pairs):
        window = _kernel_window(padded, memory, step)
        want = table_sub[select[step].long(), states[step].long()].to(torch.int64)
        assert torch.equal(label_lut[window].to(torch.int64), want), (memory, seed, step)
    # ... and a zero start is the pad every whole unit always had.
    zero = torch.zeros(SELECT_PAD + pairs, cols, dtype=torch.int32)
    zero[SELECT_PAD:] = select.to(torch.int32)
    _thread_start_state(zero, memory, None)
    assert torch.equal(zero[:SELECT_PAD], torch.zeros(SELECT_PAD, cols, dtype=torch.int32))


@pytest.mark.parametrize("span", [1, 2])
def test_pack_kernel_planes_threads_the_state_into_the_select_plane_only(span):
    """The state lands in the pad of the select plane and nowhere else: the
    label and point planes of a shard are the label and point planes of the
    same bits packed whole."""
    rate, memory = 3, 6
    steps, cols = 32, 4
    g = torch.Generator().manual_seed(5)
    body = torch.randint(0, 1 << rate, (steps, cols), generator=g).to(torch.uint8)
    state = torch.randint(0, 1 << memory, (cols,), generator=g)
    whole = pack_kernel_planes(body, rate=rate, memory=memory, span=span)
    shard = pack_kernel_planes(body, rate=rate, memory=memory, span=span, initial_state=state)
    assert len(whole) == len(shard)
    for plane_whole, plane_shard in zip(whole[1:], shard[1:]):
        assert torch.equal(plane_whole, plane_shard)
    positions = SELECT_PAD + (steps if span == 1 else steps // 2)
    bits_whole = _unpack_columns(whole[0], positions, cols)
    bits_shard = _unpack_columns(shard[0], positions, cols)
    assert torch.equal(bits_whole[SELECT_PAD:], bits_shard[SELECT_PAD:])
    assert torch.equal(bits_whole[:SELECT_PAD], torch.zeros(SELECT_PAD, cols, dtype=torch.int64))
    for j in range(memory):
        assert torch.equal(bits_shard[SELECT_PAD - memory + j], (state >> j) & 1), j
    assert torch.equal(bits_shard[:SELECT_PAD - memory],
                       torch.zeros(SELECT_PAD - memory, cols, dtype=torch.int64))


def test_a_state_that_is_not_one_register_per_column_is_refused():
    padded = torch.zeros(SELECT_PAD + 8, 3, dtype=torch.int32)
    with pytest.raises(GrammarError, match="one register per column"):
        _thread_start_state(padded, 6, torch.zeros(2, dtype=torch.int64))
    with pytest.raises(GrammarError, match="outside"):
        _thread_start_state(padded, 6, torch.tensor([0, 64, 1]))
    with pytest.raises(GrammarError, match="outside"):
        _thread_start_state(padded, 6, torch.tensor([0, -1, 1]))
    with pytest.raises(GrammarError, match="select pad"):
        _thread_start_state(padded, SELECT_PAD + 1, torch.zeros(3, dtype=torch.int64))


# --- the CUDA half: a real shard through the serving seam ---------------------

UNIT_ROWS, UNIT_COLS = 64, 512


@pytest.fixture(scope="module")
def parsed_unit():
    if not torch.cuda.is_available():
        pytest.skip("the Tessera encoder is a CUDA path")
    from tessera.alphabet import E2M1_GRID, tuple_grid
    from tessera.export import encode_linear_planes
    from tessera.unit_artifact import parse_unit_artifact

    torch.manual_seed(492)
    weight = (torch.randn(UNIT_ROWS, UNIT_COLS, device="cuda") * 0.02).contiguous()
    exported, _unit, _forests = encode_linear_planes(
        weight, grid=tuple_grid(E2M1_GRID, 2), q256=896, name="shard-me", verify=False)
    return parse_unit_artifact(exported.blob, device="cuda")


def _row_plan(rows, columns, tp_rank, tp_size):
    from tessera.serving.sharding import plan_shard

    return plan_shard("mlp.gate_up", roles=[("weight", rows)], columns=columns,
                      out_partitions=[rows // tp_size], in_size=columns, tp_rank=tp_rank,
                      tp_size=tp_size, input_size=columns, output_size=rows)


def _prepare(roles, *, allow_torch_fallback):
    from tessera.serving.ops import prepare_tessera_module

    return prepare_tessera_module(roles, device=torch.device("cuda"),
                                  allow_torch_fallback=allow_torch_fallback, prefix="test")


@needs_cuda
@pytest.mark.parametrize("tp_size", [2, 4])
def test_a_row_shard_decodes_to_the_whole_units_rows_on_the_native_decoder(parsed_unit, tp_size):
    """FAILS BEFORE: ``prepare_tessera_module`` refused a role carrying an
    INITIAL_STATE plane.  Every rank's shard now decodes, through the native
    span-2 decoder, to exactly its rows of the whole unit's decode -- and the
    load-time reference agreement inside ``prepare_tessera_module`` has already
    held each of those decodes to ``materialize_stock``."""
    from tessera.serving.ext import get_tessera_ext
    from tessera.serving.sharding import AXIS_ROWS, shard_parsed_roles
    from tessera.serving.telemetry import DECODER_NATIVE_SPAN2

    if get_tessera_ext() is None:
        pytest.skip("the native span-2 decoder could not be built here")
    whole = _prepare([("weight", parsed_unit)], allow_torch_fallback=False)
    packed_whole, scales_whole = whole.decode()
    rows_per_rank = UNIT_ROWS // tp_size
    for rank in range(tp_size):
        plan = _row_plan(UNIT_ROWS, UNIT_COLS, rank, tp_size)
        assert plan.axis == AXIS_ROWS
        roles = shard_parsed_roles([("weight", parsed_unit)], plan)
        shard_unit = roles[0][1].unit
        if rank:
            assert shard_unit.initial_state is not None and shard_unit.row_offset == rank * rows_per_rank
        prepared = _prepare(roles, allow_torch_fallback=False)
        assert prepared.decoder == DECODER_NATIVE_SPAN2
        packed, scales = prepared.decode()
        lo, hi = rank * rows_per_rank, (rank + 1) * rows_per_rank
        assert torch.equal(packed, packed_whole[lo:hi]), (tp_size, rank, "packed")
        assert torch.equal(scales, scales_whole[lo:hi]), (tp_size, rank, "scales")


@needs_cuda
def test_a_row_shard_decodes_to_the_whole_units_rows_on_the_torch_fallback(parsed_unit, monkeypatch):
    """The other decoder the route publishes (``native_extensions[].when_unavailable``):
    the same shards through ``materialize_stock`` alone, held to the same rows."""
    from tessera.serving import ext
    from tessera.serving.sharding import shard_parsed_roles
    from tessera.serving.telemetry import DECODER_TORCH_STOCK

    # ``ops.prepare_tessera_module`` imports ``get_tessera_ext`` from ``.ext``
    # at call time, so the module attribute is the one seam to close.
    monkeypatch.setattr(ext, "get_tessera_ext", lambda: None)
    whole = _prepare([("weight", parsed_unit)], allow_torch_fallback=True)
    assert whole.decoder == DECODER_TORCH_STOCK
    packed_whole, scales_whole = whole.decode()
    for rank in range(2):
        roles = shard_parsed_roles([("weight", parsed_unit)], _row_plan(UNIT_ROWS, UNIT_COLS, rank, 2))
        prepared = _prepare(roles, allow_torch_fallback=True)
        packed, scales = prepared.decode()
        lo, hi = rank * (UNIT_ROWS // 2), (rank + 1) * (UNIT_ROWS // 2)
        assert torch.equal(packed, packed_whole[lo:hi])
        assert torch.equal(scales, scales_whole[lo:hi])


@needs_cuda
def test_a_shard_whose_register_is_another_codes_width_is_refused(parsed_unit):
    """The pad carries the register of the code the planes are packed for and
    no other: a shard declaring a different ``state_bits`` is refused by name
    rather than written into the wrong pad positions."""
    from dataclasses import replace

    from tessera.lane_planes import pack_unit_for_kernel
    from tessera.serving.sharding import shard_parsed_roles

    roles = shard_parsed_roles([("weight", parsed_unit)], _row_plan(UNIT_ROWS, UNIT_COLS, 1, 2))
    shard = roles[0][1]
    wrong = replace(shard.unit, state_bits=shard.unit.state_bits + 1)
    forest = shard.forests[sorted(set(shard.unit.rates))[0]]
    with pytest.raises(GrammarError, match="register of another code"):
        pack_unit_for_kernel(wrong, forest, shard.code)
