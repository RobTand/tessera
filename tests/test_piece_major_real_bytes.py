"""Deterministic byte/layout screens, not routed GPU or served qualification."""
import dataclasses
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from tessera import kernel_window_gemv as kw, routed_fused as rf
from tessera.decode import reconstruct_unit
from tessera.errors import GrammarError
from tessera.grammar import bresenham_rate_schedule
from tessera.slicing import slice_unit
from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact
from window_lut_reference import prepare_reference, window_blob
from window_pack_reference import pack_bitstream


DATA = Path(__file__).parent / "data/legacy"
SCALAR_FIXTURES = sorted(DATA.glob("*-window-channel-*.tessera"))
RATES = sorted(rf.ROUTED_LANE_RATES)
RUNGS = sorted({rate * 256 for rate in RATES} |
               {(lower + upper) * 128 for lower, upper in zip(RATES, RATES[1:])})


def scalar_wire(path, q256):
    parsed = parse_unit_artifact(path.read_bytes(), device="cpu")
    tile_rows = kw.window_geometry.TILE_ROWS
    rows, cols = tile_rows + tile_rows // kw.PIECES_PER_TILE, parsed.unit.body_bits.shape[1]
    rates = bresenham_rate_schedule(Fraction(q256, 256), cols, cap=max(RATES))
    generator = torch.Generator().manual_seed(q256)
    body = (torch.randint(0, 1 << max(rates), (rows, cols), generator=generator)
            % (1 << torch.tensor(rates))).to(torch.uint8)
    zeros = torch.zeros((), dtype=torch.long).expand(rows, cols)
    unit = dataclasses.replace(parsed.unit, rates=rates, body_bits=body,
                               anchors=zeros, codes=zeros, completion_bits=zeros,
                               scale_rows=parsed.unit.scale_rows.repeat(rows // parsed.unit.body_bits.shape[0]))
    blob = build_unit_artifact(unit, "piece-major-cpu-screen", parsed.forests,
                               q256, fixture_id=None)[2]
    return blob, parse_unit_artifact(blob, device="cpu")


def assert_body_and_values(parsed, unit, rep):
    recovered = kw.unpack_tile_words(rep).to(torch.uint8)
    assert torch.equal(recovered, unit.body_bits)
    restored = dataclasses.replace(unit, body_bits=recovered)
    assert torch.equal(reconstruct_unit(restored, parsed.forests, parsed.code),
                       reconstruct_unit(unit, parsed.forests, parsed.code))


@pytest.mark.parametrize("path", SCALAR_FIXTURES, ids=lambda path: path.stem)
@pytest.mark.parametrize("q256", RUNGS)
def test_scalar_real_bytes_cover_routed_rates_and_layout_refusals(path, q256, monkeypatch):
    blob, parsed = scalar_wire(path, q256)
    unit = parsed.unit
    rep = pack_bitstream(unit.body_bits, unit.rates)
    assert rep.n_tiles > 1 and rep.rows_p > rep.rows
    if set(unit.rates).issubset(kw.SUPPORTED_RATES):
        production = kw.repack_window_body(unit.body_bits, unit.rates)
        assert torch.equal(production.words, rep.words)
    assert_body_and_values(parsed, unit, rep)
    original = rep.words.clone()
    if set(unit.rates) == {4}:
        relaid = rep.with_word_layout(kw.WORD_LAYOUT_PIECE_MAJOR)
        assert relaid.nbytes == rep.nbytes
        assert relaid.tile_words == rep.tile_words and relaid.n_tiles == rep.n_tiles
        assert torch.equal(relaid.perm, rep.perm) and torch.equal(relaid.runs, rep.runs)
        assert relaid.rates == rep.rates
        assert_body_and_values(parsed, unit, relaid)
        assert relaid.with_word_layout(kw.WORD_LAYOUT_PIECE_MAJOR) is relaid
        with pytest.raises(GrammarError, match="re-laid once"):
            relaid.with_word_layout(kw.WORD_LAYOUT_LEGACY)
    else:
        def refuse_reorder(*args, **kwargs):
            raise AssertionError("ineligible real bytes reached reordering")
        monkeypatch.setattr(torch.Tensor, "permute", refuse_reorder)
        with pytest.raises(GrammarError, match="one run at rate 4"):
            rep.with_word_layout(kw.WORD_LAYOUT_PIECE_MAJOR)
    assert torch.equal(rep.words, original)
    assert parsed.manifest.payload_digest == parse_unit_artifact(blob, device="cpu").manifest.payload_digest


@pytest.mark.parametrize("path", SCALAR_FIXTURES, ids=lambda path: path.stem)
@pytest.mark.parametrize("axis", ["rows", "cols"])
@pytest.mark.parametrize("rank", range(2))
def test_real_byte_tp2_shards_keep_history_and_values(path, axis, rank):
    blob, parsed = scalar_wire(path, 1024)
    extent = parsed.unit.body_bits.shape[0 if axis == "rows" else 1]
    shard = slice_unit(parsed, **{axis: (rank * extent // 2, (rank + 1) * extent // 2)})
    shard_blob = build_unit_artifact(shard, "piece-major-tp2-screen", parsed.forests,
                                     1024, fixture_id=None)[2]
    reparsed = parse_unit_artifact(shard_blob, device="cpu")
    parent_values = reconstruct_unit(parsed.unit, parsed.forests, parsed.code)
    bounds = slice(rank * extent // 2, (rank + 1) * extent // 2)
    expected = parent_values[bounds, :] if axis == "rows" else parent_values[:, bounds]
    assert torch.equal(reconstruct_unit(reparsed.unit, reparsed.forests, reparsed.code), expected)
    rep = kw.repack_window_body(reparsed.unit.body_bits, reparsed.unit.rates)
    relaid = rep.with_word_layout(kw.WORD_LAYOUT_PIECE_MAJOR)
    assert_body_and_values(reparsed, reparsed.unit, relaid)
    if axis == "rows" and rank:
        assert reparsed.unit.initial_state is not None
        states = kw.reference_states(parsed.unit.body_bits, parsed.unit.rates, parsed.unit.window_bits)
        assert torch.equal(reparsed.unit.initial_state, states[extent // 2 - 1].to(torch.int32))
        holder = kw.WindowGemvUnit(rep=relaid, table=reparsed.unit.window_codes,
                                   scale=reparsed.unit.scale_rows, window_bits=reparsed.unit.window_bits,
                                   plan=kw.Plan(), initial_state=reparsed.unit.initial_state.to(torch.int32),
                                   row_offset=reparsed.unit.row_offset)
        assert torch.equal(holder.permuted_start_state(), reparsed.unit.initial_state[relaid.perm.long()])
    else:
        assert reparsed.unit.initial_state is None
    assert parsed.manifest.payload_digest == parse_unit_artifact(blob, device="cpu").manifest.payload_digest


def test_t4_real_bytes_cannot_inherit_scalar_piece_major_qualification(monkeypatch):
    from tessera import routed_fused_e2m1 as fp4
    from tessera.compact_prep import parse_compact_wire
    wire = parse_compact_wire(window_blob(512, rows=128, cols=256))
    unit = prepare_reference(wire)
    relaid = dataclasses.replace(unit, rep=unit.rep.with_word_layout(kw.WORD_LAYOUT_PIECE_MAJOR))
    def refuse_build(*args, **kwargs):
        raise AssertionError("T4 piece-major refusal reached a native build")
    monkeypatch.setattr(fp4, "_ext", refuse_build)
    with pytest.raises(GrammarError, match="word order"):
        fp4.prepare_dense_role(relaid, 1.0)
