"""Contract tests for the piece-major resident word layout (tessera#739).

These call the PRODUCTION predicates and refusal helpers -- Repacked's re-lay,
piece_major_eligible, require_legacy_word_layout, and the readers that use
them -- not local re-implementations.  The refusal must happen BEFORE any
allocation or extension build, proven by monkeypatching the allocation/build
seams with bombs.
"""
from __future__ import annotations

import dataclasses
import sys
import types

import pytest
import torch


def _install_triton_stub():
    """``window_gemm_grouped`` imports triton at module scope, which the CPU
    image does not ship.  The intake path under test never calls a triton
    kernel, so a minimal stand-in is enough for the import."""
    if "triton" in sys.modules:
        return
    triton = types.ModuleType("triton")
    triton.jit = lambda fn=None, **kw: (fn if fn is not None else (lambda f: f))
    triton.cdiv = lambda a, b: -(-int(a) // int(b))
    language = types.ModuleType("triton.language")
    language.constexpr = type("constexpr", (), {})
    triton.language = language
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = language

from tessera.errors import GrammarError
from tessera import kernel_window_gemv as kw
from tessera.kernel_window_gemv import (
    PIECES_PER_TILE,
    WORD_LAYOUT_LEGACY,
    WORD_LAYOUT_PIECE_MAJOR,
    Repacked,
    piece_major_eligible,
    require_legacy_word_layout,
)


def _repacked(rates, cols=3):
    rate = rates[0]
    per_col = 16 * rate
    tile_words = cols * per_col
    words = torch.arange(tile_words, dtype=torch.int32)
    runs = torch.tensor([[int(rates[0]), 0, cols, 0]], dtype=torch.int32)
    return Repacked(
        words=words, tile_words=tile_words, n_tiles=1, rows=512, cols=cols, rows_p=512,
        perm=torch.arange(cols, dtype=torch.int32), runs=runs,
        rates=tuple(int(r) for r in rates), word_layout=WORD_LAYOUT_LEGACY)


def _unit(rep, family="e4m3"):
    """A WindowGemvUnit around a repack, with the fields a reader reads."""
    return kw.WindowGemvUnit(
        rep=rep, table=torch.zeros(1 << 14, dtype=torch.bfloat16),
        scale=torch.ones(rep.rows, dtype=torch.float32), window_bits=14,
        plan=kw.default_plan(rep.rows, rep.cols, M=1), family=family)


# --- the production re-lay predicate and bijection ------------------------
def test_default_layout_is_legacy():
    assert _repacked((4,)).word_layout == WORD_LAYOUT_LEGACY


def test_piece_major_eligible_only_for_one_rate_four_run():
    assert piece_major_eligible(_repacked((4,)))
    assert not piece_major_eligible(_repacked((4, 5)))
    assert not piece_major_eligible(_repacked((5,)))


def test_layout_property_carries_the_word_layout():
    legacy = _repacked((4,)).layout
    relaid = _repacked((4,)).with_word_layout(WORD_LAYOUT_PIECE_MAJOR).layout
    assert ("word_layout", WORD_LAYOUT_LEGACY) in legacy
    assert ("word_layout", WORD_LAYOUT_PIECE_MAJOR) in relaid
    assert legacy != relaid


def test_piece_major_relay_is_a_bijection_and_keeps_the_counts():
    rep = _repacked((4,), cols=3)
    relaid = rep.with_word_layout(WORD_LAYOUT_PIECE_MAJOR)
    assert relaid.word_layout == WORD_LAYOUT_PIECE_MAJOR
    assert relaid.tile_words == rep.tile_words
    assert relaid.n_tiles == rep.n_tiles
    assert torch.equal(relaid.perm, rep.perm)
    assert torch.equal(relaid.runs, rep.runs)
    assert relaid.rates == rep.rates
    assert sorted(relaid.words.tolist()) == sorted(rep.words.tolist())
    per_piece = 2 * 4
    old = rep.words.reshape(3, PIECES_PER_TILE, per_piece)
    new = relaid.words.reshape(PIECES_PER_TILE, 3, per_piece)
    for t64 in range(PIECES_PER_TILE):
        for col in range(3):
            assert torch.equal(new[t64, col], old[col, t64])


def test_piece_major_relay_refuses_a_two_run_body():
    with pytest.raises(GrammarError):
        _repacked((4, 5)).with_word_layout(WORD_LAYOUT_PIECE_MAJOR)


def test_piece_major_relay_refuses_a_non_rate_four_body():
    with pytest.raises(GrammarError):
        _repacked((5,)).with_word_layout(WORD_LAYOUT_PIECE_MAJOR)


def test_unknown_word_layout_is_refused():
    rep = _repacked((4,))
    with pytest.raises(GrammarError):
        rep.with_word_layout("f16_piece_major")


# --- the production readers refuse, before any allocation/build -----------
def _bomb(*a, **k):
    raise AssertionError("allocation/extension build must not run before the refusal")


def test_window_gemv_refuses_before_allocation(monkeypatch):
    unit = _unit(_repacked((4,)).with_word_layout(WORD_LAYOUT_PIECE_MAJOR))
    # _op_args / _gemv_op / _gemv_concrete are the allocation/extension seams.
    monkeypatch.setattr(kw, "_op_args", _bomb)
    monkeypatch.setattr(kw, "_gemv_op", _bomb)
    monkeypatch.setattr(kw, "_gemv_concrete", _bomb)
    with pytest.raises(GrammarError):
        kw.window_gemv(unit, torch.zeros(1, unit.cols, dtype=torch.bfloat16))


def test_decode_codes_refuses_before_extension(monkeypatch):
    unit = dataclasses.replace(_unit(_repacked((4,)).with_word_layout(WORD_LAYOUT_PIECE_MAJOR)),
                               codes_of_state=torch.zeros(1 << 14, dtype=torch.uint8))
    monkeypatch.setattr(kw, "_ext", _bomb)
    # _ext is the extension loader used only AFTER the layout gate.
    with pytest.raises(GrammarError):
        kw.decode_codes(unit)


def test_require_legacy_word_layout_is_the_production_helper():
    require_legacy_word_layout(WORD_LAYOUT_LEGACY, "reader")
    with pytest.raises(GrammarError):
        require_legacy_word_layout(WORD_LAYOUT_PIECE_MAJOR, "reader")


# --- production bundle + supported-check tags -----------------------------
def _bundle(word_layout, family="e4m3"):
    import types
    return types.SimpleNamespace(
        family=family, word_layout=word_layout, arithmetic="epilogue", device=torch.device("cpu"),
        window_bits=14, experts=2, cols=64, rows=64, quantizer="native",
        # one run per expert: (rate, col0, ncols, word0) rows, flattened [E, R, 4]
        runs_all=torch.tensor([[[4, 0, 64, 0]]] * 2, dtype=torch.int32),
        scale_all=torch.ones(2, 64), perm_all=torch.zeros(2, 64, dtype=torch.int32),
        init_all=torch.zeros(2, 64, dtype=torch.int32), has_init=torch.zeros(2, dtype=torch.int32),
        tile_words=torch.full((2,), 64 * 16 * 4, dtype=torch.int32),
        run_off=torch.tensor([0, 1, 2], dtype=torch.int32), library="threshold")


# --- the intake boundary: opt-in + e4m3 + R4 + fused enabled + MMA lib ----
def test_intake_admissible_matrix(monkeypatch):
    """The full production intake gate, at the exact place the transient is
    re-laid (serving/moe_route._piece_major_admissible)."""
    from tessera.serving import moe_route as mr

    monkeypatch.setenv(mr.ENV_PIECE_MAJOR, "1")
    monkeypatch.delenv("TESSERA_FUSED_E4M3_MMA", raising=False)
    monkeypatch.delenv("TESSERA_ROUTED_FUSED", raising=False)
    assert mr._piece_major_admissible("e4m3") is True
    # BF16 (A8SE layer45) stays legacy
    assert mr._piece_major_admissible("value") is False

    # explicit f16 selects the non-MMA reader: still legacy
    monkeypatch.setenv("TESSERA_FUSED_E4M3_MMA", "f16")
    assert mr._piece_major_admissible("e4m3") is False
    monkeypatch.delenv("TESSERA_FUSED_E4M3_MMA", raising=False)

    # the fused lane disabled keeps every body legacy
    monkeypatch.setenv("TESSERA_ROUTED_FUSED", "0")
    assert mr._piece_major_admissible("e4m3") is False
    monkeypatch.delenv("TESSERA_ROUTED_FUSED", raising=False)

    # opt-in off: legacy
    monkeypatch.setenv(mr.ENV_PIECE_MAJOR, "0")
    assert mr._piece_major_admissible("e4m3") is False


def test_support_refuses_piece_major_on_f16_library_before_ext(monkeypatch):
    """fused_routed_window_supported refuses PM with an alternate f16 library
    BEFORE any _ext/smem/device query."""
    from tessera import routed_fused as rf
    monkeypatch.setattr(rf, "_ext", _bomb)
    monkeypatch.setenv("TESSERA_FUSED_E4M3_MMA", "f16")
    b = _bundle(WORD_LAYOUT_PIECE_MAJOR)
    b.library = "threshold"
    why = rf.fused_routed_window_supported(b, b, b)
    assert why is not None and "MMA" in why


def test_support_admits_piece_major_on_the_mma_library(monkeypatch):
    from tessera import routed_fused as rf
    monkeypatch.delenv("TESSERA_FUSED_E4M3_MMA", raising=False)
    b = _bundle(WORD_LAYOUT_PIECE_MAJOR)
    b.library = "e4m3mma"
    b.device = torch.device("cuda")
    # the bounded shape is the only thing left that can refuse it; with a valid
    # R4 stack and CUDA metadata the reason is not the layout/library refusal.
    why = rf.fused_routed_window_supported(b, b, b)
    if why is not None:
        assert "piece_major" not in why and "MMA" not in why


# --- the production intake load path, with a fake CUDA device -------------
# ``torch.device('cuda')`` is only a spec: it constructs without a GPU, and the
# intake's check and every tensor it touches belong to the STUBBED unit, which
# stays CPU.  So the load path's own layout decision runs with no device.
_FAKE_CUDA = torch.device("cuda")

def _intake(family, *, compact=True):
    """A _RankLocalPackedIntake with one expert, no real wire."""
    _install_triton_stub()
    from tessera.serving.moe_route import _RankLocalPackedIntake
    from tessera.serving.scheme import MOE_GROUPS, TESSERA_FP8, TESSERA_BF16
    declared = {
        "family": TESSERA_FP8 if family == "e4m3" else TESSERA_BF16,
        "experts": 1,
        "groups": {g: {"roles": [{"roles": [["gate", "x"]]}], "columns": 3, "wire_stride": 1}
                   for g in MOE_GROUPS},
        "hidden_size": 3, "intermediate_size": 3,
    }
    obj = object.__new__(_RankLocalPackedIntake)
    obj.declared = declared
    obj.target = "t"
    obj.device = torch.device("cpu")
    obj.family = declared["family"]
    obj.compact = compact
    obj._has_loaded = False
    obj._scratch = {}
    from tessera.native_window_moe import WindowUnitAxis
    wf = "value" if family == "value" else "e4m3"
    obj.axis = {g: WindowUnitAxis(1, ("gate",), family=wf) for g in MOE_GROUPS}
    obj.axes = {g: None for g in MOE_GROUPS}
    return obj


def _stub_compact(monkeypatch, family, rep):
    """Route the real intake through a concrete CPU WindowGemvUnit instead of
    the CUDA kernel, so the load path's layout decision is exercised."""
    from tessera.serving import moe_route as mr

    def fake_units(blob, role, plan, target, *, device, family, scratch):
        return "gate", _unit(rep, family=("value" if family == "value" else "e4m3"))

    monkeypatch.setattr(mr, "_compact_expert_units", fake_units)


def test_intake_load_re_lays_e4m3_r4_when_opted_in(monkeypatch):
    from tessera.serving import moe_route as mr
    from tessera.kernel_window_gemv import WORD_LAYOUT_PIECE_MAJOR
    monkeypatch.setenv(mr.ENV_PIECE_MAJOR, "1")
    monkeypatch.delenv("TESSERA_FUSED_E4M3_MMA", raising=False)
    monkeypatch.delenv("TESSERA_ROUTED_FUSED", raising=False)
    rep = _repacked((4,))
    _stub_compact(monkeypatch, "e4m3", rep)
    obj = _intake("e4m3")
    obj.load("w13", 0, 0, torch.zeros(4, dtype=torch.uint8), device=_FAKE_CUDA)
    # the resident slot's tag is the piece-major one and put really ran
    slot = obj.axis["w13"]._slots["gate"]
    assert obj.axis["w13"]._word_layout["gate"] == WORD_LAYOUT_PIECE_MAJOR
    assert slot["words"].shape[0] == 1


def test_intake_load_keeps_legacy_when_opt_in_off(monkeypatch):
    from tessera.serving import moe_route as mr
    monkeypatch.setenv(mr.ENV_PIECE_MAJOR, "0")
    rep = _repacked((4,))
    _stub_compact(monkeypatch, "e4m3", rep)
    obj = _intake("e4m3")
    obj.load("w13", 0, 0, torch.zeros(4, dtype=torch.uint8), device=_FAKE_CUDA)
    assert obj.axis["w13"]._word_layout["gate"] == WORD_LAYOUT_LEGACY


def test_intake_load_keeps_bf16_folded_legacy(monkeypatch):
    """A8SE layer45 is BF16 with a nonzero start state; it must stay legacy."""
    from tessera.serving import moe_route as mr
    monkeypatch.setenv(mr.ENV_PIECE_MAJOR, "1")
    rep = _repacked((4,))
    unit = _unit(rep, family="value")
    unit = dataclasses.replace(unit, initial_state=torch.arange(rep.cols, dtype=torch.int32))
    from tessera.serving import moe_route as mr2
    monkeypatch.setattr(mr2, "_compact_expert_units",
                        lambda *a, **k: ("gate", unit))
    obj = _intake("value")
    obj.load("w13", 0, 0, torch.zeros(4, dtype=torch.uint8), device=_FAKE_CUDA)
    assert obj.axis["w13"]._word_layout["gate"] == WORD_LAYOUT_LEGACY
    # the nonzero start state survived the placement
    assert int(obj.axis["w13"]._slots["gate"]["has_init"][0]) == 1


def test_supported_check_refuses_piece_major_for_a_non_e4m3_family():
    from tessera.routed_fused import fused_routed_window_supported
    g = _bundle(WORD_LAYOUT_PIECE_MAJOR, family="value")
    why = fused_routed_window_supported(g, g, g)
    # value family is refused earlier for its arithmetic; the layout must not
    # be the thing that lets a non-E4M3 stack through.
    assert why is not None


def test_has_one_rate_four_run_matches_the_bounded_shape():
    from tessera.routed_fused import has_one_rate_four_run
    assert has_one_rate_four_run(_bundle(WORD_LAYOUT_PIECE_MAJOR), 2) is True
    two_run = _bundle(WORD_LAYOUT_PIECE_MAJOR)
    two_run.runs_all = torch.tensor([[[4, 0, 32, 0], [5, 32, 32, 2048]]] * 2, dtype=torch.int32)
    two_run.run_off = torch.tensor([0, 1, 2], dtype=torch.int32)
    assert has_one_rate_four_run(two_run, 2) is False
