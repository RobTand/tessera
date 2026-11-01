"""The routed intake's expert axis over E2M1x2 window units (the ``e2m1`` family).

``native_window_moe.WindowUnitAxis(family="e2m1")`` stacks what
``compact_prep.prepare_window_lut_compact`` reads from each expert's wire --
the tile-word body over tuple codes, the code table, the LUT16 scale plane,
its UE4M3 table and the fp32 global -- into the grouped bundle the fused
routed lane consumes.  The oracle is the per-expert compact prep itself: every
slot of the finished stack must equal what that expert's own unit holds, with
the start state in the repack's column order (tessera#729).
"""
from __future__ import annotations

import pytest
import torch

from tessera.errors import GrammarError

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="the compact window repack is a CUDA path")

ROWS, COLS, EXPERTS = 1024, 512, 3


def _blob(q256, seed, rows=ROWS, cols=COLS):
    from tessera.alphabet import E2M1_GRID, tuple_grid
    from tessera.export import encode_linear_planes
    from tessera.manifest import BodyKind, ScalePlaneKind

    torch.manual_seed(seed)
    weight = (torch.randn(rows, cols, device="cuda") * 0.02).contiguous()
    exported, _unit, _forests = encode_linear_planes(
        weight, grid=tuple_grid(E2M1_GRID, 2), q256=q256, name="w", body=BodyKind.WINDOW,
        scale_plane=ScalePlaneKind.LUT, window_bits=12, verify=False)
    return exported.blob


def _unit(blob, **cut):
    from tessera.compact_prep import parse_compact_wire, prepare_window_lut_compact

    return prepare_window_lut_compact(parse_compact_wire(blob, device="cuda", name="w"),
                                      device="cuda", **cut)


def _bundle(slot):
    from tessera.window_gemm_grouped import prepare_grouped_window_gemm_from_soa

    return prepare_grouped_window_gemm_from_soa(
        words_all=slot["words"], table_all=slot["table"], codes_all=slot["codes"],
        native_all=slot["native"], scale_all=slot["scale"], runs_all=slot["runs"],
        init_all=slot["init"], has_init=slot["has_init"], word_off=slot["word_off"],
        tile_words=slot["tile_words"], total_words=slot["total_words"],
        run_off=slot["run_off"], perm_all=slot["perm"], rows=slot["rows"],
        cols=slot["cols"], experts=EXPERTS, window_bits=slot["window_bits"],
        family="e2m1", scale_plane_all=slot["scale_plane"],
        scale_lut_all=slot["scale_lut"], global_all=slot["global_scale"])


def _check_slot(bundle, e, unit):
    rep = unit.rep
    assert torch.equal(bundle.words_all[e], rep.words.reshape(-1)), ("words", e)
    assert torch.equal(bundle.runs_all[e], rep.runs), ("runs", e)
    assert torch.equal(bundle.perm_all[e], rep.perm.to(torch.int32)), ("perm", e)
    assert int(bundle.tile_words[e]) == int(rep.tile_words), e
    assert int(bundle.total_words[e]) == int(rep.words.numel()), e
    assert torch.equal(bundle.codes_all[e], unit.codes), ("codes", e)
    assert torch.equal(bundle.scale_plane_all[e], unit.scale_plane.reshape(-1)), ("plane", e)
    assert torch.equal(bundle.scale_lut_all[e], unit.scale_lut.reshape(-1)), ("lut", e)
    assert float(bundle.global_all[e]) == float(unit.global_scale), ("global", e)
    init = unit.permuted_start_state()
    if init is None:
        assert int(bundle.has_init[e]) == 0 and not bool(bundle.init_all[e].any()), e
    else:
        assert int(bundle.has_init[e]) == 1, e
        assert torch.equal(bundle.init_all[e], init.to(torch.int32)), ("init", e)


@cuda
@pytest.mark.parametrize("q256", [512, 448])
def test_stack_equals_each_experts_unit(q256):
    """A one-run ([4]) and a two-run ([3, 4]) table: every slot is that
    expert's own compact unit, and the bundle carries the weight rows."""
    from tessera.native_window_moe import WindowUnitAxis

    blobs = [_blob(q256, seed=17 * e + q256) for e in range(EXPERTS)]
    axis = WindowUnitAxis(EXPERTS, ["gate_proj"], family="e2m1")
    for e in (2, 0, 1):                      # arrival order is the loader's, not the axis's
        axis.put("gate_proj", e, _unit(blobs[e]))
    bundle = _bundle(axis.finish()["gate_proj"])
    assert (bundle.rows, bundle.cols, bundle.family) == (ROWS, COLS, "e2m1")
    assert bundle.scale_all.numel() == 0 and bundle.table_all.numel() == 0
    for e in range(EXPERTS):
        _check_slot(bundle, e, _unit(blobs[e]))


@cuda
def test_tp2_rank1_row_cut_carries_the_permuted_start_state():
    """Rank 1's row cut of a two-run stack starts mid-stream: the slot holds
    each column's state in the repack's order (tessera#729)."""
    from tessera.native_window_moe import WindowUnitAxis

    blobs = [_blob(448, seed=5 + e) for e in range(EXPERTS)]
    cut = {"rows": (ROWS // 2, ROWS)}
    axis = WindowUnitAxis(EXPERTS, ["gate_proj"], family="e2m1")
    for e in range(EXPERTS):
        axis.put("gate_proj", e, _unit(blobs[e], **cut))
    bundle = _bundle(axis.finish()["gate_proj"])
    assert bundle.rows == ROWS // 2
    for e in range(EXPERTS):
        unit = _unit(blobs[e], **cut)
        assert unit.permuted_start_state() is not None, "rank 1 starts mid-stream"
        _check_slot(bundle, e, unit)


@cuda
def test_refusals():
    """The wrong unit kind, a layout that differs between experts, a call on
    the bundle, and the LUT scale on another family are each refused."""
    from tessera.native_window_moe import WindowUnitAxis
    from tessera.window_gemm_grouped import prepare_grouped_window_gemm_from_soa

    with pytest.raises(GrammarError, match="families are"):
        WindowUnitAxis(EXPERTS, ["gate_proj"], family="e2m1x2")

    unit = _unit(_blob(512, seed=1))
    with pytest.raises(GrammarError, match="takes a window GEMV unit"):
        WindowUnitAxis(EXPERTS, ["gate_proj"], family="e4m3").put("gate_proj", 0, unit)

    axis = WindowUnitAxis(EXPERTS, ["gate_proj"], family="e2m1")
    axis.put("gate_proj", 0, unit)
    with pytest.raises(GrammarError):
        axis.put("gate_proj", 1, _unit(_blob(448, seed=2)))
    with pytest.raises(GrammarError, match="already placed"):
        axis.put("gate_proj", 0, unit)
    for e in (1, 2):
        axis.put("gate_proj", e, _unit(_blob(512, seed=3 + e)))
    slot = axis.finish()["gate_proj"]
    bundle = _bundle(slot)
    x = torch.zeros(4, COLS, dtype=torch.bfloat16, device="cuda")
    ids = torch.zeros(4, 1, dtype=torch.int32, device="cuda")
    with pytest.raises(GrammarError, match="fused routed window lane"):
        bundle(x, ids, torch.ones(4, 1, device="cuda"))

    with pytest.raises(GrammarError, match="only its"):
        prepare_grouped_window_gemm_from_soa(
            words_all=slot["words"], table_all=slot["table"], codes_all=slot["codes"],
            native_all=slot["native"], scale_all=slot["scale"], runs_all=slot["runs"],
            init_all=slot["init"], has_init=slot["has_init"], word_off=slot["word_off"],
            tile_words=slot["tile_words"], total_words=slot["total_words"],
            run_off=slot["run_off"], perm_all=slot["perm"], rows=slot["rows"],
            cols=slot["cols"], experts=EXPERTS, window_bits=slot["window_bits"],
            family="e4m3", scale_plane_all=slot["scale_plane"],
            scale_lut_all=slot["scale_lut"], global_all=slot["global_scale"])
