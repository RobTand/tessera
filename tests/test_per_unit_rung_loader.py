"""Storage-ordered per-unit WINDOW declarations and exact CPU wire/axis checks.

Required maps/classes are constructed explicitly. Gate/up schedules are coupled
by the native reader; the standalone CPU decoding below tests per-unit byte
identity and rank cuts, not CUDA serving or old artifact compatibility.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.errors import GrammarError                          # noqa: E402
from tessera.serving import moe_route                            # noqa: E402
from tessera.serving import scheme as S                          # noqa: E402
from tessera.serving.scheme import (                             # noqa: E402
    STRUCTURE_ROUTED_MOE, TESSERA_BF16, TESSERA_FP8, TESSERA_NVFP4,
    expert_role_declarations, validate_tessera_moe_scheme)

HIDDEN, INTER, EXPERTS = 64, 32, 3

# Sorted, fused-compatible profiles. Gate/up share a schedule; down remains its own unit.
UNIT_RUNGS = {
    (0, "gate_proj"): 512, (0, "up_proj"): 512, (0, "down_proj"): 512,
    (1, "gate_proj"): 512, (1, "up_proj"): 512, (1, "down_proj"): 512,
    (2, "gate_proj"): 1088, (2, "up_proj"): 1088, (2, "down_proj"): 1088,
}
MIXED_W13 = [[UNIT_RUNGS[(e, p)] for p in ("gate_proj", "up_proj")]
             for e in range(EXPERTS)]
MIXED_W2 = [[UNIT_RUNGS[(e, "down_proj")]] for e in range(EXPERTS)]

RUNG = 1024


def _moe(q256_w13=RUNG, q256_w2=RUNG, family=TESSERA_FP8, grid="E4M3",
         body="WINDOW", plane="CHANNEL", experts=EXPERTS, hidden=HIDDEN, inter=INTER, **over):
    s = {
        "family": family, "structure": STRUCTURE_ROUTED_MOE,
        "grid": grid, "body": body, "plane": plane, "experts": experts,
        "expert_ids": list(range(experts)),
        "expert_classes": [{"start": 0, "end": experts,
                            "q256": {"w13": [RUNG, RUNG], "w2": [RUNG]}}],
        "groups": {
            "w13": {"rows": 2 * inter, "columns": hidden, "q256": q256_w13,
                    "wire_stride": 8192,
                    "roles": [["gate_proj", inter], ["up_proj", inter]]},
            "w2": {"rows": hidden, "columns": inter, "q256": q256_w2,
                   "wire_stride": 8192,
                   "roles": [["down_proj", hidden]]},
        },
    }
    # Explicit fixture construction for valid scalar/vector/matrix profiles.
    from tessera.expert_classes import build_expert_metadata
    def rows(q, arity):
        if isinstance(q, int):
            return [[q] * arity for _ in range(experts)]
        if isinstance(q, (list, tuple)) and q and all(isinstance(r, int) for r in q):
            return [list(q) for _ in range(experts)]
        return q
    matrices = {"w13": rows(q256_w13, 2), "w2": rows(q256_w2, 1)}
    if (all(isinstance(matrix, (list, tuple)) and len(matrix) == experts for matrix in matrices.values())
            and all(isinstance(row, (list, tuple)) and len(row) == len(s["groups"][group]["roles"])
                    for group, matrix in matrices.items() for row in matrix)
            and all(type(rung) is int and rung > 0
                    for matrix in matrices.values() for row in matrix for rung in row)):
        metadata = build_expert_metadata(matrices)
        s.update(metadata)
        for group, matrix in matrices.items():
            s["groups"][group]["q256"] = S.expert_group_q256([matrix[e] for e in metadata["expert_ids"]])
    s.update(over)
    return s


def _encode_fp8(rows, cols, name, seed, q256, *, window_bits=14):
    """One E4M3 unit at ``q256``: its container, and the stock tile it decodes to."""
    export, stock, alphabet, fused = _tessera()
    w = torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed)) * 0.02
    written, unit, forests = export.encode_linear_planes(
        w.contiguous(), grid=alphabet.E4M3_GRID, q256=q256, name=name,
        window_bits=window_bits, verify=False)
    blob = fused.pack_fused([(name, rows, written.blob)])
    return blob, stock.materialize_stock(unit, forests, export.DEFAULT_CODE)


def _encode_bf16(rows, cols, name, seed, q256):
    """One BF16/8-bit-window unit at ``q256``: container and decoded reference."""
    from tessera.unit_artifact import read_unit_artifact

    export, _stock, alphabet, fused = _tessera()
    w = torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed)) * 0.02
    written, _unit, _forests = export.encode_linear_planes(
        w.contiguous(), grid=alphabet.BF16_GRID, q256=q256, name=name,
        window_bits=8, verify=False)
    blob = fused.pack_fused([(name, rows, written.blob)])
    return blob, read_unit_artifact(written.blob)


def _tessera():
    return (pytest.importorskip("tessera.export"), pytest.importorskip("tessera.stock"),
            pytest.importorskip("tessera.alphabet"), pytest.importorskip("tessera.fused"))


def _mixed_stack(encode, q256_w13, q256_w2, *, family=TESSERA_FP8, grid="E4M3",
                 body="WINDOW", plane="CHANNEL", hidden=HIDDEN, inter=INTER):
    """E experts of (gate, up, down) containers at per-unit rungs, with the
    per-unit reference tile each unit must decode to."""
    w13_blobs, w2_blobs, reference = [], [], []
    for e in range(EXPERTS):
        gate, gate_ref = encode(inter, hidden, "gate_proj", 100 + e,
                                q256_w13[e][0])
        up, up_ref = encode(inter, hidden, "up_proj", 200 + e, q256_w13[e][1])
        down, down_ref = encode(hidden, inter, "down_proj", 300 + e, q256_w2[e][0])
        w13_blobs.append([gate, up])
        w2_blobs.append([down])
        reference.append({"gate": gate_ref, "up": up_ref, "down": down_ref})
    scheme = _moe(q256_w13=list(q256_w13), q256_w2=list(q256_w2), family=family,
                  grid=grid, body=body, plane=plane, hidden=hidden, inter=inter)
    scheme["groups"]["w13"]["wire_stride"] = max(
        len(b) for pair in w13_blobs for b in pair)
    scheme["groups"]["w2"]["wire_stride"] = max(len(pair[0]) for pair in w2_blobs)
    return w13_blobs, w2_blobs, scheme, reference


# ---------------------------------------------------------------- the spelling


def test_an_expert_matrix_must_cover_every_expert_and_role_exactly():
    bad = [
        _moe(q256_w13=MIXED_W13[:-1]),                       # an expert missing
        _moe(q256_w13=[row + [RUNG] for row in MIXED_W13]),  # a role missing
        _moe(q256_w13=[[512, "512"]] * EXPERTS),             # not an integer
        _moe(q256_w13=[512, 768, 1024]),                     # E rungs, not [E][roles]
    ]
    for scheme in bad:
        with pytest.raises(ValueError, match="expert-major|q256\\["):
            validate_tessera_moe_scheme(scheme, "m")


def test_every_matrix_rung_is_put_through_the_lane_rung_gate():
    with pytest.raises(ValueError, match="outside the rungs"):
        validate_tessera_moe_scheme(
            _moe(q256_w13=[[999999, RUNG]] * EXPERTS, q256_w2=MIXED_W2), "m")


def test_an_expert_matrix_is_refused_on_the_nvfp4_route_by_name():
    with pytest.raises(ValueError, match="NVFP4|A4"):
        validate_tessera_moe_scheme(
            _moe(family=TESSERA_NVFP4, grid="E2M1x2", body="TCQ", plane="LUT",
                 q256_w13=MIXED_W13, q256_w2=MIXED_W2), "m")




# ------------------------------------------------- the shared parser, per unit

def test_a_mixed_stack_decodes_bitwise_to_each_units_standalone_uniform_export():
    w13_blobs, w2_blobs, scheme, reference = _mixed_stack(_encode_fp8, MIXED_W13, MIXED_W2)
    declared = validate_tessera_moe_scheme(scheme, "m")
    prepared = moe_route.prepare_tessera_moe_experts(
        {"w13": w13_blobs, "w2": w2_blobs}, declared, "m", device="cpu")
    for e, ref in enumerate(reference):
        w13 = prepared.w13_weight[e].view(torch.uint8)
        assert torch.equal(w13[:INTER], ref["gate"]["weight"].view(torch.uint8))
        assert torch.equal(w13[INTER:], ref["up"]["weight"].view(torch.uint8))
        assert torch.equal(prepared.w2_weight[e].view(torch.uint8),
                           ref["down"]["weight"].view(torch.uint8))
        scale = prepared.w13_weight_scale[e].reshape(-1)
        assert torch.equal(scale[:INTER], ref["gate"]["weight_scale"].reshape(-1).float())
        assert torch.equal(scale[INTER:], ref["up"]["weight_scale"].reshape(-1).float())
        assert torch.equal(prepared.w2_weight_scale[e].reshape(-1),
                           ref["down"]["weight_scale"].reshape(-1).float())


def test_a_wire_at_another_experts_rung_refuses_by_name():
    w13_blobs, w2_blobs, scheme, _ref = _mixed_stack(_encode_fp8, MIXED_W13, MIXED_W2)
    declared = validate_tessera_moe_scheme(scheme, "m")
    swapped = [list(pair) for pair in w13_blobs]
    swapped[2][0] = w13_blobs[1][0]          # expert 1's gate where expert 2's belongs
    with pytest.raises(ValueError, match="sidecar scheme declares"):
        moe_route.prepare_tessera_moe_experts(
            {"w13": swapped, "w2": w2_blobs}, declared, "m", device="cpu")


def test_the_compact_reader_resolves_each_experts_declaration():
    from tessera.serving.scheme import parse_compact_tessera_expert_blob

    w13_blobs, _w2, scheme, _ref = _mixed_stack(_encode_fp8, MIXED_W13, MIXED_W2)
    declared = validate_tessera_moe_scheme(scheme, "m")
    group = declared["groups"]["w13"]
    for e in range(EXPERTS):
        for index, (name, rung) in enumerate(
                (("gate_proj", MIXED_W13[e][0]), ("up_proj", MIXED_W13[e][1]))):
            role = expert_role_declarations(group, expert=e)[index]
            assert role["q256"] == rung
            roles = parse_compact_tessera_expert_blob(
                w13_blobs[e][index], role, f"m expert {e}", device="cpu")
            assert roles[0][0] == name
    # Only a conflicting schedule must refuse, not another id at the same rung.
    wrong = expert_role_declarations(group, expert=0)[0]
    other = next(e for e in range(EXPERTS)
                 if expert_role_declarations(group, expert=e)[0]["q256"] != wrong["q256"])
    with pytest.raises(ValueError):
        parse_compact_tessera_expert_blob(
            w13_blobs[other][0], wrong, f"m expert {other}", device="cpu")


# ------------------------------------------------ research TP2 rank cuts, CPU

def test_a_mixed_bf16_stack_tp2_rank_cuts_match_standalone_uniform_tiles():
    # R1088 down cuts must cover whole256-column quota superblocks at TP2.
    inter = 512
    w13_blobs, w2_blobs, scheme, reference = _mixed_stack(
        _encode_bf16, MIXED_W13, MIXED_W2, family=TESSERA_BF16, grid="BF16", inter=inter)
    declared = validate_tessera_moe_scheme(scheme, "m")
    ids = torch.tensor([1, 0], dtype=torch.int32)
    for rank in (0, 1):
        owner = moe_route.prepare_tessera_packed_bf16_moe_experts(
            {"w13": w13_blobs, "w2": w2_blobs}, declared, "m", device="cpu",
            tp_rank=rank, tp_size=2)
        selected = owner.decode(ids, max_experts_per_chunk=2)
        lo, hi = rank * (inter // 2), (rank + 1) * (inter // 2)
        for slot, expert in enumerate(ids.tolist()):
            gate, up, down = (reference[expert][k] for k in ("gate", "up", "down"))
            full13 = torch.cat([gate, up])
            w13 = selected.w13_weight[slot].float() * selected.w13_weight_scale[slot].reshape(-1, 1)
            w2 = selected.w2_weight[slot].float() * selected.w2_weight_scale[slot].reshape(-1, 1)
            assert torch.equal(w13, torch.cat([full13[lo:hi], full13[inter + lo:inter + hi]]))
            assert torch.equal(w2, down[:, lo:hi])


def test_a_mixed_fp8_stack_tp2_rank_cuts_match_standalone_uniform_tiles():
    inter = 512
    def encode(rows, cols, name, seed, q256):
        return _encode_fp8(rows, cols, name, seed, q256, window_bits=8)
    w13_blobs, w2_blobs, scheme, reference = _mixed_stack(
        encode, MIXED_W13, MIXED_W2, inter=inter)
    declared = validate_tessera_moe_scheme(scheme, "m")
    ids = torch.tensor([2, 0], dtype=torch.int32)
    for rank in (0, 1):
        owner = moe_route.prepare_tessera_packed_moe_experts(
            {"w13": w13_blobs, "w2": w2_blobs}, declared, "m", device="cpu",
            tp_rank=rank, tp_size=2)
        selected = owner.decode(ids, max_experts_per_chunk=2)
        lo, hi = rank * (inter // 2), (rank + 1) * (inter // 2)
        for slot, expert in enumerate(ids.tolist()):
            gate, up, down = (reference[expert][k] for k in ("gate", "up", "down"))
            weight = selected.w13_weight[slot].view(torch.uint8)
            assert torch.equal(weight[:inter // 2], gate["weight"][lo:hi].view(torch.uint8))
            assert torch.equal(weight[inter // 2:], up["weight"][lo:hi].view(torch.uint8))
            assert torch.equal(selected.w2_weight[slot].view(torch.uint8),
                               down["weight"][:, lo:hi].view(torch.uint8))
            scale = selected.w13_weight_scale[slot].reshape(-1)
            assert torch.equal(scale[:inter // 2],
                               gate["weight_scale"].reshape(-1)[lo:hi])
            assert torch.equal(scale[inter // 2:], up["weight_scale"].reshape(-1)[lo:hi])

            assert torch.equal(selected.w2_weight_scale[slot].reshape(-1),
                               down["weight_scale"].reshape(-1))




def test_the_intake_derives_exact_flat_sizes_for_r1024_and_r1088_at_both_ranks():
    declared = validate_tessera_moe_scheme(_moe(q256_w13=MIXED_W13, q256_w2=MIXED_W2), "m")
    # w13 keeps the FULL source column schedule at every rank (row-cut only);
    # one 512-row tile at this geometry, so words = 16 * sum(local rates).
    # R512 -> rate 2 only; R1024 -> rate 4 only; R1088 -> quota 4.25 over 64
    # columns = 16 columns at rate 5 + 48 at rate 4 -> 4352 words, two runs.
    expected_w13 = {
        "gate_proj": [(2048, 1), (2048, 1), (4352, 2)],
        "up_proj": [(2048, 1), (2048, 1), (4352, 2)],
    }
    # w2 is column-cut: the rank's slice of the source schedule.  R1088 over
    # 32 columns is 8 columns at rate 5 + 24 at rate 4; each rank's 16-column
    # half carries exactly 4 of the uppers -> 1088 words, two runs, BOTH ranks
    # (the placement is the grammar's own, and put re-validates it exactly).
    # R512 keeps rate 2, so a rank's half is 512 words, one run.
    expected_w2 = {
        "down_proj": [(512, 1), (512, 1), (1088, 2)],
    }
    for rank in (0, 1):
        plans = {g: moe_route._packed_group_shard_plan(declared, g, "m", rank, 2)
                 for g in moe_route.MOE_GROUPS}
        sizes = moe_route._mixed_axis_word_runs(declared, plans)
        assert sizes["w13"] == expected_w13
        assert sizes["w2"] == expected_w2
    # TP1 carries the whole schedule.
    plans = {g: moe_route._packed_group_shard_plan(declared, g, "m", 0, 1)
             for g in moe_route.MOE_GROUPS}
    sizes = moe_route._mixed_axis_word_runs(declared, plans)
    assert sizes["w2"] == {"down_proj": [(1024, 1), (1024, 1), (2176, 2)]}
    # A uniform group predeclares nothing: the legacy allocation stands.
    uniform = validate_tessera_moe_scheme(_moe(), "m")
    assert moe_route._mixed_axis_word_runs(
        uniform, {g: moe_route._packed_group_shard_plan(uniform, g, "m", 0, 1)
                  for g in moe_route.MOE_GROUPS}) is None


def test_a_mixed_stack_refuses_an_incompatible_piece_major_knob_by_name(monkeypatch):
    """PM is a one-run rate-4 layout; a mixed stack would tag slots apart.

    With the experimental knob requested, a non-uniform-schedule intake
    refuses BY NAME up front instead of laying some projections piece-major
    and others legacy (and never re-tags after the fact).  That covers the
    expert matrix AND a schedule that compresses to a per-role list with no
    matrix anywhere.  With the knob unset the mixed stack is wholly legacy,
    exactly like every uniform stack today.
    """
    real_admissible = moe_route._piece_major_admissible
    monkeypatch.setattr(moe_route, "_piece_major_requested", lambda: True)
    monkeypatch.setattr(moe_route, "_piece_major_admissible", lambda family: True)
    matrix = validate_tessera_moe_scheme(_moe(q256_w13=MIXED_W13, q256_w2=MIXED_W2), "m")
    with pytest.raises(ValueError, match="piece-major|PIECE_MAJOR"):
        moe_route._RankLocalPackedIntake(matrix, "m", torch.device("cpu"), 0, 1)
    per_role = validate_tessera_moe_scheme(_moe(q256_w2=1088), "m")
    with pytest.raises(ValueError, match="piece-major|PIECE_MAJOR"):
        moe_route._RankLocalPackedIntake(per_role, "m", torch.device("cpu"), 0, 1)
    monkeypatch.setattr(moe_route, "_piece_major_requested", lambda: False)
    monkeypatch.setattr(moe_route, "_piece_major_admissible", real_admissible)
    intake = moe_route._RankLocalPackedIntake(matrix, "m", torch.device("cpu"), 0, 1)
    assert intake._piece_major is False


# ------------------------------------------------- the axis, in exact storage

def _cpu_units(rates_per_expert, rows, cols, family="value", seed=900):
    """Real ``WindowGemvUnit``s on CPU with per-expert column rates."""
    from test_window_gemm_grouped import Expert

    return [Expert(rows, cols, rates, seed + i, family=family, device="cpu").unit
            for i, rates in enumerate(rates_per_expert)]


def test_a_window_axis_holds_mixed_units_in_exact_flat_storage_matching_the_units_path():
    from tessera.native_window_moe import WindowUnitAxis
    from tessera.window_gemm_grouped import prepare_grouped_window_gemm

    for family in ("value", "e4m3"):
        rates = [(4,) * 16, (2,) * 16, (2, 4) * 8]
        units = _cpu_units(rates, 64, 16, family=family)
        sizes = [("gate_proj", [(u.rep.words.numel(), u.rep.runs.shape[0])
                                for u in units])]
        axis = WindowUnitAxis(EXPERTS, ("gate_proj",), family=family,
                              word_runs=dict(sizes))
        for e, unit in enumerate(units):
            axis.put("gate_proj", e, unit)
        soa = axis.finish()["gate_proj"]
        # Only packed constants are compared; CPU preparation needs no vLLM A quantizer.
        reference = prepare_grouped_window_gemm(units, quantizer=None)
        assert torch.equal(soa["words"].reshape(-1), reference.words_all)
        assert torch.equal(soa["runs"].reshape(-1, 4), reference.runs_all)
        # word_off is an int32 [E] here (the priced per-unit scalar); the
        # units path carries the same offsets in int64.
        assert soa["word_off"].dtype == torch.int32
        assert torch.equal(soa["word_off"].long(), reference.word_off)
        assert torch.equal(soa["run_off"], reference.run_off)
        assert torch.equal(soa["tile_words"], reference.tile_words)
        assert torch.equal(soa["total_words"], reference.total_words)
        # Exact storage: the flat words are the sum of the units' own, with
        # nothing padded and nothing left over.
        assert soa["words"].numel() == sum(u.rep.words.numel() for u in units)
        assert soa["runs"].shape[0] == sum(u.rep.runs.shape[0] for u in units)


def test_a_window_axis_keeps_the_uniform_path_allocation_unchanged():
    from tessera.native_window_moe import WindowUnitAxis

    units = _cpu_units([(4,) * 16] * EXPERTS, 64, 16)
    # No predeclared sizes: today's allocation, byte for byte.
    axis = WindowUnitAxis(EXPERTS, ("gate_proj",), family="value")
    for e, unit in enumerate(units):
        axis.put("gate_proj", e, unit)
    soa = axis.finish()["gate_proj"]
    words = units[0].rep.words.numel()
    assert soa["words"].shape == (EXPERTS, words)
    assert soa["runs"].shape == (EXPERTS, units[0].rep.runs.shape[0], 4)
    assert soa["word_off"].tolist() == [e * words for e in range(EXPERTS)]
    # The same stack through predeclared sizes must land on the SAME tensors:
    # a uniform stack never changes shape because the caller stated the sizes.
    sized = WindowUnitAxis(EXPERTS, ("gate_proj",), family="value",
                           word_runs={"gate_proj": [(u.rep.words.numel(),
                                                     u.rep.runs.shape[0])
                                                    for u in units]})
    for e, unit in enumerate(units):
        sized.put("gate_proj", e, unit)
    same = sized.finish()["gate_proj"]
    assert same["words"].shape == (EXPERTS, words)
    assert same["word_off"].tolist() == soa["word_off"].tolist()
    assert torch.equal(same["words"], soa["words"])


def test_a_window_axis_refuses_a_mixed_part_without_predeclared_sizes():
    from tessera.native_window_moe import WindowUnitAxis

    units = _cpu_units([(4,) * 16, (2,) * 16], 64, 16)
    axis = WindowUnitAxis(2, ("gate_proj",), family="value")
    axis.put("gate_proj", 0, units[0])
    with pytest.raises(GrammarError, match="predeclared|exact|word_runs"):
        axis.put("gate_proj", 1, units[1])


def test_a_window_axis_refuses_a_unit_off_its_predeclared_size():
    from tessera.native_window_moe import WindowUnitAxis

    units = _cpu_units([(4,) * 16, (2,) * 16], 64, 16)
    first = (units[0].rep.words.numel(), units[0].rep.runs.shape[0])
    # A flat part: the second expert's declared place differs from the first's,
    # so the sizes allocate flat storage and the put checks the unit against
    # ITS OWN declared size exactly.
    sizes = {"gate_proj": [first, (first[0] + 16, first[1])]}
    axis = WindowUnitAxis(2, ("gate_proj",), family="value", word_runs=sizes)
    axis.put("gate_proj", 0, units[0])
    with pytest.raises(GrammarError, match="predeclared"):
        axis.put("gate_proj", 1, units[1])


def test_grouped_soa_accepts_the_mixed_flat_stack_and_refuses_broken_offsets():
    from tessera.native_window_moe import WindowUnitAxis
    from tessera.window_gemm_grouped import prepare_grouped_window_gemm_from_soa

    units = _cpu_units([(4,) * 16, (2,) * 16, (2, 4) * 8], 64, 16)
    axis = WindowUnitAxis(EXPERTS, ("gate_proj",), family="value",
                          word_runs={"gate_proj": [(u.rep.words.numel(),
                                                    u.rep.runs.shape[0])
                                                   for u in units]})
    for e, unit in enumerate(units):
        axis.put("gate_proj", e, unit)
    soa = axis.finish()["gate_proj"]
    bundle = prepare_grouped_window_gemm_from_soa(words_all=soa["words"], table_all=soa["table"], codes_all=soa["codes"],
    native_all=soa["native"], scale_all=soa["scale"], runs_all=soa["runs"],
    init_all=soa["init"], has_init=soa["has_init"], word_off=soa["word_off"],
    tile_words=soa["tile_words"], total_words=soa["total_words"],
    run_off=soa["run_off"], perm_all=soa["perm"], rows=64, cols=16,
    experts=EXPERTS, window_bits=14, family="value", word_layout="legacy")
    assert bundle.experts == EXPERTS
    broken = dict(soa)
    broken["word_off"] = soa["word_off"].clone()
    broken["word_off"][1] += 3
    with pytest.raises(GrammarError, match="word_off"):
        prepare_grouped_window_gemm_from_soa(words_all=broken["words"], table_all=broken["table"],
        codes_all=broken["codes"], native_all=broken["native"],
        scale_all=broken["scale"], runs_all=broken["runs"],
        init_all=broken["init"], has_init=broken["has_init"],
        word_off=broken["word_off"], tile_words=broken["tile_words"],
        total_words=broken["total_words"], run_off=broken["run_off"],
        perm_all=broken["perm"], rows=64, cols=16, experts=EXPERTS,
        window_bits=14, family="value", word_layout="legacy")
