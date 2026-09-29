"""A routed stack's resident-mode bytes are the compact lane's own tensors
(tessera#624), and the MTP draft's embed/head duplicate is its own per-rank
line item (tessera#645).

The runtime serves every TESSERA_FP8/TESSERA_BF16 routed stack through
``serving.moe_route``'s compact window lane, which never allocates the decoded
1-byte tile the exporter used to price (``rows * cols + rows * 4`` per unit):
what a rank holds is ``native_window_moe.WindowUnitAxis``'s stacked planes,
tables and bookkeeping, plus the fused lane's composed tables (#685) and,
since contract v45 (tessera#694), its run pairs and block descriptors, where
the stack's shape admits it -- at every one- or two-rate shape the v45 lane
reads, not rate 4 alone.  The anchor here is the runtime's own allocation,
built on CPU from the exported wires' verified metadata by the same axis the
loader fills, and priced by the same ``resident_bytes`` the load bench
reports.
"""
from __future__ import annotations

import importlib
import inspect
import json

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from tessera import kernel_window_gemv as kg  # noqa: E402
from tessera.compact_prep import parse_compact_expert  # noqa: E402
from tessera import routed_fused, serving_parts  # noqa: E402
from tessera.routed_fused import WINDOW_BITS  # noqa: E402
from tessera.serving.scheme import MOE_GROUPS, MOE_GROUP_PROJECTIONS  # noqa: E402
from window_pack_reference import pack_bitstream  # noqa: E402

export = importlib.import_module("tessera.export_serving")
moe_write = importlib.import_module("test_export_moe_write")


def _priced(module, name):
    """The pricing helper, or a stand-in that fails the test naming it -- so a
    tree without the helper fails on the figures rather than at collection."""
    helper = getattr(module, name, None)
    if helper is not None:
        return helper

    def absent(*_args, **_kwargs):
        pytest.fail(f"{module.__name__}.{name} is not published")
    return absent


fused_routed_unit_shape_refusal = _priced(routed_fused, "fused_routed_unit_shape_refusal")
mtp_draft_embed_head_duplicate_bytes = _priced(serving_parts, "mtp_draft_embed_head_duplicate_bytes")
per_rank_fit_items = _priced(serving_parts, "per_rank_fit_items")
routed_fused_table_bytes = _priced(serving_parts, "routed_fused_table_bytes")
routed_fused_unit_bytes = _priced(serving_parts, "routed_fused_unit_bytes")
routed_window_part_resident_bytes = _priced(serving_parts, "routed_window_part_resident_bytes")
routed_window_unit_resident_bytes = _priced(serving_parts, "routed_window_unit_resident_bytes")
vocab_parallel_rows = _priced(serving_parts, "vocab_parallel_rows")

STACK = moe_write.STACK
#: q256 1024: every column at rate 4, the exported stack's rung.
RATE_4 = 4
# The smallest shape the fused lane admits whole: every part at least 128
# columns and a multiple of 32, gate/up rows a multiple of 64, down rows a
# multiple of 128 -- so the stack carries composed tables at TP1, and the
# down projection's TP2 cut (64 columns) refuses them, which is the case the
# per-rank block must price differently from a halved TP1 figure.
HIDDEN, INTER, EXPERTS, FIT_TP = 128, 128, 2, 2
FAMILY_OF = {"TESSERA_BF16": "value", "TESSERA_FP8": "e4m3"}


def _unit(family: str, rows: int, rates, bits: int = WINDOW_BITS) -> kg.WindowGemvUnit:
    """A CPU unit of the given layout, as the loader's compact preparer shapes one."""
    value = family == "value"
    return kg.WindowGemvUnit(
        rep=pack_bitstream(torch.zeros((rows, len(rates)), dtype=torch.int64), rates),
        table=torch.zeros(1 << bits, dtype=torch.bfloat16),
        scale=torch.ones(rows, dtype=torch.float32), window_bits=bits, plan=kg.Plan(),
        codes_of_state=None if value else torch.zeros(1 << bits, dtype=torch.uint8),
        native=None if value else torch.zeros(256, dtype=torch.uint8),
        family=family, initial_state=torch.zeros(len(rates), dtype=torch.int32))


def _slot_bytes(family: str, units: list, experts: int) -> int:
    """What ``WindowUnitAxis._alloc`` + ``finish`` hold for one part, spelled
    from the units' own repacked planes (the reference packer's ``Repacked``:
    padded tile words, one run per distinct rate, the permutation), stacked
    ``experts`` deep, plus the four int32 per-expert scalars and ``run_off``.

    The triton-free anchor: the grouped kernel modules import triton at
    module level, so on a CPU host the axis itself cannot be imported;
    :func:`_runtime_bytes` is the same figure off the real axis where it can.
    """
    unit = units[0]
    rep = unit.rep
    per_expert = (rep.words.numel() * 4 + rep.runs.numel() * 4 + unit.scale.numel() * 4
                  + rep.perm.numel() * 4 + rep.cols * 4        # words, runs, scale, perm, init
                  + 4 * 4)                                     # tile_words, total_words, has_init, word_off
    if family == "value":
        per_expert += unit.table.numel() * 2
    else:
        per_expert += unit.codes_of_state.numel() + unit.native.numel()
    for other in units[1:]:
        assert (other.rep.words.numel(), other.rep.runs.shape, other.rows, other.cols) == (
            rep.words.numel(), rep.runs.shape, unit.rows, unit.cols), "one layout per part"
    return experts * per_expert + 4 * (experts + 1)


def _reference_bytes(family: str, units_by_part: dict, experts: int) -> int:
    """The compact lane's allocation for a stack, by the CPU reference."""
    total = 0
    for group in MOE_GROUPS:
        for part in MOE_GROUP_PROJECTIONS[group]:
            total += _slot_bytes(family, [units_by_part[(group, part, e)] for e in range(experts)],
                                 experts)
    return total


def _runtime_bytes(family: str, units_by_part: dict, experts: int, *,
                   fused_lane: bool = True) -> "tuple[int, int | None]":
    """``(bundles.resident_bytes(), fused lane bytes)`` off the runtime's own
    axis and bundles -- the figure the load bench reports -- where the grouped
    kernel modules import (they need triton).  ``fused_lane=False`` skips the
    fused lane's tables, for run tables that lane does not read."""
    pytest.importorskip("triton")
    from tessera.native_window_moe import PackedWindowMoeBundles, WindowUnitAxis
    from tessera.routed_fused import compose_table16, projection_tables
    from tessera.window_gemm_grouped import prepare_grouped_window_gemm_from_soa

    axes = {"w13": WindowUnitAxis(experts, ("gate_proj", "up_proj"), family=family),
            "w2": WindowUnitAxis(experts, ("down_proj",), family=family)}
    for (group, part, expert), unit in units_by_part.items():
        axes[group].put(part, expert, unit)
    soa = {g: axes[g].finish() for g in MOE_GROUPS}

    def bundle(group, part):
        slot = soa[group][part]
        return prepare_grouped_window_gemm_from_soa(
            words_all=slot["words"], table_all=slot["table"], codes_all=slot["codes"],
            native_all=slot["native"], scale_all=slot["scale"], runs_all=slot["runs"],
            init_all=slot["init"], has_init=slot["has_init"], word_off=slot["word_off"],
            tile_words=slot["tile_words"], total_words=slot["total_words"],
            run_off=slot["run_off"], perm_all=slot["perm"], rows=slot["rows"],
            cols=slot["cols"], experts=experts, window_bits=slot["window_bits"],
            family=family, arithmetic="folded" if family == "value" else "epilogue")

    bundles = PackedWindowMoeBundles(gate=bundle("w13", "gate_proj"), up=bundle("w13", "up_proj"),
                                     down=bundle("w2", "down_proj"), family=family)
    if not fused_lane:
        return bundles.resident_bytes(), None
    # What ``FusedRoutedWindowMoE.resident_bytes`` adds, built by the helpers
    # ``from_bundles`` builds it with: per projection the composed table and,
    # since contract v45, the run pair and the block descriptors.
    fused = 0
    for b in (bundles.gate, bundles.up, bundles.down):
        runs, bdesc, _tile_words, _slot_words = projection_tables(b)
        fused += (compose_table16(b).numel() * 2 + runs.numel() * runs.element_size()
                  + bdesc.numel() * bdesc.element_size())
    return bundles.resident_bytes(), fused


def _stack_units(lane: str, experts: int, rows_of, rates_of, bits=WINDOW_BITS) -> dict:
    return {(group, part, expert): _unit(lane, rows_of(group), rates_of(group), bits)
            for group in MOE_GROUPS for part in MOE_GROUP_PROJECTIONS[group]
            for expert in range(experts)}


@pytest.mark.parametrize("anchor", ["reference", "runtime"])
@pytest.mark.parametrize("family", ["TESSERA_FP8", "TESSERA_BF16"])
@pytest.mark.parametrize("rows,rates", [(513, (1, 3, 5, 8)), (512, (4, 4, 4)), (19, (2, 7))])
def test_unit_pricing_is_the_axis_allocation(family, rows, rates, anchor):
    """Per unit and per part, the formula is ``WindowUnitAxis`` plus ``finish``."""
    experts = 3
    lane = FAMILY_OF[family]
    units = _stack_units(lane, experts, lambda _g: rows, lambda _g: rates)
    if anchor == "runtime":
        # None of these shapes is one the fused lane reads (four rates;
        # three columns; two rates that are not adjacent), so only the
        # compact planes are compared here.
        actual, _fused = _runtime_bytes(lane, units, experts, fused_lane=False)
    else:
        actual = _reference_bytes(lane, units, experts)
    per_unit = routed_window_unit_resident_bytes(
        family, rows, len(rates), rates, window_bits=WINDOW_BITS, tile_rows=kg.TILE_ROWS)
    assert actual == 3 * experts * per_unit + 3 * routed_window_part_resident_bytes(experts)
    # And it is not the decoded tile the exporter used to charge.
    assert per_unit != rows * len(rates) + rows * 4


def test_unit_pricing_refuses_what_the_lane_cannot_read():
    with pytest.raises(ValueError, match="window layout"):
        routed_window_unit_resident_bytes("TESSERA_FP8", 8, 2, (4, 4), window_bits=0, tile_rows=512)
    with pytest.raises(ValueError, match="geometry"):
        routed_window_unit_resident_bytes("TESSERA_FP8", 8, 3, (4, 4), window_bits=14, tile_rows=512)
    with pytest.raises(ValueError, match="family"):
        routed_window_unit_resident_bytes("TESSERA_NVFP4", 8, 2, (4, 4), window_bits=14, tile_rows=512)


def test_fused_shape_predicate_reads_the_manifest_alone():
    ok = dict(rows=128, cols=128, rates=(RATE_4,) * 128, window_bits=WINDOW_BITS)
    assert fused_routed_unit_shape_refusal("e4m3", "gate", **ok) is None
    assert fused_routed_unit_shape_refusal("value", "down", **ok) is None
    assert "columns" in fused_routed_unit_shape_refusal("e4m3", "down", **{**ok, "cols": 64, "rates": (RATE_4,) * 64})
    assert "window_bits" in fused_routed_unit_shape_refusal("e4m3", "up", **{**ok, "window_bits": 13})
    assert "hidden size" in fused_routed_unit_shape_refusal("value", "down", **{**ok, "rows": 192})
    assert "intermediate size" in fused_routed_unit_shape_refusal("value", "gate", **{**ok, "rows": 96})
    assert "column rates" in fused_routed_unit_shape_refusal("e4m3", "gate", **{**ok, "rates": (4,) * 96})
    # Contract v45 (tessera#694): one rate or two ADJACENT rates, in any
    # column order (the packer sorts them into runs).
    assert fused_routed_unit_shape_refusal("e4m3", "up", **{**ok, "rates": (3,) * 128}) is None
    assert fused_routed_unit_shape_refusal("e4m3", "gate", **{**ok, "rates": (3, 4) * 64}) is None
    assert fused_routed_unit_shape_refusal("value", "down", **{**ok, "rates": (5,) * 32 + (4,) * 96}) is None
    assert "not adjacent" in fused_routed_unit_shape_refusal("e4m3", "up", **{**ok, "rates": (2, 4) * 64})
    assert "3 runs" in fused_routed_unit_shape_refusal("e4m3", "up", **{**ok, "rates": (3, 4, 5, 4) * 32})
    assert "run table" in fused_routed_unit_shape_refusal("e4m3", "up", **{**ok, "rates": (9,) * 128})
    # The shared-memory inequality per launch, derived as the runtime derives
    # it: the two-table gate/up launch reaches the published routed set, the
    # one-table down launch every rate the kernel decodes.
    for rate in range(1, 9):
        uniform = {**ok, "rates": (rate,) * 128}
        for part in ("gate", "up"):
            refusal = fused_routed_unit_shape_refusal("e4m3", part, **uniform)
            assert (refusal is None) == (rate in routed_fused.ROUTED_LANE_RATES), (part, rate, refusal)
            if refusal is not None:
                assert "shared memory" in refusal
        assert fused_routed_unit_shape_refusal("e4m3", "down", **uniform) is None, rate
    assert routed_fused.ROUTED_LANE_RATES == (1, 2, 3, 4, 5, 6)
    assert routed_fused_table_bytes(WINDOW_BITS) == 2 * (1 << WINDOW_BITS)


def test_fused_unit_pricing_is_the_lane_tables():
    """Per unit: the composed table, the int32 [8] run pair and one int32
    [12] descriptor per 32 columns -- the constants restated in the torch-free
    ``serving_parts`` are the lane's own."""
    assert serving_parts.ROUTED_FUSED_BLOCK_COLS == routed_fused.BK
    assert serving_parts.ROUTED_FUSED_BDESC_INTS == routed_fused.BDESC_INTS
    runs, _ok = routed_fused.run_pair(torch.tensor([[4, 0, 128, 0]], dtype=torch.int32), 128)
    assert runs.numel() == serving_parts.ROUTED_FUSED_RUN_PAIR_INTS
    assert routed_fused_unit_bytes(WINDOW_BITS, 128) == 2 * (1 << WINDOW_BITS) + 4 * 8 + 4 * 12 * 4
    assert routed_fused_unit_bytes(WINDOW_BITS, 4096) == 2 * (1 << WINDOW_BITS) + 32 + 48 * 128
    with pytest.raises(ValueError, match="column"):
        routed_fused_unit_bytes(WINDOW_BITS, 100)


def _stack_layouts(rates_w13, rates_w2) -> list:
    """The write loop's per-unit layout records for an ``EXPERTS``-deep stack."""
    layouts = []
    for group in MOE_GROUPS:
        for part in MOE_GROUP_PROJECTIONS[group]:
            rates = tuple(rates_w13 if group == "w13" else rates_w2)
            for _expert in range(EXPERTS):
                layouts.append({"group": group, "projection": part, "rows": INTER if group == "w13" else HIDDEN,
                                "cols": len(rates), "rates": rates, "window_bits": WINDOW_BITS})
    return layouts


def test_a_mixed_rate_stack_is_priced_with_the_fused_lane_tables():
    """Contract v45: a stack of two adjacent rates (q256 896 as the packer
    mixes rates 3 and 4) takes the fused lane, so its per-stack figure holds
    every unit's composed table, run pair and descriptors; a stack whose
    gate/up launch the target cannot fit (rate 7) keeps the compact lane
    alone.  At v44 both were priced as compact-only."""
    part_bytes = 3 * routed_window_part_resident_bytes(EXPERTS)
    mixed = (3, 4) * 64
    _units, stack = export.routed_stack_resident_bytes("TESSERA_FP8", EXPERTS, _stack_layouts(mixed, mixed))
    assert stack == part_bytes + 3 * EXPERTS * routed_fused_unit_bytes(WINDOW_BITS, 128)
    _units, stack = export.routed_stack_resident_bytes(
        "TESSERA_BF16", EXPERTS, _stack_layouts((7,) * 128, (7,) * 128))
    assert stack == part_bytes


def _accepts_fit_flag() -> bool:
    """Whether this exporter's argument parser (built inside ``main``) knows
    ``--fit-tp-size``; read off the source since the parser is not exported."""
    return "--fit-tp-size" in inspect.getsource(export)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """One CPU export of a 2-expert 128x128 E4M3 stack at q256 1024, fit at TP2."""
    root = tmp_path_factory.mktemp("routed-pricing")
    saved = (moe_write.HIDDEN, moe_write.MOE_INTER)
    moe_write.HIDDEN, moe_write.MOE_INTER = HIDDEN, INTER
    try:
        tensors = moe_write._checkpoint(experts=EXPERTS, inter=INTER)
        config = moe_write._config()
    finally:
        moe_write.HIDDEN, moe_write.MOE_INTER = saved
    config["text_config"].update(n_routed_experts=EXPERTS, moe_intermediate_size=INTER,
                                 num_nextn_predict_layers=1)
    plan = {STACK: {"grid": "E4M3", "q256": 1024}}
    # An exporter without ``--fit-tp-size`` still exports; its manifest then
    # lacks the per-rank block, which is the failure the line-item test names.
    fit = ("--fit-tp-size", str(FIT_TP)) if _accepts_fit_flag() else ()
    with pytest.MonkeyPatch.context() as monkeypatch:
        out = moe_write._export(root, monkeypatch, tensors, plan, "--device", "cpu",
                                *fit, config=config)
    manifest = json.loads((out / "tessera_serving_manifest.json").read_text())
    with safetensors_torch.safe_open(str(out / "model.safetensors"), framework="pt") as handle:
        wires = {key: handle.get_tensor(key).numpy().tobytes()
                 for key in handle.keys() if key.startswith(STACK) and key.endswith(".wire")}
    return manifest, wires, tensors


def _layouts(wires: dict) -> dict:
    """``(group, part, expert) -> (rows, cols, rates, window_bits)`` off the wires themselves."""
    layouts = {}
    for group in MOE_GROUPS:
        for part in MOE_GROUP_PROJECTIONS[group]:
            for expert in range(EXPERTS):
                (wire,) = parse_compact_expert(wires[f"{STACK}.{expert}.{part}.wire"], device="cpu")
                meta = wire.metadata
                layouts[(group, part, expert)] = (
                    meta.rows, meta.columns, tuple(meta.rates), int(meta.manifest.window_bits))
    return layouts


@pytest.mark.parametrize("anchor", ["reference", "runtime"])
def test_routed_stack_bytes_are_the_compact_lane_allocation_plus_compose_tables(exported, anchor):
    """The manifest's TP1 figure equals what the loader's axes hold for these
    wires, plus the fused lane's three composed tables per expert and, since
    contract v45, each projection's run pair and block descriptors."""
    manifest, wires, _tensors = exported
    record = manifest["modules"][STACK]
    assert record["family"] == "TESSERA_FP8"
    layouts = _layouts(wires)
    units = {key: _unit("e4m3", rows, rates, bits) for key, (rows, _c, rates, bits) in layouts.items()}
    if anchor == "runtime":
        planes, tables = _runtime_bytes("e4m3", units, EXPERTS)
    else:
        planes = _reference_bytes("e4m3", units, EXPERTS)
        # Per unit: the 2^L-entry 16-bit table, the int32 [8] run pair and
        # one 48-byte descriptor per 32 columns.
        tables = sum(2 * (1 << bits) + 4 * 8 + 48 * (cols // 32)
                     for _rows, cols, _rates, bits in layouts.values())
    assert record["resident_bytes_resident_mode"] == planes + tables
    # The whole stack admits the fused lane at TP1 (128 columns, rate 4, L=14).
    for (group, part, _e), (rows, cols, rates, bits) in layouts.items():
        assert fused_routed_unit_shape_refusal(
            "e4m3", "down" if group == "w2" else part.removesuffix("_proj"),
            rows=rows, cols=cols, rates=rates, window_bits=bits) is None
    # The figure the exporter used to write: one decoded byte per parameter
    # and an fp32 row scale per unit, which no compact-lane tensor is.
    decoded_tile = sum(rows * cols + rows * 4 for rows, cols, _r, _b in layouts.values())
    assert record["resident_bytes_resident_mode"] != decoded_tile
    assert manifest["totals"]["resident_mode_bytes"] == sum(
        m["resident_bytes_resident_mode"] for m in manifest["modules"].values())


@pytest.mark.parametrize("anchor", ["reference", "runtime"])
def test_mtp_duplicate_is_its_own_line_item_and_the_total_is_the_sum(exported, anchor):
    manifest, wires, tensors = exported
    block = manifest["totals"]["per_rank"]
    assert block["tp_size"] == FIT_TP and block["mtp_draft_layers"] == 1
    assert [r["rank"] for r in block["ranks"]] == list(range(FIT_TP))
    # The duplicate: the target's embedding and head, vocab-parallel per rank.
    embed, head = tensors["model.embed_tokens.weight"], tensors["lm_head.weight"]
    duplicate = sum(vocab_parallel_rows(t.shape[0], FIT_TP) * t.shape[1] * t.element_size()
                    for t in (embed, head))
    assert duplicate > 0
    # The routed stacks per rank: each unit's rank-local cut through the same
    # axis allocation -- NOT the TP1 figure halved, since tables, permutations
    # and bookkeeping are per expert on every rank, and the down projection's
    # 64-column cut refuses the fused lane's tables.
    layouts = _layouts(wires)
    for rank, row in enumerate(block["ranks"]):
        units = {}
        for (group, part, expert), (rows, cols, rates, bits) in layouts.items():
            if group == "w13":
                units[(group, part, expert)] = _unit("e4m3", rows // FIT_TP, rates, bits)
            else:
                local = cols // FIT_TP
                units[(group, part, expert)] = _unit(
                    "e4m3", rows, rates[rank * local:(rank + 1) * local], bits)
        assert fused_routed_unit_shape_refusal(
            "e4m3", "down", rows=HIDDEN, cols=INTER // FIT_TP,
            rates=(RATE_4,) * (INTER // FIT_TP), window_bits=WINDOW_BITS) is not None
        if anchor == "runtime":
            routed, _tables = _runtime_bytes("e4m3", units, EXPERTS)
        else:
            routed = _reference_bytes("e4m3", units, EXPERTS)
        assert row["items"] == {"routed_moe_resident_mode_bytes": routed,
                                "mtp_draft_embed_head_duplicate_bytes": duplicate}
        assert row["total_bytes"] == sum(row["items"].values())
        assert routed != manifest["modules"][STACK]["resident_bytes_resident_mode"] // FIT_TP


def test_mtp_duplicate_is_the_issue_645_figure_and_absent_without_draft_layers():
    # GLM-5.3: vocab 154,880 x hidden 4096 in bf16 for both tables, TP2 --
    # the +1.18 GiB per rank tessera#645 measured at load peak.
    glm = [(154_880, 4096, 2), (154_880, 4096, 2)]
    assert mtp_draft_embed_head_duplicate_bytes(glm, 2) == 1_268_776_960
    assert mtp_draft_embed_head_duplicate_bytes(glm, 1) == 2 * 154_880 * 4096 * 2
    # The vocabulary is padded to 64 before the cut, as the runtime pads it.
    assert vocab_parallel_rows(100, 2) == 64
    block = per_rank_fit_items(tp_size=2, routed_bytes_by_rank=[10, 12],
                               mtp_duplicate_bytes=0, mtp_layers=0)
    assert [r["items"]["mtp_draft_embed_head_duplicate_bytes"] for r in block["ranks"]] == [0, 0]
    assert [r["total_bytes"] for r in block["ranks"]] == [10, 12]
    with pytest.raises(ValueError):
        per_rank_fit_items(tp_size=2, routed_bytes_by_rank=[10], mtp_duplicate_bytes=0, mtp_layers=0)


def test_glm_per_rank_pricing_matches_the_measured_load_bench():
    """tessera#624's measured per-rank retention (bench_routed_load, TP2, before
    #685 added the composed tables): layer 10 TESSERA_BF16 q256=1024 and layer
    43 TESSERA_FP8 q256=896, 288 experts, hidden 4096, intermediate 2048.
    The bench's ``resident_final`` is ``PackedWindowMoeBundles.resident_bytes``;
    the pricing lands within 3,468 bytes of it (3 x 4 x 289: one more int32
    ``[E + 1]``-sized row per part in that bench build), against a 3.9x
    overstatement by the decoded-tile figure the manifest used to carry."""
    experts, hidden, inter, tp = 288, 4096, 2048, 2

    def per_rank(family, rates_of):
        gate_up = 2 * routed_window_unit_resident_bytes(
            family, inter // tp, hidden, rates_of(hidden), window_bits=14, tile_rows=kg.TILE_ROWS)
        down = routed_window_unit_resident_bytes(
            family, hidden, inter // tp, rates_of(inter // tp), window_bits=14, tile_rows=kg.TILE_ROWS)
        return experts * (gate_up + down) + 3 * routed_window_part_resident_bytes(experts)

    bf16 = per_rank("TESSERA_BF16", lambda cols: (4,) * cols)
    assert abs(bf16 - 1_868_597_016) <= 3_468
    # q256=896 is 3.5 bits per parameter as two rates over the columns.
    fp8 = per_rank("TESSERA_FP8", lambda cols: (3, 4) * (cols // 2))
    assert abs(fp8 - 1_628_183_832) <= 3_468
    manifest_before = 7_257_194_496
    assert manifest_before // tp > 1.9 * bf16
