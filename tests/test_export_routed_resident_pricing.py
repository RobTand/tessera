"""Native class residency: retained storage, selected tables and per-rank items.

The paired owner test composes real CPU tables and descriptors. Only CUDA
admission, extension loading and stream/event resources are mocked; it never
executes a CPU serving substitute or reuses the owner's byte-counting walker.
"""
from __future__ import annotations

import importlib

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tessera import kernel_window_gemv as kg
from tessera import routed_fused, serving_parts
from tessera.compact_prep import parse_compact_expert
from tessera.expert_classes import build_expert_metadata, inverse_expert_ids
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera.serving.scheme import MOE_GROUPS, MOE_GROUP_PROJECTIONS
from tessera.window_geometry import TILE_ROWS
from window_pack_reference import pack_bitstream

export = importlib.import_module("tessera.export_serving")
WINDOW_BITS = routed_fused.WINDOW_BITS
FAMILY_OF = {"TESSERA_BF16": "value", "TESSERA_FP8": "e4m3"}
moe_write = importlib.import_module("test_export_moe_write")
safetensors_torch = pytest.importorskip("safetensors.torch")
STACK = moe_write.STACK
RATE_4 = 4
HIDDEN, INTER, EXPERTS, FIT_TP = 128, 256, 2, 2
fused_routed_unit_shape_refusal = routed_fused.fused_routed_unit_shape_refusal

routed_fused_unit_bytes = serving_parts.routed_fused_unit_bytes
routed_window_unit_resident_bytes = serving_parts.routed_window_unit_resident_bytes
vocab_parallel_rows = serving_parts.vocab_parallel_rows


def _preparation_part_bytes(experts):
    return 8 * (experts + 1)


def _slot_bytes(family: str, units: list, experts: int) -> int:
    """Raw preparation planes before the native owner retires its input metadata.

    The int64 run offsets hold E + 1 entries per projection.
    This control is not a current serving price.
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
    return experts * per_expert + 8 * (experts + 1)


def _reference_bytes(family: str, units_by_part: dict, experts: int) -> int:
    """Raw preparation tensors before the native owner retires its input planes."""
    total = 0
    for group in MOE_GROUPS:
        for part in MOE_GROUP_PROJECTIONS[group]:
            total += _slot_bytes(family, [units_by_part[(group, part, e)] for e in range(experts)],
                                 experts)
    return total


def _prepared_axes(family, units, experts):
    from tessera.native_window_moe import WindowUnitAxis

    axes = {group: WindowUnitAxis(experts, roles, family=family, word_runs={
        role: [(units[group, role, expert].rep.words.numel(),
                units[group, role, expert].rep.runs.shape[0]) for expert in range(experts)]
        for role in roles}) for group, roles in MOE_GROUP_PROJECTIONS.items()}
    for (group, part, expert), unit in units.items():
        axes[group].put(part, expert, unit)
    return {group: axis.finish() for group, axis in axes.items()}


def _prepared_stack(family, units, experts, expert_classes):
    from tessera.native_window_moe import PackedWindowMoeBundles
    from tessera.window_gemm_grouped import prepare_grouped_window_gemm_from_soa

    soa = _prepared_axes(family, units, experts)

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

    return PackedWindowMoeBundles(
        gate=bundle("w13", "gate_proj"), up=bundle("w13", "up_proj"),
        down=bundle("w2", "down_proj"), family=family, expert_classes=expert_classes)


@pytest.fixture
def native_cpu_control(monkeypatch):
    # Compose real CPU tables. Do not execute a CPU serving substitute.
    monkeypatch.setattr(routed_fused, "fused_routed_window_supported", lambda *_args: None)
    monkeypatch.setattr(routed_fused, "_ext", lambda _library: SimpleNamespace())
    monkeypatch.setattr(routed_fused, "_make_dispatch_resources", lambda _device, kernel: routed_fused._DispatchResources(
        streams=(object(), object()), ready=object(), finished=(object(), object()),
        empty=torch.empty(0, dtype=torch.float32), kernel=kernel))


def _native_runtime_bytes(family, units, experts, metadata, library):
    packed = _prepared_stack(family, units, experts, metadata["expert_classes"])
    adapter = packed.adapter()
    assert adapter.library == library
    inverse = torch.tensor(inverse_expert_ids(metadata["expert_ids"]), dtype=torch.int32)
    return _distinct_storage_bytes(
        [tensor for _name, tensor in packed.native_owner().named_tensors()] + [inverse])


def _native_reference_bytes(units, experts, metadata, library):
    element_bytes = 1 if routed_fused.library_mma8(library) else 2
    return (sum(_retained_unit_bytes(unit) + (1 << unit.window_bits) * element_bytes
                + 4 * 8 + 4 * routed_fused.BDESC_INTS * (unit.cols // routed_fused.BK)
                for unit in units.values()) + 4 * experts + 8 * len(metadata["expert_classes"]))


def _unit(family, rows, rates, bits=WINDOW_BITS):
    value = family == "value"
    return kg.WindowGemvUnit(
        rep=pack_bitstream(torch.zeros((rows, len(rates)), dtype=torch.int64), rates),
        table=torch.zeros(1 << bits, dtype=torch.bfloat16),
        scale=torch.ones(rows, dtype=torch.float32), window_bits=bits, plan=kg.Plan(),
        codes_of_state=None if value else torch.zeros(1 << bits, dtype=torch.uint8),
        native=None if value else torch.zeros(256, dtype=torch.uint8),
        family=family, initial_state=torch.zeros(len(rates), dtype=torch.int32))


def _retained_unit_bytes(unit):
    return (unit.rep.words.numel() * unit.rep.words.element_size()
            + unit.scale.numel() * unit.scale.element_size()
            + unit.initial_state.numel() * unit.initial_state.element_size() + 4)


@pytest.mark.parametrize("family,rows,rates", [
    ("TESSERA_BF16", 128, (4,) * 128),
    ("TESSERA_FP8", 64, (3, 4) * 64),
    ("TESSERA_FP8", 1024, (4,) * 256),
])
def test_native_unit_price_counts_retained_planes_only(family, rows, rates):
    unit = _unit(FAMILY_OF[family], rows, rates)
    priced = serving_parts.routed_window_unit_resident_bytes(
        family, rows, len(rates), rates, window_bits=WINDOW_BITS, tile_rows=TILE_ROWS)
    assert priced == _retained_unit_bytes(unit)


def test_unit_pricing_refuses_malformed_fields():
    for kwargs in ({"window_bits": 0}, {"cols": 3}, {"family": "TESSERA_NVFP4"}):
        arguments = dict(family="TESSERA_FP8", rows=128, cols=128, rates=(4,) * 128,
                         window_bits=WINDOW_BITS, tile_rows=TILE_ROWS)
        arguments.update(kwargs)
        with pytest.raises(ValueError):
            serving_parts.routed_window_unit_resident_bytes(**arguments)


def test_default_e4m3_mma_table_price_is_one_byte(monkeypatch):
    monkeypatch.setenv(routed_fused.ENV_E4M3_MMA, "e4m3")
    library = routed_fused.library_for("e4m3")
    assert serving_parts.routed_fused_table_bytes(WINDOW_BITS, element_bytes=1 if routed_fused.library_mma8(library) else 2) == 1 << WINDOW_BITS


@pytest.mark.parametrize("family,choice,element_bytes", [
    ("value", "e4m3", 2), ("e4m3", "e4m3", 1), ("e4m3", "f16", 2),
])
def test_fused_unit_pricing_uses_selected_table_and_descriptor_geometry(monkeypatch, family, choice, element_bytes):
    monkeypatch.setenv(routed_fused.ENV_E4M3_MMA, choice)
    library = routed_fused.library_for(family)
    for columns in (128, 4096):
        (runs, _) = routed_fused.run_pair(torch.tensor([[4, 0, columns, 0]], dtype=torch.int32), columns)
        expected = ((1 << WINDOW_BITS) * element_bytes + runs.numel() * runs.element_size()
                    + (columns // routed_fused.BK) * routed_fused.BDESC_INTS * 4)
        assert serving_parts.routed_fused_unit_bytes(WINDOW_BITS, columns, table_element_bytes=1 if routed_fused.library_mma8(library) else 2) == expected
    with pytest.raises(ValueError, match="column"):
        serving_parts.routed_fused_unit_bytes(WINDOW_BITS, 100, table_element_bytes=1 if routed_fused.library_mma8(library) else 2)


def _stack_layouts(rungs, *, tp_size=1, tp_rank=0):
    layouts = []
    units = {}
    for group, projections in MOE_GROUP_PROJECTIONS.items():
        rows, columns = (512, 128) if group == "w13" else (128, 512)
        for projection in projections:
            for expert, rung in enumerate(rungs):
                rates = bresenham_rate_schedule(root_from_q256(rung), columns, cap=8)
                full = {"group": group, "projection": projection, "rows": rows, "cols": columns,
                        "rates": rates, "window_bits": WINDOW_BITS}
                layouts.append(full)
                units[group, projection, expert] = export.routed_unit_rank_cut(full, tp_size, tp_rank)
    matrices = {"w13": [[q, q] for q in rungs], "w2": [[q] for q in rungs]}
    return layouts, units, build_expert_metadata(matrices)


@pytest.mark.parametrize("family,choice,element_bytes", [
    ("TESSERA_BF16", "e4m3", 2), ("TESSERA_FP8", "e4m3", 1), ("TESSERA_FP8", "f16", 2),
])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_different_expert_class_schedules_retain_all_native_storage(monkeypatch, family, choice, element_bytes, tp_size):
    monkeypatch.setenv(routed_fused.ENV_E4M3_MMA, choice)
    library = routed_fused.library_for(FAMILY_OF[family])
    layouts, cuts, metadata = _stack_layouts((896, 1024), tp_size=tp_size)
    priced = export.routed_stack_resident_bytes(
        family, 2, layouts, expert_classes=metadata["expert_classes"], native_library=library,
        tp_size=tp_size, tp_rank=0)
    planes, tables = 0, 0
    for cut in cuts.values():
        unit = _unit(FAMILY_OF[family], cut["rows"], cut["rates"])
        planes += _retained_unit_bytes(unit)
        tables += ((1 << WINDOW_BITS) * element_bytes + 4 * 8
                   + 4 * routed_fused.BDESC_INTS * (cut["cols"] // routed_fused.BK))
    assert priced == (planes, tables + 4 * 2 + 8 * 2)


def _distinct_storage_bytes(tensors):
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        storages[key] = storage.nbytes()
    return sum(storages.values())


@pytest.mark.parametrize("family,choice,element_bytes", [
    ("TESSERA_BF16", "e4m3", 2), ("TESSERA_FP8", "e4m3", 1), ("TESSERA_FP8", "f16", 2),
])
@pytest.mark.parametrize("rungs", [(1024, 1024), (896, 1024)])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_actual_native_owner_matches_exported_accounting(monkeypatch, native_cpu_control, family, choice, element_bytes, rungs, tp_size):
    monkeypatch.setenv(routed_fused.ENV_E4M3_MMA, choice)
    lane = FAMILY_OF[family]
    layouts, cuts, metadata = _stack_layouts(rungs, tp_size=tp_size)
    units = {key: _unit(lane, cut["rows"], cut["rates"]) for key, cut in cuts.items()}
    packed = _prepared_stack(lane, units, 2, metadata["expert_classes"])
    adapter = packed.adapter()
    native = packed.native_owner()
    assert all(table.element_size() == element_bytes
               for table in (adapter.table_gate, adapter.table_up, adapter.table_down))
    inverse = torch.tensor(inverse_expert_ids(metadata["expert_ids"]), dtype=torch.int32)
    actual = _distinct_storage_bytes([tensor for _name, tensor in native.named_tensors()] + [inverse])
    priced = export.routed_stack_resident_bytes(
        family, 2, layouts, expert_classes=metadata["expert_classes"], native_library=adapter.library,
        tp_size=tp_size, tp_rank=0)
    assert sum(priced) == actual


def test_fused_shape_predicate_reads_native_geometry_and_schedule():
    ok = dict(rows=128, cols=128, rates=(4,) * 128, window_bits=WINDOW_BITS)
    assert routed_fused.fused_routed_unit_shape_refusal("e4m3", "gate", **ok) is None
    assert routed_fused.fused_routed_unit_shape_refusal("value", "down", **ok) is None
    assert routed_fused.fused_routed_unit_shape_refusal("e4m3", "down", **{**ok, "cols": 64, "rates": (4,) * 64})
    assert routed_fused.fused_routed_unit_shape_refusal("value", "down", **{**ok, "rows": 192})
    for rates in ((3,) * 128, (3, 4) * 64):
        assert routed_fused.fused_routed_unit_shape_refusal("e4m3", "up", **{**ok, "rates": rates}) is None
    for rates in ((2, 4) * 64, (3, 4, 5, 4) * 32, (9,) * 128):
        assert routed_fused.fused_routed_unit_shape_refusal("e4m3", "up", **{**ok, "rates": rates})


def _stack_units(lane: str, experts: int, rows_of, rates_of, bits=WINDOW_BITS) -> dict:
    return {(group, part, expert): _unit(lane, rows_of(group), rates_of(group), bits)
            for group in MOE_GROUPS for part in MOE_GROUP_PROJECTIONS[group]
            for expert in range(experts)}


@pytest.mark.parametrize("anchor", ["reference", "runtime"])
@pytest.mark.parametrize("family", ["TESSERA_FP8", "TESSERA_BF16"])
@pytest.mark.parametrize("rows,rates", [(513, (1, 3, 5, 8)), (512, (4, 4, 4)), (19, (2, 7))])
def test_unit_pricing_is_the_axis_allocation(family, rows, rates, anchor):
    """The unit price plus retired input planes equals the preparation allocation."""
    experts = 3
    lane = FAMILY_OF[family]
    units = _stack_units(lane, experts, lambda _g: rows, lambda _g: rates)
    if anchor == "runtime":
        soa = _prepared_axes(lane, units, experts)
        actual = _distinct_storage_bytes([tensor for group in soa.values() for slot in group.values()
                                          for tensor in slot.values() if isinstance(tensor, torch.Tensor)])
    else:
        actual = _reference_bytes(lane, units, experts)
    per_unit = routed_window_unit_resident_bytes(
        family, rows, len(rates), rates, window_bits=WINDOW_BITS, tile_rows=TILE_ROWS)
    retired = 3 * _preparation_part_bytes(experts)
    for unit in units.values():
        grid = unit.table.numel() * 2 if lane == "value" else unit.codes_of_state.numel() + unit.native.numel()
        retired += unit.rep.runs.numel() * 4 + unit.rep.perm.numel() * 4 + grid + 3 * 4
    assert actual == 3 * experts * per_unit + retired
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
    # it: the two-table gate/up launch reaches the published routed set (every
    # rate since contract v49: rates 7 and 8 run at two word stages), the
    # one-table down launch every rate the kernel decodes.
    for rate in range(1, 9):
        uniform = {**ok, "rates": (rate,) * 128}
        for part in ("gate", "up"):
            refusal = fused_routed_unit_shape_refusal("e4m3", part, **uniform)
            assert (refusal is None) == (rate in routed_fused.ROUTED_LANE_RATES), (part, rate, refusal)
            if refusal is not None:
                assert "shared memory" in refusal
        assert fused_routed_unit_shape_refusal("e4m3", "down", **uniform) is None, rate



def test_fused_unit_pricing_is_the_lane_tables():
    """The explicit BF16 library owns the composed table and launch descriptors."""
    library = routed_fused.library_for("value")
    assert serving_parts.ROUTED_FUSED_BLOCK_COLS == routed_fused.BK
    assert serving_parts.ROUTED_FUSED_BDESC_INTS == routed_fused.BDESC_INTS
    runs, _ok = routed_fused.run_pair(torch.tensor([[4, 0, 128, 0]], dtype=torch.int32), 128)
    assert runs.numel() == serving_parts.ROUTED_FUSED_RUN_PAIR_INTS
    assert routed_fused_unit_bytes(WINDOW_BITS, 128, table_element_bytes=1 if routed_fused.library_mma8(library) else 2) == 2 * (1 << WINDOW_BITS) + 4 * 8 + 4 * 12 * 4
    assert routed_fused_unit_bytes(WINDOW_BITS, 4096, table_element_bytes=1 if routed_fused.library_mma8(library) else 2) == 2 * (1 << WINDOW_BITS) + 32 + 48 * 128
    with pytest.raises(ValueError, match="column"):
        routed_fused_unit_bytes(WINDOW_BITS, 100, table_element_bytes=1 if routed_fused.library_mma8(library) else 2)


def test_dense_pricing_follows_the_dense_loaders_rate_bound():
    def role(rate, bits=WINDOW_BITS):
        return [{"rows": 256, "cols": 128, "rates": [rate] * 128, "window_bits": bits, "tile_rows": 512}]

    priced = {rate: serving_parts.dense_resident_bytes_resident_mode(
        "TESSERA_BF16", 256, 128, native_roles=role(rate)) for rate in (8, 9, 14)}
    assert priced[9] - priced[8] == 512 * 128 // 8
    assert priced[14] - priced[9] == 5 * 512 * 128 // 8
    for rate, bits in ((15, WINDOW_BITS), (15, 16), (9, 8)):
        with pytest.raises(ValueError, match="native dense window layout"):
            serving_parts.dense_resident_bytes_resident_mode(
                "TESSERA_BF16", 256, 128, native_roles=role(rate, bits))


def _single_profile_stack_layouts(rates_w13, rates_w2) -> list:
    """The write loop's per-unit layout records for an ``EXPERTS``-deep stack."""
    layouts = []
    for group in MOE_GROUPS:
        for part in MOE_GROUP_PROJECTIONS[group]:
            rates = tuple(rates_w13 if group == "w13" else rates_w2)
            for _expert in range(EXPERTS):
                layouts.append({"group": group, "projection": part, "rows": INTER if group == "w13" else HIDDEN,
                                "cols": len(rates), "rates": rates, "window_bits": WINDOW_BITS})
    return layouts


def test_a_mixed_rate_stack_is_priced_with_the_fused_lane_tables(monkeypatch):
    """Mixed and high-rate stacks retain selected tables. Native admission fails closed."""
    for family, rung in (("TESSERA_FP8", 896), ("TESSERA_BF16", 1792)):
        metadata = build_expert_metadata({"w13": [[rung, rung]] * EXPERTS, "w2": [[rung]] * EXPERTS})
        rates_w13 = bresenham_rate_schedule(root_from_q256(rung), HIDDEN, cap=8)
        rates_w2 = bresenham_rate_schedule(root_from_q256(rung), INTER, cap=8)
        layouts = _single_profile_stack_layouts(rates_w13, rates_w2)
        library = routed_fused.library_for(FAMILY_OF[family])
        priced = export.routed_stack_resident_bytes(
            family, EXPERTS, layouts, expert_classes=metadata["expert_classes"], native_library=library)
        tables = sum(routed_fused_unit_bytes(WINDOW_BITS, layout["cols"], table_element_bytes=1 if routed_fused.library_mma8(library) else 2) for layout in layouts)
        assert priced[1] == tables + 4 * EXPERTS + 8 * len(metadata["expert_classes"])
    record = {"family": family, "stack": STACK, "expert_classes": metadata["expert_classes"],
              "groups": {"w13": {"rows_each": INTER, "columns": HIDDEN},
                         "w2": {"rows_each": HIDDEN, "columns": INTER}}}
    export.check_native_class_geometry(record, fit_tp_size=FIT_TP)
    monkeypatch.setattr(routed_fused, "SM121_MAX_DYNAMIC_SMEM", 0)
    with pytest.raises(SystemExit, match="unservable_native_class_geometry.*shared memory"):
        export.check_native_class_geometry(record, fit_tp_size=FIT_TP)
    # A smaller device limit cannot silently demote the price to a compact lane.
    assert export.routed_stack_resident_bytes(
        family, EXPERTS, layouts, expert_classes=metadata["expert_classes"], native_library=library) == priced

# The fixture declares its own dense and shared projection row partitions.
_FIXTURE_OUTPUT_SIZES = {
    "language_model.model.layers.*.mlp.down_proj": [HIDDEN],
    "language_model.model.layers.*.mlp.gate_up_proj": [2 * HIDDEN, 2 * HIDDEN],
    "language_model.model.layers.*.mlp.shared_experts.down_proj": [HIDDEN],
    "language_model.model.layers.*.mlp.shared_experts.gate_up_proj": [INTER, INTER],
}


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Export real E4M3 wires for two experts with native-valid TP2 cuts."""
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

    with pytest.MonkeyPatch.context() as monkeypatch:
        import copy
        from tessera.serving.contract import construction_entry as live_entry
        real = live_entry

        def _entry(architectures, contract=None):
            entry = real(architectures) if contract is None else real(architectures, contract)
            if entry is None or entry.get("architecture") != "Glm5NextForConditionalGeneration":
                return entry
            entry = copy.deepcopy(entry)
            entry.setdefault("output_sizes", {}).update(_FIXTURE_OUTPUT_SIZES)
            return entry

        monkeypatch.setenv(routed_fused.ENV_E4M3_MMA, "e4m3")
        monkeypatch.setattr(export, "construction_entry", _entry)
        out = moe_write._export(root, monkeypatch, tensors, plan, "--device", "cpu",
                                "--fit-tp-size", str(FIT_TP), config=config)
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
def test_routed_stack_bytes_are_the_native_owner_allocation(exported, anchor, monkeypatch, native_cpu_control):
    """Verified wires set the retained TP1 allocation, including the selected native table."""
    manifest, wires, _tensors = exported
    record = manifest["modules"][STACK]
    assert record["family"] == "TESSERA_FP8"
    layouts = _layouts(wires)
    units = {key: _unit("e4m3", rows, rates, bits) for key, (rows, _c, rates, bits) in layouts.items()}
    metadata = {key: record[key] for key in ("expert_ids", "expert_classes")}
    library = record["native_library"]
    monkeypatch.setenv(routed_fused.ENV_E4M3_MMA, "e4m3")
    if anchor == "runtime":
        actual = _native_runtime_bytes("e4m3", units, EXPERTS, metadata, library)
    else:
        actual = _native_reference_bytes(units, EXPERTS, metadata, library)
    assert record["resident_bytes_resident_mode"] == actual
    for (group, part, _e), (rows, cols, rates, bits) in layouts.items():
        assert fused_routed_unit_shape_refusal(
            "e4m3", "down" if group == "w2" else part.removesuffix("_proj"),
            rows=rows, cols=cols, rates=rates, window_bits=bits) is None
    decoded_tile = sum(rows * cols + rows * 4 for rows, cols, _r, _b in layouts.values())
    assert record["resident_bytes_resident_mode"] != decoded_tile
    assert manifest["totals"]["resident_mode_bytes"] == sum(
        module["resident_bytes_resident_mode"] for module in manifest["modules"].values())


@pytest.mark.parametrize("anchor", ["reference", "runtime"])
def test_mtp_duplicate_is_its_own_line_item_and_the_total_is_the_sum(exported, anchor, monkeypatch, native_cpu_control):
    manifest, wires, tensors = exported
    record = manifest["modules"][STACK]
    metadata = {key: record[key] for key in ("expert_ids", "expert_classes")}
    library = record["native_library"]
    monkeypatch.setenv(routed_fused.ENV_E4M3_MMA, "e4m3")
    block = manifest["totals"]["per_rank"]
    assert block["tp_size"] == FIT_TP and block["mtp_draft_layers"] == 1
    assert [row["rank"] for row in block["ranks"]] == list(range(FIT_TP))
    embed, head = tensors["model.embed_tokens.weight"], tensors["lm_head.weight"]
    duplicate = sum(vocab_parallel_rows(tensor.shape[0], FIT_TP) * tensor.shape[1] * tensor.element_size()
                    for tensor in (embed, head))
    assert duplicate > 0
    layouts = _layouts(wires)
    for rank, row in enumerate(block["ranks"]):
        units = {}
        for (group, part, expert), (rows, cols, rates, bits) in layouts.items():
            if group == "w13":
                units[group, part, expert] = _unit("e4m3", rows // FIT_TP, rates, bits)
            else:
                local = cols // FIT_TP
                units[group, part, expert] = _unit("e4m3", rows, rates[rank * local:(rank + 1) * local], bits)
        assert fused_routed_unit_shape_refusal(
            "e4m3", "down", rows=HIDDEN, cols=INTER // FIT_TP,
            rates=(RATE_4,) * (INTER // FIT_TP), window_bits=WINDOW_BITS) is None
        if anchor == "runtime":
            routed = _native_runtime_bytes("e4m3", units, EXPERTS, metadata, library)
        else:
            routed = _native_reference_bytes(units, EXPERTS, metadata, library)
        assert row["items"] == {"routed_moe_resident_mode_bytes": routed,
                                "mtp_draft_embed_head_duplicate_bytes": duplicate}
        assert row["total_bytes"] == sum(row["items"].values())
        assert routed != record["resident_bytes_resident_mode"] // FIT_TP


def test_mtp_duplicate_remains_separate_and_total_is_the_sum():
    glm = [(154_880, 4096, 2), (154_880, 4096, 2)]
    assert serving_parts.mtp_draft_embed_head_duplicate_bytes(glm, 2) == 1_268_776_960
    assert serving_parts.mtp_draft_embed_head_duplicate_bytes(glm, 1) == 2 * 154_880 * 4096 * 2
    assert serving_parts.vocab_parallel_rows(100, 2) == 64
    block = serving_parts.per_rank_fit_items(tp_size=2, routed_bytes_by_rank=[10, 12],
                                            mtp_duplicate_bytes=0, mtp_layers=0)
    assert [r["items"]["mtp_draft_embed_head_duplicate_bytes"] for r in block["ranks"]] == [0, 0]
    assert [r["total_bytes"] for r in block["ranks"]] == [10, 12]
    assert all(r["total_bytes"] == sum(r["items"].values()) for r in block["ranks"])
    with pytest.raises(ValueError):
        serving_parts.per_rank_fit_items(tp_size=2, routed_bytes_by_rank=[10], mtp_duplicate_bytes=0, mtp_layers=0)


def test_glm_per_rank_pricing_matches_the_measured_load_bench():
    """Keep issue 624's historical preparation bytes distinct from current native retention.

    The old TP2 load retained raw grid, run, permutation and offset planes.
    Add those retired inputs to the current unit price to recover both measured figures.
    Native tables, inverse maps and class counters have separate owner proofs above.
    """
    experts, hidden, inter, tp = 288, 4096, 2048, 2

    def prepared_unit(family, rows, cols, rates):
        retained = routed_window_unit_resident_bytes(
            family, rows, cols, rates, window_bits=WINDOW_BITS, tile_rows=TILE_ROWS)
        raw_grid = (1 << WINDOW_BITS) * (2 if family == "TESSERA_BF16" else 1)
        if family == "TESSERA_FP8":
            raw_grid += 256
        return retained + raw_grid + 16 * len(set(rates)) + 4 * cols + 3 * 4

    def per_rank(family, rates_of):
        gate_up = 2 * prepared_unit(family, inter // tp, hidden, rates_of(hidden))
        down = prepared_unit(family, hidden, inter // tp, rates_of(inter // tp))
        return experts * (gate_up + down) + 3 * _preparation_part_bytes(experts)

    bf16 = per_rank("TESSERA_BF16", lambda cols: (4,) * cols)
    assert bf16 == 1_868_597_016
    fp8 = per_rank("TESSERA_FP8", lambda cols: (3, 4) * (cols // 2))
    assert fp8 == 1_628_183_832
    manifest_before = 7_257_194_496
    assert manifest_before // tp > 1.9 * bf16
