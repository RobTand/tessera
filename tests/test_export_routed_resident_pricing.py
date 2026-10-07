"""Native class residency: retained storage, selected tables and per-rank items.

The paired owner test composes real CPU tables and descriptors. Only CUDA
admission, extension loading and stream/event resources are mocked; it never
executes a CPU serving substitute or reuses the owner's byte-counting walker.
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tessera import kernel_window_gemv as kg
from tessera import routed_fused, serving_parts
from tessera.expert_classes import build_expert_metadata, inverse_expert_ids
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera.serving.scheme import MOE_GROUP_PROJECTIONS
from window_pack_reference import pack_bitstream

export = importlib.import_module("tessera.export_serving")
WINDOW_BITS = routed_fused.WINDOW_BITS
FAMILY_OF = {"TESSERA_BF16": "value", "TESSERA_FP8": "e4m3"}


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
        family, rows, len(rates), rates, window_bits=WINDOW_BITS, tile_rows=kg.TILE_ROWS)
    assert priced == _retained_unit_bytes(unit)


def test_unit_pricing_refuses_malformed_fields():
    for kwargs in ({"window_bits": 0}, {"cols": 3}, {"family": "TESSERA_NVFP4"}):
        arguments = dict(family="TESSERA_FP8", rows=128, cols=128, rates=(4,) * 128,
                         window_bits=WINDOW_BITS, tile_rows=kg.TILE_ROWS)
        arguments.update(kwargs)
        with pytest.raises(ValueError):
            serving_parts.routed_window_unit_resident_bytes(**arguments)


def test_default_e4m3_mma_table_price_is_one_byte(monkeypatch):
    monkeypatch.setenv(routed_fused.ENV_E4M3_MMA, "e4m3")
    library = routed_fused.library_for("e4m3")
    assert serving_parts.routed_fused_table_bytes(WINDOW_BITS, library=library) == 1 << WINDOW_BITS


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
        assert serving_parts.routed_fused_unit_bytes(WINDOW_BITS, columns, library=library) == expected
    with pytest.raises(ValueError, match="column"):
        serving_parts.routed_fused_unit_bytes(WINDOW_BITS, 100, library=library)


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
def test_actual_native_owner_matches_exported_accounting(monkeypatch, family, choice, element_bytes, rungs, tp_size):
    # Requires the integrated runtime slice; no legacy owner or skip fallback.
    from tessera.native_window_moe import PackedWindowMoeBundles, WindowUnitAxis
    from tessera.window_gemm_grouped import prepare_grouped_window_gemm_from_soa

    monkeypatch.setenv(routed_fused.ENV_E4M3_MMA, choice)
    monkeypatch.setattr(routed_fused, "fused_routed_window_supported", lambda *_args: None)
    monkeypatch.setattr(routed_fused, "_ext", lambda _library: SimpleNamespace())
    monkeypatch.setattr(routed_fused, "_make_dispatch_resources", lambda _device, kernel: routed_fused._DispatchResources(
        streams=(object(), object()), ready=object(), finished=(object(), object()),
        empty=torch.empty(0, dtype=torch.float32), kernel=kernel))
    lane = FAMILY_OF[family]
    layouts, cuts, metadata = _stack_layouts(rungs, tp_size=tp_size)
    axes = {group: WindowUnitAxis(2, roles, family=lane) for group, roles in MOE_GROUP_PROJECTIONS.items()}
    for (group, projection, expert), cut in cuts.items():
        unit = _unit(lane, cut["rows"], cut["rates"])
        axes[group].put(projection, expert, unit)
    soa = {group: axis.finish() for group, axis in axes.items()}

    def bundle(group, projection):
        slot = soa[group][projection]
        return prepare_grouped_window_gemm_from_soa(
            words_all=slot["words"], table_all=slot["table"], codes_all=slot["codes"],
            native_all=slot["native"], scale_all=slot["scale"], runs_all=slot["runs"],
            init_all=slot["init"], has_init=slot["has_init"], word_off=slot["word_off"],
            tile_words=slot["tile_words"], total_words=slot["total_words"],
            run_off=slot["run_off"], perm_all=slot["perm"], rows=slot["rows"], cols=slot["cols"],
            experts=2, window_bits=slot["window_bits"], family=lane,
            arithmetic="folded" if lane == "value" else "epilogue")

    packed = PackedWindowMoeBundles(
        gate=bundle("w13", "gate_proj"), up=bundle("w13", "up_proj"),
        down=bundle("w2", "down_proj"), family=lane, expert_classes=metadata["expert_classes"])
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
