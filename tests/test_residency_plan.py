"""Hand-computed load peaks and named offline refusals."""
from __future__ import annotations

import copy
import importlib
import json
import os
import subprocess
import sys

import pytest


@pytest.fixture
def planner():
    try:
        return importlib.import_module("tessera.residency_plan")
    except ModuleNotFoundError as exc:
        if exc.name != "tessera.residency_plan":
            raise
        pytest.fail("tessera.residency_plan is not published")


def spec(**tensors):
    return {"tensors": {name: {"shape": shape, "dtype": dtype}
                        for name, (shape, dtype) in tensors.items()}}


def allocation(name, tensor=None, *, ranks=(0,), start=0, stop=None, **extra):
    return {"id": name, "tensor": tensor or name, "ranks": list(ranks),
            "start": start, "stop": stop, **extra}


def plan(*allocations, capacities=(1000,), reserves=None):
    reserves = reserves or (0,) * len(capacities)
    return {"ranks": [{"capacity_bytes": c, "reserve_bytes": r}
                      for c, r in zip(capacities, reserves)],
            "allocations": list(allocations)}


def test_peak_includes_source_output_and_workspace(planner):
    structure = spec(weight=([4, 8], "bfloat16"), output=([4, 8], "uint8"),
                     scratch=([8], "float32"))
    placement = plan(allocation("source", "weight", stop=2),
                     allocation("output", start=1), allocation("scratch", start=1, stop=2),
                     capacities=(144,), reserves=(16,))
    original = copy.deepcopy((structure, placement))
    result = planner.plan_residency(structure, placement)
    rank = result["ranks"][0]
    assert rank["peak_bytes"] == 16 + 64 + 32 + 32 == 144
    assert rank["peak_step"] == 1
    assert rank["peak_allocations"] == {"source": 64, "output": 32, "scratch": 32}
    assert rank["final_bytes"] == 16 + 32 == 48
    assert rank["headroom_bytes"] == 0
    assert result["fits"] is True
    assert (structure, placement) == original


def test_rank_placement_and_row_column_shards(planner):
    structure = spec(rows=([6, 8], "float16"), cols=([4, 10], "float32"),
                     norm=([4], "float32"), draft=([5, 8], "bfloat16"))
    placement = plan(allocation("rows", ranks=(0, 1), shard_axis=0),
                     allocation("cols", ranks=(0, 1), shard_axis=1),
                     allocation("norm", ranks=(0, 1)), allocation("draft", ranks=(1,)),
                     capacities=(144, 224))
    ranks = planner.plan_residency(structure, placement)["ranks"]
    assert [r["peak_bytes"] for r in ranks] == [48 + 80 + 16, 48 + 80 + 16 + 80]


def test_lifetimes_are_half_open(planner):
    structure = spec(a=([10], "int64"), b=([10], "int64"))
    result = planner.plan_residency(structure, plan(allocation("a", stop=1),
                                                  allocation("b", start=1), capacities=(80,)))
    assert result["ranks"][0]["peak_bytes"] == 80
    assert result["ranks"][0]["peak_step"] == 0
    assert result["ranks"][0]["final_bytes"] == 80


def test_vocab_padding_counts_each_rank(planner):
    result = planner.plan_residency(spec(head=([65, 4], "bfloat16")),
                                    plan(allocation("head", ranks=(0, 1), shard_axis=0,
                                                    padding_multiple=64), capacities=(512, 512)))
    assert [r["peak_bytes"] for r in result["ranks"]] == [64 * 4 * 2] * 2


def test_capacity_refusal_names_every_rank_and_peak(planner):
    placement = plan(allocation("weight", ranks=(0, 1)), capacities=(63, 50), reserves=(1, 2))
    with pytest.raises(planner.ResidencyRefusal) as caught:
        planner.plan_residency(spec(weight=([4, 8], "bfloat16")), placement)
    report = caught.value.report
    assert report["fits"] is False
    assert [(r["rank"], r["code"], r["excess_bytes"]) for r in report["reasons"]] == [
        (0, "capacity_exceeded", 2), (1, "capacity_exceeded", 16)]
    assert report["ranks"][0]["peak_allocations"] == {"weight": 64}
    assert "rank 0" in str(caught.value) and "rank 1" in str(caught.value)


@pytest.mark.parametrize("dtype,width", [("bool", 1), ("uint8", 1), ("int8", 1),
    ("float8_e4m3fn", 1), ("float8_e5m2", 1), ("uint16", 2), ("int16", 2),
    ("float16", 2), ("bfloat16", 2), ("uint32", 4), ("int32", 4), ("float32", 4),
    ("uint64", 8), ("int64", 8), ("float64", 8)])
def test_dtype_bytes_and_scalar_shape(planner, dtype, width):
    report = planner.plan_residency(spec(value=([], dtype)), plan(allocation("value")))
    assert report["ranks"][0]["peak_bytes"] == width


def test_native_dense_uses_rank_local_padding_and_tables(planner):
    from tessera.window_geometry import TILE_ROWS
    storage = {"kind": "dense_window", "family": "TESSERA_FP8", "rates": [4] * 32,
               "window_bits": 8, "tile_rows": TILE_ROWS}
    placement = plan(allocation("dense", ranks=(0, 1), shard_axis=0, storage=storage),
                     capacities=(9232, 9232))
    report = planner.plan_residency(spec(dense=([64, 32], "bfloat16")), placement)
    # Each rank has 32 rows. The compact loader pads these rows to its row tile.
    expected = TILE_ROWS * 32 * 4 // 8 + 256 + 256 + 32 * 4 + 16 + 32 * 8 + 32 * 4
    assert [r["peak_bytes"] for r in report["ranks"]] == [expected] * 2
    assert expected == 9232


def test_native_routed_slices_rates_in_placement_order(planner):
    from tessera.window_geometry import TILE_ROWS
    storage = {"kind": "routed_window", "family": "TESSERA_FP8", "rates": [3] * 32 + [5] * 32,
               "window_bits": 8, "tile_rows": TILE_ROWS, "fused": True}
    placement = plan(allocation("experts", ranks=(1, 0), shard_axis=2, storage=storage),
                     capacities=(23800, 15608))
    report = planner.plan_residency(spec(experts=([2, 64, 64], "bfloat16")), placement)
    def expected(rate):
        unit = TILE_ROWS * 32 * rate // 8 + 256 + 256 + 64 * 4 + 16 + 32 * 8 + 16
        fused = 2 * 256 + 4 * 8 + 4 * 12
        return 2 * (unit + fused) + 8 * (2 + 1)
    assert [r["peak_bytes"] for r in report["ranks"]] == [expected(5), expected(3)]


def test_native_a4_uses_shared_accountant(planner):
    storage = {"kind": "dense_a4", "rates": [2] * 32,
               "arity": 2, "memory": 1, "half": 16, "lut_entries": 16}
    report = planner.plan_residency(spec(weight=([32, 32], "bfloat16")),
                                    plan(allocation("weight", storage=storage)))
    # Global scale, select, label, point, scale nibbles, tables, role epilogue.
    assert report["ranks"][0]["peak_bytes"] == 4 + 72 + 64 + 64 + 32 + 16 + 16 + 24 + 4


def test_empty_inventory_has_only_reserve(planner):
    report = planner.plan_residency(spec(), plan(capacities=(30,), reserves=(30,)))
    assert report["ranks"][0]["peak_bytes"] == report["ranks"][0]["final_bytes"] == 30


@pytest.mark.parametrize("change,code", [
    (lambda s,p: s["tensors"]["x"].update(dtype="mystery"), "unknown_dtype"),
    (lambda s,p: s["tensors"]["x"].update(shape=[0, 8]), "invalid_shape"),
    (lambda s,p: s["tensors"]["x"].update(shape=[True, 8]), "invalid_shape"),
    (lambda s,p: p["allocations"][0].update(tensor="missing"), "unknown_tensor"),
    (lambda s,p: p["allocations"][0].update(ranks=[0, 0]), "invalid_placement"),
    (lambda s,p: p["allocations"][0].update(ranks=[1]), "invalid_placement"),
    (lambda s,p: p["allocations"][0].update(ranks=[]), "invalid_placement"),
    (lambda s,p: p["allocations"][0].update(shard_axis=2), "invalid_placement"),
    (lambda s,p: p["allocations"][0].update(padding_multiple=64), "invalid_placement"),
    (lambda s,p: p["allocations"][0].update(start=-1), "invalid_lifetime"),
    (lambda s,p: p["allocations"][0].update(stop=0), "invalid_lifetime"),
    (lambda s,p: p["allocations"].append(copy.deepcopy(p["allocations"][0])), "duplicate_allocation"),
    (lambda s,p: p["allocations"].clear(), "unplaced_tensor"),
    (lambda s,p: p["ranks"][0].update(capacity_bytes=1.5), "invalid_budget"),
    (lambda s,p: p["ranks"][0].update(reserve_bytes=-1), "invalid_budget"),
    (lambda s,p: p["allocations"][0].update(storag={}), "unknown_field"),
    (lambda s,p: p["allocations"][0].update(storage={"kind":"mystery"}), "unknown_storage"),
])
def test_input_refusals_name_the_field(planner, change, code):
    structure, placement = spec(x=([4, 8], "float16")), plan(allocation("x"))
    change(structure, placement)
    with pytest.raises(planner.ResidencyRefusal) as caught:
        planner.plan_residency(structure, placement)
    reason = caught.value.report["reasons"][0]
    assert reason["code"] == code
    assert reason["field"]


def test_nondivisible_shard_refuses_before_price(planner):
    with pytest.raises(planner.ResidencyRefusal, match="shard_axis"):
        planner.plan_residency(spec(x=([3, 8], "float16")),
                              plan(allocation("x", ranks=(0, 1), shard_axis=0), capacities=(99, 99)))


def test_missing_native_layout_refuses(planner):
    with pytest.raises(planner.ResidencyRefusal, match="rates"):
        planner.plan_residency(spec(x=([32, 32], "float16")),
                              plan(allocation("x", storage={"kind": "dense_window"})))


def test_cli_fit_and_capacity_refusal(planner, tmp_path):
    structure_path, plan_path = tmp_path / "structure.json", tmp_path / "plan.json"
    structure_path.write_text(json.dumps(spec(x=([4, 8], "bfloat16"))))
    command = [sys.executable, "-m", "tessera.residency_plan", "--structure-spec", str(structure_path),
               "--plan", str(plan_path)]
    environment = {**os.environ, "PYTHONPATH": "src"}
    for capacity, exit_code in [(64, 0), (63, 2)]:
        plan_path.write_text(json.dumps(plan(allocation("x"), capacities=(capacity,))))
        result = subprocess.run(command, env=environment, text=True, capture_output=True)
        assert result.returncode == exit_code, result.stderr
        report = json.loads(result.stdout)
        assert report["ranks"][0]["peak_bytes"] == 64
        assert report["fits"] is (exit_code == 0)


@pytest.mark.parametrize("kind", ["dense_window", "dense_a4"])
def test_oversized_native_table_refuses_by_name(planner, kind):
    from tessera.manifest import WINDOW_BITS_MAX
    from tessera.window_geometry import TILE_ROWS
    storage = ({"kind": kind, "family": "TESSERA_FP8", "rates": [4] * 32,
                "window_bits": WINDOW_BITS_MAX + 1, "tile_rows": TILE_ROWS}
               if kind == "dense_window" else
               {"kind": kind, "rates": [2] * 32, "arity": 2,
                "memory": 10**100, "half": 16, "lut_entries": 16})
    with pytest.raises(planner.ResidencyRefusal) as caught:
        planner.plan_residency(spec(x=([32, 32], "bfloat16")),
                              plan(allocation("x", storage=storage)))
    assert caught.value.report["reasons"][0]["code"] == "invalid_storage"


def test_import_needs_no_tensor_runtime(planner):
    code = "import sys; import tessera.residency_plan; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": "src"},
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr



@pytest.mark.parametrize("kind", ["dense_window", "routed_window"])
@pytest.mark.parametrize("tile_rows", [64, 1024])
def test_native_tile_geometry_refuses_before_pricing(planner, kind, tile_rows):
    shape = [64, 32] if kind == "dense_window" else [2, 64, 32]
    storage = {"kind": kind, "family": "TESSERA_FP8", "rates": [4] * 32,
               "window_bits": 8, "tile_rows": tile_rows}
    if kind == "routed_window":
        storage["fused"] = False
    with pytest.raises(planner.ResidencyRefusal) as caught:
        planner.plan_residency(spec(x=(shape, "bfloat16")),
                              plan(allocation("x", storage=storage), capacities=(100000,)))
    reason = caught.value.report["reasons"][0]
    assert reason["code"] == "invalid_storage"
    assert reason["field"].endswith(".tile_rows")


def test_routed_rate_wider_than_window_refuses(planner):
    from tessera.window_geometry import TILE_ROWS
    storage = {"kind": "routed_window", "family": "TESSERA_FP8", "rates": [5] * 32,
               "window_bits": 4, "tile_rows": TILE_ROWS, "fused": False}
    with pytest.raises(planner.ResidencyRefusal) as caught:
        planner.plan_residency(spec(x=([2, 64, 32], "bfloat16")),
                              plan(allocation("x", storage=storage), capacities=(100000,)))
    reason = caught.value.report["reasons"][0]
    assert reason["code"] == "invalid_storage"
    assert reason["field"].endswith(".rates")
    assert "5" in reason["message"] and "4-bit window" in reason["message"]



@pytest.mark.parametrize("rows,expected", [(31, 9224), (512, 13072), (513, 21272)])
def test_dense_row_padding_matches_loader_tiles(planner, rows, expected):
    from tessera.window_geometry import TILE_ROWS
    storage = {"kind": "dense_window", "family": "TESSERA_FP8", "rates": [4] * 32,
               "window_bits": 8, "tile_rows": TILE_ROWS}
    report = planner.plan_residency(spec(x=([rows, 32], "bfloat16")),
                                    plan(allocation("x", storage=storage), capacities=(expected,)))
    assert report["ranks"][0]["peak_bytes"] == expected


def test_dense_loader_padding_refuses_the_old_small_capacity(planner):
    from tessera.window_geometry import TILE_ROWS
    storage = {"kind": "dense_window", "family": "TESSERA_FP8", "rates": [4] * 32,
               "window_bits": 8, "tile_rows": TILE_ROWS}
    with pytest.raises(planner.ResidencyRefusal) as caught:
        planner.plan_residency(spec(x=([64, 32], "bfloat16")),
                              plan(allocation("x", ranks=(0, 1), shard_axis=0, storage=storage),
                                   capacities=(2064, 2064)))
    assert [rank["peak_bytes"] for rank in caught.value.report["ranks"]] == [9232, 9232]
    assert [reason["excess_bytes"] for reason in caught.value.report["reasons"]] == [7168, 7168]



@pytest.mark.parametrize("kind,invalid,expected", [
    ("dense_window", None, 9232), ("routed_window", None, 18520),
    ("dense_window", "tile_rows", None), ("routed_window", "rates", None),
])
def test_native_plan_and_refusal_use_only_stdlib(planner, kind, invalid, expected):
    from pathlib import Path
    storage = {"kind": kind, "family": "TESSERA_FP8", "rates": [4] * 32,
               "window_bits": 8, "tile_rows": 512}
    shape = [32, 32] if kind == "dense_window" else [2, 64, 32]
    if kind == "routed_window":
        storage["fused"] = False
    if invalid == "tile_rows":
        storage["tile_rows"] = 64
    elif invalid == "rates":
        storage.update(rates=[5] * 32, window_bits=4)
    payload = json.dumps([spec(x=(shape, "bfloat16")),
                          plan(allocation("x", storage=storage), capacities=(100000,))])
    source = str(Path(__file__).resolve().parents[1] / "src")
    code = f"""
import json
import sys
sys.path.insert(0, {source!r})
from tessera.residency_plan import ResidencyRefusal, plan_residency
structure, placement = json.loads({payload!r})
try:
    report = plan_residency(structure, placement)
except ResidencyRefusal as exc:
    report = exc.report
assert 'torch' not in sys.modules
print(json.dumps(report))
"""
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    if invalid is None:
        assert report["fits"] is True
        assert report["ranks"][0]["peak_bytes"] == expected
    else:
        assert report["fits"] is False
        assert report["reasons"][0]["code"] == "invalid_storage"
        assert report["reasons"][0]["field"].endswith("." + invalid)


def test_residency_architecture_reference_uses_current_snapshot():
    from test_issue_refs import DOCS, REF, _snapshot
    snapshot = _snapshot()
    paragraph = (DOCS / "ARCHITECTURE.md").read_text().split("\n\n", 2)[1]
    citations = REF.findall(paragraph)
    assert citations
    missing = [(repo or snapshot["default_repo"], number) for repo, number in citations
               if number not in snapshot["repos"][repo or snapshot["default_repo"]]]
    assert not missing, f"residency architecture references absent from snapshot: {missing}"

