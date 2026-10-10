"""Bench geometry preflight reads the pinned construction census (tessera#1182).

The KDA input module times its full role set. The floor (128 rows per
replicated role) and the local total (12576 rows) come from the pinned
construction receipt, never the table under test. A short table, such as
the tessera#1020 defect with 64-row replicas and a 12448 total, refuses
before timing. A short input against a sound table refuses by name.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

T8R = Path(__file__).resolve().parents[1] / "experiments" / "t8r_speed"


def load_bench_module():
    sys.path.insert(0, str(T8R))
    try:
        path = T8R / "bench_dense_module.py"
        spec = importlib.util.spec_from_file_location("bench_dense_module_geometry", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(T8R))


bench = load_bench_module()

#: The tessera#1020 defect: halved replicas and a short local total.
PREFIX_SHORT_ROLES = [("q_proj", 4096), ("k_proj", 4096), ("v_proj", 4096),
                      ("b_proj", 32), ("f_a_proj", 64), ("g_a_proj", 64)]


def contract(name="kda_in"):
    roles, cols = bench.MODULES[name]
    return list(roles), cols


def test_construction_receipt_pins_128_rows_and_12576_total():
    rows, replicated, total, floor = bench._construction_kda_tp2()
    assert replicated == [4, 5]
    assert rows == [4096, 4096, 4096, 32, 128, 128]
    assert floor == 128
    assert total == 12576


def test_full_kda_contract_passes_preflight():
    roles, cols = contract()
    assert bench.require_pinned_kda_contract() is None
    assert bench.require_module_geometry("kda_in", roles, cols) is None
    total = sum(rows for _, rows in roles)
    assert total == 12576
    assert bench.require_contract_total("kda_in", total, cols) is None


def test_prefix_short_table_refuses_before_timing(monkeypatch):
    monkeypatch.setitem(bench.MODULES, "kda_in", (list(PREFIX_SHORT_ROLES), 4096))
    with pytest.raises(ValueError, match="construction"):
        bench.require_pinned_kda_contract()
    with pytest.raises(ValueError, match="construction"):
        bench.require_module_geometry("kda_in", list(PREFIX_SHORT_ROLES), 4096)
    with pytest.raises(ValueError, match="construction"):
        bench.require_contract_total("kda_in", 12448, 4096)


def test_prefix_short_total_refuses_against_sound_table():
    _, cols = contract()
    with pytest.raises(ValueError, match="construction"):
        bench.require_contract_total("kda_in", 12448, cols)
    roles, _ = contract()
    short = [(role, 64 if role in ("f_a_proj", "g_a_proj") else rows)
             for role, rows in roles]
    with pytest.raises(ValueError, match="f_a_proj"):
        bench.require_module_geometry("kda_in", short, cols)


def test_replicated_role_below_receipt_floor_refuses_by_name():
    roles, cols = contract()
    _, _, _, floor = bench._construction_kda_tp2()
    assert floor == 128
    index = bench.KDA_REPLICATED_PARTITIONS[0]
    name, _ = roles[index]
    shrunk = [(role, floor // 2 if role == name else rows) for role, rows in roles]
    with pytest.raises(ValueError, match=name):
        bench.require_module_geometry("kda_in", shrunk, cols)


def test_dropped_replicated_role_refuses_by_name():
    roles, cols = contract()
    index = bench.KDA_REPLICATED_PARTITIONS[1]
    dropped = [role for i, role in enumerate(roles) if i != index]
    with pytest.raises(ValueError, match=roles[index][0]):
        bench.require_module_geometry("kda_in", dropped, cols)


def test_short_local_total_refuses_in_full_preflight():
    roles, cols = contract()
    short = [(role, rows) for role, rows in roles]
    name, rows = short[0]
    short[0] = (name, rows - 1)
    with pytest.raises(ValueError, match="construction"):
        bench.require_module_geometry("kda_in", short, cols)


def test_unknown_module_refuses_but_unknown_shape_total_passes():
    roles, cols = contract()
    with pytest.raises(ValueError, match="no pinned contract"):
        bench.require_module_geometry("kda_missing", roles, cols)
    assert bench.require_contract_total("research_probe", 1, 1) is None
