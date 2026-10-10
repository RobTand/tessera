"""Bench geometry preflight against the pinned bench contract (tessera#1182).

The KDA input module times its full role set. A replicated role below one
contract block, or a short local total, refuses before timing.
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


def contract(name="kda_in"):
    roles, cols = bench.MODULES[name]
    return list(roles), cols


def test_full_kda_contract_passes_preflight():
    roles, cols = contract()
    assert bench.require_module_geometry("kda_in", roles, cols) is None
    total = sum(rows for _, rows in roles)
    assert bench.require_contract_total("kda_in", total, cols) is None


def test_replicated_role_below_contract_block_refuses_by_name():
    roles, cols = contract()
    index = bench.KDA_REPLICATED_PARTITIONS[0]
    name, rows = roles[index]
    floor = min(rows for i, (_, rows) in enumerate(roles)
                if i in bench.KDA_REPLICATED_PARTITIONS)
    assert rows == floor
    shrunk = [(role, rows // 2 if role == name else rows) for role, rows in roles]
    with pytest.raises(ValueError, match=name):
        bench.require_module_geometry("kda_in", shrunk, cols)


def test_dropped_replicated_role_refuses_by_name():
    roles, cols = contract()
    index = bench.KDA_REPLICATED_PARTITIONS[1]
    dropped = [role for i, role in enumerate(roles) if i != index]
    with pytest.raises(ValueError, match=roles[index][0]):
        bench.require_module_geometry("kda_in", dropped, cols)


def test_short_local_total_refuses_by_name():
    roles, cols = contract()
    total = sum(rows for _, rows in roles)
    block = min(rows for _, rows in roles)
    with pytest.raises(ValueError, match="total"):
        bench.require_contract_total("kda_in", total - block, cols)


def test_short_local_total_refuses_in_full_preflight():
    roles, cols = contract()
    short = [(role, rows) for role, rows in roles]
    name, rows = short[0]
    short[0] = (name, rows - 1)
    with pytest.raises(ValueError, match="total"):
        bench.require_module_geometry("kda_in", short, cols)


def test_unknown_module_refuses_but_unknown_shape_total_passes():
    roles, cols = contract()
    with pytest.raises(ValueError, match="no pinned contract"):
        bench.require_module_geometry("kda_missing", roles, cols)
    assert bench.require_contract_total("research_probe", 1, 1) is None


def test_contract_derives_replicated_floor_and_total():
    roles, _ = contract()
    replicated = [rows for i, (_, rows) in enumerate(roles)
                  if i in bench.KDA_REPLICATED_PARTITIONS]
    assert replicated and all(rows == min(replicated) for rows in replicated)
    assert sum(rows for _, rows in roles) > sum(replicated)
