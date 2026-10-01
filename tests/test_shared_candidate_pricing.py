"""CPU contracts only: no CUDA allocation, engine admission or served quality."""
import copy
from contextlib import contextmanager
from functools import lru_cache
from unittest.mock import patch

import pytest

from experiments.full_engine_ownership import dense_startup_check
from experiments.full_engine_resource_partition import (
    _compose,
    _compose_terms,
    classify_allocations,
    derive_partition,
)
from tessera import decode, serving_parts
from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.export import DEFAULT_CODE

COMPONENTS = [
    {"name": "subsets", "shape": [4, 64], "dtype": "int64", "layout": "contiguous", "bytes": 2048},
    {"name": "table_next", "shape": [2, 64], "dtype": "int64", "layout": "contiguous", "bytes": 1024},
    {"name": "table_sub", "shape": [2, 64], "dtype": "int64", "layout": "contiguous", "bytes": 1024},
]


def capture(distinct=False, rank=0, process_id=101):
    groups = [{"group_id": "a" * 64, "members": ["down", "gate"], "components": copy.deepcopy(COMPONENTS)}]
    if distinct:
        groups[0]["members"] = ["gate"]
        groups.append({"group_id": "b" * 64, "members": ["down"], "components": copy.deepcopy(COMPONENTS)})
    pricing = {"schema": "tessera.shared_candidate_resident_pricing.v1", "scope": "per_process_rank_device",
               "private_resident_bytes": {"gate": 100, "down": 200}, "groups": groups,
               "resident_bytes": 300 + 4096 * len(groups)}
    rows, views, observed = [], [], []
    for module, size in pricing["private_resident_bytes"].items():
        index = len(rows)
        rows.append({"allocation_id": f"0:{10000 + index * 4096}:1", "address": 10000 + index * 4096,
                     "bytes": size, "allocate_index": index, "free_completed_index": None,
                     "scope_stack": [], "observed_owners": [f"model:buffer:{module}.private"],
                     "observed_categories": ["candidate"], "lifetime_scope": "outside_units"})
        views.append({"allocation_id": rows[-1]["allocation_id"], "class": "candidate", "unit": module,
                      "site": None, "reason": None, "rule": "census"})
    for group in groups:
        storage = []
        for component in group["components"]:
            index = len(rows)
            address = 10000 + index * 4096
            rows.append({"allocation_id": f"0:{address}:1", "address": address,
                         "bytes": component["bytes"], "allocate_index": index,
                         "free_completed_index": None, "scope_stack": [], "observed_owners": [],
                         "observed_categories": [], "lifetime_scope": "outside_units"})
            views.append({"allocation_id": rows[-1]["allocation_id"], "class": "candidate", "unit": None,
                          "site": {"package": "plugin",
                                   "relative": "encode.py" if component["name"] == "subsets" else "decode.py",
                                   "name": "_subset_table" if component["name"] == "subsets" else "_replay_tables"},
                          "reason": "no unit", "rule": "site:plugin"})
            storage.append(dict(component, address=address))
        observed.append({"group_id": group["group_id"], "members": group["members"], "components": storage})
    dense = {"schema": "tessera.full_engine_dense_startup_observation.v2", "rank": rank,
             "process_id": process_id, "device_type": "cuda", "device_id": 0,
             "memory_allocated_bytes": pricing["resident_bytes"],
             "units": {m: {"module": m, "family": "TESSERA_NVFP4",
                            "manifest_resident_bytes_resident_mode": n}
                       for m, n in pricing["private_resident_bytes"].items()},
             "shared_candidate": {"pricing": pricing, "process_id": process_id, "rank": rank,
                                  "device_type": "cuda", "device_id": 0, "groups": observed}}
    return rows, views, dense


@pytest.mark.parametrize("distinct,rank,process_id", [(False, 0, 101), (True, 0, 101), (False, 1, 202)])
def test_shared_manifest_and_dense_startup_agree(distinct, rank, process_id):
    rows, views, dense = capture(distinct, rank, process_id)
    check = dense_startup_check(rows, views, dense, ready_index=20)
    assert check is not None, "explicit shared-candidate startup pricing must be supported"
    assert check["closed"] is True
    assert check["shared_candidate_resident_bytes"] == (8192 if distinct else 4096)
    assert sum(cell["ledger_candidate_resident_bytes"] for cell in check["units"].values()) == 300
    ledger = {"schema": "tessera.full_engine_raw_resource_ledger.v2", "issues": [],
              "unattributed_external_records": [], "external_native_peak_bytes": 0,
              "torch_observed_live_peak_bytes": dense["memory_allocated_bytes"],
              "torch_allocations": rows, "identity": {"rank": rank, "process_id": process_id, "device_id": 0},
              "checkpoints": [{"label": "ready_for_workload", "trace_index": 20}],
              "owner_views": {"schema": "tessera.full_engine_ownership_observation.v1",
                              "views": {"views": views}, "dense_startup_check": check}}
    classified, unknown, _ = classify_allocations(ledger)
    assert unknown == []
    terms = _compose_terms(classified, ["gate", "down"], 21)
    assert terms["shared_candidate_resident"] == {g["group_id"]: 4096 for g in dense["shared_candidate"]["groups"]}
    assert _compose(terms) == dense["memory_allocated_bytes"]
    partition = derive_partition(ledger)
    assert partition["schema"] == "tessera.full_engine_resource_partition.v2"
    assert partition["terms"]["shared_candidate_resident"] == terms["shared_candidate_resident"]
    assert partition["uncharged_allocations"] == []
    ledger["identity"]["rank"] += 1
    with pytest.raises(ValueError, match="shared.candidate.*rank"):
        classify_allocations(ledger)
    ledger["identity"]["rank"] = rank
    ledger["identity"]["process_id"] += 1
    with pytest.raises(ValueError, match="shared.candidate.*process_id"):
        classify_allocations(ledger)


def _partition_capture():
    rows, views, dense = capture()
    check = dense_startup_check(rows, views, dense, ready_index=20)
    return {"torch_allocations": rows,
            "identity": {"rank": 0, "process_id": 101, "device_id": 0},
            "checkpoints": [{"label": "ready_for_workload", "trace_index": 20}],
            "owner_views": {"schema": "tessera.full_engine_ownership_observation.v1",
                            "views": {"views": views}, "dense_startup_check": check}}


@pytest.mark.parametrize("position", ["before", "after"])
def test_partition_rejects_original_duplicate_ownership_views(position):
    ledger = _partition_capture()
    views = ledger["owner_views"]["views"]["views"]
    conflicting = dict(views[2], **{"class": "fixed", "unit": "gate", "rule": "census"})
    views.insert(0 if position == "before" else len(views), conflicting)
    with pytest.raises(ValueError, match="shared.candidate.*ownership views"):
        classify_allocations(ledger)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "mismatch"])
def test_partition_requires_independent_unique_ledger_ready_marker(defect):
    ledger = _partition_capture()
    if defect == "missing":
        ledger["checkpoints"] = []
    elif defect == "duplicate":
        ledger["checkpoints"].append({"label": "ready_for_workload", "trace_index": 20})
    else:
        ledger["checkpoints"][0]["trace_index"] = 3
    with pytest.raises(ValueError, match="shared.candidate.*ready_for_workload"):
        classify_allocations(ledger)


def test_shared_group_union_must_cover_every_nvfp4_module():
    rows, views, dense = capture()
    proof = dense["shared_candidate"]
    proof["pricing"]["groups"][0]["members"] = ["gate"]
    proof["groups"][0]["members"] = ["gate"]
    with pytest.raises(ValueError, match="shared.candidate.*NVFP4.*roster"):
        dense_startup_check(rows, views, dense, ready_index=20)


@pytest.mark.parametrize("defect", ["rank", "process", "device", "missing", "members", "alias", "freed", "late", "dtype", "private_alias", "duplicate"])
def test_malformed_shared_proof_is_refused(defect):
    rows, views, dense = capture()
    proof = dense["shared_candidate"]
    if defect == "rank":
        proof["rank"] = 1
    if defect == "process":
        proof["process_id"] = 202
    if defect == "device":
        proof["device_id"] = 1
    if defect == "missing":
        proof["groups"][0]["components"].pop()
    if defect == "members":
        proof["groups"][0]["members"] = ["gate"]
    if defect == "alias":
        proof["groups"][0]["components"][1]["address"] = rows[2]["address"]
    if defect == "freed":
        rows[2]["free_completed_index"] = 10
    if defect == "late":
        rows[2]["allocate_index"] = 21
    if defect == "dtype":
        proof["groups"][0]["components"][0]["dtype"] = "int32"
    if defect == "private_alias":
        views[2]["rule"] = "census"
        rows[2]["observed_owners"] = ["model:buffer:gate.private"]
    if defect == "duplicate":
        proof["groups"].append(copy.deepcopy(proof["groups"][0]))
    with pytest.raises(ValueError, match="shared.candidate"):
        dense_startup_check(rows, views, dense, ready_index=20)


@pytest.mark.parametrize("distinct", [False, True])
def test_producer_prices_real_replay_tables_once_without_changing_legacy(distinct):
    with _isolated_replay_state():
        _check_real_replay_pricing(distinct)


def _check_real_replay_pricing(distinct):
    assert hasattr(decode, "replay_table_spec"), "producer needs exact shared table components"
    assert hasattr(serving_parts, "shared_candidate_pricing"), "producer needs additive shared pricing"
    forest = build_forest(7, grid=tuple_grid(E2M1_GRID, 2))
    second = build_forest(6, grid=tuple_grid(E2M1_GRID, 2)) if distinct else forest
    specs = [decode.replay_table_spec(f, DEFAULT_CODE) for f in (forest, second)]
    modules = {m: {"family": "TESSERA_NVFP4", "resident_bytes_resident_mode": private + decode.replay_table_bytes(f, DEFAULT_CODE)}
               for m, private, f in zip(("gate", "down"), (100, 200), (forest, second), strict=True)}
    old = copy.deepcopy(modules)
    pricing = serving_parts.shared_candidate_pricing(modules, {"gate": [specs[0]], "down": [specs[1]]})
    assert modules == old
    assert pricing["private_resident_bytes"] == {"gate": 100, "down": 200}
    assert len(pricing["groups"]) == (2 if distinct else 1)
    assert pricing["resident_bytes"] == 300 + sum(sum(c["bytes"] for c in g["components"]) for g in pricing["groups"])
    observation = decode.replay_resident_observation(pricing, device="cpu", rank=0)
    assert observation["process_id"] > 0
    assert len(observation["groups"]) == len(pricing["groups"])
    # Eviction/reconstruction with old tensors still held is NOT one allocation.
    held = decode._replay_tables(forest, DEFAULT_CODE, "cpu")
    decode._replay_tables.cache_clear()
    decode._replay_tables(forest, DEFAULT_CODE, "cpu")
    with pytest.raises(ValueError, match="shared.candidate.*ambiguous"):
        decode.replay_resident_observation(pricing, device="cpu", rank=0)
    assert held[0].numel() > 0


@contextmanager
def _isolated_replay_state():
    """Give one test the production memoizer policy and its own weak index."""
    original = decode._replay_tables
    fresh = lru_cache(**original.cache_parameters())(original.__wrapped__)
    with patch.object(decode, '_replay_tables', fresh), \
            patch.object(decode, '_replay_resident_entries', {}):
        try:
            yield
        finally:
            fresh.cache_clear()


@pytest.mark.parametrize('distinct', [False, True])
@pytest.mark.parametrize('prior_state', ['missing', 'ambiguous'])
def test_real_replay_pricing_owns_its_initial_state(distinct, prior_state):
    with _isolated_replay_state():
        forests = [build_forest(7, grid=tuple_grid(E2M1_GRID, 2))]
        if distinct:
            forests.append(build_forest(6, grid=tuple_grid(E2M1_GRID, 2)))
        held = [decode._replay_tables(f, DEFAULT_CODE, 'cpu') for f in forests]
        if prior_state == 'missing':
            decode._replay_resident_entries.clear()
        else:
            decode._replay_tables.cache_clear()
            held.extend(decode._replay_tables(f, DEFAULT_CODE, 'cpu') for f in forests)
        # Exercise the existing test under real stale/missing table state.
        # Its own setup must isolate that state; production still refuses it.
        test_producer_prices_real_replay_tables_once_without_changing_legacy(distinct)
        assert all(tables[0].numel() > 0 for tables in held)


def test_unknown_candidate_is_not_zero_under_shared_contract():
    rows, views, dense = capture()
    extra = dict(rows[2], allocation_id="0:99999:1", address=99999)
    rows.append(extra)
    views.append(dict(views[2], allocation_id=extra["allocation_id"]))
    check = dense_startup_check(rows, views, dense, ready_index=20)
    assert check is not None, "explicit shared-candidate startup pricing must be supported"
    assert check["closed"] is False
    assert check["unqualified_candidate_allocations"] == [extra["allocation_id"]]
