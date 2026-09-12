"""Synthetic CPU partition regressions; these fixtures are not engine/GPU evidence.

Every test here proves a refusal or an arithmetic identity on a CPU fixture. A
passing suite establishes the parser and recomputation contract only; it does not
close a domain, and it cannot establish ownership invariance for GLM.
"""
import json
from pathlib import Path

import pytest

from experiments.full_engine_resource_partition import (
    OWNER_CLASSES, TERM_DOMAINS, classify_allocations, compose_scalar_budget,
    derive_partition, qualify_domains, _simultaneous_peak,
)
from experiments.full_engine_resources import analyze_engine_resource_ledger


@pytest.fixture
def ledger():
    raw = json.loads((Path(__file__).parent
                      / "fixtures/full_engine_resource_ledger.json").read_text())
    return analyze_engine_resource_ledger(raw)


def _row(allocation_id, size, start, end, **overrides):
    row = {"allocation_id": allocation_id, "bytes": size, "allocate_index": start,
           "free_completed_index": end, "owner_class": "fixed",
           "lifetime_class": "scratch", "unit": None}
    row.update(overrides)
    return row


def test_every_term_is_null_while_its_domains_are_open(ledger):
    partition = derive_partition(ledger)
    assert set(partition["domains"]) == set(TERM_DOMAINS["fixed_resident"]) | {
        "provenance_admission", "cache_capacity", "timing_partition"}
    assert all(term is None for term in partition["terms"].values()), partition["terms"]
    assert partition["scope"]["unavailable_terms"] == sorted(TERM_DOMAINS)
    assert partition["scope"]["expressible"] is False
    assert compose_scalar_budget(partition) is None


def test_an_open_domain_names_why_and_carries_no_evidence(ledger):
    domains = qualify_domains(ledger)
    for name, domain in domains.items():
        assert domain["state"] in ("open", "refused", "closed")
        if domain["state"] != "closed":
            assert domain["evidence"] == [], (name, domain)
            assert domain["reason"], name
    assert domains["external_closure"]["state"] == "open"
    assert "disjoint observed charge" in domains["external_closure"]["reason"]
    assert domains["timing_partition"]["reason"] == "no same-run timing partition supplied"


def test_a_shared_or_unknown_owner_is_unclassified_and_never_bucketed(ledger):
    _, unclassified = classify_allocations(ledger)
    for row in unclassified:
        assert row["reason"]
        assert not (set(row["observed_categories"]) & set(OWNER_CLASSES)) or \
            len(row["observed_categories"]) != 1
    classified, _ = classify_allocations(ledger)
    assert all(row["owner_class"] in OWNER_CLASSES for row in classified)


def test_a_peak_is_a_sweep_and_not_a_sum_of_per_allocation_maxima():
    # Two allocations of 100 bytes that never coincide: the peak is 100, and
    # summing their individual maxima would say 200.
    rows = [_row("a", 100, 0, 5), _row("b", 100, 5, 9)]
    assert _simultaneous_peak(rows, terminal_index=10) == 100
    assert sum(row["bytes"] for row in rows) == 200


def test_a_free_and_an_allocation_at_one_index_do_not_double_count():
    # The free settles first, so a reused extent is charged once.
    rows = [_row("a", 64, 0, 3), _row("b", 64, 3, 7)]
    assert _simultaneous_peak(rows, terminal_index=8) == 64


def test_overlapping_allocations_sum_into_the_peak():
    rows = [_row("a", 64, 0, 9), _row("b", 32, 2, 4), _row("c", 16, 3, 8)]
    assert _simultaneous_peak(rows, terminal_index=10) == 64 + 32 + 16


def test_a_never_freed_allocation_is_live_to_the_terminal_boundary():
    rows = [_row("a", 128, 0, None)]
    assert _simultaneous_peak(rows, terminal_index=4) == 128


def test_the_composition_refuses_while_any_single_term_is_unavailable(ledger):
    partition = derive_partition(ledger)
    partition["terms"] = {name: (0 if name not in ("candidate_resident",
                                                   "candidate_activation",
                                                   "candidate_scratch") else {})
                          for name in TERM_DOMAINS}
    assert compose_scalar_budget(partition) == 0
    partition["terms"]["fixed_kv"] = None
    assert compose_scalar_budget(partition) is None


def test_the_composition_takes_maxima_over_candidates_and_sums_resident(ledger):
    partition = derive_partition(ledger)
    partition["terms"] = {
        "fixed_resident": 1000,
        "candidate_resident": {"u0": 10, "u1": 20},
        "fixed_activation": 7,
        "candidate_activation": {"u0": 300, "u1": 500},
        "fixed_scratch": 3,
        "candidate_scratch": {"u0": 40, "u1": 11},
        "fixed_kv": 64,
    }
    # 1000 + (10+20) + 7 + max(300,500) + 3 + max(40,11) + 64
    assert compose_scalar_budget(partition) == 1000 + 30 + 7 + 500 + 3 + 40 + 64


def test_the_partition_refuses_a_foreign_schema(ledger):
    ledger["schema"] = "tessera.full_engine_raw_resource_ledger.v2"
    with pytest.raises(ValueError, match="unsupported raw ledger schema"):
        derive_partition(ledger)


def test_the_scope_stays_tp1_single_device(ledger):
    scope = derive_partition(ledger)["scope"]
    assert scope["topology"] == "tp1_single_device_resident_eager"
    assert scope["allocation_scope"] == "gpu_allocations_only"
    assert scope["invariance"] == "one complete assignment, one row per unit"


def test_a_capture_collection_error_refuses_every_closed_domain(ledger):
    ledger["capture_qualification"] = {"errors": ["dropped CUPTI buffer"]}
    domains = qualify_domains(ledger)
    assert all(domain["state"] != "closed" for domain in domains.values())
    assert any(domain["state"] == "refused" for domain in domains.values())
