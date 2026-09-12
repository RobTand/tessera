"""Synthetic CPU partition regressions; these fixtures are not engine/GPU evidence.

Every test here proves a refusal or an arithmetic identity on a CPU fixture. A
passing suite establishes the parser and recomputation contract only; it does not
close a domain, and it cannot establish ownership invariance for GLM.
"""
import json
from pathlib import Path

import pytest

from experiments.full_engine_resource_partition import (
    DOMAIN_NAMES, IMPLEMENTED_DOMAINS, OWNER_CLASSES, REPORT_MEMBERS,
    TERM_DOMAINS, assemble_full_engine_resource_report, classify_allocations,
    compose_scalar_budget, derive_partition, qualify_domains, _simultaneous_peak,
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
    assert set(partition["domains"]) == set(DOMAIN_NAMES)
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


def test_a_capture_collection_error_refuses_the_domains_it_touches(ledger):
    # The analyzer folds the capture's own collection errors into ``issues``;
    # that is the key it emits, so that is the key this reads.
    assert qualify_domains(ledger)["history_join"]["state"] == "closed"
    ledger["issues"] = ["dropped CUPTI buffer"]
    domains = qualify_domains(ledger)
    assert all(domain["state"] != "closed" for domain in domains.values())
    for name in IMPLEMENTED_DOMAINS:
        assert domains[name]["state"] == "refused", name


def test_no_domain_closes_because_an_argument_was_supplied(ledger):
    # A domain that closes on the presence of a caller-supplied object is
    # ``qualified: true`` spelled differently. Four of the six have no
    # implemented closure check, and no signature lets a caller assert one.
    import inspect

    assert list(inspect.signature(qualify_domains).parameters) == ["ledger"]
    domains = qualify_domains(ledger)
    for name in set(DOMAIN_NAMES) - set(IMPLEMENTED_DOMAINS):
        assert domains[name]["state"] == "open", name
        assert domains[name]["evidence"] == []
        assert "no check here" in domains[name]["reason"] or "not covered" in \
            domains[name]["reason"] or "worker process" in domains[name]["reason"]


def test_worker_startup_stays_open_on_a_capture_from_no_engine_worker(ledger):
    # The replay refuses a capture whose recorder attached after CUDA
    # initialization, so reaching a parsed ledger proves that half. It does not
    # prove the recorder ran inside the engine's own worker process, and this
    # fixture says on its face that it is a synthetic CPU parser fixture.
    assert ledger["fixture_provenance"] == \
        "synthetic CPU-only parser fixture, not a GPU measurement"
    domain = qualify_domains(ledger)["worker_startup"]
    assert domain["state"] == "open"
    assert "worker process" in domain["reason"]


def test_an_allocation_freed_outside_every_unit_interval_is_unclassified(ledger):
    # On a live engine attention, norms, routing, sampling and every startup
    # transient land here. Charging one needs a declared step boundary saying
    # how often it recurs, and the capture emits unit intervals only.
    row = dict(ledger["torch_allocations"][0])
    row.update(allocation_id="freed-outside", lifetime_scope="outside_units",
               free_completed_index=row["allocate_index"] + 1,
               observed_categories=["fixed"], scope_stack=[])
    ledger["torch_allocations"] = [row]
    classified, unclassified = classify_allocations(ledger)
    assert classified == []
    assert len(unclassified) == 1
    assert "declared step boundary" in unclassified[0]["reason"]


def test_one_unclassified_allocation_nulls_every_term(ledger):
    # An unclassified row has neither owner nor lifetime, so it could belong to
    # any term. No term is complete while one exists, whatever the domains say.
    all_closed = {name: {"state": "closed", "evidence": ["declared by the test"],
                         "reason": None} for name in DOMAIN_NAMES}
    partition = derive_partition(ledger, domains=all_closed)
    assert partition["scope"]["unclassified_allocation_count"] == 1
    assert all(term is None for term in partition["terms"].values())
    assert compose_scalar_budget(partition) is None


def test_the_report_refuses_a_member_it_cannot_derive(ledger):
    for absent in ("reference", "workload", "execution"):
        members = {"reference": {"census": "x"}, "workload": {"tokens": "x"},
                   "execution": {"graph_mode": "eager"}}
        members[absent] = {}
        with pytest.raises(ValueError, match=f"never defaulted: {absent}"):
            assemble_full_engine_resource_report(ledger, **members)


def test_the_report_keeps_the_seven_members_and_the_synthetic_marker(ledger):
    report = assemble_full_engine_resource_report(
        ledger, reference={"census": "synthetic"},
        workload={"tokens": "synthetic"}, execution={"graph_mode": "eager"})
    assert set(report) == set(REPORT_MEMBERS) | {"schema"}
    assert report["schema"] == "tessera.full_engine_resource_report.v1"
    # A synthetic capture is never laundered into an artifact that looks measured.
    assert report["identity"]["fixture_provenance"] == ledger["fixture_provenance"]
    # Nothing is admitted today: derived is a claim, and it claims nothing.
    assert report["derived"]["scalar_budget_bytes"] is None
    assert all(term is None for term in report["derived"]["terms"].values())


def test_every_evidence_id_names_an_observation_the_report_carries(ledger):
    # Evidence that points at nothing cannot be checked by the consumer.
    report = assemble_full_engine_resource_report(
        ledger, reference={"census": "synthetic"},
        workload={"tokens": "synthetic"}, execution={"graph_mode": "eager"})
    for name, domain in report["partition"]["domains"].items():
        for observation in domain["evidence"]:
            assert observation in report["observations"], (name, observation)


def test_the_report_names_every_observation_class_a_domain_closes_on(ledger):
    # Named and null, never absent: a consumer must be able to tell "this
    # capture did not observe it" from "the producer forgot to carry it".
    # Each is what its domain would close on, and #399 owes every one.
    report = assemble_full_engine_resource_report(
        ledger, reference={"census": "synthetic"},
        workload={"tokens": "synthetic"}, execution={"graph_mode": "eager"})
    for owed in ("worker_startup_records", "runtime_provenance_relation",
                 "kv_observations", "timing_captures", "owner_views",
                 "observer_qualification"):
        assert owed in report["observations"], owed
        assert report["observations"][owed] is None, owed


def test_derived_does_not_restate_the_partition_domains(ledger):
    # Two copies of one claim invite drift; the partition owns the domains.
    report = assemble_full_engine_resource_report(
        ledger, reference={"census": "synthetic"},
        workload={"tokens": "synthetic"}, execution={"graph_mode": "eager"})
    assert "domains" not in report["derived"]
    assert set(report["partition"]["domains"]) == set(DOMAIN_NAMES)
