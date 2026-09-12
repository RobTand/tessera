"""Synthetic CPU partition regressions; these fixtures are not engine/GPU evidence.

Every test here proves a refusal or an arithmetic identity on a CPU fixture. A
passing suite establishes the parser and recomputation contract only; it does not
close a domain, and it cannot establish ownership invariance for GLM.
"""
import json
from pathlib import Path

import pytest

from experiments.full_engine_resource_partition import (
    CHARGED_CELLS, DOMAIN_NAMES, IMPLEMENTED_DOMAINS, OWNER_CLASSES, REPORT_MEMBERS,
    uncharged_allocations,
    TERM_DOMAINS, DECLARED_MEMBER_FIELDS, SUPPORTED_EXECUTION,
    assemble_full_engine_resource_report, classify_allocations,
    compose_scalar_budget, derive_partition, qualify_domains, _compose_terms,
    _simultaneous_peak, _unit_of,
)
from experiments.full_engine_resources import analyze_engine_resource_ledger


@pytest.fixture
def ledger():
    raw = json.loads((Path(__file__).parent
                      / "fixtures/full_engine_resource_ledger.json").read_text())
    return analyze_engine_resource_ledger(raw)


def _members(**overrides):
    """Declared members whose field sets are this schema's, for the assembler."""
    members = {
        "reference": {"canonical_census": "synthetic", "runtime_binding": "synthetic",
                      "selected_rows": ["synthetic"]},
        "workload": {"calibration": "synthetic", "prompt_ids": ["synthetic"],
                     "sampling": "synthetic"},
        "execution": dict(SUPPORTED_EXECUTION),
    }
    members.update(overrides)
    return members


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
    _, unclassified, _ = classify_allocations(ledger)
    for row in unclassified:
        assert row["reason"]
        assert not (set(row["observed_categories"]) & set(OWNER_CLASSES)) or \
            len(row["observed_categories"]) != 1
    classified, _, _ = classify_allocations(ledger)
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
    # how often it recurs, and THIS capture declares none -- it is the banked
    # fixture, which predates step intervals entirely.
    assert ledger["step_intervals"] is None
    assert ledger["step_coverage"]["state"] == "unobserved"
    row = dict(ledger["torch_allocations"][0])
    row.update(allocation_id="freed-outside", lifetime_scope="outside_units",
               free_completed_index=row["allocate_index"] + 1,
               observed_categories=["fixed"], scope_stack=[])
    ledger["torch_allocations"] = [row]
    classified, unclassified, non_step = classify_allocations(ledger)
    assert classified == [] and non_step == []
    assert len(unclassified) == 1
    assert "declared step boundary" in unclassified[0]["reason"]


def test_one_unclassified_allocation_nulls_every_term():
    # An unclassified row has neither owner nor lifetime, so it could belong to
    # any term. No term is complete while one exists.
    #
    # The banked fixture cannot show this. It closes history_join but not
    # external_closure, so no term's domains are all closed on it and every term
    # is null for a reason that has nothing to do with the unclassified row. On
    # a ledger where both implemented domains DO close, the two scratch terms
    # carry values -- and adding one unclassified row takes them away.
    good = _raw("a", 6, 2, 9, ["candidate"], "inside_unit", ["u"])
    baseline = derive_partition(_synthetic_ledger(6, [good]))
    assert baseline["terms"]["candidate_scratch"] == {"u": 6}
    assert baseline["scope"]["step_coverage"] is None

    stray = _raw("b", 6, 3, 8, ["candidate"], "outside_units", [])
    partition = derive_partition(_synthetic_ledger(12, [good, stray]))
    assert partition["scope"]["unclassified_allocation_count"] == 1
    assert all(term is None for term in partition["terms"].values())
    assert compose_scalar_budget(partition) is None


def test_the_report_refuses_a_member_it_cannot_derive(ledger):
    for absent in ("reference", "workload", "execution"):
        with pytest.raises(ValueError, match=f"never defaulted: {absent}"):
            assemble_full_engine_resource_report(ledger, **_members(**{absent: None}))


def test_the_report_keeps_the_seven_members_and_the_synthetic_marker(ledger):
    report = assemble_full_engine_resource_report(
        ledger, **_members())
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
        ledger, **_members())
    for name, domain in report["partition"]["domains"].items():
        for observation in domain["evidence"]:
            assert observation in report["observations"], (name, observation)


def test_the_report_names_every_observation_class_a_domain_closes_on(ledger):
    # Named and null, never absent: a consumer must be able to tell "this
    # capture did not observe it" from "the producer forgot to carry it".
    # Each is what its domain would close on, and #399 owes every one.
    report = assemble_full_engine_resource_report(
        ledger, **_members())
    for owed in ("worker_startup_records", "runtime_provenance_relation",
                 "kv_observations", "timing_captures", "owner_views",
                 "observer_qualification"):
        assert owed in report["observations"], owed
        assert report["observations"][owed] is None, owed


def test_derived_does_not_restate_the_partition_domains(ledger):
    # Two copies of one claim invite drift; the partition owns the domains.
    report = assemble_full_engine_resource_report(
        ledger, **_members())
    assert "domains" not in report["derived"]
    assert set(report["partition"]["domains"]) == set(DOMAIN_NAMES)


def test_no_caller_can_hand_the_partition_a_table_of_closed_domains(ledger):
    # Recording that a caller supplied one was not enough. The object that
    # escaped was the partition: it is public, it returns every term filled, and
    # the only refusal lived in an assembler that path never reached. The
    # parameter is gone, and the arithmetic is exercised through
    # ``_compose_terms``, which returns values and never an artifact.
    import inspect
    assert list(inspect.signature(derive_partition).parameters) == ["ledger"]
    assert derive_partition(ledger)["domains_source"] == "derived"


def _classified(**overrides):
    row = {"allocation_id": "a", "bytes": 100, "owner_class": "kv",
           "lifetime_class": "scratch", "unit": None,
           "allocate_index": 0, "free_completed_index": 2}
    row.update(overrides)
    return row


def test_a_kv_backing_with_a_transient_lifetime_is_charged_by_no_term():
    # The classifier produces nine (owner, lifetime) cells; the composition
    # charges seven. A KV backing freed inside a unit falls through all of them.
    uncharged = uncharged_allocations([_classified(lifetime_class="scratch"),
                                       _classified(allocation_id="b",
                                                   lifetime_class="activation")])
    assert {row["allocation_id"] for row in uncharged} == {"a", "b"}
    assert all("no composition term charges" in row["reason"] for row in uncharged)


def test_a_candidate_allocation_with_no_unit_is_charged_by_no_term():
    uncharged = uncharged_allocations([_classified(owner_class="candidate",
                                                   lifetime_class="resident",
                                                   free_completed_index=None,
                                                   unit=None)])
    assert len(uncharged) == 1
    assert "carrying no unit" in uncharged[0]["reason"]


def test_every_charged_cell_is_charged():
    for owner, lifetime in CHARGED_CELLS:
        unit = "u0" if owner == "candidate" else None
        assert uncharged_allocations([_classified(owner_class=owner,
                                                  lifetime_class=lifetime,
                                                  unit=unit)]) == []


def test_an_uncharged_allocation_nulls_every_term_because_undercounting_ooms():
    # An overcount wastes headroom; an undercount hands a serving gate a budget
    # smaller than the engine needs. Only one of those kills the box. Shown on a
    # ledger whose scratch terms would otherwise carry values, so that losing
    # them is the uncharged row's doing and not an open domain's.
    good = _raw("a", 6, 2, 9, ["candidate"], "inside_unit", ["u"])
    assert derive_partition(_synthetic_ledger(6, [good]))["terms"]["candidate_scratch"]

    kv_transient = _raw("kv-transient", 6, 3, 8, ["kv"], "inside_unit", ["u"])
    partition = derive_partition(_synthetic_ledger(12, [good, kv_transient]))
    assert partition["scope"]["uncharged_allocation_count"] == 1
    assert partition["unclassified_allocations"] == []
    assert all(term is None for term in partition["terms"].values())
    assert compose_scalar_budget(partition) is None


def _synthetic_ledger(peak, rows, steps=None, executed=None):
    """A minimal raw ledger, for the arithmetic the fixture cannot reach.

    The banked fixture's only intra-unit allocation carries no owner, which is
    unclassified by design, and an unclassified row nulls every term. So the
    fixture can never exercise the row-to-term selection at all. These rows can.

    ``steps`` are ``(begin_index, end_index)`` pairs; ``executed`` defaults to
    the number declared, which is the complete-coverage case. A larger
    ``executed`` is the partial-coverage case, where some engine step ran with
    no interval declared over it.
    """
    ledger = {"schema": "tessera.full_engine_raw_resource_ledger.v1", "issues": [],
              "unattributed_external_records": [], "external_native_peak_bytes": 0,
              "torch_observed_live_peak_bytes": peak, "torch_allocations": rows,
              "step_intervals": None, "step_coverage": None}
    if steps is not None:
        declared = len(steps)
        executed = declared if executed is None else executed
        ledger["step_intervals"] = [
            {"step_id": f"step:{index}", "begin_index": begin, "end_index": end}
            for index, (begin, end) in enumerate(steps, start=1)]
        ledger["step_coverage"] = {
            "state": "complete" if executed == declared and declared >= 1 else "partial",
            "declared": declared, "executed": executed}
    return ledger


def _raw(allocation_id, size, start, end, categories, scope, stack):
    return {"allocation_id": allocation_id, "bytes": size, "allocate_index": start,
            "free_completed_index": end, "observed_categories": categories,
            "lifetime_scope": scope, "scope_stack": stack}


def test_a_scratch_term_carries_a_swept_value_once_nothing_blocks_it():
    # The gap that let the maximum-over-units defect through: no test asserted a
    # non-null term value out of derive_partition on any ledger, so the row-to-
    # term selection had never executed under test. Two overlapping rows in one
    # unit sweep to 12, not to 6 and not to 12 by addition of maxima.
    rows = [_raw("a", 6, 2, 9, ["candidate"], "inside_unit", ["u"]),
            _raw("b", 6, 3, 8, ["candidate"], "inside_unit", ["u"])]
    partition = derive_partition(_synthetic_ledger(12, rows))
    assert partition["unclassified_allocations"] == []
    assert partition["uncharged_allocations"] == []
    assert partition["terms"]["candidate_scratch"] == {"u": 12}
    assert partition["terms"]["fixed_scratch"] == 0


def test_an_allocation_is_charged_to_the_unit_its_lifetime_was_decided_against():
    # The replay reads unit_invocation from the OUTERMOST containing interval and
    # decides lifetime_scope against that same interval. Charging the innermost
    # split a row's lifetime basis from its charge.
    assert _unit_of({"scope_stack": ["outer", "inner"]}) == "outer"
    assert _unit_of({"scope_stack": []}) is None


def test_two_units_that_can_be_live_at_once_are_never_alternatives_in_a_maximum():
    # Unit intervals may nest; only crossing is refused. Two sibling inner units
    # therefore hold rows that are simultaneously live, and a maximum over them
    # returns one of the two. Charged to the outermost unit, which cannot
    # overlap another outermost unit, the sweep sums them instead.
    rows = [_raw("a", 6, 2, 9, ["candidate"], "inside_unit", ["outer", "attn"]),
            _raw("b", 6, 3, 8, ["candidate"], "inside_unit", ["outer", "mlp"])]
    partition = derive_partition(_synthetic_ledger(12, rows))
    assert partition["units"] == ["outer"]
    assert partition["terms"]["candidate_scratch"] == {"outer": 12}


def test_a_composition_below_the_observed_simultaneous_peak_refuses():
    # When every allocation is classified and charged, every observed byte is
    # inside some term, so a composition under the peak the replay actually saw
    # is a contradiction rather than conservatism -- and it is the error
    # direction that OOMs a box instead of wasting headroom on it.
    rows = [_raw("a", 6, 2, 9, ["candidate"], "inside_unit", ["u"]),
            _raw("b", 6, 3, 8, ["candidate"], "inside_unit", ["u"])]
    assert derive_partition(_synthetic_ledger(12, rows))["terms"]["candidate_scratch"]
    with pytest.raises(ValueError, match="below the observed simultaneous live peak"):
        derive_partition(_synthetic_ledger(13, rows))


def test_a_free_ordered_before_its_allocation_refuses_before_any_arithmetic(ledger):
    # The sweep would settle that free first, drive the running sum negative and
    # return a peak that hides live bytes rather than inflating them.
    ledger["torch_allocations"][-1]["free_completed_index"] = 0
    with pytest.raises(ValueError, match="freed before it is allocated"):
        derive_partition(ledger)


@pytest.mark.parametrize("size", [-40, 0, True])
def test_a_size_that_cannot_bound_a_lifetime_refuses_before_any_arithmetic(ledger, size):
    ledger["torch_allocations"][-1]["bytes"] = size
    with pytest.raises(ValueError, match="no positive integer size"):
        derive_partition(ledger)


def test_history_join_stays_open_when_no_record_join_was_observed(ledger):
    # Absent and null are "never observed", exactly as they are for
    # external_closure. Only an empty list is the observation that closes it.
    assert qualify_domains(ledger)["history_join"]["state"] == "closed"
    for absent in ({}, {"unattributed_external_records": None}):
        probe = dict(ledger)
        probe.pop("unattributed_external_records", None)
        probe.update(absent)
        domain = qualify_domains(probe)["history_join"]
        assert domain["state"] == "open", domain
        assert domain["evidence"] == []


def test_the_report_refuses_a_declared_member_whose_fields_are_not_this_schemas(ledger):
    for name in DECLARED_MEMBER_FIELDS:
        with pytest.raises(ValueError, match=f"report member {name} declares"):
            assemble_full_engine_resource_report(
                ledger, **_members(**{name: {"invented": "field"}}))


def test_the_report_refuses_a_declared_member_whose_every_field_is_null(ledger):
    # A dict of nulls is truthy, so "is it empty" was never the question.
    for name, fields in DECLARED_MEMBER_FIELDS.items():
        nulls = {field: None for field in fields}
        with pytest.raises(ValueError, match=f"declares every field null: {name}"):
            assemble_full_engine_resource_report(ledger, **_members(**{name: nulls}))


def test_an_execution_coordinate_outside_the_scope_refuses_instead_of_being_projected_over():
    # The partition stamps one composite topology. A caller declaring another
    # one used to get a report whose scope contradicted its own declaration.
    raw = json.loads((Path(__file__).parent
                      / "fixtures/full_engine_resource_ledger.json").read_text())
    led = analyze_engine_resource_ledger(raw)
    for field, value in (("topology", "tp2"), ("graph_mode", "cudagraph"),
                         ("residency", "offloaded")):
        outside = dict(SUPPORTED_EXECUTION, **{field: value})
        with pytest.raises(ValueError, match="outside this schema's scope"):
            assemble_full_engine_resource_report(led, **_members(execution=outside))


# --- the declared step boundary ---------------------------------------------
#
# Without one, an allocation outside every unit interval has no lifetime class:
# charging it as scratch assumes once per step and calling it startup assumes
# never again. The capture already snapshots execute:N:begin and sample:N:end;
# declaring that pair as an interval is what turns the assumption into a read.


def test_an_outside_unit_transient_inside_a_declared_step_is_step_scratch():
    # The row the banked fixture cannot classify at all. Contained in one
    # declared step, it is that step's scratch -- and the term carries a swept
    # value rather than going null.
    stray = _raw("attn-workspace", 6, 3, 8, ["fixed"], "outside_units", [])
    partition = derive_partition(_synthetic_ledger(6, [stray], steps=[(2, 9)]))
    assert partition["unclassified_allocations"] == []
    assert partition["non_step_allocations"] == []
    assert partition["scope"]["step_coverage"] == "complete"
    assert partition["terms"]["fixed_scratch"] == 6
    assert partition["membership"][0]["lifetime_class"] == "scratch"


def test_a_transient_live_across_a_step_boundary_is_carried_not_scratch():
    # Allocated in one step and freed in the next. It is live while the second
    # step runs, so it cannot be that step's own invocation-local scratch;
    # fixed_activation and fixed_scratch are separate additive terms, so
    # charging it as carried neither double-counts it nor drops it.
    carried = _raw("carried", 6, 3, 12, ["fixed"], "outside_units", [])
    partition = derive_partition(
        _synthetic_ledger(6, [carried], steps=[(2, 9), (10, 15)]))
    assert partition["non_step_allocations"] == []
    assert partition["membership"][0]["lifetime_class"] == "activation"
    assert partition["terms"]["fixed_scratch"] == 0
    # fixed_activation carries no number on any ledger, because it depends on
    # worker_startup and nothing closes that at v1. The classification is what
    # this test can show; the price is what v1 still cannot express.
    assert partition["terms"]["fixed_activation"] is None


def test_a_transient_allocated_before_a_step_and_freed_inside_it_is_charged():
    # Liveness, not the allocation index alone. This row starts before the step
    # opens, so an index test would place it outside every step and drop bytes
    # that are live while the step runs.
    early = _raw("early", 6, 0, 5, ["fixed"], "outside_units", [])
    partition = derive_partition(_synthetic_ledger(6, [early], steps=[(2, 9)]))
    assert partition["non_step_allocations"] == []
    assert partition["membership"][0]["lifetime_class"] == "activation"


def test_a_transient_live_during_no_declared_step_is_priced_but_never_charged():
    # A startup transient: allocated and freed before the first step opens. It
    # is proven live during no step, and the seven terms compose one step, so no
    # term charges it. It is named, counted and priced beside the budget -- the
    # obligation is the maximum of the two, and an engine whose startup peak
    # exceeds its per-step budget still has to fit.
    startup = _raw("load-staging", 900, 0, 1, ["fixed"], "outside_units", [])
    step_row = _raw("in-step", 6, 3, 8, ["fixed"], "outside_units", [])
    partition = derive_partition(
        _synthetic_ledger(906, [startup, step_row], steps=[(2, 9)]))
    assert partition["unclassified_allocations"] == []
    assert partition["uncharged_allocations"] == []
    assert [row["allocation_id"] for row in partition["non_step_allocations"]] == ["load-staging"]
    assert partition["scope"]["non_step_allocation_count"] == 1
    assert partition["terms"]["fixed_scratch"] == 6
    assert partition["non_step_transient_peak_bytes"] == 900
    # 900 dwarfs the 6-byte per-step budget, which is the whole point: a gate
    # handed only the budget would admit a body that cannot finish starting.
    assert "floor" in partition["non_step_transient_peak_scope"]
    # The budget itself is null for the unrelated four-domain reason, so there
    # is nothing yet to take a maximum against.
    assert compose_scalar_budget(partition) is None


def test_partial_step_coverage_leaves_the_same_row_unclassified():
    # The inference is licensed by coverage, not by the presence of a step. An
    # allocation from an engine step nobody declared is live during a step that
    # is not in the list, so "live during no declared step" would stop being a
    # proof. Partial behaves exactly as unobserved does.
    stray = _raw("attn-workspace", 6, 3, 8, ["fixed"], "outside_units", [])
    partition = derive_partition(
        _synthetic_ledger(6, [stray], steps=[(2, 9)], executed=4))
    assert partition["scope"]["step_coverage"] == "partial"
    assert partition["scope"]["unclassified_allocation_count"] == 1
    assert "declared step boundary" in partition["unclassified_allocations"][0]["reason"]
    assert all(term is None for term in partition["terms"].values())
    assert partition["non_step_transient_peak_bytes"] is None


def test_an_unclassified_row_nulls_the_off_step_price_but_not_its_count():
    # A row with no owner could be an off-step transient too, so the peak over
    # the ones that are identified is not the peak. The count stays readable
    # regardless, so the hazard is visible before there is a price on it.
    startup = _raw("load-staging", 900, 0, 1, ["fixed"], "outside_units", [])
    blocker = _raw("no-owner", 6, 3, 8, [], "outside_units", [])
    partition = derive_partition(
        _synthetic_ledger(906, [startup, blocker], steps=[(2, 9)]))
    assert partition["scope"]["unclassified_allocation_count"] == 1
    assert partition["scope"]["non_step_allocation_count"] == 1
    assert partition["non_step_transient_peak_bytes"] is None


def test_an_off_step_price_needs_the_same_join_a_scratch_term_needs():
    # It is not one of the seven terms and does not share their gate: it needs
    # the ownership join and the external closure, and cache_capacity has no
    # bearing on it. An unresolved issue refuses both and takes the price away.
    startup = _raw("load-staging", 900, 0, 1, ["fixed"], "outside_units", [])
    ledger = _synthetic_ledger(900, [startup], steps=[(2, 9)])
    assert derive_partition(ledger)["non_step_transient_peak_bytes"] == 900
    ledger["issues"] = ["dropped CUPTI buffer"]
    assert derive_partition(ledger)["non_step_transient_peak_bytes"] is None


def test_the_report_carries_the_steps_a_consumer_reproduces_the_filter_from(ledger):
    # The off-step filter removes rows from every term. A consumer that cannot
    # read the same step intervals cannot reproduce that removal, and a producer
    # rule nobody else can check is the shape this schema exists to refuse.
    report = assemble_full_engine_resource_report(ledger, **_members())
    assert "step_intervals" in report["observations"]
    assert "step_coverage" in report["observations"]
    assert report["observations"]["step_coverage"]["state"] == "unobserved"
    assert report["derived"]["placement_obligation"] == \
        "max(scalar_budget_bytes, non_step_transient_peak_bytes)"
    assert report["derived"]["non_step_transient_peak_bytes"] is None
    assert "floor" in report["derived"]["non_step_transient_peak_scope"]


def test_no_caller_can_hand_the_classifier_a_table_of_steps():
    # The step table is read from the ledger for the same reason the domain
    # table is: a classification that closes on a caller's argument is
    # ``qualified: true`` spelled a third way.
    import inspect
    assert list(inspect.signature(classify_allocations).parameters) == ["ledger"]
