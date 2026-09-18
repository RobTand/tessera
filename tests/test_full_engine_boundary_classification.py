"""The two-capture boundary classification (tessera#548).

Every test here is a pure-dict regression: two hand-built v2 boundary ledgers,
no capture on disk, no engine, no GPU. A passing suite establishes that the
comparison is built from what both captures observed and that the rule fires
where its statement says it does. It establishes nothing about any real
engine's ownership: a classification is evidence for the two captures it
names, and nothing here makes it a universal invariance.
"""
import pytest

from experiments.full_engine_boundary_classification import (
    BOUNDARY_CLASSIFICATION_SCHEMA, boundary_classification,
)
from experiments.full_engine_ownership import RULES, derive_owner_views

LEDGER_V2 = "tessera.full_engine_raw_resource_ledger.v2"

#: Everything a capture's identity fixes. #548 holds the runtime, the roster
#: and the workload equal and changes one input: the selected assignment.
FIRST_IDENTITY = {
    "model_sha256": "m" * 64,
    "configuration_sha256": "c" * 64,
    "runtime_manifest_sha256": "r" * 64,
    "assignment_sha256": "1" * 64,
    "canonical_units_sha256": "u" * 64,
    "workload_sha256": "w" * 64,
}
SECOND_IDENTITY = dict(FIRST_IDENTITY, assignment_sha256="2" * 64,
                       configuration_sha256="d" * 64)

BOUNDARY = "native:l:model.layers.0.self_attn.o_proj:0:input.x"
ROOT = "runner:input_batch.token_ids_cpu"


def _row(allocation_id, owner, size):
    return {"allocation_id": allocation_id, "bytes": size, "allocate_index": 30,
            "free_completed_index": None, "observed_categories": ["shared"],
            "observed_owners": [owner], "scope_stack": []}


def _ledger(capture_sha256, identity, rows):
    """A v2 boundary ledger carrying one pending_548 view per row."""
    views = [{"allocation_id": row["allocation_id"], "class": None, "unit": None,
              "rule": "pending_548", "site": None,
              "reason": "census shared (native boundary tensor); assignment invariance is "
                        "tessera#548's two-assignment measurement"}
             for row in rows]
    return {"schema": LEDGER_V2, "capture_sha256": capture_sha256, "identity": identity,
            "torch_allocations": rows,
            "owner_views": {"schema": "tessera.full_engine_ownership_observation.v1",
                            "views": {"views": views}}}


def _pair(first_rows, second_rows):
    return (_ledger("a" * 64, FIRST_IDENTITY, first_rows),
            _ledger("b" * 64, SECOND_IDENTITY, second_rows))


def _views(rows, classification, capture_sha256="a" * 64):
    views, _summary = derive_owner_views(
        rows, {}, checkpoint_index={"before_model_load": 5, "ready_for_workload": 20},
        roster=[], evidence={"plugin_package_path": "/img/site-packages/tessera",
                             "plugin_files": set(), "vllm_root": "/img/site-packages/vllm",
                             "vllm_files": set(), "observer_roots": [],
                             "inventory_digests": {}},
        boundary_classification=classification, capture_sha256=capture_sha256)
    return views


# --- the artifact ------------------------------------------------------------

def test_the_classification_names_both_captures_and_carries_no_verdict():
    first, second = _pair([_row("0:1:1", BOUNDARY, 4096)], [_row("0:9:1", BOUNDARY, 4096)])
    record = boundary_classification(first, second)
    assert record["schema"] == BOUNDARY_CLASSIFICATION_SCHEMA
    assert [capture["capture_sha256"] for capture in record["captures"]] == ["a" * 64, "b" * 64]
    assert record["changed_input"] == "assignment_sha256"
    site, = record["sites"]
    assert site["owner"] == BOUNDARY
    assert site["bytes"] == {"a" * 64: 4096, "b" * 64: 4096}
    # The verdict belongs to whoever recomputes it. This artifact carries the
    # two byte figures and the two capture identities, and nothing else.
    assert "class" not in site and "verdict" not in site and "fixed" not in site


def test_a_pair_that_changed_more_than_the_assignment_is_refused():
    first, second = _pair([_row("0:1:1", BOUNDARY, 4096)], [_row("0:9:1", BOUNDARY, 4096)])
    second["identity"] = dict(SECOND_IDENTITY, workload_sha256="x" * 64)
    with pytest.raises(ValueError, match="workload_sha256"):
        boundary_classification(first, second)


def test_a_pair_under_one_assignment_is_not_a_substitution():
    first, second = _pair([_row("0:1:1", BOUNDARY, 4096)], [_row("0:9:1", BOUNDARY, 4096)])
    second["identity"] = dict(SECOND_IDENTITY, assignment_sha256="1" * 64)
    with pytest.raises(ValueError, match="one assignment"):
        boundary_classification(first, second)


def test_a_v1_ledger_carries_no_views_and_cannot_be_compared():
    first, second = _pair([_row("0:1:1", BOUNDARY, 4096)], [_row("0:9:1", BOUNDARY, 4096)])
    first["schema"] = "tessera.full_engine_raw_resource_ledger.v1"
    with pytest.raises(ValueError, match="v2 boundary ledger"):
        boundary_classification(first, second)


# --- the rule ----------------------------------------------------------------

def test_a_site_with_the_same_bytes_in_both_captures_is_evidence_for_fixed():
    first, second = _pair([_row("0:1:1", BOUNDARY, 4096)], [_row("0:9:1", BOUNDARY, 4096)])
    record = boundary_classification(first, second)
    view, = _views(first["torch_allocations"], record)
    assert (view["class"], view["rule"]) == ("fixed", "two_capture:agreed")
    # The report has to say which two captures the agreement is evidence for.
    assert "a" * 64 in view["reason"] and "b" * 64 in view["reason"]
    assert RULES["two_capture:agreed"]["class"] == "fixed"


def test_a_site_whose_bytes_move_is_a_candidate_that_owes_a_unit():
    first, second = _pair([_row("0:1:1", BOUNDARY, 4096)], [_row("0:9:1", BOUNDARY, 8192)])
    record = boundary_classification(first, second)
    view, = _views(first["torch_allocations"], record)
    assert (view["class"], view["rule"]) == ("candidate", "two_capture:moved")
    assert view["unit"] == "l:model.layers.0.self_attn.o_proj"
    assert "a" * 64 in view["reason"] and "b" * 64 in view["reason"]


def test_a_moved_site_the_owner_does_not_name_a_unit_for_says_so():
    first, second = _pair([_row("0:1:1", ROOT, 512)], [_row("0:9:1", ROOT, 1024)])
    record = boundary_classification(first, second)
    view, = _views(first["torch_allocations"], record)
    assert (view["class"], view["rule"]) == ("candidate", "two_capture:moved")
    assert view["unit"] is None and "owes a unit" in view["reason"]


def test_a_site_present_in_only_one_capture_stays_unclassified():
    first, second = _pair([_row("0:1:1", BOUNDARY, 4096)], [_row("0:9:1", ROOT, 512)])
    record = boundary_classification(first, second)
    view, = _views(first["torch_allocations"], record)
    assert view["class"] is None and view["rule"] == "pending_548"
    assert "one capture" in view["reason"]


def test_without_a_classification_every_pending_row_stays_pending():
    first, _second = _pair([_row("0:1:1", BOUNDARY, 4096)], [])
    view, = _views(first["torch_allocations"], None)
    assert view["class"] is None and view["rule"] == "pending_548"


def test_a_classification_that_does_not_name_this_capture_is_refused():
    first, second = _pair([_row("0:1:1", BOUNDARY, 4096)], [_row("0:9:1", BOUNDARY, 4096)])
    record = boundary_classification(first, second)
    with pytest.raises(ValueError, match="names captures"):
        _views(first["torch_allocations"], record, capture_sha256="f" * 64)
