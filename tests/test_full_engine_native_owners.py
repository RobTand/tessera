import copy
import pytest
from experiments.full_engine_native_owners import checkpoint_site_owners


@pytest.fixture
def witness():
    frame = {"filename": "fixture.py", "line": 12, "name": "allocate_workspace"}
    evidence = {"rule": {"schema": "tessera.native_allocation_site_owner_rule.v1", "category": "shared",
        "rule_id": "fixture", "required_source_files": {"fixture.py": {"sha256": "a" * 64}},
        "required_mapped_library": {"path": "fixture.so", "bytes": 20, "sha256": "b" * 64},
        "required_allocation_frames_in_order": [frame], "evidence_kind": "exact allocation site",
        "assignment_dependence": "unresolved"}, "sources": {"fixture.py": {"sha256": "a" * 64}},
        "mapped_libraries": {"fixture.so": {"bytes": 20, "sha256": "b" * 64}}}
    cp = {"segments": [{"device": 0, "blocks": [{"state": "active_allocated", "address": 100, "requested_size": 32}]}]}
    live = {100: {"allocate_index": 0, "allocation_id": "0:100:2", "bytes": 32, "free_requested_index": None}}
    return cp, live, [{"frames": [dict(frame)]}], evidence


def test_site_rule_uses_actual_live_generation_size(witness):
    owners = checkpoint_site_owners(*witness, 0)
    assert len(owners) == 1 and owners[0]["bytes"] == 32
    assert owners[0]["native_binding"] == {"rule_id": "fixture", "allocation_id": "0:100:2", "native_getter_observed": False}


@pytest.mark.parametrize("defect", ["source", "library", "frame", "freed", "inactive", "size", "reused"])
def test_site_rule_rejects_identity_lifetime_and_stack_changes(witness, defect):
    cp, live, history, evidence = copy.deepcopy(witness)
    if defect == "source":
        evidence["sources"]["fixture.py"]["sha256"] = "c" * 64
    elif defect == "library":
        evidence["mapped_libraries"] = {}
    elif defect == "frame":
        history[0]["frames"][0]["line"] = 13
    elif defect == "freed":
        live[100]["free_requested_index"] = 1
    elif defect == "inactive":
        cp["segments"][0]["blocks"][0]["state"] = "active_awaiting_free"
    elif defect == "size":
        cp["segments"][0]["blocks"][0]["requested_size"] = 31
    elif defect == "reused":
        history.append({"frames": []}); live[100]["allocate_index"] = 1
    if defect in {"source", "library"}:
        with pytest.raises(ValueError):
            checkpoint_site_owners(cp, live, history, evidence, 0)
    else:
        assert checkpoint_site_owners(cp, live, history, evidence, 0) == []
