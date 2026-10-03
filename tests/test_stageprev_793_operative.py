"""Changed operative controls only; fixtures are NOT actual admission evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/t8r_speed"))
import stageprev_793_prepare as prepare
import stageprev_793_prerequisites as prerequisites

NOW = 1000.0
KEY = "1" * 64


@pytest.fixture
def packet():
    return json.loads((ROOT / prepare.PACKET_PATH).read_text())


def metadata(packet):
    root_paths = [packet["environment"][root] for root, _, _ in prepare.ROOT_DECLARATIONS]
    storage_paths = [packet["environment"]["PB_CLIENT_ROOT"],
                     str(Path(packet["readset"]["path"]).parent), packet["source"]["out"]]
    return {
        "schema": prerequisites.SCHEMA, "source": prerequisites.SDK_SOURCE,
        "queue_root": prepare.QUEUE_ROOT, "observed_unix": NOW,
        "packet_sha256": hashlib.sha256((ROOT / prepare.PACKET_PATH).read_bytes()).hexdigest(),
        "gpu_submitted": False, "coordinator_host": "sparklina",
        "runtime_manifest": {"generation": "explicitCPUfixture-not-actual-runtime-proof"},
        "runtime_changed_during_read": False,
        "excluded_ledger": {"host": "sparky", "total": {"spool_gb": 0},
                            "available": {"spool_gb": 0}, "complete": True,
                            "unreadable": [], "changed_during_read": False,
                            "directories_present": True, "observed_unix": NOW},
        "refusal_action_key": KEY,
        "actual_denials": [{"host": "sparky", "action_key": KEY,
                            "reason": "never_fits_capacity", "denied_unix": NOW,
                            "evidence": {"reservation_demand": {"spool_gb": 1},
                                         "capacity_total": {"spool_gb": 0}}}],
        "excluded_offer": None,
        "offer_observation": "UNKNOWN_OMITTED_NOT_A_ZERO_MEASUREMENT",
        "used_filesystems": [
            {"host": "sparklina", "device": 1, "paths": root_paths,
             "size_bytes": 200 * (1 << 30), "available_bytes": 30 * (1 << 30),
             "aggregate_root_allowance_bytes": 3 * (1 << 30), "observed_unix": NOW},
            {"host": "sparklina", "device": 2, "paths": storage_paths,
             "size_bytes": 200 * (1 << 30), "available_bytes": 30 * (1 << 30),
             "aggregate_root_allowance_bytes": 0, "observed_unix": NOW}
        ],
        "filesystem_errors": [], "coordinator_root_census_unreadable": [],
        "conservative_current_root_reservation_bytes": 0,
        "root_growth_sum_bytes": 3 * (1 << 30)
    }


def test_actual_public_root_max_terms_and_immutable_original(packet):
    expected = json.loads((ROOT / prepare.EXPECTED_PATH).read_text())
    original = prepare.validate_successor(packet, expected)
    assert packet["cases"] == original["cases"] and packet["native_files"] == original["native_files"]
    assert hashlib.sha256((ROOT / prepare.ORIGINAL_PACKET_PATH).read_bytes()).hexdigest() == prepare.ORIGINAL_PACKET_SHA256
    demand, declarations = prepare.public_resource_demand(packet)
    assert demand == {"cpu": 2, "mem_gb": 16, "gpu": 1, "spool_gb": 3}
    assert {entry["root"] for entry in declarations} == {"/tmp", "/home/rob/tmp", "/var/lib/docker"}
    assert sum(entry["max_bytes"] for entry in declarations) == 3 * (1 << 30)
    record = prepare.proposal(packet, expected, prepare.SL_COORDINATOR)
    assert "spool_gb=3" not in record["argv_proposal_only"]
    assert "--tag" not in record["argv_proposal_only"] and "--here" not in record["argv_proposal_only"]
    assert record["process_order"] == ["master", "fix", "fix", "master"]
    assert record["status"] == "GPU_HOLD"


def test_complete_zero_and_actual_current_positive_refusal_is_not_gpu_go(packet):
    result = prerequisites.evaluate(metadata(packet), packet, now=NOW)
    assert result["control_ready"] is True
    assert result["status"] == "CONTROL_VALID_NOT_GPU_GO"
    assert result["remaining_gpu_gates"]


@pytest.mark.parametrize("defect", [
    "missing", "wrong_schema", "wrong_queue", "wrong_source", "stale", "future",
    "no_ledger", "incomplete", "unreadable", "changed", "missing_directory",
    "stale_ledger", "positive_total", "positive_available", "no_key",
    "no_denial", "null_denials", "scalar_predicate", "wrong_host", "wrong_action",
    "wrong_reason", "stale_denial", "zero_demand", "fake_kind", "nonzero_denial_capacity",
    "wrong_packet", "gpu_scope", "runtime_change", "unreadable_root_aggregate",
    "filesystem_error", "no_filesystems", "missing_root_path", "below_floor",
    "insufficient_allowance", "missing_other_reservation", "bad_paths", "root_sum"
])
def test_actual_gate_refuses_unknown_stale_changed_or_unbound_inputs(packet, defect):
    record = metadata(packet)
    if defect == "missing": record = None
    elif defect == "wrong_schema": record["schema"] = "other"
    elif defect == "wrong_queue": record["queue_root"] = "/mnt/shared/private-queue"
    elif defect == "wrong_source": record["source"] = "scalar-source-predicate"
    elif defect == "stale": record["observed_unix"] -= 121
    elif defect == "future": record["observed_unix"] += 1
    elif defect == "no_ledger": record["excluded_ledger"] = None
    elif defect == "incomplete": record["excluded_ledger"]["complete"] = False
    elif defect == "unreadable": record["excluded_ledger"]["unreadable"] = [{"holder": "unknown"}]
    elif defect == "changed": record["excluded_ledger"]["changed_during_read"] = True
    elif defect == "missing_directory": record["excluded_ledger"]["directories_present"] = False
    elif defect == "stale_ledger": record["excluded_ledger"]["observed_unix"] -= 121
    elif defect == "positive_total": record["excluded_ledger"]["total"]["spool_gb"] = 1
    elif defect == "positive_available": record["excluded_ledger"]["available"]["spool_gb"] = 1
    elif defect == "no_key": record["refusal_action_key"] = ""
    elif defect == "no_denial": record["actual_denials"] = []
    elif defect == "null_denials": record["actual_denials"] = None
    elif defect == "scalar_predicate": record["actual_denials"] = [{"source_predicate": "0 < 1"}]
    elif defect == "wrong_host": record["actual_denials"][0]["host"] = "sparklina"
    elif defect == "wrong_action": record["actual_denials"][0]["action_key"] = "2" * 64
    elif defect == "wrong_reason": record["actual_denials"][0]["reason"] = "placement_mismatch"
    elif defect == "stale_denial": record["actual_denials"][0]["denied_unix"] -= 121
    elif defect == "zero_demand": record["actual_denials"][0]["evidence"]["reservation_demand"]["spool_gb"] = 0
    elif defect == "fake_kind": record["actual_denials"][0]["evidence"]["reservation_demand"] = {"invented_root": 1}
    elif defect == "nonzero_denial_capacity": record["actual_denials"][0]["evidence"]["capacity_total"]["spool_gb"] = 1
    elif defect == "wrong_packet": record["packet_sha256"] = "0" * 64
    elif defect == "gpu_scope": record["gpu_submitted"] = True
    elif defect == "runtime_change": record["runtime_changed_during_read"] = True
    elif defect == "unreadable_root_aggregate": record["coordinator_root_census_unreadable"] = ["unknown"]
    elif defect == "filesystem_error": record["filesystem_errors"] = [{"path": "/tmp", "error": "unreadable"}]
    elif defect == "no_filesystems": record["used_filesystems"] = []
    elif defect == "missing_root_path": record["used_filesystems"][0]["paths"].pop()
    elif defect == "below_floor": record["used_filesystems"][0]["available_bytes"] = 12 * (1 << 30)
    elif defect == "insufficient_allowance": record["used_filesystems"][0]["aggregate_root_allowance_bytes"] = 0
    elif defect == "missing_other_reservation": record["conservative_current_root_reservation_bytes"] = 1 << 30
    elif defect == "bad_paths": record["used_filesystems"][0]["paths"] = None
    elif defect == "root_sum": record["root_growth_sum_bytes"] = 1 << 30
    result = prerequisites.evaluate(record, packet, now=NOW)
    assert result["status"] == "HOLD" and result["control_ready"] is False


def test_an_omitted_or_stale_offer_alone_never_establishes_zero(packet):
    record = metadata(packet)
    record["excluded_ledger"]["complete"] = False
    record["actual_denials"] = []
    record["excluded_offer"] = {"capacity": {"spool_gb": 0}, "announced_unix": NOW - 10000}
    assert prerequisites.evaluate(record, packet, now=NOW)["control_ready"] is False


@pytest.mark.parametrize("defect", ["old_hash", "fake_root", "fake_kind", "changed_gate", "fake_go"])
def test_successor_only_permits_exact_approved_operational_delta(packet, defect):
    candidate = deepcopy(packet)
    if defect == "old_hash": candidate["supersedes"]["sha256"] = "0" * 64
    elif defect == "fake_root": candidate["environment"]["TESSERA_793_DOCKER_ROOT"] = "/mnt/shared/fake-root"
    elif defect == "fake_kind": candidate["operative_admission"]["exclusion_kind"] = "fake_disk"
    elif defect == "changed_gate": candidate["requires_before_publication"].pop()
    elif defect == "fake_go": candidate["admission_approval"]["launch_authorized"] = True
    expected = json.loads((ROOT / prepare.EXPECTED_PATH).read_text())
    with pytest.raises(ValueError):
        prepare.proposal(candidate, expected, prepare.SL_COORDINATOR)
