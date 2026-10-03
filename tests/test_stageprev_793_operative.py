"""Operative CPU fixtures are not actual admission evidence; old47 retained."""
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
import stageprev_793_claim_contract as claim_contract

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
        "admission_revision_sha256": hashlib.sha256((ROOT / prerequisites.REVISION_PATH).read_bytes()).hexdigest(),
        "gpu_submitted": False, "coordinator_host": "sparklina",
        "runtime_manifest": {"generation": "explicitCPUfixture-not-actual-runtime-proof",
                             "files": {"src/prismabuild/pool.py": "a" * 64}},
        "runtime_changed_during_read": False,
        "public_claim_contract": {"owner": claim_contract.OWNER, "chain": claim_contract.CHAIN,
            "sdk_version": 4, "verified": True, "identity_verified": True,
            "chain_verified": True, "positive_reservation_refusal_verified": True,
            "observation_only": True, "claim_invoked": False, "denial_synthesized": False,
            "pool_sha256": "a" * 64, "published_pool_sha256": "a" * 64,
            "generation": "explicitCPUfixture-not-actual-runtime-proof"},
        "positive_kind_reservation": {"kind": "spool_gb", "need": 3},
        "excluded_ledger": {"host": "sparky", "total": {"spool_gb": 0},
                            "available": {"spool_gb": 0}, "complete": True,
                            "unreadable": [], "changed_during_read": False,
                            "directories_present": True, "observed_unix": NOW},
        "refusal_action_key": KEY, "actual_denials": [],
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


def test_complete_zero_and_verified_current_claim_refusal_contract_is_not_gpu_go(packet):
    result = prerequisites.evaluate(metadata(packet), packet, now=NOW)
    assert result["control_ready"] is True
    assert result["status"] == "CONTROL_VALID_NOT_GPU_GO"
    assert result["remaining_gpu_gates"]


@pytest.mark.parametrize("events", [None, [], [{"source_predicate": "0 < 1"}],
    [{"host": "sparky", "reason": "already_claimed", "denied_unix": NOW}],
    [{"host": "sparky", "reason": "never_fits_capacity", "denied_unix": NOW - 10000}]])
def test_denial_events_are_observed_only_not_a_new_execution_gate(packet, events):
    record = metadata(packet)
    record["actual_denials"] = events
    assert prerequisites.evaluate(record, packet, now=NOW)["control_ready"] is True


@pytest.mark.parametrize("defect", [
    "missing_contract", "wrong_owner", "wrong_chain", "wrong_sdk", "unverified_identity",
    "unverified_predicate", "pool_hash", "published_hash", "changed_generation",
    "claim_invoked", "denial_synthesized", "fake_kind", "zero_need", "wrong_need",
    "stale_ledger", "incomplete_ledger", "changed_ledger", "positive_capacity",
    "unreadable_ledger", "wrong_revision"
])
def test_current_source_and_effective_zero_guarantee_are_required(packet, defect):
    record = metadata(packet)
    if defect == "missing_contract": record["public_claim_contract"] = None
    elif defect == "wrong_owner": record["public_claim_contract"]["owner"] = "private-dispatcher"
    elif defect == "wrong_chain": record["public_claim_contract"]["chain"] = []
    elif defect == "wrong_sdk": record["public_claim_contract"]["sdk_version"] = 3
    elif defect == "unverified_identity": record["public_claim_contract"]["identity_verified"] = False
    elif defect == "unverified_predicate": record["public_claim_contract"]["positive_reservation_refusal_verified"] = False
    elif defect == "pool_hash": record["public_claim_contract"]["pool_sha256"] = "b" * 64
    elif defect == "published_hash": record["public_claim_contract"]["published_pool_sha256"] = "b" * 64
    elif defect == "changed_generation": record["public_claim_contract"]["generation"] = "other-generation"
    elif defect == "claim_invoked": record["public_claim_contract"]["claim_invoked"] = True
    elif defect == "denial_synthesized": record["public_claim_contract"]["denial_synthesized"] = True
    elif defect == "fake_kind": record["positive_kind_reservation"]["kind"] = "fake_disk"
    elif defect == "zero_need": record["positive_kind_reservation"]["need"] = 0
    elif defect == "wrong_need": record["positive_kind_reservation"]["need"] = 1
    elif defect == "stale_ledger": record["excluded_ledger"]["observed_unix"] -= 121
    elif defect == "incomplete_ledger": record["excluded_ledger"]["complete"] = False
    elif defect == "changed_ledger": record["excluded_ledger"]["changed_during_read"] = True
    elif defect == "positive_capacity": record["excluded_ledger"]["total"]["spool_gb"] = 1
    elif defect == "unreadable_ledger": record["excluded_ledger"]["unreadable"] = ["unknown"]
    elif defect == "wrong_revision": record["admission_revision_sha256"] = "0" * 64
    result = prerequisites.evaluate(record, packet, now=NOW)
    assert result["status"] == "HOLD" and result["control_ready"] is False


def test_an_omitted_or_stale_offer_alone_never_establishes_zero(packet):
    record = metadata(packet)
    record["excluded_ledger"]["complete"] = False
    record["public_claim_contract"] = None
    record["excluded_offer"] = {"capacity": {"spool_gb": 0}, "announced_unix": NOW - 10000}
    assert prerequisites.evaluate(record, packet, now=NOW)["control_ready"] is False


SOURCE_BRANCH = '''
def decision():
    if any(total.get(kind, 0) < need for kind, need in reservation_demand.items()):
        self.record_denial(item, "never_fits_capacity", {"capacity_total": total})
        continue
'''


def test_exact_current_refusal_predicate_requires_the_fail_closed_continuation():
    assert claim_contract.positive_kind_refusal_branch(SOURCE_BRANCH) is True
    assert claim_contract.positive_kind_refusal_branch(SOURCE_BRANCH.replace("< need", "> need")) is False
    assert claim_contract.positive_kind_refusal_branch(SOURCE_BRANCH.replace("continue", "pass")) is False
    assert claim_contract.positive_kind_refusal_branch(SOURCE_BRANCH.replace("never_fits_capacity", "other")) is False
    assert claim_contract.positive_kind_refusal_branch(SOURCE_BRANCH.replace("reservation_demand", "invented_demand")) is False


def test_actual_loaded_public_claim_identity_matches_published_manifest():
    from prismabuild.client import SDK_VERSION
    assert SDK_VERSION == 4
    manifest = json.loads((Path(prepare.PB_ROOT) / "RUNTIME_VERSION.json").read_text())
    observed = claim_contract.observe_current_claim_contract(manifest)
    assert observed["verified"] is True
    assert observed["claim_invoked"] is False and observed["denial_synthesized"] is False


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
