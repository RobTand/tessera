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
            "api_verified": True, "api_problems": [],
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
    pytest.importorskip("prismabuild")
    expected = json.loads((ROOT / prepare.EXPECTED_PATH).read_text())
    original = prepare.validate_successor(packet, expected)
    assert packet["cases"] == original["cases"] and packet["native_files"] == original["native_files"]
    assert hashlib.sha256((ROOT / prepare.ORIGINAL_PACKET_PATH).read_bytes()).hexdigest() == prepare.ORIGINAL_PACKET_SHA256
    demand, declarations = prepare.public_resource_demand(packet)
    assert demand == {"cpu": 2, "mem_gb": 16, "gpu": 1, "spool_gb": 3}
    assert {entry["root"] for entry in declarations} == {entry[2] for entry in prepare.ROOT_DECLARATIONS}
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
    "missing_contract", "wrong_owner", "wrong_chain", "missing_api", "stale_api",
    "unverified_identity", "unverified_predicate", "pool_hash", "published_hash",
    "changed_generation", "claim_invoked", "denial_synthesized", "fake_kind",
    "zero_need", "wrong_need", "stale_ledger", "incomplete_ledger", "changed_ledger",
    "positive_capacity", "unreadable_ledger", "wrong_revision"])
def test_current_source_and_effective_zero_guarantee_are_required(packet, defect):
    record = metadata(packet)
    if defect == "missing_contract": record["public_claim_contract"] = None
    elif defect == "wrong_owner": record["public_claim_contract"]["owner"] = "private-dispatcher"
    elif defect == "wrong_chain": record["public_claim_contract"]["chain"] = []
    elif defect == "missing_api": record["public_claim_contract"]["api_problems"] = ["PoolQueue.offers"]
    elif defect == "stale_api": record["public_claim_contract"]["api_verified"] = False
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
    from test_stageprev_793_prepare import _run_published_sdk
    _run_published_sdk("""
        import json
        import stageprev_793_claim_contract as contract
        manifest = json.loads((Path(sys.argv[1]) / "RUNTIME_VERSION.json").read_text())
        observed = contract.observe_current_claim_contract(manifest)
        assert observed["verified"] is True
        assert observed["claim_invoked"] is False and observed["denial_synthesized"] is False
    """)


@pytest.mark.parametrize("defect", ["old_hash", "fake_root", "fake_kind", "changed_gate", "fake_go"])
def test_successor_only_permits_exact_approved_operational_delta(packet, defect):
    candidate = deepcopy(packet)
    if defect == "old_hash": candidate["supersedes"]["sha256"] = "0" * 64
    elif defect == "fake_root": candidate["environment"]["TESSERA_793_DOCKER_ROOT"] = "/fixtures/fake-root"
    elif defect == "fake_kind": candidate["operative_admission"]["exclusion_kind"] = "fake_disk"
    elif defect == "changed_gate": candidate["requires_before_publication"].pop()
    elif defect == "fake_go": candidate["admission_approval"]["launch_authorized"] = True
    expected = json.loads((ROOT / prepare.EXPECTED_PATH).read_text())
    with pytest.raises(ValueError):
        prepare.proposal(candidate, expected, prepare.SL_COORDINATOR)


def test_pure_metadata_controls_do_not_import_the_public_sdk(packet, monkeypatch):
    import builtins
    actual_import = builtins.__import__

    def refuse_sdk(name, *args, **kwargs):
        if name == "prismabuild" or name.startswith("prismabuild."):
            raise AssertionError("Pure metadata control attempted a public SDK import")
        return actual_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_sdk)
    record = metadata(packet)
    assert prerequisites.evaluate(record, packet, now=NOW)["control_ready"] is True
    record["positive_kind_reservation"]["need"] += 1
    assert prerequisites.evaluate(record, packet, now=NOW)["control_ready"] is False

def _client_double(**overrides):
    queue = type("PoolQueue", (), {
        "claim": staticmethod(lambda **kwargs: None),
        "_claim": staticmethod(lambda **kwargs: None),
        "_claim_pass": staticmethod(lambda **kwargs: None),
        "ledger": staticmethod(lambda host=None: None),
        "latest_denials": staticmethod(lambda keys, *, include_local=True: {}),
        "offers": staticmethod(lambda *, max_age_s=120.0: []),
    })
    ledger = type("ResourceLedger", (), {
        "capacity_census": lambda self: ({}, []),
        "available": lambda self: {},
    })
    for name, value in overrides.items():
        if name == "ledger_cls":
            ledger = value
        else:
            setattr(queue, name, value)
    return type("client", (), {"PoolQueue": queue}), ledger


def test_required_api_check_passes_on_full_contract():
    module, ledger = _client_double()
    assert claim_contract.required_api_problems(module, ledger) == []


def test_required_api_check_binds_receiver_whatever_name_it_uses():
    queue = type("PoolQueue", (), {
        "claim": staticmethod(lambda **kwargs: None),
        "_claim": staticmethod(lambda **kwargs: None),
        "_claim_pass": staticmethod(lambda **kwargs: None),
        "ledger": lambda this, host=None: None,
        "latest_denials": lambda this, keys, *, include_local=True: {},
        "offers": lambda this, *, max_age_s=120.0: [],
    })
    ledger = type("ResourceLedger", (), {
        "capacity_census": lambda this: ({}, []),
        "available": lambda this: {},
    })
    module = type("client", (), {"PoolQueue": queue})
    assert claim_contract.required_api_problems(module, ledger) == []


def test_required_api_check_binds_classmethod_receiver():
    module, ledger = _client_double(ledger=classmethod(lambda cls, host=None: None))
    assert claim_contract.required_api_problems(module, ledger) == []


def test_required_api_check_refuses_static_receiver_left_in_signature():
    module, ledger = _client_double(offers=staticmethod(lambda self, *, max_age_s=120.0: []))
    assert "PoolQueue.offers(max_age_s)" in claim_contract.required_api_problems(module, ledger)


def test_required_api_check_names_each_absent_call():
    for name in claim_contract.REQUIRED_QUEUE_METHODS:
        module, ledger = _client_double(**{name: None})
        problems = claim_contract.required_api_problems(module, ledger)
        assert "PoolQueue." + name in problems
    for name in claim_contract.REQUIRED_LEDGER_METHODS:
        module, ledger = _client_double(ledger_cls=type("ResourceLedger", (), {}))
        problems = claim_contract.required_api_problems(module, ledger)
        assert "ResourceLedger." + name in problems


def test_required_api_check_names_incompatible_signatures():
    module, ledger = _client_double(ledger=staticmethod(lambda: None))
    assert "PoolQueue.ledger(host)" in claim_contract.required_api_problems(module, ledger)
    module, ledger = _client_double(latest_denials=staticmethod(lambda keys: {}))
    assert "PoolQueue.latest_denials(keys,include_local)" in claim_contract.required_api_problems(module, ledger)
    module, ledger = _client_double(offers=staticmethod(lambda: []))
    assert "PoolQueue.offers(max_age_s)" in claim_contract.required_api_problems(module, ledger)
    ledger = type("ResourceLedger", (), {
        "capacity_census": lambda self, extra: ({}, []),
        "available": lambda self: {},
    })
    module, _ = _client_double()
    assert "ResourceLedger.capacity_census()" in claim_contract.required_api_problems(module, ledger)


def test_required_api_check_refuses_extra_required_arguments():
    module, ledger = _client_double(ledger=staticmethod(lambda host, region: None))
    assert "PoolQueue.ledger(host)" in claim_contract.required_api_problems(module, ledger)
    module, ledger = _client_double(latest_denials=staticmethod(lambda keys, scope, *, include_local=False: {}))
    assert "PoolQueue.latest_denials(keys,include_local)" in claim_contract.required_api_problems(module, ledger)
    module, ledger = _client_double(offers=staticmethod(lambda *, max_age_s, limit: []))
    assert "PoolQueue.offers(max_age_s)" in claim_contract.required_api_problems(module, ledger)
    ledger = type("ResourceLedger", (), {
        "available": lambda self, unreadable: {},
        "capacity_census": lambda self: ({}, []),
    })
    module, _ = _client_double()
    assert "ResourceLedger.available()" in claim_contract.required_api_problems(module, ledger)


def test_required_api_check_refuses_denials_keyword_collision():
    module, ledger = _client_double(
        latest_denials=staticmethod(lambda include_local, keys: {}))
    assert "PoolQueue.latest_denials(keys,include_local)" in claim_contract.required_api_problems(module, ledger)
    module, ledger = _client_double(offers=staticmethod(lambda max_age_s: []))
    assert claim_contract.required_api_problems(module, ledger) == []


def test_evaluator_refuses_a_contract_with_api_problems(packet):
    record = metadata(packet)
    record["public_claim_contract"]["api_verified"] = False
    record["public_claim_contract"]["api_problems"] = ["PoolQueue.offers"]
    result = prerequisites.evaluate(record, packet, now=NOW)
    assert result["status"] == "HOLD" and result["control_ready"] is False


def test_observer_reports_receiver_name_variants_without_false_refusal(monkeypatch):
    pytest.importorskip("prismabuild.client")
    import prismabuild.client as published_client
    real_queue = published_client.PoolQueue

    class RenamedQueue(real_queue):
        def ledger(this, host=None):
            return super().ledger(host)

        def latest_denials(this, keys, *, include_local=True):
            return super().latest_denials(keys, include_local=include_local)

        def offers(this, *, max_age_s=120.0):
            return super().offers(max_age_s=max_age_s)

    monkeypatch.setattr(published_client, "PoolQueue", RenamedQueue)
    import stageprev_793_prepare as prepare_module
    monkeypatch.setattr(prepare_module, "PB_ROOT", "/mnt/shared/prismabuild-fleet/repo")
    probe = claim_contract.observe_current_claim_contract(
        {"generation": "renamed-receiver-probe", "files": {"src/prismabuild/pool.py": "0" * 64}})
    manifest = {"generation": "renamed-receiver-probe",
                "files": {"src/prismabuild/pool.py": probe["pool_sha256"]}}
    observed = claim_contract.observe_current_claim_contract(manifest)
    assert observed["api_problems"] == [] and observed["api_verified"] is True


def test_collector_refuses_static_receiver_left_in_signature(monkeypatch, packet):
    pytest.importorskip("prismabuild.client")
    import prismabuild.client as published_client
    real_queue = published_client.PoolQueue

    class StaticQueue(real_queue):
        offers = staticmethod(lambda self, *, max_age_s=120.0: [])

    monkeypatch.setattr(published_client, "PoolQueue", StaticQueue)
    with __import__("pytest").raises(ValueError, match="PoolQueue.offers"):
        prerequisites.collect(packet, "")


@pytest.mark.parametrize("defect", ["absent_name", "incompatible_signature"])
def test_observer_refuses_required_api_defects(defect):
    from test_stageprev_793_prepare import _run_published_sdk

    mutation = ("del client.PoolQueue.offers" if defect == "absent_name" else
                "client.PoolQueue.offers = staticmethod(lambda self, *, max_age_s=120.0: [])")
    problem = ("PoolQueue.offers" if defect == "absent_name" else
               "PoolQueue.offers(max_age_s)")
    _run_published_sdk(f"""
        import json
        import stageprev_793_claim_contract as contract
        manifest = json.loads((Path(sys.argv[1]) / "RUNTIME_VERSION.json").read_text())
        assert contract.observe_current_claim_contract(manifest)["verified"] is True
        {mutation}
        observed = contract.observe_current_claim_contract(manifest)
        assert observed["api_problems"] == [{problem!r}]
        assert observed["api_verified"] is False
        assert observed["identity_verified"] is False
        assert observed["verified"] is False
        assert observed["claim_invoked"] is False
        assert observed["denial_synthesized"] is False
    """)


def test_collector_refuses_absent_required_api_name():
    from test_stageprev_793_prepare import _run_published_sdk

    _run_published_sdk("""
        import json
        import pytest
        import stageprev_793_prepare as prepare
        import stageprev_793_prerequisites as prerequisites
        packet = json.loads((prepare.SOURCE_ROOT / prepare.PACKET_PATH).read_text())
        del client.PoolQueue.offers
        with pytest.raises(ValueError, match=r"Actual published public claim API lacks: PoolQueue\\.offers"):
            prerequisites.collect(packet, "")
    """)
