"""CPU regressions for the two checked observations and their two-pass join.

These fixtures are not engine evidence and close no domain on their own. What
they establish is the contract: the record shapes the consumer recomputes from,
the refusals that keep an intrusive pass from certifying itself, and the join
that binds two different processes to one configured run without ever comparing
their pointers.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.full_engine_kv import (COMMON_RUN_DIGESTS, admission_evidence,
                                        kv_observation_record, read_only_kv_observation)
from experiments.full_engine_resource_partition import (
    assemble_full_engine_resource_report, cache_capacity_closed, qualify_domains,
    worker_startup_closed)
from experiments.full_engine_resources import _identity, worker_startup_record
from experiments.report_full_engine_resources import first_existing, join_observation_passes
from experiments.capture_full_engine_resources import read_routed_owner_receipt

#: The consumer's own exact field set, restated here rather than imported: the
#: consumer is another repository and a shape that drifts silently is exactly
#: what this test exists to catch.
WORKER_STARTUP_FIELDS = ("memory_allocated_bytes", "rank", "receipt_resident_bytes",
                         "scope", "workspace_locked", "workspace_resident_bytes")
KV_STORAGE_FIELDS = ("address", "bytes", "device_id", "device_type", "owners")

_DIGEST = "a" * 64


def _run_identity(**overrides):
    identity = {name: _DIGEST for name in COMMON_RUN_DIGESTS}
    identity.update(overrides)
    return identity


def _workspace(resident=4096, locked=True):
    return {"schema": "tessera.native_moe_workspace.v1", "owner": "vllm.WorkspaceManager",
            "num_ubatches": 1, "num_lanes": 1, "locked": locked,
            "slots": [{"index": 0, "allocation": None}], "resident_bytes": resident}


def _observed(num_blocks=8, pages=(1024, 512), bytes_each=2048):
    storages = [{"device_type": "cuda", "device_id": 0, "address": 4096 + index * bytes_each,
                 "bytes": bytes_each, "owners": [f"kv_caches[{index}]"]}
                for index in range(len(pages))]
    return {"num_blocks": num_blocks, "group_page_size_bytes": list(pages),
            "resolved_limits": {"max_model_len": 4096, "max_num_seqs": 4,
                                "max_num_batched_tokens": 1024, "tensor_parallel_size": 2},
            "storage": {"storages": storages, "scope": "fixture",
                        "unique_physical_storage_bytes": sum(row["bytes"] for row in storages)}}


# --- the KV observation's pass evidence and projection -----------------------


def test_only_a_pass_with_no_recorder_and_no_snapshot_is_read_only():
    read_only = admission_evidence(mode="kv", process_id=7, recorder_attached=False, snapshot_count=0)
    assert read_only["read_only"] is True
    intrusive = admission_evidence(mode="resources", process_id=7, recorder_attached=True, snapshot_count=0)
    assert intrusive["read_only"] is False
    # A recorder that has taken no snapshot yet has still synchronized nothing,
    # but it is attached and free to; the flag follows the attachment.
    assert "admission-ineligible" in intrusive["reason"]
    assert read_only["reason"].startswith("read-only worker RPC")
    with pytest.raises(ValueError, match="negative"):
        admission_evidence(mode="kv", process_id=7, recorder_attached=False, snapshot_count=-1)


def test_the_kv_record_carries_the_consumers_required_coordinates_and_nothing_more():
    evidence = admission_evidence(mode="kv", process_id=11, recorder_attached=False, snapshot_count=0)
    record = kv_observation_record(_observed(), evidence=evidence, rank=1, world_size=2,
                                   run_identity=_run_identity(), scope="fixture")
    assert record["rank"] == 1 and record["world_size"] == 2
    assert record["process_id"] == 11
    assert record["runtime_admission"] is True
    assert record["run_identity"] == {name: _DIGEST for name in COMMON_RUN_DIGESTS}
    assert set(record["resolved_limits"]) >= {"max_num_batched_tokens", "max_num_seqs"}
    for row in record["storage"]["storages"]:
        assert tuple(sorted(row)) == tuple(sorted(KV_STORAGE_FIELDS))
    assert record["storage"]["unique_physical_storage_bytes"] == 4096


def test_an_intrusive_pass_cannot_claim_the_admission_flag():
    # The flag is derived from the evidence, so no caller can pass it beside a
    # record that its own pass contradicts.
    evidence = admission_evidence(mode="resources", process_id=11, recorder_attached=True,
                                  snapshot_count=9)
    record = kv_observation_record(_observed(), evidence=evidence, rank=0, world_size=1,
                                   run_identity=_run_identity(), scope="fixture")
    assert record["runtime_admission"] is False
    assert record["admission_evidence"]["snapshot_count"] == 9


def test_the_kv_record_refuses_a_foreign_storage_spelling_and_a_foreign_run():
    evidence = admission_evidence(mode="kv", process_id=11, recorder_attached=False, snapshot_count=0)
    observed = _observed()
    observed["storage"]["storages"][0]["scope"] = "extra"
    with pytest.raises(ValueError, match="deduplicated backing"):
        kv_observation_record(observed, evidence=evidence, rank=0, world_size=1,
                              run_identity=_run_identity(), scope="fixture")
    with pytest.raises(ValueError, match="outside world"):
        kv_observation_record(_observed(), evidence=evidence, rank=2, world_size=2,
                              run_identity=_run_identity(), scope="fixture")
    with pytest.raises(ValueError, match="run identity is missing"):
        kv_observation_record(_observed(), evidence=evidence, rank=0, world_size=1,
                              run_identity={"model_sha256": _DIGEST}, scope="fixture")


def test_the_read_only_rpc_names_the_process_that_observed(monkeypatch):
    from experiments import full_engine_kv
    monkeypatch.setattr(full_engine_kv, "inspect_worker_kv", lambda worker: {"stub": worker})
    result = read_only_kv_observation("the-worker")
    assert result["observation"] == {"stub": "the-worker"}
    assert result["process_id"] == os.getpid()


# --- the startup sample ------------------------------------------------------


def test_a_startup_sample_needs_the_runtimes_own_locked_workspace():
    torch_stub = SimpleNamespace(cuda=SimpleNamespace(memory_allocated=lambda: 9000))
    record = worker_startup_record(torch_stub, _workspace(resident=4096), rank=0,
                                   receipt_resident_bytes=2048, scope="fixture")
    assert tuple(sorted(record)) == tuple(sorted(WORKER_STARTUP_FIELDS))
    assert record["workspace_resident_bytes"] == 4096
    assert record["memory_allocated_bytes"] == 9000
    with pytest.raises(ValueError, match="workspace record"):
        worker_startup_record(torch_stub, {"schema": "other"}, rank=0,
                              receipt_resident_bytes=2048, scope="fixture")
    with pytest.raises(ValueError, match="locked"):
        worker_startup_record(torch_stub, _workspace(locked=False), rank=0,
                              receipt_resident_bytes=2048, scope="fixture")


def test_the_identity_carries_a_rank_and_a_world_together_or_not_at_all():
    base = {"schema": "tessera.full_engine_resource_identity.v1", "model_sha256": _DIGEST,
            "configuration_sha256": _DIGEST, "runtime_manifest_sha256": _DIGEST,
            "assignment_sha256": _DIGEST, "canonical_units_sha256": _DIGEST,
            "workload_sha256": _DIGEST, "device_id": 0, "device_uuid": "fixture"}
    assert _identity(dict(base))["device_uuid"] == "fixture"
    scoped = _identity(dict(base, rank=1, world_size=2))
    assert (scoped["rank"], scoped["world_size"]) == (1, 2)
    with pytest.raises(ValueError, match="together or not at all"):
        _identity(dict(base, rank=0))
    with pytest.raises(ValueError, match="outside"):
        _identity(dict(base, rank=2, world_size=2))


# --- the two closure checks --------------------------------------------------


def _resident(allocation_id, size, categories):
    return {"allocation_id": allocation_id, "bytes": size, "allocate_index": 0,
            "free_completed_index": None, "observed_categories": categories}


def _startup_ledger(record=None, fixed=2048):
    ledger = {"identity": {"rank": 0, "world_size": 2},
              "torch_allocations": [_resident("fixed-a", fixed, ["fixed"]),
                                    _resident("kv-a", 4096, ["kv"])]}
    if record is not None:
        ledger["worker_startup_records"] = [record]
    return ledger


def _startup_record(rank=0, receipt=2048, workspace=1024, allocated=4096, locked=True):
    return {"rank": rank, "memory_allocated_bytes": allocated, "receipt_resident_bytes": receipt,
            "workspace_resident_bytes": workspace, "workspace_locked": locked, "scope": "fixture"}


def test_worker_startup_closes_on_the_ledger_fixed_rows_and_refuses_every_other_way():
    assert worker_startup_closed(_startup_ledger(record=_startup_record())) is True
    assert worker_startup_closed(_startup_ledger()) is False
    # Two records are not one rank's capture.
    two = _startup_ledger(record=_startup_record())
    two["worker_startup_records"].append(_startup_record(rank=1))
    assert worker_startup_closed(two) is False
    assert worker_startup_closed(_startup_ledger(record=_startup_record(rank=1))) is False
    assert worker_startup_closed(_startup_ledger(record=_startup_record(locked=False))) is False
    # The allocator sample must cover the receipt plus the workspace slots.
    assert worker_startup_closed(_startup_ledger(record=_startup_record(allocated=1024))) is False
    # A ledger whose fixed rows do not sum to the receipt does not close.
    assert worker_startup_closed(_startup_ledger(record=_startup_record(receipt=4096),
                                                 fixed=2048)) is False


def _kv_record(runtime_admission=True, read_only=True, rank=0, world_size=2, storages=None):
    return {"rank": rank, "world_size": world_size,
            "run_identity": {name: _DIGEST for name in COMMON_RUN_DIGESTS},
            "runtime_admission": runtime_admission,
            "admission_evidence": {"read_only": read_only, "snapshot_count": 0},
            "num_blocks": 4,
            "group_page_size_bytes": [1024],
            "resolved_limits": {"max_num_batched_tokens": 1024, "max_num_seqs": 4},
            "storage": {"storages": storages if storages is not None else [
                            {"device_type": "cuda", "device_id": 0, "address": 4096,
                             "bytes": 4096, "owners": ["kv_caches[0]"]}],
                        "unique_physical_storage_bytes": sum(
                            row["bytes"] for row in (storages if storages is not None else
                                                     [{"bytes": 4096}]))}}


def _capacity_ledger(record=None, kv=4096):
    ledger = {"identity": {"rank": 0, "world_size": 2,
                           "configuration_sha256": _DIGEST, "model_sha256": _DIGEST,
                           "runtime_manifest_sha256": _DIGEST},
              "torch_allocations": [_resident("kv-a", kv, ["kv"])]}
    if record is not None:
        ledger["kv_observations"] = [record]
    return ledger


def test_cache_capacity_closes_only_on_a_read_only_pass_over_the_ledgers_own_kv_rows():
    assert cache_capacity_closed(_capacity_ledger(record=_kv_record())) is True
    assert cache_capacity_closed(_capacity_ledger()) is False
    # The intrusive pass's own record is not the one that closes this domain,
    # even when its arithmetic is otherwise perfect.
    assert cache_capacity_closed(_capacity_ledger(
        record=_kv_record(runtime_admission=False, read_only=False))) is False
    # Two overlapping backings are not one pool, whatever their sum says.
    overlapping = [{"device_type": "cuda", "device_id": 0, "address": 4096, "bytes": 4096,
                    "owners": ["kv_caches[0]"]},
                   {"device_type": "cuda", "device_id": 0, "address": 6144, "bytes": 1024,
                    "owners": ["kv_caches[1]"]}]
    assert cache_capacity_closed(_capacity_ledger(record=_kv_record(storages=overlapping))) is False
    # The ledger's kv-owned resident rows must sum to the observed extent.
    assert cache_capacity_closed(_capacity_ledger(record=_kv_record(), kv=2048)) is False


def test_the_domains_this_branch_implements_refuse_on_a_capture_that_errored():
    ledger = _capacity_ledger(record=_kv_record())
    ledger["worker_startup_records"] = [_startup_record()]
    ledger["issues"] = ["dropped CUPTI buffer"]
    domains = qualify_domains(ledger)
    assert domains["worker_startup"]["state"] == "refused"
    assert domains["cache_capacity"]["state"] == "refused"


def test_the_assembler_refuses_per_rank_observations_without_a_rank_scoped_identity():
    raw = json.loads((Path(__file__).parent / "fixtures/full_engine_resource_ledger.json").read_text())
    from experiments.full_engine_resources import analyze_engine_resource_ledger
    ledger = analyze_engine_resource_ledger(raw)
    assert "rank" not in ledger["identity"]
    ledger["worker_startup_records"] = [_startup_record()]
    members = {"reference": {"canonical_census": "synthetic", "runtime_binding": "synthetic",
                             "selected_rows": ["synthetic"]},
               "workload": {"calibration": "synthetic", "prompt_ids": ["synthetic"],
                            "sampling": "synthetic"},
               "execution": {"graph_mode": "eager", "residency": "resident", "topology": "tp1"}}
    with pytest.raises(ValueError, match="rank-scoped run identity"):
        assemble_full_engine_resource_report(ledger, **members)
    ledger["identity"] = dict(ledger["identity"], rank=0, world_size=1)
    assemble_full_engine_resource_report(ledger, **members)  # binds once the scope is named


# --- the two-pass join -------------------------------------------------------


def test_the_join_binds_one_run_and_allows_two_different_processes():
    ledger = {"identity": dict(_run_identity(), rank=0, world_size=2),
              "capture_sha256": _DIGEST}
    notes = join_observation_passes(
        ledger, resource_process_id=111, startup_records=[_startup_record()],
        kv_records=[_kv_record()],
        capacity_witness=[dict(_kv_record(runtime_admission=False, read_only=False))])
    assert notes["different_processes"] is True
    assert notes["pointer_identities_compared"] is False
    assert notes["read_only_pass"]["runtime_admission"] is True
    assert notes["read_only_pass"]["process_id"] is None or isinstance(
        notes["read_only_pass"]["process_id"], int)
    assert notes["capacity_witness"]["num_blocks"] == 4
    assert "process ids" in notes["scope"] and "never" in notes["scope"]


def test_the_join_refuses_a_second_pass_from_a_different_run_or_rank_or_pool():
    ledger = {"identity": dict(_run_identity(), rank=0, world_size=2), "capture_sha256": _DIGEST}
    foreign_run = _kv_record()
    foreign_run["run_identity"]["workload_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="not the same configured run"):
        join_observation_passes(ledger, resource_process_id=111, startup_records=None,
                                kv_records=[foreign_run])
    with pytest.raises(ValueError, match="is rank 1/2"):
        join_observation_passes(ledger, resource_process_id=111, startup_records=None,
                                kv_records=[_kv_record(rank=1)])
    witness = _kv_record(runtime_admission=False, read_only=False)
    witness["num_blocks"] = 8
    with pytest.raises(ValueError, match="different KV capacity"):
        join_observation_passes(ledger, resource_process_id=111, startup_records=None,
                                kv_records=[_kv_record()], capacity_witness=[witness])
    with pytest.raises(ValueError, match="exactly one KV observation"):
        join_observation_passes(ledger, resource_process_id=111, startup_records=None,
                                kv_records=[_kv_record(), _kv_record()])


def test_per_rank_observations_need_a_rank_scoped_identity_in_the_join():
    ledger = {"identity": _run_identity(), "capture_sha256": _DIGEST}
    with pytest.raises(ValueError, match="rank-scoped run identity"):
        join_observation_passes(ledger, resource_process_id=111,
                                startup_records=[_startup_record()], kv_records=None)


def test_first_existing_prefers_the_sidecar_that_exists(tmp_path):
    earlier, later = tmp_path / "worker/worker-startup.json", tmp_path / "worker-startup.json"
    assert first_existing(earlier, later) == later          # neither exists: the named default
    later.write_text("{}")
    assert first_existing(earlier, later) == later
    earlier.parent.mkdir()
    earlier.write_text("{}")
    assert first_existing(earlier, later) == earlier


def test_a_startup_receipt_from_another_rank_is_refused(tmp_path):
    import hashlib
    receipt = {"schema": "tessera.native_moe_operator_receipt.v1",
               "resources": {"status": "incomplete", "rank": 1, "world_size": 2,
                             "resident_bytes": 4096}}
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="is rank 1/2"):
        read_routed_owner_receipt(path, digest, rank=0, world_size=2)
    with pytest.raises(ValueError, match="declared digest"):
        read_routed_owner_receipt(path, "b" * 64, rank=1, world_size=2)
    parsed = read_routed_owner_receipt(path, digest, rank=1, world_size=2)
    assert parsed["resident_bytes"] == 4096
    receipt["schema"] = "something-else"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="must be"):
        read_routed_owner_receipt(path, hashlib.sha256(path.read_bytes()).hexdigest(),
                                  rank=1, world_size=2)
