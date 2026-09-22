"""tessera#558 (D37 ship-gate leg): the reserved-extent witness.

The allocator's search charges the allocated-block composition; the ship gate
enforces the **reserved** device extent and publishes the reservation slack as
a witness. That needs two things the v1 report does not emit:

1. the reserved peak beside the allocated one, with the sample it came from;
2. the ``PYTORCH_CUDA_ALLOC_CONF`` value the reservation is a function of,
   bound in the configuration document so ``configuration_sha256`` moves
   when it does.

Red-first: the witness tests fail on the v1 tree (no reserved sample, no
allocator binding, v1 schema) and pass on the v2 one. These fixtures are
parser-contract proofs on CPU only; a capture on the attested image emitting
both fields with non-null values is the measurement this does not claim
(``measurement-needed``).
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.full_engine_resource_partition import (
    REPORT_SCHEMA, assemble_full_engine_resource_report,
)
from experiments.full_engine_resources import worker_startup_record


def _workspace(resident=0, locked=True):
    return {"schema": "tessera.native_moe_workspace.v1", "owner": "vllm.WorkspaceManager",
            "num_ubatches": 1, "num_lanes": 1, "locked": locked,
            "slots": [{"index": 0, "allocation": None}], "resident_bytes": resident}


def _torch(allocated, reserved):
    return SimpleNamespace(cuda=SimpleNamespace(memory_allocated=lambda: allocated,
                                               memory_reserved=lambda: reserved))


def test_startup_record_samples_reserved_beside_allocated():
    # The 2026-09-18 capture's dense sample this witnesses: allocated
    # 1,098,421,248, reserved 1,201,668,096, slack 103,246,848.
    record = worker_startup_record(_torch(1098421248, 1201668096),
                                   _workspace(resident=0), rank=0,
                                   receipt_resident_bytes=0, scope="fixture")
    assert record["memory_allocated_bytes"] == 1098421248
    assert record["memory_reserved_bytes"] == 1201668096
    # The slack is derived in the report, never stored in the sample: one
    # rule, one home.
    assert "slack" not in " ".join(sorted(record))


def test_startup_record_refuses_reserved_below_allocated():
    # The reserved extent is the pool the allocations come from; below the
    # allocated sample is a contradiction, refused where it is sampled.
    with pytest.raises(ValueError, match="reserved"):
        worker_startup_record(_torch(9000, 100), _workspace(resident=0), rank=0,
                              receipt_resident_bytes=0, scope="fixture")


def _startup_record_with_reserved(rank=0, allocated=1098421248, reserved=1201668096):
    return {"rank": rank, "memory_allocated_bytes": allocated,
            "memory_reserved_bytes": reserved, "receipt_resident_bytes": 0,
            "workspace_resident_bytes": 0, "workspace_locked": True, "scope": "fixture"}


def _ledger_with_reserved():
    return {"schema": "tessera.full_engine_raw_resource_ledger.v1",
            "identity": {"rank": 0, "world_size": 1},
            "capture_sha256": "c" * 64,
            "fixture_provenance": "synthetic CPU-only parser fixture, not a GPU measurement",
            "torch_allocations": [], "checkpoints": None,
            "cuda_argument_domains": None, "unattributed_external_records": [],
            "external_native_peak_bytes": None,
            "torch_observed_live_peak_bytes": None, "torch_observed_live_peak_scope": None,
            "step_intervals": None, "step_coverage": {"state": "unobserved"},
            "issues": [], "worker_startup_records": [_startup_record_with_reserved()],
            "runtime_provenance_relation": None, "kv_observations": None,
            "timing_captures": None, "owner_views": None,
            "observer_qualification": None}


def _members():
    return {
        "reference": {"canonical_census": "synthetic", "runtime_binding": "synthetic",
                      "selected_rows": ["synthetic"]},
        "workload": {"calibration": "synthetic", "prompt_ids": ["synthetic"],
                     "sampling": "synthetic"},
        "execution": {"graph_mode": "eager", "residency": "resident", "topology": "tp1"},
    }


def test_bound_report_emits_the_reserved_peak_and_the_slack():
    report = assemble_full_engine_resource_report(
        _ledger_with_reserved(), allocator_config="unset", **_members())
    assert report["schema"] == "tessera.full_engine_resource_report.v2"
    assert report["observations"]["allocator_config"] == "unset"
    assert report["derived"]["reserved_peak_bytes"] == 1201668096
    assert report["derived"]["reservation_slack_peak_bytes"] == 1201668096 - 1098421248
    witness = report["derived"]["reservation_witness"]
    assert witness["rank"] == 0
    assert "PYTORCH_CUDA_ALLOC_CONF" in witness["scope"]
    assert "load point" in witness["scope"]


def test_a_reservation_witness_without_the_bound_allocator_config_is_refused():
    # A claimed witness whose segment policy is unbound transfers to no
    # serve; the assembler refuses it rather than emitting an unbound peak.
    with pytest.raises(ValueError, match="PYTORCH_CUDA_ALLOC_CONF"):
        assemble_full_engine_resource_report(_ledger_with_reserved(), **_members())


def test_a_negative_slack_is_a_contradiction_not_a_witness():
    ledger = _ledger_with_reserved()
    ledger["worker_startup_records"] = [_startup_record_with_reserved(allocated=5000,
                                                                      reserved=1000)]
    with pytest.raises(ValueError, match="slack"):
        assemble_full_engine_resource_report(ledger, allocator_config="unset", **_members())


def test_an_unbound_capture_without_a_witness_claim_omits_the_fields():
    # Old captures carry no reserved sample; they assemble with nulls, not a
    # refusal, so v1-era artifacts stay readable.
    ledger = _ledger_with_reserved()
    ledger["worker_startup_records"] = [
        {key: value for key, value in ledger["worker_startup_records"][0].items()
         if key != "memory_reserved_bytes"}]
    report = assemble_full_engine_resource_report(ledger, **_members())
    assert report["observations"]["allocator_config"] is None
    assert report["derived"]["reserved_peak_bytes"] is None
    assert report["derived"]["reservation_slack_peak_bytes"] is None
    assert report["derived"]["reservation_witness"] is None


def test_require_allocator_policy_reads_the_bound_value():
    from experiments.capture_full_engine_resources import require_allocator_policy
    assert require_allocator_policy(
        {"environment": {"PYTORCH_CUDA_ALLOC_CONF": "unset"}}) == "unset"
    assert require_allocator_policy(
        {"environment": {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}}) == \
        "max_split_size_mb:128"
    with pytest.raises(ValueError, match="PYTORCH_CUDA_ALLOC_CONF"):
        require_allocator_policy({"environment": {"TESSERA_SERVE_MODE": "resident"}})
    with pytest.raises(ValueError, match="PYTORCH_CUDA_ALLOC_CONF"):
        require_allocator_policy({})
    with pytest.raises(ValueError, match="PYTORCH_CUDA_ALLOC_CONF"):
        require_allocator_policy({"environment": {"PYTORCH_CUDA_ALLOC_CONF": 128}})


def test_prepare_refuses_a_config_without_the_allocator_policy(tmp_path, monkeypatch):
    from experiments import capture_full_engine_resources
    from experiments.capture_full_engine_resources import prepare
    # prepare refuses a Torch/vLLM import before anything else; the CPU test
    # session may hold torch for other files, so evict the names it checks.
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"engine_args": {"dtype": "bfloat16"},
                                  "environment": {"TESSERA_SERVE_MODE": "resident"}}))
    args = SimpleNamespace(config=config, census=None, artifact=False)
    with pytest.raises(ValueError, match="PYTORCH_CUDA_ALLOC_CONF"):
        prepare(args)
    assert capture_full_engine_resources.ALLOCATOR_POLICY_KEY == "PYTORCH_CUDA_ALLOC_CONF"


def test_allocator_config_of_reads_the_plan_or_none():
    from experiments.report_full_engine_resources import allocator_config_of
    assert allocator_config_of({"selected_configuration": {"environment": {
        "PYTORCH_CUDA_ALLOC_CONF": "unset"}}}) == "unset"
    assert allocator_config_of({"selected_configuration": {"environment": {}}}) is None
    assert allocator_config_of({}) is None


def _dense_check(rank=0, allocated=863269888, reserved=954204160, **overrides):
    """A dense artifact's startup check, as `full_engine_ownership` builds it."""
    check = {"schema": "tessera.full_engine_dense_startup_check.v1",
             "units": {}, "units_checked": 0, "units_disagreeing": [],
             "manifest_unpriced_resident_bytes": 0,
             "candidate_units_outside_manifest": [],
             "rank": rank,
             "memory_allocated_bytes": allocated,
             "memory_reserved_bytes": reserved,
             "ledger_live_bytes_at_ready_for_workload": 0,
             "allocator_sample_bounds_ledger": True, "closed": False,
             "scope": "fixture"}
    check.update(overrides)
    return check


def _dense_ledger(**overrides):
    """A DENSE capture's ledger: no routed receipt, so no startup record."""
    ledger = _ledger_with_reserved()
    ledger["worker_startup_records"] = []
    ledger["owner_views"] = {
        "schema": "tessera.full_engine_ownership_observation.v1",
        "views": {"schema": "tessera.full_engine_owner_views.v1", "rules": [],
                  "evidence": {}, "summary": {}, "views": [], "scope": "fixture"},
        "external_records": [], "boundary_geometry_witness": None,
        "transient_gap_witness": None,
        "dense_startup_check": _dense_check(**overrides)}
    return ledger


def test_a_dense_capture_witnesses_its_reserved_extent():
    """The dense sample is a resident-after-load sample at arm, and counts.

    `worker_startup_records` is written only where the plan names a routed
    owner receipt, so a dense artifact carries none and the witness read
    nothing -- while `_resource_write_startup_sample` had already sampled
    `memory_reserved` into the dense observation. The measurement existed and
    the report dropped it: tessera#399's 2026-09-21 capture sampled
    954,204,160 reserved against 863,269,888 allocated and published
    `reserved_peak_bytes: null`.
    """
    report = assemble_full_engine_resource_report(
        _dense_ledger(), allocator_config="unset", **_members())
    assert report["derived"]["reserved_peak_bytes"] == 954204160
    assert report["derived"]["reservation_slack_peak_bytes"] == 954204160 - 863269888
    assert report["derived"]["reservation_witness"]["rank"] == 0
    assert report["derived"]["reservation_witness"]["memory_allocated_bytes"] == 863269888


def test_a_dense_check_without_a_reserved_sample_witnesses_nothing():
    # A pre-#558 dense observation carries no reserved field; it stays null
    # rather than borrowing the allocated sample.
    ledger = _dense_ledger()
    del ledger["owner_views"]["dense_startup_check"]["memory_reserved_bytes"]
    report = assemble_full_engine_resource_report(ledger, **_members())
    assert report["derived"]["reserved_peak_bytes"] is None
    assert report["derived"]["reservation_witness"] is None


def test_the_dense_sample_joins_the_routed_records_rather_than_replacing_them():
    # A capture carrying both takes the peak over both, by the one rule.
    ledger = _dense_ledger(reserved=900, allocated=800)
    ledger["worker_startup_records"] = [_startup_record_with_reserved()]
    report = assemble_full_engine_resource_report(
        ledger, allocator_config="unset", **_members())
    assert report["derived"]["reserved_peak_bytes"] == 1201668096


def test_a_dense_reserved_sample_below_its_allocated_one_is_refused():
    ledger = _dense_ledger(reserved=1000, allocated=5000)
    with pytest.raises(ValueError, match="slack"):
        assemble_full_engine_resource_report(ledger, allocator_config="unset", **_members())
