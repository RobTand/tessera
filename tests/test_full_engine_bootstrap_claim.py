"""Identity-claim ordering against the bootstrap's recorded CUDA history.

Regression coverage for Tessera #664: vLLM's ``WorkerProc.init_worker``
initializes CUDA (distributed device selection) before it constructs the
configured ``worker_cls``, so ``ResourceCaptureWorker.__init__`` always runs
with CUDA already initialized. ``claim(actual_identity=...)`` must still bind
the identity there, against the pre-CUDA history the bootstrap started at
sitecustomize, and must keep refusing CUDA activity that history cannot cover.
"""

import os
from types import SimpleNamespace

import pytest

from experiments import full_engine_bootstrap as bootstrap


def _rank_identity(rank=1, world=2):
    return {
        "schema": "tessera.full_engine_resource_identity.v1",
        "model_sha256": "1" * 64,
        "configuration_sha256": "2" * 64,
        "runtime_manifest_sha256": "3" * 64,
        "assignment_sha256": "4" * 64,
        "canonical_units_sha256": "5" * 64,
        "workload_sha256": "6" * 64,
        "device_id": 0,
        "device_uuid": "GPU-b1eceeea-fec7-371e-2cf3-cd10f2e7b705",
        "rank": rank,
        "world_size": world,
        "host": {"ip": "10.100.96.1", "interface": "enp1s0f0np0", "source": "VLLM_HOST_IP"},
    }


def _fake_recorder(*, early=True, errors=(), snapshot_count=0, cuda_initialized=True,
                   collector_finished=False, collector_start_code=0):
    return SimpleNamespace(
        process_id=os.getpid(),
        _early=early,
        _errors=list(errors),
        snapshot_count=snapshot_count,
        _torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: cuda_initialized)),
        _collector=SimpleNamespace(
            _finished=collector_finished,
            start_code=collector_start_code,
            library_sha256="a" * 64,
        ),
        identity=None,
        device=None,
    )


def _install(monkeypatch, recorder, identity):
    monkeypatch.setattr(bootstrap, "_recorder", recorder)
    monkeypatch.setattr(bootstrap, "_plan", {
        "output_directory": "unused",
        "world_size": identity["world_size"],
        "identity": identity,
    })
    monkeypatch.setattr(bootstrap, "_claimed", False)


def test_claim_binds_identity_against_recorded_pre_cuda_history(monkeypatch):
    """CUDA initialized by vLLM before the worker ctor is the sanctioned order.

    The recorder started before CUDA (``_early``) and the CUPTI collector that
    recorded that initialization is still live, so the bootstrap's history
    covers the CUDA activity and the identity binds at the constructor.
    """
    identity = _rank_identity()
    recorder = _fake_recorder(cuda_initialized=True)
    _install(monkeypatch, recorder, identity)

    claimed_recorder, claimed_plan = bootstrap.claim(actual_identity=dict(identity))

    assert claimed_recorder.identity["rank"] == identity["rank"]
    assert claimed_plan["identity"]["rank"] == identity["rank"]
    assert claimed_recorder.device == identity["device_id"]
    assert bootstrap._claimed is True


def test_claim_still_binds_before_cuda_initialization_when_history_starts_first(monkeypatch):
    """The truly-early path (CUDA untouched at claim) keeps working unchanged."""
    identity = _rank_identity()
    recorder = _fake_recorder(cuda_initialized=False)
    _install(monkeypatch, recorder, identity)

    claimed_recorder, _ = bootstrap.claim(actual_identity=dict(identity))

    assert claimed_recorder.identity["rank"] == identity["rank"]


def test_claim_still_refuses_a_snapshot_taken_before_the_identity(monkeypatch):
    identity = _rank_identity()
    recorder = _fake_recorder(cuda_initialized=True, snapshot_count=2)
    _install(monkeypatch, recorder, identity)

    with pytest.raises(RuntimeError, match="before a snapshot"):
        bootstrap.claim(actual_identity=dict(identity))


@pytest.mark.parametrize("collector_finished,collector_start_code", [
    (True, 0),
    (False, 7),
])
def test_claim_refuses_cuda_activity_the_bootstrap_history_cannot_cover(
        monkeypatch, collector_finished, collector_start_code):
    """CUDA initialized at claim is only sound against live recorded history.

    A collector that already finished, or never started successfully, cannot
    cover the process's CUDA activity even when the recorder was constructed
    early; the identity binding must refuse rather than attribute blind.
    """
    identity = _rank_identity()
    recorder = _fake_recorder(cuda_initialized=True,
                              collector_finished=collector_finished,
                              collector_start_code=collector_start_code)
    _install(monkeypatch, recorder, identity)

    with pytest.raises(RuntimeError, match="bootstrap history"):
        bootstrap.claim(actual_identity=dict(identity))


def test_claim_keeps_refusing_history_that_started_after_cuda(monkeypatch):
    identity = _rank_identity()
    recorder = _fake_recorder(early=False, cuda_initialized=True)
    _install(monkeypatch, recorder, identity)

    with pytest.raises(RuntimeError, match="did not start cleanly before CUDA"):
        bootstrap.claim(actual_identity=dict(identity))
