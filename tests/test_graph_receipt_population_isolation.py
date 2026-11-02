"""Keep verifier policy units independent of producer availability (#1532).

The process denial is deliberate, not a scheduling/load simulation: an unavailable
producer must not prevent a verifier-unit positive from reaching the real rule.
Producer integration remains in test_graph_receipt_population.py.

Retain the split recalled for issuecomment-6387659195: construct the verifier
positive directly, accept it before mutation, and forbid producer launches.
Pre-fix RED through PrismaBuild (action
3a5e1cc449e05b54bacd291e041a20d0afc0f517ffc93fab90c07d8ba50f07c3):
test_extra_pass_reaches_verifier_when_producer_unavailable invoked the old
extra-pass test; the unavailable script returned 2 and _valid failed at
``assert result.returncode == 0``, before the verifier. This proves producer
coupling, not the unconfirmed historical host-load/timeout mechanism.
"""
from __future__ import annotations

import inspect
from pathlib import Path
import subprocess

import pytest

import test_graph_receipt_population as population
from test_graph_receipt_population import inputs  # noqa: F401 (integration fixture)


def _verifier_receipt():
    """An independent protocol positive, not producer output or finish's result.

    Only the runner owns the roster; don't copy the verifier's counter constant.
    No paths, timestamps, random state or producer imports enter this receipt.
    """
    total = sum(len(specs) for specs, _ in population._cases().values())

    def passes(name):
        return [{"name": p, "members": total, "choices": total}
                for p in (name, f"{name}-r2")]

    serve = {
        "image": population.IMAGE, "model_config_sha256": "d" * 64,
        "tessera_src_sha256": "b" * 64,
        "compilation_config": dict(population.CC), "speculative_tokens": 0,
        "max_model_len": 8448, "max_num_seqs": 8,
        "tensor_parallel_size": 1, "fabric": "none",
    }
    arm = {
        "name": "g", "passes": passes("g"), "equal": True,
        "replayed_everything": True,
        "compilation_config": dict(population.CC), "speculative_tokens": 0,
        "max_model_len": 8448, "max_num_seqs": 8, "tensor_parallel_size": 1,
        "graph": {
            "managers": {"ModelCudaGraphManager@fixture": {
                "captured_sizes": [1], "replayed_sizes": {"1": 1}}},
            "classes": {"captured": {}, "replays": {}},
        },
    }
    return {
        "schema": "tessera.graph_equals_eager.v2", "verdict": "equal",
        "runtime": {"image": population.IMAGE, "fabric": "none"},
        "model": {"config_sha256": serve["model_config_sha256"]},
        "tessera": {"src_sha256": serve["tessera_src_sha256"]},
        "equality_set": {"choices_per_pass": total},
        "eager_pool": [{"name": name, "passes": passes(name)} for name in ("e1", "e2")],
        "arms": [arm],
        "attests": [{**serve, "execution_mode": "compiled", "arm": "g"}],
    }


# Reuse pytest's declared attack population rather than maintaining a second roster.
_ATTACKS = next(
    mark.args[1]
    for mark in population.test_verify_rejects_forged_population_and_independence_claims.pytestmark
    if mark.name == "parametrize" and mark.args[0] == "attack"
)


@pytest.mark.parametrize("attack", _ATTACKS)
def test_verifier_attacks_need_no_producer(monkeypatch, tmp_path, attack):
    def forbidden(*args, **kwargs):
        raise AssertionError("verifier unit tried to launch a producer subprocess")

    monkeypatch.setattr(population, "PRODUCER", tmp_path / "unavailable" / "receipt.py")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    test = population.test_verify_rejects_forged_population_and_independence_claims
    assert "inputs" not in inspect.signature(test).parameters, (
        "verifier unit still requests the producer's filesystem fixture"
    )
    # This executes the independent positive-control verification and the real
    # refusal assertion, including the extra-pass population reason, not a mock.
    test(attack=attack)


def test_extra_pass_reaches_verifier_when_producer_unavailable(request, monkeypatch, tmp_path):
    # Same deterministic producer-did-not-deliver condition as check-1. On the
    # base this fails in _valid's return-code assertion, before verifier policy.
    monkeypatch.setattr(population, "PRODUCER", tmp_path / "unavailable" / "receipt.py")
    test = population.test_verify_rejects_forged_population_and_independence_claims
    kwargs = {"attack": "extra-pass"}
    if "inputs" in inspect.signature(test).parameters:
        kwargs["inputs"] = request.getfixturevalue("inputs")
    try:
        test(**kwargs)
    except AssertionError as exc:
        pytest.fail(f"extra-pass failed before reaching the verifier: {exc}", pytrace=False)


def test_producer_scratch_is_owned_by_test(inputs, monkeypatch):
    real_run = subprocess.run
    scratch_dirs = []

    def isolated_run(*args, **kwargs):
        env = kwargs.get("env", {})
        assert all(name in env for name in ("TMPDIR", "TMP", "TEMP")), (
            "producer scratch still falls back to the host temporary directory"
        )
        scratch = Path(env["TMPDIR"])
        assert scratch.is_relative_to(inputs[2].parent)
        assert scratch.is_dir()
        assert env["TMP"] == env["TEMP"] == str(scratch)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
            assert env[name] == "1"
        scratch_dirs.append(scratch)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", isolated_run)
    population._valid(inputs)
    assert len(scratch_dirs) == 1
    assert not scratch_dirs[0].exists(), "producer scratch was not cleaned up"
