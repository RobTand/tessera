"""The pytest entry point delegates to the unchanged manual A4 gate.

These CPU controls substitute device availability and the gate result. They
exercise invocation and refusal, not CUDA kernels or fixture qualification.
"""
import pytest

pytest.importorskip("torch")

import test_native_a4_serving as gate


def _entry():
    entry = getattr(gate, "test_native_a4_serving_gate", None)
    assert callable(entry), "A4 serving gate has no pytest entry point"
    return entry


def test_pytest_entry_point_runs_the_existing_gate(monkeypatch):
    entry = _entry()
    called = []
    monkeypatch.setattr(gate.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gate.box_artifacts, "skip_now", lambda *parts: called.append(parts))

    def run_gate():
        called.append("run_gate")
        return {"ok": True}

    monkeypatch.setattr(gate, "run_gate", run_gate)
    entry()
    assert called == [("a4_wires", "a4-config.json"), "run_gate"]


def test_pytest_entry_point_preserves_a_failed_gate(monkeypatch):
    entry = _entry()
    monkeypatch.setattr(gate.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gate.box_artifacts, "skip_now", lambda *parts: None)
    monkeypatch.setattr(gate, "run_gate", lambda: {"ok": False})
    with pytest.raises(AssertionError):
        entry()


def test_pytest_entry_point_names_its_cuda_precondition(monkeypatch):
    entry = _entry()
    monkeypatch.setattr(gate.torch.cuda, "is_available", lambda: False)

    def refuse_gate():
        raise AssertionError("a CUDA-less session executed the A4 gate")

    monkeypatch.setattr(gate, "run_gate", refuse_gate)
    with pytest.raises(pytest.skip.Exception, match="native A4 serving gate requires CUDA"):
        entry()


def test_pytest_entry_point_keeps_the_artifact_resolver_refusal(monkeypatch):
    entry = _entry()
    monkeypatch.setattr(gate.torch.cuda, "is_available", lambda: True)

    def missing_artifact(*parts):
        assert parts == ("a4_wires", "a4-config.json")
        pytest.skip("set TESSERA_A4_WIRE_DIR: a4-config.json is unavailable")

    monkeypatch.setattr(gate.box_artifacts, "skip_now", missing_artifact)
    monkeypatch.setattr(gate, "run_gate", lambda: pytest.fail("missing fixture reached the gate"))
    with pytest.raises(pytest.skip.Exception, match="TESSERA_A4_WIRE_DIR"):
        entry()
