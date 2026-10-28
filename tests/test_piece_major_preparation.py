"""CPU preparation binds current inputs, without executing or qualifying a GPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from test_piece_major_comparison_protocol import pp, protocol, receipt


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "experiments/t8r_speed/piece_major_pb_action.py"


def emit(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def identity(path):
    stat = path.stat()
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    doc = protocol(tmp_path)
    for relative in doc["harness"]:
        target = checkout / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
        doc["harness"][relative] = identity(target)["sha256"]
    kernel_relative = "src/tessera/serving/csrc/routed_fused_window.cu"
    kernel = checkout / kernel_relative
    kernel.parent.mkdir(parents=True)
    kernel.write_bytes((ROOT / kernel_relative).read_bytes())
    doc["kernel_sha256"] = identity(kernel)["sha256"]
    doc["source_manifest"] = emit(tmp_path / "source.json", {"files": {
        kernel_relative: identity(kernel)}})
    native = tmp_path / "fixture.so"
    native.write_bytes(b"inert binary identity fixture; never mapped or executed")
    origin = tmp_path / "origin.so"
    origin.write_bytes(native.read_bytes())
    doc["native"] = {"path": str(native), "sha256": identity(native)["sha256"],
                     "origin_path": str(origin), "files": {native.name: identity(native)}}
    doc["routing"] = emit(tmp_path / "routing.json", {"fixture": "captured routing bytes"})
    artifact = ROOT / "tests/data/legacy/e4m3-1024-window-channel-256c.tessera"
    doc["input_manifest"] = emit(tmp_path / "inputs.json", {"entries": [
        {"path": str(artifact), "offset": 0, "bytes": artifact.stat().st_size,
         "sha256": identity(artifact)["sha256"]}]})
    protocol_binding = emit(tmp_path / "protocol.json", doc)
    numeric_path, numeric_sha = receipt(tmp_path, protocol_sha256=protocol_binding["sha256"])
    evidence = tmp_path / "evidence"
    window = emit(tmp_path / "window.json", {
        "protocol": protocol_binding["path"], "protocol_sha256": protocol_binding["sha256"],
        "worktree": str(checkout), "output": str(tmp_path / "never-launched"),
        "numeric_qualification": {"receipt": numeric_path, "receipt_sha256": numeric_sha},
        "execution_owner": {"source_bindings": {}, "namespace_paths": []}})
    packet = emit(tmp_path / "packet.json", {
        "controller_sha256": identity(CONTROLLER)["sha256"], "window": window["path"],
        "window_sha256": window["sha256"], "evidence_root": str(evidence)})
    spec = importlib.util.spec_from_file_location("preparation_controller", CONTROLLER)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    monkeypatch.syspath_prepend(str(CONTROLLER.parent))
    monkeypatch.setenv("PRISMABUILD_ACTION_KEY", "unit-only-no-admitted-compute")
    monkeypatch.setattr(sys, "argv", ["controller", "--packet", packet["path"],
                                     "--packet-sha256", packet["sha256"], "--prepare"])
    return owner, doc, checkout, evidence


def test_preparation_records_observed_identities_without_qualifying(preparation):
    owner, doc, checkout, evidence = preparation
    assert owner.main() == 0
    record = json.loads((evidence / "prepare.json").read_text())
    observed = record["identities"]
    assert observed["source"] == json.loads(Path(doc["source_manifest"]["path"]).read_bytes())["files"]
    assert observed["native"]["sha256"] == doc["native"]["sha256"]
    assert observed["native"]["files"] == doc["native"]["files"]
    assert observed["artifact"] == {"path": doc["artifact"], "group": doc["group"],
                                    "input_manifest": doc["input_manifest"]}
    assert observed["routing"] == doc["routing"]
    assert observed["tp"] == [pp.TP_RANK, pp.TP_SIZE]
    assert observed["harness"] == doc["harness"]
    assert observed["scope"] == "CPU identity preparation only; FP8/BF16 and T4 GPU qualification not measured"
    assert record["benchmark_executed"] is False
    assert not (checkout.parent / "never-launched").exists()


@pytest.mark.parametrize("changed", ["source_manifest", "source", "routing", "native", "native_origin"])
def test_preparation_refuses_changed_inputs_before_publication(preparation, changed):
    owner, doc, checkout, evidence = preparation
    if changed == "source":
        target = checkout / "src/tessera/serving/csrc/routed_fused_window.cu"
    elif changed == "native_origin":
        target = Path(doc["native"]["origin_path"])
    else:
        target = Path(doc[changed]["path"])
    target.write_bytes(target.read_bytes() + b"\nchanged")
    with pytest.raises(ValueError):
        owner.main()
    assert not (evidence / "prepare.json").exists()


@pytest.mark.parametrize("change", [{"status": "failed"}, {"protocol_sha256": "f" * 64}])
def test_preparation_cannot_claim_reuse_of_a_nonadmitting_numeric_receipt(preparation, monkeypatch, change):
    owner, doc, checkout, evidence = preparation
    root = checkout.parent
    receipt_path = root / "numeric.json"
    numeric = json.loads(receipt_path.read_bytes())
    numeric.update(change)
    numeric_binding = emit(receipt_path, numeric)
    window_path = root / "window.json"
    window = json.loads(window_path.read_bytes())
    window["numeric_qualification"]["receipt_sha256"] = numeric_binding["sha256"]
    window_binding = emit(window_path, window)
    packet_path = root / "packet.json"
    packet = json.loads(packet_path.read_bytes())
    packet["window_sha256"] = window_binding["sha256"]
    packet_binding = emit(packet_path, packet)
    monkeypatch.setattr(sys, "argv", ["controller", "--packet", str(packet_path),
                                     "--packet-sha256", packet_binding["sha256"], "--prepare"])
    with pytest.raises(ValueError, match="numeric receipt"):
        owner.main()
    assert not (evidence / "prepare.json").exists()
