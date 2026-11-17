"""Independent verifier fixtures and controlled phase lifecycle tests."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest
import torch

from experiments.numerics_rows.common import Invalid, BoundaryRecorder
from experiments.numerics_rows import row


def phase_api():
    # The old tool has no split verifier. Fail with its missing feature named,
    # rather than counting an ImportError as a behavioural rejection.
    name = "experiments.numerics_rows.phase_artifact"
    assert importlib.util.find_spec(name) is not None, "feature absent: dense phase binding verifier"
    return importlib.import_module(name)


@pytest.fixture
def phase_pair(tmp_path):
    return lambda: make_pair(tmp_path)


def make_pair(tmp_path):
    api = phase_api()
    unit = row.new_unit({"cell": "img-n-dense", "symbol": "dense", "decoder": "e4m3mma", "rung": 1024,
        "m_config": {"M": 1, "k_split": 1}, "tp": {"rank": 0, "size": 2},
        "module": "module-A", "role": "unit", "launch_state_sha256": "a" * 64})
    common = {"population": "p", "source": "F", "module": "module-A", "build": "b", "rule": "r", "math": "m"}
    scope = api.unit_scope(unit, "w" * 64)
    prepared = api.Artifact(tmp_path / "prepare", "prepare", common)
    prepared.document["scopes"] = [scope]
    case = prepared.add(scope, row.unit_identity(unit)[4], "dense", {"input": torch.ones(1, 2)}, category="sweep")
    digest = prepared.finish()
    prepare, _ = api.read(prepared.root, "prepare", common, [scope])
    captured = api.Artifact(tmp_path / "capture", "capture", common)
    captured.document.update(scopes=[scope], prepare_digest=digest, cases=[copy.deepcopy(case)])
    captured.document["cases"][0]["tensors"] = {"output": captured.tensor(torch.ones(1, 2))}
    captured.finish()
    capture, _ = api.read(captured.root, "capture", common, [scope])
    assert len(api.check_capture(prepare, capture, digest)) == 1
    return api, prepared, prepare, captured, capture, digest, common, scope


@pytest.mark.parametrize("phase", ["prepare", "capture"])
def test_old_phase_pair_cannot_be_attached_to_another_module(phase_pair, phase):
    api, prepared, _, captured, _, _, common, scope = phase_pair()
    other = {**common, "module": "module-B"}
    with pytest.raises(Invalid, match="identities, module or cut"):
        api.read((prepared if phase == "prepare" else captured).root, phase, other, [scope])


@pytest.mark.parametrize("field", ["unit_key", "module", "cut", "wire_sha256"])
def test_inner_phase_scope_cannot_be_relabelled(phase_pair, field):
    api, prepared, _, _, _, _, common, scope = phase_pair()
    document = prepared.document
    document["cases"][0]["scope"][field] = "other"
    prepared.finish()
    with pytest.raises(Invalid, match="another unit"):
        api.read(prepared.root, "prepare", common, [scope])


def test_capture_must_consume_the_exact_prepare_digest(phase_pair):
    api, _, prepare, _, capture, digest, _, _ = phase_pair()
    capture["prepare_digest"] = "other"
    with pytest.raises(Invalid, match="different prepare"):
        api.check_capture(prepare, capture, digest)


@pytest.mark.parametrize("change", ["missing", "duplicate", "foreign"])
def test_compare_requires_each_case_exactly_once(phase_pair, change):
    api, _, prepare, _, capture, digest, _, _ = phase_pair()
    if change == "missing":
        capture["cases"].clear()
    elif change == "duplicate":
        capture["cases"].append(copy.deepcopy(capture["cases"][0]))
    else:
        capture["cases"][0]["id"] = "other"
    with pytest.raises(Invalid, match="exactly once"):
        api.check_capture(prepare, capture, digest)


def test_phase_payload_bytes_and_configuration_are_bound(phase_pair):
    api, prepared, _, _, _, _, common, scope = phase_pair()
    reference = prepared.document["cases"][0]["tensors"]["input"]
    assert torch.equal(api.payload(prepared.root, reference), torch.ones(1, 2))
    (prepared.root / reference["file"]).write_bytes(b"changed")
    with pytest.raises(Invalid, match="digest"):
        api.read(prepared.root, "prepare", common, [scope])
    with pytest.raises(Invalid, match="digest"):
        api.payload(prepared.root, reference)


def test_unfinished_prepare_cannot_authorize_capture(phase_pair):
    api, prepared, _, _, _, _, common, scope = phase_pair()
    prepared.document["completion"] = False
    (prepared.root / "phase.json").write_text(json.dumps(prepared.document))
    with pytest.raises(Invalid, match="incomplete"):
        api.read(prepared.root, "prepare", common, [scope])


def test_prepared_quantizer_mismatch_refuses_before_native_call():
    phase_api()
    from experiments.numerics_rows.dense_phases import quantizer_guard
    calls = []
    expected = (torch.zeros(1, 2), torch.ones(1, 1))
    actual = (torch.ones(1, 2), torch.ones(1, 1))
    recorder = BoundaryRecorder(lambda: None, [])
    recorder.before_call = lambda: quantizer_guard(actual, expected)
    with pytest.raises(Invalid, match="before native call"):
        recorder.call("native", "tessera", lambda: calls.append("native"))
    assert not calls


def test_quantizer_inside_apply_is_checked_again():
    phase_api()
    from experiments.numerics_rows.dense_phases import quantizer_guard
    recorder = BoundaryRecorder(lambda: None, [])
    recorder.quantizer_result = lambda actual: quantizer_guard(actual, (torch.ones(1),))
    assert torch.equal(recorder.call("q", "quantizer", lambda: (torch.ones(1),))[0], torch.ones(1))
    with pytest.raises(Invalid, match="before native call"):
        recorder.call("q", "quantizer", lambda: (torch.zeros(1),))


@pytest.mark.parametrize("known_fail", [False, True])
def test_handled_signal_writes_stopped_and_preserves_fail(tmp_path, monkeypatch, known_fail):
    phase_api()
    from experiments.numerics_rows.phase_lifecycle import handled_signals
    callbacks = {}
    def install(number, callback):
        previous = callbacks.get(number)
        callbacks[number] = callback
        return previous
    monkeypatch.setattr(signal, "signal", install)
    unit = row.new_unit({"cell": "dense", "role": "unit"})
    if known_fail:
        unit["findings"].append("exact mismatch")
    receipt = {"row": {}, "units": [unit]}
    with pytest.raises(SystemExit) as stopped:
        with handled_signals(receipt, tmp_path):
            callbacks[signal.SIGTERM](signal.SIGTERM, None)
    saved = json.loads((tmp_path / "numerics_row.json").read_text())
    assert stopped.value.code == 143
    assert saved["row"]["outcome"] == ("FAIL" if known_fail else "STOPPED")
    assert saved["row"]["finalization_finished"] is False


def test_progress_checkpoint_precedes_diagnostics(tmp_path, monkeypatch):
    phase_api()
    from experiments.numerics_rows.gpu_time import PhaseTimes
    from experiments.numerics_rows.phase_lifecycle import Progress
    receipt = {"row": {"gate_capture_finished": True, "finalization_finished": False}}
    phases = PhaseTimes(receipt["row"])
    meter = SimpleNamespace(snapshot=lambda **_: {"wall_minutes": 1})
    progress = Progress(tmp_path, receipt, phases, meter)
    progress.record("gate_cases")
    progress.record("diagnostics", checkpoint=True)
    saved = json.loads((tmp_path / "numerics_row.json").read_text())
    assert saved["row"]["gate_capture_finished"] is True
    assert saved["row"]["finalization_finished"] is False
    assert saved["row"]["active"]["phase"] == "diagnostics"
    assert saved["row"]["gpu_minutes"]["wall_minutes"] == 1
    assert json.loads((tmp_path / "phase-events.jsonl").read_text())["phase"] == "gate_cases"


def test_whole_job_peak_reads_flux_cgroup(tmp_path):
    phase_api()
    from experiments.numerics_rows.gpu_time import cgroup_peak
    group = tmp_path / "job"
    group.mkdir()
    (group / "memory.peak").write_text("4096\n")
    membership = tmp_path / "membership"
    membership.write_text("0::/job\n")
    assert cgroup_peak(tmp_path, membership) == 4096


def test_frozen_canary_remains_marked_uncounted(tmp_path):
    phase_api()
    from experiments.numerics_rows.common import sha256
    rule = tmp_path / "rule.md"
    rule.write_text("frozen rule")
    row.check_rule_freeze(rule, {"sha256": sha256(rule), "reviewer": "lead", "frozen_utc": "2026-11-17T02:39:51Z", "canary": False}, canary=True)


def test_lever_addendum_binds_the_frozen_record_without_rewriting_it(tmp_path):
    phase_api()
    from experiments.numerics_rows.common import sha256
    rule, freeze, addendum = (tmp_path / n for n in ("rule.md", "freeze.json", "addendum.json"))
    rule.write_text("frozen rule")
    freeze.write_text('{"frozen_utc":"2026-11-17T02:39:51Z"}')
    before = freeze.read_bytes()
    data = {"freeze_record_sha256": sha256(freeze), "rule_sha256": sha256(rule), "reviewer": "lead", "frozen_utc": "later", "launch_levers_sha256": "a" * 64}
    addendum.write_text(json.dumps(data))
    bound = row.bind_freeze_addendum(json.loads(before), freeze, rule, addendum)
    assert bound["launch_levers_sha256"] == "a" * 64 and freeze.read_bytes() == before
    data["freeze_record_sha256"] = "b" * 64
    addendum.write_text(json.dumps(data))
    with pytest.raises(Invalid, match="does not bind"):
        row.bind_freeze_addendum(json.loads(before), freeze, rule, addendum)


def test_device_capture_uses_prepared_bytes_without_a_host_reference(monkeypatch):
    phase_api()
    from experiments.numerics_rows.dense_phases import capture_case
    x = torch.tensor([[1, 0]], dtype=torch.bfloat16)
    q, sf = x.to(torch.float8_e4m3fn), torch.ones(1, 1)
    calls = []
    class Adapter:
        def quantized(self, value, recorder):
            return q, sf, None, None
        def apply(self, value):
            calls.append("apply")
            return value * 2
        def staged(self, q, sf, recorder):
            calls.append("staged")
            return q.bfloat16() * 2
        def reference(self, *args, **kwargs):
            pytest.fail("CPU reference ran in the GPU phase")
    original = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *args, **kwargs: self if args == ("cuda",) else original(self, *args, **kwargs))
    result = capture_case(Adapter(), {}, {"kind": "dense", "category": "sweep"}, {"input": x, "q": q, "sf": sf}, BoundaryRecorder(lambda: None, []))
    assert calls == ["apply", "staged"] and torch.equal(result["output"], x * 2)
    with pytest.raises(Invalid, match="quantizer bytes"):
        capture_case(Adapter(), {}, {"kind": "dense", "category": "sweep"}, {"input": x, "q": torch.zeros_like(q), "sf": sf}, BoundaryRecorder(lambda: None, []))
    assert calls == ["apply", "staged"]


def test_shared_dense_roster_prepares_every_case_without_native_calls(tmp_path):
    api = phase_api()
    from experiments.numerics_rows.dense_phases import CaseWriter
    from experiments.numerics_rows.reference import FiniteBound
    artifact = api.Artifact(tmp_path / "prepare", "prepare", {})
    writer = CaseWriter(artifact, SimpleNamespace(record=lambda *args: None), {"module-A": "wire"})
    unit = row.new_unit({"cell": "dense", "module": "module-A", "role": "unit", "m_config": {"M": 1, "chunk_size": 1, "k_split": 1}, "tp": {"rank": 0, "size": 1}})
    class Adapter:
        device, cols, rows, case_writer = "cpu", 2, 2, writer
        def quantized(self, x, recorder): return x, torch.ones(1, 1), x.double(), None
        def reference(self, a, scale, **kwargs): return a.bfloat16(), torch.ones_like(a, dtype=torch.bool), FiniteBound(a.abs())
        def apply(self, x): pytest.fail("prepare made a native call")
    module = {"complete": True, "roles": [{"separation": {"complete": True, "both": 4, "recipes": [{"column": 0, "values": [1, -1]}, {"column": 1, "values": [1]}]}}]}
    row.dense_cases(Adapter(), unit, module, BoundaryRecorder(lambda: None, []))
    assert [c["category"] for c in artifact.document["cases"]] == ["sweep", "sweep", "sweep", "chunk_pairs"]
    assert unit["exact"]["sweep"]["columns"] == 3
    assert unit["exact"]["chunk_pairs"]["rows"] == 1


def test_capture_refusal_for_a_nonfinite_prepared_reference(tmp_path):
    api = phase_api()
    from experiments.numerics_rows.dense_phases import preflight_case
    artifact = api.Artifact(tmp_path, "prepare", {})
    case = {"category": "scale_index", "tensors": {name: artifact.tensor(tensor) for name, tensor in {
        "reference": torch.tensor([float("inf")]), "finite_value": torch.ones(1), "finite_held": torch.ones(1, dtype=torch.bool)}.items()}}
    with pytest.raises(Invalid, match="reference is not finite"):
        preflight_case(tmp_path, case)


def test_host_phase_negative_reference_control_allocates_no_device(monkeypatch):
    from experiments.numerics_rows.h1 import negative_controls
    calls = []
    original = torch.tensor
    def tensor(*args, **kwargs):
        calls.append(kwargs.get("device", "cpu"))
        assert kwargs.get("device", "cpu") == "cpu"
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "tensor", tensor)
    assert all(c["flagged_exactly"] for c in negative_controls(device="cpu"))
    assert calls == ["cpu"]


def test_payload_hash_covers_the_same_read_that_is_decoded(phase_pair, monkeypatch):
    api, prepared, _, _, _, _, _, _ = phase_pair()
    reference = prepared.document["cases"][0]["tensors"]["input"]
    target = prepared.root / reference["file"]
    original, reads = Path.read_bytes, []
    def changing(path):
        value = original(path)
        if path == target:
            reads.append(path)
            return value if len(reads) == 1 else b"changed after a separate hash read"
        return value
    monkeypatch.setattr(Path, "read_bytes", changing)
    assert torch.equal(api.payload(prepared.root, reference), torch.ones(1, 2))
    assert len(reads) == 1


def test_split_diagnostic_composition_is_recorded_without_a_gate_verdict(tmp_path):
    api = phase_api()
    from experiments.numerics_rows.dense_phases import compare_case
    prepared = api.Artifact(tmp_path / "prepare", "prepare", {})
    captured = api.Artifact(tmp_path / "capture", "capture", {})
    x = torch.ones(1, 1, dtype=torch.bfloat16)
    sf = torch.ones(1, 1)
    case = {"id": "diagnostic", "kind": "dense", "category": "diagnostic", "distribution": "gauss", "seed": 0,
            "configuration": {"k_split": 1}, "tensors": {k: prepared.tensor(v) for k, v in {
                "input": x, "q": x, "sf": sf, "reference": x,
                "finite_value": sf * 10, "finite_held": torch.ones(1, 1, dtype=torch.bool),
                "diagnostic_reference": sf, "products_abs": sf}.items()}}
    observed = {"completion": True, "boundaries": [{"kind": "quantizer", "sync_before": True, "sync_after": True}] * 2,
                "profiler_names": ["kernel"], "kernel_origins": [{"profiler_name": "kernel"}],
                "tensors": {k: captured.tensor(v) for k, v in {"output": x * 2, "staged": x * 3, "quantizer_q": x, "quantizer_sf": sf}.items()}}
    unit = row.new_unit({"cell": "dense"})
    compare_case(case, observed, prepared.root, captured.root, unit)
    assert not unit["findings"] and unit["diagnostic_composition_equal"] == [False]
    assert unit["exact"]["composition"]["cases"] == 0 and len(unit["diagnostic"]) == 1
