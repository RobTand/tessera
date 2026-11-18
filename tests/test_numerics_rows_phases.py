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


def rewrite_stream(artifact, name, values):
    import hashlib
    artifact.case_ids = hashlib.sha256()
    artifact.document["streams"].pop(name, None)
    (artifact.root / (name + ".jsonl")).write_text("")
    for value in values:
        artifact.record(name, value)
    artifact.document["streams"].setdefault(name, {"count": 0})
    artifact.finish()


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
    captured.document.update(scopes=[scope], prepare_digest=digest)
    observed = copy.deepcopy(case)
    observed["tensors"] = {"output": captured.tensor(torch.ones(1, 2))}
    captured.record("cases", observed)
    captured.finish()
    capture, _ = api.read(captured.root, "capture", common, [scope])
    assert len(list(api.check_capture(prepare, capture, digest, prepared.root, captured.root))) == 1
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
    changed = list(api.records(prepared.root, "cases"))
    changed[0]["scope"][field] = "other"
    rewrite_stream(prepared, "cases", changed)
    with pytest.raises(Invalid, match="another unit|exactly once"):
        api.read(prepared.root, "prepare", common, [scope])


def test_capture_must_consume_the_exact_prepare_digest(phase_pair):
    api, prepared, prepare, captured, capture, digest, _, _ = phase_pair()
    capture["prepare_digest"] = "other"
    with pytest.raises(Invalid, match="different prepare"):
        list(api.check_capture(prepare, capture, digest, prepared.root, captured.root))


@pytest.mark.parametrize("change", ["missing", "duplicate", "foreign"])
def test_compare_requires_each_case_exactly_once(phase_pair, change):
    api, prepared, prepare, captured, capture, digest, _, _ = phase_pair()
    changed = list(api.records(captured.root, "cases"))
    if change == "missing":
        changed.clear()
    elif change == "duplicate":
        changed.append(copy.deepcopy(changed[0]))
    else:
        changed[0]["id"] = "other"
    rewrite_stream(captured, "cases", changed)
    capture.update(captured.document)
    with pytest.raises(Invalid, match="exactly once"):
        list(api.check_capture(prepare, capture, digest, prepared.root, captured.root))


def test_phase_payload_bytes_and_configuration_are_bound(phase_pair):
    api, prepared, _, _, _, _, common, scope = phase_pair()
    reference = next(api.records(prepared.root, "cases"))["tensors"]["input"]
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
    assert not (tmp_path / "numerics_row.json").exists()
    assert receipt["row"]["gate_capture_finished"] is True
    assert receipt["row"]["finalization_finished"] is False
    assert json.loads((tmp_path / "progress.json").read_text())["phase"] == "diagnostics"
    assert "gpu_minutes" not in receipt["row"]
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


def test_lever_addendum_binds_the_frozen_record_without_rewriting_it(tmp_path, monkeypatch):
    phase_api()
    from experiments.numerics_rows.common import sha256
    rule, freeze, addendum = (tmp_path / n for n in ("rule.md", "freeze.json", "addendum.json"))
    monkeypatch.setattr(row, "FREEZE_ADDENDUM", addendum, raising=False)
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
    assert [c["category"] for c in api.records(artifact.root, "cases")] == ["sweep", "sweep", "sweep", "chunk_pairs"]
    assert unit["exact"]["sweep"]["columns"] == 3
    assert unit["exact"]["chunk_pairs"]["rows"] == 1


@pytest.mark.parametrize('structure', ('dense', 'routed'))
def test_separation_claim_is_independent_of_incomplete_scale_plan(monkeypatch, structure):
    from experiments.numerics_rows import run_routed
    record = {'complete': True, 'both': 1, 'recipes': []}
    manifest = {'complete': True, 'roles': [{'expert': 0, 'separation': record,
        'scale_index': [{'reads': 1, 'reads_covered': 1}]}]}
    adapter = SimpleNamespace(cols=1, inter=1, experts=24)
    monkeypatch.setattr(run_routed, 'composition', lambda *a: None)
    for name in ('tensor', 'zeros'):
        original = getattr(torch, name)
        monkeypatch.setattr(torch, name, lambda *a, _call=original, **kw: _call(*a, **{**kw, 'device': 'cpu'}))
    for record_complete, module_complete in ((True, True), (True, False), (False, True), (False, False)):
        record['complete'] = record_complete
        manifest['complete'] = module_complete
        manifest['roles'][0]['scale_index'][0]['reads_covered'] = int(module_complete)
        unit = row.new_unit({'m_config': {'M': 1, 'chunk_size': 1, 'k_split': 1, 'superblock_width': 1, 'route_counts': [1]}})
        (row.dense_cases if structure == 'dense' else run_routed.run)(adapter, unit, manifest, None)
        assert bool(unit['unattested']) == (not record_complete and not module_complete), 'wrong local separation attestation'
        assert bool(unit['invalid']) == (not record_complete and module_complete), 'C1 false complete claim must be INVALID'


def test_capture_refusal_for_a_nonfinite_prepared_reference(tmp_path):
    api = phase_api()
    from experiments.numerics_rows.dense_phases import preflight_case
    artifact = api.Artifact(tmp_path, "prepare", {})
    case = {"category": "scale_index", "tensors": {name: artifact.tensor(tensor) for name, tensor in {
        "reference": torch.tensor([float("inf")]), "finite_value": torch.ones(1), "finite_held": torch.ones(1, dtype=torch.bool)}.items()}}
    with pytest.raises(Invalid, match="reference is not finite"):
        preflight_case(tmp_path, case)


def test_host_phase_negative_reference_control_allocates_no_device(monkeypatch):
    phase_api()
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
    reference = next(api.records(prepared.root, "cases"))["tensors"]["input"]
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


def test_split_realistic_composition_mismatch_is_fail(tmp_path):
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
    unit = row.new_unit({"cell": "dense", "role": "unit"})
    compare_case(case, observed, prepared.root, captured.root, unit)
    row.finish_unit(unit)
    assert unit["outcome"] == "FAIL" and unit["diagnostic_composition_equal"] == [False]
    assert unit["exact"]["composition"]["cases"] == 1 and len(unit["diagnostic"]) == 1


@pytest.fixture
def split_capture(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from experiments import dense_fused_oracle as dense, routed_pair_oracle
    from experiments.numerics_rows import dense_phases as dp
    from experiments.numerics_rows.common import BoundaryError, Unattested
    from experiments.numerics_rows.gpu_time import PhaseTimes
    common, runs = {"test": "bound phases"}, []
    monkeypatch.setattr(dp, "identities", lambda receipt: common)
    monkeypatch.setattr(dense, "init_vllm_world1", nullcontext)
    monkeypatch.setattr(torch.profiler, "profile", lambda **_: nullcontext())
    monkeypatch.setattr(routed_pair_oracle, "kernel_table", lambda *a, **kw: [{"name": "kernel"}])
    monkeypatch.setattr(row, "native_boundaries", lambda *a: nullcontext())
    monkeypatch.setattr(row, "Dense", lambda *a: SimpleNamespace(info={}, pairs=lambda: []))
    monkeypatch.setattr(row, "identify_served", lambda *a: (object(), [{"path": "native.so"}]))
    monkeypatch.setattr(row, "actual_geometry", lambda *a: {})
    monkeypatch.setattr(row, "module_receipt", lambda *a: {})
    monkeypatch.setattr(row, "tp_facts", lambda *a: {})
    monkeypatch.setattr(row, "scale_receipt", lambda *a: {"equal": True})
    for name in ("check_served_identity", "audit_served", "check_served_after", "dispatch"):
        monkeypatch.setattr(row, name, lambda *a: None)
    def bind_profile(adapter, unit, recorder, records, profiles, **kw):
        unit["kernel_origins"] = [{"profiler_name": "kernel", "origin": "tessera"}]
    monkeypatch.setattr(row, "bind_kernel_profile", bind_profile)
    def run(error=None, index=0, *, bind_error=False, partition=False, selection=None, prepare_refusal=False, capture_findings=False):
        if capture_findings:
            assert hasattr(dp, "cases"), "feature absent: ordered case streams"
        if partition:
            assert hasattr(dp, "cases"), "feature absent: gate and diagnostics stream union"
        if prepare_refusal:
            assert hasattr(dp, "cases"), "feature absent: prepared stream refusals"
        root = tmp_path / str(len(runs)); runs.append(root)
        unit = row.new_unit({"cell": "dense", "module": "A", "role": "unit", "admitted_by_v68": True,
            "symbol": "dense", "decoder": "native", "rung": 1024, "launch_state_sha256": "a" * 64,
            "tp": {"size": 2, "rank": 0}, "m_config": {"M": 1, "k_split": 1}, "arm": {}})
        unit["exact"]["separation"]["records"] = [{"both": 1}]
        unit["exact"]["sweep"]["elements"] = 2
        unit["exact"]["scale_index"]["arrays"] = [{"launches": 1}]
        unit["exact"]["chunk_pairs"]["x_elements"] = 1
        row.finish_unit(unit)
        receipt = {"row": {"utc_start": "2026-11-17T00:00:00Z", "finalization_finished": False},
                   "units": [unit], "modules": [], "runtime": {}}
        groups = [({"name": "A", "pair": ["dense", "native"]}, {"wire_sha256": "w", "launch_geometry": {}}, unit["tp"], [unit])]
        scope = dp.scopes(groups)[0]
        prepared = dp.Artifact(root / "prepare", "prepare", common)
        prepared.document["scopes"] = [scope]
        x = torch.ones(1, 1, dtype=torch.bfloat16); sf = torch.ones(1, 1)
        for category in ("sweep", "sweep", "diagnostic"):
            prepared.add(scope, row.unit_identity(unit)[4], "dense", {"input": x, "q": x, "sf": sf, "reference": x * 2,
                "x_mask": torch.ones_like(x, dtype=torch.bool), "finite_value": sf * 10, "finite_held": sf.bool(),
                "diagnostic_reference": sf * 2, "products_abs": sf}, category=category, distribution="gauss", seed=1)
        if prepare_refusal:
            unit["unattested"].append("incomplete scale-index coverage before the freeze")
        prepared.finish(receipt=copy.deepcopy(receipt))
        unit["unattested"].clear()
        calls = []
        def capture(adapter, current, case, tensors, recorder):
            calls.append(case["id"])
            if capture_findings:
                current["findings"].append(case["id"])
            if error and len(calls) == index + 1:
                if error == "cuda":
                    raise BoundaryError({"call": "native", "kind": "tessera", "sync_before": True, "sync_after": False}, RuntimeError("CUDA error"))
                raise (Invalid if error == "invalid" else Unattested)("refused capture")
            current["boundaries"].extend([{"kind": "quantizer", "sync_before": True, "sync_after": True}] * 2)
            return {"output": x * 2, "staged": x * 2, "quantizer_q": x, "quantizer_sf": sf}
        monkeypatch.setattr(dp, "capture_case", capture)
        out = root / "capture-job"; out.mkdir()
        args = SimpleNamespace(out=out, prepare_artifact=[prepared.root],
            code_audit=SimpleNamespace(records=[], unit=None, context=lambda *a: nullcontext()),
            meter=SimpleNamespace(snapshot=lambda **kw: {}, finish=lambda: {}),
            launch_state=SimpleNamespace(check=lambda: None, diagnostics=lambda: {}, latched_diagnostic=lambda *a: {}))
        dp.device_capture(args, receipt, PhaseTimes(receipt["row"]), groups, {"key": "e4m3mma"}, [{"key": "e4m3mma", "native_source_path": "kernel.cu"}])
        args.capture_artifact = [out / "capture"]
        if partition:
            from shutil import copytree
            paired = []
            original_prepare = list(dp.cases(prepared.root, prepared.document))
            original_capture = list(dp.cases(args.capture_artifact[0], json.loads((args.capture_artifact[0] / "phase.json").read_text())))
            for part, indices in (("gate", [0, 1]), ("diagnostics", [2])):
                prep = dp.Artifact(root / part / "prepare", "prepare", common)
                prep.document.update(scopes=[scope], case_set=part, ordinal_start=indices[0])
                copytree(prepared.root / "bytes", prep.root / "bytes")
                for position in indices:
                    for reference in original_prepare[position]["tensors"].values():
                        prep.file(reference["file"])
                    prep.record("cases", original_prepare[position])
                part_digest = prep.finish(receipt=copy.deepcopy(prepared.document["receipt"]))
                cap = dp.Artifact(root / part / "capture", "capture", common)
                cap.document.update(scopes=[scope], case_set=part, ordinal_start=indices[0], prepare_digest=part_digest)
                copytree(args.capture_artifact[0] / "bytes", cap.root / "bytes")
                for position in indices:
                    for reference in original_capture[position]["tensors"].values():
                        cap.file(reference["file"])
                    cap.record("cases", original_capture[position])
                cap.finish(receipt=copy.deepcopy(receipt))
                paired.append((prep.root, cap.root))
            if selection == "missing": paired.pop()
            if selection == "duplicate": paired[1] = paired[0]
            args.prepare_artifact = [pair[0] for pair in reversed(paired)]
            args.capture_artifact = [pair[1] for pair in reversed(paired)]
        args.out = root / "compare-job"; args.out.mkdir()
        if bind_error:
            monkeypatch.setattr(dp, "bind_evidence", lambda *a: (_ for _ in ()).throw(RuntimeError("compare binding failed")))
            with pytest.raises(RuntimeError, match="compare binding failed"):
                dp.host_compare(args, receipt, PhaseTimes({}), groups)
            return receipt, calls
        if selection:
            with pytest.raises(Invalid, match="union exactly once"):
                dp.host_compare(args, receipt, PhaseTimes({}), groups)
            return receipt, {}
        dp.host_compare(args, receipt, PhaseTimes({}), groups)
        if partition:
            return receipt, json.loads((args.out / "compare/phase.json").read_text())
        captured = json.loads((args.capture_artifact[0] / "phase.json").read_text())
        observed = list(dp.cases(args.capture_artifact[0], captured))
        assert len(observed) == prepared.document["streams"]["cases"]["count"]
        return receipt, {**captured, "cases": observed}
    return run


def test_split_first_cuda_error_finishes_compare_as_fail(split_capture):
    positive, _ = split_capture()
    assert positive["row"]["finalization_finished"] and positive["units"][0]["outcome"] == "PASS"
    receipt, captured = split_capture("cuda")
    assert receipt["row"]["finalization_finished"] and receipt["units"][0]["outcome"] == receipt["row"]["outcome"] == "FAIL"
    assert all(c["not_run"] and not c["completion"] for c in captured["cases"][1:])


@pytest.mark.parametrize("error", ("invalid", "unattested"))
def test_split_last_case_preserves_refusal_class(split_capture, error):
    positive, _ = split_capture()
    assert positive["row"]["finalization_finished"] and positive["units"][0]["outcome"] == "PASS"
    receipt, _ = split_capture(error, index=2)
    assert receipt["row"]["finalization_finished"] and receipt["units"][0]["outcome"] == receipt["row"]["outcome"] == error.upper()


def test_split_compare_binds_its_artifact_before_finalization(split_capture):
    positive, _ = split_capture()
    assert positive["row"]["finalization_finished"] and "compare" in positive["phase_artifacts"]
    partial, _ = split_capture(bind_error=True)
    assert partial["row"]["finalization_finished"] is False



def test_split_gate_and_diagnostics_finish_one_receipt(split_capture):
    receipt, compared = split_capture(partition=True)
    assert receipt["row"]["finalization_finished"] and receipt["row"]["outcome"] == "PASS"
    assert len(receipt["units"]) == 1 and compared["streams"]["cases"]["count"] == 3
    assert compared["ranges"] == [[0, 2], [2, 1]]


@pytest.mark.parametrize("selection", ("missing", "duplicate"))
def test_split_compare_refuses_a_partial_or_duplicate_case_set(split_capture, selection):
    receipt, _ = split_capture(partition=True, selection=selection)
    assert not receipt["row"]["finalization_finished"]


def test_split_preserves_prepare_refusals_before_capture_and_compare(split_capture):
    receipt, captured = split_capture(prepare_refusal=True)
    assert receipt["row"]["finalization_finished"] and receipt["row"]["outcome"] == "UNATTESTED"
    assert all(case["not_run"] and not case["completion"] for case in captured["cases"])
    assert receipt["units"][0]["unattested"] == ["incomplete scale-index coverage before the freeze"]


@pytest.mark.parametrize("error,refusal_index", [(error, refusal_index) for error in ("invalid", "unattested") for refusal_index in range(3)] + [("prepare", 0)])
def test_split_compare_keeps_each_refusal_in_its_case_record(split_capture, error, refusal_index):
    receipt, captured = split_capture(prepare_refusal=True) if error == "prepare" else split_capture(error, index=refusal_index)
    field = "unattested" if error == "prepare" else error
    reason = "incomplete scale-index coverage before the freeze" if error == "prepare" else "refused capture"
    api = phase_api()
    root = Path(receipt["phase_artifacts"]["compare"]["path"]).parent
    compared = list(api.cases(root, json.loads((root / "phase.json").read_text())))
    for source, result in zip(captured["cases"][refusal_index:], compared[refusal_index:], strict=True):
        assert result[field] == source["status"][field] == [reason]
        assert result["outcome"] == field.upper()
    assert receipt["units"][0]["outcome"] == receipt["row"]["outcome"] == field.upper()


def test_capture_keeps_case_findings_in_records_and_a_bounded_summary(split_capture):
    receipt, captured = split_capture(capture_findings=True)
    assert receipt["row"]["outcome"] == "FAIL"
    assert len(captured["receipt"]["units"][0]["findings"]) == 1
    assert all(case["status"]["findings"] == [case["id"]] for case in captured["cases"])


def module_phase_pairs(root, *, case_sets=("gate", "diagnostics"), cuts=None, finding=False):
    from experiments.numerics_rows import dense_phases
    api = phase_api()
    cuts = cuts or [{"size": 2, "rank": 0}, {"size": 2, "rank": 1}]
    receipt = {"canary": False, "row": {"name": "module compare", "finalization_finished": False},
        "tool": dict.fromkeys(("commit", "tree_sha256"), "tool"),
        "tessera": dict.fromkeys(("commit", "contract_sha256", "src_tree_sha256", "kernel_source_sha256", "build_provenance_sha256", "retained_source"), "F"),
        "runtime": dict.fromkeys(("image_digest", "vllm", "torch", "launch_state_names", "launch_state_sha256", "start_environment_spec_sha256", "dependency_math", "dependency_math_sha256"), "runtime"),
        "inputs": {**dict.fromkeys(("population_sha256", "case_manifest_sha256", "build_sha256", "q1_table_sha256", "launch_levers_sha256"), "input"),
                   "selected_modules": ["A"], "selected_cuts": cuts},
        "rule": dict.fromkeys(("sha256", "h1_sha256", "frozen_utc"), "rule"), "units": [], "modules": []}
    groups, pairs, ranges = [], [], {}
    for cut in cuts:
        units = [row.new_unit({"cell": "dense", "module": "A", "symbol": "dense", "decoder": "native", "rung": 1024,
            "role": "unit", "tp": cut, "m_config": {"M": m, "k_split": 1}, "launch_state_sha256": "a" * 64, "arm": {}}) for m in (1, 2)]
        groups.append(({"name": "A"}, {"wire_sha256": "w"}, cut, units))
        receipt["units"].extend(units)
    for index, selected in enumerate(dense_phases.phase_units(groups)):
        selected_cuts, units = [g[2] for g in selected], [u for g in selected for u in g[3]]
        common = api.identities({**receipt, "inputs": {**receipt["inputs"], "selected_cuts": selected_cuts}})
        for part in case_sets:
            job = root / str(index) / part
            start = cuts.index(selected_cuts[0]) * (4 if part == "all" else 2) + (len(groups) * 2 if part == "diagnostics" else 0)
            prepared = api.Artifact(job / "prepare", "prepare", common)
            prepared.document.update(scopes=dense_phases.scopes(selected), case_set=part, ordinal_start=start)
            previous = copy.deepcopy({**receipt, "inputs": {**receipt["inputs"], "selected_cuts": selected_cuts}, "units": units})
            for unit in previous["units"]:
                unit["exact"]["separation"]["records"] = [{"both": 1}]
                unit["exact"]["sweep"]["elements"] = 10 * (cuts.index(unit["tp"]) + 1) + unit["m_config"]["M"] if part != "diagnostics" else 0
                unit["exact"]["scale_index"]["arrays"] = [{"launches": 1}]
                unit["exact"]["chunk_pairs"]["x_elements"] = 1
            x, sf = torch.ones(1, 1, dtype=torch.bfloat16), torch.ones(1, 1)
            for category in (("sweep", "diagnostic") if part == "all" else ("diagnostic" if part == "diagnostics" else "sweep",)):
                for unit in units:
                    prepared.add(api.unit_scope(unit, "w"), row.unit_identity(unit)[4], "dense", {
                        "input": x, "q": x, "sf": sf, "reference": x, "x_mask": sf.bool(),
                        "finite_value": sf * 10, "finite_held": sf.bool(), "diagnostic_reference": sf, "products_abs": sf},
                        category=category, distribution="gauss", seed=1)
            previous["units"].reverse()
            pd = prepared.finish(receipt=previous)
            captured = api.Artifact(job / "capture", "capture", common)
            captured.document.update(scopes=dense_phases.scopes(selected), case_set=part, ordinal_start=start, prepare_digest=pd)
            current = copy.deepcopy({**previous, "units": units, "modules": [{"name": "A", "tp": cut} for cut in selected_cuts]})
            current["row"] = {"name": f"cut {index} {part}", "finalization_finished": False}
            for unit in current["units"]:
                unit["arm"]["capture_fact"] = row.unit_identity(unit)
                if finding and unit["tp"] == {"size": 2, "rank": 1} and part != "diagnostics" and unit["m_config"]["M"] == 2:
                    unit["findings"] = ["cut 2 exact mismatch"]
            for case in api.cases(prepared.root, prepared.document):
                captured.record("cases", {**case, "completion": True,
                    "boundaries": [{"kind": "quantizer", "sync_before": True, "sync_after": True}] * 2,
                    "profiler_names": ["kernel"], "kernel_origins": [{"profiler_name": "kernel"}],
                    "code": {"unit_id": case["scope"]["unit_key"], "proof": case["id"]},
                    "tensors": {k: captured.tensor(v) for k, v in sorted({"output": x, "staged": x, "quantizer_q": x, "quantizer_sf": sf}.items())}})
            captured.finish(receipt=current)
            pairs.append((prepared.root, captured.root))
            for offset, cut in enumerate(selected_cuts):
                count = 4 if part == "all" else 2
                ranges[("A", cut["size"], cut["rank"], part)] = [start + offset * count, count]
    out = root / "compare-job"
    out.mkdir()
    args = SimpleNamespace(out=out, case_ranges=ranges, prepare_artifact=[p[0] for p in reversed(pairs)], capture_artifact=[p[1] for p in reversed(pairs)])
    return dense_phases, api, args, receipt, groups


@pytest.mark.parametrize("case_sets", [("gate", "diagnostics"), ("all",)])
def test_module_compare_finishes_one_receipt_with_every_cut(tmp_path, case_sets):
    from experiments.numerics_rows import dense_phases
    assert hasattr(dense_phases, "unit_map"), "feature absent: module cut artifact union"
    from experiments.numerics_rows.gpu_time import PhaseTimes
    dp, api, args, receipt, groups = module_phase_pairs(tmp_path, case_sets=case_sets)
    dp.host_compare(args, receipt, PhaseTimes({}), groups)
    assert receipt["row"]["finalization_finished"] and receipt["row"]["outcome"] == "PASS"
    assert len(receipt["units"]) == 4 and len(receipt["modules"]) == 2
    keys = {f'A:tp2r0:{part}:{phase}' for part in case_sets for phase in ("prepare", "capture")}
    assert set(receipt["phase_artifacts"]) == set(receipt["row"]["phase_jobs"]) == keys | {"compare"}
    for key in keys:
        reference = receipt["phase_artifacts"][key]
        assert reference["sha256"] == row.sha256(Path(reference["path"]))
        assert not receipt["row"]["phase_jobs"][key]["finalization_finished"]
    for unit in receipt["units"]:
        assert unit["exact"]["sweep"]["elements"] == 10 * (unit["tp"]["rank"] + 1) + unit["m_config"]["M"]
        assert unit["arm"]["capture_fact"] == unit["unit_id"]
        assert unit["code_stream"] == receipt["phase_artifacts"]["compare"]
    document, digest = api.read(args.out / "compare", "compare", api.identities(receipt), dp.scopes(groups))
    expected = sorted((case for root in args.capture_artifact for case in api.cases(root, json.loads((root / "phase.json").read_text()))), key=lambda case: case["ordinal"])
    compared = list(api.cases(args.out / "compare", document))
    assert len(compared) == 8 and digest == receipt["phase_artifacts"]["compare"]["sha256"]
    for source, result in zip(expected, compared, strict=True):
        assert all(result[field] == source[field] for field in ("id", "ordinal", "scope", "configuration", "code"))
        assert result["identities"] == api.identities(receipt)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "foreign", "mixed"])
def test_module_compare_refuses_invalid_cut_unions(tmp_path, fault):
    from experiments.numerics_rows import dense_phases
    assert hasattr(dense_phases, "unit_map"), "feature absent: module cut artifact union"
    from experiments.numerics_rows.gpu_time import PhaseTimes
    dp, _, args, receipt, groups = module_phase_pairs(tmp_path / "module")
    if fault == "missing":
        args.prepare_artifact, args.capture_artifact = args.prepare_artifact[2:], args.capture_artifact[2:]
    else:
        extra = args if fault == "duplicate" else module_phase_pairs(tmp_path / "extra", case_sets=("all",),
            cuts=[{"size": 1, "rank": 0}] if fault == "foreign" else [g[2] for g in groups])[2]
        args.prepare_artifact.append(extra.prepare_artifact[0])
        args.capture_artifact.append(extra.capture_artifact[0])
    with pytest.raises(Invalid, match="foreign module or cut|union exactly once"):
        dp.host_compare(args, receipt, PhaseTimes({}), groups)
    assert not receipt["row"]["finalization_finished"]


def test_module_compare_keeps_cut_two_fail_after_diagnostics(tmp_path):
    from experiments.numerics_rows import dense_phases
    assert hasattr(dense_phases, "unit_map"), "feature absent: module cut artifact union"
    from experiments.numerics_rows.gpu_time import PhaseTimes
    dp, _, args, receipt, groups = module_phase_pairs(tmp_path, finding=True)
    dp.host_compare(args, receipt, PhaseTimes({}), groups)
    assert receipt["row"]["finalization_finished"] and receipt["row"]["outcome"] == "FAIL"
    failed = [unit for unit in receipt["units"] if unit["outcome"] == "FAIL"]
    assert len(failed) == 1 and failed[0]["tp"]["rank"] == 1 and failed[0]["m_config"]["M"] == 2
    assert failed[0]["findings"] == ["cut 2 exact mismatch"] and receipt["row"]["exit_status"] == 3


@pytest.mark.parametrize("fault", ["cut identity", "tool identity", "range", "receipt units"])
def test_module_compare_keeps_exact_cut_bindings(tmp_path, fault):
    from experiments.numerics_rows import dense_phases
    assert hasattr(dense_phases, "unit_map"), "feature absent: module cut artifact union"
    from experiments.numerics_rows.gpu_time import PhaseTimes
    dp, _, args, receipt, groups = module_phase_pairs(tmp_path)
    path = args.prepare_artifact[0] / "phase.json"
    document = json.loads(path.read_text())
    if fault == "cut identity": document["identities"]["inputs"]["selected_cuts"] = receipt["inputs"]["selected_cuts"][:1]
    elif fault == "tool identity": document["identities"]["tool"]["commit"] = "other"
    elif fault == "receipt units": document["receipt"]["units"].pop()
    else: args.case_ranges[("A", 2, 1, "diagnostics")][0] += 1
    path.write_text(json.dumps(document))
    with pytest.raises(Invalid, match="identities, module or cut|deterministic full row|receipt identities or units"):
        dp.host_compare(args, receipt, PhaseTimes({}), groups)
    assert not receipt["row"]["finalization_finished"]


def test_module_compare_refuses_cuts_with_different_launch_states(tmp_path):
    from experiments.numerics_rows.gpu_time import PhaseTimes
    dp, _, args, receipt, groups = module_phase_pairs(tmp_path, case_sets=("all",))
    path = args.capture_artifact[0] / "phase.json"
    document = json.loads(path.read_text())
    assert [s["cut"] for s in document["scopes"][::2]] == [g[2] for g in groups]
    document["identities"]["runtime"]["launch_state_sha256"] = "b" * 64
    document["receipt"]["runtime"]["launch_state_sha256"] = "b" * 64
    path.write_text(json.dumps(document))
    with pytest.raises(Invalid, match="^row phase common identities, module or cut differ$"):
        dp.host_compare(args, receipt, PhaseTimes({}), groups)
    assert receipt["row"]["finalization_finished"] is not True


def test_tp2_phase_artifact_refuses_one_rank(tmp_path):
    from experiments.numerics_rows import dense_phases
    assert hasattr(dense_phases, "phase_units"), "feature absent: P5 refuses a one-rank TP2 artifact"
    from experiments.numerics_rows.gpu_time import PhaseTimes
    dp, _, args, receipt, groups = module_phase_pairs(tmp_path)
    dp.host_compare(args, receipt, PhaseTimes({}), groups)
    assert receipt["row"]["outcome"] == "PASS"
    receipt["row"]["finalization_finished"] = False
    path = args.prepare_artifact[0] / "phase.json"
    document = json.loads(path.read_text())
    document["scopes"] = document["scopes"][:2]
    path.write_text(json.dumps(document))
    with pytest.raises(Invalid, match="rule P5 requires both TP2 ranks in one process"):
        dp.host_compare(args, receipt, PhaseTimes({}), groups)
    assert not receipt["row"]["finalization_finished"]


@pytest.mark.parametrize("case_sets", [("gate", "diagnostics"), ("all",)])
def test_module_compare_finishes_tp1_and_tp2_pair_in_one_receipt(tmp_path, case_sets):
    from experiments.numerics_rows import dense_phases
    assert hasattr(dense_phases, "phase_units"), "feature absent: P5 module receipt holds TP1 and the TP2 pair"
    from experiments.numerics_rows.gpu_time import PhaseTimes
    cuts = [{"size": 1, "rank": 0}, {"size": 2, "rank": 0}, {"size": 2, "rank": 1}]
    dp, api, args, receipt, groups = module_phase_pairs(tmp_path, cuts=cuts, case_sets=case_sets)
    assert [len(unit) for unit in dp.phase_units(groups)] == [1, 2]
    dp.host_compare(args, receipt, PhaseTimes({}), groups)
    assert receipt["row"]["finalization_finished"] and receipt["row"]["outcome"] == "PASS"
    assert receipt["inputs"]["selected_cuts"] == cuts and len(receipt["units"]) == 6
    assert [m["tp"] for m in receipt["modules"]] == cuts
    assert set(receipt["phase_artifacts"]) == {"compare"} | {
        f"A:tp{size}r0:{part}:{phase}" for size in (1, 2) for part in case_sets for phase in ("prepare", "capture")}
    document, _ = api.read(args.out / "compare", "compare", api.identities(receipt), dp.scopes(groups))
    assert len(list(api.cases(args.out / "compare", document))) == 12
    assert all(u["code_stream"] == receipt["phase_artifacts"]["compare"] for u in receipt["units"])


@pytest.mark.parametrize("rank, field", [(0, 0), (1, 0), (0, 1), (1, 1)], ids=["rank0-start", "rank1-start", "rank0-count", "rank1-count"])
def test_tp2_pair_range_requires_both_rank_ranges(tmp_path, rank, field):
    from experiments.numerics_rows import dense_phases
    assert hasattr(dense_phases, "phase_units"), "feature absent: P5 checks the whole TP2 pair range"
    from experiments.numerics_rows.gpu_time import PhaseTimes
    dp, _, args, receipt, groups = module_phase_pairs(tmp_path)
    dp.host_compare(args, receipt, PhaseTimes({}), groups)
    assert receipt["row"]["outcome"] == "PASS"
    receipt["row"]["finalization_finished"] = False
    args.case_ranges[("A", 2, rank, "diagnostics")][field] += 1
    with pytest.raises(Invalid, match="deterministic full row enumeration"):
        dp.host_compare(args, receipt, PhaseTimes({}), groups)
    assert not receipt["row"]["finalization_finished"]


def test_module_compare_arguments_keep_every_artifact_pair(tmp_path):
    from experiments.numerics_rows import dense_phases, prepare_split
    assert hasattr(dense_phases, "unit_map"), "feature absent: module cut artifact union"
    import shlex
    pairs = [[tmp_path / f"cut{cut}/{part}/{phase}" for phase in ("prepare", "capture")] for cut in (0, 1) for part in ("gate", "diagnostics")]
    config = {"module": "A", "canary": False, "phase_artifacts": pairs}
    command = shlex.split(prepare_split.row_arguments(config, tmp_path, tmp_path, tmp_path, {}, "compare"))
    for flag, position in (("--prepare-artifact", 0), ("--capture-artifact", 1)):
        assert [command[i + 1] for i, value in enumerate(command) if value == flag] == [str(pair[position]) for pair in pairs]
    assert not {"--tp-size", "--tp-rank"} & set(command)

def test_lever_addendum_refuses_another_path_with_identical_bytes(tmp_path, monkeypatch):
    from experiments.numerics_rows.common import sha256
    rule, freeze, addendum = (tmp_path / n for n in ("rule.md", "freeze.json", "freeze-addendum-v2.json"))
    rule.write_text("frozen rule"); freeze.write_text("{}")
    monkeypatch.setattr(row, "FREEZE_ADDENDUM", addendum, raising=False)
    addendum.write_text(json.dumps({"freeze_record_sha256": sha256(freeze), "rule_sha256": sha256(rule),
        "reviewer": "lead", "frozen_utc": "2026-11-17T04:00:00Z", "launch_levers_sha256": "a" * 64}))
    assert row.bind_freeze_addendum({}, freeze, rule, addendum)["freeze_addendum_sha256"] == sha256(addendum)
    other = tmp_path / "copied-addendum.json"; other.write_bytes(addendum.read_bytes())
    with pytest.raises(Invalid, match="fixed rule path"):
        row.bind_freeze_addendum({}, freeze, rule, other)


@pytest.mark.parametrize("phase", ("prepare", "capture", "compare"))
def test_counted_split_launch_selects_a_whole_module_and_preserves_artifact_inputs(tmp_path, phase):
    import shlex
    from experiments.numerics_rows import prepare_split as api
    assert hasattr(api, "row_arguments"), "feature absent: counted whole-module split preparation"
    paths = {"source": "/retained/F", "manifest": "/shared/C1.json", "build": "/shared/build.json"}
    config = {"module": "module-A", "canary": False}
    command = shlex.split(api.row_arguments(config, tmp_path, Path(paths["source"]), "/signed/population.json", paths, phase))
    assert command[command.index("--module") + 1] == "module-A"
    assert not {"--tp-size", "--tp-rank", "--canary"} & set(command)
    assert ("--prepare-artifact" in command) == (phase != "prepare")
    assert ("--capture-artifact" in command) == (phase == "compare")
    canary = shlex.split(api.row_arguments({**config, "canary": True}, tmp_path, Path(paths["source"]), "/signed/population.json", paths, phase))
    assert "--canary" in canary and canary[canary.index("--tp-size") + 1] == "2"
    assert "--tp-rank" not in canary


@pytest.fixture
def split_canary_evidence(tmp_path):
    from experiments.numerics_rows import prepare_split as api
    from experiments.numerics_rows.common import sha256, write_json
    paths = {key: tmp_path / key for key in api.INPUTS}
    for path in paths.values():
        path.write_text("bound bytes")
    directory = tmp_path / "canary"
    directory.mkdir()
    write_json(directory / "campaign.json", {"canary": True, "tool_commit": "tool", "paths": {k: str(v) for k, v in paths.items()},
        "digests": {k: sha256(v) for k, v in paths.items() if k != "source"}})
    for phase in ("prepare", "capture", "compare"):
        (directory / phase / "results").mkdir(parents=True)
        (directory / phase / "tool-exit.txt").write_text("0\n")
    receipt = {"canary": True, "tool": {"commit": "tool"}, "tessera": {"commit": api.SOURCE}, "rule": {"sha256": api.RULE_SHA},
        "row": {"outcome": "PASS", "exit_status": 0, "finalization_finished": True}, "units": [{"outcome": "PASS"}]}
    write_json(directory / "compare/results/numerics_row.json", receipt)
    meter = {"wall_minutes": 1, "gpu_busy_minutes": 0.2, "peak_device_memory_mib": 800, "peak_power_watts": 15,
        "mean_sample_power_watts": 12, "whole_job_peak_memory_bytes": 2**30, "sampled_seconds": 60,
        "busy_sampled_seconds": 60, "samples": [{"compute_process_count": 1}] * 2, "errors": []}
    write_json(directory / "capture/results/numerics_row.json", {"row": {"gpu_minutes": meter}})
    return api, directory, paths, receipt, meter


@pytest.mark.parametrize("mutation", ("unfinished", "fail", "tool", "source", "input", "exit"))
def test_counted_split_refuses_an_unfinished_failed_or_mixed_canary(split_canary_evidence, mutation):
    from experiments.numerics_rows.common import write_json
    api, directory, paths, receipt, _ = split_canary_evidence
    api.check_canary(directory, "tool", paths)
    if mutation == "unfinished": receipt["row"]["finalization_finished"] = False
    elif mutation == "fail": receipt["units"][0]["outcome"] = "FAIL"
    elif mutation == "tool": receipt["tool"]["commit"] = "other"
    elif mutation == "source": receipt["tessera"]["commit"] = "other"
    elif mutation == "input": paths["build"].write_text("changed bytes")
    else: (directory / "capture/tool-exit.txt").write_text("3\n")
    write_json(directory / "compare/results/numerics_row.json", receipt)
    with pytest.raises(Invalid, match="canary|F chain"):
        api.check_canary(directory, "tool", paths)


@pytest.mark.parametrize("field", ("gpu_busy_minutes", "peak_device_memory_mib", "peak_power_watts", "whole_job_peak_memory_bytes", "busy_sampled_seconds"))
def test_capture_job_refuses_success_without_its_own_telemetry(split_canary_evidence, field):
    from experiments.numerics_rows.common import write_json
    api, directory, _, _, meter = split_canary_evidence
    path = directory / "capture/results/numerics_row.json"
    api.check_telemetry(path)
    meter[field] = None
    write_json(path, {"row": {"gpu_minutes": meter}})
    with pytest.raises(Invalid, match="capture telemetry"):
        api.check_telemetry(path)


@pytest.mark.parametrize("mutation", ("no-process", "unknown-process", "zero-memory", "false-zero"))
def test_capture_job_refuses_unobserved_compute_process_or_zero_memory(split_canary_evidence, mutation):
    from experiments.numerics_rows.common import write_json
    api, directory, _, _, meter = split_canary_evidence
    path = directory / "capture/results/numerics_row.json"
    meter["gpu_busy_minutes"] = 0
    write_json(path, {"row": {"gpu_minutes": meter}})
    api.check_telemetry(path)
    if mutation in ("no-process", "unknown-process", "false-zero"):
        meter["samples"] = [{"compute_process_count": None if mutation == "unknown-process" else 0}] * 2
    if mutation in ("zero-memory", "false-zero"):
        meter["peak_device_memory_mib"] = 0
    write_json(path, {"row": {"gpu_minutes": meter}})
    with pytest.raises(Invalid, match="capture telemetry"):
        api.check_telemetry(path)


def test_capture_telemetry_flags_multiple_processes_without_refusal(split_canary_evidence):
    from experiments.numerics_rows.common import write_json
    from experiments.numerics_rows.gpu_time import summarize
    api, directory, _, _, meter = split_canary_evidence
    path = directory / "capture/results/numerics_row.json"
    meter["samples"] = [{"compute_process_count": 2}] * 2
    write_json(path, {"row": {"gpu_minutes": meter}})
    result = api.check_telemetry(path)
    assert result.get("max_compute_process_count") == 2 and result.get("flag") == "count_above_1"
    samples = [{"t": t, "compute_process_count": 2, "busy_percent": 96} for t in (0, 1)]
    result = summarize(samples, 0, 1)
    assert result.get("max_compute_process_count") == 2 and result.get("flag") == "count_above_1"


def test_capture_telemetry_records_known_sample_coverage(split_canary_evidence):
    api, directory, _, _, _ = split_canary_evidence
    result = api.check_telemetry(directory / "capture/results/numerics_row.json")
    assert result["known_sample_coverage_fraction"] == 1 and result["busy_sampled_seconds"] == 60


@pytest.fixture
def feeder_booking(tmp_path, monkeypatch):
    from experiments.numerics_rows import counted_feeder as api, prepare_split
    canary, rowdir = tmp_path / "canary", tmp_path / "row"
    canary.mkdir(); rowdir.mkdir()
    def write(path, value):
        path.write_text(json.dumps(value))
    write(canary / "counted-rows.json", {"module-A": str(rowdir)})
    write(canary / "counted-chain-inputs.json", {"module-A": {}})
    write(canary / "campaign.json", {"tool_commit": "tool", "module": "module-A"})
    write(rowdir / "campaign.json", {"bound": True})
    bookings = tmp_path / "bookings.json"
    write(bookings, {"module-A": {phase: {"memory_gib": 8 if phase == "prepare" else 5, "minutes": 15 if phase == "capture" else 10} for phase in ("prepare", "capture", "compare")}})
    for phase in ("prepare", "capture", "compare"):
        job = rowdir / phase
        (job / "results" / phase).mkdir(parents=True)
        (job / "job.sh").write_text("timeout --kill-after=20s 100s command\n")
        (job / "submit-after-review.sh").write_text("flux submit -t5m --resources-shape='slot/[core=1;memory=4]' --requires=host:sparklina command\n")
        (job / "tool-exit.txt").write_text("0")
        for kind in ("payload", "podman"):
            (job / f"results/{kind}-exit.txt").write_text("0")
        write(job / "results" / phase / "phase.json", {"completion": True})
    write(rowdir / "compare/results/numerics_row.json", {"row": {"outcome": "PASS", "finalization_finished": True}})
    monkeypatch.setattr(prepare_split, "check_canary", lambda *a: None)
    monkeypatch.setattr(api, "sha", lambda path: "a" * 64)
    write(canary/'counted-feeder-state.json', {'bookings_sha256': 'a'*64, 'settings_sha256': 'a'*64,
          'jobs': {}, 'lanes': [['fv1e-sparklina', 'sparklina']], 'fv1e_admission': 'admitted'})
    monkeypatch.setattr(api, 'client', lambda name: ['flux'])
    monkeypatch.setattr(api, 'observe', lambda name: {'name': name})
    from experiments.numerics_rows import phase_artifact
    monkeypatch.setattr(phase_artifact, 'tensor_subset_equal', lambda *a: {'cases': 1})
    def run(command, **kw):
        if command[0] == "bash":
            job = Path(command[1]).parent
            jobid = {'prepare': 'f123', 'capture': 'f124', 'compare': 'f125'}[job.name]
            (job/'jobid').write_text(jobid)
            write(job/'instance.json', {'name': (job/'instance-name.txt').read_text().strip(), 'job_id': jobid})
    monkeypatch.setattr(api.subprocess, "run", run)
    (canary/'prepare').mkdir()
    (canary/'prepare/jobid').write_text('f111')
    settings = api.read(api.SETTINGS)
    settings['fv1e_jobmem_row_dirs'] = [str(tmp_path/'rows-anon'), str(tmp_path/'rows')]
    settings_path = tmp_path/'settings.json'
    write(settings_path, settings)
    monkeypatch.setattr(api, 'SETTINGS', settings_path)
    for jobid, prefix in [('f111', ''), ('f123', 'fv1e-sparklina-')]:
        sample_path = tmp_path/'rows-anon'/(prefix + jobid + '-3.tsv')
        sample_path.parent.mkdir(exist_ok=True)
        sample_path.write_text('0 0 0 0 0 30000000000 0 0 3000000000 50 ok\n')
    monkeypatch.setattr(api.subprocess, "check_output", lambda command, **kw: '100' if '{runtime:.6f}' in command else '0')
    monkeypatch.setattr(api, "wait_dense_slot", lambda: None, raising=False)
    return api, canary, rowdir, bookings


@pytest.mark.parametrize('fault', ('none', 'missing', 'short', 'unknown', 'status', 'samples', 'above-total', 'negative', 'runtime', 'query', 'malformed'))
def test_feeder_anon_peak_requires_valid_fleet_row(feeder_booking, monkeypatch, fault):
    api, canary, _, _ = feeder_booking
    assert hasattr(api, 'anon_peak'), 'feature absent: sampled anon admission'
    settings = api.read(api.SETTINGS)
    path = Path(settings['fv1e_jobmem_row_dirs'][0])/'f111-3.tsv'
    fields = path.read_text().split()
    mutations = {'unknown': (8, '?'), 'status': (10, 'invalid'), 'samples': (9, '49'),
                 'above-total': (8, '30000000001'), 'negative': (8, '-1'), 'malformed': (9, 'x')}
    if fault in mutations:
        index, value = mutations[fault]; fields[index] = value
    path.write_text(' '.join(fields[:10] if fault == 'short' else fields))
    if fault == 'missing':
        path.unlink()
    if fault in ('runtime', 'query'):
        def query(*a, **kw):
            if fault == 'query':
                raise api.subprocess.CalledProcessError(1, 'flux')
            return '0'
        monkeypatch.setattr(api.subprocess, 'check_output', query)
    assert api.anon_peak('f111', 'fv1', settings) == (3000000000 if fault == 'none' else None)


def test_feeder_anon_peak_uses_configured_column_dirs_and_host_ranks(feeder_booking):
    api, _, _, _ = feeder_booking
    assert hasattr(api, 'anon_peak'), 'feature absent: sampled anon admission'
    settings = api.read(api.SETTINGS)
    first, second = map(Path, settings['fv1e_jobmem_row_dirs'])
    second.mkdir()
    source = first/'fv1e-sparklina-f123-3.tsv'
    (second/source.name).write_text(source.read_text().replace('3000000000', '4000000000'))
    assert api.anon_peak('f123', 'fv1e-sparklina', settings) == 3000000000
    source.unlink()
    assert api.anon_peak('f123', 'fv1e-sparklina', settings) == 4000000000
    settings['fv1e_anon_peak_column'] = 8
    (second/source.name).write_text('0 0 0 0 0 30000000000 0 4000000001 ? 50 ok')
    assert api.anon_peak('f123', 'fv1e-sparklina', settings) == 4000000001
    (second/'fv1e-sparklina-f123-7.tsv').write_text('0 0 0 0 0 30000000000 0 ? ? 50 ok')
    assert api.anon_peak('f123', 'fv1e-sparklina', settings) is None


@pytest.fixture
def anon_wave(feeder_booking, monkeypatch):
    import shutil
    api, canary, rowdir, bookings = feeder_booking
    settings = api.read(api.SETTINGS)
    settings['fv1e_max_prepares_per_spark'] = 2
    api.SETTINGS.write_text(json.dumps(settings))
    (canary/'counted-feeder-state.json').unlink()
    rows, requested = api.read(canary/'counted-rows.json'), api.read(bookings)
    for suffix in 'BCDEFG':
        target = rowdir.with_name('row' + suffix)
        shutil.copytree(rowdir, target)
        rows['module-' + suffix] = str(target)
        requested['module-' + suffix] = requested['module-A']
    (canary/'counted-rows.json').write_text(json.dumps(rows))
    (canary/'counted-chain-inputs.json').write_text(json.dumps(dict.fromkeys(rows, {})))
    bookings.write_text(json.dumps(requested))
    peaks, submits = {}, []
    original = api.subprocess.run
    ids = iter('fAA' + char for char in 'ABCDEFGHJKLMNPQRSTUVWXYZ')
    def run(command, **kw):
        if command[0] != 'bash':
            return original(command, **kw)
        job = Path(command[1]).parent
        name = (job/'instance-name.txt').read_text().strip()
        module = next(m for m, path in rows.items() if Path(path) == job.parent)
        jobid = next(ids)
        (job/'jobid').write_text(jobid)
        (job/'instance.json').write_text(json.dumps({'name': name, 'job_id': jobid}))
        submits.append((module, job.name, name))
        prefix = '' if name == 'fv1' else name + '-'
        peak = peaks.get(module, 3 * 2**30)
        if peak is not None:
            (Path(settings['fv1e_jobmem_row_dirs'][0])/(prefix+jobid+'-3.tsv')).write_text(f'0 0 0 0 0 30000000000 0 0 {peak} 50 ok')
    monkeypatch.setattr(api.subprocess, 'run', run)
    def observe(name):
        if name == 'fv1e-sparky':
            raise OSError('not installed')
        return {'name': name}
    monkeypatch.setattr(api, 'observe', observe)
    return api, canary, rows, bookings, peaks, submits


@pytest.mark.parametrize('canary_peak', (4 * 2**30, 4 * 2**30 + 1, None))
def test_feeder_anon_admission_falls_back_and_rechecks(anon_wave, canary_peak):
    api, canary, _, bookings, _, submits = anon_wave
    assert hasattr(api, 'anon_peak'), 'feature absent: sampled anon admission'
    path = Path(api.read(api.SETTINGS)['fv1e_jobmem_row_dirs'][0])/'f111-3.tsv'
    path.write_text(path.read_text().replace('3000000000', str(canary_peak) if canary_peak is not None else '?'))
    api.run(canary, bookings)
    prepares = [(m, n) for m, p, n in submits if p == 'prepare']
    events = [json.loads(line) for line in (canary/'counted-feeder.jsonl').read_text().splitlines()]
    if canary_peak == 4 * 2**30 + 1:
        assert all(name == 'fv1' for _, name in prepares)
        assert prepares[-1][0] == 'module-A'
        assert next(e for e in events if e['event'] == 'EP_NOT_RUN')['ran'] is False
    else:
        assert next(m for m, n in prepares if n != 'fv1') == 'module-A'
        if canary_peak is None:
            assert prepares[0][0] != 'module-A' and prepares[0][1] == 'fv1'
        ep = next(i for i, e in enumerate(events) if e['event'] == 'EP_TENSOR_EQUAL')
        e_ids = {v['id'] for v in api.read(canary/'counted-feeder-state.json')['jobs'].values() if v['instance']['name'] != 'fv1'}
        assert not any(e['event'] == 'WAIT' and e.get('module') != 'module-A' and e.get('id') in e_ids for e in events[:ep])
    assert any('LOWER bound' in e.get('policy', '') and 'MemoryMax' in e.get('policy', '') for e in events)


@pytest.mark.parametrize('late_peak', (3 * 2**30, 4 * 2**30 + 1))
def test_feeder_invalid_rows_hold_until_a_valid_fv1_probe(anon_wave, late_peak):
    api, canary, rows, bookings, peaks, submits = anon_wave
    assert hasattr(api, 'anon_peak'), 'feature absent: sampled anon admission'
    root = Path(api.read(api.SETTINGS)['fv1e_jobmem_row_dirs'][0])
    (root/'f111-3.tsv').unlink()
    peaks.update(dict.fromkeys(rows, None))
    peaks.update({'module-C': late_peak, 'module-A': 3 * 2**30})
    api.run(canary, bookings)
    prepares = [(m, n) for m, p, n in submits if p == 'prepare']
    if late_peak > 4 * 2**30:
        assert prepares[-1] == ('module-A', 'fv1')
    else:
        assert prepares[:2] == [('module-B', 'fv1'), ('module-C', 'fv1')]
        assert next(m for m, n in prepares if n != 'fv1') == 'module-A'


def test_feeder_invalid_anon_wave_releases_held_row_last(anon_wave, monkeypatch):
    api, canary, rows, bookings, peaks, submits = anon_wave
    root = Path(api.read(api.SETTINGS)['fv1e_jobmem_row_dirs'][0])
    (root/'f111-3.tsv').unlink()
    peaks.update(dict.fromkeys(rows, None))
    def pause(seconds):
        assert False, 'an exhausted invalid anon wave must finish without HOLD'
    monkeypatch.setattr(api.time, 'sleep', pause)
    api.run(canary, bookings)
    prepares = [(m, n) for m, p, n in submits if p == 'prepare']
    assert len(prepares) == len(rows) and all(n == 'fv1' for _, n in prepares)
    assert prepares[-1][0] == 'module-A'
    events = [json.loads(line) for line in (canary/'counted-feeder.jsonl').read_text().splitlines()]
    reason = 'no valid anon sample in this wave'
    assert next(e for e in events if e['event'] == 'STOP')['reason'] == reason
    assert next(e for e in events if e['event'] == 'EP_NOT_RUN')['reason'] == reason
    assert api.read(canary/'counted-feeder-state.json')['fv1e_admission'] == 'stopped'
    assert events[-1]['event'] == 'COMPLETE'


def test_feeder_anon_rolling_stop_finishes_running_and_keeps_fv1(anon_wave):
    api, canary, _, bookings, peaks, submits = anon_wave
    assert hasattr(api, 'anon_peak'), 'feature absent: sampled anon admission'
    peaks['module-B'] = 4 * 2**30 + 1
    api.run(canary, bookings)
    events = [json.loads(line) for line in (canary/'counted-feeder.jsonl').read_text().splitlines()]
    stop = next(i for i, e in enumerate(events) if e['event'] == 'STOP')
    assert any(e['event'] == 'END' and e.get('module') == 'module-C' and e.get('phase') == 'prepare' for e in events[stop:])
    assert all(name == 'fv1' for module, phase, name in submits if module in ('module-F', 'module-G'))
    assert api.read(canary/'counted-feeder-state.json')['fv1e_admission'] == 'stopped'
    assert events[-1]['event'] == 'COMPLETE'


def test_feeder_fv1e_pause_holds_first_and_preserves_saved_instances(anon_wave, monkeypatch):
    api, canary, rows, bookings, _, submits = anon_wave
    assert hasattr(api, 'anon_peak'), 'feature absent: sampled anon admission'
    pause = canary/'pause-fv1e-sparklina'
    pause.touch()
    def release(seconds):
        assert all(n == 'fv1' for _, p, n in submits if p == 'prepare')
        assert not (Path(rows['module-A'])/'prepare/jobid').exists()
        pause.unlink()
    monkeypatch.setattr(api.time, 'sleep', release)
    api.run(canary, bookings)
    assert next(m for m, p, n in submits if p == 'prepare' and n != 'fv1') == 'module-A'
    before = list(submits)
    pause.touch()
    api.run(canary, bookings)
    assert submits == before
    assert api.read(Path(rows['module-A'])/'prepare/instance.json')['name'] == 'fv1e-sparklina'


@pytest.mark.parametrize("field", ("inner-timeout", "walltime", "shape"))
@pytest.mark.parametrize("mutation", ("missing", "duplicate"))
def test_feeder_refuses_missing_or_ambiguous_booking_fields(feeder_booking, field, mutation):
    api, canary, rowdir, bookings = feeder_booking
    api.run(canary, bookings)
    for path in rowdir.glob("*/jobid"):
        path.unlink()
    job = rowdir / "prepare" / ("job.sh" if field == "inner-timeout" else "submit-after-review.sh")
    text = job.read_text()
    token = {"inner-timeout": "--kill-after=20s 540s ", "walltime": " -t10m ", "shape": "slot/[core=1;memory=6]"}[field]
    assert token in text
    job.write_text(text.replace(token, "missing" if mutation == "missing" else token + token))
    with pytest.raises(ValueError, match="EXACTLY_ONE_MATCH"):
        api.run(canary, bookings)
    assert not list(rowdir.glob("*/jobid"))


def test_dense_prepare_waits_for_pending_aarch64_jobs(monkeypatch):
    from experiments.numerics_rows import counted_feeder as api
    assert hasattr(api, 'wait_dense_slot'), 'feature absent: dense prepare admission'
    replies = iter([{'jobs': [{'id': 1}]}, {'attributes': {'system': {'constraints': {'properties': ['aarch64']}}}}, {'jobs': []}])
    monkeypatch.setattr(api.subprocess, 'check_output', lambda *a, **kw: json.dumps(next(replies)))
    pauses = []
    monkeypatch.setattr(api.time, 'sleep', pauses.append)
    api.wait_dense_slot()
    assert pauses == [15]
    assert api.excludes_sparks({'properties': ['host:dl380g10']})
    assert not api.excludes_sparks({})
    assert not api.excludes_sparks({'or': [{'properties': ['amd64']}, {'properties': ['aarch64']}]})


def test_dense_admission_ignores_held_jobs_but_waits_for_active_jobs(monkeypatch):
    from experiments.numerics_rows import counted_feeder as api
    polls, queried, pauses = [], [], []
    def query(command, **kw):
        if command[1] == 'jobs':
            polls.append(True)
            return json.dumps({'jobs': [{'id': 11, 'urgency': 0}, {'id': 22, 'urgency': 8}]} if len(polls) == 1 else {'jobs': []})
        queried.append(command[3])
        return json.dumps({'attributes': {'system': {'constraints': {'properties': ['aarch64']}}}})
    monkeypatch.setattr(api.subprocess, 'check_output', query)
    monkeypatch.setattr(api, 'time', SimpleNamespace(monotonic=lambda: 0, sleep=pauses.append))
    api.wait_dense_slot()
    assert queried == ['22']
    assert pauses == [15]


def test_dense_admission_reports_one_stall_after_thirty_minutes(monkeypatch, capsys):
    from experiments.numerics_rows import counted_feeder as api
    elapsed, polls, stalls = [0], [], []
    def query(command, **kw):
        if command[1] == 'jobs':
            polls.append(True)
            return json.dumps({'jobs': [{'id': 33, 'urgency': 8}]} if len(polls) <= 4 else {'jobs': []})
        return json.dumps({'attributes': {'system': {'constraints': {}}}})
    def pause(seconds):
        stalls.extend((elapsed[0], line) for line in capsys.readouterr().out.splitlines() if 'STALL' in line)
        elapsed[0] += 900
    monkeypatch.setattr(api.subprocess, 'check_output', query)
    monkeypatch.setattr(api, 'time', SimpleNamespace(monotonic=lambda: elapsed[0], sleep=pause))
    api.wait_dense_slot()
    assert len(stalls) == 1, 'one STALL report is required'
    assert stalls[0][0] == 1800 and '33' in stalls[0][1]


@pytest.fixture
def fv1e_client(tmp_path, monkeypatch):
    import shlex
    from experiments.numerics_rows import counted_feeder as api
    log, state, client = tmp_path / 'calls', tmp_path / 'polls', tmp_path / 'fxe.sh'
    reason = tmp_path / 'reason'
    client.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> ' + shlex.quote(str(log)) + '\n'
        'if test "$2 $3" = "resource drain"; then\n'
        '  if test "$4" = 0; then printf "%s\\n" "$5" > ' + shlex.quote(str(reason)) + ';\n'
        '  else cat ' + shlex.quote(str(reason)) + '; fi\nfi\n'
        'if test "$2 $3" = "resource undrain"; then rm ' + shlex.quote(str(reason)) + '; fi\n'
        'if test "$2" = jobs; then\n'
        '  if test -e ' + shlex.quote(str(state)) + '; then echo \'{"jobs":[]}\';\n'
        '  else touch ' + shlex.quote(str(state)) + '; echo \'{"jobs":[{"id":1}]}\'; fi\nfi\n')
    client.chmod(0o700)
    monkeypatch.setattr(api, 'FXE', str(client), raising=False)
    elapsed, pauses = [0], []
    def pause(seconds):
        elapsed[0] += seconds
        pauses.append(seconds)
    monkeypatch.setattr(api, 'time', SimpleNamespace(monotonic=lambda: elapsed[0], sleep=pause))
    return api, log, pauses


@pytest.mark.parametrize('memory', (32, 36, 37))
def test_fv1e_drain_only_above_memory_limit(fv1e_client, memory):
    api, log, pauses = fv1e_client
    assert hasattr(api, 'drain_ecores_before_submit'), 'feature absent: fv1e memory admission'
    api.drain_ecores_before_submit(memory, 'flux submit --requires=host:sparklina command', 'test-drain')
    calls = log.read_text().splitlines() if log.exists() else []
    if memory <= 36:
        assert calls == [] and pauses == []
    else:
        assert calls[0].startswith('sparklina resource drain 0 ')
        assert calls[1:] == ['sparklina jobs --json', 'sparklina jobs --json']
        assert pauses


def test_fv1e_large_job_requires_one_spark(fv1e_client):
    api, log, pauses = fv1e_client
    assert hasattr(api, 'drain_ecores_before_submit'), 'feature absent: fv1e memory admission'
    for submit in ('flux submit', 'flux submit --requires=host:dl380g10',
                   'flux submit --requires=host:sparky --requires=host:sparklina'):
        with pytest.raises(api.FeederError, match='FV1E_REQUIRES_ONE_SPARK_HOST'):
            api.drain_ecores_before_submit(37, submit, 'test-drain')
    assert not log.exists() and not pauses


@pytest.mark.parametrize('error', ('deadline', 'query-timeout', 'query-error', 'parse-error'))
def test_fv1e_drain_refuses_when_jobs_do_not_end(fv1e_client, monkeypatch, error):
    api, log, pauses = fv1e_client
    assert hasattr(api, 'drain_ecores_before_submit'), 'feature absent: fv1e memory admission'
    original = api.subprocess.check_output
    def query(command, **kw):
        if command[2] != 'jobs':
            return original(command, **kw)
        if error == 'query-timeout':
            raise api.subprocess.TimeoutExpired(command, 1)
        if error == 'query-error':
            raise RuntimeError('query failed')
        return 'not JSON' if error == 'parse-error' else '{"jobs":[{"id":1}]}'
    monkeypatch.setattr(api.subprocess, 'check_output', query)
    expected = RuntimeError if error == 'query-error' else json.JSONDecodeError if error == 'parse-error' else api.FeederError
    with pytest.raises(expected):
        api.drain_ecores_before_submit(37, 'flux submit --requires=host:sparky command', 'test-drain')
    assert log.read_text().startswith('sparky resource drain 0 ')
    assert log.read_text().splitlines()[-1] == 'sparky resource undrain 0 test-drain', 'failed drain must release its own reason'
    assert not (log.parent / 'reason').exists()
    assert 0 <= sum(pauses) <= 300


@pytest.mark.parametrize('reason', ('test-drain', 'TP2 smoke'))
def test_fv1e_undrain_keeps_another_owners_reason(fv1e_client, reason):
    api, log, _ = fv1e_client
    assert hasattr(api, 'undrain_ecores'), 'feature absent: owned fv1e undrain'
    drain = api.drain_ecores_before_submit(37, 'flux submit --requires=host:sparky command', 'test-drain')
    (log.parent / 'reason').write_text(reason + '\n')
    api.undrain_ecores(**drain)
    releases = [line for line in log.read_text().splitlines() if 'resource undrain' in line]
    assert releases == (['sparky resource undrain 0 test-drain'] if reason == 'test-drain' else [])


@pytest.mark.parametrize('failure', (None, 'native', 'artifact'))
def test_feeder_undrains_only_after_clean_and_checks(feeder_booking, monkeypatch, failure):
    api, canary, rowdir, bookings = feeder_booking
    assert hasattr(api, 'undrain_ecores'), 'feature absent: owned fv1e undrain'
    requested = json.loads(bookings.read_text())
    requested['module-A']['capture']['memory_gib'] = 37
    bookings.write_text(json.dumps(requested))
    events = []
    monkeypatch.setattr(api, 'drain_ecores_before_submit', lambda memory, *a: {'host': 'sparky', 'reason': 'test-drain'} if memory > 36 else None)
    monkeypatch.setattr(api, 'undrain_ecores', lambda **kw: events.append('undrain'))
    original = api.subprocess.run
    def run(command, **kw):
        if command[:3] == ['flux', 'job', 'wait-event']:
            events.append('clean')
        return original(command, **kw)
    monkeypatch.setattr(api.subprocess, 'run', run)
    if failure == 'native':
        (rowdir / 'capture/results/podman-exit.txt').write_text('1')
    if failure == 'artifact':
        (rowdir / 'capture/results/capture/phase.json').write_text('{"completion":false}')
    if failure:
        with pytest.raises(api.FeederError, match='COUNTED_PHASE_'):
            api.run(canary, bookings)
        assert 'undrain' not in events
    else:
        api.run(canary, bookings)
        assert events == ['clean', 'clean', 'undrain', 'clean']


def test_dense_prepare_booking_and_low_urgency(feeder_booking):
    api, canary, rowdir, bookings = feeder_booking
    waits = []
    api.wait_dense_slot = lambda: waits.append(True)
    api.run(canary, bookings)
    assert len(waits) == 2
    assert all('--urgency=8' in p.read_text() for p in rowdir.glob('*/submit-after-review.sh'))
    bad = json.loads(bookings.read_text())
    bad['module-A']['prepare']['memory_gib'] = 32
    bookings.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='REQUIRES_8_GIB'):
        api.run(canary, bookings)
    bad['module-A']['prepare']['memory_gib'] = 8
    bad['module-A']['capture']['minutes'] = 60
    bookings.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='REQUIRES_15_MINUTES'):
        api.run(canary, bookings)
