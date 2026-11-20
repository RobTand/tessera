"""Check each bound case's streamed code proof."""
import copy
import inspect

import pytest

from experiments.numerics_rows import dependency_code as code, phase_artifact as streams, row


def native_code_proof(unit_id):
    return {"unit_id": unit_id, "dependency_code": {"observation_complete": True},
            "triton_specializations": [], "unresolved_kernel_origins": [],
            "arm": {"profiler_kernels": [{"name": "case-kernel"}], "preparation_kernels": [{"name": "prepare-kernel"}],
                    "loaded_modules": [{"path": "/mapped/native.so", "library_so_sha256": "a" * 64}]},
            "kernel_origins": [{"profiler_name": name, "origin": "tessera", "path": "/mapped/native.so", "sha256": "a" * 64}
                               for name in ("case-kernel", "prepare-kernel")]}


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "unit", "origin", "observer", "digest", "stream-bytes"))
def test_join_code_reader_checks_each_bound_case(tmp_path, mutation):
    assert "common" in inspect.signature(code.check_unit_code).parameters, "feature absent: streamed unit code proofs"
    pytest.importorskip("torch")
    from tests.test_numerics_rows_phases import rewrite_stream
    unit = row.new_unit({"cell": "cell", "module": "A", "m_config": {"M": 1}, "tp": {"size": 1, "rank": 0}})
    row.finish_unit(unit)
    common = {"runtime": {"image_digest": "image"}}
    proof = native_code_proof(unit["unit_id"])
    code.check_unit_code(proof, "image")
    artifact = streams.Artifact(tmp_path / "compare", "compare", common)
    scope = streams.unit_scope(unit, "wire")
    artifact.document["scopes"] = [scope]
    case = artifact.add(scope, {"M": 1}, "dense", {}, category="sweep", completion=True, code=proof)
    unit["code_stream"] = {"path": str(artifact.root / "phase.json"), "sha256": artifact.finish()}
    code.check_unit_code(unit, "image", common)
    values = [case]
    if mutation == "missing": values.clear()
    if mutation == "duplicate": values.append(copy.deepcopy(case))
    if mutation == "unit": proof["unit_id"] = ["another-unit"]
    if mutation == "origin": proof["kernel_origins"].pop()
    if mutation == "observer": proof["dependency_code"]["observation_complete"] = False
    if mutation == "digest": unit["code_stream"]["sha256"] = "b" * 64
    elif mutation == "stream-bytes":
        with (artifact.root / "cases.jsonl").open("a") as target:
            target.write("{}\n")
    else:
        rewrite_stream(artifact, "cases", values)
        from experiments.numerics_rows.common import sha256
        unit["code_stream"]["sha256"] = sha256(artifact.root / "phase.json")
    with pytest.raises(ValueError):
        code.check_unit_code(unit, "image", common)


def test_join_reads_one_code_stream_once_per_receipt(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from experiments.numerics_rows.join import join
    from tests.test_numerics_rows_receipts import joined_inputs

    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    receipt = receipts[0][1]
    receipt["canary"] = False
    receipt["tessera"].update(dict.fromkeys(("src_tree_sha256", "kernel_source_sha256", "build_provenance_sha256", "retained_source"), "source"))
    receipt["runtime"].update(vllm="vllm", torch="torch")
    receipt["inputs"]["selected_cuts"] = [unit["tp"] for unit in receipt["units"]]
    artifact = streams.Artifact(tmp_path / "compare", "compare", streams.identities(receipt))
    passed = [unit for unit in receipt["units"] if unit["role"] == "unit" and unit["admitted_by_v68"] and unit["outcome"] == "PASS"]
    assert len(passed) > 1
    for unit in passed:
        scope = streams.unit_scope(unit, "wire")
        artifact.document["scopes"].append(scope)
        for _ in range(2):
            artifact.add(scope, unit["unit_id"][4], "dense", {}, completion=True, code=copy.deepcopy(unit))
    foreign = copy.deepcopy(artifact.document["scopes"][0])
    foreign["module"] = "outside-receipt"
    artifact.document["scopes"].append(foreign)
    artifact.add(foreign, foreign["unit_key"][4], "dense", {}, completion=False, code={"code_stream": {}})
    reference = {"path": str(artifact.root / "phase.json"), "sha256": artifact.finish()}
    for unit in passed:
        unit["code_stream"] = dict(reference)
    calls, read_cases = [], streams.cases
    def counted_cases(root, document):
        calls.append(root)
        yield from read_cases(root, document)
    monkeypatch.setattr(streams, "cases", counted_cases)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    assert len(calls) == 1, f"stream read once per receipt: {len(calls)} reads"


def test_real_writer_compare_join_with_synthetic_capture(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    torch = pytest.importorskip("torch")
    from experiments.numerics_rows import cases, dense_phases as dp, scope, scale_launch
    from experiments.numerics_rows.common import BoundaryRecorder, Invalid, sha256
    from experiments.numerics_rows.gpu_time import PhaseTimes
    from experiments.numerics_rows.join import join, rule_unit_id
    from tests.test_numerics_rows_receipts import joined_inputs

    # Fixture substitutions (not production): joined_inputs patches scope.wire_geometries, scope.contract_document,
    # join.RULE_SHA and join.H1_SHA to fixture values; below, configurations force chunk_size 1. The capture is synthetic.
    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    entry = population["modules"][0]
    population["modules"] = [entry]
    role_names = ("gate_proj", "up_proj")
    entry["roles"] = [[name, 2] for name in role_names]
    configurations = cases.configurations

    def tiny_configurations(*args, **kwargs):
        return [{**config, "chunk_size": 1} for config in configurations(*args, **kwargs)]

    monkeypatch.setattr(cases, "configurations", tiny_configurations)
    monkeypatch.setattr(scope, "configurations", tiny_configurations)
    whole, = manifests[0][1]["modules"]
    for cut_manifest in whole["cuts"]:
        cut_manifest["roles"] = [{"name": name, "expert": 0} for name in role_names]
        for config in cut_manifest["m_configs"]:
            config["chunk_size"] = 1

    receipt = receipts[0][1]
    cut = {"size": 1, "rank": 0}
    unit, = [u for u in receipt["units"] if u["tp"] == cut and u["m_config"]["M"] == 1]
    unit["m_config"]["chunk_size"] = 1
    row.finish_unit(unit)
    receipt.update(canary=False, row={"name": "compare", "finalization_finished": False}, units=[unit], modules=[])
    receipt["tessera"].update(dict.fromkeys(("src_tree_sha256", "kernel_source_sha256", "build_provenance_sha256", "retained_source"), "source"))
    receipt["runtime"].update(vllm="vllm", torch="torch")
    receipt["inputs"]["selected_cuts"] = [cut]

    class Quantizer:
        gs, normal_j = 1.0, None

        def __call__(self, value):
            return float(torch.tensor(value).clamp(-448, 448).to(torch.float8_e4m3fn).float()), 1.0

    class TinyDense(dp.HostDense):
        def __init__(self):
            self.device, self.family, self.gs, self.splits = "cpu", "e4m3", 1.0, unit["m_config"]["k_split"]
            self.cols, self.rows, self.quantizer = 2, 4, Quantizer()
            self.roles = [(name, w, torch.ones(2), None) for name, w in zip(role_names, (torch.eye(2), -torch.eye(2)))]

    adapter = TinyDense()
    roles = [{"name": name, "expert": 0, "cols": adapter.cols,
              "separation": {"complete": True, "both": 2, "recipes": [{"column": 0, "values": [1]}]},
              "scale_index": cases.scale_index_manifest(w, s, "e4m3", "dense", None, None, None,
                                                        adapter.quantizer, 1, 0, [unit["m_config"]])}
             for name, w, s, _ in adapter.roles]
    module = {**whole, **whole["cuts"][0], "roles": roles, "wire_sha256": "tiny-wire", "complete": True}
    groups = [(entry, module, cut, [unit])]
    common, scopes = streams.identities(receipt), dp.scopes(groups)
    args = SimpleNamespace(out=tmp_path / "compare-job", case_ranges=dp.case_ranges(groups),
                           prepare_artifact=[], capture_artifact=[])
    recorder = BoundaryRecorder(lambda: None, [])
    proof = native_code_proof(unit["unit_id"])
    weights = {name: (w, s) for name, w, s, _ in adapter.roles}
    for part in ("gate", "diagnostics"):
        key = f"tp1-{part}-" + streams.json_digest([entry["name"], 1, part])
        attempt = tmp_path / "units" / key / "attempts" / "1"
        start, count = dp.unit_range(args, groups, part)
        prepare = streams.Artifact(attempt / "prepare", "prepare", common)
        prepare.document.update(scopes=scopes, case_set=part, ordinal_start=start, expected_cases=count)
        adapter.case_writer = dp.CaseWriter(prepare, SimpleNamespace(record=lambda *a, **k: None), {entry["name"]: module["wire_sha256"]})
        previous = copy.deepcopy(receipt)
        prepared_unit, = previous["units"]
        if part == "gate":
            row.run_dense_case(adapter, prepared_unit, [], recorder, category="profile")
            row.dense_cases(adapter, prepared_unit, module, recorder)
            scale_launch.check(adapter, prepared_unit, module, recorder)
        else:
            row.dense_diagnostics(adapter, prepared_unit, recorder)

        capture = streams.Artifact(attempt / "capture", "capture", common)
        capture.document.update(scopes=scopes, case_set=part, ordinal_start=start, prepare_digest=prepare.finish(receipt=previous))
        for case in streams.cases(prepare.root, prepare.document):
            tensors = {k: streams.payload(prepare.root, v) for k, v in case["tensors"].items()}
            q, sf = tensors["q"], tensors["sf"]
            outputs = {"quantizer_q": q, "quantizer_sf": sf}
            if case["kind"] == "scale":
                w, _ = weights[case["role"]]
                outputs["output"] = ((q.float() @ w.T) * tensors["coded_sf"] * tensors["coded_values"]).bfloat16()
            else:
                outputs["output"] = torch.cat([((q.float() @ w.T) * sf * s).bfloat16() for w, s in weights.values()], dim=1)
                outputs["staged"] = outputs["output"].clone()
            capture.record("cases", {**case, "completion": True, "code": proof, "status": dp.case_status(unit),
                "boundaries": [{"kind": "quantizer", "sync_before": True, "sync_after": True}] * 2,
                "profiler_names": ["case-kernel"], "kernel_origins": proof["kernel_origins"][:1],
                "tensors": {k: capture.tensor(v) for k, v in sorted(outputs.items())}})
        capture.finish(receipt={**receipt, "modules": [{"name": entry["name"], "tp": cut}]})
        args.prepare_artifact.append(prepare.root)
        args.capture_artifact.append(capture.root)

    arrays = list(streams.records(args.prepare_artifact[0], "arrays"))
    assert [(a["array_index"], a["record"]["launches"]) for a in arrays] == [(0, 2), (1, 0), (2, 2), (3, 0)]
    try:
        dp.host_compare(args, receipt, PhaseTimes({}), groups)
    except Invalid as error:
        raise AssertionError(f"REAL_WRITER_COMPARE: {error}") from error
    assert receipt["row"]["finalization_finished"] and receipt["row"]["outcome"] == unit["outcome"] == "PASS", unit["reasons"]
    compared, compare_sha = streams.read(args.out / "compare", "compare", common, scopes)
    assert list(streams.records(args.out / "compare", "arrays", compared)) == [arrays[0], arrays[2]]
    stream = {"path": str(args.out / "compare/phase.json"), "sha256": compare_sha}
    assert unit["code_stream"] == receipt["phase_artifacts"]["compare"] == stream
    receipt_path = args.out / "numerics_row.json"
    receipt_sha = sha256(receipt_path)
    joined = join(population, digest, manifests[:1], [(receipt_sha, json.loads(receipt_path.read_text()))])
    covered, = [u for u in joined["units"] if u["unit_id"] == rule_unit_id(unit)]
    assert covered["outcome"] == "PASS" and covered["modules"][0]["receipt_sha256s"] == [receipt_sha]
    assert joined["verdict"] == "UNATTESTED"
