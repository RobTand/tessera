"""Check dense fault controls with streamed unit artifacts in tmp_path."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

try:
    import numerics_dense_controls as controls
except ModuleNotFoundError as error:
    if error.name != "numerics_dense_controls":
        raise
    controls = None

from experiments.numerics_rows import dense_phases as dp, row, scale_launch, scope as source_scope
from experiments.numerics_rows.common import Invalid, sha256, write_json
from experiments.numerics_rows.launch_state import json_digest
from experiments.numerics_rows.scale_index import passes
from experiments.numerics_rows.population import select_modules
from test_numerics_rows_receipts import joined_inputs


def scale_plan():
    return {**passes(2, 0, 1), "array": {"N": 2, "mode": 2}, "scale": "row", "reads": 1, "reads_covered": 1,
            "read_cases": [{"read": [1], "scale": "row", "input": 1.0, "column": 0, "input_row": 0, "output_row": 1,
                            "g": 1.0, "u": 0.0, "g_scale": 1.0, "u_scale": 1.0, "a_scale": 1.0}]}


def control_pair(tmp_path, monkeypatch, *, canary=False):
    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    document = manifests[0][1]
    document["launch_selectors"] = {}
    module = document["modules"][0]
    module["wire_sha256"] = "wire"
    population["modules"][0]["roles"] = [["gate_proj", 64], ["up_proj", 64]]
    for cut in module["cuts"]:
        cut["roles"] = [{"name": name, "expert": 0, "cut_axis": 0,
                         "cut_range": [0, 128 // cut["tp"]["size"]], "scale_index": [scale_plan()]}
                        for name in ("gate_proj", "up_proj")]
    q1 = tmp_path / "Q1.json"
    write_json(q1, {})
    document["q1_table_sha256"] = sha256(q1)
    manifest = tmp_path / "C1.json"
    write_json(manifest, document)
    receipt = copy.deepcopy(receipts[0][1])
    receipt.update(canary=canary, modules=[])
    receipt["row"]["utc_start"] = "2026-11-17T00:00:00Z"
    receipt["inputs"].update(case_manifest_sha256=sha256(manifest), q1_table_sha256=sha256(q1))
    receipt["inputs"]["selected_cuts"] = [{"size": n, "rank": r} for n, r in ((1, 0), (2, 0), (2, 1))]
    receipt["runtime"].update(vllm="fixture", torch="fixture", launch_selectors={})
    receipt["tessera"].update(src_tree_sha256="source", kernel_source_sha256="kernel",
                            build_provenance_sha256="build", retained_source={"commit": "source"})
    groups = row.planned_units(source_scope.contract_document()[0], receipt["runtime"]["image_digest"],
                              select_modules(population, receipt["inputs"]["selected_modules"]), document)
    template = receipts[0][1]["units"][0]
    for _, selected_module, _, units in groups:
        for unit in units:
            unit["launch_state_sha256"] = receipt["runtime"]["launch_state_sha256"]
            unit.update({k: copy.deepcopy(v) for k, v in template.items() if k.startswith("dependency_math")})
            unit["tp"].update(cut_axis=[r["cut_axis"] for r in selected_module["roles"]],
                              cut_range=[r["cut_range"] for r in selected_module["roles"]])
    receipt["units"] = [row.new_unit(u) for *_, units in groups for u in units]
    prepared_roots, captured_roots, offset = [], [], 0
    x, sf = torch.ones(1, 2, dtype=torch.bfloat16), torch.ones(1, 1)
    for part in ("gate", "diagnostics"):
        for selected in dp.phase_units(groups):
            cuts = [g[2] for g in selected]
            local = copy.deepcopy({**receipt, "inputs": {**receipt["inputs"], "selected_cuts": cuts},
                                   "units": [u for g in selected for u in g[3]]})
            for unit in local["units"]:
                unit.update(completed=True, triton_specializations=[], kernel_origins=[], unresolved_kernel_origins=[],
                            dependency_code={"observation_complete": True})
                unit["exact"]["separation"]["records"] = [{"both": 1}]
                unit["exact"]["sweep"]["elements"] = 2
                unit["exact"]["scale_index"]["arrays"] = [{"launches": 2, "mismatches": 0}]
                unit["exact"]["chunk_pairs"]["x_elements"] = 1
            common = dp.identities(local)
            key = f'tp{cuts[0]["size"]}-{part}-' + json_digest([module["name"], cuts[0]["size"], part])
            attempt = tmp_path / "original/units" / key / "attempts/1"
            prepared = dp.Artifact(attempt / "prepare", "prepare", common)
            captured = dp.Artifact(attempt / "capture", "capture", common)
            for artifact in (prepared, captured):
                artifact.document.update(scopes=dp.scopes(selected), case_set=part, ordinal_start=offset)
            for unit in local["units"]:
                scope = dp.unit_scope(unit, "wire")
                categories = ("diagnostic",) if part == "diagnostics" else ("sweep", "chunk_pairs", "scale_index", "scale_index")
                for category_index, category in enumerate(categories):
                    tensors = {"input": x, "q": x * 2, "sf": sf, "reference": x * 2,
                               "finite_value": x.double() * 10, "finite_held": x.bool(),
                               "x_mask": torch.tensor([[True, category != "chunk_pairs"]]),
                               "diagnostic_reference": x.double() * 2, "products_abs": x.double()}
                    if category == "scale_index":
                        tensors.update(reference=torch.tensor([2], dtype=x.dtype), coded_values=torch.tensor([1.0, 2.0]))
                        prepared.record("arrays", {"scope": scope, "array_index": category_index - 2,
                                                   "record": {**scale_plan(), "launches": 1, "mismatches": 0}})
                    case = prepared.add(scope, row.unit_identity(unit)[4], "scale" if category == "scale_index" else "dense",
                                        tensors, category=category, distribution="gauss", seed=1, output_index=[0, 0],
                                        array_index=category_index - 2 if category == "scale_index" else None,
                                        role="up_proj" if category_index == 3 else "gate_proj",
                                        scale="row", read=[1], coding=scale_plan()["coding"][0])
                    proof = {**unit, "unit_id": row.unit_identity(unit)}
                    captured.add(scope, case["configuration"], case["kind"],
                                 {"output": x * 2, "staged": x * 2, "quantizer_q": x * 2, "quantizer_sf": sf},
                                 id=case["id"], category=category, completion=True, code=proof,
                                 boundaries=[{"kind": "quantizer", "sync_before": True, "sync_after": True}] * 2,
                                 profiler_names=["kernel"], kernel_origins=[{"profiler_name": "kernel"}])
            pd = prepared.finish(receipt=copy.deepcopy(local))
            for unit in local["units"]:
                unit["tp"].update(has_init=False, start_state_sha256="0" * 64)
            captured.finish(prepare_digest=pd, receipt=local)
            offset += prepared.document["streams"]["cases"]["count"]
            prepared_roots.append(prepared.root)
            captured_roots.append(captured.root)
    args = SimpleNamespace(prepare=prepared_roots, capture=captured_roots, manifest=manifest,
                           contract=None, scratch=tmp_path / "replays", only=None, group=None, assemble=None,
                           out=tmp_path / "report.json", q1=q1, population=tmp_path / "population.json")
    monkeypatch.setattr(controls, "experiments_tree", lambda *args: "tree")
    monkeypatch.setattr(controls, "load_signed_population", lambda path: (population, digest))
    runner = controls.Controls(args)
    runner.baseline = controls.replay(args.prepare, args.capture, args.scratch / "baseline",
                                      groups=runner.groups, receipt=runner.receipt)
    assert runner.baseline["row"]["outcome"] == "PASS"
    return runner


def scale_probe(monkeypatch):
    unit = row.new_unit({"m_config": {"M": 1, "k_split": 1}})
    adapter = SimpleNamespace(device="cpu", family="e4m3", gs=1.0, cols=2, rows=2,
                              roles=[("linear", torch.ones(2, 2), torch.ones(2), None)],
                              quantized=lambda x, _: (x, torch.ones(1, 1), x.double(), torch.ones(1)))
    def forbidden(*args, **kwargs):
        pytest.fail("a CPU control attempted a native launch")
    monkeypatch.setattr(scale_launch, "dense_launch", forbidden)
    return adapter, unit, scale_plan()


def test_replay_capture_tp_scope(tmp_path, monkeypatch):
    assert hasattr(controls, "module_inputs"), "feature absent: streamed unit fault controls"
    runner = control_pair(tmp_path, monkeypatch)
    assert len(runner.inputs) == 4
    assert [len(selected) for selected in dp.phase_units(runner.groups)] == [1, 2]
    assert {case["role"] for case, _ in runner.pairs if case["kind"] == "scale"} == {"gate_proj", "up_proj"}
    assert all({"has_init", "start_state_sha256"} <= u["tp"].keys() for u in runner.baseline["units"])
    assert runner.baseline["row"]["outcome"] == "PASS"
    assert len(runner.baseline["attempts"]["listing"]) == 4


@pytest.mark.parametrize("category", ["sweep", "chunk_pairs", "diagnostic", "scale_index"])
def test_dense_controls_detect_faults_and_record_unchecked_scope(tmp_path, monkeypatch, category):
    assert hasattr(controls, "module_inputs"), "feature absent: streamed unit fault controls"
    runner = control_pair(tmp_path, monkeypatch)
    before = controls.artifact_hashes(*controls.source_roots(runner.args))
    runner.output_controls(category)
    detected = [r for r in runner.records if r["control_result"] == "DETECTED"]
    assert detected and all(r["row_outcome"] == r["join_outcome"] == "FAIL" and r["exit_status"] == 3 for r in detected)
    outside = runner.records[-1]
    assert outside["control_result"] == ("NOT_APPLICABLE" if category == "sweep" else "NOT_CHECKED")
    assert outside["not_checked_elements"] == (0 if category == "sweep" else 1)
    if category == "diagnostic":
        assert outside["fault_indices"][0] != detected[0]["fault_indices"][0]
    assert controls.artifact_hashes(*controls.source_roots(runner.args)) == before
    receipt = {**runner.baseline, "canary": True}
    assert runner.joined(receipt, receipt["units"][0]) == "CANARY_REFUSED"


def test_dense_control_changes_only_selected_case_and_fields(tmp_path, monkeypatch):
    assert hasattr(controls, "module_inputs"), "feature absent: streamed unit fault controls"
    runner = control_pair(tmp_path, monkeypatch)
    case, _ = next(runner.pairs)
    original_compare, roots = dp.compare_case, []
    def compare_on_copy(case, observed, prepare_root, capture_root, unit):
        roots.extend((prepare_root, capture_root))
        return original_compare(case, observed, prepare_root, capture_root, unit)
    monkeypatch.setattr(dp, "compare_case", compare_on_copy)
    result = controls.replay(runner.args.prepare, runner.args.capture, runner.args.scratch / "one-case",
                            [(case["id"], field, (0, 0), 0.0) for field in ("output", "staged")],
                            groups=runner.groups, receipt=runner.receipt)
    assert sum(u["exact"]["sweep"]["mismatches"] for u in result["units"]) == 1
    assert sum(bool(u["findings"]) for u in result["units"]) == 1
    assert roots and all(root.is_relative_to(runner.args.scratch / "one-case/artifacts/units") for root in roots)
    assert all((root / "cases.jsonl").is_file() for root in roots)


@pytest.mark.parametrize("plant,expected", [("none", "PASS"), ("incomplete", "UNATTESTED"), ("domain", "INVALID")])
def test_dense_controls_recompute_and_propagate_to_compare_and_join(tmp_path, monkeypatch, plant, expected):
    assert hasattr(controls, "module_inputs"), "feature absent: streamed unit fault controls"
    runner = control_pair(tmp_path, monkeypatch)
    adapter, _, plan = scale_probe(monkeypatch)
    case, _ = next((c, o) for c, o in runner.pairs if c["category"] == "scale_index")
    unit = controls.selected_unit(runner.baseline, case["scope"])
    if plant == "incomplete":
        plan["reads_covered"] = 0
    elif plant == "domain":
        plan["array"]["N"] = 3
    result, verified = controls.plan_probe(adapter, unit, {"structure": "dense"}, {"name": "linear", "expert": 0}, plan)
    assert verified == {"reads": 1, "reads_covered": 1, "uncovered_reads": []}
    assert bool(result["invalid"]) == (plant == "domain")
    assert bool(result["unattested"]) == (plant == "incomplete")
    check = (("invalid", "C1 scale case has the wrong read tuple or array size") if plant == "domain" else
             ("unattested", "incomplete scale-index coverage before the freeze") if plant == "incomplete" else None)
    runner.run(plant, case, expected, refusal=(case["scope"], result), check=check)
    assert runner.records[-1]["row_outcome"] == runner.records[-1]["join_outcome"] == expected
    assert runner.records[-1]["control_result"] == "MAPPING_ONLY"


def test_wrong_detecting_check_fails(tmp_path, monkeypatch):
    assert hasattr(controls, "module_inputs"), "feature absent: streamed unit fault controls"
    runner = control_pair(tmp_path, monkeypatch)
    case, _ = next((c, o) for c, o in runner.pairs if c["category"] == "diagnostic")
    with pytest.raises(Invalid, match="control was not flagged"):
        runner.run("wrong-check", case, "FAIL", faults=[(case["id"], "output", (0, 0), 0.0)],
                   check=("findings", case["id"] + ": exact output differs"))


def test_one_named_control(tmp_path, monkeypatch):
    assert hasattr(controls, "module_inputs"), "feature absent: streamed unit fault controls"
    runner = control_pair(tmp_path, monkeypatch)
    runner.args.only = "chunk_pairs-outside"
    runner.output_controls("chunk_pairs")
    assert [r["name"] for r in runner.records] == ["chunk_pairs-outside"]
    assert runner.records[0]["control_result"] == "NOT_CHECKED"


@pytest.mark.parametrize("change", ["tracked", "untracked", "head"])
def test_dense_controls_bind_actual_experiments_tree(tmp_path, change):
    assert hasattr(controls, "module_inputs"), "feature absent: streamed unit fault controls"
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()
    git("init", "-q")
    (tmp_path / "experiments").mkdir()
    original = tmp_path / "experiments/original.py"
    original.write_text("original = True\n")
    git("add", "experiments")
    git("-c", "user.name=CPU test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    head = git("rev-parse", "HEAD")
    assert controls.experiments_tree(tmp_path, head, tmp_path / "scratch") == git("rev-parse", "HEAD:experiments")
    if change == "head":
        with pytest.raises(Invalid, match="checkout differs"):
            controls.experiments_tree(tmp_path, "0" * 40, tmp_path / "scratch")
    else:
        (original if change == "tracked" else tmp_path / "experiments/added.py").write_text("changed = True\n")
        with pytest.raises(Invalid, match="experiments tree differs"):
            controls.experiments_tree(tmp_path, head, tmp_path / "scratch")


@pytest.mark.parametrize("mutation", ["no-effect", "extra", "same-case"])
def test_combined_replay_requires_every_plant(tmp_path, monkeypatch, mutation):
    assert hasattr(controls, "module_inputs"), "feature absent: streamed unit fault controls"
    runner = control_pair(tmp_path, monkeypatch)
    runner.plan()
    plants = [p for p in runner.plants if p["name"] in controls.REPLAY_GROUPS["R2"]]
    assert len(plants) == len({p["case"]["id"] for p in plants}) == 4
    if mutation == "same-case":
        plants[-1] = plants[0]
    else:
        plant = plants[0]
        if mutation == "no-effect":
            observed = next(o for c, o in runner.pairs if c["id"] == plant["case"]["id"])
            plant["faults"] = [(case_id, field, index, float(runner.tensor(plant["case"], observed["tensors"][field], captured=True)[index]))
                               for case_id, field, index, _ in plant["faults"]]
        else:
            plant["faults"] = plant["faults"][:1]
    with pytest.raises(Invalid, match="distinct cases" if mutation == "same-case" else "control was not flagged"):
        runner.run_plants("R2", plants, "FAIL")
    if mutation == "no-effect":
        assert all(r["control_result"] == "MISSED" for r in runner.records)


@pytest.mark.parametrize("canary", [True, False], ids=["canary", "counted"])
def test_independent_groups_and_assembly(tmp_path, monkeypatch, canary):
    assert hasattr(controls, "module_inputs"), "feature absent: streamed unit fault controls"
    runner = control_pair(tmp_path, monkeypatch, canary=canary)
    excluded = copy.deepcopy(runner.receipt)
    for unit in excluded["units"]:
        unit["role"] = "observation"
    assert controls.eligible_cases(runner.pairs, excluded) == set()
    for unit in excluded["units"]:
        unit.update(role="unit", admitted_by_v68=False)
    assert controls.eligible_cases(runner.pairs, excluded) == set()
    adapter, _, _ = scale_probe(monkeypatch)
    adapter.roles = [(name, *adapter.roles[0][1:]) for name in ("gate_proj", "up_proj")]
    monkeypatch.setattr(dp, "HostDense", lambda *args: adapter)
    monkeypatch.setattr(controls, "experiments_tree", lambda *args: "tree")
    original_replay, launches = controls.replay, []
    def count_replay(*args, **kwargs):
        assert len(json.loads(runner.args.out.read_text())["control_table"]) == 13
        launches.append(args[2].name)
        return original_replay(*args, **kwargs)
    monkeypatch.setattr(controls, "replay", count_replay)
    reports = []
    for name in reversed(controls.REPLAY_GROUPS):
        runner.args.group = name
        argv = ["controls", "--group", name]
        for field in ("prepare", "capture", "manifest", "population", "q1", "contract", "scratch", "out"):
            value = getattr(runner.args, field)
            if field == "scratch":
                value = value / name
            argv.extend(["--" + field, *(map(str, value) if isinstance(value, list) else [str(value)])])
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit) as stopped:
            controls.main()
        assert stopped.value.code == 0
        reports.append(json.loads(runner.args.out.read_text()))
        assert reports[-1]["group"] == name and reports[-1]["counting_allowed"] is False
        assert all(r["valid"] for r in reports[-1]["controls"])
        assert reports[-1]["artifact_hashes_before"] == reports[-1]["artifact_hashes_after"]
        assert reports[-1]["experiments_tree_before"] == reports[-1]["experiments_tree_after"] == "tree"
    report_paths = [runner.args.out.with_name(report["group"] + ".json") for report in reports]
    for report_path, report in zip(report_paths, reports):
        write_json(report_path, report)
    monkeypatch.setattr("sys.argv", ["controls", "--assemble", *map(str, report_paths), *argv[3:]])
    with pytest.raises(SystemExit) as stopped:
        controls.main()
    assert stopped.value.code == 0
    assembled = json.loads(runner.args.out.read_text())
    assert assembled["verdict"] == "PASS" and assembled["counting_allowed"] is (not canary)
    runner.plan()
    assert runner.assemble(reports) is (not canary)
    assert len(launches) == (5 if canary else 7)
    assert set(launches) == set(controls.REPLAY_GROUPS) | (set() if canary else {"incomplete-C1", "wrong-C1-domain"})
    assert len(runner.records) == 13
    assert all(r["control_result"] == ("DEFERRED_TO_COUNTED_ROW" if canary else "MAPPING_ONLY")
               for r in runner.records if r["name"] in controls.REPLAY_GROUPS["R0"])
    r2 = next(report for report in reports if report["group"] == "R2")
    target = next(target for target in r2["controls"][0]["targets"] if target["stream_reasons"]["findings"])
    assert len(target["stream_reasons"]["findings"]) == 4 and len(target["reasons"]["findings"]) == 1
    damaged = copy.deepcopy(reports)
    next(report for report in damaged if report["group"] == "R0")["baseline"]["row"]["exit_status"] = 3
    with pytest.raises(Invalid, match="PASS row with rc 0"):
        runner.assemble(damaged)
    damaged = copy.deepcopy(reports)
    damaged[0]["artifact_hashes_after"] = {}
    with pytest.raises(Invalid, match="invalid group report"):
        runner.assemble(damaged)
    with pytest.raises(Invalid, match="one report for each"):
        runner.assemble(reports[:-1])
    damaged = copy.deepcopy(reports)
    damaged[0]["experiments_tree_after"] = "other"
    with pytest.raises(Invalid, match="invalid group report"):
        runner.assemble(damaged)


def test_combined_only_keeps_single_fallback(tmp_path, monkeypatch):
    assert hasattr(controls, "module_inputs"), "feature absent: streamed unit fault controls"
    runner = control_pair(tmp_path, monkeypatch)
    runner.args.only = "wrong-scale-index"
    runner.plan()
    plant, = [p for p in runner.plants if p["name"] == runner.args.only]
    runner.run_plants(plant["name"], [plant], plant["expected"])
    assert [r["name"] for r in runner.records] == ["wrong-scale-index"]
    assert runner.records[0]["control_result"] == "DETECTED"


@pytest.mark.parametrize("fault", ["missing", "duplicate", "identity", "stream", "payload"])
def test_streamed_controls_refuse_bad_artifacts(tmp_path, monkeypatch, fault):
    assert hasattr(controls, "module_inputs"), "feature absent: streamed unit fault controls"
    runner = control_pair(tmp_path, monkeypatch)
    if fault == "missing":
        runner.args.prepare.pop()
        runner.args.capture.pop()
    elif fault == "duplicate":
        runner.args.prepare.append(runner.args.prepare[0])
        runner.args.capture.append(runner.args.capture[0])
    else:
        path = runner.args.capture[0] / "phase.json"
        document = json.loads(path.read_text())
        if fault == "identity":
            document["identities"]["tool"]["commit"] = "other"
            write_json(path, document)
        elif fault == "stream":
            with (path.parent / "cases.jsonl").open("a") as stream:
                stream.write("{}\n")
        else:
            case, observed = next(runner.pairs)
            path = runner.args.capture[0] / observed["tensors"]["output"]["file"]
            path.write_bytes(b"wrong")
    reason = {"missing": "union exactly once", "duplicate": "union exactly once",
              "identity": "identities, module or cut", "stream": "ordered case", "payload": "file digest differs"}[fault]
    with pytest.raises(Invalid, match=reason):
        controls.Controls(runner.args)


@pytest.mark.parametrize("field", ["scratch", "out"])
def test_control_paths_refuse_original_root(tmp_path, monkeypatch, field):
    assert hasattr(controls, "module_inputs"), "feature absent: streamed unit fault controls"
    runner = control_pair(tmp_path, monkeypatch)
    assert controls.check_paths(runner.args)
    before = controls.artifact_hashes(*controls.source_roots(runner.args))
    with pytest.raises(Invalid, match="outside original artifacts"):
        controls.replay(runner.args.prepare, runner.args.capture, tmp_path / "original/forbidden-replay",
                        groups=runner.groups, receipt=runner.receipt)
    assert controls.artifact_hashes(*controls.source_roots(runner.args)) == before
    assert not (tmp_path / "original/forbidden-replay").exists()
    setattr(runner.args, field, tmp_path / "original/control-output")
    with pytest.raises(Invalid, match="outside original artifacts"):
        controls.check_paths(runner.args)


def test_controls_copy_and_keep_all_attempts(tmp_path, monkeypatch):
    assert hasattr(controls, "module_inputs"), "feature absent: streamed unit fault controls"
    runner = control_pair(tmp_path, monkeypatch, canary=True)
    attempt = runner.args.prepare[0].parent
    shutil.copytree(attempt, attempt.with_name("2"))
    rebuilt = controls.Controls(runner.args)
    assert rebuilt.inputs[0][0].parent.name == "2"
    before = controls.artifact_hashes(*controls.source_roots(rebuilt.args))
    rebuilt.plan()
    report = rebuilt.group("R0")
    listing = report["baseline"]["attempts"]["listing"]
    assert len(listing) == 5 and {entry["attempt"] for entry in listing} == {1, 2}
    assert report["artifact_hashes_before"] == report["artifact_hashes_after"] == before
    for entry in listing:
        assert (rebuilt.args.scratch / "R0/artifacts/units" / entry["unit"] / "attempts" / str(entry["attempt"])).is_dir()
    partial = attempt / "capture/numerics_row.json"
    failed = json.loads((attempt / "capture/phase.json").read_text())["receipt"]
    failed["units"][0]["findings"] = ["retained FAIL"]
    write_json(partial, failed)
    with_fail = controls.artifact_hashes(*controls.source_roots(rebuilt.args))
    assert with_fail != before
    result = controls.replay(rebuilt.args.prepare, rebuilt.args.capture, rebuilt.args.scratch / "prior-fail",
                             groups=rebuilt.groups, receipt=rebuilt.receipt)
    assert result["row"]["outcome"] == "FAIL"
    assert any(unit.get("failed_attempts") for unit in result["units"])
    assert controls.artifact_hashes(*controls.source_roots(rebuilt.args)) == with_fail


@pytest.mark.parametrize("field,value", [("role", "observation"), ("admitted_by_v68", False)])
def test_control_eligibility_uses_capture_receipts(tmp_path, monkeypatch, field, value):
    assert hasattr(controls, "module_inputs"), "feature absent: streamed unit fault controls"
    runner = control_pair(tmp_path, monkeypatch)
    first, _ = next(runner.pairs)
    assert runner.eligible(first)
    for root in runner.args.capture:
        path = root / "phase.json"
        document = json.loads(path.read_text())
        for unit in document["receipt"]["units"]:
            if row.unit_identity(unit) == first["scope"]["unit_key"]:
                unit[field] = value
        write_json(path, document)
    rebuilt = controls.Controls(runner.args)
    assert not rebuilt.eligible(first)
    rebuilt.plan()
    assert all(plant["case"]["scope"] != first["scope"] for plant in rebuilt.plants)
