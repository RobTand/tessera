"""Check reviewed unit templates without a Flux client or numerical work."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

from experiments.numerics_rows import phase_instance, prepare_split
from experiments.numerics_rows.common import Invalid


def templates(root, canary=False, instance_lists=None, module="module-A"):
    root.mkdir()
    source = Path(prepare_split.__file__).parent
    for name in ("phase_instance.py", "feeder-settings.json"):
        shutil.copyfile(source / name, root / {"phase_instance.py": "instance.py"}.get(name, name))
    if instance_lists is not None:
        settings = json.loads((root / "feeder-settings.json").read_text())
        settings["template_instances"] = instance_lists
        (root / "feeder-settings.json").write_text(json.dumps(settings))
    config = {"module": module, "tool_commit": "a" * 40, "canary": canary,
              "paths": {"source": str(root / "source")}, "staged_population": str(root / "population.json")}
    prepare_split.write_templates(root, config, root / "repository.git")
    prepare_split.bind_templates(root, config)
    return config, [root / "templates" / unit["key"] / unit["phase"] for unit in config["templates"]]


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() if p.is_file() else None for p in root.rglob("*")}


def refuse(script, arguments, text, root):
    before = snapshot(root)
    result = subprocess.run(["bash", str(script), *map(str, arguments)], text=True, capture_output=True)
    assert result.returncode == 3, result.stderr
    assert text in result.stderr
    assert snapshot(root) == before


def prepared(attempt):
    directory = attempt / "prepare"
    directory.mkdir()
    raw = json.dumps({"schema": "tessera.dense_row_phase.v2", "phase": "prepare", "completion": True}).encode()
    (directory / "phase.json").write_bytes(raw)
    (directory / "completion.sha256").write_text(hashlib.sha256(raw).hexdigest() + "  phase.json\n")


def job_world(root, fault=""):
    tools, scratch, calls = root / "tools", root / "scratch", root / "calls"
    tools.mkdir()
    scratch.mkdir()
    scripts = {
        "git": '[[ "$*" != *rev-parse* ]] || printf "%s\\n" ' + "a" * 40,
        "python3": f'''if [[ "$1" == */cdi_env.py ]]; then shift 6; exec "$@"; fi
if [[ "$*" == *--verify* ]]; then step=verify; elif [[ "$2" == check ]]; then step=instance; else exit 0; fi
printf '%s\\n' "$step" >> {shlex.quote(str(calls))}
[[ "$step" != {shlex.quote(fault)} ]] || {{ echo "refused $step" >&2; exit 3; }}
[[ "$step" != instance ]] || printf '{{"checked":true}}\\n'
''',
        "podman": '[[ "$1" != run ]] || exit 9',
    }
    for name, body in scripts.items():
        path = tools / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o700)
    return {"PATH": str(tools) + os.pathsep + os.defpath, "FLUX_SCRATCH": str(scratch), "FLUX_JOB_ID": "unit-job"}, calls


@pytest.mark.parametrize("fault", ["verify", "instance", ""])
def test_job_checks_documents_before_claim(tmp_path, fault):
    assert (Path(prepare_split.__file__).parent / "templates/job.sh").read_text().index("--verify @ROOT@") < (Path(prepare_split.__file__).parent / "templates/job.sh").read_text().index('mkdir "$attempt/@PHASE@"'), "feature absent: document checks precede the phase claim"
    _, directories = templates(tmp_path / "module")
    directory = next(p for p in directories if p.name == "prepare")
    attempt = tmp_path / "module/units" / directory.parent.name / "attempts/1"
    attempt.mkdir(parents=True)
    environment, calls = job_world(tmp_path, fault)
    before = snapshot(tmp_path / "module")
    result = subprocess.run([str(directory / "job.sh"), str(attempt)], env=environment, text=True, capture_output=True)
    assert result.returncode == (3 if fault else 9)
    assert calls.read_text().splitlines() == (["verify"] if fault == "verify" else ["verify", "instance"])
    assert not any((tmp_path / "scratch").iterdir())
    if fault:
        assert "refused " + fault in result.stderr
        assert snapshot(tmp_path / "module") == before
    else:
        assert json.loads((attempt / "prepare/instance.json").read_text()) == {"checked": True}


def test_every_template_uses_cdi_replay_or_gpu_device(tmp_path):
    _, directories = templates(tmp_path / "module")
    for directory in directories:
        launch, = [line for line in (directory / "job.sh").read_text().splitlines() if line.startswith("timeout ")]
        replay = "/experiments/numerics_rows/cdi_env.py" in launch
        device = "--device nvidia.com/gpu=all" in launch
        if directory.name == "capture":
            assert device and not replay, "capture must request the CDI GPU device"
        else:
            assert replay and not device, f"{directory.name} lacks CPU CDI replay"


def test_templates_cover_units_phases_and_manifest(tmp_path):
    assert hasattr(prepare_split, "write_templates"), "feature absent: one reviewed template per unit and phase"
    config, directories = templates(tmp_path / "module")
    assert len(directories) == 9
    assert len(config["unit_keys"]) == len(set(config["unit_keys"])) == 4
    for size in (1, 2):
        for part in ("gate", "diagnostics"):
            key, = [k for k in config["unit_keys"] if k.startswith(f"tp{size}-{part}-")]
            for phase in ("prepare", "capture"):
                directory = tmp_path / "module/templates" / key / phase
                payload = shlex.split((directory / "payload.sh").read_text().split("exec ", 1)[1])
                assert payload[payload.index("--tp-size") + 1] == str(size)
                assert payload[payload.index("--case-set") + 1] == part
                assert "--tp-rank" not in payload
                assert payload[payload.index("--out") + 1] == "$attempt"
                assert "--phase-local-output" in payload
    for directory in directories:
        manifest = (directory / "template.sha256").read_text()
        assert "  payload.sh\n" in manifest
        assert all("  " + str(tmp_path / "module" / name) + "\n" in manifest
                   for name in ("instance.py", "campaign.json", "feeder-settings.json"))
        assert "  job.sh\n" in manifest, "template manifest omits job.sh"
        assert "  table.json\n" not in manifest
        result = subprocess.run(["sha256sum", "-c", "template.sha256"], cwd=directory, capture_output=True)
        assert result.returncode == 0, result.stderr
    assert not (tmp_path / "module/units").exists()


@pytest.mark.parametrize("phase", ("prepare", "capture", "compare"))
@pytest.mark.parametrize("count", (0, 2))
def test_template_refuses_argument_count(tmp_path, phase, count):
    assert hasattr(prepare_split, "write_templates"), "feature absent: template requires exactly one argument"
    _, directories = templates(tmp_path / "module")
    script = next(p for p in directories if p.name == phase) / "job.sh"
    refuse(script, ["unused"] * count, "require one attempt argument", tmp_path)


@pytest.mark.parametrize("mutation", ("module", "unit", "zero", "symlink", "parent"))
@pytest.mark.parametrize("phase", ("prepare", "capture", "compare"))
def test_template_refuses_other_paths_and_attempt_ids(tmp_path, mutation, phase):
    assert hasattr(prepare_split, "write_templates"), "feature absent: template fixes module unit and canonical attempt"
    config, directories = templates(tmp_path / "module")
    directory = next(p for p in directories if p.name == phase)
    attempt = tmp_path / "module/units" / directory.parent.name / "attempts/1"
    attempt.mkdir(parents=True)
    reason = "attempt path differs from fixed module and unit"
    if mutation == "module":
        attempt = tmp_path / "other/units" / directory.parent.name / "attempts/1"
    elif mutation == "unit":
        attempt = attempt.parents[2] / config["unit_keys"][1] / "attempts/1"
    elif mutation == "zero":
        attempt = attempt.with_name("0")
    elif mutation == "symlink":
        target = tmp_path / "target"
        target.mkdir()
        attempt.rmdir()
        attempt.symlink_to(target, target_is_directory=True)
        reason = "attempt path is not canonical"
    else:
        attempt = str(attempt) + "/../1"
    refuse(directory / "job.sh", [attempt], reason, tmp_path)


@pytest.mark.parametrize("phase", ("prepare", "capture", "compare"))
def test_template_refuses_existing_phase_claim(tmp_path, phase):
    assert hasattr(prepare_split, "write_templates"), "feature absent: template claims phase with one atomic mkdir"
    _, directories = templates(tmp_path / "module")
    directory = next(p for p in directories if p.name == phase)
    attempt = tmp_path / "module/units" / directory.parent.name / "attempts/1"
    attempt.mkdir(parents=True)
    if phase == "capture":
        prepared(attempt)
    (attempt / phase).mkdir()
    environment, _ = job_world(tmp_path)
    before = snapshot(tmp_path / "module")
    result = subprocess.run([str(directory / "job.sh"), str(attempt)], env=environment, text=True, capture_output=True)
    assert result.returncode == 3 and "phase claim already exists or failed" in result.stderr
    assert snapshot(tmp_path / "module") == before
    script = (directory / "job.sh").read_text()
    assert script.count('mkdir "$attempt/' + phase + '"') == 1
    assert 'mkdir -p "$attempt/' not in script


@pytest.mark.parametrize("state", ("missing", "incomplete", "no-hashes", "bad-hash"))
def test_capture_template_requires_complete_prepare(tmp_path, state):
    assert hasattr(prepare_split, "write_templates"), "feature absent: capture template checks prepare completion and hashes before writes"
    _, directories = templates(tmp_path / "module")
    directory = next(p for p in directories if p.name == "capture")
    attempt = tmp_path / "module/units" / directory.parent.name / "attempts/1"
    attempt.mkdir(parents=True)
    if state != "missing":
        prepared(attempt)
        if state == "no-hashes":
            (attempt / "prepare/completion.sha256").unlink()
        elif state == "incomplete":
            (attempt / "prepare/phase.json").write_text('{"completion":false}')
        else:
            with (attempt / "prepare/phase.json").open("a") as stream:
                stream.write("\n")
    refuse(directory / "job.sh", [attempt], "prepare is not complete or hashes differ", tmp_path)


@pytest.mark.parametrize("canary", (False, True))
@pytest.mark.parametrize("prepare_instances", (None, ["fv1"], ["fv1e-sparky"], ["fv1", "fv1e-sparklina"]))
def test_every_template_uses_fleet_table_schema(tmp_path, canary, prepare_instances):
    assert "template_instances" in json.loads(Path(prepare_split.__file__).with_name("feeder-settings.json").read_text()), "feature absent: every template uses Fleet table schema"
    placement = {"canary": ["fv1"], "prepare": prepare_instances, "capture": ["fv1"], "compare": ["fv1e-sparklina"]}
    _, directories = templates(tmp_path / "module", canary, placement if prepare_instances is not None else None)
    for attempt_id, directory in enumerate(directories, start=4):
        assert (directory / "table.json").is_file(), "feature absent: every template uses Fleet table schema"
        phase = directory.name
        fields = json.loads((directory / "table.json").read_text())
        defaults = {"prepare": ["fv1", "fv1e-sparky", "fv1e-sparklina"], "compare": ["fv1"]}
        names = ["fv1"] if canary or phase == "capture" else (defaults[phase] if prepare_instances is None else placement[phase])
        instances = {}
        for name in names:
            hosts = {"fv1": ["sparky", "sparklina"], "fv1e-sparky": ["sparky"], "fv1e-sparklina": ["sparklina"]}[name]
            memory = {"fv1": 8, "fv1e-sparky": 9, "fv1e-sparklina": 9}[name] if phase == "prepare" else 32
            instances[name] = {"constraints": [{"hostlist": [host]} for host in hosts],
                               "shapes": [[{"type": "slot", "count": 1, "label": "task", "with":
                                           [{"type": "core", "count": 4 if canary and phase == "capture" else 1},
                                            {"type": "memory", "count": memory}]
                                           + ([{"type": "gpu", "count": 1}] if phase == "capture" else [])}]],
                               "max_duration_s": {"prepare": 28800 if "-gate-" in directory.parent.name else 18000,
                                                  "capture": 36000 if "-gate-" in directory.parent.name else 18000,
                                                  "compare": 86400}[phase]}
        assert fields == {"schema": 1, "instances": instances,
                          "job_name": f"tessera-1717-{directory.parent.name}-{phase}", "args": 1,
                          "max_attempts": 5, "urgency": 8,
                          "cwd": str(directory), "env_names": ["PATH", "HOME", "XDG_RUNTIME_DIR", "FLUX_F58_FORCE_ASCII", "FLUX_SCRATCH",
                                                                 "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]}
        assert not {"policy.json", "submit-after-review.sh", "submit-reviewed.sh"} & {p.name for p in directory.iterdir()}
        assert (directory / "job.sh").stat().st_mode & 0o100
        attempt = tmp_path / "module/units" / directory.parent.name / f"attempts/{attempt_id}"
        attempt.mkdir(parents=True)
        if phase == "capture":
            prepared(attempt)
        result = subprocess.run([str(directory / "job.sh"), str(attempt)], text=True, capture_output=True,
                                env={"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"})
        assert result.returncode != 0 and "FLUX_SCRATCH" in result.stderr
        assert not (attempt / phase).exists()


def test_compare_template_lists_all_units_and_uses_fv1e(tmp_path):
    assert hasattr(prepare_split, "write_templates"), "feature absent: one module compare template selects every latest unit on fv1"
    config, directories = templates(tmp_path / "module")
    directory = directories[-1]
    payload = (directory / "payload.sh").read_text()
    command = shlex.split(payload.split("exec ", 1)[1])
    assert command.count("--prepare-artifact") == command.count("--capture-artifact") == 4
    assert command[command.index("--case-set") + 1] == "all"
    assert not {"--tp-size", "--tp-rank"} & set(command)
    for key in config["unit_keys"]:
        assert key in payload
    assert "compare-inputs.json" in payload and "find " not in payload
    assert list(json.loads((directory / "table.json").read_text())["instances"]) == ["fv1"]


def test_compare_payload_uses_saved_attempts(tmp_path):
    assert "compare-inputs.json" in inspect.getsource(prepare_split.bind_templates), "feature absent: compare payload reads saved input attempts"
    config, directories = templates(tmp_path / "module")
    directory = next(p for p in directories if p.name == "compare")
    attempt = tmp_path / "module/units" / directory.parent.name / "attempts/1"
    attempt.mkdir(parents=True)
    selected = {key: str(tmp_path / "module/units" / key / "attempts/1") for key in config["unit_keys"]}
    for path in selected.values():
        Path(path).mkdir(parents=True)
        Path(path).with_name("2").mkdir()
    (attempt / "compare-inputs.json").write_text(json.dumps(selected))
    prefix = (directory / "payload.sh").read_text().split("exec ", 1)[0]
    script = prefix + "printf '%s\\n' " + " ".join(f'"$pair_{index}"' for index in range(len(selected)))
    result = subprocess.run(["bash", "-c", script, "payload", str(attempt)], text=True, capture_output=True)
    assert result.returncode == 0 and result.stdout.splitlines() == list(selected.values())


@pytest.mark.parametrize('outcome,finalized', [('FAIL', True), ('STOPPED', True), ('FAIL', False)])
def test_compare_exit_three_keeps_checked_completion_hashes(tmp_path, outcome, finalized):
    assert 'r.get("finalization_finished")' in (Path(prepare_split.__file__).parent / 'templates/job.sh').read_text(), 'feature absent: finalized compare exit 3 has completion hashes'
    _, directories = templates(tmp_path / 'module')
    directory = next(p for p in directories if p.name == 'compare')
    job = tmp_path / 'compare-output'
    job.mkdir()
    (job / 'phase.json').write_text('{"completion":true}')
    receipt = job / 'numerics_row.json'
    receipt.write_text(json.dumps({'row': {'outcome': outcome, 'finalization_finished': finalized}}))
    tail = 'rc=0\n' + (directory / 'job.sh').read_text().split('rc=0\n', 1)[1]
    script = 'set -euo pipefail\ntimeout() { return 3; }\n'
    script += 'task_root=$PWD\ntask_scratch=$PWD\nname=stub\nphase_mounts=()\npopulation_mount=()\nclaimed=("$PWD")\n' + tail
    result = subprocess.run(['bash', '-c', script], cwd=job, text=True, capture_output=True)
    assert result.returncode == 3
    assert (job / 'completion.sha256').exists() is finalized
    if finalized:
        checked = subprocess.run(['sha256sum', '-c', 'completion.sha256'], cwd=job, capture_output=True)
        assert checked.returncode == 0
        receipt.write_text('{}')
        changed = subprocess.run(['sha256sum', '-c', 'completion.sha256'], cwd=job, capture_output=True)
        assert changed.returncode != 0


@pytest.mark.parametrize("phase", ("prepare", "capture"))
def test_phase_local_outputs_keep_partial_fail_evidence(tmp_path, phase):
    assert hasattr(prepare_split, "write_templates"), "feature absent: phase-local files retain partial FAIL and artifact paths"
    pytest.importorskip("torch")
    from experiments.numerics_rows import row
    from experiments.numerics_rows.gpu_time import PhaseTimes
    from tests.test_numerics_rows_phases import retry_phase_attempts
    dp, api, args, receipt, groups, attempt = retry_phase_attempts(tmp_path)
    document_path = attempt / phase / "phase.json"
    partial = json.loads(document_path.read_text())["receipt"]
    wanted = row.unit_identity(partial["units"][0])
    partial["units"][0]["findings"] = ["known mismatch"]
    partial["units"][0]["outcome"] = "FAIL"
    local_receipt = attempt / phase / "numerics_row.json"
    local_receipt.write_text(json.dumps(partial))
    document_path.unlink()
    args.prepare_artifact = [attempt.with_name("2") / "prepare" if path.parent == attempt else path for path in args.prepare_artifact]
    args.capture_artifact = [attempt.with_name("2") / "capture" if path.parent == attempt else path for path in args.capture_artifact]
    args.phase_root = args.out / "compare"
    args.out = args.phase_root
    args.out.mkdir()
    before = snapshot(tmp_path)
    dp.host_compare(args, receipt, PhaseTimes({}), groups)
    failed = next(unit for unit in receipt["units"] if row.unit_identity(unit) == wanted)
    assert receipt["row"]["outcome"] == failed["outcome"] == "FAIL"
    assert (args.out / "phase.json").is_file()
    assert not (args.out / "compare").exists()
    listing = next(item for item in receipt["attempts"]["listing"] if item["unit"] == attempt.parent.parent.name and item["attempt"] == 1)
    assert listing["phases"][phase + "_partial"] == hashlib.sha256(local_receipt.read_bytes()).hexdigest()
    prefix = str(args.out.relative_to(tmp_path)) + "/"
    assert all(name.startswith(prefix) for name, value in snapshot(tmp_path).items() if name not in before or before[name] != value)
    api.read(args.out, "compare", api.identities(receipt), dp.scopes(groups))


def test_prepare_takes_output_root_and_refuses_existing_root(tmp_path, monkeypatch):
    assert "output_root" in inspect.signature(prepare_split.prepare).parameters, "feature absent: explicit campaign output root with existing root refusal"
    assert inspect.signature(prepare_split.prepare).parameters["output_root"].default is None
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}")
    monkeypatch.setattr(prepare_split, "FREEZE", freeze)
    monkeypatch.setattr(prepare_split.subprocess, "check_output", lambda *args, **kwargs: "a" * 40)
    clones = []
    monkeypatch.setattr(prepare_split.subprocess, "run", lambda command, **kwargs: clones.append(command))
    output = tmp_path / "chosen root"
    module = "model.language_model.layers.2.mlp.gate_up_proj"
    assert prepare_split.prepare(tmp_path / "checks.json", module=module, output_root=output) == output
    config = json.loads((output / "campaign.json").read_text())
    assert config["module"] == module and config["canary"] is True
    assert len(config["templates"]) == 9 and len(clones) == 1
    before = snapshot(tmp_path)
    with pytest.raises(FileExistsError):
        prepare_split.prepare(tmp_path / "checks.json", output_root=output)
    assert snapshot(tmp_path) == before and len(clones) == 1
    monkeypatch.undo()
    help_result = subprocess.run([sys.executable, "-m", "experiments.numerics_rows.prepare_split", "--help"],
                                 cwd=prepare_split.ROOT, capture_output=True, text=True)
    assert help_result.returncode == 0
    assert "--output-root" in help_result.stdout and "--module" in help_result.stdout


@pytest.mark.parametrize("canary", [True, False])
def test_phase_durations_from_settings_reach_table_and_job(tmp_path, canary):
    assert "phase_minutes" in json.loads(Path(prepare_split.__file__).with_name("feeder-settings.json").read_text()), "feature absent: phase and case set durations come from settings"
    config, directories = templates(tmp_path / "module", canary=canary)
    expected = {"prepare": {"gate": 480, "diagnostics": 300},
                "capture": {"gate": 600, "diagnostics": 300}, "compare": {"all": 1440}}
    for custom in (False, True):
        if custom:
            shutil.rmtree(tmp_path / "module/templates")
            expected = {phase: {part: minutes + 7 for part, minutes in parts.items()} for phase, parts in expected.items()}
            settings_path = tmp_path / "module/feeder-settings.json"
            settings = json.loads(settings_path.read_text())
            settings["phase_minutes"] = expected
            settings_path.write_text(json.dumps(settings))
            prepare_split.write_templates(tmp_path / "module", config, tmp_path / "module/repository.git")
        for unit, directory in zip(config["templates"], directories):
            seconds = expected[unit["phase"]][unit["case_set"]] * 60
            table = json.loads((directory / "table.json").read_text())
            assert all(instance["max_duration_s"] == seconds for instance in table["instances"].values())
            assert f"timeout --signal=TERM --kill-after=20s {seconds - 60}s " in (directory / "job.sh").read_text()


def cpu_campaign(tmp_path, monkeypatch):
    files = ["tests/test_numerics_alpha.py", "tests/test_numerics_beta.py"]
    rule, freeze, data, checks_path = [tmp_path / name for name in ("rule.md", "freeze.json", "population.json", "checks.json")]
    for path in (rule, freeze, data):
        path.write_text("{}")
    monkeypatch.setattr(prepare_split, "RULE", rule)
    monkeypatch.setattr(prepare_split, "RULE_SHA", prepare_split.sha256(rule))
    monkeypatch.setattr(prepare_split, "FREEZE", freeze)
    config = {"bound": True, "source_revision": prepare_split.SOURCE, "rule_sha256": prepare_split.RULE_SHA,
              "freeze_sha256": prepare_split.sha256(freeze), "tool_commit": "a" * 40, "cpu_check": str(checks_path),
              "canary": True, "paths": {"source": str(tmp_path), "population": str(data)},
              "digests": {"population": prepare_split.sha256(data)}}
    (tmp_path / "campaign.json").write_text(json.dumps(config))
    def git_output(command, **kwargs):
        if command[1] == "ls-tree":
            assert command == ["git", "ls-tree", "-r", "--name-only", config["tool_commit"], "--", "tests"]
            assert kwargs["cwd"] == prepare_split.ROOT
            return "\n".join(files + ["tests/test_other.py", "tests/nested/test_numerics_other.py"])
        assert command == ["git", "rev-parse", "HEAD"] and kwargs["cwd"] == str(tmp_path)
        return prepare_split.SOURCE
    monkeypatch.setattr(prepare_split.subprocess, "check_output", git_output)
    record = {"verdict": "PASS", "before": {"source": "b" * 40, "exit": 1, "failures": 4},
              "after": [dict(source=config["tool_commit"], exit=0, failures=0, errors=0, skips=0, selected=1234, passed=True)
                        for _ in range(2)], "test_files": files[::-1]}
    checks_path.write_text(json.dumps(record))
    return config, record, checks_path


def test_verify_accepts_v2_cpu_check(tmp_path, monkeypatch):
    assert hasattr(prepare_split, "check_cpu_check"), "feature absent: verify accepts the whole package v2 CPU check"
    config, record, checks_path = cpu_campaign(tmp_path, monkeypatch)
    assert "source" not in record
    assert prepare_split.verify(tmp_path) == config
    for arm in record["after"]:
        arm["selected"] = 4321
    checks_path.write_text(json.dumps(record))
    assert prepare_split.verify(tmp_path) == config


@pytest.mark.parametrize("arm_index", [0, 1])
@pytest.mark.parametrize("fault", ["source", "exit", "failures", "errors", "skips", "passed", "zero", "count", "missing-count", "bool-count",
                                   "verdict", "subset", "other-file", "one-arm", "three-arms"])
def test_verify_refuses_invalid_v2_cpu_check(tmp_path, monkeypatch, arm_index, fault):
    assert hasattr(prepare_split, "check_cpu_check"), "feature absent: verify refuses failed arms other sources and partial packages"
    config, record, checks_path = cpu_campaign(tmp_path, monkeypatch)
    assert prepare_split.verify(tmp_path) == config
    arm = record["after"][arm_index]
    if fault == "source":
        arm["source"] = "b" * 40
    elif fault in ("exit", "failures", "errors", "skips"):
        arm[fault] = 1
    elif fault == "passed":
        arm["passed"] = False
    elif fault == "zero":
        arm["selected"] = 0
    elif fault == "count":
        arm["selected"] += 1
    elif fault == "missing-count":
        del arm["selected"]
    elif fault == "bool-count":
        arm["selected"] = True
    elif fault == "verdict":
        record["verdict"] = "FAIL"
    elif fault == "subset":
        record["test_files"].pop()
    elif fault == "other-file":
        record["test_files"].append("tests/test_numerics_other.py")
    elif fault == "one-arm":
        record["after"].pop()
    else:
        record["after"].append(dict(arm))
    checks_path.write_text(json.dumps(record))
    with pytest.raises(Invalid, match="CPU check"):
        prepare_split.verify(tmp_path)
