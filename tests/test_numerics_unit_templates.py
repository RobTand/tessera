"""Check reviewed unit templates without a Flux client or numerical work."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

from experiments.numerics_rows import phase_instance, prepare_split


def templates(root, canary=False, instance_lists=None):
    root.mkdir()
    source = Path(prepare_split.__file__).parent
    for name in ("phase_instance.py", "feeder-settings.json"):
        shutil.copyfile(source / name, root / {"phase_instance.py": "instance.py"}.get(name, name))
    if instance_lists is not None:
        settings = json.loads((root / "feeder-settings.json").read_text())
        settings["template_instances"] = instance_lists
        (root / "feeder-settings.json").write_text(json.dumps(settings))
    config = {"module": "module-A", "tool_commit": "a" * 40, "canary": canary,
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
        assert all("  " + name + "\n" not in manifest for name in ("job.sh", "table.json"))
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
    refuse(directory / "job.sh", [attempt], "phase claim already exists or failed", tmp_path)
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
    placement = {"canary": ["fv1"], "prepare": prepare_instances, "capture": ["fv1"], "compare": ["fv1e-sparklina"]}
    _, directories = templates(tmp_path / "module", canary, placement if prepare_instances is not None else None)
    for directory in directories:
        assert (directory / "table.json").is_file(), "feature absent: every template uses Fleet table schema"
        phase = directory.name
        fields = json.loads((directory / "table.json").read_text())
        defaults = {"prepare": ["fv1", "fv1e-sparky", "fv1e-sparklina"], "compare": ["fv1e-sparky", "fv1e-sparklina"]}
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
                               "max_duration_s": {"prepare": 21600 if canary else 86400,
                                                  "capture": 900, "compare": 7200 if canary else 21600}[phase]}
        assert fields == {"schema": 1, "instances": instances,
                          "job_name": f"tessera-1717-{directory.parent.name}-{phase}", "args": 1,
                          "max_attempts": 3,
                          "cwd": str(directory), "env_names": ["PATH", "HOME", "XDG_RUNTIME_DIR", "FLUX_F58_FORCE_ASCII", "FLUX_SCRATCH",
                                                                 "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"],
                          "shell_options": {}}
        assert not {"policy.json", "submit-after-review.sh", "submit-reviewed.sh"} & {p.name for p in directory.iterdir()}
        assert (directory / "job.sh").stat().st_mode & 0o100
        attempt = tmp_path / "module/units" / directory.parent.name / "attempts/4"
        attempt.mkdir(parents=True)
        if phase == "capture":
            prepared(attempt)
        result = subprocess.run([str(directory / "job.sh"), str(attempt)], text=True, capture_output=True,
                                env={"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"})
        assert result.returncode != 0 and "FLUX_SCRATCH" in result.stderr
        assert (attempt / phase).is_dir()


def test_compare_template_lists_all_units_and_uses_fv1e(tmp_path, monkeypatch):
    assert hasattr(prepare_split, "write_templates"), "feature absent: one module compare template selects every latest unit on fv1e"
    config, directories = templates(tmp_path / "module")
    directory = directories[-1]
    payload = (directory / "payload.sh").read_text()
    command = shlex.split(payload.split("exec ", 1)[1])
    assert command.count("--prepare-artifact") == command.count("--capture-artifact") == 4
    assert command[command.index("--case-set") + 1] == "all"
    assert not {"--tp-size", "--tp-rank"} & set(command)
    for key in config["unit_keys"]:
        assert str(tmp_path / "module/units" / key / "attempts") in payload
    job = tmp_path / "module/units" / directory.parent.name / "attempts/1"
    job.mkdir(parents=True)
    (job / "compare-instance-name.txt").write_text("fv1e-sparklina\n")
    calls = []
    monkeypatch.setattr(phase_instance, "observe", lambda name: {"name": name})
    monkeypatch.setattr(phase_instance, "query", lambda command, *args: calls.append((command, args)) or "f123")
    phase_instance.submit(job, ["submit", "--urgency=8", "payload"], "compare")
    assert json.loads((job / "compare-instance.json").read_text())["name"] == "fv1e-sparklina"
    receipt = {}
    phase_instance.bind(receipt, "compare", {"name": "fv1e-sparklina"})
    assert receipt["phase_jobs"]["compare"]["name"] == "fv1e-sparklina"
    assert calls[0][0] == phase_instance.client("fv1e-sparklina")


@pytest.mark.parametrize("phase", ("prepare", "capture"))
def test_phase_local_outputs_keep_partial_fail_evidence(tmp_path, phase):
    assert hasattr(prepare_split, "write_templates"), "feature absent: phase-local files retain partial FAIL and artifact paths"
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
