"""Check capture groups with local files and fake payloads. Submit nothing."""
import json
import os
from pathlib import Path
import select
import shlex
import shutil
import signal
import subprocess
import sys

import pytest

from experiments.numerics_rows import prepare_split
from experiments.numerics_rows.common import Invalid
from experiments.numerics_rows.launch_state import json_digest
from tests.test_numerics_unit_templates import prepared, snapshot, templates


def inputs(root):
    root.mkdir()
    source = Path(prepare_split.__file__).parent
    for name, target in (("phase_instance.py", "instance.py"), ("feeder-settings.json", "feeder-settings.json")):
        shutil.copyfile(source / name, root / target)
    keys = [f"tp{size}-{part}-" + json_digest(["module-A", size, part])
            for size in (1, 2) for part in ("gate", "diagnostics")]
    return {"module": "module-A", "tool_commit": "a" * 40, "canary": False,
            "paths": {"source": str(root / "source")}, "staged_population": str(root / "population.json"),
            "capture_groups": [{"key": "group-A", "unit_keys": keys[:3]}]}


def set_canary(root):
    settings = json.loads((root / "feeder-settings.json").read_text())
    settings["capture_group_canary"] = {"cores_per_capture": 2, "anon_peak_gib_per_capture": 2.25,
                                        "margin_gib": 1.5, "max_k": 3}
    (root / "feeder-settings.json").write_text(json.dumps(settings))


def build_group(root):
    config = inputs(root)
    set_canary(root)
    prepare_split.write_templates(root, config, root / "repository.git")
    prepare_split.bind_templates(root, config)
    directory = root / "templates/group-A/capture"
    attempts = [root / "units" / key / "attempts/1" for key in config["capture_groups"][0]["unit_keys"]]
    for attempt in attempts:
        attempt.mkdir(parents=True)
        prepared(attempt)
    return config, directory, attempts


def run_job(directory, attempts, path=os.defpath):
    return subprocess.run(["bash", str(directory / "job.sh"), *map(str, attempts)], text=True, capture_output=True,
                          env={"PATH": path, "PYTHONDONTWRITEBYTECODE": "1"})


@pytest.mark.parametrize("count", (2, 3, 4))
def test_group_requires_exactly_k_arguments(tmp_path, count):
    assert hasattr(prepare_split, "capture_templates"), "feature absent: capture group requires exactly K arguments"
    _, directory, attempts = build_group(tmp_path / "module")
    before = snapshot(tmp_path)
    result = run_job(directory, (attempts + [attempts[0]])[:count])
    if count == 3:
        assert result.returncode != 0 and "FLUX_SCRATCH" in result.stderr
        assert all((attempt / "capture").is_dir() for attempt in attempts)
    else:
        assert result.returncode == 3 and "require 3 attempt arguments" in result.stderr
        assert snapshot(tmp_path) == before


@pytest.mark.parametrize("index", (0, 1, 2))
@pytest.mark.parametrize("change", ("key", "zero", "symlink", "parent", "prepare", "hash", "capture", "capture-link"))
def test_group_validates_every_argument_before_any_claim(tmp_path, index, change):
    assert hasattr(prepare_split, "capture_templates"), "feature absent: capture group validates all arguments before writes"
    _, directory, attempts = build_group(tmp_path / "module")
    target = attempts[index]
    if change == "key":
        attempts[index] = target.parents[2] / "wrong-key/attempts/1"
    elif change == "zero":
        attempts[index] = target.with_name("0")
    elif change == "symlink":
        moved = tmp_path / "target"
        target.rename(moved)
        target.symlink_to(moved, target_is_directory=True)
    elif change == "parent":
        attempts[index] = str(target) + "/../1"
    elif change == "prepare":
        (target / "prepare/phase.json").write_text('{"completion":false}')
    elif change == "hash":
        with (target / "prepare/phase.json").open("a") as stream:
            stream.write("\n")
    elif change == "capture":
        (target / "capture").mkdir()
    else:
        (target / "capture").symlink_to(tmp_path / "absent", target_is_directory=True)
    before = snapshot(tmp_path)
    result = run_job(directory, attempts)
    assert result.returncode == 3, result.stderr
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("index", (1, 2))
def test_group_rolls_back_only_its_own_claims(tmp_path, index):
    assert hasattr(prepare_split, "capture_templates"), "feature absent: capture group rolls back partial claims"
    _, directory, attempts = build_group(tmp_path / "module")
    commands = tmp_path / "bin"
    commands.mkdir()
    mkdir = shlex.quote(shutil.which("mkdir"))
    contested = str(attempts[index] / "capture")
    fake = commands / "mkdir"
    fake.write_text(f'#!/bin/bash\nif test "$1" = {shlex.quote(contested)}; then {mkdir} "$1"; fi\nexec {mkdir} "$@"\n')
    fake.chmod(0o700)
    before = snapshot(tmp_path)
    result = run_job(directory, attempts, str(commands) + os.pathsep + os.defpath)
    assert result.returncode == 3 and "phase claim already exists or failed" in result.stderr
    before[str(Path(contested).relative_to(tmp_path))] = None
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("duplicate", ("within", "across"))
def test_group_refuses_duplicate_capture_membership(tmp_path, duplicate):
    assert hasattr(prepare_split, "capture_templates"), "feature absent: each unit has exactly one capture template"
    root = tmp_path / "module"
    config = inputs(root)
    set_canary(root)
    if duplicate == "within":
        config["capture_groups"][0]["unit_keys"][1] = config["capture_groups"][0]["unit_keys"][0]
    else:
        config["capture_groups"].append({"key": "group-B", "unit_keys": config["capture_groups"][0]["unit_keys"][-2:]})
    before = snapshot(tmp_path)
    with pytest.raises(Invalid, match="unit key appears in two capture templates"):
        prepare_split.write_templates(root, config, root / "repository.git")
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("field", (None, "cores_per_capture", "anon_peak_gib_per_capture", "margin_gib", "max_k"))
def test_group_refuses_unset_canary_table(tmp_path, field):
    assert hasattr(prepare_split, "capture_templates"), "feature absent: capture group requires a set canary table"
    root = tmp_path / "module"
    config = inputs(root)
    if field is not None:
        set_canary(root)
        settings = json.loads((root / "feeder-settings.json").read_text())
        settings["capture_group_canary"][field] = None
        (root / "feeder-settings.json").write_text(json.dumps(settings))
    before = snapshot(tmp_path)
    with pytest.raises(Invalid, match="capture group canary table is unset or invalid"):
        prepare_split.write_templates(root, config, root / "repository.git")
    assert snapshot(tmp_path) == before
    _, singles = templates(tmp_path / "singles")
    assert sum(directory.name == "capture" for directory in singles) == 4


@pytest.mark.parametrize("max_k", (2, 3))
def test_group_uses_canary_shape_and_fleet_table(tmp_path, max_k):
    assert hasattr(prepare_split, "capture_templates"), "feature absent: capture group uses canary size and Fleet table"
    root = tmp_path / "module"
    config = inputs(root)
    set_canary(root)
    settings = json.loads((root / "feeder-settings.json").read_text())
    settings["capture_group_canary"]["max_k"] = max_k
    (root / "feeder-settings.json").write_text(json.dumps(settings))
    if max_k == 2:
        with pytest.raises(Invalid, match="K exceeds canary admission"):
            prepare_split.write_templates(root, config, root / "repository.git")
        return
    prepare_split.write_templates(root, config, root / "repository.git")
    prepare_split.bind_templates(root, config)
    directory = root / "templates/group-A/capture"
    table = json.loads((directory / "table.json").read_text())
    assert table == {"schema": 1, "instances": {"fv1": {"constraints": [{"hostlist": ["sparky"]}, {"hostlist": ["sparklina"]}],
                     "shapes": [[{"type": "slot", "count": 1, "label": "task", "with":
                                 [{"type": "core", "count": 6}, {"type": "memory", "count": 9}, {"type": "gpu", "count": 1}]}]],
                     "max_duration_s": 36000}}, "job_name": "tessera-1717-group-A-capture", "args": 3,
                     "max_attempts": 3, "urgency": 8, "cwd": str(directory),
                     "env_names": ["PATH", "HOME", "XDG_RUNTIME_DIR", "FLUX_F58_FORCE_ASCII", "FLUX_SCRATCH",
                                   "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"],
                     "shell_options": {}}
    grouped, = [item for item in config["templates"] if item["key"] == "group-A"]
    assert grouped["shape"] == "slot=1/[core=6;memory=9;gpu=1]"
    captures = [key for item in config["templates"] if item["phase"] == "capture" for key in item.get("unit_keys", [item["key"]])]
    assert sorted(captures) == sorted(config["unit_keys"]) and len(set(captures)) == 4
    assert not (directory / "submit-after-review.sh").exists() and not (directory / "policy.json").exists()
    assert (directory / "job.sh").stat().st_mode & 0o100
    assert "  job.sh\n" not in (directory / "template.sha256").read_text()
    command = json.loads((directory / "commands.json").read_text())[0]["command"]
    assert command[command.index("--out") + 1] == "__ATTEMPT__" and "--phase-local-output" in command


def test_group_leaves_attempt_cap_to_fleet(tmp_path):
    assert hasattr(prepare_split, "capture_templates"), "feature absent: capture group leaves attempt cap to Fleet"
    _, directory, attempts = build_group(tmp_path / "module")
    for index, attempt in enumerate(attempts):
        attempt.rename(attempt.with_name("4"))
        attempts[index] = attempt.with_name("4")
    assert json.loads((directory / "table.json").read_text())["max_attempts"] == 3
    assert "MAX_ATTEMPTS" not in (directory / "job.sh").read_text()
    result = run_job(directory, attempts)
    assert result.returncode != 0 and "FLUX_SCRATCH" in result.stderr
    assert all((attempt / "capture").is_dir() for attempt in attempts)


@pytest.mark.parametrize("terminate", (False, True))
def test_group_term_reaches_all_children_and_waits_for_exits(tmp_path, terminate):
    assert hasattr(prepare_split, "capture_templates"), "feature absent: capture group forwards TERM to every child"
    from experiments.numerics_rows import capture_group
    _, directory, attempts = build_group(tmp_path / "module")
    for attempt in attempts:
        (attempt / "capture").mkdir()
    payload = tmp_path / "fake.py"
    payload.write_text('''import os, signal, sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index('--out') + 1]) / 'capture'
def stop(_signal, _frame):
    (out / 'term').write_text('TERM')
    raise SystemExit(143)
signal.signal(signal.SIGTERM, stop)
(out / 'pid').write_text(str(os.getpid()))
fd = os.open(sys.argv[sys.argv.index('--ready-fifo') + 1], os.O_WRONLY)
os.write(fd, b'R')
os.close(fd)
if 'fail' in sys.argv:
    raise SystemExit(7)
if 'wait' in sys.argv:
    signal.pause()
''')
    units = [{"out": str(attempt), "command": [sys.executable, str(payload), "--phase", "capture", "--out", str(attempt),
              "wait" if terminate else "fail" if index == 1 else "pass"]} for index, attempt in enumerate(attempts)]
    driver = tmp_path / "driver.py"
    driver.write_text("import json,sys\nfrom experiments.numerics_rows import capture_group,gpu_time\n"
                      "gpu_time.preflight = lambda: {'verdict': 'FREE'}\n"
                      "raise SystemExit(capture_group.capture_group(3,json.loads(sys.argv[1]),claimed=True))\n")
    readers = []
    for index, item in enumerate(units):
        fifo = tmp_path / f"ready-{index}"
        os.mkfifo(fifo)
        readers.append(os.open(fifo, os.O_RDONLY | os.O_NONBLOCK))
        item["command"] += ["--ready-fifo", str(fifo)]
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "containers.conf").write_text("fake")
    stop = "stop() {" + (directory / "job.sh").read_text().split("stop() {", 1)[1].split("\ntrap cleanup EXIT", 1)[0]
    wrapper = tmp_path / "stop.sh"
    wrapper.write_text(f'task_scratch={shlex.quote(str(scratch))}\nname=fake\nchild=\n'
                       'podman() { wait "$child" || true; }\n' + stop + '\ntrap \'stop 143\' TERM\n'
                       + shlex.join([sys.executable, str(driver), json.dumps(units)]) + ' &\nchild=$!\nwait "$child"\n')
    process = subprocess.Popen(["bash", str(wrapper)],
                               env={"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1",
                                    "PYTHONPATH": str(Path(capture_group.__file__).parents[2])})
    try:
        pending = set(readers)
        while pending:
            ready, _, _ = select.select(list(pending), [], [], 20)
            assert ready, "capture children did not become ready"
            for descriptor in ready:
                assert os.read(descriptor, 1) == b"R"
                pending.remove(descriptor)
        if terminate:
            process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=20) == (143 if terminate else 7)
        for attempt in attempts:
            output = attempt / "capture"
            assert (output / "term").exists() is terminate
            status = json.loads((output / "exit.json").read_text())["exit_status"]
            assert status == (143 if terminate else 7 if attempt == attempts[1] else 0)
            with pytest.raises(ProcessLookupError):
                os.kill(int((output / "pid").read_text()), 0)
            assert (output / "slot.json").is_file()
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=20)
        for descriptor in readers:
            os.close(descriptor)
