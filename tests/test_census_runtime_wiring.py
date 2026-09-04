"""The census receives the selected image, including through shell wrappers."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "example/runtime@sha256:" + "1" * 64


def _replay():
    spec = importlib.util.spec_from_file_location(
        "scoped_replay", ROOT / "experiments" / "ts111_replay_cell_agreement.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plugin_wrapper_injects_selected_image_after_caller_environment(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    docker = fake / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1:3] == ['image', 'inspect']:\n"
        "    print('sha256:' + '0' * 64 + '\\t' + json.dumps([os.environ['IMG']]))\n"
        "else:\n"
        "    env = {}\n"
        "    for i, value in enumerate(sys.argv[:-1]):\n"
        "        if value == '-e':\n"
        "            key, _, val = sys.argv[i+1].partition('=')\n"
        "            env[key] = val\n"
        "    print(json.dumps({'container_env': env}))\n"
    )
    docker.chmod(0o755)
    env = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}",
               RUNTIME_IMAGE_PY=sys.executable, IMG=IMAGE, TS=str(ROOT),
               EXT=str(tmp_path / "ext"), RUNS=str(tmp_path / "runs"),
               TESSERA_CENSUS_RUNTIME_IMAGE="forged-host-value")
    proc = subprocess.run(
        ["bash", str(ROOT / "experiments" / "tessera_plugin_run.sh"),
         "-e", "TESSERA_CENSUS_RUNTIME_IMAGE=forged-caller-value", "--", "true"],
        env=env, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout.splitlines()[-1])
    assert payload["container_env"]["TESSERA_CENSUS_RUNTIME_IMAGE"] == IMAGE


def test_every_shell_census_invocation_passes_the_verified_wrapper_image():
    invocations = []
    for path in sorted((ROOT / "experiments").rglob("*.sh")):
        text = path.read_text()
        for match in re.finditer(r"python3 (?:/work/)?tools/tessera_route_census\.py", text):
            command = text[match.start():].split('" \\', 1)[0]
            invocations.append((path.relative_to(ROOT), command))
    assert invocations
    for path, command in invocations:
        assert "--runtime-image" in command, str(path)
        assert r"\$TESSERA_CENSUS_RUNTIME_IMAGE" in command, str(path)


@pytest.mark.parametrize("compiled", [False, True])
def test_historical_replay_does_not_infer_runtime_from_global_pin_or_compiled(compiled):
    got = _replay().replay_runtime_context({"compiled": compiled})
    assert got == {"runtime_image": None, "execution_mode": None}


def test_replay_uses_only_recorded_runtime_context():
    got = _replay().replay_runtime_context({
        "runtime": {"image": IMAGE, "execution_mode": "eager"}, "compiled": False})
    assert got == {"runtime_image": IMAGE, "execution_mode": "eager"}


@pytest.mark.parametrize("runtime,compiled", [
    ({"image": "example/runtime:latest", "execution_mode": "eager"}, False),
    ({"image": IMAGE, "execution_mode": "compiled"}, False),
    ({"image": IMAGE, "execution_mode": "eager"}, True),
    ({"image": IMAGE, "execution_mode": "unknown"}, False),
])
def test_replay_refuses_malformed_or_contradictory_runtime_context(runtime, compiled):
    with pytest.raises(ValueError):
        _replay().replay_runtime_context({"runtime": runtime, "compiled": compiled})
