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
sys.path.insert(0, str(ROOT / "src"))

from tessera.serving.runtime_image import (  # noqa: E402
    ATTESTATION_SOURCE, CENSUS_ATTESTATION_ENV, CENSUS_IMAGE_ENV, container_env, resolve)

IMAGE = "example/runtime@sha256:" + "1" * 64
OTHER_IMAGE = "example/runtime@sha256:" + "2" * 64


def _launcher_env(reference=IMAGE):
    """What the wrapper exports, built by the code the wrapper calls."""
    return container_env(resolve(reference, inspector=lambda _r: {
        "present": True, "local_id": "sha256:" + "ab" * 32,
        "repo_digests": [reference]}))


def _census():
    spec = importlib.util.spec_from_file_location(
        "attested_census", ROOT / "tools" / "tessera_route_census.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
               **{CENSUS_IMAGE_ENV: "forged-host-value",
                  CENSUS_ATTESTATION_ENV: "forged-host-attestation"})
    proc = subprocess.run(
        ["bash", str(ROOT / "experiments" / "tessera_plugin_run.sh"),
         "-e", f"{CENSUS_IMAGE_ENV}=forged-caller-value",
         "-e", f"{CENSUS_ATTESTATION_ENV}=forged-caller-attestation", "--", "true"],
        env=env, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout.splitlines()[-1])
    assert payload["container_env"][CENSUS_IMAGE_ENV] == IMAGE
    # The daemon's own record travels beside the name, so the process inside
    # can check its claim against a table instead of trusting one string.
    record = json.loads(payload["container_env"][CENSUS_ATTESTATION_ENV])
    assert record["resolved_reference"] == IMAGE and record["repo_digests"] == [IMAGE]


def test_the_wrapper_exports_the_resolved_digest_not_the_tag_it_was_asked_for(tmp_path):
    """A tag identifies no runtime; what the census must receive is the bytes."""
    fake = tmp_path / "bin"
    fake.mkdir()
    docker = fake / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1:3] == ['image', 'inspect']:\n"
        "    print('sha256:' + '0' * 64 + '\\t' + json.dumps([os.environ['RESOLVED']]))\n"
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
               RUNTIME_IMAGE_PY=sys.executable, IMG="example/runtime:latest",
               RESOLVED=IMAGE, TS=str(ROOT), EXT=str(tmp_path / "ext"),
               RUNS=str(tmp_path / "runs"))
    proc = subprocess.run(
        ["bash", str(ROOT / "experiments" / "tessera_plugin_run.sh"), "--", "true"],
        env=env, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout.splitlines()[-1])
    assert payload["container_env"][CENSUS_IMAGE_ENV] == IMAGE


# --------------------------------------- the census checks its own claim ---

def test_the_census_refuses_an_image_the_container_did_not_attest(capsys):
    """Issue #132: the operator's string was the whole of the scope."""
    with pytest.raises(SystemExit) as raised:
        _census().parse_args(["ckpt", "out.json", "--runtime-image", OTHER_IMAGE],
                             env=_launcher_env())
    assert raised.value.code == 2
    err = capsys.readouterr().err
    # Both bytes named, and the variable holding the right answer: a refusal a
    # reader cannot act on gets worked around rather than honoured.
    assert "--runtime-image" in err and OTHER_IMAGE in err and IMAGE in err
    assert CENSUS_IMAGE_ENV in err


def test_the_census_refuses_when_nothing_attested_the_image(capsys):
    """A hand ``docker run`` outside the launcher writes no receipt at all."""
    with pytest.raises(SystemExit) as raised:
        _census().parse_args(["ckpt", "out.json", "--runtime-image", IMAGE], env={})
    assert raised.value.code == 2
    assert CENSUS_IMAGE_ENV in capsys.readouterr().err


def test_the_census_carries_the_mechanism_that_attested_its_scope():
    args = _census().parse_args(
        ["ckpt", "out.json", "--runtime-image", IMAGE], env=_launcher_env())
    assert args.runtime_image == IMAGE
    assert args.runtime_image_attestation["source"] == ATTESTATION_SOURCE
    assert args.runtime_image_attestation["image"] == IMAGE


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
