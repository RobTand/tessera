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
    CENSUS_DECLARATION_ENV, CENSUS_IMAGE_ENV, DECLARATION_SOURCE, container_env, resolve)

IMAGE = "example/runtime@sha256:" + "1" * 64
OTHER_IMAGE = "example/runtime@sha256:" + "2" * 64


def _launcher_env(reference=IMAGE):
    """What the wrapper exports, built by the code the wrapper calls."""
    return container_env(resolve(reference, inspector=lambda _r: {
        "present": True, "local_id": "sha256:" + "ab" * 32,
        "repo_digests": [reference]}))


def _census():
    spec = importlib.util.spec_from_file_location(
        "census_under_test", ROOT / "tools" / "tessera_route_census.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replay():
    spec = importlib.util.spec_from_file_location(
        "scoped_replay", ROOT / "experiments" / "ts111_replay_cell_agreement.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plugin_runtime(tmp_path, reference=IMAGE, resolved=IMAGE):
    binary = tmp_path / 'bin'; binary.mkdir()
    podman = binary / 'podman'
    podman.write_text(f"#!{sys.executable}\n" + r'''
import json, os, pathlib, sys
args=sys.argv[1:]; root=pathlib.Path(os.environ['CONTROL'])
if args[:2]==['image','inspect']: print('sha256:'+'0'*64+'\t'+json.dumps([os.environ['RESOLVED']]))
elif args[0]=='info': print('true')
elif args[0]=='create':
 env={}
 for i,value in enumerate(args[:-1]):
  if value in ('-e','--env'):
   key,_,val=args[i+1].partition('=');env[key]=val
 (root/'env.json').write_text(json.dumps({'container_env':env}))
 pathlib.Path(args[args.index('--cidfile')+1]).write_text('b'*64)
 scratch=pathlib.Path(args[args.index('--volume')+1].split(':')[0])
 (scratch/'container-status').write_text(pathlib.Path('/proc/self/status').read_text())
 (root/'owned').touch()
elif args[0]=='start': print((root/'env.json').read_text())
elif args[0]=='rm': (root/'owned').unlink()
elif args[:2]==['container','exists']: sys.exit(int(not (root/'owned').exists()))
''')
    podman.chmod(0o755)
    for name, value in [('flux', 'f123'), ('findmnt', 'ext4')]:
        p=binary/name
        p.write_text('#!/bin/sh\nif [ "$3" = --to=dec ]; then echo 123; else echo '+value+'; fi\n')
        p.chmod(0o755)
    return dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", CONTROL=str(tmp_path),
                RUNTIME_IMAGE_PY=sys.executable, IMG=reference, RESOLVED=resolved, TS=str(ROOT),
                EXT=str(tmp_path/'ext'), RUNS=str(tmp_path/'runs'),
                FLUX_JOB_ID='f123', CUDA_VISIBLE_DEVICES='0', FLUX_SCRATCH=str(tmp_path/'scratch'))


def test_plugin_wrapper_injects_selected_image_after_caller_environment(tmp_path):
    env = _plugin_runtime(tmp_path)
    env.update({CENSUS_IMAGE_ENV: 'forged-host-value', CENSUS_DECLARATION_ENV: 'forged-host-declaration'})
    proc = subprocess.run(
        ['bash', str(ROOT/'experiments/tessera_plugin_run.sh'),
         '-e', f'{CENSUS_IMAGE_ENV}=forged-caller-value',
         '-e', f'{CENSUS_DECLARATION_ENV}=forged-caller-declaration', '--', 'true'],
        env=env, text=True, capture_output=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout.splitlines()[-1])
    assert payload['container_env'][CENSUS_IMAGE_ENV] == IMAGE
    record = json.loads(payload['container_env'][CENSUS_DECLARATION_ENV])
    assert record['resolved_reference'] == IMAGE and record['repo_digests'] == [IMAGE]


def test_the_wrapper_exports_the_resolved_digest_not_the_tag_it_was_asked_for(tmp_path):
    env = _plugin_runtime(tmp_path, reference='example/runtime:latest')
    proc = subprocess.run(
        ['bash', str(ROOT/'experiments/tessera_plugin_run.sh'), '--', 'true'],
        env=env, text=True, capture_output=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout.splitlines()[-1])
    assert payload['container_env'][CENSUS_IMAGE_ENV] == IMAGE


# --------------------------------------- the census checks its own claim ---

def test_the_census_refuses_an_image_the_launcher_did_not_declare(capsys):
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


def test_the_census_refuses_when_the_launcher_declared_no_image(capsys):
    """A hand ``docker run`` outside the launcher writes no record at all."""
    with pytest.raises(SystemExit) as raised:
        _census().parse_args(["ckpt", "out.json", "--runtime-image", IMAGE], env={})
    assert raised.value.code == 2
    assert CENSUS_IMAGE_ENV in capsys.readouterr().err


def test_the_census_carries_the_mechanism_that_established_its_scope():
    args = _census().parse_args(
        ["ckpt", "out.json", "--runtime-image", IMAGE], env=_launcher_env())
    assert args.runtime_image == IMAGE
    assert args.runtime_image_declaration["source"] == DECLARATION_SOURCE
    assert args.runtime_image_declaration["image"] == IMAGE


#: The census tool, named once.  A launcher calls it by PATH or as a MODULE,
#: through whatever interpreter the image ships, from whatever mount the
#: container gave it -- and the rule below is about the CALL, not about one
#: box's spelling of it.  Matching an interpreter in front of the name is what
#: keeps a docstring or an ``import`` of the same module from reading as a
#: launch; an example invocation in a comment does match, and it should, since
#: an example that omits ``--runtime-image`` teaches the defect this gate
#: exists to refuse.
CENSUS_TOOL = ROOT / "tools" / "tessera_route_census.py"
_INTERPRETER = r"(?:[\w./+-]*python[\w.]*|uvx|uv run|poetry run|pipx run)"
_CENSUS_CALL = re.compile(
    rf"{_INTERPRETER}\s+(?:\S+\s+){{0,3}}?"
    rf"(?:[\w./-]*{re.escape(CENSUS_TOOL.name)}|-m\s+[\w.]*{re.escape(CENSUS_TOOL.stem)})")


def census_invocations(text: str) -> list:
    """Every call of the census tool in ``text``, one command's worth each.

    One invocation's worth of text: the shell form ends at a line
    continuation, the python form is an adjacent-string block, and neither
    spans a blank line.
    """
    return [text[m.start():m.start() + 500].split("\n\n", 1)[0]
            for m in _CENSUS_CALL.finditer(text)]


def test_a_census_call_is_recognised_however_the_launcher_spells_it():
    """The pre-fix failure this test was written for::

        AssertionError: python -m tools.tessera_route_census ckpt out.json
        assert [] == ['python -m t...kpt out.json']

    Detection was a regex over ONE spelling, ``python3 (/work/)?tools/
    tessera_route_census.py``.  A launcher using ``python -m``, another mount,
    another interpreter or ``uv run`` was not seen as a census caller at all,
    so the assertion below silently narrowed to nothing on the day somebody
    changed how the campaign starts it -- a gate that stops looking is worse
    than one that refuses.
    """
    for command in (
            "python3 tools/tessera_route_census.py ckpt out.json",
            "python3 /work/tools/tessera_route_census.py ckpt out.json",
            "python -m tools.tessera_route_census ckpt out.json",
            "/usr/bin/python3.12 /opt/ts/tools/tessera_route_census.py ckpt out.json",
            "uv run tools/tessera_route_census.py ckpt out.json",
            "python3 -X faulthandler tools/tessera_route_census.py ckpt out.json"):
        assert census_invocations(command) == [command], command
    # Naming the module is not calling it: the replay tool's docstring and the
    # check tool's import both mention it and neither starts a census.
    assert census_invocations("``tools/tessera_route_census.py`` writes the receipt") == []
    assert census_invocations("from tools.tessera_route_census import parse_args") == []


def test_every_census_invocation_passes_the_verified_wrapper_image():
    """A driver that spells the digest itself reads its own mind, not the run.

    Since #132 a reconstructed image is a refusal rather than a silent
    rescope -- but a driver that learns at census time what its own text says
    has burned two model loads to find out. Both suffixes: the .sh-only scan
    was blind to ``experiments/ts5_lfm_served_bound.py``, the one census
    caller that interpolated the digest from a host-side constant.
    """
    invocations = []
    for path in sorted(p for p in (ROOT / "experiments").rglob("*")
                       if p.suffix in {".sh", ".py"}):
        for command in census_invocations(path.read_text()):
            invocations.append((path.relative_to(ROOT), command))
    assert invocations
    for path, command in invocations:
        assert "--runtime-image" in command, str(path)
        assert f"${CENSUS_IMAGE_ENV}" in command, str(path)


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
