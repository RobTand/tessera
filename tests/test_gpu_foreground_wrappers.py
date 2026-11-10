"""Foreground entry points use real lifetime policy at a controlled runtime."""
import json
import subprocess
from pathlib import Path

import pytest

from gpu_podman_fixtures import controlled_runtime, IMAGE, ROOT


@pytest.mark.parametrize('family', ['regdirect', 'regdirect-pytest', 'plugin', 'fp4'])
def test_foreground_wrapper_owns_its_container(tmp_path, family):
    _, env, out = controlled_runtime(tmp_path)
    # A legacy Docker call must fail immediately rather than touch a daemon.
    docker = tmp_path / 'bin/docker'
    docker.write_text('#!/bin/sh\necho legacy-docker-boundary >&2\nexit 64\n')
    docker.chmod(0o755)
    env.update(ORACLE_IMAGE=IMAGE, IMG=IMAGE, TS=str(ROOT), RUNS=str(out),
               EXT=str(out / 'ext'), FLUX_SCRATCH=str(tmp_path / 'scratch'))
    if family == 'regdirect':
        script = ROOT / 'experiments/regdirect_stage1/run.sh'
        args = [str(ROOT), str(out), '--probe-marker']
    elif family == 'regdirect-pytest':
        script = ROOT / 'experiments/regdirect_stage1/run_pytest.sh'
        packages = tmp_path / 'packages'
        packages.mkdir()
        for name in ('pytest', '_pytest', 'pluggy', 'iniconfig', 'packaging', 'pygments'):
            (packages / name).mkdir()
        (packages / 'py.py').touch()
        env['TEST_RUNNER_SP'] = str(packages)
        args = [str(ROOT), str(out), '--probe-marker']
    elif family == 'plugin':
        script = ROOT / 'experiments/tessera_plugin_run.sh'
        args = ['--', 'echo --probe-marker']
    else:
        script = ROOT / 'experiments/t4_code/fp4_arithmetic_action.sh'
        args = [str(out), '--probe-marker']
    result = subprocess.run(['bash', str(script), *args], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in (tmp_path / 'calls').read_text().splitlines()]
    create, = [c for c in calls if c[0] == 'create']
    name = create[create.index('--name') + 1]
    record = json.loads((out / f'container-{name}.json').read_text())
    assert record['name'] == name
    assert record['removed'] and record['scratch_removed']
    assert any('--probe-marker' in a for a in create)
    assert '--network=none' in create and 'nvidia.com/gpu=0' in create
    if family == 'regdirect-pytest':
        assert not any('pip install' in a for a in create)
