"""Legacy Docker paths refuse in a Flux job, before doing any work."""
import os
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('job_id', ['fABC', ''])
def test_legacy_entry_refuses_inside_flux(tmp_path, job_id):
    fake = tmp_path / 'bin'
    fake.mkdir()
    docker = fake / 'docker'
    docker.write_text('#!/bin/sh\necho legacy-docker-invoked >&2\nexit 42\n')
    docker.chmod(0o755)
    result = subprocess.run(['bash', str(ROOT / 'experiments/container_limits_probe.sh')],
                            env={**os.environ, 'FLUX_JOB_ID': job_id, 'PATH': str(fake) + ':' + os.environ['PATH']},
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert 'ROOT_DOCKER_REQUIRES_REVIEWED_SERVING_WRAPPER' in result.stderr
    assert 'anything else goes to the lead' in result.stderr
    assert 'legacy-docker-invoked' not in result.stderr
