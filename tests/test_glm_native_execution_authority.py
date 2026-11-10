"""Launch evidence states only the Flux context it observed."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def test_encode_launch_records_flux_environment_without_claiming_admission(tmp_path, monkeypatch):
    from experiments import run_glm_native_construction as driver
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    request = tmp_path / 'request.json'
    request.write_text(json.dumps({'runtime_image': 'image@sha256:fixture'}))
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['driver', '--request', str(request), '--out', str(out),
                                     '--stage', 'encode', '--cpus', '1'])
    monkeypatch.setenv('FLUX_JOB_ID', 'unit-context')
    monkeypatch.setattr(driver.os, 'sched_getaffinity', lambda pid: {1})
    monkeypatch.setattr(driver.subprocess, 'check_output', lambda argv: json.dumps(
        [{'RepoDigests': ['image@sha256:fixture'], 'Id': 'image-id'}]).encode())
    monkeypatch.setattr(driver.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=7))
    assert driver.main() == 7
    launch = json.loads((out / 'launch.json').read_text())
    assert launch['execution_authority'] == 'Flux job unit-context (environment value, not an admission token)'
