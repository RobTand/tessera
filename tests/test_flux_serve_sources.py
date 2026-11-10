"""Bind every mounted plugin and refuse a failed plugin install before vLLM."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from tools import _flux_serve as policy

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'registry/serve@sha256:' + 'a' * 64


def checkout(path, text):
    path.mkdir()
    subprocess.run(['git', 'init', '-q', str(path)], check=True)
    (path / 'plugin').write_text(text)
    subprocess.run(['git', '-C', str(path), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(path), '-c', 'user.name=Test', '-c',
                    'user.email=test@example.invalid', 'commit', '-qm', 'fixture'], check=True)
    return path


def request(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'weights').write_bytes(b'independent model')
    primary = checkout(tmp_path / 'primary', 'primary')
    supporting = checkout(tmp_path / 'supporting', 'supporting')
    return {'model': str(model), 'plugin': str(primary), 'supporting_plugins': [str(supporting)],
            'port': 8000, 'util': '.35', 'image': IMAGE,
            'options': ['-v', f'{primary}:/gb:ro', '-v', f'{supporting}:/tessera:ro'],
            'command': ['serve', str(model), '--gpu-memory-utilization', '.35']}


def test_supporting_plugin_bytes_are_bound_and_checkout_locations_are_neutral(tmp_path):
    req = request(tmp_path)
    first = policy.make_spec(req)
    for field, path in [('plugin', req['plugin']), ('supporting_plugins', req['supporting_plugins'][0])]:
        identity = first[field][0] if field == 'supporting_plugins' else first[field]
        assert identity == {
            'commit': subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'], text=True).strip(),
            'tree': subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD^{tree}'], text=True).strip()}
    old = req['supporting_plugins'][0]
    moved = tmp_path / 'moved'
    subprocess.run(['git', 'clone', '-q', old, str(moved)], check=True)
    req['supporting_plugins'] = [str(moved)]
    req['options'] = [arg.replace(old, str(moved)) for arg in req['options']]
    assert policy.make_spec(req) == first
    (moved / 'plugin').write_text('changed supporting code')
    subprocess.run(['git', '-C', str(moved), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(moved), '-c', 'user.name=Test', '-c',
                    'user.email=test@example.invalid', 'commit', '-qm', 'change'], check=True)
    assert policy.spec_id(policy.make_spec(req)) != policy.spec_id(first)


def test_supporting_plugin_writable_mount_refuses_before_run(tmp_path, monkeypatch):
    req = request(tmp_path)
    req['options'][-1] = req['options'][-1].removesuffix(':ro')
    monkeypatch.setenv('FLUX_JOB_ID', 'fABC')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('FLUX_MEMORY_GIB', '47')
    calls = []
    def docker(*args, **kwargs):
        calls.append(args)
        if args[:2] == ('image', 'inspect'):
            return json.dumps([{'Id': 'sha256:image', 'RepoDigests': [IMAGE]}])
        if args[0] == 'ps':
            return ''
        raise AssertionError('Docker reached a launch before mount validation')
    serve = policy.Serve(policy.make_spec(req), tmp_path / 'receipt.json', docker=docker,
                         decimal_id=lambda _: '12345', cpus=lambda: '5')
    with pytest.raises(policy.Refused, match='PLUGIN_MOUNT_READ_ONLY'):
        serve.start(req)
    assert not any(call[0] == 'run' for call in calls)


@pytest.mark.parametrize('wrapper', ['tessera_plugin_served.sh', 'decode_regime_kl.sh',
                                     'window_gemv_latency.sh', 'gridbook_lane_served.sh'])
def test_actual_plugin_startup_never_falls_back_after_install_failure(tmp_path, wrapper):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    shutil.copyfile(ROOT / 'experiments' / wrapper, scripts / wrapper)
    (scripts / 'runtime_image.sh').write_text('runtime_image_pin() { echo "$IMAGE"; }\nruntime_image_require() { :; }\n')
    (scripts / 'build_identity.sh').write_text('build_identity_docker_env() { :; }\n')
    (scripts / 'flux_serve.sh').write_text('''flux_serve_start() {
 "$CAPTURE_PY" -c 'import json,os,sys; open(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))' "$@"
 return 19
}
''')
    fake = tmp_path / 'bin'
    fake.mkdir()
    for name, body in {
        'mkdir': 'exit 0', 'chmod': 'exit 0',
        'python3': 'exit 0', 'mktemp': 'echo "$PLUGIN_TEMP"',
        'cp': 'exit 0',
        'pip': 'touch "$PIP_CALLED"; exit 23',
        'vllm': 'touch "$VLLM_CALLED"; exit 0',
    }.items():
        path = fake / name
        path.write_text('#!/bin/sh\n' + body + '\n')
        path.chmod(0o755)
    model = tmp_path / 'model'
    model.mkdir()
    project = tmp_path / 'project'
    project.mkdir()
    env = os.environ | {'PATH': f'{fake}:/usr/bin:/bin', 'IMAGE': IMAGE,
                        'TS': str(tmp_path / 'tessera'), 'GB': str(tmp_path / 'gridbook'),
                        'RUNS': str(tmp_path / 'runs'), 'CAPTURE_PY': sys.executable,
                        'CAPTURE': str(tmp_path / 'argv.json'), 'PLUGIN_TEMP': str(project),
                        'PIP_CALLED': str(tmp_path / 'pip-called'),
                        'VLLM_CALLED': str(tmp_path / 'vllm-called')}
    captured = subprocess.run(['bash', str(scripts / wrapper), str(model), 'fixture'],
                              env=env, text=True, capture_output=True)
    assert captured.returncode == 19, captured.stderr
    arguments = json.loads((tmp_path / 'argv.json').read_text())
    assert arguments[-2] == '-c'
    if wrapper == 'gridbook_lane_served.sh':
        assert arguments[arguments.index('--supporting-plugin') + 1] == env['TS']
    executed = subprocess.run(['bash', '-c', arguments[-1]], env=env, capture_output=True, text=True)
    assert (tmp_path / 'pip-called').exists(), executed.stderr
    assert executed.returncode == 23, executed.stderr
    assert not (tmp_path / 'vllm-called').exists()
