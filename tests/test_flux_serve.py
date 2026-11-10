"""CPU-only serving policy tests. Docker and HTTP are controlled boundaries."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
from box_artifacts import FLUX_RENDER_ROOT

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'tools/_flux_serve.py'
IMAGE = 'registry/serve@sha256:' + 'a' * 64
CID = 'b' * 64


def policy():
    spec = importlib.util.spec_from_file_location('_flux_serve_test', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Docker:
    def __init__(self):
        self.calls = []
        self.exists = False
        self.running = True
        self.image = 'sha256:' + 'c' * 64
        self.digests = [IMAGE]
        self.running_digests = None
        self.absent = False
        self.remove_fails = False

    def __call__(self, *args, check=True):
        self.calls.append(list(args))
        if args[:2] == ('image', 'inspect'):
            if self.absent:
                raise RuntimeError('not found')
            digests = self.digests if args[2] == IMAGE or self.running_digests is None else self.running_digests
            return json.dumps([{'Id': self.image, 'RepoDigests': digests}])
        if args[0] == 'run':
            self.exists = True
            return CID
        if args[:2] == ('container', 'inspect'):
            if not self.exists:
                raise RuntimeError('not found')
            return json.dumps([{'Id': CID, 'Image': self.image, 'State': {
                'Running': self.running, 'Pid': 123, 'StartedAt': '2026-11-10T00:00:00Z'}}])
        if args[0] == 'rm' and not self.remove_fails:
            self.exists = False
        if args[0] == 'ps':
            return CID if self.exists else ''
        return ''


@pytest.fixture
def setup(tmp_path, monkeypatch):
    m = policy()
    monkeypatch.setenv('FLUX_JOB_ID', 'fABC')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('FLUX_MEMORY_GIB', '59')
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'weights').write_bytes(b'weights')
    request = {'model': str(model), 'plugin': None, 'port': 8000, 'util': '0.45',
               'options': ['-v', f'{model}:{model}:ro'],
               'image': IMAGE, 'command': [str(model), '--gpu-memory-utilization', '0.45']}
    spec = m.make_spec(request)
    peak = tmp_path / 'memory.peak'
    peak.write_text('123456\n')
    d = Docker()
    serve = m.Serve(spec, tmp_path / 'receipt.json', docker=d,
                    decimal_id=lambda _: '12345', cpus=lambda: '5-6,15',
                    models=lambda url, timeout: {'data': [{'id': 'kl-target'}]},
                    peak_path=lambda inspection: peak)
    return m, serve, d, request, tmp_path


def test_exact_run_flags_and_receipt(setup):
    m, s, d, request, root = setup
    s.start(request)
    run = next(c for c in d.calls if c[0] == 'run')
    for value in ['-d', '--init', '--pull=never', '--name=flux-12345',
                  '--label=flux.jobid=12345', '--cpuset-cpus=5-6,15',
                  '--memory=55g', '--memory-swap=55g', '--gpus=device=0',
                  '--publish=127.0.0.1:8000:8000',
                  f'--user={os.getuid()}:{os.getgid()}', IMAGE]:
        assert value in run
    assert not any('fABC' in arg for arg in run)
    s.stop()
    receipt = json.loads((root / 'receipt.json').read_text())
    assert receipt['container_id'] == CID
    assert receipt['image_id'] == d.image
    assert receipt['repo_digest'] == IMAGE
    assert receipt['started_at'] == '2026-11-10T00:00:00Z'
    assert receipt['models'] == {'data': [{'id': 'kl-target'}]}
    assert receipt['spec_id'] == m.spec_id(s.spec)
    assert receipt['declared_serve_memory_gib'] == 55
    assert receipt['declared_job_memory_gib'] == 59
    assert receipt['container_memory_peak_bytes'] == 123456
    assert receipt['cpus_allowed_list'] == '5-6,15'
    assert receipt['removed'] is True
    assert [c[0] for c in d.calls].count('stop') == 1


@pytest.mark.parametrize('case,reason', [
    ('outside', 'OUTSIDE_FLUX_JOB'), ('gpu', 'GPU_ALLOCATION_MISSING'),
    ('absent', 'IMAGE_DIGEST_ABSENT'), ('floating', 'IMAGE_DIGEST_REQUIRED'),
    ('digest', 'RUNNING_DIGEST_MISMATCH'), ('spec', 'SPEC_ID_MISMATCH'),
    ('health', 'HEALTH_DEADLINE'), ('dead', 'CONTAINER_EXITED'),
    ('bound', 'JOB_MEMORY_BOUND'), ('peak', 'CONTAINER_MEMORY_PEAK'),
    ('cleanup', 'CLEANUP_UNVERIFIED'), ('payload', 'CONTAINER_OPTION'),
])
def test_named_refusals(setup, monkeypatch, case, reason):
    m, s, d, request, root = setup
    if case == 'outside': monkeypatch.delenv('FLUX_JOB_ID')
    if case == 'gpu': monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    if case == 'absent': d.absent = True
    if case == 'floating': request['image'] = 'serve:latest'
    if case == 'digest': d.running_digests = ['other@sha256:' + 'd' * 64]
    if case == 'spec': request['command'].append('--enforce-eager')
    if case == 'health': s.models = lambda *a: (_ for _ in ()).throw(OSError('unhealthy'))
    if case == 'dead': d.running = False
    if case == 'bound': monkeypatch.setenv('FLUX_MEMORY_GIB', '54')
    if case == 'peak': (root / 'memory.peak').unlink()
    if case == 'cleanup': d.remove_fails = True
    if case == 'payload': request['options'].extend(['--privileged'])
    with pytest.raises(m.Refused, match=reason):
        try:
            s.start(request, health_seconds=0)
            if case == 'cleanup': s.stop()
        finally:
            if case != 'cleanup': s.stop()
    assert not d.exists or case == 'cleanup'
    if case == 'digest':
        assert any(call[0] == 'run' for call in d.calls), 'running digest must be checked after launch'
        assert ['image', 'inspect', d.image] in d.calls


def test_container_dies_during_scoring(setup):
    m, s, d, request, root = setup
    s.start(request)
    d.running = False
    with pytest.raises(m.Refused, match='CONTAINER_EXITED'):
        s.check_running()
    s.stop()
    assert not d.exists


def test_spec_is_neutral_and_binds_model_bytes(setup):
    m, s, d, request, root = setup
    request['repository'] = 'anything'
    assert m.make_spec(request) == s.spec
    (Path(request['model']) / 'weights').write_bytes(b'changed')
    assert m.spec_id(m.make_spec(request)) != m.spec_id(s.spec)


def test_peak_resolves_docker_process_cgroup(tmp_path):
    m = policy()
    proc = tmp_path / 'proc/123'
    proc.mkdir(parents=True)
    (proc / 'cgroup').write_text('0::/system.slice/docker-' + CID + '.scope\n')
    expected = tmp_path / ('cgroups/system.slice/docker-' + CID + '.scope/memory.peak')
    assert m.container_peak_path({'Id': CID, 'State': {'Pid': 123}},
                                 proc_root=tmp_path / 'proc', cgroup_root=tmp_path / 'cgroups') == expected


def test_wrapper_outside_flux_refuses_before_docker(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('FLUX_')}
    result = subprocess.run(['bash', str(ROOT / 'experiments/serve_smoke_graph.sh'), '/absent'],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'OUTSIDE_FLUX_JOB' in result.stderr


def test_real_term_and_exit_cleanup(tmp_path):
    """A real child receives TERM after a readiness file, using fake executables."""
    m = policy()
    m  # The subprocess fixture is shared with the CLI smoke below.
    fake = tmp_path / 'bin'
    fake.mkdir()
    state = tmp_path / 'state'
    state.mkdir()
    docker = fake / 'docker'
    docker.write_text('''#!/usr/bin/env python3
import json,os,pathlib,sys
p=pathlib.Path(os.environ['FAKE_STATE']); a=sys.argv[1:]
with (p/'calls').open('a') as f: f.write(json.dumps(a)+'\\n')
if a[:2]==['image','inspect']: print(json.dumps([{'Id':'sha256:cccc','RepoDigests':[os.environ['FAKE_IMAGE']]}]))
elif a[0]=='run':
 (p/'exists').touch(); print('b'*64)
elif a[:2]==['container','inspect']:
 if not (p/'exists').exists(): sys.exit(1)
 print(json.dumps([{'Id':'b'*64,'Image':'sha256:cccc','State':{'Running':not (p/'dead').exists(),'Pid':os.getpid(),'StartedAt':'now'}}]))
elif a[0]=='rm': (p/'exists').unlink(missing_ok=True)
elif a[0]=='ps': print('b'*64 if (p/'exists').exists() else '')
''')
    docker.chmod(0o755)
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'w').write_text('w')
    req = {'model': str(model), 'plugin': None, 'port': 8000, 'util': '0.45',
           'options': ['-v', f'{model}:{model}:ro'], 'image': IMAGE,
           'command': [str(model), '--gpu-memory-utilization', '0.45']}
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps(m.make_spec(req)))
    peak = tmp_path / 'peak'
    peak.write_text('42')
    fixture = tmp_path / 'driver.py'
    fixture.write_text('''import importlib.util,json,os,pathlib,sys
s=importlib.util.spec_from_file_location('policy',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
m.container_peak_path=lambda info: pathlib.Path(os.environ['FAKE_PEAK'])
def models(url,timeout):
 if os.environ['MODE']=='startup-term': raise OSError('waiting')
 return {'data':[]}
m.fetch_models=models
m.decimal_job_id=lambda value: '12345'
sys.exit(m.supervise(json.loads(pathlib.Path(sys.argv[2]).read_text()),sys.argv[3],sys.argv[4:]))
''')
    score = tmp_path / 'score.sh'
    score.write_text('''#!/bin/bash
set -euo pipefail
source "$LIB"
flux_serve_start --model "$MODEL" --port 8000 --util 0.45 -- -v "$MODEL:$MODEL:ro" "$FAKE_IMAGE" "$MODEL" --gpu-memory-utilization 0.45
: > "$FAKE_STATE/ready"
if [ "$MODE" = term ] || [ "$MODE" = dead ]; then read -r line < "$FAKE_STATE/fifo"; fi
if [ "$MODE" = score-failed ]; then exit 17; fi
flux_serve_stop
''')
    os.mkfifo(state / 'fifo')
    env = {**os.environ, 'PATH': str(fake) + ':' + os.environ['PATH'], 'FAKE_STATE': str(state),
           'FAKE_IMAGE': IMAGE, 'FAKE_PEAK': str(peak), 'FLUX_JOB_ID': 'fABC',
           'CUDA_VISIBLE_DEVICES': '0', 'FLUX_MEMORY_GIB': '59', 'FLUX_SCRATCH': str(tmp_path / 'scratch'), 'MODEL': str(model),
           'LIB': str(ROOT / 'experiments/flux_serve.sh')}
    for mode in ['exit', 'term', 'dead', 'startup-term', 'score-failed']:
        (state / 'ready').unlink(missing_ok=True)
        (state / 'dead').unlink(missing_ok=True)
        proc = subprocess.Popen([sys.executable, str(fixture), str(MODULE), str(spec),
                                 str(tmp_path / f'{mode}.json'), 'bash', str(score)], env={**env, 'MODE': mode})
        deadline = time.monotonic() + 20
        try:
            readiness = state / ('exists' if mode == 'startup-term' else 'ready')
            while not readiness.exists():
                assert proc.poll() is None, 'child exited before readiness'
                assert time.monotonic() < deadline, 'readiness deadline'
                time.sleep(.02)
            if mode in {'term', 'startup-term'}: proc.send_signal(signal.SIGTERM)
            if mode == 'dead': (state / 'dead').touch()
            assert proc.wait(timeout=20) == {'exit': 0, 'term': 143, 'dead': 2, 'startup-term': 143, 'score-failed': 17}[mode]
        finally:
            if proc.poll() is None:
                proc.kill(); proc.wait()
        assert not (state / 'exists').exists()
        receipt = json.loads((tmp_path / f'{mode}.json').read_text())
        assert receipt['removed']
        if mode != 'startup-term':
            assert receipt['container_memory_peak_bytes'] == 42


def test_decimal_job_id_uses_native_conversion(tmp_path, monkeypatch):
    m = policy()
    sys.path.insert(0, str(ROOT / 'tools'))
    import _flux_gpu_jobspec
    client = tmp_path / 'flux'
    client.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CONVERSION_CALL"\necho 12345\n')
    client.chmod(0o755)
    monkeypatch.setenv('CONVERSION_CALL', str(tmp_path / 'call'))
    monkeypatch.setattr(_flux_gpu_jobspec, 'flux_client', lambda: (str(client), None))
    assert m.decimal_job_id('fABC') == '12345'
    assert (tmp_path / 'call').read_text().splitlines() == ['job', 'id', '--to=dec', 'fABC']


def test_render_declares_whole_serve_and_client(setup, capsys):
    m, s, d, request, root = setup
    spec = root / 'spec.json'
    spec.write_text(json.dumps(s.spec))
    assert m.main(['render', '--spec', str(spec), '--out', str(FLUX_RENDER_ROOT / 'owned-serving-run'),
                   '--', 'experiments/serve_smoke_graph.sh', request['model']]) == 0
    command = capsys.readouterr().out
    assert '--resources-shape=slot/[core=4;memory=59;gpu=1]' in command
    assert '--requires' not in command
    assert '--env=-*' in command
    assert '--env=FLUX_F58_FORCE_ASCII=1' in command
    assert '--env=FLUX_MEMORY_GIB=59' in command


def test_long_gpu_run_requires_duration_approval(setup):
    m, s, d, request, root = setup
    arguments = dict(out=str(FLUX_RENDER_ROOT / 'owned-serving-run'), walltime='11m', spec_path=root / 'spec.json')
    command = [str(ROOT / 'experiments/serve_smoke_graph.sh'), request['model']]
    with pytest.raises(m.Refused, match='SERVE_DURATION_APPROVAL_REQUIRED'):
        m.render_serving(command, s.spec, **arguments)
    assert m.render_serving(command, s.spec, duration_approval='fleet-master test decision', **arguments)


def test_spec_refusal_still_records_the_final_client_peak(tmp_path, monkeypatch):
    m = policy()
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'gpu_memory_utilization': '.35'}))
    (tmp_path / 'cgroup').write_text('0::/controlled-job\n')
    group = tmp_path / 'cgroups/controlled-job'
    group.mkdir(parents=True)
    (group / 'memory.peak').write_text('987654')
    real_path = m.Path
    def observed_path(value):
        if value == '/proc/self/cgroup': return tmp_path / 'cgroup'
        if value == '/sys/fs/cgroup': return tmp_path / 'cgroups'
        return real_path(value)
    monkeypatch.setattr(m, 'Path', observed_path)
    monkeypatch.setenv('TESSERA_SERVE_EXPECTED_SPEC_ID', 'different')
    with pytest.raises(m.Refused, match='SPEC_ID_MISMATCH'):
        m.main(['wrapper', '--spec', str(spec), '--out', str(tmp_path), '--', 'unreached-score'])
    assert json.loads((tmp_path / 'job-memory.json').read_text()) == {'memory_peak_bytes': 987654}


def test_model_mount_cannot_be_writable_or_duplicated(setup):
    m, s, d, request, root = setup
    s.start(request)
    run = next(c for c in d.calls if c[0] == 'run')
    mounts = [run[i + 1] for i, arg in enumerate(run[:-1]) if arg in {'-v', '--volume'}]
    assert mounts.count(request['model'] + ':' + request['model'] + ':ro') == 1
    s.stop()


def test_model_alias_mount_must_be_read_only(setup):
    m, s, d, request, root = setup
    request['options'] += ['-v', request['model'] + ':/writable-alias']
    s.spec = m.make_spec(request)
    with pytest.raises(m.Refused, match='MODEL_MOUNT_READ_ONLY'):
        s.start(request)
    assert not any(call[0] == 'run' for call in d.calls)


def test_campaign_dispatch_selects_exact_plan_and_waits(setup, monkeypatch):
    m, s, d, request, root = setup
    monkeypatch.delenv('FLUX_JOB_ID')
    spec = root / 'spec.json'
    spec.write_text(json.dumps(s.spec))
    plan = {'wrapper': 'serve_smoke_graph.sh', 'arguments': [request['model']],
            'spec': str(spec), 'out': str(FLUX_RENDER_ROOT / 'owned-run'), 'walltime': '10m'}
    plans = root / 'plans.json'
    plans.write_text(json.dumps([plan]))
    monkeypatch.setenv('TESSERA_SERVE_PLANS', str(plans))
    seen = []
    monkeypatch.setattr(m, 'submit_and_collect', lambda argv, out: seen.append((argv, out)) or 0)
    assert m.dispatch([str(ROOT / 'experiments/serve_smoke_graph.sh'), request['model']]) == 0
    argv, out = seen[0]
    assert '--resources-shape=slot/[core=4;memory=59;gpu=1]' in argv
    assert out == plan['out']
    with pytest.raises(m.Refused, match='SERVE_PLAN_MISSING'):
        m.dispatch([str(ROOT / 'experiments/serve_smoke_graph.sh'), 'other-model'])


def test_campaign_cannot_nest_inside_gpu_job(setup):
    m, s, d, request, root = setup
    with pytest.raises(m.Refused, match='CAMPAIGN_INSIDE_FLUX_JOB'):
        m.dispatch([str(ROOT / 'experiments/serve_smoke_graph.sh'), request['model']])


def test_failed_run_still_cleans_registered_name(setup):
    m, s, d, request, root = setup
    original = d.__call__
    def docker(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[0] == 'run':
            raise m.Refused('DOCKER_COMMAND_FAILED: run')
        return result
    s.docker = docker
    with pytest.raises(m.Refused, match='DOCKER_COMMAND_FAILED'):
        try:
            s.start(request)
        finally:
            s.stop()
    assert not d.exists
    assert ['rm', '-f', 'flux-12345'] in d.calls


def test_preexisting_name_is_never_removed(setup):
    m, s, d, request, root = setup
    d.exists = True
    with pytest.raises(m.Refused, match='CONTAINER_NAME_TAKEN'):
        try:
            s.start(request)
        finally:
            s.stop()
    assert d.exists
    assert not any(c[0] in {'run', 'stop', 'rm'} for c in d.calls)


@pytest.mark.parametrize('case,reason', [('clean', 'NATIVE_TERMINAL_EVIDENCE_MISSING'),
                                        ('gpu', 'GPU_ALLOCATION_MISSING'),
                                        ('receipt', 'SERVE_RECEIPT_INCOMPLETE'),
                                        ('spec', 'SPEC_ID_MISMATCH'),
                                        ('jobpeak', 'JOB_MEMORY_PEAK_MISSING'),
                                        ('external', 'EXTERNAL_GPU_LOAD')])
def test_native_collection_requires_its_own_complete_evidence(tmp_path, monkeypatch, case, reason):
    m = policy()
    sys.path.insert(0, str(ROOT / 'tools'))
    import _flux_gpu_jobspec
    client = tmp_path / 'flux'
    client.write_text('''#!/usr/bin/env python3
import json,os,pathlib,sys
out=pathlib.Path(os.environ['CONTROL_OUT']); case=os.environ['CONTROL_CASE']; a=sys.argv[1:]
with (out/'calls').open('a') as f: f.write(json.dumps(a)+'\\n')
if a[0]=='submit':
 r={'returncode':0,'removed':case!='receipt','scratch_removed':True,'container_memory_peak_bytes':17,'spec_id':'different' if case=='spec' else 'expected'}
 (out/'serve-receipt.json').write_text(json.dumps(r))
 if case!='jobpeak': (out/'job-memory.json').write_text('{"memory_peak_bytes":42}')
 print('fABC')
elif a[:2]==['job','status']:
 assert (out/'jobid').read_text().strip()=='fABC'
 print(json.dumps({'waitstatus':0,'exception_occurred':False}))
elif a[:2]==['job','wait-event']:
 if case=='clean': sys.exit(1)
 print('{"name":"clean"}')
elif a[:2]==['job','eventlog']:
 if case!='clean': print('{"name":"clean"}')
elif a[:2]==['job','info']:
 children={'core':'0-3'}
 if case!='gpu': children['gpu']='0'
 print(json.dumps({'execution':{'R_lite':[{'children':children}]}}))
else: sys.exit(92)
''')
    client.chmod(0o755)
    ssh = tmp_path / 'ssh'
    ssh.write_text('#!/bin/sh\n[ "$CONTROL_CASE" != external ] || echo external-gpu-process\nexit 0\n')
    ssh.chmod(0o755)
    monkeypatch.setenv('PATH', str(tmp_path) + ':' + os.environ['PATH'])
    argv = ['flux', 'submit', '--env=TESSERA_SERVE_EXPECTED_SPEC_ID=expected', '--', 'reviewed-wrapper']
    for control in ['positive', case]:
        out = tmp_path / control
        monkeypatch.setenv('CONTROL_CASE', control)
        monkeypatch.setattr(_flux_gpu_jobspec, 'flux_client', lambda: (
            str(client), {**os.environ, 'CONTROL_OUT': str(out), 'CONTROL_CASE': control}))
        if control == 'positive':
            assert m.submit_and_collect(argv, str(out)) == 0
        else:
            with pytest.raises(m.Refused, match=reason):
                m.submit_and_collect(argv, str(out))
        if control == 'external':
            assert not (out / 'calls').exists(), 'external load must refuse before submission'
            continue
        calls = [json.loads(line) for line in (out / 'calls').read_text().splitlines()]
        assert [call[0] for call in calls].count('submit') == 1
        assert [call[:2] for call in calls[1:]] == [
            ['job', 'status'], ['job', 'wait-event'], ['job', 'eventlog'], ['job', 'info']]
