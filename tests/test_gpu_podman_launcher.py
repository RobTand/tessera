"""Drive container policy through executable boundaries, including real TERM."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'tools/_flux_gpu_jobspec.py'
IMAGE = 'example.invalid/cuda@sha256:' + 'a' * 64


def controlled_runtime(tmp_path, *, hold=False):
    binary = tmp_path / 'bin'
    binary.mkdir()
    podman = binary / 'podman'
    podman.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys, threading
p=pathlib.Path(os.environ['CONTROL'])
a=sys.argv[1:]
with (p/'calls').open('a') as f: f.write(json.dumps(a)+'\\n')
if a[:2]==['image','inspect']: print('sha256:'+'c'*64+'\\t'+json.dumps([a[2]]))
elif a[0]=='info': print('true')
elif a[0]=='create':
 (p/'created').touch()
 pathlib.Path(a[a.index('--cidfile')+1]).write_text('b'*64)
 print('b'*64)
elif a[0]=='start':
 (p/'ready').touch()
 if os.environ.get('HOLD')=='1': threading.Event().wait()
 # The fake drives a known observable container affinity, not a verdict.
 mount=next(json.loads(l) for l in reversed((p/'calls').read_text().splitlines()) if json.loads(l)[0]=='create')
 scratch=mount[mount.index('--volume')+1].split(':')[0]
 pathlib.Path(scratch,'container-status').write_text('Cpus_allowed_list:\\t'+','.join(map(str,sorted(os.sched_getaffinity(0))))+'\\n')
elif a[0]=='rm': (p/'created').unlink(missing_ok=True)
elif a[:2]==['container','exists']: sys.exit(int(not (p/'created').exists()))
''')
    podman.chmod(0o755)
    flux = binary / 'flux'
    flux.write_text('#!/bin/sh\nprintf \"%s\\n\" \"$*\" > \"$CONTROL/flux-id-call\"\necho f123\n')
    flux.chmod(0o755)
    findmnt = binary / 'findmnt'
    findmnt.write_text('#!/bin/sh\nprintf "%s\\n" "${FILESYSTEM:-ext4}"\n')
    findmnt.chmod(0o755)
    env = {**os.environ, 'PATH': f'{binary}:{os.environ["PATH"]}',
           'CONTROL': str(tmp_path), 'FLUX_JOB_ID': 'f123', 'CUDA_VISIBLE_DEVICES': '0',
           'HOLD': str(int(hold))}
    out = tmp_path / 'out'
    command = [sys.executable, str(LAUNCHER), 'run', '--out', str(out),
               '--scratch-root', str(tmp_path / 'scratch'), '--',
               '--entrypoint', '/bin/sh', IMAGE, '-c', 'exit 0']
    return command, env, out


def test_real_term_removes_only_owned_container(tmp_path):
    command, env, out = controlled_runtime(tmp_path, hold=True)
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 30
        while not (tmp_path / 'ready').exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise AssertionError(process.communicate(timeout=5))
            time.sleep(0.01)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 143, (stdout, stderr)
        record, = [json.loads(p.read_text()) for p in out.glob('container-*.json')]
        assert record['removed'] and record['scratch_removed']
        assert not (tmp_path / 'created').exists()
        calls = [json.loads(line) for line in (tmp_path / 'calls').read_text().splitlines()]
        create = next(call for call in calls if call[0] == 'create')
        assert next(call for call in calls if call[0] == 'rm')[-1] == create[create.index('--name') + 1]
        assert '--cgroups=disabled' in create and '--network=none' in create
        assert create[create.index('--device') + 1] == 'nvidia.com/gpu=all'
        assert create[create.index('--cpuset-cpus') + 1] == ','.join(map(str, sorted(os.sched_getaffinity(0))))
        assert '--read-only' not in create
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_success_records_affinity_and_cleanup(tmp_path):
    command, env, out = controlled_runtime(tmp_path)
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    record, = [json.loads(p.read_text()) for p in out.glob('container-*.json')]
    assert record['container_affinity'] == sorted(os.sched_getaffinity(0))
    assert record['removed'] and record['scratch_removed']


def test_non_ext4_refused_before_container(tmp_path):
    command, env, out = controlled_runtime(tmp_path)
    result = subprocess.run(command, env={**env, 'FILESYSTEM': 'nfs4'}, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'ext4' in result.stderr
    assert not (tmp_path / 'created').exists()


@pytest.mark.parametrize('wrapper', [
    'routed_pair_oracle.sh', 'dense_fused_oracle.sh', 'mhc/mhc_probe.sh',
    'routed_census.sh', 'routed_fused_census.sh', 'routed_fused_tests.sh',
])
def test_oracle_wrapper_runs_owned_podman_policy(tmp_path, wrapper):
    _, env, out = controlled_runtime(tmp_path)
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    (model / 'model.safetensors.index.json').write_text('{}')
    packages = tmp_path / 'packages'
    packages.mkdir()
    for package in ('pytest', '_pytest', 'pluggy', 'iniconfig', 'packaging', 'pygments'):
        (packages / package).mkdir()
    (packages / 'py.py').touch()
    env.update(ORACLE_IMAGE=IMAGE, FLUX_SCRATCH=str(tmp_path / 'scratch'),
               DENSE_ORACLE_STUB=str(model), MHC_PROBE_MODEL=str(model),
               TEST_RUNNER_SP=str(packages))
    scripts = {
        'routed_pair_oracle.sh': ROOT / 'experiments/routed_pair_oracle.sh',
        'dense_fused_oracle.sh': ROOT / 'experiments/dense_fused_oracle.sh',
        'mhc/mhc_probe.sh': ROOT / 'experiments/mhc/mhc_probe.sh',
        'routed_census.sh': ROOT / 'experiments/routed_census.sh',
        'routed_fused_census.sh': ROOT / 'experiments/routed_fused_census.sh',
        'routed_fused_tests.sh': ROOT / 'experiments/routed_fused_tests.sh',
    }
    command = ['bash', str(scripts[wrapper]), str(ROOT), str(out)]
    if wrapper == 'routed_fused_census.sh':
        command.append(str(model))
    command.append('--probe-marker')
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(p.read_text()) for p in out.glob('container-*.json')]
    assert len(records) == 1
    assert records[0]['removed'] and records[0]['scratch_removed']
    calls = [json.loads(line) for line in (tmp_path / 'calls').read_text().splitlines()]
    create = next(call for call in calls if call[0] == 'create')
    assert '--probe-marker' in create
    assert str(ROOT) + ':/work:ro' in create
    assert 'nvidia.com/gpu=all' in create
    if wrapper == 'routed_fused_tests.sh':
        assert (out / 'runner-sp/pygments').is_dir()


@pytest.mark.parametrize('wrapper,expected_containers', [
    ('t4_code/run_in_image.sh', 1), ('mla_prefill/mask_row.sh', 1),
    ('mla_prefill/d1_row.sh', 3), ('glm53_508_graph_qual/mhc_split_run.sh', 1),
])
def test_kernel_wrapper_runs_owned_podman_policy(tmp_path, wrapper, expected_containers):
    _, env, out = controlled_runtime(tmp_path)
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    profiler = tmp_path / 'profiler'
    profiler.mkdir()
    (profiler / 'ncu').write_text('#!/bin/sh\nexit 0\n')
    (profiler / 'ncu').chmod(0o755)
    env.update(ORACLE_IMAGE=IMAGE, MLA_IMAGE=IMAGE, FLUX_SCRATCH=str(tmp_path / 'scratch'),
               T4_SOURCE_ROOT=str(inputs), T4_ACTIVATION_ROOT=str(inputs), NCU_ROOT=str(profiler))
    if wrapper.startswith('t4_code/'):
        args = [str(ROOT), str(out), 'fp4_arithmetic_attest.py', '--probe-marker']
    elif wrapper.startswith('glm53_'):
        args = [IMAGE, str(out)]
    else:
        args = [str(out)]
    scripts = {
        't4_code/run_in_image.sh': ROOT / 'experiments/t4_code/run_in_image.sh',
        'mla_prefill/mask_row.sh': ROOT / 'experiments/mla_prefill/mask_row.sh',
        'mla_prefill/d1_row.sh': ROOT / 'experiments/mla_prefill/d1_row.sh',
        'glm53_508_graph_qual/mhc_split_run.sh': ROOT / 'experiments/glm53_508_graph_qual/mhc_split_run.sh',
    }
    result = subprocess.run(['bash', str(scripts[wrapper]), *args],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(p.read_text()) for p in out.glob('container-*.json')]
    assert len(records) == expected_containers
    assert all(r['removed'] and r['scratch_removed'] for r in records)


def test_submit_collects_native_terminal_records(tmp_path, monkeypatch):
    import runpy
    module = runpy.run_path(str(LAUNCHER))
    command, env, out = controlled_runtime(tmp_path)
    binary = tmp_path / 'bin'
    flux = binary / 'flux'
    flux.write_text('''#!/usr/bin/env python3
import json,os,pathlib,sys
args=sys.argv[1:]
with pathlib.Path(os.environ['CONTROL'],'flux-calls').open('a') as f: f.write(json.dumps(args)+'\\n')
if args[:2]==['resource','list']: print('free sparky sparklina dl380g10')
elif args[0]=='submit': print('f123')
elif args[:2]==['job','status']: print('{"waitstatus":0,"exception_occurred":false}')
elif args[:2]==['job','eventlog']: print(json.dumps({'name':'finish','context':{'status':0}})); print(json.dumps({'name':'clean'}))
elif args[:2]==['job','info']: print(json.dumps({'execution':{'R_lite':[{'rank':'1','children':{'core':'0','gpu':'0'}}],'nodelist':['sparky']}}))
''')
    flux.chmod(0o755)
    ssh = binary / 'ssh'
    ssh.write_text('#!/bin/sh\ncase "$*" in *query-gpu*) echo 0;; esac\n')
    ssh.chmod(0o755)
    for key in ('PATH', 'CONTROL'):
        monkeypatch.setenv(key, env[key])
    argv = ['flux', 'submit', '--env=-*', '--resources-shape=slot/[core=1;memory=8;gpu=1]',
            '-n1', '-t', '10m', '--', '/bin/true']
    assert module['submit_and_collect'](argv, out, cores=1, memory_gib=8, walltime='10m') == 0
    receipt = json.loads((out / 'receipt.json').read_text())
    assert receipt['job_id'] == 'f123' and receipt['returncode'] == 0
    assert (out / 'R.json').is_file() and (out / 'eventlog.jsonl').is_file()
    calls = [json.loads(line) for line in (tmp_path / 'flux-calls').read_text().splitlines()]
    assert sum(call[0] == 'submit' for call in calls) == 1
    assert ['job', 'status', '--json', '--exception-exit-code=125', 'f123'] in calls
    assert '--env=-*' in next(call for call in calls if call[0] == 'submit')


def test_submit_refuses_caps_before_any_native_boundary(tmp_path):
    import runpy
    module = runpy.run_path(str(LAUNCHER))
    with pytest.raises(ValueError, match='10 minutes'):
        module['submit_and_collect'](['flux', 'submit'], tmp_path / 'out', cores=4,
                                     memory_gib=8, walltime='45m')
    assert not (tmp_path / 'out').exists()


def test_launcher_converts_the_native_job_id_at_one_boundary(tmp_path):
    command, env, out = controlled_runtime(tmp_path)
    env['FLUX_JOB_ID'] = 'ƒ123'
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    record, = [json.loads(p.read_text()) for p in out.glob('container-*.json')]
    assert Path(record['scratch']).parent.name == 'f123'
    assert (tmp_path / 'flux-id-call').read_text().strip() == 'job id --to=f58plain ƒ123'
