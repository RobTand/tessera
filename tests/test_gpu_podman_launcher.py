"""Drive container policy through executable boundaries, including real TERM."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest

from gpu_podman_fixtures import controlled_runtime, controlled_submission, IMAGE, ROOT, LAUNCHER


def owned_environment_names():
    import runpy
    return sorted({'TMPDIR', 'CUDA_VISIBLE_DEVICES', *runpy.run_path(str(LAUNCHER))['THREADS']})


@pytest.mark.parametrize('name', owned_environment_names())
@pytest.mark.parametrize('flag', ['-e', '--env', '--env='])
def test_payload_cannot_override_launcher_environment(tmp_path, name, flag):
    command, env, out = controlled_runtime(tmp_path)
    at = command.index(IMAGE)
    option = [flag + name + '=untrusted'] if flag.endswith('=') else [flag, name + '=untrusted']
    command[at:at] = option
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2, result.stderr
    assert name in result.stderr and 'owned' in result.stderr
    assert not (tmp_path / 'created').exists()


def test_wrapper_tmpdir_is_owned_once_by_launcher(tmp_path):
    _, env, out = controlled_runtime(tmp_path)
    env.update(ORACLE_IMAGE=IMAGE, FLUX_SCRATCH=str(tmp_path / 'scratch'))
    result = subprocess.run(['bash', str(ROOT / 'experiments/routed_pair_oracle.sh'), str(ROOT), str(out)],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in (tmp_path / 'calls').read_text().splitlines()]
    create = next(call for call in calls if call[0] == 'create')
    tmpdirs = [create[i + 1] for i, arg in enumerate(create[:-1])
               if arg in {'--env', '-e'} and create[i + 1].startswith('TMPDIR=')]
    assert tmpdirs == ['TMPDIR=/tessera-scratch/tmp'], tmpdirs
    assert str(out) + ':' + str(out) in create

@pytest.mark.parametrize('signum', [signal.SIGTERM, signal.SIGINT])
def test_real_term_removes_only_owned_container(tmp_path, signum):
    command, env, out = controlled_runtime(tmp_path, hold=True)
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 30
        while not (tmp_path / 'ready').exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise AssertionError(process.communicate(timeout=5))
            time.sleep(0.01)
        process.send_signal(signum)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 128 + signum, (stdout, stderr)
        record, = [json.loads(p.read_text()) for p in out.glob('container-*.json')]
        assert record['removed'] and record['scratch_removed']
        assert not (tmp_path / 'created').exists()
        calls = [json.loads(line) for line in (tmp_path / 'calls').read_text().splitlines()]
        create = next(call for call in calls if call[0] == 'create')
        assert next(call for call in calls if call[0] == 'rm')[-1] == create[create.index('--name') + 1]
        assert '--cgroups=disabled' in create and '--network=none' in create
        assert create[create.index('--device') + 1] == 'nvidia.com/gpu=0'
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
    assert 'nvidia.com/gpu=0' in create
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


def test_owned_tmpfs_payload_option_is_forwarded(tmp_path):
    command, env, out = controlled_runtime(tmp_path)
    image_at = command.index(IMAGE)
    command[image_at:image_at] = ['--tmpfs', '/pbtmp:rw,exec,size=8g']
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (tmp_path / 'calls').read_text().splitlines()]
    create = next(c for c in calls if c[0] == 'create')
    assert create[create.index('--tmpfs') + 1] == '/pbtmp:rw,exec,size=8g'
    record, = [json.loads(p.read_text()) for p in out.glob('container-*.json')]
    assert record['removed'] and record['scratch_removed']


def test_rootful_runtime_refused_without_owned_scratch(tmp_path):
    command, env, out = controlled_runtime(tmp_path)
    result = subprocess.run(command, env={**env, 'ROOTLESS': 'false'}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and 'rootless' in result.stderr
    assert not (tmp_path / 'scratch/f123').exists()
    assert not (tmp_path / 'created').exists()


def test_changed_container_affinity_is_refused_and_cleaned(tmp_path):
    # A CPU outside the host's mask, independent of host CPU count or load.
    other = str(max(os.sched_getaffinity(0)) + 1)
    command, env, out = controlled_runtime(tmp_path, affinity=other)
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and 'changed the Flux CPU affinity' in result.stderr
    record, = [json.loads(p.read_text()) for p in out.glob('container-*.json')]
    assert record['container_affinity'] == [int(other)]
    assert record['removed'] and record['scratch_removed']


@pytest.mark.parametrize('case,message', [('tag', 'digest-pinned'), ('entrypoint', 'explicit entrypoint')])
def test_image_and_entrypoint_refused_before_runtime(tmp_path, case, message):
    command, env, out = controlled_runtime(tmp_path)
    if case == 'tag':
        command[command.index(IMAGE)] = 'example.invalid/cuda:latest'
    else:
        at = command.index('--entrypoint')
        del command[at:at + 2]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and message in result.stderr
    assert not (tmp_path / 'calls').exists()
    assert not (tmp_path / 'scratch').exists()


@pytest.mark.parametrize('option', [
    ['--privileged'], ['--network=host'], ['--gpus', 'all'],
    ['--cgroups', 'enabled'], ['--memory', '1g'],
])
def test_containment_flags_are_refused_before_runtime(tmp_path, option):
    command, env, out = controlled_runtime(tmp_path)
    command[command.index(IMAGE):command.index(IMAGE)] = option
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and 'owned by launcher' in result.stderr
    assert option[0].partition('=')[0] in result.stderr
    assert not (tmp_path / 'calls').exists()


def test_surviving_container_records_cleanup_error_and_preserves_scratch(tmp_path):
    command, env, out = controlled_runtime(tmp_path, survives_rm=True)
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and 'cleanup failed' in result.stderr
    record, = [json.loads(p.read_text()) for p in out.glob('container-*.json')]
    assert record['cleanup_errors'] and 'container remains' in record['cleanup_errors'][0]
    assert record['removed'] is False and record['scratch_removed'] is False
    assert Path(record['scratch']).is_dir() and (tmp_path / 'created').exists()


def test_create_failure_removes_owned_scratch_without_start(tmp_path):
    command, env, out = controlled_runtime(tmp_path, create_failure=True)
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and 'create failed' in result.stderr
    record, = [json.loads(p.read_text()) for p in out.glob('container-*.json')]
    assert record['create_returncode'] == 17
    assert record['removed'] and record['scratch_removed'] and not record['cleanup_errors']
    assert not Path(record['scratch']).exists() and not (tmp_path / 'ready').exists()


@pytest.mark.parametrize('user,keep_id', [('0:0', False), ('1001:1002', True)])
def test_container_create_contract_and_identity(tmp_path, user, keep_id):
    import runpy
    command, env, out = controlled_runtime(tmp_path)
    env.update(HOME='/secret-home', HOST_SECRET='must-not-be-forwarded')
    command[command.index(IMAGE):command.index(IMAGE)] = ['--user', user]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (tmp_path / 'calls').read_text().splitlines()]
    create = next(call for call in calls if call[0] == 'create')
    assert '--pull=never' in create and '--init' in create
    assert ('--userns=keep-id' in create) is keep_id
    assert create[create.index('--user') + 1] == user
    entries = [create[i + 1] for i, arg in enumerate(create[:-1]) if arg == '--env']
    expected = {'TMPDIR': '/tessera-scratch/tmp', 'CUDA_VISIBLE_DEVICES': '0',
                **runpy.run_path(str(LAUNCHER))['THREADS']}
    assert sorted(entries) == sorted(k + '=' + v for k, v in expected.items())
    assert '-e' not in create and '--env-host' not in create
    assert not any('must-not-be-forwarded' in arg or '/secret-home' in arg for arg in create)


@pytest.mark.parametrize('case,error', [
    ('waitstatus', 'native job failed or raised an exception'),
    ('exception', 'native job failed or raised an exception'),
    ('clean', 'native clean event missing'),
    ('gpu', 'allocation is not one whole Spark GPU'),
    ('cores', 'allocated core count differs from request'),
    ('unreadable', 'unreadable native terminal/allocation evidence'),
    ('unknown', 'submission unknown'),
])
def test_submission_negative_evidence_refuses(tmp_path, monkeypatch, case, error):
    positive = tmp_path / 'positive'
    positive.mkdir()
    module, argv, out = controlled_submission(positive, monkeypatch)
    assert module['submit_and_collect'](argv, out, cores=1, memory_gib=8, walltime='10m') == 0
    assert json.loads((out / 'receipt.json').read_text())['evidence_errors'] == []
    negative = tmp_path / 'negative'
    negative.mkdir()
    module, argv, out = controlled_submission(negative, monkeypatch, evidence=case)
    code = module['submit_and_collect'](argv, out, cores=1, memory_gib=8, walltime='10m')
    assert code == 2
    receipt = json.loads((out / 'receipt.json').read_text())
    assert any(error in entry for entry in receipt['evidence_errors'])
    if case == 'unknown':
        assert json.loads((out / 'submission.json').read_text())['state'] == 'unknown'
        calls = [json.loads(line) for line in (negative / 'flux-calls').read_text().splitlines()]
        assert sum(call[0] == 'submit' for call in calls) == 1
        assert not any(call[:2] == ['job', 'status'] for call in calls)


@pytest.mark.parametrize('payload_code', [0, 7])
def test_generated_payload_executes_shared_memory_receipt(tmp_path, monkeypatch, payload_code):
    module, argv, out = controlled_submission(tmp_path, monkeypatch, evidence='payload')
    argv[-1:] = ['/bin/sh', '-c', f'exit {payload_code}']
    assert module['submit_and_collect'](argv, out, cores=1, memory_gib=8, walltime='10m') == (2 if payload_code else 0)
    assert (tmp_path / 'payload-exit').read_text() == str(payload_code)
    assert 'memory-receipt' in (out / 'payload.sh').read_text()
    receipt = json.loads((out / 'memory.json').read_text())
    assert receipt['declared_memory_gib'] == 8
    assert 'memory_peak_bytes' in receipt


def test_memory_receipt_cli_preserves_unified_cgroup_colons(tmp_path):
    proc = tmp_path / 'cgroup'
    proc.write_text('5:cpu:/wrong\n0::/job:sys.service\n')
    root = tmp_path / 'cgroups'
    peak = root / 'job:sys.service/memory.peak'
    peak.parent.mkdir(parents=True)
    peak.write_text('123456')
    out = tmp_path / 'memory.json'
    result = subprocess.run([os.sys.executable, str(LAUNCHER), 'memory-receipt', '--out', str(out),
                             '--memory-gib', '8', '--proc', str(proc), '--root', str(root)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(out.read_text())
    assert receipt == {'declared_memory_gib': 8, 'memory_peak_bytes': 123456}


def test_created_container_has_decimal_flux_job_label(tmp_path):
    command, env, out = controlled_runtime(tmp_path)
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (tmp_path / 'calls').read_text().splitlines()]
    create = next(call for call in calls if call[0] == 'create')
    assert create[create.index('--label') + 1] == 'flux.jobid=123'


@pytest.mark.parametrize('devices', ['', '0,1', 'all', '-1', '0 1'])
def test_gpu_requires_exactly_one_allocated_device(tmp_path, devices):
    command, env, out = controlled_runtime(tmp_path)
    env['CUDA_VISIBLE_DEVICES'] = devices
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert 'CUDA_VISIBLE_DEVICES' in result.stderr and 'exactly one' in result.stderr
    assert not (tmp_path / 'calls').exists()


def test_cdi_device_matches_the_single_allocation(tmp_path):
    command, env, out = controlled_runtime(tmp_path)
    env['CUDA_VISIBLE_DEVICES'] = '3'
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    create = next(json.loads(line) for line in (tmp_path / 'calls').read_text().splitlines()
                  if json.loads(line)[0] == 'create')
    assert create[create.index('--device') + 1] == 'nvidia.com/gpu=3'
    assert 'CUDA_VISIBLE_DEVICES=3' in create


def test_submit_runtime_uid_is_current_and_cuda_is_left_to_flux(tmp_path, monkeypatch):
    module, argv, out = controlled_submission(tmp_path, monkeypatch)
    monkeypatch.setattr(module['os'], 'getuid', lambda: 1234)
    assert module['submit_and_collect'](argv, out, cores=1, memory_gib=8, walltime='10m') == 0
    submit = next(json.loads(line) for line in (tmp_path / 'flux-calls').read_text().splitlines()
                  if json.loads(line)[0] == 'submit')
    assert '--env=XDG_RUNTIME_DIR=/run/user/1234' in submit
    assert not any(arg.startswith('--env=CUDA_VISIBLE_DEVICES=') for arg in submit)


def test_container_memory_declaration_is_an_integer(tmp_path):
    command, env, out = controlled_runtime(tmp_path)
    env['FLUX_MEMORY_GIB'] = '8'
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    record, = [json.loads(p.read_text()) for p in out.glob('container-*.json')]
    assert record['declared_memory_gib'] == 8


def test_invalid_memory_declaration_is_named_before_runtime(tmp_path):
    command, env, out = controlled_runtime(tmp_path)
    env['FLUX_MEMORY_GIB'] = 'eight'
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and 'FLUX_MEMORY_GIB' in result.stderr
    assert not (tmp_path / 'calls').exists()
