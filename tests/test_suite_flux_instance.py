"""Live-instance policy through a controlled native CLI boundary."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from suite_flux_fixtures import ROOT, load, publication, resource_set, instance, suite



def test_changed_config_with_same_shape_submits_and_records_observation(instance, tmp_path):
    instance.submit(tmp_path / 'first')
    first = json.loads((tmp_path / 'first/launch.json').read_bytes())
    instance.config['operator-note'] = 'a changed configuration, same capacity'
    instance.submit(tmp_path / 'second')
    second = json.loads((tmp_path / 'second/launch.json').read_bytes())
    assert len([a for a in instance.calls if a[:2] == ['job', 'submit']]) == 2
    assert first['qualification']['config_sha256'] != second['qualification']['config_sha256']
    assert second['qualification']['config_sha256'] == hashlib.sha256(json.dumps(instance.config).encode()).hexdigest()
    assert second['qualification']['flux_version'] == 'test-flux-version'
    assert second['qualification']['broker_starttime'] == 'instance-A'
    assert second['qualification']['model_sha256'] == hashlib.sha256(instance.flux.encoded(instance.model['scheduling'])).hexdigest()
    assert 'profile' not in second and 'profile' not in second['qualification']
    assert second['uri'] == 'local:///operator/selected-instance'


@pytest.mark.parametrize('field', ['cores', 'memory_gib', 'gpus', 'architecture', 'online', 'host', 'split-hosts'])
def test_live_capacity_refuses_by_field_before_submission(instance, tmp_path, field):
    # Positive control runs real shape policy, not a configured verifier result.
    instance.submit(tmp_path / 'positive')
    instance.calls.clear()
    kw = {}
    if field in ('cores', 'memory_gib'):
        kind = {'cores': 'core', 'memory_gib': 'memory'}[field]
        instance.model['scheduling']['graph']['nodes'] = [n for n in instance.model['scheduling']['graph']['nodes']
            if n['metadata']['type'] != kind or n['metadata']['rank'] != 4 or n['metadata']['id'] == 0]
    elif field == 'gpus':
        # Exercise GPU capacity policy directly; submit requires a container.
        demand = instance.flux.resources(2, 8, 30, gpus=1, architecture='aarch64')
        instance.model['scheduling']['graph']['nodes'] = [n for n in instance.model['scheduling']['graph']['nodes']
                                                           if n['metadata']['type'] != 'gpu']
    elif field == 'architecture':
        instance.model['execution']['properties'].pop('x86_64')
    elif field == 'online':
        instance.online = '9'
    elif field == 'host':
        demand = instance.flux.resources(2, 8, 30, host='new-image-host')
    else:
        # Global totals fit, but neither eligible host has both resources.
        instance.model = resource_set()
        for n in instance.model['scheduling']['graph']['nodes']:
            if n['metadata']['type'] == 'node':
                n['metadata']['properties']['arch'] = 'x86_64'
        instance.model['execution']['properties']['x86_64'] = '4,9'
        instance.model['scheduling']['graph']['nodes'] = [n for n in instance.model['scheduling']['graph']['nodes']
            if not (n['metadata']['type'] == 'core' and n['metadata']['rank'] == 4 and n['metadata']['id'] > 0)
            and not (n['metadata']['type'] == 'memory' and n['metadata']['rank'] == 9 and n['metadata']['id'] > 0)]
    expected = 'cores|memory_gib' if field == 'split-hosts' else field
    with pytest.raises(ValueError, match=expected):
        if field in ('gpus', 'host'):
            instance.flux.require_capacity(demand)
        else:
            instance.submit(tmp_path / 'refused', **kw)
    assert not any(a[:2] == ['job', 'submit'] for a in instance.calls)
    assert not (tmp_path / 'refused').exists()


def test_resume_restart_names_broker_starttime_before_reading_job(suite, tmp_path, monkeypatch):
    from suite_flux_fixtures import NativeJobs
    published, native, stdout = publication(suite, tmp_path)
    jobs = NativeJobs(suite, monkeypatch)
    world = jobs.add(tmp_path, 'x86', published.payload)
    assert suite.suite_flux.collect(world.directory)['returncode'] == 0
    jobs.calls.clear()
    jobs.qualification = {**jobs.qualification, 'broker_starttime': 'restarted-instance'}
    record = suite.suite_flux.collect(world.directory)
    assert record['state'] == 'unknown'
    assert any('broker_starttime' in error for error in record['query_errors'])
    assert jobs.calls == []


def test_connection_requires_named_client(suite, monkeypatch):
    flux = suite.suite_flux
    monkeypatch.setenv('FLUX_URI', 'local:///selected')
    monkeypatch.setenv('FLUX_CLIENT', '/missing/operator-client')
    monkeypatch.setattr(flux, 'instance_settings', lambda: Path('/operator/prefix'))
    with pytest.raises(ValueError, match='FLUX_CLIENT.*PATH'):
        flux.native(['getattr', 'version'])


def test_legacy_profile_receipt_reads_without_reinstating_profile_gate(suite, tmp_path, monkeypatch):
    from suite_flux_fixtures import NativeJobs
    published, _, _ = publication(suite, tmp_path)
    jobs = NativeJobs(suite, monkeypatch)
    world = jobs.add(tmp_path, 'x86', published.payload)
    assert suite.suite_flux.collect(world.directory)['returncode'] == 0
    jobs.qualification = {'config_sha256': 'c' * 64, 'model_sha256': 'd' * 64,
                          'broker_starttime': 'generation-A', 'flux_version': 'new-client'}
    record = suite.suite_flux.collect(world.directory)
    assert record['returncode'] == 0, record
    assert record['qualification']['profile'] == 'fixture'


@pytest.mark.parametrize('rank', ['0', '4', '9'])
def test_broker_generation_uses_rank_zero_rpc_from_any_client(suite, monkeypatch, rank):
    flux = suite.suite_flux
    calls = []
    def checked(argv, **kw):
        calls.append(argv)
        if argv == ['getattr', 'rank']:
            return rank
        if argv == ['getattr', 'broker.starttime']:
            return 'rank-local-start'
        assert argv[:2] == ['python', '-c']
        assert 'nodeid=0' in argv[2]
        return 'root-start'
    monkeypatch.setattr(flux, 'checked', checked)
    assert flux.broker_starttime() == 'root-start'
    assert all(argv[:1] == ['python'] for argv in calls)


def test_native_uses_caller_connection_without_inheriting_credentials(suite, monkeypatch):
    flux = suite.suite_flux
    monkeypatch.setattr(flux, 'instance_settings', lambda: Path('/operator/prefix'))
    monkeypatch.setenv('FLUX_URI', 'local:///operator/other-instance')
    monkeypatch.setenv('FLUX_CLIENT', sys.executable)
    monkeypatch.setenv('UNRELATED_SECRET', 'must-not-travel')
    calls = []
    def run(argv, **kw):
        calls.append((argv, kw))
        return SimpleNamespace(returncode=0, stdout='version', stderr='')
    monkeypatch.setattr(flux.subprocess, 'run', run)
    flux.native(['getattr', 'version'])
    argv, kw = calls[-1]
    assert argv == [sys.executable, 'getattr', 'version']
    assert kw['env']['FLUX_URI'] == 'local:///operator/other-instance'
    assert 'UNRELATED_SECRET' not in kw['env']
    monkeypatch.delenv('FLUX_CLIENT')
    monkeypatch.setenv('PATH', str(Path(sys.executable).parent))
    monkeypatch.setattr(flux.shutil, 'which', lambda command, **kw: sys.executable if command.endswith('/install/bin/flux') else None)
    flux.native(['getattr', 'version'])
    assert calls[-1][0] == argv


def test_gpu_renderer_has_no_implicit_host_roster(suite):
    spec = load('portable_gpu_render', ROOT / 'tools/_flux_gpu_jobspec.py')
    argv = spec.render(['payload'], cores=1, memory_gib=1, walltime='30s', directory='/mnt/shared/render-fixture')
    assert not any(token.startswith('--requires=') for token in argv)
    explicit = spec.render(['payload'], cores=1, memory_gib=1, walltime='30s', hosts='image-owner', directory='/mnt/shared/render-fixture')
    assert '--requires=host:image-owner' in explicit


def test_submit_keeps_instance_client_paths_out_of_job_environment(suite, monkeypatch, tmp_path):
    flux = suite.suite_flux
    prefix = tmp_path / 'INSTANCE-client'
    manifest_read = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **kw:
        f'{prefix} operator-run\n' if str(path) == '/mnt/shared/flux-migration/INSTANCE'
        else manifest_read(path, *a, **kw))
    monkeypatch.setenv('FLUX_URI', 'local:///selected-instance')
    monkeypatch.setenv('FLUX_CLIENT', sys.executable)
    monkeypatch.setenv('PATH', f'{prefix}/deps/bin:/usr/bin:/bin')
    monkeypatch.setenv('LD_LIBRARY_PATH', f'{prefix}/deps/lib')
    monkeypatch.setattr(flux, 'source', lambda path: {'sha256': 'a' * 64})
    monkeypatch.setattr(flux, 'runtime_identity', lambda *a, **kw: {'schema': 'tessera.suite_runtime.v1'})
    monkeypatch.setattr(flux, 'collect', lambda directory: {'state': 'running'})
    client_environments, submitted = [], []
    def client(command, **options):
        client_environments.append(options['env'])
        argv = command[1:]
        if argv == ['kvs', 'get', 'resource.R']:
            value = json.dumps(resource_set())
        elif argv[:2] == ['resource', 'status']:
            value = '4,9'
        elif argv[:1] == ['python']:
            value = 'generation-A'
        elif argv == ['config', 'get']:
            value = '{}'
        elif argv == ['getattr', 'version']:
            value = 'fake-client'
        elif argv[:3] == ['submit', '--env=-*', '--dry-run']:
            # The fake CLI renders the actual submitted allow-list, not an
            # independently configured successful environment.
            environment = dict(token.removeprefix('--env=').split('=', 1)
                               for token in argv[3:] if token.startswith('--env='))
            value = json.dumps({'attributes': {'system': {'environment': environment}}})
        elif argv[:2] == ['job', 'submit']:
            submitted.append(json.loads(Path(argv[2]).read_bytes()))
            value = 'f123'
        else:
            raise AssertionError(argv)
        return SimpleNamespace(returncode=0, stdout=value, stderr='')
    monkeypatch.setattr(flux.subprocess, 'run', client)
    directory = tmp_path / 'isolated-launch-fixture'
    flux.submit(directory, tmp_path, [sys.executable, '-s', '-c', 'pass'], 1, 1, 30, {})
    assert submitted, 'submit must reach the fake client with a real saved jobspec'
    assert all(env['PATH'].startswith(f'{prefix}/install/bin:{prefix}/deps/bin:')
               and env['LD_LIBRARY_PATH'] == f'{prefix}/install/lib:{prefix}/deps/lib'
               for env in client_environments)
    environment = submitted[0]['attributes']['system']['environment']
    assert environment['PATH'] == '/usr/local/bin:/usr/bin:/bin', 'job PATH must use system compilers'
    assert str(prefix) not in environment['PATH'], 'INSTANCE prefix leaked into job PATH'
    assert 'LD_LIBRARY_PATH' not in environment, 'INSTANCE client libraries leaked into job LD_LIBRARY_PATH'
