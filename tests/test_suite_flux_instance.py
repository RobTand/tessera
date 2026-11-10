"""Live-instance policy through a controlled native CLI boundary."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from test_suite_flux import suite


def resource_set():
    nodes = []
    for rank, arch, cores, memory, gpus in [(4, 'x86_64', 4, 16, 0), (9, 'aarch64', 4, 16, 1)]:
        items = [{'type': 'node', 'name': f'worker-{rank}', 'properties': {'arch': arch}}]
        for kind, count in [('core', cores), ('memory', memory), ('gpu', gpus)]:
            items += [{'type': kind, 'id': i, 'size': 1, 'unit': 'GiB' if kind == 'memory' else '',
                       'properties': {'core-type': 'eff'}} for i in range(count)]
        nodes += [{'id': str(len(nodes) + i), 'metadata': {**item, 'rank': rank}} for i, item in enumerate(items)]
    return {'scheduling': {'graph': {'nodes': nodes}}, 'execution': {
        'R_lite': [{'rank': '4', 'children': {'core': '0-3'}},
                   {'rank': '9', 'children': {'core': '0-3', 'gpu': '0'}}],
        'properties': {'x86_64': '4', 'aarch64': '9'}}}


@pytest.fixture
def instance(suite, monkeypatch, tmp_path):
    flux = suite.suite_flux
    state = SimpleNamespace(config={'resource': {'path': '/operator/new-model.json'}},
                            model=resource_set(), generation='instance-A', online='4,9', calls=[])
    monkeypatch.setenv('FLUX_URI', 'local:///operator/selected-instance')
    monkeypatch.setenv('FLUX_CLIENT', sys.executable)
    monkeypatch.setattr(flux, 'source', lambda path: {'sha256': 'a' * 64})
    monkeypatch.setattr(flux, 'runtime_identity', lambda *a, **kw: {'schema': 'tessera.suite_runtime.v1'})
    monkeypatch.setattr(flux, 'collect', lambda directory: {'state': 'running'})
    # The old implementation reads deployment files before reaching the CLI.
    # Give it real changed files so its pre-fix refusal identifies the defect.
    for name, value in [('CONFIG', state.config), ('MODEL', state.model)]:
        if hasattr(flux, name):
            path = tmp_path / name
            path.write_text(json.dumps(value))
            monkeypatch.setattr(flux, name, path)
    def native(argv, **kwargs):
        state.calls.append(argv)
        if argv == ['config', 'get']:
            value = json.dumps(state.config)
        elif argv == ['kvs', 'get', 'resource.R']:
            value = json.dumps(state.model)
        elif argv[:2] == ['resource', 'status']:
            value = state.online
        elif argv == ['getattr', 'rank']:
            value = '0'
        elif argv == ['getattr', 'broker.starttime'] or argv[:1] == ['python']:
            value = state.generation
        elif argv == ['getattr', 'version']:
            value = 'test-flux-version'
        elif argv[:3] == ['submit', '--env=-*', '--dry-run']:
            value = '{}'
        elif argv[:2] == ['job', 'submit']:
            value = 'f123'
        else:
            raise AssertionError(argv)
        return SimpleNamespace(returncode=0, stdout=value, stderr='')
    monkeypatch.setattr(flux, 'native', native)
    state.submit = lambda directory, **kw: flux.submit(directory, tmp_path,
        [sys.executable, '-s', '-c', 'pass'], 2, 8, 30, {}, **kw)
    state.flux = flux
    return state


def test_changed_config_with_same_shape_submits_and_records_observation(instance, tmp_path):
    instance.submit(tmp_path / 'first')
    first = json.loads((tmp_path / 'first/launch.json').read_bytes())
    instance.config['operator-note'] = 'a changed configuration, same capacity'
    instance.submit(tmp_path / 'second')
    second = json.loads((tmp_path / 'second/launch.json').read_bytes())
    assert len([a for a in instance.calls if a[:2] == ['job', 'submit']]) == 2
    assert first['qualification']['config_sha256'] != second['qualification']['config_sha256']
    assert second['qualification']['config_sha256'] == hashlib.sha256(instance.flux.encoded(instance.config)).hexdigest()
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
    from test_suite_flux import publication
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


def test_connection_requires_named_uri_and_client(suite, monkeypatch):
    flux = suite.suite_flux
    monkeypatch.delenv('FLUX_URI', raising=False)
    with pytest.raises(ValueError, match='FLUX_URI'):
        flux.native(['getattr', 'version'])
    monkeypatch.setenv('FLUX_URI', 'local:///selected')
    monkeypatch.delenv('FLUX_CLIENT', raising=False)
    monkeypatch.setenv('PATH', '')
    with pytest.raises(ValueError, match='FLUX_CLIENT.*PATH'):
        flux.native(['getattr', 'version'])


def test_legacy_profile_receipt_reads_without_reinstating_profile_gate(suite, tmp_path, monkeypatch):
    from suite_flux_fixtures import NativeJobs
    from test_suite_flux import publication
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
    monkeypatch.setattr(flux.shutil, 'which', lambda command: sys.executable if command == 'flux' else None)
    flux.native(['getattr', 'version'])
    assert calls[-1][0] == argv


def test_gpu_renderer_has_no_implicit_host_roster(suite):
    from test_suite_flux import load, ROOT
    spec = load('portable_gpu_render', ROOT / 'tools/_flux_gpu_jobspec.py')
    argv = spec.render(['payload'], cores=1, memory_gib=1, walltime='30s')
    assert '--requires=aarch64' in argv
    assert not any('host:' in token for token in argv)
    explicit = spec.render(['payload'], cores=1, memory_gib=1, walltime='30s', hosts='image-owner')
    assert '--requires=aarch64 and host:image-owner' in explicit
