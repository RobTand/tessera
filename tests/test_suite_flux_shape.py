"""SHAPE v1 resolver and allocation boundaries, without a live submission."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from suite_flux_fixtures import suite, instance


@pytest.mark.parametrize('override', [None, 'local:///caller-selected'])
def test_instance_manifest_and_resolver_select_connection(suite, monkeypatch, tmp_path, override):
    flux = suite.suite_flux
    prefix = tmp_path / 'operator-client'
    original = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **kw:
        f'{prefix} selected-run\n' if str(path) == '/mnt/shared/flux-migration/INSTANCE' else original(path, *a, **kw))
    monkeypatch.setenv('FLUX_CLIENT', sys.executable)
    monkeypatch.delenv('FLUX_URI', raising=False)
    if override:
        monkeypatch.setenv('FLUX_URI', override)
    calls = []
    def run(argv, **kw):
        calls.append((argv, kw))
        return SimpleNamespace(returncode=0, stdout='local:///resolved-at-runtime\n' if argv[0].endswith('flux-uri.sh') else 'version', stderr='')
    monkeypatch.setattr(flux.subprocess, 'run', run)
    flux.native(['getattr', 'version'])
    query, options = calls[-1]
    assert query == [sys.executable, 'getattr', 'version']
    assert options['env']['FLUX_URI'] == (override or 'local:///resolved-at-runtime')
    assert options['env']['PATH'].startswith(f'{prefix}/install/bin:{prefix}/deps/bin:')
    assert options['env']['LD_LIBRARY_PATH'] == f'{prefix}/install/lib:{prefix}/deps/lib'
    assert any(argv == ['/mnt/shared/flux-migration/flux-uri.sh'] for argv, _ in calls) == (override is None)


def test_resolver_refusal_names_resolver_and_preserves_stderr(suite, monkeypatch):
    flux = suite.suite_flux
    monkeypatch.delenv('FLUX_URI', raising=False)
    monkeypatch.setattr(flux, 'instance_settings', lambda: Path('/operator/prefix'))
    monkeypatch.setattr(flux.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        returncode=3, stdout='', stderr='flux-uri.sh: broker socket missing: operator-selected\n'))
    with pytest.raises(ValueError, match='flux-uri.sh.*broker socket missing: operator-selected'):
        flux.native(['getattr', 'version'])


@pytest.mark.parametrize('fact', ['core-count', 'gpu', 'host', 'rank'])
def test_spark_gpu_efficiency_allocation_accepts_class_and_binds_execution(suite, fact):
    flux = suite.suite_flux
    demand = flux.resources(2, 2, 30, gpus=1, architecture='aarch64')
    metadata = [
        {'type': 'node', 'rank': 1, 'name': 'sparky', 'properties': {'arch': 'aarch64'}},
        *[{'type': 'core', 'rank': 1, 'id': i, 'properties': {'core-type': 'eff', 'os-pus': str(i)}} for i in (0, 10)],
        *[{'type': 'memory', 'rank': 1, 'id': i} for i in (0, 1)],
        {'type': 'gpu', 'rank': 1, 'id': 0},
    ]
    allocation = {'version': 1, 'execution': {'R_lite': [{'rank': '1', 'children': {'core': '0,10', 'gpu': '0'}}],
                  'nodelist': ['sparky']}, 'scheduling': {'graph': {'nodes': [{'metadata': m} for m in metadata]}}}
    # v1 legitimately allocates these efficiency cores to GPU work.
    flux.require_allocation(allocation, demand, [0, 10], [0, 10])
    changed = copy.deepcopy(allocation)
    row = changed['execution']['R_lite'][0]
    if fact == 'core-count':
        row['children']['core'] = '0'
    elif fact == 'gpu':
        row['children']['gpu'] = '1'
    elif fact == 'host':
        changed['execution']['nodelist'] = ['dl380g10']
    else:
        row['rank'] = '0'
    with pytest.raises(ValueError, match='allocation'):
        flux.require_allocation(changed, demand, [0, 10], [0, 10])


def test_config_receipt_hashes_flux_config_get_bytes(instance):
    identity = instance.flux.instance_identity()
    assert identity['config_sha256'] == hashlib.sha256(json.dumps(instance.config).encode()).hexdigest()
    assert identity['r_sha256'] == hashlib.sha256(json.dumps(instance.model).encode()).hexdigest()
