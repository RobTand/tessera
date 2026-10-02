"""CPU controls for the native app; public plugin calls are explicit stand-ins."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest
import torch

from test_native_timing_panel import panel, canonical_wire
from tools import tessera_shape_time_panel as app
from tessera.serving import contract, telemetry, backend, source_identity, runtime_image


def request_file(panel, monkeypatch):
    monkeypatch.setattr(contract, 'contract_path', lambda: Path(panel['evidence']['contract']['path']))
    value = {'schema': app.REQUEST_SCHEMA, 'expected_runtime': copy.deepcopy(panel['runtime']),
             'scope': copy.deepcopy(panel['plan']['rows'][0]['scope']), 'prefix': 'test.dense',
             'scheme': copy.deepcopy(panel['rows'][0]['scheme']), 'wire': panel['evidence']['wire'],
             'sampling': {'samples': 4, 'warmup_iterations': 1, 'steady_s': 20.0, 'seed': 688},
             'netdata_hosts': {'sparky': 'sparky', 'sparklina': 'sparklina'}}
    path = Path(panel['evidence']['wire']['path']).parent / 'request.json'
    path.write_bytes(app.tp.canonical(value))
    return path, value


def test_preflight_is_pure_and_positive(panel, monkeypatch):
    path, value = request_file(panel, monkeypatch)
    got, plan, wire = app.read_request(path)
    assert got == value and wire == app.tp.read_bound(value['wire'])
    assert plan['gpu_executed'] is False
    assert app.main(['check-request', str(path)]) == 0


@pytest.mark.parametrize('fault', ['contract', 'cell_source', 'tp2', 'samples_bool', 'empty_hosts', 'extra'])
def test_request_refuses_before_native_preparation(panel, monkeypatch, fault):
    path, value = request_file(panel, monkeypatch)
    if fault == 'contract': value['expected_runtime']['contract_sha256'] = '0' * 64
    elif fault == 'cell_source': value['expected_runtime']['serving_source_sha256'] = '0' * 64
    elif fault == 'tp2': value['scope']['tp_degree'] = 2
    elif fault == 'samples_bool': value['sampling']['samples'] = True
    elif fault == 'empty_hosts': value['netdata_hosts']['sparklina'] = ''
    else: value['scope']['implicit_rank'] = 0
    path.write_bytes(app.tp.canonical(value))
    with pytest.raises(ValueError): app.read_request(path)


def test_preparation_uses_public_exact_wire_path(panel, monkeypatch):
    from tessera.serving import lane
    events = []
    class Method:
        def create_weights(self, layer, **kw):
            events.append(('create', kw))
            layer.wire_bytes = torch.nn.Parameter(torch.empty(len(wire), dtype=torch.uint8), requires_grad=False)
        def process_weights_after_loading(self, layer):
            events.append(('finalize', bytes(layer.wire_bytes.detach().numpy())))
            layer.tessera_shard_plan = SimpleNamespace(tp_rank=0, tp_size=1)
            layer.tessera_native = SimpleNamespace(rows=16, columns=128, packed_bytes=lambda: 123)
    method = Method()
    monkeypatch.setattr(lane, 'build_tessera_method', lambda *a: (events.append(('build', a)) or method))
    path, request = request_file(panel, monkeypatch);wire = app.tp.read_bound(request['wire'])
    layer, got, prep = app.prepare_dense(request, wire)
    assert got is method and [e[0] for e in events] == ['build', 'create', 'finalize']
    assert events[-1][1] == wire and events[1][1]['input_size_per_partition'] == 128
    assert events[1][1]['output_partition_sizes'] == [16]
    assert prep['wire_sha256'] == hashlib.sha256(wire).hexdigest()
    assert prep['native_packed_bytes'] == 123
    assert (layer.tp_rank, layer.tp_size) == (0, 1)


@pytest.mark.parametrize('fresh', [False, True])
def test_route_must_be_emitted_by_this_apply(monkeypatch, fresh):
    layer = SimpleNamespace()
    setattr(layer, telemetry.ATTR_PREFIX + 'state', {'state': 'served', 'symbol': 'old'})
    monkeypatch.setattr(telemetry, 'read_route', lambda obj: getattr(obj, telemetry.ATTR_PREFIX + 'state'))
    def apply(obj, x):
        if fresh: setattr(obj, telemetry.ATTR_PREFIX + 'state', {'state': 'served', 'symbol': 'actual'})
        return x
    method = SimpleNamespace(apply=apply)
    if fresh:
        _, record = app.fresh_call(layer, method, object());assert record['symbol'] == 'actual'
    else:
        with pytest.raises(ValueError, match='fresh'): app.fresh_call(layer, method, object())


@pytest.mark.parametrize('failed', [False, True])
def test_publish_file_then_directory_sync(tmp_path, monkeypatch, failed):
    events = []; real_fsync, real_replace = app.os.fsync, app.os.replace
    monkeypatch.setattr(app.os, 'fsync', lambda fd: (events.append('file_sync'), real_fsync(fd))[-1])
    monkeypatch.setattr(app.os, 'replace', lambda a, b: (events.append('rename'), real_replace(a,b))[-1])
    def directory(path):
        assert Path(path) == tmp_path;events.append('directory_sync')
        if failed: raise OSError('directory sync failure')
    monkeypatch.setattr(app, 'fsync_path', directory)
    if failed:
        with pytest.raises(OSError): app.publish_json({'committed': True}, tmp_path/'receipt.json')
    else:
        bound = app.publish_json({'committed': True}, tmp_path/'receipt.json')
        assert app.tp.json_bytes(app.tp.read_bound(bound)) == {'committed': True}
    assert events == ['file_sync', 'rename', 'directory_sync']


def test_runtime_commit_comes_from_actual_imported_checkout(tmp_path, monkeypatch):
    package = SimpleNamespace(__file__=str(tmp_path/'imported'/'__init__.py'))
    (tmp_path/'imported').mkdir();Path(package.__file__).write_text('')
    seen=[]
    def run(argv, **kw):
        seen.append(argv)
        if '--show-toplevel' in argv: out=str(tmp_path)
        elif 'status' in argv: out=''
        elif 'ls-files' in argv: out='imported/__init__.py'
        else: out='a'*40
        return SimpleNamespace(returncode=0,stdout=out)
    monkeypatch.setattr(app.subprocess,'run',run)
    assert app.observed_commit(package) == 'a'*40
    assert seen[0][2] == str(Path(package.__file__).parent)


def test_dirty_runtime_and_missing_vcs_identity_refuse(tmp_path,monkeypatch):
    package=SimpleNamespace(__file__=str(tmp_path/'__init__.py'));Path(package.__file__).write_text('')
    def run(argv,**kw):
        return SimpleNamespace(returncode=0,stdout=str(tmp_path) if '--show-toplevel' in argv else ' M source.py')
    monkeypatch.setattr(app.subprocess,'run',run)
    with pytest.raises(ValueError,match='dirty'):app.observed_commit(package)
    monkeypatch.setattr(app.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stdout=''))
    monkeypatch.setattr(app.importlib.metadata,'distribution',lambda *a:SimpleNamespace(read_text=lambda *a:None,files=[Path('__init__.py')],locate_file=lambda p:tmp_path/p))
    with pytest.raises(ValueError,match='immutable'):app.observed_commit(package)


def test_foreign_installed_metadata_cannot_label_actual_runtime(tmp_path,monkeypatch):
    package=SimpleNamespace(__file__=str(tmp_path/'foreign'/'__init__.py'));Path(package.__file__).parent.mkdir();Path(package.__file__).write_text('')
    monkeypatch.setattr(app.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stdout=''))
    dist=SimpleNamespace(read_text=lambda *a:json.dumps({'vcs_info':{'commit_id':'a'*40}}),
                         files=[Path('tessera/__init__.py')],locate_file=lambda p:tmp_path/p)
    monkeypatch.setattr(app.importlib.metadata,'distribution',lambda *a:dist)
    with pytest.raises(ValueError,match='imported'):app.observed_commit(package)
