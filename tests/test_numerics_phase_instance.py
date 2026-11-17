"""CPU checks with stub Flux clients and real saved tensor files."""
import copy
import hashlib
import json
import sys
from types import SimpleNamespace
import pytest
from experiments.numerics_rows import phase_instance as api
from experiments.numerics_rows.phase_artifact import tensor_subset_equal


def test_prepare_capture_bind_instances_and_hold_until_recorded(tmp_path, monkeypatch):
    calls, live = [], ['fv1e-sparklina']
    def query(command, *args):
        name = live[0] if command[0] == 'env' else 'fv1e-' + command[1] if len(command) == 2 else 'fv1'
        calls.append((name, args))
        if args[0] == 'python':
            return '{}' if 'config.get' in args[2] else json.dumps({'broker.starttime': name, 'local-uri': 'local://' + name})
        if args[0] == 'submit':
            assert '--urgency=0' in args
            return 'f123'
        if args[:2] == ('job', 'id'):
            return args[-1]
        assert (job/'instance.json').exists()
        return ''
    monkeypatch.setattr(api, 'query', query)
    monkeypatch.setenv('FLUX_JOB_ID', 'f123')
    (tmp_path/'campaign.json').write_text('{"canary":false}')
    row = {}
    for phase, name in [('prepare', 'fv1e-sparklina'), ('capture', 'fv1')]:
        job = tmp_path/phase; job.mkdir()
        (job/'instance-name.txt').write_text(name)
        api.submit(job, ['submit', '--urgency=8', 'payload'])
        live[0] = name
        api.bind(row, phase, api.check(job))
    assert {p: v['name'] for p, v in row['phase_jobs'].items()} == {'prepare': 'fv1e-sparklina', 'capture': 'fv1'}
    live[0] = 'fv1e-sparky'
    with pytest.raises(ValueError, match='live phase instance'):
        api.check(job)
    with pytest.raises(ValueError, match='cannot mix'):
        api.bind(row, 'prepare', {**row['phase_jobs']['prepare'], 'name': 'fv1'})
    with pytest.raises(ValueError, match='cannot mix'):
        api.bind({}, 'capture', row['phase_jobs']['prepare'])
    assert any(args == ('job', 'urgency', 'f123', '8') for _, args in calls)


def test_observe_uses_rank_zero_config(monkeypatch, capsys):
    local = ['submit-broker']
    def rpc(topic, payload=None, nodeid=None):
        value = {'broker': 'rank-zero' if nodeid == 0 else local[0]} if topic == 'config.get' else {'value': payload['name']}
        return SimpleNamespace(get=lambda: value)
    monkeypatch.setitem(sys.modules, 'flux', SimpleNamespace(Flux=lambda: SimpleNamespace(rpc=rpc)))
    def query(command, *args):
        if args[:2] == ('config', 'get'):
            return json.dumps({'broker': local[0]})
        exec(args[2], {})
        return capsys.readouterr().out
    monkeypatch.setattr(api, 'query', query)
    submitted = api.observe('fv1')
    local[0] = 'job-broker'
    assert api.observe('fv1', ['job-client']) == submitted
    assert submitted['config_sha256'] == hashlib.sha256(b'{"broker":"rank-zero"}').hexdigest()


@pytest.mark.parametrize('change', ['bytes', 'dtype', 'shape', 'seed'])
def test_ep_tensor_subset_ignores_records_and_refuses_tensor_changes(tmp_path, change):
    scopes = [{'unit_key': ['u'], 'cut': {'size': 2, 'rank': 0}}]
    ref = {'file': 'tensor', 'sha256': hashlib.sha256(b'1234').hexdigest(), 'dtype': 'float32', 'shape': [1]}
    case = {'scope': scopes[0], 'category': 'diagnostic', 'configuration': {'M': 1}, 'seed': 0,
            'tensors': {'input': ref}, 'id': 'ordinal-0', 'identities': {'canary': True}, 'seconds': 1}
    roots = [tmp_path/'canary', tmp_path/'counted']
    for i, root in enumerate(roots):
        root.mkdir(); (root/'tensor').write_bytes(b'1234')
        record = {**copy.deepcopy(case), 'id': str(i), 'seconds': i+2, 'identities': {'canary': not i}}
        document = {'completion': True, 'scopes': scopes, 'cases': [record]}
        if i:
            document['cases'].insert(0, {**copy.deepcopy(record), 'scope': {'unit_key': ['other-cut']}})
        (root/'phase.json').write_text(json.dumps(document))
    assert tensor_subset_equal(*roots)['cases'] == 1
    if change == 'bytes':
        (roots[1]/'tensor').write_bytes(b'4321')
    else:
        target = record if change == 'seed' else record['tensors']['input']
        target[change] = {'dtype': 'int32', 'shape': [2], 'seed': 1}[change]
        (roots[1]/'phase.json').write_text(json.dumps(document))
    with pytest.raises(ValueError, match='E/P'):
        tensor_subset_equal(*roots)
