"""CPU checks with stub Flux clients and real saved tensor files."""
import copy
import hashlib
import json
import sys
from types import SimpleNamespace
import pytest
from experiments.numerics_rows import phase_instance as api
from experiments.numerics_rows.phase_artifact import tensor_subset_equal


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
    pytest.importorskip("torch")
    from experiments.numerics_rows.phase_artifact import Artifact, records
    from tests.test_numerics_rows_phases import rewrite_stream
    scopes = [{'unit_key': ['cell', 'symbol', 'decoder', 1024, {'M': 1}], 'cut': {'size': 2, 'rank': 0}}]
    ref = {'file': 'tensor', 'sha256': hashlib.sha256(b'1234').hexdigest(), 'dtype': 'float32', 'shape': [1]}
    artifacts = []
    for index, name in enumerate(('canary', 'counted')):
        artifact = Artifact(tmp_path / name, 'prepare', {'canary': not index})
        artifact.document['scopes'] = scopes
        (artifact.root / 'tensor').write_bytes(b'1234')
        if index:
            other = {**scopes[0], 'cut': {'size': 1, 'rank': 0}}
            artifact.document['scopes'] = [other, *scopes]
            artifact.add(other, {'M': 1}, 'dense', {}, category='diagnostic', seed=0)
        record = artifact.add(scopes[0], {'M': 1}, 'dense', {}, category='diagnostic', seed=0, seconds=index + 2)
        record['tensors'] = {'input': copy.deepcopy(ref)}
        changed = list(records(artifact.root, 'cases'))
        changed[-1] = record
        rewrite_stream(artifact, 'cases', changed)
        artifacts.append(artifact)
    assert tensor_subset_equal(*(a.root for a in artifacts))['cases'] == 1
    if change == 'bytes':
        (artifacts[1].root / 'tensor').write_bytes(b'4321')
    else:
        target = record if change == 'seed' else record['tensors']['input']
        target[change] = {'dtype': 'int32', 'shape': [2], 'seed': 1}[change]
        changed[-1] = record
        rewrite_stream(artifacts[1], 'cases', changed)
    with pytest.raises(ValueError, match='E/P'):
        tensor_subset_equal(*(a.root for a in artifacts))
