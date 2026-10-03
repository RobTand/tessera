"""CPU orchestration model only; real PyBind and CUDA proof are separate."""
import functools
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile
import pytest
import torch
from experiments.t8r_speed import value_prefetch_numeric as numeric
import test_routed_fused_window as helpers


@pytest.mark.parametrize('fail', [False, True], ids=['two-whole-arm-epochs', 'failure-fences-before-release'])
def test_matrix_epoch_loads_each_arm_once_and_releases_after_fences(tmp_path, monkeypatch, fail):
    bank, out = tmp_path / 'bank', tmp_path / 'out'
    qs = [256, 384, 512]
    monkeypatch.setattr(numeric, 'Q256_CASES', qs)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda value: value)
    monkeypatch.setattr(helpers, '_bundles', lambda *args: None)
    events, owners, loads = [], [], []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: events.append('fence'))
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w'):
        pass
    files = {str(bank / 'runner.zip'): archive.getvalue()}
    cases = [{'q256': q, 'path': str(bank / f'q{q}.pt'), 'ms': numeric.MS} for q in qs]
    files[str(bank / 'packet.json')] = json.dumps({'scope': numeric.SCOPE, 'source_sha256': numeric.SOURCE_SHA,
                                                 'arms': numeric.ARMS, 'cases': cases}).encode()
    for q in qs:
        samples = [{'m': m, 'x': torch.tensor([[float(q), float(m)]]),
                    'ids': torch.zeros(1, 1, dtype=torch.int32), 'weights': torch.ones(1, 1)} for m in numeric.MS]
        stream = io.BytesIO()
        torch.save({'scope': numeric.SCOPE, 'q256': q, 'stacks': [[], [], []], 'samples': samples}, stream)
        files[str(bank / f'q{q}.pt')] = stream.getvalue()
    class Reader:
        manifest_sha256 = 'model-only'
        def __init__(self, path):
            pass
        def read(self, path):
            return files[str(path)]
        def close(self):
            for owner in owners:
                with pytest.raises(OSError):
                    os.fstat(owner.fd)
            events.append('release')
    monkeypatch.setattr(numeric, 'StagedInputs', Reader)
    class Owner:
        def __init__(self, reader, path, rf, directory, **kwargs):
            directory.mkdir(parents=True)
            self.fd = os.open(__file__, os.O_RDONLY)
            self.module = None
            self.record = {'expected_sha256': kwargs['expected_sha256']}
            owners.append(self)
        def attest_mapped(self, module):
            assert module is self.module
            os.fstat(self.fd)
        def finish(self, fence, *, keep_load_fd):
            assert keep_load_fd
            os.fstat(self.fd)
            fence()
            events.append('finish')
    monkeypatch.setattr(numeric, 'NativeCallback', Owner)
    @functools.lru_cache(None)
    def ext(library):
        loads.append(library)
        module = SimpleNamespace(VALUE_A_PREFETCH=int(os.environ['TESSERA_ROUTED_FUSED_VALUE_A_PREFETCH']))
        owners[-1].module = module
        return module
    monkeypatch.setattr(numeric.rf, '_ext', ext)
    def stage(stacks, bundles, x, ids, weights, family, what, **kwargs):
        # Exercise the owner's ordinary cache at every model cell.
        assert numeric.rf._ext('value') is owners[-1].module
        if fail and x[0, 0].item() == 384 and x[0, 1].item() == 7:
            raise RuntimeError('model stage failure')
        return (x.clone(), x.clone(), x.clone(), x.clone())
    monkeypatch.setattr(helpers, '_staged_check', stage)
    if fail:
        with pytest.raises(RuntimeError, match='model stage failure'):
            numeric.consume(bank, tmp_path / 'manifest.json', out)
        assert len(loads) == len(owners) == 1
        assert not (out / 'numeric.json').exists()
    else:
        numeric.consume(bank, tmp_path / 'manifest.json', out)
        result = json.loads((out / 'numeric.json').read_text())
        assert len(loads) == len(owners) == 2
        assert len(result['results']) == 9
        assert len(result['owners']) == 2
        assert all(len(owner['completed_cases']) == 9 for owner in result['owners'])
    assert events[-2:] == ['fence', 'release']
    assert events.count('finish') == len(owners)
