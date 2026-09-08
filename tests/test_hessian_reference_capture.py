"""The public capture reader consumes committed canonical H one unit at a time."""
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tessera.cached_unit import tensor_identity
from tessera.export import ActivationSource
from tessera.errors import GrammarError

SCHEMA = 'tessera.hessian_capture.references.v1'
POLICY = dict(schema='tessera.hessian_reference_load.v1',
              max_metadata_bytes=1024**2, max_file_bytes=1024**2,
              max_hessian_bytes=1024**2)


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def reference(tmp_path):
    H = {'a': torch.eye(4)*3, 'b': torch.eye(4)*7}
    provenance = dict(text_sha256='a'*64, fit_ids_sha256='b'*64,
                      fit_tokens=8, model='fixture', seqlen=8, source='fixture',
                      hessian_role='fit')
    counts = {'a': 8, 'b': 8}
    census = tmp_path/'census.json'
    census_sha = write_json(census, dict(unit_shapes={n:[4,4] for n in H},
        counts=counts, max_abs={n:1.0 for n in H}))
    root=tmp_path/'capture';(root/'inputs').mkdir(parents=True)
    entries={}
    for name, value in H.items():
        p=root/'inputs'/f'{name}.pt'
        torch.save(dict(inputs=torch.ones(2,4), hessian=value, name=name,
            source='tessera_campaign_prefix_f32_v1',count=8,max_abs=1.0),p)
        entries[name]=dict(path=f'inputs/{name}.pt',sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    canonical=root/'capture_manifest.json'
    manifest=dict(schema='prismaquant.tessera_calibration_cache.v2',status='complete',
        identity=dict(schema='prismaquant.tessera_calibration_cache.v2',
            census_sha256=census_sha,units={n:[4,4] for n in H},
            storage_source='tessera_campaign_prefix_f32_v1',max_act_rows=2,
            calibration={k:v for k,v in provenance.items() if k!='hessian_role'}),
        entries=entries)
    manifest_sha=write_json(canonical,manifest)
    digest=ActivationSource(H,provenance).capture_sha256()
    payload=dict(schema=SCHEMA,canonical_capture=dict(path=str(canonical),sha256=manifest_sha),
        census=dict(path=str(census),sha256=census_sha),provenance=provenance,
        counts=counts,hessians={n:tensor_identity(v) for n,v in H.items()},
        capture_sha256=digest,rows=[dict(units=sorted(H),capture_sha256=digest)],
        load_policy=dict(POLICY))
    handoff=tmp_path/'hessian_capture.references.json';write_json(handoff,payload)
    return handoff,payload,H,canonical,manifest


def test_public_reader_binds_commitments_without_loading_h_then_verifies_access(reference, monkeypatch):
    handoff,payload,H,_,_=reference
    original=torch.load;loads=[]
    def observed(path,*args,**kwargs):
        assert not str(path).endswith('.json'), 'reference JSON must not take the eager torch.load route'
        loads.append((str(path),dict(kwargs)))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(torch,'load',observed)
    source=ActivationSource.from_capture(handoff)
    assert source.capture_sha256()==payload['capture_sha256']
    assert source.config_block()['hessian']['capture_sha256']==payload['capture_sha256']
    assert set(source.hessians)==set(H) and 'a' in source.hessians
    assert loads==[], 'metadata/config/key lookups must not read H payloads'
    actual=source.hessians['a']
    torch.testing.assert_close(actual,H['a'],rtol=0,atol=0)
    assert len(loads)==1 and loads[0][1]['mmap'] is True and loads[0][1]['weights_only'] is True
    assert source.hessians.receipt()['verified_units']==['a']
    assert source.hessians.receipt()['live_payloads']==0
    source.hessians.close()


@pytest.mark.parametrize('target', ['handoff', 'canonical', 'census'])
def test_metadata_replacement_invalidates_held_owner(reference, target):
    handoff,payload,_,canonical,_=reference
    source=ActivationSource.from_capture(handoff)
    path={'handoff':handoff,'canonical':canonical,'census':Path(payload['census']['path'])}[target]
    replacement=path.with_suffix('.new');replacement.write_bytes(path.read_bytes());replacement.replace(path)
    with pytest.raises(GrammarError, match='changed or was replaced'):
        source.capture_sha256()
    source.hessians.close()


@pytest.mark.parametrize('mode', ['bytes', 'missing', 'oversize'])
def test_each_consumption_reauthenticates_source(reference, mode):
    handoff,payload,H,canonical,_=reference
    source=ActivationSource.from_capture(handoff)
    torch.testing.assert_close(source.hessians['a'],H['a'])
    path=canonical.parent/'inputs/a.pt'
    if mode=='missing': path.unlink()
    elif mode=='oversize': path.write_bytes(b'x'*(POLICY['max_file_bytes']+1))
    else:
        raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
    with pytest.raises((GrammarError,OSError)):
        source.hessians['a']
    assert source.hessians.receipt()['loaded_entries']==1
    assert source.hessians.receipt()['live_payloads']==0
    source.hessians.close()


@pytest.mark.parametrize('edit', [
    lambda p:p.update(capture_sha256='0'*64),
    lambda p:p['rows'][0].update(capture_sha256='0'*64),
    lambda p:p['rows'][0].update(units=['a']),
    lambda p:p['hessians'].pop('b'),
    lambda p:p['load_policy'].update(max_hessian_bytes=16),
    lambda p:p['load_policy'].update(max_metadata_bytes=16),
    lambda p:p['counts'].update(a=9),
    lambda p:p['canonical_capture'].update(sha256='0'*64),
    lambda p:p['census'].update(sha256='0'*64),
])
def test_forged_commitments_or_bounds_refuse_without_tensor_load(reference,monkeypatch,edit):
    handoff,payload,*_=reference
    edit(payload);write_json(handoff,payload)
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('payload read before metadata accepted'))
    with pytest.raises(GrammarError): ActivationSource.from_capture(handoff)


def test_commitment_cannot_forge_actual_h_with_resealed_row(reference):
    from tessera.hessian_capture import capture_sha256_from_units
    handoff,payload,_,_,_=reference
    payload['hessians']['a']['sha256']='0'*64
    digest=capture_sha256_from_units(payload['provenance'],{n:v['sha256'] for n,v in payload['hessians'].items()})
    payload['capture_sha256']=digest;payload['rows'][0]['capture_sha256']=digest
    write_json(handoff,payload)
    source=ActivationSource.from_capture(handoff)
    assert source.hessians.receipt()['verified_units']==[]
    with pytest.raises(GrammarError,match='differs from its commitment'):source.hessians['a']
    assert source.hessians.receipt()['verified_units']==[]
    source.hessians.close()


def test_returned_h_is_detached_and_owner_does_not_retain_it(reference):
    import gc, weakref
    handoff,_,H,*_=reference
    source=ActivationSource.from_capture(handoff)
    returned=source.hessians['a']; weak=weakref.ref(returned)
    returned.mul_(8);del returned;gc.collect()
    assert weak() is None
    torch.testing.assert_close(source.hessians['a'],H['a'],rtol=0,atol=0)
    detached=source.hessians.descriptor;detached['hessians']['a']['sha256']='0'*64
    torch.testing.assert_close(source.hessians['a'],H['a'],rtol=0,atol=0)
    source.hessians.close()


def test_for_unit_and_cached_wire_identity_consume_verified_h(reference):
    from tessera.cached_unit import encoding_input_identity
    from types import SimpleNamespace
    handoff,_,_,canonical,_=reference
    source=ActivationSource.from_capture(handoff,ldlq_sigma=None)
    from tessera.manifest import ScalePlaneKind
    source.for_unit('a.weight',4,'cpu',scale_plane=ScalePlaneKind.CHANNEL)
    assert source.hessians.receipt()['loaded_entries']==1
    path=canonical.parent/'inputs/a.pt';raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
    with pytest.raises(GrammarError,match='checksum'):
        encoding_input_identity(torch.ones(4,4),'a',SimpleNamespace(name='unused_before_checksum'),256,activation=source)
    source.hessians.close()


def test_priced_v2_requires_exact_reference_binding_even_with_same_old_seal(reference,tmp_path):
    from test_priced_inputs_snapshot import exporter
    handoff,payload,H,*_=reference
    source=ActivationSource.from_capture(handoff)
    block=dict(schema='tessera.priced_export_inputs.v2',hessian_capture_sha256=source.capture_sha256(),
        input_global_scales={},hessian_reference_binding=source.reference_binding())
    build=tmp_path/'build.json'
    def snapshot():
        digest=write_json(build,{'priced_inputs':block})
        return exporter.PricedInputsSnapshot(build,digest)
    snapshot().require(source,{})
    assert source.hessians.receipt()['loaded_entries']==0
    with pytest.raises(SystemExit,match='canonical Hessian reference'):
        snapshot().require(ActivationSource(H,payload['provenance']),{})
    block['hessian_reference_binding']['canonical_capture_sha256']='0'*64
    with pytest.raises(SystemExit,match='canonical Hessian reference'):snapshot().require(source,{})
    block['schema']='tessera.priced_export_inputs.v1';block.pop('hessian_reference_binding')
    with pytest.raises(SystemExit,match='canonical Hessian reference'):snapshot().require(source,{})
    source.hessians.close()


def test_legacy_pt_reader_still_loads_eagerly(reference,tmp_path,monkeypatch):
    _,payload,H,*_=reference
    path=tmp_path/'legacy.pt';torch.save(dict(H=H,provenance=payload['provenance']),path)
    calls=[];original=torch.load
    def observed(*a,**kw):calls.append(a[0]);return original(*a,**kw)
    monkeypatch.setattr(torch,'load',observed)
    source=ActivationSource.from_capture(path)
    assert len(calls)==1 and isinstance(source.hessians,dict)
    assert source.capture_sha256()==payload['capture_sha256']
