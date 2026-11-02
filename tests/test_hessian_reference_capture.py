"""The public capture reader consumes committed canonical H one unit at a time."""
import json
import os
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tessera.export import ActivationSource
from tessera.errors import GrammarError
from reuse_authority_fixture import CANONICAL_CAPTURE
from hessian_reference_support import POLICY, reference, write_json


def test_public_reader_binds_commitments_without_loading_h_then_verifies_access(reference, monkeypatch):
    handoff,payload,H,_,_=reference
    original=torch.load;loads=[]
    def observed(path,*args,**kwargs):
        assert not str(path).endswith('.json'), 'reference JSON must not take the eager torch.load route'
        loads.append((str(path),dict(kwargs)))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(torch,'load',observed)
    source=ActivationSource.from_capture(handoff, canonical_capture=CANONICAL_CAPTURE)
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
    source=ActivationSource.from_capture(handoff, canonical_capture=CANONICAL_CAPTURE)
    path={'handoff':handoff,'canonical':canonical,'census':Path(payload['census']['path'])}[target]
    replacement=path.with_suffix('.new');replacement.write_bytes(path.read_bytes());replacement.replace(path)
    with pytest.raises(GrammarError, match='changed or was replaced'):
        source.capture_sha256()
    source.hessians.close()


@pytest.mark.parametrize('mode', ['bytes', 'missing', 'oversize'])
def test_each_consumption_reauthenticates_source(reference, mode):
    handoff,payload,H,canonical,_=reference
    source=ActivationSource.from_capture(handoff, canonical_capture=CANONICAL_CAPTURE)
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
    with pytest.raises(GrammarError): ActivationSource.from_capture(handoff, canonical_capture=CANONICAL_CAPTURE)


def test_commitment_cannot_forge_actual_h_with_resealed_row(reference):
    from tessera.hessian_capture import capture_sha256_from_units
    handoff,payload,_,_,_=reference
    payload['hessians']['a']['sha256']='0'*64
    digest=capture_sha256_from_units(payload['provenance'],{n:v['sha256'] for n,v in payload['hessians'].items()})
    payload['capture_sha256']=digest;payload['rows'][0]['capture_sha256']=digest
    write_json(handoff,payload)
    source=ActivationSource.from_capture(handoff, canonical_capture=CANONICAL_CAPTURE)
    assert source.hessians.receipt()['verified_units']==[]
    with pytest.raises(GrammarError,match='differs from its commitment'):source.hessians['a']
    assert source.hessians.receipt()['verified_units']==[]
    source.hessians.close()


def test_returned_h_is_detached_and_owner_does_not_retain_it(reference):
    import gc, weakref
    handoff,_,H,*_=reference
    source=ActivationSource.from_capture(handoff, canonical_capture=CANONICAL_CAPTURE)
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
    source=ActivationSource.from_capture(handoff, canonical_capture=CANONICAL_CAPTURE,ldlq_sigma=None)
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
    source=ActivationSource.from_capture(handoff, canonical_capture=CANONICAL_CAPTURE)
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
    source=ActivationSource.from_capture(path, canonical_capture=CANONICAL_CAPTURE)
    assert len(calls)==1 and isinstance(source.hessians,dict)
    assert source.capture_sha256()==payload['capture_sha256']


def test_unsorted_reference_json_preserves_existing_capture_seal(reference):
    handoff,payload,H,*_=reference
    payload['hessians']={name:payload['hessians'][name] for name in ('b','a')}
    handoff.write_text(json.dumps(payload,sort_keys=False))
    source=ActivationSource.from_capture(handoff, canonical_capture=CANONICAL_CAPTURE)
    assert source.capture_sha256()==ActivationSource(H,payload['provenance']).capture_sha256()
    assert source.hessians.receipt()['loaded_entries']==0
    source.hessians.close()


def test_source_change_during_mapping_load_refuses_before_return(reference,monkeypatch):
    handoff,_,_,canonical,_=reference
    source=ActivationSource.from_capture(handoff, canonical_capture=CANONICAL_CAPTURE);original=torch.load
    def mutate_after_mapping(*args,**kwargs):
        result=original(*args,**kwargs)
        path=canonical.parent/'inputs/a.pt'
        before=path.stat()
        with path.open('r+b') as handle:
            handle.seek(-1,2);value=handle.read(1);handle.seek(-1,2);handle.write(bytes([value[0]^1]))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        return result
    monkeypatch.setattr(torch,'load',mutate_after_mapping)
    with pytest.raises(GrammarError,match='changed or was replaced'):
        source.hessians['a']
    assert source.hessians.receipt()['verified_units']==[]
    assert source.hessians.receipt()['live_payloads']==0
    source.hessians.close()
