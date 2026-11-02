"""Shared committed Hessian reference documents and pytest fixture."""
import hashlib
import json

import pytest
import torch

from tessera.cached_unit import tensor_identity
from tessera.export import ActivationSource
from reuse_authority_fixture import CANONICAL_CAPTURE

SCHEMA = 'tessera.hessian_capture.references.v1'



POLICY = dict(schema='tessera.hessian_reference_load.v1',
              max_metadata_bytes=1024**2, max_file_bytes=1024**2,
              max_hessian_bytes=1024**2)



def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()



def build_reference(tmp_path, canonical_capture=CANONICAL_CAPTURE):
    """A reference document over a canonical capture named ``canonical_capture``."""
    schema, source = canonical_capture
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
            source=source,count=8,max_abs=1.0),p)
        entries[name]=dict(path=f'inputs/{name}.pt',sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    canonical=root/'capture_manifest.json'
    manifest=dict(schema=schema,status='complete',
        identity=dict(schema=schema,
            census_sha256=census_sha,units={n:[4,4] for n in H},
            storage_source=source,max_act_rows=2,
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



@pytest.fixture
def reference(tmp_path):
    return build_reference(tmp_path)
