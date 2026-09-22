import hashlib
import json
import os
from pathlib import Path
import pytest
from experiments.native_local_inputs import LocalNativeInputs


def fixture(tmp_path):
    root=tmp_path/'local';root.mkdir(mode=0o700)
    path=root/'wire';path.write_bytes(b'actual bytes');path.chmod(0o400)
    bound={'path':str(path),'bytes':12,'sha256':hashlib.sha256(b'actual bytes').hexdigest()}
    bundle={'schema':'prismaquant.native_local_handoff.v1','action_key':'owner',
        'max_bytes':10000,'reserved_bytes':12,'files':{'wire':bound}}
    raw=json.dumps(bundle).encode();manifest=root/'bundle.json';manifest.write_bytes(raw)
    return {'path':str(manifest),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)},bound


def test_independent_consumer_reads_exact_local_bytes(tmp_path):
    bundle,bound=fixture(tmp_path);owner=LocalNativeInputs(bundle,'owner')
    try:assert owner.read(bound['path'],bound['sha256'],12)==b'actual bytes'
    finally:owner.close()


def test_local_bundle_refuses_other_action(tmp_path):
    bundle,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='different admitted action'):LocalNativeInputs(bundle,'other')


def test_consumer_refuses_payload_symlink_swap(tmp_path):
    bundle,bound=fixture(tmp_path);owner=LocalNativeInputs(bundle,'owner')
    target=tmp_path/'canonical';target.write_bytes(b'actual bytes')
    path=Path(bound['path']);path.unlink();path.symlink_to(target)
    try:
        with pytest.raises(OSError):owner.read(path,bound['sha256'],12)
    finally:owner.close()


def test_consumer_refuses_canonical_alias_and_oversize(tmp_path):
    bundle,bound=fixture(tmp_path);owner=LocalNativeInputs(bundle,'owner')
    try:
        with pytest.raises(ValueError,match='outside'):owner.read(tmp_path/'canonical',bound['sha256'],12)
        with pytest.raises(ValueError,match='invalid local payload'):owner.read(bound['path'],bound['sha256'],11)
    finally:owner.close()


def test_consumer_refuses_replaced_same_size_payload(tmp_path):
    bundle,bound=fixture(tmp_path);owner=LocalNativeInputs(bundle,'owner')
    path=Path(bound['path']);path.chmod(0o600);path.write_bytes(b'changed data')
    try:
        with pytest.raises(ValueError,match='digest differs'):owner.read(path,bound['sha256'],12)
    finally:owner.close()
