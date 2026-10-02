"""Real fused framing controls; explicitly require its Torch dependency."""
import hashlib
import importlib.util
from pathlib import Path
import pytest
from tessera.fused import pack_fused
ROOT = Path(__file__).resolve().parents[1]

def frame_checker():
    path = ROOT / 'experiments/t8r_speed/pb_staged_store.py'
    spec = importlib.util.spec_from_file_location('pb_staged_store', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_cached_frame


def frame_fixture():
    from tessera.fused import pack_fused
    unit = b'independently-hashed-cached-unit'
    raw = bytearray(pack_fused([('gate',2048,unit)]))
    role = {'role':'gate', 'rows':2048, 'blob_bytes':len(raw),
            'cached_blob_sha256':hashlib.sha256(unit).hexdigest()}
    return raw, role


def test_cached_digest_covers_inner_not_exported_outer():
    raw, role = frame_fixture()
    proof = frame_checker()(raw,role)
    assert proof['inner_sha256'] == role['cached_blob_sha256']
    assert proof['outer_sha256'] == hashlib.sha256(raw).hexdigest()
    assert proof['outer_sha256'] != proof['inner_sha256']
    assert proof['outer_bytes'] == len(raw)
    assert proof['inner_bytes'] == len(b'independently-hashed-cached-unit')


@pytest.mark.parametrize('fault', ['role','rows','inner','length','extra','trailing','magic'])
def test_frame_rejects_wrong_or_extra_member_before_delivery(fault):
    from tessera.fused import pack_fused
    raw, role = frame_fixture()
    if fault == 'role': role['role'] = 'up'
    elif fault == 'rows': role['rows'] += 1
    elif fault == 'inner': role['cached_blob_sha256'] = '0'*64
    elif fault == 'length': role['blob_bytes'] += 1
    elif fault == 'extra': raw = bytearray(pack_fused([('gate',2048,b'independently-hashed-cached-unit'),('up',2048,b'extra')]))
    elif fault == 'trailing': raw += b'extra'
    elif fault == 'magic': raw[0] ^= 1
    from tessera.errors import GrammarError
    with pytest.raises((ValueError,GrammarError)):
        frame_checker()(raw,role)


