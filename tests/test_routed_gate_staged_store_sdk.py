"""SDK schema/lease controls; explicitly require PrismaBuild at collection."""
import hashlib
import os
import pytest
pytest.importorskip("prismabuild", reason="public schema/lease controls require the published PB SDK")
from prismabuild import client
from _routed_gate_sdk_fixture import fixture
from test_routed_gate_staged_store import reader_class, forbidden_origin

def test_owned_bytes_survive_origin_and_stage_replacement(tmp_path, monkeypatch):
    reader, staged, opened, sdk = fixture(tmp_path)
    monkeypatch.setattr('builtins.open', forbidden_origin)
    owned = reader.read('/forbidden-origin/wire', 9)
    # Even a later same-name publication cannot change the already authenticated
    # owner returned to the intake. There is no hash-then-origin-reread seam.
    replacement = tmp_path / 'replacement'
    replacement.write_bytes(b'other-wire')
    replacement.replace(staged)
    assert owned == b'owned-wire'
    assert hashlib.sha256(owned).hexdigest() == reader.reads[0]['sha256']
    with pytest.raises(OSError):
        os.fstat(opened[0])
    reader.close()
    with pytest.raises(ValueError, match='released'):
        reader.read('/forbidden-origin/wire', 9)


@pytest.mark.parametrize('payload,reason', [(b'wrong-wire', 'digest'),
    (b'owned', 'short'), (b'owned-wire-plus', 'oversized')])
def test_bad_owned_bytes_refuse_and_close_descriptor(tmp_path, payload, reason):
    reader, staged, opened, sdk = fixture(tmp_path, payload)
    with pytest.raises(ValueError, match=reason):
        reader.read('/forbidden-origin/wire', 9)
    with pytest.raises(OSError):
        os.fstat(opened[0])
    reader.close()


def test_wrong_offset_refuses_without_open_or_origin_fallback(tmp_path):
    reader, staged, opened, sdk = fixture(tmp_path)
    with pytest.raises(ValueError, match='undeclared'):
        reader.read('/forbidden-origin/wire', 8)
    assert opened == []
    reader.close()


def test_pin_open_failure_does_not_fall_back(tmp_path):
    reader, staged, opened, sdk = fixture(tmp_path)
    sdk.open_pinned = forbidden_origin
    with pytest.raises(PermissionError):
        reader.read('/forbidden-origin/wire', 9)
    assert opened == []
    reader.close()


def test_missing_launch_context_refuses_before_acquire(tmp_path):
    reader, staged, opened, sdk = fixture(tmp_path)
    reader.close()
    sdk.injected_context = lambda: {'ok': False, 'refusal': 'no-launch-context'}
    sdk.acquire_for = lambda *a, **k: pytest.fail('must not acquire')
    with pytest.raises(ValueError, match='no-launch-context'):
        reader_class()(tmp_path/'manifest.json', sdk=sdk)


def test_failed_release_is_visible(tmp_path):
    reader, staged, opened, sdk = fixture(tmp_path)
    sdk.release = lambda *a, **k: False
    with pytest.raises(ValueError, match='release failed'):
        reader.close()
    assert reader.closed is False


def test_missing_public_cover_refuses_before_acquire(tmp_path):
    reader,staged,opened,sdk=fixture(tmp_path)
    reader.close()
    sdk.covers_for_keys=lambda *a,**k:{'ok':True,'covers':[],'expected':{}}
    sdk.acquire_for=lambda *a,**k:pytest.fail('missing proof must not acquire')
    with pytest.raises(ValueError,match='prove every'):
        reader_class()(tmp_path/'manifest.json',sdk=sdk)


def test_final_public_covers_can_resolve_a_lagging_composed_map(tmp_path):
    reader,staged,opened,sdk=fixture(tmp_path)
    reader.close()
    old_map=sdk.read_residency_map
    sdk.read_residency_map=lambda p:{**old_map(p),'entries':{}}
    second=reader_class()(tmp_path/'manifest.json',sdk=sdk)
    assert second.read('/forbidden-origin/wire',9)==b'owned-wire'
    second.close()


def test_complete_file_descriptor_uses_pin_and_stays_open_until_consumer_closes(tmp_path, monkeypatch):
    reader, staged, opened, sdk = fixture(tmp_path, offset=0)
    monkeypatch.setattr('builtins.open', forbidden_origin)
    fd, entry, serving = reader.pinned_file('/forbidden-origin/wire')
    assert os.read(fd, entry['bytes']) == b'owned-wire'
    assert serving['range_ref'] == client.residency_map_key('/forbidden-origin/wire', 0)
    assert not reader.closed
    os.close(fd)
    reader.close()


def test_complete_file_descriptor_refuses_partial_length_and_closes_fd(tmp_path):
    reader, staged, opened, sdk = fixture(tmp_path, b'owned', offset=0)
    with pytest.raises(ValueError, match='byte length'):
        reader.pinned_file('/forbidden-origin/wire')
    with pytest.raises(OSError):
        os.fstat(opened[0])
    reader.close()


