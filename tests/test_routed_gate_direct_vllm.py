"""Actual input/native owner controls for the direct vLLM execution exemption."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_routed_gate_staged_store import reader_class


def direct_fixture(tmp_path, *, digest=None, offset=7, length=10):
    original=tmp_path/'original.so'
    original.write_bytes(b'prefix!owned-wireignored-tail')
    manifest={'entries':[{'path':str(original),'offset':offset,'bytes':length,
        'sha256':digest or hashlib.sha256(b'owned-wire').hexdigest()}],
        'entry_count':1,'total_bytes':length}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    sdk=SimpleNamespace(read_data_manifest=lambda p:(json.loads(Path(p).read_bytes()),'identity'),
        injected_context=lambda:pytest.fail('direct mode must not acquire a PB context'))
    return reader_class()(path,sdk=sdk,direct_vllm=True),original


def test_direct_reads_only_the_declared_range_and_holds_its_descriptor(tmp_path):
    reader,path=direct_fixture(tmp_path)
    fd=reader.direct_files[str(path)]['fd']
    assert reader.read(path,7)==b'owned-wire'
    assert os.fstat(fd).st_size==29
    with pytest.raises(ValueError,match='undeclared'):reader.read(path,8)
    reader.close()
    with pytest.raises(OSError):os.fstat(fd)
    with pytest.raises(ValueError,match='released'):reader.read(path,7)
    assert reader.direct_record['after']==reader.direct_record['before']


def test_direct_path_replacement_never_adopts_replacement_bytes(tmp_path):
    reader,path=direct_fixture(tmp_path)
    replacement=tmp_path/'replacement';replacement.write_bytes(b'prefix!other-wireignored-tail')
    replacement.replace(path)
    # Unlink normally changes the held inode's ctime and refuses. On a
    # filesystem preserving its fingerprint, only the old authenticated FD
    # can be consumed; the replacement can never pass as the sealed input.
    try:assert reader.read(path,7)==b'owned-wire'
    except ValueError as exc:assert 'identity changed' in str(exc)
    try:reader.close()
    except ValueError as exc:assert 'identity changed' in str(exc)
    assert reader.closed


def test_direct_inplace_mutation_refuses_and_closes_all_held_files(tmp_path):
    reader,path=direct_fixture(tmp_path)
    fd=reader.direct_files[str(path)]['fd']
    path.write_bytes(b'prefix!other-wireignored-tail')
    with pytest.raises(ValueError,match='identity changed'):reader.read(path,7)
    with pytest.raises(ValueError,match='identity changed'):reader.close()
    assert reader.closed
    with pytest.raises(OSError):os.fstat(fd)


def test_direct_range_digest_refuses(tmp_path):
    reader,path=direct_fixture(tmp_path,digest='0'*64)
    with pytest.raises(ValueError,match='digest differs'):reader.read(path,7)
    reader.close()


def test_direct_short_read_refuses_even_if_file_stat_is_stable(tmp_path,monkeypatch):
    reader,path=direct_fixture(tmp_path)
    monkeypatch.setattr(os,'pread',lambda *a:b'')
    with pytest.raises(ValueError,match='short'):reader.read(path,7)
    reader.close()


def test_direct_out_of_file_range_refuses_before_read(tmp_path):
    with pytest.raises(ValueError,match='range exceeds'):direct_fixture(tmp_path,offset=27)


def test_direct_symlink_refuses(tmp_path):
    target=tmp_path/'target';target.write_bytes(b'prefix!owned-wireignored-tail')
    original=tmp_path/'original.so';original.symlink_to(target)
    with pytest.raises(OSError):direct_fixture(tmp_path)
