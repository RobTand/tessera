"""Actual input/native owner controls for the direct vLLM execution exemption."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from routed_gate_staged_store_support import reader_class


def direct_fixture(tmp_path, *, digest=None, offset=7, length=10):
    original=tmp_path/'original.so'
    original.write_bytes(b'prefix!owned-wireignored-tail')
    manifest={'schema':'tessera.declared_inputs.v1','mount_prefix':str(tmp_path),'entries':[{'path':str(original),'offset':offset,'bytes':length,
        'sha256':digest or hashlib.sha256(b'owned-wire').hexdigest()}],
        'entry_count':1,'total_bytes':length}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    return reader_class()(path),original


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
    before = path.stat()
    path.write_bytes(b'prefix!other-wireignored-tail')
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
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


@pytest.mark.parametrize('fault',['schema','pb-schema','digest-missing','digest-short','duplicate','count','total','empty','not-plain-json'])
def test_declared_record_refuses_every_malformed_declaration(tmp_path,fault):
    original=tmp_path/'original.so';original.write_bytes(b'prefix!owned-wireignored-tail')
    entry={'path':str(original),'offset':7,'bytes':10,'sha256':hashlib.sha256(b'owned-wire').hexdigest()}
    record={'schema':'tessera.declared_inputs.v1','mount_prefix':str(tmp_path),'entries':[entry],'entry_count':1,'total_bytes':10}
    path=tmp_path/'declared.json'
    path.write_text(json.dumps(record))
    reader_class()(path).close()  # the accepted control beside every refusal
    if fault=='schema':del record['schema']
    elif fault=='pb-schema':record['schema']='prismaquant.prismabuild.data_manifest.v1'
    elif fault=='digest-missing':entry['sha256']=None
    elif fault=='digest-short':entry['sha256']='ab'
    elif fault=='duplicate':record['entries']=[entry,dict(entry)];record['entry_count']=2;record['total_bytes']=20
    elif fault=='count':record['entry_count']=2
    elif fault=='total':record['total_bytes']=11
    elif fault=='empty':record.update(entries=[],entry_count=0,total_bytes=0)
    path.write_text(json.dumps(record))
    if fault=='not-plain-json':
        import gzip;path.write_bytes(gzip.compress(path.read_bytes()))
    with pytest.raises(ValueError):reader_class()(path)


@pytest.mark.parametrize('fault', ['relative', 'outside', 'missing-mount', 'uppercase-digest'])
def test_declared_paths_are_bound_to_the_named_mount(tmp_path, fault):
    from declared_inputs_support import manifest_value, write_manifest
    from experiments.t8r_speed.declared_inputs import read_declared_inputs
    source = tmp_path / 'owned'
    source.write_bytes(b'owned')
    record = manifest_value([{'path': str(source), 'offset': 0, 'bytes': 5,
                              'sha256': hashlib.sha256(b'owned').hexdigest()}], 'whole')
    path = write_manifest(tmp_path / 'declaration.json', record)
    read_declared_inputs(path)
    if fault == 'relative':
        record['entries'][0]['path'] = 'owned'
    elif fault == 'outside':
        record['mount_prefix'] = str(tmp_path / 'other')
    elif fault == 'missing-mount':
        del record['mount_prefix']
    else:
        record['entries'][0]['sha256'] = record['entries'][0]['sha256'].upper()
    write_manifest(path, record)
    with pytest.raises(ValueError):
        read_declared_inputs(path)


@pytest.mark.parametrize('frontier', [0, None, '5', True, 6])
def test_phase_frontiers_are_monotone_integer_byte_counts(tmp_path, frontier):
    from declared_inputs_support import manifest_value, write_manifest
    from experiments.t8r_speed.declared_inputs import validate_whole_phase_readset
    record = manifest_value([{'path': str(tmp_path / 'owned'), 'offset': 0, 'bytes': 5,
                              'sha256': 'a' * 64}], 'whole')
    path = write_manifest(tmp_path / 'declaration.json', record)
    validate_whole_phase_readset(path, 'whole')
    record['annotations']['phases'].append({'name': 'trailing', 'cumulative_bytes': frontier})
    write_manifest(path, record)
    with pytest.raises(ValueError, match='cumulative_bytes'):
        validate_whole_phase_readset(path, 'whole')
