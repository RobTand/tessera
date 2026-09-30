"""Additive package source identities; no wire or qualified record migration."""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys

import pytest

from tessera import cached_unit
from tessera.serving import source_identity

V1 = 'tessera.package_source.v1'
ENCODER_V1 = 'tessera.encoder_source.v1'
V2 = 'tessera.package_source.v2'
CPP = (b'constexpr char embedded[] = R"BIN(before\0after)BIN";\n'
       b'static_assert(sizeof(embedded) == 13);\n'
       b'static_assert(embedded[6] == 0);\n'
       b'static_assert(embedded[7] == \'a\');\n'
       b'int main() { return embedded[12]; }\n')


def literal(records, *, profile=None, framed=False):
    h = hashlib.sha256(b'' if profile is None else profile.encode('ascii') + b'\0')
    for name, raw in sorted(records.items(), key=lambda entry: entry[0].encode('utf-8')):
        name = name.encode('utf-8')
        h.update(len(name).to_bytes(8, 'big') + name if framed else name + b'\0')
        h.update(len(raw).to_bytes(8, 'big') + raw if framed else raw + b'\0')
    return h.hexdigest()


def package(root, records):
    directory = root / 'tessera'
    (directory / 'serving').mkdir(parents=True)
    for name, raw in records.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return directory


@pytest.mark.parametrize('owner', ['encoder', 'serving'])
def test_counterexample_separates_v2_without_changing_v1(tmp_path, owner):
    embedded_name = b'tessera/b.py' if owner == 'serving' else b'b.py'
    merged = {'__init__.py': b'', 'a.py': b'first\0' + embedded_name + b'\0second'}
    split = {'__init__.py': b'', 'a.py': b'first', 'b.py': b'second'}
    maps = []
    for name, files in [('merged', merged), ('split', split)]:
        root = tmp_path / name
        directory = package(root, files)
        if owner == 'encoder':
            observed = cached_unit._encoder_source_profiles(directory)
            expected_v1 = literal(files)
        else:
            observed = source_identity.serving_source_profiles(root)
            prefixed = {'tessera/' + key: value for key, value in files.items()}
            expected_v1 = literal(prefixed, profile=V1)
        maps.append(observed)
        assert observed[ENCODER_V1 if owner == 'encoder' else V1] == expected_v1
    legacy_name = ENCODER_V1 if owner == 'encoder' else V1
    assert maps[0][legacy_name] == maps[1][legacy_name]
    assert maps[0][V2] != maps[1][V2]


@pytest.mark.parametrize('owner', ['encoder', 'serving'])
def test_legitimate_cpp_nul_and_exact_v2_frame(tmp_path, owner):
    records = {'__init__.py': b'', 'embedded.cpp': CPP, 'é.py': b'utf8 name'}
    directory = package(tmp_path, records)
    if owner == 'encoder':
        actual = cached_unit._encoder_source_profiles(directory)
        legacy_name, legacy_tag = ENCODER_V1, None
    else:
        actual = source_identity.serving_source_profiles(tmp_path)
        records = {'tessera/' + key: value for key, value in records.items()}
        legacy_name, legacy_tag = V1, V1
    assert actual == {legacy_name: literal(records, profile=legacy_tag),
                      V2: literal(records, profile=V2, framed=True)}
    assert CPP.count(b'\0') == 1 and len(CPP) == 196


def test_serving_snapshot_cache_and_result_isolation(tmp_path):
    root = package(tmp_path, {'__init__.py': b'original'})
    first = source_identity.serving_source_profiles(tmp_path)
    assert source_identity.serving_source_sha256(tmp_path) == first[V1]
    first[V2] = '0' * 64
    observed = source_identity.serving_source_profiles(tmp_path)
    assert observed[V2] != '0' * 64
    (root / '__init__.py').write_bytes(b'changed')
    assert source_identity.serving_source_profiles(tmp_path) == observed


def test_cached_encoder_profiles_preserve_the_legacy_api():
    cached_unit.encoder_source_sha256.cache_clear()
    expected = cached_unit.encoder_source_sha256()
    profiles = cached_unit.encoder_source_profiles()
    assert profiles[ENCODER_V1] == expected
    assert len(profiles[V2]) == 64
    profiles[V2] = '0' * 64
    assert cached_unit.encoder_source_profiles()[V2] != '0' * 64


def test_actual_installed_git_package_keeps_legacy_bytes():
    dist = metadata.distribution('tessera-quant')
    direct_url = dist.read_text('direct_url.json')
    if direct_url is None:
        pytest.skip('the installed-source control requires a Git-provenanced package')
    url = json.loads(direct_url)
    if 'vcs_info' not in url or url.get('dir_info', {}).get('editable', False):
        pytest.skip('the installed-source control requires a noneditable Git package')
    commit = url['vcs_info']['commit_id']
    assert len(commit) == 40 and all(c in '0123456789abcdef' for c in commit)
    root = Path(dist.locate_file('tessera')).resolve()
    assert root.is_relative_to(Path(sys.prefix).resolve())
    files = {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob('*'))
             if p.is_file() and p.suffix in {'.py', '.cu', '.cuh', '.cpp', '.h'}}
    result = cached_unit._encoder_source_profiles(root)
    legacy_digest = hashlib.sha256()
    for name in sorted(files, key=Path):
        legacy_digest.update(name.encode() + b'\0' + files[name] + b'\0')
    assert result[ENCODER_V1] == legacy_digest.hexdigest()
    assert result[V2] == literal(files, profile=V2, framed=True)
    print(json.dumps({'actual_package': str(root), 'commit': commit,
                      'files': len(files), 'legacy_sha256': result[ENCODER_V1]},
                     sort_keys=True))
