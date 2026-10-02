"""The exact replay Store consumes owned staged bytes, never original paths."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def store_class():
    path = ROOT / 'experiments/t8r_speed/bench_t8r.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Store')
    scope = {'os': __import__('os'), 'json': __import__('json'), 'safe_open': forbidden_origin}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['Store']


def forbidden_origin(*args, **kwargs):
    raise PermissionError('origin read forbidden by regression tripwire')


def test_existing_store_uses_owned_wire_under_origin_denial():
    cls = store_class()
    store = cls.__new__(cls)
    store.root = '/forbidden-origin'
    store.index = {'wire': 'shard.safetensors'}
    store._open = {}
    store.inputs = type('Inputs', (), {'tensor': lambda self, root, key, **kwargs: b'owned-wire'})()
    assert store.get('wire') == b'owned-wire'


def test_existing_store_uses_owned_metadata_under_origin_denial(monkeypatch):
    monkeypatch.setattr('builtins.open', forbidden_origin)
    cls = store_class()
    store = cls.__new__(cls)
    store.root = '/forbidden-origin'
    store.inputs = type('Inputs', (), {'json': lambda self, path: {'config_groups': {}}})()
    assert store.metadata('config.json') == {'config_groups': {}}


def test_legacy_store_keeps_origin_behavior():
    cls = store_class()
    store = cls.__new__(cls)
    store.root = '/forbidden-origin'
    store.index = {'wire': 'shard.safetensors'}
    store._open = {}
    store.inputs = None
    with pytest.raises(PermissionError, match='origin read forbidden'):
        store.get('wire')


def reader_class():
    path = ROOT / 'experiments/t8r_speed/pb_staged_store.py'
    spec = importlib.util.spec_from_file_location('pb_staged_store', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.StagedInputs


def fixture(tmp_path, payload=b'owned-wire', expected=None):
    staged = tmp_path / 'staged'
    staged.write_bytes(payload)
    origin = '/forbidden-origin/wire'
    entry = {'path': origin, 'offset': 9, 'bytes': len(b'owned-wire'),
             'sha256': expected or hashlib.sha256(b'owned-wire').hexdigest()}
    manifest = {'entries': [entry], 'entry_count': 1, 'total_bytes': entry['bytes']}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    key = f'{origin}@9'
    action = 'a'*64
    opened = []
    sdk = SimpleNamespace(
        read_data_manifest=lambda p: (json.loads(Path(p).read_text()), 'identity'),
        injected_context=lambda: {'ok': True, 'ctx': {'action_key': action,
            'queue_root': str(tmp_path/'queue'), 'map_path': str(tmp_path/'maps/map.json')}},
        read_residency_map=lambda p: {'consumer_action_key': action,
            'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'tier_id': 'fixture', 'entries': {key: entry}},
        residency_map_key=lambda p,o: f'{p}@{o}',
        covers_for_keys=lambda *a,**k: {'ok': True, 'covers': []},
        acquire_for=lambda *a,**k: {'ok': True, 'pin_id': 'pin', 'ref_id': 'ref',
                                  'pin': {'stage_root': str(tmp_path)}},
        release=lambda *a,**k: True,
    )
    def pinned(*args, **kwargs):
        fd = os.open(staged, os.O_RDONLY)
        opened.append(fd)
        return fd, {'range_ref': key}
    sdk.open_pinned = pinned
    return reader_class()(path, sdk=sdk), staged, opened, sdk


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


def options():
    return SimpleNamespace(single_routing_file='recorded.pt',
        groups='experts.R1024.L10', ms='2048', input_manifest='sealed.json',
        routing=None, no_graph=True, power_s=30, warmup=10, iters=30,
        artifact='/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported')


def require_options(args, **kwargs):
    path = ROOT / 'experiments/t8r_speed/bench_t8r.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'require_single_replay_options')
    scope = {'ARTIFACT': '/legacy'}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    scope[node.name](args, **kwargs)


def test_exact_single_options_admit():
    require_options(options())


@pytest.mark.parametrize('field,value', [('groups','all'), ('ms','512,2048'),
    ('input_manifest',None), ('routing','unsealed-dir'), ('no_graph',False),
    ('power_s',2), ('artifact','older-T8R-artifact')])
def test_single_options_refuse_menu_or_unqualified_inputs(field, value):
    args = options()
    setattr(args, field, value)
    with pytest.raises(ValueError):
        require_options(args)


def test_single_options_refuse_stubbed_runtime():
    with pytest.raises(ValueError, match='stubbed'):
        require_options(options(), stubbed=True)
