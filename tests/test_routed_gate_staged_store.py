"""The exact replay Store consumes owned staged bytes, never original paths."""
import ast
import hashlib
from pathlib import Path

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
    store.inputs = type('Inputs', (), {'tensor': lambda self, root, key: b'owned-wire'})()
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
