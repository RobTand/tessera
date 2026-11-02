"""Imports point from the slow layers to the fast ones, never back (#1588).

tests/conftest.py imports the byte layer for every test, so whatever the byte
layer names, statically, every test depends on: Bazel reruns the whole gate
when it changes. A name inside a function or behind a module __getattr__ is
still a dependency. These read the source, so they hold for imports a run
never reaches.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: The modules the pure job proves import without torch (tools/ci/gate_boundary.py).
BYTE_LAYER = ('alphabet', 'canonical', 'container', 'errors', 'exact', 'grammar', 'layout', 'manifest',
              'planes', 'provenance', 'trellis')


def imported(module):
    """Every tessera module a module names in an import statement, anywhere in it."""
    tree = ast.parse((ROOT / 'src/tessera' / f'{module}.py').read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.removeprefix('tessera.') for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = (node.module or '') if node.level else (node.module or '').removeprefix('tessera')
            base = base.lstrip('.')
            names |= {f'{base}.{alias.name}'.strip('.') if base else alias.name for alias in node.names}
            if base:
                names.add(base)
    return names


def test_the_byte_layer_names_no_cutter():
    """tessera.slicing needs torch and the encoder; the byte layer re-exporting
    it lazily put the encoder, export and serving stack under every test."""
    offenders = {module: sorted(name for name in imported(module) if name.split('.')[0] == 'slicing')
                 for module in BYTE_LAYER}
    assert {module: names for module, names in offenders.items() if names} == {}


def test_the_encoder_names_no_identity_built_from_it():
    """tessera.encoder_identity runs the exporter, and so the encoder, to
    compute the encoder's identity. The encoder asking it whether that build
    was running closed an import cycle through export and the serving stack;
    it asks tessera.fixture_build instead."""
    assert 'encoder_identity' not in {name.split('.')[0] for name in imported('encode')}
