"""A canonical package hash must not hide a foreign loaded serving source."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest


def load_probe():
    path = Path(os.environ.get('TESSERA_ORIGIN_PROBE_SOURCE',
                               'experiments/original_wire_generation.py'))
    spec = importlib.util.spec_from_file_location('_original_generation_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_loaded_origins_are_accepted():
    pytest.importorskip('torch', reason='serving telemetry requires torch')
    record = load_probe().observed(SimpleNamespace(named_modules=lambda: []))['package_identity']
    assert record.get('loaded_module_origins_verified', True)
    assert record['fresh_encoder_source_sha256'] == record['independently_recomputed_source_sha256']


def test_probe_import_does_not_consume_worker_arguments(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['worker'])
    assert callable(load_probe().observed)


def test_foreign_loaded_source_is_rejected_even_when_package_hash_matches(tmp_path, monkeypatch):
    pytest.importorskip('torch', reason='serving telemetry requires torch')
    probe = load_probe()
    empty_model = SimpleNamespace(named_modules=lambda: [])
    before = probe.observed(empty_model)['package_identity']
    path = tmp_path / 'foreign_serving.py'
    path.write_text('VALUE = 7\n')
    name = 'tessera._origin_regression'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, name, module)
    after = probe.observed(empty_model)['package_identity']
    assert after['fresh_encoder_source_sha256'] == before['fresh_encoder_source_sha256']
    assert after['package_source_files'] == before['package_source_files']
    # The old proof checked only these unchanged package facts, retaining the
    # foreign module's paths without joining them to the measured package.
    assert not after.get('loaded_module_origins_verified', True), 'foreign loaded source passed the canonical package hash guard'
