"""A seeded break counts as caught only when an expected target fails inside the selection (#1589)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/ci'))
SPEC = importlib.util.spec_from_file_location('seeded_breaks', ROOT / 'tools/ci/seeded_breaks.py')
seeded_breaks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seeded_breaks)


def test_an_expected_failure_inside_the_selection_is_caught():
    row = seeded_breaks.seed_verdict(['//tests:test_wire', '//tests:test_wire.bytes'],
                                     {'//tests:test_wire': 'FAILED', '//tests:test_wire.bytes': 'PASSED'},
                                     ['//tests:test_wire'])
    assert row == {'impacted_tests': 2, 'failing_selected': ['//tests:test_wire'], 'caught': True,
                   'example': '//tests:test_wire'}


def test_an_unexpected_failure_or_an_unselected_test_is_not():
    assert not seeded_breaks.seed_verdict(['//tests:test_pool'], {'//tests:test_pool': 'FAILED'},
                                          ['//tests:test_wire'])['caught']
    assert not seeded_breaks.seed_verdict([], {}, ['//tests:test_wire'])['caught']


def test_seeds_cover_at_least_eight_places_and_name_existing_files():
    assert len({place for place, *_ in seeded_breaks.SEEDS}) >= 8
    for _, path, _, expected in seeded_breaks.SEEDS:
        assert (ROOT / path).is_file(), path
        assert expected and all(label.startswith('//tests:test_') for label in expected)
