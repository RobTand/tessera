"""Registered grid identities reuse bytes already hashed for the wire registry."""
from dataclasses import FrozenInstanceError, replace
import hashlib

import pytest

from tessera.alphabet import E2M1_GRID, PayloadGrid, SERIALISABLE_GRIDS, grid_digest


@pytest.mark.parametrize('digest,grid', list(SERIALISABLE_GRIDS.items()),
                         ids=lambda value: value.name if isinstance(value, PayloadGrid) else value[:8])
def test_registered_grid_digest_does_not_rehash_payload(monkeypatch, digest, grid):
    calls = 0
    original = hashlib.sha256

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(hashlib, 'sha256', counted)
    assert grid_digest(grid) == digest
    assert grid_digest(grid) == digest
    assert calls == 0, 'registered immutable grid digest rehashed its entire payload'


@pytest.mark.parametrize('field,value', [
    ('name', 'changed-grid'),
    ('values', (0.25,) + E2M1_GRID.values[1:]),
    ('native', (1,) + E2M1_GRID.native[1:]),
    ('arity', 2),
    ('keys', ((15,),) + E2M1_GRID.keys[1:]),
    ('partition', 'coset'),
])
def test_replaced_registered_field_cannot_reuse_old_digest(field, value):
    original = getattr(E2M1_GRID, field)
    before = grid_digest(E2M1_GRID)
    try:
        # Even bypassing frozen dataclass assignment must not return stale wire identity.
        object.__setattr__(E2M1_GRID, field, value)
        assert grid_digest(E2M1_GRID) != before
    finally:
        object.__setattr__(E2M1_GRID, field, original)
    assert grid_digest(E2M1_GRID) == before


def test_unregistered_equal_grid_then_changed_values_remains_content_addressed():
    values = list(E2M1_GRID.values)
    other = replace(E2M1_GRID, values=values)
    before = grid_digest(other)
    assert before == grid_digest(E2M1_GRID)
    values[0] = 0.25
    assert grid_digest(other) != before
    assert grid_digest(other) not in SERIALISABLE_GRIDS


def test_dynamic_registry_entry_does_not_cache_mutable_payload(monkeypatch):
    values = list(E2M1_GRID.values)
    other = replace(E2M1_GRID, name='dynamic-grid', values=values)
    before = grid_digest(other)
    monkeypatch.setitem(SERIALISABLE_GRIDS, before, other)
    assert grid_digest(other) == before
    values[0] = 0.25
    assert grid_digest(other) != before
    assert grid_digest(other) not in SERIALISABLE_GRIDS


def test_registered_grid_still_refuses_normal_mutation():
    with pytest.raises(FrozenInstanceError):
        E2M1_GRID.name = 'changed-grid'
