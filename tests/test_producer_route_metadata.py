from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tessera import metadata
from tessera.serving import scheme


def test_public_family_lookup_reads_the_served_route_owner():
    assert metadata.ROUTES is scheme.ROUTES
    assert metadata.TESSERA_FAMILIES is scheme.TESSERA_FAMILIES
    for family, route in scheme.ROUTES.items():
        for grid in route["grids"]:
            first = next(label for label, row in scheme.ROUTES.items() if grid in row["grids"])
            assert metadata.route_family_for_grid(grid) == first


def test_public_first_match_preserves_serving_ambiguity_refusal(monkeypatch):
    monkeypatch.setitem(scheme.ROUTES, "SYNTHETIC_FIRST", {"grids": ("SYNTHETIC_GRID",)})
    monkeypatch.setitem(scheme.ROUTES, "SYNTHETIC_SECOND", {"grids": ("SYNTHETIC_GRID",)})
    assert metadata.route_family_for_grid("SYNTHETIC_GRID") == "SYNTHETIC_FIRST"
    with pytest.raises(ValueError, match="held by more than one route"):
        scheme.route_for_grid("SYNTHETIC_GRID")


def test_public_lookup_refuses_an_unowned_grid():
    with pytest.raises(ValueError, match="no route family holds grid 'UNOWNED_GRID'"):
        metadata.route_family_for_grid("UNOWNED_GRID")
    assert scheme.route_for_grid("UNOWNED_GRID") is None


def test_api_extraction_keeps_the_exact_reviewed_v60_contract():
    source = Path(scheme.__file__).with_name("runtime_contract.json").read_bytes()
    assert hashlib.sha256(source).hexdigest() == "ee065629b081d913a0351e43160c5c6e1bd38fa628cafd51e756e9caf3bb334e"
