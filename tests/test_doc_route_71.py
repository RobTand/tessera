"""route_for_grid's docstring must name the shipped BF16 route (#71).

Issue #9 ("BF16 has a wire and no serving route") is closed: ``TESSERA_BF16``
is a ``ROUTES`` entry and ``route_for_grid("BF16")`` returns it, feeding the
export gate. The docstring still said the route does not exist, which is the
opposite of the shipped behavior.
"""
from __future__ import annotations


def test_route_for_grid_docstring_names_the_shipped_bf16_route():
    """BF16 resolves through ROUTES; the docstring must not call it routeless."""
    from tessera.serving.scheme import ROUTES, route_for_grid

    assert route_for_grid("BF16") == "TESSERA_BF16"
    assert "BF16" in ROUTES["TESSERA_BF16"]["grids"]
    doc = route_for_grid.__doc__ or ""
    assert "whose route does not" not in doc
    assert not any(
        "BF16" in line and "no decoder" in line for line in doc.splitlines()
    ), "docstring still lists BF16 as a grid with no decoder"
    assert "TESSERA_BF16" in doc
