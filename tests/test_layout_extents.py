"""Pricing shares the writer's extents without allocating or hashing payloads."""
from dataclasses import fields

import pytest

from tessera import calculator, layout
from tessera.canonical import Writer
from tessera.manifest import Geometry
from tessera.planes import PlaneDescriptor, PlaneLayout


def _forbid(*args, **kwargs):
    raise AssertionError("pricing allocated or hashed a payload")


def test_calculator_never_materializes_payloads(monkeypatch):
    monkeypatch.setattr(layout, "bytes", _forbid, raising=False)
    monkeypatch.setattr(calculator, "bytes", _forbid, raising=False)
    monkeypatch.setattr(layout.hashlib, "sha256", _forbid)
    assert calculator.terminal_rate(512, 4096, 4096, window_bits=16) > 2


@pytest.mark.parametrize("wire_layout", list(PlaneLayout))
@pytest.mark.parametrize("mode", ["plain", "completion", "release", "row", "shard", "arity", "span", "refine"])
def test_extent_matches_real_plane_and_terminal(wire_layout, mode):
    from tessera.layout import build_plane_extents, build_terminal_extent

    geometry = Geometry(32, 40, 16, 32, 16, 32 * 40)
    rates = (1, 2) * 20
    spec = layout.TerminalSpec(
        "extent", (1 if mode == "completion" else 0,) * 40,
        released_positions=5 if mode == "release" else 0,
        with_scale_base=mode != "row", with_scale_refine=mode == "refine",
        with_diagonals=mode == "plain", with_row_scale=mode == "row",
        state_bits=7 if mode == "shard" else 0,
        scale_refine_halves=3 if mode == "refine" else None,
    )
    kwargs = dict(spec=spec, with_diagonals=spec.with_diagonals,
                  with_row_scale=spec.with_row_scale, state_bits=spec.state_bits,
                  arity=2 if mode == "arity" else 1,
                  span=2 if mode == "span" else 1,
                  alignment_bytes=16, layout=wire_layout)
    alphabet, descendant = b"alphabet", b"descendants"
    extents = build_plane_extents(geometry, rates, len(alphabet), len(descendant), **kwargs)
    planes = layout.build_planes(geometry, rates, alphabet, descendant, **kwargs)
    for extent, plane in zip(extents, planes, strict=True):
        assert not isinstance(extent, PlaneDescriptor)
        assert not hasattr(extent, "content_digest")
        assert not hasattr(extent, "encode")
        for field in fields(extent):
            assert getattr(extent, field.name) == getattr(plane, field.name)
        assert extent.byte_length() == plane.byte_length()
        # The full descriptor remains serializable with its real digest.
        writer = Writer()
        plane.encode(writer)
        assert writer.bytes
    terminal_kwargs = dict(arity=kwargs["arity"], span=kwargs["span"])
    extent = build_terminal_extent(geometry, rates, spec, extents, len(alphabet), len(descendant), **terminal_kwargs)
    record = layout.build_terminal(geometry, rates, spec, planes, len(alphabet), len(descendant), **terminal_kwargs)
    assert not hasattr(extent, "payload_digest")
    assert (extent.plane_elements, extent.exact_bytes, extent.exact_bpp) == (
        record.plane_elements, record.exact_bytes, record.exact_bpp)


def test_large_extent_has_no_payload_allocation_or_digest(monkeypatch):
    from tessera.layout import build_plane_extents, build_terminal_extent

    monkeypatch.setattr(layout, "bytes", _forbid, raising=False)
    monkeypatch.setattr(layout.hashlib, "sha256", _forbid)
    rows, columns = 1 << 30, 32
    geometry = Geometry(rows, columns, 16, 32, 16, rows * columns)
    rates = (2,) * columns
    spec = layout.TerminalSpec("large", (0,) * columns, with_scale_base=False)
    planes = build_plane_extents(geometry, rates, 0, 0, spec=spec, with_diagonals=False)
    extent = build_terminal_extent(geometry, rates, spec, planes, 0, 0)
    assert extent.exact_bytes == rows * columns // 4
    assert extent.exact_bpp == 2
