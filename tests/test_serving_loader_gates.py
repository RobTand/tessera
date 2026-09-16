"""The load-time prepare gates derive from scheme.ROUTES, not from literals.

(The NVFP4 half of this file tested the retired ``ops.prepare_tessera_module``
wrapper; that wrapper is gone with the A4 whole-weight expansion, and the A4
lanes validate the compact reader's sidecar and per-role facts directly --
their refusals are covered by ``tests/test_serving_nvfp4_route.py`` and the
A4 owner's suite.  What remains here is the FP8 route's gate.)

``validate_tessera_scheme`` + ``refuse_unserveable_wire`` accept a wire at
export off ``ROUTES``; the ``prepare_*_module`` gates run AFTER, at load, and
restated the same facts in string literals (``{"TCQ"}``, ``{"LUT"}``,
``_GRID = "E4M3"`` / ``"BF16"``, ``!= "WINDOW"``, ``is not CHANNEL``) -- and
never checked span at all.  A ROUTES change adding a grid, moving a body or a
plane, or a span the sidecar never carried, was accepted at export and refused
at load (or served where no receipt describes it).

Every test below derives its expectation from ``ROUTES`` itself: it mutates
the table and asserts the gate follows, or feeds a wire the table does not
name and asserts a refusal.  Nothing here restates today's grid/body/plane
names -- a test that did would pass because of the roster, which is the
defect, not the check.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tessera.serving import bf16_route, fp8_route               # noqa: E402
from tessera.serving.scheme import (                             # noqa: E402
    ROUTES, TESSERA_BF16, TESSERA_FP8, TESSERA_NVFP4,
    parse_tessera_blob_for_scheme, validate_tessera_scheme)


def _ns(**kw):
    return SimpleNamespace(**kw)


# --- fakes -----------------------------------------------------------------
#
# The gates under test run before any packing/decoding, so refusal paths need
# no stubs at all.  Acceptance paths (a mutated ROUTES entry the gate must
# follow) stub the steps AFTER the gate: the packer, the reference decoder,
# the shared-global move and the extension handle.

def _window_roles(*, grid="E4M3", body="WINDOW", plane="CHANNEL", span=1,
                  steps=8, cols=64):
    from tessera.manifest import ScalePlaneKind

    unit = _ns(scale_plane=ScalePlaneKind[plane],
               body_bits=torch.zeros(steps, cols, dtype=torch.uint8),
               rates=(4,) * cols, window_bits=8,
               window_codes=torch.zeros(16, dtype=torch.uint8), span=span)
    grid_ns = _ns(name=grid, arity=1, native=list(range(256)), size=256)
    return [("weight", _ns(body=_ns(name=body), grid=grid_ns,
                            unit=unit, forests={}, code=None))]


def _stub_fp8_tail(monkeypatch):
    import tessera.decode

    tile = torch.zeros(8, 64, dtype=torch.uint8)
    scale = torch.zeros(8, dtype=torch.float32)

    def _window(body_bits, rates, window_bits, window_codes, device,
                code_map=None, initial_state=None):
        return _ns(cols=64, decode=lambda: tile.clone(),
                   resident_bytes=lambda: 8)

    monkeypatch.setattr(fp8_route, "prepare_window", _window)
    monkeypatch.setattr(tessera.decode, "materialize_fp8",
                        lambda unit, forests, code: (tile.clone(), scale.clone()))


def _stub_bf16_tail(monkeypatch):
    import tessera.decode

    tile = torch.zeros(8, 64, dtype=torch.bfloat16)
    scale = torch.zeros(8, dtype=torch.float32)

    def _window(body_bits, rates, window_bits, table, device,
                initial_state=None):
        return _ns(cols=64, decode=lambda: tile.clone(),
                   resident_bytes=lambda: 8)

    monkeypatch.setattr(bf16_route, "_window_table_values",
                        lambda parsed: torch.zeros(4, dtype=torch.bfloat16))
    monkeypatch.setattr(bf16_route, "prepare_window", _window)
    monkeypatch.setattr(tessera.decode, "materialize_bf16",
                        lambda unit, forests, code: (tile.clone(), scale.clone()))


# --- FP8: fp8_route.prepare_tessera_fp8_module --------------------------------

def test_fp8_control_wire_prepares(monkeypatch):
    _stub_fp8_tail(monkeypatch)
    module = fp8_route.prepare_tessera_fp8_module(_window_roles(), device="cpu")
    assert module.rows == 8 and module.columns == 64


def test_fp8_grid_follows_the_route_table(monkeypatch):
    """A grid the route's own entry lists is served; the ``_GRID`` literal
    refused it."""
    route = ROUTES[TESSERA_FP8]
    monkeypatch.setitem(route, "grids", tuple(route["grids"]) + ("E4M3X",))
    _stub_fp8_tail(monkeypatch)
    module = fp8_route.prepare_tessera_fp8_module(
        _window_roles(grid="E4M3X"), device="cpu")
    assert module.rows == 8


def test_fp8_body_follows_the_route_table(monkeypatch):
    route = ROUTES[TESSERA_FP8]
    monkeypatch.setitem(route, "body", "WOBBLE")
    _stub_fp8_tail(monkeypatch)
    module = fp8_route.prepare_tessera_fp8_module(
        _window_roles(body="WOBBLE"), device="cpu")
    assert module.rows == 8


def test_fp8_plane_follows_the_route_table(monkeypatch):
    route = ROUTES[TESSERA_FP8]
    monkeypatch.setitem(route, "plane", "LUT")
    _stub_fp8_tail(monkeypatch)
    module = fp8_route.prepare_tessera_fp8_module(
        _window_roles(plane="LUT"), device="cpu")
    assert module.rows == 8


def test_fp8_refuses_a_span_no_route_reads(monkeypatch):
    assert ROUTES[TESSERA_FP8]["span"] != 7
    _stub_fp8_tail(monkeypatch)  # the tail is stubbed so only the gate can refuse
    with pytest.raises(ValueError, match="span"):
        fp8_route.prepare_tessera_fp8_module(_window_roles(span=7), device="cpu")


# --- BF16: bf16_route.prepare_tessera_bf16_module ------------------------------

def test_bf16_control_wire_prepares(monkeypatch):
    _stub_bf16_tail(monkeypatch)
    module = bf16_route.prepare_tessera_bf16_module(
        _window_roles(grid="BF16"), device="cpu")
    assert module.rows == 8 and module.columns == 64


def test_bf16_grid_follows_the_route_table(monkeypatch):
    route = ROUTES[TESSERA_BF16]
    monkeypatch.setitem(route, "grids", tuple(route["grids"]) + ("BF16X",))
    _stub_bf16_tail(monkeypatch)
    module = bf16_route.prepare_tessera_bf16_module(
        _window_roles(grid="BF16X"), device="cpu")
    assert module.rows == 8


def test_bf16_body_follows_the_route_table(monkeypatch):
    route = ROUTES[TESSERA_BF16]
    monkeypatch.setitem(route, "body", "WOBBLE")
    _stub_bf16_tail(monkeypatch)
    module = bf16_route.prepare_tessera_bf16_module(
        _window_roles(grid="BF16", body="WOBBLE"), device="cpu")
    assert module.rows == 8


def test_bf16_plane_follows_the_route_table(monkeypatch):
    route = ROUTES[TESSERA_BF16]
    monkeypatch.setitem(route, "plane", "LUT")
    _stub_bf16_tail(monkeypatch)
    module = bf16_route.prepare_tessera_bf16_module(
        _window_roles(grid="BF16", plane="LUT"), device="cpu")
    assert module.rows == 8


def test_bf16_refuses_a_span_no_route_reads(monkeypatch):
    assert ROUTES[TESSERA_BF16]["span"] != 7
    _stub_bf16_tail(monkeypatch)  # the tail is stubbed so only the gate can refuse
    with pytest.raises(ValueError, match="span"):
        bf16_route.prepare_tessera_bf16_module(
            _window_roles(grid="BF16", span=7), device="cpu")


# --- the blob/sidecar comparison checks span too ------------------------------

def _fp8_scheme(rows=8, columns=64, **over):
    s = {"family": TESSERA_FP8, "grid": "E4M3", "body": "WINDOW",
         "plane": "CHANNEL", "q256": 1024, "rows": rows, "columns": columns,
         "wire_bytes": 32, "roles": [["weight", rows]]}
    s.update(over)
    return s


def _parsed_with_span(span):
    return _ns(
        body=_ns(name="WINDOW"), grid=_ns(name="E4M3", arity=1),
        manifest=_ns(geometry=_ns(rows=8, columns=64),
                     branch=_ns(root_q256=1024),
                     scale_plane=_ns(kind=_ns(name="CHANNEL")), span=span),
        forests={}, code=None)


def test_blob_and_scheme_must_agree_on_span(monkeypatch):
    """``parse_tessera_blob_for_scheme`` compared grid/body/plane/rate/shape
    but not span, and ``validate_tessera_scheme`` cannot (the sidecar carries
    no span field) -- so a span mismatch was refused nowhere in the repo.  The
    wire's span must equal the span ROUTES publishes for the family."""
    import tessera.fused
    import tessera.unit_artifact

    blob = b"\x00" * 32
    member = _ns(name="weight", rows=8, blob=blob)
    monkeypatch.setattr(tessera.fused, "parse_fused", lambda raw: [member])
    scheme = _fp8_scheme()
    validate_tessera_scheme(scheme, "t")  # the control parses clean
    monkeypatch.setattr(tessera.unit_artifact, "parse_unit_artifact",
                        lambda raw, device="cpu": _parsed_with_span(span=7))
    assert ROUTES[TESSERA_FP8]["span"] != 7
    with pytest.raises(ValueError, match="span"):
        parse_tessera_blob_for_scheme(blob, scheme, "t")
