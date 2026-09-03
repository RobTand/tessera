"""Container and accounting discipline (math audit 2026-09-02, §3).

None of this moves a byte.  Every finding here is about a number the package
*reports* or a state it *permits*, and the through-line is the same one: a
figure nothing consumes is not a figure, it is a confession.  A ``wire_bpp``
that means two things depending on which side computed it, a plane storage
mode charged zero bytes, a bit order nothing can pack and nothing would
verify, a rate helper with no callers and the wrong arithmetic -- each is
harmless today and each is quotable, constructible or reachable tomorrow.
"""

import hashlib
import pathlib
from fractions import Fraction

import pytest
import torch

from tessera import calculator, container, wire
from tessera.codebook import _hoist
from tessera.container import HEADER_BYTES, parse, serialize
from tessera.errors import GrammarError, PlaneLayoutError
from tessera.planes import (
    BitOrder,
    CountGranularity,
    IndexDomain,
    PayloadDtype,
    PlaneDescriptor,
    PlaneKind,
    Storage,
)
from tessera.unit_artifact import build_unit_artifact
from tessera.alphabet import E2M1_GRID, build_forest
from tessera.encode import encode_unit
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera.trellis import ConvCode

_CODE = ConvCode(memory=6)


def _toy_artifact():
    """One real serialised unit: small, cheap, and genuinely round-tripped."""
    torch.manual_seed(0)
    weights = torch.randn(8, 256) * 0.02
    q256 = 512
    rates = bresenham_rate_schedule(root_from_q256(q256), 256, cap=E2M1_GRID.rate_cap)
    forests = {rate: build_forest(rate, grid=E2M1_GRID) for rate in sorted(set(rates))}
    unit = encode_unit(weights, forests, rates, _CODE, completion=0)
    return build_unit_artifact(unit, "unit0", forests, q256, _CODE)


# --- C1: side_bytes means one thing ----------------------------------------


def test_side_bytes_means_the_same_thing_on_both_sides(monkeypatch):
    """``serialize`` passed 0, ``parse`` passed header+manifest.

    Both discard the ``FootprintReport``, so no byte moves -- but the same
    ``wire_bpp`` formula reported two numbers for one artifact (23/4 vs 543/64
    on an 8x256 toy) and one of them is quotable and wrong.
    """
    manifest, plane_region, _blob = _toy_artifact()
    seen = []
    real = container.account_terminal

    def recorder(manifest, terminal, *, side_bytes, physical_bytes):
        seen.append(side_bytes)
        return real(
            manifest, terminal, side_bytes=side_bytes, physical_bytes=physical_bytes
        )

    monkeypatch.setattr(container, "account_terminal", recorder)
    blob = serialize(manifest, plane_region)
    parsed = parse(blob)

    manifest_bytes = len(manifest.encode(manifest.schema_minor))
    assert seen == [HEADER_BYTES + manifest_bytes, HEADER_BYTES + manifest_bytes]
    assert parsed.side_bytes == HEADER_BYTES + manifest_bytes
    assert len(blob) == parsed.side_bytes + len(parsed.plane_region)


# --- C3 / C4: the enum members nothing can produce -------------------------


def _descriptor(**overrides):
    fields = dict(
        kind=PlaneKind.BODY,
        index_domain=IndexDomain.POSITION,
        storage=Storage.INLINE,
        element_bits=1,
        bit_order=BitOrder.MSB_FIRST,
        alignment_bytes=1,
        count_granularity=CountGranularity.WHOLE_PLANE,
        counts=(100,),
        restart_offsets=(),
        payload_dtype=PayloadDtype.RAW_BITS,
        content_digest=bytes(32),
    )
    fields.update(overrides)
    return PlaneDescriptor(**fields)


def test_reference_storage_is_refused_not_charged_zero():
    """A REFERENCE plane returned ``byte_length == 0`` and the bundle summed zeros."""
    with pytest.raises(PlaneLayoutError):
        _descriptor(storage=Storage.REFERENCE)


def test_reference_storage_has_no_skip_left_to_reach():
    """#24: the branches that skipped a REFERENCE plane are gone.

    A pin, not a fail-before: the refusal above is 249dc9a's, and this states
    the consequence the deletions rest on -- no module in the package branches
    on that storage any more.  The enum member survives in ``planes.py`` as the
    named future the refusal there reserves.
    """
    package = pathlib.Path(container.__file__).parent
    branching = {
        path.name
        for path in sorted(package.rglob("*.py"))
        if "storage is Storage.REFERENCE" in path.read_text()
    }
    assert branching == set(), f"a REFERENCE branch is back in {branching}"


def test_no_reader_can_meet_an_arity_above_two():
    """#24: what makes ``layout._steps_of``'s 4 and 8 unreachable.

    A pin.  ``arity`` is the grid's tuple order, so the arities a parsed
    manifest can carry are the tuple orders in ``SERIALISABLE_GRIDS`` -- 1 and
    2.  E2M1^3 is refused twice over, by the registry and by the 256-code
    ceiling on the byte-wide ALPHABET/DESCENDANT planes; E2M1^4 (65536 codes)
    constructs, and is refused the same two ways.
    """
    from tessera.alphabet import SERIALISABLE_GRIDS, grid_digest, tuple_grid

    assert {grid.arity for grid in SERIALISABLE_GRIDS.values()} == {1, 2}
    for k in (3, 4):
        grid = tuple_grid(E2M1_GRID, k)
        assert grid.arity == k
        assert grid_digest(grid) not in SERIALISABLE_GRIDS
        assert grid.size > 256
        with pytest.raises(GrammarError, match="SERIALISABLE_GRIDS"):
            build_forest(3, grid=grid).alphabet_plane()


def test_lsb_first_is_refused_not_half_honoured():
    """The packer is MSB-first unconditionally; a verifier alone is half a feature."""
    with pytest.raises(PlaneLayoutError):
        _descriptor(bit_order=BitOrder.LSB_FIRST)


def test_msb_first_inline_is_still_the_wire():
    descriptor = _descriptor()
    assert descriptor.byte_length() == 13  # 100 bits -> 13 bytes


# --- C5: the dead second accountant ----------------------------------------


def test_plane_rate_is_gone():
    """It had zero callers, ignored padding and alignment, and was 8x wrong on a
    1-element plane.  ``PlaneDescriptor.byte_length`` is the exact-byte
    authority; a second, weaker one inside the module whose thesis is
    "derived means derived" is the drift this package keeps paying for.
    """
    assert not hasattr(calculator, "plane_rate")
    assert "plane_rate" not in calculator.__all__


# --- C6: the calculator says what it cannot derive -------------------------


def test_cited_wire_bpp_is_not_derivable_and_says_so():
    figures = {figure.name: figure for figure in calculator.figure_table()}
    derived = figures["body_e4m3_16_q256_512"]
    cited = figures["exact_wire_bpp_q256_512"]

    assert derived.quotable_as_derived() == Fraction(5, 2)
    with pytest.raises(ValueError):
        cited.quotable_as_derived()
    assert not calculator.assert_derivation_matches(derived, cited)

    # The note must name the gap's components, and must not pretend the cited
    # figure is exact: 1/1250 bpp at 4096^2 is 1677.7216 bytes, not an integer.
    gap_bytes = (cited.value - derived.value) * 4096 * 4096 / 8
    assert gap_bytes.denominator != 1
    for token in ("5/2", "header", "forest", "manifest", "not derivable"):
        assert token in cited.note


# --- G2a / G2b: the wire's own slack ---------------------------------------


def test_bit_order_is_spelled_at_every_packbits_call():
    """numpy's default *is* big, which is why the omission was invisible.

    This is the one property that can regress silently: a future edit that
    changes the default, or a reader who assumes the opposite, costs an
    artifact.  The fix is explicitness, so the test is about the spelling.
    """
    source = pathlib.Path(wire.__file__).read_text()
    assert source.count('bitorder="big"') == 4


def test_unpack_uniform_refuses_dirty_trailing_slack():
    """``ac80`` and ``acff`` unpacked identically: the slack was sliced, not checked.

    ``parse(verify=True)`` catches this one byte earlier, so the live path is
    covered -- ``verify=False`` and every direct caller are not.
    """
    clean = wire.pack_uniform(torch.tensor([5, 3, 1]), 3)
    assert clean.hex() == "ac80"
    dirty = bytes([clean[0], clean[1] | 0x7F])
    assert torch.equal(
        wire.unpack_uniform(clean, 3, 3), torch.tensor([5, 3, 1], dtype=torch.int64)
    )
    with pytest.raises(GrammarError):
        wire.unpack_uniform(dirty, 3, 3)


def test_unpack_body_refuses_dirty_trailing_slack():
    rates = (3, 3)
    body = torch.tensor([[5, 1], [3, 2], [1, 7]], dtype=torch.int64)
    clean = wire.pack_body(body, rates)
    assert torch.equal(wire.unpack_body(clean, rates, 3).long(), body)
    dirty = bytearray(clean)
    dirty[-1] |= 0x0F
    with pytest.raises(GrammarError):
        wire.unpack_body(bytes(dirty), rates, 3)


# --- G1: the tie-break is deterministic and now stated ----------------------


def test_hoist_breaks_ties_left():
    points = torch.tensor([[-1.0], [1.0]])
    leaves = torch.tensor([[-1.0], [1.0]])
    assign = torch.tensor([0, 1])
    order = _hoist(points, assign, leaves, depth=1)
    assert torch.equal(order, torch.tensor([0, 1]))
    assert "tie" in _hoist.__doc__.lower()


# --- G2c: the boundary, now that both sides have landed --------------------


def test_calculator_terminal_record_carries_no_digest_it_did_not_compute():
    """This was a pin on the other worktree's bug and it fired on the merge,
    which is what a pin is for.  ``layout.build_terminal(plane_region=None)``
    used to digest ``bytes(total_bytes)`` -- a look-valid sha256 of data never
    hashed -- and ``calculator.terminal_rate`` hits that path on every call.
    It now stores ``ZERO_DIGEST``, a sentinel no payload can produce, so a
    terminal built without a region fails ``container``'s unconditional digest
    comparison loudly instead of matching an all-zero payload.  The calculator
    itself reads only ``exact_bpp``, so neither form escaped this module.
    """
    from tessera.grammar import C_FULL_BITS, bresenham_rate_schedule, root_from_q256
    from tessera.layout import (
        ZERO_DIGEST,
        TerminalSpec,
        build_planes,
        build_terminal,
    )
    from tessera.manifest import Geometry

    geometry = Geometry(
        rows=64,
        columns=256,
        superblock_columns=256,
        group_weights=32,
        half_weights=16,
        quantizable_params=64 * 256,
    )
    rates = bresenham_rate_schedule(root_from_q256(512), 256, C_FULL_BITS)
    spec = TerminalSpec(
        slot_id="calc",
        completion_bits=tuple(0 for _ in rates),
        released_positions=0,
    )
    planes = build_planes(geometry, rates, b"", b"", cap=C_FULL_BITS, arity=1, spec=spec)
    record = build_terminal(
        geometry, rates, spec, planes, 0, 0, cap=C_FULL_BITS, arity=1
    )
    assert record.payload_digest == ZERO_DIGEST
    # The point of the sentinel: it is not the digest of the payload it would
    # otherwise have claimed, so nothing can match it by accident.
    assert record.payload_digest != hashlib.sha256(bytes(record.exact_bytes)).digest()
