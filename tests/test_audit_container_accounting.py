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
from tessera.errors import GrammarError, PlaneLayoutError, TruncationError
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


# --- tessera#144: what the terminal ladder actually is on an encode --------


def test_an_encoded_artifact_declares_exactly_one_terminal():
    """A pin on current behaviour, not a fail-before test.

    Four modules described the multi-terminal truncation ladder as live.  It is
    a capability of the layout and container layers -- ``layout.build_terminal``
    prices a rung, ``container.parse`` resolves a byte length against the
    declared rungs -- and the tests that exercise a legal truncation lay their
    three-rung artifact out directly (``tests/conftest.py::make_artifact``).
    No **encode** produces one: ``build_unit_artifact`` writes
    ``terminals=(terminal,)``, so a written unit has exactly one legal length
    and every truncation of it is refused.

    This pins that, so the day a writer emits a ladder it also has to answer
    the second half: ``parse_unit_artifact`` reads the scale and body planes at
    counts derived from the geometry, not at the terminal's declared counts, so
    a short rung would fail in ``unpack_uniform`` after passing the match in
    ``container.parse``.  Whether truncatable encodes are a planned capability
    is tessera#144 and is not decided here.
    """
    manifest, region, blob = _toy_artifact()
    assert [t.slot_id for t in manifest.terminals] == ["t-nvfp4"]
    assert manifest.terminals[0].exact_bytes == len(region)
    with pytest.raises(TruncationError, match="match no declared terminal"):
        parse(blob[:-1])


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


def test_steps_of_searches_the_registry_arities_not_a_hardcoded_pair(monkeypatch):
    """#34.1: ``layout._steps_of`` must derive its search from SERIALISABLE_GRIDS.

    The arity is not on the wire, so it is recovered by trial.  Hardcoding
    ``(1, 2)`` repeats the drift 654be54 just fixed (docstring and refusal
    naming 4 and 8 three commits after the loop narrowed): the next serialised
    arity would need three coordinated edits.  The rule is the registry --
    every entry is a permanent wire commitment -- so the loop searches exactly
    the arities the registry holds.  Extending the registry must extend the
    search, which a hardcoded pair cannot do.
    """
    from types import SimpleNamespace

    from tessera import alphabet as _alphabet
    from tessera import layout as _layout
    from tessera.planes import CANONICAL_PLANE_ORDER, PlaneKind

    wire = CANONICAL_PLANE_ORDER
    body = wire.index(PlaneKind.BODY)
    elements = [0] * len(wire)
    elements[body] = 10**9  # no arity explains it, so this always refuses

    def _refusing_manifest():
        return SimpleNamespace(
            shard=None,
            geometry=SimpleNamespace(rows=8, columns=256),
            span=1,
            rates=(1,),
            terminals=[SimpleNamespace(plane_elements=tuple(elements))],
        )

    with pytest.raises(GrammarError) as exc:
        _layout._steps_of(_refusing_manifest())
    expected = tuple(sorted({grid.arity for grid in _alphabet.SERIALISABLE_GRIDS.values()}))
    assert str(expected) in str(exc.value), (
        f"the refusal names {exc.value}, not the registry arities {expected}"
    )

    # The discriminating half: a registry holding an arity-3 grid must widen
    # the search.  A hardcoded ``(1, 2)`` keeps refusing with the old set.
    extended = dict(_alphabet.SERIALISABLE_GRIDS)
    extended["test-arity-3"] = SimpleNamespace(arity=3)
    monkeypatch.setattr(_alphabet, "SERIALISABLE_GRIDS", extended)
    with pytest.raises(GrammarError) as exc2:
        _layout._steps_of(_refusing_manifest())
    expected2 = tuple(sorted({grid.arity for grid in extended.values()}))
    assert 3 in expected2
    assert str(expected2) in str(exc2.value), (
        f"extending SERIALISABLE_GRIDS to {expected2} left the refusal at "
        f"{exc2.value}: the search is a hardcoded pair, not the registry"
    )


def test_no_shorter_terminal_survives_the_wire_on_an_encode():
    """A pin on current behaviour, not a fail-before test (tessera#144, part 2).

    The pin above says the encoder declares one terminal.  This one says why a
    second, shorter one cannot be *added* to an encode without a wire change,
    in the order the obstacles are met -- so the day the wire moves, the line
    that fires here names which obstacle moved.  Measured on the one place the
    recipe table still writes a completion axis at all (E2M1, TCQ body, below
    the cap, with ``completion`` asked for explicitly: the exporter's default
    is ``completion=0`` and a window body refuses any other value):

    1. Canonical plane order (schema D5) puts the scale index plane
       ``SCALE_REFINE`` -- the LUT plane's nibble, which nothing decodes without
       -- *after* COMPLETION.  A terminal that shortens COMPLETION by any amount
       therefore has to drop the scales, and the manifest refuses it as not a
       prefix.  ``t-c1`` on two superblocks below shows the cut is at a legal
       granule boundary and still refused.
    2. The COMPLETION granule is a superblock of columns at full depth, not a
       depth level of every column: a shallower reading is the top bits of
       each per-position word (``decode.reconstruct_unit`` shifts them), never
       a byte prefix of the plane.  ``t-c1`` on one superblock shows the
       depth-1 count is not a quota boundary.
    3. Only a terminal that keeps COMPLETION whole and drops SCALE_REFINE
       passes the manifest -- the schema's T-po2/T-C3 classes on an S6B plane --
       and the reader then reads SCALE_REFINE at the geometry's count and fails
       in ``unpack_uniform``.  That is the obstacle the issue named; it is the
       third, and the exporter's default planes (LUT, CHANNEL) never reach it.

    ``experiments/tessera_embedded_ladder.py`` measured what the axis is worth
    (weight-space, E2M1x2 under the TCQ body the recipe table has since left):
    a truncated reading costs 1.03-1.28x over an encode at that depth.  What
    the ladder would add is bytes-on-disk for that reading; the reading itself
    is already available from a full artifact via ``completion=``.
    """
    import dataclasses

    from tessera.alphabet import SERIALISABLE_GRIDS
    from tessera.export import encode_linear_planes, tcq_cap_q256
    from tessera.grammar import completion_capacity
    from tessera.layout import TerminalSpec, build_terminal
    from tessera.manifest import ManifestError, ScalePlaneKind
    from tessera.planes import PlaneKind
    from tessera.unit_artifact import read_unit_artifact

    torch.manual_seed(0)

    # The exporter's default path writes an empty completion axis and an
    # empty release plane on every serialisable grid, so no rate rung exists
    # to declare a shorter terminal for.
    for grid in SERIALISABLE_GRIDS.values():
        exported, _, _ = encode_linear_planes(
            torch.randn(16, 256) * 0.02, grid=grid, q256=tcq_cap_q256(grid) // 2,
            name="u",
        )
        manifest = parse(exported.blob).manifest
        wire = manifest.plane_order
        counts = manifest.terminals[0].plane_elements
        assert counts[wire.index(PlaneKind.COMPLETION)] == 0, grid.name
        assert counts[wire.index(PlaneKind.RELEASE)] == 0, grid.name

    e2m1 = next(g for g in SERIALISABLE_GRIDS.values() if g.name == "E2M1")
    cap = e2m1.rate_cap

    def encoded(columns, **kwargs):
        exported, _, _ = encode_linear_planes(
            torch.randn(16, columns) * 0.02, grid=e2m1, q256=256, name="u",
            completion=None, **kwargs,
        )
        art = parse(exported.blob)
        return art, art.manifest, art.manifest.terminals[0]

    def shorter(art, manifest, spec):
        wire = manifest.plane_order
        alphabet = manifest.terminals[0].plane_elements[wire.index(PlaneKind.ALPHABET)]
        descendant = manifest.terminals[0].plane_elements[wire.index(PlaneKind.DESCENDANT)]
        short = build_terminal(
            manifest.geometry, manifest.rates, spec, manifest.planes, alphabet,
            descendant, plane_region=art.plane_region, cap=cap,
            arity=e2m1.arity, span=manifest.span,
        )
        ladder = dataclasses.replace(manifest, terminals=(manifest.terminals[0], short))
        blob = serialize(ladder, art.plane_region)
        return blob[: len(blob) - (len(art.plane_region) - short.exact_bytes)]

    def depth(c, rates, **flags):
        return TerminalSpec(
            f"t-c{c}", tuple(min(c, completion_capacity(r, cap)) for r in rates), **flags
        )

    # 1. The default LUT plane: every shorter completion drops the scale index.
    art, manifest, full = encoded(512)
    wire = manifest.plane_order
    assert full.plane_elements[wire.index(PlaneKind.COMPLETION)] == 16384
    assert list(manifest.plane(PlaneKind.COMPLETION).counts) == [8192, 8192]
    lut = dict(with_scale_base=False, with_scale_refine=True)
    for c in (0, 1):
        with pytest.raises(ManifestError, match="not a prefix: SCALE_REFINE carries"):
            shorter(art, manifest, depth(c, manifest.rates, **lut))

    # 2. One superblock: the depth-1 count is not a granule boundary at all.
    art, manifest, _ = encoded(256)
    with pytest.raises(ManifestError, match="not a per-superblock quota boundary"):
        shorter(art, manifest, depth(1, manifest.rates, **lut))

    # 3. The S6B plane's T-C3 class passes the manifest and fails in the reader.
    art, manifest, _ = encoded(512, scale_plane=ScalePlaneKind.S6B)
    t_c3 = depth(cap, manifest.rates, with_scale_base=True, with_scale_refine=False)
    cut = shorter(art, manifest, t_c3)
    with pytest.raises(GrammarError, match="need 2048 bits for 512 elements of 4 bits"):
        read_unit_artifact(cut)
