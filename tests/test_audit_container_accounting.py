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
    # The rule, not the roster: every packbits/unpackbits call spells it.
    calls = source.count("np.packbits(") + source.count("np.unpackbits(")
    assert calls >= 4
    assert source.count('bitorder="big"') == calls


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
            plane_order=wire,
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


# --- tessera#144: the truncation ladder on an encode -----------------------
#
# The wire's ladder was the layout's capability and the encoder's refusal:
# ``test_an_encoded_artifact_declares_exactly_one_terminal`` above says the
# exporter writes one terminal, and at master ``da2b371`` its sibling
# ``test_no_shorter_terminal_survives_the_wire_on_an_encode`` pinned why a
# shorter one could not even be *added* to an encode, in the order the
# obstacles were met: (1) the minor 0-6 plane order put the LUT plane's index
# after COMPLETION, (2) the COMPLETION granule was a superblock of columns at
# full depth, (3) the reader sized the refinement plane from the geometry.
# Schema minor 7 removes 1 and 2 on the wire (``planes.PlaneLayout.LADDER``,
# ``wire.pack_levels``); the tests below are that pin inverted, and the third
# obstacle pinned as it still stands.


def _e2m1_completion_encode(columns, **kwargs):
    """The one recipe rung with a completion axis: E2M1, TCQ body, below the
    cap, ``completion`` asked for explicitly.  Returns the exported blob, its
    parse, the manifest and the exporter's single terminal."""
    from tessera.alphabet import SERIALISABLE_GRIDS
    from tessera.export import encode_linear_planes

    e2m1 = next(g for g in SERIALISABLE_GRIDS.values() if g.name == "E2M1")
    exported, _, _ = encode_linear_planes(
        torch.randn(16, columns) * 0.02, grid=e2m1, q256=256, name="u",
        completion=None, **kwargs,
    )
    art = parse(exported.blob)
    return exported.blob, art, art.manifest, art.manifest.terminals[0]


def _rung(slot, c, rates, **flags):
    from tessera.grammar import completion_capacity
    from tessera.layout import TerminalSpec

    return TerminalSpec(
        slot, tuple(min(c, completion_capacity(r, 3)) for r in rates), **flags
    )


def _laddered(art, manifest, specs):
    """The exporter's terminal plus ``specs`` as shorter rungs, serialised
    over the exporter's own plane region.  Returns the blob and the rungs."""
    import dataclasses

    from tessera.layout import build_terminal
    from tessera.planes import PlaneKind

    wire = manifest.plane_order
    full = manifest.terminals[0]
    alphabet = full.plane_elements[wire.index(PlaneKind.ALPHABET)]
    descendant = full.plane_elements[wire.index(PlaneKind.DESCENDANT)]
    rungs = tuple(
        build_terminal(
            manifest.geometry, manifest.rates, spec, manifest.planes, alphabet,
            descendant, plane_region=art.plane_region, cap=3, arity=1,
            span=manifest.span,
        )
        for spec in specs
    )
    ladder = dataclasses.replace(manifest, terminals=(full, *rungs))
    return serialize(ladder, art.plane_region), rungs


def _cut(blob, art, rung):
    """The artifact truncated to ``rung``: a byte prefix, nothing rewritten."""
    return blob[: len(blob) - (len(art.plane_region) - rung.exact_bytes)]


def test_the_exporters_default_path_still_writes_no_completion_and_no_release():
    """Unchanged fact, stated so the ladder tests are read at their scope:
    on every serialisable grid the exporter's default terminal has an empty
    completion axis and an empty release plane.  The wire can carry a ladder
    since minor 7; the recipe table still has no rung to shorten, and nothing
    here claims one would be worth bytes."""
    from tessera.alphabet import SERIALISABLE_GRIDS
    from tessera.export import encode_linear_planes, tcq_cap_q256
    from tessera.planes import PlaneKind

    torch.manual_seed(0)
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
        assert len(manifest.terminals) == 1, grid.name


def test_a_shallower_completion_rung_reads_from_a_byte_prefix_of_an_encode():
    """tessera#144, obstacles 1 and 2, inverted at schema minor 7.

    At master ``da2b371`` the predecessor of this test pinned, on this exact
    encode, ``not a prefix: SCALE_REFINE carries 512 elements after an
    earlier plane was left incomplete`` for every shorter completion rung on
    two superblocks, and ``not a per-superblock quota boundary of [0, 8192]``
    for the depth-1 rung on one.  Minor 7 puts COMPLETION behind the scale
    planes and cuts it by depth level, so every shorter rung is accepted by
    the manifest, is a byte prefix of the full artifact, and reads from bytes
    alone to exactly what the full artifact decodes to at that depth.  The
    exporter still declares one terminal; the ladder is laid on its encode.
    """
    from tessera.decode import reconstruct_unit
    from tessera.planes import CountGranularity, PlaneKind
    from tessera.unit_artifact import parse_unit_artifact, read_unit_artifact

    torch.manual_seed(0)
    lut = dict(with_scale_base=False, with_scale_refine=True)
    for columns in (512, 256):
        blob, art, manifest, full = _e2m1_completion_encode(columns)
        wire = manifest.plane_order
        # Obstacle 1: everything a decode needs now precedes COMPLETION, and
        # only RELEASE follows it.
        after = wire[wire.index(PlaneKind.COMPLETION) + 1 :]
        assert after == (PlaneKind.RELEASE,)
        # Obstacle 2: the granules are depth levels -- capacity 2 at rate 1,
        # 16 steps per column -- and the depth-1 count is the first one.
        descriptor = manifest.plane(PlaneKind.COMPLETION)
        assert descriptor.count_granularity is CountGranularity.PER_LEVEL
        assert list(descriptor.counts) == [16 * columns, 16 * columns]
        assert full.plane_elements[wire.index(PlaneKind.COMPLETION)] == 32 * columns

        whole = parse_unit_artifact(blob)
        rungs = [_rung(f"t-c{c}", c, manifest.rates, **lut) for c in (0, 1)]
        laddered, terminals = _laddered(art, manifest, rungs)
        sizes = [t.exact_bytes for t in terminals]
        assert sizes == sorted(sizes) and sizes[-1] < full.exact_bytes
        for c, rung in zip((0, 1), terminals):
            assert rung.plane_elements[wire.index(PlaneKind.COMPLETION)] == 16 * columns * c
            cut = _cut(laddered, art, rung)
            back = parse_unit_artifact(cut)
            assert back.manifest.terminals == (full, *terminals)
            assert back.unit.completion_limit == c
            expect = reconstruct_unit(whole.unit, whole.forests, whole.code, completion=c)
            assert torch.equal(read_unit_artifact(cut), expect), (columns, c)
        # The axis is not decorative: the depth-0 reading is a different
        # tensor from the full one, and the full artifact still reads.
        assert not torch.equal(
            read_unit_artifact(_cut(laddered, art, terminals[0])),
            read_unit_artifact(laddered),
        )
        assert torch.equal(read_unit_artifact(laddered), read_unit_artifact(blob))


def test_a_level_cut_is_the_first_levels_of_the_plane_not_the_top_bits_of_each_word():
    """The byte-prefix property itself, on the exporter's bytes: the depth-1
    rung's COMPLETION bytes are the leading bytes of the full plane, and they
    are what ``wire.pack_levels`` emits for the first level alone."""
    from tessera.container import plane_ranges
    from tessera.planes import PlaneKind
    from tessera.unit_artifact import parse_unit_artifact
    from tessera.wire import pack_levels

    torch.manual_seed(0)
    blob, art, manifest, full = _e2m1_completion_encode(256)
    whole = parse_unit_artifact(blob)
    laddered, (rung,) = _laddered(
        art, manifest,
        [_rung("t-c1", 1, manifest.rates, with_scale_base=False, with_scale_refine=True)],
    )
    ranges = {d.kind: (offset, content) for d, offset, content, _ in plane_ranges(manifest, full)}
    offset, content = ranges[PlaneKind.COMPLETION]
    full_plane = art.plane_region[offset : offset + content]
    short = {d.kind: (o, c) for d, o, c, _ in plane_ranges(manifest, rung)}
    s_offset, s_content = short[PlaneKind.COMPLETION]
    assert s_offset == offset
    level_one = art.plane_region[s_offset : s_offset + s_content]
    assert full_plane.startswith(level_one) and 0 < len(level_one) < len(full_plane)
    top_bits = whole.unit.completion_bits >> 1
    assert level_one == pack_levels(top_bits, (1,) * 256)


def _record(manifest, art, slot, **counts):
    """A terminal at explicit per-plane counts (``KIND=count``), priced the
    way ``build_terminal`` prices one, for rungs ``TerminalSpec`` cannot
    spell.  Every plane after the first shortened one is emptied, so the
    record is a prefix and only the shortened plane is under test."""
    import dataclasses

    from tessera.container import plane_ranges

    full = manifest.terminals[0]
    wire = manifest.plane_order
    extent = {d.kind: d.element_count for d in manifest.planes}
    elements = list(full.plane_elements)
    for kind, count in counts.items():
        elements[wire.index(PlaneKind[kind])] = count
    short = next((i for i, kind in enumerate(wire) if elements[i] < extent[kind]), None)
    if short is not None:
        elements[short + 1:] = [0] * (len(elements) - short - 1)
    probe = dataclasses.replace(full, slot_id=slot, plane_elements=tuple(elements))
    exact = sum(total for _d, _o, _c, total in plane_ranges(manifest, probe))
    return dataclasses.replace(
        probe, exact_bytes=exact,
        exact_bpp=Fraction(8 * exact, manifest.geometry.quantizable_params),
        payload_digest=hashlib.sha256(art.plane_region[:exact]).digest(),
    )


def _with_records(art, manifest, records):
    import dataclasses

    ladder = dataclasses.replace(manifest, terminals=(manifest.terminals[0], *records))
    return serialize(ladder, art.plane_region)


def test_a_po2_rung_of_an_s6b_unit_reads_at_the_po2_base():
    """tessera#144, obstacle 3, inverted: the reader reads at the terminal's
    count.

    Pinned before this commit, on this exact encode: the T-po2 rung (po2
    base, no refinement, no completion) passed the manifest and the reader
    died in ``unpack_uniform`` -- ``need 2048 bits for 512 elements of 4
    bits, the plane holds 0`` -- because it sized SCALE_REFINE from the
    geometry.  D3 says what the rung means: every half at its group's po2
    base, which is the all-zero refinement word.
    """
    import dataclasses

    from tessera.decode import reconstruct_unit
    from tessera.manifest import ScalePlaneKind
    from tessera.unit_artifact import parse_unit_artifact, read_unit_artifact

    torch.manual_seed(0)
    blob, art, manifest, full = _e2m1_completion_encode(
        512, scale_plane=ScalePlaneKind.S6B
    )
    whole = parse_unit_artifact(blob)
    assert int(whole.unit.scale_refine.sum()) > 0   # the refinement is not decorative
    s6b = dict(with_scale_base=True, with_scale_refine=False)
    laddered, (t_po2,) = _laddered(art, manifest, [_rung("t-po2", 0, manifest.rates, **s6b)])
    cut = _cut(laddered, art, t_po2)
    back = parse_unit_artifact(cut)
    assert back.manifest.terminals == (full, t_po2)
    assert back.unit.completion_limit == 0
    assert back.unit.scale_refine.shape == whole.unit.scale_refine.shape
    assert int(back.unit.scale_refine.sum()) == 0
    assert torch.equal(back.unit.scale_base, whole.unit.scale_base)
    base_only = dataclasses.replace(
        whole.unit, scale_refine=torch.zeros_like(whole.unit.scale_refine)
    )
    expect = reconstruct_unit(base_only, whole.forests, whole.code, completion=0)
    assert torch.equal(read_unit_artifact(cut), expect)
    assert not torch.equal(
        expect, reconstruct_unit(whole.unit, whole.forests, whole.code, completion=0)
    )


def test_a_refinement_prefix_leaves_the_later_halves_at_the_po2_base():
    """D3's prefix semantics on the wire: ``scale_refine_halves=k`` carries
    the first ``k`` halves' words, the rest read as the po2 base.  An odd
    number of 4-bit halves is not a byte and the manifest refuses it."""
    import dataclasses

    from tessera.decode import reconstruct_unit
    from tessera.manifest import ManifestError, ScalePlaneKind
    from tessera.unit_artifact import parse_unit_artifact, read_unit_artifact

    torch.manual_seed(0)
    blob, art, manifest, full = _e2m1_completion_encode(
        512, scale_plane=ScalePlaneKind.S6B
    )
    wire = manifest.plane_order
    whole = parse_unit_artifact(blob)
    halves = whole.unit.scale_refine.numel()
    k = halves // 2
    assert int(whole.unit.scale_refine[k:].sum()) > 0
    rung = _rung("t-half", 0, manifest.rates, with_scale_base=True,
                 with_scale_refine=True, scale_refine_halves=k)
    laddered, (t_half,) = _laddered(art, manifest, [rung])
    assert t_half.plane_elements[wire.index(PlaneKind.SCALE_REFINE)] == k
    back = parse_unit_artifact(_cut(laddered, art, t_half))
    assert torch.equal(back.unit.scale_refine[:k], whole.unit.scale_refine[:k])
    assert int(back.unit.scale_refine[k:].sum()) == 0
    prefix = dataclasses.replace(
        whole.unit,
        scale_refine=torch.cat([
            whole.unit.scale_refine[:k],
            torch.zeros_like(whole.unit.scale_refine[k:]),
        ]),
    )
    expect = reconstruct_unit(prefix, whole.forests, whole.code, completion=0)
    assert torch.equal(read_unit_artifact(_cut(laddered, art, t_half)), expect)
    odd = _rung("t-odd", 0, manifest.rates, with_scale_base=True,
                with_scale_refine=True, scale_refine_halves=k + 1)
    with pytest.raises(ManifestError, match=f"{4 * (k + 1)} bits, which is not a whole number of bytes"):
        _laddered(art, manifest, [odd])


def test_planes_with_no_prefix_meaning_are_refused_by_name():
    """A LUT index plane, the body, the forests and the rank-1 pair have no
    prefix semantics; the manifest admits a byte-aligned cut of each (cutting
    is the layout's business) and the reader refuses it by name, where it
    used to die in ``wire.unpack_*`` naming neither the plane nor the rule."""
    from tessera.manifest import ManifestError
    from tessera.unit_artifact import parse_unit_artifact

    torch.manual_seed(0)
    blob, art, manifest, full = _e2m1_completion_encode(512)
    wire = manifest.plane_order
    # The LUT plane's index nibble: T-po2's spelling on a LUT unit is a unit
    # with no scales at all.
    laddered, (no_index,) = _laddered(
        art, manifest,
        [_rung("t-noidx", 0, manifest.rates, with_scale_base=False, with_scale_refine=False)],
    )
    with pytest.raises(GrammarError, match="SCALE_REFINE is not truncatable"):
        parse_unit_artifact(_cut(laddered, art, no_index))
    # BODY at its first superblock: a granule boundary, whole bytes, a prefix.
    body = manifest.plane(PlaneKind.BODY)
    assert len(body.counts) == 2 and body.counts[0] * body.element_bits % 8 == 0
    half_body = _record(manifest, art, "t-body", BODY=body.counts[0])
    with pytest.raises(GrammarError, match="BODY is not truncatable"):
        parse_unit_artifact(_cut(_with_records(art, manifest, [half_body]), art, half_body))
    # ALPHABET short by one code (8-bit words, so any count is a byte).
    alphabet = full.plane_elements[wire.index(PlaneKind.ALPHABET)]
    short_alphabet = _record(manifest, art, "t-alpha", ALPHABET=alphabet - 1)
    with pytest.raises(GrammarError, match="ALPHABET is not truncatable"):
        parse_unit_artifact(
            _cut(_with_records(art, manifest, [short_alphabet]), art, short_alphabet)
        )
    # The rank-1 pair travels together: DIAG_SU whole and DIAG_SV absent is a
    # prefix in the minor-7 order and means nothing.
    torch.manual_seed(1)
    blob, art, manifest, full = _e2m1_completion_encode(256, with_diagonals=True)
    wire = manifest.plane_order
    assert full.plane_elements[wire.index(PlaneKind.DIAG_SU)] == 256
    su_only = _record(manifest, art, "t-su", DIAG_SV=0)
    assert su_only.plane_elements[wire.index(PlaneKind.DIAG_SU)] == 256
    with pytest.raises(GrammarError, match="DIAG_SU/DIAG_SV travels whole or not at all"):
        parse_unit_artifact(_cut(_with_records(art, manifest, [su_only]), art, su_only))
    # And a cut inside the pair is not a byte-aligned cut the manifest admits
    # either way: fp16 words are whole bytes, so this one reaches the reader.
    part_sv = _record(manifest, art, "t-sv", DIAG_SV=8)
    with pytest.raises(GrammarError, match="DIAG_SU/DIAG_SV travels whole or not at all"):
        parse_unit_artifact(_cut(_with_records(art, manifest, [part_sv]), art, part_sv))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the encoder is a CUDA path")
def test_a_release_rung_is_the_first_codes_in_plane_order():
    """A shorter RELEASE terminal reads its codes against the first positions
    of the placement the writer ranked at the plane's full count -- on two
    superblocks, a 128-of-256 rung is the first superblock's 128 -- not
    against a quota respread at 128, which would put 64 of them on the second
    superblock's positions and the codes on positions the encoder never
    chose."""
    import dataclasses

    from tessera.alphabet import SERIALISABLE_GRIDS
    from tessera.decode import reconstruct_unit
    from tessera.export import (
        DEFAULT_CODE, DEFAULT_GROUP, DEFAULT_HALF, DEFAULT_SCALE_REFIT, _plan_for,
        wire_recipe,
    )
    from tessera.layout import TerminalSpec
    from tessera.unit_artifact import parse_unit_artifact, read_unit_artifact

    grid = next(g for g in SERIALISABLE_GRIDS.values() if g.name == "E2M1")
    q256, columns, released = 768, 512, 256
    recipe = wire_recipe(grid, q256)
    rates, forests = _plan_for(grid, q256, columns, recipe.body, None)
    weight = (torch.randn(16, columns, generator=torch.Generator().manual_seed(10)) * 0.02).cuda()
    unit = encode_unit(
        weight, forests, rates, DEFAULT_CODE, completion=0, released_positions=released,
        group=DEFAULT_GROUP, half=DEFAULT_HALF, scale_refit=DEFAULT_SCALE_REFIT,
        span=recipe.span, scale_plane=recipe.scale_plane, trellis_weighting="scale",
        body=recipe.body, window_bits=recipe.window_bits, window_seed=recipe.window_seed,
        window_sigma=recipe.window_sigma, channel_sigma=recipe.channel_sigma,
    )
    _m, _r, blob = build_unit_artifact(unit, "u", forests, q256 * grid.arity, DEFAULT_CODE)
    art = parse(blob)
    manifest, full = art.manifest, art.manifest.terminals[0]
    wire = manifest.plane_order
    superblock = manifest.geometry.superblock_columns
    assert columns // superblock == 2
    assert full.plane_elements[wire.index(PlaneKind.RELEASE)] == released
    whole = parse_unit_artifact(blob, device="cuda")
    half = released // 2
    spec = TerminalSpec(
        "t-r128", (0,) * columns, released_positions=half,
        with_scale_base=full.plane_elements[wire.index(PlaneKind.SCALE_BASE)] > 0,
        with_scale_refine=full.plane_elements[wire.index(PlaneKind.SCALE_REFINE)] > 0,
        with_diagonals=full.plane_elements[wire.index(PlaneKind.DIAG_SU)] > 0,
    )
    laddered, (rung,) = _laddered(art, manifest, [spec])
    assert rung.plane_elements[wire.index(PlaneKind.RELEASE)] == half
    cut = _cut(laddered, art, rung)
    back = parse_unit_artifact(cut, device="cuda")
    assert torch.equal(back.unit.release_index, whole.unit.release_index[:half])
    assert torch.equal(back.unit.release_code, whole.unit.release_code[:half])
    # The first superblock's share, and only its: the observable difference
    # from a respread at 128, whose quota would be 64 + 64.
    assert bool(((back.unit.release_index % columns) < superblock).all())
    prefix = dataclasses.replace(
        whole.unit,
        release_index=whole.unit.release_index[:half],
        release_code=whole.unit.release_code[:half],
    )
    expect = reconstruct_unit(prefix, whole.forests, whole.code)
    assert torch.equal(read_unit_artifact(cut, device="cuda"), expect)
    assert not torch.equal(expect, reconstruct_unit(whole.unit, whole.forests, whole.code))
