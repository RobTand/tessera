"""Schema minor 7 (tessera#144): the plane layout that makes the ladder legal.

Torch-free -- this is the byte layer.  Three things and their refusals:

* the order: every plane a decode at *any* completion depth needs precedes
  COMPLETION, and only RELEASE follows it;
* the cut: COMPLETION is one granule per depth level (``PER_LEVEL``), a
  terminal cut at a level boundary is legal, inside a level it is not, and a
  granule boundary that is not a byte boundary is not a legal cut either;
* the minor: the header minor is the layout, ``LEGACY`` below 7 and
  ``LADDER`` from it, checked against the descriptors on decode.

The pre-change bytes themselves are held in ``tests/test_ladder_wire.py``.
"""

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from conftest import ALPHABET_BLOB, DESCENDANT_BLOB, make_artifact
from tessera.container import SCHEMA_MINOR, SCHEMA_MINORS_READ, parse, serialize
from tessera.errors import GrammarError, ManifestError, PlaneLayoutError, TesseraError
from tessera.grammar import (
    bresenham_rate_schedule,
    completion_level_counts,
    root_from_q256,
)
from tessera.layout import TerminalSpec, build_plane_region, build_planes, build_terminal
from tessera.manifest import (
    ArrangementMode,
    BranchIdentity,
    ContainerClass,
    Geometry,
    Manifest,
    RotationState,
)
from tessera.planes import (
    CANONICAL_PLANE_ORDER,
    LEGACY_PLANE_ORDER,
    LEGACY_SHARD_PLANE_ORDER,
    SHARD_PLANE_ORDER,
    CountGranularity,
    PlaneKind,
    PlaneLayout,
    layout_of,
    plane_order,
)

LEGACY_SIDECAR = Path(__file__).parent / "data" / "legacy" / "manifest.json"


# --- the order ---------------------------------------------------------------


@pytest.mark.parametrize("has_state", [False, True])
def test_every_decode_input_precedes_completion_and_only_release_follows(has_state):
    wire = plane_order(has_state, PlaneLayout.LADDER)
    completion = wire.index(PlaneKind.COMPLETION)
    needed = {
        PlaneKind.ALPHABET, PlaneKind.DESCENDANT, PlaneKind.BODY,
        PlaneKind.SCALE_BASE, PlaneKind.DIAG_SU, PlaneKind.DIAG_SV,
        PlaneKind.SCALE_REFINE,
    } | ({PlaneKind.INITIAL_STATE} if has_state else set())
    assert {kind for kind in wire[:completion]} == needed
    assert wire[completion + 1 :] == (PlaneKind.RELEASE,)


def test_only_completion_moved_between_the_two_layouts():
    """The minimal move: take COMPLETION out of either order and the two are
    the same sequence, whole unit and shard alike."""
    for legacy, ladder in (
        (LEGACY_PLANE_ORDER, CANONICAL_PLANE_ORDER),
        (LEGACY_SHARD_PLANE_ORDER, SHARD_PLANE_ORDER),
    ):
        strip = lambda order: [k for k in order if k is not PlaneKind.COMPLETION]
        assert strip(legacy) == strip(ladder)
        assert len(legacy) == len(ladder)
        assert legacy.index(PlaneKind.COMPLETION) == legacy.index(PlaneKind.SCALE_BASE) + 1
        assert ladder.index(PlaneKind.COMPLETION) == ladder.index(PlaneKind.RELEASE) - 1


def test_the_legacy_orders_are_the_orders_the_pre_change_writer_used():
    """Derived from the recorded artifacts, not restated: every blob under
    ``tests/data/legacy`` carries the order master ``da2b371`` wrote it in."""
    meta = json.loads(LEGACY_SIDECAR.read_text())
    whole = [c for c in meta["cases"].values() if len(c["plane_elements"]) == len(LEGACY_PLANE_ORDER)]
    shards = [c for c in meta["cases"].values() if len(c["plane_elements"]) == len(LEGACY_SHARD_PLANE_ORDER)]
    assert whole and shards
    for case in whole:
        assert case["plane_order"] == [k.name for k in LEGACY_PLANE_ORDER]
        assert case["header_minor"] < 7
    for case in shards:
        assert case["plane_order"] == [k.name for k in LEGACY_SHARD_PLANE_ORDER]
        assert case["header_minor"] < 7


def test_layout_of_reads_the_layout_off_a_full_descriptor_sequence():
    for has_state in (False, True):
        for layout in PlaneLayout:
            assert layout_of(plane_order(has_state, layout), has_state) is layout
    with pytest.raises(PlaneLayoutError, match="neither"):
        layout_of(CANONICAL_PLANE_ORDER, True)
    with pytest.raises(PlaneLayoutError, match="neither"):
        layout_of(CANONICAL_PLANE_ORDER[:-1], False)


def test_plane_order_has_no_default_layout():
    """A caller that does not know which wire it is on must not be handed
    the current one."""
    with pytest.raises(TypeError):
        plane_order(False)  # type: ignore[call-arg]


# --- the cut -----------------------------------------------------------------


def test_level_counts_prefix_to_the_count_the_depth_recovery_inverts():
    widths = (2, 0, 1, 2, 2, 1)
    steps = 8
    counts = completion_level_counts(widths, steps)
    assert counts == (5 * steps, 3 * steps)
    running = 0
    for level, count in enumerate(counts, start=1):
        running += count
        assert running == sum(min(level, w) for w in widths) * steps
    assert completion_level_counts((0, 0), 8) == ()
    assert completion_level_counts((), 8) == ()


def _unit(q256=256, rows=8, columns=32, superblock_columns=8, layout=PlaneLayout.LADDER,
          alignment_bytes=1, spec=None):
    geometry = Geometry(
        rows=rows, columns=columns, superblock_columns=superblock_columns,
        group_weights=32, half_weights=16, quantizable_params=rows * columns,
    )
    rates = bresenham_rate_schedule(root_from_q256(q256), columns)
    payloads = {PlaneKind.ALPHABET: ALPHABET_BLOB, PlaneKind.DESCENDANT: DESCENDANT_BLOB}
    planes = build_planes(
        geometry, rates, ALPHABET_BLOB, DESCENDANT_BLOB, payloads=payloads,
        alignment_bytes=alignment_bytes, layout=layout, spec=spec,
    )
    region = build_plane_region(planes, payloads)
    return geometry, rates, planes, region


def _manifest(geometry, rates, planes, region, terminals, layout=PlaneLayout.LADDER, q256=256):
    return Manifest(
        encoder_profile_id=hashlib.sha256(b"profile").digest(),
        branch=BranchIdentity(
            unit_id="u", root_q256=q256, rotation=RotationState.NONE,
            container=ContainerClass.GRIDBOOK,
        ),
        geometry=geometry, arrangement=ArrangementMode.BRESENHAM, rates=rates,
        planes=planes, terminals=terminals,
        payload_digest=hashlib.sha256(region).digest(), layout=layout,
    )


def _terminal(geometry, rates, planes, region, spec):
    return build_terminal(
        geometry, rates, spec, planes, len(ALPHABET_BLOB), len(DESCENDANT_BLOB),
        plane_region=region,
    )


def test_completion_is_one_granule_per_level_and_a_level_cut_is_legal():
    """Rate 1 under cap 3: capacity 2, two levels of ``steps x columns`` bits.
    A terminal at each depth is a legal prefix; one inside a level is not."""
    geometry, rates, planes, region = _unit()
    descriptor = next(p for p in planes if p.kind is PlaneKind.COMPLETION)
    assert descriptor.count_granularity is CountGranularity.PER_LEVEL
    assert list(descriptor.counts) == [8 * 32, 8 * 32]
    assert list(descriptor.restart_offsets) == [0, 8 * 32]
    full = dict(with_scale_base=True, with_scale_refine=True, with_diagonals=True)
    rungs = tuple(
        _terminal(geometry, rates, planes, region, TerminalSpec(f"t-c{c}", (c,) * 32, **full))
        for c in (0, 1, 2)
    )
    manifest = _manifest(geometry, rates, planes, region, rungs)
    sizes = [t.exact_bytes for t in rungs]
    assert sizes == sorted(sizes) and len(set(sizes)) == 3
    assert sizes[1] - sizes[0] == sizes[2] - sizes[1] == 8 * 32 // 8
    blob = serialize(manifest, region)
    assert blob[10] == SCHEMA_MINOR == 7
    for rung in rungs:
        cut = blob[: len(blob) - (len(region) - rung.exact_bytes)]
        assert parse(cut).terminal == rung

    wire = manifest.plane_order
    inside = list(rungs[1].plane_elements)
    inside[wire.index(PlaneKind.COMPLETION)] = 8 * 32 // 2
    mid = dataclasses.replace(rungs[1], plane_elements=tuple(inside), exact_bytes=rungs[1].exact_bytes - 16)
    with pytest.raises(ManifestError, match="not a granule boundary"):
        _manifest(geometry, rates, planes, region, (rungs[2], mid))


def test_a_granule_boundary_off_a_byte_is_not_a_legal_cut():
    """Level count ``steps x N`` at one step and 36 columns is 36 bits, and a
    BODY superblock of six rate-1 columns is 6 bits: real granule boundaries,
    and neither ends on a byte.  D4 needs the final byte's slack to be zero,
    and the bits sharing it would be the next granule's real content."""
    spec = TerminalSpec("t", (2,) * 36, with_scale_base=False, with_scale_refine=False)
    geometry, rates, planes, region = _unit(rows=1, columns=36, superblock_columns=6, spec=spec)
    assert list(next(p for p in planes if p.kind is PlaneKind.COMPLETION).counts) == [36, 36]
    accepted = _terminal(geometry, rates, planes, region, spec)
    depth_one = _terminal(
        geometry, rates, planes, region,
        TerminalSpec("t-c1", (1,) * 36, with_scale_base=False, with_scale_refine=False),
    )
    with pytest.raises(ManifestError, match="36 bits, which is not a whole number of bytes"):
        _manifest(geometry, rates, planes, region, (accepted, depth_one))
    wire = plane_order(False, PlaneLayout.LADDER)
    body_cut = [0] * len(wire)
    body_cut[wire.index(PlaneKind.ALPHABET)] = len(ALPHABET_BLOB)
    body_cut[wire.index(PlaneKind.DESCENDANT)] = len(DESCENDANT_BLOB)
    body_cut[wire.index(PlaneKind.BODY)] = 6
    short = dataclasses.replace(
        accepted, slot_id="t-body", plane_elements=tuple(body_cut),
        exact_bytes=len(ALPHABET_BLOB) + len(DESCENDANT_BLOB) + 1,
    )
    with pytest.raises(ManifestError, match="6 bits, which is not a whole number of bytes"):
        _manifest(geometry, rates, planes, region, (accepted, short))
    # Eight steps make the same boundaries byte-aligned, and the cuts legal.
    geometry8, rates8, planes8, region8 = _unit(rows=8, columns=36, superblock_columns=6, spec=spec)
    ok = _terminal(geometry8, rates8, planes8, region8, spec)
    one = _terminal(
        geometry8, rates8, planes8, region8,
        TerminalSpec("t-c1", (1,) * 36, with_scale_base=False, with_scale_refine=False),
    )
    assert _manifest(geometry8, rates8, planes8, region8, (ok, one)).terminals == (ok, one)


def test_a_cut_off_the_planes_alignment_boundary_is_not_a_legal_cut():
    """The byte rule's sibling above ``alignment_bytes=1``.

    A partial cut's byte length rounds up to the descriptor's alignment
    (``PlaneDescriptor.byte_length``), and the reader requires the padding
    bytes to be zero (``container.verify_plane_region``) -- but in the full
    region the bytes at that offset are the plane's *next* granule's real
    content.  A rate-1 unit at 3 rows has 12-byte completion levels; at
    alignment 8 the depth-1 rung's COMPLETION extent rounds to 16 bytes, so
    the four bytes it calls padding are the depth-2 level's nonzero words:
    the writer used to emit that artifact and its own declared rung was then
    refused by the parser.  Refused at manifest construction instead, where
    the bytes are decided.  At alignment 4 the same cut ends on the boundary
    and every declared rung reads back from its own byte prefix, nonzero
    next-level content and all."""
    geometry = Geometry(
        rows=3, columns=32, superblock_columns=8, group_weights=32,
        half_weights=16, quantizable_params=96,
    )
    rates = bresenham_rate_schedule(root_from_q256(256), 32)
    payloads = {
        PlaneKind.ALPHABET: ALPHABET_BLOB, PlaneKind.DESCENDANT: DESCENDANT_BLOB,
        PlaneKind.COMPLETION: bytes([255]) * 24,
    }
    full = TerminalSpec("t-c2", (2,) * 32, with_scale_base=False)
    shallow = TerminalSpec("t-c1", (1,) * 32, with_scale_base=False)

    def ladder(alignment_bytes):
        planes = build_planes(
            geometry, rates, ALPHABET_BLOB, DESCENDANT_BLOB, payloads=payloads,
            alignment_bytes=alignment_bytes, spec=full,
        )
        region = build_plane_region(planes, payloads)
        rungs = tuple(
            build_terminal(geometry, rates, spec, planes, len(ALPHABET_BLOB),
                           len(DESCENDANT_BLOB), plane_region=region)
            for spec in (full, shallow)
        )
        return planes, region, rungs

    planes, region, rungs = ladder(alignment_bytes=8)
    with pytest.raises(ManifestError, match="alignment boundary"):
        _manifest(geometry, rates, planes, region, rungs)
    planes, region, rungs = ladder(alignment_bytes=4)
    blob = serialize(_manifest(geometry, rates, planes, region, rungs), region)
    for rung in rungs:
        cut = blob[: len(blob) - (len(region) - rung.exact_bytes)]
        assert parse(cut).terminal == rung


def test_a_cut_inside_a_whole_plane_must_end_on_a_byte_too():
    """The byte rule is not a granule rule.  An S6b refinement rung
    (``TerminalSpec.scale_refine_halves``, schema D3) cuts a WHOLE_PLANE
    plane of 4-bit words: three halves are 12 bits and the fourth half's
    leading nibble would share the terminal's final byte; two halves are a
    byte and the cut is legal."""
    spec = TerminalSpec(
        "t", tuple(3 - rate for rate in bresenham_rate_schedule(root_from_q256(256), 32)),
        with_scale_base=True, with_scale_refine=True,
    )
    geometry, rates, planes, region = _unit(spec=spec)
    refine = next(p for p in planes if p.kind is PlaneKind.SCALE_REFINE)
    assert refine.count_granularity is CountGranularity.WHOLE_PLANE
    assert refine.element_count == 16 and refine.element_bits == 4
    full = _terminal(geometry, rates, planes, region, spec)
    three = _terminal(
        geometry, rates, planes, region,
        TerminalSpec("t-3h", (0,) * 32, with_scale_base=True,
                     with_scale_refine=True, scale_refine_halves=3),
    )
    with pytest.raises(ManifestError, match="12 bits, which is not a whole number of bytes"):
        _manifest(geometry, rates, planes, region, (full, three))
    two = _terminal(
        geometry, rates, planes, region,
        TerminalSpec("t-2h", (0,) * 32, with_scale_base=True,
                     with_scale_refine=True, scale_refine_halves=2),
    )
    wire = plane_order(False, PlaneLayout.LADDER)
    assert two.plane_elements[wire.index(PlaneKind.SCALE_REFINE)] == 2
    assert _manifest(geometry, rates, planes, region, (full, two)).terminals == (full, two)
    with pytest.raises(GrammarError, match="17 refinement halves of 16"):
        _terminal(
            geometry, rates, planes, region,
            TerminalSpec("t-17h", (0,) * 32, with_scale_base=True,
                         with_scale_refine=True, scale_refine_halves=17),
        )
    with pytest.raises(GrammarError, match="but declares no SCALE_REFINE plane"):
        _terminal(
            geometry, rates, planes, region,
            TerminalSpec("t-none", (0,) * 32, with_scale_base=True,
                         with_scale_refine=False, scale_refine_halves=2),
        )


def test_a_zero_depth_completion_plane_declares_no_levels():
    """Rate 3 under cap 3 has no completion axis: the plane is PER_LEVEL with
    no granules and no elements, which is what a plane with no levels is.
    A superblock-cut plane may not be empty of granules."""
    geometry, rates, planes, region = _unit(q256=768)
    descriptor = next(p for p in planes if p.kind is PlaneKind.COMPLETION)
    assert descriptor.count_granularity is CountGranularity.PER_LEVEL
    assert descriptor.counts == () and descriptor.element_count == 0
    with pytest.raises(PlaneLayoutError, match="declares no granules"):
        dataclasses.replace(descriptor, count_granularity=CountGranularity.PER_SUPERBLOCK)


def test_a_completion_granularity_that_contradicts_the_layout_is_refused():
    manifest, region, _blob = make_artifact()
    descriptor = manifest.plane(PlaneKind.COMPLETION)
    assert descriptor.count_granularity is CountGranularity.PER_LEVEL
    as_superblock = dataclasses.replace(
        descriptor, count_granularity=CountGranularity.PER_SUPERBLOCK,
    )
    planes = tuple(as_superblock if p.kind is PlaneKind.COMPLETION else p for p in manifest.planes)
    with pytest.raises(ManifestError, match="cut by depth level"):
        dataclasses.replace(manifest, planes=planes)
    geometry, rates, legacy_planes, legacy_region = _unit(layout=PlaneLayout.LEGACY)
    legacy = next(p for p in legacy_planes if p.kind is PlaneKind.COMPLETION)
    assert legacy.count_granularity is CountGranularity.PER_SUPERBLOCK
    as_level = dataclasses.replace(legacy, count_granularity=CountGranularity.PER_LEVEL)
    mixed = tuple(as_level if p.kind is PlaneKind.COMPLETION else p for p in legacy_planes)
    spec = TerminalSpec("t", (2,) * 32, with_scale_refine=True, with_diagonals=True)
    terminal = _terminal(geometry, rates, legacy_planes, legacy_region, spec)
    with pytest.raises(ManifestError, match="cut by superblock"):
        _manifest(geometry, rates, mixed, legacy_region, (terminal,), layout=PlaneLayout.LEGACY)


# --- the minor ---------------------------------------------------------------


def test_the_header_minor_is_the_layout():
    manifest, region, blob = make_artifact()
    assert manifest.layout is PlaneLayout.LADDER
    assert manifest.schema_minor == SCHEMA_MINOR == 7
    assert blob[10] == 7 and tuple(SCHEMA_MINORS_READ) == tuple(range(8))
    assert parse(blob).manifest.layout is PlaneLayout.LADDER
    assert parse(blob).manifest.plane_order is CANONICAL_PLANE_ORDER
    with pytest.raises(ManifestError, match="LADDER plane layout; needs minor 7"):
        manifest.encode(6)
    # A minor-7 artifact whose header claims an older minor decodes under the
    # legacy order, and the descriptors give it away.
    stale = bytearray(blob)
    stale[10] = 6
    with pytest.raises(ManifestError, match="LEGACY layout's wire order"):
        parse(bytes(stale))


def test_a_legacy_layout_writes_the_minor_it_always_did_and_reads_back_legacy():
    geometry, rates, planes, region = _unit(layout=PlaneLayout.LEGACY)
    assert [p.kind for p in planes] == list(LEGACY_PLANE_ORDER)
    spec = TerminalSpec("t", (2,) * 32, with_scale_refine=True, with_diagonals=True)
    terminal = _terminal(geometry, rates, planes, region, spec)
    manifest = _manifest(geometry, rates, planes, region, (terminal,), layout=PlaneLayout.LEGACY)
    assert manifest.schema_minor == 0
    blob = serialize(manifest, region)
    assert blob[10] == 0
    back = parse(blob).manifest
    assert back.layout is PlaneLayout.LEGACY and back.plane_order is LEGACY_PLANE_ORDER
    assert back == manifest
    # Claiming the new minor over the old order is refused the same way.
    forged = bytearray(blob)
    forged[10] = 7
    with pytest.raises(TesseraError):
        parse(bytes(forged))
    # The same geometry in the current layout: the descriptors reorder, the
    # region here does not, because every position-domain plane of this toy
    # is zero-filled -- which is also why a default recipe-table artifact
    # (empty COMPLETION) keeps its plane bytes across the minor.
    _g, _r, ladder_planes, ladder_region = _unit()
    assert [p.kind for p in ladder_planes] == list(CANONICAL_PLANE_ORDER)
    assert ladder_region == region


def test_build_terminal_refuses_a_descriptor_sequence_in_neither_order():
    geometry, rates, planes, region = _unit()
    spec = TerminalSpec("t", (2,) * 32, with_scale_refine=True, with_diagonals=True)
    with pytest.raises(PlaneLayoutError, match="neither"):
        build_terminal(
            geometry, rates, spec, planes[1:] + planes[:1], len(ALPHABET_BLOB),
            len(DESCENDANT_BLOB), plane_region=region,
        )
