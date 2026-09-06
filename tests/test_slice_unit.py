"""Schema minor 4: a Tessera artifact is tensor-parallel by construction.

One artifact, written by an exporter that never learns the TP degree, and
every rank cuts its own shard out of those bytes at load.  What makes that
possible is that a column's body is a bit stream whose only carried state is
the trellis register -- so the stream can be entered in the middle, provided
the register at that point travels with it.  ``layout.slice_unit`` does the
cutting; the INITIAL_STATE plane is the one thing it adds.

These tests hold the mechanism to what a serving loader has to be able to
assume:

  * a shard's decode is the parent's decode sliced, **bit for bit**, over both
    bodies (TCQ span 2, WINDOW), all three scale planes (S6B, LUT, CHANNEL),
    both axes, at tp in {2, 4, 8}, on synthetic units and on units cut out of
    the shipped Qwen3-0.6B E4M3 checkpoint;
  * a shard round-trips through the wire and decodes from **bytes alone**;
  * nothing at offset 0 moved in the slicing wire: encoder-free artifacts keep
    their schema-minor-4 baseline, and the identity slice of any unit is that
    unit;
  * the trellis correction is checked against the scalar ``TCQ.decode``
    oracle, not against another vectorised path;
  * the RELEASE plane restricts consistently, and its per-superblock counts
    are on the wire because no spread reproduces them;
  * an illegal cut is refused with a message naming the granularity;
  * a shard of a shard is a shard: re-slicing equals slicing directly.

The follow-on work these tests do *not* cover is named in
``docs/design/tensor-parallel.md``: the serving plugin's per-rank loader, and
a served two-box TP=2 gate.
"""
import hashlib
import pathlib
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import box_artifacts

from tessera.alphabet import SERIALISABLE_GRIDS
from tessera.decode import (
    decode_codes,
    decode_codes_mixed,
    reconstruct_unit,
    release_order,
    replay_body,
    replay_window,
    unit_scale_field,
)
from tessera.encode import (
    _canonical_release_order,
    encode_unit,
    grid_value_table,
)
from tessera.grammar import (
    RELEASE_BITS,
    release_defined_on,
    release_quota,
    superblock_count,
)
from tessera.errors import GrammarError, ManifestError
from tessera.export import _plan_for, tcq_cap_q256, wire_recipe
from tessera.layout import (
    SlicedUnit,
    _scale_columns_per_row,
    _slice_release,
    can_shard,
    shard_granularity,
    slice_unit,
)
from tessera.container import SCHEMA_MINOR
from tessera.manifest import BodyKind, ScalePlaneKind, ShardOrigin
from tessera.planes import (
    CANONICAL_PLANE_ORDER,
    SHARD_PLANE_ORDER,
    CountGranularity,
    PlaneKind,
    PlaneLayout,
    plane_order,
)
from tessera.trellis import TCQ, ConvCode
from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact

CODE = ConvCode(memory=6)
GRIDS = {g.name: g for g in SERIALISABLE_GRIDS.values()}

#: The bytes the encoder writes for three small units, one per shipping wire
#: recipe, in the minor-7 envelope (tessera#144): the COMPLETION descriptor
#: moved behind SCALE_REFINE, the header minor byte is 7, and the
#: behaviour-derived encoder identity moved with them.  Their payload and
#: envelope are pinned together because those are the bytes served. Issue #360
#: updates the CHANNEL refit and derived identity; the two LUT payloads stay
#: identical under their old stamp. The CHANNEL unit changes two scale words
#: and seven body choices. Receipts: channel-refit-cancellation-2026-09-06.md.
CURRENT_ENCODER_UNIT_DIGESTS = {
    "e4m3-window-channel": (
        "482e1ae55bac89f68513bfab0a3ce613d28180bfe9ff9d65ec38956a6be6724c", 21192),
    "e2m1-tcq-lut-release": (
        "a7dccfb6bcbcaf20373b849609089306acf34e22f0911c208964ac84e0aa89a0", 8431),
    "e2m1x2-subcap-window-lut": (
        "bddea472944c748065b7ce17095ef7f59e46fed318814b73b4c4b3b7284bcea1", 8092),
}

#: The same, for the encoder-free artifact ``conftest.make_artifact`` builds --
#: the one that exercises the layout, manifest and container alone, which is
#: precisely the code a schema minor touches.  The LEGACY rows are the
#: constants pinned at minor 6, untouched: the minor 0-6 writer still writes
#: them byte for byte.  The LADDER rows are what minor 7 writes for the same
#: payloads -- the COMPLETION descriptor moved and the header minor byte
#: reads 7 -- derived by building them, never by editing the constant.
LAYOUT_DIGESTS = {
    (PlaneLayout.LEGACY, 512, 8, 32, 8):
        "cb39f35e686e8858485917c81ab4c53c3b1464d83559ac49dcbbda24fdd783a3",
    (PlaneLayout.LEGACY, 640, 16, 64, 16):
        "0e88573aaa13af73d8fc13d8dab0a16b5f52257f70085493ee5f20bd30b73787",
    (PlaneLayout.LADDER, 512, 8, 32, 8):
        "db62a1ef6896890c54bb5d82b3319a377526a1be0ad34fea4d827b315c84cad1",
    (PlaneLayout.LADDER, 640, 16, 64, 16):
        "232154ae4d05dd579f29da520abb8d7f0ac81d5ee44bdfdcfbdd455d0142863b",
}

#: The shipped Qwen3-0.6B E4M3 checkpoint: real units at the shipping wire.
GBFAM = box_artifacts.path("runs", *("gbfam", "qwen3-0.6b-tessera-e4m3-reach-gridbook"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the encoder is a CUDA path"
)


# --------------------------------------------------------------- fixtures


def _encode(label, grid, q256, released=0, rows=32, cols=256, seed=7,
            device=DEVICE):
    """One unit at its grid's shipping recipe, plus the forests it needs."""
    torch.manual_seed(seed)
    weight = (torch.randn(rows, cols) * 0.02).to(device)
    recipe = wire_recipe(grid, q256)
    sigma = (recipe.channel_sigma
             if recipe.scale_plane is ScalePlaneKind.CHANNEL else None)
    rates, forests = _plan_for(grid, q256, cols, recipe.body, sigma)
    unit = encode_unit(
        weight, forests, rates, CODE, completion=0,
        released_positions=released, span=recipe.span,
        scale_plane=recipe.scale_plane, body=recipe.body,
        window_bits=recipe.window_bits, window_seed=recipe.window_seed,
        window_sigma=recipe.window_sigma, channel_sigma=recipe.channel_sigma,
        scale_refit=2,
    )
    _m, _r, blob = build_unit_artifact(
        unit, label, forests, q256 * grid.arity, CODE)
    return unit, forests, grid, blob


#: ``(label, grid name, q256 offset from the cap, released)`` -- the three
#: shipping wires plus an S6b unit, which no recipe selects but which the
#: format still writes and a reader still has to slice.
CASES = [
    ("e4m3-window-channel", "E4M3", 1024, 0, 256),
    # 512 columns so the release case cuts along columns too: a released unit
    # cuts only on superblock boundaries, and a superblock is 256 columns.
    ("e2m1-tcq-lut-release", "E2M1", None, 96, 512),
    ("e2m1x2-subcap-window-lut", "E2M1x2", -128, 0, 256),
    ("e2m1x2-cap-tcq-lut", "E2M1x2", None, 0, 256),
]


def _case(label):
    entry = next(c for c in CASES if c[0] == label)
    _label, name, q256, released, cols = entry
    grid = GRIDS[name]
    if q256 is None:
        q256 = tcq_cap_q256(grid)
    elif q256 < 0:
        q256 = tcq_cap_q256(grid) + q256
    return _encode(label, grid, q256, released, cols=cols)


@pytest.fixture(scope="module")
def units():
    if not torch.cuda.is_available():
        pytest.skip("the encoder is a CUDA path")
    return {label: _case(label) for label, *_rest in CASES}


def _s6b_unit(device=DEVICE, rows=32):
    """A unit over the S6b plane: the older default, still a legal wire."""
    grid = GRIDS["E2M1"]
    torch.manual_seed(11)
    weight = (torch.randn(rows, 256) * 0.02).to(device)
    q256 = tcq_cap_q256(grid)
    rates, forests = _plan_for(grid, q256, 256, BodyKind.TCQ, None)
    unit = encode_unit(
        weight, forests, rates, CODE, completion=0, span=2,
        scale_plane=ScalePlaneKind.S6B, body=BodyKind.TCQ, scale_refit=2,
    )
    _m, _r, blob = build_unit_artifact(unit, "s6b", forests, q256 * grid.arity, CODE)
    return unit, forests, grid, blob


# ------------------------------------------------- nothing at offset 0 moved


@pytest.mark.parametrize("key,digest", sorted(LAYOUT_DIGESTS.items()))
def test_layout_bytes_are_what_each_layout_writes(key, digest):
    """The encoder-free artifact is pinned per plane layout.

    ``make_artifact`` builds a complete unit out of fixed payloads through
    ``build_planes`` / ``build_terminal`` / ``Manifest.encode`` / ``serialize``
    -- every file a schema minor touches and no encoder at all.  Minor 7
    moved these bytes on purpose (the COMPLETION descriptor now follows
    SCALE_REFINE, and the header says 7), so the LADDER rows are new; the
    LEGACY rows are the minor-6 constants, and ``layout=PlaneLayout.LEGACY``
    must still write them exactly -- that is the reader's compatibility
    promise seen from the writer's side.
    """
    from conftest import make_artifact

    layout, q256, rows, columns, superblock = key
    _m, _region, blob = make_artifact(
        q256=q256, rows=rows, columns=columns, superblock_columns=superblock,
        layout=layout,
    )
    assert blob[10] == (SCHEMA_MINOR if layout is PlaneLayout.LADDER else 0)
    assert hashlib.sha256(blob).hexdigest() == digest


@needs_cuda
@pytest.mark.parametrize("label", sorted(CURRENT_ENCODER_UNIT_DIGESTS))
def test_encoded_unit_bytes_match_encoder_identity_baseline(label):
    """A real unit pins payload bytes together with its encoder identity."""
    digest, size = CURRENT_ENCODER_UNIT_DIGESTS[label]
    _unit, _forests, _grid, blob = _case(label)
    assert (len(blob), hashlib.sha256(blob).hexdigest()) == (size, digest)


@needs_cuda
def test_identity_slice_is_the_unit(units):
    """``slice_unit`` over the whole extent reproduces the parent's bytes.

    Not a convenience: it is the statement that a shard record and a state
    plane appear only where a cut needs them, so an artifact that is nobody's
    shard is written exactly as it was before this schema minor existed.
    """
    for label, (_unit, forests, grid, blob) in units.items():
        parsed = parse_unit_artifact(blob, device=DEVICE)
        whole = slice_unit(parsed)
        _m, _r, again = build_unit_artifact(
            whole, parsed.manifest.branch.unit_id, forests,
            parsed.manifest.branch.root_q256, CODE)
        assert again == blob, label
        assert parse_unit_artifact(again).manifest.shard is None


@box_artifacts.require("runs", *("gbfam", "qwen3-0.6b-tessera-e4m3-reach-gridbook"))
def test_shipped_checkpoint_units_reparse_identically():
    """Units written at 8070ec6 still parse and rebuild to the same bytes."""
    from safetensors import safe_open

    from tessera.fused import parse_fused

    key = "model.layers.0.self_attn.qkv_proj.wire_bytes"
    with safe_open(str(GBFAM / "model.safetensors"), framework="pt") as handle:
        container = bytes(handle.get_tensor(key).numpy().tobytes())
    for member in parse_fused(container):
        parsed = parse_unit_artifact(member.blob, device=DEVICE)
        manifest = parsed.manifest
        _m, _r, again = build_unit_artifact(
            slice_unit(parsed), manifest.branch.unit_id, parsed.forests,
            manifest.branch.root_q256, parsed.code or CODE,
            superblock=manifest.geometry.superblock_columns,
            container=manifest.branch.container,
            fixture_id=manifest.encoder_fixture_id,
            layout=manifest.layout)
        assert again == member.blob, member.name


# ------------------------------------------------------- the slice contract


def _tp_ranges(extent, tp):
    step = extent // tp
    return [(rank * step, (rank + 1) * step) for rank in range(tp)]


@needs_cuda
@pytest.mark.parametrize("label", [c[0] for c in CASES])
@pytest.mark.parametrize("axis", ["row", "column"])
@pytest.mark.parametrize("tp", [2, 4, 8])
def test_shard_decode_is_the_parent_decode_sliced(units, label, axis, tp):
    """The property the whole mechanism exists for, over both bodies and all
    three planes: a shard decodes to its window of the parent, bit for bit."""
    unit, forests, grid, blob = units[label]
    parsed = parse_unit_artifact(blob, device=DEVICE)
    geometry = parsed.manifest.geometry
    full = reconstruct_unit(parsed.unit, parsed.forests, parsed.code)
    if not can_shard(parsed.unit, tp, axis,
                     geometry.superblock_columns, grid.arity):
        pytest.skip(f"{label} does not cut {tp} ways along {axis}s")
    extent = geometry.rows if axis == "row" else geometry.columns
    for lo, hi in _tp_ranges(extent, tp):
        kwargs = {"rows": (lo, hi)} if axis == "row" else {"cols": (lo, hi)}
        shard = slice_unit(parsed, **kwargs)
        want = full[lo:hi, :] if axis == "row" else full[:, lo:hi]
        assert torch.equal(
            reconstruct_unit(shard, parsed.forests, parsed.code), want
        ), f"{label} {axis} shard [{lo}, {hi})"


@needs_cuda
@pytest.mark.parametrize("label", ["e2m1-tcq-lut-release", "e2m1x2-cap-tcq-lut"])
@pytest.mark.parametrize("axis", ["row", "column"])
@pytest.mark.parametrize("tp", [2, 4, 8])
def test_decode_codes_of_a_shard_is_the_parent_sliced(units, label, axis, tp):
    """The single-forest TCQ wrapper is an entry point too, and it is public.

    ``reconstruct_unit`` reaches the trellis through ``decode_codes_mixed``,
    which prefers a fused kernel and only falls back to ``replay_body``; the
    uniform-rate ``decode_codes`` calls ``replay_body`` directly and threads
    the start state itself.  A shard whose state reached one path and not the
    other would decode to plausible wrong weights on whichever caller happened
    to hold a single forest, so both paths are held to the same property here.
    Only the two TCQ cases: ``decode_codes`` refuses a window body by design.
    """
    unit, forests, grid, blob = units[label]
    parsed = parse_unit_artifact(blob, device=DEVICE)
    geometry = parsed.manifest.geometry
    rates = sorted(parsed.forests)
    assert len(rates) == 1, f"{label} is not uniform-rate: {rates}"
    forest = parsed.forests[rates[0]]
    if not can_shard(parsed.unit, tp, axis,
                     geometry.superblock_columns, grid.arity):
        pytest.skip(f"{label} does not cut {tp} ways along {axis}s")
    whole = decode_codes(parsed.unit, forest, parsed.code)
    extent = geometry.rows if axis == "row" else geometry.columns
    for lo, hi in _tp_ranges(extent, tp):
        kwargs = {"rows": (lo, hi)} if axis == "row" else {"cols": (lo, hi)}
        shard = slice_unit(parsed, **kwargs)
        want = (whole[lo // grid.arity : hi // grid.arity]
                if axis == "row" else whole[:, lo:hi])
        assert torch.equal(decode_codes(shard, forest, parsed.code), want), (
            f"{label} {axis} shard [{lo}, {hi})"
        )


@needs_cuda
@pytest.mark.parametrize("label", [c[0] for c in CASES])
@pytest.mark.parametrize("axis", ["row", "column"])
def test_shard_round_trips_through_bytes(units, label, axis):
    """A shard is a whole artifact: serialise it, and a reader holding only
    bytes decodes the same window."""
    unit, forests, grid, blob = units[label]
    parsed = parse_unit_artifact(blob, device=DEVICE)
    manifest = parsed.manifest
    geometry = manifest.geometry
    full = reconstruct_unit(parsed.unit, parsed.forests, parsed.code)
    if not can_shard(parsed.unit, 2, axis,
                     geometry.superblock_columns, grid.arity):
        pytest.skip(f"{label} does not cut two ways along {axis}s")
    extent = geometry.rows if axis == "row" else geometry.columns
    for rank, (lo, hi) in enumerate(_tp_ranges(extent, 2)):
        kwargs = {"rows": (lo, hi)} if axis == "row" else {"cols": (lo, hi)}
        shard = slice_unit(parsed, **kwargs)
        _m, _r, shard_blob = build_unit_artifact(
            shard, f"{label}.rank{rank}", forests,
            manifest.branch.root_q256, CODE, fixture_id=None)
        back = parse_unit_artifact(shard_blob, device=DEVICE)
        want = full[lo:hi, :] if axis == "row" else full[:, lo:hi]
        assert torch.equal(
            reconstruct_unit(back.unit, back.forests, back.code), want
        )
        record = back.manifest.shard
        assert record is not None
        assert (record.row_offset, record.col_offset) == (
            (lo, 0) if axis == "row" else (0, lo)
        )
        assert (record.parent_rows, record.parent_columns) == (
            geometry.rows, geometry.columns
        )
        assert record.parent_digest == manifest.manifest_digest()
        # The state plane exists exactly where a cut needs it.
        state = back.manifest.plane(PlaneKind.INITIAL_STATE)
        if record.row_offset:
            assert state is not None
            assert state.element_count == back.manifest.geometry.columns
            assert back.manifest.plane_order is SHARD_PLANE_ORDER
            assert back.manifest.schema_minor == SCHEMA_MINOR
        else:
            assert state is None
            assert back.manifest.plane_order is CANONICAL_PLANE_ORDER


@needs_cuda
@pytest.mark.parametrize("label", [c[0] for c in CASES])
def test_random_legal_slices(units, label):
    """Random cuts at legal granularities, on both axes at once."""
    unit, forests, grid, blob = units[label]
    parsed = parse_unit_artifact(blob, device=DEVICE)
    geometry = parsed.manifest.geometry
    rows, cols = geometry.rows, geometry.columns
    row_gran, col_gran = shard_granularity(
        parsed.unit, geometry.superblock_columns, grid.arity
    )
    full = reconstruct_unit(parsed.unit, parsed.forests, parsed.code)
    rng = random.Random(19)
    for _ in range(8):
        r0 = rng.randrange(0, rows // row_gran) * row_gran
        r1 = rng.randrange(r0 // row_gran + 1, rows // row_gran + 1) * row_gran
        c0 = rng.randrange(0, cols // col_gran) * col_gran
        c1 = rng.randrange(c0 // col_gran + 1, cols // col_gran + 1) * col_gran
        shard = slice_unit(parsed, rows=(r0, r1), cols=(c0, c1))
        got = reconstruct_unit(shard, parsed.forests, parsed.code)
        assert torch.equal(got, full[r0:r1, c0:c1]), (label, r0, r1, c0, c1)


@needs_cuda
def test_s6b_plane_slices():
    """The S6b plane -- a base byte per 32 and a nibble per 16 -- slices too."""
    unit, forests, grid, blob = _s6b_unit()
    parsed = parse_unit_artifact(blob, device=DEVICE)
    full = reconstruct_unit(parsed.unit, parsed.forests, parsed.code)
    assert parsed.manifest.scale_plane.kind is ScalePlaneKind.S6B
    assert shard_granularity(parsed.unit, 256, grid.arity) == (2, 32)
    for lo, hi in _tp_ranges(parsed.manifest.geometry.rows, 4):
        shard = slice_unit(parsed, rows=(lo, hi))
        assert torch.equal(
            reconstruct_unit(shard, parsed.forests, parsed.code), full[lo:hi]
        )
    for lo, hi in _tp_ranges(parsed.manifest.geometry.columns, 4):
        shard = slice_unit(parsed, cols=(lo, hi))
        assert torch.equal(
            reconstruct_unit(shard, parsed.forests, parsed.code), full[:, lo:hi]
        )


@needs_cuda
@pytest.mark.parametrize("label", [c[0] for c in CASES])
def test_reslice_equals_direct_slice(units, label):
    """A shard of a shard is a shard, and it is the shard you would have cut.

    This is what makes the start state a *composition*: the sub-shard inherits
    the register its parent carries, so the two paths land on the same bytes.
    """
    unit, forests, grid, blob = units[label]
    parsed = parse_unit_artifact(blob, device=DEVICE)
    geometry = parsed.manifest.geometry
    rows = geometry.rows
    row_gran, _col = shard_granularity(
        parsed.unit, geometry.superblock_columns, grid.arity
    )
    quarter = rows // 4
    if quarter % row_gran:
        pytest.skip("no legal quarter cut")
    half = slice_unit(parsed, rows=(rows // 2, rows))
    composed = slice_unit(
        half, rows=(quarter, 2 * quarter), arity=grid.arity, code=parsed.code,
        grid=grid, superblock=geometry.superblock_columns,
    )
    direct = slice_unit(parsed, rows=(rows // 2 + quarter, rows))
    assert composed.row_offset == direct.row_offset == rows // 2 + quarter
    assert torch.equal(
        reconstruct_unit(composed, parsed.forests, parsed.code),
        reconstruct_unit(direct, parsed.forests, parsed.code),
    )


# ------------------------------------------ the shard record's frame (#140)
#
# A shard record names the ORIGINAL -- the whole unit the exporter wrote --
# whatever depth of re-slicing produced the shard.  The offsets already
# composed into that frame (``test_reslice_equals_direct_slice`` asserts it)
# and ``parent_digest`` already inherited it; ``parent_rows``/``parent_columns``
# were read off the immediate parent instead, so a re-slice wrote a record
# whose four fields described two units.  These run on CPU: the encoder is not
# a CUDA path at this size, and the slicing surface above is gated on one.


@pytest.fixture(scope="module")
def cpu_units():
    """CPU-built parents, 64 rows so a quarter cut lands on a super-symbol at
    both depths: the shipping E4M3 window/CHANNEL wire (the one the serving
    loader shards) and the S6b TCQ unit tessera#140 was reproduced on."""
    e4m3 = _encode("e4m3-window-channel", GRIDS["E4M3"], 1024, 0,
                   rows=64, cols=256, device="cpu")
    s6b = _s6b_unit(device="cpu", rows=64)
    return {"e4m3-window-channel": e4m3, "s6b-tcq": s6b}


def _cpu_parse(blob):
    return parse_unit_artifact(blob, device="cpu")


def _build_shard(shard, forests, parsed, label):
    return build_unit_artifact(
        shard, label, forests, parsed.manifest.branch.root_q256, CODE,
        fixture_id=None,
    )


def _origin_record(parsed, row_offset, col_offset, state_bits):
    geometry = parsed.manifest.geometry
    return ShardOrigin(
        row_offset=row_offset, col_offset=col_offset,
        parent_rows=geometry.rows, parent_columns=geometry.columns,
        parent_digest=parsed.manifest.manifest_digest(), state_bits=state_bits,
    )


@pytest.mark.parametrize("label", ["e4m3-window-channel", "s6b-tcq"])
@pytest.mark.parametrize("axis", ["row", "column"])
def test_reslice_record_names_the_original(cpu_units, label, axis):
    """A shard of a shard serialises the record the direct cut writes: the
    composed offset, the ORIGINAL's extent and the ORIGINAL's digest -- and
    the same bytes, because nothing else about the cut differs."""
    _unit, forests, grid, blob = cpu_units[label]
    parsed = _cpu_parse(blob)
    geometry = parsed.manifest.geometry
    extent = geometry.rows if axis == "row" else geometry.columns
    half, quarter = extent // 2, extent // 4
    cut = (lambda lo, hi: {"rows": (lo, hi)}) if axis == "row" else (
        lambda lo, hi: {"cols": (lo, hi)})
    upper = slice_unit(parsed, **cut(half, extent))
    composed = slice_unit(
        upper, **cut(quarter, 2 * quarter), arity=grid.arity, code=parsed.code,
        grid=grid, superblock=geometry.superblock_columns,
    )
    direct = slice_unit(parsed, **cut(half + quarter, extent))
    m_composed, _r, composed_blob = _build_shard(composed, forests, parsed, "q")
    m_direct, _r, direct_blob = _build_shard(direct, forests, parsed, "q")
    want = _origin_record(
        parsed,
        row_offset=half + quarter if axis == "row" else 0,
        col_offset=half + quarter if axis == "column" else 0,
        state_bits=m_direct.shard.state_bits,
    )
    assert m_composed.shard == want
    assert m_direct.shard == want
    assert composed_blob == direct_blob
    full = reconstruct_unit(parsed.unit, parsed.forests, parsed.code)
    back = _cpu_parse(composed_blob)
    window = (full[half + quarter:] if axis == "row"
              else full[:, half + quarter:])
    assert torch.equal(reconstruct_unit(back.unit, back.forests, back.code), window)


@pytest.fixture(scope="module")
def cpu_released_unit():
    """A CPU parent that actually carries a RELEASE plane.

    ``cpu_units`` builds both of its parents at ``released=0``, so every
    record-frame test above runs over an eight-plane unit and the ninth plane
    -- whose restriction is the only thing a cut does to a plane that is not a
    slice -- is serialised by no re-slice test at all.  The released unit
    otherwise lives only in ``units``, which is CUDA-gated, and
    ``test_reslice_equals_direct_slice`` compares reconstructions rather than
    bytes.  That is the gap tessera#183 (M11) names.

    Three things about the shape, each so the test above cannot pass
    vacuously.  32 rows rather than ``cpu_units``' 64: the cut granularity is
    ``arity * span`` = 2, so two depths of halving land legally, and the CPU
    encode is the expensive part of this file.  512 columns: a RELEASE plane's
    counts are **per superblock**, and over one superblock that vector is a
    scalar which binning and a recomputed quota agree on whatever either
    means.  And a row ramp, opposed between the two superblocks, so the two
    disagree in fact and not just in principle -- on a plain Gaussian the
    restricted set splits evenly and a quota reproduces it by luck.

    Rows are the only axis a re-slice can use here: a RELEASE plane raises the
    column granularity to the superblock (``shard_granularity``), so the legal
    column cuts of 512 columns are the two halves and neither halves again.
    """
    grid = GRIDS["E2M1"]
    rows, cols, superblock = 32, 512, 256
    q256 = tcq_cap_q256(grid)
    recipe = wire_recipe(grid, q256)
    torch.manual_seed(7)
    weight = torch.randn(rows, cols) * 0.02
    ramp = torch.linspace(0.25, 4.0, rows).unsqueeze(1)
    weight[:, :superblock] *= ramp
    weight[:, superblock:] *= ramp.flip(0)
    rates, forests = _plan_for(grid, q256, cols, recipe.body, None)
    unit = encode_unit(
        weight, forests, rates, CODE, completion=0, released_positions=48,
        span=recipe.span, scale_plane=recipe.scale_plane, body=recipe.body,
        scale_refit=2,
    )
    _m, _r, blob = build_unit_artifact(
        unit, "e2m1-tcq-lut-release", forests, q256 * grid.arity, CODE)
    return unit, forests, grid, blob


def test_a_reslice_of_a_released_shard_serialises_its_record(cpu_released_unit):
    """A re-sliced released shard writes the direct cut's bytes, RELEASE plane
    and all.

    Coverage, not a bug: the reviewer looked for a defect behind this gap and
    found none.  ``slicing._slice_release`` bins the already-restricted index
    per superblock rather than recomputing a quota from the shard's extent, so
    restricting twice is restricting once and composition is correct by
    construction.  What was missing was any test that *serialised* one, which
    is what would catch a future change to that reasoning.
    """
    _unit, forests, grid, blob = cpu_released_unit
    parsed = _cpu_parse(blob)
    geometry = parsed.manifest.geometry
    rows, cols = geometry.rows, geometry.columns
    assert parsed.unit.release_index.numel(), "this case exists to carry releases"
    row_gran, col_gran = shard_granularity(
        parsed.unit, geometry.superblock_columns, grid.arity
    )
    # The premises of the row-only cut over two superblocks, derived rather
    # than asserted.
    assert col_gran == geometry.superblock_columns
    assert superblock_count(cols, col_gran) > 1
    quarter = rows // 4
    assert quarter % row_gran == 0

    half = slice_unit(parsed, rows=(rows // 2, rows))
    composed = slice_unit(
        half, rows=(quarter, 2 * quarter), arity=grid.arity, code=parsed.code,
        grid=grid, superblock=geometry.superblock_columns,
    )
    direct = slice_unit(parsed, rows=(rows // 2 + quarter, rows))
    # The reviewer's reason the composition is correct, made observable: the
    # shard's per-superblock counts are the parent's set *binned*, not a quota
    # recomputed from the shard's own total.  This weight makes the two
    # disagree, so what follows is not a restatement of ``release_quota``.
    assert tuple(direct.release_counts) != tuple(
        release_quota(int(direct.release_index.numel()), cols, col_gran)
    )
    m_composed, _r, composed_blob = _build_shard(composed, forests, parsed, "q")
    m_direct, _r, direct_blob = _build_shard(direct, forests, parsed, "q")
    want = _origin_record(
        parsed, rows // 2 + quarter, 0, m_direct.shard.state_bits
    )
    assert m_composed.shard == want
    assert m_direct.shard == want
    assert composed_blob == direct_blob

    # The plane is on the wire and it is the parent's set restricted -- read
    # back from the shard's own bytes, so a shard that dropped or renumbered
    # its releases could not pass by agreeing with the other shard object.
    low = rows // 2 + quarter
    expected = {
        (flat // cols - low) * cols + flat % cols
        for flat in parsed.unit.release_index.tolist()
        if flat // cols >= low
    }
    assert expected, "the cut has to keep some of the parent's releases"
    back = _cpu_parse(composed_blob)
    assert set(back.unit.release_index.tolist()) == expected
    full = reconstruct_unit(parsed.unit, parsed.forests, parsed.code)
    assert torch.equal(
        reconstruct_unit(back.unit, back.forests, back.code), full[low:]
    )


# ------------------------- a parsed shard rewrites to itself (tessera#336)
#
# The test above cuts twice **in memory** and serialises once, so the shard
# object it writes is always one ``slice_unit`` built, counts and all.  What no
# test held was the other direction: bytes -> ``SlicedUnit`` -> bytes.  A rank
# that reads a shard and writes it back out -- a re-shard, a merge, a cache
# round trip -- went through ``unit_artifact._as_unit``, which restored the
# origin and the start state and left ``release_counts`` empty, and
# ``build_unit_artifact`` then wrote the shard's total as a whole unit's, at
# WHOLE_PLANE granularity, for the reader to respread onto a different set of
# positions.  Both artifacts parsed; the weights differed.


def _shard_blob(shard, forests, root_q256, label="released-shard"):
    """One shard artifact, written the way a rank writes one.

    ``fixture_id=None`` throughout, so the first write and the rewrite are
    compared on the bytes the cut decides and not on an encoder identity
    neither of them moves.
    """
    return build_unit_artifact(
        shard, label, forests, root_q256, CODE, fixture_id=None,
    )


def _rewrite(parsed, root_q256, label="released-shard"):
    """The parsed unit written straight back out, with nothing else changed."""
    return _shard_blob(parsed.unit, parsed.forests, root_q256, label)


def _release_counts_of(manifest):
    return tuple(manifest.plane(PlaneKind.RELEASE).counts)


@pytest.fixture(scope="module")
def s6b_released_parent():
    """tessera#336's own construction: the committed S6b artifact, given 96
    ordinary releases and written back out as a whole unit.

    No encoder runs.  The releases are placed by the canonical rule at the
    parent's own decode, which is what an encode would have placed them at, so
    the parent is an artifact the reader accepts on its own terms -- and the
    cuts below then carry a *restriction* of a quota, which is the vector no
    reader regenerates.
    """
    legacy = pathlib.Path(__file__).parent / "data" / "legacy"
    parsed = _cpu_parse((legacy / "e2m1-256-cfull-s6b-512c.tessera").read_bytes())
    geometry = parsed.manifest.geometry
    cols, superblock = geometry.columns, geometry.superblock_columns
    unit = parsed.unit
    decoded = reconstruct_unit(unit, parsed.forests, parsed.code)
    unit.release_index = _canonical_release_order(decoded, cols, superblock, 96)
    unit.release_code = torch.arange(96) % 16
    _m, _r, blob = build_unit_artifact(
        unit, "released-parent", parsed.forests, parsed.manifest.branch.root_q256,
        parsed.code, fixture_id=None,
    )
    return _cpu_parse(blob)


@pytest.mark.parametrize("rows", [(0, 8), (8, 16), (4, 12)])
def test_the_s6b_reproduction_rewrites_a_released_shard_to_the_same_bytes(
    s6b_released_parent, rows
):
    """tessera#336, fail-before: parse a released shard and write it again.

    Measured on the unfixed reader at the middle cut: the shard's descriptor
    said ``(19, 17)``, ``parsed.unit.release_counts`` came back ``()``, the
    rewrite declared ``(36,)`` at WHOLE_PLANE granularity, the reader
    substituted the quota ``(18, 18)``, and 19 of 4096 weights moved by up to
    0.10546875 -- with both artifacts parsing clean, which is the whole reason
    this is a P0 and not a crash.  The other two cuts moved 32 and 27 weights.

    A rewrite must be the identity here: nothing about the shard changed
    between the two writes, so anything but the same bytes is the writer
    deciding something the cut already decided.
    """
    parsed = s6b_released_parent
    low, high = rows
    shard = slice_unit(parsed, rows=rows)
    manifest, _r, blob = _shard_blob(
        shard, parsed.forests, parsed.manifest.branch.root_q256
    )
    # The premises, derived: two superblocks, and counts a quota does not give.
    assert len(shard.release_counts) == 2
    assert tuple(shard.release_counts) != release_quota(
        int(shard.release_index.numel()),
        parsed.manifest.geometry.columns,
        parsed.manifest.geometry.superblock_columns,
    )
    assert _release_counts_of(manifest) == tuple(shard.release_counts)

    first = _cpu_parse(blob)
    # The vector no reader regenerates, back on the unit the reader returns.
    assert tuple(first.unit.release_counts) == tuple(shard.release_counts)
    rewritten_manifest, _r, rewritten = _rewrite(
        first, parsed.manifest.branch.root_q256
    )
    assert _release_counts_of(rewritten_manifest) == tuple(shard.release_counts)
    assert rewritten == blob
    second = _cpu_parse(rewritten)
    assert torch.equal(second.unit.release_index, first.unit.release_index)
    assert torch.equal(second.unit.release_code, first.unit.release_code)
    parent = reconstruct_unit(parsed.unit, parsed.forests, parsed.code)
    window = parent[low:high]
    assert torch.equal(
        reconstruct_unit(first.unit, first.forests, first.code), window)
    assert torch.equal(
        reconstruct_unit(second.unit, second.forests, second.code), window)


@pytest.fixture(scope="module")
def window_released_unit():
    """A WINDOW-body parent that carries releases, on the one grid a release
    is defined over.

    ``grammar.release_defined_on`` admits exactly the grids whose code space
    fits ``RELEASE_BITS``, which today is E2M1 alone, and no shipping recipe
    pairs E2M1 with a window body -- so this is built off the recipe table
    rather than through ``wire_recipe``.  The format writes it (the window
    reader has its own release branch, ``unit_artifact._read_window_unit``) and
    a reader still has to slice it, which is the same reason this file carries
    an S6b case no recipe selects.  A window body has no trellis, so the encode
    is a fraction of a second.

    Row-opposed ramps across the two superblocks, so a row cut keeps most of
    one superblock's releases and none of the other's: that is the zero count
    the acceptance list asks for, and no quota produces it.
    """
    grid = GRIDS["E2M1"]
    rows, cols, superblock = 16, 512, 256
    q256 = tcq_cap_q256(grid)
    torch.manual_seed(7)
    weight = torch.randn(rows, cols) * 0.02
    ramp = torch.linspace(0.25, 4.0, rows).unsqueeze(1)
    weight[:, :superblock] *= ramp
    weight[:, superblock:] *= ramp.flip(0)
    rates, forests = _plan_for(grid, q256, cols, BodyKind.WINDOW, None)
    unit = encode_unit(
        weight, forests, rates, CODE, completion=0, released_positions=96,
        span=1, scale_plane=ScalePlaneKind.LUT, body=BodyKind.WINDOW,
        window_bits=12, window_seed=1234, scale_refit=2,
    )
    _m, _r, blob = build_unit_artifact(
        unit, "e2m1-window-lut-release", forests, q256 * grid.arity, CODE,
        fixture_id=None,
    )
    return _cpu_parse(blob)


#: ``(body, rows)``.  The two parents have different heights, so the cuts are
#: stated per body rather than scaled: each is a row window that empties one
#: superblock's releases and keeps the other's, one of them at row zero.
_REWRITE_CUTS = [
    ("tcq", (0, 16)), ("tcq", (16, 32)), ("tcq", (12, 20)),
    ("window", (0, 8)), ("window", (12, 16)), ("window", (4, 8)),
]


@pytest.mark.parametrize("body,rows", _REWRITE_CUTS)
def test_a_parsed_released_shard_rewrites_to_itself(
    cpu_released_unit, window_released_unit, body, rows
):
    """tessera#336 over both bodies, at row-zero and nonzero-row cuts, with a
    superblock that releases nothing.

    The cuts are chosen for what they make the count vector do, and the test
    asserts that rather than trusting it: ``(0, n)`` at the row-zero cut, which
    carries no INITIAL_STATE plane, and ``(n, 0)`` at the nonzero-row cuts,
    which do.  A zero count is the case a respread cannot even approximate --
    the quota puts half the releases in a superblock the cut released nothing
    in -- and it is also the case that must NOT be confused with "this shard
    has no counts": that spelling belongs to a shard of a parent that released
    nothing, and it keeps its whole-unit descriptor here.
    """
    if body == "tcq":
        _u, forests, _grid, blob = cpu_released_unit
        parsed = _cpu_parse(blob)
    else:
        parsed = window_released_unit
        forests = parsed.forests
    low, high = rows
    root_q256 = parsed.manifest.branch.root_q256
    geometry = parsed.manifest.geometry
    cols, superblock = geometry.columns, geometry.superblock_columns
    assert superblock_count(cols, superblock) == 2

    shard = slice_unit(parsed, rows=(low, high))
    counts = tuple(shard.release_counts)
    assert len(counts) == 2 and sum(counts) == int(shard.release_index.numel())
    assert 0 in counts and sum(counts), (
        f"cut [{low}, {high}) of the {body} parent was meant to empty one "
        f"superblock and fill the other; it binned {counts}"
    )
    assert (shard.initial_state is not None) == (low > 0)

    manifest, _r, first_blob = _shard_blob(shard, forests, root_q256)
    assert _release_counts_of(manifest) == counts
    first = _cpu_parse(first_blob)
    assert tuple(first.unit.release_counts) == counts

    rewritten_manifest, _r, rewritten = _rewrite(first, root_q256)
    assert _release_counts_of(rewritten_manifest) == counts
    assert rewritten == first_blob
    second = _cpu_parse(rewritten)
    assert torch.equal(second.unit.release_index, first.unit.release_index)
    assert torch.equal(second.unit.release_code, first.unit.release_code)
    window = reconstruct_unit(parsed.unit, parsed.forests, parsed.code)[low:high]
    assert torch.equal(
        reconstruct_unit(first.unit, first.forests, first.code), window)
    assert torch.equal(
        reconstruct_unit(second.unit, second.forests, second.code), window)


def test_a_shard_of_an_unreleased_parent_keeps_the_whole_unit_spelling(cpu_units):
    """Byte compatibility, both directions of the round trip.

    A parent that released nothing gives a shard with no count vector at all,
    and that shard's RELEASE descriptor is the whole-unit one -- an empty plane
    at WHOLE_PLANE granularity, exactly as it was before tessera#336.  A
    reader must give that shard back an empty ``release_counts`` and not a
    vector of zeros, or the rewrite would move bytes on every unreleased shard
    in existence.
    """
    _unit, forests, _grid, blob = cpu_units["s6b-tcq"]
    parsed = _cpu_parse(blob)
    assert parsed.unit.released_positions == 0
    shard = slice_unit(parsed, rows=(16, 48))
    assert tuple(shard.release_counts) == ()
    manifest, _r, first_blob = _shard_blob(shard, forests,
                                           parsed.manifest.branch.root_q256)
    descriptor = manifest.plane(PlaneKind.RELEASE)
    assert descriptor.count_granularity is CountGranularity.WHOLE_PLANE
    assert descriptor.element_count == 0
    first = _cpu_parse(first_blob)
    assert tuple(first.unit.release_counts) == ()
    _m, _r, rewritten = _rewrite(first, parsed.manifest.branch.root_q256)
    assert rewritten == first_blob


def _laddered(art, manifest, spec, cap, arity):
    """``art`` re-serialised with one shorter rung appended, and that rung."""
    from dataclasses import replace as _replace

    from tessera.container import serialize
    from tessera.layout import build_terminal

    wire = manifest.plane_order
    full = manifest.terminals[0]
    rung = build_terminal(
        manifest.geometry, manifest.rates, spec, manifest.planes,
        full.plane_elements[wire.index(PlaneKind.ALPHABET)],
        full.plane_elements[wire.index(PlaneKind.DESCENDANT)],
        plane_region=art.plane_region, cap=cap, arity=arity, span=manifest.span,
    )
    blob = serialize(_replace(manifest, terminals=(full, rung)), art.plane_region)
    return blob[: len(blob) - (len(art.plane_region) - rung.exact_bytes)], rung


def _release_spec(manifest, released, unit, grid):
    """The full terminal's shape with only the RELEASE count moved.

    The completion widths are derived from the unit's own limit, never zeroed:
    §9 ranks the **pre-release decode at the written depth**, so a rung that
    also shortened COMPLETION would place its releases somewhere else and the
    prefix this test asserts would not be one.
    """
    from tessera.grammar import completion_widths
    from tessera.layout import TerminalSpec

    wire = manifest.plane_order
    full = manifest.terminals[0]

    def count(kind):
        return full.plane_elements[wire.index(kind)]

    return TerminalSpec(
        "t-release-rung",
        completion_widths(manifest.rates, grid.rate_cap, unit.completion_limit),
        released_positions=released,
        with_scale_base=count(PlaneKind.SCALE_BASE) > 0,
        with_scale_refine=count(PlaneKind.SCALE_REFINE) > 0,
        with_diagonals=count(PlaneKind.DIAG_SU) > 0,
        with_row_scale=manifest.scale_plane.kind is ScalePlaneKind.CHANNEL,
        state_bits=0 if manifest.shard is None else manifest.shard.state_bits,
    )


@pytest.mark.parametrize("parent_fixture", [
    "s6b_released_parent", "window_released_unit",
])
@pytest.mark.parametrize("selected", [24, 48, 72])
def test_whole_release_prefix_refuses_a_rewrite_that_would_move_positions(
    request, parent_fixture, selected,
):
    """#349: a whole-unit terminal keeps the full descriptor's ranked prefix.

    The three cuts end inside the first block, at its boundary, and inside
    the second. The first decode must retain those positions; the writer must
    refuse a placement its total-only descriptor would redistribute.
    """
    from tessera.container import parse as parse_container

    parent = request.getfixturevalue(parent_fixture)
    root = parent.manifest.branch.root_q256
    manifest, _r, full_blob = _rewrite(parent, root)
    geometry = manifest.geometry
    cols, superblock = geometry.columns, geometry.superblock_columns
    assert manifest.shard is None
    assert release_quota(parent.unit.released_positions, cols, superblock) == (48, 48)
    spec = _release_spec(manifest, selected, parent.unit, parent.grid)
    cap = (parent.grid.payload_bits if manifest.body is BodyKind.WINDOW
           else parent.grid.rate_cap)
    short_blob, _rung = _laddered(
        parse_container(full_blob), manifest, spec, cap, parent.grid.arity)
    first = _cpu_parse(short_blob)
    assert first.unit.released_positions == selected
    assert torch.equal(first.unit.release_index, parent.unit.release_index[:selected])
    assert torch.equal(first.unit.release_code, parent.unit.release_code[:selected])
    expected = replace(
        parent.unit, release_index=parent.unit.release_index[:selected],
        release_code=parent.unit.release_code[:selected],
    )
    before = reconstruct_unit(first.unit, first.forests, first.code)
    assert torch.equal(before, reconstruct_unit(expected, parent.forests, parent.code))
    held = tuple(int((((first.unit.release_index % cols) // superblock) == b).sum())
                 for b in range(superblock_count(cols, superblock)))
    assert held != release_quota(selected, cols, superblock)

    with pytest.raises(GrammarError, match="whole unit.*RELEASE.*quota"):
        _rewrite(first, root)
    # Refusal must leave the valid parsed terminal usable.
    assert torch.equal(before, reconstruct_unit(first.unit, first.forests, first.code))


@pytest.mark.parametrize("parent_fixture", [
    "s6b_released_parent", "window_released_unit",
])
@pytest.mark.parametrize("selected", [0, 24, 48, 72, 96])
def test_whole_release_rewrite_keeps_canonical_counts_and_zero_terminal(
    request, parent_fixture, selected,
):
    """A refusal is about placement, not the count or the body family.

    Whole units canonically encoded at the same shorter totals still write
    byte-identically on rewrite. A zero terminal from the full descriptor
    remains writable, and drops no information that affects reconstruction.
    """
    from tessera.container import parse as parse_container

    parent = request.getfixturevalue(parent_fixture)
    root = parent.manifest.branch.root_q256
    geometry = parent.manifest.geometry
    empty = parent.unit.release_index[:0]
    base = replace(parent.unit, release_index=empty, release_code=empty)
    decoded = reconstruct_unit(base, parent.forests, parent.code)
    index = _canonical_release_order(
        decoded, geometry.columns, geometry.superblock_columns, selected)
    canonical = replace(parent.unit, release_index=index,
                        release_code=parent.unit.release_code[:selected])
    manifest, _r, blob = _shard_blob(canonical, parent.forests, root)
    assert manifest.shard is None
    assert manifest.plane(PlaneKind.RELEASE).count_granularity is CountGranularity.WHOLE_PLANE
    parsed = _cpu_parse(blob)
    _m, _r, rewritten = _rewrite(parsed, root)
    assert rewritten == blob
    assert torch.equal(parsed.unit.release_index, canonical.release_index)
    assert torch.equal(parsed.unit.release_code, canonical.release_code)
    assert torch.equal(reconstruct_unit(parsed.unit, parsed.forests, parsed.code),
                       reconstruct_unit(canonical, parent.forests, parent.code))
    if selected == 0:
        full_manifest, _r, full_blob = _rewrite(parent, root)
        cap = (parent.grid.payload_bits if full_manifest.body is BodyKind.WINDOW
               else parent.grid.rate_cap)
        short_blob, _rung = _laddered(
            parse_container(full_blob), full_manifest,
            _release_spec(full_manifest, 0, parent.unit, parent.grid),
            cap, parent.grid.arity,
        )
        _m, _r, zero_rewrite = _rewrite(_cpu_parse(short_blob), root)
        assert zero_rewrite == blob


def test_a_representable_whole_release_prefix_still_rewrites():
    """A shorter terminal over one superblock already has its own quota."""
    from tessera.container import parse as parse_container

    legacy = pathlib.Path(__file__).parent / "data" / "legacy"
    parent = _cpu_parse((legacy / "e2m1-768-release256-256c.tessera").read_bytes())
    root = parent.manifest.branch.root_q256
    manifest, _r, full_blob = _rewrite(parent, root)
    assert superblock_count(manifest.geometry.columns,
                            manifest.geometry.superblock_columns) == 1
    selected = parent.unit.released_positions // 2
    assert 0 < selected < parent.unit.released_positions
    short_blob, _rung = _laddered(
        parse_container(full_blob), manifest,
        _release_spec(manifest, selected, parent.unit, parent.grid),
        parent.grid.rate_cap, parent.grid.arity,
    )
    first = _cpu_parse(short_blob)
    assert torch.equal(first.unit.release_index, parent.unit.release_index[:selected])
    assert torch.equal(first.unit.release_code, parent.unit.release_code[:selected])
    _m, _r, rewritten = _rewrite(first, root)
    second = _cpu_parse(rewritten)
    assert torch.equal(second.unit.release_index, first.unit.release_index)
    assert torch.equal(second.unit.release_code, first.unit.release_code)
    assert torch.equal(reconstruct_unit(first.unit, first.forests, first.code),
                       reconstruct_unit(second.unit, second.forests, second.code))
    assert _rewrite(second, root)[2] == rewritten


def test_a_shorter_release_terminal_rewrites_to_its_own_prefix(
    s6b_released_parent
):
    """A rung that keeps the shard's first superblock and drops the second.

    The prefix rule is the descriptor's, not the total's: a shard's RELEASE
    plane is written superblock by superblock, so the only counts a terminal
    may declare are prefix sums of the descriptor's vector.  What the reader
    must then hand back is that vector **cut at the same boundary** -- ``(n,
    0)``, not the full-extent ``(n, m)`` -- because the next writer prices the
    plane from it, and a shard that claimed releases its terminal never carried
    would declare an extent its own region cannot fill.
    """
    from tessera.container import parse as parse_container

    parsed = s6b_released_parent
    grid = GRIDS["E2M1"]
    geometry = parsed.manifest.geometry
    cols, superblock = geometry.columns, geometry.superblock_columns
    root_q256 = parsed.manifest.branch.root_q256
    # A cut with both superblocks non-empty, so "the first superblock's share"
    # is a real prefix and not the whole plane under another name.
    shard = slice_unit(parsed, rows=(4, 12))
    counts = tuple(shard.release_counts)
    assert len(counts) == 2 and all(counts), (
        f"this rung needs both superblocks populated; the cut binned {counts}")
    # A terminal also ends on a byte (schema D4), and a RELEASE element is
    # ``RELEASE_BITS`` wide, so the boundary this rung cuts at has to be both a
    # superblock prefix and a whole number of bytes.
    assert not (counts[0] * RELEASE_BITS) % 8, counts
    manifest, _r, first_blob = _shard_blob(shard, parsed.forests, root_q256)
    whole = _cpu_parse(first_blob)
    art = parse_container(first_blob)

    head = counts[0]
    spec = _release_spec(manifest, head, whole.unit, grid)
    cut, rung = _laddered(art, manifest, spec, grid.rate_cap, grid.arity)
    assert rung.plane_elements[
        manifest.plane_order.index(PlaneKind.RELEASE)] == head

    rung_parsed = _cpu_parse(cut)
    # The descriptor's vector at the terminal's boundary, not the descriptor's
    # vector: the second superblock is past the cut and holds nothing.
    assert tuple(rung_parsed.unit.release_counts) == (head, 0)
    assert torch.equal(
        rung_parsed.unit.release_index, whole.unit.release_index[:head])
    assert torch.equal(
        rung_parsed.unit.release_code, whole.unit.release_code[:head])
    assert bool(((rung_parsed.unit.release_index % cols) < superblock).all())

    rewritten_manifest, _r, rewritten = _rewrite(rung_parsed, root_q256)
    assert _release_counts_of(rewritten_manifest) == (head, 0)
    again = _cpu_parse(rewritten)
    assert tuple(again.unit.release_counts) == (head, 0)
    assert torch.equal(again.unit.release_index, rung_parsed.unit.release_index)
    assert torch.equal(again.unit.release_code, rung_parsed.unit.release_code)
    assert torch.equal(
        reconstruct_unit(again.unit, again.forests, again.code),
        reconstruct_unit(rung_parsed.unit, rung_parsed.forests, rung_parsed.code),
    )
    # ...and the rung is not the whole plane by another name.
    assert not torch.equal(
        reconstruct_unit(rung_parsed.unit, rung_parsed.forests, rung_parsed.code),
        reconstruct_unit(whole.unit, whole.forests, whole.code),
    )

    # A count that is not a superblock prefix names positions in a superblock
    # whose codes the plane never carried.  The manifest refuses to describe
    # such a terminal at all -- the descriptor's counts are its granules --
    # so the bytes cannot be built...
    mid = head - 2  # still a whole number of bytes; still not a prefix sum
    assert mid not in (0, counts[0], sum(counts)) and not (mid * RELEASE_BITS) % 8
    with pytest.raises(ManifestError, match="granule boundary"):
        _laddered(art, manifest, _release_spec(manifest, mid, whole.unit, grid),
                  grid.rate_cap, grid.arity)
    # ...and the reader that would place the codes keeps its own backstop, so
    # a manifest reaching it by another route is refused rather than respread.
    from tessera.unit_artifact import _shard_release_counts

    with pytest.raises(GrammarError, match="superblock prefix"):
        _shard_release_counts(manifest, mid)


def test_the_writer_refuses_a_released_shard_whose_counts_it_cannot_carry(
    cpu_released_unit
):
    """tessera#336's other half, where the bytes are decided.

    Restoring the vector at the reader closes the path that produced a wrong
    artifact; it does not make the wrong artifact unwritable.  A shard that
    carries releases and declares no per-superblock counts has a placement the
    RELEASE descriptor cannot express, and the writer must say so rather than
    write the total and let the reader respread it -- which is precisely what
    it did, silently, for every parsed shard.
    """
    _unit, forests, _grid, blob = cpu_released_unit
    parsed = _cpu_parse(blob)
    root_q256 = parsed.manifest.branch.root_q256
    shard = slice_unit(parsed, rows=(16, 32))
    assert int(shard.release_index.numel())

    stripped = replace(shard, release_counts=())
    with pytest.raises(GrammarError, match="no per-superblock RELEASE counts"):
        _shard_blob(stripped, forests, root_q256)

    # A vector that disagrees with the placement is the same defect stated the
    # other way round, and is refused with the placement it actually holds.
    counts = tuple(shard.release_counts)
    swapped = replace(shard, release_counts=counts[::-1])
    assert swapped.release_counts != counts, "the swap has to change the vector"
    with pytest.raises(GrammarError, match="one object"):
        _shard_blob(swapped, forests, root_q256)

    # The unaltered shard still writes, so the refusal is about the
    # disagreement and not about shards carrying releases at all.
    _shard_blob(shard, forests, root_q256)


def test_the_reader_refuses_a_released_shard_written_without_its_counts(
    cpu_released_unit
):
    """The same rule, read back -- the bytes the unfixed writer produced.

    An artifact of this shape is byte-self-consistent: every digest agrees with
    it, the RELEASE plane is a legal 4-bit field, and nothing in the reader
    would notice that the codes had landed on a quota's positions instead of
    the cut's.  That is the silent misdecode the refusal exists to prevent, and
    it is the same doctrine ``grammar.require_release_defined`` already applies
    at both ends.
    """
    from dataclasses import replace as _replace

    from tessera.container import parse as parse_container, serialize

    _unit, forests, _grid, blob = cpu_released_unit
    parsed = _cpu_parse(blob)
    shard = slice_unit(parsed, rows=(16, 32))
    manifest, _r, first_blob = _shard_blob(
        shard, forests, parsed.manifest.branch.root_q256)
    art = parse_container(first_blob)
    descriptor = manifest.plane(PlaneKind.RELEASE)
    assert descriptor.count_granularity is CountGranularity.PER_SUPERBLOCK

    # Exactly what the pre-fix rewrite wrote: the shard's total, spelled as a
    # whole unit's, over the shard's own plane region.
    respread = _replace(
        descriptor,
        count_granularity=CountGranularity.WHOLE_PLANE,
        counts=(descriptor.element_count,),
        restart_offsets=(0,),
    )
    planes = tuple(
        respread if p.kind is PlaneKind.RELEASE else p for p in manifest.planes
    )
    forged = serialize(_replace(manifest, planes=planes), art.plane_region)
    with pytest.raises(GrammarError, match="per-superblock counts"):
        _cpu_parse(forged)


def test_the_reslice_the_issue_refused_builds(cpu_units):
    """tessera#140, first reproduction: rows (32, 64) then (16, 32) is the
    legal shard [48, 64) of a 64-row parent, and it was refused as running
    past a parent of 32 rows -- the immediate parent's extent under the
    original's offset."""
    _unit, forests, grid, blob = cpu_units["s6b-tcq"]
    parsed = _cpu_parse(blob)
    assert parsed.manifest.geometry.rows == 64
    half = slice_unit(parsed, rows=(32, 64))
    sub = slice_unit(half, rows=(16, 32), code=parsed.code, grid=grid)
    assert (sub.row_offset, sub.parent_rows) == (48, 64)
    manifest, _r, _blob = _build_shard(sub, forests, parsed, "sub")
    assert manifest.shard == _origin_record(parsed, 48, 0, manifest.shard.state_bits)
    assert manifest.shard.state_bits == CODE.memory


def test_the_reslice_the_issue_serialised_names_one_parent(cpu_units):
    """tessera#140, second reproduction: rows (0, 32) then (0, 16) serialised
    a record claiming a 32-row parent under the 64-row original's digest --
    a lie no reader holding only the shard could catch.  Every field now
    names the original."""
    _unit, forests, grid, blob = cpu_units["s6b-tcq"]
    parsed = _cpu_parse(blob)
    top = slice_unit(parsed, rows=(0, 32))
    sub = slice_unit(top, rows=(0, 16), code=parsed.code, grid=grid)
    manifest, _r, sub_blob = _build_shard(sub, forests, parsed, "sub")
    assert manifest.shard == _origin_record(parsed, 0, 0, 0)
    assert _cpu_parse(sub_blob).manifest.shard == manifest.shard


def test_a_parsed_shard_reslices_into_the_original_frame(cpu_units):
    """Round trip through the reader: a shard written to bytes and parsed
    back is a ``ParsedUnit`` whose manifest is the SHARD's, and cutting it
    again must not take the shard's own geometry and digest for the parent's.
    The composed shard rebuilds to the direct cut's bytes and decodes to the
    original's window."""
    _unit, forests, grid, blob = cpu_units["e4m3-window-channel"]
    parsed = _cpu_parse(blob)
    rows = parsed.manifest.geometry.rows
    half = slice_unit(parsed, rows=(rows // 2, rows))
    _m, _r, half_blob = _build_shard(half, forests, parsed, "half")
    parsed_half = _cpu_parse(half_blob)
    assert parsed_half.manifest.shard is not None
    composed = slice_unit(parsed_half, rows=(rows // 4, rows // 2))
    direct = slice_unit(parsed, rows=(3 * rows // 4, rows))
    m_composed, _r, composed_blob = _build_shard(composed, forests, parsed, "q")
    m_direct, _r, direct_blob = _build_shard(direct, forests, parsed, "q")
    assert m_composed.shard == m_direct.shard == _origin_record(
        parsed, 3 * rows // 4, 0, m_direct.shard.state_bits)
    assert composed_blob == direct_blob
    full = reconstruct_unit(parsed.unit, parsed.forests, parsed.code)
    back = _cpu_parse(composed_blob)
    assert torch.equal(
        reconstruct_unit(back.unit, back.forests, back.code), full[3 * rows // 4:]
    )


def test_a_shard_refuses_a_parent_that_contradicts_its_record(cpu_units):
    """An explicit ``parent_shape`` or ``parent_digest`` that disagrees with
    the record a shard already carries is refused by name: the record names
    the original, and a caller cannot rename it to the immediate parent."""
    _unit, forests, grid, blob = cpu_units["s6b-tcq"]
    parsed = _cpu_parse(blob)
    half = slice_unit(parsed, rows=(32, 64))
    with pytest.raises(GrammarError, match="parent_shape"):
        slice_unit(half, rows=(16, 32), code=parsed.code, grid=grid,
                   parent_shape=(32, 256))
    with pytest.raises(GrammarError, match="parent_digest"):
        slice_unit(half, rows=(16, 32), code=parsed.code, grid=grid,
                   parent_digest=bytes(32))
    # The original's own record, restated, is not a contradiction.
    same = slice_unit(half, rows=(16, 32), code=parsed.code, grid=grid,
                      parent_shape=(64, 256),
                      parent_digest=parsed.manifest.manifest_digest())
    assert (same.row_offset, same.parent_rows) == (48, 64)


@pytest.mark.parametrize("label", ["e4m3-window-channel", "s6b-tcq"])
def test_an_unsliced_unit_and_a_first_cut_do_not_move(cpu_units, label):
    """The frame fix touches re-slices only: the identity slice is still the
    unit, byte for byte and with no record, and a first cut's record is the
    one every reader already assumes -- offsets into, extent of, and digest
    of the unit it was cut from."""
    _unit, forests, grid, blob = cpu_units[label]
    parsed = _cpu_parse(blob)
    whole = slice_unit(parsed)
    _m, _r, again = build_unit_artifact(
        whole, parsed.manifest.branch.unit_id, forests,
        parsed.manifest.branch.root_q256, CODE)
    assert again == blob
    assert _cpu_parse(again).manifest.shard is None
    rows = parsed.manifest.geometry.rows
    manifest, _r, _blob = _build_shard(
        slice_unit(parsed, rows=(rows // 2, rows)), forests, parsed, "half")
    assert manifest.shard == _origin_record(
        parsed, rows // 2, 0, manifest.shard.state_bits)


# ------------------------------------------------------- the trellis oracle


def test_conv_state_correction_against_the_scalar_oracle():
    """The vectorised start-state correction equals ``TCQ.decode``'s walk.

    ``_conv_state_stream`` corrects a shard's first ``memory`` steps with a
    closed-form ``init >> t``.  The oracle is the scalar reference trellis,
    stepped from the same register -- a different implementation of the same
    machine, not another vectorisation of the same expression.
    """
    from tessera.alphabet import build_forest

    grid = GRIDS["E2M1"]
    forest = build_forest(3, grid=grid)
    tcq = TCQ(forest, CODE)
    rng = random.Random(3)
    positions = 24
    for span in (1, 2):
        for start in (0, 1, 5, 37, 63):
            bits = [rng.randrange(2)
                    for _ in range(_body_bits(3, positions, span))]
            want = tcq.decode(bits, positions, span, initial_state=start)
            body = _bits_to_body(bits, 3, positions, span)
            got = replay_body(
                body, forest, CODE, span,
                torch.tensor([start], dtype=torch.long),
            )
            assert got.reshape(-1).tolist() == want, (span, start)


def _body_bits(rate, positions, span):
    from tessera.trellis import body_bits

    return body_bits(rate, positions, span)


def _bits_to_body(bits, rate, positions, span):
    """One column's bit list -> the ``[positions, 1]`` field tensor."""
    from tessera.wire import field_widths

    widths = field_widths(rate, span)
    out, cursor = [], 0
    for _ in range(positions // span):
        for width in widths:
            value = 0
            for _bit in range(width):
                value = (value << 1) | bits[cursor]
                cursor += 1
            out.append(value)
    return torch.tensor(out, dtype=torch.uint8).reshape(positions, 1)


def test_window_state_correction_matches_a_scalar_walk():
    """``replay_window`` from a start state equals the recursion, stepped."""
    rng = random.Random(5)
    for window_bits, rate in ((12, 4), (14, 4), (8, 3), (6, 6)):
        steps = 20
        bits = torch.tensor(
            [[rng.randrange(1 << rate)] for _ in range(steps)], dtype=torch.uint8
        )
        for start in (0, 1, (1 << window_bits) - 1, rng.randrange(1 << window_bits)):
            want, state = [], start
            for step in range(steps):
                state = ((state << rate) | int(bits[step, 0])) & ((1 << window_bits) - 1)
                want.append(state)
            got = replay_window(
                bits, window_bits, rate, torch.tensor([start], dtype=torch.long)
            )
            assert got.reshape(-1).tolist() == want, (window_bits, rate, start)


@needs_cuda
@pytest.mark.parametrize("fused", ["1", "0"])
def test_fused_and_eager_replay_agree_on_a_shard(units, monkeypatch, fused):
    """The compiled chain and the eager fallback decode a shard identically."""
    import tessera.decode as decode

    unit, forests, grid, blob = units["e2m1-tcq-lut-release"]
    parsed = parse_unit_artifact(blob, device=DEVICE)
    shard = slice_unit(parsed, rows=(16, 32))
    # The env var is read per call, so setting it is enough; the compile
    # cache holds compiled functions and never the decision to use them.
    monkeypatch.setenv("TESSERA_FUSED_REPLAY", fused)
    codes = decode_codes_mixed(shard, parsed.forests, parsed.code)
    full = decode_codes_mixed(parsed.unit, parsed.forests, parsed.code)
    assert torch.equal(codes, full[16:32])


# ------------------------------------------------------------- the RELEASE plane


def test_release_partition_counts_the_trailing_partial_superblock():
    """``block_of = column // superblock`` produces ``ceil(cols/superblock)``
    block indices, so the quota has to run over that many.  Flooring it meant a
    640-column unit's positions 512..639 could never be released, while
    ``layout.build_planes`` -- which ceilings -- allocated their granule anyway.
    """
    torch.manual_seed(3)
    superblock, rows = 256, 16
    for cols, blocks in ((512, 2), (640, 3), (384, 2), (320, 2), (128, 1)):
        assert superblock_count(cols, superblock) == blocks
        decoded = torch.randn(rows, cols)
        index = _canonical_release_order(decoded, cols, superblock, 3 * blocks)
        touched = sorted(set(((index % cols) // superblock).tolist()))
        assert touched == list(range(blocks)), cols


@pytest.mark.parametrize(
    "width,ok",
    [(256, True), (512, True), (128, True), (1, True), (384, False), (640, False)],
)
def test_slice_release_admits_only_widths_where_ceiling_equals_floor(width, ok):
    """A regression pin, not a fail-before test: ``_slice_release``'s guard
    already admitted only a union of whole superblocks or a cut inside one, and
    on exactly those widths the ceiling and the floor agree.  That is why
    moving the shard path onto ``superblock_count`` cannot move a shard's
    counts -- and this is the property that has to keep holding for it.
    """
    superblock, rows, columns = 256, 4, 1024
    unit = SimpleNamespace(
        body_bits=torch.zeros(rows, columns),
        release_index=torch.tensor([0, 1, columns + 2]),
        release_code=torch.tensor([3, 5, 7]),
    )
    if ok:
        assert superblock_count(width, superblock) == max(1, width // superblock)
        _, _, counts = _slice_release(
            unit, rows, columns, 0, rows, 0, width, superblock
        )
        assert len(counts) == superblock_count(width, superblock)
    else:
        assert superblock_count(width, superblock) != max(1, width // superblock)
        with pytest.raises(GrammarError, match="only on superblock boundaries"):
            _slice_release(unit, rows, columns, 0, rows, 0, width, superblock)


@pytest.mark.parametrize("cols", [256, 257, 320, 512, 640])
def test_release_order_generalises_the_release_quota(cols):
    """``decode.release_order`` at ``grammar.release_quota`` *is* the encoder's
    own placement.  The shard needs the general form; this binds the two so
    they cannot drift -- at a partial trailing superblock too, which is where
    the quota and an equal count stop agreeing."""
    torch.manual_seed(2)
    rows, superblock = 16, 256
    decoded = torch.randn(rows, cols)
    for total in (0, 1, 7, 64, 129, rows * cols):
        counts = release_quota(total, cols, superblock)
        assert torch.equal(
            release_order(decoded, cols, superblock, counts),
            _canonical_release_order(decoded, cols, superblock, total),
        )


@pytest.mark.parametrize(
    "body,plane,q256",
    [
        (BodyKind.WINDOW, ScalePlaneKind.CHANNEL, 1024),
        (BodyKind.TCQ, ScalePlaneKind.LUT, 512),
    ],
    ids=["window", "tcq"],
)
def test_release_is_refused_at_read_on_a_grid_wider_than_the_plane(body, plane, q256):
    """Both readers refuse a release on a grid the RELEASE plane cannot name.

    This test used to prove the other doctrine.  Under #163 it pinned the
    *acceptance* of exactly these bytes: the readers were taught to rank
    released positions by the resolved grid's value table rather than by the
    16-entry E2M1 one, so that an E4M3 unit carrying releases parsed instead of
    walking off the end of that table.  The ranking fix was right and stays --
    ``grid_value_table(grid)`` is still what both readers use -- but accepting
    the artifact was the wrong half of it, because the same PR taught
    ``encode.encode_unit`` to call this byte string undefined.  Two rules for
    one wire is the defect (tessera#180, finding S5), and AGENTS rule 5 picks
    the refusing side: a release names a whole payload code, ``RELEASE_BITS``
    cannot spell most of E4M3's 256, and the codes that *do* fit land on
    positions the reader itself ranked and decode to values no encoder chose.
    Nothing is silent about it any more.

    The unit is built by hand rather than by ``encode_unit`` for the same
    reason: the encoder refuses this call, so only a non-conforming writer
    could produce these bytes -- which is precisely the writer a reader has to
    be closed against.  ``build_unit_artifact`` is left unguarded so this test
    can *be* that writer; the artifact it makes is wire-legal in every other
    respect (the manifest, the terminal and the plane all validate), and it is
    the reader that says no.
    """
    grid = GRIDS["E4M3"]
    rows, cols, superblock, released = 8, 256, 256, 8
    rates, forests = _plan_for(grid, q256, cols, body, None)
    torch.manual_seed(7)
    weight = torch.randn(rows, cols) * 0.02
    extra = {}
    if body is BodyKind.WINDOW:
        recipe = wire_recipe(grid, q256)
        extra = dict(
            window_bits=recipe.window_bits, window_seed=recipe.window_seed,
            window_sigma=recipe.window_sigma, channel_sigma=recipe.channel_sigma,
        )
    unit = encode_unit(
        weight, forests, rates, CODE, completion=0, released_positions=0,
        span=1, scale_plane=plane, body=body, scale_refit=2, **extra,
    )

    forest = grid if body is BodyKind.WINDOW else forests
    code = None if body is BodyKind.WINDOW else CODE
    assert not release_defined_on(grid), (
        "this case exists to carry a release on a grid wider than the plane"
    )
    pre = decode_codes_mixed(unit, forest, code, apply_release=False)
    decoded = grid_value_table(grid)[pre.int()] * unit_scale_field(unit, rows, cols)
    want = _canonical_release_order(decoded, cols, superblock, released)

    unit.release_index = want
    unit.release_code = torch.arange(released) % (1 << RELEASE_BITS)
    _m, _r, blob = build_unit_artifact(
        unit, "e4m3-release", forest, q256 * grid.arity, code
    )

    with pytest.raises(GrammarError, match=f"stores {RELEASE_BITS} bits"):
        parse_unit_artifact(blob)


@pytest.mark.parametrize(
    "body,plane",
    [(BodyKind.WINDOW, ScalePlaneKind.CHANNEL), (BodyKind.TCQ, ScalePlaneKind.LUT)],
    ids=["window", "tcq"],
)
def test_the_release_element_width_is_one_number(monkeypatch, body, plane):
    """Move ``grammar.RELEASE_BITS`` and the bytes move with the descriptor.

    The RELEASE plane's element width is derived once --
    ``planes.NORMATIVE_ELEMENT_BITS[RELEASE]`` is ``RELEASE_BITS`` -- and was
    then spelled ``4`` at the three sites that actually touch the bits: the
    writer's ``pack_uniform`` and both readers' ``unpack_uniform``
    (tessera#183, M10).  Three literals agreeing with a constant are not the
    same object as the constant, and the failure mode is the one the audit
    names: the descriptor says one width, the payload is packed at another,
    and the two halves of the wire disagree.

    Widening the constant is exactly the move that catches it, so the test
    makes it: the three homes of the number are patched together and the
    artifact has to round-trip.  Nothing here is a wire change -- the patch is
    undone with the test, and at the shipping ``RELEASE_BITS`` the bytes are
    what they always were, which
    ``test_encoded_unit_bytes_match_encoder_identity_baseline`` pins.

    E2M1 rather than a wider grid, and codes the grid can name, because
    widening the plane widens ``grammar.release_defined_on`` with it: the
    admissible grids are derived from ``RELEASE_BITS`` (tessera#180), so a
    grid that is undefined at 4 bits is undefined at 5 too, and both readers
    would refuse the artifact before they ever unpacked the plane.  The
    premise is asserted from that predicate rather than restated.

    Both readers are covered because there are two: ``parse_unit_artifact``
    reads a TCQ unit's RELEASE plane and ``_read_window_unit`` reads a window
    unit's.  The unit is encoded with ``released_positions=0`` and released by
    hand so the codes are this test's and not an argmin's; they are distinct
    and non-zero, so a reader left at the old width reads different numbers
    out of the same bits rather than the same ones by luck.
    """
    from tessera import grammar, planes as planes_module, unit_artifact

    widened = RELEASE_BITS + 1
    monkeypatch.setattr(grammar, "RELEASE_BITS", widened)
    monkeypatch.setitem(
        planes_module.NORMATIVE_ELEMENT_BITS, PlaneKind.RELEASE, widened
    )
    monkeypatch.setattr(unit_artifact, "RELEASE_BITS", widened, raising=False)

    grid = GRIDS["E2M1"]
    assert release_defined_on(grid), "the widened plane still has to name this grid"
    rows, cols, superblock, released, q256 = 8, 256, 256, 8, 512
    rates, forests = _plan_for(grid, q256, cols, body, None)
    torch.manual_seed(7)
    weight = torch.randn(rows, cols) * 0.02
    extra = {}
    if body is BodyKind.WINDOW:
        recipe = wire_recipe(grid, q256)
        extra = dict(
            window_bits=max(rates), window_seed=recipe.window_seed,
            window_sigma=recipe.window_sigma, channel_sigma=recipe.channel_sigma,
        )
    unit = encode_unit(
        weight, forests, rates, CODE, completion=0, released_positions=0,
        span=1, scale_plane=plane, body=body, scale_refit=2, **extra,
    )

    forest = grid if body is BodyKind.WINDOW else forests
    code = None if body is BodyKind.WINDOW else CODE
    pre = decode_codes_mixed(unit, forest, code, apply_release=False)
    decoded = grid_value_table(grid)[pre.int()] * unit_scale_field(unit, rows, cols)
    unit.release_index = _canonical_release_order(decoded, cols, superblock, released)
    unit.release_code = torch.arange(1, released + 1)
    assert int(unit.release_code.max()) < grid.size, "codes the grid can name"

    _m, _r, blob = build_unit_artifact(
        unit, "widened-release", forest, q256 * grid.arity, code
    )
    parsed = parse_unit_artifact(blob)
    assert torch.equal(parsed.unit.release_code.cpu(), unit.release_code.cpu())
    assert torch.equal(parsed.unit.release_index.cpu(), unit.release_index.cpu())


def test_the_block_scale_plane_widths_are_one_number(monkeypatch):
    """The same statement as above, for the two block scale planes.

    Found while fixing the RELEASE plane's copy of it (tessera#183, M10) and
    fixed here rather than filed: ``pack_uniform(unit.scale_base, 8)`` and
    ``pack_uniform(unit.scale_refine, 4)`` restate
    ``planes.NORMATIVE_ELEMENT_BITS`` at the writer, and each reader restates
    it once more.  Unlike RELEASE these two have no constant in ``grammar`` to
    point at, so the descriptor table -- which is what a reader consults and
    what ``layout.build_planes`` charges the bytes against -- is their one
    home.

    Both widths move together because one unit carries both planes, and an
    S6b unit is the only kind that does: a LUT plane carries no SCALE_BASE and
    a CHANNEL plane carries neither.
    """
    from tessera import planes as planes_module

    for kind in (PlaneKind.SCALE_BASE, PlaneKind.SCALE_REFINE):
        monkeypatch.setitem(
            planes_module.NORMATIVE_ELEMENT_BITS, kind,
            planes_module.NORMATIVE_ELEMENT_BITS[kind] + 1,
        )
    unit, _forests, _grid, blob = _s6b_unit(device="cpu")
    parsed = _cpu_parse(blob)
    assert unit.scale_base.numel() and unit.scale_refine.numel()
    assert torch.equal(parsed.unit.scale_base.cpu(), unit.scale_base.cpu())
    assert torch.equal(parsed.unit.scale_refine.cpu(), unit.scale_refine.cpu())


def test_release_is_refused_on_a_grid_wider_than_the_release_plane():
    """The encoder says which dial does not exist, and says it before it works.

    A release stores a whole payload code and the RELEASE plane is
    ``grammar.RELEASE_BITS`` wide whatever the grid, so release is a 16-code
    grid's dial.  Without this refusal the pass ran to completion, picked
    release codes by argmin over all 256 E4M3 values, and failed at write with
    ``wire.pack_uniform``'s "value out of range for a 4-bit field: [113, 251]"
    -- which names neither release nor the grid, and arrives one call after the
    encoder that chose them.  Widening the plane per grid is a wire change.
    """
    grid = GRIDS["E4M3"]
    rows, cols, q256 = 8, 256, 1024
    rates, forests = _plan_for(grid, q256, cols, BodyKind.WINDOW, None)
    recipe = wire_recipe(grid, q256)
    torch.manual_seed(7)
    weight = torch.randn(rows, cols) * 0.02
    assert grid.size > (1 << RELEASE_BITS)
    with pytest.raises(GrammarError, match=f"stores {RELEASE_BITS} bits"):
        encode_unit(
            weight, forests, rates, CODE, completion=0, released_positions=8,
            span=1, scale_plane=ScalePlaneKind.CHANNEL, body=BodyKind.WINDOW,
            scale_refit=2, window_bits=recipe.window_bits,
            window_seed=recipe.window_seed, window_sigma=recipe.window_sigma,
            channel_sigma=recipe.channel_sigma,
        )


@needs_cuda
def test_release_restricts_to_the_shard(units):
    """The released set of a shard is the parent's, restricted -- and the
    reader recomputes exactly that placement from the shard's own decode."""
    unit, forests, grid, blob = units["e2m1-tcq-lut-release"]
    parsed = parse_unit_artifact(blob, device=DEVICE)
    geometry = parsed.manifest.geometry
    rows, cols = geometry.rows, geometry.columns
    parent = set(parsed.unit.release_index.tolist())
    assert parent, "this case exists to carry releases"
    seen = 0
    for lo, hi in _tp_ranges(rows, 4):
        shard = slice_unit(parsed, rows=(lo, hi))
        want = {
            (flat // cols - lo) * cols + flat % cols
            for flat in parent
            if lo <= flat // cols < hi
        }
        assert set(shard.release_index.tolist()) == want
        assert sum(shard.release_counts) == len(want)
        seen += len(want)
        # ...and the reader lands on the same positions from bytes alone.
        _m, _r, shard_blob = build_unit_artifact(
            shard, "shard", forests, parsed.manifest.branch.root_q256, CODE)
        back = parse_unit_artifact(shard_blob, device=DEVICE)
        assert set(back.unit.release_index.tolist()) == want
        assert torch.equal(back.unit.release_code, shard.release_code)
    assert seen == len(parent), "every parent release lands in exactly one shard"


@needs_cuda
def test_release_counts_travel_per_superblock(units):
    """A shard's RELEASE plane declares its counts; a whole unit's does not."""
    from tessera.planes import CountGranularity

    unit, forests, grid, blob = units["e2m1-tcq-lut-release"]
    parsed = parse_unit_artifact(blob, device=DEVICE)
    whole = parsed.manifest.plane(PlaneKind.RELEASE)
    assert whole.count_granularity is CountGranularity.WHOLE_PLANE
    shard = slice_unit(parsed, rows=(16, 32))
    _m, _r, shard_blob = build_unit_artifact(
        shard, "shard", forests, parsed.manifest.branch.root_q256, CODE)
    plane = parse_unit_artifact(shard_blob).manifest.plane(PlaneKind.RELEASE)
    assert plane.count_granularity is CountGranularity.PER_SUPERBLOCK
    assert plane.counts == shard.release_counts


# ---------------------------------------------------------------- refusals


@needs_cuda
def test_illegal_granularity_is_refused(units):
    """Every refusal names the granularity it is refusing against."""
    unit, forests, grid, blob = units["e2m1-tcq-lut-release"]
    parsed = parse_unit_artifact(blob, device=DEVICE)
    row_gran, col_gran = shard_granularity(
        parsed.unit, parsed.manifest.geometry.superblock_columns, grid.arity
    )
    assert (row_gran, col_gran) == (2, 256)
    with pytest.raises(GrammarError, match="row offset 1 is not a multiple"):
        slice_unit(parsed, rows=(1, 17))
    with pytest.raises(GrammarError, match="column offset 16 is not a multiple"):
        slice_unit(parsed, cols=(16, 256))
    with pytest.raises(GrammarError, match="not inside"):
        slice_unit(parsed, rows=(0, 4096))


@needs_cuda
def test_a_rotated_unit_is_refused():
    """Rotation is a column-block structure a cut would break in silence."""
    from tessera.manifest import RotationState

    unit, forests, grid, blob = _s6b_unit()
    parsed = parse_unit_artifact(blob, device=DEVICE)
    parsed.unit.rotation = RotationState.R_IN_ONLY
    with pytest.raises(GrammarError, match="refusing to slice a rotated unit"):
        slice_unit(parsed, rows=(0, 16))


def test_a_manifest_cannot_declare_a_state_it_does_not_carry():
    """The two halves of the shard record have to agree with each other."""
    with pytest.raises(ManifestError, match="carries its start state"):
        ShardOrigin(row_offset=16, col_offset=0, parent_rows=32,
                    parent_columns=64, parent_digest=bytes(32), state_bits=0)
    with pytest.raises(ManifestError, match="carries its start state"):
        ShardOrigin(row_offset=0, col_offset=0, parent_rows=32,
                    parent_columns=64, parent_digest=bytes(32), state_bits=14)


@needs_cuda
def test_the_span2_kernel_lane_refuses_a_shard(units):
    """The kernel lane fails closed where it cannot take a start state."""
    from tessera.kernel import pack_unit_for_kernel

    unit, forests, grid, blob = units["e2m1x2-cap-tcq-lut"]
    parsed = parse_unit_artifact(blob, device=DEVICE)
    shard = slice_unit(parsed, rows=(16, 32))
    forest = parsed.forests[parsed.unit.rates[0]]
    with pytest.raises(GrammarError, match="does not yet take a start state"):
        pack_unit_for_kernel(shard, forest, parsed.code)


@needs_cuda
def test_the_window_kernel_lane_decodes_a_shard(units):
    """...and where it *can*, it does, with no kernel change at all: the
    window plane's ``L``-bit pad is the start state, so a shard's first
    window read is the recursion's own first step."""
    from tessera.kernel import gemv_from_packed, pack_unit_for_kernel

    unit, forests, grid, blob = units["e4m3-window-channel"]
    parsed = parse_unit_artifact(blob, device="cuda")
    cols = parsed.manifest.geometry.columns
    torch.manual_seed(1)
    x = torch.randn(cols, device="cuda")
    for lo, hi in _tp_ranges(parsed.manifest.geometry.rows, 2):
        shard = slice_unit(parsed, rows=(lo, hi))
        reference = reconstruct_unit(shard, parsed.forests, parsed.code).float() @ x
        got = gemv_from_packed(x, pack_unit_for_kernel(shard, parsed.grid, None))
        assert torch.allclose(got.float(), reference, rtol=1e-4, atol=1e-4)


@needs_cuda
def test_a_state_width_that_contradicts_the_record_is_refused(units):
    """INITIAL_STATE has no normative width, so the manifest supplies one.

    ``NORMATIVE_ELEMENT_BITS`` deliberately omits this plane -- its width is
    the body's, 14 bits under the shipped window wire and the code's memory
    under the coset trellis -- which means the descriptor alone constrains
    nothing.  The shard record is what binds it, and a manifest whose plane
    and record disagree has to be refused rather than read: a state read at
    the wrong width is a start state that decodes to plausible wrong weights.
    """
    from dataclasses import replace

    unit, forests, grid, blob = units["e4m3-window-channel"]
    parsed = parse_unit_artifact(blob, device=DEVICE)
    rows = parsed.manifest.geometry.rows
    shard = slice_unit(parsed, rows=(rows // 2, rows))
    _m, _r, shard_blob = build_unit_artifact(
        shard, "half", forests, parsed.manifest.branch.root_q256, CODE)
    manifest = parse_unit_artifact(shard_blob, device=DEVICE).manifest
    assert manifest.shard.state_bits == manifest.window_bits
    with pytest.raises(ManifestError, match="bits wide"):
        replace(manifest, shard=replace(manifest.shard, state_bits=13))


# ------------------------------------------------- the serving materialisers


@needs_cuda
@pytest.mark.parametrize("label", [c[0] for c in CASES])
@pytest.mark.parametrize("axis", ["row", "column"])
def test_stock_materialisation_of_a_shard_is_the_parent_sliced(units, label, axis):
    """The tensors a runtime is handed slice with the unit.

    ``materialize_stock`` is what the resident serving route calls -- the
    NVFP4 triple on an E2M1 unit, the per-channel FP8 pair on an E4M3 one --
    and the plugin calls it on the *shard*, never on the parent.  Decoding
    correctly is not enough on its own: the packed nibble pairs and the
    per-16 scale bytes are a second layout over the same values, and a
    column cut that landed off a nibble pair or off a scale group would
    still decode right and pack wrong.
    """
    from tessera.stock import materialize_stock, stock_kind

    unit, forests, grid, blob = units[label]
    parsed = parse_unit_artifact(blob, device=DEVICE)
    geometry = parsed.manifest.geometry
    if not can_shard(parsed, 2, axis):
        pytest.skip(f"{label} does not cut two ways along {axis}s")
    whole = materialize_stock(parsed.unit, parsed.forests, parsed.code)
    #: Columns per element of each tensor's second axis, or ``None`` where the
    #: second axis is not columns at all: the FP8 pair's scale is one word per
    #: output ROW, so a column cut keeps every row's scale untouched.
    strides = (
        {"weight": 1, "weight_scale": None}
        if stock_kind(whole) == "fp8"
        else {"weight_packed": 2, "weight_scale": parsed.unit.half,
              "weight_global_scale": None}
    )
    extent = geometry.rows if axis == "row" else geometry.columns
    for lo, hi in _tp_ranges(extent, 2):
        kwargs = {"rows": (lo, hi)} if axis == "row" else {"cols": (lo, hi)}
        shard = materialize_stock(
            slice_unit(parsed, **kwargs), parsed.forests, parsed.code
        )
        assert set(shard) == set(whole)
        for name, got in shard.items():
            want = whole[name]
            stride = strides[name]
            if want.ndim == 2 and axis == "row":
                want = want[lo:hi]
            elif want.ndim == 2 and axis == "column" and stride is not None:
                want = want[:, lo // stride : hi // stride]
            assert torch.equal(got.view(torch.uint8), want.view(torch.uint8)), (
                f"{label} {axis} [{lo}, {hi}) {name}"
            )


@needs_cuda
@pytest.mark.parametrize("axis", ["row", "column"])
def test_materialize_fp8_of_a_shard_is_the_parent_sliced(units, axis):
    """``materialize_fp8`` is the FP8 route's entry point, called directly.

    Both returns have to slice: the byte plane on both axes, and the
    per-output-row scale on rows only -- a column shard keeps every row's
    scale, because a column cut does not change which rows exist.
    """
    from tessera.decode import materialize_fp8

    unit, forests, grid, blob = units["e4m3-window-channel"]
    parsed = parse_unit_artifact(blob, device=DEVICE)
    geometry = parsed.manifest.geometry
    bytes_whole, scale_whole = materialize_fp8(
        parsed.unit, parsed.forests, parsed.code
    )
    extent = geometry.rows if axis == "row" else geometry.columns
    for lo, hi in _tp_ranges(extent, 4):
        kwargs = {"rows": (lo, hi)} if axis == "row" else {"cols": (lo, hi)}
        got, scale = materialize_fp8(
            slice_unit(parsed, **kwargs), parsed.forests, parsed.code
        )
        if axis == "row":
            assert torch.equal(got, bytes_whole[lo:hi])
            assert torch.equal(scale, scale_whole[lo:hi])
        else:
            assert torch.equal(got, bytes_whole[:, lo:hi])
            assert torch.equal(scale, scale_whole)


# ------------------------------------------------------------- granularity


@needs_cuda
@pytest.mark.parametrize("label", [c[0] for c in CASES])
def test_granularity_agrees_between_a_unit_and_a_manifest(units, label):
    """A reader holding only bytes computes the granularity a holder of the
    planes computes.  The serving plugin has the manifest; the exporter has
    the unit; a disagreement would be a shard one of them refuses to make.

    The third form is the one a loader actually holds -- a ``ParsedUnit``,
    which supplies its own superblock and arity.  It has to agree without
    being told them, because the defaults (256, arity 1) are wrong for a
    k-tuple grid and a caller who passed nothing would get a granularity that
    ``slice_unit`` then refuses."""
    unit, forests, grid, blob = units[label]
    parsed = parse_unit_artifact(blob, device=DEVICE)
    explicit = shard_granularity(
        parsed.unit, parsed.manifest.geometry.superblock_columns, grid.arity
    )
    assert explicit == shard_granularity(parsed.manifest)
    assert explicit == shard_granularity(parsed)


@needs_cuda
def test_can_shard_matches_what_slice_unit_accepts(units):
    """``can_shard`` is the question the plugin asks before it cuts; it must
    answer with the same rule ``slice_unit`` enforces."""
    for label, (unit, forests, grid, blob) in units.items():
        parsed = parse_unit_artifact(blob, device=DEVICE)
        geometry = parsed.manifest.geometry
        for axis in ("row", "column"):
            extent = geometry.rows if axis == "row" else geometry.columns
            for tp in (2, 4, 8, 16):
                allowed = can_shard(parsed.unit, tp, axis,
                                    geometry.superblock_columns, grid.arity)
                # A loader holds the parse, not the planes, and passes no
                # superblock or arity; it must get the same answer.
                assert can_shard(parsed, tp, axis) is allowed
                if extent % tp:
                    assert not allowed
                    continue
                lo, hi = _tp_ranges(extent, tp)[-1]
                kwargs = {"rows": (lo, hi)} if axis == "row" else {"cols": (lo, hi)}
                try:
                    slice_unit(parsed, **kwargs)
                except GrammarError:
                    assert not allowed, (label, axis, tp)
                else:
                    assert allowed, (label, axis, tp)


def _narrowed(unit, rows, width):
    """The same encoded unit restricted to its first ``width`` columns.

    Every plane a column restriction touches, and nothing re-encoded: this is
    the wire at a narrower width, which is what a capability query is asked
    about.  It is built by restriction rather than by ``encode_unit`` because
    the encoder refuses a width whose S6b groups cross rows (tessera#57) --
    which is exactly the population the two answers disagreed on.
    """
    fields = dict(
        rates=unit.rates[:width],
        body_bits=unit.body_bits[:, :width].contiguous(),
        scale_base=unit.scale_base[: rows * width // unit.group].clone(),
        scale_refine=unit.scale_refine[: rows * width // unit.half].clone(),
    )
    for key in ("completion_bits", "anchors", "codes"):
        plane = getattr(unit, key)
        if plane is not None and plane.ndim == 2 and plane.numel():
            fields[key] = plane[:, :width].contiguous()
    return replace(unit, **fields)


def test_can_shard_is_exactly_what_the_slicer_accepts_at_every_block_width(cpu_units):
    """The predicate and the cutter answer from one rule, at every width.

    ``shard_granularity`` used to *raise the row granularity* for a unit whose
    columns are not a whole number of scale blocks, on the reasoning that a run
    of rows closes a straddling block -- so ``can_shard(unit, 2, "row")`` said
    yes for a 48-column S6b unit.  ``_slice_block_plane`` refuses every cut of
    such a unit, the identity slice included, because the plane is indexed
    ``(row * cols + col) // block`` and no rectangle of the weight is a run of
    it.  A loader that asks first and cuts second got ``True`` and then a
    ``GrammarError``, which is a capability answer it cannot act on.

    The widths come from the plane's own block and half --
    ``_scale_columns_per_row``, the number ``shard_granularity`` itself reads
    -- rather than from 48, and both kinds have to appear or the test says so.
    """
    unit, _forests, grid, _blob = cpu_units["s6b-tcq"]
    block = _scale_columns_per_row(unit)
    steps, columns = unit.body_bits.shape
    rows = steps * grid.arity
    widths = tuple(range(2 * unit.half, columns + 1, unit.half))
    assert {bool(w % block) for w in widths} == {True, False}, (
        "the roster must hold both a width the block tiles and one it does not")
    seen = set()
    for width in widths:
        narrow = _narrowed(unit, rows, width)
        for axis in ("row", "column"):
            extent = rows if axis == "row" else width
            for tp in (1, 2, 4):
                allowed = can_shard(narrow, tp, axis, 256, grid.arity)
                if extent % tp:
                    assert not allowed, (width, axis, tp)
                    continue
                lo, hi = _tp_ranges(extent, tp)[-1]
                kwargs = {"rows": (lo, hi)} if axis == "row" else {"cols": (lo, hi)}
                try:
                    slice_unit(narrow, arity=grid.arity, code=CODE, **kwargs)
                except GrammarError:
                    assert not allowed, (width, axis, tp)
                    seen.add(("refused", bool(width % block)))
                else:
                    assert allowed, (width, axis, tp)
                    seen.add(("cut", bool(width % block)))
    assert ("refused", True) in seen, "a straddling width must be refused by both"
    assert ("cut", False) in seen, "a tiled width must still cut, or nothing is tested"


def _declared_rotated(parsed):
    """The same wire, declared ``R_in``-only, parsed back.

    The rotation block is the one the reader derives from the width
    (``diagonals.rotation_block_for``) because that is the only block
    ``build_unit_artifact`` accepts (tessera#210) -- so this is a rotated
    artifact a loader can really be handed, not a hand-made unit.
    """
    from tessera.diagonals import rotation_block_for
    from tessera.manifest import RotationState

    columns = parsed.manifest.geometry.columns
    unit = replace(
        parsed.unit,
        rotation=RotationState.R_IN_ONLY,
        rotation_block=rotation_block_for(RotationState.R_IN_ONLY, columns),
    )
    _m, _r, blob = build_unit_artifact(
        unit, "rotated", parsed.forests, parsed.manifest.branch.root_q256,
        parsed.code or CODE, fixture_id=None,
    )
    return _cpu_parse(blob)


def test_can_shard_refuses_the_rotated_units_the_cutter_refuses(cpu_units):
    """tessera#304: capability is the cutter's answer over the *transform*
    domain too, not only over block geometry.

    ``slice_unit`` refuses every rotated unit -- ``R_in``-only rotation is a
    column-block structure a cut would break into pieces that decode to
    plausible wrong weights -- while ``can_shard`` inspected only straddling
    blocks and arithmetic granularity, so a producer or operator reading the
    capability API was promised a cut that always raises.  #235 aligned the two
    on block geometry; this is the population that alignment did not reach.

    Both bodies and both block plane kinds the CPU fixtures carry, both axes,
    every view a caller holds (bare unit, ``ParsedUnit``, ``Manifest``), and
    TP1 -- the identity slice, which is refused too -- as well as TP>1.
    """
    from tessera.manifest import RotationState

    cut = set()
    for label, (_unit, _forests, grid, blob) in cpu_units.items():
        parsed = _cpu_parse(blob)
        rotated = _declared_rotated(parsed)
        assert rotated.unit.rotation is RotationState.R_IN_ONLY
        assert rotated.manifest.branch.rotation is RotationState.R_IN_ONLY
        geometry = rotated.manifest.geometry
        for axis in ("row", "column"):
            extent = geometry.rows if axis == "row" else geometry.columns
            for tp in (1, 2, 4):
                assert can_shard(rotated, tp, axis) is False, (label, axis, tp)
                assert can_shard(rotated.manifest, tp, axis) is False, (
                    label, axis, tp)
                assert can_shard(
                    rotated.unit, tp, axis, geometry.superblock_columns,
                    grid.arity) is False, (label, axis, tp)
                lo, hi = _tp_ranges(extent, tp)[-1]
                kwargs = {"rows": (lo, hi)} if axis == "row" else {"cols": (lo, hi)}
                with pytest.raises(GrammarError, match="refusing to slice a rotated"):
                    slice_unit(rotated, **kwargs)
                # The control: unrotated, the same wire still cuts, and
                # ``can_shard`` still says so.
                if can_shard(parsed, tp, axis):
                    slice_unit(parsed, **kwargs)
                    cut.add((label, axis, tp))
    assert cut, "no unrotated cut was accepted, so the refusal proves nothing"


def test_the_rotation_refusal_has_one_home_over_every_view():
    """The issue's own reproduction, on committed bytes and no encoder: a
    ``R_in``-only E4M3 window artifact whose dimensions are otherwise
    shardable.  ``can_shard`` answered ``True`` for both axes and ``slice_unit``
    raised.  One predicate now answers both, so the refusal a loader is given
    is the sentence the cutter would have raised."""
    import pathlib

    from tessera.slicing import _unsliceable_reason

    legacy = pathlib.Path(__file__).parent / "data" / "legacy"
    parsed = _cpu_parse(
        (legacy / "e4m3-1024-window-channel-256c.tessera").read_bytes())
    rotated = _declared_rotated(parsed)
    assert can_shard(rotated, 2, "row") is False
    assert can_shard(rotated.manifest, 2, "column") is False
    with pytest.raises(GrammarError) as raised:
        slice_unit(rotated, rows=(8, 16))
    reason = _unsliceable_reason(
        rotated.unit.rotation, _scale_columns_per_row(rotated.unit),
        rotated.manifest.geometry.columns,
    )
    assert reason and str(raised.value) == reason
    # The predicate is the whole of the disagreement: the same bytes without
    # the declared rotation have no reason and cut.
    assert _unsliceable_reason(
        parsed.unit.rotation, _scale_columns_per_row(parsed.unit),
        parsed.manifest.geometry.columns) is None
    assert can_shard(parsed, 2, "row") is True
    slice_unit(parsed, rows=(8, 16))


def test_plane_order_is_the_only_place_the_two_orders_live():
    """The shard order is the canonical order with one plane wedged in ahead
    of BODY -- the only position a legal truncation cannot separate them.
    True of both layouts: the minor-7 pair and the minor 0-6 pair it reads."""
    from tessera.planes import (
        LEGACY_PLANE_ORDER, LEGACY_SHARD_PLANE_ORDER, PlaneLayout,
    )

    for layout, whole, shard in (
        (PlaneLayout.LADDER, CANONICAL_PLANE_ORDER, SHARD_PLANE_ORDER),
        (PlaneLayout.LEGACY, LEGACY_PLANE_ORDER, LEGACY_SHARD_PLANE_ORDER),
    ):
        assert plane_order(False, layout) is whole
        assert plane_order(True, layout) is shard
        assert PlaneKind.INITIAL_STATE not in whole
        assert [k for k in shard if k is not PlaneKind.INITIAL_STATE] == list(whole)
        body = shard.index(PlaneKind.BODY)
        assert shard.index(PlaneKind.INITIAL_STATE) == body - 1
