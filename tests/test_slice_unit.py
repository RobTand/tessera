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
  * nothing at offset 0 moved: the artifacts HEAD writes are byte-identical,
    and the identity slice of any unit is that unit;
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
from types import SimpleNamespace

import pytest
import torch

from tessera.alphabet import SERIALISABLE_GRIDS
from tessera.decode import (
    decode_codes,
    decode_codes_mixed,
    reconstruct_unit,
    release_order,
    replay_body,
    replay_window,
)
from tessera.encode import _canonical_release_order, encode_unit
from tessera.grammar import release_quota, superblock_count
from tessera.errors import GrammarError, ManifestError
from tessera.export import _plan_for, tcq_cap_q256, wire_recipe
from tessera.layout import (
    SlicedUnit,
    _slice_release,
    can_shard,
    shard_granularity,
    slice_unit,
)
from tessera.manifest import BodyKind, ScalePlaneKind, ShardOrigin
from tessera.planes import (
    CANONICAL_PLANE_ORDER,
    SHARD_PLANE_ORDER,
    PlaneKind,
    plane_order,
)
from tessera.trellis import TCQ, ConvCode
from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact

CODE = ConvCode(memory=6)
GRIDS = {g.name: g for g in SERIALISABLE_GRIDS.values()}

#: The bytes HEAD (3d419e7) writes for three small units, one per shipping
#: wire recipe.  Recorded by encoding these exact weights against the HEAD
#: tree; the point of the constants is that schema minor 4 moved none of them.
HEAD_UNIT_DIGESTS = {
    "e4m3-window-channel": (
        "bca30ebbc1687d1a753525ceb1148edd1469504d2cd2c18cbf9aa5f052dd802c", 21159),
    "e2m1-tcq-lut-release": (
        "1840b3f9dfe3929d9aa86006f207ad427658135d2d5c95fa5dc0b6ad0e532f31", 8398),
    "e2m1x2-subcap-window-lut": (
        "ae0e675dc7b0fbf40a5cee88f22baf5bb74b650d07b6aaa30f6008a443196ec5", 8059),
}

#: The same, for the encoder-free artifact ``conftest.make_artifact`` builds --
#: the one that exercises the layout, manifest and container alone, which is
#: precisely the code this schema minor touches.
HEAD_LAYOUT_DIGESTS = {
    (512, 8, 32, 8):
        "cb39f35e686e8858485917c81ab4c53c3b1464d83559ac49dcbbda24fdd783a3",
    (640, 16, 64, 16):
        "0e88573aaa13af73d8fc13d8dab0a16b5f52257f70085493ee5f20bd30b73787",
}

#: The shipped Qwen3-0.6B E4M3 checkpoint: real units at the shipping wire.
GBFAM = pathlib.Path(
    "/home/rob/tessera-runs/gbfam/qwen3-0.6b-tessera-e4m3-reach-gridbook"
)

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


def _s6b_unit(device=DEVICE):
    """A unit over the S6b plane: the older default, still a legal wire."""
    grid = GRIDS["E2M1"]
    torch.manual_seed(11)
    weight = (torch.randn(32, 256) * 0.02).to(device)
    q256 = tcq_cap_q256(grid)
    rates, forests = _plan_for(grid, q256, 256, BodyKind.TCQ, None)
    unit = encode_unit(
        weight, forests, rates, CODE, completion=0, span=2,
        scale_plane=ScalePlaneKind.S6B, body=BodyKind.TCQ, scale_refit=2,
    )
    _m, _r, blob = build_unit_artifact(unit, "s6b", forests, q256 * grid.arity, CODE)
    return unit, forests, grid, blob


# ------------------------------------------------- nothing at offset 0 moved


@pytest.mark.parametrize("key,digest", sorted(HEAD_LAYOUT_DIGESTS.items()))
def test_layout_bytes_are_what_head_wrote(key, digest):
    """The encoder-free artifact is byte-identical across the schema bump.

    ``make_artifact`` builds a complete unit out of fixed payloads through
    ``build_planes`` / ``build_terminal`` / ``Manifest.encode`` / ``serialize``
    -- every file this minor touches and no encoder at all.  A minor that
    added a tenth plane descriptor, or a tenth entry to every count array,
    would move these hashes; this one does not, because a whole unit still
    writes the nine-plane canonical order it always wrote.
    """
    from conftest import make_artifact

    q256, rows, columns, superblock = key
    _m, _region, blob = make_artifact(
        q256=q256, rows=rows, columns=columns, superblock_columns=superblock
    )
    assert hashlib.sha256(blob).hexdigest() == digest


@needs_cuda
@pytest.mark.parametrize("label", sorted(HEAD_UNIT_DIGESTS))
def test_encoded_unit_bytes_are_what_head_wrote(label):
    """A real unit at each shipping recipe is byte-identical to HEAD's."""
    digest, size = HEAD_UNIT_DIGESTS[label]
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


@pytest.mark.skipif(not GBFAM.exists(), reason="the shipped checkpoint is not here")
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
            container=manifest.branch.container)
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
            manifest.branch.root_q256, CODE)
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
            assert back.manifest.schema_minor == 4
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
    monkeypatch.setenv("TESSERA_FUSED_REPLAY", fused)
    decode._fused_replay.cache_clear()
    decode._fused_decode.cache_clear()
    try:
        codes = decode_codes_mixed(shard, parsed.forests, parsed.code)
        full = decode_codes_mixed(parsed.unit, parsed.forests, parsed.code)
        assert torch.equal(codes, full[16:32])
    finally:
        decode._fused_replay.cache_clear()
        decode._fused_decode.cache_clear()


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


def test_plane_order_is_the_only_place_the_two_orders_live():
    """The shard order is the canonical order with one plane wedged in ahead
    of BODY -- the only position a legal truncation cannot separate them."""
    assert plane_order(False) is CANONICAL_PLANE_ORDER
    assert plane_order(True) is SHARD_PLANE_ORDER
    assert PlaneKind.INITIAL_STATE not in CANONICAL_PLANE_ORDER
    assert [k for k in SHARD_PLANE_ORDER if k is not PlaneKind.INITIAL_STATE] == list(
        CANONICAL_PLANE_ORDER
    )
    body = SHARD_PLANE_ORDER.index(PlaneKind.BODY)
    assert SHARD_PLANE_ORDER.index(PlaneKind.INITIAL_STATE) == body - 1
