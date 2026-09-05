"""Schema minor 7 (tessera#144) against the bytes that predate it.

Two hard constraints from the wire change, each proved on bytes and not on a
round trip through code that shares the change:

* **Existing artifacts still read.**  ``tests/data/legacy`` holds eleven
  artifacts written by master ``da2b371`` -- the last tree before minor 7 --
  with a sidecar recording the tensor that tree decoded each to.  The current
  reader must decode every one to the same tensor, under the same minor and
  the same plane order.
* **The legacy writer is exact.**  ``layout=PlaneLayout.LEGACY`` with the
  identity that tree stamped must reproduce every one of those blobs byte for
  byte; otherwise a test "building an old artifact" would only prove the
  reader agrees with the writer it was changed alongside.

Plus the packing the minor introduces (``wire.pack_levels``) and the one
fact worth stating about the default recipe table: an empty completion axis
keeps its plane bytes across the minor.
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from tessera.alphabet import SERIALISABLE_GRIDS
from tessera.container import SCHEMA_MINOR, parse
from tessera.encode import encode_unit
from tessera.errors import GrammarError
from tessera.export import (
    DEFAULT_CODE,
    DEFAULT_GROUP,
    DEFAULT_HALF,
    DEFAULT_SCALE_REFIT,
    _plan_for,
    encode_linear_planes,
    wire_recipe,
)
from tessera.grammar import completion_level_counts
from tessera.manifest import ScalePlaneKind
from tessera.planes import PlaneKind, PlaneLayout
from tessera.slicing import slice_unit
from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact, read_unit_artifact
from tessera.wire import pack_body, pack_levels, unpack_body, unpack_levels

LEGACY = Path(__file__).parent / "data" / "legacy"
META = json.loads((LEGACY / "manifest.json").read_text())
GRIDS = {g.name: g for g in SERIALISABLE_GRIDS.values()}


def _weight(rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=g) * 0.02


def _tensor_digest(tensor):
    return hashlib.sha256(tensor.contiguous().float().numpy().tobytes()).hexdigest()


# --- existing artifacts still read -------------------------------------------


@pytest.mark.parametrize("label", sorted(META["cases"]))
def test_a_pre_change_artifact_reads_to_the_tensor_its_writer_decoded(label):
    case = META["cases"][label]
    blob = (LEGACY / f"{label}.tessera").read_bytes()
    assert hashlib.sha256(blob).hexdigest() == case["blob_sha256"]
    assert blob[10] == case["header_minor"] < 7
    art = parse(blob)
    assert art.manifest.layout is PlaneLayout.LEGACY
    assert art.manifest.schema_minor == case["schema_minor"]
    assert [k.name for k in art.manifest.plane_order] == case["plane_order"]
    assert list(art.terminal.plane_elements) == case["plane_elements"]
    assert list(art.manifest.plane(PlaneKind.COMPLETION).counts) == case["completion_counts"]
    decoded = read_unit_artifact(blob)
    assert list(decoded.shape) == case["decoded_shape"]
    assert _tensor_digest(decoded) == case["decoded_sha256"]


def test_the_legacy_set_covers_the_planes_and_bodies_the_minor_touches():
    """The blobs are not a roster to trust: say what they span."""
    cases = META["cases"]
    labels = set(cases)
    assert any(c["completion_counts"] and sum(c["completion_counts"]) for c in cases.values())
    assert any(len(c["plane_elements"]) == 10 for c in cases.values()), "a shard"
    assert any("release" in label for label in labels)
    assert any("s6b" in label for label in labels)
    assert any("diag" in label for label in labels)
    assert {"e2m1", "e2m1x2", "e4m3", "bf16"} <= {label.split("-")[0] for label in labels}


# --- the legacy writer is exact ----------------------------------------------


def _legacy_builders():
    """The eleven encodes of ``tests/data/legacy/generate_legacy_blobs.py``,
    as (label, build) where build(layout, fixture_id) -> blob.  The encodes
    are deterministic on CPU (fixed generator seeds, no torch RNG)."""
    old = bytes.fromhex(META["encoder_fixture_id"])

    def artifact(label, unit, forests, code=DEFAULT_CODE):
        q256 = parse((LEGACY / f"{label}.tessera").read_bytes()).manifest.branch.root_q256
        return build_unit_artifact(
            unit, "u", forests, q256, code, fixture_id=old, layout=PlaneLayout.LEGACY,
        )[2]

    def planes(label, rows, cols, seed, grid, q256, **kwargs):
        _e, unit, forests = encode_linear_planes(
            _weight(rows, cols, seed), grid=GRIDS[grid], q256=q256, name="u", **kwargs
        )
        return artifact(label, unit, forests)

    yield "e2m1-256-cfull-lut-512c", lambda: planes("e2m1-256-cfull-lut-512c", 16, 512, 1, "E2M1", 256, completion=None)
    yield "e2m1-256-c1-lut-512c", lambda: planes("e2m1-256-c1-lut-512c", 16, 512, 2, "E2M1", 256, completion=1)
    yield "e2m1-256-cfull-s6b-512c", lambda: planes(
        "e2m1-256-cfull-s6b-512c", 16, 512, 3, "E2M1", 256, completion=None,
        scale_plane=ScalePlaneKind.S6B,
    )
    yield "e2m1-256-cfull-lut-diag-256c", lambda: planes(
        "e2m1-256-cfull-lut-diag-256c", 16, 256, 4, "E2M1", 256, completion=None,
        with_diagonals=True,
    )
    yield "e2m1x2-896-tcq-lut-256c", lambda: planes("e2m1x2-896-tcq-lut-256c", 16, 256, 5, "E2M1x2", 896)
    yield "e2m1x2-640-window-lut-256c", lambda: planes("e2m1x2-640-window-lut-256c", 16, 256, 6, "E2M1x2", 640)
    yield "e4m3-1024-window-channel-256c", lambda: planes("e4m3-1024-window-channel-256c", 16, 256, 7, "E4M3", 1024)
    yield "bf16-1024-window-channel-256c", lambda: planes("bf16-1024-window-channel-256c", 16, 256, 8, "BF16", 1024)

    def shard_pair():
        parent = planes("e2m1-256-cfull-lut-512c-parent", 16, 512, 9, "E2M1", 256, completion=None)
        parsed = parse_unit_artifact(parent)
        shard = slice_unit(parsed, rows=(8, 16))
        return parent, artifact("e2m1-256-cfull-lut-512c-shard-r8-16", shard, parsed.forests, parsed.code)

    yield "e2m1-256-cfull-lut-512c-parent", lambda: shard_pair()[0]
    yield "e2m1-256-cfull-lut-512c-shard-r8-16", lambda: shard_pair()[1]

    def released():
        grid, q256 = GRIDS["E2M1"], 768
        recipe = wire_recipe(grid, q256)
        rates, forests = _plan_for(grid, q256, 256, recipe.body, None)
        unit = encode_unit(
            _weight(16, 256, 10), forests, rates, DEFAULT_CODE, completion=0,
            released_positions=256, group=DEFAULT_GROUP, half=DEFAULT_HALF,
            scale_refit=DEFAULT_SCALE_REFIT, span=recipe.span,
            scale_plane=recipe.scale_plane, trellis_weighting="scale",
            body=recipe.body, window_bits=recipe.window_bits,
            window_seed=recipe.window_seed, window_sigma=recipe.window_sigma,
            channel_sigma=recipe.channel_sigma,
        )
        return artifact("e2m1-768-release256-256c", unit, forests)

    yield "e2m1-768-release256-256c", released


@pytest.mark.parametrize("label,build", list(_legacy_builders()))
def test_the_legacy_writer_reproduces_the_pre_change_bytes(label, build):
    assert label in META["cases"]
    assert build() == (LEGACY / f"{label}.tessera").read_bytes()


# --- the packing -------------------------------------------------------------


def _words(steps, widths, seed):
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(steps, len(widths), dtype=torch.long)
    for j, width in enumerate(widths):
        out[:, j] = torch.randint(0, 1 << width, (steps,), generator=g)
    return out


def test_level_major_packing_round_trips_and_each_level_is_a_byte_prefix():
    steps, widths = 16, (2, 0, 1, 2, 2, 1, 0, 2)
    words = _words(steps, widths, 0)
    packed = pack_levels(words, widths)
    assert torch.equal(unpack_levels(packed, widths, steps), words)
    counts = completion_level_counts(widths, steps)
    assert len(packed) * 8 >= sum(counts) > 0
    running = 0
    for level, count in enumerate(counts, start=1):
        running += count
        assert running % 8 == 0
        narrowed = tuple(min(level, w) for w in widths)
        prefix = packed[: running // 8]
        assert torch.equal(
            unpack_levels(prefix, narrowed, steps),
            words >> torch.tensor([w - n for w, n in zip(widths, narrowed)]),
        )
        assert prefix == pack_levels(
            words >> torch.tensor([w - n for w, n in zip(widths, narrowed)]), narrowed
        )


def test_level_major_and_per_position_packing_coincide_at_one_level():
    """At depth 1 there is one bit per position either way, column-major in
    both -- the one place the two packings must agree, and do."""
    steps, widths = 8, (1, 0, 1, 1, 0, 1)
    words = _words(steps, widths, 1)
    assert pack_levels(words, widths) == pack_body(words, widths)
    assert torch.equal(
        unpack_levels(pack_levels(words, widths), widths, steps),
        unpack_body(pack_body(words, widths), widths, steps).long(),
    )


def test_level_major_packing_differs_from_per_position_packing_at_depth_two():
    steps, widths = 8, (2,) * 4
    words = _words(steps, widths, 2)
    assert pack_levels(words, widths) != pack_body(words, widths)


def test_level_major_packing_refuses_a_word_out_of_its_columns_width():
    words = torch.tensor([[3, 1]])
    with pytest.raises(GrammarError, match="out of range"):
        pack_levels(words, (1, 1))
    with pytest.raises(GrammarError, match="COMPLETION needs"):
        unpack_levels(b"", (1, 1), 4)
    with pytest.raises(GrammarError, match="canonical"):
        unpack_levels(b"\xff", (1,), 4)


def test_unpack_levels_reads_an_empty_plane_at_depth_zero():
    assert unpack_levels(b"", (0, 0, 0), 5).shape == (5, 3)
    assert pack_levels(torch.zeros(5, 3, dtype=torch.long), (0, 0, 0)) == b""


# --- the default recipe table across the minor -------------------------------


def test_an_empty_completion_axis_keeps_its_plane_bytes_across_the_minor():
    """Every unit today's recipe table writes has an empty COMPLETION plane,
    so its plane region is byte-identical in either layout; only the
    manifest's descriptor order and the terminal's count array move -- which
    is exactly what moves the encoder identity (``encoder_identity``)."""
    for grid, q256 in (("E4M3", 1024), ("E2M1x2", 896), ("E2M1x2", 640), ("BF16", 1024)):
        _e, unit, forests = encode_linear_planes(
            _weight(16, 256, 11), grid=GRIDS[grid], q256=q256, name="u"
        )
        q = parse(_e.blob).manifest.branch.root_q256
        ladder = build_unit_artifact(unit, "u", forests, q, fixture_id=None)
        legacy = build_unit_artifact(unit, "u", forests, q, fixture_id=None, layout=PlaneLayout.LEGACY)
        assert ladder[1] == legacy[1], grid
        assert ladder[0].payload_digest == legacy[0].payload_digest
        assert ladder[0].schema_minor == SCHEMA_MINOR and legacy[0].schema_minor < 7
        assert [p.kind for p in ladder[0].planes] != [p.kind for p in legacy[0].planes]
        assert sorted(ladder[0].terminals[0].plane_elements) == sorted(legacy[0].terminals[0].plane_elements)
        assert ladder[0].terminals[0].plane_elements != legacy[0].terminals[0].plane_elements
        assert torch.equal(read_unit_artifact(ladder[2]), read_unit_artifact(legacy[2]))
