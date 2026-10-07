"""Served paired E2M1: actual bytes, exact prices and independent CPU replay.

The oracle reads MSB-first body fields, tuple-code bytes and scale nibbles
itself. It deliberately calls neither the CUDA compact repacker nor the
materialising reader's unpack/replay/scale helpers.
"""
from __future__ import annotations

from functools import lru_cache
import pytest
import torch

from tessera.alphabet import E2M1_GRID, tuple_grid
from tessera.compact_prep import parse_compact_wire
from tessera.container import parse, plane_ranges
from tessera.control import unit_wire_bits
from tessera.export import encode_linear, served_recipe
from tessera.manifest import BodyKind, ScalePlaneKind
from tessera.planes import PlaneKind
from tessera.structure import STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE
from tessera.unit_artifact import read_unit_artifact

GRID = tuple_grid(E2M1_GRID, 2)
ROWS, COLS = 32, 32


def _fields(data, widths):
    """Independent bit reader: consecutive fields, most significant bit first."""
    cursor = 0
    for width in widths:
        value = 0
        for _ in range(width):
            value = (value << 1) | ((data[cursor // 8] >> (7 - cursor % 8)) & 1)
            cursor += 1
        yield value


def _ue4m3(byte):
    assert 0 <= byte < 127
    exponent, mantissa = byte >> 3, byte & 7
    return mantissa * 2.0 ** -9 if exponent == 0 else (1 + mantissa / 8) * 2.0 ** (exponent - 7)


def decode_window_bytes(blob):
    """Reconstruct this transform-free paired LUT wire from serialized bytes."""
    artifact = parse(blob)
    manifest = artifact.manifest
    assert (manifest.body, manifest.span, manifest.scale_plane.kind) == (
        BodyKind.WINDOW, 1, ScalePlaneKind.LUT)
    rows, cols = manifest.geometry.rows, manifest.geometry.columns
    half = manifest.geometry.half_weights
    chunks = {d.kind: artifact.plane_region[offset:offset + content]
              for d, offset, content, _total in plane_ranges(manifest, artifact.terminal)}
    steps = rows // 2
    initial = [0] * cols
    if manifest.shard is not None and manifest.shard.has_initial_state:
        initial = list(_fields(chunks[PlaneKind.INITIAL_STATE],
                               [manifest.shard.state_bits] * cols))
    fields = iter(_fields(chunks[PlaneKind.BODY],
                          [rate for rate in manifest.rates for _ in range(steps)]))
    nibbles = list(_fields(chunks[PlaneKind.SCALE_REFINE], [4] * (rows * cols // half)))
    table = chunks[PlaneKind.ALPHABET]
    assert len(table) == 1 << manifest.window_bits
    lut = [_ue4m3(byte) for byte in manifest.scale_plane.table]
    # Hardware E2M1 magnitude codes, with bit 3 the sign (including -0).
    magnitudes = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    scales = torch.tensor([lut[index] for index in nibbles], dtype=torch.float32)
    scales = scales.reshape(rows, cols // half) * float(manifest.scale_plane.global_scale)
    out = torch.empty(rows, cols, dtype=torch.float32)
    mask = (1 << manifest.window_bits) - 1
    for column, rate in enumerate(manifest.rates):
        state = initial[column]
        for step in range(steps):
            state = ((state << rate) | next(fields)) & mask
            code = table[state]
            for member, nibble in enumerate((code >> 4, code & 15)):
                value = -magnitudes[nibble & 7] if nibble & 8 else magnitudes[nibble]
                row = 2 * step + member
                out[row, column] = value * scales[row, column // half]
    return out

@lru_cache(maxsize=None)
def _encode(q256, structure, rows=ROWS, cols=COLS):
    recipe = served_recipe(GRID, q256, structure)
    weight = torch.linspace(-0.25, 0.375, rows * cols).reshape(rows, cols)
    return encode_linear(weight, grid=GRID, q256=q256, name="served", verify=False,
                         body=recipe.body, span=recipe.span, scale_plane=recipe.scale_plane,
                         window_bits=recipe.window_bits, window_seed=recipe.window_seed,
                         window_sigma=recipe.window_sigma, channel_sigma=recipe.channel_sigma)


@pytest.mark.parametrize("q256", range(128, 1025, 128))
@pytest.mark.parametrize("structure", [STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE])
def test_served_pure_classes_are_l14_bytes_with_exact_prices_and_reconstruction(q256, structure):
    encoded = _encode(q256, structure)
    artifact = parse(encoded.blob)
    manifest = artifact.manifest
    assert (manifest.body, manifest.span, manifest.scale_plane.kind, manifest.window_bits) == (
        BodyKind.WINDOW, 1, ScalePlaneKind.LUT, 14)
    assert set(manifest.rates) == {q256 // 128}
    chunks = {d.kind: content for d, _offset, content, _total in plane_ranges(manifest, artifact.terminal)}
    assert chunks[PlaneKind.ALPHABET] == 16384
    assert chunks[PlaneKind.BODY] == ROWS * COLS * (q256 // 128) // 16
    assert chunks[PlaneKind.SCALE_REFINE] == ROWS * COLS // 32
    assert chunks[PlaneKind.DESCENDANT] == chunks[PlaneKind.COMPLETION] == 0
    assert encoded.exact_bytes * 8 == unit_wire_bits(GRID, q256, ROWS, COLS)
    assert len(encoded.blob) > encoded.exact_bytes  # Header, manifest and LUT are not free.
    from tessera.footprint import account_terminal
    report = account_terminal(manifest, artifact.terminal,
        side_bytes=len(encoded.blob) - len(artifact.plane_region),
        physical_bytes=len(artifact.plane_region))
    assert report.wire_bpp * (ROWS * COLS) == len(encoded.blob) * 8
    assert torch.equal(decode_window_bytes(encoded.blob), read_unit_artifact(encoded.blob))
    wire = parse_compact_wire(encoded.blob, device="cpu")
    assert (wire.metadata.grid.arity, wire.metadata.span, wire.metadata.manifest.window_bits,
            wire.metadata.manifest.geometry.half_weights) == (2, 1, 14, 16)
    assert wire.blob_len == len(encoded.blob)


@pytest.mark.parametrize("q256", [128, 896, 1024])
def test_dense_and_routed_write_identical_served_bytes(q256):
    assert _encode(q256, STRUCTURE_DENSE).blob == _encode(q256, STRUCTURE_ROUTED_MOE).blob


@pytest.mark.parametrize("q256", [128, 896, 1024])
@pytest.mark.parametrize("cut", [{"rows": (16, 32)}, {"cols": (0, 16)}], ids=["row", "column"])
def test_serialized_rank_cuts_replay_the_parent_values(q256, cut):
    from tessera.slicing import slice_unit
    from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact
    encoded = _encode(q256, STRUCTURE_DENSE)
    parsed = parse_unit_artifact(encoded.blob)
    shard = slice_unit(parsed, **cut)
    manifest, _region, blob = build_unit_artifact(
        shard, "rank", parsed.forests, parsed.manifest.branch.root_q256, fixture_id=None)
    if "rows" in cut:
        assert manifest.shard.has_initial_state and manifest.shard.state_bits == 14
        assert bool(shard.initial_state.any())
    whole = decode_window_bytes(encoded.blob)
    (r0, r1), (c0, c1) = cut.get("rows", (0, ROWS)), cut.get("cols", (0, COLS))
    assert torch.equal(decode_window_bytes(blob), whole[r0:r1, c0:c1])
    assert torch.equal(read_unit_artifact(blob), whole[r0:r1, c0:c1])


@pytest.mark.parametrize("q256", range(128, 1025, 128))
def test_routed_unit_price_matches_cpu_reference_packed_tensors(q256):
    from window_pack_reference import pack_bitstream
    from tessera.unit_artifact import parse_unit_artifact
    from tessera.serving_parts import routed_window_unit_resident_bytes
    parsed = parse_unit_artifact(_encode(q256, STRUCTURE_ROUTED_MOE).blob)
    unit = parsed.unit
    rep = pack_bitstream(unit.body_bits, tuple(unit.rates))
    scale = unit.scale_refine.reshape(ROWS, COLS // 16).T.contiguous()
    plane = ((scale[:, 0::2] << 4) | scale[:, 1::2]).to(torch.uint8)
    tensors = [rep.words, rep.runs, rep.perm, unit.window_codes.to(torch.uint8),
               torch.zeros(COLS, dtype=torch.int32), plane, unit.scale_lut,
               torch.tensor([unit.scale_global], dtype=torch.float32),
               torch.zeros(4, dtype=torch.int32)]
    actual = sum(t.numel() * t.element_size() for t in tensors)
    assert routed_window_unit_resident_bytes("TESSERA_NVFP4", ROWS, COLS, unit.rates,
        window_bits=14, tile_rows=512) == actual


def test_joined_export_owner_writes_the_complete_served_recipe():
    from tessera.export_serving import fresh_joined_encode
    from tessera.fused import parse_fused
    weight = torch.linspace(-0.25, 0.375, ROWS * COLS).reshape(ROWS, COLS)
    members = [({"tensor": f"expert.{expert}.gate_proj.weight", "stack": "experts",
                 "projection": "gate_proj", "source_layout": "unpacked_per_expert",
                 "rows": ROWS, "cols": COLS}, weight) for expert in range(2)]
    observed = []
    encoded = fresh_joined_encode(members, stack_plan={"experts": {"grid": GRID, "q256": 1024}},
        activation=None, device="cpu", no_verify=True, batch_observed=observed)
    assert observed == [2]
    for exported, framed, _payload, _global, manifest in encoded:
        assert manifest.window_bits == 14 and manifest.body is BodyKind.WINDOW
        assert parse_fused(framed)[0].blob == exported.blob
        assert torch.equal(decode_window_bytes(exported.blob), read_unit_artifact(exported.blob))


def test_fused_routed_launch_price_counts_actual_native_table_tensors():
    from types import SimpleNamespace
    from window_pack_reference import pack_bitstream
    from tessera.unit_artifact import parse_unit_artifact
    from tessera.routed_fused_e2m1 import projection_tables
    from tessera.serving_parts import routed_fused_unit_bytes
    parsed = parse_unit_artifact(_encode(1024, STRUCTURE_ROUTED_MOE, 256, 256).blob)
    rep = pack_bitstream(parsed.unit.body_bits, tuple(parsed.unit.rates))
    bundle = SimpleNamespace(experts=2, cols=256,
        perm_all=rep.perm.reshape(1, -1).expand(2, -1).contiguous(),
        runs_all=rep.runs.reshape(1, -1, 4).expand(2, -1, -1).contiguous())
    runs, desc, _tile_words, _slot_words = projection_tables(bundle)
    ratio = torch.full((2,), float(parsed.unit.scale_global), dtype=torch.float32)
    actual = sum(t.numel() * t.element_size() for t in (runs, desc, ratio))
    assert 2 * routed_fused_unit_bytes(14, 256, family="TESSERA_NVFP4") == actual
