"""The compact loader (``tessera.compact_prep``) against the materialising reader.

Two contracts, in the two halves this file is split along:

* **CPU -- metadata and refusals.**  ``parse_unit_metadata`` must accept
  exactly the bytes ``parse_unit_artifact`` accepts and refuse exactly the
  bytes it refuses, with the same words.  The legacy corpus is the shipped
  wire population (span-1/span-2 TCQ, LUT/S6b/CHANNEL planes, a row shard
  with INITIAL_STATE, a RELEASE unit, a diagonals unit), and the mutations
  walk truncations and byte flips through the header, the manifest and the
  plane region.

* **GPU -- the rank-local planes.**  ``prepare_span2_compact`` must be
  byte-equal to ``lane_planes.prepare_span2_planes`` on the shard
  ``sharding.shard_parsed_roles`` produces -- every key, ``torch.equal`` --
  whole unit and TP2/TP4 row and column cuts; and
  ``prepare_window_compact`` must be byte-equal to
  ``kernel_window_gemv.repack_window_body``'s tile-word layout with the
  sliced CHANNEL row scale, carrying the row cut's incoming window history as
  ``initial_state`` int32[cols] in original column order.

The actual A4 expert wires (``a4_export`` root) exercise the shipping
geometry when this box carries the checkpoint; the encoded units exercise the
same cuts everywhere else.  ``no multiple-of-16 rule`` is asserted directly:
M in {9, 15, 17} builds a window unit when the caller supplies the plan.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import box_artifacts

from tessera.errors import GrammarError

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="the compact plane packers are CUDA paths")

LEGACY = Path(__file__).resolve().parent / "data" / "legacy"
CORPUS = sorted(LEGACY.glob("*.tessera"))

A4_TENSOR = "model.language_model.layers.3.mlp.experts.0.{projection}.wire"


# ---------------------------------------------------------------------------
# CPU: metadata parity and identical refusals
# ---------------------------------------------------------------------------


def _facts_from_parsed(parsed) -> dict:
    geometry = parsed.manifest.geometry
    root = int(parsed.manifest.branch.root_q256)
    return {
        "grid": parsed.grid.name,
        "body": parsed.body.name,
        "plane": parsed.manifest.scale_plane.kind.name,
        "q256": root // parsed.grid.arity,
        "rows": int(geometry.rows),
        "columns": int(geometry.columns),
        "span": int(parsed.manifest.span),
    }


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: p.stem)
def test_metadata_agrees_with_the_materialising_reader(path):
    """The compact reader's facts are the reference reader's facts, on every
    shipped legacy wire -- including the shard's INITIAL_STATE and the
    RELEASE unit's override count."""
    from tessera.unit_artifact import parse_unit_artifact, parse_unit_metadata

    blob = path.read_bytes()
    parsed = parse_unit_artifact(blob)
    metadata = parse_unit_metadata(blob)
    assert metadata.manifest.manifest_digest() == parsed.manifest.manifest_digest()
    assert metadata.role_facts() == _facts_from_parsed(parsed)
    assert metadata.rates == tuple(int(r) for r in parsed.unit.rates)
    assert metadata.n_released == int(parsed.unit.release_index.numel())
    if parsed.manifest.shard is not None and parsed.manifest.shard.has_initial_state:
        assert metadata.shard_state is not None
        assert torch.equal(metadata.shard_state, parsed.unit.initial_state)
    else:
        assert metadata.shard_state is None
    if parsed.body.name == "TCQ":
        assert metadata.completion_limit == parsed.unit.completion_limit


def _refusal(fn, blob):
    try:
        fn(blob)
    except Exception as exc:  # noqa: BLE001 -- the refusal IS the value
        return (type(exc).__name__, str(exc))
    return None


def _mutations(blob: bytes):
    """Truncations through every region and flips at the first, middle and
    last plane bytes, plus manifest and header bytes."""
    out = []
    for cut in (0, 8, 12, 24, 25, len(blob) // 2, len(blob) - 1):
        if cut < len(blob):
            out.append(blob[:cut])
    for offset in (0, 12, 20, 28, len(blob) // 2, len(blob) - 1):
        mutated = bytearray(blob)
        mutated[offset] ^= 0x01
        out.append(bytes(mutated))
    return out


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: p.stem)
def test_mutated_wires_are_refused_identically(path):
    """One validation path: whatever the reference reader says about a
    mutated wire -- accept or refuse, and in what words -- the compact reader
    says exactly the same."""
    from tessera.unit_artifact import parse_unit_artifact, parse_unit_metadata

    blob = path.read_bytes()
    for mutated in _mutations(blob):
        got = _refusal(parse_unit_metadata, mutated)
        want = _refusal(parse_unit_artifact, mutated)
        assert got == want, (len(mutated), got, want)


def _legacy(name: str) -> bytes:
    return (LEGACY / name).read_bytes()


@pytest.mark.parametrize(
    "name, message",
    [
        # A RELEASE unit: the span-2 lane reads no RELEASE plane.
        ("e2m1-768-release256-256c.tessera", "released positions"),
        # A diagonals unit: the rank-1 pair is outside the dot product.
        ("e2m1-256-cfull-lut-diag-256c.tessera", "carries diagonals"),
        # A span-1 unit: not this path's body.
        ("e2m1-256-cfull-s6b-512c.tessera", ""),
    ],
)
def test_compact_span2_refusals_are_the_packers(name, message):
    """The compact span-2 path refuses the same units, in the same words,
    before any device work -- so the refusal is testable without a GPU."""
    from tessera.compact_prep import parse_compact_wire, prepare_span2_compact
    from tessera.unit_artifact import parse_unit_artifact

    blob = _legacy(name)
    parsed = parse_unit_artifact(blob)
    wire = parse_compact_wire(blob, device="cpu")
    with pytest.raises(GrammarError) as info:
        prepare_span2_compact(wire, device="cpu")
    if message:
        assert message in str(info.value)
        return
    # The span-1 S6b unit: the body check owns it, whatever the lane.
    assert "span-2" in str(info.value) or "span 1" in str(info.value)


def test_compact_span2_refusal_matches_the_packer_on_the_release_unit():
    """The exact sentence ``prepare_span2_planes`` raises for the RELEASE
    unit, from the compact path's own call."""
    from tessera import lane_planes as lp
    from tessera.compact_prep import parse_compact_wire, prepare_span2_compact

    blob = _legacy("e2m1-768-release256-256c.tessera")
    wire = parse_compact_wire(blob, device="cpu")
    with pytest.raises(GrammarError) as compact:
        prepare_span2_compact(wire, device="cpu")
    with pytest.raises(GrammarError) as packed:
        lp.require_no_post_decode_transforms(
            release_positions=wire.metadata.release_positions,
            diagonals=wire.metadata.has_diagonals, rotation=wire.metadata.rotation)
    assert str(compact.value) == str(packed.value)


# ---------------------------------------------------------------------------
# GPU: span-2 packed planes, byte-equal to the materialising packer
# ---------------------------------------------------------------------------


def _row_plan(rows, columns, rank, tp, name="w"):
    from tessera.serving.sharding import plan_shard

    return plan_shard("m", roles=[(name, rows)], columns=columns,
                      out_partitions=[rows // tp], in_size=columns,
                      tp_rank=rank, tp_size=tp, input_size=columns, output_size=rows)


def _col_plan(rows, columns, rank, tp, name="w"):
    from tessera.serving.sharding import plan_shard

    return plan_shard("m", roles=[(name, rows)], columns=columns,
                      out_partitions=[rows], in_size=columns // tp,
                      tp_rank=rank, tp_size=tp, input_size=columns, output_size=rows)


def _old_span2(blob, plan, name, device="cuda"):
    from tessera.lane_planes import prepare_span2_planes
    from tessera.serving.sharding import shard_parsed_roles
    from tessera.unit_artifact import parse_unit_artifact

    parsed = parse_unit_artifact(blob, device=device)
    roles = shard_parsed_roles([(name, parsed)], plan)
    return prepare_span2_planes(roles[0][1], device=device)


def _assert_planes_equal(got: dict, want: dict):
    assert set(got) == set(want)
    for key, value in want.items():
        if torch.is_tensor(value):
            assert torch.equal(got[key], value), key
        else:
            assert got[key] == value, key


def _cut_kwargs(plan):
    from tessera.serving.sharding import AXIS_ROWS

    role = plan.roles[0]
    if plan.axis == AXIS_ROWS:
        return {"rows": (role.lo, role.hi)}
    return {"cols": (role.lo, role.hi)}


@cuda
@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("axis", ["row", "column"])
def test_encoded_span2_unit_matches_the_packer_at_every_rank(axis, tp):
    """A freshly encoded E2M1x2 unit (q256 896, the shipping rung): every
    rank's compact planes equal ``prepare_span2_planes`` on the shard the
    route would have cut."""
    from tessera.alphabet import E2M1_GRID, tuple_grid
    from tessera.compact_prep import parse_compact_wire, prepare_span2_compact
    from tessera.export import encode_linear_planes

    rows, cols = 128, 512
    torch.manual_seed(1000 + tp)
    weight = (torch.randn(rows, cols, device="cuda") * 0.02).contiguous()
    exported, _unit, _forests = encode_linear_planes(
        weight, grid=tuple_grid(E2M1_GRID, 2), q256=896, name="compact-me",
        verify=False)
    blob = exported.blob
    wire = parse_compact_wire(blob, device="cuda", name="w")
    for rank in range(tp):
        plan = (_row_plan if axis == "row" else _col_plan)(rows, cols, rank, tp)
        got = prepare_span2_compact(wire, device="cuda", **_cut_kwargs(plan))
        want = _old_span2(blob, plan, "w")
        _assert_planes_equal(got, want)


def _old_whole(blob, name, device="cuda"):
    from tessera.lane_planes import prepare_span2_planes
    from tessera.unit_artifact import parse_unit_artifact

    return prepare_span2_planes(parse_unit_artifact(blob, device=device), device=device)


@cuda
def test_encoded_span2_unit_matches_the_packer_whole():
    """The whole unit, no cut: the rank-0 identity case (no INITIAL_STATE)."""
    from tessera.alphabet import E2M1_GRID, tuple_grid
    from tessera.compact_prep import parse_compact_wire, prepare_span2_compact
    from tessera.export import encode_linear_planes

    rows, cols = 128, 512
    torch.manual_seed(7)
    weight = (torch.randn(rows, cols, device="cuda") * 0.02).contiguous()
    exported, _unit, _forests = encode_linear_planes(
        weight, grid=tuple_grid(E2M1_GRID, 2), q256=896, name="compact-me",
        verify=False)
    wire = parse_compact_wire(exported.blob, device="cuda", name="w")
    _assert_planes_equal(
        prepare_span2_compact(wire, device="cuda"),
        _old_whole(exported.blob, "w"))


def _a4_wire_blob(projection: str) -> bytes:
    """One actual expert wire out of the routed A4 export."""
    index_path = box_artifacts.skip_now("a4_export", "model.safetensors.index.json")
    weight_map = json.loads(Path(index_path).read_text())["weight_map"]
    tensor = A4_TENSOR.format(projection=projection)
    shard = box_artifacts.skip_now("a4_export", weight_map[tensor])
    from safetensors import safe_open

    with safe_open(str(shard), framework="pt") as handle:
        return bytes(handle.get_tensor(tensor).detach().cpu().numpy().tobytes())


@cuda
@pytest.mark.parametrize(
    "projection, rows, columns, axis",
    [
        ("gate_proj", 2048, 4096, "row"),
        ("up_proj", 2048, 4096, "row"),
        ("down_proj", 4096, 2048, "column"),
    ],
)
@pytest.mark.parametrize("rank", [0, 1])
def test_actual_a4_wire_tp2_both_rank_shapes_are_byte_equal(
        projection, rows, columns, axis, rank):
    """The shipping A4 expert container, both TP2 rank shapes: the compact
    planes are the materialising packer's, byte for byte, and the wire's role
    facts are the sidecar's declaration.

    ``w13`` is cut on rows (1024 of 2048 per rank, each projection), ``w2``
    on columns (1024 of 2048); rank 1's row cut carries the trellis register
    in the select pad and is only equal if the packed-tail replay produced the
    parent's own state.
    """
    from tessera.compact_prep import parse_compact_wire, prepare_span2_compact
    from tessera.fused import parse_fused

    blob = _a4_wire_blob(projection)
    members = parse_fused(blob)
    assert len(members) == 1 and members[0].name == projection
    wire = parse_compact_wire(members[0].blob, device="cuda", name=projection)
    assert wire.role_facts == {
        "grid": "E2M1x2", "body": "TCQ", "plane": "LUT", "q256": 896,
        "rows": rows, "columns": columns, "span": 2,
    }
    plan = (_row_plan if axis == "row" else _col_plan)(
        rows, columns, rank, 2, name=projection)
    got = prepare_span2_compact(wire, device="cuda", **_cut_kwargs(plan))
    want = _old_span2(members[0].blob, plan, projection)
    _assert_planes_equal(got, want)
    role = plan.roles[0]
    if axis == "row":
        assert got["rows"] == role.hi - role.lo
        assert got["cols"] == columns
    else:
        assert got["cols"] == role.hi - role.lo


@cuda
def test_actual_a4_wire_decodes_identically_through_the_native_symbol():
    """The packed planes are the decoder's input, so plane equality already
    settles the decode; this runs the actual symbol when it can build, on the
    whole unit and on TP2 rank 1's row cut, and holds both to
    ``prepare_span2_planes``' planes through the same op."""
    from tessera.lane_planes import prepare_span2_planes
    from tessera.serving.ext import get_tessera_ext
    from tessera.serving.sharding import shard_parsed_roles
    from tessera.unit_artifact import parse_unit_artifact
    from tessera.compact_prep import parse_compact_wire, prepare_span2_compact
    from tessera.fused import parse_fused

    if get_tessera_ext() is None:
        pytest.skip("the native span-2 decoder could not be built here")
    from tessera.serving.ops import _decode_impl

    blob = _a4_wire_blob("gate_proj")
    member = parse_fused(blob)[0]
    wire = parse_compact_wire(member.blob, device="cuda", name="gate_proj")

    def decode(planes):
        packed = torch.empty((planes["rows"], planes["cols"] // 2),
                             dtype=torch.uint8, device="cuda")
        scales = torch.empty((planes["rows"], planes["cols"] // 16),
                             dtype=torch.uint8, device="cuda")
        _decode_impl(
            planes["select"], planes["label"], planes["point"], planes["nibbles"],
            planes["lut_bytes"], planes["label_lut"], planes["subset_nibbles"],
            int(planes["rows"]), int(planes["cols"]), int(planes["rate"]),
            int(planes["arity"]), int(planes["memory"]), int(planes["half"]),
            packed, scales)
        return packed, scales

    for rank in (0, 1):
        plan = _row_plan(2048, 4096, rank, 2, name="gate_proj")
        compact_planes = prepare_span2_compact(wire, device="cuda", **_cut_kwargs(plan))
        parsed = parse_unit_artifact(member.blob, device="cuda")
        shard = shard_parsed_roles([("gate_proj", parsed)], plan)[0][1]
        old_planes = prepare_span2_planes(shard, device="cuda")
        got, got_scale = decode(compact_planes)
        want, want_scale = decode(old_planes)
        assert torch.equal(got, want), ("packed", rank)
        assert torch.equal(got_scale, want_scale), ("scales", rank)


# ---------------------------------------------------------------------------
# GPU: window tile-word layout straight from the packed BODY
# ---------------------------------------------------------------------------


def _encoded_window(rows, cols, grid, seed, q256=1024):
    from tessera.export import encode_linear_planes

    torch.manual_seed(seed)
    weight = (torch.randn(rows, cols, device="cuda") * 0.02).contiguous()
    exported, _unit, _forests = encode_linear_planes(
        weight, grid=grid, q256=q256, name="window-compact", verify=False)
    return exported.blob


def _sliced_window(blob, plan, name, device="cuda"):
    """The old reader's shard: the unit the route would have cut."""
    from tessera.serving.sharding import shard_parsed_roles
    from tessera.unit_artifact import parse_unit_artifact

    parsed = parse_unit_artifact(blob, device=device)
    if plan is None:
        return parsed
    return shard_parsed_roles([(name, parsed)], plan)[0][1]


def _assert_repack_equal(got, want):
    for field in ("tile_words", "n_tiles", "rows", "cols", "rows_p"):
        assert int(getattr(got, field)) == int(getattr(want, field)), field
    assert torch.equal(got.words, want.words), "words"
    assert torch.equal(got.perm, want.perm), "perm"
    assert torch.equal(got.runs, want.runs), "runs"
    assert got.rates == want.rates, "rates"


def _reference_window_repack(shard):
    from tessera.kernel_window_gemv import repack_window_body

    return repack_window_body(shard.unit.body_bits,
                              tuple(int(r) for r in shard.unit.rates))


@cuda
@pytest.mark.parametrize("family, grid_name", [("e4m3", "E4M3"), ("value", "BF16")])
def test_window_compact_whole_unit_is_the_repacked_layout(family, grid_name):
    """Whole unit: the loader's ``WindowGemvUnit`` equals the reference
    repack (words/perm/runs), carries the sliced row scale, the route's table
    and an explicit zero start state -- none of it through an expanded codes
    tensor."""
    from tessera import kernel_window_gemv as kg
    from tessera.alphabet import BF16_GRID, E4M3_GRID
    from tessera.bf16_route import window_table_values
    from tessera.compact_prep import parse_compact_wire, prepare_window_compact
    from tessera.unit_artifact import parse_unit_artifact

    rows, cols = 128, 512
    grid = E4M3_GRID if grid_name == "E4M3" else BF16_GRID
    blob = _encoded_window(rows, cols, grid, seed=31)
    parsed = parse_unit_artifact(blob, device="cuda")
    wire = parse_compact_wire(blob, device="cuda", name="w")
    unit = prepare_window_compact(wire, device="cuda", family=family)
    _assert_repack_equal(unit.rep, _reference_window_repack(parsed))
    assert int(unit.rows) == rows and int(unit.cols) == cols
    assert unit.initial_state is not None
    assert unit.initial_state.dtype is torch.int32
    assert int(unit.initial_state.numel()) == cols
    assert not bool(unit.initial_state.any()), "a whole unit starts from zero"
    assert int(unit.row_offset) == 0
    assert not bool(unit.permuted_start_state().any()), "zero start, permuted"
    assert torch.equal(
        unit.scale,
        (parsed.unit.scale_rows.float()
         * float(parsed.unit.scale_global)).reshape(-1).contiguous())
    # The table is the route's own: ``native[codes]`` for E4M3, the grid's
    # values for BF16, over the ALPHABET plane's codes.
    if family == "e4m3":
        native = torch.tensor(grid.native, dtype=torch.uint8, device="cuda")
        want = native.view(torch.float8_e4m3fn).float()[
            parsed.unit.window_codes.long()]
        assert torch.equal(unit.table, want.to(unit.table.dtype))
        assert torch.equal(unit.codes_of_state, parsed.unit.window_codes.to(torch.uint8))
        assert torch.equal(unit.native, native)
        assert unit.family == "e4m3"
    else:
        want = window_table_values(parsed.unit.window_codes, parsed.grid)
        assert torch.equal(unit.table, want.to(unit.table.dtype))
        assert unit.codes_of_state is None and unit.native is None
        assert unit.family == "value"


@cuda
@pytest.mark.parametrize("family, grid_name", [("e4m3", "E4M3"), ("value", "BF16")])
def test_window_compact_matches_the_packaged_unit_preparers(family, grid_name):
    """The whole-unit bundle against the lane's own preparers
    (``prepare_from_parsed`` / ``prepare_value_unit``), when the packaged
    window extension builds here."""
    from tessera import kernel_window_gemv as kg
    from tessera.alphabet import BF16_GRID, E4M3_GRID
    from tessera.bf16_route import window_table_values
    from tessera.compact_prep import parse_compact_wire, prepare_window_compact
    from tessera.unit_artifact import parse_unit_artifact

    try:
        kg._ext()
    except Exception as exc:  # noqa: BLE001 -- the build result IS the capability
        pytest.skip(f"the packaged window GEMV extension did not build here: {exc}")
    rows, cols = 128, 512
    grid = E4M3_GRID if grid_name == "E4M3" else BF16_GRID
    blob = _encoded_window(rows, cols, grid, seed=32)
    parsed = parse_unit_artifact(blob, device="cuda")
    wire = parse_compact_wire(blob, device="cuda", name="w")
    unit = prepare_window_compact(wire, device="cuda", family=family)
    if family == "e4m3":
        old = kg.prepare_from_parsed(parsed)
    else:
        values = window_table_values(parsed.unit.window_codes, parsed.grid)
        old = kg.prepare_value_unit(
            parsed.unit.body_bits, tuple(int(r) for r in parsed.unit.rates),
            int(parsed.unit.window_bits), values,
            scale=(parsed.unit.scale_rows.float()
                   * float(parsed.unit.scale_global)).reshape(-1).contiguous(),
            table_dtype=torch.bfloat16)
    _assert_repack_equal(unit.rep, old.rep)
    assert torch.equal(unit.table, old.table)
    assert torch.equal(unit.scale, old.scale)
    assert int(unit.row_offset) == 0 and not bool(unit.initial_state.any())


@cuda
@pytest.mark.parametrize("family, grid_name", [("e4m3", "E4M3"), ("value", "BF16")])
@pytest.mark.parametrize("axis", ["row", "column"])
def test_window_compact_tp_cuts_match_the_reference_repack(family, grid_name, axis):
    """TP2 both ranks: the tile-word body equals ``repack_window_body`` on the
    sliced unit, the row scale is the sliced one, and the row cut's
    ``initial_state`` is the shard's own register -- rank 1 is nonzero."""
    from tessera.alphabet import BF16_GRID, E4M3_GRID
    from tessera.compact_prep import parse_compact_wire, prepare_window_compact

    rows, cols = 256, 512
    grid = E4M3_GRID if grid_name == "E4M3" else BF16_GRID
    blob = _encoded_window(rows, cols, grid, seed=41 + (family == "value"))
    wire = parse_compact_wire(blob, device="cuda", name="w")
    for rank in (0, 1):
        plan = (_row_plan if axis == "row" else _col_plan)(rows, cols, rank, 2)
        shard = _sliced_window(blob, plan, "w")
        unit = prepare_window_compact(
            wire, device="cuda", family=family, **_cut_kwargs(plan))
        _assert_repack_equal(unit.rep, _reference_window_repack(shard))
        assert torch.equal(unit.scale, (
            shard.unit.scale_rows.float()
            * float(shard.unit.scale_global)).reshape(-1).contiguous()), "scale"
        expected = shard.unit.initial_state
        if expected is None:
            assert not bool(unit.initial_state.any()), "no history means zeros"
        else:
            assert torch.equal(unit.initial_state, expected.to(torch.int32)), "state"
            # The kernel reads history through the repack's column order; the
            # accessor is silent about the base state's dtype, so compare it
            # against the shard's register permuted by the same reference perm.
            assert torch.equal(
                unit.permuted_start_state(),
                expected.index_select(0, unit.rep.perm.long()).to(torch.int32)), "permuted state"
            if bool(expected.any()):
                assert bool(unit.initial_state.any()), "the inherited register is real"
        assert int(unit.row_offset) == int(shard.unit.row_offset)


@cuda
@pytest.mark.parametrize("M", [9, 15, 17])
def test_window_compact_carries_no_multiple_of_16_rule(M):
    """The loader holds no M ladder: M in {9, 15, 17} builds a unit when the
    caller supplies the plan, and the bundle carries the history fields the
    GEMM's tail rows read."""
    from tessera import kernel_window_gemv as kg
    from tessera.alphabet import E4M3_GRID
    from tessera.compact_prep import parse_compact_wire, prepare_window_compact

    blob = _encoded_window(128, 512, E4M3_GRID, seed=57)
    wire = parse_compact_wire(blob, device="cuda", name="w")
    unit = prepare_window_compact(wire, device="cuda", family="e4m3",
                                  plan=kg.Plan(), M=M)
    assert int(unit.rep.cols) == 512
    assert unit.initial_state is not None and unit.initial_state.dtype is torch.int32
    assert int(unit.row_offset) == 0


@cuda
def test_window_compact_row_cut_requires_history_at_construction():
    """A ``WindowGemvUnit`` with a nonzero row offset and no start state is
    refused where it is built: no consumer can decode a cut from zero."""
    from tessera import kernel_window_gemv as kg

    rep = kg.Repacked(
        words=torch.zeros(16, dtype=torch.int32, device="cuda"),
        tile_words=4, n_tiles=1, rows=16, cols=1, rows_p=512,
        perm=torch.zeros(1, dtype=torch.int32, device="cuda"),
        runs=torch.tensor([[4, 0, 1, 0]], dtype=torch.int32, device="cuda"),
        rates=(4,),
    )
    with pytest.raises(GrammarError, match="carries no start state"):
        kg.WindowGemvUnit(
            rep=rep, table=torch.zeros(1 << 4, device="cuda"),
            scale=torch.ones(16, device="cuda"), window_bits=4,
            plan=kg.Plan(), family="value", initial_state=None, row_offset=16)
    with pytest.raises(GrammarError, match="int32"):
        kg.WindowGemvUnit(
            rep=rep, table=torch.zeros(1 << 4, device="cuda"),
            scale=torch.ones(16, device="cuda"), window_bits=4,
            plan=kg.Plan(), family="value",
            initial_state=torch.zeros(1, dtype=torch.int64, device="cuda"),
            row_offset=16)
    with pytest.raises(GrammarError, match="one register per column"):
        kg.WindowGemvUnit(
            rep=rep, table=torch.zeros(1 << 4, device="cuda"),
            scale=torch.ones(16, device="cuda"), window_bits=4,
            plan=kg.Plan(), family="value",
            initial_state=torch.zeros(2, dtype=torch.int32, device="cuda"),
            row_offset=16)


@cuda
def test_window_compact_refuses_a_non_channel_plane_by_name():
    """The packaged window unit is the CHANNEL plane; a LUT-plane window unit
    (research lane) is refused by the plane's name, not silently served."""
    from tessera.compact_prep import parse_compact_wire, prepare_window_compact

    blob = _legacy("e2m1x2-640-window-lut-256c.tessera")
    wire = parse_compact_wire(blob, device="cuda", name="w")
    with pytest.raises(GrammarError, match="CHANNEL"):
        prepare_window_compact(wire, device="cuda")
