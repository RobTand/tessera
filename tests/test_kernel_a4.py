"""Native A4 span-2 compute: plane decode and W4A4 parity on real expert wires.

The candidate is ``tessera.kernel_a4`` (fused plane decode + block-scaled FP4
MMA).  The oracle is the repository's own reference decode
(``tessera.stock.materialize_stock``, which the CUDA decoder is held to byte
for byte) and, for the multiply, both the exact fp64 arithmetic on the decoded
codes/scales and the executed W4A4 contract (vLLM's ``scaled_fp4_quant`` plus
``torch._scaled_mm``, the dense route's arithmetic).

The wires are external bytes, not fixtures: point ``TESSERA_A4_WIRE_DIR`` at a
directory holding ``gate_proj_wire.bin``, ``up_proj_wire.bin``,
``down_proj_wire.bin`` and ``a4-config.json`` (the checkout's own tester
publishes them under ``/mnt/shared/astra-native-a4/data``).  A missing
directory fails with the variable named, it does not silently skip.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

try:
    import pytest
except ImportError:  # the stock serve image runs the gate without pytest
    class _Mark:
        def __getattr__(self, name):
            def decorator(*args, **kwargs):
                return lambda fn: fn
            return decorator

    class _Raises:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            return self

        def __exit__(self, kind, value, tb):
            if kind is None:
                raise AssertionError(f"expected {self.exc.__name__}")
            return issubclass(kind, self.exc)

    class _GatePytest:
        def __init__(self):
            self.mark = _Mark()

        def fixture(self, *args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda fn: fn

        def raises(self, exc):
            return _Raises(exc)

        @staticmethod
        def skip(reason=""):
            raise RuntimeError(f"skipped: {reason}")

        @staticmethod
        def fail(reason=""):
            raise AssertionError(reason)

    pytest = _GatePytest()

from tessera.errors import GrammarError
from tessera.kernel_a4 import (
    A4Unit,
    A4UnitStack,
    a4_decode_span2_tile,
    a4_decode_states_at,
    a4_quantize_activation,
    a4_span2_gemm,
    a4_span2_grouped_gemm,
    build_code_nibbles,
    native_fp4_backend,
    require_native_fp4_mma,
)
from tessera.lane_planes import prepare_span2_planes

DATA = Path(os.environ.get("TESSERA_A4_WIRE_DIR", "/mnt/shared/astra-native-a4/data"))
PREFIX = "model.language_model.layers.3.mlp.experts"
LAYER = "layers_3"
TP = 2
E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


def _require_data() -> None:
    if not DATA.is_dir() or not (DATA / "a4-config.json").is_file():
        pytest.fail(
            f"the A4 wire fixtures are absent: TESSERA_A4_WIRE_DIR={DATA} holds no "
            "a4-config.json. Point it at the shared A4 fixture directory instead of "
            "skipping the gate.")


def declared():
    _require_data()
    cfg = json.loads((DATA / "a4-config.json").read_text())
    scheme = cfg["quantization_config"]["config_groups"][
        f"tessera_model_language_model_{LAYER}_mlp_experts"]["scheme"]
    from tessera.serving.scheme import validate_tessera_moe_scheme

    return validate_tessera_moe_scheme(scheme, PREFIX)


def rank_roles(rank: int, device: str = "cpu"):
    """``{"w13": [gate, up], "w2": [down]}`` shard-parsed for one TP2 rank.

    The full container is parsed and verified (the integrity path the loader
    uses), then cut to the rank: gate/up rows and down columns.
    """
    _require_data()
    from tessera.serving.moe_route import _packed_group_shard_plan
    from tessera.serving.scheme import expert_role_declarations, parse_tessera_expert_blob
    from tessera.serving.sharding import shard_parsed_roles

    declared_scheme = declared()
    roles13 = expert_role_declarations(declared_scheme["groups"]["w13"])
    roles2 = expert_role_declarations(declared_scheme["groups"]["w2"])
    plan13 = _packed_group_shard_plan(declared_scheme, "w13", PREFIX, rank, TP)
    plan2 = _packed_group_shard_plan(declared_scheme, "w2", PREFIX, rank, TP)
    blobs = {name: (DATA / f"{name}_wire.bin").read_bytes()
             for name in ("gate_proj", "up_proj", "down_proj")}
    parsed13 = [
        parse_tessera_expert_blob(blobs["gate_proj"], roles13[0], f"{PREFIX} gate",
                                  device=device)[0],
        parse_tessera_expert_blob(blobs["up_proj"], roles13[1], f"{PREFIX} up",
                                  device=device)[0],
    ]
    parsed2 = [parse_tessera_expert_blob(blobs["down_proj"], roles2[0],
                                         f"{PREFIX} down", device=device)[0]]
    return {"w13": shard_parsed_roles(parsed13, plan13),
            "w2": shard_parsed_roles(parsed2, plan2)}


def units_for(rank: int, device="cpu"):
    """The prepared bundles + oracle tiles for one TP2 rank."""
    from tessera.stock import materialize_stock

    out = {}
    for group, roles in rank_roles(rank).items():
        group_out = []
        for name, parsed in roles:
            prepared = prepare_span2_planes(parsed, device=device)
            stock = materialize_stock(parsed.unit, parsed.forests, parsed.code)
            unit = A4Unit.from_prepared(prepared)
            group_out.append((name, parsed, unit, stock, prepared))
        out[group] = group_out
    return out


def _bits(tensor: torch.Tensor) -> np.ndarray:
    return np.unpackbits(tensor.cpu().numpy().view(np.uint8)).astype(np.int64)


def _read_bits(bits: np.ndarray, bit: np.ndarray, width: int) -> np.ndarray:
    """The kernel's big-endian byte read, spelled one bit at a time.

    ``bits`` is an unpacked MSB-first bit plane; the read opens on ``bit``'s
    byte and keeps ``width`` bits.  Written this way so the CPU oracle mirrors
    the kernel's arithmetic rather than re-deriving the window in a more
    forgiving form (the anchor bug the GPU kernel had is exactly what the
    forgiving form fails to catch).
    """
    byte = np.asarray(bit, dtype=np.int64) // 8
    value = np.zeros(byte.shape, dtype=np.int64)
    for i in range(width):
        value = (value << 1) | bits[byte * 8 + i]
    return value


def decode_states_reference(parsed, prepared, ks, ps):
    """The kernel's nine decode intermediates, bit-for-bit, in numpy.

    Mirrors ``_decode_pair_states``: the select window is anchored at the
    oldest of its ``memory+1`` bits and read as a big-endian 16-bit field; the
    label byte is the kernel's; the pair's two ``rate-1``-bit points are one
    16-bit read with absolute shifts.  The tile this produces is held byte for
    byte to ``materialize_stock`` by the CPU tests, so the formulas are the
    repository's reference decoder, not a re-derivation.
    """
    rows, _cols = int(prepared["rows"]), int(prepared["cols"])
    rate, arity = int(prepared["rate"]), int(prepared["arity"])
    memory = int(prepared["memory"])
    field = rate - 1
    steps = rows // arity
    pairs = steps // 2
    points = 1 << field
    select = _bits(prepared["select"])
    label = _bits(prepared["label"])
    point = _bits(prepared["point"])
    label_lut = prepared["label_lut"].cpu().numpy()
    subset = prepared["subset_nibbles"].cpu().numpy().astype(np.int64).reshape(4, points, arity)
    code = subset[:, :, 0] | (subset[:, :, 1] << 4)
    ks = np.asarray(ks, dtype=np.int64)
    ps = np.asarray(ps, dtype=np.int64)
    q = ks * (pairs + 8) + (8 - memory) + ps                 # anchor, oldest bit
    window = (_read_bits(select, q, 16) >> (15 - memory - (q % 8))) & ((1 << (memory + 1)) - 1)
    ell = label_lut[window]
    lab_bit = ks * (pairs * 2) + ps * 2
    stored = (_read_bits(label, lab_bit, 8) >> (6 - (ps * 2) % 8)) & 3
    lab0 = (ell - stored) & 3
    lab1 = stored
    t = ks * (steps * field) + ps * (2 * field)
    u = _read_bits(point, t, 16)
    pt0 = (u >> (16 - field - (t % 8))) & (points - 1)
    pt1 = (u >> (16 - 2 * field - (t % 8))) & (points - 1)
    return window, ell, stored, lab0, lab1, pt0, pt1, code[lab0, pt0], code[lab1, pt1]


def decode_reference(parsed, prepared):
    """The kernel's decode as ``materialize_stock``'s ``[rows, cols//2]`` tile.

    Assembled from ``decode_states_reference`` over every ``(k, pair)``, plus
    the LUT scale plane as ``[rows, cols//16]`` E4M3 bytes: the whole oracle is
    the kernel's own arithmetic, checked against the repository's reference
    decoder.
    """
    rows, cols = int(prepared["rows"]), int(prepared["cols"])
    rate, arity = int(prepared["rate"]), int(prepared["arity"])
    half = int(prepared["half"])
    steps = rows // arity
    pairs = steps // 2
    ks = np.repeat(np.arange(cols), pairs)
    ps = np.tile(np.arange(pairs), cols)
    _w, _e, _s, _l0, _l1, _p0, _p1, code0, code1 = decode_states_reference(
        parsed, prepared, ks, ps)
    nibbles = np.stack([code0 & 0xF, code0 >> 4, code1 & 0xF, code1 >> 4], axis=-1)
    nibbles = nibbles.reshape(cols, pairs * 4)               # (k, p, r) -> [cols, rows]
    even, odd = nibbles[0::2], nibbles[1::2]
    tile = (even & 0xF) | ((odd & 0xF) << 4)
    tile = tile.T.astype(np.uint8)                           # [rows, cols//2]

    # The LUT plane is one flat byte plane: two rows per byte, even row high.
    nib_plane = prepared["nibbles"].cpu().numpy().astype(np.uint8).reshape(-1)
    lut_bytes = prepared["lut_bytes"].cpu().numpy().astype(np.uint8)
    groups = cols // half
    row_index = np.arange(rows)
    scales = np.empty((rows, groups), dtype=np.uint8)
    for g in range(groups):
        gn = g * rows + row_index
        byte = nib_plane[gn // 2]
        nib = np.where(row_index % 2 == 0, byte >> 4, byte & 0xF)
        scales[:, g] = lut_bytes[nib]
    return tile, scales


@pytest.mark.parametrize("rank", [0, 1])
def test_code_nibbles_match_subset_semantics(rank):
    """Every code-table entry is the subset table's own two nibbles."""
    roles = units_for(rank)
    _name, _parsed, unit, _stock, prepared = roles["w13"][0]
    points = 1 << (unit.rate - 1)
    subset = prepared["subset_nibbles"].cpu().numpy().astype(np.int64)
    table = unit.code_nibbles.cpu().numpy().astype(np.int64)
    expected = np.zeros(4 * points, dtype=np.int64)
    arity = unit.arity
    for label in range(4):
        for point in range(points):
            expected[label * points + point] = (
                subset[(label * points + point) * arity + 0]
                | (subset[(label * points + point) * arity + 1] << 4))
    assert list(table) == list(expected)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("group", ["w13", "w2"])
def test_plane_decode_matches_materialize_stock(rank, group):
    """The kernel's addressing, in numpy, is the reference decode byte for byte."""
    for name, parsed, unit, stock, prepared in units_for(rank)[group]:
        tile, scales = decode_reference(parsed, prepared)
        oracle_tile = stock["weight_packed"].cpu().numpy()
        oracle_scales = stock["weight_scale"].view(torch.uint8).cpu().numpy()
        assert tile.shape == oracle_tile.shape, name
        assert scales.shape == oracle_scales.shape, name
        assert np.array_equal(tile, oracle_tile), f"{name}: code tile differs"
        assert np.array_equal(scales, oracle_scales), f"{name}: scale tile differs"


# ---------------------------------------------------------------------------
# GPU: the kernels
#
# The checks are plain functions because the stock serve image carries torch,
# Triton and vLLM's operators but no pytest: ``run_gate`` (``python3
# tests/test_kernel_a4.py``) is the same gate the pytest wrappers below run.
# ---------------------------------------------------------------------------


def cuda_units():
    """The rank-local bundles on the CUDA device, oracle tiles kept on CPU."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    require_native_fp4_mma("test_kernel_a4")
    return {rank: {group: [(name, parsed, unit.to("cuda"), stock, prepared)
                           for name, parsed, unit, stock, prepared in units_for(rank)[group]]
                   for group in ("w13", "w2")}
            for rank in (0, 1)}


@pytest.fixture(scope="module")
def gpu_units():
    return cuda_units()


def check_native_fp4_mma():
    """The probe's own PTX must carry the block-scaled FP4 MMA instruction."""
    from tessera.kernel_a4 import native_fp4_mma_ptx_tokens

    require_native_fp4_mma("test_kernel_a4")
    tokens = native_fp4_mma_ptx_tokens()
    assert "mxf4nvf4" in tokens, tokens
    assert "mma.sync" in tokens, tokens
    return {"backend": native_fp4_backend(), "ptx_tokens": tokens}


def check_quantizer():
    """The linear-layout quantizer's codes are the swizzled op's codes."""
    x = torch.randn(32, 512, dtype=torch.bfloat16, device="cuda")
    gscale = torch.tensor([448.0 * 6.0 / float(x.abs().max())],
                          dtype=torch.float32, device="cuda")
    packed, scales = a4_quantize_activation(x, gscale)
    ref_packed, _ = torch.ops._C.scaled_fp4_quant(x.contiguous(), gscale, True)
    assert torch.equal(packed, ref_packed.view(torch.uint8).reshape(packed.shape))
    assert scales.shape == (32, 32)
    return {"codes_equal": True, "scales_shape": list(scales.shape)}


STATE_NAMES = ("window", "ell", "stored", "lab0", "lab1", "pt0", "pt1", "code0", "code1")


def check_decode_states(units, rank, group, limit=8):
    """The kernel's decode intermediates vs the planes, at explicit indices.

    Bounded diagnostic: the first and last 64 columns crossed with the first
    and last 32 row pairs, so a failure names the field and the index instead
    of only that the tile differs.
    """
    for name, parsed, unit, stock, prepared in units[rank][group]:
        pairs = unit.rows // 4
        columns = np.concatenate([np.arange(0, 64), np.arange(unit.cols - 64, unit.cols)])
        pair_ids = np.concatenate([np.arange(0, 32), np.arange(pairs - 32, pairs)])
        ks = np.repeat(columns, pair_ids.size)
        ps = np.tile(pair_ids, columns.size)
        got = a4_decode_states_at(unit, torch.as_tensor(ks, dtype=torch.int32, device="cuda"),
                                  torch.as_tensor(ps, dtype=torch.int32, device="cuda"))
        expected = decode_states_reference(parsed, prepared, ks, ps)
        reported = []
        for field_name, got_field, want in zip(STATE_NAMES, got, expected):
            got_np = got_field.cpu().numpy().astype(np.int64)
            for index in np.nonzero(got_np != want)[0][:limit]:
                reported.append((int(ks[index]), int(ps[index]), field_name,
                                 int(got_np[index]), int(want[index])))
        if reported:
            detail = "; ".join(f"k={k} p={p} {field}: got {g} want {w}"
                               for k, p, field, g, w in reported[:limit])
            raise AssertionError(f"{name}: decode states differ -> {detail}")
    return {"indexed": int(ks.size)}


def check_decode_kernel(units, rank, group):
    """Decode-kernel codes and scales vs ``materialize_stock``, byte for byte."""
    for name, parsed, unit, stock, prepared in units[rank][group]:
        va, vb, scale = a4_decode_span2_tile(unit)
        va = va.cpu().numpy().astype(np.int64)
        vb = vb.cpu().numpy().astype(np.int64)
        pairs = va.shape[1]
        tile = np.empty((unit.cols // 2, unit.rows), dtype=np.uint8)
        for r in range(4):
            even = (va >> (4 * r)) & 0xF
            odd = (vb >> (4 * r)) & 0xF
            tile[:, 4 * np.arange(pairs) + r] = (even | (odd << 4)).astype(np.uint8)
        tile = tile.T
        oracle_tile = stock["weight_packed"].cpu().numpy()
        oracle_scales = stock["weight_scale"].view(torch.uint8).cpu().numpy()
        if not np.array_equal(tile, oracle_tile):
            bad = np.argwhere(tile != oracle_tile)
            first = bad[0]
            raise AssertionError(
                f"{name}: decode kernel codes differ at {len(bad)} of {tile.size} "
                f"bytes; first (row={int(first[0])}, packed_col={int(first[1])}) "
                f"got 0x{tile[first[0], first[1]]:02x} want "
                f"0x{oracle_tile[first[0], first[1]]:02x}")
        if not np.array_equal(scale.view(torch.uint8).cpu().numpy(), oracle_scales):
            bad = np.argwhere(scale.view(torch.uint8).cpu().numpy() != oracle_scales)
            first = bad[0]
            raise AssertionError(
                f"{name}: decode kernel scales differ at {len(bad)} of "
                f"{oracle_scales.size}; first (row={int(first[0])}, group="
                f"{int(first[1])})")
    return {"roles": len(units[rank][group])}


def _e4m3_to_float64(scale_bytes) -> np.ndarray:
    """E4M3 bytes as numbers through torch's own converter (numpy has no fp8)."""
    if isinstance(scale_bytes, torch.Tensor):
        tensor = scale_bytes.view(torch.uint8).cpu()
    else:
        tensor = torch.as_tensor(np.asarray(scale_bytes, dtype=np.uint8))
    return tensor.view(torch.float8_e4m3fn).to(torch.float64).numpy()


def _dequant_weights(packed, scales, rows: int, cols: int):
    """The stock tile's codes and E4M3 scales as exact fp64 weights."""
    values = np.array([(-1 if i >> 3 else 1) * E2M1_VALUES[i & 7] for i in range(16)])
    packed = np.asarray(packed).astype(np.int64)
    codes = np.empty((rows, cols), dtype=np.int64)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = (packed >> 4) & 0xF
    return values[codes] * np.repeat(_e4m3_to_float64(scales), 16, axis=1)


def _dequant_activation(packed: torch.Tensor, scales: torch.Tensor, cols: int):
    """The quantizer's codes and E4M3 scales as exact fp64 activations."""
    values = np.array([(-1 if i >> 3 else 1) * E2M1_VALUES[i & 7] for i in range(16)])
    packed_np = packed.cpu().numpy().astype(np.int64)
    codes = np.empty((packed_np.shape[0], cols), dtype=np.int64)
    codes[:, 0::2] = packed_np & 0xF
    codes[:, 1::2] = (packed_np >> 4) & 0xF
    return values[codes] * np.repeat(_e4m3_to_float64(scales), 16, axis=1)


def check_gemm_arithmetic(units, rank, group, m):
    """Output vs the exact fp64 product of the decoded codes and scales."""
    details = {}
    for name, parsed, unit, stock, prepared in units[rank][group]:
        torch.manual_seed(0)
        cols, rows = unit.cols, unit.rows
        x = (torch.randn(m, cols, dtype=torch.bfloat16, device="cuda") * 0.25)
        gscale = torch.tensor([448.0 * 6.0 / max(float(x.abs().max()), 1e-6)],
                              dtype=torch.float32, device="cuda")
        packed, scales = a4_quantize_activation(x, gscale)
        y = a4_span2_gemm(packed, scales, unit, unit.epilogue_for(gscale),
                          out_dtype=torch.float32)
        assert y.shape == (m, rows)
        a = _dequant_activation(packed, scales, cols)
        w = _dequant_weights(stock["weight_packed"].cpu().numpy(),
                             stock["weight_scale"].view(torch.uint8).cpu().numpy(),
                             rows, cols)
        epilogue = unit.global_scale / float(gscale)
        ref = (a @ w.T) * epilogue
        got = y.cpu().numpy().astype(np.float64)
        denom = max(np.abs(ref).max(), 1e-12)
        rel = float(np.abs(got - ref).max() / denom)
        assert rel < 2e-3, f"{name} M={m}: rel={rel}"
        details[name] = rel
    return details


def check_gemm_contract(units, rank, group, m):
    """Output vs vLLM quantizer + ``torch._scaled_mm``: the route's own arithmetic."""
    from tessera.serving.nvfp4_route import blocked_scales

    details = {}
    for name, parsed, unit, stock, prepared in units[rank][group]:
        torch.manual_seed(0)
        cols, rows = unit.cols, unit.rows
        x = (torch.randn(m, cols, dtype=torch.bfloat16, device="cuda") * 0.25)
        gscale = torch.tensor([448.0 * 6.0 / max(float(x.abs().max()), 1e-6)],
                              dtype=torch.float32, device="cuda")
        packed, scales = a4_quantize_activation(x, gscale)
        y = a4_span2_gemm(packed, scales, unit, unit.epilogue_for(gscale),
                          out_dtype=torch.float32)

        a_q, a_s = torch.ops._C.scaled_fp4_quant(x.contiguous(), gscale, True)
        a_q = a_q.view(torch.float4_e2m1fn_x2)
        a_s = a_s.view(torch.uint8).view(torch.float8_e4m3fn).contiguous()
        b_q = stock["weight_packed"].to("cuda").view(torch.float4_e2m1fn_x2)
        b_s = blocked_scales(
            stock["weight_scale"].view(torch.uint8).view(torch.float8_e4m3fn).to("cuda"))
        try:
            ref = torch._scaled_mm(a_q, b_q.t(), scale_a=a_s, scale_b=b_s,
                                   out_dtype=torch.float32)
        except RuntimeError:
            ref = torch._scaled_mm(a_q, b_q.t(), scale_a=a_s, scale_b=b_s,
                                   out_dtype=torch.bfloat16).to(torch.float32)
        ref = ref * (unit.global_scale / float(gscale))
        rel = float((y - ref).abs().max() / ref.abs().max().clamp_min(1e-12))
        assert rel < 2e-2, f"{name} M={m}: rel={rel}"
        details[name] = rel
    return details


def check_grouped_gemm(units, rank, group, num_tokens=6, top_k=2):
    """The grouped operator vs the dense kernel on the same routing.

    The stack is the group's own roles (w13: gate and up -- distinct bytes,
    distinct scales, one geometry), which is the expert axis the kernel
    addresses; per-route outputs are held to the dense kernel's rows for the
    routed token, with repeated expert ids and an empty expert in the
    dispatch.  Routing weights are deliberately absent: the operator keeps
    per-route rows, and a routed MoE applies weights after the down
    projection.
    """
    bundle = units[rank][group]
    stack = A4UnitStack.stack([unit for _name, _parsed, unit, _stock, _prepared in bundle])
    experts = stack.experts
    cols, rows = stack.cols, stack.rows
    torch.manual_seed(7)
    x = torch.randn(num_tokens, cols, dtype=torch.bfloat16, device="cuda") * 0.25
    gscale = torch.tensor([448.0 * 6.0 / max(float(x.abs().max()), 1e-6)],
                          dtype=torch.float32, device="cuda")
    packed, scales = a4_quantize_activation(x, gscale)

    # Dispatch with a repeated expert id (token 0 twice on expert 0) and, when
    # there are two tiles, an expert that receives nothing.
    if experts == 1:
        chosen = torch.zeros((num_tokens, top_k), dtype=torch.long, device="cuda")
    else:
        chosen = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [1, 0]],
                              dtype=torch.long, device="cuda")[:num_tokens, :top_k]
        chosen[0, 1] = 0                              # repeated expert id
    flat_expert = chosen.reshape(-1)
    flat_token = torch.arange(num_tokens, device="cuda").repeat_interleave(top_k)
    order = torch.argsort(flat_expert, stable=True)
    token_ids = flat_token[order].to(torch.int32)
    counts = torch.bincount(flat_expert[order], minlength=experts)
    expert_offsets = torch.zeros(experts + 1, dtype=torch.int32, device="cuda")
    expert_offsets[1:] = torch.cumsum(counts, 0).to(torch.int32)

    epilogues = torch.stack([unit.epilogue_for(gscale)[0]
                             for _n, _p, unit, _s, _pr in bundle])
    got = a4_span2_grouped_gemm(
        packed, scales, stack, epilogues, expert_offsets=expert_offsets,
        token_ids=token_ids)

    assert got.shape == (token_ids.numel(), rows), (got.shape, token_ids.numel(), rows)
    ref = torch.zeros_like(got)
    for index in range(experts):
        start = int(expert_offsets[index].item())
        end = int(expert_offsets[index + 1].item())
        if start == end:
            continue
        toks = token_ids[start:end].to(torch.int64)
        dense = a4_span2_gemm(packed.index_select(0, toks), scales.index_select(0, toks),
                              bundle[index][2], epilogues[index:index + 1],
                              out_dtype=torch.float32)
        ref[start:end] = dense
    rel = float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-12))
    empty = int((counts == 0).sum().item())

    # A dispatch whose second expert is empty must be the same operator with an
    # expert axis nobody addresses: the early-exit path is the one tested.
    if experts > 1:
        one_way = torch.zeros((num_tokens, top_k), dtype=torch.long, device="cuda")
        flat = one_way.reshape(-1)
        order = torch.argsort(flat, stable=True)
        ids = flat_token[order].to(torch.int32)
        counts_one = torch.bincount(flat[order], minlength=experts)
        offsets_one = torch.zeros(experts + 1, dtype=torch.int32, device="cuda")
        offsets_one[1:] = torch.cumsum(counts_one, 0).to(torch.int32)
        got_one = a4_span2_grouped_gemm(packed, scales, stack, epilogues,
                                        expert_offsets=offsets_one, token_ids=ids)
        ref_one = a4_span2_gemm(packed.index_select(0, ids.to(torch.int64)),
                                scales.index_select(0, ids.to(torch.int64)),
                                bundle[0][2], epilogues[0:1], out_dtype=torch.float32)
        rel_one = float((got_one - ref_one).abs().max()
                        / ref_one.abs().max().clamp_min(1e-12))
        assert rel_one < 1e-6, f"grouped rank{rank} {group} empty-expert: rel={rel_one}"
        empty = int((counts_one == 0).sum().item())
    assert rel < 1e-6, f"grouped rank{rank} {group}: rel={rel}"
    return {"experts": experts, "routes": int(token_ids.numel()), "empty_experts": empty,
            "rel": rel}


def check_graph_capture(units, rank, group):
    """A dense forward captures and replays inside a CUDA graph.

    The epilogue is a device tensor and the quantizer takes a device scalar, so
    no host read sits in the call; a host synchronization during capture is an
    error in torch, which makes capture success the test.
    """
    bundle = units[rank][group]
    cols = bundle[0][2].cols
    torch.manual_seed(3)
    x = torch.randn(8, cols, dtype=torch.bfloat16, device="cuda") * 0.25
    gscale = torch.tensor([448.0 * 6.0 / max(float(x.abs().max()), 1e-6)],
                          dtype=torch.float32, device="cuda")
    details = {}
    for name, _parsed, unit, _stock, _prepared in bundle:
        packed, scales = a4_quantize_activation(x, gscale)
        epilogue = unit.epilogue_for(gscale)
        eager = a4_span2_gemm(packed, scales, unit, epilogue, out_dtype=torch.float32)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                a4_span2_gemm(packed, scales, unit, epilogue, out_dtype=torch.float32)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = a4_span2_gemm(packed, scales, unit, epilogue, out_dtype=torch.float32)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, eager), f"{name}: replay differs from eager"
        details[name] = "captured"
    return details


def check_gemm_refusal(units):
    _name, _parsed, unit, _stock, _prepared = units[0]["w2"][0]
    x = torch.randn(4, unit.cols, dtype=torch.bfloat16, device="cuda")
    gscale = torch.tensor([448.0], dtype=torch.float32, device="cuda")
    packed, scales = a4_quantize_activation(x, gscale)
    with pytest.raises(GrammarError):
        a4_span2_gemm(packed, scales, unit, gscale, block_n=unit.rows + 8)
    return {"refused": True}


# --- pytest wrappers --------------------------------------------------------


def test_native_fp4_backend_reports():
    assert native_fp4_backend() in ("tokenspeed_triton", "triton")


@CUDA
def test_native_fp4_mma_is_emitted_not_emulated(gpu_units):
    check_native_fp4_mma()


@CUDA
def test_quantizer_matches_vllm_swizzled_codes(gpu_units):
    check_quantizer()


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("group", ["w13", "w2"])
def test_decode_states_match_the_planes(gpu_units, rank, group):
    check_decode_states(gpu_units, rank, group)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("group", ["w13", "w2"])
def test_decode_kernel_matches_stock_tile(gpu_units, rank, group):
    check_decode_kernel(gpu_units, rank, group)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("group", ["w13", "w2"])
@pytest.mark.parametrize("m", [1, 8, 32, 128])
def test_gemm_matches_arithmetic_reference(gpu_units, rank, group, m):
    check_gemm_arithmetic(gpu_units, rank, group, m)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("group", ["w13", "w2"])
@pytest.mark.parametrize("m", [8, 32, 128])
def test_gemm_matches_executed_w4a4_contract(gpu_units, rank, group, m):
    check_gemm_contract(gpu_units, rank, group, m)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("group", ["w13", "w2"])
def test_grouped_gemm_matches_the_dense_kernel(gpu_units, rank, group):
    check_grouped_gemm(gpu_units, rank, group)


@CUDA
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("group", ["w13", "w2"])
def test_dense_forward_captures_in_a_cuda_graph(gpu_units, rank, group):
    check_graph_capture(gpu_units, rank, group)


def test_gemm_refuses_a_remainder_shape(gpu_units):
    check_gemm_refusal(gpu_units)


# --- the same gate without pytest (the serve image has no pytest) -----------


def run_gate(*, ranks=(0, 1), groups=("w13", "w2"), ms=(1, 8, 32, 128),
             only="all", report_path=None) -> dict:
    import json
    import traceback

    report = {"backend": native_fp4_backend(), "checks": [], "ok": False}
    units = cuda_units()

    def record(name, fn):
        entry = {"check": name}
        try:
            entry["detail"] = fn()
            entry["ok"] = True
        except Exception as exc:  # noqa: BLE001 -- the gate records failures
            entry["ok"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc()[-2000:]
        report["checks"].append(entry)
        shown = entry.get("detail") if entry["ok"] else str(entry.get("error", "")).splitlines()[0]
        print(f"[{ 'PASS' if entry['ok'] else 'FAIL' }] {name} {shown}", flush=True)

    record("native_fp4_mma", check_native_fp4_mma)
    record("quantizer", check_quantizer)
    for rank in ranks:
        for group in groups:
            record(f"decode_states.rank{rank}.{group}",
                   lambda rank=rank, group=group: check_decode_states(units, rank, group))
            record(f"decode_kernel.rank{rank}.{group}",
                   lambda rank=rank, group=group: check_decode_kernel(units, rank, group))
            if only == "decode":
                continue
            record(f"grouped.rank{rank}.{group}",
                   lambda rank=rank, group=group: check_grouped_gemm(units, rank, group))
            record(f"graph_capture.rank{rank}.{group}",
                   lambda rank=rank, group=group: check_graph_capture(units, rank, group))
            for m in ms:
                record(f"gemm_arithmetic.rank{rank}.{group}.M{m}",
                       lambda rank=rank, group=group, m=m:
                       check_gemm_arithmetic(units, rank, group, m))
            for m in (8, 32, 128):
                record(f"gemm_contract.rank{rank}.{group}.M{m}",
                       lambda rank=rank, group=group, m=m:
                       check_gemm_contract(units, rank, group, m))
    record("gemm_refusal", lambda: check_gemm_refusal(units))

    report["ok"] = all(entry["ok"] for entry in report["checks"])
    if report_path:
        with open(report_path, "w") as handle:
            json.dump(report, handle, indent=1)
    print("REPORT " + json.dumps(report))
    print("gate:", "PASS" if report["ok"] else "FAIL")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=None)
    parser.add_argument("--ms", default="1,8,32,128")
    parser.add_argument("--ranks", default="0,1")
    parser.add_argument("--groups", default="w13,w2")
    parser.add_argument("--only", default="all", choices=("all", "decode", "gemm"))
    args = parser.parse_args()
    result = run_gate(ranks=tuple(int(r) for r in args.ranks.split(",")),
                      groups=tuple(args.groups.split(",")),
                      ms=tuple(int(m) for m in args.ms.split(",")),
                      only=args.only,
                      report_path=args.report)
    raise SystemExit(0 if result["ok"] else 1)
