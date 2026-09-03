"""The BF16 route's decode-regime lane: the window GEMV over the value family.

The route used to decode every window body through the pure-torch window
decoder in both residency modes (every census record said
``decoder: torch_window``) while ``tessera.kernel_window_gemv`` already
carried the instantiation this route wants: the value family
(``prepare_value_unit`` -- a window body whose table holds bf16 values --
with ``decode_values`` as its reference and ``window_linear`` as the module
seam, ``x [..., K] bf16 -> [..., rows] bf16``).  These tests pin the dispatch:
the lane prepares where the unit is in range, the torch path serves where it
is not, and the census record names which engine each forward ran.

What "in range" means lives in ``bf16_route.gemv_eligible_for_unit`` and is
derived from the kernel module's own declared support constants -- the tests
below read the same constants rather than restating them, so a wider kernel
widens the expectation with no edit.

STUBBED: vLLM's ``LinearMethodBase`` and parameters, as in
``test_serving_bf16_route``.  There is no A-side quantiser to stub -- the A
side is bf16 as it arrives, which is the whole of this route's activation
contract.
"""
from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from tessera.serving import bf16_route as route                      # noqa: E402
from tessera.serving import lane as serving_lane                     # noqa: E402
from tessera.serving import telemetry                                # noqa: E402
from tessera.serving.lane import (                                   # noqa: E402
    MODE_RESIDENT, MODE_STREAMED, TESSERA_MODE_ENV, build_tessera_method)
from tessera.serving.scheme import TESSERA_BF16                      # noqa: E402

CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA device")


def _tessera():
    return (pytest.importorskip("tessera.fused"), pytest.importorskip("tessera.export"),
            pytest.importorskip("tessera.decode"), pytest.importorskip("tessera.alphabet"))


def _kg():
    return pytest.importorskip("tessera.kernel_window_gemv")


@pytest.fixture(autouse=True)
def _fresh_env(monkeypatch):
    serving_lane.reset_for_tests()
    monkeypatch.delenv(TESSERA_MODE_ENV, raising=False)
    yield
    serving_lane.reset_for_tests()


# --- the eligibility rule, on real units --------------------------------------
#
# CPU throughout: eligibility reads the unit's rates, window bits and start
# state -- no device, no extension -- so small CPU encodes decide it.


def _encode_cpu(q256, rows=32, cols=64, seed=0):
    fused, export, decode, alphabet = _tessera()
    torch.manual_seed(seed)
    w = torch.randn(rows, cols) * 0.02
    exported, unit, forests = export.encode_linear_planes(
        w, grid=alphabet.BF16_GRID, q256=q256, name="unit", verify=False)
    return exported, unit, forests


def test_eligibility_is_derived_from_the_kernel_constants():
    """The rule, not a roster: every supported rate and window is eligible."""
    kg = _kg()
    assert tuple(kg.SUPPORTED_RATES) == (1, 2, 4)
    assert tuple(kg.WINDOW_BITS_SUPPORTED) == (14,)
    from types import SimpleNamespace
    for rate in kg.SUPPORTED_RATES:
        unit = SimpleNamespace(rates=(rate,) * 8, window_bits=14, initial_state=None)
        assert route.gemv_eligible_for_unit(unit), rate
    for window_bits in kg.WINDOW_BITS_SUPPORTED:
        unit = SimpleNamespace(rates=(4,) * 8, window_bits=window_bits, initial_state=None)
        assert route.gemv_eligible_for_unit(unit), window_bits


def test_a_bresenham_mix_inside_the_supported_set_is_eligible():
    """Rates are per column: a schedule mixing supported rates is in range even
    though no uniform rung sits between them."""
    from types import SimpleNamespace
    unit = SimpleNamespace(rates=(2,) * 4 + (4,) * 4, window_bits=14, initial_state=None)
    assert route.gemv_eligible_for_unit(unit)


@pytest.mark.parametrize("q256", [256, 512, 1024])
def test_low_rungs_are_eligible(q256):
    """Real BF16 units at rate 1, 2 and 4: the kernel reads all three."""
    _exported, unit, _forests = _encode_cpu(q256)
    assert set(int(r) for r in unit.rates) <= {1, 2, 4}, [int(r) for r in unit.rates]
    assert route.gemv_eligible_for_unit(unit)


@pytest.mark.parametrize("q256", [768, 1792])
def test_a_rate_outside_the_supported_set_is_not_eligible(q256):
    """R = 3 has no lane here, and neither does R = 7: the torch path serves."""
    _exported, unit, _forests = _encode_cpu(q256)
    assert not route.gemv_eligible_for_unit(unit)


def test_a_mixed_schedule_with_one_unsupported_column_is_not_eligible():
    """q256=896 mixes 3 and 4: the 4-columns are readable, the unit is not --
    dispatch is per module, so one unsupported column keeps the torch path."""
    _exported, unit, _forests = _encode_cpu(896, cols=96)
    assert set(int(r) for r in unit.rates) == {3, 4}
    assert not route.gemv_eligible_for_unit(unit)


def test_a_window_outside_the_supported_set_is_not_eligible():
    from types import SimpleNamespace
    unit = SimpleNamespace(rates=(4,) * 8, window_bits=15, initial_state=None)
    assert not route.gemv_eligible_for_unit(unit)


def test_a_shard_start_state_is_not_eligible():
    """A TP row shard starts mid-stream; the kernel supplies state_{-1} = 0
    itself, so a shard keeps the torch lane that threads the state."""
    from tessera.layout import slice_unit
    from tessera.trellis import ConvCode
    from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact

    torch.manual_seed(0)
    fused, export, decode, alphabet = _tessera()
    weight = torch.randn(32, 96) * 0.02
    exported, _unit, _forests = export.encode_linear_planes(
        weight, grid=alphabet.BF16_GRID, q256=512, name="unit", verify=False)
    parsed = parse_unit_artifact(exported.blob, device="cpu")
    assert route.gemv_eligible_for_unit(parsed.unit)
    shard = slice_unit(parsed, rows=(8, 24))
    manifest = parsed.manifest
    _m, _region, blob = build_unit_artifact(
        shard, "rank", parsed.forests, int(manifest.branch.root_q256),
        parsed.code or ConvCode(),
        superblock=int(manifest.geometry.superblock_columns),
        container=manifest.branch.container)
    reparsed = parse_unit_artifact(blob, device="cpu")
    assert reparsed.unit.initial_state is not None
    assert not route.gemv_eligible_for_unit(reparsed.unit)


# --- the lane's names ----------------------------------------------------------

def test_gemv_max_m_is_the_kernel_build_limit():
    """A wider kernel widens this route with no edit: one place to remember."""
    kg = _kg()
    assert route.GEMV_MAX_M == kg.GEMV_MAX_M == 8


def test_gemv_symbol_and_module_name_are_the_kernel_and_contract_values():
    """The record's symbol is the op the lane invokes; the module name is the
    contract table's constant, not a second literal."""
    from tessera.serving import ext
    assert route.GEMV_MODULE_NAME == ext.WINDOW_GEMV_MODULE_NAME == "tessera_window_gemv"
    assert route.GEMV_SYMBOL == "tessera_window_gemv::gemv"


def test_the_census_expectations_come_from_the_route():
    """The decode phase can report every path the dispatch takes; batch two."""
    go = route.census_expected(compiled=False)
    assert (route.GEMV_SYMBOL, telemetry.DECODER_WINDOW_GEMV) in go["decode"]
    assert (route.GEMM_SYMBOL, telemetry.DECODER_TORCH_WINDOW) in go["decode"]
    assert (route.GEMM_SYMBOL, telemetry.DECODER_WINDOW_GEMV) in go["decode"]
    assert go["batch"] == {(route.GEMM_SYMBOL, telemetry.DECODER_TORCH_WINDOW),
                           (route.GEMM_SYMBOL, telemetry.DECODER_WINDOW_GEMV)}
    gc = route.census_expected(compiled=True)
    assert (route.COMPILED_SYMBOL, route.COMPILED_DECODER) in gc["decode"]
    assert (route.COMPILED_SYMBOL, route.COMPILED_DECODER) in gc["batch"]
    assert route.GEMM_SYMBOL == "torch.mm"


def test_m_tile_is_the_kernel_build_rule():
    """The telemetry's tile is the lane's rule, off the lane itself."""
    kg = _kg()
    for m in (1, 2, 3, 4, 5, 8):
        assert route.m_tile(m) == kg._m_tile(m)


def _synthetic_holder(rate_one=False):
    """A one-role holder of small CPU tensors: only the shapes and the meta
    the dispatch reads, for the kernel-free tests below."""
    words = torch.zeros(8, dtype=torch.int32)
    items = torch.zeros(2, 8, dtype=torch.int32)
    perm = torch.arange(32, dtype=torch.int32)
    table = torch.ones(64, dtype=torch.bfloat16)
    scale = torch.ones(16, dtype=torch.float32)
    runs = torch.zeros(1, 4, dtype=torch.int32)
    tensors = (words, items, items[:0].clone(), perm, table, scale, runs)
    meta = (4, 16, 6, 16, 8, 4, 32, 0, int(rate_one), 1, 1)
    role = route._Bf16GemvRole("weight", 0, tensors, meta)
    return route.PreparedBf16Gemv([role], rows=16, columns=32, device=torch.device("cpu"))


def test_decode_is_gemv_is_the_m_rule_in_one_place():
    """M past the lane's max is prefill; M >= 4 over a rate-1 column has no
    8-row lane; everything else in the decode regime is the GEMV."""
    plain = _synthetic_holder(rate_one=False)
    assert route.GEMV_MAX_M == 8
    for m in (1, 2, 3, 4, 5, 8):
        assert route.decode_is_gemv(plain, m), m
    assert not route.decode_is_gemv(plain, 9)
    assert not route.decode_is_gemv(plain, 64)
    racy = _synthetic_holder(rate_one=True)
    assert racy.rate_one
    assert route.decode_is_gemv(racy, 2)
    assert not route.decode_is_gemv(racy, 4)
    assert not route.decode_is_gemv(racy, 8)


def test_streamed_apply_routes_by_m_and_rate_one(monkeypatch):
    """The custom op's dispatch, with the kernel behind fakes: the GEMV branch
    in the decode regime, the kernel-decode + GEMM branch past the lane's max
    and over a rate-1 column at M >= 4.  What the fakes return is distinctive
    per branch, so the assertions read which path ran."""
    kg = _kg()
    calls = []

    def _fake_gemv_concrete(x, *args):
        calls.append("gemv")
        return torch.ones(x.shape[0], int(args[7]), dtype=torch.float32) * 2.0

    class _FakeExt:
        def window_decode(self, words, tile_words, n_tiles, runs, perm, table,
                          window_bits, out):
            calls.append("decode")
            out.fill_(3.0)

    monkeypatch.setattr(kg, "_gemv_concrete", _fake_gemv_concrete)
    monkeypatch.setattr(kg, "_ext", lambda: _FakeExt())
    # ``torch.mm(..., out_dtype=)`` over bf16 is a CUDA kernel; on this CPU box
    # the materialised branch needs the same product spelled promotably.
    _real_mm = torch.mm
    monkeypatch.setattr(torch, "mm",
                        lambda a, b, **kw: _real_mm(a.float(), b.float())
                        if kw.get("out_dtype") is not None else _real_mm(a, b, **kw))

    plain = _synthetic_holder(rate_one=False)
    tensors, meta, rows, cols = plain.op_args()
    x = torch.ones(2, cols, dtype=torch.bfloat16)
    y = route.streamed_apply(x, tensors, meta, rows, cols)
    assert calls == ["gemv"] and tuple(y.shape) == (2, 16) and y.dtype == torch.bfloat16
    assert bool((y == torch.tensor(2.0, dtype=torch.bfloat16)).all())

    calls.clear()
    y64 = route.streamed_apply(torch.ones(64, cols, dtype=torch.bfloat16),
                               tensors, meta, rows, cols)
    assert calls == ["decode"] and tuple(y64.shape) == (64, 16)
    # tile 3.0, scale 1.0: each output is 32 * 3.0, in bf16.
    assert bool((y64 == torch.tensor(96.0, dtype=torch.bfloat16)).all())

    racy = _synthetic_holder(rate_one=True)
    rtensors, rmeta, rrows, rcols = racy.op_args()
    calls.clear()
    y2 = route.streamed_apply(torch.ones(2, cols, dtype=torch.bfloat16),
                              rtensors, rmeta, rrows, rcols)
    assert calls == ["gemv"] and tuple(y2.shape) == (2, 16)
    calls.clear()
    y4 = route.streamed_apply(torch.ones(4, cols, dtype=torch.bfloat16),
                              rtensors, rmeta, rrows, rcols)
    assert calls == ["decode"] and tuple(y4.shape) == (4, 16)


# --- the serve, on a CUDA box ---------------------------------------------------
#
# Everything below drives the route for real.  Without a GPU these skip; the
# CPU tests above are what pin the rule here.


def _install_vllm_stubs(monkeypatch):
    class _LinearMethodBase:
        pass

    def _param(data, **_kw):
        return torch.nn.Parameter(data, requires_grad=False)

    linear = types.ModuleType("vllm.model_executor.layers.linear")
    linear.LinearMethodBase = _LinearMethodBase
    parameter = types.ModuleType("vllm.model_executor.parameter")
    parameter.ModelWeightParameter = _param
    parameter.BasevLLMParameter = _param
    for name, mod in (("vllm", types.ModuleType("vllm")),
                      ("vllm.model_executor", types.ModuleType("vllm.model_executor")),
                      ("vllm.model_executor.layers", types.ModuleType("vllm.model_executor.layers")),
                      ("vllm.model_executor.layers.linear", linear),
                      ("vllm.model_executor.parameter", parameter)):
        monkeypatch.setitem(sys.modules, name, mod)


class _Layer(torch.nn.Module):
    pass


def _scheme(rows, columns, roles, q256, wire_bytes):
    return {"family": TESSERA_BF16, "grid": "BF16", "body": "WINDOW", "plane": "CHANNEL",
            "q256": q256, "rows": rows, "columns": columns, "wire_bytes": wire_bytes,
            "roles": roles}


def _drive(monkeypatch, mode, roles=(("weight", 64),), cols=256, m=4, seed=0, q256=1024):
    serving_lane.reset_for_tests()
    monkeypatch.setenv(TESSERA_MODE_ENV, mode)
    _install_vllm_stubs(monkeypatch)
    fused, export, decode, alphabet = _tessera()
    torch.manual_seed(seed)
    values, scales, blobs = [], [], []
    for i, (name, rows) in enumerate(roles):
        w = torch.randn(rows, cols, device="cuda") * 0.02
        w[: max(1, rows // 8)] *= 2.0 ** (i + 1)
        exported, unit, forests = export.encode_linear_planes(
            w.contiguous(), grid=alphabet.BF16_GRID, q256=q256, name=name, verify=False)
        tile, scale = decode.materialize_bf16(unit, forests, export.DEFAULT_CODE)
        values.append(tile)
        scales.append(scale.reshape(-1))
        blobs.append((name, rows, exported.blob))
    blob = fused.pack_fused(blobs)
    total = sum(r for _, r in roles)
    scheme = _scheme(total, cols, [[n, r] for n, r in roles], q256, len(blob))
    method = build_tessera_method(scheme, "test.layer")
    assert type(method).__name__ == "TesseraBf16LinearMethod"
    layer = _Layer()
    method.create_weights(layer, input_size_per_partition=cols,
                          output_partition_sizes=[r for _, r in roles],
                          input_size=cols, output_size=total, params_dtype=torch.bfloat16)
    layer.wire_bytes.data = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
    layer.to(torch.device("cuda"))
    method.process_weights_after_loading(layer)
    x = torch.randn(m, cols, dtype=torch.bfloat16, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(seed))
    got = method.apply(layer, x)
    return got, layer, method, x, (torch.cat(values), torch.cat(scales))


def _fp32_bound(tile_f32, scale, x):
    """A deterministic fp32 accumulation bound: ``2K * 2^-23 * sum_j |w_ij x_j|``
    (each of the K partial sums is rounded once, in either order; the factor 2
    covers the reference's own rounding of the same size).  The kernel's own
    GEMV tests derive this same bound; it is not a picked tolerance."""
    K = x.shape[1]
    mag = (tile_f32 * scale[:, None]).abs().double() @ x.abs().double().t()
    return (2 * K * 2.0 ** -23) * mag.t() + 1e-30


@requires_cuda
def test_streamed_prepares_the_gemv_holder_and_drops_the_torch_planes(monkeypatch):
    """An in-range unit (R = 4) holds the repacked wire, not the torch planes --
    and the lane's kernel decode is the verified tile, bit for bit."""
    _g, layer, _m, _x, (values, scale) = _drive(monkeypatch, MODE_STREAMED, q256=1024)
    assert layer.tessera_gemv is not None
    assert layer.tessera_prepared is None
    assert not hasattr(layer, "weight_bf16")
    assert layer.tessera_decoder == telemetry.DECODER_WINDOW_GEMV
    got_tile, got_scale = route.holder_decode(layer.tessera_gemv)
    assert torch.equal(got_tile, values.cuda())
    assert torch.equal(got_scale, scale.cuda())


@requires_cuda
def test_an_out_of_range_unit_serves_through_torch_and_says_so(monkeypatch):
    """R = 7 is outside the lane's support set: the module serves exactly as
    before -- torch decode, torch.mm -- and the census says so.  Silently
    producing wrong bytes, or refusing a unit that used to serve, are both
    worse than the status quo."""
    from tessera.serving.telemetry import read_route
    got, layer, _m, x, (values, scale) = _drive(monkeypatch, MODE_STREAMED, q256=1792,
                                                m=2, cols=512)
    assert layer.tessera_gemv is None
    assert layer.tessera_prepared is not None
    rec = read_route(layer)
    assert rec["symbol"] == "torch.mm" and rec["decoder"] == telemetry.DECODER_TORCH_WINDOW
    assert rec["contract"] == route.ACTIVATION_CONTRACT == "bf16_unquantized"
    assert rec["state"] == "served"
    exact = (x.float() @ (values.float().cuda() * scale.float().cuda()[:, None]).t())
    assert bool(((got.float() - exact).abs() <= _fp32_bound(
        values.float().cuda(), scale.float().cuda(), x) + 2.0 ** -8 * exact.abs()).all())


@requires_cuda
def test_resident_does_not_prepare_the_gemv(monkeypatch):
    """The resident lane already holds the tile; the GEMV is the streamed route's."""
    _g, layer, _m, _x, _r = _drive(monkeypatch, MODE_RESIDENT, q256=1024)
    assert layer.tessera_gemv is None
    assert layer.tessera_prepared is None
    assert layer.weight_bf16.dtype == torch.bfloat16


@requires_cuda
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 8])
def test_decode_regime_serves_the_gemv(monkeypatch, m):
    """M <= 8 runs the wire GEMV: the record names the lane, and the answer
    lands within the derived fp32 bound of the exact product.

    The max absolute and relative differences between the two decoders of the
    same bytes are printed, not asserted past the bound: an unexplained
    numerical difference would be a finding, not a rounding detail.
    """
    from tessera.serving.telemetry import read_route
    got, layer, _m, x, (values, scale) = _drive(monkeypatch, MODE_STREAMED, q256=1024,
                                                m=m, seed=11)
    rec = read_route(layer)
    assert rec["symbol"] == route.GEMV_SYMBOL
    assert rec["decoder"] == telemetry.DECODER_WINDOW_GEMV
    assert rec["tile_m"] == route.m_tile(m)
    assert rec["contract"] == route.ACTIVATION_CONTRACT == "bf16_unquantized"
    assert rec["state"] == "served"
    tile, scl = values.float().cuda(), scale.float().cuda()
    exact = (x.float() @ (tile * scl[:, None]).t())
    bound = _fp32_bound(tile, scl, x) + 2.0 ** -8 * exact.abs()
    assert bool(((got.float() - exact).abs() <= bound).all())
    assert layer.tessera_gemv is not None


@requires_cuda
@pytest.mark.parametrize("m", [16, 64])
def test_prefill_on_a_gemv_module_stamps_the_lane_decoder(monkeypatch, m):
    """M > 8 kernel-decodes the tile and runs torch.mm: the GEMV refuses these
    shapes, so the dispatch serves them materialised instead of raising -- and
    the tile came from the lane's kernel decode, so the record names the
    lane's decoder with the route's own GEMM symbol."""
    from tessera.serving.telemetry import read_route
    got, layer, _m, x, (values, scale) = _drive(monkeypatch, MODE_STREAMED, q256=1024,
                                                m=m, seed=12)
    rec = read_route(layer)
    assert rec["symbol"] == route.GEMM_SYMBOL == "torch.mm"
    assert rec["decoder"] == telemetry.DECODER_WINDOW_GEMV
    assert rec["state"] == "served"
    tile, scl = values.float().cuda(), scale.float().cuda()
    exact = (x.float() @ (tile * scl[:, None]).t())
    bound = _fp32_bound(tile, scl, x) + 2.0 ** -8 * exact.abs()
    assert bool(((got.float() - exact).abs() <= bound).all())


@requires_cuda
def test_gemv_and_torch_agree_with_measured_differences(monkeypatch):
    """Same bytes, two engines: the decode-regime GEMV against the torch
    decode + torch.mm on the same module, with the exact max absolute and
    relative differences reported.

    Bit-exactness is asserted where the bytes are exact (the lane's decode of
    the tile); the two fp32 products may differ by summation order, so they
    are held to the derived fp32 bound -- and the measured differences are
    printed for the record.
    """
    _g, layer, _m, x, (values, scale) = _drive(monkeypatch, MODE_STREAMED, q256=1024,
                                               m=4, seed=13)
    holder = layer.tessera_gemv
    assert holder is not None
    tensors, meta, rows, cols = holder.op_args()
    y_gemv = route.streamed_apply(x.contiguous(), tensors, meta, rows, cols)
    tile, scl = values.cuda(), scale.float().cuda()
    y_torch = (torch.mm(x.contiguous(), tile.t(), out_dtype=torch.float32)
               * scl).to(torch.bfloat16)
    absdiff = (y_gemv.float() - y_torch.float()).abs()
    max_abs = float(absdiff.max())
    denom = float(y_torch.float().abs().max())
    print(f"\nBF16 GEMV-vs-torch on 64x256 R=4 M=4: max_abs={max_abs:.3e} "
          f"max_rel={max_abs / max(denom, 1e-9):.3e}")
    ref = (x.float() @ (tile.float() * scl[:, None]).t())
    bound = _fp32_bound(tile.float(), scl, x) + 2.0 ** -8 * ref.abs()
    assert bool(((y_gemv.float() - ref).abs() <= bound).all())
    assert bool(((y_torch.float() - ref).abs() <= bound).all())


@requires_cuda
def test_rate1_columns_fall_back_inside_the_decode_regime(monkeypatch):
    """M >= 4 over a rate-1 column has no 8-row lane: the dispatch serves it
    materialised instead of raising, at M <= 8 -- still on the lane's decode."""
    from tessera.serving.telemetry import read_route
    got2, layer, _m, _x, _r = _drive(monkeypatch, MODE_STREAMED, q256=256, m=2, seed=20)
    assert layer.tessera_gemv is not None and layer.tessera_gemv.rate_one
    assert read_route(layer)["symbol"] == route.GEMV_SYMBOL
    assert tuple(got2.shape) == (2, layer.tessera_rows)
    got4, _l, _m2, _x2, _r2 = _drive(monkeypatch, MODE_STREAMED, q256=256, m=4, seed=21)
    rec = read_route(_l)
    assert rec["symbol"] == route.GEMM_SYMBOL == "torch.mm"
    assert rec["decoder"] == telemetry.DECODER_WINDOW_GEMV
    assert tuple(got4.shape) == (4, _l.tessera_rows)


@requires_cuda
def test_without_the_extension_streamed_falls_back_to_the_torch_path(monkeypatch):
    """No toolchain, no GEMV: the streamed route serves exactly as before, by name."""
    from tessera import kernel_window_gemv
    from tessera.serving.telemetry import read_route

    def _no_toolchain():
        raise RuntimeError("no nvcc on this box")

    monkeypatch.setattr(kernel_window_gemv, "_ext", _no_toolchain)
    got, layer, _m, _x, _r = _drive(monkeypatch, MODE_STREAMED, q256=1024, m=2, seed=14)
    assert layer.tessera_gemv is None
    assert layer.tessera_prepared is not None
    rec = read_route(layer)
    assert rec["symbol"] == "torch.mm" and rec["decoder"] == telemetry.DECODER_TORCH_WINDOW


@requires_cuda
def test_the_dispatch_survives_a_compiled_forward_with_a_dynamic_token_dim(monkeypatch):
    """One graph serves M = 1..8 (GEMV) and M = 64 (materialised) without a
    recompile: no int() on the token dim in the trace."""
    from tessera.serving.telemetry import read_route
    torch._dynamo.reset()
    _got, layer, method, _x, _r = _drive(monkeypatch, MODE_STREAMED, q256=1024,
                                         m=2, seed=15)
    assert layer.tessera_gemv is not None
    compiled = torch.compile(lambda x: method.apply(layer, x))

    def x_for(M, seed):
        x = torch.randn(M, layer.tessera_columns, device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(seed)).bfloat16()
        torch._dynamo.decorators.mark_unbacked(x, 0)
        return x

    for i, M in enumerate((2, 1, 4, 8, 64)):
        with torch._dynamo.config.patch(error_on_recompile=(i > 0)):
            y = compiled(x_for(M, 500 + M))
        assert tuple(y.shape) == (M, layer.tessera_rows) and y.dtype == torch.bfloat16
    rec = read_route(layer)
    assert rec["shape"].startswith("M*:")
    assert rec["symbol"] == route.COMPILED_SYMBOL
    assert rec["decoder"] == route.COMPILED_DECODER
