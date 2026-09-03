"""The streamed FP8 route's decode-regime GEMV: the wire read once, never materialised.

Wires ``tessera.kernel_window_gemv`` (the fused window-body GEMV, bit-exact on
196/196 reach units) into the FP8 route's ``apply()`` for M <= 8, keeping the
materialised decode + ``torch._scaled_mm`` path for prefill.  The activation
contract does NOT change: the GEMV runs on the dequantised per-token-dynamic
FP8 values, so what it computes is the W8A8 product up to fp32 summation
order, and the census contract field is untouched.

STUBBED like its sibling: vLLM's ``LinearMethodBase`` / parameters, the
per-token FP8 activation quantiser and the ABI attestation.
"""
from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from tessera.serving import fp8_route as route                       # noqa: E402
from tessera.serving import fp8_gemv                                 # noqa: E402
from tessera.serving import lane as serving_lane                     # noqa: E402
from tessera.serving import native_ops, telemetry                    # noqa: E402
from tessera.serving.lane import (                                   # noqa: E402
    MODE_RESIDENT, MODE_STREAMED, TESSERA_MODE_ENV, build_tessera_method)
from tessera.serving.scheme import TESSERA_FP8                       # noqa: E402

CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA device")


@pytest.fixture(autouse=True)
def _fresh_env(monkeypatch):
    serving_lane.reset_for_tests()
    monkeypatch.delenv(TESSERA_MODE_ENV, raising=False)
    yield
    serving_lane.reset_for_tests()


def _scheme(rows=256, columns=1024, roles=None, **over):
    s = {"family": TESSERA_FP8, "grid": "E4M3", "body": "WINDOW", "plane": "CHANNEL", "q256": 1024,
         "rows": rows, "columns": columns, "wire_bytes": 4096,
         "roles": roles if roles is not None else [["weight", rows]]}
    s.update(over)
    return s


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


_LAST_A = {}
_ATTESTED = []
FP8_MAX = 448.0


def _reference_fp8_quant(x):
    xf = x.float()
    amax = xf.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = amax / FP8_MAX
    q = (xf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    _LAST_A["value"] = q.float() * scale
    return q, scale.contiguous()


class _Layer(torch.nn.Module):
    pass


def _tessera():
    return (pytest.importorskip("tessera.fused"), pytest.importorskip("tessera.export"),
            pytest.importorskip("tessera.stock"), pytest.importorskip("tessera.alphabet"))


def _encode_module(roles, cols=1024, q256=1024, seed=0):
    fused, export, stock, alphabet = _tessera()
    torch.manual_seed(seed)
    tensors, blobs = {}, []
    for i, (name, rows) in enumerate(roles):
        w = (torch.randn(rows, cols, device="cuda") * 0.02)
        w[: rows // 8] *= 2.0 ** (i + 1)
        exported, unit, forests = export.encode_linear_planes(
            w.contiguous(), grid=alphabet.E4M3_GRID, q256=q256, name=name, verify=False)
        tensors[name] = stock.materialize_stock(unit, forests, export.DEFAULT_CODE)
        blobs.append((name, rows, exported.blob))
    blob = fused.pack_fused(blobs)
    scheme = _scheme(rows=sum(r for _, r in roles), columns=cols, wire_bytes=len(blob),
                     roles=[[n, r] for n, r in roles], q256=q256)
    weight = torch.cat([tensors[n]["weight"].view(torch.uint8) for n, _ in roles])
    scale = torch.cat([tensors[n]["weight_scale"].reshape(-1) for n, _ in roles])
    ref_w = torch.cat([stock.stock_dequant(tensors[n]) for n, _ in roles])
    return blob, scheme, weight, scale, ref_w


def _drive(monkeypatch, mode, roles=(("weight", 256),), cols=1024, m=32, seed=0, q256=1024):
    serving_lane.reset_for_tests()
    monkeypatch.setenv(TESSERA_MODE_ENV, mode)
    _install_vllm_stubs(monkeypatch)
    monkeypatch.setattr(native_ops, "native_fp8_quant", _reference_fp8_quant)
    _ATTESTED.clear()
    monkeypatch.setattr(native_ops, "require_native_fp8_quant",
                        lambda context: _ATTESTED.append(context))
    blob, scheme, weight, scale, ref_w = _encode_module(list(roles), cols=cols, seed=seed, q256=q256)
    method = build_tessera_method(scheme, "test.layer")
    assert type(method).__name__ == "TesseraFp8LinearMethod"
    layer = _Layer()
    rows = scheme["rows"]
    method.create_weights(layer, input_size_per_partition=cols, output_partition_sizes=[r for _, r in roles],
                          input_size=cols, output_size=rows, params_dtype=torch.bfloat16)
    layer.wire_bytes.data = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
    layer.to(torch.device("cuda"))
    method.process_weights_after_loading(layer)
    x = torch.randn(m, cols, dtype=torch.bfloat16, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(seed))
    got = method.apply(layer, x)
    want = (_LAST_A["value"] @ ref_w.t()).to(torch.bfloat16)
    return got, want, layer, method, (weight, scale)


def _fp32_bound(w_f32, scale_f32, x_bf16):
    """The kernel lane's own accumulation bound: ``2K * 2^-23 * sum_j |w_ij x_j|``."""
    K = x_bf16.shape[1]
    mag = (w_f32 * scale_f32[:, None]).abs().double() @ x_bf16.abs().double().t()
    return (2 * K * 2.0 ** -23) * mag.t() + 1e-30


def _bf16_tol(bound, ref):
    """The fp32 bound plus one bf16 rounding of the reference: two fp32 sums
    within the bound round to bf16 values at most one ulp apart (the straddle),
    the same tolerance the kernel lane's own seam test holds."""
    return 2 * bound + 2.0 ** -7 * ref.abs()


# --- the wiring --------------------------------------------------------------

@requires_cuda
def test_streamed_prepares_the_gemv_holder_and_drops_the_torch_planes(monkeypatch):
    """The streamed route holds the repacked wire, not the torch window planes."""
    _g, _w, layer, _m, (weight, scale) = _drive(monkeypatch, MODE_STREAMED)
    assert layer.tessera_gemv is not None
    assert layer.tessera_prepared is None
    assert not hasattr(layer, "weight_fp8")
    got_bytes, got_scale = fp8_gemv.holder_decode(layer.tessera_gemv)
    assert torch.equal(got_bytes, weight)
    assert torch.equal(got_scale, scale)


@requires_cuda
def test_resident_does_not_prepare_the_gemv(monkeypatch):
    """The resident lane already holds the tile; the GEMV is the streamed route's."""
    _g, _w, layer, _m, _ = _drive(monkeypatch, MODE_RESIDENT)
    assert layer.tessera_gemv is None
    assert layer.tessera_prepared is None
    assert layer.weight_fp8.dtype == torch.float8_e4m3fn


@requires_cuda
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 8])
def test_decode_regime_serves_the_gemv(monkeypatch, m):
    """M <= 8 runs the wire GEMV on the quantised activations, at the route's own bar."""
    from tessera.serving.telemetry import read_route
    got, want, layer, _m, _ = _drive(monkeypatch, MODE_STREAMED, m=m, seed=11)
    rec = read_route(layer)
    assert rec["symbol"] == fp8_gemv.GEMV_SYMBOL
    assert rec["decoder"] == telemetry.DECODER_WINDOW_GEMV
    assert rec["contract"] == route.ACTIVATION_CONTRACT == "fp8_per_token_dynamic"
    assert rec["state"] == "served"
    err = (got.float() - want.float()).abs().max().item()
    assert err / max(want.float().abs().max().item(), 1e-9) < 8e-3


@requires_cuda
@pytest.mark.parametrize("m", [9, 32, 64])
def test_prefill_keeps_the_materialised_path(monkeypatch, m):
    """M > 8 decodes the tile and runs _scaled_mm: the GEMV refuses these shapes.

    The tile comes from the lane's kernel decode (the dispatch never
    materialises through the torch decoder), so the record names the lane's
    decoder with the route's own GEMM symbol."""
    from tessera.serving.telemetry import read_route
    got, want, layer, _m, _ = _drive(monkeypatch, MODE_STREAMED, m=m, seed=12)
    rec = read_route(layer)
    assert rec["symbol"] == route.GEMM_SYMBOL == "torch._scaled_mm"
    assert rec["decoder"] == telemetry.DECODER_WINDOW_GEMV
    assert rec["state"] == "served"
    err = (got.float() - want.float()).abs().max().item()
    assert err / max(want.float().abs().max().item(), 1e-9) < 8e-3


@requires_cuda
def test_gemv_and_materialised_agree_within_fp32_summation_order(monkeypatch):
    """Same weights (bit-exact) and same A values: only the fp32 reduction order differs."""
    _g, _w, layer, _m, (weight, scale) = _drive(monkeypatch, MODE_STREAMED, m=4, seed=13)
    holder = layer.tessera_gemv
    g = torch.Generator(device="cuda").manual_seed(13)
    x = torch.randn(4, layer.tessera_columns, device="cuda", generator=g).bfloat16()
    a_q, a_scale = _reference_fp8_quant(x.contiguous())
    y_gemv = fp8_gemv.streamed_apply(a_q, a_scale, layer.scale_b, *holder.op_args())
    b = weight.view(torch.float8_e4m3fn)
    y_mm = torch._scaled_mm(a_q, b.t(), scale_a=a_scale, scale_b=layer.scale_b,
                            out_dtype=torch.bfloat16)
    w = weight.view(torch.float8_e4m3fn).float()
    bound = _fp32_bound(w, scale.float().cuda(), x)
    tol = _bf16_tol(bound, y_mm.double())
    assert bool((((y_gemv.float() - y_mm.float()).abs()) <= tol).all())


@requires_cuda
def test_without_the_extension_streamed_falls_back_to_the_torch_path(monkeypatch):
    """No toolchain, no GEMV: the streamed route serves exactly as before, by name."""
    from tessera import kernel_window_gemv
    from tessera.serving.telemetry import read_route

    def _no_toolchain():
        raise RuntimeError("no nvcc on this box")

    monkeypatch.setattr(kernel_window_gemv, "_ext", _no_toolchain)
    got, want, layer, _m, _ = _drive(monkeypatch, MODE_STREAMED, m=2, seed=14)
    assert layer.tessera_gemv is None
    assert layer.tessera_prepared is not None
    rec = read_route(layer)
    assert rec["symbol"] == "torch._scaled_mm" and rec["decoder"] == telemetry.DECODER_TORCH_WINDOW
    err = (got.float() - want.float()).abs().max().item()
    assert err / max(want.float().abs().max().item(), 1e-9) < 8e-3


def _synthetic_parsed(rows, cols, rates, seed=0):
    """A ParsedUnit stand-in carrying a raw window body (only what prepare reads)."""
    from types import SimpleNamespace

    from tessera.alphabet import E4M3_GRID
    from tessera.manifest import BodyKind, RotationState, ScalePlaneKind

    g = torch.Generator(device="cpu").manual_seed(seed)
    rate = torch.tensor(list(rates), dtype=torch.int64)
    body = (torch.randint(0, 1 << 16, (rows, cols), generator=g) & ((1 << rate) - 1)).to(torch.uint8)
    codes = torch.randint(0, 256, (1 << 14,), generator=g).to(torch.uint8)
    scale_rows = (torch.rand(rows, generator=g, dtype=torch.float16) + 0.5)
    unit = SimpleNamespace(
        body=BodyKind.WINDOW, scale_plane=ScalePlaneKind.CHANNEL,
        release_index=torch.zeros(0, dtype=torch.int64), diagonals=None,
        rotation=RotationState.NONE, window_codes=codes.cpu(), window_bits=14,
        scale_rows=scale_rows, scale_global=0.75, body_bits=body, rates=tuple(rates),
        initial_state=None, span=1)
    return SimpleNamespace(name="weight", unit=unit, grid=E4M3_GRID,
                           forests=None, code=None, body=BodyKind.WINDOW)


@requires_cuda
def test_rate1_columns_fall_back_inside_the_decode_regime():
    """M >= 4 over a rate-1 column has no 8-row lane: the dispatch serves it
    materialised instead of raising, at M <= 8."""
    cols = 64
    rates = tuple(1 if c % 4 == 0 else 4 for c in range(cols))
    parsed = _synthetic_parsed(512, cols, rates)
    expected = fp8_gemv.reference_bytes_for_test(parsed)
    holder = fp8_gemv.prepare_fp8_gemv([("weight", parsed)], device="cuda", expected=expected)
    assert holder.rate_one
    x2 = torch.randn(2, cols, device="cuda").bfloat16()
    a_q, a_scale = _reference_fp8_quant(x2.contiguous())
    scale_b = expected[1].view(1, -1)
    y2 = fp8_gemv.streamed_apply(a_q, a_scale, scale_b, *holder.op_args())
    assert y2.shape == (2, 512)
    x4 = torch.randn(4, cols, device="cuda").bfloat16()
    b_q, b_scale = _reference_fp8_quant(x4.contiguous())
    y4 = fp8_gemv.streamed_apply(b_q, b_scale, scale_b, *holder.op_args())
    assert y4.shape == (4, 512)
    w = expected[0].view(torch.float8_e4m3fn).float()
    # The reference runs the SAME quantised activations the dispatch consumed
    # (``_reference_fp8_quant`` records the values it represents): raw-x would
    # charge the test the quantiser's own error.  Each path is referenced in
    # the rounding IT consumes: the GEMV path rounds the dequant to bf16
    # before the kernel reads it, while ``_scaled_mm`` multiplies the fp8
    # values directly -- half a bf16 ulp per element either way, which
    # cancellation amplifies past any accumulation bound.
    cases = ((y2, (a_q.float() * a_scale).to(torch.bfloat16)),
             (y4, b_q.float() * b_scale))
    for y, a_val in cases:
        ref = ((w.double() * expected[1].double()[:, None]) @ a_val.double().t()).t()
        assert bool(((y.double() - ref).abs()
                     <= _bf16_tol(_fp32_bound(w, expected[1], a_val), ref)).all())


@requires_cuda
def test_the_dispatch_survives_a_compiled_forward_with_a_dynamic_token_dim(monkeypatch):
    """One graph serves M = 1..8 (GEMV) and M = 64 (materialised) without a
    recompile: no int() on the token dim in the trace."""
    from tessera.serving.telemetry import read_route
    torch._dynamo.reset()
    # _drive is a plain helper (no decorator); call it directly.
    _got, _want, layer, method, _ = _drive(monkeypatch, MODE_STREAMED, m=2, seed=15)
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
    assert rec["symbol"] == fp8_gemv.COMPILED_SYMBOL
    assert rec["decoder"] == fp8_gemv.COMPILED_DECODER


# --- the contract publishes what the serve loads --------------------------------

def test_the_gemv_prefix_is_the_constant_the_load_path_asks_for():
    """The published prefix IS the module name the JIT load asks for (source, not value)."""
    from pathlib import Path

    from tessera.serving import ext
    assert fp8_gemv.GEMV_MODULE_NAME == "tessera_window_gemv"
    entry = next(e for e in ext.NATIVE_EXTENSIONS
                 if e["module_name_prefix"] == fp8_gemv.GEMV_MODULE_NAME)
    assert entry["filename_glob"] == "tessera_window_gemv*.so"
    src = (Path(__file__).resolve().parents[1] / "src" / "tessera" / "kernel_window_gemv.py"
           ).read_text(encoding="utf-8")
    assert 'name="tessera_window_gemv"' in src


def test_the_serving_csrc_is_the_kernel_source_byte_for_byte():
    """The contract's published source is the kernel the serve builds, not a copy
    that drifted: byte-identical to the library's own csrc."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "src" / "tessera"
    assert (root / "serving" / "csrc" / "window_gemv.cu").read_bytes() == \
        (root / "csrc" / "window_gemv.cu").read_bytes()


def test_the_gemv_fallback_the_table_publishes_is_the_one_the_route_takes():
    """Both residencies substitute the torch window decode without the .so."""
    from tessera.serving import ext
    assert ext.substitutes_when_unavailable(MODE_STREAMED, fp8_gemv.GEMV_MODULE_NAME) is True
    assert ext.substitutes_when_unavailable(MODE_RESIDENT, fp8_gemv.GEMV_MODULE_NAME) is True
    entry = next(e for e in ext.NATIVE_EXTENSIONS
                 if e["module_name_prefix"] == fp8_gemv.GEMV_MODULE_NAME)
    assert entry["routes"] == [TESSERA_FP8]
    assert entry["when_unavailable"]["streamed"]["decoder"] in telemetry.DECODERS
    assert entry["when_unavailable"]["resident"]["decoder"] in telemetry.DECODERS
    assert entry["when_unavailable"]["streamed"]["decoder"] == telemetry.DECODER_TORCH_WINDOW
    # The value the fallback actually stamps: the torch window decode, which is
    # what the route serves without the lane.
    src = open(route.__file__).read()
    assert "substitutes_when_unavailable" in src and "fp8_gemv" in src


def test_the_census_expectations_come_from_the_route():
    """The decode phase can report either path the dispatch takes; batch only one."""
    go = fp8_gemv.census_expected(compiled=False)
    assert (fp8_gemv.GEMV_SYMBOL, telemetry.DECODER_WINDOW_GEMV) in go["decode"]
    assert (route.GEMM_SYMBOL, telemetry.DECODER_TORCH_WINDOW) in go["decode"]
    assert (route.GEMM_SYMBOL, telemetry.DECODER_WINDOW_GEMV) in go["decode"]
    assert go["batch"] == {(route.GEMM_SYMBOL, telemetry.DECODER_TORCH_WINDOW),
                           (route.GEMM_SYMBOL, telemetry.DECODER_WINDOW_GEMV)}
    gc = fp8_gemv.census_expected(compiled=True)
    assert (fp8_gemv.COMPILED_SYMBOL, fp8_gemv.COMPILED_DECODER) in gc["decode"]
    assert (fp8_gemv.COMPILED_SYMBOL, fp8_gemv.COMPILED_DECODER) in gc["batch"]
