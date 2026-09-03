"""The Tessera FP8 W8A8 dense serving route.

Exercised for real on a CUDA box with ``tessera`` importable: the container
parse, the packed-window decode, the reference-decoder cross-check at
preparation, the per-row ``scale_b``, ``torch._scaled_mm`` W8A8, both
residency modes, the compiled decode and the refusals.  The load-bearing
assertion is BYTE identity of the decoded pair with ``tessera.stock.
materialize_stock`` -- the tensors the compressed-tensors stock lane served
on vanilla vLLM -- so the numbers this route produces are, by construction,
the stock lane's served numbers.  STUBBED: vLLM's ``LinearMethodBase`` /
parameters, the per-token FP8 activation quantiser (an audited upstream op)
and the ABI attestation that would otherwise reach for the real vLLM operator
library.  vLLM loading this route owes a container run, as for every route.

Ported from Gridbook's ``test_tessera_fp8_lane.py``; the enable flag is gone
(the checkpoint selects the plugin) and the residency is the only flag left.
"""
from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from tessera.serving import fp8_route as route                       # noqa: E402
from tessera.serving import lane as serving_lane                     # noqa: E402
from tessera.serving import native_ops, telemetry                    # noqa: E402
from tessera.serving.lane import (                                   # noqa: E402
    MODE_RESIDENT, MODE_STREAMED, TESSERA_MODE_ENV, build_tessera_method)
from tessera.serving.scheme import (                                 # noqa: E402
    TESSERA_FP8, TESSERA_NVFP4, validate_tessera_scheme)

CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA device")


def _tessera():
    return (pytest.importorskip("tessera.fused"), pytest.importorskip("tessera.export"),
            pytest.importorskip("tessera.stock"), pytest.importorskip("tessera.alphabet"))


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


# --- the scheme --------------------------------------------------------------

def test_fp8_scheme_normalises_and_refuses_the_other_routes_vocabulary():
    norm = validate_tessera_scheme(_scheme(roles=[["q_proj", 128], ["k_proj", 128]]), "t")
    assert norm["family"] == TESSERA_FP8 and norm["plane"] == "CHANNEL"
    assert norm["roles"] == [("q_proj", 128), ("k_proj", 128)]
    with pytest.raises(ValueError, match="scalar E4M3 grid"):
        validate_tessera_scheme(_scheme(grid="E2M1x2"), "t")
    with pytest.raises(ValueError, match="no FP8 tile"):
        validate_tessera_scheme(_scheme(plane="LUT"), "t")
    with pytest.raises(ValueError, match="K % 16"):
        validate_tessera_scheme(_scheme(columns=1000), "t")
    with pytest.raises(ValueError, match="no NVFP4 tile"):
        validate_tessera_scheme({**_scheme(), "family": TESSERA_NVFP4, "grid": "E2M1x2"}, "t")


def test_the_route_refuses_before_vllm_and_the_family_picks_the_route(monkeypatch):
    """FAMILY = ROUTE.  The checkpoint's fact, not the operator's: the only
    thing left for the operator to declare is the residency."""
    assert not hasattr(serving_lane, "TESSERA_FLAG")
    with pytest.raises(ValueError, match=TESSERA_MODE_ENV):
        build_tessera_method(_scheme(), "test.layer")
    serving_lane.reset_for_tests()
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    with pytest.raises(ValueError, match="family must be one of"):
        build_tessera_method({**_scheme(), "family": "TESSERA_INT4"}, "test.layer")
    with pytest.raises(ValueError, match=f"serves {TESSERA_FP8}, not"):
        # A COHERENT NVFP4 scheme, handed to the wrong builder.  q256 is 896
        # because that is the only rate the E2M1x2 reader takes -- leaving the
        # FP8 default here would be refused by the rung gate first, which is a
        # true refusal but not the one under test.
        route.build_tessera_fp8_method({**_scheme(), "family": TESSERA_NVFP4, "grid": "E2M1x2",
                                        "plane": "LUT", "body": "TCQ", "q256": 896},
                                       "test.layer", "resident")


def test_the_fp8_routes_decoder_is_pure_torch():
    """No CUDA extension at all on this route: the window decoder is torch."""
    assert route.ACTIVATION_CONTRACT == "fp8_per_token_dynamic"
    assert telemetry.DECODER_TORCH_WINDOW in telemetry.DECODERS


# --- the numerics ------------------------------------------------------------

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
    """Per-token dynamic E4M3 (the arithmetic of vLLM's op), recording the value
    it represents so the expectation is the A side the route consumed."""
    xf = x.float()
    amax = xf.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = amax / FP8_MAX
    q = (xf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    _LAST_A["value"] = q.float() * scale
    return q, scale.contiguous()


class _Layer(torch.nn.Module):
    pass


def _encode_module(roles, cols=1024, q256=1024, seed=0):
    """Encode ``roles`` = [(name, rows)] with Tessera on the E4M3 grid; return
    the container blob, the scheme and the stock reference pair."""
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
    # The residency is latched to the first value this process read, so a test
    # that drives both modes clears it here rather than only between tests.
    serving_lane.reset_for_tests()
    monkeypatch.setenv(TESSERA_MODE_ENV, mode)
    _install_vllm_stubs(monkeypatch)
    monkeypatch.setattr(native_ops, "native_fp8_quant", _reference_fp8_quant)
    # With sys.modules['vllm'] stubbed, the real operator library is not
    # importable; record that the route ATTESTS the ABI rather than executing
    # an attestation the stub cannot satisfy.  See the report.
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


@requires_cuda
@pytest.mark.parametrize("mode", [MODE_RESIDENT, MODE_STREAMED])
def test_pair_is_the_stock_pair_byte_for_byte(monkeypatch, mode):
    """The decoded bytes and per-row scale ARE materialize_stock's."""
    got, want, layer, method, (weight, scale) = _drive(monkeypatch, mode)
    if mode == MODE_RESIDENT:
        tile = layer.weight_fp8.view(torch.uint8)
    elif getattr(layer, "tessera_gemv", None) is not None:
        # The GEMV lane verified its repack against these same bytes at load;
        # read them back through the lane's own decode.
        from tessera.serving import fp8_gemv as _gemv
        tile, lane_scale = _gemv.holder_decode(layer.tessera_gemv)
        assert torch.equal(lane_scale, scale)
    else:
        tile = layer.tessera_prepared.decode()
    assert torch.equal(tile, weight)
    assert torch.equal(layer.scale_b.reshape(-1), scale)
    assert layer.scale_b.dtype == torch.float32 and tuple(layer.scale_b.shape) == (1, weight.shape[0])
    err = (got.float() - want.float()).abs().max().item()
    assert err / max(want.float().abs().max().item(), 1e-9) < 8e-3


@requires_cuda
def test_the_route_attests_the_native_a_side_abi(monkeypatch):
    _drive(monkeypatch, MODE_RESIDENT)
    assert len(_ATTESTED) == 1 and "test.layer" in _ATTESTED[0]


@requires_cuda
def test_fused_roles_decode_into_row_slices_with_their_own_row_scales(monkeypatch):
    roles = (("q_proj", 256), ("k_proj", 128), ("v_proj", 128))
    got, want, layer, _m, (weight, scale) = _drive(monkeypatch, MODE_RESIDENT, roles=roles, seed=3)
    assert layer.tessera_roles == ("q_proj", "k_proj", "v_proj")
    assert torch.equal(layer.weight_fp8.view(torch.uint8), weight)
    assert torch.equal(layer.scale_b.reshape(-1), scale)
    err = (got.float() - want.float()).abs().max().item()
    assert err / max(want.float().abs().max().item(), 1e-9) < 8e-3


@requires_cuda
def test_a_mixed_rate_schedule_decodes_through_the_group_permutation(monkeypatch):
    got, want, layer, _m, (weight, scale) = _drive(monkeypatch, MODE_STREAMED, cols=512, seed=4, q256=1000)
    prepared = layer.tessera_prepared
    assert torch.equal(prepared.decode(), weight)


@requires_cuda
def test_the_two_modes_are_numerically_identical(monkeypatch):
    a, _w, _l, _m, _ = _drive(monkeypatch, MODE_RESIDENT, seed=7)
    b, _w2, _l2, _m2, _ = _drive(monkeypatch, MODE_STREAMED, seed=7)
    assert torch.equal(a, b)


@requires_cuda
def test_streamed_holds_the_packed_wire_and_no_resident_tile(monkeypatch):
    _g, _w, a, _m, _ = _drive(monkeypatch, MODE_STREAMED, roles=(("weight", 512),), cols=512)
    for name in ("wire_bytes", "weight_fp8", "decode_buf"):
        assert not hasattr(a, name), name
    if getattr(a, "tessera_gemv", None) is not None:
        # The repacked wire, not the torch planes and not a tile: rows are a
        # multiple of the lane's 512-row tile here, so no padding enters and
        # the same wire-vs-tile cap the torch planes met applies.
        from tessera.serving import fp8_gemv as _gemv
        holder = a.tessera_gemv
        assert holder is not None and a.tessera_prepared is None
        assert holder.resident_bytes() < 512 * 512 * 4.5 / 8 + 65536
        p1, p2 = _gemv.holder_decode(holder), _gemv.holder_decode(holder)
        assert torch.equal(p1[0], p2[0]) and p1[0].data_ptr() != p2[0].data_ptr()
        return
    prepared = a.tessera_prepared
    assert prepared is not None
    assert prepared.wire_bytes_resident() < 512 * 512 * 4.5 / 8 + 65536   # ~ the wire, not the 8-bit tile
    p1, p2 = prepared.decode(), prepared.decode()
    assert torch.equal(p1, p2) and p1.data_ptr() != p2.data_ptr()


@requires_cuda
def test_resident_drops_the_wire(monkeypatch):
    _g, _w, r, _m, _ = _drive(monkeypatch, MODE_RESIDENT)
    assert not hasattr(r, "wire_bytes") and r.tessera_prepared is None
    assert r.weight_fp8.dtype == torch.float8_e4m3fn
    assert tuple(r.weight_fp8.shape) == (r.tessera_rows, r.tessera_columns)


@requires_cuda
def test_streamed_decode_traces_under_torch_compile(monkeypatch):
    _g, _w, layer, _m, (weight, _s) = _drive(monkeypatch, MODE_STREAMED, cols=512, seed=9, q256=1000)
    prepared = layer.tessera_prepared
    compiled = torch.compile(prepared.decode, fullgraph=True)
    assert torch.equal(compiled(), weight)


@requires_cuda
def test_preparation_refuses_a_decoder_that_disagrees_with_the_reference(monkeypatch):
    import tessera.decode as tdecode
    from tessera.serving.scheme import parse_tessera_blob_for_scheme
    blob, scheme, *_ = _encode_module([("weight", 128)], cols=512)
    roles = parse_tessera_blob_for_scheme(blob, scheme, "t")
    real = tdecode.materialize_fp8

    def _wrong(unit, forest, code):
        b, s = real(unit, forest, code)
        b = b.clone(); b[0, 0] ^= 1
        return b, s

    monkeypatch.setattr(tdecode, "materialize_fp8", _wrong)
    with pytest.raises(RuntimeError, match="disagrees with tessera.decode.materialize_fp8 on 1 of"):
        route.prepare_tessera_fp8_module(roles, device="cuda")


@requires_cuda
def test_scheme_and_blob_must_agree(monkeypatch):
    from tessera.serving.scheme import parse_tessera_blob_for_scheme
    blob, scheme, *_ = _encode_module([("weight", 128)], cols=512)
    parse_tessera_blob_for_scheme(blob, scheme, "t")
    with pytest.raises(ValueError, match="sidecar scheme declares"):
        parse_tessera_blob_for_scheme(blob, {**scheme, "q256": 896}, "t")
    with pytest.raises(ValueError, match="sidecar scheme declares"):
        parse_tessera_blob_for_scheme(blob, {**scheme, "body": "TCQ"}, "t")


@requires_cuda
def test_route_record_names_the_family_mode_contract_and_decoder(monkeypatch):
    from tessera.serving.telemetry import read_route
    _g, _w, layer, _m, _ = _drive(monkeypatch, MODE_STREAMED)
    rec = read_route(layer)
    assert rec is not None and rec["policy"] == f"{TESSERA_FP8}:streamed" and rec["state"] == "served"
    assert rec["contract"] == route.ACTIVATION_CONTRACT == "fp8_per_token_dynamic"
    assert rec["symbol"] == "torch._scaled_mm"
    # The drive above is a 32-row prefill: the tile path either way, but the
    # tile's producer is the lane that prepared the module.
    if getattr(layer, "tessera_gemv", None) is not None:
        assert rec["decoder"] == telemetry.DECODER_WINDOW_GEMV == layer.tessera_decoder
    else:
        assert rec["decoder"] == telemetry.DECODER_TORCH_WINDOW == layer.tessera_decoder
