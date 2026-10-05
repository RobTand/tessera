"""Decode-once E4M3 weights for large-M dense prefill (tessera#931).

What this pins, on encoded modules (``export.encode_linear_planes`` on the E4M3
grid, packed and prepared the way a serve loads them):

* the decoded bytes are the stock materialisation's (``stock.materialize_stock``),
  value for value, and the scale is the module's fp32 row scale bitwise;
* the scaled FP8 GEMM over them sits inside the derived bound of the fp64
  definition (``fused_bound.dense_bound``) and within twice it of the served
  window lane, at prefill and small M alike.

The bound is charged with a K split equal to K (``s = cols``): every product
its own partial, which covers any summation order cuBLAS may choose, so the
test does not depend on which FP8 kernel the library picks.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fused_bound as fb                                   # noqa: E402
from test_dense_fused_window import _module, _scheme, _tessera  # noqa: E402
from test_window_gemm_grouped import _quant                # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the lane is a CUDA GEMM")


def _encode(roles, cols, q256=1024, seed=0):
    """``(blob, scheme, ref bytes e4m3 [rows, cols], ref scale fp32 [rows])``."""
    fused, export, stock, _decode, alphabet = _tessera()
    torch.manual_seed(seed)
    blobs, weights, scales = [], [], []
    for i, (name, rows) in enumerate(roles):
        w = torch.randn(rows, cols, device="cuda") * 0.02
        w[: max(1, rows // 8)] *= 2.0 ** (i + 1)
        exported, unit, forests = export.encode_linear_planes(
            w.contiguous(), grid=alphabet.E4M3_GRID, q256=q256, name=name, verify=False)
        tiles = stock.materialize_stock(unit, forests, export.DEFAULT_CODE)
        weights.append(tiles["weight"].to("cuda"))
        scales.append(tiles["weight_scale"].to("cuda").float().reshape(-1))
        blobs.append((name, rows, exported.blob))
    blob = fused.pack_fused(blobs)
    scheme = _scheme("e4m3", sum(r for _, r in roles), cols, roles, len(blob), q256)
    return blob, scheme, torch.cat(weights), torch.cat(scales)


ROLES = [("q_proj", 256), ("k_proj", 128), ("b_proj", 32)]


@cuda
@pytest.mark.parametrize("q256", [1024, 1088, 832])
def test_decode_once_reproduces_the_stock_bytes_and_row_scale(q256):
    from tessera.serving.e4m3_prefill import decode_e4m3

    blob, scheme, ref_w, ref_s = _encode(ROLES, cols=512, q256=q256, seed=q256)
    module = _module(blob, scheme)
    dec = decode_e4m3(module, chunk=192)     # a chunk that does not divide cols
    assert dec.weight.dtype == torch.float8_e4m3fn and dec.weight.shape == ref_w.shape
    ref8 = ref_w.to(torch.float8_e4m3fn) if ref_w.dtype != torch.float8_e4m3fn else ref_w
    assert torch.equal(dec.weight.float(), ref8.float()), "decoded E4M3 values differ from stock"
    # bytes too, except that a zero may come back unsigned (an exact sum of zeros)
    a, b = dec.weight.view(torch.uint8), ref8.view(torch.uint8)
    assert bool(((a == b) | ((a & 0x7F) == 0) & ((b & 0x7F) == 0)).all())
    assert torch.equal(dec.scale, ref_s), "row scale differs from the stock weight_scale"
    assert torch.equal(dec.scale, module.row_scale())
    assert dec.nbytes == ref_w.numel() + 4 * ref_w.shape[0]


@cuda
@pytest.mark.parametrize("m", [1, 7, 64, 512, 2048])
def test_the_scaled_gemm_is_the_e4m3_contract_within_the_derived_bound(m):
    from tessera.serving.e4m3_prefill import decode_e4m3, prefill_apply

    cols = 512
    blob, scheme, ref_w, ref_s = _encode(ROLES, cols=cols, seed=11)
    module = _module(blob, scheme)
    dec = decode_e4m3(module)
    x = torch.randn(m, cols, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(m)).bfloat16()
    xq, a = _quant(x)
    xq, a = xq.contiguous(), a.reshape(-1).contiguous().float()
    got = prefill_apply(dec, xq, a)
    served = module.apply(xq, a)
    assert got.shape == served.shape == (m, ref_w.shape[0]) and got.dtype == torch.bfloat16
    a64 = xq.double() * a.double()[:, None]
    w64 = ref_w.to(torch.float8_e4m3fn).double() * ref_s.double()[:, None]
    r, bound = fb.dense_bound("e4m3", a64, w64, cols, s=cols)
    ratio = fb.check_within(got, r, bound, f"decoded M={m} vs the fp64 definition")
    pair = fb.check_within(got, served.double(), bound, f"decoded vs served M={m}", scale=2.0)
    same = float((got == served).float().mean())
    print(f"E4M3-PREFILL M={m} decoded/bound={ratio:.4f} pair/(2*bound)={pair:.4f} "
          f"bitwise_equal_frac={same:.6f}")


# --- the served integration (contract v56) ---------------------------------------

def _inputs(m, cols, seed):
    x = torch.randn(m, cols, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(seed)).bfloat16()
    xq, a = _quant(x)
    return x, xq.contiguous(), a.reshape(-1).contiguous().float()


@cuda
def test_the_module_serves_from_its_decoded_copy_at_and_above_min_m():
    """Below ``MIN_M`` the window lane runs and is stamped; at and above it the
    decoded copy runs and is stamped -- each bitwise the function it names."""
    from tessera.serving.e4m3_prefill import MIN_M, decode_e4m3, prefill_apply
    from tessera.serving.scheme import DECODE_ONCE_DENSE_SYMBOL
    from tessera.serving.telemetry import DECODER_NATIVE_WINDOW_DECODE_ONCE_E4M3

    blob, scheme, _w, _s = _encode(ROLES, cols=512, seed=21)
    module, twin = _module(blob, scheme), _module(blob, scheme)
    window_pair = module.launch_pair
    dec = decode_e4m3(module)
    module.attach_decoded(dec)
    assert module.decoded is dec and twin.decoded is None
    for m in (1, MIN_M - 1):
        _x, xq, a = _inputs(m, 512, m)
        assert module.launch_pair_for(m) == window_pair
        assert torch.equal(module.apply(xq, a), twin.apply(xq, a)), m
    for m in (MIN_M, 2 * MIN_M + 3):
        _x, xq, a = _inputs(m, 512, m)
        assert module.launch_pair_for(m) == (DECODE_ONCE_DENSE_SYMBOL,
                                             DECODER_NATIVE_WINDOW_DECODE_ONCE_E4M3)
        assert torch.equal(module.apply(xq, a), prefill_apply(dec, xq, a)), m
        # the copy-less twin never takes it
        assert twin.launch_pair_for(m) == window_pair


@cuda
def test_the_decoded_copy_is_counted_and_attached_once_and_only_where_it_fits():
    from tessera.serving.e4m3_prefill import DecodedE4M3, decode_e4m3

    blob, scheme, _w, _s = _encode(ROLES, cols=512, seed=22)
    module = _module(blob, scheme)
    before, names_before = module.packed_bytes(), dict(module.named_tensors())
    dec = decode_e4m3(module)
    module.attach_decoded(dec)
    named = dict(module.named_tensors())
    assert set(named) - set(names_before) == {"decoded.weight", "decoded.scale"}
    assert named["decoded.weight"] is dec.weight and named["decoded.scale"] is dec.scale
    assert module.packed_bytes() - before == dec.nbytes == 416 * 512 + 4 * 416
    with pytest.raises(ValueError, match="already holds"):
        module.attach_decoded(dec)
    other = _module(blob, scheme)
    wrong = DecodedE4M3(weight=dec.weight[:, :256].contiguous(), scale=dec.scale)
    with pytest.raises(ValueError, match="does not fit"):
        other.attach_decoded(wrong)


def _route_layer(monkeypatch, flag, mode, seed):
    """``(method, layer)``: the FP8 route built, loaded and prepared on one
    rank of one, under ``TESSERA_E4M3_DECODE_ONCE=flag``, with a compile
    identity record so the dispatch fact is observable."""
    from types import SimpleNamespace

    import vllm.model_executor.parameter as vllm_parameter
    from tessera.serving import compile_identity, e4m3_prefill, flags
    from tessera.serving.lane import build_tessera_method

    # one rank of one: vLLM's parameters read the TP group, which no test
    # process initialises
    monkeypatch.setattr(vllm_parameter, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(vllm_parameter, "get_tensor_model_parallel_world_size", lambda: 1,
                        raising=False)
    monkeypatch.delitem(flags._LATCHED, e4m3_prefill.FLAG, raising=False)
    monkeypatch.setenv(e4m3_prefill.FLAG, flag)
    blob, scheme, _w, _s = _encode(ROLES, cols=512, seed=seed)

    class _Layer(torch.nn.Module):
        tp_rank, tp_size = 0, 1

    compile_identity.reset_for_tests()
    compile_identity.declare_compile_identity_in(SimpleNamespace(
        additional_config={},
        compilation_config=SimpleNamespace(mode=SimpleNamespace(name="NONE"))), serve_mode=mode)
    method = build_tessera_method(scheme, "test.layer", mode=mode)
    layer = _Layer()
    method.create_weights(layer, input_size_per_partition=512,
                          output_partition_sizes=[r for _, r in ROLES], input_size=512,
                          output_size=scheme["rows"], params_dtype=torch.bfloat16)
    layer.wire_bytes.data = torch.frombuffer(bytearray(blob), dtype=torch.uint8).cuda()
    method.process_weights_after_loading(layer)
    return method, layer


@cuda
@pytest.mark.parametrize("flag,mode,attached", [("1", "resident", True), ("", "resident", False),
                                                ("0", "resident", False), ("1", "streamed", False)])
def test_the_fp8_route_attaches_only_under_the_flag_and_resident(monkeypatch, flag, mode, attached):
    """The route's load path decides: flag on and a resident module, or no
    copy at all; ``apply`` stamps the pair that actually ran, and the
    compile-cache dispatch fact tells the two apart (issue #91's rule)."""
    pytest.importorskip("vllm")   # the route's A side is vLLM's native FP8 quantiser
    from tessera.serving import compile_identity, e4m3_prefill, telemetry
    from tessera.serving.scheme import DECODE_ONCE_DENSE_SYMBOL

    method, layer = _route_layer(monkeypatch, flag, mode, seed=23)
    native = layer.tessera_native
    assert (native.decoded is not None) == attached
    fact = compile_identity.traced_dispatch()["test.layer"]
    assert fact == (f"{native.symbol}|{DECODE_ONCE_DENSE_SYMBOL}" if attached else native.symbol)
    compile_identity.reset_for_tests()
    for m in (8, e4m3_prefill.MIN_M):
        x = torch.randn(m, 512, device="cuda").bfloat16()
        y = method.apply(layer, x)
        assert y.shape == (m, sum(r for _, r in ROLES))
        record = telemetry.read_route(layer)
        assert (record["symbol"], record["decoder"]) == native.launch_pair_for(m)
        took = (record["symbol"], record["decoder"]) == (
            DECODE_ONCE_DENSE_SYMBOL, telemetry.DECODER_NATIVE_WINDOW_DECODE_ONCE_E4M3)
        assert took == (attached and m >= e4m3_prefill.MIN_M), (flag, mode, m)


@cuda
@pytest.mark.parametrize("flag", ["", "1"])
def test_a_dynamic_token_dimension_compiles_without_a_copy_and_refuses_with_one(monkeypatch, flag):
    """The compiled-serve failure mode, reproduced at the route: vLLM marks
    the token dimension dynamic, and a forward that reads ``int(M)``
    specialises it and raises ``ConstraintViolationError``.  With the flag
    unset the route's ``apply`` compiles once and serves two M; with a
    decode-once copy attached it refuses by name (the lane is eager-only)."""
    pytest.importorskip("vllm")
    from tessera.serving.e4m3_prefill import FLAG

    method, layer = _route_layer(monkeypatch, flag, "resident", seed=25)
    torch._dynamo.reset()
    compiled = torch.compile(lambda x: method.apply(layer, x), fullgraph=False)
    for m in (16, 300):
        x = torch.randn(m, 512, device="cuda").bfloat16()
        torch._dynamo.mark_dynamic(x, 0)
        if flag == "1":
            with pytest.raises(Exception, match=f"eager-only.*{FLAG}"):
                compiled(x)
            continue
        got = compiled(x)
        torch.testing.assert_close(got, method.apply(layer, x), rtol=0, atol=0)
    torch._dynamo.reset()


@cuda
def test_the_decode_once_lane_refuses_a_compiled_forward_and_the_window_lane_does_not(monkeypatch):
    """Eager-only by name: with a copy attached, ``apply`` under
    ``torch.compile`` refuses (its M branch would pin or bake the token
    count); a copy-less module is untouched."""
    from tessera.serving.e4m3_prefill import FLAG, decode_e4m3

    blob, scheme, _w, _s = _encode(ROLES, cols=512, seed=24)
    module, twin = _module(blob, scheme), _module(blob, scheme)
    module.attach_decoded(decode_e4m3(module))
    _x, xq, a = _inputs(8, 512, 8)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(RuntimeError, match=f"eager-only.*{FLAG}"):
        module.apply(xq, a)
    assert twin.apply(xq, a).shape == (8, 416)
