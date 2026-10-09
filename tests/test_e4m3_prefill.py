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


def _route_layer(monkeypatch, flag, mode, seed, vllm_mode="NONE", roles=ROLES):
    """``(method, layer)``: the FP8 route built, loaded and prepared on one
    rank of one, under ``TESSERA_E4M3_DECODE_ONCE=flag``, with a compile
    identity record so the dispatch fact is observable, and vLLM's current
    config at load in compilation mode ``vllm_mode``."""
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
    if flag is None:
        monkeypatch.delenv(e4m3_prefill.FLAG, raising=False)
    else:
        monkeypatch.setenv(e4m3_prefill.FLAG, flag)
    blob, scheme, _w, _s = _encode(roles, cols=512, seed=seed)

    class _Layer(torch.nn.Module):
        tp_rank, tp_size = 0, 1

    config = SimpleNamespace(
        additional_config={},
        compilation_config=SimpleNamespace(mode=SimpleNamespace(name=vllm_mode)))
    monkeypatch.setattr(compile_identity, "_current_vllm_config", lambda: config)
    compile_identity.reset_for_tests()
    compile_identity.declare_compile_identity_in(config, serve_mode=mode)
    method = build_tessera_method(scheme, "test.layer", mode=mode)
    layer = _Layer()
    method.create_weights(layer, input_size_per_partition=512,
                          output_partition_sizes=[r for _, r in roles], input_size=512,
                          output_size=scheme["rows"], params_dtype=torch.bfloat16)
    layer.wire_bytes.data = torch.frombuffer(bytearray(blob), dtype=torch.uint8).cuda()
    method.process_weights_after_loading(layer)
    return method, layer


@cuda
@pytest.mark.parametrize("roles", [ROLES, [("gate_proj", 128), ("up_proj", 128)],
                                  [("down_proj", 128)]], ids=["dense", "shared-gate-up", "shared-down"])
@pytest.mark.parametrize("flag,mode,attached", [(None, "resident", True), ("1", "resident", True),
                                                ("", "resident", True), ("0", "resident", False),
                                                (None, "streamed", False), ("1", "streamed", False)])
def test_the_fp8_route_attaches_only_under_the_flag_and_resident(monkeypatch, flag, mode, attached, roles):
    """The real load and forward cover dense and shared projection modules."""
    pytest.importorskip("vllm")   # the route's A side is vLLM's native FP8 quantiser
    from tessera.serving import compile_identity, e4m3_prefill, telemetry
    from tessera.serving.scheme import DECODE_ONCE_DENSE_SYMBOL

    method, layer = _route_layer(monkeypatch, flag, mode, seed=23, roles=roles)
    native = layer.tessera_native
    assert (native.decoded is not None) == attached
    fact = compile_identity.traced_dispatch()["test.layer"]
    assert fact == (f"{native.symbol}|{DECODE_ONCE_DENSE_SYMBOL}" if attached else native.symbol)
    compile_identity.reset_for_tests()
    for m in (8, e4m3_prefill.MIN_M):
        x = torch.randn(m, 512, device="cuda").bfloat16()
        y = method.apply(layer, x)
        assert y.shape == (m, layer.tessera_native.rows)
        record = telemetry.read_route(layer)
        assert (record["symbol"], record["decoder"]) == native.launch_pair_for(m)
        took = (record["symbol"], record["decoder"]) == (
            DECODE_ONCE_DENSE_SYMBOL, telemetry.DECODER_NATIVE_WINDOW_DECODE_ONCE_E4M3)
        assert took == (attached and m >= e4m3_prefill.MIN_M), (flag, mode, m)


@cuda
@pytest.mark.parametrize("flag", ["", "1"])
def test_a_compiled_vllm_forward_refuses_the_flag_at_load(monkeypatch, flag):
    """The eager-only gate is the LOAD: with vLLM's compilation mode not NONE,
    the flag refuses by name before any copy is made; unset, the module loads
    as on master."""
    pytest.importorskip("vllm")
    from tessera.serving.e4m3_prefill import FLAG

    if flag == "1":
        with pytest.raises(RuntimeError, match=f"{FLAG}=1 serves an eager-only lane"):
            _route_layer(monkeypatch, flag, "resident", seed=26, vllm_mode="VLLM_COMPILE")
        return
    _method, layer = _route_layer(monkeypatch, flag, "resident", seed=26, vllm_mode="VLLM_COMPILE")
    assert layer.tessera_native.decoded is None


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

    from tessera.serving import fp8_route

    method, layer = _route_layer(monkeypatch, flag, "resident", seed=25,
                                 vllm_mode="VLLM_COMPILE" if flag == "" else "NONE")
    # vLLM compiles the whole forward (fullgraph); the route record is
    # host-side telemetry the eager tests above check, and the token-count
    # read this test is about happens before it is called
    monkeypatch.setattr(fp8_route, "emit_route", lambda *a, **k: None)
    torch._dynamo.reset()
    compiled = torch.compile(lambda x: method.apply(layer, x), fullgraph=True)
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

# CPU policy lifecycle: external vLLM/CUDA preparation seams are stand-ins;
# the route, compile owner, decoded-copy attachment and accounting are real.
def _cpu_load_entry(monkeypatch, flag, mode, vllm_mode):
    from types import SimpleNamespace

    from test_piece_major_reader_boundaries import prepared
    from test_serving_fp8_route import _install_vllm_stubs, _scheme
    from tessera.serving import compile_identity, e4m3_prefill, fp8_route, native_ops
    from tessera.serving.native_window import PreparedDenseNativeModule

    _install_vllm_stubs(monkeypatch)
    if flag is None:
        monkeypatch.delenv(e4m3_prefill.FLAG, raising=False)
    else:
        monkeypatch.setenv(e4m3_prefill.FLAG, flag)
    bundle = prepared()
    native = PreparedDenseNativeModule(
        [SimpleNamespace(name="weight", rows=bundle.rows, bundle=bundle)],
        rows=bundle.rows, columns=bundle.cols, device=bundle.device, family=bundle.family)
    monkeypatch.setattr(fp8_route, "parse_compact_blob_for_scheme", lambda *a, **k: ())
    monkeypatch.setattr(fp8_route, "prepare_dense_native_module", lambda *a, **k: native)
    monkeypatch.setattr(native_ops, "require_native_fp8_quant", lambda *a: None)
    decoded_roles = []

    def decode_role(unit, chunk):
        decoded_roles.append(unit)
        return torch.zeros(unit.rows, unit.cols, dtype=torch.float8_e4m3fn)

    monkeypatch.setattr(e4m3_prefill, "_decode_role", decode_role)
    config = SimpleNamespace(
        additional_config={},
        compilation_config=SimpleNamespace(mode=SimpleNamespace(name=vllm_mode)))
    compile_identity.declare_compile_identity_in(config, serve_mode=mode)
    # Model construction has left set_current_vllm_config before weight load.
    monkeypatch.setattr(compile_identity, "_current_vllm_config", lambda: None)
    scheme = _scheme(rows=bundle.rows, columns=bundle.cols)
    method = fp8_route.build_tessera_fp8_method(scheme, "test.layer", mode)

    class Layer(torch.nn.Module):
        tp_rank, tp_size = 0, 1

    layer = Layer()
    method.create_weights(layer, input_size_per_partition=bundle.cols,
                          output_partition_sizes=[bundle.rows], input_size=bundle.cols,
                          output_size=bundle.rows, params_dtype=torch.bfloat16)
    layer.wire_bytes.data.zero_()
    return method, layer, native, config, decoded_roles


@pytest.fixture
def cpu_load_entry(monkeypatch):
    from tessera.serving import compile_identity, e4m3_prefill, flags

    compile_identity.reset_for_tests()
    flags.reset_for_tests(e4m3_prefill.FLAG)
    yield lambda **kwargs: _cpu_load_entry(monkeypatch, **kwargs)
    compile_identity.reset_for_tests()
    flags.reset_for_tests(e4m3_prefill.FLAG)


def test_declared_compiled_forward_refuses_decode_once_after_config_exits(cpu_load_entry):
    from tessera.serving.e4m3_prefill import FLAG

    method, layer, native, config, decoded_roles = cpu_load_entry(
        flag="1", mode="resident", vllm_mode="VLLM_COMPILE")
    # Changing the old config cannot change the construction-time declaration.
    config.compilation_config.mode.name = "NONE"
    with pytest.raises(RuntimeError, match=f"{FLAG}=1 serves an eager-only lane"):
        method.process_weights_after_loading(layer)
    assert decoded_roles == [] and native.decoded is None
    assert hasattr(layer, "wire_bytes") and not hasattr(layer, "tessera_native")


@pytest.mark.parametrize("flag,mode,vllm_mode,attached", [
    (None, "resident", "NONE", True),
    ("", "resident", "NONE", True),
    (None, "resident", "VLLM_COMPILE", False),
    ("", "resident", "VLLM_COMPILE", False),
    ("0", "resident", "VLLM_COMPILE", False),
    ("0", "resident", "NONE", False),
    ("1", "resident", "NONE", True),
    (None, "streamed", "NONE", False),
    ("1", "streamed", "VLLM_COMPILE", False),
])
def test_load_boundaries_survive_the_current_config_exiting(
        cpu_load_entry, flag, mode, vllm_mode, attached):
    from tessera.serving import compile_identity
    from tessera.serving.scheme import DECODE_ONCE_DENSE_SYMBOL

    method, layer, native, _config, decoded_roles = cpu_load_entry(
        flag=flag, mode=mode, vllm_mode=vllm_mode)
    before = native.packed_bytes()
    method.process_weights_after_loading(layer)
    assert layer.tessera_native is native and not hasattr(layer, "wire_bytes")
    assert torch.equal(layer.scale_b.reshape(-1), native.row_scale())
    assert (native.decoded is not None) is attached
    assert len(decoded_roles) == int(attached)
    expected = f"{native.symbol}|{DECODE_ONCE_DENSE_SYMBOL}" if attached else native.symbol
    assert compile_identity.traced_dispatch() == {"test.layer": expected}
    assert native.packed_bytes() - before == (native.decoded.nbytes if attached else 0)
    held = dict(method.resident_tensors(layer))
    for name, tensor in native.named_tensors():
        assert held[f"tessera_native.{name}"] is tensor


def test_decode_once_forward_backstop_remains_after_eager_load(cpu_load_entry, monkeypatch):
    from tessera.serving import native_ops
    from tessera.serving.e4m3_prefill import FLAG

    method, layer, native, _config, _decoded_roles = cpu_load_entry(
        flag="1", mode="resident", vllm_mode="NONE")
    method.process_weights_after_loading(layer)
    assert native.decoded is not None
    monkeypatch.setattr(native_ops, "native_fp8_quant",
                        lambda x: (x.to(torch.float8_e4m3fn), torch.ones(x.shape[0])))
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(RuntimeError, match=f"eager-only.*{FLAG}"):
        method.apply(layer, torch.zeros(1, native.columns, dtype=torch.bfloat16))
