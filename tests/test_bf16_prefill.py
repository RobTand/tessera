"""Behavior checks for the shared BF16 prefill scratch and the packed WINDOW wire.

The numerical limits come from fused_bound. Graph replay has external serialization.
These tests do not qualify concurrent graph replay or whole-model service.
"""

import importlib
import sys
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl

from tessera.window_gemm import _scratch_indices

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fused_bound as fb
from test_dense_fused_window import _encode_module, _module, _role, _sched

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 prefill requires CUDA")
ROLES = [("q_proj", 128), ("b_proj", 32)]
COLS = 64


def _api():
    # A missing public implementation must fail, not skip, on the pre-fix tree.
    return importlib.import_module("tessera.serving.bf16_prefill")


@triton.jit(do_not_specialize=["row_offset", "cols"])
def _scratch_index_probe(out, row_offset, cols):
    rows = tl.arange(0, 2)
    columns = tl.arange(0, 2)
    indices = _scratch_indices(row_offset, rows, cols, columns)
    tl.store(out + columns[:, None] * 2 + rows[None, :], indices)


@cuda
def test_scratch_addresses_do_not_wrap_at_the_int32_limit():
    cols = 4096
    first = torch.iinfo(torch.int32).max + 1
    row_offset = first // cols
    got = torch.empty(2, 2, dtype=torch.int64, device="cuda")
    _scratch_index_probe[(1,)](got, row_offset, cols)
    expected = torch.tensor([[first, first + cols],
                             [first + 1, first + cols + 1]],
                            dtype=torch.int64, device="cuda")
    assert torch.equal(got, expected), f"scratch addresses wrap: {got.tolist()}"


@pytest.fixture(scope="module")
def encoded():
    return _encode_module("value", ROLES, COLS, q256=1088, seed=31)


@pytest.mark.parametrize("raw", ["", "0", "-1", "1.5", "true"])
def test_route_admission_requires_an_explicit_positive_min_m(monkeypatch, raw):
    api = _api()
    from tessera.serving import flags

    flags.reset_for_tests(api.MIN_M_FLAG)
    monkeypatch.setenv(api.MIN_M_FLAG, raw)
    with pytest.raises(ValueError, match=api.MIN_M_FLAG):
        api.configured_min_m()
    flags.reset_for_tests(api.MIN_M_FLAG)


def test_route_admission_is_stable_and_the_flag_is_off_by_default(monkeypatch):
    api = _api()
    from tessera.serving import flags

    flags.reset_for_tests(api.FLAG)
    flags.reset_for_tests(api.MIN_M_FLAG)
    monkeypatch.delenv(api.FLAG, raising=False)
    assert not api.enabled()
    monkeypatch.setenv(api.MIN_M_FLAG, "7")
    assert api.configured_min_m() == 7
    monkeypatch.setenv(api.MIN_M_FLAG, "8")
    with pytest.raises(RuntimeError, match="changed"):
        api.configured_min_m()
    flags.reset_for_tests(api.FLAG)
    flags.reset_for_tests(api.MIN_M_FLAG)


@cuda
def test_encoded_wire_materialization_has_exact_role_and_column_order(encoded):
    api = _api()
    blob, scheme, reference = encoded
    module = _module(blob, scheme)
    prepared = api.prepare_bf16_prefill(module)
    assert prepared.role_bundles == module.role_bundles
    assert prepared.weight.dtype == torch.bfloat16
    assert prepared.weight.shape == reference.shape
    assert api.decode_into(prepared) is prepared.weight
    assert torch.equal(prepared.weight.double(), reference)
    eye = torch.eye(COLS, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(api.prefill_apply(prepared, eye), reference.t().bfloat16())
    assert prepared.nbytes == reference.numel() * 2


@cuda
@pytest.mark.parametrize("q256", [256, 1088, 2176, 3456])
@pytest.mark.parametrize("has_init", [False, True])
def test_direct_decode_preserves_mixed_rates_and_initial_state(q256, has_init):
    api = _api()
    from tessera.serving.native_window import PreparedDenseNativeModule, _NativeRole, _RoleFacts

    cols, rows = 64, 513
    init = (torch.randint(0, 1 << 14, (cols,), dtype=torch.int32,
                          generator=torch.Generator().manual_seed(51)) if has_init else None)
    rates = _sched(cols, q256, cap=14)
    expert, bundle = _role("value", rows=rows, cols=cols, rates=rates, init=init, seed=52)
    facts = _RoleFacts(tuple(rates), int(expert.unit.rep.rows_p), cols, 0, has_init)
    role = _NativeRole("projection", rows, bundle, facts)
    module = PreparedDenseNativeModule([role], rows=rows, columns=cols,
                                       device=bundle.device, family="value")
    prepared = api.prepare_bf16_prefill(module)
    reference = fb.decoded_weight(expert, "value").bfloat16()
    assert torch.equal(api.decode_into(prepared), reference)
    eye = torch.eye(cols, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(api.prefill_apply(prepared, eye), reference.t())
    # The shared gather must also preserve the original streaming decoder.
    assert torch.equal(bundle(eye), reference.t())


@cuda
@pytest.mark.parametrize("m", [0, 1, 16, 2048])
def test_full_lane_matches_the_fp64_bound_on_the_same_encoded_wire(encoded, m):
    api = _api()
    blob, scheme, reference = encoded
    prepared = api.prepare_bf16_prefill(_module(blob, scheme))
    x = torch.randn(m, COLS, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(70 + m)).bfloat16()
    got = api.prefill_apply(prepared, x)
    assert got.shape == (m, reference.shape[0]) and got.dtype == torch.bfloat16
    if m:
        r, bound = fb.dense_bound("value", x.double(), reference, COLS, s=COLS)
        fb.check_within(got, r, bound, f"BF16 prefill M={m}")


@cuda
def test_same_shape_modules_share_scratch_and_refresh_it_on_every_step(encoded):
    api = _api()
    blob, scheme, first_ref = encoded
    other_blob, other_scheme, second_ref = _encode_module("value", ROLES, COLS,
                                                         q256=1088, seed=32)
    first = api.prepare_bf16_prefill(_module(blob, scheme))
    second = api.prepare_bf16_prefill(_module(other_blob, other_scheme))
    assert first.weight is second.weight
    eye = torch.eye(COLS, device="cuda", dtype=torch.bfloat16)
    for prepared, reference in [(first, first_ref), (second, second_ref), (first, first_ref)]:
        prepared.weight.fill_(float("nan"))
        assert torch.equal(api.prefill_apply(prepared, eye), reference.t().bfloat16())
    footprint = api.scratch_pool_footprint()
    key = (str(first.weight.device), *first.weight.shape)
    assert footprint[key] == first.nbytes


@cuda
def test_the_pool_releases_scratch_after_its_last_user(encoded):
    import gc
    import weakref

    api = _api()
    blob, scheme, _reference = encoded
    module = _module(blob, scheme)
    first = api.prepare_bf16_prefill(module)
    second = api.prepare_bf16_prefill(module)
    key = (str(first.weight.device), *first.weight.shape)
    storage = weakref.ref(first.weight)
    del first
    gc.collect()
    assert storage() is second.weight
    del second
    gc.collect()
    assert storage() is None
    assert key not in api.scratch_pool_footprint()



@cuda
def test_eager_streams_do_not_overwrite_another_modules_pending_gemm(encoded):
    api = _api()
    blob, scheme, first_ref = encoded
    other_blob, other_scheme, second_ref = _encode_module("value", ROLES, COLS,
                                                         q256=1088, seed=33)
    first = api.prepare_bf16_prefill(_module(blob, scheme))
    second = api.prepare_bf16_prefill(_module(other_blob, other_scheme))
    eye = torch.eye(COLS, device="cuda", dtype=torch.bfloat16)
    # Compile both calls before the deliberate stream delay.
    api.prefill_apply(first, eye)
    api.prefill_apply(second, eye)
    torch.cuda.synchronize()
    one, two = torch.cuda.Stream(), torch.cuda.Stream()
    one.wait_stream(torch.cuda.current_stream())
    two.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(one):
        torch.cuda._sleep(10_000_000)
        a = api.prefill_apply(first, eye)
    with torch.cuda.stream(two):
        b = api.prefill_apply(second, eye)
    torch.cuda.current_stream().wait_stream(one)
    torch.cuda.current_stream().wait_stream(two)
    assert torch.equal(a, first_ref.t().bfloat16())
    assert torch.equal(b, second_ref.t().bfloat16())


@cuda
def test_graph_replay_refreshes_the_wire_under_external_serialization(encoded):
    api = _api()
    blob, scheme, reference = encoded
    module = _module(blob, scheme)
    prepared = api.prepare_bf16_prefill(module)
    eye = torch.eye(COLS, device="cuda", dtype=torch.bfloat16)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        api.prefill_apply(prepared, eye)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        output = api.prefill_apply(prepared, eye)
    torch.cuda.synchronize()
    # Replay uses the current stream, not the side stream that captured it.
    prepared.weight.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, reference.t().bfloat16())
    for bundle in module.role_bundles:
        bundle.scale.mul_(2)
    prepared.weight.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, (reference * 2).t().bfloat16())
    # Every graph and eager scratch user must finish before the next user.
    assert torch.equal(api.prefill_apply(prepared, eye), (reference * 2).t().bfloat16())


@cuda
def test_native_admission_and_resource_accounting_are_explicit(encoded):
    _api()
    from tessera.serving.scheme import BF16_DECODE_ONCE_DENSE_SYMBOL
    from tessera.serving.telemetry import DECODER_NATIVE_WINDOW_DECODE_ONCE_BF16_FOLDED

    blob, scheme, reference = encoded
    module, baseline = _module(blob, scheme), _module(blob, scheme)
    assert module.bf16_prefill is None
    with pytest.raises(TypeError):
        module.enable_bf16_prefill()
    with pytest.raises(ValueError, match="min_m"):
        module.enable_bf16_prefill(min_m=0)
    before = module.packed_bytes()
    module.enable_bf16_prefill(min_m=7)
    assert module.packed_bytes() - before == module.bf16_prefill.nbytes
    assert dict(module.named_tensors())["bf16_prefill.weight"] is module.bf16_prefill.weight
    expected_pair = (BF16_DECODE_ONCE_DENSE_SYMBOL,
                     DECODER_NATIVE_WINDOW_DECODE_ONCE_BF16_FOLDED)
    for m in [1, 6, 7, 16]:
        x = torch.randn(m, COLS, device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(110 + m)).bfloat16()
        got = module.apply(x)
        if m < 7:
            assert module.launch_pair_for(m) == module.launch_pair
            assert torch.equal(got, baseline.apply(x))
        else:
            assert module.launch_pair_for(m) == expected_pair
            r, bound = fb.dense_bound("value", x.double(), reference, COLS, s=COLS)
            fb.check_within(got, r, bound, f"native BF16 prefill M={m}")
    from tessera.serving.residency import resident_storage_bytes

    shared = _module(blob, scheme)
    physical_before = resident_storage_bytes(list(module.named_tensors())
                                            + list(shared.named_tensors()))
    shared.enable_bf16_prefill(min_m=7)
    assert shared.bf16_prefill.weight is module.bf16_prefill.weight
    physical_after = resident_storage_bytes(list(module.named_tensors())
                                           + list(shared.named_tensors()))
    assert physical_after == physical_before

    with pytest.raises(ValueError, match="already"):
        module.enable_bf16_prefill(min_m=8)
    with pytest.raises(ValueError, match="activation scale"):
        module.apply(torch.zeros(7, COLS, device="cuda", dtype=torch.bfloat16),
                     torch.ones(7, device="cuda"))


def _route_layer(monkeypatch, flag, min_m, *, mode="resident", compile_mode="NONE"):
    from types import SimpleNamespace

    import vllm.model_executor.parameter as parameters
    from tessera.serving import compile_identity, flags
    from tessera.serving.lane import build_tessera_method

    api = _api()
    monkeypatch.setattr(parameters, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(parameters, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    flags.reset_for_tests(api.FLAG)
    flags.reset_for_tests(api.MIN_M_FLAG)
    monkeypatch.setenv(api.FLAG, flag)
    monkeypatch.setenv(api.MIN_M_FLAG, min_m)
    blob, scheme, _reference = _encode_module("value", ROLES, COLS, q256=1088, seed=41)
    config = SimpleNamespace(additional_config={}, compilation_config=SimpleNamespace(
        mode=SimpleNamespace(name=compile_mode)))
    monkeypatch.setattr(compile_identity, "_current_vllm_config", lambda: config)
    compile_identity.reset_for_tests()
    compile_identity.declare_compile_identity_in(config, serve_mode=mode)
    method = build_tessera_method(scheme, "test.bf16", mode=mode)

    class Layer(torch.nn.Module):
        tp_rank, tp_size = 0, 1

    layer = Layer()
    method.create_weights(layer, input_size_per_partition=COLS,
                          output_partition_sizes=[rows for _, rows in ROLES], input_size=COLS,
                          output_size=scheme["rows"], params_dtype=torch.bfloat16)
    layer.wire_bytes.data = torch.frombuffer(bytearray(blob), dtype=torch.uint8).cuda()
    method.process_weights_after_loading(layer)
    flags.reset_for_tests(api.FLAG)
    flags.reset_for_tests(api.MIN_M_FLAG)
    return method, layer


@cuda
@pytest.mark.parametrize("flag,mode,attached", [("1", "resident", True),
                                               ("0", "resident", False),
                                               ("", "resident", False),
                                               ("1", "streamed", False)])
def test_route_stamps_the_actual_lane_and_keeps_default_paths(monkeypatch, flag, mode, attached):
    pytest.importorskip("vllm", reason="the route needs the real vLLM loader")
    _api()
    from tessera.serving import compile_identity, telemetry
    from tessera.serving.scheme import BF16_DECODE_ONCE_DENSE_SYMBOL

    method, layer = _route_layer(monkeypatch, flag, "7", mode=mode)
    native = layer.tessera_native
    assert (native.bf16_prefill is not None) == attached
    assert compile_identity.traced_dispatch()["test.bf16"] == (
        f"{native.symbol}|{BF16_DECODE_ONCE_DENSE_SYMBOL}" if attached else native.symbol)
    for m in [1, 7]:
        x = torch.randn(m, COLS, device="cuda").bfloat16()
        output = method.apply(layer, x)
        assert output.shape == (m, sum(rows for _, rows in ROLES))
        record = telemetry.read_route(layer)
        assert (record["symbol"], record["decoder"]) == native.launch_pair_for(m)
    compile_identity.reset_for_tests()


@cuda
def test_route_refuses_compile_and_an_unmeasured_threshold_before_scratch(monkeypatch):
    pytest.importorskip("vllm", reason="the route needs the real vLLM loader")
    api = _api()
    with pytest.raises(RuntimeError, match=f"{api.FLAG}=1 serves an eager-only lane"):
        _route_layer(monkeypatch, "1", "7", compile_mode="VLLM_COMPILE")
    with pytest.raises(ValueError, match=api.MIN_M_FLAG):
        _route_layer(monkeypatch, "1", "")
    _method, layer = _route_layer(monkeypatch, "0", "", compile_mode="VLLM_COMPILE")
    assert layer.tessera_native.bf16_prefill is None
