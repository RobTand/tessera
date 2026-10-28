"""CPU lifecycle evidence for the experimental raw-BF16 resident prefill lane.

Kernel launches are stand-ins here; these tests do not qualify CUDA numerics,
graphs, serving cells, speed or energy.
"""
from types import SimpleNamespace

import pytest
import torch

from test_piece_major_reader_boundaries import prepared
from tessera.serving import flags
from tessera.serving.native_window import PreparedDenseNativeModule


@pytest.fixture(autouse=True)
def reset_flags():
    from tessera.serving import compile_identity

    compile_identity.reset_for_tests()
    flags.reset_for_tests()
    yield
    flags.reset_for_tests()
    compile_identity.reset_for_tests()


def module(family="value"):
    bundle = prepared(family)
    roles = [SimpleNamespace(name="weight", rows=bundle.rows, bundle=bundle)]
    return PreparedDenseNativeModule(roles, rows=bundle.rows, columns=bundle.cols,
                                     device=bundle.device, family=family)


def decoded(owner, min_m=256):
    from tessera.serving.bf16_prefill import DecodedBF16

    bundle = owner.role_bundles[0]
    return DecodedBF16(
        weight=torch.ones(owner.rows, owner.columns, dtype=torch.bfloat16),
        scale=torch.linspace(0.25, 1.25, owner.rows), min_m=min_m,
        block_m=bundle.block_m, block_n=bundle.block_n, block_k=bundle.block_k)


@pytest.mark.parametrize("setting", [None, "0", "1", "typo"])
def test_prefill_is_strictly_default_off(monkeypatch, setting):
    from tessera.serving import bf16_prefill as prefill

    monkeypatch.delenv(prefill.FLAG, raising=False)
    if setting is not None:
        monkeypatch.setenv(prefill.FLAG, setting)
    if setting == "typo":
        with pytest.raises(ValueError, match=prefill.FLAG):
            prefill.enabled()
    else:
        assert prefill.enabled() is (setting == "1")


@pytest.mark.parametrize("setting", [None, "", "0", "-1", "1.5", "256"])
def test_crossover_requires_an_explicit_positive_integer(monkeypatch, setting):
    from tessera.serving import bf16_prefill as prefill

    monkeypatch.delenv(prefill.MIN_M_FLAG, raising=False)
    if setting is not None:
        monkeypatch.setenv(prefill.MIN_M_FLAG, setting)
    if setting == "256":
        assert prefill.minimum_m() == 256
        monkeypatch.setenv(prefill.MIN_M_FLAG, "512")
        with pytest.raises(RuntimeError, match="changed"):
            prefill.minimum_m()
    else:
        with pytest.raises(ValueError, match=prefill.MIN_M_FLAG):
            prefill.minimum_m()


def test_raw_copy_attaches_once_and_every_resident_byte_is_declared():
    owner = module()
    copy = decoded(owner)
    before = owner.packed_bytes()
    owner.attach_decoded(copy)
    named = dict(owner.named_tensors())
    assert named["decoded.weight"] is copy.weight
    assert named["decoded.scale"] is copy.scale
    assert owner.packed_bytes() - before == copy.nbytes
    assert copy.nbytes == 2 * owner.rows * owner.columns + 4 * owner.rows
    with pytest.raises(ValueError, match="already holds"):
        owner.attach_decoded(copy)
    with pytest.raises(ValueError, match="not"):
        module("e4m3").attach_decoded(copy)


@pytest.mark.parametrize("tokens", [1, 255, 256, 8192])
def test_dispatch_and_telemetry_name_the_same_path(monkeypatch, tokens):
    from tessera.serving import bf16_prefill as prefill, native_window

    owner = module()
    copy = decoded(owner)
    packed_pair = owner.launch_pair
    owner.attach_decoded(copy)
    calls = []

    def raw_gemm(inputs, weight, scale, *blocks):
        calls.append("decoded")
        return ((inputs.float() @ weight.float().t()) * scale).bfloat16()

    def packed(inputs, *args):
        calls.append("packed")
        return torch.zeros(inputs.shape[0], owner.rows, dtype=torch.bfloat16)

    monkeypatch.setattr(prefill, "_bf16_prefill", raw_gemm)
    monkeypatch.setattr(native_window, "_window_gemm_dense", packed)
    activations = torch.ones(tokens, owner.columns, dtype=torch.bfloat16)
    result = owner.apply(activations)
    took_copy = tokens >= copy.min_m
    assert calls == ["decoded" if took_copy else "packed"]
    assert owner.launch_pair_for(tokens) == (copy.launch_pair if took_copy else packed_pair)
    assert result.shape == (tokens, owner.rows) and result.dtype == torch.bfloat16


def test_raw_decode_uses_ones_not_folded_scales_and_accounts_for_scratch(monkeypatch):
    from tessera.serving.bf16_prefill import decode_bf16, decode_scratch_bytes
    from tessera.window_gemm import PreparedWindowGemm

    owner = module()
    bundle = owner.role_bundles[0]
    raw = torch.arange(owner.rows * owner.columns).reshape(owner.rows, owner.columns)
    raw = (raw.remainder(17) / 16).bfloat16()
    calls = []

    def identity_decode(unit, identity, a_scale=None, **kwargs):
        calls.append(identity.shape[0])
        assert unit.family == "value"
        assert torch.equal(unit.scale, torch.ones_like(bundle.scale))
        assert identity.dtype == torch.bfloat16 and a_scale is None
        return (identity.float() @ raw.float().t()).bfloat16()

    monkeypatch.setattr(PreparedWindowGemm, "__call__", identity_decode)
    copy = decode_bf16(owner, min_m=256, chunk=33)
    assert torch.equal(copy.weight, raw)
    assert torch.equal(copy.scale, owner.row_scale())
    assert sum(calls) == owner.columns and max(calls) == 33
    assert decode_scratch_bytes(owner, chunk=33) == (
        4 * 33 * owner.columns + 2 * 33 * owner.rows + 4 * owner.rows + 4)


@pytest.mark.parametrize("field", ["weight", "scale", "min_m"])
def test_invalid_copy_metadata_refuses_by_field(field):
    import dataclasses

    copy = decoded(module())
    wrong = {"weight": copy.weight.float(), "scale": copy.scale.bfloat16(), "min_m": 0}
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(copy, **{field: wrong[field]})


def test_compiled_use_refuses_before_launch(monkeypatch):
    from tessera.serving.bf16_prefill import FLAG

    owner = module()
    owner.attach_decoded(decoded(owner))
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(RuntimeError, match=FLAG):
        owner.apply(torch.zeros(256, owner.columns, dtype=torch.bfloat16))


def test_prefill_op_has_shape_only_fake_implementation():
    from tessera.serving.bf16_prefill import _bf16_prefill

    result = _bf16_prefill(
        torch.empty(8192, 128, device="meta", dtype=torch.bfloat16),
        torch.empty(256, 128, device="meta", dtype=torch.bfloat16),
        torch.empty(256, device="meta"), 64, 64, 64)
    assert result.shape == (8192, 256) and result.dtype == torch.bfloat16


@pytest.mark.parametrize("flag,mode,compiled,expected", [
    (None, "resident", False, "packed"),
    ("0", "resident", False, "packed"),
    ("1", "resident", False, "decoded"),
    ("1", "resident", True, "refuse"),
    ("1", "streamed", False, "refuse"),
])
def test_route_load_owns_opt_in_and_records_the_executed_launch(
        monkeypatch, flag, mode, compiled, expected):
    from test_serving_bf16_route import _install_vllm_stubs, _scheme
    from tessera.serving import bf16_prefill as prefill, bf16_route, compile_identity

    _install_vllm_stubs(monkeypatch)
    owner = module()
    copy = decoded(owner)
    monkeypatch.delenv(prefill.FLAG, raising=False)
    if flag is not None:
        monkeypatch.setenv(prefill.FLAG, flag)
    monkeypatch.setenv(prefill.MIN_M_FLAG, str(copy.min_m))
    monkeypatch.setattr(bf16_route, "parse_compact_blob_for_scheme", lambda *args, **kwargs: ())
    monkeypatch.setattr(bf16_route, "prepare_dense_native_module", lambda *args, **kwargs: owner)
    config = SimpleNamespace(additional_config={}, compilation_config=SimpleNamespace(
        mode=SimpleNamespace(name="VLLM_COMPILE" if compiled else "NONE")))
    compile_identity.declare_compile_identity_in(config, serve_mode=mode)
    monkeypatch.setattr(compile_identity, "_current_vllm_config", lambda: None)
    method = bf16_route.build_tessera_bf16_method(
        _scheme(rows=owner.rows, columns=owner.columns), "test.prefill", mode)

    class Layer(torch.nn.Module):
        tp_rank, tp_size = 0, 1

    layer = Layer()
    method.create_weights(layer, input_size_per_partition=owner.columns,
                          output_partition_sizes=[owner.rows], input_size=owner.columns,
                          output_size=owner.rows, params_dtype=torch.bfloat16)
    layer.wire_bytes.data.zero_()
    decode_calls = []

    def decode(*args, **kwargs):
        decode_calls.append(kwargs)
        return copy

    monkeypatch.setattr(prefill, "decode_bf16", decode)
    if expected == "refuse":
        with pytest.raises(RuntimeError, match=prefill.FLAG):
            method.process_weights_after_loading(layer)
        assert decode_calls == []
        return
    method.process_weights_after_loading(layer)
    assert bool(decode_calls) == (expected == "decoded")
    assert (owner.decoded is copy) == (expected == "decoded")
    if expected == "decoded":
        named = dict(method.resident_tensors(layer))
        assert named["tessera_native.decoded.weight"] is copy.weight
        assert named["tessera_native.decoded.scale"] is copy.scale
        events = []
        monkeypatch.setattr(prefill, "_bf16_prefill", lambda inputs, *args: torch.zeros(
            inputs.shape[0], owner.rows, dtype=torch.bfloat16))
        monkeypatch.setattr(bf16_route, "emit_route", lambda *args, **kwargs: events.append(kwargs))
        method.apply(layer, torch.ones(copy.min_m, owner.columns, dtype=torch.bfloat16))
        assert (events[0]["symbol"], events[0]["decoder"]) == copy.launch_pair


def test_bf16_prefill_is_experimental_and_resident_eager_only():
    from tessera.serving.bf16_prefill import DecodedBF16
    from tessera.serving.scheme import (EAGER_ONLY_LAUNCHES, EXPERIMENTAL_LAUNCHES,
                                        TESSERA_BF16, launch_pairs)
    from tessera.serving.bf16_route import census_expected

    pair = decoded(module()).launch_pair
    assert pair in EXPERIMENTAL_LAUNCHES and pair in EAGER_ONLY_LAUNCHES
    assert pair not in launch_pairs(TESSERA_BF16)
    assert pair in launch_pairs(TESSERA_BF16, mode="resident", include_experimental=True)
    assert pair not in launch_pairs(TESSERA_BF16, mode="streamed", include_experimental=True)
    assert all(pair not in pairs for pairs in census_expected(compiled=True).values())


def test_invalid_crossover_refuses_before_decode_allocation(monkeypatch):
    from tessera.serving.bf16_prefill import decode_bf16

    owner = module()

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("decode allocated before validating min_m")

    monkeypatch.setattr(torch, "empty", unexpected_allocation)
    with pytest.raises(ValueError, match="min_m"):
        decode_bf16(owner, min_m=0)


def test_prefill_launch_keeps_raw_weight_and_fp32_scale_separate(monkeypatch):
    from tessera.serving import bf16_prefill_kernel as kernel

    copy = decoded(module())
    activations = torch.ones(copy.min_m, copy.weight.shape[1], dtype=torch.bfloat16)
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, args, kwargs))
            return launch

    monkeypatch.setattr(kernel, "_bf16_prefill_kernel", Kernel())
    result = copy.apply(activations)
    grid, operands, config = launches.pop()
    assert operands[0] is activations
    assert operands[1] is copy.weight and operands[2] is copy.scale
    assert operands[3] is result and result.dtype == torch.bfloat16
    assert config["enable_fp_fusion"] is False
    assert grid == (kernel.triton.cdiv(copy.weight.shape[0], copy.block_n),
                    kernel.triton.cdiv(copy.min_m, copy.block_m))
