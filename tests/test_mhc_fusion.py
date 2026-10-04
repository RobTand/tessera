"""The fused mHC override's install, dispatch and contract rules (#783), device-less.

The kernel's bitwise identity is the CUDA-gated ``test_mhc_fusion_cuda.py``
and ``experiments/mhc/mhc_fused_probe.py``; this file pins the rules that
decide whether that kernel is ever called.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

from tessera.serving import contract, ext, flags
from tessera.serving import glm53_prefill as gp
from tessera.serving import mhc_fusion as mf


@pytest.fixture(autouse=True)
def _flag(monkeypatch):
    flags.reset_for_tests(mf.FLAG)
    monkeypatch.delenv(mf.FLAG, raising=False)
    yield
    flags.reset_for_tests(mf.FLAG)


def test_contract_publishes_the_override_this_module_installs():
    assert contract.stock_kernel_override_refusal(
        mf.FLAG, kind="model_method", overrides=mf.STOCK_OBJECT,
        loaded_by=mf.__name__, library_prefix=mf.MODULE_PREFIX) is None
    entry = next(e for e in ext.STOCK_KERNEL_OVERRIDES if e["enabled_by"] == mf.FLAG)
    assert entry["default"] == ext.OVERRIDE_DEFAULT_OFF
    assert entry["required_identity"] == ext.IDENTITY_BITWISE_VS_STOCK
    assert mf.source_path().name == entry["library"]["source"].split("/")[-1]
    assert mf.source_path().is_file()


def test_contract_refuses_an_install_that_drifted_from_its_entry():
    drifted = dict(mf.STOCK_OBJECT, method=mf.STOCK_OBJECT["method"] + "_other")
    assert contract.stock_kernel_override_refusal(
        mf.FLAG, kind="model_method", overrides=drifted,
        loaded_by=mf.__name__, library_prefix=mf.MODULE_PREFIX) is not None


def test_interfaces_are_the_sp_table_pins_for_the_same_modules():
    for interface, sp in zip(mf._interfaces(), gp._INTERFACES):
        assert interface.name == sp.name
        assert interface.digests == tuple(sp.digests[gp.SP_MODULES.index(n)] for n in mf.MODULES)


def test_kernel_constants_and_params_layout_match_the_cuda_source():
    src = mf.source_path().read_text()
    assert int(re.search(r"constexpr int TM_MAX = (\d+);", src).group(1)) == mf.TILE_MAX
    assert int(re.search(r"constexpr int HIDDEN = (\d+);", src).group(1)) == mf.HIDDEN
    assert int(re.search(r"constexpr int HC = (\d+);", src).group(1)) == mf.HC_MULT
    body = re.search(r"struct Params \{(.*?)\};", src, re.S).group(1)
    names = []
    for line in body.split("\n"):
        code = line.split("//")[0].strip().rstrip(";")
        if not code:
            continue
        names += [part.strip().lstrip("*").split()[-1].lstrip("*") for part in code.split(",")]
    assert names == [field for field, _ in mf._Params._fields_]


@pytest.mark.parametrize("tokens,ctas", [(33, 48), (1024, 48), (1024, 32), (2048, 48), (2049, 48),
                                         (8192, 48), (100, 1)])
def test_tiles_are_whole_m16_tiles_covering_the_site_in_the_fewest_waves(tokens, ctas):
    tm = mf.tile_tokens(tokens, ctas)
    assert tm % mf.TILE_QUANTUM == 0 and mf.TILE_QUANTUM <= tm <= mf.TILE_MAX
    waves = -(-tokens // (tm * ctas))
    if tm < mf.TILE_MAX:
        assert waves == 1  # one wave whenever the cap does not bind
        if tm > mf.TILE_QUANTUM:  # and no smaller tile would also have been one wave
            assert -(-tokens // ((tm - mf.TILE_QUANTUM) * ctas)) > 1


def test_unset_flag_installs_nothing():
    assert mf.install_mhc_fusion(object()) is False


def test_flag_is_strict(monkeypatch):
    monkeypatch.setenv(mf.FLAG, "on")
    with pytest.raises(ValueError):
        mf.install_mhc_fusion(None)


def test_non_glm5next_serve_declines(monkeypatch):
    monkeypatch.setenv(mf.FLAG, "1")
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])))
    assert mf.install_mhc_fusion(config) is False


def _call_tensors(tokens, hidden=mf.HIDDEN):
    return dict(
        x=torch.zeros(tokens, hidden, dtype=torch.bfloat16),
        residual=torch.zeros(tokens, mf.HC_MULT, hidden, dtype=torch.bfloat16),
        post_layer_mix=torch.zeros(tokens, mf.HC_MULT, 1),
        comb_res_mix=torch.zeros(tokens, mf.HC_MULT, mf.HC_MULT),
        fn=torch.zeros(mf.NMIX, mf.HC_MULT * hidden), hc_scale=torch.zeros(3), hc_base=torch.zeros(mf.NMIX),
        sinkhorn_repeat=20, norm_weight=torch.zeros(hidden, dtype=torch.bfloat16))


def _deps(small_batch_config, deep_gemm=True, split=1):
    kernels = SimpleNamespace(mhc_fused_post_pre_split_config=lambda t, h, n: small_batch_config,
                              compute_num_split=lambda block_k, k, grid: split)
    return kernels, SimpleNamespace(is_deep_gemm_supported=lambda: deep_gemm)


def test_decline_defers_to_the_stock_small_batch_dispatch():
    kernels, dg = _deps(small_batch_config=(6, 8, 128))
    assert mf.decline_reason(**_call_tensors(64), kernels=kernels, deep_gemm=dg) == "stock small-batch fused kernel"


def test_decline_without_deepgemm_the_stock_gemm_is_another_kernel():
    kernels, dg = _deps(None, deep_gemm=False)
    assert "DeepGEMM" in mf.decline_reason(**_call_tensors(64), kernels=kernels, deep_gemm=dg)


def test_decline_when_the_stock_split_is_above_one():
    kernels, dg = _deps(None, split=3)
    assert "split" in mf.decline_reason(**_call_tensors(1024), kernels=kernels, deep_gemm=dg)


def test_the_split_is_read_for_this_calls_tokens_at_call_time():
    seen = []
    kernels, dg = _deps(None)
    kernels.compute_num_split = lambda block_k, k, grid: seen.append((block_k, k, grid)) or 1
    mf.decline_reason(**_call_tensors(1000), kernels=kernels, deep_gemm=dg)
    assert seen == [(64, mf.HC_MULT * mf.HIDDEN, -(-1000 // 64))]


def test_decline_shape_norm_and_device():
    kernels, dg = _deps(None)
    assert "residual streams" in mf.decline_reason(**_call_tensors(64, hidden=1024), kernels=kernels, deep_gemm=dg)
    args = _call_tensors(64)
    args["norm_weight"] = None
    assert mf.decline_reason(**args, kernels=kernels, deep_gemm=dg) == "no fused norm"
    # Every check above passes on these; host tensors are not the kernel's.
    assert "device" in mf.decline_reason(**_call_tensors(64), kernels=kernels, deep_gemm=dg)


class _Op:
    def forward_cuda(self, *a, **k):  # stands in for MHCFusedPostPreOp.forward_cuda
        raise AssertionError("never called directly")

    def forward_native(self, *a, **k):
        raise AssertionError("never called directly")


def _layer(on_cuda=True):
    op = _Op()
    op._forward_method = op.forward_cuda if on_cuda else op.forward_native
    return SimpleNamespace(mhc_fused_post_pre_op=op, rms_norm_eps=1e-5, hc_eps=1e-6,
                           mhc_post_mult_value=2.0, mhc_sinkhorn_iterations=20)


def _wrapper(monkeypatch, decline):
    calls = []
    monkeypatch.setattr(mf, "decline_reason", lambda *a, **k: decline)
    monkeypatch.setattr(mf, "fused_post_pre", lambda *a, **k: calls.append(("fused", a)) or "fused")

    def stock(self, *a, **k):
        calls.append(("stock", a, k))
        return "stock"
    return mf.make_hc_fused_post_pre(stock, _Op, lambda: "lib", "kernels", "deep_gemm"), calls


def test_eligible_call_takes_the_fused_kernel_with_the_layers_constants(monkeypatch):
    fn, calls = _wrapper(monkeypatch, decline=None)
    assert fn(_layer(), "x", "res", "post", "comb", "hc_fn", "scale", "base",
              norm_weight="w", norm_eps=1e-5) == "fused"
    (_, args), = calls
    assert args[:2] == ("lib", "kernels")
    assert args[2:9] == ("x", "res", "post", "comb", "hc_fn", "scale", "base")
    assert args[9:] == (1e-5, 1e-6, 1e-6, 2.0, 20, "w", 1e-5)


def test_declined_call_is_the_stock_method_unchanged(monkeypatch):
    fn, calls = _wrapper(monkeypatch, decline="stock small-batch fused kernel")
    assert fn(_layer(), "x", "res", "post", "comb", "hc_fn", "scale", "base",
              norm_weight="w", norm_eps=1e-5) == "stock"
    assert calls == [("stock", ("x", "res", "post", "comb", "hc_fn", "scale", "base"),
                      {"norm_weight": "w", "norm_eps": 1e-5})]


def test_a_layer_whose_op_is_not_on_forward_cuda_stays_stock(monkeypatch):
    fn, calls = _wrapper(monkeypatch, decline=None)
    assert fn(_layer(on_cuda=False), "x", "res", "post", "comb", "hc_fn", "scale", "base") == "stock"
    assert [c[0] for c in calls] == ["stock"]
