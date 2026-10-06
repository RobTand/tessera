"""The fused mHC override's install, dispatch and contract rules (#783), device-less.

The kernel's bitwise identity is the CUDA-gated ``test_mhc_fusion_cuda.py``
and ``experiments/mhc/mhc_fused_probe.py``; this file pins the rules that
decide whether that kernel is ever called.
"""
from __future__ import annotations

import re
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tessera.serving import contract, ext, flags
from tessera.serving import glm53_prefill as gp
from tessera.serving import mhc_fusion as mf


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_probe_bitwise_gate_rejects_signed_zero(dtype):
    probe = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                               "experiments/mhc/mhc_probe.py"))
    positive = torch.tensor([0.0, 1.0], dtype=dtype)
    negative = torch.tensor([-0.0, 1.0], dtype=dtype)
    assert torch.equal(positive, negative)  # The old value gate accepts this mutation.
    assert not probe["compare"](positive, negative)["equal"]
    assert probe["bits_compare"](positive, negative)["bits_differing"] == 1


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


@pytest.mark.parametrize("tokens,ctas", [(33, 48), (768, 48), (769, 48), (1024, 48), (1024, 32),
                                         (1536, 48), (1537, 48), (2048, 48), (8192, 48), (100, 1)])
def test_a_second_m16_tile_only_when_it_removes_a_wave(tokens, ctas):
    tm = mf.tile_tokens(tokens, ctas)
    assert tm in (mf.TILE_QUANTUM, 2 * mf.TILE_QUANTUM) and tm <= mf.TILE_MAX
    waves = lambda t: -(-tokens // (t * ctas))  # noqa: E731
    assert (tm == 2 * mf.TILE_QUANTUM) == (waves(mf.TILE_QUANTUM) > 1 and waves(2 * mf.TILE_QUANTUM) == 1)


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


def _install_modules(monkeypatch, *, incompatible=False):
    class Layer:
        def hc_fused_post_pre(self, x, residual, post, comb, hc_fn, hc_scale, hc_base,
                              norm_weight=None, norm_eps=0.0):
            return "stock"
    if incompatible:
        Layer.hc_fused_post_pre = lambda self, x: "stock"
    kernels, dg = _deps(None)
    modules = (SimpleNamespace(Glm5NextDecoderLayer=Layer),
               SimpleNamespace(MHCFusedPostPreOp=_Op), SimpleNamespace(), kernels, dg)
    monkeypatch.setattr(mf, "import_modules", lambda names: (modules, ""))
    monkeypatch.setattr(mf, "_DECIDED", {})
    monkeypatch.setenv(mf.FLAG, "1")
    config = SimpleNamespace(model_config=SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["Glm5NextForCausalLM"])))
    monkeypatch.setattr(gp, "is_glm5next", lambda c: True)
    return config, Layer, modules


def test_recorded_source_identity_stamps_without_hashing_in_dev_mode(monkeypatch, capsys):
    config, layer, _ = _install_modules(monkeypatch)
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    def no_hash(*args):
        raise AssertionError("development mode must not compute historical source digests")
    monkeypatch.setattr(mf, "match_modules", no_hash)
    assert mf.install_mhc_fusion(config)
    assert layer.hc_fused_post_pre._tessera_mhc_fused
    assert "[DEV-MODE]" in capsys.readouterr().out


def test_recorded_source_identity_still_declines_in_certified_mode(monkeypatch):
    config, layer, _ = _install_modules(monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    monkeypatch.setattr(mf, "match_modules", lambda *args: (None, "recorded source differs"))
    assert not mf.install_mhc_fusion(config)
    assert not getattr(layer.hc_fused_post_pre, "_tessera_mhc_fused", False)


def test_actual_method_interface_still_declines_in_dev_mode(monkeypatch):
    config, layer, _ = _install_modules(monkeypatch, incompatible=True)
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    assert not mf.install_mhc_fusion(config)
    assert not getattr(layer.hc_fused_post_pre, "_tessera_mhc_fused", False)


def test_actual_missing_dispatch_still_declines_in_dev_mode(monkeypatch):
    config, layer, modules = _install_modules(monkeypatch)
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    modules[3].compute_num_split = None
    assert not mf.install_mhc_fusion(config)
    assert not getattr(layer.hc_fused_post_pre, "_tessera_mhc_fused", False)


@pytest.mark.parametrize("certified", [False, True])
def test_compiler_path_identity_uses_returned_library_or_refuses(monkeypatch, tmp_path, capsys, certified):
    import torch.utils.cpp_extension as cpp
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0" if certified else "1")
    source = tmp_path / "kernel.cu"
    source.write_text("// source for the injected compiler result\n")
    returned = tmp_path / "compiler-selected.so"
    monkeypatch.setattr(mf, "source_path", lambda: source)
    monkeypatch.setattr(ext, "_nvcc_for_build", lambda: "/nvcc")
    monkeypatch.setattr(ext, "toolchain_report", lambda torch: {"complete": True})
    monkeypatch.setattr(mf.subprocess, "check_output", lambda *a, **k: "nvcc test version")
    monkeypatch.setattr(cpp, "load", lambda **kwargs: str(returned))
    opened = []
    launch = SimpleNamespace()
    monkeypatch.setattr(mf.ctypes, "CDLL", lambda path: opened.append(path) or
                        SimpleNamespace(tessera_mhc_fused_post_pre=launch))
    if certified:
        with pytest.raises(RuntimeError, match="JIT returned"):
            mf.MhcFusedLibrary(str(tmp_path))
        assert not opened
    else:
        lib = mf.MhcFusedLibrary(str(tmp_path))
        assert lib.path == str(returned) and opened == [str(returned)]
        assert "[DEV-MODE]" in capsys.readouterr().out


def test_actual_launch_failure_remains_a_refusal():
    lib = mf.MhcFusedLibrary.__new__(mf.MhcFusedLibrary)
    lib._launch = lambda *args: 9
    with pytest.raises(RuntimeError, match="cudaError 9"):
        lib.launch(mf._Params(), 1, 0)
