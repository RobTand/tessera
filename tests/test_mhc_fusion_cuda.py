"""The fused mHC kernel is bitwise equal to the stock split-k sequence (#783).

Needs a CUDA device, the serving image's vLLM (its TileLang mHC kernels and
DeepGEMM) and the toolchain to build ``csrc/mhc_fused.cu``.  Random projections
here; the served checkpoint's own layers, adversarial rows and the timing are
``experiments/mhc/mhc_fused_probe.py``.
"""
from __future__ import annotations

import importlib.util

import pytest
import torch

CUDA = torch.cuda.is_available()
VLLM = importlib.util.find_spec("vllm") is not None
pytestmark = [pytest.mark.skipif(not CUDA, reason="needs a CUDA device"),
              pytest.mark.skipif(not VLLM, reason="needs vLLM's stock mHC kernels (the serving image)")]

RMS_EPS, HC_EPS, POST_MULT, SINKHORN = 1e-5, 1e-6, 2.0, 20


@pytest.fixture(scope="module")
def stack():
    import vllm.model_executor.kernels.mhc.tilelang_kernels as tk
    from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_tilelang
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    from tessera.serving import mhc_fusion as mf

    if not is_deep_gemm_supported():
        pytest.skip("DeepGEMM unsupported here: the stock pre-norm GEMM is another kernel")
    return SimpleStack(tk, mhc_fused_post_pre_tilelang, mf, mf.library())


class SimpleStack:
    def __init__(self, tk, stock, mf, lib):
        self.tk, self.stock, self.mf, self.lib = tk, stock, mf, lib


class _Forced:
    def __init__(self, tk, full):
        self.tk, self.orig, self.full = tk, tk.compute_num_split, full

    def __enter__(self):
        self.tk.compute_num_split = lambda bk, k, g: self.orig(bk, k, -(-self.full // 64))

    def __exit__(self, *exc):
        self.tk.compute_num_split = self.orig


def _inputs(tokens, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(tokens, 4096, device="cuda", generator=g).bfloat16()
    res = torch.randn(tokens, 4, 4096, device="cuda", generator=g).bfloat16()
    post = torch.rand(tokens, 4, 1, device="cuda", generator=g) * POST_MULT
    comb = torch.softmax(torch.randn(tokens, 4, 4, device="cuda", generator=g), -1)
    fn = torch.randn(24, 16384, device="cuda", generator=g) * 0.01
    scale = torch.rand(3, device="cuda", generator=g)
    base = torch.randn(24, device="cuda", generator=g) * 0.1
    norm = (torch.rand(4096, device="cuda", generator=g) + 0.5).bfloat16()
    return x, res, post, comb, fn, scale, base, norm


@pytest.mark.parametrize("tokens,full_batch", [(33, 33), (1024, 2048), (2048, 2048), (2049, 2049)])
def test_fused_equals_stock_bitwise(stack, tokens, full_batch):
    x, res, post, comb, fn, scale, base, norm = _inputs(tokens, tokens + full_batch)
    with _Forced(stack.tk, full_batch):
        ref = stack.stock(x, res, post, comb, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT,
                          SINKHORN, 1, 1, norm_weight=norm, norm_eps=RMS_EPS)
        got = stack.mf.fused_post_pre(stack.lib, stack.tk, x, res, post, comb, fn, scale, base, RMS_EPS,
                                      HC_EPS, HC_EPS, POST_MULT, SINKHORN, norm, RMS_EPS)
    for name, a, b in zip(("residual", "post_mix", "comb_mix", "layer_input"), got, ref):
        assert a.shape == b.shape, name
        assert torch.equal(a, b), f"{name}: {int((a != b).sum())} elements differ"
