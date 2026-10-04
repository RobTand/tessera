"""One-pass mHC post + pre for GLM-5.3 prefill (#783): build, eligibility, install.

Stock vLLM runs each mHC site of a prefill chunk as three passes over the
residual stream (``mhc_post_tilelang_kernel``, DeepGEMM's
``sm120_tf32_hc_prenorm_gemm_impl`` at the split ``compute_num_split`` picks,
``mhc_pre_big_fuse_with_norm_tilelang_kernel``): 144 KiB of DRAM traffic per
token per site against an 80 KiB floor.  ``csrc/mhc_fused.cu`` runs all three
per 16-token tile in one kernel: the new residual is written once and read back
from L2 by the GEMM and the pre, so DRAM moves the floor.

**Identity.**  Every output -- residual, post mix, comb mix, layer input -- is
required to be bitwise equal to the stock sequence at the same split.  The
kernel restates the stock arithmetic (see its header and
``docs/design/mhc-fusion-783.md``); the gate is the GPU comparison in
``tests/test_mhc_fusion_cuda.py`` and ``experiments/mhc/mhc_fused_probe.py``.
The split is read at call time from ``tilelang_kernels.compute_num_split``,
the attribute :class:`tessera.serving.glm53_prefill.SplitForcer` replaces, so
an exact-SP shard keeps the full batch's split here exactly as on stock.

**Install.**  ``TESSERA_GLM53_MHC_FUSED``: unset or ``0`` (default, stock) or
``1``.  The packaged contract must publish the override
(``stock_kernel_overrides``, kind ``model_method``) under that flag, naming
this loader and library, or the install refuses.  On a
Glm5Next serve whose touched vLLM modules are the inspected ones,
:func:`install_mhc_fusion` rebinds ``Glm5NextDecoderLayer.hc_fused_post_pre``
to a wrapper that takes the fused kernel when :func:`decline_reason` is ``None``
and calls the stock method otherwise.  A shape the stock op sends to its
small-batch fused kernel (``mhc_fused_post_pre_split_config`` not ``None``),
any other dtype, layout or constant, or a missing DeepGEMM declines to stock
for that call.  It never changes which split a call runs at.
"""
from __future__ import annotations

import ctypes
import hashlib
import logging
import os
from pathlib import Path
import subprocess
import threading
from typing import Any, Callable

from .stock_interface import InspectedInterface, import_modules, match_modules

_log = logging.getLogger(__name__)

HC_MULT = 4
HIDDEN = 4096
NMIX = HC_MULT * 2 + HC_MULT * HC_MULT
#: The kernel's token tile: one m16 MMA tile.  Must match ``TM`` in ``csrc/mhc_fused.cu``.
TILE_TOKENS = 16

MODULE_PREFIX = "tessera_mhc_fused_"
SOURCE = "mhc_fused.cu"
#: TileLang compiles the stock post/pre with ``-O3`` and no fast math (pass config
#: ``tl.enable_fast_math`` defaults off); the stock SASS has IEEE division and full
#: ``expf``.  Matching that is part of the identity, so no ``-use_fast_math`` here.
FLAGS = ("-O3", "-std=c++17", "--threads", "1", "-gencode=arch=compute_121a,code=sm_121a")

#: The vLLM modules the rebind reads or replaces, in digest order.
MODULES = (
    "vllm.models.glm5next.common.model",
    "vllm.model_executor.layers.mhc",
    "vllm.model_executor.kernels.mhc.tilelang",
    "vllm.model_executor.kernels.mhc.tilelang_kernels",
    "vllm.utils.deep_gemm",
)


def _interfaces() -> tuple[InspectedInterface, ...]:
    """These modules' pins, read from the SP interface table that already holds them."""
    from .glm53_prefill import SP_MODULES, _INTERFACES

    index = [SP_MODULES.index(name) for name in MODULES]
    return tuple(InspectedInterface(i.name, tuple(i.digests[k] for k in index)) for i in _INTERFACES)


FLAG = "TESSERA_GLM53_MHC_FUSED"
#: What the override replaces, as the contract's ``model_method`` kind names it.
STOCK_OBJECT = {
    "method": "vllm.models.glm5next.common.model.Glm5NextDecoderLayer.hc_fused_post_pre",
    "kernels": "mhc_post_tilelang_kernel+sm120_tf32_hc_prenorm_gemm_impl+"
               "mhc_pre_big_fuse_with_norm_tilelang_kernel",
}


def enabled() -> bool:
    from . import flags

    return flags.latched_bool(FLAG, meaning="the fused mHC post/pre kernel (#783)")


# ------------------------------------------------------------------ build and load


class _Params(ctypes.Structure):
    """``tessera_mhc::Params`` field for field."""

    _fields_ = [(name, ctypes.c_void_p) for name in (
        "x", "residual", "post", "comb", "fn", "scale", "base", "norm_w",
        "residual_out", "post_mix", "comb_mix", "layer_input", "part", "sqrsum")] + [
        ("tokens", ctypes.c_int), ("splits", ctypes.c_int),
        ("rms_eps", ctypes.c_float), ("pre_eps", ctypes.c_float), ("sinkhorn_eps", ctypes.c_float),
        ("post_mult", ctypes.c_float), ("norm_eps", ctypes.c_float), ("sinkhorn_repeat", ctypes.c_int),
        ("clocks", ctypes.c_void_p)]


def source_path() -> Path:
    from . import ext

    return Path(ext.csrc_dir()) / SOURCE


class MhcFusedLibrary:
    """The JIT-built kernel, identified by its source, flags and compiler."""

    def __init__(self, build_directory: str | None = None):
        import torch
        from torch.utils.cpp_extension import load

        from . import ext

        nvcc = ext._nvcc_for_build()
        if not nvcc or not ext.toolchain_report(torch)["complete"]:
            raise RuntimeError("mHC fusion build requires the complete resolved CUDA/ninja toolchain")
        self.nvcc_version = subprocess.check_output([nvcc, "--version"], text=True)
        source = source_path()
        self.source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        self.build_id = hashlib.sha256(
            (self.source_sha256 + repr(FLAGS) + self.nvcc_version).encode()).hexdigest()
        self.name = f"{MODULE_PREFIX}{self.build_id[:16]}"
        if build_directory is None:
            from torch.utils.cpp_extension import _get_build_directory

            build_directory = _get_build_directory("tessera_mhc_fused", verbose=False)
        Path(build_directory).mkdir(parents=True, exist_ok=True)
        self.library_path = Path(build_directory) / f"{self.name}.so"
        built = load(name=self.name, sources=[str(source)], build_directory=build_directory,
                     extra_cuda_cflags=list(FLAGS), extra_cflags=["-O3"],
                     is_python_module=False, verbose=False)
        if Path(built).resolve() != self.library_path.resolve():
            raise RuntimeError(f"JIT returned {built}, not the declared {self.library_path}")
        self.path = str(self.library_path)
        lib = ctypes.CDLL(str(self.library_path))
        self._launch = lib.tessera_mhc_fused_post_pre
        self._launch.argtypes = [ctypes.POINTER(_Params), ctypes.c_int, ctypes.c_void_p]
        self._launch.restype = ctypes.c_int

    def launch(self, params: _Params, grid: int, stream: int) -> None:
        err = self._launch(ctypes.byref(params), int(grid), ctypes.c_void_p(stream))
        if err:
            raise RuntimeError(f"tessera mHC fused kernel launch failed: cudaError {err}")


_LIBRARY: MhcFusedLibrary | None = None
_LIBRARY_LOCK = threading.Lock()


def library() -> MhcFusedLibrary:
    global _LIBRARY
    with _LIBRARY_LOCK:
        if _LIBRARY is None:
            _LIBRARY = MhcFusedLibrary()
        return _LIBRARY


def default_grid(torch: Any, device: Any) -> int:
    """One persistent CTA per SM: the kernel's register use admits one 256-thread CTA per SM.

    Measured, not assumed: on GB10 at 1024 tokens, split 1, the grid sweep
    (12/16/24/32/48) was monotone and 48 was fastest (timing receipt
    ``timing-6f19bf03``, PB ``7b0c4d46``); an L2-capacity cap on resident
    tiles (24) was slower.  ``TESSERA_MHC_FUSED_CTAS`` overrides it for
    measurement only.
    """
    override = os.environ.get("TESSERA_MHC_FUSED_CTAS")
    if override:
        return max(1, int(override))
    return torch.cuda.get_device_properties(device).multi_processor_count


# ------------------------------------------------------------------ the call


def decline_reason(x: Any, residual: Any, post_layer_mix: Any, comb_res_mix: Any, fn: Any,
                   hc_scale: Any, hc_base: Any, sinkhorn_repeat: int, norm_weight: Any,
                   kernels: Any, deep_gemm: Any) -> str | None:
    """Why this call must run the stock op; ``None`` when the fused kernel computes it exactly."""
    import torch

    if norm_weight is None:
        return "no fused norm"
    if residual.dim() < 2 or tuple(residual.shape[-2:]) != (HC_MULT, HIDDEN):
        return f"residual streams {tuple(residual.shape[-2:])}, kernel is ({HC_MULT}, {HIDDEN})"
    tokens = residual.numel() // (HC_MULT * HIDDEN)
    if tokens == 0:
        return "no tokens"
    if kernels.mhc_fused_post_pre_split_config(tokens, HIDDEN, HC_MULT) is not None:
        return "stock small-batch fused kernel"
    if not deep_gemm.is_deep_gemm_supported():
        return "stock pre-norm GEMM is not DeepGEMM's"
    if sinkhorn_repeat < 1:
        return "sinkhorn_repeat < 1"
    expect = ((x, torch.bfloat16, tokens * HIDDEN), (residual, torch.bfloat16, tokens * HC_MULT * HIDDEN),
              (post_layer_mix, torch.float32, tokens * HC_MULT),
              (comb_res_mix, torch.float32, tokens * HC_MULT * HC_MULT),
              (fn, torch.float32, NMIX * HC_MULT * HIDDEN), (hc_scale, torch.float32, 3),
              (hc_base, torch.float32, NMIX), (norm_weight, torch.bfloat16, HIDDEN))
    for tensor, dtype, numel in expect:
        if not tensor.is_cuda or tensor.dtype != dtype or tensor.numel() != numel \
                or not tensor.is_contiguous() or tensor.device != residual.device:
            return "dtype, device, layout or size is not the kernel's"
    return None


def fused_post_pre(lib: MhcFusedLibrary, kernels: Any, x: Any, residual: Any, post_layer_mix: Any,
                   comb_res_mix: Any, fn: Any, hc_scale: Any, hc_base: Any, rms_eps: float,
                   hc_pre_eps: float, hc_sinkhorn_eps: float, hc_post_mult_value: float,
                   sinkhorn_repeat: int, norm_weight: Any, norm_eps: float,
                   grid: int | None = None, clocks: Any = None) -> tuple[Any, Any, Any, Any]:
    """``mhc_fused_post_pre_tilelang``'s outputs from one kernel (caller has checked eligibility).

    ``clocks`` (diagnostic): an int64 CUDA tensor of ``tiles * 5`` that receives each
    tile's phase-boundary ``%globaltimer`` stamps.
    """
    import torch

    outer = residual.shape[:-2]
    tokens = residual.numel() // (HC_MULT * HIDDEN)
    # The stock rule, read now: under SplitForcer.full_batch this is the full batch's split.
    splits = kernels.compute_num_split(64, HC_MULT * HIDDEN, -(-tokens // 64))
    dev = residual.device
    residual_cur = torch.empty((tokens, HC_MULT, HIDDEN), dtype=torch.bfloat16, device=dev)
    post_mix = torch.empty((tokens, HC_MULT), dtype=torch.float32, device=dev)
    comb_mix = torch.empty((tokens, HC_MULT * HC_MULT), dtype=torch.float32, device=dev)
    layer_input = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=dev)
    part = torch.empty((splits, tokens, NMIX), dtype=torch.float32, device=dev)
    sqrsum = torch.empty((splits, tokens), dtype=torch.float32, device=dev)
    params = _Params(
        x.data_ptr(), residual.data_ptr(), post_layer_mix.data_ptr(), comb_res_mix.data_ptr(),
        fn.data_ptr(), hc_scale.data_ptr(), hc_base.data_ptr(), norm_weight.data_ptr(),
        residual_cur.data_ptr(), post_mix.data_ptr(), comb_mix.data_ptr(), layer_input.data_ptr(),
        part.data_ptr(), sqrsum.data_ptr(), tokens, int(splits),
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, norm_eps, int(sinkhorn_repeat),
        None if clocks is None else clocks.data_ptr())
    tiles = -(-tokens // TILE_TOKENS)
    grid = min(tiles, grid if grid is not None else default_grid(torch, dev))
    lib.launch(params, grid, torch.cuda.current_stream(dev).cuda_stream)
    return (residual_cur.view(*outer, HC_MULT, HIDDEN), post_mix.view(*outer, HC_MULT, 1),
            comb_mix.view(*outer, HC_MULT, HC_MULT), layer_input.view(*outer, HIDDEN))


def make_hc_fused_post_pre(stock: Callable, op_cls: Any, lib_getter: Callable[[], MhcFusedLibrary],
                           kernels: Any, deep_gemm: Any) -> Callable:
    """The rebound ``Glm5NextDecoderLayer.hc_fused_post_pre``: fused when exact, else stock.

    The layer method is the hook, not ``MHCFusedPostPreOp.forward_cuda``: a
    ``CustomOp`` binds its forward when it is constructed, while the layer
    method is looked up on every call (by the stock forward and the SP one
    alike).  A layer whose op does not dispatch to the stock ``forward_cuda``
    (a disabled custom op, another platform) keeps the stock method.
    """

    def hc_fused_post_pre(self, x, residual, post, comb, hc_fn, hc_scale, hc_base,
                          norm_weight=None, norm_eps=0.0):
        op = self.mhc_fused_post_pre_op
        bound = getattr(op, "_forward_method", None)
        why = None if getattr(bound, "__func__", None) is op_cls.forward_cuda else "op not on forward_cuda"
        if why is None:
            why = decline_reason(x, residual, post, comb, hc_fn, hc_scale, hc_base,
                                 self.mhc_sinkhorn_iterations, norm_weight, kernels, deep_gemm)
        if why is not None:
            return stock(self, x, residual, post, comb, hc_fn, hc_scale, hc_base,
                         norm_weight=norm_weight, norm_eps=norm_eps)
        return fused_post_pre(lib_getter(), kernels, x, residual, post, comb, hc_fn, hc_scale, hc_base,
                              self.rms_norm_eps, self.hc_eps, self.hc_eps, self.mhc_post_mult_value,
                              self.mhc_sinkhorn_iterations, norm_weight, norm_eps)

    hc_fused_post_pre._tessera_mhc_fused = True  # type: ignore[attr-defined]
    return hc_fused_post_pre


_DECIDED: dict[int, bool] = {}


def install_mhc_fusion(config: Any) -> bool:
    """Rebind ``Glm5NextDecoderLayer.hc_fused_post_pre`` on an inspected serve; True when active."""
    if not enabled():
        return False
    from . import contract

    refusal = contract.stock_kernel_override_refusal(FLAG, kind="model_method", overrides=STOCK_OBJECT,
                                                     loaded_by=__name__, library_prefix=MODULE_PREFIX)
    if refusal:
        raise RuntimeError(refusal)
    if id(config) in _DECIDED:
        return _DECIDED[id(config)]
    from .glm53_prefill import is_glm5next

    reasons = [] if config is not None and is_glm5next(config) else ["not a Glm5Next model"]
    modules = None
    if not reasons:
        modules, why = import_modules(MODULES)
        if modules is None:
            reasons = [why]
        else:
            interface, why = match_modules(modules, MODULES, _interfaces())
            if interface is None:
                reasons = [why]
    if reasons:
        _log.warning("tessera.mhc_fusion: declined, stock mHC: %s", "; ".join(reasons))
        _DECIDED[id(config)] = False
        return False
    model, layers, _tilelang, kernels, deep_gemm = modules
    cls = model.Glm5NextDecoderLayer
    if not getattr(cls.hc_fused_post_pre, "_tessera_mhc_fused", False):
        cls.hc_fused_post_pre = make_hc_fused_post_pre(cls.hc_fused_post_pre, layers.MHCFusedPostPreOp,
                                                       library, kernels, deep_gemm)
    _log.warning("tessera.mhc_fusion: installed (interface %s): split-k mHC sites run one fused "
                 "post/GEMM/pre kernel, bitwise to stock at the stock split", interface.name)
    _DECIDED[id(config)] = True
    return True
