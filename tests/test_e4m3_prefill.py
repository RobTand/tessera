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
