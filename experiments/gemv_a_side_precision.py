#!/usr/bin/env python3
"""Where the window-GEMV lane and its ``_scaled_mm`` fallback actually differ (issue #110).

Both arms serve the SAME bytes -- ``prepare_fp8_gemv`` refuses unless the
lane's kernel decode equals the torch decoder's byte for byte -- and both run
the SAME activation quantiser.  What is left is arithmetic, and this script
prices each candidate term against an fp64 reference of the product both arms
claim to compute:

    ref[m, n] = sum_k (a_q[m, k] * a_scale[m]) * (w[n, k] * scale_b[n])   in fp64

Arms, each read as the serve consumes it (bf16):

* ``scaled_mm``     -- the fallback: fp8 x fp8, both scales in the epilogue.
* ``gemv_scaled_A`` -- the lane as it stands: ``(a_q.float() * a_scale)`` is
                       rounded to **bf16** before the kernel reads it.
* ``gemv_code_A``   -- the same kernel handed the raw E4M3 codes (exact in
                       bf16), with ``a_scale`` applied to the fp32 output.
* ``bf16_A_only``   -- an fp64 control carrying ONLY ``gemv_scaled_A``'s A-side
                       rounding, so that term reads apart from any kernel.

``identical`` is the fraction of output elements bit-identical to the
fallback's -- the quantity a served KL actually sees.  ``self`` is the lane
run twice on identical input: the reduction is an ``atomicAdd`` tree, so this
is the accumulation-order floor, measured rather than asserted.

Run it on a GPU box through the pool; it needs nvcc to build the lane.
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tessera.serving import fp8_gemv  # noqa: E402

FP8_MAX = 448.0


def fp8_quant(x: torch.Tensor):
    """Per-token dynamic E4M3, the shape ``native_fp8_quant`` returns."""
    xf = x.float()
    amax = xf.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = amax / FP8_MAX
    q = (xf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale.contiguous()


def synthetic_parsed(rows: int, cols: int, rates, seed: int = 0):
    from tessera.alphabet import E4M3_GRID
    from tessera.manifest import BodyKind, RotationState, ScalePlaneKind

    g = torch.Generator(device="cpu").manual_seed(seed)
    rate = torch.tensor(list(rates), dtype=torch.int64)
    body = (torch.randint(0, 1 << 16, (rows, cols), generator=g) & ((1 << rate) - 1)).to(torch.uint8)
    codes = torch.randint(0, 256, (1 << 14,), generator=g).to(torch.uint8)
    scale_rows = (torch.rand(rows, generator=g, dtype=torch.float16) + 0.5)
    unit = SimpleNamespace(
        body=BodyKind.WINDOW, scale_plane=ScalePlaneKind.CHANNEL,
        release_index=torch.zeros(0, dtype=torch.int64), diagonals=None,
        rotation=RotationState.NONE, window_codes=codes.cpu(), window_bits=14,
        scale_rows=scale_rows, scale_global=0.75, body_bits=body, rates=tuple(rates),
        initial_state=None, span=1)
    return SimpleNamespace(name="weight", unit=unit, grid=E4M3_GRID,
                           forests=None, code=None, body=BodyKind.WINDOW)


def rel(a: torch.Tensor, ref: torch.Tensor) -> float:
    return float((a.double() - ref).norm() / ref.norm())


def worst(a: torch.Tensor, ref: torch.Tensor) -> float:
    """Largest per-element error as a fraction of the row's own RMS output."""
    d = (a.double() - ref).abs()
    rms = ref.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-300)
    return float((d / rms).max())


def identical(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.view(torch.int16) == b.view(torch.int16)).double().mean())


def analytic_leg(rows: int = 1024, cols: int = 1024, seed: int = 5) -> dict:
    """The A-side term with NO kernel in it: CPU, fp64, three lines of arithmetic.

    Reproducible on any box.  It answers two questions the kernel cannot be
    asked to answer about itself: is an E4M3 code exact in bf16 (yes), and is
    ``code * a_scale`` (no) -- and prices what the second costs at the output
    of a K-term dot product.
    """
    torch.manual_seed(seed)
    x = torch.randn(8, cols).bfloat16()
    a_q, a_scale = fp8_quant(x)
    codes = a_q.float()
    exact_codes = bool(torch.equal(codes.to(torch.bfloat16).float(), codes))
    dequant = codes.double() * a_scale.double()
    folded = (codes * a_scale).to(torch.bfloat16).double()
    a_term = float((folded - dequant).norm() / dequant.norm())

    w = (torch.randint(0, 254, (rows, cols)).to(torch.uint8)
         .view(torch.float8_e4m3fn).float().double())
    w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
    sb = (torch.rand(rows).double() + 0.5)[:, None]
    ref = dequant @ (w * sb).t()
    got = folded @ (w * sb).t()
    out_term = float((got - ref).norm() / ref.norm())
    return {"e4m3_codes_exact_in_bf16": exact_codes,
            "a_side_rel_rms": a_term, "output_rel_rms": out_term,
            "bf16_half_ulp": 2.0 ** -9, "cols": cols, "rows": rows}


def main() -> int:
    leg = analytic_leg()
    print("analytic leg (CPU, no kernel):")
    print(f"  every E4M3 code exact in bf16          : {leg['e4m3_codes_exact_in_bf16']}")
    print(f"  rel rms of bf16(code * a_scale)        : {leg['a_side_rel_rms']:.3e} "
          f"(half a bf16 ulp is {leg['bf16_half_ulp']:.3e})")
    print(f"  rel rms it costs at the K=%d output   : %.3e" % (leg["cols"], leg["output_rel_rms"]))
    if not torch.cuda.is_available():
        print("no CUDA device: the kernel leg needs one")
        dest = os.environ.get("OUT")
        if dest:
            with open(dest, "w") as fh:
                json.dump({"analytic": leg, "arms": None}, fh, indent=2)
        return 0
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, cols = int(os.environ.get("ROWS", 1024)), int(os.environ.get("COLS", 1024))
    parsed = synthetic_parsed(rows, cols, (4,) * cols)
    expected = fp8_gemv.reference_bytes_for_test(parsed)
    holder = fp8_gemv.prepare_fp8_gemv([("weight", parsed)], device="cuda", expected=expected)
    tensors, meta, _rows, _cols = holder.op_args()

    w_bytes, scale_b = expected
    w = w_bytes.view(torch.float8_e4m3fn)
    w_f64 = w.float().double()
    sb64 = scale_b.double()[:, None]
    assert torch.isfinite(w.float()).all(), "an E4M3 NaN byte reached the reference"

    out = []
    for m in (1, 2, 4, 8):
        g = torch.Generator(device="cuda").manual_seed(1234 + m)
        x = torch.randn(m, cols, device="cuda", generator=g).bfloat16()
        a_q, a_scale = fp8_quant(x.contiguous())
        a64 = a_q.float().double() * a_scale.double()
        ref = a64 @ (w_f64 * sb64).t()
        assert torch.isfinite(ref).all()

        y_mm = torch._scaled_mm(a_q, w.t(), scale_a=a_scale,
                                scale_b=scale_b.view(1, -1), out_dtype=torch.bfloat16)
        a_now = (a_q.float() * a_scale).to(torch.bfloat16)
        f_now = fp8_gemv._gemv_path(a_now, tensors, meta)
        a_code = a_q.float().to(torch.bfloat16)     # exact: every E4M3 code fits bf16
        f_fix = fp8_gemv._gemv_path(a_code, tensors, meta) * a_scale
        f_self = fp8_gemv._gemv_path(a_code, tensors, meta) * a_scale
        y_now, y_fix = f_now.to(torch.bfloat16), f_fix.to(torch.bfloat16)
        y_ctrl = (a_now.double() @ (w_f64 * sb64).t()).to(torch.bfloat16)

        row = {
            "M": m,
            "rel": {"scaled_mm": rel(y_mm, ref), "gemv_scaled_A": rel(y_now, ref),
                    "gemv_code_A": rel(y_fix, ref), "bf16_A_only": rel(y_ctrl, ref),
                    "gemv_code_A_fp32": rel(f_fix, ref), "gemv_scaled_A_fp32": rel(f_now, ref)},
            "worst": {"scaled_mm": worst(y_mm, ref), "gemv_scaled_A": worst(y_now, ref),
                      "gemv_code_A": worst(y_fix, ref)},
            "identical_to_fallback": {"gemv_scaled_A": identical(y_now, y_mm),
                                      "gemv_code_A": identical(y_fix, y_mm)},
            "lane_self_rel_fp32": rel(f_self, f_fix.double()),
            "lane_self_identical": identical(f_self.to(torch.bfloat16), y_fix),
        }
        out.append(row)
        r, w_, i = row["rel"], row["worst"], row["identical_to_fallback"]
        print(f"M={m:2d}  vs fp64 (bf16 out): scaled_mm={r['scaled_mm']:.3e} "
              f"gemv_scaled_A={r['gemv_scaled_A']:.3e} gemv_code_A={r['gemv_code_A']:.3e} "
              f"bf16_A_only={r['bf16_A_only']:.3e}")
        print(f"      fp32 out: gemv_scaled_A={r['gemv_scaled_A_fp32']:.3e} "
              f"gemv_code_A={r['gemv_code_A_fp32']:.3e}   lane self={row['lane_self_rel_fp32']:.3e}")
        print(f"      worst elt / row-rms: mm={w_['scaled_mm']:.3e} now={w_['gemv_scaled_A']:.3e} "
              f"fixed={w_['gemv_code_A']:.3e}")
        print(f"      bf16 outputs identical to the fallback: now={i['gemv_scaled_A']:.4%} "
              f"fixed={i['gemv_code_A']:.4%}   (lane vs itself {row['lane_self_identical']:.4%})")

    dest = os.environ.get("OUT")
    if dest:
        with open(dest, "w") as fh:
            json.dump({"analytic": leg, "rows": rows, "cols": cols, "arms": out}, fh, indent=2)
        print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
