"""Where the fused-role A4 product's 1.67% comes from, component by component.

``tests/test_serving_nvfp4_route.py::test_fused_roles_stack_with_their_own_row_slices``
compares the native fused product against a reference whose three inputs are
the A-side VALUES, the shared-global weight decode and the epilogue scale. A
residual of ``0.765625/45.75 = 1.67%`` against an ``8e-3`` bound is too large
to be fp32 accumulation and too small to be a scale error, so this walks the
components the test folds together:

1. per-role error localization (an error in the shared-global move or the row
   cut concentrates on the roles that were moved; an A-side error is spread);
2. the A-side: the reference's own ``argmin`` level choice against the codes
   the registered ``torch.ops._C.scaled_fp4_quant`` actually emits, with the
   tie geometry of every differing element;
3. the same product rebuilt from the STOCK op's codes, which decides whether
   the residual is the reference's rounding rule or the route's numbers.

Run inside the serve image (CUDA) with ``PYTHONPATH=src:tests``:

    python3 experiments/a4_fused_discriminator.py
"""
from __future__ import annotations

import sys

import torch

# The suite registers the runtime's NVFP4 operator as a side effect of its
# first CUDA test; a standalone probe has to do it BEFORE the route's stubs
# replace sys.modules["vllm"], or the op the route executes never registers.
try:
    import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)
except Exception as _exc:  # noqa: BLE001 -- reported, not swallowed
    print(f"[bootstrap] real vllm._custom_ops import failed: "
          f"{type(_exc).__name__}: {_exc}")

if not callable(getattr(torch.ops._C, "scaled_fp4_quant", None)):
    print("[bootstrap] torch.ops._C.scaled_fp4_quant is NOT registered; the probe "
          "needs the real operator library in the image")
    sys.exit(2)
print("[bootstrap] torch.ops._C.scaled_fp4_quant registered")

import test_serving_nvfp4_route as T  # tests/ is on PYTHONPATH

from _pytest.monkeypatch import MonkeyPatch

ROLES = (("q_proj", 256), ("k_proj", 128), ("v_proj", 128))
SEED = 3
M, COLS = 32, 1024


def _levels():
    return torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                        dtype=torch.float32, device="cuda")


def _reference_codes(x, global_scale):
    """The reference helper's own decision, exposed as (codes, sf) so the
    element a stock op disagrees about can be inspected."""
    m, k = x.shape
    groups = k // T.GROUP
    xf = x.float().view(m, groups, T.GROUP)
    amax = xf.abs().amax(dim=2, keepdim=True).clamp_min(1e-12)
    sf = (amax / 6.0 * float(global_scale)).to(torch.float8_e4m3fn)
    sf_f = sf.float().clamp_min(1e-12)
    q = xf * float(global_scale) / sf_f
    levels = _levels()
    idx = (q.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1)
    sign = (q < 0).to(torch.uint8)
    codes = (idx.to(torch.uint8) | (sign * 8)).view(m, k)
    return codes, sf_f.repeat_interleave(T.GROUP, dim=2).view(m, k), q.view(m, k)


def main() -> None:
    torch.manual_seed(0)
    monkey = MonkeyPatch()
    got, want, layer, _method, (packed, scale, global_) = T._drive(
        monkey, T.MODE_RESIDENT, roles=ROLES, seed=SEED)

    err = (got.float() - want.float()).abs()
    hess = want.float().abs().max().item()
    print(f"[full] err={err.max().item():.6f} max|want|={hess:.6f} "
          f"ratio={err.max().item() / hess:.6f} bound=8e-3 (the test's bound)")

    # 1. per-role localization
    start = 0
    for name, rows in ROLES:
        block = err[:, start:start + rows]
        ref = want.float()[:, start:start + rows].abs().max().item()
        print(f"[role {name:7s}] rows[{start}:{start + rows}] "
              f"err={block.max().item():.6f} max|want|={ref:.6f} "
              f"ratio={block.max().item() / max(ref, 1e-9):.6f} "
              f"n_over_half_max={(block > 0.5 * block.max().item()).sum().item()}")
        start += rows

    # 2. A-side: reference decision vs the registered stock op
    gs = float(layer.trellis_input_global_scale.data.reshape(-1)[0])
    x = torch.randn(M, COLS, dtype=torch.bfloat16, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(SEED))
    ref_codes, ref_sf, q = _reference_codes(x, gs)
    gs_tensor = torch.tensor([gs], dtype=torch.float32, device="cuda")
    op_packed, _op_scale = torch.ops._C.scaled_fp4_quant(x, gs_tensor, True)
    op_codes = torch.empty_like(ref_codes)
    op_codes[:, 0::2] = op_packed & 0xF
    op_codes[:, 1::2] = (op_packed >> 4) & 0xF
    diff = ref_codes != op_codes
    n_diff = int(diff.sum())
    print(f"[a-side] differing codes: {n_diff} of {ref_codes.numel()} "
          f"({100.0 * n_diff / ref_codes.numel():.4f}%)")
    if n_diff:
        mag = q[diff].abs()
        levels = _levels()
        # nearest two levels for each differing magnitude
        upper = torch.searchsorted(levels, mag)
        upper = upper.clamp(1, len(levels) - 1)
        lo = levels[upper - 1]
        hi = levels[upper]
        mid = 0.5 * (lo + hi)
        print(f"[a-side] differing |q| vs level midpoints: "
              f"exact_midpoints={(mag == mid).sum().item()} of {n_diff} "
              f"max|q-mid|={(mag - mid).abs().max().item():.3e}")
        print(f"[a-side] ref levels on those: {ref_codes[diff].tolist()[:12]} "
              f"op levels: {op_codes[diff].tolist()[:12]}")

    # 3. rebuild the product from the STOCK op's codes
    levels = _levels()
    sign = (op_codes & 8).to(torch.bool)
    value_level = levels[(op_codes & 7).to(torch.int64)]
    stock_vals = torch.where(sign, -value_level, value_level) * ref_sf
    _blob, _scheme, _packed, _scale, _global, ref_w = T._encode_module(
        list(ROLES), cols=COLS, seed=SEED)
    stock_want = (stock_vals / gs) @ ref_w.t()
    ref_vals = T._reference_fp4_quant_value(x, gs)
    ref_want = (ref_vals / gs) @ ref_w.t()
    stock_err = stock_want.sub(got.float()).abs().max().item()
    ref_err = ref_want.sub(got.float()).abs().max().item()
    print(f"[rebuild] max|want_from_stock_codes - got| = {stock_err:.6f} "
          f"(ratio={stock_err / max(stock_want.abs().max().item(), 1e-9):.6f})")
    print(f"[rebuild] max|want_from_reference - got| = {ref_err:.6f} "
          f"(ratio={ref_err / max(ref_want.abs().max().item(), 1e-9):.6f})")
    print(f"[rebuild] max|want_from_stock - want_from_reference| = "
          f"{stock_want.sub(ref_want).abs().max().item():.6f}")


if __name__ == "__main__":
    main()
