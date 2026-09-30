"""vLLM's per-token dynamic E4M3 quantiser, tested on its own.

``tessera.serving.native_ops.native_fp8_quant`` calls the pinned image's
``torch.ops._C.dynamic_per_token_scaled_fp8_quant``.  Every E4M3 oracle in this
tree feeds the SAME quantised activations to the kernel under test and to its
reference, so a defect in the quantiser is invisible there.  This module holds
the quantiser to two independent statements:

* the tests' own reference, ``test_serving_fp8_route._reference_fp8_quant``
  (duplicated in ``test_serving_fp8_gemv``): codes bitwise equal and the
  per-token scale within one fp32 ulp;
* the arithmetic of the pinned kernel's source, restated here from vLLM
  ``fd4a1512`` (``csrc/libtorch_stable/quantization/w8a8/fp8/common.cu``,
  ``dynamic_per_token_scaled_fp8_quant_kernel_strided``): ``scale =
  fmaxf(amax / 448, 1 / (448 * 512))`` (``min_scaling_factor``), then
  ``scaled_fp8_conversion<false>`` -- a true fp32 division ``x / scale``,
  clamped to +-448, converted round-to-nearest-even with saturation.  That
  translation unit is not built with ``--use_fast_math``, so the division is
  correctly rounded, and the restatement computes it in fp64 and rounds once
  to fp32 (exact: 53 >= 2 * 24 + 2).

Input classes: random bf16 rows of several shapes (hidden sizes that are and
are not multiples of the 16-byte vector width), rows with a tiny amax (below,
at and above the scale floor's ``amax = 2^-9``), rows with one large outlier,
exact powers of two, and values that land exactly on E4M3 rounding midpoints
after the division.

FINDING (measured on the pinned image, vLLM 0.28.1rc1.dev397+gfd4a15126,
2026-09-28).  The op IS the restated kernel arithmetic, bitwise, on every
class.  The tests' reference is NOT, in two ways, so the reference comparison
is xfail on the classes where they show (strict, each reason carries its
counts):

* reciprocal vs division: the reference's ``amax / FP8_MAX`` is a CUDA
  division by a Python scalar, which torch computes as ``amax * fl32(1/448)``;
  the kernel divides, ``fl32(amax / 448)``.  The two scales are 1 fp32 ulp
  apart in about half the rows, the reference's always the larger.  A
  quotient that is an exact E4M3 midpoint under the kernel's division (a tie;
  round-to-nearest-even takes the even code) lands just below the midpoint
  under the reference and rounds down, so every differing code is one step
  LARGER in magnitude on the native side.
* the scale floor: the kernel floors the scale at ``1 / (448 * 512)``, which
  binds when ``amax < 2^-9``; the reference only clamps ``amax`` at 1e-12.
  Below the floor the native scale is larger and every nonzero native code
  smaller in magnitude (up to 125 code steps); an all-zero row agrees on its
  codes and differs only in its scale.

All cases are CUDA; they run through PrismaBuild inside the pinned serving
image (``experiments/routed_fused_tests.sh``).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_serving_fp8_route import FP8_MAX, _reference_fp8_quant  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the quantiser is a CUDA op")

#: ``min_scaling_factor<c10::Float8_e4m3fn>::val()`` = ``1.0f / (448.0f * 512.0f)``.
MIN_SCALE = float(torch.tensor(1.0, dtype=torch.float32)
                  / (torch.tensor(FP8_MAX, dtype=torch.float32) * 512.0))
#: ``1.0f / 448.0f``: the reciprocal a multiply-by-reciprocal division uses.
INV_FP8_MAX = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(FP8_MAX, dtype=torch.float32))


def _e4m3_positive_values():
    """Every finite positive E4M3FN value, ascending (bytes 0x01 .. 0x7E)."""
    return torch.arange(1, 0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn).double()


def _bf16_exact(values):
    """``values`` (fp64) as bf16, refusing any value bf16 cannot hold exactly."""
    out = values.to(torch.bfloat16)
    assert torch.equal(out.double(), values), "a constructed value is not bf16-exact"
    return out


def _random(rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    sigma = torch.exp(torch.empty(rows, 1).uniform_(-3.0, 3.0, generator=g))
    return (torch.randn(rows, cols, generator=g) * sigma).to(torch.bfloat16)


def _tiny_amax(seed):
    """Rows whose amax straddles the scale floor: ``fmaxf(amax / 448,
    1 / (448 * 512))`` binds exactly when ``amax < 2^-9``."""
    amaxes = [0.0, 2.0 ** -40, 1e-6, 1e-4, 1e-3, 1.9e-3, 2.0 ** -9, 2.5e-3, 1e-2]
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i, amax in enumerate(amaxes):
        row = torch.rand(4096, generator=g) * 2.0 - 1.0          # (-1, 1)
        row = (row * amax).to(torch.bfloat16)
        row[(97 * i) % 4096] = amax * (-1.0 if i % 2 else 1.0)   # the row's amax
        rows.append(row)
    return torch.stack(rows)


def _outliers(seed):
    """One large element in a row of small ones, at the head, the middle and
    the unaligned tail of the row."""
    g = torch.Generator().manual_seed(seed)
    spikes = [(0, 1000.0), (2047, -3.0e4), (4095, 448.0), (1, 7.5), (4094, -1.0e2), (3000, 65504.0)]
    x = (torch.randn(len(spikes), 4096, generator=g) * 0.01)
    for i, (at, value) in enumerate(spikes):
        x[i, at] = value
    return x.to(torch.bfloat16)


def _powers_of_two():
    """Every element a signed power of two.  Rows whose amax is ``448 * 2^j``
    have an exact scale ``2^j``; rows whose amax is ``2^k`` do not."""
    rows = []
    exps = torch.arange(-24, 16, dtype=torch.float64)
    base = torch.cat([torch.exp2(exps), -torch.exp2(exps)])             # 80 values
    for top in [448.0 * 2.0 ** j for j in (-8, -3, 0, 2, 5)] + [2.0 ** k for k in (-5, 0, 3, 9, 15)]:
        row = base.clone()
        row = row[row.abs() <= top]
        row = torch.cat([row, torch.tensor([top]),
                         torch.zeros(96 - 1 - row.numel(), dtype=torch.float64)])
        rows.append(row)
    return _bf16_exact(torch.stack(rows))


#: Row amax ``448 * c``: an exact scale ``c`` under a true division; ``c``
#: carries at most three significant bits so every ``midpoint * c`` below is
#: bf16-exact.
MIDPOINT_SCALES = (1.0, 0.75, 1.25, 1.5, 1.75, 0.875, 3.0, 0.375)


def _midpoints():
    """Values whose quotient by a true-division scale IS an E4M3 rounding
    midpoint (a tie: round-to-nearest-even decides), plus every representable
    E4M3 value, both signs, scaled by ``c`` per row; the row's amax is
    ``448 c``.  503 columns: not a multiple of the vector width."""
    grid = _e4m3_positive_values()
    mids = (grid[:-1] + grid[1:]) / 2.0
    first = grid[0] / 2.0                                   # the tie between 0 and the least subnormal
    pos = torch.cat([torch.tensor([first]), mids, grid])
    body = torch.cat([pos, -pos])
    rows = []
    for c in MIDPOINT_SCALES:
        row = torch.cat([body * c, torch.tensor([0.0])])
        assert float(row.abs().max()) == FP8_MAX * c
        rows.append(row)
    return _bf16_exact(torch.stack(rows))


CLASSES = {
    "random-1x4096": lambda: _random(1, 4096, 11),
    "random-7x6144": lambda: _random(7, 6144, 12),
    "random-64x3072": lambda: _random(64, 3072, 13),
    "random-333x2048": lambda: _random(333, 2048, 14),
    "random-5x1000": lambda: _random(5, 1000, 15),
    "random-3x129": lambda: _random(3, 129, 16),
    "random-2048x4096": lambda: _random(2048, 4096, 17),
    "tiny-amax": lambda: _tiny_amax(21),
    "outliers": lambda: _outliers(22),
    "powers-of-two": _powers_of_two,
    "e4m3-midpoints": _midpoints,
}

_RECIPROCAL = ("the reference scale is amax * fl32(1/448) (torch's CUDA division by a Python "
               "scalar), the kernel's is fl32(amax / 448): 1 fp32 ulp apart, the reference's "
               "the larger, so exact-midpoint quotients round down on the reference side and "
               "every differing native code is one step larger in magnitude")
_FLOOR = ("the kernel floors the scale at 1/(448*512) (min_scaling_factor, binding at "
          "amax < 2^-9), the reference does not: below it the native scale is larger and the "
          "native codes smaller in magnitude")

#: Measured on the pinned image (the first GPU row of this module): codes that
#: differ from the reference / all codes, and rows whose scales differ / rows.
REFERENCE_DISAGREES = {
    "random-7x6144": f"215/43008 codes, 6/7 scales 1 ulp apart; {_RECIPROCAL}",
    "random-64x3072": f"173/196608 codes, 39/64 scales 1 ulp apart; {_RECIPROCAL}",
    "random-333x2048": f"839/681984 codes, 175/333 scales 1 ulp apart; {_RECIPROCAL}",
    "random-3x129": f"3/387 codes, 3/3 scales 1 ulp apart; {_RECIPROCAL}",
    "random-2048x4096": f"7073/8388608 codes, 1127/2048 scales 1 ulp apart; {_RECIPROCAL}",
    "outliers": f"4/24576 codes, 3/6 scales 1 ulp apart; {_RECIPROCAL}",
    "e4m3-midpoints": f"756/4040 codes, 6/8 scales 1 ulp apart; {_RECIPROCAL}",
    "tiny-amax": (f"17730/36864 codes in the 5 nonzero rows under the floor (up to 125 code "
                  f"steps), 6/9 scales apart by up to 2.6e8 ulps (the zero row: scale only); "
                  f"{_FLOOR}"),
}


def _against_the_reference(name):
    reason = REFERENCE_DISAGREES.get(name)
    if reason is None:
        return name
    return pytest.param(name, marks=pytest.mark.xfail(
        strict=True, reason=f"_reference_fp8_quant is not vLLM's arithmetic: {reason}"))


def _native(x):
    from tessera.serving.native_ops import native_fp8_quant, require_native_fp8_quant

    require_native_fp8_quant("native fp8 quantiser test")
    q, s = native_fp8_quant(x.contiguous())
    torch.cuda.synchronize()
    return q, s.reshape(-1, 1)


def _codes(q):
    return q.view(torch.uint8)


def _ulps_apart(a, b):
    """fp32 ulps between two tensors of non-negative finite floats (their bit
    patterns order like the values)."""
    return (a.float().contiguous().view(torch.int32).long()
            - b.float().contiguous().view(torch.int32).long()).abs()


def _kernel_arithmetic(x):
    """The pinned kernel's arithmetic, restated (see the module docstring)."""
    xf = x.float()
    amax = xf.abs().amax(dim=1, keepdim=True)
    scale = (amax.double() / FP8_MAX).float().clamp_min(MIN_SCALE)
    q = (xf.double() / scale.double()).float().clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


def _diagnose(x, q_n, s_n, q_r, s_r):
    """Per-row facts of a native/reference disagreement: the scales, their fp32
    ulp distance, which arithmetic each scale is (a true fp32 division
    ``fl(amax / 448)``, a reciprocal multiply ``fl(amax * fl(1/448))``, the
    ``1/(448*512)`` floor), and the differing codes with their direction."""
    xf = x.float()
    amax = xf.abs().amax(dim=1, keepdim=True)
    true_div = (amax.double() / FP8_MAX).float()
    recip = (amax.clamp_min(1e-12).double() * INV_FP8_MAX).float()
    cn, cr = _codes(q_n), _codes(q_r)
    differ = cn != cr
    vn, vr = q_n.float().abs(), q_r.float().abs()
    rows = []
    for i in range(x.shape[0]):
        n_diff = int(differ[i].sum())
        ulps = int(_ulps_apart(s_n[i], s_r[i]).item())
        if n_diff == 0 and ulps == 0:
            continue
        rows.append({
            "row": i, "amax": float(amax[i]), "native_scale": float(s_n[i]),
            "reference_scale": float(s_r[i]), "scale_ulps": ulps,
            "native_is_true_div": bool(s_n[i] == true_div[i]),
            "native_is_floor": bool(float(s_n[i]) == MIN_SCALE),
            "reference_is_recip_mul": bool(s_r[i] == recip[i]),
            "reference_is_true_div": bool(s_r[i] == true_div[i]),
            "codes_differ": n_diff,
            "native_larger_magnitude": int((differ[i] & (vn[i] > vr[i])).sum()),
            "native_smaller_magnitude": int((differ[i] & (vn[i] < vr[i])).sum()),
            "max_code_step": int((cn[i].int() - cr[i].int()).abs().max()),
        })
    return rows


@cuda
@pytest.mark.parametrize("name", [_against_the_reference(n) for n in CLASSES])
def test_native_fp8_quant_matches_the_tests_reference(name):
    """Codes bitwise equal; the per-token scale within one fp32 ulp."""
    x = CLASSES[name]().cuda()
    q_n, s_n = _native(x)
    q_r, s_r = _reference_fp8_quant(x)
    s_r = s_r.reshape(-1, 1)
    assert q_n.shape == q_r.shape and q_n.dtype == q_r.dtype == torch.float8_e4m3fn
    ulps = _ulps_apart(s_n, s_r)
    codes_differ = int((_codes(q_n) != _codes(q_r)).sum())
    rows = _diagnose(x, q_n, s_n, q_r, s_r)
    print(f"FP8-QUANT {name} shape={tuple(x.shape)} codes_differ={codes_differ}/{x.numel()} "
          f"max_scale_ulps={int(ulps.max())} rows_with_scale_diff={int((ulps > 0).sum())}/{x.shape[0]}")
    for row in rows[:40]:
        print(f"FP8-QUANT {name} {row}")
    assert codes_differ == 0 and int(ulps.max()) <= 1, (
        f"{name}: {codes_differ} codes differ, max scale distance {int(ulps.max())} fp32 ulps; "
        f"rows: {rows[:8]}")


@cuda
@pytest.mark.parametrize("name", list(CLASSES))
def test_native_fp8_quant_is_the_pinned_kernels_arithmetic(name):
    """Bitwise: the codes and the scales the op returns are the restated
    source arithmetic's (true division, the ``1/(448*512)`` floor, RNE)."""
    x = CLASSES[name]().cuda()
    q_n, s_n = _native(x)
    q_k, s_k = _kernel_arithmetic(x)
    codes_differ = int((_codes(q_n) != _codes(q_k)).sum())
    scales_differ = int((s_n != s_k).sum())
    print(f"FP8-QUANT-SOURCE {name} shape={tuple(x.shape)} codes_differ={codes_differ} "
          f"scales_differ={scales_differ}")
    assert codes_differ == 0 and scales_differ == 0, (
        f"{name}: {codes_differ} codes and {scales_differ} scales differ from the restated kernel")
