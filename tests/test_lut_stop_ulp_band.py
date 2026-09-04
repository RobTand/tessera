"""Issue #106: what ``_fit_lut``'s eps stop test actually decides differently.

``2f6a15a`` replaced the swap accept test ``cost < base * (1 - 1e-9)`` with
``cost < base * (1 - torch.finfo(cost.dtype).eps)``.  On the float32 costs
``_lut_cost`` accumulates in production that threshold is ~119x looser, so it
can only reject swaps the literal took -- and issue #106 asks *which* ones,
because the table those swaps choose is on the wire.

The band is one or two float32 steps wide and never more.  ``base`` is a
float32 value widened to a Python float, so ``base * (1 - step)`` is exact in
float64 and the test accepts iff ``base - cost > base * step``:

* ``base * eps`` is one ulp of ``base``'s binade -- ``2^(e-23)`` for
  ``base in [2^e, 2^(e+1))`` -- so the eps test rejects the single smallest
  representable step, and at an exact power of two, where the steps *below*
  are half-width, the two smallest.
* ``base * 1e-9`` is far below one ulp (``eps / 2 = 5.96e-8`` is the smallest
  relative step float32 can express at all), so the literal accepted every
  improvement the dtype can represent, including those.

So the whole behavioural difference is: **trials that improve the running cost
by at most one ulp of its binade changed hands.**  Everything larger is taken
by both tests, unchanged.

These are the arithmetic facts the byte receipt in
``docs/measurements/tessera-lut-stop-eps-bytes-2026-09-04.md`` rests on, so
they are pinned here rather than restated in prose: the receipt's claim is that
the one decision that could move never fires on a real encode, and that claim
is empty unless "the one decision" is exactly this one.

``tests/test_lut_stop_dtype.py`` pins the *rule* (the threshold comes from the
dtype).  This pins the *consequence* (which trials change hands), which is what
a wire question needs.
"""

import math

import pytest
import torch

EPS32 = float(torch.finfo(torch.float32).eps)
OLD_LITERAL = 1e-9


def _f32(x: float) -> float:
    return float(torch.tensor(x, dtype=torch.float32))


def _ulps_below(base: float, n: int) -> float:
    """``base`` moved ``n`` float32 steps toward zero, widened like the code."""
    t = torch.tensor(base, dtype=torch.float32)
    neg_inf = torch.tensor(-math.inf, dtype=torch.float32)
    for _ in range(n):
        t = torch.nextafter(t, neg_inf)
    return float(t)


def _accepts(cost: float, base: float, step: float) -> bool:
    """``_fit_lut``'s accept test, spelled exactly as the swap loop spells it."""
    return cost < base * (1.0 - step)


# A binade's foot, just above it, its middle and its top, at three scales.
# The foot is the case the bound is tight at and is why the band is stated as
# "at most one ulp of the binade" rather than "one step".
BASES = [
    _f32(v) * s
    for s in (2.0 ** -20, 1.0, 2.0 ** 20)
    for v in (1.0, 1.0000001, 1.3, 1.7, 1.9999999)
]


@pytest.mark.parametrize("base", BASES)
def test_the_band_is_exactly_one_ulp_of_the_binade(base):
    """Improvements up to ``base * eps`` change hands; nothing above does.

    Walked step by step from ``base`` downward: every step whose improvement is
    at most ``base * eps`` is taken by the literal and refused by the eps test,
    and the first step past it is taken by both.  That is the difference set,
    enumerated rather than argued.
    """
    seen_band = 0
    for n in range(1, 6):
        cost = _ulps_below(base, n)
        improvement = base - cost
        old, new = _accepts(cost, base, OLD_LITERAL), _accepts(cost, base, EPS32)
        assert old, "1e-9 accepts every representable float32 improvement"
        if improvement <= base * EPS32:
            assert not new, (base, n, improvement)
            seen_band += 1
        else:
            assert new, (base, n, improvement)
    # One step wide inside a binade, two at its foot where steps are half-width.
    assert seen_band in (1, 2)
    assert (seen_band == 2) == (math.log2(base).is_integer())


@pytest.mark.parametrize("base", BASES)
def test_no_improvement_is_rejected_by_both(base):
    """An equal cost is not a descent step under either test."""
    assert not _accepts(base, base, OLD_LITERAL)
    assert not _accepts(base, base, EPS32)


def test_the_literal_sat_below_every_representable_float32_step():
    """Why the band is only ulps wide: float32 has nothing finer to offer.

    The smallest positive relative improvement a float32 cost can express is
    one step, i.e. at least ``eps / 2``.  ``1e-9`` is nearly two orders below
    that, so on this dtype the literal was never a threshold at all -- it
    accepted every improvement the arithmetic could represent, and the eps test
    is the first threshold in this code that refuses one.
    """
    assert OLD_LITERAL < EPS32 / 2.0
    for base in BASES:
        rel = (base - _ulps_below(base, 1)) / base
        assert rel >= EPS32 / 2.0 - 1e-18
