"""Every rung of every family is realisable -- the property allocation rests on.

PrismaQuant's allocator wants to place a per-Linear byte budget anywhere on the
rate axis, not on a handful of rungs.  That is only honest if a requested rung
can actually be *encoded*, so this file tests realisability rather than quality:
given a family's cap and a root rate, does a legal per-column schedule exist
whose superblock quota is exact?

The answer is yes everywhere, and the reason is structural rather than lucky.
A root is realisable over ``n`` columns when ``(root - floor(root)) * n`` is an
integer.  Rates are quoted in q256 -- 256ths of a bit -- and the superblock is
256 columns, so that product is ``root_q256 mod 256``, an integer by
construction.  **The q256 grid is exactly the realisable set at superblock
scale.**  Picking 256 for both was the design decision that makes continuous
allocation exact instead of approximate.
"""
from fractions import Fraction
from math import log2

import pytest

from tessera.errors import GrammarError
from tessera.grammar import (
    C_FULL_BITS,
    Q256_UNIT,
    bresenham_rate_schedule,
    bits_per_position,
    root_from_q256,
    superblock_quota_ok,
    validate_rate_schedule,
)

SUPERBLOCK = 256

# (base grid size, arity).  Cap is arity*log2(size) - 1.
GRID_SPACE = [(8, 1), (8, 2), (16, 1), (16, 2), (32, 1), (32, 2), (64, 1), (256, 1)]


def _cap(base_size: int, arity: int) -> int:
    return int(log2(base_size)) * arity - 1


@pytest.mark.parametrize("base_size,arity", GRID_SPACE)
def test_every_rung_of_every_family_is_realisable(base_size, arity):
    cap = _cap(base_size, arity)
    lo = -(-Q256_UNIT // arity)
    hi = cap * Q256_UNIT // arity
    for per_position_q256 in range(lo, hi + 1):
        root = root_from_q256(per_position_q256 * arity)
        rates = bresenham_rate_schedule(root, SUPERBLOCK, cap)
        assert superblock_quota_ok(rates, SUPERBLOCK, root)
        validate_rate_schedule(rates, root, cap)
        assert min(rates) >= 1 and max(rates) <= cap


@pytest.mark.parametrize("base_size,arity", GRID_SPACE)
def test_the_schedule_mixes_only_the_two_bracketing_rates(base_size, arity):
    """A quota met by mixing distant rates would not be the canonical one."""
    cap = _cap(base_size, arity)
    for code_q256 in (Q256_UNIT + 1, (cap * Q256_UNIT) // 2, cap * Q256_UNIT - 1):
        root = root_from_q256(code_q256)
        rates = bresenham_rate_schedule(root, SUPERBLOCK, cap)
        assert set(rates) <= {int(root), int(root) + 1}


def test_a_rate_above_the_cap_is_still_refused():
    """Threading the cap must not turn the guard off."""
    with pytest.raises(GrammarError, match="outside the shaped domain"):
        bresenham_rate_schedule(Fraction(4), SUPERBLOCK, cap=3)
    with pytest.raises(GrammarError, match="outside the shaped domain"):
        bresenham_rate_schedule(Fraction(8), SUPERBLOCK, cap=7)


def test_the_cap_default_still_reproduces_every_pre_families_artifact():
    """Omitting `cap` must behave exactly as it did before it existed."""
    for code_q256 in range(Q256_UNIT, C_FULL_BITS * Q256_UNIT + 1, 17):
        root = root_from_q256(code_q256)
        assert bresenham_rate_schedule(root, SUPERBLOCK) == bresenham_rate_schedule(
            root, SUPERBLOCK, C_FULL_BITS
        )
    for rate in (1, 2, 3):
        assert bits_per_position(rate, 0) == bits_per_position(rate, 0, cap=C_FULL_BITS)


def test_high_rates_cost_what_they_should_per_position():
    """E4M3 reaches rate 7; the body cost is the rate, plus completion bits."""
    assert bits_per_position(7, 0, cap=7) == 7
    assert bits_per_position(4, 0, cap=7) == 4
    with pytest.raises(GrammarError, match="outside the shaped domain"):
        bits_per_position(4, 0)  # cap 3: rate 4 is not a TESSERA-4 rate
