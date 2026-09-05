"""The byte-matched uniform control for a rate-axis plan.

A Tessera family is a rate *axis*, so an allocation over it is a claim: that
choosing a rung per Linear beats spending the same bytes everywhere.  Nothing
else in the pipeline tests that claim.  The byte check asks "did we build what
we said", the census asks "does it route", the twin asks "do two renderings
agree" -- all three passed on 2026-09-02 while the allocation they certified
served **2.00x worse** than the uniform arm at the same bytes
(``docs/measurements/tessera-allocated-served-2026-09-02.md``).  The control is
the only check that asks whether the allocation was worth making.

This module is that control, promoted out of the receipt's drivers:

* :func:`unit_wire_bits` prices one planned unit through
  :func:`tessera.calculator.terminal_rate`, the wire's own accountant;
* :func:`rate_menu` prices *every* rung at one shape and says which of them a
  higher rung already matches or beats on bytes -- the rungs an allocator must
  not be offered, and the reason (issue #43);
* :func:`uniform_control` searches the family's rungs for the one whose whole
  plan weighs what the candidate weighs;
* :func:`assert_byte_matched` refuses a pair that does not, on integer bit
  totals, so a post-export check can hand it two manifests;
* :func:`control_block` renders the pair -- and, when the two KLs are known,
  the verdict -- as the JSON an artifact carries beside its bpp.
* :func:`selection_requirement` says whether a plan embodies a rung
  selection at all, and stamps the menu's requirement when it does:
  validated-surrogate selection (``docs/ARCHITECTURE.md`` §4.10, tessera#2).
* :func:`assert_plane_promotion` refuses a per-plane promotion its evidence
  does not carry -- a geomean without per-unit wins, a served number for an
  arm other than the one promoted, a screen taken off the wire -- under the
  receipt's GLM gate, which it restates without moving (tessera#65, #85).
* :func:`landing_ordering` puts the on-wire arm ordering beside the
  landing-disabled one and says, as a value, whether they agree (tessera#85).

**What is held fixed.**  The control varies *only the rung*.  A unit the
candidate plans as BF16 stays BF16 in the control, and the control uses the
candidate's own grid.  The null hypothesis is "spend the same bytes at one
rate", not "quantize a different set of Linears": which units to quantize is
the format question the format menu already answers, and folding it in here
would make a lost comparison unattributable.

**Matching is on bytes, never on a label.**  ``artifact_bpp`` labels are
rounded, and a rung's rate depends on the *shape* (a CHANNEL plane amortises
one fp16 over the row; a window table amortises ``2^L`` bytes over the unit),
so the search prices every unit at its own shape and compares integer bit
totals.  The rung quantum is 1/256 body bits per code -- 0.0039 bpp -- so a
nearest-rung control usually lands within half of that, 0.05% at 4 bpp against
the 0.1% :func:`assert_byte_matched` allows.  The receipt's control sits at
65 ppm.  The axis is not uniformly dense, though: E2M1x2 jumps **0.241 bpp**
between R895 and R896 where the recipe changes from the window body to the
coset trellis, and near that hole no control this tight exists.  The assertion
fires there rather than quietly comparing two different byte budgets.

The axis is not *monotone* either, and on a small unit not by a little: the
window table below the cap is a fixed 4096 bytes, so R896 can weigh less than
R895 while decoding better.  :func:`rate_menu` is where that is measured and
screened; ``uniform_control`` is immune by construction, since it ranks by
bits and never by rung, and issue #43 is why that is now stated rather than
incidental.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .alphabet import BF16_GRID, E2M1_GRID, E4M3_GRID, PayloadGrid, tuple_grid
from .calculator import terminal_rate
from .encode import LUT_LANDING_MODES, LUT_LANDING_WIRE
from .errors import (
    ControlNotByteMatchedError,
    GrammarError,
    PromotionRefusedError,
    TesseraError,
)
from .export import rung_ceiling, wire_recipe
from .manifest import BodyKind, ScalePlaneKind

__all__ = [
    "BF16",
    "CONTROL_SCHEMA",
    "GRID_NAMES",
    "DEFAULT_MAX_RELATIVE_SLACK",
    "REQUIRED_SELECTION_MODE",
    "SELECTION_SCHEMA",
    "PROMOTION_SCHEMA",
    "GLM_GATE",
    "LANDING_SCHEMA",
    "ByteMatch",
    "LandingOrdering",
    "PlannedUnit",
    "PlanePromotion",
    "RateMenu",
    "RungPrice",
    "UniformControl",
    "assert_byte_matched",
    "assert_plane_promotion",
    "bits_from_manifest",
    "control_block",
    "grid_for_name",
    "landing_ordering",
    "plan_wire_bits",
    "promotion_block",
    "rate_menu",
    "require_kl",
    "selection_requirement",
    "uniform_control",
    "unit_wire_bits",
    "units_from_plan",
]

#: The exporter's spelling for a passthrough module, in a ``--plan-json``.
BF16 = "BF16"

CONTROL_SCHEMA = "tessera.uniform_control.v1"

#: Half a rung step at 4 bpp is ~0.05%; this is the next round number above it,
#: so a nearest-rung control satisfies it by construction and the assertion
#: fires only on a forced rung, a mixed multiset the axis cannot reach, or a
#: manifest that disagrees with the plan.  The 2026-09-02 control sat at
#: 65 ppm.  Issue #3: "Two arms at 4.0 bpp that differ 1% in bytes are not a
#: control."
DEFAULT_MAX_RELATIVE_SLACK = Fraction(1, 1000)

#: How the search picks a rung.  ``nearest`` minimises |candidate - control|
#: and is the default, because the direction that flatters an arm depends on
#: which arm wins and only a small |slack| is neutral to both.  ``no_larger``
#: takes the heaviest rung that does not outweigh the candidate, for a claim
#: that must be conservative against the allocation whatever the outcome.
MATCH_RULES = ("nearest", "no_larger")

_GRID_BY_NAME = {
    "E2M1": E2M1_GRID,
    "E4M3": E4M3_GRID,
    "BF16": BF16_GRID,
}

#: The exporter's ``--grid`` vocabulary, as one tuple rather than as a sentence
#: in a refusal message.  A test enumerates it to cross every rung the wire can
#: emit against the rungs the serving plugin publishes a decode for (#41), so a
#: grid added here without a served range is a failing test rather than a
#: checkpoint that refuses at load.
GRID_NAMES = ("E2M1", "E2M1x2", "E4M3", "BF16")

_BITS_CACHE: "dict[tuple, Fraction]" = {}


# ------------------------------------------------------------- the domains
#
# Every number a verdict in this module reads has a domain its own definition
# gives it, and a gate that converts a number and then compares it has already
# lost the argument: NaN is False against every ordered comparison, an infinity
# satisfies any ``<=`` bar it is handed, a zero denominator turns a ratio into
# whatever the fallback branch says, and ``int()`` truncates a fractional count
# into a denominator nobody has.  So conversion and domain are one step here,
# and a value outside its domain is refused **by field name** before anything
# is compared (AGENTS.md rule 5; tessera#224, tessera#225).
#
# The domains are read off the quantities, never off round numbers (rule 2): a
# wire bit total is counted, a parameter count is counted, a byte-match
# tolerance is a fraction of the candidate's own bits, an error ratio is a
# quotient of two positive errors, and a KL divergence is non-negative.


def _bit_total(value, *, field: str, where: str) -> Fraction:
    """A wire bit total: a whole number of bits, and more than none of them.

    Bits are *counted* -- :func:`unit_wire_bits` prices integral at every rung
    of every grid on every shape ``tests/test_uniform_control.py`` sweeps, and
    :func:`bits_from_manifest` reads whole bytes -- so a fractional total is an
    accounting defect rather than a rounding, and an arm of zero or negative
    bits is not an arm.  Zero is the one that mattered: it made
    :attr:`ByteMatch.relative_slack` report a perfect match for a candidate
    nobody weighed (tessera#225).
    """
    reason = (
        f"{where}: {field} is a whole positive number of wire bits, got "
        f"{value!r} -- the byte match is a ratio over the candidate's own "
        "bits, so a zero, negative, fractional or non-numeric total is "
        "refused here rather than compared"
    )
    try:
        bits = Fraction(value)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise TesseraError(reason) from exc
    if bits.denominator != 1 or bits <= 0:
        raise TesseraError(reason)
    return bits


def _exact_count(value, *, field: str, where: str) -> int:
    """A count of things: an exact positive integer, never a truncation."""
    reason = (
        f"{where}: {field} is an exact positive count, got {value!r} -- "
        "int() would truncate a fractional count silently and report a bpp "
        "over a denominator nothing has"
    )
    try:
        count = Fraction(value)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise TesseraError(reason) from exc
    if count.denominator != 1 or count <= 0:
        raise TesseraError(reason)
    return int(count)


def _tolerance(value, *, field: str, where: str) -> Fraction:
    """A byte-match tolerance: a fraction of the candidate's bits in [0, 1).

    Zero is legal and means "exact".  One is not: at a relative slack of 1 the
    control weighs twice the candidate's bytes or none of them, which is the
    comparison this module exists to refuse, so a tolerance that would admit
    it is refused where it enters instead of at every arm it later passes.
    """
    reason = (
        f"{where}: {field} is a fraction of the candidate's bits in [0, 1), "
        f"got {value!r} -- at 1 the control may weigh twice the candidate or "
        "none of it, which is not a control"
    )
    try:
        slack = Fraction(value)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise TesseraError(reason) from exc
    if not 0 <= slack < 1:
        raise TesseraError(reason)
    return slack


def _error_ratio(value, *, field: str, where: str, error=TesseraError) -> float:
    """A quotient of two positive errors: finite and strictly positive."""
    reason = (
        f"{where}: {field} is a positive finite ratio of two errors, got "
        f"{value!r} -- NaN loses every ordered comparison silently and an "
        "infinity clears any bar written as one"
    )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise error(reason) from exc
    if not math.isfinite(number) or number <= 0:
        raise error(reason)
    return number


def require_kl(value, *, field: str, where: str, error=TesseraError) -> float:
    """A KL divergence: finite and non-negative, by its own definition.

    Public, because ``experiments/uniform_control.py verify`` builds its
    verdict from two KLs off the command line rather than through
    :func:`control_block`, and one rule has one home (AGENTS.md rule 4).

    Zero is admissible and means the two distributions agree; it is not a
    number any served arm has produced, and it is not this gate's business to
    say it cannot be.  Negative and non-finite are, because a divergence that
    is neither orders below every bar it is compared against and reads as a
    pass (tessera#224, tessera#225).
    """
    reason = (
        f"{where}: {field} is a finite non-negative KL divergence, got "
        f"{value!r} -- a negative or unmeasurable divergence sorts below "
        "every bar it is compared to and reads as a pass"
    )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise error(reason) from exc
    if not math.isfinite(number) or number < 0:
        raise error(reason)
    return number


def _unit_ratios(values, *, where: str) -> "tuple[float, ...]":
    """The per-unit error ratios a screen is made of: at least one, each valid."""
    ratios = tuple(values)
    if not ratios:
        raise TesseraError(
            f"{where}: no per-unit ratios -- a geomean presented without its "
            "units is exactly what this gate exists to refuse"
        )
    return tuple(_error_ratio(r, field="unit_ratios", where=where) for r in ratios)


def _unit_geomean(ratios: "Sequence[float]") -> float:
    """The geometric mean of a unit set, in logs so a long set does not drift."""
    return math.exp(math.fsum(math.log(r) for r in ratios) / len(ratios))


def grid_for_name(name: str) -> PayloadGrid:
    """``"E2M1x2" -> tuple_grid(E2M1_GRID, 2)``, the exporter's ``--grid`` vocabulary.

    The same four names ``experiments/export_tessera_serving.py`` accepts, so a
    plan written for the exporter prices here without translation.
    """
    text = str(name)
    grid = _GRID_BY_NAME.get(text)
    if grid is not None:
        return grid
    if text.startswith("E2M1x"):
        suffix = text[len("E2M1x"):]
        if suffix.isdigit() and int(suffix) >= 1:
            return tuple_grid(E2M1_GRID, int(suffix))
    raise GrammarError(
        f"unknown grid {name!r}; one of {', '.join(GRID_NAMES)} "
        "(E2M1/E2M1x2 the NVFP4 route, E4M3 the FP8 route, BF16 the 16-bit route)"
    )


def unit_wire_bits(grid: "str | PayloadGrid", q256: int, rows: int, columns: int) -> Fraction:
    """Exact plane-region bits one Tessera unit costs on the wire.

    The recipe comes from :func:`tessera.export.wire_recipe` -- the same
    function the exporter writes with -- and the count from
    :func:`tessera.calculator.terminal_rate`, which is the accountant the
    parser and the serializer both defer to.  Nothing here is a rate times a
    parameter count: a CHANNEL plane and a window table are charged per *unit*,
    so the shape is an argument and not a convenience.

    Header and manifest side bytes are outside this figure, exactly as they are
    outside ``TerminalRecord.exact_bpp``; both arms of a control carry the same
    module count, so they cancel, and ``check_wire_against_plan.py`` compares
    the same quantity against the manifest's ``wire_bytes``.
    """
    payload = grid if isinstance(grid, PayloadGrid) else grid_for_name(grid)
    rows, columns, q256 = int(rows), int(columns), int(q256)
    key = (payload.name, payload.arity, q256, rows, columns)
    hit = _BITS_CACHE.get(key)
    if hit is not None:
        return hit
    recipe = wire_recipe(payload, q256)
    plane = ScalePlaneKind(recipe.scale_plane)
    body = BodyKind(recipe.body)
    # The window body may spend the grid's whole payload width at a position;
    # the coset trellis spends one bit of it on the code.  The same dispatch
    # ``export.plan_for`` makes when it builds the schedule the encoder runs.
    cap = payload.payload_bits if body is BodyKind.WINDOW else payload.rate_cap
    rate = terminal_rate(
        q256 * payload.arity,
        rows,
        columns,
        with_scale_base=plane is ScalePlaneKind.S6B,
        with_scale_refine=plane in (ScalePlaneKind.S6B, ScalePlaneKind.LUT),
        with_row_scale=plane is ScalePlaneKind.CHANNEL,
        with_diagonals=False,
        completion=0,
        cap=cap,
        arity=payload.arity,
        span=recipe.span,
        window_bits=recipe.window_bits,
        code_bytes=payload.code_bytes,
        # A TCQ body's forest is on the wire, so it is in this figure.  It was
        # not until 2026-09-02: the accountant priced the position planes only
        # and every TCQ unit came out 512 B (E2M1x2 at the coset cap) or
        # 20-44 B (arity-1 E2M1) light, while every window unit was exact.  A
        # control that matches a window arm against a TCQ arm on those numbers
        # is matching two different quantities, which is the one thing it
        # exists not to do.  Measured against ``encode_linear`` in
        # ``tests/test_wire_bits_match_exported_bytes``.
        with_forest=body is BodyKind.TCQ,
    )
    bits = rate * rows * columns
    _BITS_CACHE[key] = bits
    return bits


@dataclass(frozen=True)
class RungPrice:
    """One rung of one unit: its exact wire bits, and what already beats it.

    ``dominated_by`` names the *higher* rung that costs no more bytes -- the
    cheapest such rung, and the highest of those when several tie, which is
    the best alternative on both axes.  ``None`` means nothing above this rung
    matches it, so it is on the frontier and may be offered.
    """

    q256: int
    bits: Fraction
    dominated_by: "int | None" = None

    @property
    def is_offered(self) -> bool:
        return self.dominated_by is None


@dataclass(frozen=True)
class RateMenu:
    """Every rung one Tessera unit admits at its own shape, priced and screened.

    A rung is **dominated** when a higher rung of the same unit costs no more
    bytes.  Both legs of that are measured, not assumed:

    * *no more bytes* is exact integer arithmetic through
      :func:`unit_wire_bits`, which agrees with ``encode_linear`` byte for byte
      (``tests/test_rate_menu.py``);
    * *no worse* would be an inference across a recipe change, so it was
      measured.  At (64, 512) on E2M1x2 the cap rung R896 weighs exactly what
      R736 weighs and 2544 B less than R895, and its relative SSE on a
      Gaussian unit is 0.00877 against R736's 0.02353 and R895's 0.01152 --
      better on both axes than everything it dominates
      (``experiments/tessera_dominated_rungs.py --quality``, weight space, one
      unit).  Error fell monotonically with the rung on the five sub-cap rungs
      measured (736, 800, 860, 894, 895), which is evidence for -- not proof of
      -- the ordering holding between them.

    So :attr:`offered` is what a menu builder should expose and
    :attr:`dominated` is what it should not -- and the pruning is *recorded*
    rather than silent, because a rung disappearing from a menu with no reason
    attached is how the next reader files issue #43 again.

    Measured shape dependence, since the whole effect is a fixed per-unit table
    amortised over the unit: on E2M1x2 the dominated count is 87 of 385 legal
    rungs at 96x320, 160/769 at 64x512, 35/769 at 96x768, and **0** at
    512x2048 and 1024x3072.  Production-shaped units have nothing to prune;
    small ones have a third of the axis to prune.
    """

    grid: str
    rows: int
    columns: int
    prices: "tuple[RungPrice, ...]"

    @property
    def params(self) -> int:
        return int(self.rows) * int(self.columns)

    @property
    def offered(self) -> "tuple[RungPrice, ...]":
        """The frontier: strictly increasing in bits as the rung rises."""
        return tuple(price for price in self.prices if price.is_offered)

    @property
    def dominated(self) -> "tuple[RungPrice, ...]":
        return tuple(price for price in self.prices if not price.is_offered)

    def price(self, q256: int) -> RungPrice:
        """This rung's row.  A rung the grammar refuses here is not in the menu."""
        for price in self.prices:
            if price.q256 == int(q256):
                return price
        raise GrammarError(
            f"R{q256} is not a legal rung of {self.grid} at "
            f"{self.rows}x{self.columns}"
        )

    def bpp(self, q256: int) -> Fraction:
        return Fraction(self.price(q256).bits, self.params)

    def to_json(self) -> dict:
        return {
            "grid": self.grid,
            "shape": [int(self.rows), int(self.columns)],
            "legal_rungs": len(self.prices),
            "offered": [price.q256 for price in self.offered],
            "dominated": {
                str(price.q256): price.dominated_by for price in self.dominated
            },
            "reason": (
                "a rung a higher rung matches or beats on bytes is worse on "
                "both axes and is not offered (tessera#43, measured in "
                "experiments/tessera_dominated_rungs.py)"
            ),
        }


def rate_menu(
    grid: "str | PayloadGrid",
    rows: int,
    columns: int,
    *,
    rungs: "Iterable[int] | None" = None,
) -> RateMenu:
    """The rungs of ``grid`` an allocator may be offered for a ``rows x columns`` unit.

    Prices every rung the grammar admits -- a rung it refuses at this width is
    skipped, never approximated -- and then sweeps from the top down, keeping a
    rung only when it is strictly cheaper than everything above it.

    The shape is an argument and not a convenience: the axis is non-monotone
    only because a *per-unit* term (a 4096-byte window table below the E2M1x2
    coset cap, one forest per distinct rate on arity-1 E2M1) is a large share
    of a small unit and rounding error on a large one.  A menu pruned at one
    shape and reused at another is wrong in both directions.
    """
    payload = grid if isinstance(grid, PayloadGrid) else grid_for_name(grid)
    rows, columns = int(rows), int(columns)
    ceiling = int(rung_ceiling(payload))
    candidates = [int(q) for q in rungs] if rungs is not None else range(1, ceiling + 1)

    priced: "list[tuple[int, Fraction]]" = []
    for q in candidates:
        try:
            priced.append((q, unit_wire_bits(payload, q, rows, columns)))
        except (GrammarError, ValueError, ZeroDivisionError):
            continue
    if not priced:
        raise GrammarError(
            f"no rung of {payload.name} prices at {rows}x{columns}; the "
            "grammar admits none of the rungs searched"
        )
    priced.sort()

    out: "list[RungPrice]" = []
    best_q: "int | None" = None
    best_bits: "Fraction | None" = None
    for q, bits in reversed(priced):
        out.append(RungPrice(q, bits, None if best_bits is None or bits < best_bits
                             else best_q))
        if best_bits is None or bits < best_bits:
            best_q, best_bits = q, bits
    return RateMenu(payload.name, rows, columns, tuple(reversed(out)))


@dataclass(frozen=True)
class PlannedUnit:
    """One body Linear of a ``--plan-json``: a Tessera rung, or a passthrough.

    ``q256 is None`` spells BF16, which is what the exporter's plan spells with
    the string ``"BF16"``.  A BF16 unit is carried into the control untouched
    and contributes the same bits to both arms.
    """

    tensor: str
    grid: str
    q256: "int | None"
    rows: int
    columns: int

    @property
    def params(self) -> int:
        return int(self.rows) * int(self.columns)

    @property
    def is_tessera(self) -> bool:
        return self.q256 is not None

    @property
    def shape(self) -> "tuple[int, int]":
        return (int(self.rows), int(self.columns))

    @property
    def wire_bits(self) -> Fraction:
        """This unit's wire bits: the accountant's figure, or 16 per parameter."""
        if not self.is_tessera:
            return Fraction(16 * self.params)
        return unit_wire_bits(self.grid, int(self.q256), self.rows, self.columns)

    def at_rung(self, grid: str, q256: int) -> "PlannedUnit":
        """The same tensor at another (grid, rung); a BF16 unit is unchanged."""
        if not self.is_tessera:
            return self
        return PlannedUnit(self.tensor, str(grid), int(q256), self.rows, self.columns)

    def to_plan_entry(self):
        return BF16 if not self.is_tessera else {"grid": self.grid, "q256": int(self.q256)}


def units_from_plan(
    plan: Mapping[str, object], shapes: Mapping[str, Sequence[int]]
) -> "tuple[PlannedUnit, ...]":
    """Read an exporter ``--plan-json`` plus a ``{tensor: (rows, columns)}`` table.

    Every tensor the plan names must have a shape: a rung is a rate on a shape,
    so a unit whose shape is unknown cannot be priced, and guessing one is how
    the control stops being a control.
    """
    units = []
    for tensor in sorted(plan):
        entry = plan[tensor]
        shape = shapes.get(tensor)
        if shape is None:
            raise TesseraError(
                f"{tensor} is planned but has no shape; a Tessera rung is a rate "
                "on a shape and cannot be priced without one"
            )
        rows, columns = (int(v) for v in shape)
        if isinstance(entry, str):
            if entry != BF16:
                raise TesseraError(f"{tensor}: unreadable plan entry {entry!r}")
            units.append(PlannedUnit(tensor, BF16, None, rows, columns))
            continue
        if not isinstance(entry, Mapping):
            raise TesseraError(f"{tensor}: unreadable plan entry {entry!r}")
        units.append(
            PlannedUnit(tensor, str(entry["grid"]), int(entry["q256"]), rows, columns)
        )
    return tuple(units)


def plan_wire_bits(units: Iterable[PlannedUnit]) -> Fraction:
    """Total wire bits of a plan, BF16 passthroughs included at 16 per parameter."""
    return sum((unit.wire_bits for unit in units), Fraction(0))


@dataclass(frozen=True)
class ByteMatch:
    """Two arms' bit totals, and whether they are close enough to be a control.

    ``candidate_bits`` and ``control_bits`` are over the units the control
    *varies* -- the Tessera ones.  A BF16 unit is identical in both arms, so
    including it would shrink the reported slack without changing anything, and
    the honest denominator for a rate-axis claim is the rate axis.

    **The four numbers are validated here and not above here** (tessera#225).
    This class is public and :func:`uniform_control` builds one directly, so a
    domain enforced only in :func:`assert_byte_matched` would be a domain with
    two doors.  Every field is converted exactly and refused by name: whole
    positive bit totals, an exact positive parameter count, and a tolerance
    that is a fraction of the candidate's own bits.  Before that,
    ``assert_byte_matched(0, 800, 1)`` returned an *accepted* match whose
    ``relative_slack`` was 0 -- a perfect control for an arm of no bytes.
    """

    candidate_bits: Fraction
    control_bits: Fraction
    varying_params: int
    max_relative_slack: Fraction = DEFAULT_MAX_RELATIVE_SLACK

    def __post_init__(self) -> None:
        where = "a byte match"
        for field in ("candidate_bits", "control_bits"):
            object.__setattr__(self, field, _bit_total(
                getattr(self, field), field=field, where=where))
        object.__setattr__(self, "varying_params", _exact_count(
            self.varying_params, field="varying_params", where=where))
        object.__setattr__(self, "max_relative_slack", _tolerance(
            self.max_relative_slack, field="max_relative_slack", where=where))

    @property
    def slack_bits(self) -> Fraction:
        """``control - candidate``: positive when the control is the fatter arm."""
        return self.control_bits - self.candidate_bits

    @property
    def relative_slack(self) -> Fraction:
        """``|slack| / candidate``, always over a denominator that exists."""
        return abs(self.slack_bits) / self.candidate_bits

    @property
    def candidate_bpp(self) -> Fraction:
        return Fraction(self.candidate_bits, self.varying_params)

    @property
    def control_bpp(self) -> Fraction:
        return Fraction(self.control_bits, self.varying_params)

    @property
    def fatter_arm(self) -> str:
        if self.slack_bits > 0:
            return "control"
        if self.slack_bits < 0:
            return "candidate"
        return "neither"

    @property
    def control_is_no_larger(self) -> bool:
        return self.control_bits <= self.candidate_bits

    @property
    def byte_matched(self) -> bool:
        return self.relative_slack <= self.max_relative_slack

    def to_json(self) -> dict:
        return {
            "candidate_bits": int(self.candidate_bits),
            "control_bits": int(self.control_bits),
            "varying_params": int(self.varying_params),
            "candidate_bpp": float(self.candidate_bpp),
            "control_bpp": float(self.control_bpp),
            "slack_bits": int(self.slack_bits),
            "relative_slack": float(self.relative_slack),
            "relative_slack_ppm": float(self.relative_slack * 1_000_000),
            "fatter_arm": self.fatter_arm,
            "control_is_no_larger": bool(self.control_is_no_larger),
            "max_relative_slack": [
                self.max_relative_slack.numerator,
                self.max_relative_slack.denominator,
            ],
            "byte_matched": bool(self.byte_matched),
        }


def assert_byte_matched(
    candidate_bits,
    control_bits,
    varying_params: int,
    *,
    max_relative_slack: Fraction = DEFAULT_MAX_RELATIVE_SLACK,
    require_no_larger: bool = False,
    where: str = "the uniform control",
) -> ByteMatch:
    """Refuse a candidate/control pair whose bytes do not match.

    Takes two bit totals rather than two plans, so the same assertion serves
    before the export (from :func:`plan_wire_bits`) and after it (from
    :func:`bits_from_manifest`) -- and a disagreement between those two
    readings is itself a defect this will name.

    ``require_no_larger`` additionally refuses a control that outweighs the
    candidate, for a claim that must stay conservative against the allocation
    whichever arm wins.

    The four inputs are converted and domain-checked by :class:`ByteMatch`
    itself, so an invalid total refuses by field name here rather than
    arriving as a ``relative_slack`` of zero (tessera#225).
    """
    match = ByteMatch(candidate_bits, control_bits, varying_params,
                      max_relative_slack)
    if not match.byte_matched:
        raise ControlNotByteMatchedError(
            f"{where}: {int(match.control_bits)} bits against the candidate's "
            f"{int(match.candidate_bits)} -- {float(match.relative_slack) * 100:.4f}% "
            f"apart, over the {float(match.max_relative_slack) * 100:.4f}% a control "
            f"may be.  The {match.fatter_arm} arm is the fatter one, so the "
            "comparison would price those bytes as quality."
        )
    if require_no_larger and not match.control_is_no_larger:
        raise ControlNotByteMatchedError(
            f"{where}: the control outweighs the candidate by "
            f"{int(match.slack_bits)} bits, and require_no_larger was asked for."
        )
    return match


@dataclass(frozen=True)
class UniformControl:
    """The uniform arm a rate-axis candidate has to beat, and by how much it may not.

    ``units`` is the whole control plan -- BF16 passthroughs carried through
    from the candidate, every Tessera unit at ``(grid, q256)``.

    ``dominated_by`` names a higher rung that weighs no more than the one the
    byte match chose, over this plan's *own* shape multiset -- ``None`` when
    there is none, which is every production-shaped plan.  It is reported and
    not corrected: matching bytes is what this class promises, and the nearest
    rung is the nearest rung.  But a control sitting on a rung that a better
    one already matches on bytes is a handicapped uniform arm, and an
    allocation that beats it has beaten something it should not have been
    offered either (issue #43).
    """

    grid: str
    q256: int
    units: "tuple[PlannedUnit, ...]"
    match: ByteMatch
    rule: str
    searched: "tuple[int, int]"
    legal_rungs: int
    bracket: dict
    dominated_by: "int | None" = None

    @property
    def plan(self) -> dict:
        """The control as an exporter ``--plan-json``."""
        return {unit.tensor: unit.to_plan_entry() for unit in self.units}

    @property
    def tessera_units(self) -> "tuple[PlannedUnit, ...]":
        return tuple(unit for unit in self.units if unit.is_tessera)

    def to_json(self) -> dict:
        return {
            "grid": self.grid,
            "q256": int(self.q256),
            "rule": self.rule,
            "searched_q256": [int(self.searched[0]), int(self.searched[1])],
            "legal_rungs": int(self.legal_rungs),
            "bracket": self.bracket,
            "units": len(self.tessera_units),
            "bf16_carried": len(self.units) - len(self.tessera_units),
            "dominated_by": self.dominated_by,
            "match": self.match.to_json(),
        }


def uniform_control(
    units: Iterable[PlannedUnit],
    *,
    grid: "str | None" = None,
    rule: str = "nearest",
    rungs: "Iterable[int] | None" = None,
    max_relative_slack: Fraction = DEFAULT_MAX_RELATIVE_SLACK,
    assert_match: bool = True,
) -> UniformControl:
    """The one-rung plan that weighs what this candidate weighs.

    The search is a brute-force scan of every rung the grid admits, priced at
    each unit's own shape, and it ranks by **bits** rather than by rung.  The two
    orders agree at the production shape
    ``test_wire_bits_rise_with_the_rung_on_every_grid`` sweeps (1024x3072) and
    **disagree on small units** -- ``wire_recipe`` chooses body and plane per
    rung, and below the E2M1x2 coset cap a 4096-byte window table buys a rung
    that a 512-byte forest undercuts, so on a 64x512 unit R736..R895 all cost
    more bits than R896 (measured in ``tests/test_rate_menu.py``, issue
    tessera#43).  Bits, not rung, is what a byte match means.  Rungs the
    grammar refuses are skipped rather than approximated.

    Raises when the candidate's Tessera units span more than one grid and no
    ``grid`` is named: "one uniform rung" has no meaning across two families,
    and picking one silently would answer a question nobody asked.  Raises,
    unless ``assert_match=False``, when the nearest rung is further from the
    candidate than a control may be -- which is what the 0.239-bpp hole below
    the E2M1x2 coset cap produces, and is a refusal rather than a warning
    because the alternative is discovering it after two serves.
    """
    if rule not in MATCH_RULES:
        raise TesseraError(f"unknown match rule {rule!r}; one of {MATCH_RULES}")
    units = tuple(units)
    varying = tuple(unit for unit in units if unit.is_tessera)
    if not varying:
        raise TesseraError(
            "this plan quantizes nothing with Tessera, so it has no rate axis "
            "and no uniform control"
        )

    grids = sorted({unit.grid for unit in varying})
    if grid is None:
        if len(grids) != 1:
            raise TesseraError(
                f"the candidate spans {len(grids)} grids ({', '.join(grids)}); "
                "a single uniform rung is not defined across families.  Name "
                "grid= to say which family the control should be built on."
            )
        grid = grids[0]
    payload = grid_for_name(grid)

    candidate_bits = sum((unit.wire_bits for unit in varying), Fraction(0))
    varying_params = sum(unit.params for unit in varying)

    # Five distinct shapes carry 196 units on Qwen3-0.6B, so price the shape
    # multiset once per rung rather than the unit list.
    multiset = Counter(unit.shape for unit in varying)
    ceiling = int(rung_ceiling(payload))
    candidates = list(rungs) if rungs is not None else list(range(1, ceiling + 1))

    priced: "list[tuple[int, Fraction]]" = []
    for q in candidates:
        try:
            total = sum(
                (unit_wire_bits(payload, q, rows, columns) * n
                 for (rows, columns), n in multiset.items()),
                Fraction(0),
            )
        except (GrammarError, ValueError, ZeroDivisionError):
            continue
        priced.append((int(q), total))
    if not priced:
        raise TesseraError(
            f"no rung of {grid} prices on this plan's shapes; searched "
            f"{min(candidates)}..{max(candidates)}"
        )

    if rule == "nearest":
        # Ties to the lighter arm: a control that does not outweigh the
        # candidate is the conservative one to prefer when both are equally far.
        best = min(priced, key=lambda row: (abs(row[1] - candidate_bits), row[1]))
    else:
        feasible = [row for row in priced if row[1] <= candidate_bits]
        if not feasible:
            raise TesseraError(
                f"no rung of {grid} weighs {int(candidate_bits)} bits or less on "
                "this plan's shapes, so no no_larger control exists; the "
                "candidate is below the family's lightest rung"
            )
        best = max(feasible, key=lambda row: row[1])
    q256, control_bits = best

    below = max((row for row in priced if row[1] <= candidate_bits),
                key=lambda row: row[1], default=None)
    above = min((row for row in priced if row[1] > candidate_bits),
                key=lambda row: row[1], default=None)
    bracket = {
        "below": None if below is None else {"q256": below[0], "bits": int(below[1])},
        "above": None if above is None else {"q256": above[0], "bits": int(above[1])},
        # The axis's local step at this candidate.  Half of it is the best any
        # nearest-rung control can do, so a slack near it says the *axis* is
        # coarse here (the E2M1x2 coset cap) rather than the search sloppy.
        "quantum_bits": (None if below is None or above is None
                         else int(above[1] - below[1])),
    }

    match = ByteMatch(candidate_bits, control_bits, varying_params,
                      Fraction(max_relative_slack))
    if assert_match:
        assert_byte_matched(
            candidate_bits, control_bits, varying_params,
            max_relative_slack=max_relative_slack,
            where=(f"the nearest {grid} rung to this candidate is R{q256}"
                   if rule == "nearest" else
                   f"the heaviest {grid} rung that does not outweigh this "
                   f"candidate is R{q256}"),
        )
    # The cheapest rung above the chosen one that does not outweigh it, and the
    # highest of those when several tie -- the rung that dominates this control,
    # or None.  Computed over this plan's shape multiset, since domination is a
    # property of the shapes and not of the family.
    dominating = [row for row in priced if row[0] > q256 and row[1] <= control_bits]
    dominated_by = (
        min(dominating, key=lambda row: (row[1], -row[0]))[0] if dominating else None
    )
    control_units = tuple(unit.at_rung(grid, q256) for unit in units)
    return UniformControl(
        grid=str(grid),
        q256=int(q256),
        units=control_units,
        match=match,
        rule=rule,
        searched=(min(row[0] for row in priced), max(row[0] for row in priced)),
        legal_rungs=len(priced),
        bracket=bracket,
        dominated_by=dominated_by,
    )


def bits_from_manifest(checkpoint: "str | Path") -> "tuple[Fraction, dict]":
    """``(total wire bits, {tensor: bits})`` from an exported checkpoint's manifest.

    The same read ``experiments/check_wire_against_plan.py`` makes, so a
    post-export assertion compares the bytes that shipped rather than the bytes
    a plan predicted.
    """
    path = Path(checkpoint)
    if path.is_dir():
        path = path / "tessera_serving_manifest.json"
    manifest = json.loads(path.read_text())
    per_tensor = {}
    for module in manifest["modules"].values():
        for role in module["roles"]:
            per_tensor[role["tensor"]] = Fraction(int(role["wire_bytes"]) * 8)
    return sum(per_tensor.values(), Fraction(0)), per_tensor


def control_block(
    control: UniformControl,
    *,
    candidate_kl=None,
    control_kl=None,
    metric: str = "kl_vs_bf16",
    candidate_label: str = "allocated",
    note: "str | None" = None,
) -> dict:
    """The JSON an artifact carries beside its bpp (principle 12, issue #3).

    With both KLs it states the verdict the gate exists to record: *this
    candidate beat / did not beat its uniform control by X at matched bytes*.
    Without them it states, equally explicitly, that the control was built and
    **not served** -- which is a different claim from a passing gate and must
    not read like one.

    **A measured verdict requires the byte match to have held** (tessera#225).
    ``uniform_control(..., assert_match=False)`` builds an unmatched pair on
    purpose -- that is how the E2M1x2 coset hole is *reported* rather than
    papered over -- and such a pair stays representable here, as the unserved
    block, which is an explicitly unqualified diagnostic.  What it may not
    become is the measured verdict, whose own sentence is "against the
    byte-matched uniform": on the issue's pair that read as a victory while
    the arms were 3.16% apart in bytes, 31.6x the tolerance, so the winning
    arm was simply the one holding the extra bytes.  Both KLs are validated
    for the same reason the totals are: the verdict divides them.
    """
    block = {
        "schema": CONTROL_SCHEMA,
        "candidate_label": candidate_label,
        "control": control.to_json(),
        "reason": (
            "a candidate on a continuous rate axis is a claim that choosing "
            "rungs beats spending the same bytes at one rung; this is the arm "
            "that tests it (tessera#3, measured in "
            "docs/measurements/tessera-allocated-served-2026-09-02.md)"
        ),
    }
    if candidate_kl is None or control_kl is None:
        block["verdict"] = {
            "metric": metric,
            "measured": False,
            "detail": "the control was built and priced; neither arm was served",
        }
    else:
        where = f"the {candidate_label} arm against its uniform control"
        if not control.match.byte_matched:
            raise ControlNotByteMatchedError(
                f"{where}: byte_matched is False -- {int(control.match.control_bits)} "
                f"bits against the candidate's {int(control.match.candidate_bits)}, "
                f"{float(control.match.relative_slack) * 100:.4f}% apart against the "
                f"{float(control.match.max_relative_slack) * 100:.4f}% a control may "
                "be.  A measured verdict here would read 'against the byte-matched "
                "uniform' for a pair that is not byte matched, pricing the "
                "difference in bytes as quality; the unserved block is what an "
                "unmatched plan may carry."
            )
        candidate_kl = require_kl(candidate_kl, field="candidate_kl", where=where)
        control_kl = require_kl(control_kl, field="control_kl", where=where)
        ratio = candidate_kl / control_kl if control_kl else float("inf")
        block["verdict"] = {
            "metric": metric,
            "measured": True,
            "candidate": candidate_kl,
            "control": control_kl,
            "candidate_over_control": ratio,
            "beat_control": candidate_kl < control_kl,
            "detail": (
                f"{candidate_label} {candidate_kl:.6g} against the byte-matched "
                f"uniform {control.grid} R{control.q256} {control_kl:.6g} "
                f"({ratio:.4g}x); the control is "
                f"{float(control.match.relative_slack) * 1e6:.1f} ppm "
                f"{'fatter' if control.match.slack_bits > 0 else 'lighter'} "
                "in bytes"
            ),
        }
    if note:
        block["note"] = note
    return block


#: The selection mode the continuous Tessera menu requires
#: (``docs/ARCHITECTURE.md`` §4.10, tessera#2): the rungs were chosen by the
#: surrogate, so a served KL -- the candidate against its byte-matched uniform
#: control -- selects before anything ships.  ``COST_MODE=aura`` is not an
#: accepted substitute until someone measures AURA on a rung sweep; until
#: then it is unmeasured on exactly the failure of tessera#1.
REQUIRED_SELECTION_MODE = "validated-surrogate"

SELECTION_SCHEMA = "tessera.menu_selection.v1"

_SELECTION_REASON = (
    "a plan at more than one (grid, rung) embodies a rung selection, and the "
    "surrogate that chose it is measured to invert the answer: 2.00x worse "
    "served KL than the byte-matched uniform arm at 4.0 bpp, 95% of the gap "
    "on the seven units the surrogate itself priced (tessera#1, measured in "
    "docs/measurements/tessera-allocated-served-2026-09-02.md §5 and §7).  "
    "docs/ARCHITECTURE.md §4.10 requires validated-surrogate selection for "
    "this menu (tessera#2)."
)


def selection_requirement(units: Iterable[PlannedUnit]) -> dict:
    """Whether this plan embodies a rung selection, and what that requires.

    The test is on the plan itself, never on a roster: the distinct (grid,
    rung) pairs over the plan's own Tessera units.  One pair means the plan
    is the null hypothesis -- one rung at matched bytes is what the gate
    tests *against* -- so there is no selection for the validated-surrogate
    gate to check.  More than one pair means the surrogate chose rungs, and
    the block says so with ``validated: False``: no KL has been served here,
    and serving the byte-matched uniform control
    (``experiments/uniform_control.py verify``, whose verdict
    :func:`control_block` records) is what flips that answer, not building
    the plan.  A plan with no Tessera units has no rate axis at all.
    """
    members = tuple(units)
    pairs = sorted(
        {(unit.grid, int(unit.q256)) for unit in members if unit.is_tessera}
    )
    if not pairs:
        return {
            "schema": SELECTION_SCHEMA,
            "mode_required": REQUIRED_SELECTION_MODE,
            "requires_validation": False,
            "validated": True,
            "distinct_rungs": [],
            "detail": (
                "no Tessera units: no rate axis, so no rung selection"
            ),
            "reason": _SELECTION_REASON,
        }
    if len(pairs) == 1:
        grid, q256 = pairs[0]
        return {
            "schema": SELECTION_SCHEMA,
            "mode_required": REQUIRED_SELECTION_MODE,
            "requires_validation": False,
            "validated": True,
            "distinct_rungs": [[grid, q256]],
            "detail": (
                f"one (grid, rung) pair ({grid} R{q256}): the plan embodies "
                "no rung selection, so there is nothing for the "
                "validated-surrogate gate to check"
            ),
            "reason": _SELECTION_REASON,
        }
    return {
        "schema": SELECTION_SCHEMA,
        "mode_required": REQUIRED_SELECTION_MODE,
        "requires_validation": True,
        "validated": False,
        "distinct_rungs": [[grid, q256] for grid, q256 in pairs],
        "detail": (
            f"{len(pairs)} distinct (grid, rung) pairs: a rung selection the "
            "surrogate made and no served KL has validated.  Serve the "
            "byte-matched uniform control (experiments/uniform_control.py "
            "verify) and record its verdict before this plan ships."
        ),
        "reason": _SELECTION_REASON,
    }


LANDING_SCHEMA = "tessera.landing_ordering.v1"

_LANDING_REASON = (
    "a LUT-plane arm score is a joint measurement of the refit and the "
    "sixteen-entry landing, and the landing reorders the arms: on the wire "
    "Gauss-Seidel wins and Jacobi is third, with the landing removed Jacobi is "
    "first (tessera#85, six dense Qwen3-0.6B units, E2M1x2 q256=896, LDLQ "
    "1.0/32, held-out `out` geomeans).  So an ordering quoted from one landing "
    "is not an ordering of the refit objectives, and this block carries both "
    "or says which one is missing."
)


@dataclass(frozen=True)
class LandingOrdering:
    """Two arm orderings of one screen, and whether they agree -- as a value.

    The LUT plane's per-block scales land on sixteen E4M3 entries, so every
    arm score this repo has taken on that plane measures the refit **and**
    the landing.  Issue #85 measured what that costs the comparison: the two
    orderings differ, so "which refit objective is best" has two answers and
    the receipts quoted one of them as if it were the other.

    This is the pair, rendered so a reader and a test see the same object:
    :attr:`arms` in on-wire order, both score columns, and the agreement
    read off them.  Lower is better in both columns -- these are errors or
    error ratios -- and the two columns are never compared to each other,
    only ranked within themselves, because one of them is not a wire and its
    absolute level means nothing a decision may read.

    **Disagreement is recorded here and refused nowhere.**  What ships is the
    landed wire, so the on-wire ordering is the correct measurement of the
    shipped object rather than a confound in it; the landing-disabled column
    is the ceiling a *different* landing would compete for (issue #50), and
    no such landing exists.  The argument is in
    :func:`assert_plane_promotion`, whose fifth leg is the one that does
    refuse: it demands the screen be taken on the wire at all.
    """

    arms: "tuple[str, ...]"
    on_wire: "tuple[float, ...]"
    landing_disabled: "tuple[float, ...]"
    wire_landing: str
    disabled_landing: str
    where: str

    @staticmethod
    def _rank(arms, scores) -> "tuple[str, ...]":
        return tuple(name for _, name in sorted(zip(scores, arms)))

    @property
    def wire_order(self) -> "tuple[str, ...]":
        """Arms best-first under the wire's landing, ties broken by name."""
        return self._rank(self.arms, self.on_wire)

    @property
    def disabled_order(self) -> "tuple[str, ...]":
        """Arms best-first with the landing removed, ties broken by name."""
        return self._rank(self.arms, self.landing_disabled)

    @staticmethod
    def _best(arms, scores) -> "tuple[str, ...]":
        floor = min(scores)
        return tuple(sorted(a for a, v in zip(arms, scores) if v == floor))

    @property
    def wire_best(self) -> "tuple[str, ...]":
        return self._best(self.arms, self.on_wire)

    @property
    def disabled_best(self) -> "tuple[str, ...]":
        return self._best(self.arms, self.landing_disabled)

    @property
    def same_best(self) -> bool:
        """Do the two columns choose the same winner (or the same tied set)?"""
        return self.wire_best == self.disabled_best

    @property
    def inversions(self) -> "tuple[tuple[str, str], ...]":
        """Every arm pair the two columns order differently, ``(a, b)`` sorted.

        A pair is an inversion when ``sign(wire_a - wire_b)`` differs from
        ``sign(disabled_a - disabled_b)``, with an exact tie its own sign.  No
        tolerance: "disagree by more than x%" would be a threshold from
        intuition (AGENTS.md rule 2), and a tie that is not a tie in both
        columns is a genuine difference of order, not noise to be absorbed.
        """
        wire = dict(zip(self.arms, self.on_wire))
        free = dict(zip(self.arms, self.landing_disabled))
        out = []
        for i, a in enumerate(self.arms):
            for b in self.arms[i + 1:]:
                lo, hi = (a, b) if a <= b else (b, a)
                sw = (wire[lo] > wire[hi]) - (wire[lo] < wire[hi])
                sf = (free[lo] > free[hi]) - (free[lo] < free[hi])
                if sw != sf:
                    out.append((lo, hi))
        return tuple(sorted(out))

    @property
    def same_order(self) -> bool:
        return not self.inversions

    def to_json(self) -> dict:
        return {
            "schema": LANDING_SCHEMA,
            "wire_landing": self.wire_landing,
            "disabled_landing": self.disabled_landing,
            "arms": list(self.arms),
            "on_wire": [float(v) for v in self.on_wire],
            "landing_disabled": [float(v) for v in self.landing_disabled],
            "wire_order": list(self.wire_order),
            "disabled_order": list(self.disabled_order),
            "wire_best": list(self.wire_best),
            "disabled_best": list(self.disabled_best),
            "same_best": self.same_best,
            "same_order": self.same_order,
            "inversions": [list(pair) for pair in self.inversions],
            "where": self.where,
            "reason": _LANDING_REASON,
        }


def landing_ordering(
    on_wire: "Mapping[str, float]",
    landing_disabled: "Mapping[str, float]",
    *,
    wire_landing: str = LUT_LANDING_WIRE,
    disabled_landing: str = "none",
    where: str = "the LUT-plane arm screen",
) -> LandingOrdering:
    """Rank the same arms under both landings and say whether the orders agree.

    Both mappings are ``arm -> score``, lower better, over the **same** arm
    set: a column missing an arm is a pair that was never taken, and it
    refuses rather than ranking the arms it happens to have.  Two arms is the
    minimum -- one arm has no ordering to invert.

    The landing-disabled column costs one extra encode per arm
    (``lut_landing("none")``, ``experiments/lut_landing_ceiling.py``).  It is
    **not** readable off ``refit_diagnostics``: that instrument's
    ``continuous`` leg is a within-call quantity by its own contract -- for a
    1-D metric it records the separable parabola, equal to the weighted error
    only up to a constant -- and the arms being ranked here are 1-D
    (``h^1.0``) against full-H (Jacobi, Gauss-Seidel).  So the diagnostics
    give the *size* of the landing leg within one arm and cannot give the
    ordering across arms; issue #85's "reporting change, not a new
    measurement" holds for the former and not for the latter.
    """
    wire = {str(k): float(v) for k, v in dict(on_wire).items()}
    free = {str(k): float(v) for k, v in dict(landing_disabled).items()}
    if set(wire) != set(free):
        missing = sorted(set(wire) ^ set(free))
        raise TesseraError(
            f"{where}: the two landings must rank the same arms; "
            f"{missing!r} appears in one column only, and a column ranked "
            "over a different arm set is not the same ordering"
        )
    if len(wire) < 2:
        raise TesseraError(
            f"{where}: an ordering needs at least two arms, got {sorted(wire)!r}"
        )
    for label, column in (("on_wire", wire), ("landing_disabled", free)):
        bad = {k: v for k, v in column.items()
               if not math.isfinite(v) or v <= 0}
        if bad:
            raise TesseraError(
                f"{where}: {label} scores are positive finite errors, got {bad!r}"
            )
    if wire_landing != LUT_LANDING_WIRE:
        raise GrammarError(
            f"{where}: the on-wire column must be the wire "
            f"({LUT_LANDING_WIRE!r}), got {wire_landing!r}"
        )
    if disabled_landing not in LUT_LANDING_MODES or disabled_landing == LUT_LANDING_WIRE:
        raise GrammarError(
            f"{where}: the landing-disabled column must be a non-wire landing, "
            f"one of {[m for m in LUT_LANDING_MODES if m != LUT_LANDING_WIRE]}, "
            f"got {disabled_landing!r}"
        )
    arms = tuple(sorted(wire, key=lambda a: (wire[a], a)))
    return LandingOrdering(
        arms=arms,
        on_wire=tuple(wire[a] for a in arms),
        landing_disabled=tuple(free[a] for a in arms),
        wire_landing=wire_landing,
        disabled_landing=disabled_landing,
        where=where,
    )


PROMOTION_SCHEMA = "tessera.plane_promotion.v1"

#: The coordinator's cross-check on a LUT-plane promotion, restated here and
#: not moved: the candidate's GLM six-expert geomean against the same wire
#: without levers is no worse than 1.00x.  The 2026-09-02 receipt wrote it;
#: issue #65 pins it.  A caller that holds a tighter bar passes it in --
#: ``glm_bar`` is a *tightening* override, enforced as one by
#: :func:`_require_pinned_glm_bar` (tessera#224).
GLM_GATE = 1.00


def _require_pinned_glm_bar(glm_bar: float, *, where: str) -> None:
    """A caller may hold a tighter GLM bar; it may not hold a looser one.

    The one home for this rule (AGENTS.md rule 4), called by
    :func:`assert_plane_promotion` before the cross-check it protects and by
    :meth:`PlanePromotion.__post_init__` so the class cannot be built around
    it.  Until tessera#224 the gate compared only against the caller's own
    ``glm_bar``, so ``glm_ratio=1.5`` promoted under ``glm_bar=2.0``: a 50%
    six-expert regression clearing a cross-check the arm being checked had
    written.  Moving :data:`GLM_GATE` itself is a decision this gate does not
    make; refusing a caller who tries to move it here is.
    """
    if glm_bar > GLM_GATE:
        raise PromotionRefusedError(
            f"{where}: glm_bar {glm_bar:.4g}x is looser than the pinned "
            f"{GLM_GATE:.4g}x GLM gate, which no caller relaxes -- the "
            "override exists to tighten the coordinator's cross-check, and at "
            f"glm_bar={glm_bar:.4g} a {glm_bar:.4g}x six-expert regression "
            "would promote (tessera#65, #224)"
        )


_PROMOTION_REASON = (
    "a per-plane default is set by a screen and a cross-check, and the screen "
    "that sets it must be won by the arm that ships: the 2026-09-02 receipt "
    "promoted `hessian` on a 1.38% six-unit geomean it won on 2 of 6 units, "
    "while the served KL quoted for the pick measured `h^1.0`, the arm not "
    "selected (tessera#65, measured in "
    "docs/measurements/tessera-ldlq-lut-plane-served-2026-09-02.md and "
    "docs/measurements/tessera-lut-refit-gauss-seidel-2026-09-03.md)."
)


@dataclass(frozen=True)
class PlanePromotion:
    """A per-plane promotion its evidence carries, refused otherwise.

    ``unit_ratios`` is the unit set: the candidate's per-unit error over the
    incumbent's, in the receipt's own unit order, one entry per unit.  A ratio
    below one is a unit the candidate wins.  The geomean is derived from
    these, never presented beside them, so a geomean cannot arrive without
    the units that compose it.  ``served_arm`` names the arm the served KL
    was measured on -- ``None`` when nothing was served.  ``landing`` names
    the landing the per-unit ratios were taken under; only the wire promotes
    (tessera#85), so on an accepted promotion it is always
    :data:`tessera.encode.LUT_LANDING_WIRE` -- stamped rather than implied,
    because the number it certifies outlives the run that produced it.
    """

    candidate: str
    served_arm: "str | None"
    unit_ratios: "tuple[float, ...]"
    geomean: float
    wins: int
    glm_ratio: float
    glm_bar: float
    served_kl: "float | None"
    served_bar: float
    landing: str
    where: str

    def __post_init__(self) -> None:
        """The domains, on the object :func:`promotion_block` publishes.

        ``promotion_block`` says "only a promotion this gate accepted reaches
        here", and this is what makes that true rather than conventional: the
        class is public, so a domain enforced only in
        :func:`assert_plane_promotion` would be a domain with two doors
        (tessera#224).  Each number is refused by field name, and the derived
        pair -- ``geomean`` and ``wins`` -- must be the pair these very ratios
        make, so a summary can never arrive beside a unit set it does not
        summarise.
        """
        object.__setattr__(self, "unit_ratios",
                           _unit_ratios(self.unit_ratios, where=self.where))
        for field in ("glm_ratio", "glm_bar"):
            object.__setattr__(self, field, _error_ratio(
                getattr(self, field), field=field, where=self.where))
        object.__setattr__(self, "served_bar", require_kl(
            self.served_bar, field="served_bar", where=self.where))
        if self.served_kl is not None:
            object.__setattr__(self, "served_kl", require_kl(
                self.served_kl, field="served_kl", where=self.where))
        _require_pinned_glm_bar(self.glm_bar, where=self.where)
        geomean = _unit_geomean(self.unit_ratios)
        # The tolerance is float64's, not a judgement: an n-term reduction in
        # the log domain rounds at most once per term, so n ulps of the value
        # itself is the whole room a second correct computation of this number
        # has.  Anything wider would be a different number wearing this one's
        # name (AGENTS.md rule 2).
        if not abs(float(self.geomean) - geomean) <= len(self.unit_ratios) * math.ulp(geomean):
            raise TesseraError(
                f"{self.where}: geomean {float(self.geomean):.6g} is not the "
                f"geomean of these {len(self.unit_ratios)} unit ratios "
                f"({geomean:.6g}) -- it is derived from them, never presented "
                "beside them"
            )
        wins = sum(1 for r in self.unit_ratios if r < 1)
        if int(self.wins) != wins:
            raise TesseraError(
                f"{self.where}: unit_wins {self.wins!r} is not the number of "
                f"these unit ratios below 1 ({wins})"
            )

    def to_json(self) -> dict:
        return {
            "schema": PROMOTION_SCHEMA,
            "candidate": self.candidate,
            "served_arm": self.served_arm,
            "landing": self.landing,
            "units": len(self.unit_ratios),
            "unit_wins": int(self.wins),
            "unit_ratios": [float(r) for r in self.unit_ratios],
            "geomean": float(self.geomean),
            "glm_ratio": float(self.glm_ratio),
            "glm_bar": float(self.glm_bar),
            "served_kl": None if self.served_kl is None else float(self.served_kl),
            "served_bar": float(self.served_bar),
            "reason": _PROMOTION_REASON,
        }


def assert_plane_promotion(
    *,
    candidate: str,
    served_arm: "str | None",
    unit_ratios: Sequence[float],
    glm_ratio: float,
    served_kl: "float | None",
    served_bar: float,
    glm_bar: float = GLM_GATE,
    landing: str = LUT_LANDING_WIRE,
    where: str = "the per-plane promotion",
) -> PlanePromotion:
    """Refuse a per-plane promotion its evidence does not carry.

    **Before any leg, the evidence has to be evidence** (tessera#224).  Each
    per-unit ratio, ``glm_ratio``, ``glm_bar``, ``served_kl`` and
    ``served_bar`` is checked against the domain its own definition gives it
    and refused by field name, because an ordered comparison is not a
    validity check: ``not (nan <= bar)`` refuses, but ``not (-inf <= bar)``
    passes, ``served_kl < inf`` passes for every KL there is, and a
    ``glm_bar`` above :data:`GLM_GATE` passes a regression the pinned gate
    forbids.  All six of the issue's cases promoted before this.  A ratio of
    two errors is strictly positive -- zero is a division artifact, and the
    geomean reads it in logs -- while a KL divergence may be zero, so those
    two domains are spelled apart rather than sharing one word that fits
    neither.  ``glm_bar`` may only ever tighten
    (:func:`_require_pinned_glm_bar`).

    Five legs, in the order the receipt learned them.  The GLM cross-check
    is first and exactly as written: above ``glm_bar`` refuses, whatever the
    screen says.  Then the screen itself: the geomean must beat the
    incumbent, and -- never on the geomean alone -- the candidate must win a
    strict majority of the receipt's own units.  Then the identity the
    promotion stands on: the served KL must measure the promoted arm, and it
    must beat its bar.  A served number for a different arm is not evidence
    for the promoted one, and no served number at all is a screen, not a
    result.  Then, last learned and first checked, the ``landing``: the
    per-unit ratios must have been taken on the wire.

    **The fifth leg (tessera#85), and the leg it is deliberately not.**  On
    the LUT plane a per-block scale lands on one of sixteen E4M3 entries, and
    ``tessera.encode.lut_landing`` can remove that landing to read issue
    #50's ceiling.  The arms reorder when it does: on the wire Gauss-Seidel
    wins and Jacobi is third; with the landing removed Jacobi is first.  Two
    consequences, and only one of them is a refusal.

    * *A screen taken off the wire is refused.*  The landing-disabled column
      holds the most attractive numbers anyone has measured on this plane
      (0.7057 against the wire's 1.0000), and until now the gate had no way
      to tell them from wire numbers -- it read a ratio and could not ask
      what it was a ratio of.  ``landing`` is that value.  It is
      caller-asserted, exactly as ``served_arm`` is, and for the same reason
      it is knowable: non-wire ratios can only be produced inside a
      ``lut_landing`` context, whose sink already reports
      ``serialisable=False``.  The default is the wire because that is the
      state every encode runs in.
    * *A disagreement between the two orderings is recorded and not refused*
      (:func:`landing_ordering`).  What ships is the landed wire, so the
      on-wire ordering is the correct measurement of the shipped object
      rather than a confound in it: "Gauss-Seidel plus this landing beats
      Jacobi plus this landing" is true, and it is the sentence a default
      selection needs.  What #85 corrects is the *attribution* -- that
      sentence was written as "Gauss-Seidel is the better refit" -- and an
      attribution error is fixed by reporting the pair, not by blocking a
      promotion.  Refusing on the disagreement would also pin one
      measurement (one wire, one ``(sigma, block)``, six weight-space Qwen
      units, no serve) as a standing rule about the plane, which is the
      roster-not-rule failure AGENTS.md rule 3 names.  The disagreement is a
      **re-run trigger** for the day a better landing lands, and #50 is where
      that is owed.

    ``served_bar`` has no default on purpose.  It is *the incumbent's own
    served KL at matched bytes* -- the same quantity ``unit_ratios`` is a
    ratio against -- so it moves every time the incumbent does, and a
    module-level constant would be the wrong number the moment one is
    promoted.  The 2026-09-02 receipt's own bar, 0.640, is the *stock* wire's
    served KL, which was the incumbent for "levers vs no levers" and is not
    the incumbent for anything since: the LUT plane's incumbent is `h^1.0` at
    0.5310.  Defaulting to 0.640 would have let a candidate serving 0.60 clear
    this leg while regressing the arm it replaces by 13%, which is the same
    class of error as the unit legs above and would have been made by the
    gate written to refuse it.
    """
    if not isinstance(candidate, str) or not candidate:
        raise TesseraError(f"{where}: the promoted arm must be named, got {candidate!r}")
    ratios = _unit_ratios(unit_ratios, where=where)
    if landing not in LUT_LANDING_MODES:
        raise GrammarError(
            f"{where}: unknown landing {landing!r}; one of "
            f"{list(LUT_LANDING_MODES)} (tessera.encode.lut_landing)"
        )
    if landing != LUT_LANDING_WIRE:
        raise PromotionRefusedError(
            f"{where}: the per-unit ratios were taken at landing "
            f"{landing!r}, which is not a wire -- a ceiling read is the most "
            "any landing could return, not a number this one reaches, and it "
            f"reorders the arms (tessera#85).  Only {LUT_LANDING_WIRE!r} "
            "promotes"
        )
    # Domains before comparisons (tessera#224).  An ordered comparison is not
    # a validity check: `not (nan <= bar)` refuses but `not (-inf <= bar)`
    # passes, `served_kl < inf` passes for every KL, and a caller-supplied
    # `glm_bar` above the pinned one turns the coordinator's cross-check into
    # a number the arm being checked chooses.  A ratio of two errors is
    # strictly positive -- zero is a division artifact and its log is not a
    # number -- while a KL divergence may be zero, so the two domains are
    # spelled apart rather than sharing a "positive" that fits neither.
    glm_ratio = _error_ratio(glm_ratio, field="glm_ratio", where=where)
    glm_bar = _error_ratio(glm_bar, field="glm_bar", where=where)
    served_bar = require_kl(served_bar, field="served_bar", where=where)
    if served_kl is not None:
        served_kl = require_kl(served_kl, field="served_kl", where=where)
    _require_pinned_glm_bar(glm_bar, where=where)
    geomean = _unit_geomean(ratios)
    wins = sum(1 for r in ratios if r < 1)

    if not glm_ratio <= glm_bar:
        raise PromotionRefusedError(
            f"{where}: GLM six-expert {glm_ratio:.4g}x is above the "
            f"{glm_bar:.4g}x gate -- the cross-check the coordinator's gate "
            "requires, and no screen overrules it"
        )
    if not geomean < 1:
        raise PromotionRefusedError(
            f"{where}: {candidate} geomean {geomean:.4g}x does not beat the "
            "incumbent -- there is nothing to promote"
        )
    if not 2 * wins > len(ratios):
        raise PromotionRefusedError(
            f"{where}: never promote on geomean alone -- require per-unit "
            f"wins.  {candidate} takes {wins} of {len(ratios)} units at "
            f"geomean {geomean:.4g}x: the aggregate is carried by a minority "
            "of the unit set"
        )
    if served_arm != candidate:
        raise PromotionRefusedError(
            f"{where}: the served KL measures arm {served_arm!r}, not the "
            f"promoted arm {candidate!r} -- a served number for a different "
            "arm is not evidence for it"
        )
    if served_kl is None or not served_kl < served_bar:
        detail = (
            "no served KL: a screen is not a result"
            if served_kl is None
            else f"served KL {served_kl:.4g} does not beat {served_bar:.4g}"
        )
        raise PromotionRefusedError(f"{where}: {detail}")
    return PlanePromotion(
        candidate=candidate,
        served_arm=served_arm,
        unit_ratios=ratios,
        geomean=geomean,
        wins=wins,
        glm_ratio=glm_ratio,
        glm_bar=glm_bar,
        served_kl=served_kl,
        served_bar=served_bar,
        landing=landing,
        where=where,
    )


def promotion_block(promotion: PlanePromotion) -> dict:
    """The JSON a promoted plane carries beside its default (principle 12).

    Only a promotion :func:`assert_plane_promotion` accepted reaches here,
    so the verdict is the record of which bars it cleared, not a second
    reading of them.
    """
    block = promotion.to_json()
    block["verdict"] = {
        "promoted": True,
        "detail": (
            f"{promotion.candidate} wins {promotion.wins} of "
            f"{len(promotion.unit_ratios)} units at geomean "
            f"{promotion.geomean:.4g}x, GLM {promotion.glm_ratio:.4g}x "
            f"against the {promotion.glm_bar:.4g}x gate, served "
            f"{promotion.served_kl:.4g} against {promotion.served_bar:.4g} "
            f"on the promoted arm, screened at landing "
            f"{promotion.landing!r} (the wire)"
        ),
    }
    return block
