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
* :func:`uniform_control` searches the family's rungs for the one whose whole
  plan weighs what the candidate weighs;
* :func:`assert_byte_matched` refuses a pair that does not, on integer bit
  totals, so a post-export check can hand it two manifests;
* :func:`control_block` renders the pair -- and, when the two KLs are known,
  the verdict -- as the JSON an artifact carries beside its bpp.

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
65 ppm.  The axis is not uniformly dense, though: E2M1x2 jumps **0.239 bpp**
between R895 and R896 where the recipe changes from the window body to the
coset trellis, and near that hole no control this tight exists.  The assertion
fires there rather than quietly comparing two different byte budgets.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .alphabet import BF16_GRID, E2M1_GRID, E4M3_GRID, PayloadGrid, tuple_grid
from .calculator import terminal_rate
from .errors import ControlNotByteMatchedError, GrammarError, TesseraError
from .export import rung_ceiling, wire_recipe
from .manifest import BodyKind, ScalePlaneKind

__all__ = [
    "BF16",
    "CONTROL_SCHEMA",
    "GRID_NAMES",
    "DEFAULT_MAX_RELATIVE_SLACK",
    "ByteMatch",
    "PlannedUnit",
    "UniformControl",
    "assert_byte_matched",
    "bits_from_manifest",
    "control_block",
    "grid_for_name",
    "plan_wire_bits",
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
    )
    bits = rate * rows * columns
    _BITS_CACHE[key] = bits
    return bits


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
    """

    candidate_bits: Fraction
    control_bits: Fraction
    varying_params: int
    max_relative_slack: Fraction = DEFAULT_MAX_RELATIVE_SLACK

    @property
    def slack_bits(self) -> Fraction:
        """``control - candidate``: positive when the control is the fatter arm."""
        return self.control_bits - self.candidate_bits

    @property
    def relative_slack(self) -> Fraction:
        if not self.candidate_bits:
            return Fraction(0)
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
    """
    match = ByteMatch(
        Fraction(candidate_bits),
        Fraction(control_bits),
        int(varying_params),
        Fraction(max_relative_slack),
    )
    if not match.byte_matched:
        raise ControlNotByteMatchedError(
            f"{where}: {int(match.control_bits)} bits against the candidate's "
            f"{int(match.candidate_bits)} -- {float(match.relative_slack) * 100:.4f}% "
            f"apart, over the {float(max_relative_slack) * 100:.4f}% a control may "
            f"be.  The {match.fatter_arm} arm is the fatter one, so the "
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
    """

    grid: str
    q256: int
    units: "tuple[PlannedUnit, ...]"
    match: ByteMatch
    rule: str
    searched: "tuple[int, int]"
    legal_rungs: int
    bracket: dict

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
    each unit's own shape, and it ranks by **bits** rather than by rung.  Today
    the two orders agree -- ``test_wire_bits_rise_with_the_rung_on_every_grid``
    pins that as a measured property of the current recipe table, not as an
    axiom -- but ``wire_recipe`` chooses body and plane per rung, so the day a
    boundary moves the bit order is what a byte match means and the rung order
    is not.  Rungs the grammar refuses are skipped rather than approximated.

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
        candidate_kl, control_kl = float(candidate_kl), float(control_kl)
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
