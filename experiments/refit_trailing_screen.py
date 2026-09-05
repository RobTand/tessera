#!/usr/bin/env python
"""The screen receipt's own proofs, read by whoever reads its ratios (#250).

``refit_trailing_pair.py`` records, beside every number it prints, the evidence
that can invalidate the experiment that produced it:

* ``drift_control_identical`` on the control arm -- the FIRST and LAST runs of
  the same arm in one process reconstructed the same weights.  When they do
  not, the arms were not encoded by one deterministic encoder and no ratio in
  the document is a controlled comparison.
* ``landing``/``serialisable``/``sink_vs_wire_bit_identical`` on every wire
  arm -- the sink the arm was scored off IS the wire that ships.  That identity
  is what licenses scoring an arm off the landing sink at all, and only the
  wire promotes (tessera#85).
* ``matched_pair`` on a trailing arm -- ``codes_identical``, ``bytes_equal``,
  ``inner_objectives_equal`` and ``inner_refits_identical``: the pair differs
  in the last scale plane and in nothing else.

The producer wrote them and printed them; nothing read them.  A screen whose
control DIFFERS, or whose trailing arm changed its packed codes, reached
``assert_plane_promotion`` on its ratios alone and could print PROMOTED
(tessera#250).  This module is the one place those recordings are turned into
a refusal, so the producer and the gate cannot drift into two opinions about
what a valid screen is.

**Which arms owe the matched-pair proof is derived, not listed.**  A trailing
pair is an arm whose recorded refit schedule has the control's inner
objectives and a swapped trailing one -- ``1,1,1,2`` against ``1,1,1,1`` --
and which ran no coupled landing, because #50's coupled landing re-assigns
blocks and is *expected* to move the codes the next trellis pass sees.  Naming
``B-Jac``/``B-GS`` instead would pass on the day the roster is wrong
(AGENTS.md rule 3), and would let a receipt exempt an arm by deleting its
proof.

``plane_moved`` is recorded and deliberately **not** required: an arm whose
lever reached nothing is an ineffective arm, which is a result, not a broken
comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.encode import LUT_LANDING_WIRE  # noqa: E402
from tessera.errors import PromotionRefusedError  # noqa: E402

#: The control arm's name begins with this; the producer runs it first and
#: again last in one process and compares the two reconstructions.
CONTROL_FIRST_PREFIX = "A drift control FIRST"

#: An off-wire row carries its landing in its own arm name and is labelled
#: ``[NOT A WIRE]``; it is never a promotion input, so it is not validated as
#: one.
OFF_WIRE_MARK = "landing="

#: The proof the control arm owes, per unit.
CONTROL_LEG = "drift_control_identical"

#: The proofs every wire arm owes, per unit.
WIRE_LEGS = ("serialisable", "sink_vs_wire_bit_identical")

#: The matched-pair legs that are invariants of the pair.  ``plane_moved`` is
#: not among them on purpose (see the module docstring).
MATCHED_PAIR_LEGS = ("codes_identical", "bytes_equal",
                     "inner_objectives_equal", "inner_refits_identical")

MATCHED_PAIR_FIELD = "matched_pair"


def wire_arms(unit_record: dict) -> "list[str]":
    """The arms of one unit that were scored at the wire landing."""
    return [arm for arm in unit_record if OFF_WIRE_MARK not in arm]


def control_arm(unit_record: dict, *, where: str, unit: str) -> str:
    """The one FIRST drift control of a unit's record, refusing any other count."""
    hits = [arm for arm in wire_arms(unit_record)
            if arm.startswith(CONTROL_FIRST_PREFIX)]
    if len(hits) != 1:
        raise PromotionRefusedError(
            f"{where}: unit {unit!r} carries {len(hits)} arms named "
            f"{CONTROL_FIRST_PREFIX!r} ({hits!r}); the screen's ratios are "
            "ratios against exactly one drift control"
        )
    return hits[0]


def _schedule_objectives(record: dict) -> "list[int] | None":
    """The metric dimension of each refit call, in order, or None if unrecorded.

    ``schedule`` is ``[(metric_ndim, gauss_seidel), ...]``.  Only the objective
    is read here: the sweep flag is handed to the inner 1-D passes as well,
    where it is provably the parallel step, so the flag differs between two
    arms that ran identical inner refits.
    """
    schedule = record.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        return None
    try:
        return [int(step[0]) for step in schedule]
    except (TypeError, ValueError, IndexError):
        return None


def _ran_coupled_landing(record: dict) -> bool:
    """True when any refit call re-assigned blocks under #50's coupled landing.

    A coupled landing is expected to move the codes the NEXT trellis pass sees,
    so a coupled arm is not a codes-identical trailing pair and owes no
    matched-pair proof.  Read off the diagnostics the refit itself wrote, not
    off the arm's name.
    """
    return any(isinstance(step, dict) and "coupled" in step
               for step in record.get("refit", []) or [])


def is_trailing_pair(record: dict, control: dict) -> bool:
    """Is this arm the control's schedule with only the trailing objective swapped?"""
    mine, base = _schedule_objectives(record), _schedule_objectives(control)
    if mine is None or base is None or len(mine) != len(base):
        return False
    return (mine[:-1] == base[:-1] and mine[-1] != base[-1]
            and not _ran_coupled_landing(record))


def _leg_true(record: dict, leg: str) -> "str | None":
    """None when the leg is recorded and true; otherwise why it is not."""
    if leg not in record:
        return f"{leg} is absent -- a proof that was not recorded is not a proof that passed"
    if record[leg] is not True:
        return f"{leg}={record[leg]!r}"
    return None


def assert_screen_receipt(document: dict, *, name: str, where: str) -> "dict[str, tuple[str, ...]]":
    """Refuse a screen document whose own recorded proofs do not hold.

    Raises :class:`tessera.errors.PromotionRefusedError`, naming the document,
    the unit and the field, for the legs that invalidate the whole screen: the
    drift control and the wire/landing identity.  Returns the per-arm
    matched-pair failures, which invalidate one arm's claim rather than the
    document -- keyed by arm name, one entry per arm that failed at least one
    leg, so the caller refuses that arm and reads the others.
    """
    units = document.get("units")
    if not isinstance(units, dict) or not units:
        raise PromotionRefusedError(
            f"{where}: {name} carries no 'units' record; there is no screen to read")
    arm_failures: "dict[str, list[str]]" = {}
    roster: "tuple[str, ...] | None" = None
    for unit, record in units.items():
        if not isinstance(record, dict) or not record:
            raise PromotionRefusedError(
                f"{where}: {name} unit {unit!r} carries no arms")
        control = control_arm(record, where=where, unit=unit)
        arms = tuple(sorted(wire_arms(record)))
        if roster is None:
            roster = arms
        elif arms != roster:
            raise PromotionRefusedError(
                f"{where}: {name} unit {unit!r} was measured on wire arms "
                f"{list(arms)!r}, not the {list(roster)!r} the first unit "
                "carries; a geomean over two arm sets is not a ratio")
        for arm in arms:
            arm_record = record[arm]
            landing = arm_record.get("landing")
            if landing != LUT_LANDING_WIRE:
                raise PromotionRefusedError(
                    f"{where}: {name} unit {unit!r} arm {arm!r} records "
                    f"landing={landing!r}, not {LUT_LANDING_WIRE!r} -- an "
                    "off-wire row is not named as one and only the wire "
                    "promotes (tessera#85)")
            for leg in WIRE_LEGS:
                bad = _leg_true(arm_record, leg)
                if bad is not None:
                    raise PromotionRefusedError(
                        f"{where}: {name} unit {unit!r} arm {arm!r}: {bad}.  "
                        "The sink these arms were scored off is not the wire "
                        "that ships, so the numbers are not the shipped "
                        "object's")
            if arm == control:
                bad = _leg_true(arm_record, CONTROL_LEG)
                if bad is not None:
                    raise PromotionRefusedError(
                        f"{where}: {name} unit {unit!r} drift control: {bad}.  "
                        "The FIRST and LAST runs of the control did not "
                        "reconstruct the same weights, so an arm-to-arm gap "
                        "in this document is not attributable to the arm")
                continue
            if not is_trailing_pair(arm_record, record[control]):
                continue
            pair = arm_record.get(MATCHED_PAIR_FIELD)
            if not isinstance(pair, dict):
                arm_failures.setdefault(arm, []).append(
                    f"{name} unit {unit!r}: {MATCHED_PAIR_FIELD} is absent, and "
                    "this arm's schedule says it is the control's with the "
                    "trailing objective swapped -- the pair it claims is "
                    "unproven")
                continue
            for leg in MATCHED_PAIR_LEGS:
                bad = _leg_true(pair, leg)
                if bad is not None:
                    arm_failures.setdefault(arm, []).append(
                        f"{name} unit {unit!r}: {MATCHED_PAIR_FIELD}.{bad}")
    return {arm: tuple(reasons) for arm, reasons in arm_failures.items()}


def assert_arm_proofs(arm: str, failures: "tuple[str, ...]", *, where: str) -> None:
    """Refuse one arm whose recorded matched-pair proofs do not hold."""
    if not failures:
        return
    raise PromotionRefusedError(
        f"{where}: {arm} does not carry the matched pair it is measured as -- "
        + "; ".join(failures)
        + ".  A trailing refit that moved a code or a blob length is comparing "
        "two encodings, not two scale planes"
    )
