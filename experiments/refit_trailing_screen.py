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

**The evidence that derives it is established before the obligations are.**
``schedule`` and the refit's coupled-landing diagnostics decide which arm owes
what, so they are evidence exactly as the proofs are, and a receipt that does
not record them has not shown an arm to be exempt -- it has failed to say what
the arm is.  Reading them with a ``.get()`` and answering "not a trailing
pair" made *deleting one field* a way past every matched-pair check, over an
explicitly recorded ``codes_identical: false`` (tessera#299).  Unknown is
refused by name: the candidate's, on the arm that cannot be classified; the
control's, on the whole document, because the control's schedule is the
baseline every other arm in the unit is classified against and losing it
exempts all of them at once, exactly as a failed drift control does.

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

#: The two recordings that classify an arm -- its refit schedule, and the
#: per-call diagnostics that say whether a coupled landing ran.  Neither is a
#: proof the arm owes; both are the evidence that decides which proofs it owes,
#: and an arm missing either is UNCLASSIFIED rather than exempt (tessera#299).
SCHEDULE_FIELD = "schedule"
REFIT_FIELD = "refit"


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


def schedule_objectives(record: dict) -> "list[int] | str":
    """The metric dimension of each refit call, in order, or why it is unknown.

    ``schedule`` is ``[(metric_ndim, gauss_seidel), ...]``.  Only the objective
    is read here: the sweep flag is handed to the inner 1-D passes as well,
    where it is provably the parallel step, so the flag differs between two
    arms that ran identical inner refits.

    Returns the objectives, or a **string** naming what the receipt does not
    record.  Never ``None``-as-"nothing to check": an unreadable schedule used
    to answer "not a trailing pair", which exempted the arm from every
    matched-pair proof it had recorded -- including a FAILED one (tessera#299).
    """
    if SCHEDULE_FIELD not in record:
        return (f"{SCHEDULE_FIELD} is absent, so which comparison this arm is "
                "cannot be established")
    schedule = record[SCHEDULE_FIELD]
    if not isinstance(schedule, list) or not schedule:
        return (f"{SCHEDULE_FIELD}={schedule!r} records no refit call, so "
                "which comparison this arm is cannot be established")
    try:
        return [int(step[0]) for step in schedule]
    except (TypeError, ValueError, IndexError, KeyError):
        return (f"{SCHEDULE_FIELD}={schedule!r} does not read as "
                "[(metric_ndim, gauss_seidel), ...], so which comparison this "
                "arm is cannot be established")


def ran_coupled_landing(record: dict) -> "bool | str":
    """Did any refit call re-assign blocks under #50's coupled landing?

    A coupled landing is expected to move the codes the NEXT trellis pass sees,
    so a coupled arm is not a codes-identical trailing pair and owes no
    matched-pair proof.  Read off the diagnostics the refit itself wrote, not
    off the arm's name -- and when there are none, the answer is the reason
    string and not ``False``: an arm that recorded no diagnostics has not shown
    that no coupled landing ran (tessera#299).
    """
    steps = record.get(REFIT_FIELD)
    if not isinstance(steps, list) or not steps:
        return (f"{REFIT_FIELD}={steps!r} records no per-call diagnostics, so "
                "whether a coupled landing re-assigned blocks is unknown")
    if not all(isinstance(step, dict) for step in steps):
        return (f"{REFIT_FIELD}={steps!r} does not read as one diagnostics "
                "record per refit call, so whether a coupled landing "
                "re-assigned blocks is unknown")
    return any("coupled" in step for step in steps)


def classify_trailing_pair(record: dict, control: dict) -> "tuple[bool, str | None]":
    """``(owes the matched-pair proof, why it could not be classified)``.

    A trailing pair is an arm whose recorded schedule carries the control's
    inner objectives with the trailing one swapped and which ran no coupled
    landing.  When the evidence that decides that is missing the second element
    says which recording is absent and the first is ``False`` meaning
    **unknown** -- never *exempt*.  Callers refuse an unclassified arm; they do
    not read it as owing nothing (tessera#299).
    """
    mine = schedule_objectives(record)
    if isinstance(mine, str):
        return False, mine
    base = schedule_objectives(control)
    if isinstance(base, str):
        return False, f"the drift control's {base}"
    if len(mine) != len(base) or mine[:-1] != base[:-1] or mine[-1] == base[-1]:
        return False, None
    coupled = ran_coupled_landing(record)
    if isinstance(coupled, str):
        return False, coupled
    return not coupled, None


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
    drift control (its proof AND its schedule, the baseline every other arm is
    classified against) and the wire/landing identity.  Returns the per-arm
    matched-pair failures, which invalidate one arm's claim rather than the
    document -- keyed by arm name, one entry per arm that failed at least one
    leg, so the caller refuses that arm and reads the others.  An arm whose own
    classification evidence cannot be read is one of those failures: unknown is
    refused, never exempted (tessera#299).
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
        # The control's schedule is the baseline every other arm is classified
        # against, so a control that does not record one exempts the whole unit
        # at a stroke.  That is a broken document, not one arm's failed claim
        # (tessera#299).
        baseline = schedule_objectives(record[control])
        if isinstance(baseline, str):
            raise PromotionRefusedError(
                f"{where}: {name} unit {unit!r} drift control {control!r}: "
                f"{baseline}.  Which arms owe the matched-pair proof is derived "
                "from the control's own schedule, so a control that records "
                "none exempts every arm in the unit from every pair check")
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
            # Establish what this arm IS before deciding what it owes.  An arm
            # the receipt cannot classify is refused by name; it is not read as
            # an arm with no obligations, and any matched-pair leg it did
            # record is still read, so a recorded failure cannot be cleared by
            # deleting the field that classifies the comparison (tessera#299).
            owes_pair, unknown = classify_trailing_pair(arm_record, record[control])
            if unknown is not None:
                arm_failures.setdefault(arm, []).append(
                    f"{name} unit {unit!r}: {unknown}; an arm that cannot be "
                    "told from the control's trailing pair is not an arm that "
                    "owes no matched-pair proof")
            elif not owes_pair:
                continue
            pair = arm_record.get(MATCHED_PAIR_FIELD)
            if not isinstance(pair, dict):
                if unknown is None:
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
