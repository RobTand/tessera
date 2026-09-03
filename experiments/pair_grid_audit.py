"""The pair sweep's grid, and what it means for a cell to be absent.

Issue #93.  The ``pair-dense``/``pair-glm`` stage of
``bf16_l_sigma_sweep.py`` compares each ``(L, ratio)`` candidate against a
byte-matched reference, and skips a cell whose reference it could not build
or whose encode failed.  Skipping is correct; doing it *silently* was not.
The stage's own audit line read

    byte match: N of M arms sit at their reference's exact bpp

with ``M`` counted from the comparison it had just built -- that is, from the
survivors.  A denominator computed from the survivors cannot report a drop,
and over a broken reference lookup it printed a clean ``2 of 2`` above a
table missing half its rows.  The stored ``best on <gate>`` line then named
the wrong arm, by 28% on the gate metric.

So the denominator moves here, where it is derived from the grid the run
**asked for** rather than from the rows it managed to fill.  The invariant,
stated once and read by both the writer and the reader:

    at every rung ``q256``, the number of cells the stage owes is
    ``|{L in pair_bits : L * 256 >= q256}| x |pair_ratios|``

-- the ``L * 256 >= q256`` filter being the stage's own legality rule (a
table wider than the rung's whole budget is not a candidate).  On a
``{12, 14, 16} x {1.0, 1.25, 1.4142, 1.75}`` grid a trustworthy file has 12
entries per rung and the broken one had 4.

This module is deliberately torch-free and depends on nothing in ``src/``:
it is the one place the label grammar and the completeness rule live, so the
sweep that writes a file and the reader that checks one cannot drift apart,
and the rule can be tested in an interpreter that has no GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Why a cell is not in the comparison.  Ordered from "the stage could never
#: have run it" to "it ran and disagreed", because that is the order in which
#: a reader should stop worrying about the file.
NO_BYTE_MATCH = "no integral byte-matched rung on this shape"
REFERENCE_MISSING = "the byte-matched reference arm was not encoded"
CANDIDATE_MISSING = "the candidate arm was not encoded"
#: Not a reason a cell is absent -- the cell is present and wrong -- so it
#: rides in ``unmatched`` rather than ``missing``.  Named here anyway because
#: the two failures are read together and a reader should not have to invent
#: the wording for one of them.
BYTES_UNMATCHED = "encoded, but not at its reference's exact bpp"

#: Why a rung has no cross-arm contamination check (#96).  The repeat of the
#: shipped arm is the stage's only evidence that no arm leaked state into a
#: later one, and it is the arm most likely to die -- it runs last, after
#: every wide table has churned the allocator.  It used to be recorded only
#: ``if last in res``, so a rung whose repeat failed carried no ``_control``
#: key at all and read exactly like a rung whose control passed.
CONTROL_REPEAT_MISSING = "the repeat arm was not encoded, so nothing checked "\
                         "this rung for cross-arm contamination"
CONTROL_BASELINE_MISSING = "the first arm was not encoded, so the repeat had "\
                           "nothing to be compared against"
CONTROL_ABSENT = "no control was recorded at all -- a file written before #96, "\
                 "or a rung whose repeat vanished"


def control_status(entry: "dict | None") -> dict:
    """Did this rung's repeat control run, and did it agree?

    Three outcomes, not two.  Folding "did not run" into "did not agree"
    would be as wrong as folding it into "agreed": the first is a gap in the
    evidence and the second is evidence of a bug, and only the second is a
    reason to distrust the numbers rather than to re-run.
    """
    if entry is None:
        return {"ran": False, "passed": None, "reason": CONTROL_ABSENT}
    if entry.get("ran") is False:
        return {"ran": False, "passed": None,
                "reason": entry.get("reason", CONTROL_ABSENT)}
    passed = bool(entry.get("bytes_identical")) and bool(entry.get("tensor_identical"))
    return {"ran": True, "passed": passed,
            "reason": None if passed else
            "the repeat is not byte- and tensor-identical to the first arm"}


def control_line(q256: int, status: dict) -> "str | None":
    """The one sentence a rung's control is worth.  ``None`` when it passed."""
    if status["ran"] and status["passed"]:
        return None
    if not status["ran"]:
        return (f"CONTROL MISSING at R{q256}: {status['reason']}; this rung's "
                "arms are unchecked for contamination and read exactly like "
                "checked ones (#96)")
    return f"CONTROL FAILED at R{q256}: {status['reason']}"


def pair_arm_key(q256: int, window_bits: int, ratio: float) -> str:
    """The key an arm is stored under.  One spelling, shared by all readers."""
    return f"R{q256} L={window_bits} r={ratio:g}"


def cell_label(window_bits: int, ratio: float) -> str:
    """A cell of the ``(L, ratio)`` grid, rung-free."""
    return f"L={window_bits} r={ratio:g}"


def legal_widths(q256: int, pair_bits) -> list:
    """The widths this rung can carry: a table wider than the budget is not one."""
    return [int(L) for L in pair_bits if int(L) * 256 >= int(q256)]


def requested_cells(q256: int, pair_bits, pair_ratios) -> list:
    """Every ``(L, ratio)`` cell the grid owes at this rung, in table order."""
    return [(L, float(r)) for L in legal_widths(q256, pair_bits)
            for r in pair_ratios]


def audit_rung(q256: int, pair_bits, pair_ratios, comparison: dict,
               *, reasons: "dict | None" = None) -> dict:
    """Score one rung's comparison against the grid the run asked for.

    ``comparison`` is the stage's ``R{q}_vs_shipped`` mapping, keyed by arm.
    ``reasons`` optionally carries a per-label explanation the caller already
    knows (the stage does; a reader of a finished file does not, and gets
    ``CANDIDATE_MISSING`` as the honest "it is not here and I cannot say
    why").  Nothing here raises: the caller decides whether an incomplete
    grid is fatal, and this returns the facts it needs to decide.
    """
    reasons = dict(reasons or {})
    want = requested_cells(q256, pair_bits, pair_ratios)
    present, missing, unmatched = [], {}, []
    for L, ratio in want:
        label = cell_label(L, ratio)
        entry = comparison.get(pair_arm_key(q256, L, ratio))
        if entry is None:
            missing[label] = reasons.get(label, CANDIDATE_MISSING)
            continue
        present.append(label)
        if not entry.get("bytes_matched", False):
            unmatched.append(label)
    # A key in the comparison that the grid never asked for is its own bug --
    # a label-grammar drift between writer and reader looks exactly like this.
    extra = sorted(set(comparison) - {pair_arm_key(q256, L, r) for L, r in want})
    return {
        "q256": int(q256),
        "expected": len(want),
        "present": len(present),
        "complete": len(missing) == 0,
        "missing": dict(sorted(missing.items())),
        "unmatched": sorted(unmatched),
        "unexpected": extra,
    }


def audit_lines(audit: dict) -> list:
    """The log the stage prints and the reader prints -- the same sentences."""
    q = audit["q256"]
    out = [f"byte match: {audit['present'] - len(audit['unmatched'])} of "
           f"{audit['expected']} arms the grid asked for sit at their "
           f"reference's exact bpp"
           + (f"; UNMATCHED {audit['unmatched']}" if audit["unmatched"] else "")]
    if audit["missing"]:
        out.append(f"GRID INCOMPLETE at R{q}: {len(audit['missing'])} of "
                   f"{audit['expected']} cells absent")
        out += [f"  absent: {label} -- {why}"
                for label, why in audit["missing"].items()]
    if audit["unexpected"]:
        out.append(f"GRID UNEXPECTED at R{q}: {audit['unexpected']} -- "
                   "keys no cell of the requested grid names")
    return out


# ------------------------------------------------------------------ reader
#
# The same check, run against a finished ``pair-*`` JSON by someone who did
# not launch it.  It reads the grid out of the file's own ``args`` record, so
# a file cannot pass by having been asked for less than it claims.

def audit_doc(doc: dict) -> dict:
    """Audit every (unit, rung) cell of a loaded ``pair-*`` document."""
    args = doc.get("args") or {}
    for field in ("pair_bits", "pair_ratios", "rungs"):
        if field not in args:
            raise SystemExit(
                f"this file records no args[{field!r}], so the grid it asked "
                "for is unknown and its completeness cannot be checked")
    units = doc.get("units") or {}
    out = {"complete": True, "controls_ok": True, "units": {}}
    for unit in sorted(units):
        res = units[unit]
        per = {}
        for q in args["rungs"]:
            cmp_ = res.get(f"R{int(q)}_vs_shipped")
            if cmp_ is None:
                per[f"R{int(q)}"] = {
                    "q256": int(q), "expected": len(requested_cells(
                        q, args["pair_bits"], args["pair_ratios"])),
                    "present": 0, "complete": False,
                    "missing": {cell_label(L, r): "the rung has no comparison "
                                                  "block at all"
                                for L, r in requested_cells(
                                    q, args["pair_bits"], args["pair_ratios"])},
                    "unmatched": [], "unexpected": [],
                }
            else:
                # A file written by a fixed stage carries the reasons it knew
                # at the time; prefer them over this reader's guess.
                stored = (res.get(f"R{int(q)}_grid_audit") or {}).get("missing")
                per[f"R{int(q)}"] = audit_rung(
                    q, args["pair_bits"], args["pair_ratios"], cmp_,
                    reasons=stored)
            per[f"R{int(q)}"]["control"] = control_status(
                res.get(f"R{int(q)}_control"))
            out["complete"] &= per[f"R{int(q)}"]["complete"]
            out["controls_ok"] = out.get("controls_ok", True) and bool(
                per[f"R{int(q)}"]["control"]["passed"])
        out["units"][unit] = per
    return out


def report(path: str) -> bool:
    doc = json.load(open(path))
    result = audit_doc(doc)
    print(f"\n########## {path}")
    args = doc["args"]
    print(f"grid asked for: L={list(args['pair_bits'])} x "
          f"ratios={[round(float(r), 4) for r in args['pair_ratios']]} at "
          f"rungs={list(args['rungs'])}; {len(result['units'])} unit(s)")
    for unit, per in result["units"].items():
        for rung, audit in per.items():
            head = f"  {unit} {rung}: "
            ctl = control_line(audit["q256"], audit["control"])
            if audit["complete"] and not audit["unmatched"] and ctl is None:
                print(head + f"{audit['present']} of {audit['expected']} cells "
                             "present, all byte-matched, control passed")
                continue
            print(head.rstrip())
            for line in audit_lines(audit) + ([ctl] if ctl else []):
                print(f"    {line}")
    print("\nVERDICT: " + ("every rung carries the grid its own args record"
                           if result["complete"] else
                           "GRID INCOMPLETE -- this file's summary is derived "
                           "from a subset of the grid and cannot be read as a "
                           "comparison over it (#93)")
          + ("" if result["controls_ok"] else
             "; and at least one rung has no passing repeat control, so its "
             "arms are not known to be free of cross-arm contamination (#96)"))
    return result["complete"] and result["controls_ok"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help="pair-* JSON files to check")
    a = ap.parse_args(argv)
    ok = True
    for p in a.paths:
        ok &= report(str(Path(p)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
