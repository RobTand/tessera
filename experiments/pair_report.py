"""Read a finished ``pair-*`` sweep the way its receipt has to read it.

The sweep already writes a per-rung comparison. This does **not** trust it.
Every ratio here is rebuilt from the run's **raw arms**: each candidate is
matched to the shipped-pair arm (the file's own ``default_L``, ratio 1.0)
whose ``bpp`` equals its own to 1e-9, and that match is asserted rather than
assumed. A number below
1.0 is therefore that candidate beating the shipped pair at bytes the two
provably share, established from the encodes themselves.

The reason for the second derivation is #93: a stored comparison is exactly
the thing that was wrong there -- a broken reference lookup dropped half the
grid, and the summary derived from it named the wrong arm by 28% on the gate
metric. A reader that re-derives from the arms cannot inherit that class of
bug; a reader that re-implements the *completeness rule* can, and does, which
is why the rule is imported rather than restated (#97).

``pair_grid_audit`` is ``experiments/pair_grid_audit.py``, which the sweep
stage also calls: one completeness rule and one label grammar, in one place,
so a writer and a reader cannot come to disagree about which cells exist.

Scope this printer does not know and the caller must state: which grid, which
axis (tracking vs ratio), how many units, and whether the metric is weight
space or served.
"""
from __future__ import annotations

import json
import math
import sys

from pair_grid_audit import audit_doc, audit_lines, cell_label, control_line

#: The width the shipped recipe carried *when this reader was written*. It is
#: a fallback and a cross-check, never the value used: every unit records its
#: own ``default_L``, and reading the constant instead is how a reader keeps
#: scoring against ``L=14`` after the recipe moves -- silently, because a
#: candidate at the shipped width would then find no reference and the arm
#: would simply vanish from the table.
DEFAULT_L_WHEN_WRITTEN = 14
AXES = ("wt", "h", "out")


def default_L(doc) -> int:
    """The shipped width this run byte-matched against, from the run.

    One value for the whole document: a file whose units disagree is not a
    grid with a reference, and summarising it would compare arms scored
    against different recipes.
    """
    seen = {res["default_L"] for res in doc["units"].values()
            if "default_L" in res}
    if len(seen) != 1:
        raise SystemExit(
            f"pair_report: units declare {sorted(seen) or 'no'} default_L; a "
            "byte-matched report needs exactly one shipped width")
    return int(next(iter(seen)))


def requested_units(doc):
    """The unit set the run was *asked* for, or ``None`` if it never said.

    A pair sweep writes its units as they finish, into one fixed path with no
    per-run identity, so a partial file is indistinguishable from a finished
    one by inspection -- and a re-run of the same sweep truncates the artifact
    it is reproducing. The denominator therefore has to come from ``args``,
    never from the units that arrived, for the same reason #93's cell count
    does: a count taken from the survivors reads ``N of N`` over any drop.
    """
    args = doc.get("args") or {}
    # The discriminator is the stage, not which keys are populated: the sweep
    # writes *every* selector into ``args`` with its default, so ``units``
    # holds the dense list even on a GLM run, and reading whichever key is
    # non-empty answers "4" to a six-expert grid.
    stage = args.get("stage")
    if stage == "pair-dense":
        return [str(u) for u in args.get("units") or []] or None
    if stage == "pair-glm":
        layers, projs = args.get("layers"), args.get("projs")
        experts = args.get("experts")
        if layers and projs and experts is not None:
            return [f"L{l}.e{e}.{proj}"
                    for l in layers for e in experts for proj in projs]
    return None


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def arms_of(res):
    """The encoded arms of one unit, minus the repeat control."""
    return {k: v for k, v in res.items()
            if isinstance(v, dict) and "L" in v and "[repeat]" not in k}


def refs_of(res, shipped_L):
    """Every shipped-pair arm of one unit, keyed by bpp.

    These are the byte-matched references: the recipe as it ships, run at
    whatever rung spends the candidate's bytes. Their ``rung`` is deliberately
    *not* the candidate's -- that shift is the byte match -- so keying by rung
    would break the design rather than guard it. The guard that does apply is
    ``same_rung_group``: a reference may be shifted by a table's worth of
    bytes and no more.
    """
    return {round(v["bpp"], 12): v for v in arms_of(res).values()
            if v["L"] == shipped_L and v["ratio"] == 1.0}


def same_rung_group(candidate, ref) -> bool:
    """True when a bpp match is the intended shift and not a collision.

    Byte matching runs the shipped recipe at a shifted rung, so the reference
    sits within one table's bytes of the candidate -- under 0.5 bpp on any
    width this grid carries. Rungs are 2 bpp apart, so anything further is a
    match across rung groups: two arms with equal bytes that are not the same
    comparison. Unreachable on the shipped grid, and exactly the assumption
    that should be asserted rather than believed.
    """
    return abs(candidate["rung"] - ref["rung"]) / 256.0 < 0.5


def report(path: str) -> bool:
    doc = json.load(open(path))
    gate, units = doc["gate"], doc["units"]
    audit = audit_doc(doc)
    shipped_L = default_L(doc)
    print(f"\n########## {path}")
    print(f"population={doc['population']}  grid={doc['grid']}  gate={gate}  "
          f"units={len(units)}  rows={doc['args'].get('eval_rows')}  "
          f"shipped L={shipped_L}"
          + ("" if shipped_L == DEFAULT_L_WHEN_WRITTEN
             else f" (not the {DEFAULT_L_WHEN_WRITTEN} this reader was "
                  "written against -- the file wins)"))
    print("units: " + ", ".join(sorted(units)))

    # The unit set, against what the run asked for. A sweep in flight looks
    # exactly like a finished one, and its numbers are a subset of a
    # population rather than the population.
    want = requested_units(doc)
    units_ok = True
    if want is None:
        print("unit set: args declares no unit set -- NOT checked")
    elif sorted(want) != sorted(units):
        units_ok = False
        missing = sorted(set(want) - set(units))
        extra = sorted(set(units) - set(want))
        print(f"UNIT SET INCOMPLETE: {len(units)} of {len(want)} requested "
              "units present -- this file is a partial run, or a re-run "
              "overwriting a finished one; the geomeans below are over the "
              "units that arrived, which is not the population asked for"
              + (f"\n  missing: {', '.join(missing)}" if missing else "")
              + (f"\n  unrequested: {', '.join(extra)}" if extra else ""))
    else:
        print(f"unit set: {len(units)} of {len(want)} requested units present")
    rungs = sorted({v["rung"] for res in units.values()
                    for v in arms_of(res).values()})

    # Controls: three outcomes, not two -- "did not run" is a gap in the
    # evidence and "did not agree" is evidence of a bug (#96).
    ok, drift, problems = 0, [], []
    for u in sorted(units):
        for q in rungs:
            st = audit["units"].get(u, {}).get(f"R{q}", {}).get("control")
            if st is None:
                problems.append(f"{u} R{q}: no audit entry")
                continue
            line = control_line(q, st)
            if line is None:
                ok += 1
                c = units[u][f"R{q}_control"]
                if c.get("secs_first"):
                    drift.append(c["secs_repeat"] / c["secs_first"])
            else:
                problems.append(f"{u}: {line}")
    print(f"\ncontrols: {ok} of {len(units) * len(rungs)} (unit, rung) repeats "
          "byte- AND tensor-identical"
          + ("".join(f"\n  {p}" for p in problems) if problems else ""))
    if drift:
        # The process's FIRST control pays Triton/torch compile on its first
        # encode, so its repeat/first ratio measures JIT warmup, not the box.
        # Quote it separately; the drift range is the rest.
        warm, rest = drift[0], drift[1:]
        print("  wall drift between the paired baselines: "
              + (f"{min(rest):.2f}x-{max(rest):.2f}x over {len(rest)} pairs"
                 if rest else "no pairs after warmup")
              + f"  (first control excluded at {warm:.2f}x -- that one is JIT "
                "warmup on the process's first encode, not box drift)")

    unmatched, grid_ok = [], True
    for q in rungs:
        print(f"\n== R={q / 256:g}  ({len(units)} units, gate={gate}, "
              f"vs byte-matched shipped pair L={shipped_L} r=1)")
        # The completeness rule lives in pair_grid_audit and is read, not
        # restated: a second copy is how a rule stops describing the same grid.
        # Say so when it holds, too -- a receipt that reports completeness only
        # by staying silent cannot be told from one that never checked.
        # An arm can carry a rung the audit's own rule never enumerated -- a
        # stray reference, a rung nobody asked for -- and that is a cell
        # outside the counted grid, which is #93 again. Name it; do not die of
        # a KeyError, and do not let it read as clean.
        rung_audits, uncounted = {}, []
        for u in sorted(units):
            a = audit["units"].get(u, {}).get(f"R{q}")
            if a is None:
                uncounted.append(u)
            else:
                rung_audits[u] = a
        for u in uncounted:
            print(f"  {u}: R{q} carries arms the completeness rule does not "
                  "enumerate -- this rung is outside the audited grid")
        clean = [a for a in rung_audits.values()
                 if a["complete"] and not a["unmatched"] and not a["unexpected"]]
        if uncounted:
            grid_ok = False
        if len(clean) == len(units) and not uncounted:
            a0 = next(iter(rung_audits.values()))
            print(f"  grid: all {len(units)} unit(s) carry {a0['expected']} of "
                  f"{a0['expected']} (L, ratio) cells, every one at its "
                  "reference's exact bpp")
        else:
            for u, a in rung_audits.items():
                for line in audit_lines(a):
                    print(f"  {u}: {line}")
        cells = {}
        for u in sorted(units):
            res, refs = units[u], refs_of(units[u], shipped_L)
            for k, v in arms_of(res).items():
                if v["rung"] != q or "[bytematch" in k:
                    continue
                ref = refs.get(round(v["bpp"], 12))
                if ref is None:
                    unmatched.append((u, k))
                    continue
                if not same_rung_group(v, ref):
                    unmatched.append((u, f"{k} matched across rung groups "
                                         f"(ref at R{ref['rung']})"))
                    continue
                cells.setdefault(cell_label(v["L"], v["ratio"]), {})[u] = (v, ref)
        print(f"{'arm':<12}{'wins':>6}{'>1%':>6}"
              + "".join(f"{ax:>9}" for ax in AXES)
              + f"{'bpp':>9}   per-unit {gate}")
        for lab in sorted(cells, key=lambda s: (int(s.split("L=")[1].split()[0]),
                                                float(s.split("r=")[1]))):
            got = cells[lab]
            if len(got) != len(units):
                print(f"{lab:<12}  incomplete: {len(got)} of {len(units)} units")
                continue
            per = {}
            for ax in AXES:
                vals = [v[ax] / r[ax] for v, r in got.values()
                        if ax in v and ax in r and r[ax] > 0]
                per[ax] = vals if len(vals) == len(units) else None
            g = per[gate]
            print(f"{lab:<12}{sum(1 for x in g if x < 1):>6}"
                  f"{sum(1 for x in g if x < 0.99):>6}"
                  + "".join(f"{geo(per[ax]):9.4f}" if per[ax] else f"{'-':>9}"
                            for ax in AXES)
                  + f"{next(iter(got.values()))[0]['bpp']:9.4f}   "
                  + " ".join(f"{x:.3f}" for x in g))
        # Separability: the two axes are only separate if the best ratio does
        # not depend on L and the best L does not depend on the ratio.
        by_L, by_r = {}, {}
        for lab, got in cells.items():
            if len(got) != len(units):
                continue
            L = int(lab.split("L=")[1].split()[0])
            r = float(lab.split("r=")[1])
            g = geo([v[gate] / rf[gate] for v, rf in got.values()])
            by_L.setdefault(L, []).append((g, r))
            by_r.setdefault(r, []).append((g, L))
        if by_L:
            print("  best ratio at each L: " + "  ".join(
                f"L{L}->r{min(v)[1]:g} ({min(v)[0]:.4f}x)"
                for L, v in sorted(by_L.items())))
            print("  best L at each ratio: " + "  ".join(
                f"r{r:g}->L{min(v)[1]} ({min(v)[0]:.4f}x)"
                for r, v in sorted(by_r.items())))
            print("  separable? ratio-argmin constant across L: "
                  f"{len({min(v)[1] for v in by_L.values()}) == 1}; "
                  "L-argmin constant across ratio: "
                  f"{len({min(v)[1] for v in by_r.values()}) == 1}")
            best = min((min(v)[0], L, min(v)[1]) for L, v in by_L.items())
            print(f"  best pair at this rung: L={best[1]} r={best[2]:g} "
                  f"at {best[0]:.4f}x")
        # Reach at the shipped width: the ratio's mechanism, in the units the
        # ratio is measured in.  Reach costs no bytes; only L does.
        for lab, got in sorted(cells.items()):
            if not lab.startswith(f"L={shipped_L} "):
                continue
            ov = [v["over"] for v, _ in got.values()]
            rr = [v["reach_rms"] for v, _ in got.values()]
            print(f"    {lab:<12} reach_rms {min(rr):.3f}-{max(rr):.3f}  "
                  f"rows over reach {min(ov):.4f}-{max(ov):.4f}")
    print("\nbyte match: " + ("every candidate found an exact-bpp reference"
                              if not unmatched else f"UNMATCHED {unmatched}"))
    return (audit["complete"] and audit["controls_ok"] and not unmatched
            and units_ok and grid_ok)


def main(argv=None) -> int:
    paths = list(argv if argv is not None else sys.argv[1:])
    if not paths:
        raise SystemExit("usage: pair_report.py <pair-*.json> [...]")
    ok = True
    for p in paths:
        ok &= report(p)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
