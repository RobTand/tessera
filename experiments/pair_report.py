"""Read a finished ``pair-*`` sweep the way its receipt has to read it.

The sweep already writes a per-rung comparison. This does **not** trust it.
Every ratio here is rebuilt from the run's **raw arms**: each candidate is
matched to the shipped-pair arm (``L=14``, ratio 1.0) whose ``bpp`` equals its
own to 1e-9, and that match is asserted rather than assumed. A number below
1.0 is therefore that candidate beating the shipped pair at bytes the two
provably share, established from the encodes themselves.

The reason for the second derivation is #93: a stored comparison is exactly
the thing that was wrong there -- a broken reference lookup dropped half the
grid, and the summary derived from it named the wrong arm by 28% on the gate
metric. A reader that re-derives from the arms cannot inherit that class of
bug; a reader that re-implements the *completeness rule* can, and does, which
is why the rule is imported rather than restated (#97).

Dependency, stated because it is not on this branch: ``pair_grid_audit`` is
``experiments/pair_grid_audit.py`` from ``muse/ts-93-griddrop``, last touched
there by ``8b5424a`` ("Read the rungs from args too, and say when a cell is
missing both ways"); it was added by ``b76a836``. This module owes that
branch's file and does not carry a copy of it -- one completeness rule, one
label grammar, one place.

Scope this printer does not know and the caller must state: which grid, which
axis (tracking vs ratio), how many units, and whether the metric is weight
space or served.
"""
from __future__ import annotations

import json
import math
import sys

from pair_grid_audit import audit_doc, audit_lines, cell_label, control_line

#: The width the shipped BF16/E4M3 recipe carries, and so the width whose
#: arms are the byte-matched references every candidate is scored against.
DEFAULT_L = 14
AXES = ("wt", "h", "out")


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def arms_of(res):
    """The encoded arms of one unit, minus the repeat control."""
    return {k: v for k, v in res.items()
            if isinstance(v, dict) and "L" in v and "[repeat]" not in k}


def refs_of(res):
    """Every shipped-pair arm of one unit, keyed by bpp.

    These are the byte-matched references: the recipe as it ships, run at
    whatever rung spends the candidate's bytes.
    """
    return {round(v["bpp"], 12): v for v in arms_of(res).values()
            if v["L"] == DEFAULT_L and v["ratio"] == 1.0}


def report(path: str) -> bool:
    doc = json.load(open(path))
    gate, units = doc["gate"], doc["units"]
    audit = audit_doc(doc)
    print(f"\n########## {path}")
    print(f"population={doc['population']}  grid={doc['grid']}  gate={gate}  "
          f"units={len(units)}  rows={doc['args'].get('eval_rows')}")
    print("units: " + ", ".join(sorted(units)))
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

    unmatched = []
    for q in rungs:
        print(f"\n== R={q / 256:g}  ({len(units)} units, gate={gate}, "
              "vs byte-matched shipped pair L=14 r=1)")
        # The completeness rule lives in pair_grid_audit and is read, not
        # restated: a second copy is how a rule stops describing the same grid.
        # Say so when it holds, too -- a receipt that reports completeness only
        # by staying silent cannot be told from one that never checked.
        rung_audits = {u: audit["units"][u][f"R{q}"] for u in sorted(units)}
        clean = [a for a in rung_audits.values()
                 if a["complete"] and not a["unmatched"] and not a["unexpected"]]
        if len(clean) == len(units):
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
            res, refs = units[u], refs_of(units[u])
            for k, v in arms_of(res).items():
                if v["rung"] != q or "[bytematch" in k:
                    continue
                ref = refs.get(round(v["bpp"], 12))
                if ref is None:
                    unmatched.append((u, k))
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
            if not lab.startswith(f"L={DEFAULT_L} "):
                continue
            ov = [v["over"] for v, _ in got.values()]
            rr = [v["reach_rms"] for v, _ in got.values()]
            print(f"    {lab:<12} reach_rms {min(rr):.3f}-{max(rr):.3f}  "
                  f"rows over reach {min(ov):.4f}-{max(ov):.4f}")
    print("\nbyte match: " + ("every candidate found an exact-bpp reference"
                              if not unmatched else f"UNMATCHED {unmatched}"))
    return audit["complete"] and audit["controls_ok"] and not unmatched


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
