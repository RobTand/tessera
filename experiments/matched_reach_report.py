"""Read the matched-reach rows as one factorial, and refuse a bundle for a split.

Issue #18.  The rows written by ``bf16_matched_reach_run.sh`` are separate
processes -- one window width each -- so the thing a reader must establish
before it prints anything is that they are the same experiment: the same
units, the same rungs, and the same shipped baseline down to the byte.  The
sweep's own control is an in-process repeat, which cannot see across runs;
this is the cross-run half, and it is a refusal rather than a warning.

What it then prints is a 2-D table the one-dimensional ``L`` sweeps could
not produce:

* a **row** is a table entry count.  It costs bytes, and every cell in it is
  scored against the shipped pair built at the rung that spends those same
  bytes -- the sweep does that matching and asserts it to 1e-9.
* a **column** is a realised reach, keyed off the built table rather than
  off the requested spread, because the request is snapped.  Moving along a
  row costs nothing: an artifact is the same size at every spread.

So the row marginal is the price of entries at a fixed reach, the column
marginal is the value of reach at a fixed entry count, and the diagonal is
the shipped recipe at each width.  ``recovered`` is the fraction of an
L-arm's log-effect that the shipped width reproduces at that arm's reach
and zero extra bytes -- the number the run's header registers a threshold
for before the numbers exist.

Scope this printer does not know and the caller must state: which grid,
weight space or served, and how many units stand behind each geomean.  It
prints the counts it can see and nothing about what they mean.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pair_grid_audit import control_status, pair_arm_key  # noqa: E402

AXES = ("wt", "h", "out")


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def baseline_arms(doc) -> dict:
    """Every shipped-pair arm in a document, keyed by ``(unit, rung)``.

    The shipped pair is the file's own ``default_L`` at ratio 1.0, which is
    the arm every run repeats first and last.  Two runs of the same units
    must agree on it exactly; the encoder is deterministic, so a difference
    is a difference in the code or the input, never noise.
    """
    L = {res["default_L"] for res in doc["units"].values() if "default_L" in res}
    if len(L) != 1:
        raise SystemExit("matched_reach_report: a file whose units disagree "
                         "about the shipped width is not one experiment")
    L = next(iter(L))
    out = {}
    for unit, res in doc["units"].items():
        for key, arm in res.items():
            if not isinstance(arm, dict) or "sha" not in arm:
                continue
            if arm.get("L") == L and arm.get("ratio") == 1.0 \
                    and arm.get("q256") == arm.get("rung"):
                out[(unit, int(arm["rung"]))] = arm
    return out


def cross_run_control(docs: "dict[str, dict]") -> dict:
    """The shipped baseline, compared across every file that carries it.

    Byte identity AND tensor identity, the same pair the in-process repeat
    asserts, so a run that shares a name with another and not a codebase is
    caught here rather than in a table of ratios.
    """
    base = {name: baseline_arms(doc) for name, doc in docs.items()}
    names = sorted(base)
    shared, bad = 0, []
    for i, a in enumerate(names):
        for bname in names[i + 1:]:
            common = set(base[a]) & set(base[bname])
            for key in sorted(common):
                shared += 1
                x, y = base[a][key], base[bname][key]
                if x["sha"] != y["sha"] or x["tsha"] != y["tsha"]:
                    bad.append((a, bname, key, x["sha"][:8], y["sha"][:8]))
    return {"pairs": shared, "mismatched": bad, "files": names}


def cells(doc) -> dict:
    """``(unit, rung, L, reach) -> {gate ratios, reach, over}`` from raw arms.

    Rebuilt from the arms and their recorded comparison, so a cell exists
    here only if the sweep found it a reference it is provably the same size
    as.
    """
    out = {}
    for unit, res in doc["units"].items():
        for rung_key, cmp in [(k, v) for k, v in res.items()
                              if k.endswith("_vs_shipped")]:
            rung = int(rung_key[1:].split("_")[0])
            for arm_key, c in cmp.items():
                arm = res[arm_key]
                cell = {"unit": unit, "rung": rung, "L": int(arm["L"]),
                        "ratio": float(arm["ratio"]),
                        "reach": float(arm["reach_grid_units"]),
                        "over": float(arm["rows_over_reach"]),
                        "bpp": float(arm["bpp"]), "ref": c["ref"],
                        "bytes_matched": bool(c["bytes_matched"])}
                for ax in AXES:
                    if f"{ax}_ratio" in c:
                        cell[ax] = float(c[f"{ax}_ratio"])
                out[(unit, rung, int(arm["L"]), cell["reach"])] = cell
    return out



#: The conditioning floor the run script registers before any number exists:
#: ``f = log B / log A`` is dominated by its third digit wherever the
#: byte-matched L arm barely moves, so the verdict is READ only where that arm
#: moves its gate by at least 1%.  It is a reporting convention, not a noise
#: floor -- the encoder is deterministic and the repeat control is byte
#: identity, so there is no noise here to clear.  ``f`` is printed either way;
#: what the floor gates is the sentence, not the number.
LOG_A_FLOOR = math.log(1.01)


def read_split(A: float, B: float) -> str:
    """The verdict the run script's header registered, applied to one cell."""

    if A <= 0 or B <= 0:
        return "unreadable (a non-positive ratio)"
    if abs(math.log(A)) < LOG_A_FLOOR:
        return (f"NOT READ: the L arm moves {abs(A - 1) * 100:.2f}% < 1%, "
                "below the registered conditioning floor")
    f = math.log(B) / math.log(A)
    if f < 0:
        # The two axes moved the gate in OPPOSITE directions, so there is no
        # fraction of the L effect for the spread to have recovered: a
        # negative f is not "under 15% recovered", it is a different fact.
        # The header registers one orientation of it (`B > 1 > A`, the L win
        # that the spread alone reverses); this is that test written as the
        # sign it actually is, so the mirror -- an L arm that HURTS while the
        # spread alone helps -- is not silently filed as "entry count".
        worse, better = ("the spread", "more entries") if A < 1 else (
            "more entries", "the spread")
        return (f"OPPOSITE SIGNS: {better} helps and {worse} hurts, so no "
                f"fraction of one is recoverable from the other (f={f:+.2f})")
    if f >= 0.5:
        return f"majority SPREAD ({f:.0%} of the L win), and spread is free"
    if f <= 0.15:
        return f"ENTRY COUNT ({f:.0%} recovered by spread); it costs bytes"
    return f"BOTH halves are real ({f:.0%} spread)"


def geo(values) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def report(paths, *, gate=None) -> None:
    docs = {Path(p).name: load(p) for p in paths}
    ctl = cross_run_control(docs)
    print("########## matched-reach factorial")
    print("files: " + ", ".join(ctl["files"]))
    pops = {d["population"] for d in docs.values()}
    gates = {d["gate"] for d in docs.values()}
    if len(pops) != 1:
        raise SystemExit(f"matched_reach_report: {sorted(pops)} in one report; "
                         "two populations are two experiments")
    gate = gate or (next(iter(gates)) if len(gates) == 1 else None)
    if gate is None:
        raise SystemExit(f"matched_reach_report: files disagree on the gate "
                         f"({sorted(gates)}); name one with --gate")
    print(f"population={next(iter(pops))}  gate={gate}")

    if ctl["mismatched"]:
        for row in ctl["mismatched"]:
            print(f"  MISMATCH {row}")
        raise SystemExit(
            "matched_reach_report: the shipped baseline differs between runs, "
            "so their cells are not on one reference. Nothing is summarised.")
    # A count of zero passes "identical in every pair" vacuously, which is
    # how a guard comes to be reported as green over nothing at all.  Say
    # which of the three cases this is.
    if len(docs) == 1:
        print("cross-run control: one file, so there is none -- every number "
              "below rests on this run's own in-process repeat")
    elif ctl["pairs"] == 0:
        raise SystemExit(
            "matched_reach_report: these files share no (unit, rung) shipped "
            "baseline, so nothing ties them together and their cells are not "
            "on one reference. Check the unit lists and the rungs.")
    else:
        print(f"cross-run control: {ctl['pairs']} shared (unit, rung) "
              "baselines, byte- AND tensor-identical in every pair")

    all_cells = {}
    for doc in docs.values():
        all_cells.update(cells(doc))
    units = sorted({k[0] for k in all_cells})
    rungs = sorted({k[1] for k in all_cells})
    Ls = sorted({k[2] for k in all_cells})
    # The columns of this factorial are the widths' OWN reaches -- the three
    # values a shipped-ratio arm lands on.  A file swept for something else
    # (the landed pair grid's ratios of 1.25 and up) also carries arms at
    # reaches no width has, and those are not columns of this table: they
    # have no entry-count row to be read against.  They are dropped, and
    # counted, rather than widening the table with cells that can only ever
    # read "n/N units".
    own = sorted({k[3] for k, c in all_cells.items() if c["ratio"] == 1.0})
    other = sorted({k[3] for k in all_cells} - set(own))
    reaches = own
    if other:
        print(f"columns dropped (reaches no width lands on at ratio 1): "
              f"{[round(r, 4) for r in other]}")
    print(f"units={len(units)}  rungs={[r / 256 for r in rungs]}  "
          f"widths={Ls}  reaches={reaches}")

    for rung in rungs:
        print(f"\n== R={rung / 256:g}   rows = table entries (L, costs bytes), "
              f"columns = realised reach (costs none)")
        head = "  L    " + "".join(f"reach {r:<8g}".ljust(20) for r in reaches)
        print(head)
        grid_geo = {}
        for L in Ls:
            row = [f"  {L:<5}"]
            for reach in reaches:
                got = [all_cells[(u, rung, L, reach)] for u in units
                       if (u, rung, L, reach) in all_cells]
                got = [c for c in got if gate in c]
                if len(got) != len(units):
                    row.append(f"{len(got)}/{len(units)} units".ljust(20))
                    continue
                g = geo([c[gate] for c in got])
                wins = sum(1 for c in got if c[gate] < 1.0)
                grid_geo[(L, reach)] = (g, wins, got)
                row.append(f"{g:.4f}x  {wins}/{len(units)}".ljust(20))
            print("".join(row))
        # The unbundling, stated as the run's header registers it.
        print("  -- the split, where both halves are present --")
        for L in [x for x in Ls if x != 14]:
            own = {c["reach"] for c in all_cells.values() if c["L"] == L
                   and abs(c["ratio"] - 1.0) < 1e-12}
            if len(own) != 1:
                continue
            reach = next(iter(own))
            bundle = grid_geo.get((L, reach))
            free = grid_geo.get((14, reach))
            diag = grid_geo.get((14, 4.0))
            if not (bundle and free):
                continue
            A, B = bundle[0], free[0]
            frac = math.log(B) / math.log(A) if A not in (0, 1.0) else float("nan")
            verdict = read_split(A, B)
            print(f"  L={L} at its own reach {reach:g}: bundle {A:.4f}x "
                  f"({bundle[1]}/{len(units)}), spread-only at L=14 {B:.4f}x "
                  f"({free[1]}/{len(units)}), recovered {frac:+.3f}"
                  f"  -> {verdict}")
            if diag:
                print(f"      (the shipped cell reads {diag[0]:.4f}x, which is "
                      "1.0000 by construction -- it IS the reference)")
            per = []
            for u in units:
                a = all_cells[(u, rung, L, reach)][gate]
                b = all_cells[(u, rung, 14, reach)][gate]
                per.append(f"{math.log(b) / math.log(a):+.2f}" if a != 1.0 else "  -  ")
            print("      per-unit recovered: " + " ".join(per))

    # The physical check: the same reach clips the same rows.
    bad_over = []
    for (u, rung, L, reach), c in sorted(all_cells.items()):
        peer = all_cells.get((u, rung, 14, reach))
        if peer is not None and peer["over"] != c["over"]:
            bad_over.append((u, rung, L, reach, c["over"], peer["over"]))
    print(f"\nreach match: rows-over-reach equal at matched reach for "
          f"{'ALL' if not bad_over else 'NOT ALL'} cells"
          + ("" if not bad_over else f"  ({len(bad_over)} disagree)"))
    for row in bad_over[:6]:
        print(f"  {row}")

    for name, doc in docs.items():
        states = [control_status(v) for res in doc["units"].values()
                  for k, v in res.items() if k.endswith("_control")]
        ok = sum(1 for st in states if st["ran"] and st["passed"])
        print(f"in-run controls {name}: {ok} of {len(states)} repeats "
              "byte- AND tensor-identical"
              + ("" if ok == len(states) else "  -- NOT ALL; read the run's log"))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    gate = None
    for a in sys.argv[1:]:
        if a.startswith("--gate="):
            gate = a.split("=", 1)[1]
    if not args:
        raise SystemExit("usage: matched_reach_report.py FILE... [--gate=out]")
    report(args, gate=gate)
