#!/usr/bin/env python
"""Read issue #50's two JSONs into the tables the verdict needs.

``lut_landing_oracle.py`` writes the matched pair on the landing alone (every
refit call replayed with the encoder's landing and with the coupled one) and
the end-state ladder up to the continuous ceiling; ``ldlq_window_sweep.py
--coupled-landing --drift-control`` writes the in-encoder arms, where the
coupled landing also changes the codes the next trellis pass sees.  This prints:

1. **the drift controls** of both runs -- the noise floor;
2. **the end-state ladder per unit**, out and hfit, for the Gauss-Seidel arm:
   landed, coupled-assign, coupled-table, free-e4m3, free -- how far the
   coupled landing goes towards the ceiling, at the codes the wire holds;
3. **the per-pass matched pair pooled over units** (cost-weighted): step,
   landing loss, what the coupled landing recovers, against the sink's
   ``continuous`` and against the exact joint minimiser;
4. **which candidate the encoder's accept guard chose** on every pass -- the
   count of ``new-table`` wins, which is the empirical weight of the issue's
   claim that the separable TABLE FIT is where the loss lives;
5. **the in-encoder arms per unit** -- control, GS, GS + coupled landing,
   Jacobi + coupled landing -- and the pre-registered verdict.

    python experiments/lut_landing_report.py oracle.json coupled.json
"""
from __future__ import annotations

import argparse
import json
import math


def geo(units, arm, field):
    return math.exp(sum(math.log(units[u][arm][field]) for u in units) / len(units))


def short(u):
    return u.replace("model.layers.", "L").replace("self_attn.", "").replace("mlp.", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("oracle")
    ap.add_argument("coupled", nargs="?")
    a = ap.parse_args()
    o = json.load(open(a.oracle))
    ou = o["units"]
    names = list(ou)
    CONTROL = "control [LDLQ 1.0/32 + refit h^1.0]"
    JAC = "LDLQ 1.0/32 + refit full-H (Jacobi)"
    GS = "LDLQ 1.0/32 + refit full-H (Gauss-Seidel)"
    SWAP = "control codes + one full-H GS refit"
    LADDER = ["coupled-assign", "coupled-table", "free-e4m3", "free"]

    print("== 1. drift control (oracle run): the served default first and last, per unit")
    for u in names:
        f, l = ou[u][CONTROL], ou[u][CONTROL + " REPEAT"]
        print(f"   {short(u):<16} out {f['out']:.6f} -> {l['out']:.6f}  bytes "
              f"{'IDENTICAL' if f['sha256'] == l['sha256'] else 'DIFFER'}  "
              f"ref-digests: ctl {ou[u][CONTROL].get('matches_reference')} "
              f"jac {ou[u][JAC].get('matches_reference')} gs {ou[u][GS].get('matches_reference')}  "
              f"replay==wire: {ou[u][GS].get('replay_matches_wire')}")

    print("\n== 2. end-state ladder, GS arm, per unit (out = held-out activation space; hfit = fit-row quadratic)")
    print(f"   {'unit':<16} {'landed':>8} {'assign':>8} {'table':>8} {'e4m3':>8} {'free':>8} | "
          f"{'assign/l':>8} {'table/l':>8} {'free/l':>8} | {'ctl':>8} {'table/ctl':>9}")
    for field in ("out", "hfit"):
        print(f"   -- {field}")
        for u in names:
            r = ou[u]
            l = r[GS][field]
            v = [r[f"{k} [{GS}]"][field] for k in LADDER]
            c = r[CONTROL][field]
            print(f"   {short(u):<16} {l:8.5f} {v[0]:8.5f} {v[1]:8.5f} {v[2]:8.5f} {v[3]:8.5f} | "
                  f"{v[0] / l:8.4f} {v[1] / l:8.4f} {v[3] / l:8.4f} | {c:8.5f} {v[1] / c:9.4f}")
        l = geo(ou, GS, field)
        v = [geo(ou, f"{k} [{GS}]", field) for k in LADDER]
        c = geo(ou, CONTROL, field)
        print(f"   {'GEOMEAN':<16} {l:8.5f} {v[0]:8.5f} {v[1]:8.5f} {v[2]:8.5f} {v[3]:8.5f} | "
              f"{v[0] / l:8.4f} {v[1] / l:8.4f} {v[3] / l:8.4f} | {c:8.5f} {v[1] / c:9.4f}")

    print("\n== 2b. the same ladder for the Jacobi arm and the control (out geomean, ratios to landed)")
    for base in (JAC, CONTROL, SWAP):
        l = geo(ou, base, "out")
        v = [geo(ou, f"{k} [{base}]", "out") for k in LADDER]
        print(f"   {base:<44} landed {l:.5f}  assign {v[0] / l:.4f}x  table {v[1] / l:.4f}x  "
              f"e4m3 {v[2] / l:.4f}x  free {v[3] / l:.4f}x")

    print("\n== 3. the per-pass matched pair, pooled over units and passes (cost-weighted)")
    print("   fractions of the pass's starting cost; recov = share of (landed - continuous) the")
    print("   coupled landing gets back; vs free = the same share of (landed - free)")
    print(f"   {'arm':<8} {'step':>8} {'landing':>8} {'coupled':>8} {'recov':>7} {'vs free':>8} "
          f"{'freegap':>8} {'entry moves':>11} {'new-table wins':>14}")
    for label, arm in (("h^1.0", CONTROL), ("jacobi", JAC), ("gs", GS)):
        B = S = C = Ld = T = F = 0.0
        entry = wins = passes = 0
        for u in names:
            for rec in ou[u][arm]["replay"]:
                B += rec["before"]; S += rec["stepped"]; C += rec["continuous"]
                Ld += rec["landed"]; T += rec["coupled_table"]; F += rec["free"]
                entry += rec["table_entry_moves"]; passes += 1
                wins += rec["candidate"].startswith("new-table")
        land = Ld - C
        print(f"   {label:<8} {(B - S) / B:8.3%} {land / B:8.3%} {(Ld - T) / B:8.3%} "
              f"{((Ld - T) / land if land > 0 else float('nan')):7.1%} "
              f"{(Ld - T) / (Ld - F):8.1%} {(Ld - F) / B:8.3%} {entry:11d} {wins:8d}/{passes:<5d}")
    print("   per unit and pass, GS arm:")
    print(f"   {'unit':<16} {'p':>2} {'step':>8} {'landing':>8} {'coupled':>8} {'recov':>7} "
          f"{'vs free':>8} {'candidate':>12} {'moves':>6} {'entry':>5} {'noise':>8}")
    for u in names:
        for p, rec in enumerate(ou[u][GS]["replay"]):
            b, s, c, ld, t, f = (rec[k] for k in ("before", "stepped", "continuous", "landed",
                                                  "coupled_table", "free"))
            print(f"   {short(u):<16} {p:2d} {(b - s) / b:8.3%} {(ld - c) / b:8.3%} {(ld - t) / b:8.3%} "
                  f"{((ld - t) / (ld - c) if ld > c else float('nan')):7.1%} "
                  f"{((ld - t) / (ld - f) if ld > f else float('nan')):8.1%} {rec['candidate']:>12} "
                  f"{rec['assign_moves'] + rec['table_block_moves']:6d} {rec['table_entry_moves']:5d} "
                  f"{rec.get('replay_rel_discrepancy', float('nan')):8.1e}")

    if "verdict_issue_50" in o:
        print("\n== 4. the oracle run's pre-registered verdict")
        print(json.dumps(o["verdict_issue_50"], indent=1))

    if a.coupled:
        c = json.load(open(a.coupled))
        cu = c["units"]
        arms = sorted(set.intersection(*[{k for k in v if not k.startswith("_")} for v in cu.values()]))
        ctl = next(x for x in arms if x.startswith("drift control FIRST"))
        last = next(x for x in arms if x.startswith("drift control LAST"))
        gs = next(x for x in arms if x.endswith("(Gauss-Seidel)"))
        jac = next(x for x in arms if x.startswith("LDLQ") and x.endswith("refit full-H"))
        gsc = next(x for x in arms if x.endswith("(Gauss-Seidel, coupled landing)"))
        jc = next(x for x in arms if x.endswith("(Jacobi, coupled landing)"))
        gst = next((x for x in arms if x.endswith("(Gauss-Seidel, trailing coupled landing)")), None)
        jt = next((x for x in arms if x.endswith("(Jacobi, trailing coupled landing)")), None)
        print("\n== 5. in-encoder arms (sweep run): the coupled landing inside the alternation")
        print("   drift: " + "  ".join(
            f"{short(u)}:{'ID' if cu[u][ctl]['sha256'] == cu[u][last]['sha256'] else 'DIFFER'}" for u in cu))
        cols = [("ctl", ctl), ("jac", jac), ("gs", gs), ("jac+cpl", jc), ("gs+cpl", gsc)]
        if jt and gst:
            cols += [("jac+trl", jt), ("gs+trl", gst)]
        for field in ("out", "hfit"):
            print(f"   -- {field}   (cpl = coupled landing on every pass; trl = on the trailing refit only)")
            print("   " + f"{'unit':<16}" + "".join(f"{n:>9}" for n, _ in cols) + " |"
                  + f"{'gs+cpl/gs':>10}{'gs+cpl/ctl':>11}" + (f"{'gs+trl/gs':>10}{'gs+trl/ctl':>11}" if gst else ""))
            for u in cu:
                r = cu[u]
                line = f"   {short(u):<16}" + "".join(f"{r[k][field]:9.5f}" for _, k in cols) + " |"
                line += f"{r[gsc][field] / r[gs][field]:10.4f}{r[gsc][field] / r[ctl][field]:11.4f}"
                if gst:
                    line += f"{r[gst][field] / r[gs][field]:10.4f}{r[gst][field] / r[ctl][field]:11.4f}"
                print(line)
            g = {k: geo(cu, k, field) for _, k in cols}
            line = f"   {'GEOMEAN':<16}" + "".join(f"{g[k]:9.5f}" for _, k in cols) + " |"
            line += f"{g[gsc] / g[gs]:10.4f}{g[gsc] / g[ctl]:11.4f}"
            if gst:
                line += f"{g[gst] / g[gs]:10.4f}{g[gst] / g[ctl]:11.4f}"
            print(line)
        print("   coupled-landing sweeps and moves per pass (GS + coupled arm):")
        for u in cu:
            recs = cu[u][gsc].get("refit", [])
            print(f"   {short(u):<16} " + "  ".join(
                f"p{i}: {r.get('coupled_sweeps', '-')} sweeps {r.get('coupled_moves', '-')} moves "
                f"{(r['coupled'] - r['landed']) / r['before']:+.2%}" for i, r in enumerate(recs)))
        bar = o.get("bar", 0.0138)
        for label, arm in (("every pass", gsc), ("trailing refit only", gst)):
            if arm is None:
                continue
            gain = 1.0 - geo(cu, arm, "out") / geo(cu, gs, "out")
            print(f"\n   in-encoder: GS + coupled landing ({label}) vs GS on out geomean: {gain:+.2%} "
                  f"(bar {bar:.2%}: {'CLEARS' if gain > bar else 'does not clear'}); "
                  f"vs control {geo(cu, arm, 'out') / geo(cu, ctl, 'out'):.4f}x")


if __name__ == "__main__":
    main()
