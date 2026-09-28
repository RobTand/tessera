"""tessera#695: equality-suite membership with pools of one size for eager and graph runs.

  member-pools-695.py RECEIPTS EAGER_SERVE,EAGER_SERVE,... GRAPH_ARM,... [--out OUT_JSON]
  member-pools-695.py --relabel RECEIPTS EAGER_SERVE,... MODE=ARM[+ARM...] [MODE=...] [--out OUT_JSON]

Every run is judged with the predicate of ../glm53_508_graph_qual/eq-member-508.py
against pools of n - 1 eager serves, where n is the number of eager serves listed;
a serve brings its first pass and, when its receipts exist, its second pass
(<serve>-r2).

  eager serve S   S and S-r2 are each judged against the other n - 1 serves.
  graph arm G     G and G-r2 are each judged against every (n - 1)-subset of the
                  eager serves, and the counts are averaged over the subsets.

Judging eager runs against pools of the size the graph runs face makes the two
counts comparable: the eager counts are the noise floor for the graph counts.

A choice equals an outcome of a pool exactly when it equals an outcome of one of
the pool's serves, so a pool's members are the union of its serves' members. Each
run is compared with each other serve once, and every pool is read off those match
sets. Nothing is written next to the receipts.

--relabel applies criterion 2's serve-level relabelling to the non-member counts
(criterion 3 in step 2). For each MODE its graph serves and the eager serves are
pooled, and every way to label |graph| of them "graph" is scored with the pools
recomputed under that labelling:

  eager-labelled serve   each pass judged against the other eager-labelled serves
  graph-labelled serve   each pass judged against every (e - 1)-subset of the e
                         eager-labelled serves, averaged over the subsets
  serve count            the mean of its passes' counts
  statistic              mean graph-labelled serve count minus mean eager-labelled
                         serve count, as an exact fraction
  p_high                 the fraction of labellings whose statistic is at or above
                         the observed one, ties counted; the mode is refused when
                         p_high < 0.10. The smallest attainable p_high is
                         1 / C(e + g, g), the floor; a floor above 0.10 cannot refuse.

No run is ever in its own pool, and neither is its serve's other pass. The max
rule (no graph run's count above the largest eager run's) is printed beside it
at the observed labelling, as a diagnostic that never decides.
"""
import importlib.util
import itertools
import json
import math
import pathlib
import sys
from fractions import Fraction

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "eq_member_508", _HERE.parent / "glm53_508_graph_qual" / "eq-member-508.py")
eqm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eqm)

REFUSE_BELOW = Fraction(1, 10)


class Membership:
    """Each run's choices, and its match set against each other serve, computed once."""

    def __init__(self, receipts, serves):
        self.receipts = receipts
        self.passes, self.runs, self.order = {}, {}, {}
        for s in serves:
            if not (receipts / f"{s}.eq.summary.json").exists():
                raise SystemExit(f"{s}: no equality-suite receipts in {receipts}")
            self.passes[s] = [a for a in (s, f"{s}-r2") if (receipts / f"{a}.eq.summary.json").exists()]
            for a in self.passes[s]:
                self.runs[a] = eqm.load_arm(receipts, a)
                self.order[a] = [(case, i) for case, choices in self.runs[a].items()
                                 for i in range(len(choices))]
        self.serve_of = {a: s for s, runs in self.passes.items() for a in runs}
        self._match = {}

    def match(self, run, serve):
        """Bit mask over RUN's choices that equal an outcome of SERVE's passes."""
        if serve == self.serve_of[run]:
            raise SystemExit(f"{run} would be judged against its own serve")
        k = (run, serve)
        if k not in self._match:
            pool = {a: self.runs[a] for a in self.passes[serve] if a != run}
            hits = eqm.members(self.runs[run], pool)
            self._match[k] = sum(1 << j for j, c in enumerate(self.order[run]) if c in hits)
        return self._match[k]

    def misses(self, run, pool):
        """RUN's non-member (case, choice) pairs against the serves of POOL, in record order."""
        got = 0
        for s in pool:
            got |= self.match(run, s)
        return [[case, i] for j, (case, i) in enumerate(self.order[run]) if not got >> j & 1]

    def count(self, run, pool):
        got = 0
        for s in pool:
            got |= self.match(run, s)
        return len(self.order[run]) - bin(got).count("1")

    def choices(self, run):
        return len(self.order[run])

    def run_count(self, run, eager, graph_labelled):
        """A run's pool-averaged non-member count, as a fraction, under one labelling."""
        if not graph_labelled:
            return Fraction(self.count(run, [e for e in eager if e != self.serve_of[run]]))
        subsets = list(itertools.combinations(eager, len(eager) - 1))
        return Fraction(sum(self.count(run, t) for t in subsets), len(subsets))

    def serve_count(self, serve, eager, graph_labelled):
        runs = self.passes[serve]
        return sum((self.run_count(a, eager, graph_labelled) for a in runs), Fraction(0)) / len(runs)


def pools(args):
    """Criterion 3 as run on the stub: per-run counts at the listed labelling."""
    out = None
    if "--out" in args:
        i = args.index("--out")
        out = pathlib.Path(args[i + 1]).resolve()
        del args[i:i + 2]
    receipts = pathlib.Path(args[0]).resolve()
    serves, graph = args[1].split(","), [g for g in args[2].split(",") if g]
    if len(serves) < 2:
        raise SystemExit("need at least two eager serves")
    m = Membership(receipts, serves + graph)

    record = {}
    for s in serves:
        for arm in m.passes[s]:
            miss = m.misses(arm, [x for x in serves if x != s])
            record[arm] = dict(kind="eager", pools=1, nonmember_mean=len(miss), choices=m.choices(arm),
                               misses=[miss])
    for g in graph:
        for arm in m.passes[g]:
            runs = [m.misses(arm, list(sub)) for sub in itertools.combinations(serves, len(serves) - 1)]
            record[arm] = dict(kind="graph", pools=len(runs), choices=m.choices(arm),
                               nonmember_mean=sum(len(x) for x in runs) / len(runs), misses=runs)
    eager_counts = [r["nonmember_mean"] for r in record.values() if r["kind"] == "eager"]
    for arm, r in record.items():
        cases = sorted({c for x in r["misses"] for c, _ in x})
        print(f"{arm:12s} {r['kind']:5s} non-member choices {r['nonmember_mean']:.1f} of {r['choices']} "
              f"over {r['pools']} pool(s) of {len(serves) - 1} serves; cases {cases}")
    if eager_counts:
        print(f"eager range {min(eager_counts):.1f} to {max(eager_counts):.1f}")
    if out:
        out.write_text(json.dumps(dict(serves=serves, graph=graph, pool_serves=len(serves) - 1, runs=record),
                                  indent=1))
        print(f"wrote {out}")


def relabel(m, eager, graph):
    """Serve-level relabelling of the mean non-member counts for one mode."""
    serves = eager + graph
    if len(eager) < 2 or not graph:
        raise SystemExit(f"relabelling needs at least two eager serves and one graph serve "
                         f"({len(eager)}E+{len(graph)}G)")

    def score(g_lab):
        e_lab = [s for s in serves if s not in g_lab]
        counts = {s: m.serve_count(s, e_lab, s in g_lab) for s in serves}
        stat = (sum((counts[s] for s in g_lab), Fraction(0)) / len(g_lab)
                - sum((counts[s] for s in e_lab), Fraction(0)) / len(e_lab))
        return stat, counts

    observed, counts = score(tuple(graph))
    table = []
    for g_lab in itertools.combinations(serves, len(graph)):
        stat, _ = score(g_lab)
        table.append((list(g_lab), stat))
    labellings = len(table)
    assert labellings == math.comb(len(serves), len(graph))
    p_high = Fraction(sum(1 for _, s in table if s >= observed), labellings)
    floor = Fraction(1, labellings)
    runs = {a: m.run_count(a, eager, s in graph) for s in serves for a in m.passes[s]}
    g_max = max(runs[a] for s in graph for a in m.passes[s])
    e_max = max(runs[a] for s in eager for a in m.passes[s])
    verdict = ("cannot refuse" if floor >= REFUSE_BELOW
               else "refuses" if p_high < REFUSE_BELOW else "holds")
    return dict(
        eager=eager, graph=graph, labellings=labellings,
        floor=float(floor), floor_fraction=str(floor),
        serve_counts={s: float(c) for s, c in counts.items()},
        serve_counts_fraction={s: str(c) for s, c in counts.items()},
        run_counts={a: float(c) for a, c in runs.items()},
        choices={a: m.choices(a) for a in runs},
        mean_graph=float(sum((counts[s] for s in graph), Fraction(0)) / len(graph)),
        mean_eager=float(sum((counts[s] for s in eager), Fraction(0)) / len(eager)),
        statistic=float(observed), statistic_fraction=str(observed),
        p_high=float(p_high), p_high_fraction=str(p_high),
        verdict=verdict,
        max_rule=dict(largest_graph_run=float(g_max), largest_eager_run=float(e_max),
                      reading="holds" if g_max <= e_max else "fails", decides=False),
        table=[dict(graph=g, statistic=float(s), statistic_fraction=str(s)) for g, s in table],
    )


def relabel_main(args):
    out = None
    if "--out" in args:
        i = args.index("--out")
        out = pathlib.Path(args[i + 1]).resolve()
        del args[i:i + 2]
    receipts, eager = pathlib.Path(args[0]).resolve(), args[1].split(",")
    modes = dict(x.split("=", 1) for x in args[2:])
    if not modes:
        raise SystemExit("name at least one MODE=ARM[+ARM...]")
    arms = eager + [a for v in modes.values() for a in v.split("+")]
    if len(set(arms)) != len(arms):
        raise SystemExit("a serve is listed twice")
    m = Membership(receipts, arms)
    record = {}
    for mode, members in modes.items():
        graph = members.split("+")
        r = relabel(m, eager, graph)
        record[mode] = r
        print(f"{mode}: {len(eager)} eager and {len(graph)} graph serves, {r['labellings']} labellings, "
              f"floor {r['floor']:.3f}")
        for s in eager + graph:
            kind = "graph" if s in graph else "eager"
            per = ", ".join(f"{a} {r['run_counts'][a]:.2f}" for a in m.passes[s])
            print(f"  {s:12s} {kind} serve count {r['serve_counts'][s]:.3f} of {m.choices(s)} ({per})")
        print(f"  mean graph {r['mean_graph']:.3f} minus mean eager {r['mean_eager']:.3f} = "
              f"{r['statistic']:+.3f}; p_high {r['p_high']:.3f} ({r['p_high_fraction']}): {r['verdict']}")
        mr = r["max_rule"]
        print(f"  max rule (diagnostic, does not decide): largest graph run {mr['largest_graph_run']:.2f} "
              f"vs largest eager run {mr['largest_eager_run']:.2f}: {mr['reading']}")
    if out:
        out.write_text(json.dumps(record, indent=1))
        print(f"wrote {out}")


def main():
    args = sys.argv[1:]
    if args[:1] == ["--relabel"]:
        relabel_main(args[1:])
    else:
        pools(args)


if __name__ == "__main__":
    main()
