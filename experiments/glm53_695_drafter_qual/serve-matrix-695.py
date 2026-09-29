"""tessera#695: drafter identity between serves, with the serve as the unit of replication.

  serve-matrix-695.py RECEIPTS EAGER_ARM,EAGER_ARM,... MODE=ARM[+ARM...] [MODE=...] [--out OUT_JSON]

Reads RECEIPTS/<arm>.draft.<pid>.jsonl for every arm, as draftlog-695.py does, and
for each MODE compares that mode's graph serves with the eager serves.

  contexts   the contexts that every eager serve and every serve of the mode
             reached, each weighted once (a context a batch case sends n times
             counts once, at the mean agreement of its n x m pairs).
  pairs      for every two serves, the fraction of those contexts at which their
             draft token at position i agrees, per position, and at which the
             whole draft vector agrees.
  statistic  the mean identity of the mode's graph~eager pairs minus the mean
             identity of the eager~eager pairs, per position and for the vector.
  p_low      the one-sided relabelling p-value: the fraction of the ways to label
             |graph| of the pooled serves "graph" whose statistic is at or below
             the observed one. Under the null hypothesis the serves are
             exchangeable, so every labelling is equally likely. The smallest
             attainable p is 1 / C(|eager| + |graph|, |graph|); it is reported
             as the floor, and a p cannot fall below it.

The per-context intervals of draftlog-695.py resample contexts inside fixed
serves, so they do not see serve-to-serve variation: on the four-layer stub two
FULL_DECODE_ONLY serves at k = 2 fell on opposite sides of the eager spread
under them. This test resamples serves.
"""
import importlib.util
import itertools
import json
import math
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("draftlog695", _HERE / "draftlog-695.py")
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)


def rates(by, common, k, a, b):
    """Per-position and whole-vector identity of serves a and b over the common contexts."""
    tot = [0.0] * (k + 1)
    for c in common:
        _, pos, whole = dl.agreement(by[a][c], by[b][c])
        for i in range(k):
            tot[i] += pos[i]
        tot[k] += whole
    return [t / len(common) for t in tot]


def statistic(pairs, graph, eager, width):
    ge = [pairs[frozenset((g, e))] for g in graph for e in eager]
    ee = [pairs[frozenset(p)] for p in itertools.combinations(eager, 2)]
    return [sum(x[i] for x in ge) / len(ge) - sum(x[i] for x in ee) / len(ee) for i in range(width)]


def compare(by, eager, graph):
    serves = eager + graph
    common = set.intersection(*(set(by[s]) for s in serves))
    if not common:
        raise SystemExit(f"{'+'.join(graph)}: no context is reached by every serve")
    ks = {len(d) for s in serves for ds in by[s].values() for d in ds}
    if len(ks) != 1:
        raise SystemExit(f"serves propose different draft counts {sorted(ks)}; compare like with like")
    k = ks.pop()
    cols = [f"position {i}" for i in range(k)] + ["whole vector"]
    pairs = {frozenset(p): rates(by, common, k, *p) for p in itertools.combinations(serves, 2)}
    observed = statistic(pairs, graph, eager, k + 1)
    labellings = [statistic(pairs, list(g), [s for s in serves if s not in g], k + 1)
                  for g in itertools.combinations(serves, len(graph))]
    p_low = [sum(1 for s in labellings if s[i] <= observed[i] + 1e-12) / len(labellings) for i in range(k + 1)]
    mean = lambda xs: [sum(x[i] for x in xs) / len(xs) for i in range(k + 1)] if xs else None
    return dict(
        k=k, columns=cols, contexts=len(common), eager=eager, graph=graph,
        pairs={"~".join(sorted(p)): v for p, v in pairs.items()},
        mean_eager_eager=mean([pairs[frozenset(p)] for p in itertools.combinations(eager, 2)]),
        mean_graph_eager=mean([pairs[frozenset((g, e))] for g in graph for e in eager]),
        mean_graph_graph=mean([pairs[frozenset(p)] for p in itertools.combinations(graph, 2)]),
        statistic=observed, p_low=p_low, labellings=len(labellings),
        p_floor=1 / math.comb(len(serves), len(graph)))


def main():
    args = sys.argv[1:]
    out = None
    if "--out" in args:
        i = args.index("--out")
        out = pathlib.Path(args[i + 1])
        del args[i:i + 2]
    receipts, eager, modes = args[0], args[1].split(","), dict(m.split("=", 1) for m in args[2:])
    if len(eager) < 2:
        raise SystemExit("need at least two eager serves: their pairs are the reference")
    arms = eager + [a for v in modes.values() for a in v.split("+")]
    by = {a: dl.by_context(dl.steps_of(dl.load(receipts, a))) for a in dict.fromkeys(arms)}
    record = {}
    fmt = lambda xs: " ".join(f"{x:.4f}" for x in xs)
    for mode, members in modes.items():
        r = compare(by, eager, members.split("+"))
        record[mode] = r
        print(f"{mode}: {len(r['graph'])} graph and {len(r['eager'])} eager serves, {r['contexts']} common "
              f"contexts, k = {r['k']} ({', '.join(r['columns'])})")
        for pair, v in sorted(r["pairs"].items()):
            print(f"  {pair:24s} {fmt(v)}")
        print(f"  mean eager~eager {fmt(r['mean_eager_eager'])}")
        print(f"  mean graph~eager {fmt(r['mean_graph_eager'])}")
        if r["mean_graph_graph"] is not None:
            print(f"  mean graph~graph {fmt(r['mean_graph_graph'])}")
        print(f"  graph~eager minus eager~eager " + "; ".join(
            f"{c} {s:+.4f} p_low {p:.3f}" for c, s, p in zip(r["columns"], r["statistic"], r["p_low"]))
              + f" (over {r['labellings']} labellings, floor {r['p_floor']:.3f})")
    if out:
        out.write_text(json.dumps(record, indent=1))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
