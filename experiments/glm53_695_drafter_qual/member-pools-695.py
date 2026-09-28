"""tessera#695: equality-suite membership with pools of one size for eager and graph runs.

  member-pools-695.py RECEIPTS EAGER_SERVE,EAGER_SERVE,... GRAPH_ARM,... [--out OUT_JSON]

Every run is judged by ../glm53_508_graph_qual/eq-member-508.py against pools of
n - 1 eager serves, where n is the number of eager serves listed; a serve brings
its first pass and, when its receipts exist, its second pass (<serve>-r2).

  eager serve S   S and S-r2 are each judged against the other n - 1 serves.
  graph arm G     G and G-r2 are each judged against every (n - 1)-subset of the
                  eager serves, and the counts are averaged over the subsets.

Judging eager runs against pools of the size the graph runs face makes the two
counts comparable: the eager counts are the noise floor for the graph counts.
eq-member-508.py writes member-<arm>.json next to the receipts it reads, so the
runs are judged in a work directory of links (OUT_JSON's directory, or RECEIPTS
when no --out is given), never in RECEIPTS itself.
"""
import itertools
import json
import pathlib
import subprocess
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_MEMBER = _HERE.parent / "glm53_508_graph_qual" / "eq-member-508.py"


def main():
    args = sys.argv[1:]
    out = None
    if "--out" in args:
        i = args.index("--out")
        out = pathlib.Path(args[i + 1]).resolve()
        del args[i:i + 2]
    receipts = pathlib.Path(args[0]).resolve()
    serves, graph = args[1].split(","), [g for g in args[2].split(",") if g]
    if len(serves) < 2:
        raise SystemExit("need at least two eager serves")
    work = (out.parent if out else receipts) / "member-pools.work"
    work.mkdir(parents=True, exist_ok=True)
    for f in receipts.glob("*.eq.*.json"):
        link = work / f.name
        if not link.exists():
            link.symlink_to(f)

    def judge(arm, pool):
        subprocess.run([sys.executable, str(_MEMBER), str(work), arm, ",".join(pool)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rec = json.loads((work / f"member-{arm}.json").read_text())
        if arm in rec["pool"]:
            raise SystemExit(f"{arm} was judged against itself")
        miss = [[case, row["choice"]] for case, rows in rec["cases"].items() for row in rows if not row["member"]]
        return miss, sum(len(rows) for rows in rec["cases"].values())

    record = {}
    for s in serves:
        for arm in (s, f"{s}-r2"):
            if not (receipts / f"{arm}.eq.summary.json").exists():
                continue
            miss, total = judge(arm, [x for x in serves if x != s])
            record[arm] = dict(kind="eager", pools=1, nonmember_mean=len(miss), choices=total, misses=[miss])
    for g in graph:
        for arm in (g, f"{g}-r2"):
            if not (receipts / f"{arm}.eq.summary.json").exists():
                continue
            runs = [judge(arm, list(sub)) for sub in itertools.combinations(serves, len(serves) - 1)]
            record[arm] = dict(kind="graph", pools=len(runs), choices=runs[0][1],
                               nonmember_mean=sum(len(m) for m, _ in runs) / len(runs),
                               misses=[m for m, _ in runs])
    eager_counts = [r["nonmember_mean"] for r in record.values() if r["kind"] == "eager"]
    for arm, r in record.items():
        cases = sorted({c for m in r["misses"] for c, _ in m})
        print(f"{arm:12s} {r['kind']:5s} non-member choices {r['nonmember_mean']:.1f} of {r['choices']} "
              f"over {r['pools']} pool(s) of {len(serves) - 1} serves; cases {cases}")
    if eager_counts:
        print(f"eager range {min(eager_counts):.1f} to {max(eager_counts):.1f}")
    if out:
        out.write_text(json.dumps(dict(serves=serves, graph=graph, pool_serves=len(serves) - 1, runs=record),
                                  indent=1))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
