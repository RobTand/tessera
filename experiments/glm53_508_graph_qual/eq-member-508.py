"""tessera#508: is every graph-mode result one that eager itself produces?

Eager is not repeat-exact on this stub: a few cases move by a recurring discrete
amount between fresh serves, and within one serve (eagerF1 vs eagerF2, eagerN2 vs
its second pass). A graph arm is therefore compared with the SET of eager outcomes,
not with one eager serve: for every case of the equality suite and every choice in
it, the graph arm's greedy token ids and top-20 logprobs must be bit-identical to
the same choice of at least one eager run in the pool. Every eager run in the pool
must have admitted its batches paused (VLLM_SERVER_DEV_MODE=1), so all runs share
one step structure.

Cases that send the same batch (the same prompt token ids in the same order and the
same generation length) are one outcome pool: rep_len17_a, rep_len17_b and b1_len17
send one 17-token prompt, so an eager outcome of any of them is an eager outcome of
each. Cases that differ in batch composition are never pooled (b5's first prompt is
b2's first prompt, but b5 decodes five rows per step and b2 two).

For a choice that matches no eager run, the report names the eager outcome
nearest to it: the one with the smallest largest logprob difference on an
identical prefix (then the fewest differing positions). That difference is taken
over the tokens both top-20 lists name, at the positions up to and including the
first differing generated token id, where both runs condition on the same
tokens; past that position the two lists answer different contexts, so their
differences measure nothing. The report also lists the positions whose generated
token id differs from that outcome's.

  eq-member-508.py RECEIPTS ARM EAGER_ARM[,EAGER_ARM...]
Writes RECEIPTS/member-<ARM>.json; exit 0 when every choice of every case matches.

Each pool name's second pass (<name>-r2) joins the pool when its receipts exist,
except when it is ARM itself: a run is never judged against its own outcomes. ARM
may not appear in the pool. To judge an eager run against the other eager runs,
leave its serve out of the pool, or list only the serve's other pass.
"""
import json
import pathlib
import sys

recs = pathlib.Path(sys.argv[1])
arm = sys.argv[2]
pool = sys.argv[3].split(",")


def load_arm(name):
    summary = json.loads((recs / f"{name}.eq.summary.json").read_text())
    if summary.get("_admission") != "paused":
        raise SystemExit(f"{name}: batches were not admitted paused ({summary.get('_admission')!r})")
    cases = {}
    for case in summary:
        path = recs / f"{name}.eq.{case}.json"
        if isinstance(summary[case], dict) and path.exists():
            cases[case] = json.loads(path.read_text())["choices"]
    return cases


def signature(choices):
    """The batch a case sends: every prompt's token ids, in order, and the length generated."""
    return json.dumps([[c["prompt_token_ids"], len(c["token_ids"])] for c in choices])


def key(choice):
    return json.dumps([choice["token_ids"], choice["logprobs"]["top_logprobs"]], sort_keys=True)


def distance(a, b):
    la, lb = a["logprobs"]["top_logprobs"], b["logprobs"]["top_logprobs"]
    positions = sum(1 for x, y in zip(la, lb) if x != y) + abs(len(la) - len(lb))
    dmax = 0.0
    for x, y in zip(la, lb):
        for tok in set(x) & set(y):
            dmax = max(dmax, abs(x[tok] - y[tok]))
    return positions, dmax


def same_prefix(a, b):
    """Token-id divergence and the largest logprob difference on an identical prefix."""
    ta, tb = a["token_ids"], b["token_ids"]
    diverge = [p for p, (x, y) in enumerate(zip(ta, tb)) if x != y]
    last = diverge[0] if diverge else min(len(ta), len(tb)) - 1
    la, lb = a["logprobs"]["top_logprobs"], b["logprobs"]["top_logprobs"]
    dmax = 0.0
    for x, y in zip(la[:last + 1], lb[:last + 1]):
        for tok in set(x) & set(y):
            dmax = max(dmax, abs(x[tok] - y[tok]))
    return dict(token_ids_differ_at=diverge, max_same_prefix_delta=dmax)


if arm in pool:
    raise SystemExit(f"{arm} is in its own pool")
target = load_arm(arm)
runs = {}
for name in pool:
    runs[name] = load_arm(name)
    second = f"{name}-r2"
    if second != arm and (recs / f"{second}.eq.summary.json").exists():
        runs[second] = load_arm(second)

# Outcome pool per target case: (run, case) pairs whose case sends the same batch.
outcomes = {}
for case, choices in target.items():
    sig = signature(choices)
    outcomes[case] = [(f"{r}:{c}" if c != case else r, cs) for r, cases in runs.items()
                      for c, cs in cases.items() if signature(cs) == sig]

report, all_ok = {}, True
for case, choices in target.items():
    rows = []
    for i, choice in enumerate(choices):
        hits = [r for r, cs in outcomes[case] if key(cs[i]) == key(choice)]
        row = dict(choice=i, member=bool(hits), matches=hits)
        if not hits:
            near = [(same_prefix(choice, cs[i]), distance(choice, cs[i]), r)
                    for r, cs in outcomes[case]]
            best = min(near, default=None,
                       key=lambda t: (t[0]["max_same_prefix_delta"], t[1][0], t[2]))
            if best is not None:
                prefix, (pos, dmax), r = best
                row.update(closest=r, differing_positions=pos, max_shared_top20_delta=dmax,
                           **prefix)
            all_ok = False
        rows.append(row)
    report[case] = rows
    miss = [r for r in rows if not r["member"]]
    if miss:
        print(f"{case:12s} NOT A MEMBER: " + "; ".join(
            f"choice {r['choice']} closest {r.get('closest')} "
            f"({r.get('differing_positions')} positions, token ids differ at "
            f"{r.get('token_ids_differ_at')}, same-prefix max {r.get('max_same_prefix_delta', 0):.6g})"
            for r in miss))
    else:
        print(f"{case:12s} member: every choice equals an eager outcome "
              f"({len(set(h for r in rows for h in r['matches']))} of {len(outcomes[case])} pooled outcomes "
              f"match some choice)")
(recs / f"member-{arm}.json").write_text(json.dumps(dict(arm=arm, pool=sorted(runs), cases=report),
                                                      indent=1))
print("ALL MEMBERS" if all_ok else "NOT ALL MEMBERS", f"(pool: {', '.join(sorted(runs))})")
sys.exit(0 if all_ok else 1)
