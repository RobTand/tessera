"""tessera#702/#695: one table of every arm on the vLLM nightly stack.

For each arm: its serve configuration (engine-args-<arm>.txt), the CUDA graphs
its runner captured and the token counts it replayed (<arm>.dispatch.<pid>.json,
GA_DISPATCH_LOG), and its equality-suite outcome against the eager pool with the
tessera#508 membership criterion (eq-member-508.py): a choice is a member when
its generated token ids and every top-20 logprob list equal the same choice of
some eager run of the same batch. Eager runs are judged against the other eager
serves (never against themselves); an arm with no other eager run in its pool
is named unjudged in the table and carries no membership numbers.

  summarize-arms.py RECEIPTS POOL ARM [ARM ...]    POOL: comma list of eager serves
Writes RECEIPTS/summary-<first POOL name>.json and prints one line per arm.
"""
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "eq_member", HERE.parent / "glm53_508_graph_qual" / "eq-member-508.py")
eq = importlib.util.module_from_spec(spec)
sys.argv, _argv = sys.argv[:1], sys.argv  # the module reads no argv at import
spec.loader.exec_module(eq)
sys.argv = _argv


def engine_args(recs, arm):
    out = {}
    path = recs / f"engine-args-{arm}.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            key, _, value = line.partition("=")
            out.setdefault(key, value)
    return out


def dispatch(recs, arm):
    """The engine process's totals (the process that captured graphs, else the busiest)."""
    best = None
    for path in sorted(recs.glob(f"{arm}.dispatch.*.json")):
        rec = json.loads(path.read_text())
        score = (bool(rec.get("captured")), sum(rec.get("counts", {}).values()))
        if best is None or score > best[0]:
            best = (score, rec)
    if best is None:
        return None
    rec = best[1]
    full, none = {}, {}
    for key, n in rec["counts"].items():
        mgr, mode, tokens, _reqs = key.split("|")
        tokens = int(tokens.split("=")[1])
        bucket = full if mode == "FULL" else none if mode == "NONE" else {}
        name = f"{mgr}:{tokens}"
        bucket[name] = bucket.get(name, 0) + n
    return dict(captured=rec.get("captured"), full_replays=full,
                eager_steps=sum(none.values()), pid=rec.get("pid"))


def first_difference(a, b):
    """The first generated step at which token id or top-20 list differ, or None."""
    la, lb = a["logprobs"]["top_logprobs"], b["logprobs"]["top_logprobs"]
    for step, (ta, tb, xa, xb) in enumerate(zip(a["token_ids"], b["token_ids"], la, lb)):
        if ta != tb or xa != xb:
            return step
    return None


def judge(recs, arm, pool):
    runs = eq.pool_runs(recs, arm, [p for p in pool if p != arm and f"{p}-r2" != arm])
    if not runs:
        # The membership criterion compares an arm with the eager outcomes other
        # serves produced. An arm with no other eager run in its pool (a lone
        # eager serve, or one judged only against itself) has no criterion, so
        # it is named unjudged rather than scored against nothing.
        return dict(judged=False, why="no other eager run in the pool", pool=[])
    target = eq.load_arm(recs, arm)
    outcomes = eq.outcome_pools(target, runs)
    total = members = changed = 0
    worst = 0.0
    misses = []
    # Where a non-member first departs from its nearest eager outcome: the step
    # (0 = the token prefill samples) and the context the step attended to
    # (prompt tokens + tokens generated before it, the sequence length the
    # step's attention and indexer see).
    first_steps, first_contexts = [], []
    for case, choices in target.items():
        for i, choice in enumerate(choices):
            total += 1
            if any(eq.key(cs[i]) == eq.key(choice) for _, cs in outcomes[case]):
                members += 1
                continue
            near = [(eq.same_prefix(choice, cs[i]), r) for r, cs in outcomes[case]]
            best = min(near, key=lambda t: (t[0]["max_same_prefix_delta"], t[1]))
            worst = max(worst, best[0]["max_same_prefix_delta"])
            changed += bool(best[0]["token_ids_differ_at"])
            misses.append(f"{case}[{i}]")
            step = min(s for s in (first_difference(choice, cs[i]) for _, cs in outcomes[case])
                       if s is not None)
            first_steps.append(step)
            first_contexts.append(len(choice["prompt_token_ids"]) + step)
    return dict(judged=True, choices=total, members=members, non_members=total - members,
                changed_a_token=changed, worst_same_prefix_delta=worst,
                pool=sorted(runs), non_member_choices=misses,
                first_difference_step_min=min(first_steps, default=None),
                first_difference_context_min=min(first_contexts, default=None),
                prefill_token_equal=all(s > 0 for s in first_steps))


def main():
    recs = pathlib.Path(sys.argv[1])
    pool = sys.argv[2].split(",")
    arms = sys.argv[3:]
    table = {}
    for arm in arms:
        for name in (arm, f"{arm}-r2"):
            if not (recs / f"{name}.eq.summary.json").exists():
                continue
            args = engine_args(recs, arm)
            row = dict(eager=args.get("eager"), compilation=args.get("compilation_json"),
                       spec=args.get("spec_json"), extra_env=args.get("extra_env"),
                       max_num_seqs=args.get("max_num_seqs"), image=args.get("image"),
                       tree_sha=args.get("tree_sha"), src_tree=args.get("src_tree"),
                       model=args.get("model"), dispatch=dispatch(recs, arm),
                       equality=judge(recs, name, pool))
            table[name] = row
            e = row["equality"]
            d = row["dispatch"] or {}
            if not e["judged"]:
                print(f"{name:10s} NOT JUDGED ({e['why']})  captured {d.get('captured')}",
                      flush=True)
                continue
            print(f"{name:10s} members {e['members']}/{e['choices']}  changed-token "
                  f"{e['changed_a_token']}  worst {e['worst_same_prefix_delta']:.6g}  "
                  f"first-diff step>={e['first_difference_step_min']} "
                  f"ctx>={e['first_difference_context_min']}  "
                  f"captured {d.get('captured')}  full {d.get('full_replays')}", flush=True)
    out = recs / f"summary-{pool[0]}.json"
    out.write_text(json.dumps(table, indent=1, sort_keys=True))
    print("wrote", out)


if __name__ == "__main__":
    main()
