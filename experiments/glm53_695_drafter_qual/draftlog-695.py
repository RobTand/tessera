"""tessera#695: compare an arm's drafter proposals with an eager pool's, at the same context.

  draftlog-695.py RECEIPTS ARM EAGER_ARM[,EAGER_ARM...] [OUT_JSON]

Reads RECEIPTS/<arm>.draft.<pid>.jsonl, written by the T695_DRAFT_LOG research
hook (../glm53_508_graph_qual/digest/usercustomize.py), for ARM and for each
eager arm. A request's context at a step is its prompt (by hash) plus every
token the target emitted through that step: the drafts it accepted from the
previous step and the token it sampled. Drafts are compared only where two
contexts are identical, because the drafter's input is that context.

For every step of ARM whose context the eager pool also reached:

  member         ARM's k drafts equal a draft vector an eager arm proposed there.
  per_position   agreement with the pool at each draft position, where the pool
                 proposed one vector: position 0 comes from the drafter's
                 prefill pass, positions 1 and up from its decode steps, which
                 run as separate graphs.

Steps whose context the pool never reached are counted and not judged. With two
or more eager arms, each is also judged against the others, which measures the
pool's own spread (the stub's eager serve is not bitwise repeatable).

  identity       for two arms, over every pair of steps (one from each) at an
                 identical context: the fraction whose draft token at position i
                 is the same, per position, and whose whole draft vector is.
                 ARM is paired with each eager arm, and the eager arms with each
                 other: the eager pairs are the noise floor a graph arm's
                 identity rate is read against. A context a batch case sends
                 n times in each arm contributes n*n pairs, so the rates are
                 also given with every shared context weighted once.
  common         every rate again on the contexts that ARM and every eager arm
                 reached, each context weighted once. The arms' context sets
                 differ when their probes differ (the acceptance prompts' contexts
                 agree more often than the equality suite's), so only these
                 rates compare one pair with another:
                   spread         per position, the lowest and highest identity
                                  over the eager pairs;
                   vs_eager       ARM's identity with each eager arm, and whether
                                  it lies below, inside or above that spread;
                   per_reference  with each eager arm R as the reference, ARM's
                                  identity with R minus the other eager arms'
                                  mean identity with R, per position and for the
                                  whole vector, with a 95% bootstrap interval
                                  over contexts. An interval that holds zero
                                  means ARM agrees with R as often as eager
                                  agrees with R.

Exit 0 when at least one step aligned and every aligned step is a member, 1 on a
non-member, 2 when nothing aligned.
"""
import collections
import glob
import hashlib
import json
import pathlib
import sys


def load(receipts, arm):
    """Every recorded step of an arm as (context key, drafts), per engine process file."""
    files = sorted(glob.glob(f"{receipts}/{arm}.draft.*.jsonl"))
    if not files:
        raise SystemExit(f"{arm}: no draft log under {receipts}")
    streams = []
    for path in files:
        prompt, ctx, prev = {}, {}, {}
        steps, counts, ks = [], collections.Counter(), set()
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["ev"] == "new":
                    prompt[rec["req"]] = rec["prompt_sha"]
                    ctx[rec["req"]] = (rec["prompt_sha"], 0)
                    prev.pop(rec["req"], None)
                    counts["requests"] += 1
                    continue
                k = rec["k"]
                ks.add(k)
                for req, row in zip(rec["reqs"], rec["rows"]):
                    drafts = tuple(row[:k])
                    last, sampled = row[k], row[k + 1]
                    if req not in ctx:
                        counts["unknown_request"] += 1
                        continue
                    state, emitted_n = ctx[req]
                    if state is None:
                        counts["context_lost"] += 1
                        continue
                    if sampled == 0:
                        # A prefill chunk: nothing emitted, the drafts are discarded.
                        counts["no_emission"] += 1
                        continue
                    if sampled > 1:
                        before = prev.get(req)
                        if before is None or sampled - 1 > len(before):
                            ctx[req] = (None, emitted_n)
                            counts["context_lost"] += 1
                            continue
                        emitted = list(before[:sampled - 1]) + [last]
                    else:
                        emitted = [last]
                    for token in emitted:
                        state = hashlib.sha1(f"{state}:{token}".encode()).hexdigest()
                    emitted_n += len(emitted)
                    ctx[req] = (state, emitted_n)
                    prev[req] = drafts
                    steps.append(((state, emitted_n), drafts))
                    counts["steps"] += 1
                    counts["accepted_tokens"] += sampled - 1
        streams.append(dict(file=path, k=sorted(ks), counts=dict(counts), steps=steps))
    return streams


def pool_of(streams_by_arm):
    pool = collections.defaultdict(set)
    for streams in streams_by_arm:
        for stream in streams:
            for key, drafts in stream["steps"]:
                pool[key[0]].add(drafts)
    return pool


def judge(steps, pool):
    aligned = member = single = 0
    unaligned = 0
    pos_n, pos_eq = collections.Counter(), collections.Counter()
    nonmembers = []
    for (state, emitted_n), drafts in steps:
        seen = pool.get(state)
        if not seen:
            unaligned += 1
            continue
        aligned += 1
        if drafts in seen:
            member += 1
        elif len(nonmembers) < 20:
            nonmembers.append(dict(context=state[:12], emitted=emitted_n, drafts=list(drafts),
                                   pool=[list(d) for d in sorted(seen)]))
        if len(seen) == 1:
            single += 1
            (ref,) = seen
            for i, (a, b) in enumerate(zip(drafts, ref)):
                pos_n[i] += 1
                pos_eq[i] += a == b
    return dict(steps=len(steps), aligned=aligned, unaligned=unaligned, member=member,
                nonmember=aligned - member, aligned_single_vector=single,
                per_position_agreement=[pos_eq[i] / pos_n[i] for i in sorted(pos_n)],
                nonmember_examples=nonmembers)


def by_context(steps):
    out = collections.defaultdict(list)
    for (state, _), drafts in steps:
        out[state].append(drafts)
    return out


def agreement(a_list, b_list):
    """Per-position and whole-vector agreement over every pair at one context."""
    n = len(a_list) * len(b_list)
    k = min(len(a_list[0]), len(b_list[0]))
    pos = [sum(a[i] == b[i] for a in a_list for b in b_list) / n for i in range(k)]
    whole = sum(a == b for a in a_list for b in b_list) / n
    return n, pos, whole


def identity(steps_a, steps_b):
    """Per-position draft identity over every pair of steps at an identical context."""
    by_a, by_b = by_context(steps_a), by_context(steps_b)
    shared = by_a.keys() & by_b.keys()
    pairs = 0
    whole = 0.0
    pos_eq = collections.Counter()
    ctx_pos, ctx_whole = collections.Counter(), 0.0
    for state in shared:
        n, pos, w = agreement(by_a[state], by_b[state])
        pairs += n
        whole += w * n
        ctx_whole += w
        for i, v in enumerate(pos):
            pos_eq[i] += v * n
            ctx_pos[i] += v
    c = len(shared)
    return dict(contexts_a=len(by_a), contexts_b=len(by_b), contexts_shared=c, pairs=pairs,
                per_position_identity=[pos_eq[i] / pairs for i in sorted(pos_eq)] if pairs else [],
                vector_identity=whole / pairs if pairs else None,
                per_position_identity_context_weighted=[ctx_pos[i] / c for i in sorted(ctx_pos)] if c else [],
                vector_identity_context_weighted=ctx_whole / c if c else None)


def per_reference(by_j, by_r, by_others, common, draws=2000, seed=0):
    """ARM's identity with reference R minus the other eager arms' mean identity with R."""
    import random
    rows = []
    for state in common:
        _, pj, wj = agreement(by_j[state], by_r[state])
        others = [agreement(by_o[state], by_r[state]) for by_o in by_others]
        po = [sum(o[1][i] for o in others) / len(others) for i in range(len(pj))]
        wo = sum(o[2] for o in others) / len(others)
        rows.append((pj + [wj], po + [wo]))
    if not rows:
        return dict(contexts=0)
    width = min(len(r[0]) for r in rows)
    rng = random.Random(seed)
    boot = [[rng.randrange(len(rows)) for _ in rows] for _ in range(draws)]
    out = dict(contexts=len(rows), arm=[], others=[], diff=[], diff_ci95=[],
               columns=[f"position {i}" for i in range(width - 1)] + ["whole vector"])
    for i in range(width):
        a = sum(r[0][i] for r in rows) / len(rows)
        b = sum(r[1][i] for r in rows) / len(rows)
        diffs = sorted(sum(rows[t][0][i] - rows[t][1][i] for t in idx) / len(idx) for idx in boot)
        out["arm"].append(a)
        out["others"].append(b)
        out["diff"].append(a - b)
        out["diff_ci95"].append([diffs[int(0.025 * draws)], diffs[int(0.975 * draws) - 1]])
    return out


def common_rates(judged_steps, eager_steps):
    """Every identity rate on the contexts all arms reached, each context weighted once."""
    by_j = by_context(judged_steps)
    by_e = {e: by_context(steps) for e, steps in eager_steps.items()}
    common = set(by_j)
    for by in by_e.values():
        common &= set(by)
    common = sorted(common)

    def rate(a, b):
        if not common:
            return None
        per = [agreement(a[c], b[c]) for c in common]
        width = min(len(p[1]) for p in per)
        return dict(per_position=[sum(p[1][i] for p in per) / len(per) for i in range(width)],
                    vector=sum(p[2] for p in per) / len(per))

    names = list(by_e)
    eager_pairs = {f"{a}~{b}": rate(by_e[a], by_e[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    out = dict(contexts=len(common), eager_pairs=eager_pairs)
    if eager_pairs and common:
        width = min(len(r["per_position"]) for r in eager_pairs.values())
        lo = [min(r["per_position"][i] for r in eager_pairs.values()) for i in range(width)]
        hi = [max(r["per_position"][i] for r in eager_pairs.values()) for i in range(width)]
        vlo = min(r["vector"] for r in eager_pairs.values())
        vhi = max(r["vector"] for r in eager_pairs.values())
        out["spread"] = dict(per_position=[[a, b] for a, b in zip(lo, hi)], vector=[vlo, vhi])
        vs = {}
        for e in names:
            r = rate(by_j, by_e[e])
            where = ["below" if x < lo[i] else "above" if x > hi[i] else "inside"
                     for i, x in enumerate(r["per_position"][:width])]
            vwhere = "below" if r["vector"] < vlo else "above" if r["vector"] > vhi else "inside"
            vs[e] = dict(r, position_vs_spread=where, vector_vs_spread=vwhere)
        out["vs_eager"] = vs
    if len(names) >= 2 and common:
        out["per_reference"] = {
            r: per_reference(by_j, by_e[r], [by_e[o] for o in names if o != r], common)
            for r in names}
    return out


def steps_of(streams):
    return [step for stream in streams for step in stream["steps"]]


def main():
    receipts, arm, eager_arms = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    out = pathlib.Path(sys.argv[4]) if len(sys.argv) > 4 else pathlib.Path(receipts) / f"{arm}.draftcmp.json"
    judged = load(receipts, arm)
    eager = {e: load(receipts, e) for e in eager_arms}
    ks = {tuple(s["k"]) for s in judged} | {tuple(s["k"]) for v in eager.values() for s in v}
    if len(ks) != 1:
        raise SystemExit(f"arms propose different draft counts {sorted(ks)}; compare like with like")
    pool = pool_of(eager.values())
    multi = sum(1 for v in pool.values() if len(v) > 1)
    record = dict(arm=arm, eager_pool=eager_arms, k=list(ks.pop()), pool_contexts=len(pool),
                  pool_contexts_with_several_vectors=multi,
                  judged=[dict(file=s["file"], counts=s["counts"], **judge(s["steps"], pool)) for s in judged])
    record["identity_vs_eager"] = {e: identity(steps_of(judged), steps_of(eager[e])) for e in eager_arms}
    record["identity_eager_pairs"] = {
        f"{a}~{b}": identity(steps_of(eager[a]), steps_of(eager[b]))
        for i, a in enumerate(eager_arms) for b in eager_arms[i + 1:]}
    record["common"] = common_rates(steps_of(judged), {e: steps_of(eager[e]) for e in eager_arms})
    if len(eager_arms) >= 2:
        record["eager_leave_one_out"] = {
            e: [judge(s["steps"], pool_of(v for o, v in eager.items() if o != e)) for s in eager[e]]
            for e in eager_arms}
    out.write_text(json.dumps(record, indent=1))
    for j in record["judged"]:
        print(f"{arm} vs {'+'.join(eager_arms)}: steps {j['steps']} aligned {j['aligned']} "
              f"member {j['member']} nonmember {j['nonmember']} unaligned {j['unaligned']} "
              f"per-position {[round(x, 4) for x in j['per_position_agreement']]}")
    for e, rows in record.get("eager_leave_one_out", {}).items():
        for j in rows:
            print(f"  eager spread {e} vs rest: aligned {j['aligned']} member {j['member']} "
                  f"nonmember {j['nonmember']} per-position {[round(x, 4) for x in j['per_position_agreement']]}")
    for name, rows in (("vs", record["identity_vs_eager"]), ("eager", record["identity_eager_pairs"])):
        for pair, j in rows.items():
            label = f"{arm} vs {pair}" if name == "vs" else f"eager {pair}"
            print(f"  identity {label}: shared contexts {j['contexts_shared']} pairs {j['pairs']} "
                  f"per-position {[round(x, 4) for x in j['per_position_identity']]} "
                  f"vector {None if j['vector_identity'] is None else round(j['vector_identity'], 4)}; "
                  f"contexts weighted once {[round(x, 4) for x in j['per_position_identity_context_weighted']]} "
                  f"vector {None if j['vector_identity_context_weighted'] is None else round(j['vector_identity_context_weighted'], 4)}")
    com = record["common"]
    fmt = lambda xs: "[" + ", ".join(f"{x:.4f}" for x in xs) + "]"
    print(f"  common contexts (reached by every arm): {com['contexts']}")
    for pair, r in com.get("eager_pairs", {}).items():
        if r:
            print(f"    eager {pair}: per-position {fmt(r['per_position'])} vector {r['vector']:.4f}")
    if "spread" in com:
        sp = com["spread"]
        print(f"    eager spread: per-position {[fmt(x) for x in sp['per_position']]} vector {fmt(sp['vector'])}")
        for e, r in com["vs_eager"].items():
            print(f"    {arm} vs {e}: per-position {fmt(r['per_position'])} {r['position_vs_spread']} "
                  f"vector {r['vector']:.4f} {r['vector_vs_spread']}")
    for r, j in com.get("per_reference", {}).items():
        if not j.get("contexts"):
            continue
        cells = "; ".join(f"{col} {a:.4f} vs {b:.4f} diff {d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                          for col, a, b, d, (lo, hi) in zip(j["columns"], j["arm"], j["others"],
                                                             j["diff"], j["diff_ci95"]))
        print(f"    reference {r}, {arm} minus the other eager arms: {cells}")
    print(f"pool contexts {len(pool)}, with several vectors {multi}; wrote {out}")
    if any(j["nonmember"] for j in record["judged"]):
        sys.exit(1)
    sys.exit(0 if all(j["aligned"] for j in record["judged"]) else 2)


if __name__ == "__main__":
    main()
