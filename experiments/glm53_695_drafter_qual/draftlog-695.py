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
  paired         the same comparison on the same contexts: with an eager arm R
                 as the reference and another eager arm O, over the contexts
                 ARM, R and O all reached, ARM's per-context identity with R
                 minus O's, per position (contexts weighted once), with a 95%
                 bootstrap interval over contexts. An interval that holds zero
                 means ARM agrees with eager as often as eager agrees with
                 itself.

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


def paired(steps_j, steps_r, steps_o, draws=2000, seed=0):
    """ARM's identity with reference R minus eager arm O's, on the contexts all three reached."""
    import random
    by_j, by_r, by_o = by_context(steps_j), by_context(steps_r), by_context(steps_o)
    common = sorted(by_j.keys() & by_r.keys() & by_o.keys())
    if not common:
        return dict(contexts=0)
    rows = []
    for state in common:
        _, pj, wj = agreement(by_j[state], by_r[state])
        _, po, wo = agreement(by_o[state], by_r[state])
        rows.append((pj + [wj], po + [wo]))
    width = min(len(r[0]) for r in rows)
    mean = lambda idx, which: sum(rows[t][which][i] for t in idx) / len(idx)
    all_idx = range(len(rows))
    rng = random.Random(seed)
    out = dict(contexts=len(common), arm=[], other=[], diff=[], diff_ci95=[])
    boot = [[rng.randrange(len(rows)) for _ in rows] for _ in range(draws)]
    for i in range(width):
        a = sum(rows[t][0][i] for t in all_idx) / len(rows)
        b = sum(rows[t][1][i] for t in all_idx) / len(rows)
        diffs = sorted(sum(rows[t][0][i] - rows[t][1][i] for t in idx) / len(idx) for idx in boot)
        out["arm"].append(a)
        out["other"].append(b)
        out["diff"].append(a - b)
        out["diff_ci95"].append([diffs[int(0.025 * draws)], diffs[int(0.975 * draws) - 1]])
    out["columns"] = [f"position {i}" for i in range(width - 1)] + ["whole vector"]
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
    record["paired"] = {
        f"ref {r}, other {o}": paired(steps_of(judged), steps_of(eager[r]), steps_of(eager[o]))
        for r in eager_arms for o in eager_arms if o != r}
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
    for name, j in record["paired"].items():
        if not j.get("contexts"):
            print(f"  paired {name}: no common context")
            continue
        cells = "; ".join(f"{col} {a:.4f} vs {b:.4f} diff {d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                          for col, a, b, d, (lo, hi) in zip(j["columns"], j["arm"], j["other"],
                                                             j["diff"], j["diff_ci95"]))
        print(f"  paired {name} over {j['contexts']} contexts: {cells}")
    print(f"pool contexts {len(pool)}, with several vectors {multi}; wrote {out}")
    if any(j["nonmember"] for j in record["judged"]):
        sys.exit(1)
    sys.exit(0 if all(j["aligned"] for j in record["judged"]) else 2)


if __name__ == "__main__":
    main()
