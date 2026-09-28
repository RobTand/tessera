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
    print(f"pool contexts {len(pool)}, with several vectors {multi}; wrote {out}")
    if any(j["nonmember"] for j in record["judged"]):
        sys.exit(1)
    sys.exit(0 if all(j["aligned"] for j in record["judged"]) else 2)


if __name__ == "__main__":
    main()
