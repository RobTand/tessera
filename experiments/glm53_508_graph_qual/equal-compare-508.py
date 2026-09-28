"""tessera#508: compare two arms' equality-suite records, bit for bit.

For every case and every choice: are the generated token ids equal, and is
every top-20 logprob entry at every position the same float (JSON round-trips
a Python float exactly, so equal parsed values are equal bits)? Where they
differ, report the first differing position, the number of shared generated
tokens, and the largest |dlogprob| of the sampled token over the shared prefix.

Suite v2 records each case's engine-step structure; a batched case is only
comparable across arms when both ran it in the same steps (same_steps).

  equal-compare-508.py RECEIPTS_DIR ARM_A ARM_B     -> compare-eq-<B>-vs-<A>.json
"""
import json
import pathlib
import sys

recs, a, b = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sa = json.loads((recs / f"{a}.eq.summary.json").read_text())
sb = json.loads((recs / f"{b}.eq.summary.json").read_text())


def choices(arm, case):
    res = json.loads((recs / f"{arm}.eq.{case}.json").read_text())
    return sorted(res["choices"], key=lambda c: c["index"])


report, exact_all = {"_admission": [sa.get("_admission"), sb.get("_admission")]}, True
for case in sa:
    if case.startswith("_") or case not in sb:
        continue
    rows = []
    for ca, cb in zip(choices(a, case), choices(b, case)):
        ta, tb = ca["token_ids"], cb["token_ids"]
        shared = 0
        while shared < min(len(ta), len(tb)) and ta[shared] == tb[shared]:
            shared += 1
        la, lb = ca["logprobs"]["token_logprobs"], cb["logprobs"]["token_logprobs"]
        pa, pb = ca["logprobs"]["top_logprobs"], cb["logprobs"]["top_logprobs"]
        first_top_diff = next((i for i in range(min(len(pa), len(pb))) if pa[i] != pb[i]), None)
        top_max = 0.0
        for i in range(shared):
            for tok, v in pa[i].items():
                if tok in pb[i]:
                    top_max = max(top_max, abs(v - pb[i][tok]))
        exact = ta == tb and pa == pb
        exact_all &= exact
        rows.append(dict(index=ca["index"], exact=exact, shared_generated_tokens=shared,
                         generated=len(ta), first_top20_difference=first_top_diff,
                         max_sampled_logprob_abs=max((abs(la[i] - lb[i]) for i in range(shared)), default=0.0),
                         max_top20_logprob_abs_shared=top_max))
    # Engine-step structure (suite v2 records it): the same number of steps and
    # tokens is the precondition for comparing a batched case across arms.
    ka = {k: v for k, v in (sa[case].get("steps") or {}).items() if k.startswith("iteration_tokens")}
    kb = {k: v for k, v in (sb[case].get("steps") or {}).items() if k.startswith("iteration_tokens")}
    same_steps = (ka == kb) if ka and kb else None
    report[case] = dict(batch=sa[case]["batch"], exact=all(r["exact"] for r in rows),
                        same_steps=same_steps, steps=[ka, kb], choices=rows)
    worst = max(r["max_top20_logprob_abs_shared"] for r in rows)
    steps_note = {None: "", True: "  steps=", False: "  STEPS DIFFER"}[same_steps]
    print(f"{case:11s} batch {sa[case]['batch']}: {'EXACT' if report[case]['exact'] else 'DIFF'}"
          f"  shared {[r['shared_generated_tokens'] for r in rows]}  max|d| {worst:.6g}{steps_note}", flush=True)
report["_all_exact"] = exact_all
(recs / f"compare-eq-{b}-vs-{a}.json").write_text(json.dumps(report, indent=1))
print("ALL EXACT" if exact_all else "NOT ALL EXACT", flush=True)
