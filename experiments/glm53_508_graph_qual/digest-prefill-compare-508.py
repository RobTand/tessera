"""tessera#508: place a prefill difference on the first module that differs.

For whole-tensor digest serves (T508_DIGEST_FULL=1) that ran the equality suite,
every eager forward left one line with exact integer checksums of every hooked
module's first tensor input and output. A prefill line is identified by its token
count: every count above the largest batch size (8) is a prefill or a prefill chunk
(decode lines carry one token per request). This pairs prefill lines by token
count and occurrence order:

  one arm:   the suite ran twice (eq, eq2): occurrence i of a count in the first
             half against occurrence i in the second half;
  two arms:  occurrence i of a count in arm A against occurrence i in arm B.

For every pair that differs it names the first differing slot in the order the
modules ran and the origin: the first slot whose module output differs while
that module's input matched. The indexer's output is vLLM's persistent top-k
buffer, whose rows past the forward's tokens hold earlier requests' values: it
is compared on its first row only and never named as an origin. Writes
RECEIPTS/digpre-<A>[-vs-<B>].json.

  digest-prefill-compare-508.py RECEIPTS ARM_A [ARM_B]
"""
import json
import pathlib
import sys
from collections import defaultdict

recs = pathlib.Path(sys.argv[1])
arm_a = sys.argv[2]
arm_b = sys.argv[3] if len(sys.argv) > 3 else None
MAX_BATCH = 8


def prefill_lines(arm):
    rows = [json.loads(line) for line in (recs / f"{arm}.dig.jsonl").read_text().splitlines()
            if line.strip()]
    by_count = defaultdict(list)
    for row in rows:
        if row.get("real") and row["real"] > MAX_BATCH and row["digests"]:
            by_count[row["real"]].append(row)
    return by_count


def rows_of(value):
    """Element count of a whole-tensor digest ("sum.sum.numel|first|last"), or None."""
    try:
        return int(str(value).split("|")[0].split(".")[2], 16)
    except (IndexError, ValueError):
        return None


# The sparse-attention indexer returns vLLM's persistent top-k buffer: all
# max_num_batched_tokens rows, of which only the first num_tokens are reset and
# written by this forward. A whole-tensor checksum of it therefore includes rows
# left by earlier requests, so it differs between two runs of the same prompt
# whenever their histories differ. Its first-row checksum is exact; its whole
# and last-row checksums are not evidence, and it is never named as an origin.
PERSISTENT_BUFFERS = (".indexer.indexer_op|out", ".indexer|out")


def persistent(key):
    return key.split("#")[0].endswith(PERSISTENT_BUFFERS)


def diff(fa, fb):
    keys = [k for k in fa["digests"] if k in fb["digests"]]
    differ = [k for k in keys if fa["digests"][k] != fb["digests"][k]
              and not (persistent(k) and str(fa["digests"][k]).split("|")[1:2]
                       == str(fb["digests"][k]).split("|")[1:2])]
    origin = None
    for k in differ:
        base, _, rep = k.partition("#")
        if persistent(k):
            continue
        if base.endswith("|out") and not base.startswith("language_model.logits_processor"):
            twin = base[:-4] + "|in" + (f"#{rep}" if rep else "")
            if twin in fa["digests"] and fa["digests"][twin] == fb["digests"].get(twin):
                origin = k
                break
    # The logits processor runs after the root forward, so its slots hold the
    # previous step's values; they are never an origin and are listed last.
    model = [k for k in differ if not k.startswith("language_model.logits_processor")]
    return dict(n_slots=len(keys), n_differ=len(model), first=model[0] if model else None,
                origin=origin, differ=model[:30])


pairs = []
if arm_b is None:
    lines = prefill_lines(arm_a)
    for count, occ in sorted(lines.items()):
        if len(occ) % 2:
            print(f"count {count}: {len(occ)} occurrences, not two equal passes; skipped")
            continue
        half = len(occ) // 2
        pairs += [(count, i, occ[i], occ[half + i]) for i in range(half)]
    tag = arm_a
else:
    la, lb = prefill_lines(arm_a), prefill_lines(arm_b)
    for count in sorted(set(la) | set(lb)):
        if len(la.get(count, [])) != len(lb.get(count, [])):
            print(f"count {count}: {len(la.get(count, []))} vs {len(lb.get(count, []))} occurrences; "
                  "paired in order up to the shorter")
        pairs += [(count, i, a, b) for i, (a, b) in enumerate(zip(la.get(count, []), lb.get(count, [])))]
    tag = f"{arm_a}-vs-{arm_b}"

report = []
for count, i, a, b in pairs:
    d = diff(a, b)
    report.append(dict(tokens=count, occurrence=i, step_a=a["step"], step_b=b["step"], **d))
    if d["n_differ"]:
        print(f"prefill {count:5d} #{i}: {d['n_differ']}/{d['n_slots']} slots differ; "
              f"first {d['first']}; origin {d['origin']}", flush=True)
    else:
        print(f"prefill {count:5d} #{i}: identical ({d['n_slots']} slots)", flush=True)
(recs / f"digpre-{tag}.json").write_text(json.dumps(report, indent=1))
