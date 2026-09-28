"""tessera#508: place a graph-vs-eager divergence on the first module that differs.

Reads two arms' $OUT/<arm>.dig.jsonl (DIGEST=1 serves, digest/usercustomize.py)
and aligns their LAST n forwards (the probe's requests; start-up profile and
capture forwards differ between modes and are dropped). For each aligned pair
of forwards with the same real token count it lists, in the order the modules
first ran, every slot whose digest differs, and names the first one: a module
whose output differs while its input matched is where the difference starts.
Writes $OUT/compare-dig-<B>-vs-<A>.json.

  digest-compare-508.py RECEIPTS_DIR ARM_A ARM_B N_FORWARDS
"""
import json
import pathlib
import sys

recs, a, b, n = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])


def load(arm):
    rows = [json.loads(line) for line in (recs / f"{arm}.dig.jsonl").read_text().splitlines() if line.strip()]
    return rows[-n:]


ra, rb = load(a), load(b)
report = {"arms": [a, b], "forwards": []}
for i, (fa, fb) in enumerate(zip(ra, rb)):
    entry = dict(index=i, real=[fa["real"], fb["real"]], padded=[fa["padded"], fb["padded"]],
                 mode=[fa["mode"], fb["mode"]])
    if fa["real"] != fb["real"]:
        entry["misaligned"] = True
        report["forwards"].append(entry)
        continue
    keys = [k for k in fa["digests"] if k in fb["digests"]]
    differ = [k for k in keys if fa["digests"][k] != fb["digests"][k]]
    entry["n_slots"] = len(keys)
    entry["differ"] = differ
    first_out = None
    for k in differ:
        if k.endswith("|out"):
            twin = k[:-4] + "|in"
            if twin in fa["digests"] and fa["digests"][twin] == fb["digests"].get(twin):
                first_out = k
                break
    entry["first_output_diff_with_equal_input"] = first_out
    entry["first_diff"] = differ[0] if differ else None
    report["forwards"].append(entry)
    print(f"fwd {i:3d} real {fa['real']} padded {fa['padded']}/{fb['padded']} "
          f"{fa['mode']}/{fb['mode']}: {len(differ)}/{len(keys)} differ; "
          f"first {entry['first_diff']}; origin {first_out}", flush=True)
(recs / f"compare-dig-{b}-vs-{a}.json").write_text(json.dumps(report, indent=1))
