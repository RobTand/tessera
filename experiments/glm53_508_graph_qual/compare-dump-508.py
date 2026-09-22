"""tessera#508: diff repeats of one request in a TESSERA_RESEARCH_GLM53_NOPE_DUMP file.

Records are one JSON line per Tessera forward_mqa call (digests of q, topk,
topk_sorted, physical, counts, out). An episode starts at a prefill record
(num_tokens > 1) on the first MLA layer seen; consecutive episodes with the same
shape sequence are compared record by record and the first differing field per
record is reported. Usage: compare-dump-508.py DUMP.jsonl
"""
import json, sys, collections

recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
recs = [r for r in recs if r.get("site") == "forward_mqa"]
layers = []
for r in recs:
    if r["layer"] not in layers:
        layers.append(r["layer"])
first_layer = layers[0] if layers else None
episodes, cur = [], None
for r in recs:
    if r["num_tokens"] > 1 and r["layer"] == first_layer:
        cur = []; episodes.append(cur)
    if cur is not None:
        cur.append(r)
print(f"{len(recs)} records, layers={layers}, {len(episodes)} episodes:",
      [(e[0]["num_tokens"], len(e)) for e in episodes])
FIELDS = ["q", "topk", "topk_sorted", "physical", "counts", "out"]
for i in range(len(episodes) - 1):
    a, b = episodes[i], episodes[i + 1]
    sig = lambda e: [(r["num_tokens"], r["layer"]) for r in e]
    if sig(a) != sig(b):
        print(f"episodes {i},{i+1}: different shape sequence; skipped"); continue
    print(f"== episodes {i} vs {i+1} (prefill rows {a[0]['num_tokens']}, {len(a)} calls)")
    verdict = collections.Counter()
    for k, (ra, rb) in enumerate(zip(a, b)):
        diff = [f for f in FIELDS if ra[f]["sha"] != rb[f]["sha"]]
        same = [f for f in FIELDS if f not in diff]
        tag = "EQUAL" if not diff else "differs:" + ",".join(diff)
        verdict[tag] += 1
        if k < 6 or diff:
            print(f"  call {k:2d} n={ra['num_tokens']:5d} {ra['layer']}: {tag}"
                  + (f"  (equal: {','.join(same)})" if diff else ""))
    print("  summary:", dict(verdict))
