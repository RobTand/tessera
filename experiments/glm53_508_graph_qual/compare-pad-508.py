"""tessera#508 diagnostic compare for probe-pad-508.py records: per exact prompt
length, shared-prefix max |dlogprob| and first-token delta between two arms.
Usage: compare-pad-508.py OUT BASE_ARM ARM"""
import json, pathlib, sys
recs = pathlib.Path(sys.argv[1]); a, b = sys.argv[2], sys.argv[3]
report = {}
for f in sorted(recs.glob(f'{a}.pad*.json')):
    n = f.name[len(a) + 4:-5]
    if not n.isdigit(): continue
    g = recs / f'{b}.pad{n}.json'
    if not g.exists(): continue
    ca = json.loads(f.read_text())['choices'][0]; cb = json.loads(g.read_text())['choices'][0]
    ta, tb = ca['token_ids'], cb['token_ids']
    la, lb = ca['logprobs']['token_logprobs'], cb['logprobs']['token_logprobs']
    shared = 0
    for i in range(min(len(ta), len(tb))):
        if ta[i] != tb[i]: break
        shared += 1
    d = [abs(la[i] - lb[i]) for i in range(shared)]
    report[int(n)] = dict(shared=shared, n=len(ta), first=abs(la[0] - lb[0]), max=max(d) if d else None)
for n in sorted(report):
    print(f'len {n:3d}: shared {report[n]["shared"]:2d}/{report[n]["n"]}  first |d| {report[n]["first"]:.6g}  max |d| {report[n]["max"]}')
(recs / f'compare-pad-{b}-vs-{a}.json').write_text(json.dumps(report, indent=1))
