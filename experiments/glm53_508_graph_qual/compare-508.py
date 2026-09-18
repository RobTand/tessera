"""tessera#508: compare two arms' smoke records — shared-prefix |dlogprob| stats.

Metric definitions follow the historical eager-graph-comparison.json: for each
prompt, walk generated tokens while the token ids agree; report mean/max
same-position |dlogprob| over that shared prefix, plus first divergence token.
Usage: compare-508.py RECEIPTS_DIR EAGER_ARM GRAPH_ARM
"""
import json, math, pathlib, sys

recs = pathlib.Path(sys.argv[1])
a, b = sys.argv[2], sys.argv[3]

def load(arm, name):
    return json.loads((recs / f'{arm}.{name}.json').read_text())['choices'][0]

report = {}
for name in ('short1', 'short2', 'long'):
    ca, cb = load(a, name), load(b, name)
    ta, tb = ca['token_ids'], cb['token_ids']
    la, lb = ca['logprobs']['token_logprobs'], cb['logprobs']['token_logprobs']
    shared = 0
    for i in range(min(len(ta), len(tb))):
        if ta[i] != tb[i]:
            break
        shared += 1
    deltas = [abs(la[i] - lb[i]) for i in range(shared)]
    report[name] = dict(
        equal_text=ca['text'] == cb['text'],
        equal_token_ids=ta == tb,
        shared_generated_prefix_tokens=shared,
        mean_shared_prefix_logprob_abs=(sum(deltas) / len(deltas)) if deltas else None,
        max_shared_prefix_logprob_abs=max(deltas) if deltas else None,
        first_token_logprob_abs=abs(la[0] - lb[0]) if la and lb else None,
        eager_first_token=ca['logprobs']['tokens'][0],
        graph_first_token=cb['logprobs']['tokens'][0])
    print(name, report[name], flush=True)

(recs / f'compare-{b}-vs-{a}.json').write_text(json.dumps(report, indent=1))
worst = max((r['max_shared_prefix_logprob_abs'] or 0) for r in report.values())
print(f'WORST max |dlogprob| = {worst:.6g} nats', flush=True)
