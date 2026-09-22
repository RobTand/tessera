"""tessera#508 diagnostic (not the fixed smoke set): long token-id prompts, repeated.

Sends exact-length token-id prompts (default 1500, 2100, 3649 tokens) REPEATS
times each with max_tokens 8, greedy, logprobs. Records $OUT/<arm>.len<N>.r<i>.json
and prints per length whether the repeats are bit-identical within this serve.
With prefix caching enabled a repeat hits the cache (then it is not a repeat of
the prefill); run the serve with EXTRA_ARGS=--no-enable-prefix-caching for the
within-process determinism question.  Usage: probe-long-508.py PORT OUT ARM [N,N,...] [REPEATS]
"""
import json, math, pathlib, sys, urllib.request

port = sys.argv[1]; out = pathlib.Path(sys.argv[2]); arm = sys.argv[3]
lengths = [int(x) for x in (sys.argv[4].split(',') if len(sys.argv) > 4 else '1500,2100,3649'.split(','))]
repeats = int(sys.argv[5]) if len(sys.argv) > 5 else 2
out.mkdir(parents=True, exist_ok=True)

def post(path, payload):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)

para = ('The quick brown fox jumps over the lazy dog near the riverbank while '
        'seventeen curious ravens observe the scene from a weathered oak branch. ')
ids = post('/tokenize', dict(model='glm53-stub', prompt=para * 200, add_special_tokens=False))['tokens']
assert len(ids) >= max(lengths), (len(ids), max(lengths))
summary = {}
for n in lengths:
    runs = []
    for i in range(repeats):
        res = post('/v1/completions', dict(model='glm53-stub', prompt=ids[:n], max_tokens=8,
                                           temperature=0, logprobs=1, return_token_ids=True))
        (out / f'{arm}.len{n}.r{i}.json').write_text(json.dumps(res, indent=1))
        ch = res['choices'][0]
        assert res['usage']['prompt_tokens'] == n, (n, res['usage'])
        assert all(math.isfinite(v) for v in ch['logprobs']['token_logprobs'])
        runs.append((ch['token_ids'], ch['logprobs']['token_logprobs']))
    same = all(r == runs[0] for r in runs)
    summary[n] = dict(repeat_exact=same, first_logprob=runs[0][1][0], token_ids=runs[0][0],
                      first_logprobs=[r[1][0] for r in runs])
    print(n, 'repeat_exact' if same else 'REPEATS DIFFER', [round(r[1][0], 6) for r in runs], flush=True)
(out / f'{arm}.long.summary.json').write_text(json.dumps(summary, indent=1))
