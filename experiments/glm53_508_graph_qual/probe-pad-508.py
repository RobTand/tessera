"""tessera#508 diagnostic (not the fixed smoke set): exact-length token-id prompts.

Sends prompts of exactly N prompt tokens (token-id arrays taken from a fixed
paragraph's tokenization on the served model) so a piecewise-graph arm can be
compared against eager at lengths that ARE capture sizes (no padding) and at
lengths that are padded up to one. Writes $OUT/<arm>.pad<N>.json per length and
$OUT/<arm>.pad.summary.json.  Usage: probe-pad-508.py PORT OUT ARM [N,N,...]
"""
import json, math, pathlib, sys, urllib.request

port = sys.argv[1]; out = pathlib.Path(sys.argv[2]); arm = sys.argv[3]
lengths = [int(x) for x in (sys.argv[4].split(',') if len(sys.argv) > 4 else
           '1,2,3,4,5,6,7,8,12,16,17,24,32'.split(','))]
out.mkdir(parents=True, exist_ok=True)

def post(path, payload):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)

para = ('The quick brown fox jumps over the lazy dog near the riverbank while '
        'seventeen curious ravens observe the scene from a weathered oak branch. ') * 4
ids = post('/tokenize', dict(model='glm53-stub', prompt=para, add_special_tokens=False))['tokens']
assert len(ids) >= max(lengths), (len(ids), max(lengths))
summary = {}
for n in lengths:
    res = post('/v1/completions', dict(model='glm53-stub', prompt=ids[:n], max_tokens=16,
                                       temperature=0, logprobs=1, return_token_ids=True))
    (out / f'{arm}.pad{n}.json').write_text(json.dumps(res, indent=1))
    ch = res['choices'][0]
    assert res['usage']['prompt_tokens'] == n, (n, res['usage'])
    assert all(math.isfinite(v) for v in ch['logprobs']['token_logprobs'])
    summary[n] = dict(first_logprob=ch['logprobs']['token_logprobs'][0], first_token=ch['logprobs']['tokens'][0],
                      token_ids=ch['token_ids'])
    print(n, summary[n]['first_logprob'], repr(summary[n]['first_token']), flush=True)
(out / f'{arm}.pad.summary.json').write_text(json.dumps(summary, indent=1))
