"""t508: decode throughput probe - 8 concurrent greedy decodes, 128 tokens each."""
import json, time, urllib.request, sys
port, arm, out = sys.argv[1], sys.argv[2], sys.argv[3]
para = ('The quick brown fox jumps over the lazy dog near the riverbank while '
        'seventeen curious ravens observe the scene from a weathered oak branch. ')
# 8 concurrent sequences, each a distinct long-context prompt (forces decode-dominant phase)
reqs = []
for i in range(8):
    p = (f'Article {i}. ' + para * 60).strip()
    payload = dict(model='glm53-stub', prompt=p, max_tokens=128,
                   temperature=0, logprobs=0, return_token_ids=True)
    reqs.append(urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions',
                 data=json.dumps(payload).encode(),
                 headers={'Content-Type': 'application/json'}))
t0 = time.monotonic()
rs = [urllib.request.urlopen(r, timeout=600) for r in reqs]
res = [json.load(x) for x in rs]
dt = time.monotonic() - t0
for x in rs: x.close()
toks = sum(r['usage']['completion_tokens'] for r in res)
# second pass to warm any allocator path
t1 = time.monotonic()
rs = [urllib.request.urlopen(r, timeout=600) for r in reqs]
res2 = [json.load(x) for x in rs]
dt2 = time.monotonic() - t1
for x in rs: x.close()
toks2 = sum(r['usage']['completion tokens' if False else 'completion_tokens'] for r in res2)
report = dict(arm=arm, n=8, max_tokens=128, pass1=dict(elapsed_s=dt, tokens=toks, tok_per_s=toks/dt),
             pass2=dict(elapsed_s=dt2, tokens=toks2, tok_per_s=toks2/dt2))
print(json.dumps(report, indent=1))
open(out, 'w').write(json.dumps(report, indent=1))
