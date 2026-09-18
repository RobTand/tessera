"""tessera#508 fixed prompt set: greedy text + logprobs + token ids per arm.

Same prompts/params as the 2026-09-13 divergence arms (campaign-takeover
smoke.py), plus return_token_ids so token ids are recorded, not just text.
Run on the serving host against 127.0.0.1:$PORT; writes $OUT/<arm>.<name>.json.
"""
import json, math, pathlib, sys, urllib.request

port = sys.argv[1] if len(sys.argv) > 1 else "8139"
out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "/home/rob/tmp/t508-serve/receipts")
arm = sys.argv[3] if len(sys.argv) > 3 else "arm"
out.mkdir(parents=True, exist_ok=True)

para = ('The quick brown fox jumps over the lazy dog near the riverbank while '
        'seventeen curious ravens observe the scene from a weathered oak branch. ')
PROMPTS = [('short1', 'The capital of France is', 32),
           ('short2', 'The capital of France is', 32),
           ('long', para * 130 + '\nIn one word, what animal jumps?', 16)]

summary = {}
for name, prompt, max_tokens in PROMPTS:
    payload = dict(model='glm53-stub', prompt=prompt, max_tokens=max_tokens,
                   temperature=0, logprobs=1, return_token_ids=True)
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions',
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.load(response)
    (out / f'{arm}.{name}.json').write_text(json.dumps(result, indent=1))
    choice = result['choices'][0]
    assert result['usage']['completion_tokens'] == max_tokens, (name, result['usage'])
    assert all(math.isfinite(v) for v in choice['logprobs']['token_logprobs']), name
    assert choice.get('token_ids') is not None, f'{name}: no token_ids'
    summary[name] = dict(prompt_tokens=result['usage']['prompt_tokens'],
                         completion_tokens=result['usage']['completion_tokens'],
                         first_token=choice['logprobs']['tokens'][0],
                         first_logprob=choice['logprobs']['token_logprobs'][0])
    print(name, summary[name], flush=True)

a = json.loads((out / f'{arm}.short1.json').read_text())['choices'][0]
b = json.loads((out / f'{arm}.short2.json').read_text())['choices'][0]
assert a['text'] == b['text'] and a['logprobs'] == b['logprobs'] and a['token_ids'] == b['token_ids'], \
    'short1 != short2: repeat not deterministic'
assert summary['long']['prompt_tokens'] > 2048, summary['long']
summary['repeat_exact'] = True
(out / f'{arm}.summary.json').write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1), flush=True)
