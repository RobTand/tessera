"""tessera#508: matrix table over every arm's smoke records in a receipts dir.

For each arm with all three records, compares against BASE (default eager1) and
against every other arm sharing its config prefix (repeat determinism): shared
generated prefix length, first-token |dlogprob|, max shared-prefix |dlogprob|.
Arms whose smoke never wrote a record are listed from engine-args-<arm>.txt
with their recorded smoke_rc.  Usage: matrix-508.py RECEIPTS_DIR [BASE]
"""
import json, pathlib, re, sys

recs = pathlib.Path(sys.argv[1]); base = sys.argv[2] if len(sys.argv) > 2 else 'eager1'
PROMPTS = ('short1', 'short2', 'long')

def load(arm, name):
    p = recs / f'{arm}.{name}.json'
    return json.loads(p.read_text())['choices'][0] if p.exists() else None

def cmp(ca, cb):
    ta, tb = ca['token_ids'], cb['token_ids']
    la, lb = ca['logprobs']['token_logprobs'], cb['logprobs']['token_logprobs']
    shared = 0
    for i in range(min(len(ta), len(tb))):
        if ta[i] != tb[i]:
            break
        shared += 1
    d = [abs(la[i] - lb[i]) for i in range(shared)]
    return shared, len(ta), abs(la[0] - lb[0]), (max(d) if d else 0.0)

def fmt(a, b, name):
    ca, cb = load(a, name), load(b, name)
    if ca is None or cb is None:
        return 'n/a'
    s, n, f, m = cmp(ca, cb)
    return f'{s}/{n} f{f:.4g} m{m:.4g}' if s < n or m > 0 else f'{s}/{n} exact'

arms = sorted({p.name.split('.')[0] for p in recs.glob('engine-args-*.txt')} - {'engine-args-EAGER0', 'engine-args-EAGER1'})
arms = [a[len('engine-args-'):] for a in sorted(p.name[:-4] for p in recs.glob('engine-args-*.txt')) if not a.endswith(('EAGER0', 'EAGER1'))]
print(f'{"arm":10s} {"cfg":52s} {"rc":>3s}  {"vs " + base + " short1":26s} {"long":26s}')
cfgs = {}
for arm in arms:
    meta = dict(l.split('=', 1) for l in (recs / f'engine-args-{arm}.txt').read_text().splitlines() if '=' in l)
    cfg = ('eager' if meta.get('eager') == '1' else 'graph') + ' ' + (meta.get('compilation_json') or '{}') + (' ' + meta['bisect_env'] if meta.get('bisect_env') else '')
    cfgs[arm] = cfg
    rc = meta.get('smoke_rc', '?')
    print(f'{arm:10s} {cfg[:52]:52s} {rc:>3s}  {fmt(base, arm, "short1"):26s} {fmt(base, arm, "long"):26s}')
print('\nrepeat pairs (same config):')
for i, a in enumerate(arms):
    for b in arms[i + 1:]:
        if cfgs[a] == cfgs[b] and load(a, 'short1') and load(b, 'short1'):
            print(f'  {a} vs {b}: short1 {fmt(a, b, "short1")} | short2 {fmt(a, b, "short2")} | long {fmt(a, b, "long")}')
