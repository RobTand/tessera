import json, collections
m = json.load(open('/mnt/shared/tessera-runs/allocated/qwen3-0.6b-alloc-4.0/tessera_serving_manifest.json'))
t = m['totals']
print('alloc totals', {k: t[k] for k in t if 'bpp' in k or 'bytes' in k})
fam = collections.Counter()
rung = collections.Counter()
for name, mod in m['modules'].items():
    for r in mod['roles']:
        fam[r['grid']] += 1
        rung[(r['tensor'].split('.')[-2], r['grid'], r['q256'])] += 1
print('modules', len(m['modules']), 'roles', sum(fam.values()))
print('by grid', dict(fam))
for k, v in sorted(rung.items()):
    print('  ', k, v)
p = json.load(open('/home/rob/tmp/alloc-plans/plan_full_4.0_bcast.json.provenance.json'))
print('coverage', p['coverage'])
print('fused_disagreements', p['fused_disagreements'])
print('totals', p['totals'])
