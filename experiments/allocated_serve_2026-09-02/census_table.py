import glob
import json
import os

rows = []
for path in sorted(glob.glob('/home/rob/tessera-runs/allocated/census_*.json')):
    d = json.load(open(path))
    name = os.path.basename(path)[len('census_'):-len('.json')]
    for phase in ('prefill', 'decode'):
        h = d['histogram'][phase]
        for r in h['routes']:
            rows.append((name, d['verdict'], len(d.get('problems', [])), phase,
                         r['modules'], h['other_route_modules'], r['policy'],
                         r['contract'], r['decoder'], r['symbol'], r['state'],
                         d['compiled'], d['versions']['tessera_commit'][:7]))
for r in rows:
    print(' | '.join(str(x) for x in r))
