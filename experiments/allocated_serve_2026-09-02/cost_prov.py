import pickle

cost = pickle.load(open('/mnt/shared/tessera-runs/pq-continuous/qwen06b/cost.pkl', 'rb'))
print('top-level keys:', sorted(k for k in cost if k != 'costs'))
for k in cost:
    if k == 'costs':
        continue
    v = cost[k]
    print(f'  {k}: {str(v)[:400]}')
q = next(iter(cost['costs']))
f = next(iter(cost['costs'][q]))
print('example cost row:', q, f, cost['costs'][q][f])
