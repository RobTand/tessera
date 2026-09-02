"""Where does the allocation claim its layer-0 win, and where does it pay?

Both sides re-measured (not interpolated): 0.5 * h_trace * measured output MSE,
on layer 0 only -- exactly the seven units the served separator pair contains.
"""
import json
import pickle

ROOT = '/mnt/shared/tessera-runs/pq-continuous/qwen06b'
ALLOC = {'self_attn.q_proj': 1083, 'self_attn.k_proj': 1083, 'self_attn.v_proj': 1083,
         'self_attn.o_proj': 934, 'mlp.gate_proj': 1107, 'mlp.up_proj': 1107,
         'mlp.down_proj': 749}
UNIF = 1006

stats = pickle.load(open(f'{ROOT}/probe.pkl', 'rb'))
stats = stats.get('stats', stats)
measured = {(a['qname'], a['format_name']): a['dloss']
            for a in json.load(open(f'{ROOT}/cost.anchors.json'))['anchors']}
for row in json.load(open(f'{ROOT}/regret_full.json'))['per_unit']:
    measured[(row['qname'], row['format'])] = row['true']
for row in json.load(open('/mnt/shared/tessera-runs/allocated/regret_uniform.json'))['per_unit']:
    measured[(row['qname'], row['format'])] = row['true']

print(f"{'role':22s} {'rung':>6s} {'bits':>7s} {'alloc dloss':>12s} {'unif dloss':>12s} {'a/u':>7s}")
ta = tu = 0.0
for role in sorted(ALLOC):
    q = f'model.layers.0.{role}'
    h = float(stats[q]['h_trace'] if isinstance(stats[q], dict) else stats[q])
    a = 0.5 * h * measured[(q, f'TESSERA_E4M3_K1_R{ALLOC[role]}')]
    u = 0.5 * h * measured[(q, f'TESSERA_E4M3_K1_R{UNIF}')]
    ta += a
    tu += u
    print(f"{role:22s} {'R' + str(ALLOC[role]):>6s} {ALLOC[role] / 256:7.3f} "
          f"{a:12.5g} {u:12.5g} {a / u:7.3f}")
print(f"{'TOTAL':22s} {'':>6s} {'':>7s} {ta:12.5g} {tu:12.5g} {ta / tu:7.3f}")
