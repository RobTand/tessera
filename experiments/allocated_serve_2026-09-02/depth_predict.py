"""What would the allocator have predicted if it had looked past layer 0?

The allocation is a layer-0 answer broadcast to 28 depths.  The *cost* was
measured on layer 0 only, but the *probe* covers all 196 Linears, so the
first-order depth correction is free: a unit's Delta-loss is 0.5 * h_trace *
output_mse, and re-weighting layer 0's measured Delta-loss by each layer's own
h_trace holds the rate-distortion curve fixed while letting the sensitivity vary
with depth.

Two arms, both re-weighted the same way: the allocation's per-role rungs, and
the byte-matched uniform R1006.  If the served gap is the broadcast rather than
the allocator, this reweighting should already show it.
"""
import json
import pickle
import re
import statistics

ROOT = '/mnt/shared/tessera-runs/pq-continuous/qwen06b'
import sys
ARMS = {
    '3.0': ({'self_attn.q_proj': 814, 'self_attn.k_proj': 814, 'self_attn.v_proj': 814,
             'self_attn.o_proj': 785, 'mlp.gate_proj': 824, 'mlp.up_proj': 824,
             'mlp.down_proj': 493}, 750),
    '4.0': ({'self_attn.q_proj': 1083, 'self_attn.k_proj': 1083, 'self_attn.v_proj': 1083,
             'self_attn.o_proj': 934, 'mlp.gate_proj': 1107, 'mlp.up_proj': 1107,
             'mlp.down_proj': 749}, 1006),
    '5.0': ({'self_attn.q_proj': 1366, 'self_attn.k_proj': 1366, 'self_attn.v_proj': 1366,
             'self_attn.o_proj': 1217, 'mlp.gate_proj': 1384, 'mlp.up_proj': 1384,
             'mlp.down_proj': 909}, 1262),
}
TARGET = sys.argv[1] if len(sys.argv) > 1 else '4.0'
ALLOC, UNIF = ARMS[TARGET]

stats = pickle.load(open(f'{ROOT}/probe.pkl', 'rb'))
stats = stats.get('stats', stats)
h = {}
for name, rec in stats.items():
    m = re.match(r'model\.layers\.(\d+)\.(.+)$', name)
    if m:
        h[(int(m.group(1)), m.group(2))] = float(rec['h_trace'] if isinstance(rec, dict) else rec)

anchors = json.loads(open(f'{ROOT}/cost.anchors.json').read())['anchors']
measured = {(a['qname'], a['format_name']): a['dloss'] for a in anchors}
uni = json.load(open('/mnt/shared/tessera-runs/allocated/regret_uniform.json'))
for row in uni['per_unit']:
    measured[(row['qname'], row['format'])] = row['true']
regret = json.load(open(f'{ROOT}/regret_full.json'))
for row in regret['per_unit']:
    measured[(row['qname'], row['format'])] = row['true']

roles = sorted(ALLOC)
layers = sorted({l for l, _ in h})


def dloss0(role, rung):
    key = (f'model.layers.0.{role}', f'TESSERA_E4M3_K1_R{rung}')
    if key not in measured:
        raise SystemExit(f'no measured layer-0 output MSE for {key}')
    # The anchor's ``dloss`` field is the unit's output MSE, not a Delta-loss:
    # PrismaQuant multiplies it by 0.5 * h_trace to get one.  Scaling the raw
    # MSE by a depth ratio would weight the roles by 1/h_0 -- a 25x distortion
    # between q_proj and v_proj on this model.
    return 0.5 * h[(0, role)] * measured[key]


def whole_body(rung_of):
    total = 0.0
    per_role = {}
    for role in roles:
        d0 = dloss0(role, rung_of(role))
        scale = sum(h[(l, role)] for l in layers) / h[(0, role)]
        per_role[role] = d0 * scale
        total += d0 * scale
    return total, per_role


ta, pa = whole_body(lambda r: ALLOC[r])
tu, pu = whole_body(lambda r: UNIF)
print(f'target {TARGET}: allocated rungs vs uniform R{UNIF}')
print(f"{'role':22s} {'rung':>6s} {'allocated':>12s} {'uniform R1006':>14s} {'alloc/unif':>11s}")
for role in roles:
    print(f"{role:22s} {'R' + str(ALLOC[role]):>6s} {pa[role]:12.4g} {pu[role]:14.4g} "
          f"{pa[role] / pu[role]:11.3f}")
print(f"{'TOTAL (depth-aware)':22s} {'':>6s} {ta:12.6g} {tu:14.6g} {ta / tu:11.3f}")
print()
la = sum(dloss0(r, ALLOC[r]) for r in roles)
lu = sum(dloss0(r, UNIF) for r in roles)
print(f"layer-0 only (what the allocator saw): allocated {la:.6g}  uniform {lu:.6g}  "
      f"ratio {la / lu:.3f}")
