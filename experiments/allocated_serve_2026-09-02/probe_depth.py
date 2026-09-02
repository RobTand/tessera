"""Is layer 0's role ordering representative of the other 27 depths?

The broadcast assumes it is.  ``probe.pkl`` carries the empirical Fisher
``h_trace`` per Linear; if the probe covered the whole body, the per-role
profile across depth is readable straight off it.
"""
import pickle
import re
import statistics

stats = pickle.load(open('/mnt/shared/tessera-runs/pq-continuous/qwen06b/probe.pkl', 'rb'))
stats = stats.get('stats', stats)
rows = {}
for name, rec in stats.items():
    m = re.match(r'model\.layers\.(\d+)\.(.+)$', name)
    if not m:
        continue
    h = rec['h_trace'] if isinstance(rec, dict) else rec
    rows[(int(m.group(1)), m.group(2))] = float(h)
layers = sorted({l for l, _ in rows})
roles = sorted({r for _, r in rows})
print(f"probe covers {len(layers)} layers, {len(roles)} roles, {len(rows)} Linears")
if len(layers) < 2:
    raise SystemExit("probe covers one layer only -- no depth profile to read")
print(f"{'role':22s} {'layer0':>12s} {'median 1..27':>14s} {'ratio l0/med':>13s} "
      f"{'l0 rank':>8s} {'median rank':>12s}")


def ranks(vals):
    order = sorted(vals, key=vals.get, reverse=True)
    return {r: i + 1 for i, r in enumerate(order)}


l0 = {r: rows[(0, r)] for r in roles if (0, r) in rows}
med = {r: statistics.median([rows[(l, r)] for l in layers if l and (l, r) in rows]) for r in roles}
r0, rm = ranks(l0), ranks(med)
for role in roles:
    print(f"{role:22s} {l0[role]:12.4g} {med[role]:14.4g} "
          f"{l0[role] / med[role]:13.3f} {r0[role]:8d} {rm[role]:12d}")
