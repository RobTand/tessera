import glob, json, os
R = '/home/rob/tessera-runs/allocated'
S = '/mnt/shared/tessera-runs/allocated'
BPP = {}
for d in sorted(glob.glob(f'{S}/qwen3-0.6b-*/tessera_serving_manifest.json')):
    m = json.load(open(d))
    BPP[os.path.basename(os.path.dirname(d))] = (m['totals']['wire_bpp'], m['totals']['resident_mode_bpp'])
ARM = {
 'alloc3-resident': 'qwen3-0.6b-alloc-3.0', 'unif750-resident': 'qwen3-0.6b-uniform-R750',
 'alloc4-resident': 'qwen3-0.6b-alloc-4.0', 'alloc4-resident-graph': 'qwen3-0.6b-alloc-4.0',
 'alloc4-streamed': 'qwen3-0.6b-alloc-4.0', 'alloc4-twin': 'qwen3-0.6b-alloc-4.0',
 'unif1006-resident': 'qwen3-0.6b-uniform-R1006', 'unif1006-resident-graph': 'qwen3-0.6b-uniform-R1006',
 'alloc5-resident': 'qwen3-0.6b-alloc-5.0', 'unif1262-resident': 'qwen3-0.6b-uniform-R1262',
 'l0-alloc': 'qwen3-0.6b-l0-alloc', 'l0-unif1006': 'qwen3-0.6b-l0-unif1006'}
print(f"{'arm':26s} {'wire':>9s} {'res':>7s} {'KLall':>9s} {'KLconf':>9s} {'top1':>7s} {'p99':>7s}")
for arm in ARM:
    p = f'{R}/kl_tessera_{arm}.json'
    if not os.path.exists(p):
        print(arm, 'MISSING')
        continue
    d = json.load(open(p))
    w, r = BPP.get(ARM[arm], (float('nan'), float('nan')))
    print(f"{arm:26s} {w:9.6f} {r:7.3f} {d['all']['kl_lower_mean']:9.6f} "
          f"{d['confident']['kl_lower_mean']:9.6f} {d['all']['top1_agree_pct']:6.2f}% {d['all']['kl_lower_p99']:7.3f}")
print()
print('=== mutual (student vs student)')
for p in sorted(glob.glob(f'{R}/mutual_*.json')):
    d = json.load(open(p))
    n = os.path.basename(p)[len('mutual_'):-len('.json')].replace('__', ' vs ')
    print(f"{n:52s} {d['all']['kl_lower_mean']:9.6f}  top1={d['all']['top1_agree_pct']:.2f}%")
