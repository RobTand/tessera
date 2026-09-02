"""One table of every KL arm this run produced, read off the compare JSONs."""
import glob
import json
import os
import re

BPP = {}
for d in sorted(glob.glob('/mnt/shared/tessera-runs/allocated/qwen3-0.6b-*/tessera_serving_manifest.json')):
    m = json.load(open(d))
    BPP[os.path.basename(os.path.dirname(d))] = (m['totals']['wire_bpp'], m['totals']['resident_mode_bpp'])

ARM_CKPT = {
    'alloc4-resident': 'qwen3-0.6b-alloc-4.0', 'alloc4-resident-graph': 'qwen3-0.6b-alloc-4.0',
    'alloc4-streamed': 'qwen3-0.6b-alloc-4.0', 'alloc4-twin': 'qwen3-0.6b-alloc-4.0',
    'unif1006-resident': 'qwen3-0.6b-uniform-R1006',
    'unif1006-resident-graph': 'qwen3-0.6b-uniform-R1006',
    'l0-alloc': 'qwen3-0.6b-l0-alloc', 'l0-unif1006': 'qwen3-0.6b-l0-unif1006',
    'alloc3-resident': 'qwen3-0.6b-alloc-3.0', 'unif750-resident': 'qwen3-0.6b-uniform-R750',
    'alloc5-resident': 'qwen3-0.6b-alloc-5.0', 'unif1262-resident': 'qwen3-0.6b-uniform-R1262',
}

print(f"{'arm':26s} {'wire bpp':>9s} {'res bpp':>8s} {'KL all':>10s} {'KL conf':>9s} {'top-1':>7s}")
for path in sorted(glob.glob('/home/rob/tessera-runs/allocated/kl_tessera_*.json')):
    arm = os.path.basename(path)[len('kl_tessera_'):-len('.json')]
    d = json.load(open(path))
    log = f'/home/rob/tessera-runs/allocated/arm_{arm}.log'
    kl_all = kl_conf = top1 = None
    if os.path.exists(log):
        text = open(log, errors='replace').read()
        m = re.search(r'ALL\s+KL >= ([0-9.]+)', text)
        kl_all = m.group(1) if m else None
        m = re.search(r'CONFIDENT.*?KL >= ([0-9.]+)', text)
        kl_conf = m.group(1) if m else None
        m = re.search(r'top1_agree=([0-9.]+)%', text)
        top1 = m.group(1) if m else None
    w, r = BPP.get(ARM_CKPT.get(arm, ''), (None, None))
    print(f"{arm:26s} {w if w else float('nan'):9.6f} {r if r else float('nan'):8.3f} "
          f"{kl_all or '?':>10s} {kl_conf or '?':>9s} {(top1 + '%') if top1 else '?':>7s}")
