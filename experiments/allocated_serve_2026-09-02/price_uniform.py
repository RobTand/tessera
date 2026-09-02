"""What the allocator's own cost table says about the matched-bytes UNIFORM arm.

The served comparison is allocated-vs-uniform at matched wire bytes.  The
surrogate-vs-served check needs the same pair in the DP's currency, so both are
priced here on the SAME full interpolated table PrismaQuant allocated against
(`price_mink.py`'s function, unchanged in substance).
"""
import json
import pickle
import sys

sys.path.insert(0, "/home/rob/tessera/src")
sys.path.insert(0, "/home/rob/pq-wt/tessera-continuous")
from prismaquant.allocator_solver import predicted_dloss
from prismaquant import format_registry as fr

ROOT = "/mnt/shared/tessera-runs/pq-continuous/qwen06b"
costs = pickle.load(open(f"{ROOT}/cost.pkl", "rb"))["costs"]
probe = pickle.load(open(f"{ROOT}/probe.pkl", "rb"))
stats = probe["stats"] if isinstance(probe, dict) and "stats" in probe else probe


def price(rows):
    """rows: {qname: format name} -> (dloss, bpp, distinct rungs)"""
    total, bits, params, rungs = 0.0, 0.0, 0, set()
    for q, fmt in rows.items():
        rungs.add(fmt)
        entry = costs[q][fmt]
        total += predicted_dloss(float(stats[q]["h_trace"]), float(entry["output_mse"]), 1.0)
        n = int(stats[q]["n_params"])
        params += n
        shape = (int(stats[q]["out_features"]), int(stats[q]["in_features"]))
        bits += float(fr.get_format(fmt).effective_bits_for_shape(shape)) * n
    return total, bits / params, len(rungs)


def from_layer_config(path):
    cfg = json.load(open(path))
    return {k: v["tessera_format"] for k, v in cfg.items() if not k.startswith("__")}


UNITS = sorted(k for k in costs if k.startswith("model.layers.0."))
MATCHED = {"3.0": 750, "4.0": 1006, "5.0": 1262}

print(f"{'target':7s} {'arm':10s} {'bpp':>10s} {'dloss':>14s} {'rungs':>6s}")
results = {}
for target, rung in MATCHED.items():
    alloc_rows = from_layer_config(f"{ROOT}/alloc/lc_full_{target}.json")
    uni_rows = {q: f"TESSERA_E4M3_K1_R{rung}" for q in UNITS}
    for arm, rows in (("allocated", alloc_rows), (f"uniform R{rung}", uni_rows)):
        try:
            dl, bpp, n = price(rows)
        except KeyError as exc:
            print(f"{target:7s} {arm:10s}  MISSING from the cost table: {exc}")
            continue
        results[(target, arm)] = (dl, bpp)
        print(f"{target:7s} {arm:14s} {bpp:10.6f} {dl:14.6f} {n:6d}")
    key_a, key_u = (target, "allocated"), (target, f"uniform R{rung}")
    if key_a in results and key_u in results:
        ratio = results[key_u][0] / results[key_a][0]
        print(f"        -> the surrogate says the allocation is {ratio:.4f}x better in dloss "
              f"at {results[key_u][1] - results[key_a][1]:+.6f} bpp of extra budget for the uniform")
