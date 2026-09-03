"""Per-row decomposition of the R=3 (and R=5) reach loss: which rows pay?

Rows are bucketed by z = amax/rms against the two reaches being compared, so
the loss is attributed to (a) rows unclamped under both, (b) rows that change
clamp status, (c) rows clamped under both.  Deterministic encoder; identical
bytes per arm are asserted by the sha.
"""
import json, sys, math
import torch
_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT / "experiments"))
from tessera.alphabet import BF16_GRID
from tessera.export import encode_linear_planes, wire_recipe, BF16_CHANNEL_SIGMA
from tessera.unit_artifact import read_unit_artifact
from bf16_l_sigma_sweep import DENSE_UNITS, reach_stats
from bf16_route_weight_space import DENSE_SRC, open_all

src = open_all(DENSE_SRC)
out = {}
for q, sigmas in ((768, (1.0, math.sqrt(3/4))), (1280, (1.0, math.sqrt(5/4)))):
    for name in DENSE_UNITS:
        w = src[name + ".weight"].get_tensor(name + ".weight").cuda().float().contiguous()
        rms = w.pow(2).mean(1).sqrt(); amax = w.abs().amax(1); z = amax / rms
        recipe = wire_recipe(BF16_GRID, q)
        sse, reach = {}, {}
        for s in sigmas:
            st = reach_stats(w, BF16_GRID, recipe.window_bits, s, BF16_CHANNEL_SIGMA)
            reach[s] = st["reach_row_rms"]
            ex, *_ = encode_linear_planes(w, grid=BF16_GRID, q256=q, name=name,
                                          channel_sigma=BF16_CHANNEL_SIGMA, window_sigma=s, verify=True)
            hat = read_unit_artifact(ex.blob, device=w.device)
            sse[s] = (hat - w).pow(2).sum(1)
        a, b = sigmas  # a = pinned 1.0, b = law
        lo, hi = sorted((reach[a], reach[b]))
        buckets = {"unclamped_both": z < lo, "changes_status": (z >= lo) & (z < hi), "clamped_both": z >= hi}
        tot_a, tot_b = float(sse[a].sum()), float(sse[b].sum())
        rec = {"rows": int(w.shape[0]), "reach_pinned": reach[a], "reach_law": reach[b],
               "wt_ratio_law_over_pinned": math.sqrt(tot_b / tot_a), "buckets": {}}
        for k, m in buckets.items():
            da, db = float(sse[a][m].sum()), float(sse[b][m].sum())
            rec["buckets"][k] = {"rows": int(m.sum()), "frac_rows": float(m.float().mean()),
                                 "sse_pinned_share": da / tot_a, "sse_law_over_pinned": (db / da if da > 0 else None),
                                 "share_of_delta": ((db - da) / (tot_b - tot_a)) if tot_b != tot_a else None}
        out[f"R{q} {name}"] = rec
        print(f"R={q/256:g} {name}: wt law/pinned {rec['wt_ratio_law_over_pinned']:.4f}  reach {reach[a]:.2f}->{reach[b]:.2f}")
        for k, v in rec["buckets"].items():
            print(f"    {k:<16} rows {v['frac_rows']:.3f}  sse share(pinned) {v['sse_pinned_share']:.3f}  "
                  f"law/pinned {v['sse_law_over_pinned'] if v['sse_law_over_pinned'] is None else round(v['sse_law_over_pinned'],4)}  "
                  f"share of delta {v['share_of_delta'] if v['share_of_delta'] is None else round(v['share_of_delta'],3)}")
        del w, hat; torch.cuda.empty_cache()
json.dump(out, open(sys.argv[1], "w"), indent=1)
