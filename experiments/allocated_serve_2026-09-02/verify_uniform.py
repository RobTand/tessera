"""The true (re-measured) cost of the matched-bytes UNIFORM arms.

PrismaQuant's ``verify_chosen.py`` re-encoded and re-scored every rung the
allocator chose, giving a *true* Delta-loss beside the interpolated one.  The
served comparison here is allocated-vs-uniform, so the uniform arm needs the
same treatment or the two sides of the surrogate check are not the same
measurement.  Same campaign functions, same activations, same render path.
"""
import argparse
import collections
import json
import pathlib
import pickle
import sys

sys.path.insert(0, "/home/rob/pq-wt/tessera-continuous")
sys.path.insert(0, "/home/rob/tessera/.claude/worktrees/agent-a6d34c0d5bba700a6/src")

from prismaquant.tessera_campaign import (
    _calibration_tokens, _collect_activations, _measure_anchor)

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/mnt/shared/tessera-runs/pq-continuous/qwen06b")
ap.add_argument("--model", default="/home/rob/models/Qwen3-0.6B")
ap.add_argument("--rungs", default="750,1006,1262")
ap.add_argument("--out", default="/mnt/shared/tessera-runs/allocated/regret_uniform.json")
ap.add_argument("--nsamples", type=int, default=4)
ap.add_argument("--seqlen", type=int, default=512)
ap.add_argument("--max-act-rows", type=int, default=256)
args = ap.parse_args()
ROOT = pathlib.Path(args.root)

cost = pickle.load(open(ROOT / "cost.pkl", "rb"))
anchors = json.loads((ROOT / "cost.anchors.json").read_text())["anchors"]
measured = {(x["qname"], x["format_name"]): x["dloss"] for x in anchors}
stats = pickle.load(open(ROOT / "probe.pkl", "rb"))
stats = stats.get("stats", stats)

units = sorted(k for k in cost["costs"] if k.startswith("model.layers.0."))
rungs = [int(r) for r in args.rungs.split(",")]
chosen = {(q, f"TESSERA_E4M3_K1_R{r}"): [str(r)] for r in rungs for q in units}
todo = [kv for kv in chosen if kv not in measured]
print(f"[verify] {len(chosen)} (unit, rung) pairs; {len(todo)} never measured", flush=True)

import torch
from transformers import AutoModelForCausalLM
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=device)
model.eval()
tokens = _calibration_tokens(args.model, args.nsamples, args.seqlen, 0)
acts, _hess, _rows = _collect_activations(model, units, tokens, args.max_act_rows, device)
weights = {n: dict(model.named_modules())[n].weight.detach() for n in units}
del model
torch.cuda.empty_cache()

from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.tessera_campaign import SCHEMA
from prismaquant.tessera_menu import menu_mode
cache = ProductionWeightCache(
    weights={}, levers={"tessera_campaign": True},
    cache_dir="/mnt/shared/tessera-runs/allocated/verify_cache",
    metadata={"schema": SCHEMA, "menu_mode": menu_mode(None)})
wire_dir = pathlib.Path("/mnt/shared/tessera-runs/allocated/verify_cache/wire")
wire_dir.mkdir(parents=True, exist_ok=True)

true = dict(measured)
for i, (q, f) in enumerate(todo):
    anc = _measure_anchor(qname=q, weight=weights[q], activations=acts[q],
                          format_name=f, cache=cache, wire_dir=wire_dir,
                          hessian_required=False)
    true[(q, f)] = anc.dloss
    print(f"[verify] {i + 1}/{len(todo)} {q} {f} true={anc.dloss:.6g}", flush=True)

out = {"per_unit": [], "per_rung": {}}
for r in rungs:
    fmt = f"TESSERA_E4M3_K1_R{r}"
    sp = st = 0.0
    for q in units:
        h = float(stats[q]["h_trace"])
        pred = cost["costs"][q][fmt]["output_mse"]
        tv = true[(q, fmt)]
        sp += 0.5 * h * pred
        st += 0.5 * h * tv
        out["per_unit"].append({"qname": q, "format": fmt, "predicted": pred, "true": tv,
                                "ratio": tv / pred})
    out["per_rung"][str(r)] = {"predicted_dloss": sp, "true_dloss": st, "ratio": st / sp}
    print(f"[uniform R{r}] predicted dloss={sp:.6g}  true dloss={st:.6g}  ratio={st / sp:.4f}")
pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
print(f"[verify] wrote {args.out}")
