"""CPU check of real_units.py on the T-8 release artifact (D38 for the real-wire path).

For one expert, both TP2 ranks: the window-rule decode of the cut planes, times the row scale,
must equal today's full-unit reconstruction (``read_unit_artifact``) sliced to the same cut.
"""
import argparse, collections, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_units import Artifact, layer_units, reference_weights  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--artifact", default="/mnt/shared/tessera-measurements/pact-e4m3-accuracy-20260928/release-t8/exported")
ap.add_argument("--layer", type=int, default=10)
ap.add_argument("--expert", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
from tessera.unit_artifact import read_unit_artifact
art = Artifact(a.artifact)
rep = {}
for rank in (0, 1):
    units = layer_units(art, a.layer, a.expert, rank)
    for proj, u in units.items():
        name = f"model.language_model.layers.{a.layer}.mlp.experts.{a.expert}.{proj}.wire"
        full = read_unit_artifact(art.blob(name), device="cpu").float()
        rows, cols = u.codes.shape
        if proj == "down_proj":
            ref_full = full[:, rank * cols:(rank + 1) * cols]
        else:
            ref_full = full[rank * rows:(rank + 1) * rows]
        mine = reference_weights(u).view(torch.float8_e4m3fn).float() * u.scale[:, None]
        rep[f"rank{rank}.{proj}"] = {
            "shape": [rows, cols], "rates": dict(collections.Counter(u.rates)),
            "start_nonzero": int((u.start != 0).sum()),
            "equal": bool(torch.equal(mine, ref_full)),
            "max_abs_diff": float((mine - ref_full).abs().max())}
        print(rank, proj, rep[f"rank{rank}.{proj}"], flush=True)
json.dump(rep, open(os.path.join(a.out, "real-units-check.json"), "w"), indent=1)
assert all(v["equal"] for v in rep.values()), "real-unit decode differs from read_unit_artifact"
print("real-unit check passed")
