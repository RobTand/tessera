"""Is Gridbook's LDLQ certificate an in-window win that does not travel?

One tensor (L42.gate_proj), FP8-CB K32, the same fields the followups run
built.  Scores raw vs the gate's shipped LDLQ candidate on (a) the first 7168
rows the Hessian and the certificate were fit on, (b) the last 1024 rows the
followups run reports.  Same reconstruct call the gate itself uses.
"""
import json, torch
from safetensors import safe_open
from tessera8_targets import ACT, SRC
from prismaquant.nvfp4_cb_formats import (ldlq_reassign_cb_fields_gated,
                                          nvfp4_cb_fields, nvfp4_cb_reconstruct)

layer, proj, K = 42, "gate_proj", 32
blob = torch.load(f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                  map_location="cpu", weights_only=False)
xa = blob["inputs"].float()
x_fit = xa[:-1024].contiguous().cuda()
x_ev = xa[-1024:].contiguous().cuda()
cw = x_fit.pow(2).mean(dim=0).float().contiguous()
index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
    w = f.get_tensor(name).contiguous().cuda().float()

fields = nvfp4_cb_fields(w, K, grid="fp8", mode="product", col_weights=cw)
fields2, info = ldlq_reassign_cb_fields_gated(w, fields, cw, x_fit,
                                              grid="fp8", mode="product", k=K)
raw = nvfp4_cb_reconstruct(fields, K, grid="fp8", mode="product").float()
ldq = nvfp4_cb_reconstruct(fields2, K, grid="fp8", mode="product").float()
print("gate:", {k: v for k, v in info.items() if isinstance(v, (str, bool, float))})
print(f"weight rel-err   raw {float((raw-w).norm()/w.norm()):.5f}  "
      f"ldlq {float((ldq-w).norm()/w.norm()):.5f}")
for tag, X in (("fit window (first 7168)", x_fit), ("eval window (last 1024)", x_ev)):
    y = X @ w.T; n = y.norm()
    r = float((X @ raw.T - y).norm() / n); l = float((X @ ldq.T - y).norm() / n)
    print(f"{tag:<26} out raw {r:.5f}  ldlq {l:.5f}  ratio {l/r:.3f}")
