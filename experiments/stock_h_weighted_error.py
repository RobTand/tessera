"""Diagonal-Hessian-weighted weight error per served arm.

Plain weight MSE ranks the arms one way and the served KL another.  The
standard proxy for a Linear's output error is sum_j h_j ||dW[:, j]||^2 with
h_j = E[x_j^2] the input column's second moment (GPTQ's diagonal); Qwen's
outlier input channels carry h_j hundreds of times the median.  This scores
every arm on that proxy, from a BF16 forward over held-out-disjoint text.
"""
import json, sys, torch
from safetensors import safe_open
sys.path.insert(0, "/home/rob/tessera/src")
from tessera.stock import stock_dequant
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = "/home/rob/models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(SRC)
model = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.bfloat16).cuda().eval()
# Calibration text disjoint from the KL corpus: the model card / README and
# the tokenizer's own vocabulary file are not the WikiText the corpus draws on.
text = open(f"{SRC}/README.md").read() + "\n" + open("/home/rob/tessera/docs/tessera-one-format.md").read()
ids = tok(text, return_tensors="pt").input_ids[:, : 8 * 512].cuda()
ids = ids[:, : (ids.shape[1] // 512) * 512].reshape(-1, 512)
h, counts = {}, {}
def hook(name):
    def fn(mod, inp, out):
        x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
        h[name] = h.get(name, 0) + (x * x).sum(0)
        counts[name] = counts.get(name, 0) + x.shape[0]
    return fn
handles = [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "proj" in n and "layers." in n]
with torch.no_grad():
    for row in ids:
        model(row[None])
for hd in handles: hd.remove()
H = {n: (v / counts[n]) for n, v in h.items()}
torch.save({n: v.cpu() for n, v in H.items()}, "/home/rob/tessera-runs/stock/h_diag.pt")
ratios = [(v.max() / v.median()).item() for v in H.values()]
print(f"calib tokens {ids.numel()}  Linears {len(H)}  max/median h: median {sorted(ratios)[len(ratios)//2]:.0f} max {max(ratios):.0f}")
del model; torch.cuda.empty_cache()

src = safe_open(f"{SRC}/model.safetensors", "pt")
arms = {"nvfp4-prod": "/home/rob/dq-runs/fc45-0p6b-nvfp4/exported", "tessera-k2": "/home/rob/tessera-runs/stock/qwen3-0.6b-tessera-k2-q896-nvfp4",
        "tessera-e8": "/home/rob/tessera-runs/stock/qwen3-0.6b-tessera-e4m3-q1024-fp8", "fp8-rtn": "/home/rob/tessera-runs/stock/qwen3-0.6b-fp8-rtn"}
out = {}
for arm, d in arms.items():
    f = safe_open(f"{d}/model.safetensors", "pt"); keys = set(f.keys())
    num = den = 0.0; per = {}
    for n, hv in H.items():
        w = src.get_tensor(n + ".weight").cuda().float(); hv = hv.cuda()
        if n + ".weight_packed" in keys:
            t = {s: f.get_tensor(f"{n}.{s}").cuda() for s in ("weight_packed", "weight_scale", "weight_global_scale")}
        else:
            t = {s: f.get_tensor(f"{n}.{s}").cuda() for s in ("weight", "weight_scale")}
        dw = stock_dequant(t) - w
        e = float((dw.pow(2).sum(0) * hv).sum()); r = float((w.pow(2).sum(0) * hv).sum())
        per[n] = e / r; num += e; den += r
    vals = sorted(per.values())
    print(f"{arm:12s} H-weighted rel err: total {num/den:.3e}  geomean {torch.tensor(vals).log().mean().exp().item():.3e}  median {vals[len(vals)//2]:.3e}  max {vals[-1]:.3e}")
    out[arm] = {"total": num / den, "per_unit": per}
json.dump(out, open("/home/rob/tessera-runs/stock/h_weighted_error.json", "w"), indent=1)
