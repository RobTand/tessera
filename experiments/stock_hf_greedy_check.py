"""Greedy completion with the checkpoint's DEQUANTISED weights in HF (W16A16-equivalent):
separates 'the weights are bad for this prompt' from 'the served path is bad'."""
import sys, torch
from pathlib import Path
from safetensors import safe_open
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tessera.stock import stock_dequant
from transformers import AutoModelForCausalLM, AutoTokenizer
SRC = "/home/rob/models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(SRC)
prompt = "The capital of France is"
for label, d in [("bf16", None), ("tessera-e8", "/home/rob/tessera-runs/stock/qwen3-0.6b-tessera-e4m3-q1024-fp8"),
                 ("fp8-rtn", "/home/rob/tessera-runs/stock/qwen3-0.6b-fp8-rtn"), ("tessera-k2", "/home/rob/tessera-runs/stock/qwen3-0.6b-tessera-k2-q896-nvfp4"),
                 ("nvfp4-prod", "/home/rob/dq-runs/fc45-0p6b-nvfp4/exported")]:
    model = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.bfloat16).cuda().eval()
    if d:
        f = safe_open(f"{d}/model.safetensors", "pt"); keys = set(f.keys()); n_sub = 0
        sd = model.state_dict()
        for name, mod in model.named_modules():
            if not isinstance(mod, torch.nn.Linear) or "layers." not in name: continue
            if name + ".weight_packed" in keys:
                t = {s: f.get_tensor(f"{name}.{s}").cuda() for s in ("weight_packed", "weight_scale", "weight_global_scale")}
            elif name + ".weight" in keys and f.get_slice(name + ".weight").get_dtype() == "F8_E4M3":
                t = {s: f.get_tensor(f"{name}.{s}").cuda() for s in ("weight", "weight_scale")}
            else: continue
            mod.weight.data.copy_(stock_dequant(t).to(torch.bfloat16)); n_sub += 1
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=24, do_sample=False)
        logits = model(ids).logits[0, -1].float()
    top = torch.topk(torch.log_softmax(logits, -1), 5)
    print(f"{label:11s} subst={n_sub if d else 0:3d}  {tok.decode(out[0, ids.shape[1]:])!r}")
    print("             top-5 next:", [(tok.decode([int(i)]), round(float(v), 2)) for v, i in zip(top.values, top.indices)])
    del model; torch.cuda.empty_cache()
