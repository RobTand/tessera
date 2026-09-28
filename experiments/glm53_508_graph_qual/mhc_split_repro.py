"""tessera#508: does vLLM's mHC TileLang op give the same bits for a token at every batch size?

A decode graph captured for 8 tokens and replayed for 5 real tokens runs every op at
8 rows; eager runs them at 5. An op whose kernel configuration depends on the row
count can then reduce in a different order in the two runs. vLLM's
``mhc_fused_post_pre_tilelang`` (model_executor/kernels/mhc/tilelang.py) picks its
split-K factor from the token count: at most 16 tokens take the small-FMA path with
``tile_n = 2 if num_tokens < 8 else 3`` and ``n_splits = 8 if num_tokens < 8 (and
hidden_size <= 4096) else 4``; above 16 tokens it sizes ``n_splits`` with
``compute_num_split(64, K, cdiv(num_tokens, 64))``. ``mhc_pre_tilelang`` (layer 0's
attention-side pre block) always uses ``compute_num_split``, which is constant up to
64 tokens.

For each pair (M, P) this runs the op once on M rows and once on P >= M rows whose
first M rows are the same bytes, and compares the first M rows of every output bit
for bit. Each size is also run twice, so a difference is a function of the row
count, not run-to-run noise. The mHC parameters are layer 1's real ``hc_attn_*``
tensors and RMSNorm weight from the 4-layer stub; the activations are seeded
random values (the property tested is the op's reduction order, not a value).

  mhc_split_repro.py OUT_JSON [MODEL_DIR]
"""
import json
import pathlib
import sys

import torch
from safetensors import safe_open

from vllm.model_executor.kernels.mhc.tilelang import (
    mhc_fused_post_pre_tilelang,
    mhc_pre_tilelang,
)
from vllm.model_executor.kernels.mhc.tilelang_kernels import compute_num_split
from vllm.utils.math_utils import cdiv

out_path = pathlib.Path(sys.argv[1])
model_dir = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else
                         "/mnt/shared/tessera-runs/moe/glm53-4layer-a4-e2m1x2-q896-l2")
cfg = json.loads((model_dir / "config.json").read_text())
text = cfg.get("text_config", cfg)
HC = int(text.get("hc_mult", 4))
H = int(text["hidden_size"])
SINKHORN = int(text.get("hc_sinkhorn_iters", 20))
HC_EPS = float(text.get("hc_eps", 1e-6))
RMS_EPS = float(text["rms_norm_eps"])
POST_MULT = 2.0  # Glm5NextConfig default; the stub's config does not set it

index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
prefix = "model.language_model.layers.1."
names = {k: prefix + k for k in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base",
                                 "input_layernorm.weight")}
params = {}
for key, name in names.items():
    with safe_open(str(model_dir / index[name]), framework="pt", device="cuda") as fh:
        params[key] = fh.get_tensor(name)
fn = params["hc_attn_fn"].float().contiguous()
scale = params["hc_attn_scale"].float().contiguous()
base = params["hc_attn_base"].float().contiguous()
norm_w = params["input_layernorm.weight"].to(torch.bfloat16).contiguous()

PMAX = 96
g = torch.Generator(device="cuda").manual_seed(508)
x_all = torch.randn(PMAX, H, device="cuda", generator=g).to(torch.bfloat16)
res_all = torch.randn(PMAX, HC, H, device="cuda", generator=g).to(torch.bfloat16)
post_all = (torch.rand(PMAX, HC, 1, device="cuda", generator=g) * 2.0).float()
comb_raw = torch.rand(PMAX, HC, HC, device="cuda", generator=g) + 0.1
comb_all = (comb_raw / comb_raw.sum(-1, keepdim=True)).float()


def fused(m):
    outs = mhc_fused_post_pre_tilelang(
        x_all[:m].clone(), res_all[:m].clone(), post_all[:m].clone(), comb_all[:m].clone(),
        fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
        norm_weight=norm_w, norm_eps=RMS_EPS)
    return [t.reshape(m, -1).clone() for t in outs]


def pre(m):
    outs = mhc_pre_tilelang(
        res_all[:m].clone(), fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
        norm_weight=norm_w, norm_eps=RMS_EPS)
    return [t.reshape(m, -1).clone() for t in outs]


N_SMS = torch.cuda.get_device_properties(0).multi_processor_count


def fused_config(m):
    if m <= 16:
        return dict(path="small_fma", tile_n=2 if m < 8 else 3,
                    n_splits=8 if (m < 8 and H <= 4096) else 4)
    return dict(path="post+prenorm_gemm",
                n_splits_if_deep_gemm=compute_num_split(64, HC * H, cdiv(m, 64)))


def pre_config(m):
    return dict(n_splits_if_deep_gemm=compute_num_split(64, HC * H, cdiv(m, 64)))


FUSED_OUT = ["residual_cur", "post_mix_cur", "comb_mix_cur", "layer_input_cur"]
PRE_OUT = ["post_mix", "comb_mix", "layer_input"]


def compare(op, out_names, m, p):
    a1, a2, b = op(m), op(m), op(p)
    torch.cuda.synchronize()
    rep = all(torch.equal(u, v) for u, v in zip(a1, a2))
    per = {}
    for name, u, v in zip(out_names, a1, b):
        vm = v[:m]
        diff = (u.float() - vm.float()).abs()
        per[name] = dict(bit_equal=bool(torch.equal(u, vm)),
                         elements_differing=int((u != vm).sum().item()),
                         max_abs=float(diff.max().item()))
    return dict(m=m, p=p, repeat_exact_at_m=rep,
                all_outputs_bit_equal=all(v["bit_equal"] for v in per.values()), outputs=per)


FUSED_PAIRS = [(1, 2), (2, 4), (3, 4), (4, 4), (5, 8), (6, 8), (7, 8), (8, 8), (8, 16),
               (9, 16), (12, 16), (16, 16), (13, 24), (17, 24), (24, 32), (40, 48), (60, 72)]
PRE_PAIRS = [(3, 4), (5, 8), (7, 8), (9, 16), (17, 24), (60, 64), (60, 72), (64, 72)]

report = dict(model=str(model_dir), hc_mult=HC, hidden_size=H, sinkhorn_iters=SINKHORN,
              n_sms=N_SMS, torch=torch.__version__, device=torch.cuda.get_device_name(0),
              fused=[], pre=[])
for m, p in FUSED_PAIRS:
    r = compare(fused, FUSED_OUT, m, p)
    r["config_m"], r["config_p"] = fused_config(m), fused_config(p)
    report["fused"].append(r)
    print(f"fused {m:>3} -> {p:>3}: bit-equal={r['all_outputs_bit_equal']!s:5} "
          f"repeat-exact={r['repeat_exact_at_m']!s:5} "
          f"cfg {r['config_m']} -> {r['config_p']} "
          + " ".join(f"{k}:{v['elements_differing']}/{v['max_abs']:.3g}"
                     for k, v in r["outputs"].items()), flush=True)
for m, p in PRE_PAIRS:
    r = compare(pre, PRE_OUT, m, p)
    r["config_m"], r["config_p"] = pre_config(m), pre_config(p)
    report["pre"].append(r)
    print(f"pre   {m:>3} -> {p:>3}: bit-equal={r['all_outputs_bit_equal']!s:5} "
          f"repeat-exact={r['repeat_exact_at_m']!s:5} "
          f"cfg {r['config_m']} -> {r['config_p']} "
          + " ".join(f"{k}:{v['elements_differing']}/{v['max_abs']:.3g}"
                     for k, v in r["outputs"].items()), flush=True)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(report, indent=1))
print(f"wrote {out_path}")
