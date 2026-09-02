#!/usr/bin/env python3
"""Per-Linear GEMM error of each served arm on ITS OWN captured input, against an
fp64 reference built from the stock tiles and vLLM's own activation quantiser.
usage: gemm_real.py <layers_stock.npz> <layers_gridbook.npz> <layer> [<layer>...]"""
import sys, glob, numpy as np, torch
from safetensors import safe_open
from vllm import _custom_ops as ops
CK = "/home/rob/tessera-runs/stock/qwen3-0.6b-tessera-k2-q896-nvfp4"
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64, device="cuda")
A = np.load(sys.argv[1]); B = np.load(sys.argv[2]); layers = [int(x) for x in sys.argv[3:]]
stock = {}
for fn in glob.glob(f"{CK}/*.safetensors"):
    with safe_open(fn, "pt") as f:
        for k in f.keys():
            if any(f"layers.{L}." in k for L in layers): stock[k] = f.get_tensor(k)
def dequant(packed, scale_lin, g):
    lo = (packed & 0xF).long(); hi = (packed >> 4).long()
    vals = torch.stack([E2M1[lo], E2M1[hi]], dim=-1).reshape(packed.shape[0], -1)
    s = scale_lin.view(torch.float8_e4m3fn).to(torch.float64).repeat_interleave(16, dim=1)
    return vals * s / float(g)
FUSED = {"qkv_proj": ("self_attn", ["q_proj", "k_proj", "v_proj"]), "o_proj": ("self_attn", ["o_proj"]),
         "gate_up_proj": ("mlp", ["gate_proj", "up_proj"]), "down_proj": ("mlp", ["down_proj"])}
def ref_forward(x_bf16, gs, W64):
    gs_t = torch.tensor(gs, device="cuda", dtype=torch.float32)
    xq, xs = ops.scaled_fp4_quant(x_bf16.cuda(), gs_t, is_sf_swizzled_layout=False)
    M, K = x_bf16.shape
    return dequant(xq, xs.reshape(M, K // 16), gs) @ W64.t()
print(f"{'module':36s} {'stock err':>10s} {'lane err':>10s} {'propagated':>11s} {'total':>9s} {'|x_gb-x_st|/|x|':>15s}")
for L in layers:
    for fused, (sub, sibs) in FUSED.items():
        prefix = f"model.layers.{L}.{sub}"
        Ws = []; gss = []; wgss = []
        for s in sibs:
            m = f"{prefix}.{s}"
            Ws.append(dequant(stock[f"{m}.weight_packed"].cuda(), stock[f"{m}.weight_scale"].cuda(), float(stock[f"{m}.weight_global_scale"])))
            gss.append(float(stock[f"{m}.input_global_scale"])); wgss.append(float(stock[f"{m}.weight_global_scale"]))
        W64 = torch.cat(Ws, dim=0); gs = max(gss)
        key = f"{prefix}.{fused}"
        x_st = torch.from_numpy(A[key + ":in"]).to(torch.bfloat16); x_gb = torch.from_numpy(B[key + ":in"]).to(torch.bfloat16)
        y_st = torch.from_numpy(A[key + ":out"]).to(torch.float64).cuda(); y_gb = torch.from_numpy(B[key + ":out"]).to(torch.float64).cuda()
        r_st = ref_forward(x_st, gs, W64); r_gb = ref_forward(x_gb, gs, W64)
        n = r_st.norm()
        e_st = (y_st - r_st).norm() / n; e_gb = (y_gb - r_gb).norm() / n
        prop = (r_gb - r_st).norm() / n; tot = (y_gb - y_st).norm() / y_st.norm()
        dx = (x_gb.double() - x_st.double()).norm() / x_st.double().norm()
        note = "" if len(set(gss)) == 1 else f"  gs siblings {gss}"
        print(f"{key:36s} {float(e_st):10.3e} {float(e_gb):10.3e} {float(prop):11.3e} {float(tot):9.3e} {float(dx):15.3e}{note}")
