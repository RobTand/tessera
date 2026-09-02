"""2026-09-02 dense-outlier study (Qwen3-0.6B, E4M3/CHANNEL/window L=14 wire). Run with
PYTHONPATH=src TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache <prismaquant-cu130 python>.

All 196 Qwen3-0.6B Linears: production Tessera-8 (stock E4M3 checkpoint, deterministic
encoder) vs the amax-aware per-row initial sigma re-encode vs production NVFP4 GPTQ+JSO
vs FP8 RTN.  H-weighted and plain RMS relative error per tensor."""
import glob, json, math, sys, time, torch
from safetensors import safe_open
import tessera.scale_channel as sc
sc._orig = sc.initial_channel_scale
from tessera.alphabet import E4M3_GRID
from tessera.export import DEFAULT_CODE, encode_linear_planes
from tessera.encode import window_table
from tessera.stock import materialize_stock
from tessera.scale_channel import default_channel_sigma, channel_global, land_channel_scale

dev = "cuda"
H = {k: v.to(dev, torch.float32) for k, v in torch.load("/home/rob/tessera-runs/stock/h_diag.pt").items()}
sig0 = default_channel_sigma(E4M3_GRID)
tab = window_table(E4M3_GRID, 14, sigma=sig0, seed=0, half=16, device="cpu")
reach = float(torch.tensor(E4M3_GRID.values)[tab.long()].abs().max()) / sig0

def amax_aware(work, sigma):
    r = work.float().pow(2).mean(1).sqrt(); z = work.float().abs().amax(1) / r
    row_sigma = torch.full_like(r, float(sigma)); over = z > reach
    row_sigma[over] = float(sigma) * reach / z[over]
    scale = r / row_sigma; g = channel_global(scale)
    stored, eff = land_channel_scale(scale, g); return stored, eff, g

def open_all(d):
    hs = [safe_open(f, framework="pt") for f in sorted(glob.glob(d + "/*.safetensors"))]
    idx = {}
    for h in hs:
        for k in h.keys(): idx[k] = h
    return idx
src = open_all("/home/rob/models/Qwen3-0.6B")
e8 = open_all("/home/rob/tessera-runs/stock/qwen3-0.6b-tessera-e4m3-q1024-fp8")
nv = open_all("/home/rob/dq-runs/fc45-0p6b-nvfp4/exported")
f8 = open_all("/home/rob/tessera-runs/stock/qwen3-0.6b-fp8-rtn")
e2m1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=dev)
def nvfp4(name, rows, cols):
    pk = nv[name + ".weight_packed"].get_tensor(name + ".weight_packed").to(dev)
    s = nv[name + ".weight_scale"].get_tensor(name + ".weight_scale").to(dev).float()
    g = nv[name + ".weight_global_scale"].get_tensor(name + ".weight_global_scale").to(dev).float()
    lo, hi = (pk & 0xF).long(), (pk >> 4).long()
    dq = lambda n: e2m1[n & 7] * torch.where(n >= 8, -1.0, 1.0)
    return torch.stack([dq(lo), dq(hi)], -1).reshape(rows, cols) * (s / g).repeat_interleave(16, dim=1)
def fp8pair(idx, name):
    return idx[name + ".weight"].get_tensor(name + ".weight").to(dev).float() * idx[name + ".weight_scale"].get_tensor(name + ".weight_scale").to(dev).float()

def err(W, Wd, h):
    E = Wd - W
    return (math.sqrt((E*E).sum().item() / (W*W).sum().item()),
            math.sqrt(((E*E).sum(0)*h).sum().item() / ((W*W).sum(0)*h).sum().item()))

out = {}; t0 = time.time()
for i, name in enumerate(sorted(H)):
    W = src[name + ".weight"].get_tensor(name + ".weight").to(dev, torch.float32).contiguous()
    rows, cols = W.shape; h = H[name]
    zmax = (W.abs().amax(1) / W.pow(2).mean(1).sqrt())
    rec = {"rows": rows, "cols": cols, "rows_over_reach": int((zmax > reach).sum()), "zmax": float(zmax.max()),
           "h_ratio": float(h.max() / h.median())}
    rec["A"] = err(W, fp8pair(e8, name), h)
    sc.initial_channel_scale = amax_aware
    try:
        _, unit, forests = encode_linear_planes(W, grid=E4M3_GRID, q256=1024, name=name, verify=False)
    finally:
        sc.initial_channel_scale = sc.__dict__.get("_orig", sc.initial_channel_scale)
    st = materialize_stock(unit, forests, DEFAULT_CODE)
    rec["B"] = err(W, st["weight"].to(dev).float() * st["weight_scale"].to(dev).float(), h)
    rec["nvfp4"] = err(W, nvfp4(name, rows, cols), h)
    rec["fp8rtn"] = err(W, fp8pair(f8, name), h)
    out[name] = rec
    if i % 20 == 0: print(f"[{i}/196] {name} {time.time()-t0:.0f}s A {rec['A'][1]:.4f} B {rec['B'][1]:.4f} nv {rec['nvfp4'][1]:.4f}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
def gm(xs): return math.exp(sum(math.log(x) for x in xs) / len(xs))
roles = sorted({n.rsplit(".", 1)[-1] for n in out})
print(f"\ntable reach {reach:.2f} sigma (sigma0 {sig0:.1f}); {sum(r['rows_over_reach'] for r in out.values())}/{sum(r['rows'] for r in out.values())} rows exceed it")
print(f"{'role':10s} {'n':>3s} | {'A plain':>8s} {'B plain':>8s} {'nv plain':>8s} | {'A wtd':>8s} {'B wtd':>8s} {'nv wtd':>8s} {'fp8 wtd':>8s} | B<nv  A<nv")
for role in roles + ["ALL"]:
    rs = [r for n, r in out.items() if role == "ALL" or n.endswith("." + role)]
    print(f"{role:10s} {len(rs):3d} | {gm([r['A'][0] for r in rs]):8.4f} {gm([r['B'][0] for r in rs]):8.4f} {gm([r['nvfp4'][0] for r in rs]):8.4f} | "
          f"{gm([r['A'][1] for r in rs]):8.4f} {gm([r['B'][1] for r in rs]):8.4f} {gm([r['nvfp4'][1] for r in rs]):8.4f} {gm([r['fp8rtn'][1] for r in rs]):8.4f} | "
          f"{sum(r['B'][1] < r['nvfp4'][1] for r in rs):3d}  {sum(r['A'][1] < r['nvfp4'][1] for r in rs):3d}")
worst = sorted(out.items(), key=lambda kv: -kv[1]["B"][1] / kv[1]["nvfp4"][1])[:5]
print("worst B/nvfp4:", [(n, round(r['B'][1], 4), round(r['nvfp4'][1], 4), r['rows_over_reach']) for n, r in worst])
