"""2026-09-02 dense-outlier study (Qwen3-0.6B, E4M3/CHANNEL/window L=14 wire). Run with
PYTHONPATH=src TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache <prismaquant-cu130 python>.

Alpha sweep of the h weighting on layer-2 down_proj (B + h^alpha on refit, metric on/off)."""
import math, sys, torch
from safetensors import safe_open
import tessera.encode as enc
import tessera.scale_channel as sc
from tessera.alphabet import E4M3_GRID
from tessera.export import DEFAULT_CODE, encode_linear_planes
from tessera.encode import window_table
from tessera.stock import materialize_stock
from tessera.scale_channel import default_channel_sigma, channel_global, land_channel_scale
name = "model.layers.2.mlp.down_proj"; dev = "cuda"
h = torch.load("/home/rob/tessera-runs/stock/h_diag.pt")[name].to(dev, torch.float32)
sig0 = default_channel_sigma(E4M3_GRID)
tab = window_table(E4M3_GRID, 14, sigma=sig0, seed=0, half=16, device="cpu")
reach = float(torch.tensor(E4M3_GRID.values)[tab.long()].abs().max()) / sig0
orig_refit, orig_vw = sc.refit_channel_scale, enc.viterbi_window
CUR = {"calls": 0}
def amax_aware(work, sigma):
    r = work.float().pow(2).mean(1).sqrt(); z = work.float().abs().amax(1) / r
    row_sigma = torch.full_like(r, float(sigma)); over = z > reach
    row_sigma[over] = float(sigma) * reach / z[over]
    scale = r / row_sigma; g = channel_global(scale)
    stored, eff = land_channel_scale(scale, g); return stored, eff, g
sc.initial_channel_scale = amax_aware
def refit_h(work, units, stored, g):
    sq = CUR["w"].sqrt()[None, :]
    return orig_refit(work.float() * sq, units.float() * sq, stored, g)
def vw_h(sub, vectors, window_bits, present, weights=None, **kw):
    CUR["calls"] += 1
    hw = CUR["w"][None, :].expand_as(sub)
    weights = hw if weights is None else weights * hw
    return orig_vw(sub, vectors, window_bits, present, weights=weights, **kw)
with safe_open("/home/rob/models/Qwen3-0.6B/model.safetensors", framework="pt") as f:
    W = f.get_tensor(name + ".weight").to(dev, torch.float32).contiguous()
def err(Wd):
    E = Wd - W
    return (math.sqrt((E*E).sum().item() / (W*W).sum().item()),
            math.sqrt(((E*E).sum(0)*h).sum().item() / ((W*W).sum(0)*h).sum().item()))
def run():
    _, unit, forests = encode_linear_planes(W, grid=E4M3_GRID, q256=1024, verify=False)
    st = materialize_stock(unit, forests, DEFAULT_CODE)
    return err(st["weight"].to(dev).float() * st["weight_scale"].to(dev).float())
for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
    w = (h / h.mean()) ** alpha; CUR["w"] = w / w.mean()
    sc.refit_channel_scale = refit_h; enc.viterbi_window = orig_vw
    r_only = run()
    enc.viterbi_window = vw_h; CUR["calls"] = 0
    both = run(); calls = CUR["calls"]
    sc.refit_channel_scale = orig_refit
    m_only = run()
    print(f"alpha {alpha:.2f}: refit-only {r_only[0]:.4f}/{r_only[1]:.4f}  metric-only {m_only[0]:.4f}/{m_only[1]:.4f}  both {both[0]:.4f}/{both[1]:.4f}  (metric calls {calls})")
