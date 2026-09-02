"""2026-09-02 dense-outlier study (Qwen3-0.6B, E4M3/CHANNEL/window L=14 wire). Run with
PYTHONPATH=src TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache <prismaquant-cu130 python>.

Arm E on layer-2 down_proj: amax-aware per-row sigma (B) + h on the branch metric and
on the LS refit.  Same table, same wire."""
import math, sys, torch
from safetensors import safe_open
import tessera.encode as enc
import tessera.scale_channel as sc
from tessera.alphabet import E4M3_GRID
from tessera.export import DEFAULT_CODE, encode_linear_planes
from tessera.encode import window_table
from tessera.stock import materialize_stock
from tessera.scale_channel import default_channel_sigma, channel_global, land_channel_scale

names = sys.argv[1:] or ["model.layers.2.mlp.down_proj"]
dev = "cuda"
Hall = torch.load("/home/rob/tessera-runs/stock/h_diag.pt")
sig0 = default_channel_sigma(E4M3_GRID)
tab = window_table(E4M3_GRID, 14, sigma=sig0, seed=0, half=16, device="cpu")
reach = float(torch.tensor(E4M3_GRID.values)[tab.long()].abs().max()) / sig0
orig_init, orig_refit, orig_vw = sc.initial_channel_scale, sc.refit_channel_scale, enc.viterbi_window
CUR = {}
def amax_aware(work, sigma):
    r = work.float().pow(2).mean(1).sqrt(); z = work.float().abs().amax(1) / r
    row_sigma = torch.full_like(r, float(sigma)); over = z > reach
    row_sigma[over] = float(sigma) * reach / z[over]
    scale = r / row_sigma; g = channel_global(scale)
    stored, eff = land_channel_scale(scale, g); return stored, eff, g
def refit_h(work, units, stored, g):
    sq = CUR["h"].sqrt()[None, :]
    return orig_refit(work.float() * sq, units.float() * sq, stored, g)
def vw_h(sub, vectors, window_bits, present, weights=None, **kw):
    hw = CUR["h"][None, :].expand_as(sub)
    weights = hw if weights is None else weights * hw
    return orig_vw(sub, vectors, window_bits, present, weights=weights, **kw)
def err(W, Wd, h):
    E = Wd - W
    return (math.sqrt((E*E).sum().item() / (W*W).sum().item()),
            math.sqrt(((E*E).sum(0)*h).sum().item() / ((W*W).sum(0)*h).sum().item()))
def run(W, tag):
    _, unit, forests = encode_linear_planes(W, grid=E4M3_GRID, q256=1024, verify=False)
    st = materialize_stock(unit, forests, DEFAULT_CODE)
    return err(W, st["weight"].to(dev).float() * st["weight_scale"].to(dev).float(), CUR["h"])
with safe_open("/home/rob/models/Qwen3-0.6B/model.safetensors", framework="pt") as f:
    for name in names:
        W = f.get_tensor(name + ".weight").to(dev, torch.float32).contiguous()
        h = Hall[name].to(dev, torch.float32); CUR["h"] = h / h.mean()
        sc.initial_channel_scale, sc.refit_channel_scale, enc.viterbi_window = orig_init, orig_refit, orig_vw
        a = run(W, "A")
        sc.initial_channel_scale = amax_aware
        b = run(W, "B")
        sc.refit_channel_scale = refit_h
        enc.viterbi_window = vw_h
        e = run(W, "E")
        enc.viterbi_window = orig_vw
        e2 = run(W, "E-refit-only")
        print(f"{name}: plain/weighted  A {a[0]:.4f}/{a[1]:.4f}  B {b[0]:.4f}/{b[1]:.4f}  B+h-refit {e2[0]:.4f}/{e2[1]:.4f}  B+h-metric+h-refit {e[0]:.4f}/{e[1]:.4f}")
