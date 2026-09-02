"""2026-09-02 dense-outlier study (Qwen3-0.6B, E4M3/CHANNEL/window L=14 wire). Run with
PYTHONPATH=src TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache <prismaquant-cu130 python>.

Layer-2 down_proj on the E4M3 route: production encode vs an amax-aware per-row
initial sigma vs a global lower sigma.  Weighted error = sqrt(sum_j h_j sum_i e_ij^2 /
sum_j h_j sum_i w_ij^2); plain = sqrt(sum e^2 / sum w^2)."""
import math, sys, time, torch
from safetensors import safe_open
import tessera.scale_channel as sc
from tessera.alphabet import E4M3_GRID
from tessera.export import DEFAULT_CODE, encode_linear_planes
from tessera.encode import window_table
from tessera.stock import materialize_stock
from tessera.scale_channel import default_channel_sigma, channel_global, land_channel_scale

NAME = "model.layers.2.mlp.down_proj"
dev = "cuda" if torch.cuda.is_available() else "cpu"
src = "/home/rob/models/Qwen3-0.6B/model.safetensors"
with safe_open(src, framework="pt") as f:
    W = f.get_tensor(NAME + ".weight").to(dev, torch.float32).contiguous()
H = torch.load("/home/rob/tessera-runs/stock/h_diag.pt")[NAME].to(dev, torch.float32)
rows, cols = W.shape
rms = W.pow(2).mean(1).sqrt()
zmax = W.abs().amax(1) / rms
sig0 = default_channel_sigma(E4M3_GRID)
tab = window_table(E4M3_GRID, 14, sigma=sig0, seed=0, half=16, device="cpu")
vals = torch.tensor(E4M3_GRID.values)[tab.long()].abs()
reach = float(vals.max()) / sig0
print(f"unit {NAME} {tuple(W.shape)}  default sigma {sig0:.1f} units  table max |code| {float(vals.max()):.0f} = {reach:.2f} sigma")
print(f"rows with max z > reach: {(zmax > reach).sum().item()}  max z {zmax.max().item():.2f} (row {zmax.argmax().item()})")
print(f"h: max/median {(H.max()/H.median()).item():.3g}  top-4 columns share {(H.topk(4).values.sum()/H.sum()).item():.3f}")

def errors(tag, Wd):
    E = Wd - W
    plain = math.sqrt((E * E).sum().item() / (W * W).sum().item())
    wt = math.sqrt(((E * E).sum(0) * H).sum().item() / ((W * W).sum(0) * H).sum().item())
    c = int(H.argmax()); r = int(W[:, c].abs().argmax())
    col_sse = (E[:, c] ** 2).sum().item(); ent = (E[r, c] ** 2).item()
    print(f"{tag:26s} plain {plain:.4f}  weighted {wt:.4f}  | col {c}: sse {col_sse:.3e} outlier entry {ent:.3e} ({100*ent/col_sse:.0f}%)"
          f"  w {W[r,c].item():+.4f} -> {Wd[r,c].item():+.4f}  | weighted w/o that entry {math.sqrt((((E*E).sum(0)*H).sum().item() - ent*H[c].item()) / ((W*W).sum(0)*H).sum().item()):.4f}")

def encode(tag, **kw):
    t0 = time.time()
    exported, unit, forests = encode_linear_planes(W, grid=E4M3_GRID, q256=1024, name=NAME, verify=False, **kw)
    st = materialize_stock(unit, forests, DEFAULT_CODE)
    Wd = st["weight"].to(dev).float() * st["weight_scale"].to(dev).float()
    errors(f"{tag} ({time.time()-t0:.0f}s)", Wd)
    return Wd

# A: production
encode("A production")

# C: global lower sigma so a 9-sigma row fits the table
sigC = sig0 * reach / float(zmax.max())
encode(f"C global sigma {sigC:.1f}", channel_sigma=sigC)

# B: per-row amax-aware initial sigma (monkeypatch), default elsewhere
orig = sc.initial_channel_scale
def amax_aware(work, sigma):
    r = work.float().pow(2).mean(1).sqrt(); z = work.float().abs().amax(1) / r
    row_sigma = torch.full_like(r, float(sigma))
    over = z > reach
    row_sigma[over] = float(sigma) * reach / z[over]
    scale = r / row_sigma
    g = channel_global(scale)
    stored, eff = land_channel_scale(scale, g)
    return stored, eff, g
sc.initial_channel_scale = amax_aware
try:
    encode("B per-row amax-aware")
finally:
    sc.initial_channel_scale = orig

# reference: production NVFP4 (GPTQ+JSO, 4.5 bpp) on the same unit
p = "/home/rob/dq-runs/fc45-0p6b-nvfp4/exported"
import glob, json
for fn in glob.glob(p + "/*.safetensors"):
    with safe_open(fn, framework="pt") as f:
        if NAME + ".weight_packed" in f.keys():
            pk = f.get_tensor(NAME + ".weight_packed").to(dev); sc_ = f.get_tensor(NAME + ".weight_scale").to(dev).float()
            gs = f.get_tensor(NAME + ".weight_global_scale").to(dev).float()
            break
e2m1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=dev)
lo, hi = (pk & 0xF).long(), (pk >> 4).long()
def dq(n): return e2m1[n & 7] * torch.where(n >= 8, -1.0, 1.0)
q = torch.stack([dq(lo), dq(hi)], -1).reshape(rows, cols)
Wn = q * (sc_ / gs).repeat_interleave(16, dim=1)
errors("ref NVFP4 GPTQ+JSO 4.5", Wn)
