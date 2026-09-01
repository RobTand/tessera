"""Does the 4.0 bpp parity result survive more than one tensor?

experiments/ktuple.py measures one Linear of one model.  The claim it supports
-- free-16 k=2 at 4.0 bpp matches NVFP4 at 4.5 -- is the one worth building a
kernel for, so it is the one that has to hold on tensors it was not tuned on.
Two models, several roles, kurtosis reported because that is what should
predict where a Gaussian-fit grid stops working.
"""
import sys, glob, torch; sys.path.insert(0, "/home/rob/tessera/src")
from safetensors import safe_open
from tessera.alphabet import build_forest, tuple_grid, lloyd_max_grid, E2M1_GRID
from tessera.encode import encode_unit, _pack_scales
from tessera.decode import reconstruct_unit
from tessera.trellis import ConvCode
from tessera.manifest import RotationState

dev = "cuda"; CC = ConvCode(memory=6)
LM16 = lloyd_max_grid(16)
GRIDS = {
    "E2M1 k=1 R=3 (3.5)":   (E2M1_GRID, 3, 3.5),
    "E2M1 k=2 R=7 (4.0)":   (tuple_grid(E2M1_GRID, 2), 7, 4.0),
    "free16 k=1 R=3 (3.5)": (LM16, 3, 3.5),
    "free16 k=2 R=7 (4.0)": (tuple_grid(LM16, 2), 7, 4.0),
}
FORESTS = {name: {R: build_forest(R, grid=g)} for name, (g, R, _) in GRIDS.items()}

def nvfp4_rtn(W):
    from tessera.encode import e2m1_value_table
    _, _, eff = _pack_scales(W, 32, 16)
    scale = torch.repeat_interleave(eff, 16).reshape(W.shape)
    v = e2m1_value_table(dev)
    t = W / scale
    return ((v[((t.unsqueeze(-1) - v) ** 2).argmin(-1)] * scale - W).norm() / W.norm()).item()

def measure(W):
    out = {}
    cols = W.shape[1]
    for name, (grid, R, _) in GRIDS.items():
        u = encode_unit(W, FORESTS[name], (R,) * cols, CC,
                        rotation=RotationState.NONE, with_diagonals=False, completion=0)
        rec = reconstruct_unit(u, FORESTS[name], CC, completion=0).float()
        out[name] = ((rec - W).norm() / W.norm()).item()
    out["NVFP4 RTN (4.5)"] = nvfp4_rtn(W)
    return out

WANT = ("gate_proj", "down_proj", "up_proj", "q_proj", "o_proj",
        "in_proj_qkv", "out_proj")
MODELS = {
    "Qwen3.8-27B": "models--Qwen--Qwen3.8-27B",
    "Qwen3-4B": "models--Qwen--Qwen3-4B",
}
cols_hdr = list(GRIDS) + ["NVFP4 RTN (4.5)"]
print(f"{'model / tensor':<34}{'kurt':>6}" + "".join(f"{c.split(' ')[0]+c.split('(')[1][:-1]:>13}" for c in cols_hdr))
ratios = {n: [] for n in cols_hdr}
for label, repo in MODELS.items():
    seen = set()
    for path in sorted(glob.glob(f"/home/rob/.cache/huggingface/hub/{repo}/snapshots/*/model*.safetensors"))[:3]:
        with safe_open(path, "pt") as f:
            for key in f.keys():
                role = next((w for w in WANT if key.endswith(w + ".weight")), None)
                if role is None or role in seen or ".layers.1." not in key:
                    continue
                W = f.get_tensor(key)
                if W.ndim != 2 or W.shape[0] < 512 or W.shape[1] < 512:
                    continue
                seen.add(role)
                W = W[:1024, :2048].to(dev).float().contiguous()
                k = ((W - W.mean()) ** 4).mean() / ((W - W.mean()) ** 2).mean() ** 2
                res = measure(W)
                base = res["NVFP4 RTN (4.5)"]
                for n in cols_hdr:
                    ratios[n].append(res[n] / base)
                print(f"{label + ' ' + role:<34}{k:>6.2f}" +
                      "".join(f"{res[n]:>13.5f}" for n in cols_hdr))
print()
print(f"{'MEAN ratio vs NVFP4@4.5 bpp':<40}" +
      "".join(f"{sum(ratios[n])/len(ratios[n]):>13.3f}" for n in cols_hdr))
print(f"{'WORST ratio vs NVFP4@4.5 bpp':<40}" +
      "".join(f"{max(ratios[n]):>13.3f}" for n in cols_hdr))
