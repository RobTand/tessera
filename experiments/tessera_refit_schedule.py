"""Which refit schedule, at which Viterbi count -- measured on the encoder
itself, not a re-implementation.

`61df165` made `scale_refit=k` mean k trellis passes and k refits with the
last refit trailing (`T(RT)^(k-1)R`); `cf82b00` had k refits BETWEEN k+1
passes (`(TR)^k T`).  This runs both encoders on the six battery tensors and
scores every schedule on the held-out output-space weight leg, so the doc's
schedule table is backed by a JSON and the encode time is the real encoder's,
one job on the box.

    PYTHONPATH=src:experiments:/home/rob/prismaquant python experiments/tessera_refit_schedule.py
"""
import argparse, importlib.util, json, subprocess, sys, time
from fractions import Fraction
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tessera_fp4_native_levers import ACT, CC, GRID, GROUP, HALF, SRC, Scorer, quant_a4  # noqa: E402
from tessera.alphabet import build_forest  # noqa: E402
from tessera.decode import reconstruct_unit  # noqa: E402
from tessera.encode import encode_unit  # noqa: E402
from tessera.grammar import bresenham_rate_schedule  # noqa: E402
from tessera.manifest import RotationState  # noqa: E402
from prismaquant.nvfp4_activation_contract import (  # noqa: E402
    nvfp4_activation_qdq_served, select_mse_grid_input_global_scale)

OLD_REV = "cf82b00"


def load_old_encoder(scratch: Path):
    """`cf82b00`'s encode.py as `tessera.encode_cf82b00`, so its relative
    imports resolve against the current package."""
    src = subprocess.check_output(["git", "show", f"{OLD_REV}:src/tessera/encode.py"], text=True)
    path = scratch / "encode_cf82b00.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("tessera.encode_cf82b00", path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "tessera"
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.encode_unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--new", type=int, nargs="+", default=[0, 1, 2, 3, 4, 6])
    ap.add_argument("--old", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--scratch", default="/home/rob/tmp")
    ap.add_argument("--out", default="experiments/results/tessera_refit_schedule.json")
    a = ap.parse_args()
    logf = open(a.out.replace(".json", ".log"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    old_encode = load_old_encoder(Path(a.scratch))
    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    schedules = [("new", k, "T" if k == 0 else "TR" * k) for k in a.new] + \
                [("old", k, "TR" * k + "T") for k in a.old]
    out = {"tensors": [], "act": [], "arms": {}, "args": vars(a), "old_rev": OLD_REV,
           "new_rev": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()}
    for layer in a.layers:
        blob = torch.load(f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                          map_location="cpu", weights_only=False)
        xa = blob["inputs"].float().cuda()
        n = xa.shape[0] // 2
        x_fit, x_ev = xa[:n].contiguous(), xa[n:].contiguous()
        g = select_mse_grid_input_global_scale([x_fit])
        xq_s = nvfp4_activation_qdq_served(x_ev, g).float()
        xq_a = quant_a4(x_ev)
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name).contiguous().cuda().float()
            sc = Scorer(w, x_ev, xq_s, xq_a)
            act = float((xq_s @ w.T - sc.y).norm() / sc.ny)
            out["tensors"].append(f"L{layer}.{proj}"); out["act"].append(act)
            rates = bresenham_rate_schedule(Fraction(GRID.rate_cap), w.shape[1], cap=GRID.rate_cap)
            forests = {r: build_forest(r, grid=GRID) for r in sorted(set(rates))}
            log(f"\n== L{layer} {proj} {tuple(w.shape)}  act leg served {act:.5f}")
            for which, k, sched in schedules:
                enc = encode_unit if which == "new" else old_encode
                torch.cuda.synchronize(); t0 = time.time()
                unit = enc(w, forests, rates, CC, rotation=RotationState.NONE,
                           with_diagonals=False, completion=0, group=GROUP, half=HALF,
                           scale_refit=k)
                torch.cuda.synchronize(); dt = time.time() - t0
                hat = reconstruct_unit(unit, forests, CC)
                r = sc(hat, bpp=4.0, passes=(1 if k == 0 else k) if which == "new" else k + 1,
                       schedule=sched, seconds=dt, encoder=which, scale_refit=k)
                key = f"{sched} ({which} k={k})"
                out["arms"].setdefault(key, []).append(r)
                log(f"    {key:<26} passes={r['passes']} wt={r['wt']:.5f} out={r['out']:.5f} "
                    f"W4A4={r['both_served']:.5f}  {dt:6.2f}s")
            json.dump(out, open(a.out, "w"), indent=1)
            del w, sc; torch.cuda.empty_cache()
        del xa, x_fit, x_ev, xq_s, xq_a; torch.cuda.empty_cache()
    base = out["arms"]["T (new k=0)"]
    log("\n| schedule | passes | vs amax plane | min-max | encode s |")
    for key, v in sorted(out["arms"].items(), key=lambda kv: (kv[1][0]["passes"], kv[0])):
        ratio = [b["out"] / r["out"] for b, r in zip(base, v)]
        log(f"| {key} | {v[0]['passes']} | {sum(ratio)/len(ratio):.3f}x | {min(ratio):.3f}-{max(ratio):.3f} "
            f"| {sum(r['seconds'] for r in v)/len(v):.2f} |")
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
