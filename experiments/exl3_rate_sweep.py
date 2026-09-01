"""EXL3's own rate-distortion curve, so the comparison can be made at matched
*payload* bits and not only at matched *total* bytes.

`exl3-head-to-head-2026-09-01.md` compared Tessera at 4.0000 bpp against EXL3 at
4.0117 bpw and found EXL3 1.72x better.  Matched total bytes is the right axis
for shipping and the wrong axis for asking "is the coder worse", because the
two formats spend that budget very differently:

    Tessera E2M1_K2 @ 4.0000 bpp = 3.5 payload + 0.5 scale plane
        (E8M0 base per 32 columns + a 4-bit refinement per 16 = 8/32 + 4/16)
    EXL3 K=4      @ 4.0117 bpw  = 4.0 payload + 0.0117 (fp16 suh/svh diagonals)

So the head-to-head handed EXL3 **14% more payload bits** at the same size, and
Tessera cannot answer: body and completion sum to the grid's cap, so 3.5 is the
ceiling of the serialisable K2 family (`tessera-rung-is-not-a-rate`).  That is a
real shipping disadvantage and it is also not a coder defect -- it is a rate
allocation, and the two claims need separating before either steers work.

Running EXL3 at K in {2, 3, 4} gives its slope.  **K=3 is the decisive point:**
it has 3.0 payload bits, exactly Tessera's ``E2M1_K1`` rung, so
``EXL3@K3 vs Tessera_K1 = 0.12666`` is a matched-payload comparison with no
extrapolation in it.  If they land together, the coders are equally efficient
per payload bit and the whole 1.72x is overhead plus calibration.  If EXL3 is
still well ahead at matched payload, the coder gap is real and the rate-ceiling
work will not close it.

Same six tensors, same fit/eval split, scored on the disjoint eval half, same
reader-drift assertion as the v2 arm -- the serialised tensors are what gets
scored.
"""
import importlib
import json
import statistics as st
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
from vllm.model_executor.layers.quantization.exl3 import (   # noqa: E402
    make_linear_exl3, _install_exllamav3_namespace,
)

_install_exllamav3_namespace()
quantize_exl3 = importlib.import_module(
    "exllamav3.modules.quant.exl3_lib.quantize").quantize_exl3

BF16 = "/mnt/shared/models/GLM-5.3-Flash-BF16"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830/act"
OUT = f"/work/experiments/results/exl3_rate_sweep_K{sys.argv[1]}.json"
# ONE K PER PROCESS.  Running K=2,3,4 in a single process gives correct output
# for the first K and rel_err ~55-65 for the rest, at unchanged reader drift --
# so the reader agrees with a quantizer that produced garbage.  Something in
# quantize_exl3's call path is stateful across K; rather than diagnose someone
# else's cache, isolate.  The K=4 arm re-measuring exl3_arm_glm_experts_v2's
# 0.05653 is the check that isolation worked.
KS = (int(sys.argv[1]),)
# Tessera, same six tensors, uncompensated.  payload bits -> mean rel_err.
TESSERA = {3.0: 0.12666, 3.5: 0.09738}


def main():
    mapping = json.load(open(f"{BF16}/model.safetensors.index.json"))["weight_map"]
    arms, shapes = {}, []
    for layer in (5, 20, 42):
        blob = torch.load(
            f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
            map_location="cpu", weights_only=False)
        xa = blob["inputs"].float().cuda()
        half = xa.shape[0] // 2
        x_fit, x = xa[:half].contiguous(), xa[half:].contiguous()
        H = (x_fit.T @ x_fit).double().float().contiguous()
        for proj in ("gate_proj", "up_proj"):
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{BF16}/{mapping[name]}", framework="pt") as handle:
                w = handle.get_tensor(name).cuda()
            w_in = w.float().T.contiguous()
            ref = x @ w.float().T
            den = ref.norm()
            shapes.append(list(w.shape))
            for K in KS:
                H_data = {"H": H.clone(), "count": x_fit.shape[0],
                          "finalized": False, "device": "cuda:0"}
                wq, proxy, out = quantize_exl3(
                    w_in, H_data,
                    {"K": K, "seed": 0, "devices": ["cuda:0"], "mcg": True,
                     "sigma_reg": 0.025, "apply_out_scales": None},
                    return_weight_q=True)
                lin = make_linear_exl3(out["trellis"], out["suh"], out["svh"],
                                       out["mcg"], out_dtype=torch.float16)
                w_hat = lin.get_weight_tensor().float()
                drift = float((w_hat - wq.float()).norm() / wq.float().norm())
                if drift > 1e-2:
                    raise SystemExit(
                        f"{name} K={K}: reader disagrees with the quantizer by "
                        f"{drift:.5f} -- the serialised tensors are not what was scored")
                rel = float(((x @ w_hat - ref).norm() / den))
                arms.setdefault(K, []).append(rel)
                print(f"{layer:>3} {proj:<10} K={K}  rel_err {rel:.5f}  "
                      f"drift {drift:.1e}", flush=True)
                del wq, out, lin, w_hat
                torch.cuda.empty_cache()
            del w, w_in, ref
            torch.cuda.empty_cache()
        del H, x, x_fit, xa
        torch.cuda.empty_cache()

    rows, cols = shapes[0]
    overhead = 16 * (rows + cols) / (rows * cols)
    print(f"\n{'K':>3} {'payload':>8} {'total bpw':>10} {'rel_err':>9}   "
          f"{'Tessera @ same payload':>24}")
    means = {}
    for K in KS:
        mean = st.mean(arms[K])
        means[K] = mean
        match = TESSERA.get(float(K))
        note = f"{match:.5f}  ({match / mean:.3f}x)" if match else "-"
        print(f"{K:>3} {float(K):>8.2f} {K + overhead:>10.4f} {mean:>9.5f}   {note:>24}")
    print(f"\nTessera payload ceiling is 3.5 bits (body + completion = cap); its "
          f"scale plane costs 0.5 bpp on top.")
    json.dump(dict(means={str(k): v for k, v in means.items()},
                   raw={str(k): v for k, v in arms.items()},
                   overhead=overhead, shape=shapes[0], tessera=TESSERA),
              open(OUT, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
