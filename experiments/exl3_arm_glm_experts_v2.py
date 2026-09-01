"""EXL3 at 4.0117 bpw vs Tessera at 4.0000, same GLM experts, same eval tokens.

**This is the comparator that matters.**  Tessera's kernel lane and EXL3 are
both **W4A16**, so unlike the NVFP4 comparison there is no activation-contract
mismatch to correct for -- the two formats answer the same question with the
same budget.

**Why quantize rather than decode Mia's artifact.**  ``exl3_arm_glm_experts.py``
tried to read ``GLM-5.3-Flash-EXL3-TR3-4bpw`` directly and got the classic
garbage signature -- ``norm_ratio`` 0.999, ``cos`` 0.001, ``rel_err`` sqrt(2):
right magnitude, no correlation.  Chasing it established two things worth
keeping:

  * The reader path is **exactly right**.  ``LinearEXL3.get_weight_tensor()``
    reproduces the quantizer's own verified reconstruction -- the formula
    ``quantize_exl3`` itself scores ``block_nmse`` against -- to rel 0.00044,
    cos 1.0000.  So a wrong reconstruction cannot be blamed on the reader.
  * The reconstruction of Mia's expert 0 correlates with **no** expert in that
    layer (top |cos| over all 288 is 0.0014, and expert 0 ranks 6th), so it is
    not an expert permutation either.

That leaves her stored trellis, and her artifact says as much itself:
``exl3-mcg-storage-abi.json`` carries ``serving_reader_qualified: false`` and
an empty ``qualified_tp_sizes``, with the reason *"ExLlamaV3 v0.0.43 has no
audited GLM-5.3 TP model load/inference receipt"*.  ``storage_checkpoint_verified``
is true -- the bytes were checked, the decode never was.  So rather than assert
a defect in someone else's checkpoint from a negative result, this runs EXL3's
**own quantizer** on the same weights.  That is a stronger arm anyway: it is
EXL3 at its best on this data, not EXL3 as one artifact happened to store it.

The rate matches by construction: K=4 trellis plus fp16 ``suh``/``svh``
diagonals is 4 + 16/2048 + 16/4096 = **4.0117 bpw** on these shapes -- the exact
figure measured from Mia's safetensors headers.

**Asymmetry, stated not netted:** ``quantize_exl3`` gets a real Hessian built
from ``X_fit`` and does LDL-ordered error compensation.  Tessera's encoder sees
no activations at all.  Every arm is scored on the disjoint ``X_eval``, so this
is held out, but the comparison hands EXL3 the same advantage NVFP4 had.
"""
import importlib, json, statistics as st, sys
import torch
from safetensors import safe_open

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
from vllm.model_executor.layers.quantization.exl3 import (   # noqa: E402
    make_linear_exl3, _install_exllamav3_namespace,
)

_install_exllamav3_namespace()          # stub the __init__ that drags in flash_attn
quantize_exl3 = importlib.import_module(
    "exllamav3.modules.quant.exl3_lib.quantize").quantize_exl3

BF16 = "/mnt/shared/models/GLM-5.3-Flash-BF16"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830/act"


def _get(mapping, name):
    with safe_open(f"{BF16}/{mapping[name]}", framework="pt") as h:
        return h.get_tensor(name)


def main():
    m = json.load(open(f"{BF16}/model.safetensors.index.json"))["weight_map"]
    rows, rels = [], []
    for layer in (5, 20, 42):
        blob = torch.load(
            f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
            map_location="cpu", weights_only=False)
        xa = blob["inputs"].float().cuda()
        half = xa.shape[0] // 2
        x_fit, x = xa[:half].contiguous(), xa[half:].contiguous()
        # H is the uncentred second moment over the FIT half only; quantize_exl3
        # divides by count and regularizes the diagonal itself.
        H = (x_fit.T @ x_fit).double().float().contiguous()
        for proj in ("gate_proj", "up_proj"):
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            w = _get(m, name).cuda()
            w_in = w.float().T.contiguous()          # (in_features, out_features)
            ref = x @ w.float().T
            H_data = {"H": H.clone(), "count": x_fit.shape[0],
                      "finalized": False, "device": "cuda:0"}
            wq, proxy, out = quantize_exl3(
                w_in, H_data,
                {"K": 4, "seed": 0, "devices": ["cuda:0"], "mcg": True,
                 "sigma_reg": 0.025, "apply_out_scales": None},
                return_weight_q=True,
            )
            # Score the SERIALISED tensors through the reader, not the in-loop
            # weight_q: what ships is what must be measured.
            lin = make_linear_exl3(out["trellis"], out["suh"], out["svh"],
                                   out["mcg"], out_dtype=torch.float16)
            w_hat = lin.get_weight_tensor().float()   # (in, out)
            drift = ((w_hat - wq.float()).norm() / wq.float().norm()).item()
            if drift > 1e-2:
                raise SystemExit(
                    f"{name}: reader disagrees with the quantizer by {drift:.5f} -- "
                    "the serialised tensors are not the reconstruction that was scored"
                )
            rel = ((x @ w_hat - ref).norm() / ref.norm()).item()
            cos = torch.nn.functional.cosine_similarity(
                (x @ w_hat).flatten(), ref.flatten(), dim=0).item()
            bpw = (4 * w.numel() + 16 * (w.shape[0] + w.shape[1])) / w.numel()
            rows.append(dict(layer=layer, proj=proj, rel=rel, bpw=bpw,
                             proxy=proxy, reader_drift=drift))
            rels.append(rel)
            print(f"{layer:>3} {proj:<10} rel_err {rel:.5f}  cos {cos:.4f}  "
                  f"bpw {bpw:.4f}  proxy {proxy:.5f}  reader_drift {drift:.2e}",
                  flush=True)
            del w, w_in, ref, wq, w_hat, lin
            torch.cuda.empty_cache()
        del H, x, x_fit, xa
        torch.cuda.empty_cache()

    print(f"\nEXL3 mean rel_err: {st.mean(rels):.5f}  (n={len(rels)}, "
          f"{st.mean(r['bpw'] for r in rows):.4f} bpw)")
    print("Tessera E2M1_K2 @ 4.0000 on the same six: 0.09738")
    print(f"ratio Tessera/EXL3 = {0.09738 / st.mean(rels):.4f}  "
          "(<1 means Tessera wins)")
    json.dump(rows, open("/work/experiments/results/exl3_arm_v2.json", "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
