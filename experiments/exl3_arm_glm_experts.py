"""SUPERSEDED 2026-09-01 by ``exl3_arm_glm_experts_v2.py``.

This reads Mia's TR3 artifact directly and gets the garbage signature --
norm_ratio 0.999, cos 0.001, rel_err ~sqrt(2).  The reader is not at fault: it
reproduces the quantizer's own verified reconstruction to rel 0.00044 / cos
1.0000, it is not an expert permutation (top |cos| over all 288 experts is
0.0014), and the version and MCG codebook match the artifact's own ABI.  The
artifact declares ``serving_reader_qualified: false`` -- its bytes were
verified, its decode never was.  v2 quantizes the same weights with EXL3's own
quantizer instead: a fairer arm and a stronger one.  Kept as the record of what
was excluded.  See docs/measurements/exl3-head-to-head-2026-09-01.md.

Original docstring follows.

EXL3's own functional error on the same GLM experts, same eval tokens.

This is the comparator that matters.  Tessera's kernel lane and EXL3 are both
**W4A16** at ~4.0 bpw, so unlike the NVFP4 comparison there is no activation-
contract mismatch to correct for -- the two formats are answering the same
question with the same budget.

Runs inside Mia's image, which is where ``exllamav3_ext`` and vLLM's
``exl3.py`` live (the base vLLM image has neither -- exl3 support is Mia's
addition, not stock).  The decode goes through ``execute_exl3_linear``, which
requires ``suh``/``svh``: they are the input/output Hadamard diagonals, and
without them the result comes back with the right *norm* and no correlation to
the reference at all -- correct magnitude in the wrong basis, which is exactly
what a silent omission of them looks like.
"""
import json, statistics as st, sys
import torch
from safetensors import safe_open

BF16 = "/mnt/shared/models/GLM-5.3-Flash-BF16"
EXL3 = "/mnt/shared/models/GLM-5.3-Flash-EXL3-TR3-4bpw"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830/act"

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
from vllm.model_executor.layers.quantization.exl3 import execute_exl3_linear


def _map(path):
    return json.load(open(f"{path}/model.safetensors.index.json"))["weight_map"]


def _get(path, mapping, name):
    with safe_open(f"{path}/{mapping[name]}", framework="pt") as h:
        return h.get_tensor(name)


def main():
    bf16_map, exl3_map = _map(BF16), _map(EXL3)
    out = {}
    ratios = []
    for layer in (5, 20, 42):
        blob = torch.load(
            f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
            map_location="cpu", weights_only=False)
        x_all = blob["inputs"].float().cuda()
        x = x_all[x_all.shape[0] // 2:].contiguous()      # same eval half
        for proj in ("gate_proj", "up_proj"):
            base = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}"
            w = _get(BF16, bf16_map, base + ".weight").cuda().float()
            ref = x @ w.T
            parts = {s: _get(EXL3, exl3_map, f"{base}.{s}").cuda()
                     for s in ("trellis", "suh", "svh", "mcg")}
            y = execute_exl3_linear(x, parts["trellis"], parts["suh"],
                                    parts["svh"], parts["mcg"],
                                    out_dtype=torch.float32)
            rel = ((y - ref).norm() / ref.norm()).item()
            cos = torch.nn.functional.cosine_similarity(
                y.flatten(), ref.flatten(), dim=0).item()
            nrm = (y.norm() / ref.norm()).item()
            out[f"{layer}.{proj}"] = rel
            ratios.append(rel)
            print(f"{layer:>3} {proj:<10} rel_err {rel:.5f}  cos {cos:.4f}  "
                  f"norm_ratio {nrm:.4f}", flush=True)
            del w, ref, parts, y
            torch.cuda.empty_cache()
    print(f"\nEXL3 mean rel_err: {st.mean(ratios):.5f}  (n={len(ratios)})")
    print("JSON " + json.dumps(out))


if __name__ == "__main__":
    sys.exit(main())
