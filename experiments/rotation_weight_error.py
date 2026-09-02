"""Where a rotated checkpoint's 4-bit loss sits: weight space or output space.

R1 folds an orthogonal R into the checkpoint, so ``||W||_F`` and ``||dW||_F``
are directly comparable between a rotated arm and its unrotated twin -- the
Frobenius norm does not see the rotation.  The *output* error is not
rotation-invariant in the same trivial way: it is ``sum_j h_j ||dW[:, j]||^2``
with ``h_j = E[x_j^2]`` measured on whichever model the arm was built from, and
rotation flattens that h (measured: max/median 645 -> 3.8 on Qwen3-0.6B) by
moving the anisotropy into off-diagonal terms a diag-h encoder cannot see.

Reporting both separates two hypotheses for a rotated arm that serves worse:

* relative ``||dW||_F`` much worse   -> the grid/alphabet interacts badly with
  the Gaussianised rows (a weight-space failure);
* relative ``||dW||_F`` about equal but h-weighted error worse -> the encoder
  is spending its bits uniformly because its cost model went uniform, while the
  true output geometry did not (an objective failure, not a grid failure).

usage: rotation_weight_error.py <out.json> [--arms name=dir ...]
Arms are named ``rot:<label>=<dir>`` or ``unrot:<label>=<dir>`` so each is
scored against the source model it was actually built from.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tessera.stock import stock_dequant  # noqa: E402

SOURCES = {
    "unrot": "/home/rob/models/Qwen3-0.6B",
    "rot": "/mnt/shared/tessera-runs/rotation/qwen3-0.6b-rot-seed0",
}


def diag_h(src: str, calib_text: str) -> tuple[dict[str, torch.Tensor], str]:
    """diag(H) per Linear from a BF16 forward, on text disjoint from the KL corpus."""
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).cuda().eval()
    ids = tok(calib_text, return_tensors="pt").input_ids[:, : 8 * 512].cuda()
    ids = ids[:, : (ids.shape[1] // 512) * 512].reshape(-1, 512)
    h: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    def hook(name):
        def fn(_mod, inp, _out):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            h[name] = h.get(name, 0) + (x * x).sum(0)
            counts[name] = counts.get(name, 0) + x.shape[0]
        return fn

    handles = [m.register_forward_hook(hook(n)) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and "proj" in n and "layers." in n]
    with torch.no_grad():
        for row in ids:
            model(row[None])
    for handle in handles:
        handle.remove()
    out = {n: (v / counts[n]) for n, v in h.items()}
    del model
    torch.cuda.empty_cache()
    return out, f"{ids.numel()} tokens"


def score(arm_dir: str, src: str, hs: dict[str, torch.Tensor]) -> dict:
    source = safe_open(f"{src}/model.safetensors", "pt")
    handle = safe_open(f"{arm_dir}/model.safetensors", "pt")
    keys = set(handle.keys())
    fro_num = fro_den = hw_num = hw_den = 0.0
    per: dict[str, dict[str, float]] = {}
    for name, hv in hs.items():
        if f"{name}.weight_packed" not in keys and f"{name}.weight_scale" not in keys:
            continue
        weight = source.get_tensor(f"{name}.weight").cuda().float()
        hv = hv.cuda()
        if f"{name}.weight_packed" in keys:
            tensors = {s: handle.get_tensor(f"{name}.{s}").cuda()
                       for s in ("weight_packed", "weight_scale", "weight_global_scale")}
        else:
            tensors = {s: handle.get_tensor(f"{name}.{s}").cuda() for s in ("weight", "weight_scale")}
        delta = stock_dequant(tensors) - weight
        f_e, f_r = float(delta.pow(2).sum()), float(weight.pow(2).sum())
        h_e = float((delta.pow(2).sum(0) * hv).sum())
        h_r = float((weight.pow(2).sum(0) * hv).sum())
        per[name] = {"fro": f_e / f_r, "hw": h_e / h_r}
        fro_num += f_e; fro_den += f_r; hw_num += h_e; hw_den += h_r
    return {"fro_total": fro_num / fro_den, "hw_total": hw_num / hw_den,
            "units": len(per), "per_unit": per}


def main() -> int:
    out_path = Path(sys.argv[1])
    arms = [a for a in sys.argv[2:] if "=" in a]
    if not arms:
        print("no arms given", file=sys.stderr)
        return 2
    text = (open(f"{SOURCES['unrot']}/README.md").read() + "\n"
            + open(Path(__file__).resolve().parent.parent / "docs/tessera-one-format.md").read())
    result: dict[str, dict] = {}
    for side in ("unrot", "rot"):
        wanted = [a for a in arms if a.startswith(f"{side}:")]
        if not wanted:
            continue
        hs, note = diag_h(SOURCES[side], text)
        ratios = sorted((v.max() / v.median()).item() for v in hs.values())
        print(f"[{side}] diag(H) over {len(hs)} Linears ({note}): "
              f"max/median median {ratios[len(ratios)//2]:.1f} max {ratios[-1]:.1f}", flush=True)
        result[f"{side}_h_max_over_median_median"] = ratios[len(ratios) // 2]
        for spec in wanted:
            label, arm_dir = spec.split("=", 1)
            label = label.split(":", 1)[1]
            scored = score(arm_dir, SOURCES[side], hs)
            result[label] = {"side": side, "dir": arm_dir, **scored}
            print(f"{label:24s} rel ||dW||_F^2 {scored['fro_total']:.4e}   "
                  f"h-weighted {scored['hw_total']:.4e}   ({scored['units']} units)", flush=True)
    out_path.write_text(json.dumps(result, indent=1))
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
