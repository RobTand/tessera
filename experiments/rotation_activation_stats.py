#!/usr/bin/env python
"""Calibrate NVFP4 ``input_global_scale`` and census the input distribution.

Two jobs, one forward, because they must see the same activations:

1. **The A-side static scale.**  compressed-tensors' W4A4 NVFP4 scheme stores
   one ``input_global_scale`` per Linear and derives every serve-time per-16
   block scale from it.  PrismaQuant's default policy is the legacy
   ``FP4_MAX / amax = 6 / amax``, so a block whose own amax is more than 512x
   below the calibrated tensor amax lands in the E4M3 subnormals and loses
   most of its precision.  On a dense model with residual outliers that is
   almost every block, which is the thing an R1 rotation is supposed to fix.
   Fused siblings share one value (they read one tensor, and vLLM builds one
   method per fused module), taken as the min scale = max amax.

2. **The census.**  Per input column ``j`` of each Linear, the second moment
   ``h_j = E[x_j^2]`` -- the diagonal Hessian proxy the stock-lane receipt
   uses.  Reported as ``max/median`` and the kurtosis of ``h``, plus the
   kurtosis of the raw activation values.  A per-row scale plane cannot see a
   heavy ``h``; a per-16 block scale can.

The same script runs on the unrotated and the rotated checkpoint, so the two
numbers are the same measurement.  Only the modules that read the residual
stream change under R1 (q/k/v/gate/up); ``o_proj`` (R2) and ``down_proj``
(R4) read bases R1 does not touch, and their rows are the control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

FP4_MAX = 6.0
FUSED = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
         ("mlp.gate_proj", "mlp.up_proj"))


def kurtosis(x: torch.Tensor) -> float:
    x = x.double().flatten()
    x = x - x.mean()
    var = x.pow(2).mean()
    return float((x.pow(4).mean() / var.pow(2)).item()) if var > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="stats json")
    ap.add_argument("--scales-out", type=Path, default=None,
                    help="safetensors of <module>.input_global_scale, the exporter's donor shape")
    ap.add_argument("--text", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.model))
    # bf16 forward: the production amax values are bf16-representable, so this
    # is the arithmetic the calibrated scale was and is taken in.
    model = AutoModelForCausalLM.from_pretrained(str(args.model), dtype=torch.bfloat16)
    model = model.to(args.device).eval()

    ids = tok(args.text.read_text(), return_tensors="pt").input_ids[0, : args.tokens]
    chunks = ids[: (len(ids) // args.seqlen) * args.seqlen].view(-1, args.seqlen)
    print(f"{chunks.shape[0]} chunks x {args.seqlen} tokens")

    amax: dict[str, float] = {}
    hsum: dict[str, torch.Tensor] = {}
    m2: dict[str, torch.Tensor] = {}
    m4: dict[str, torch.Tensor] = {}
    count: dict[str, int] = {}
    handles = []

    def hook(name):
        def fn(_module, inputs, _output):
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()
            amax[name] = max(amax.get(name, 0.0), float(x.abs().max()))
            sq = x.pow(2).sum(0)
            hsum[name] = sq if name not in hsum else hsum[name] + sq
            m2[name] = sq.sum() if name not in m2 else m2[name] + sq.sum()
            m4[name] = x.pow(4).sum() if name not in m4 else m4[name] + x.pow(4).sum()
            count[name] = count.get(name, 0) + x.shape[0]
        return fn

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.startswith("model.layers."):
            handles.append(module.register_forward_hook(hook(name)))

    with torch.no_grad():
        for i in range(chunks.shape[0]):
            model(chunks[i : i + 1].to(args.device))
    for handle in handles:
        handle.remove()

    stats = {}
    for name in sorted(amax):
        h = hsum[name] / count[name]
        n = count[name] * h.numel()
        mean2 = float(m2[name] / n)
        stats[name] = {
            "amax": amax[name],
            "input_global_scale": FP4_MAX / amax[name],
            "h_max_over_median": float(h.max() / h.median()),
            "h_max_over_mean": float(h.max() / h.mean()),
            "h_kurtosis": kurtosis(h),
            "value_kurtosis": float(m4[name] / n) / (mean2 ** 2) if mean2 > 0 else float("nan"),
            "columns": int(h.numel()),
        }

    # fused siblings serve as one module: one scale, the most conservative
    fused_groups = {}
    for name in list(stats):
        for group in FUSED:
            for member in group:
                if name.endswith("." + member):
                    prefix = name[: -len(member)]
                    fused_groups.setdefault(prefix + "/".join(group), []).append(name)
    for group, members in fused_groups.items():
        shared = min(stats[m]["input_global_scale"] for m in members)
        for m in members:
            stats[m]["input_global_scale_fused"] = shared

    payload = {"model": str(args.model), "text": str(args.text),
               "tokens": int(chunks.numel()), "seqlen": args.seqlen,
               "policy": "legacy compressed-tensors FP4_MAX/amax (PrismaQuant default)",
               "units": stats}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"-> {args.out}  ({len(stats)} Linears)")

    if args.scales_out:
        tensors = {f"{name}.input_global_scale":
                   torch.tensor([values.get("input_global_scale_fused", values["input_global_scale"])],
                                dtype=torch.float32)
                   for name, values in stats.items()}
        args.scales_out.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(args.scales_out), metadata={"format": "pt"})
        print(f"-> {args.scales_out}  ({len(tensors)} scales)")


if __name__ == "__main__":
    main()
