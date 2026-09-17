#!/usr/bin/env python3
"""How the runtime's activation quantizers answer a TP2-sized perturbation (tessera#514).

At a world of two, vLLM's row-parallel Linears round each rank's partial sum to
BF16 before the all-reduce, so the activation the next quantizer sees differs
from the single-rank one by at most a BF16 ulp on some elements.  This probe
feeds the SAME runtime operators the Tessera routes call -- ``scaled_fp4_quant``
with the checkpoint's static global (``native_ops.native_fp4_quant``) and
``dynamic_per_token_scaled_fp8_quant`` (``native_ops.native_fp8_quant``) -- an
activation ``x`` and its TP2-rounded twin ``x + e``, and reports what each
quantizer does with ``e``:

* ``flip_fraction``: the share of codes that changed;
* ``err_var_tp1`` / ``err_var_tp2``: the quantization error variance against
  the fp32 truth on each arm.  Equal means the perturbation re-draws the error
  without adding to it (the re-roll reading of #514); larger on the TP2 arm
  means added error;
* ``reroll_fraction``: 1 - corr(err_tp1, err_tp2), how much of the error was
  re-drawn;
* ``amplification``: var(deq(Q(x+e)) - deq(Q(x))) / var(e), the power gain of
  the perturbation through the quantizer.  BF16 (no quantizer) is 1 by
  construction and is the control.

The perturbation is the exact TP2 arithmetic: ``x_tp1 = bf16(y1 + y2)`` and
``x_tp2 = bf16(bf16(y1) + bf16(y2))`` for a random split of the fp32 pre-sum
``y`` into two partials.  The activations are SYNTHETIC -- no hidden state of
the stub is captured here -- and are calibrated so the draw's amax matches the
checkpoint's static global (``global = 6 * 448 / amax``, vLLM's convention),
because a global far from the draw's amax would measure a saturation regime
the serve never ran.  The E2M1 dequantizer below is a decoder of the operator's
codes; it is self-checked by reconstruction error and by agreement of the codes
with the route binding's own call.

    python3 experiments/tp2_partial_sum_reroll_probe.py --out probe.json
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_CAPACITY = 6.0 * 448.0   # E2M1 max times E4M3 max: vLLM's global = capacity / amax

#: The stub's static activation globals, read off the checkpoint
#: (``glm53-4layer-a4-e2m1x2-q896-l2``): the two dense E2M1 units of layer 1.
STUB_GLOBALS = {
    "layers.1.mlp.shared_experts.gate_up_proj": (4096, 983.04),
    "layers.1.mlp.shared_experts.down_proj": (2048, 28.294737),
}


def tp2_rounded_pair(y: torch.Tensor, gen: torch.Generator):
    """(bf16(y1+y2), bf16(bf16(y1)+bf16(y2))) for a random split y = y1 + y2."""
    u = torch.rand(y.shape, generator=gen, device=y.device, dtype=torch.float32)
    y1 = y * u
    y2 = y - y1
    tp1 = (y1 + y2).to(torch.bfloat16)
    tp2 = (y1.to(torch.bfloat16).float() + y2.to(torch.bfloat16).float()).to(torch.bfloat16)
    return tp1, tp2


def draw_activation(rows: int, cols: int, amax: float, kind: str, gen: torch.Generator,
                    device) -> torch.Tensor:
    if kind == "gaussian":
        y = torch.randn(rows, cols, generator=gen, device=device, dtype=torch.float32)
    elif kind == "student_t4":
        z = torch.randn(rows, cols, generator=gen, device=device, dtype=torch.float32)
        chi = torch.randn(rows, cols, 4, generator=gen, device=device, dtype=torch.float32).pow(2).sum(-1)
        y = z / torch.sqrt(chi / 4.0)
    else:
        raise ValueError(kind)
    return y * (amax / y.abs().max())


def fp4_dequant(packed: torch.Tensor, scales_u8: torch.Tensor, global_scale: float, low_first: bool):
    """Decode the operator's E2M1 codes and per-16 E4M3 scales back to floats."""
    lo = (packed & 0x0F).to(torch.int64)
    hi = (packed >> 4).to(torch.int64)
    codes = torch.stack([lo, hi] if low_first else [hi, lo], dim=-1).reshape(packed.shape[0], -1)
    mag = E2M1_VALUES.to(packed.device)[codes & 0x7]
    sign = torch.where(codes & 0x8 > 0, -1.0, 1.0)
    vals = (mag * sign).reshape(packed.shape[0], -1, 16)
    sf = scales_u8.view(torch.float8_e4m3fn).float()[:, : vals.shape[1]]
    return (vals * sf.unsqueeze(-1) / global_scale).reshape(packed.shape[0], -1)


def probe_fp4(x_tp1: torch.Tensor, x_tp2: torch.Tensor, truth: torch.Tensor, global_scale: float):
    from tessera.serving import native_ops

    gs = torch.tensor([global_scale], dtype=torch.float32, device=x_tp1.device)
    out = {}
    packed_ref, _ = native_ops.native_fp4_quant(x_tp1.contiguous(), gs)   # the route's own call
    deq = {}
    codes = {}
    for arm, x in (("tp1", x_tp1), ("tp2", x_tp2)):
        packed, sf = torch.ops._C.scaled_fp4_quant(x.contiguous(), gs, False)   # row-major scales
        codes[arm] = packed
        if arm == "tp1":
            assert torch.equal(packed, packed_ref), "route binding and raw op disagree on the codes"
        # Pick the nibble order by reconstruction: the wrong order does not decode.
        cands = {lf: fp4_dequant(packed, sf, global_scale, lf) for lf in (True, False)}
        errs = {lf: float((cands[lf] - x.float()).pow(2).mean()) for lf in cands}
        lf = min(errs, key=errs.get)
        out.setdefault("nibble_order", "low_first" if lf else "high_first")
        out.setdefault("dequant_check_mse_chosen_vs_other", [errs[lf], errs[not lf]])
        deq[arm] = cands[lf]
    return out, codes, deq


def probe_fp8(x_tp1: torch.Tensor, x_tp2: torch.Tensor):
    from tessera.serving import native_ops

    codes, deq = {}, {}
    for arm, x in (("tp1", x_tp1), ("tp2", x_tp2)):
        q, scale = native_ops.native_fp8_quant(x.contiguous())
        codes[arm] = q.view(torch.uint8)
        deq[arm] = q.float() * scale
    return {}, codes, deq


def stats(codes, deq, x_tp1, x_tp2, truth):
    e = x_tp2.float() - x_tp1.float()
    err1 = deq["tp1"] - truth
    err2 = deq["tp2"] - truth
    d = deq["tp2"] - deq["tp1"]
    c1, c2 = err1.flatten(), err2.flatten()
    corr = float(((c1 - c1.mean()) * (c2 - c2.mean())).mean() / (c1.std() * c2.std()))
    return {
        "flip_fraction": float((codes["tp1"] != codes["tp2"]).float().mean()),
        "perturbed_element_fraction": float((e != 0).float().mean()),
        "perturbation_rms_over_x_rms": float(e.pow(2).mean().sqrt() / truth.pow(2).mean().sqrt()),
        "err_var_tp1": float(err1.var()),
        "err_var_tp2": float(err2.var()),
        "err_var_ratio_tp2_over_tp1": float(err2.var() / err1.var()) if float(err1.var()) > 0 else None,
        "reroll_fraction": 1.0 - corr,
        "amplification": float(d.var() / e.var()) if float(e.var()) > 0 else None,
        "delta_rms_over_err_rms": float(d.pow(2).mean().sqrt() / err1.pow(2).mean().sqrt())
        if float(err1.pow(2).mean()) > 0 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=514)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("the runtime operators need a CUDA device")
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(args.seed)
    import vllm  # noqa: F401  -- the operator library the routes bind to

    result = {
        "schema": "tessera.tp2_partial_sum_reroll_probe/1",
        "issue": "tessera#514",
        "activations": "SYNTHETIC draws calibrated to the checkpoint's static global (amax = 6*448/global); "
                       "no hidden state of the stub is captured here",
        "perturbation": "x_tp1 = bf16(y1+y2), x_tp2 = bf16(bf16(y1)+bf16(y2)), random split of the fp32 pre-sum",
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__, "vllm": getattr(vllm, "__version__", None),
        "python": platform.python_version(), "rows": args.rows, "seed": args.seed,
        "cells": [],
    }
    for unit, (cols, gscale) in STUB_GLOBALS.items():
        amax = FP4_CAPACITY / gscale
        for kind in ("gaussian", "student_t4"):
            truth = draw_activation(args.rows, cols, amax, kind, gen, device)
            x_tp1, x_tp2 = tp2_rounded_pair(truth, gen)
            cell = {"unit": unit, "K": cols, "global_scale": gscale, "amax": amax, "draw": kind,
                    "quantizers": {}}
            extra, codes, deq = probe_fp4(x_tp1, x_tp2, truth, gscale)
            cell["quantizers"]["e2m1_group16_ue4m3_static"] = {**extra, **stats(codes, deq, x_tp1, x_tp2, truth)}
            extra, codes, deq = probe_fp8(x_tp1, x_tp2)
            cell["quantizers"]["fp8_per_token_dynamic"] = {**extra, **stats(codes, deq, x_tp1, x_tp2, truth)}
            bf = {"tp1": x_tp1.view(torch.int16), "tp2": x_tp2.view(torch.int16)}
            cell["quantizers"]["bf16_unquantized"] = stats(bf, {"tp1": x_tp1.float(), "tp2": x_tp2.float()},
                                                          x_tp1, x_tp2, truth)
            result["cells"].append(cell)
            for name, q in cell["quantizers"].items():
                print(f"{unit} K={cols} {kind:10s} {name:28s} flips={q['flip_fraction']:.4f} "
                      f"errvar tp2/tp1={q['err_var_ratio_tp2_over_tp1']!s:>8.8} reroll={q['reroll_fraction']:.4f} "
                      f"amplification={q['amplification']!s:>10.10} pert={q['perturbed_element_fraction']:.3f}")
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print("->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
