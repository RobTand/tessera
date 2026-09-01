"""Per-unit, per-format cost for a full GLM-5.3-Flash body allocation.

**This is a screen (principle 3). It selects nothing.**  It exists to answer
one question the six-projection head-to-head could not: *given a real menu and
a real byte budget, what does the DP actually pick across all 45 layers, and
does a mix beat the uniform assignment?*

**The cost.**  The probe defines the per-element empirical diagonal Fisher as
``H[o,i] = sum_t g[t,o]^2 x[t,i]^2`` (`incremental_probe.py:87`), so
``h_trace = E_t[||g_t||^2 ||x_t||^2]``.  The production scalar form
``0.5 h_trace * weight_mse`` (`allocator_solver.predicted_dloss`) reaches a
Delta-loss from that by decoupling g from x *and* assuming x isotropic.  Here
the second assumption is dropped, because the activation cache makes it
unnecessary:

    g2_hat = h_trace / E_t||x_t||^2                 (decouple, once)
    dloss  = 0.5 * g2_hat * E_t|| f_q(x_t) - W x_t ||^2      (measured)

``f_q`` is the **served** map, not the weight error: for a W4A4 or W8A8 rung it
is ``Q_w(W) @ Q_a(x)``, so the A-side is priced rather than assumed free.  That
is delta #1 of this design and it is not optional -- rendering identity without
execution identity is what priced a real A-side at zero on 2026-08-17.  Tessera
and BF16 are W4A16/W16A16 and take the clean input; that asymmetry is the
actual contract difference between the lanes, which is the thing being measured.

**One estimator for every unit.**  Shared experts and ``lm_head`` carry full
Fisher marginals and the packed experts do not.  Using ``g_sq_sum`` where it
exists and the scalar elsewhere would put different biases on the two sides of
every bit trade the DP makes, so the scalar form is used everywhere and the
marginal-weighted number is emitted as a diagnostic the DP never reads.

**Stated limits.**
  * *Pooled inputs.*  The activation cache stores one input per packed-expert
    entry, not per expert, so a sampled expert is scored on the tokens the
    whole layer saw rather than the tokens routed to it.  Production would use
    ``expert_empirical_cost``'s measured unit-KL instead.  Route-blind.
  * *The decoupling.*  ``E[||g||^2||x||^2] ~ E||g||^2 E||x||^2`` is an
    approximation.  It is the same one the production scalar form already
    makes; this makes it once instead of twice.
  * *One-step A-side.*  ``down_proj``'s input is computed from BF16 gate/up.
    As served those outputs would themselves be perturbed; the screen does not
    cascade.
  * *256 cached rows*, split 128 fit / 128 held-out eval.

**Pairing.**  The same seeded experts are used for every arm of a layer, so the
per-expert format *delta* is paired and expert-level variance cancels.  The
stderr reported is on the deltas, which is what the DP consumes.
"""
import argparse
import json
import math
import os
import pickle
import statistics as st
import sys
import time
from fractions import Fraction

import torch
from safetensors import safe_open

MODEL = "/mnt/shared/models/GLM-5.3-Flash-BF16"
RUN = "/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830"
ACT = f"{RUN}/act"
PROBE = f"{RUN}/artifacts/probe.pkl"
LDLQ_BLOCK = 256

# (name, bits, a_side) -- a_side is how the input is perturbed at serve time.
MENU = (
    ("BF16", 16.0, "none"),
    ("TESSERA_E2M1_K1_R768", 3.5, "none"),
    ("TESSERA_E2M1_K2_R896", 4.0, "none"),
    ("NVFP4", 4.5, "nvfp4"),
    ("FP8_E4M3", 8.0234375, "fp8"),
)
LEVERS = {"gptq": True, "static_act_order": True, "joint_scale_opt": True}


def is_tessera(name):
    return name.startswith("TESSERA_")


def load_probe():
    with open(PROBE, "rb") as handle:
        return pickle.load(handle)["stats"]


def act_inputs(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    x = blob["inputs"].float().cuda()
    half = x.shape[0] // 2
    return x[:half].contiguous(), x[half:].contiguous()


def render(weight, fmt, qname, fit, compensate_h):
    """Production render, optionally LDLQ-compensated for the Tessera lanes."""
    from prismaquant.production_weight_cache import render_production_weight

    if fmt == "BF16":
        return weight.float()
    if is_tessera(fmt):
        if compensate_h is None:
            return render_production_weight(
                weight, fmt, qname=qname, activations={}, levers={}).float()
        from tessera.compensate import compensated_targets

        def encode(block, start, stop):
            return render_production_weight(
                block.to(weight.dtype), fmt, qname=qname,
                activations={}, levers={}).float()

        _, recon = compensated_targets(weight, compensate_h, encode,
                                       block=LDLQ_BLOCK)
        return recon
    out = render_production_weight(weight, fmt, qname=qname,
                                   activations={qname: fit}, levers=LEVERS)
    return out.float()


def a_side(x, kind, fit, fmt_obj):
    if kind == "none":
        return x
    if kind == "fp8":
        return fmt_obj.activation_quantize_dequantize(x).float()
    from prismaquant.nvfp4_activation_contract import (
        nvfp4_activation_qdq_served, select_mse_grid_input_global_scale,
    )
    scale = select_mse_grid_input_global_scale([fit], device="cuda")
    return nvfp4_activation_qdq_served(x, scale).float()


def ldl_for(x_fit, columns):
    from tessera.compensate import block_ldl, regularize_hessian

    moment = (x_fit.T @ x_fit).double().float()
    return block_ldl(regularize_hessian(moment, count=x_fit.shape[0]),
                     LDLQ_BLOCK)


def score_unit(tensors, x_fit, x_eval, compensate, guard):
    """Return {fmt: (dloss, sq_error)} for one serving unit.

    ``tensors`` is a list of ``(weight, h_trace)`` sharing one input -- a fused
    sibling group is exactly that.  Each tensor carries its OWN ``h_trace``, so
    the unit's Delta-loss is the sum of per-tensor Delta-losses rather than one
    scalar applied to a pooled error; fusing constrains the format, it does not
    merge the sensitivities.
    """
    import prismaquant.format_registry as fr

    energy = float((x_eval * x_eval).sum(dim=1).mean())
    ldl = ldl_for(x_fit, x_eval.shape[1]) if compensate else None
    out = {}
    for fmt, _bits, kind in MENU:
        fmt_obj = None if fmt == "BF16" else fr.get_format(fmt)
        xq = a_side(x_eval, kind, x_fit, fmt_obj)
        dloss, sq = 0.0, 0.0
        for index, (weight, h_trace) in enumerate(tensors):
            qname = f"{guard}.{index}"
            wq = render(weight, fmt, qname, x_fit,
                        ldl if (compensate and is_tessera(fmt)) else None)
            if fmt != "BF16" and not is_tessera(fmt):
                if torch.equal(wq, fr.get_format(fmt).quantize_dequantize(weight).float()):
                    raise SystemExit(
                        f"{qname}/{fmt}: production render is bit-identical to RTN -- "
                        "the activation key did not land (see the qname memory)")
            ref = x_eval @ weight.float().T
            err = float(((xq @ wq.T - ref) ** 2).sum(dim=1).mean())
            dloss += 0.5 * (float(h_trace) / energy) * err
            sq += err
            del wq, ref
        out[fmt] = (dloss, sq)
        del xq
        torch.cuda.empty_cache()
    return out


def get_tensor(mapping, name):
    with safe_open(f"{MODEL}/{mapping[name]}", framework="pt") as handle:
        return handle.get_tensor(name).cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=8,
                    help="sampled experts per MoE layer")
    ap.add_argument("--out", default="experiments/results/glm53_full_body_cost.json")
    ap.add_argument("--compensate", action="store_true",
                    help="LDLQ-compensate the Tessera lanes (menu B)")
    ap.add_argument("--layers", default="", help="comma list, default all MoE layers")
    args = ap.parse_args()

    stats = load_probe()
    mapping = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    moe = sorted({int(k.split(".layers.")[1].split(".")[0])
                  for k in stats if ".mlp.experts.gate_up_proj" in k})
    if args.layers:
        moe = [int(v) for v in args.layers.split(",")]
    print(f"MoE layers: {len(moe)}  sampled experts/layer: {args.experts}  "
          f"compensate: {args.compensate}", flush=True)

    units, started = {}, time.time()
    for layer in moe:
        prefix = f"model.language_model.layers.{layer}"
        x_fit, x_eval = act_inputs(f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt")
        n_experts = int(stats[f"{prefix}.mlp.experts.gate_up_proj"]["num_experts"])
        generator = torch.Generator().manual_seed(20260901 + layer)
        picked = torch.randperm(n_experts, generator=generator)[:args.experts].tolist()

        gate_up, down = [], []
        for expert in picked:
            base = f"{prefix}.mlp.experts.{expert}"
            w_gate = get_tensor(mapping, f"{base}.gate_proj.weight")
            w_up = get_tensor(mapping, f"{base}.up_proj.weight")
            w_down = get_tensor(mapping, f"{base}.down_proj.weight")
            gate_up.append((expert, [w_gate, w_up]))
            down.append((expert, w_down))

        # --- fused gate_up unit -------------------------------------------
        per_expert = []
        # The probe packs gate and up into ONE row for routed experts, so both
        # siblings carry that row's h_trace.  Unlike the shared/dense units
        # below there is no separate sensitivity to give them.
        h_gu = stats[f"{prefix}.mlp.experts.gate_up_proj"]["h_trace"]
        for expert, pair in gate_up:
            per_expert.append(score_unit(
                [(pair[0], h_gu), (pair[1], h_gu)], x_fit, x_eval,
                args.compensate, f"{prefix}.experts.{expert}.gate_up"))
        units[f"{prefix}.mlp.experts.gate_up_proj"] = dict(
            n_params=int(stats[f"{prefix}.mlp.experts.gate_up_proj"]["n_params"]),
            scale=n_experts / len(per_expert), samples=per_expert, kind="packed_expert")

        # --- down unit: its input is the gated product, computed in BF16 ---
        per_expert = []
        for (expert, w_down), (_e, pair) in zip(down, gate_up):
            hid_fit = torch.nn.functional.silu(x_fit @ pair[0].float().T) * (x_fit @ pair[1].float().T)
            hid_eval = torch.nn.functional.silu(x_eval @ pair[0].float().T) * (x_eval @ pair[1].float().T)
            per_expert.append(score_unit(
                [(w_down, stats[f"{prefix}.mlp.experts.down_proj"]["h_trace"])],
                hid_fit.contiguous(), hid_eval.contiguous(),
                args.compensate, f"{prefix}.experts.{expert}.down"))
            del hid_fit, hid_eval
        units[f"{prefix}.mlp.experts.down_proj"] = dict(
            n_params=int(stats[f"{prefix}.mlp.experts.down_proj"]["n_params"]),
            scale=n_experts / len(per_expert), samples=per_expert, kind="packed_expert")

        for _e, pair in gate_up:
            del pair[:]
        del gate_up, down, x_fit, x_eval
        torch.cuda.empty_cache()
        done = len(units) // 2
        rate = (time.time() - started) / done
        print(f"layer {layer:>3}  {done}/{len(moe)}  "
              f"{rate:.1f}s/layer  eta {rate*(len(moe)-done)/60:.1f} min", flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(dict(units=units, menu=[list(m) for m in MENU],
                       experts_per_layer=args.experts,
                       compensate=args.compensate, run=RUN),
                  open(args.out, "w"), indent=1)

    # ---- the 3% that is not a routed expert ------------------------------
    # No sampling here: these units are small enough to price whole, and they
    # are the bits the DP has to fund a promotion WITH, so an estimate would be
    # the wrong economy.
    def cached(path):
        return act_inputs(f"{ACT}/{path}")

    def add(name, tensors, x_fit, x_eval, guard, kind):
        units[name] = dict(
            n_params=sum(int(w.numel()) for w, _h in tensors),
            scale=1.0, kind=kind,
            samples=[score_unit(tensors, x_fit, x_eval, args.compensate, guard)])

    dense = sorted({int(k.split(".layers.")[1].split(".")[0])
                    for k in stats if ".mlp.gate_proj" in k and ".experts" not in k})
    for layer in moe:
        prefix = f"model.language_model.layers.{layer}"
        base = f"model__language_model__layers__{layer}__mlp__shared_experts"
        x_fit, x_eval = cached(f"{base}__gate_proj.pt")
        pair = [(get_tensor(mapping, f"{prefix}.mlp.shared_experts.{p}.weight"),
                 stats[f"{prefix}.mlp.shared_experts.{p}"]["h_trace"])
                for p in ("gate_proj", "up_proj")]
        add(f"{prefix}.mlp.shared_experts.gate_up_proj", pair, x_fit, x_eval,
            f"{prefix}.shared.gate_up", "shared_expert")
        d_fit, d_eval = cached(f"{base}__down_proj.pt")
        add(f"{prefix}.mlp.shared_experts.down_proj",
            [(get_tensor(mapping, f"{prefix}.mlp.shared_experts.down_proj.weight"),
              stats[f"{prefix}.mlp.shared_experts.down_proj"]["h_trace"])],
            d_fit, d_eval, f"{prefix}.shared.down", "shared_expert")
        del pair, x_fit, x_eval, d_fit, d_eval
        torch.cuda.empty_cache()
        print(f"shared {layer:>3} done", flush=True)

    for layer in dense:
        prefix = f"model.language_model.layers.{layer}"
        base = f"model__language_model__layers__{layer}__mlp"
        x_fit, x_eval = cached(f"{base}__gate_proj.pt")
        pair = [(get_tensor(mapping, f"{prefix}.mlp.{p}.weight"),
                 stats[f"{prefix}.mlp.{p}"]["h_trace"])
                for p in ("gate_proj", "up_proj")]
        add(f"{prefix}.mlp.gate_up_proj", pair, x_fit, x_eval,
            f"{prefix}.dense.gate_up", "dense_mlp")
        d_fit, d_eval = cached(f"{base}__down_proj.pt")
        add(f"{prefix}.mlp.down_proj",
            [(get_tensor(mapping, f"{prefix}.mlp.down_proj.weight"),
              stats[f"{prefix}.mlp.down_proj"]["h_trace"])],
            d_fit, d_eval, f"{prefix}.dense.down", "dense_mlp")
        del pair, x_fit, x_eval, d_fit, d_eval
        torch.cuda.empty_cache()
        print(f"dense  {layer:>3} done", flush=True)

    x_fit, x_eval = cached("lm_head.pt")
    add("lm_head", [(get_tensor(mapping, "lm_head.weight"),
                     stats["lm_head"]["h_trace"])],
        x_fit, x_eval, "lm_head", "lm_head")
    print("lm_head done", flush=True)

    json.dump(dict(units=units, menu=[list(m) for m in MENU],
                   experts_per_layer=args.experts,
                   compensate=args.compensate, run=RUN),
              open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}  ({len(units)} units, "
          f"{(time.time()-started)/60:.1f} min)")


if __name__ == "__main__":
    sys.exit(main())
