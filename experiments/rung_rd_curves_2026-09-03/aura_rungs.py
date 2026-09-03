"""Candidate 1: does AURA's KL-adjoint rank Tessera rungs?

AURA is PrismaQuant's default cost and it replaces the output-MSE objective
with a KL-adjoint one, so it is the obvious first candidate.  It has never been
run on a rung axis, and issue #4 is explicit that it must not be assumed to
inherit its format-menu win: what a rung sweep needs is the adjoint's
*gradient* along rate, not its level on a menu of discretely different formats.

THE FORM IS PRISMAQUANT'S, NOT A RECONSTRUCTION.  ``prismaquant/aura_cost.py``
defines

    predicted_dloss[i, f] = 0.5 * mean_k ( <gW_i^(k), dW_{i,f}> )^2
    gW_i^(k) = d/dW_i fisher_probe_scalar(logits; seed = seed_base + k)

and this script imports ``fisher_probe_scalar`` from that tree rather than
re-deriving it, runs at its shipped defaults (``n_probes=16``,
``token_scope="all"``, ``temperature=1.0``, ``seed_base=7000``, fp32-resident
model (``aura_cost.py --dtype`` defaults to ``float32``),
eager attention, ``dw_dtype=bfloat16``) and takes its calibration from the same
loader at the same defaults (``load_wikitext_calibration_windowed``, wikitext-2
train, ``n=4``, ``seqlen=256``, ``seed=42``).

``dW`` is the production-rendered error the allocator would have seen: the
decoded FP8 tile times its per-row scale, minus the source weight -- the same
rendering the served checkpoint carries, byte-checked by ``encoder_identity.py``.

Also harvested, because it comes free from the same gradients and is the other
half of the L1 factorisation: ``g_trace`` = the KL-Fisher weight-gradient
energy per unit, AURA's own analogue of the CE-empirical ``h_trace`` the L1
currency multiplies by.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/home/rob/pq-wt/tessera-continuous")

MODEL = Path("/home/rob/models/Qwen3-0.6B")
ROLES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
         "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/home/rob/tmp/ts-rung-rd-out/tiles")
    ap.add_argument("--rungs", required=True)
    ap.add_argument("--n-probes", type=int, default=16)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--n-calib-samples", type=int, default=4)
    ap.add_argument("--calib-seqlen", type=int, default=256)
    ap.add_argument("--calib-seed", type=int, default=42)
    ap.add_argument("--token-scope", default="all")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"),
                    help="Resident model dtype for the probe.  aura_cost.py's "
                         "own --dtype default is float32 ('the historical "
                         "default'), so that is the default here too.")
    args = ap.parse_args()
    rungs = tuple(int(r) for r in args.rungs.split(","))
    device = torch.device("cuda")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prismaquant.calibration_data import load_wikitext_calibration_windowed
    from prismaquant.kl_fisher import fisher_probe_scalar

    tok = AutoTokenizer.from_pretrained(str(MODEL))
    calib = load_wikitext_calibration_windowed(
        tok, args.n_calib_samples, args.calib_seqlen,
        split="train", seed=args.calib_seed).to(device)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL),
        dtype=torch.float32 if args.dtype == "float32" else torch.bfloat16,
        attn_implementation="eager").to(device).eval()

    mods = dict(model.named_modules())
    weights = {r: mods[f"model.layers.0.{r}"].weight for r in ROLES}
    for w in weights.values():
        w.requires_grad_(True)

    cache = Path(args.cache)
    dW: dict[tuple[str, int], torch.Tensor] = {}
    for role in ROLES:
        src = weights[role].detach().float()
        for rung in rungs:
            blob = torch.load(cache / f"{role.replace('.', '__')}_R{rung}.pt", map_location=device)
            tile = blob["tile"].view(torch.float8_e4m3fn).to(device).float()
            sc = blob["scale"].to(device).float()
            dW[(role, rung)] = (tile * sc[:, None] - src).to(torch.bfloat16)

    s2 = {k: 0.0 for k in dW}
    s4 = {k: 0.0 for k in dW}
    g_trace = {r: 0.0 for r in ROLES}
    t0 = time.time()
    for k in range(args.n_probes):
        logits = model(calib).logits
        scalar = fisher_probe_scalar(logits, seed=args.seed_base + k,
                                     token_scope=args.token_scope, temperature=1.0)
        grads = torch.autograd.grad(scalar, [weights[r] for r in ROLES])
        for role, g in zip(ROLES, grads):
            gf = g.float()
            g_trace[role] += float((gf * gf).sum().item())
            for rung in rungs:
                x2 = float((gf * dW[(role, rung)].float()).sum().item()) ** 2
                s2[(role, rung)] += x2
                s4[(role, rung)] += x2 * x2
        del logits, scalar, grads
        print(f"probe {k+1}/{args.n_probes} {time.time() - t0:.1f}s", flush=True)

    inv = 1.0 / args.n_probes
    out = {"schema": "tessera.rung_aura_cost/1", "rungs": list(rungs), "roles": list(ROLES),
           "n_probes": args.n_probes, "seed_base": args.seed_base,
           "measurement_dtype": args.dtype,
           "token_scope": args.token_scope,
           "calib": {"loader": "prismaquant.calibration_data.load_wikitext_calibration_windowed",
                     "split": "train", "n": args.n_calib_samples,
                     "seqlen": args.calib_seqlen, "seed": args.calib_seed},
           "g_trace": {r: g_trace[r] * inv for r in ROLES},
           "predicted_dloss": {}, "predicted_dloss_stderr": {}}
    for (role, rung), v in s2.items():
        mean_x2 = inv * v
        var = max((s4[(role, rung)] - args.n_probes * mean_x2 * mean_x2)
                  / max(1, args.n_probes - 1), 0.0)
        out["predicted_dloss"][f"{role}|{rung}"] = 0.5 * mean_x2
        out["predicted_dloss_stderr"][f"{role}|{rung}"] = 0.5 * (var * inv) ** 0.5
    Path(args.out).write_text(json.dumps(out, indent=2))
    for role in ROLES:
        print(f"{role:22s} g_trace {out['g_trace'][role]:.6g}  "
              + "  ".join(f"R{q}:{out['predicted_dloss'][f'{role}|{q}']:.4g}"
                          for q in (749, 1006, 1107, 1262) if q in rungs), flush=True)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
