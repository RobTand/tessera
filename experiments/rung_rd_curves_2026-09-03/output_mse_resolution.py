"""Candidate 4: is the L1 currency's rung ranking a resolution bug?

The allocation audited in Tessera #1 was priced by PrismaQuant's
``tessera_campaign`` in the currency ``output_mse_under_route_activation_contract``
at ``nsamples: 4, seqlen: 512, max_act_rows: 256``.  Issue #4's cheapest
candidate is that those 256 activation rows are simply too few to resolve
third-digit differences at the top of the curve, and that a 10x re-price flips
the ranking toward the served answer.  That is the best possible outcome, so it
is measured first.

WHAT IS REPRODUCED.  The campaign's scorer, exactly
(``production_weight_cache._local_forward_render_score``):

    mse = mean over (row, out) of ( x @ W_ref^T  -  QDQ_a(x) @ W_rendered^T )^2

with ``QDQ_a`` the route's own per-token dynamic E4M3 quantiser
(``format_registry._make_plain_fp8_activation_vllm_rtn``, i.e. absmax/448 with
vLLM's ``1/(448*512)`` floor).  Note both legs: the reference is the *exact*
activation against the *exact* weight, so the A side's own error is inside the
score.  ``W_rendered`` here is the decoded FP8 tile times its per-row scale --
the object the W8A8 route multiplies, and byte-identical to the blobs the
campaign priced (``encoder_identity.py``).

The calibration is the campaign's: wikitext-2-raw-v1 **train**, windows of
``seqlen`` drawn by ``torch.Generator().manual_seed(seed)``, and the row cap
applied the way ``_collect_activations`` applies it -- rows are taken in order
from the first sequence forward, so ``max_act_rows=256`` reads 256 rows of ONE
sequence however many ``nsamples`` names.  The KL corpus is wikitext **test**;
the two are disjoint.

SETTINGS MEASURED
  ``s0..s3``   the shipped setting (256 rows) at four calibration seeds -- the
               seed-variance pre-screen: if the currency's own scatter is
               smaller than the effect, resolution is excluded before the 10x
               sweep is paid for.
  ``rows2560`` 10x the rows (five sequences).
  ``rows20480`` 40 sequences, every row.
  ``a_only``   the A side alone: the exact weight under the same activation
               contract.  Not a resolution setting -- the floor every rung's
               score is measured against.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import torch

MODEL = Path("/home/rob/models/Qwen3-0.6B")
PROBE = Path("/mnt/shared/tessera-runs/pq-continuous/qwen06b/probe.pkl")
ROLES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
         "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
FP8_MAX = 448.0
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)


def act_qdq(x: torch.Tensor) -> torch.Tensor:
    """The route's per-token dynamic E4M3 QDQ, in fp32."""
    rows = x.reshape(-1, x.shape[-1]).float()
    scale = (rows.abs().amax(dim=-1, keepdim=True) / FP8_MAX).clamp_min(FP8_MIN_SCALE)
    q = (rows / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (q.float() * scale).reshape(x.shape)


def calibration_tokens(n: int, seqlen: int, seed: int):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(data["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    gen = torch.Generator().manual_seed(int(seed))
    out = []
    for _ in range(int(n)):
        start = int(torch.randint(0, max(1, ids.shape[0] - seqlen - 1), (1,),
                                  generator=gen).item())
        out.append(ids[start:start + seqlen].unsqueeze(0))
    return out


def collect(model, batches, max_rows: int, device):
    store = {r: [] for r in ROLES}
    kept = {r: 0 for r in ROLES}
    handles = []

    def hook_for(role):
        def hook(_m, args):
            if not args or not isinstance(args[0], torch.Tensor):
                return
            flat = args[0].detach().reshape(-1, args[0].shape[-1])
            room = max_rows - kept[role]
            if room <= 0:
                return
            take = flat[:room].to(dtype=torch.float32)
            store[role].append(take)
            kept[role] += int(take.shape[0])
        return hook

    modules = dict(model.named_modules())
    for role in ROLES:
        handles.append(modules[f"model.layers.0.{role}"].register_forward_pre_hook(hook_for(role)))
    try:
        with torch.no_grad():
            for b in batches:
                model(b.to(device))
    finally:
        for h in handles:
            h.remove()
    return {r: torch.cat(v, 0) for r, v in store.items()}


def score(ref_w: torch.Tensor, rendered_w: torch.Tensor, x: torch.Tensor,
          row_chunk: int = 512) -> float:
    """``_local_forward_render_score``, same reference and same reduction."""
    ref_t = ref_w.float().t()
    ren_t = rendered_w.float().t()
    total = torch.zeros((), dtype=torch.float64, device=x.device)
    with torch.no_grad():
        for s in range(0, x.shape[0], row_chunk):
            xr = x[s:s + row_chunk]
            xq = act_qdq(xr)
            total = total + (xr @ ref_t - xq @ ren_t).pow(2).sum().double()
    return float(total.item()) / (int(x.shape[0]) * int(ref_w.shape[0]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/home/rob/tmp/ts-rung-rd-out/tiles")
    ap.add_argument("--rungs", required=True)
    args = ap.parse_args()
    rungs = tuple(int(r) for r in args.rungs.split(","))
    device = torch.device("cuda")
    cache = Path(args.cache)

    from safetensors import safe_open
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()

    source, rendered = {}, {}
    with safe_open(str(MODEL / "model.safetensors"), framework="pt") as h:
        for role in ROLES:
            source[role] = h.get_tensor(f"model.layers.0.{role}.weight").to(device, torch.float32)
    for role in ROLES:
        for rung in rungs:
            blob = torch.load(cache / f"{role.replace('.', '__')}_R{rung}.pt", map_location=device)
            tile = blob["tile"].view(torch.float8_e4m3fn).to(device)
            sc = blob["scale"].to(device).float()
            rendered[(role, rung)] = tile.float() * sc[:, None]

    stats = pickle.load(PROBE.open("rb"))["stats"]
    h_trace = {r: float(stats[f"model.layers.0.{r}"]["h_trace"]) for r in ROLES}

    settings = [("seed0_rows256", 4, 256, 0), ("seed1_rows256", 4, 256, 1),
                ("seed2_rows256", 4, 256, 2), ("seed3_rows256", 4, 256, 3),
                ("seed0_rows2560", 40, 2560, 0), ("seed0_rows20480", 40, 20480, 0)]

    out = {"schema": "tessera.rung_output_mse_resolution/1", "rungs": list(rungs),
           "roles": list(ROLES), "h_trace": h_trace, "settings": {}}
    for label, nsamples, max_rows, seed in settings:
        t0 = time.time()
        acts = collect(model, calibration_tokens(nsamples, 512, seed), max_rows, device)
        rec = {"nsamples": nsamples, "max_act_rows": max_rows, "seed": seed,
               "rows_used": {r: int(acts[r].shape[0]) for r in ROLES},
               "output_mse": {}, "a_only_mse": {}}
        for role in ROLES:
            x = acts[role]
            rec["a_only_mse"][role] = score(source[role], source[role], x)
            for rung in rungs:
                rec["output_mse"][f"{role}|{rung}"] = score(source[role], rendered[(role, rung)], x)
        out["settings"][label] = rec
        print(f"{label:18s} rows={rec['rows_used']['mlp.down_proj']:6d} "
              f"in {time.time() - t0:6.1f}s", flush=True)
        for role in ROLES:
            print(f"   {role:22s} a_only {rec['a_only_mse'][role]:.6g}  "
                  + "  ".join(f"R{q}:{rec['output_mse'][f'{role}|{q}']:.4g}"
                              for q in (749, 1006, 1107, 1262) if q in rungs), flush=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
