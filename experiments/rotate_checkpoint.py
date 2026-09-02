#!/usr/bin/env python
"""Fold a QuaRot/SpinQuant R1 residual-stream rotation into a Llama-family checkpoint.

R1 is a *checkpoint* transform, not a runtime op: after it the model is an
ordinary ``Qwen3ForCausalLM`` that vanilla vLLM loads with no plugin, and its
logits are the source model's to round-off.  What changes is the basis the
residual stream lives in, and therefore the input distribution every Linear
that reads the residual sees.  A dense model's residual carries a handful of
massive channels; an orthogonal mixing spreads them, which is what an
activation quantiser with a per-tensor calibrated scale is exposed to.

THE ALGEBRA.  Torch stores a Linear as ``W`` of shape ``[out, in]`` and
computes ``y = x Wᵀ``.  With an orthogonal ``R`` on the residual:

  * every RMSNorm gamma is folded into the Linears that read that norm's
    output (``input_layernorm`` -> q/k/v, ``post_attention_layernorm`` ->
    gate/up, the final ``model.norm`` -> ``lm_head``) and set to ones.  This
    must happen first: ``RMSNorm(h) = h/rms(h) * g`` only commutes with ``R``
    once ``g`` is the identity, because ``rms`` is rotation-invariant and the
    elementwise product is not.
  * ``embed_tokens``: ``E -> E R`` (its rows are residual vectors).
  * reads the residual (q/k/v/gate/up, and ``lm_head``): ``W -> W R``, so
    ``y = (x R) (W R)ᵀ = x R Rᵀ Wᵀ = x Wᵀ`` is unchanged.
  * writes the residual (``o_proj``, ``down_proj``): ``W -> Rᵀ W``, so
    ``y = x (Rᵀ W)ᵀ = x Wᵀ R`` lands in the rotated basis.

Qwen3's per-head ``q_norm``/``k_norm`` act on ``head_dim`` after ``q_proj``
and are untouched.  ``o_proj``'s per-head input basis (R2) and
``down_proj``'s input basis (R4) are runtime ops and are NOT applied here.

TIED EMBEDDINGS.  The final norm's gamma folds into ``lm_head`` only, so
after the fold ``lm_head != embed_tokens`` and the checkpoint must be untied
(``tie_word_embeddings: false``).  Qwen3-0.6B already stores both tensors on
disk, so untying costs no bytes there; the script reports the delta either
way.

Everything is computed in float64 and stored in the source dtype.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# gamma -> the Linears that read that norm's output
NORM_FOLD = (
    ("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
    ("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj")),
)
READS_RESIDUAL = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                  "mlp.gate_proj", "mlp.up_proj")
WRITES_RESIDUAL = ("self_attn.o_proj", "mlp.down_proj")


def hadamard(n: int) -> torch.Tensor:
    """The Sylvester Hadamard of order ``n`` (a power of two), unnormalised."""
    if n & (n - 1):
        raise SystemExit(f"hidden size {n} is not a power of two; a Sylvester "
                         "Hadamard does not exist for it (use a Kronecker "
                         "factorisation or a random orthogonal instead)")
    h = torch.ones((1, 1), dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def rotation(n: int, seed: int) -> torch.Tensor:
    """A randomised Hadamard ``R = D H/sqrt(n)``, orthogonal, in float64.

    The sign diagonal is what makes it *randomised*: a bare Hadamard maps the
    all-ones direction onto a single coordinate, which would concentrate
    rather than spread whatever energy sits there.
    """
    generator = torch.Generator().manual_seed(seed)
    signs = (torch.randint(0, 2, (n,), generator=generator, dtype=torch.int64) * 2 - 1).to(torch.float64)
    return signs[:, None] * hadamard(n) / (n ** 0.5)


def load_state_dict(src: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Every tensor in the checkpoint, plus the tensor -> shard map."""
    index = src / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = [p.name for p in sorted(src.glob("*.safetensors"))]
        weight_map = {}
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(str(src / shard), framework="pt") as handle:
            for key in handle.keys():
                state[key] = handle.get_tensor(key)
                weight_map.setdefault(key, shard)
    return state, weight_map


def rotate_state_dict(state: dict[str, torch.Tensor], n_layers: int,
                      rot: torch.Tensor) -> dict[str, torch.Tensor]:
    """Fold the norms, then rotate.  Returns a new float64 state dict.

    The input may be tied (``lm_head.weight is embed_tokens.weight``); the
    output is always untied.
    """
    out = {k: v.to(torch.float64) for k, v in state.items()}
    embed = out["model.embed_tokens.weight"]

    # --- fold every RMSNorm gamma into the Linears that read it ---
    for layer in range(n_layers):
        prefix = f"model.layers.{layer}."
        for norm, targets in NORM_FOLD:
            gamma = out[prefix + norm + ".weight"]
            for target in targets:
                key = prefix + target + ".weight"
                out[key] = out[key] * gamma[None, :]
            out[prefix + norm + ".weight"] = torch.ones_like(gamma)
    final = out["model.norm.weight"]
    # the final gamma has exactly one consumer, and it is why the pair unties
    out["lm_head.weight"] = embed * final[None, :]
    out["model.norm.weight"] = torch.ones_like(final)

    # --- rotate ---
    out["model.embed_tokens.weight"] = embed @ rot
    out["lm_head.weight"] = out["lm_head.weight"] @ rot
    for layer in range(n_layers):
        prefix = f"model.layers.{layer}."
        for target in READS_RESIDUAL:
            key = prefix + target + ".weight"
            out[key] = out[key] @ rot
        for target in WRITES_RESIDUAL:
            key = prefix + target + ".weight"
            out[key] = rot.T @ out[key]
    return out


def token_ids(model_dir: Path, text: str, tokens: int, device: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    return tok(text, return_tensors="pt").input_ids[:, :tokens].to(device)


def _run(model, ids):
    with torch.no_grad():
        logits = model(ids).logits.float().cpu()
    del model
    torch.cuda.empty_cache()
    return logits


def hf_logits(model_dir: Path, dtype, ids, device: str):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype).to(device).eval()
    return _run(model, ids)


def hf_logits_from_state(src: Path, state: dict[str, torch.Tensor], dtype, ids, device: str):
    """Forward an in-memory (untied) state dict, so storage rounding is out of the way."""
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(str(src))
    config.tie_word_embeddings = False
    model = AutoModelForCausalLM.from_config(config).to(dtype)
    missing, unexpected = model.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=False)
    unexpected = [k for k in unexpected]
    missing = [k for k in missing if "inv_freq" not in k and "rotary" not in k]
    if missing or unexpected:
        raise SystemExit(f"state dict does not match the model: missing={missing[:5]} unexpected={unexpected[:5]}")
    return _run(model.to(device).eval(), ids)


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    """max |Δlogit| and the mean KL of b's softmax from a's, over all positions."""
    la = torch.log_softmax(a.double(), -1)
    lb = torch.log_softmax(b.double(), -1)
    kl = (la.exp() * (la - lb)).sum(-1).mean().item()
    return {"max_abs_logit_diff": (a - b).abs().max().item(),
            "logit_rms": a.pow(2).mean().sqrt().item(),
            "mean_kl_nats": kl}


def sample_text(path: Path | None, chars: int = 8000) -> str:
    if path is not None:
        return path.read_text()[:chars]
    return (" ".join(["The Battle of Hastings was fought on 14 October 1066 between the "
                      "Norman-French army of William, the Duke of Normandy, and an English "
                      "army under the Anglo-Saxon King Harold Godwinson."] * 40))[:chars]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true",
                    help="run both models on --tokens tokens and report max |Δlogit| and KL, "
                         "in float32 (the algebra) and in the stored dtype (what serves)")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--verify-text", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    config = json.loads((args.src / "config.json").read_text())
    hidden = int(config["hidden_size"])
    n_layers = int(config["num_hidden_layers"])
    state, _ = load_state_dict(args.src)
    dtype = state["model.embed_tokens.weight"].dtype

    rot = rotation(hidden, args.seed)
    identity = (rot @ rot.T - torch.eye(hidden, dtype=torch.float64)).abs().max().item()
    print(f"R: {hidden}x{hidden} randomised Hadamard, seed {args.seed}, "
          f"max |R Rᵀ - I| = {identity:.3e}")
    assert identity < 1e-10, "R is not orthogonal"

    rotated = rotate_state_dict(state, n_layers, rot)
    args.dst.mkdir(parents=True, exist_ok=True)
    payload = {k: v.to(dtype).contiguous() for k, v in rotated.items()}
    save_file(payload, str(args.dst / "model.safetensors"), metadata={"format": "pt"})

    config = dict(config)
    config["tie_word_embeddings"] = False
    (args.dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json",
                 "vocab.json", "merges.txt", "special_tokens_map.json"):
        if (args.src / name).exists():
            shutil.copy2(args.src / name, args.dst / name)

    src_bytes = sum(p.stat().st_size for p in args.src.glob("*.safetensors"))
    dst_bytes = sum(p.stat().st_size for p in args.dst.glob("*.safetensors"))
    head_bytes = payload["lm_head.weight"].numel() * payload["lm_head.weight"].element_size()
    sidecar = {
        "transform": "QuaRot/SpinQuant R1 (residual-stream randomised Hadamard), folded",
        "source": str(args.src), "seed": args.seed, "hidden": hidden,
        "rotation_sha256": hashlib.sha256(rot.numpy().tobytes()).hexdigest(),
        "orthogonality_residual": identity,
        "tie_word_embeddings": False,
        "lm_head_bytes": head_bytes,
        "source_stored_lm_head": "lm_head.weight" in state,
        "safetensors_bytes_src": src_bytes, "safetensors_bytes_dst": dst_bytes,
        "not_applied": ["R2 (per-head o_proj input)", "R4 (down_proj input)"],
    }
    (args.dst / "rotation.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(json.dumps(sidecar, indent=2))

    if args.verify:
        text = sample_text(args.verify_text)
        ids = token_ids(args.src, text, args.tokens, args.device)
        stored = str(dtype).split(".")[-1]
        report = {}
        # (a) the algebra: same source bytes, rotation in float64, forward in
        #     float32.  Nothing here has been through the stored dtype twice,
        #     so a residual above float32 round-off is a folding bug.
        base32 = hf_logits(args.src, torch.float32, ids, args.device)
        report["algebra_float32"] = compare(
            base32, hf_logits_from_state(args.src, rotated, torch.float32, ids, args.device))
        # (b) what serves: both checkpoints as stored, forward in the stored dtype.
        report[f"stored_{stored}"] = compare(
            hf_logits(args.src, dtype, ids, args.device),
            hf_logits(args.dst, dtype, ids, args.device))
        # (c) the storage cost alone: rotated weights rounded to the stored
        #     dtype, forward in float32, against (a)'s baseline.
        report["stored_weights_float32_compute"] = compare(
            base32, hf_logits(args.dst, torch.float32, ids, args.device))
        for name, values in report.items():
            print(f"verify [{name}] {json.dumps(values)}", flush=True)
        (args.dst / "verify.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
