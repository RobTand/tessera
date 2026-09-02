"""R1 must be exactly invisible before quantisation, or it is not a basis change.

The whole argument for a folded residual-stream rotation is that it changes
the *distribution* every Linear sees and nothing else: the model it produces
computes the same function.  A sign error in the transpose, or a gamma folded
into the wrong side, still produces a plausible-looking checkpoint that serves
and generates text -- and every quantised number measured on it is then
measuring a different model.  So the test is logit equality on a real forward
of a real (tiny, random) Qwen3, in float64, not a property of the matrices.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from rotate_checkpoint import hadamard, rotate_state_dict, rotation  # noqa: E402

transformers = pytest.importorskip("transformers")


def tiny_config():
    from transformers import Qwen3Config
    return Qwen3Config(
        vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=64, rms_norm_eps=1e-6, tie_word_embeddings=True,
    )


def build_tiny(seed: int = 7):
    from transformers import Qwen3ForCausalLM
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(tiny_config()).to(torch.float64).eval()
    # A freshly initialised model has gammas of exactly one, which would let a
    # broken fold pass.  Make every norm non-trivial, including the final one.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight") or "layernorm" in name:
                param.copy_(1.0 + 0.5 * torch.randn_like(param))
    return model


def test_hadamard_is_orthogonal():
    h = hadamard(64) / 8.0
    assert torch.allclose(h @ h.T, torch.eye(64, dtype=torch.float64), atol=1e-12)
    with pytest.raises(SystemExit):
        hadamard(96)


def test_rotation_is_orthogonal_and_seeded():
    a = rotation(64, 3)
    assert torch.allclose(a @ a.T, torch.eye(64, dtype=torch.float64), atol=1e-12)
    assert torch.equal(a, rotation(64, 3))
    assert not torch.equal(a, rotation(64, 4))
    # a bare Hadamard would leave the sign row constant; the randomisation is
    # the point, so at least one row must carry both signs of the first row
    assert not torch.equal(a[0].sign(), a[1].sign() * a[1].sign()[0] * a[0].sign()[0])


def test_rotated_model_computes_the_same_function():
    model = build_tiny()
    config = model.config
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # the source is tied: lm_head is not a separate parameter
    assert "lm_head.weight" not in state or torch.equal(
        state["lm_head.weight"], state["model.embed_tokens.weight"])

    rot = rotation(config.hidden_size, seed=11)
    rotated = rotate_state_dict(state, config.num_hidden_layers, rot)

    from transformers import Qwen3ForCausalLM
    untied = tiny_config()
    untied.tie_word_embeddings = False
    other = Qwen3ForCausalLM(untied).to(torch.float64).eval()
    missing, unexpected = other.load_state_dict(rotated, strict=False)
    assert not unexpected
    assert all("inv_freq" in k or "rotary" in k for k in missing), missing

    ids = torch.randint(0, config.vocab_size, (2, 24))
    with torch.no_grad():
        a = model(ids).logits
        b = other(ids).logits
    # The floor is float32, not float64: transformers' RMSNorm casts to
    # float32 for the variance regardless of the parameter dtype, so the two
    # paths round differently at ~1e-7 relative.  A folding error is 1e-3+
    # (the last test in this file holds that separation).
    scale = a.abs().max().item()
    assert (a - b).abs().max().item() < 1e-6 * max(scale, 1.0), (a - b).abs().max().item()

    # and the transform is not a no-op: the residual basis really moved
    assert not torch.allclose(rotated["model.embed_tokens.weight"],
                              state["model.embed_tokens.weight"].to(torch.float64))
    # every gamma is one afterwards
    for key, value in rotated.items():
        if key.endswith("layernorm.weight") or key == "model.norm.weight":
            assert torch.equal(value, torch.ones_like(value)), key
    # q_norm/k_norm act on head_dim and must be untouched
    for key in state:
        if "q_norm" in key or "k_norm" in key:
            assert torch.equal(rotated[key], state[key].to(torch.float64)), key


def test_wrong_transpose_is_caught_by_the_same_check():
    """The guard has teeth: transposing the write side breaks logit equality."""
    model = build_tiny()
    config = model.config
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    rot = rotation(config.hidden_size, seed=11)
    rotated = rotate_state_dict(state, config.num_hidden_layers, rot)
    broken = dict(rotated)
    key = "model.layers.0.self_attn.o_proj.weight"
    broken[key] = rot @ (rot.T.inverse() @ broken[key])  # i.e. R W instead of Rᵀ W

    from transformers import Qwen3ForCausalLM
    untied = tiny_config()
    untied.tie_word_embeddings = False
    other = Qwen3ForCausalLM(untied).to(torch.float64).eval()
    other.load_state_dict(broken, strict=False)
    ids = torch.randint(0, config.vocab_size, (2, 24))
    with torch.no_grad():
        a = model(ids).logits
        b = other(ids).logits
    assert (a - b).abs().max().item() > 1e-3
