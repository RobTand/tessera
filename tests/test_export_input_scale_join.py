"""The fused A-side static scale is the MIN member scale, or a refusal.

``input_global_scale`` is capacity over amax -- the route hands it unmodified
to vLLM's quantiser, which stores ``e4m3(block_amax / 6 * scale)`` clamped at
448, so a value too large for the tensor's true amax saturates the stored
block scale and the peak activations clip silently.  A fused module's one GEMM
quantises one input tensor for every member, so the module must carry the
scale of the largest calibrated amax: the MINIMUM member scale.  The exporter
used to take the MAX (the smallest calibrated range -- the clipping
direction), copying the reduction vLLM's stock scheme applies to checkpoints
its calibrators already unified (flagged by RobTand/prismaquant#196; the
contract helpers there join min-scale / max-amax).

Members of a vLLM-fused module read the same input tensor, so scales from one
calibration agree to within one bf16 ULP of the lattice the served A side is
cast to; a wider spread is two calibrations and is refused by
``tessera.fused.shared_input_global_scale`` rather than joined silently.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from tessera.errors import GrammarError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "export_tessera_serving", ROOT / "experiments" / "export_tessera_serving.py")
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)

BODY = "model.layers."
Q = BODY + "0.self_attn.q_proj.weight"
K = BODY + "0.self_attn.k_proj.weight"
V = BODY + "0.self_attn.v_proj.weight"
FUSED = BODY + "0.self_attn.qkv_proj"

#: One bf16 ULP at 1.0 -- the derived noise bound the join must accept.
BF16_ULP = 2.0 ** -7


def _tensor(rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=g).bfloat16()


def _run(tmp_path, monkeypatch, scales, *extra):
    src = tmp_path / "src"
    src.mkdir()
    save_file({name: _tensor(64, 32, i).contiguous()
               for i, name in enumerate((Q, K, V))},
              str(src / "model.safetensors"), metadata={"format": "pt"})
    (src / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 32, "intermediate_size": 32,
    }))
    donor = tmp_path / "input_scales.safetensors"
    save_file({name[: -len(".weight")] + ".input_global_scale":
               torch.tensor([value], dtype=torch.float32)
               for name, value in zip((Q, K, V), scales)},
              str(donor), metadata={"format": "pt"})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(
        {name: {"grid": "E2M1x2", "q256": 896} for name in (Q, K, V)}))
    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        ["export_tessera_serving.py", str(src), str(out),
         "--grid", "E2M1x2", "--q256", "896", "--device", "cpu", "--no-verify",
         "--plan-json", str(plan_path), "--input-scales", str(donor), *extra])
    exporter.main()
    return out


def _read_scale(path, key):
    with safe_open(str(path), framework="pt") as handle:
        return float(handle.get_tensor(key).float().reshape(-1)[0])


def test_the_fused_a_side_scale_is_the_min_member_scale(tmp_path, monkeypatch):
    """Members one calibration apart by rounding: the served scale is the MIN
    (the largest calibrated amax), not the max.  The pre-fix exporter wrote
    ``max(input_scales)`` -- the member with the SMALLEST range -- and every
    activation of the wider members above that range clipped silently."""
    out = _run(tmp_path, monkeypatch, (4.0, 4.0 * (1.0 + 2.0 ** -8), 4.0))
    written = _read_scale(out / "model.safetensors",
                          FUSED + ".trellis_input_global_scale")
    assert written == 4.0


def test_the_twin_members_all_carry_the_joined_scale(tmp_path, monkeypatch):
    """The stock twin exists to execute the same A side this export serves.
    vLLM reduces whatever the members carry into one scale per fused module
    (with only a warning when they differ), so per-member donor values in the
    twin serve vLLM's reduction, not this export's join.  Every member carries
    the joined value instead."""
    out = _run(tmp_path, monkeypatch, (4.0, 4.0 * (1.0 + 2.0 ** -8), 4.0),
               "--stock-twin", str(tmp_path / "twin"))
    del out
    for name in (Q, K, V):
        key = name[: -len(".weight")] + ".input_global_scale"
        assert _read_scale(tmp_path / "twin" / "model.safetensors", key) == 4.0


def test_member_scales_beyond_one_bf16_ulp_are_refused_by_name(tmp_path, monkeypatch):
    """A 2x spread is not calibration noise: it is two calibrations, and a
    joined value would serve a distribution nobody measured.  Refused where
    the bytes are decided, naming the members and the derived bound; the
    pre-fix exporter joined it silently (to the max -- the clipping side)."""
    with pytest.raises(GrammarError) as caught:
        _run(tmp_path, monkeypatch, (4.0, 2.0, 4.0))
    message = str(caught.value)
    assert "input_global_scale" in message
    assert BODY + "0.self_attn.k_proj" in message
    assert "bf16" in message
    assert not (tmp_path / "out" / "config.json").exists()


def test_a_spread_of_exactly_one_bf16_ulp_is_still_one_calibration(tmp_path, monkeypatch):
    """The bound is derived (one bf16 ULP -- the lattice the route casts the
    A tensor to), and the boundary itself passes: two faithful readings of one
    bf16 amax can land one lattice step apart."""
    out = _run(tmp_path, monkeypatch, (4.0, 4.0 * (1.0 + BF16_ULP), 4.0))
    written = _read_scale(out / "model.safetensors",
                          FUSED + ".trellis_input_global_scale")
    assert written == 4.0
