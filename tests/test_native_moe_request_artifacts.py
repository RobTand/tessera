"""The request loader's shared-source split and verification lifetime, on CPU.

No model and no device: what is under test is WHERE each tensor comes from,
which device it is on, and that the proof tensors are dropped once every member
is qualified.  The containers are real safetensors files written here, so the
reader is the format's own and the device policy is the one the harness applies.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

torch = pytest.importorskip("torch")
from experiments import bench_native_moe_operator as moe


def _save(path: Path, tensors: dict) -> Path:
    from safetensors.torch import save_file

    save_file(tensors, str(path))
    return path


def _members(*units):
    return [{"unit": unit, "expert": 0, "role": "w1"} for unit in units]


def _routing():
    return {"source_protocol": {"selection_bias": {"width": 8}}}


def test_the_shared_source_is_read_on_the_cpu_one_unit_at_a_time(tmp_path):
    """The source container is a host artifact, and its roster is the owner's."""
    path = _save(tmp_path / "shared.safetensors",
                 {f"source_weight/{unit}": torch.zeros(2, 4, dtype=torch.bfloat16)
                  for unit in ("u.0.w1", "u.0.w2")})
    with moe.open_shared_source(path) as source:
        assert source.keys == frozenset({"source_weight/u.0.w1", "source_weight/u.0.w2"})
        read = source.read("u.0.w1")
        assert read.device.type == "cpu" and read.dtype == torch.bfloat16
        assert moe.validate_shared_source(source, _members("u.0.w1", "u.0.w2")) is source
        with pytest.raises(ValueError, match="shared source roster"):
            moe.validate_shared_source(source, _members("u.0.w1", "u.0.w3"))


def test_the_request_roster_moves_the_sources_out_of_the_tensor_file():
    """A split request carries renders/phases/bias; the shared file the sources."""
    members = _members("u.0.w1", "u.0.w2")
    monolithic = moe.request_tensor_roster(_routing(), members)
    split = moe.request_tensor_roster(_routing(), members, shared_source=True)
    assert {key for key in monolithic if key.startswith("source_weight/")} == {
        "source_weight/u.0.w1", "source_weight/u.0.w2"}
    assert not any(key.startswith("source_weight/") for key in split)
    assert {key for key in split if key.startswith("rendered_weight/")} == {
        "rendered_weight/u.0.w1", "rendered_weight/u.0.w2"}
    assert "routing_bias" in split and "prefill.input" in split
    assert moe.shared_source_roster(members) == {
        "source_weight/u.0.w1", "source_weight/u.0.w2"}
    # The legacy LFM request keeps both kinds in its one file, key for key.
    assert monolithic == split | moe.shared_source_roster(members)


def test_the_verification_tensors_are_released_and_the_phases_survive():
    """After qualification the proof population must not be measured with.

    The owner holds the loaded wire containers and the layer holds the bias; the
    source and render tensors are read to prove the wires and are never read
    again.  What a benchmark keeps is the phases, the bias and the owner.
    """
    tensors = {
        "source_weight/u.0.w1": torch.zeros(2, 4, dtype=torch.bfloat16),
        "source_weight/u.0.w2": torch.zeros(2, 4, dtype=torch.bfloat16),
        "rendered_weight/u.0.w1": torch.zeros(2, 4, dtype=torch.bfloat16),
        "rendered_weight/u.0.w2": torch.zeros(2, 4, dtype=torch.bfloat16),
        "routing_bias": torch.zeros(8, dtype=torch.float32),
        "prefill.input": torch.zeros(8, 4, dtype=torch.bfloat16),
        "decode.reference_output": torch.zeros(1, 4, dtype=torch.bfloat16),
    }
    evidence = moe.release_verification_tensors(tensors, _members("u.0.w1", "u.0.w2"))
    assert evidence["dropped_tensors"] == 4
    assert evidence["dropped_bytes"] == 4 * (2 * 4 * 2)
    assert evidence["retained"] == ["decode.reference_output", "prefill.input", "routing_bias"]
    assert set(tensors) == {"routing_bias", "prefill.input", "decode.reference_output"}
    assert "released" in evidence["scope"]


def test_a_source_tensor_on_neither_side_of_the_split_is_refused():
    """The device policy is explicit: host for a shared source, CUDA for a render."""
    source = torch.zeros(2, 4, dtype=torch.bfloat16)
    assert moe._require_source_tensor(source) is None
    with pytest.raises(ValueError, match="2-D BF16"):
        moe._require_source_tensor(torch.zeros(2, 4, dtype=torch.float32))
    with pytest.raises(ValueError, match="2-D BF16"):
        moe._require_source_tensor(torch.zeros(4, dtype=torch.bfloat16))
