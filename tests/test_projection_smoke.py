"""Consumer-visible checks for the small projection artifacts and numeric screen."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import tessera_projection_smoke as smoke


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    return smoke.prepare(tmp_path_factory.mktemp("projection-smoke") / "artifact")


def _alter_manifest(path, mutator):
    value = json.loads(path.read_text())
    changed = copy.deepcopy(value)
    mutator(changed)
    return value, changed


def test_cpu_dry_run_reads_every_route_and_bias(artifact, tmp_path):
    out = tmp_path / "dry.json"
    assert smoke.main(["--artifact", str(artifact.parent), "--dry-run", "--mode", "graph", "--m", "1", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["device_forward"] is False
    assert report["status"] == "cpu-input-proof"
    assert len(report["modules"]) == len(smoke.CASES)
    assert all(row["input_shape"] == [1, row["columns"]] for row in report["modules"])
    assert all(set(row["wire_bytes"]) == {"T-8", "T-16"} for row in report["modules"])
    inputs = smoke.read_inputs(artifact, 3)
    for item in inputs:
        if item["row"]["prefix"].startswith("visual."):
            assert item["bias"].abs().sum() > 0


@pytest.mark.parametrize("defect", ["missing", "duplicate", "shape", "role_order", "digest"])
def test_cpu_dry_run_refuses_consumer_input_defects(artifact, defect):
    def damage(value):
        if defect == "missing":
            value["modules"].pop()
        elif defect == "duplicate":
            value["modules"][-1] = copy.deepcopy(value["modules"][0])
        elif defect == "shape":
            value["modules"][0]["columns"] += 1
        elif defect == "role_order":
            value["modules"][0]["roles"][-2:] = value["modules"][0]["roles"][-2:][::-1]
        else:
            value["modules"][0]["routes"]["T-8"]["sha256"] = "0" * 64
    original, changed = _alter_manifest(artifact, damage)
    smoke._json(artifact, changed)
    try:
        with pytest.raises(ValueError, match="exactly once|shape or role order|integrity"):
            smoke.read_inputs(artifact, 3)
    finally:
        smoke._json(artifact, original)


@pytest.mark.parametrize("defect", ["width", "dtype", "bias"])
def test_cpu_dry_run_reads_and_refuses_bad_tensor_bytes(artifact, defect):
    manifest = json.loads(artifact.read_text())
    row = next(r for r in manifest["modules"] if r["prefix"].startswith("visual."))
    file = artifact.parent / row["file"]
    original = load_file(str(file))
    damaged = {name: tensor.clone() for name, tensor in original.items()}
    if defect == "width":
        damaged["input"] = damaged["input"][:, :-1].contiguous()
    elif defect == "dtype":
        damaged["input"] = damaged["input"].float()
    else:
        damaged["bias"].zero_()
    save_file(damaged, str(file))
    try:
        with pytest.raises(ValueError, match="invalid input|nonzero"):
            smoke.read_inputs(artifact, 3)
    finally:
        save_file(original, str(file))


@pytest.mark.parametrize("family", ["T-8", "T-16"])
def test_wire_reader_preserves_numeric_rows_and_role_order(artifact, family):
    from tessera.alphabet import BF16_GRID, E4M3_GRID
    inputs = smoke.read_inputs(artifact, 3)
    index = 0
    item = inputs[index]
    row = item["row"]
    grid = E4M3_GRID if family == "T-8" else BF16_GRID
    expected = []
    for role_index, (_name, rows) in enumerate(row["roles"]):
        unit = smoke._unit(rows, row["columns"], grid, 211 + index * 17 + role_index)
        codes = unit.codes.long()
        native = torch.tensor(grid.native, dtype=torch.long)
        values = torch.tensor(grid.values, dtype=torch.float32)[native[codes]]
        weight = values * unit.scale_rows.float()[:, None]
        expected.append(weight if family == "T-8" else weight.bfloat16().float())
    got = smoke._dense_weight(item["parsed"][family], smoke.FAMILIES[family][0])
    assert torch.equal(got, torch.cat(expected)), "A decoded role changed its source rows or row scale"
    # The two replicated KDA roles must remain complete on both TP ranks.
    assert row["roles"][-2:] == [["f_a_proj", 128], ["g_a_proj", 128]]
    assert not torch.equal(got[-256:-128], got[-128:])


def test_numeric_screen_refuses_wrong_head_gate_dtype_and_values():
    expected = torch.tensor([[0.25, -0.75]], dtype=torch.float32)
    with pytest.raises(AssertionError, match="dtype"):
        smoke.compare(expected.bfloat16(), expected, name="head gate", dtype=torch.float32, exact=True)
    with pytest.raises(AssertionError, match="bits changed"):
        smoke.compare(expected + 0.125, expected, name="head gate", dtype=torch.float32, exact=True)


def test_numeric_screen_refuses_error_below_the_old_fixed_limit():
    left = torch.full((1, 16), 0.001, dtype=torch.bfloat16).double()
    weight = torch.ones(1, 16, dtype=torch.bfloat16).double()
    reference, bound = smoke.fb.dense_bound("value", left, weight, 16, 16)
    bad = reference.bfloat16() + torch.tensor(0.01, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="derived bound"):
        smoke.compare(bad, reference, name="dense", dtype=torch.bfloat16, bound=bound)


def test_numeric_screen_refuses_missing_vision_bias():
    left = torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16).double()
    weight = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.bfloat16).double()
    reference, error = smoke.fb.dense_bound("value", left, weight, 2, 2)
    biased, bound = smoke._bias_bound(reference, error, torch.tensor([[0.25, -0.25]]).double())
    with pytest.raises(AssertionError, match="vision"):
        smoke.compare(reference.bfloat16(), biased, name="vision", dtype=torch.bfloat16, bound=bound)


def test_bitwise_screen_distinguishes_signed_zero():
    with pytest.raises(AssertionError, match="bits changed"):
        smoke.compare(torch.tensor([-0.0]), torch.tensor([0.0]), name="stock", dtype=torch.float32, exact=True)


def test_numeric_screen_refuses_nonfinite_output():
    with pytest.raises(AssertionError, match="nonfinite"):
        smoke.compare(torch.tensor([[float("nan")]]), torch.zeros(1, 1), name="router", dtype=torch.float32)


def test_resident_footprint_counts_shared_tail_storage_once():
    layer = torch.nn.Module()
    layer.register_parameter("weight", torch.nn.Parameter(torch.ones(8, 16)))
    layer.register_buffer("tail", layer.weight.detach()[4:])
    layer.register_buffer("transposed_tail", layer.tail.t())
    footprint = smoke.resident_tensors(layer)
    assert footprint["bytes_by_device"] == {"cpu": 8 * 16 * 4}
    assert sum(row["unique_storage"] for row in footprint["tensors"]) == 1
