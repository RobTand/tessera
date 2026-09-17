"""The GLM routed-owner input producer's own contract, on the CPU.

No model, no GPU, no census mount: what is under test is the part this driver
OWNS -- the member roster's order and naming, the cached-units binding, and the
streamed safetensors container, which is compared byte for byte against
``safetensors.torch.save_file`` for the same tensors rather than against a
second implementation of the format here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments import glm_routed_owner_inputs as inputs


def _entry(count=2):
    """A producer plan entry in the layout ``project_expert_plan`` returns."""
    units = []
    for expert in range(count):
        for projection, group, shape in (("gate_proj", "w13", (8, 4)),
                                         ("up_proj", "w13", (8, 4)),
                                         ("down_proj", "w2", (4, 8))):
            tensor = f"model.language_model.layers.3.mlp.experts.{expert}.{projection}.weight"
            units.append({"tensor": tensor, "wire": tensor[:-len(".weight")] + ".wire",
                          "source_tensor": tensor, "source_layout": "unpacked_per_expert",
                          "source_slice": {"expert": expert, "selector": "whole", "transpose": False},
                          "expert": expert, "projection": projection, "group": group,
                          "rows": shape[0], "cols": shape[1]})
    return {"stack": "model.language_model.layers.3.mlp.experts", "experts": count,
            "source_layout": "unpacked_per_expert", "units": units}


def test_the_roster_is_the_owners_expert_role_order():
    members = inputs.member_roster(_entry())
    assert [(m["expert"], m["role"]) for m in members] == [
        (0, "w1"), (0, "w3"), (0, "w2"), (1, "w1"), (1, "w3"), (1, "w2")]
    assert [m["projection"] for m in members] == [
        "gate_proj", "up_proj", "down_proj"] * 2
    assert members[0]["unit"] == "model.language_model.layers.3.mlp.experts.0.gate_proj"


def test_a_scrambled_plan_is_refused_rather_than_sorted():
    entry = _entry()
    entry["units"][0], entry["units"][1] = entry["units"][1], entry["units"][0]
    with pytest.raises(SystemExit, match="roster position"):
        inputs.member_roster(entry)


def test_a_unit_that_is_not_its_own_source_tensor_is_refused():
    entry = _entry()
    entry["units"][0]["source_tensor"] = "model.language_model.layers.3.mlp.experts.gate_up_proj"
    with pytest.raises(SystemExit, match="whole source tensor"):
        inputs.member_roster(entry)


def test_the_research_selected_owner_comes_from_the_exports_own_record():
    block = {"input_sha256": "ab" * 32, "input_utf8": "{}"}
    served = {"export_identity": {"options": {"research_selected_moe": block}}}
    assert inputs.research_selected_record(served) == block
    assert inputs.research_selected_record({"export_identity": {"options": {}}}) is None
    assert inputs.research_selected_record({}) is None


def _provenance_served(seal="a4" * 32):
    return {"cached_units": {"historical_producer": {"package": "/pkg", "source_sha256": seal}}}


def test_the_records_keep_the_producer_that_wrote_the_wire():
    """A newer pin's seal is reported, never stamped into an old wire's record."""
    served = _provenance_served()
    carried = {"encoder_source_sha256": "a4" * 32, "encoder_fixture_id": "03" * 32}
    current = {"encoder_source_sha256": "59" * 32, "encoder_fixture_id": "03" * 32}
    block = inputs.wire_producer_provenance(served, carried, current)
    assert block["carried_encoder_source_sha256"] == "a4" * 32
    assert block["current_process_encoder_source_sha256"] == "59" * 32
    assert block["restamped_with_the_current_encoder"] is False
    assert block["carried_source_seal_is_this_processes"] is False
    assert block["carried_encoder_fixture_id"] == block["current_process_encoder_fixture_id"]


def test_a_record_whose_seal_is_not_the_declared_producer_is_refused():
    served = _provenance_served()
    with pytest.raises(SystemExit, match="the records carry encoder_source_sha256"):
        inputs.wire_producer_provenance(
            served, {"encoder_source_sha256": "59" * 32, "encoder_fixture_id": "03" * 32},
            {"encoder_source_sha256": "59" * 32, "encoder_fixture_id": "03" * 32})


def test_an_export_with_no_declared_producer_is_refused():
    with pytest.raises(SystemExit, match="no historical producer"):
        inputs.wire_producer_provenance(
            {"cached_units": {}},
            {"encoder_source_sha256": "a4" * 32, "encoder_fixture_id": "03" * 32},
            {"encoder_source_sha256": "59" * 32, "encoder_fixture_id": "03" * 32})


def _write_bundle(tmp_path: Path, value) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return path


def test_a_bundle_that_is_not_the_export_declared_bundle_is_refused(tmp_path):
    bundle = {"schema": "tessera.cached_units.v1", "source": {"x": 1}, "units": {"u": {}}}
    path = _write_bundle(tmp_path, bundle)
    served = {"cached_units": {"manifest_sha256": inputs.canonical_sha256(bundle),
                               "planned_units": 1}}
    assert inputs.bundle_manifest(served, path)["schema"] == "tessera.cached_units.v1"
    served["cached_units"]["manifest_sha256"] = "00" * 32
    with pytest.raises(SystemExit, match="not the one"):
        inputs.bundle_manifest(served, path)


def test_a_bundle_with_a_different_unit_count_is_refused(tmp_path):
    bundle = {"schema": "tessera.cached_units.v1", "source": {"x": 1}, "units": {"u": {}}}
    path = _write_bundle(tmp_path, bundle)
    served = {"cached_units": {"manifest_sha256": inputs.canonical_sha256(bundle),
                               "planned_units": 2}}
    with pytest.raises(SystemExit, match="1 units, not 2"):
        inputs.bundle_manifest(served, path)


def _tensors():
    values = {"source_weight/b": torch.arange(4, dtype=torch.float32).reshape(2, 2),
              "source_weight/a": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)}
    plan = [(key, str(values[key].dtype), tuple(values[key].shape),
             values[key].numel() * values[key].element_size()) for key in sorted(values)]
    return values, plan


def _container_header(path: Path):
    """Parse the container by hand: length word, padded JSON, then the region."""
    import struct

    raw = path.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    assert (8 + length) % 8 == 0, "the header region is not 8-byte aligned"
    header = json.loads(raw[8:8 + length])
    return header, raw[8 + length:]


def test_the_streamed_writer_is_a_container_safetensors_itself_reads(tmp_path):
    """The bytes are the format's, and the payload is the plan's, exactly.

    Ordering inside the header is the format's to choose (``safetensors``
    builds it from an unordered map), so this does not compare files byte for
    byte with ``save_file``: it reads the container back through
    ``safetensors`` -- the reader the harness will use -- and checks every
    tensor, then checks the header arithmetic that a writer can get wrong.
    """
    from safetensors import safe_open

    values, plan = _tensors()
    streamed = tmp_path / "streamed.safetensors"
    checked = {}
    written = inputs.streamed_safetensors(
        streamed, plan, lambda key: values[key],
        check=lambda path, header: checked.update(header))
    assert set(checked) == set(values)
    assert written["device_bytes"] == sum(entry["data_offsets"][1] - entry["data_offsets"][0]
                                          for entry in checked.values())

    with safe_open(str(streamed), framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == set(values)
        for key, value in values.items():
            assert str(handle.get_tensor(key).dtype) == str(value.dtype)
            assert torch.equal(handle.get_tensor(key), value)

    header, region = _container_header(streamed)
    offsets = sorted((entry["data_offsets"] for entry in header.values()))
    assert offsets[0][0] == 0 and offsets[-1][1] == len(region)
    assert all(next_offset == end for (_, end), (next_offset, _) in zip(offsets, offsets[1:])), \
        "the payload region has a gap or an overlap"
    for key, dtype, shape, length in plan:
        assert header[key]["dtype"] == inputs.DTYPE_NAMES[dtype]
        assert header[key]["shape"] == list(shape)
        assert header[key]["data_offsets"][1] - header[key]["data_offsets"][0] == length


def test_the_streamed_writer_refuses_a_tensor_that_is_not_the_plan(tmp_path):
    values, plan = _tensors()
    plan[0] = (plan[0][0], plan[0][1], (3, 3), plan[0][3])
    with pytest.raises(SystemExit, match="is not the plan's"):
        inputs.streamed_safetensors(tmp_path / "out.safetensors", plan,
                                    lambda key: values[key], check=lambda path, header: None)
    assert not (tmp_path / "out.safetensors").exists()
