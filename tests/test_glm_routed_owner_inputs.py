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


def _sealed_manifest(units, source_seal="a4" * 32, fixture="03" * 32):
    return {"units": {unit: {"identity": {"encoder_source_sha256": source_seal,
                                          "encoder_fixture_id": fixture}}
                      for unit in units}}


def test_the_records_keep_the_producer_that_wrote_the_wire():
    """A newer pin's seal is reported, never stamped into an old wire's record."""
    served = _provenance_served()
    settled = inputs.preflight_producer_provenance(
        served, _sealed_manifest(["u.0.w1", "u.0.w2"]), [{"unit": "u.0.w1"}, {"unit": "u.0.w2"}])
    current = {"encoder_source_sha256": "59" * 32, "encoder_fixture_id": "03" * 32}
    block = inputs.wire_producer_provenance(served, settled, current)
    assert block["carried_encoder_source_sha256"] == "a4" * 32
    assert block["current_process_encoder_source_sha256"] == "59" * 32
    assert block["restamped_with_the_current_encoder"] is False
    assert block["carried_source_seal_is_this_processes"] is False
    assert block["carried_encoder_fixture_id"] == block["current_process_encoder_fixture_id"]
    assert block["sealed_records_agreeing"] == 2


def test_a_record_whose_seal_is_not_the_declared_producer_is_refused_before_any_output():
    served = _provenance_served()
    with pytest.raises(SystemExit, match="the records carry encoder_source_sha256"):
        inputs.preflight_producer_provenance(
            served, _sealed_manifest(["u.0.w1"], source_seal="59" * 32), [{"unit": "u.0.w1"}])


def test_an_export_with_no_declared_producer_is_refused():
    with pytest.raises(SystemExit, match="no historical producer"):
        inputs.preflight_producer_provenance(
            {"cached_units": {}}, _sealed_manifest(["u.0.w1"]), [{"unit": "u.0.w1"}])


def test_records_that_disagree_on_one_producer_are_refused():
    served = _provenance_served()
    manifest = _sealed_manifest(["u.0.w1"])
    manifest["units"]["u.0.w2"] = {"identity": {"encoder_source_sha256": "a4" * 32,
                                                "encoder_fixture_id": "04" * 32}}
    with pytest.raises(SystemExit, match="do not agree on one producer"):
        inputs.preflight_producer_provenance(
            served, manifest, [{"unit": "u.0.w1"}, {"unit": "u.0.w2"}])


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


def test_a_candidate_that_fails_its_check_never_replaces_the_previous_file(tmp_path):
    """A refusal must leave the last good artifact where a reader finds it.

    The destination is only ever replaced after the candidate has been flushed,
    fsynced and verified IN PLACE, so a verification failure leaves the prior
    container untouched and retains the rejected candidate beside it.  This is
    the ordering the previous writer got backwards: it renamed first and
    checked the destination afterwards, so a bad candidate could overwrite a
    good one.
    """
    values, plan = _tensors()
    destination = tmp_path / "source.safetensors"
    destination.write_bytes(b"the previous, accepted container\n")
    with pytest.raises(SystemExit, match="planted refusal"):
        inputs.streamed_safetensors(
            destination, plan, lambda key: values[key],
            check=lambda path, header: (_ for _ in ()).throw(
                SystemExit(f"{path}: planted refusal")))
    assert destination.read_bytes() == b"the previous, accepted container\n"
    partials = sorted(tmp_path.glob("source.safetensors.partial-*"))
    assert len(partials) == 1, partials
    assert partials[0].stat().st_size > 0, "the rejected candidate is retained as evidence"


def test_progress_is_reported_only_after_the_candidate_is_published(tmp_path):
    """The phase watcher hears about durable work, not about a loop position.

    Verification runs on the partial file; the report is made once, after the
    rename, and names a path that is the verified container.  A counter that
    advanced before the fsync would let a stalled write keep a phase alive
    while no reader could use the file.
    """
    values, plan = _tensors()
    seen = []
    written = inputs.streamed_safetensors(
        tmp_path / "streamed.safetensors", plan, lambda key: values[key],
        check=lambda path, header: seen.append(("check", path.name)),
        progress=lambda count, key: seen.append(("progress", count, Path(key).name)))
    assert [entry[0] for entry in seen] == ["check", "progress"], seen
    assert seen[-1] == ("progress", len(plan), "streamed.safetensors")
    assert Path(written["path"]).is_file()
    assert written["sha256"] == inputs.sha256_file(tmp_path / "streamed.safetensors")


def test_a_durable_record_is_replaced_atomically(tmp_path):
    """``dump`` publishes by rename, so a reader sees old or whole-new bytes."""
    path = tmp_path / "receipt.json"
    inputs.dump(path, {"n": 1})
    first = path.read_bytes()
    assert json.loads(first) == {"n": 1}
    inputs.dump(path, {"n": 2})
    assert json.loads(path.read_bytes()) == {"n": 2}
    assert first != path.read_bytes()
    assert not list(tmp_path.glob("receipt.json.tmp*")), "the candidate is not left behind"


def _directory_only_fsync(failure):
    """The real ``os.fsync``, except that a directory gets ``failure()``."""
    import os
    import stat

    real = os.fsync

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise failure()
        return real(fd)

    return fsync


def test_a_filesystem_that_cannot_sync_a_directory_is_named_not_swallowed(tmp_path, monkeypatch):
    """An unsupported directory sync is a policy, not a silent success.

    The one errno set that means "this filesystem defines no directory fsync"
    is reported back to the caller and recorded in the artifact; a genuine
    failure (EIO here) is not in that set and propagates instead.
    """
    import errno

    monkeypatch.setattr(inputs.os, "fsync",
                        _directory_only_fsync(lambda: OSError(errno.EINVAL, "unsupported")))
    assert inputs.fsync_directory(tmp_path) == "unsupported_by_filesystem"
    monkeypatch.setattr(inputs.os, "fsync",
                        _directory_only_fsync(lambda: OSError(errno.EIO, "planted io")))
    with pytest.raises(OSError, match="planted io"):
        inputs.fsync_directory(tmp_path)


def test_a_failed_directory_sync_is_never_reported_as_committed(tmp_path, monkeypatch):
    """The regression: a rename that is not durable must not commit progress.

    The candidate is already renamed at this point, so the failure is reported
    for what it is -- the verified file is in place and the publication is not
    durable -- and ``progress`` is never called.  Claiming the previous file was
    left untouched would be false here: a rename has happened.
    """
    import errno

    values, plan = _tensors()
    seen = []
    monkeypatch.setattr(inputs.os, "fsync",
                        _directory_only_fsync(lambda: OSError(errno.EIO, "planted dirsync io")))
    with pytest.raises(OSError, match="planted dirsync io"):
        inputs.streamed_safetensors(
            tmp_path / "streamed.safetensors", plan, lambda key: values[key],
            check=lambda path, header: None,
            progress=lambda count, key: seen.append((count, key)))
    assert seen == [], "a publication that is not durable is not committed work"
    assert (tmp_path / "streamed.safetensors").is_file(), \
        "the rewritten file is in place; the report says so instead of claiming otherwise"


def test_an_unsupported_directory_sync_still_publishes_and_names_the_policy(tmp_path, monkeypatch):
    """Where the filesystem defines no directory fsync, the rename still stands."""
    import errno

    values, plan = _tensors()
    seen = []
    monkeypatch.setattr(inputs.os, "fsync",
                        _directory_only_fsync(lambda: OSError(errno.ENOTSUP, "unsupported")))
    written = inputs.streamed_safetensors(
        tmp_path / "streamed.safetensors", plan, lambda key: values[key],
        check=lambda path, header: None,
        progress=lambda count, key: seen.append((count, key)))
    assert written["directory_sync"] == "unsupported_by_filesystem"
    assert seen == [(len(plan), written["path"])]
    assert Path(written["path"]).is_file()
