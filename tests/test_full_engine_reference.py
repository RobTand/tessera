"""Reference binding controls use synthetic bytes, not a served checkpoint."""
import json

import pytest

from experiments.full_engine_reference import digest, verify_reference_checkpoint


@pytest.fixture
def reference(tmp_path):
    def write(path, data):
        path.write_text(json.dumps(data))
        return digest(path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    write(checkpoint / "config.json", {"quantization_config": {"quant_method": "tessera"}})
    (checkpoint / "model.safetensors").write_bytes(b"synthetic checkpoint bytes")
    source, wire = {"files": {"source": "a" * 64}}, "b" * 64
    proof = {"schema": "tessera.full_model_original_wire_checkpoint.v1", "status": "passed",
        "anchors_sha256": "c" * 64, "wires": {"dense": {"sha256": wire}},
        "originals_sha256": write(tmp_path / "originals.json", {"anchors_sha256": "c" * 64, "census_sha256": "e" * 64,
            "units": {"dense": {"original_blob_sha256": wire}}}),
        "plan_sha256": write(tmp_path / "plan.json", {"dense.weight": {"grid": "E4M3", "q256": 1024}}),
        "source_identity_sha256": write(tmp_path / "source-identity.json", source),
        "checkpoint_files": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in checkpoint.iterdir()}}
    path = tmp_path / "export-proof.json"
    write(path, proof)
    return path, source, [{"unit_id": "l:dense", "module": "dense", "members": ["dense"]}], "e" * 64


def test_reference_binds_complete_members_and_actual_checkpoint_bytes(reference):
    result = verify_reference_checkpoint(*reference)
    assert result["assignment"]["units"] == {"l:dense": {"dense": "b" * 64}}
    assert result["proof_sha256"] == digest(reference[0])


@pytest.mark.parametrize("defect", ["source", "census", "wire", "missing_member", "changed_file", "extra_file", "input_digest", "symlink"])
def test_reference_ambiguity_refuses(reference, defect):
    path, source, roster, census_sha256 = reference
    proof = json.loads(path.read_text())
    if defect == "source":
        source = {"files": {"other": "a" * 64}}
    elif defect == "census":
        census_sha256 = "f" * 64
    elif defect == "wire":
        proof["wires"]["dense"]["sha256"] = "d" * 64
    elif defect == "missing_member":
        roster.append({"unit_id": "l:other", "module": "other", "members": ["other"]})
    elif defect == "changed_file":
        (path.parent / "checkpoint/model.safetensors").write_bytes(b"different bytes")
    elif defect == "extra_file":
        (path.parent / "checkpoint/unlisted.bin").write_bytes(b"extra")
    elif defect == "input_digest":
        (path.parent / "plan.json").write_text("{}")
    else:
        file = path.parent / "checkpoint/model.safetensors"
        target = path.parent / "elsewhere"
        file.rename(target)
        file.symlink_to(target)
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError):
        verify_reference_checkpoint(path, source, roster, census_sha256)
