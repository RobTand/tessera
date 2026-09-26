"""Independent original cache cohorts share one checked source and plan."""
import copy
import hashlib
import json

import pytest

from tessera.cached_unit import CachedUnitBundle
from test_rooted_cached_bundle import rooted


def _bound(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def cohorts(tmp_path):
    body, roots = rooted(tmp_path)
    mtp = body["units"].pop("expert")
    del body["unit_roots"]["expert"]
    del body["wire_roots"]["added"]
    del body["producer_packages"]["b" * 64]
    body["encoder_adoptions"] = {}
    body["reuse_authority"]["encoder_source_proofs"] = []
    body["source"] = {"sha256": "same-whole-checkpoint"}
    original_mtp = {"schema": "tessera.cached_units.v1", "source": body["source"],
                    "units": {"expert": mtp}}
    child = [
        {"manifest": _bound(tmp_path / "body.json", body), "producer_package": None},
        {"manifest": _bound(roots["added"] / "manifest.json", original_mtp),
         "producer_package": {"path": str(tmp_path / "producer-b"), "sha256": "b" * 64}},
    ]
    return {"schema": "tessera.cached_units.v3", "source": body["source"],
            "children": child}, body, original_mtp


def test_independent_cohorts_require_composition(tmp_path):
    parent, body, mtp = cohorts(tmp_path)
    invalid_migration = copy.deepcopy(body)
    invalid_migration["units"].update(mtp["units"])
    invalid_migration["unit_roots"]["expert"] = "added"
    invalid_migration["wire_roots"]["added"] = str(tmp_path / "added")
    invalid_migration["producer_packages"]["b" * 64] = parent["children"][1]["producer_package"]
    with pytest.raises(ValueError, match="adoption coverage"):
        CachedUnitBundle(invalid_migration, tmp_path, {"dense", "expert"}, parent["source"])

    bundle = CachedUnitBundle(parent, tmp_path, {"dense", "expert"}, parent["source"])
    assert set(bundle.units) == {"dense", "expert"}
    assert set(bundle.producer_packages) == {"a" * 64, "b" * 64}
    for name in bundle.units:
        blob, record = bundle.read(name)
        assert blob == b"unchanged wire" and record["identity"]["unit"] == name
    assert [child["schema"] for child in bundle.child_manifests] == [
        "tessera.cached_units.v2", "tessera.cached_units.v1"]


def test_composition_preserves_source_part_proof(tmp_path):
    parent, body, mtp = cohorts(tmp_path)
    whole = {"config_sha256": "a" * 64, "auxiliary_sha256": "b" * 64,
             "files": {"shard-0": "c" * 64, "shard-1": "d" * 64},
             "tensors": {"dense.weight": "shard-0", "expert.weight": "shard-1"}}
    parent["source"] = body["source"] = mtp["source"] = whole
    parent["children"][0]["manifest"] = _bound(tmp_path / "body.json", body)
    parent["children"][1]["manifest"] = _bound(tmp_path / "added" / "manifest.json", mtp)
    part = {"schema": "tessera.source-part.v1", **whole,
            "files": {"shard-0": whole["files"]["shard-0"]}}
    assert set(CachedUnitBundle(parent, tmp_path, {"dense", "expert"}, part).units) == {
        "dense", "expert"}
    part["files"]["shard-0"] = "e" * 64
    with pytest.raises(ValueError, match="source identity changed"):
        CachedUnitBundle(parent, tmp_path, {"dense", "expert"}, part)


def test_parent_manifest_mutation_cannot_change_bound_provenance(tmp_path):
    parent, _, _ = cohorts(tmp_path)
    bundle = CachedUnitBundle(parent, tmp_path, {"dense", "expert"}, parent["source"])
    original = copy.deepcopy(bundle.child_manifests)
    packages = copy.deepcopy(bundle.producer_packages)
    parent["children"][0]["manifest"]["sha256"] = "0" * 64
    parent["children"][1]["producer_package"]["path"] = "/untrusted"
    assert bundle.child_manifests == original
    assert bundle.producer_packages == packages


@pytest.mark.parametrize("change", ["missing", "overlap", "source", "tampered", "alias",
                                     "package", "unbound", "recursive", "adoption",
                                     "served_policy", "producer_conflict", "malformed_seal"])
def test_composition_refuses_bad_children(tmp_path, change):
    parent, body, mtp = cohorts(tmp_path)
    if change == "missing":
        parent["children"].pop()
    elif change == "overlap":
        mtp["units"]["dense"] = body["units"]["dense"]
        parent["children"][1]["manifest"] = _bound(tmp_path / "added" / "manifest.json", mtp)
    elif change == "source":
        mtp["source"] = {"sha256": "different"}
        parent["children"][1]["manifest"] = _bound(tmp_path / "added" / "manifest.json", mtp)
    elif change == "tampered":
        (tmp_path / "body.json").write_text("{}")
    elif change == "alias":
        parent["children"].append(copy.deepcopy(parent["children"][0]))
    elif change == "package":
        parent["children"][1]["producer_package"]["sha256"] = "a" * 64
    elif change == "unbound":
        parent["children"][1]["producer_package"] = None
    elif change == "recursive":
        mtp["schema"] = "tessera.cached_units.v3"
        parent["children"][1]["manifest"] = _bound(tmp_path / "added" / "manifest.json", mtp)
    elif change == "served_policy":
        body["served_activations"] = {"dense": {"group": "invented", "input_global_scale": 1.0}}
        parent["children"][0]["manifest"] = _bound(tmp_path / "body.json", body)
    elif change == "producer_conflict":
        mtp["units"]["expert"]["identity"]["encoder_source_sha256"] = "a" * 64
        parent["children"][1]["manifest"] = _bound(tmp_path / "added" / "manifest.json", mtp)
        parent["children"][1]["producer_package"]["sha256"] = "a" * 64
    elif change == "malformed_seal":
        mtp["units"]["expert"]["identity"]["encoder_source_sha256"] = "short"
        parent["children"][1]["manifest"] = _bound(tmp_path / "added" / "manifest.json", mtp)
        parent["children"][1]["producer_package"]["sha256"] = "short"
    else:
        body["encoder_adoptions"]["dense"] = {"invented": True}
        parent["children"][0]["manifest"] = _bound(tmp_path / "body.json", body)
    with pytest.raises(ValueError):
        CachedUnitBundle(parent, tmp_path, {"dense", "expert"}, parent["source"])
