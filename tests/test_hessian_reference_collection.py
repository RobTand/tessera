"""A closed collection composes existing exact Hessian references (#625)."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from tessera.alphabet import E4M3_GRID
from tessera.cached_unit import CachedUnitIdentity, encoding_input_identity, tensor_identity
from tessera.errors import GrammarError
from tessera.export import ActivationSource
from tessera.hessian_capture import capture_sha256_from_units
from tessera.manifest import ScalePlaneKind

from test_hessian_reference_capture import POLICY, SCHEMA, write_json


COLLECTION_SCHEMA = "tessera.hessian_capture.collection.v1"


def _reference(root, hessians, provenance):
    (root / "inputs").mkdir(parents=True)
    shapes = {name: [4, 4] for name in hessians}
    counts = {name: 8 for name in hessians}
    census = root / "census.json"
    census_sha = write_json(census, {"unit_shapes": shapes, "counts": counts,
                                     "max_abs": {name: 1.0 for name in hessians}})
    entries = {}
    for name, H in hessians.items():
        path = root / "inputs" / f"{name}.pt"
        torch.save({"inputs": torch.ones(2, 4), "hessian": H, "name": name,
                    "source": "tessera_campaign_prefix_f32_v1", "count": 8,
                    "max_abs": 1.0}, path)
        entries[name] = {"path": f"inputs/{name}.pt",
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    canonical = root / "capture_manifest.json"
    canonical_sha = write_json(canonical, {
        "schema": "prismaquant.tessera_calibration_cache.v2", "status": "complete",
        "identity": {"schema": "prismaquant.tessera_calibration_cache.v2",
                     "census_sha256": census_sha, "units": shapes,
                     "storage_source": "tessera_campaign_prefix_f32_v1", "max_act_rows": 2,
                     "calibration": {k: v for k, v in provenance.items() if k != "hessian_role"}},
        "entries": entries})
    digest = ActivationSource(hessians, provenance).capture_sha256()
    handoff = root / "capture.references.json"
    write_json(handoff, {"schema": SCHEMA,
        "canonical_capture": {"path": str(canonical), "sha256": canonical_sha},
        "census": {"path": str(census), "sha256": census_sha},
        "provenance": provenance, "counts": counts,
        "hessians": {name: tensor_identity(H) for name, H in hessians.items()},
        "capture_sha256": digest,
        "rows": [{"units": sorted(hessians), "capture_sha256": digest}],
        "load_policy": dict(POLICY)})
    return handoff


@pytest.fixture
def collection(tmp_path):
    provenance = {"text_sha256": "a" * 64, "fit_ids_sha256": "b" * 64,
                  "fit_tokens": 8, "model": "fixture", "seqlen": 8,
                  "source": "fixture", "hessian_role": "fit"}
    a = {"a": torch.eye(4) * 3, "b": torch.eye(4) * 7}
    b = {"c": torch.eye(4) * 11}
    paths = [_reference(tmp_path / "one", a, provenance),
             _reference(tmp_path / "two", b, provenance)]
    merged = a | b
    document = tmp_path / "all.collection.references.json"
    write_json(document, {"schema": COLLECTION_SCHEMA,
        "references": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                       for path in paths],
        "units": sorted(merged),
        "capture_sha256": capture_sha256_from_units(
            provenance, {name: tensor_identity(H)["sha256"] for name, H in merged.items()})})
    return document, paths, merged, provenance


def test_collection_preserves_per_unit_cached_identity_and_lazy_uncached_read(collection, monkeypatch):
    document, paths, hessians, _ = collection
    original = torch.load
    loads = []
    def observed(*args, **kwargs):
        loads.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "load", observed)
    combined = ActivationSource.from_capture(document, ldlq_sigma=None)
    singles = [ActivationSource.from_capture(path, ldlq_sigma=None) for path in paths]
    assert set(combined.hessians) == set(hessians)
    assert combined.capture_sha256() == json.loads(document.read_text())["capture_sha256"]
    assert loads == [], "collection intake and sealing must not load H payloads"
    derive = lambda weight, unit_name, unit, grid, q256, *, activation: \
        encoding_input_identity(weight, unit_name, grid, q256, activation=activation)
    packed = CachedUnitIdentity(derive, combined, mode="committed")
    individual = [CachedUnitIdentity(derive, source, mode="committed") for source in singles]
    weight = torch.ones(4, 4, dtype=torch.bfloat16)
    for index, name in enumerate(sorted(hessians)):
        expected = individual[0 if name in "ab" else 1](weight, name, None, E4M3_GRID, 1024)
        assert packed(weight, name, None, E4M3_GRID, 1024) == expected
    assert combined.hessians.receipt()["verified_units"] == ["a"]
    assert combined.hessians.receipt()["committed_units_served"] == ["a", "b", "c"]
    assert [len(child["verified_units"]) for child in combined.hessians.receipt()["references"]] == [1, 0]
    recorded = packed.record()["reference"]
    assert recorded["binding"] == combined.reference_binding()
    assert [child["verified_units"] for child in recorded["consumption"]["references"]] == [["a"], []]
    assert len(loads) == 3, "each factory witnesses one unit; the collection witnesses one"
    combined_kwargs = combined.for_unit("c.weight", 4, "cpu", scale_plane=ScalePlaneKind.CHANNEL)
    individual_kwargs = singles[1].for_unit("c.weight", 4, "cpu", scale_plane=ScalePlaneKind.CHANNEL)
    assert set(combined_kwargs) == set(individual_kwargs)
    for key in combined_kwargs:
        if isinstance(combined_kwargs[key], torch.Tensor):
            torch.testing.assert_close(combined_kwargs[key], individual_kwargs[key], rtol=0, atol=0)
        else:
            assert combined_kwargs[key] == individual_kwargs[key]
    assert combined.hessians.receipt()["verified_units"] == ["a", "c"]
    for source in [combined, *singles]:
        source.hessians.close()


@pytest.mark.parametrize("problem", ["overlap", "mismatch", "omission", "mutation", "missing", "collection_mutation"])
def test_collection_refuses_invalid_member_or_roster(collection, problem):
    document, paths, _hessians, provenance = collection
    payload = json.loads(document.read_text())
    if problem == "overlap":
        payload["references"][1] = payload["references"][0]
    elif problem == "mismatch":
        other = _reference(document.parent / "other", {"d": torch.eye(4)},
                           dict(provenance, seqlen=9))
        payload["references"][1] = {"path": str(other),
                                    "sha256": hashlib.sha256(other.read_bytes()).hexdigest()}
    elif problem == "omission":
        payload["units"].remove("c")
    elif problem == "mutation":
        paths[1].write_text(paths[1].read_text() + " ")
    elif problem == "missing":
        paths[1].unlink()
    elif problem == "collection_mutation":
        payload["capture_sha256"] = "0" * 64
    if problem in {"overlap", "mismatch", "omission", "collection_mutation"}:
        write_json(document, payload)
    with pytest.raises((GrammarError, OSError, ValueError)):
        ActivationSource.from_capture(document)


def test_collection_mutation_after_open_refuses_and_closes_every_child(collection):
    document, paths, _hessians, _provenance = collection
    source = ActivationSource.from_capture(document)
    paths[1].write_text(paths[1].read_text() + " ")
    with pytest.raises(GrammarError, match="changed or was replaced"):
        source.capture_sha256()
    source.hessians.close()
    assert all(item["closed"] for item in source.hessians.receipt()["references"])


def test_failed_second_child_closes_both_opened_reference_owners(collection, monkeypatch):
    from tessera.hessian_capture import ReferenceHessians
    document, paths, _hessians, _provenance = collection
    payload = json.loads(document.read_text())
    payload["references"][1]["sha256"] = "0" * 64
    write_json(document, payload)
    original = ReferenceHessians.close
    closed = set()
    def observed(self):
        path = str(self._held[0].path)
        original(self)
        assert all(held.fd is None for held in self._held)
        closed.add(path)
    monkeypatch.setattr(ReferenceHessians, "close", observed)
    with pytest.raises(GrammarError, match="child checksum differs"):
        ActivationSource.from_capture(document)
    assert closed == {str(path) for path in paths}


def test_priced_inputs_binds_the_collection_and_every_child(collection, tmp_path):
    from test_priced_inputs_snapshot import exporter
    document, paths, _hessians, _provenance = collection
    source = ActivationSource.from_capture(document)
    block = {"schema": "tessera.priced_export_inputs.v2",
             "hessian_capture_sha256": source.capture_sha256(),
             "hessian_reference_binding": source.reference_binding(),
             "input_global_scales": {}}
    build = tmp_path / "build.json"
    def snapshot():
        return exporter.PricedInputsSnapshot(build, write_json(build, {"priced_inputs": block}))
    snapshot().require(source, {})
    block["hessian_reference_binding"]["references"][1]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="canonical Hessian reference differs"):
        snapshot().require(source, {})
    source.hessians.close()
