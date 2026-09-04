"""Exact campaign blobs cross the producer boundary without another encode."""
from __future__ import annotations

import copy
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tessera.alphabet import E4M3_GRID
from tessera.export import ActivationSource, encode_linear
from tessera.fused import parse_fused

ROOT = Path(__file__).resolve().parents[1]
STACK = "model.layers.2.feed_forward.experts"
TENSOR = STACK + ".0.w1.weight"
UNIT = TENSOR.removesuffix(".weight")


def _api():
    return importlib.import_module("tessera.cached_unit")


def _exporter():
    spec = importlib.util.spec_from_file_location(
        "cached_test_exporter", ROOT / "experiments/export_tessera_serving.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def encoded():
    weight = torch.randn(32, 32, generator=torch.Generator().manual_seed(183)).bfloat16()
    unit = encode_linear(weight.float(), grid=E4M3_GRID, q256=1024,
                         name="TESSERA_E4M3_K1_R1024", verify=False)
    return weight, unit.blob


def _projection():
    return {"tensor": TENSOR, "source_tensor": TENSOR,
            "source_layout": "unpacked_per_expert", "expert": 0,
            "source_slice": {"expert": 0, "selector": "whole", "transpose": False},
            "projection": "gate_proj", "group": "w13", "rows": 32, "cols": 32}


def _record(encoded, activation=None):
    api = _api()
    weight, blob = encoded
    identity = api.unit_input_identity(weight, _projection(), E4M3_GRID, 1024,
                                       activation=activation)
    return api.make_unit_record(blob, identity, filename="unit.tessera"), identity


def test_cached_blob_round_trip_keeps_original_unit_name_and_bytes(encoded):
    api = _api()
    record, expected = _record(encoded)
    accepted = api.verify_cached_unit(encoded[1], record, expected)
    assert accepted.blob == encoded[1]
    assert accepted.manifest.branch.unit_id == "TESSERA_E4M3_K1_R1024"
    assert accepted.wire_bytes < len(accepted.blob)


@pytest.mark.parametrize("field", ["source", "projection", "calibration", "recipe",
                                   "encoder_source_sha256", "encoder_fixture_id"])
def test_each_input_identity_mismatch_refuses(encoded, field):
    api = _api()
    record, expected = _record(encoded)
    record["identity"][field] = {"wrong": True}
    with pytest.raises(ValueError, match=field):
        api.verify_cached_unit(encoded[1], record, expected)


def test_source_values_are_bound_not_only_shape(encoded):
    api = _api()
    record, _ = _record(encoded)
    altered = encoded[0].clone()
    altered[0, 0] += 1
    expected = api.unit_input_identity(altered, _projection(), E4M3_GRID, 1024)
    with pytest.raises(ValueError, match="source"):
        api.verify_cached_unit(encoded[1], record, expected)


def test_hessian_values_and_full_activation_settings_are_bound(encoded):
    api = _api()
    provenance = {"text_sha256": "a" * 64, "fit_ids_sha256": "b" * 64,
                  "fit_tokens": 32}
    first = ActivationSource({UNIT: torch.eye(32)}, provenance)
    changed = ActivationSource({UNIT: torch.eye(32) * 2}, provenance)
    settings = ActivationSource({UNIT: torch.eye(32)}, provenance,
                                refit_objective_trailing="h^0.5")
    identities = [api.unit_input_identity(encoded[0], _projection(), E4M3_GRID,
                                          1024, activation=a)
                  for a in (first, changed, settings)]
    assert identities[0]["calibration"] != identities[1]["calibration"]
    assert identities[0]["calibration"] != identities[2]["calibration"]


def test_missing_expert_hessian_refuses(encoded):
    api = _api()
    activation = ActivationSource({}, {"text_sha256": "a" * 64,
                                        "fit_ids_sha256": "b" * 64, "fit_tokens": 32})
    with pytest.raises(ValueError, match="Hessian"):
        api.unit_input_identity(encoded[0], _projection(), E4M3_GRID, 1024,
                                activation=activation)


def test_digest_covers_header_and_payload(encoded):
    api = _api()
    record, expected = _record(encoded)
    corrupted = bytearray(encoded[1])
    corrupted[-1] ^= 1
    with pytest.raises(ValueError, match="sha256"):
        api.verify_cached_unit(bytes(corrupted), record, expected)


def test_relabelled_wrong_rung_cannot_pass_receipt_comparison(encoded):
    api = _api()
    record, _ = _record(encoded)
    expected = api.unit_input_identity(encoded[0], _projection(), E4M3_GRID, 1280)
    record["identity"] = copy.deepcopy(expected)
    with pytest.raises(ValueError, match="rung|profile|recipe"):
        api.verify_cached_unit(encoded[1], record, expected)


@pytest.mark.parametrize("problem", ["missing", "extra", "duplicate", "escape"])
def test_manifest_refuses_ambiguous_coverage_before_loading_blobs(tmp_path, encoded, problem):
    api = _api()
    record, _ = _record(encoded)
    entries = {UNIT: record}
    if problem == "missing":
        entries = {}
    elif problem == "extra":
        entries[UNIT + "_other"] = copy.deepcopy(record)
    elif problem == "duplicate":
        entries[UNIT + "_other"] = copy.deepcopy(record)
        entries[UNIT + "_other"]["identity"]["unit"] = UNIT + "_other"
    else:
        entries[UNIT]["file"] = "../unit.tessera"
    manifest = {"schema": api.CACHE_SCHEMA, "source": {"sha256": "source"},
                "units": entries}
    with pytest.raises(ValueError, match="coverage|filename|duplicate"):
        expected = set(entries) if problem == "duplicate" else {UNIT}
        api.CachedUnitBundle(manifest, tmp_path, expected, {"sha256": "source"})


def test_projection_uses_producer_role_order_and_group_geometry():
    exporter = _exporter()
    shapes = {f"{STACK}.{expert}.{role}.weight": [32, 32]
              for expert in range(2) for role in ("w1", "w2", "w3")}
    projected = exporter.project_expert_plan(shapes, {},
                                             {STACK: {"grid": "E4M3", "q256": 1024}})
    units = projected["stacks"][STACK]["units"]
    assert [u["tensor"].split(".")[-2] for u in units] == ["w1", "w3", "w2"] * 2
    assert [u["projection"] for u in units] == ["gate_proj", "up_proj", "down_proj"] * 2
    assert projected["stacks"][STACK]["groups"]["w13"]["rows"] == 64
    assert json.loads(json.dumps(projected)) == projected


def test_projection_refuses_partial_source_stack():
    exporter = _exporter()
    with pytest.raises(SystemExit, match="missing"):
        exporter.project_expert_plan({TENSOR: [32, 32]}, {},
                                    {STACK: {"grid": "E4M3", "q256": 1024}})


def test_cached_packaging_never_calls_encoder(encoded, monkeypatch):
    exporter = _exporter()
    record, expected = _record(encoded)
    def forbidden(*args, **kwargs):
        raise AssertionError("cached input was re-encoded")
    monkeypatch.setattr(exporter, "encode_linear_planes", forbidden)
    accepted, packed = exporter.pack_cached_expert_unit(encoded[1], record, expected)
    members = parse_fused(packed)
    assert len(members) == 1
    assert members[0].blob == encoded[1]
    assert accepted.blob == encoded[1]


def test_export_consumes_complete_bundle_without_encoder(tmp_path, encoded, monkeypatch):
    from safetensors import safe_open
    from safetensors.torch import save_file
    from tessera.serving_parts import source_identity

    api, exporter = _api(), _exporter()
    src = tmp_path / "src"
    src.mkdir()
    tensors = {f"{STACK}.0.{role}.weight": encoded[0].clone() for role in ("w1", "w2", "w3")}
    save_file(tensors, str(src / "model.safetensors"))
    (src / "config.json").write_text(json.dumps({"architectures": ["Lfm2MoeForCausalLM"],
                                                "hidden_size": 32, "moe_intermediate_size": 32,
                                                "num_experts": 1}))
    choices = {STACK: {"grid": "E4M3", "q256": 1024}}
    projection = exporter.project_expert_plan({k: list(v.shape) for k, v in tensors.items()},
                                               {}, choices)
    cache = tmp_path / "cache"
    cache.mkdir()
    records = {}
    for index, unit in enumerate(projection["stacks"][STACK]["units"]):
        identity = api.unit_input_identity(tensors[unit["source_tensor"]], unit, E4M3_GRID, 1024)
        filename = f"unit-{index}.tessera"
        (cache / filename).write_bytes(encoded[1])
        records[identity["unit"]] = api.make_unit_record(encoded[1], identity, filename=filename)
    manifest = {"schema": api.CACHE_SCHEMA, "source": source_identity(src), "units": records}
    manifest_path = cache / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(choices))
    def forbidden(*args, **kwargs):
        raise AssertionError("cached export attempted an encode")
    monkeypatch.setattr(exporter, "encode_linear_planes", forbidden)
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["export", str(src), str(out), "--plan-json", str(plan_path),
                                    "--cached-expert-units", str(manifest_path), "--device", "cpu",
                                    "--allow-unrouted", "--allow-unserveable"])
    exporter.main()
    with safe_open(str(out / "model.safetensors"), framework="pt") as handle:
        for name in tensors:
            packed = bytes(handle.get_tensor(name.removesuffix(".weight") + ".wire").tolist())
            assert parse_fused(packed)[0].blob == encoded[1]
    receipt = json.loads((out / "tessera_serving_manifest.json").read_text())
    assert receipt["cached_expert_units"]["planned_units"] == 3
    assert len(receipt["modules"][STACK]["roles"]) == 3
    assert all(role["cached_blob_sha256"] for role in receipt["modules"][STACK]["roles"])


def test_projection_packed_layout_is_explicit_and_serialized():
    exporter = _exporter()
    stack = "model.layers.2.mlp.experts"
    shapes = {stack + ".gate_up_proj": [2, 64, 32], stack + ".down_proj": [2, 32, 32]}
    config = {"num_experts": 2, "hidden_size": 32, "moe_intermediate_size": 32}
    with pytest.raises(SystemExit, match="source_layout"):
        exporter.project_expert_plan(shapes, config, {stack: {"grid": "E4M3", "q256": 1024}})
    result = exporter.project_expert_plan(shapes, config,
        {stack: {"grid": "E4M3", "q256": 1024, "source_layout": "out_first_chunked"}})
    units = result["stacks"][stack]["units"]
    assert [u["source_slice"]["selector"] for u in units] == ["first_half", "second_half", "whole"] * 2
    assert units[0]["source_tensor"] == stack + ".gate_up_proj"
    assert units[0]["tensor"].endswith(".0.gate_proj.weight")


def test_cache_manifest_rejects_duplicate_json_keys(tmp_path):
    api = _api()
    path = tmp_path / "bad.json"
    path.write_text('{"units": {"same": 1, "same": 2}}')
    with pytest.raises(ValueError, match="duplicate"):
        api.read_manifest(path)


def test_encoding_identity_is_shared_by_dense_and_projected_callers():
    api = _api()
    weight = torch.ones(32, 32, dtype=torch.bfloat16)
    common = api.encoding_input_identity(weight, UNIT, E4M3_GRID, 1024)
    projected = api.unit_input_identity(weight, _projection(), E4M3_GRID, 1024)
    assert set(projected) - set(common) == {"projection"}
    assert {key: value for key, value in common.items() if key != "schema"} == {
        key: value for key, value in projected.items() if key not in {"schema", "projection"}}
    dense = api.encoding_input_identity(weight, "model.layers.0.self_attn.q_proj",
                                        E4M3_GRID, 1024)
    assert "projection" not in dense
    assert dense["source"] == common["source"]
    assert dense["unit"] != common["unit"]


def test_dense_resume_uses_same_record_and_wire_validation(encoded):
    api = _api()
    expected = api.encoding_input_identity(encoded[0], "model.layers.0.self_attn.q_proj",
                                           E4M3_GRID, 1024)
    record = api.make_unit_record(encoded[1], expected, filename="dense.tessera")
    accepted = api.verify_cached_unit(encoded[1], record, expected)
    assert accepted.blob == encoded[1]
    changed = api.encoding_input_identity(encoded[0], "model.layers.0.self_attn.q_proj",
                                          E4M3_GRID, 1280)
    with pytest.raises(ValueError, match="recipe"):
        api.verify_cached_unit(encoded[1], record, changed)


def test_cached_record_refuses_reach_disagreeing_with_profile(encoded):
    from dataclasses import replace
    from tessera.container import parse, serialize
    from tessera.manifest import ReachParams

    api = _api()
    _record_, expected = _record(encoded)
    artifact = parse(encoded[1])
    changed = replace(artifact.manifest, reach=ReachParams(window_seed=1))
    blob = serialize(changed, artifact.plane_region)
    with pytest.raises(ValueError, match="profile|reach"):
        api.make_unit_record(blob, expected, filename="changed-reach.tessera")


def test_cached_record_refuses_single_incomplete_terminal(encoded):
    from dataclasses import replace
    from fractions import Fraction
    import hashlib
    from tessera.container import parse, serialize
    from tessera.footprint import plane_region_bytes

    api = _api()
    _record_, expected = _record(encoded)
    artifact = parse(encoded[1])
    counts = list(artifact.terminal.plane_elements)
    last = max(index for index, count in enumerate(counts) if count)
    counts[last] = 0
    terminal = replace(artifact.terminal, plane_elements=tuple(counts))
    length = plane_region_bytes(artifact.manifest, terminal)
    prefix = artifact.plane_region[:length]
    digest = hashlib.sha256(prefix).digest()
    terminal = replace(terminal, exact_bytes=length,
                       exact_bpp=Fraction(8 * length, artifact.manifest.geometry.quantizable_params),
                       payload_digest=digest)
    manifest = replace(artifact.manifest, terminals=(terminal,), payload_digest=digest)
    blob = serialize(manifest, prefix)
    with pytest.raises(ValueError, match="complete|prefix"):
        api.make_unit_record(blob, expected, filename="prefix.tessera")
