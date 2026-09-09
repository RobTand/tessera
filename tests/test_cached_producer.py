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
    config = {"num_experts": 2, "hidden_size": 32, "moe_intermediate_size": 32}
    projected = exporter.project_expert_plan(shapes, config,
                                             {STACK: {"grid": "E4M3", "q256": 1024}})
    units = projected["stacks"][STACK]["units"]
    assert [u["tensor"].split(".")[-2] for u in units] == ["w1", "w3", "w2"] * 2
    assert [u["projection"] for u in units] == ["gate_proj", "up_proj", "down_proj"] * 2
    assert projected["stacks"][STACK]["groups"]["w13"]["rows"] == 64
    assert json.loads(json.dumps(projected)) == projected


def test_projection_refuses_partial_source_stack():
    exporter = _exporter()
    with pytest.raises(SystemExit, match="missing"):
        exporter.project_expert_plan(
            {TENSOR: [32, 32]},
            {"num_experts": 1, "hidden_size": 32, "moe_intermediate_size": 32},
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
    config = {"architectures": ["Lfm2MoeForCausalLM"],
              "hidden_size": 32, "moe_intermediate_size": 32, "num_experts": 1}
    (src / "config.json").write_text(json.dumps(config))
    choices = {STACK: {"grid": "E4M3", "q256": 1024}}
    projection = exporter.project_expert_plan({k: list(v.shape) for k, v in tensors.items()},
                                               config, choices)
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


@pytest.mark.parametrize("include_experts", [False, True])
@pytest.mark.parametrize("omit_dense", [False, True])
def test_complete_cached_export_preserves_dense_and_expert_originals(
        tmp_path, encoded, monkeypatch, include_experts, omit_dense):
    from safetensors import safe_open
    from safetensors.torch import save_file
    from tessera.serving_parts import source_identity

    api, exporter = _api(), _exporter()
    src, cache = tmp_path / "src", tmp_path / "cache"
    src.mkdir(); cache.mkdir()
    dense = {f"model.layers.0.feed_forward.{role}.weight": encoded[0].clone()
             for role in ("w1", "w3", "w2")}
    tensors = dict(dense)
    choices = {name: {"grid": "E4M3", "q256": 1024} for name in dense}
    config = {"architectures": ["Lfm2MoeForCausalLM"],
              "hidden_size": 32, "moe_intermediate_size": 32, "num_experts": 1}
    if include_experts:
        tensors.update({f"{STACK}.0.{role}.weight": encoded[0].clone()
                        for role in ("w1", "w3", "w2")})
        choices[STACK] = {"grid": "E4M3", "q256": 1024}
    save_file(tensors, str(src / "model.safetensors"))
    (src / "config.json").write_text(json.dumps(config))
    identities = [api.encoding_input_identity(weight, name, E4M3_GRID, 1024)
                  for name, weight in dense.items()]
    if include_experts:
        projection = exporter.project_expert_plan({k: list(v.shape) for k, v in tensors.items()},
                                                  config, {STACK: choices[STACK]})
        identities.extend(api.unit_input_identity(tensors[unit["source_tensor"]], unit,
                          E4M3_GRID, 1024) for unit in projection["stacks"][STACK]["units"])
    records = {}
    for index, identity in enumerate(identities):
        filename = f"unit-{index}.tessera"
        (cache / filename).write_bytes(encoded[1])
        records[identity["unit"]] = api.make_unit_record(encoded[1], identity, filename=filename)
    if omit_dense:
        del records[next(iter(dense)).removesuffix(".weight")]
    manifest_path = cache / "manifest.json"
    manifest_path.write_text(json.dumps({"schema": api.CACHE_SCHEMA,
        "source": source_identity(src), "units": records}))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(choices))
    def forbidden(*args, **kwargs):
        raise AssertionError("an original dense or expert wire was re-encoded")
    monkeypatch.setattr(exporter, "encode_linear_planes", forbidden)
    # The packaged LFM census has real 7168-row MLP roles; this miniature
    # checkpoint has 32-row roles and exercises the same fusion partitioner.
    monkeypatch.setattr(exporter, "output_partitions",
                        lambda census, module: [32, 32] if module.endswith('.w13') else [32])
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["export", str(src), str(out), "--plan-json", str(plan_path),
        "--cached-units", str(manifest_path), "--device", "cpu", "--allow-unrouted", "--allow-unserveable"])
    if omit_dense:
        with pytest.raises(ValueError, match="coverage"):
            exporter.main()
        assert not out.exists()
        return
    exporter.main()
    with safe_open(str(out / "model.safetensors"), framework="pt") as handle:
        members = [unit for name in handle.keys() if name.endswith((".wire", ".wire_bytes"))
                   for unit in parse_fused(handle.get_tensor(name).numpy().tobytes())]
    assert len(members) == len(tensors)
    assert all(unit.blob == encoded[1] for unit in members)
    receipt = json.loads((out / "tessera_serving_manifest.json").read_text())
    assert receipt["cached_units"]["planned_units"] == len(tensors)
    assert all(role["cached_blob_sha256"] for module in receipt["modules"].values()
               for role in module["roles"])


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


@pytest.fixture(scope="module")
def calibrated_wires():
    """Real CPU wires fitted to distinct source/H pairs for every expert role."""
    from tessera.export import wire_recipe

    provenance = dict(text_sha256="a" * 64, fit_ids_sha256="b" * 64,
                      fit_tokens=32, model="fixture", seqlen=32, source="fixture",
                      hessian_role="fit")
    result = []
    for index in range(6):
        weight = torch.randn(32, 32, generator=torch.Generator().manual_seed(200 + index)).bfloat16()
        hessian = torch.diag(torch.linspace(1 + index, 2 + index, 32))
        activation = ActivationSource({"fixture": hessian}, provenance)
        extra = activation.for_unit("fixture", 32, "cpu",
                                    scale_plane=wire_recipe(E4M3_GRID, 1024).scale_plane)
        blob = encode_linear(weight.float(), grid=E4M3_GRID, q256=1024,
                             name="fixture", verify=False, **extra).blob
        result.append((weight, hessian, blob))
    return provenance, result


def _canonical_handoff(root, hessians, provenance):
    """Use the public canonical reference contract; keep payloads per unit."""
    import hashlib
    from test_hessian_reference_capture import POLICY, SCHEMA, write_json

    root.mkdir()
    (root / "inputs").mkdir()
    counts = {name: 32 for name in hessians}
    shapes = {name: [32, 32] for name in hessians}
    census = root / "census.json"
    census_sha = write_json(census, dict(unit_shapes=shapes, counts=counts,
                                        max_abs={name: 1.0 for name in hessians}))
    entries = {}
    for index, (name, hessian) in enumerate(hessians.items()):
        path = root / "inputs" / f"unit-{index}.pt"
        torch.save(dict(inputs=torch.ones(2, 32), hessian=hessian, name=name,
                        source="tessera_campaign_prefix_f32_v1", count=32, max_abs=1.0), path)
        entries[name] = dict(path="inputs/" + path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    canonical = root / "capture_manifest.json"
    manifest_sha = write_json(canonical, dict(
        schema="prismaquant.tessera_calibration_cache.v2", status="complete",
        identity=dict(schema="prismaquant.tessera_calibration_cache.v2", census_sha256=census_sha,
                      units=shapes, storage_source="tessera_campaign_prefix_f32_v1", max_act_rows=2,
                      calibration={k: v for k, v in provenance.items() if k != "hessian_role"}),
        entries=entries))
    seal = ActivationSource(hessians, provenance).capture_sha256()
    handoff = root / "capture.references.json"
    write_json(handoff, dict(schema=SCHEMA,
        canonical_capture=dict(path=str(canonical), sha256=manifest_sha),
        census=dict(path=str(census), sha256=census_sha), provenance=provenance, counts=counts,
        hessians={name: _api().tensor_identity(h) for name, h in hessians.items()},
        capture_sha256=seal, rows=[dict(units=sorted(hessians), capture_sha256=seal)],
        load_policy=dict(POLICY)))
    return handoff


def _calibrated_packed_export(tmp_path, calibrated_wires, monkeypatch, layout):
    from safetensors.torch import save_file
    from tessera.serving_parts import source_identity

    api, exporter = _api(), _exporter()
    provenance, triples = calibrated_wires
    stack = "model.language_model.layers.1.mlp.experts"
    roles = ("gate_proj", "up_proj", "down_proj")
    dense = {f"model.language_model.layers.0.mlp.{role}.weight": triples[i][0].clone()
             for i, role in enumerate(roles)}
    tensors = dict(dense)
    gate, up, down = [torch.stack([triples[3 * expert + role][0] for expert in range(2)])
                      for role in range(3)]
    if layout == "out_first_chunked":
        tensors[stack + ".gate_up_proj.weight"] = torch.cat((gate, up), dim=1)
        tensors[stack + ".down_proj.weight"] = down
    else:
        packed = torch.empty(2, 32, 64, dtype=gate.dtype)
        packed[:, :, 0::2] = gate.transpose(1, 2)
        packed[:, :, 1::2] = up.transpose(1, 2)
        tensors[stack + ".gate_up_proj"] = packed
        tensors[stack + ".down_proj"] = down.transpose(1, 2).contiguous()
    config = dict(architectures=["Glm5NextForConditionalGeneration"],
                  text_config=dict(hidden_size=32, moe_intermediate_size=32,
                                   num_hidden_layers=2, n_routed_experts=2))
    choices = {name: dict(grid="E4M3", q256=1024) for name in dense}
    choices[stack] = dict(grid="E4M3", q256=1024, source_layout=layout)
    src, cache = tmp_path / "src", tmp_path / "cache"
    src.mkdir(); cache.mkdir()
    save_file(tensors, str(src / "model.safetensors"))
    (src / "config.json").write_text(json.dumps(config))
    units = exporter.project_expert_plan({k: list(v.shape) for k, v in tensors.items()},
        config, {stack: choices[stack]})["stacks"][stack]["units"]
    logical = {name.removesuffix(".weight"): triples[i] for i, name in enumerate(dense)}
    logical.update({unit["tensor"].removesuffix(".weight"): triples[3 * unit["expert"] + roles.index(unit["projection"])]
                    for unit in units})
    hessians = {name: triple[1] for name, triple in logical.items()}
    handoff = _canonical_handoff(tmp_path / "capture", hessians, provenance)
    activation = ActivationSource.from_capture(handoff)
    identities = [api.encoding_input_identity(weight, name, E4M3_GRID, 1024, activation=activation)
                  for name, weight in dense.items()]
    identities.extend(api.unit_input_identity(
        exporter.packed_expert_weight(tensors[unit["source_tensor"]], unit), unit,
        E4M3_GRID, 1024, activation=activation) for unit in units)
    activation.hessians.close()
    records = {}
    for index, identity in enumerate(identities):
        blob = logical[identity["unit"]][2]
        filename = f"unit-{index}.tessera"
        (cache / filename).write_bytes(blob)
        records[identity["unit"]] = api.make_unit_record(blob, identity, filename=filename)
    manifest = dict(schema=api.CACHE_SCHEMA, source=source_identity(src), units=records)
    manifest_path, plan_path = cache / "manifest.json", tmp_path / "plan.json"
    manifest_path.write_text(json.dumps(manifest))
    plan_path.write_text(json.dumps(choices))
    def forbidden(*args, **kwargs):
        raise AssertionError("cached calibrated export invoked the encoder")
    monkeypatch.setattr(exporter, "encode_linear_planes", forbidden)
    monkeypatch.setattr(exporter, "output_partitions",
                        lambda census, module: [32, 32] if module.endswith(".gate_up_proj") else [32])
    out = tmp_path / "out"
    argv = ["export", str(src), str(out), "--plan-json", str(plan_path),
            "--cached-units", str(manifest_path), "--hessian", str(handoff),
            "--device", "cpu", "--allow-unrouted", "--allow-unserveable"]
    monkeypatch.setattr("sys.argv", argv)
    return dict(exporter=exporter, argv=argv, src=src, out=out, manifest=manifest,
                manifest_path=manifest_path, hessians=hessians, handoff=handoff,
                provenance=provenance, logical=logical, units=units, tensors=tensors)


@pytest.mark.parametrize("layout", ["out_first_chunked", "in_first_interleaved"])
def test_calibrated_packed_cached_cli_preserves_originals(tmp_path, calibrated_wires, monkeypatch, layout):
    import weakref
    from safetensors import safe_open
    from tessera.hessian_capture import ReferenceHessians

    case = _calibrated_packed_export(tmp_path, calibrated_wires, monkeypatch, layout)
    original = ReferenceHessians.__getitem__
    returned, consumed = [], []
    def observed(self, key):
        assert all(ref() is None for ref in returned), "export retained an earlier H"
        result = original(self, key)
        assert self.receipt()["live_payloads"] == 0
        consumed.append(key)
        returned.append(weakref.ref(result))
        return result
    monkeypatch.setattr(ReferenceHessians, "__getitem__", observed)
    case["exporter"].main()
    assert sorted(consumed) == sorted(case["logical"])
    assert all(ref() is None for ref in returned)
    with safe_open(str(case["out"] / "model.safetensors"), framework="pt") as handle:
        members = [unit for name in handle.keys() if name.endswith((".wire", ".wire_bytes"))
                   for unit in parse_fused(handle.get_tensor(name).numpy().tobytes())]
        for unit in case["units"]:
            actual = parse_fused(handle.get_tensor(unit["wire"]).numpy().tobytes())[0].blob
            assert actual == case["logical"][unit["tensor"].removesuffix(".weight")][2]
    from collections import Counter
    assert Counter(unit.blob for unit in members) == Counter(v[2] for v in case["logical"].values())
    receipt = json.loads((case["out"] / "tessera_serving_manifest.json").read_text())
    assert receipt["cached_units"]["planned_units"] == 9
    assert all(role["cached_blob_sha256"] for module in receipt["modules"].values()
               for role in module["roles"])


@pytest.mark.parametrize("problem, error", [
    ("missing_h", "Hessian key"), ("physical_h", "Hessian key"),
    ("changed_h", "calibration"), ("wrong_h_shape", "Hessian shape"),
    ("wrong_h_owner", "calibration"), ("missing_calibration", "calibration"),
    ("changed_settings", "calibration"), ("changed_provenance", "calibration"),
    ("changed_source", "source"), ("wrong_source_slice", "projection"),
    ("changed_blob", "sha256"), ("wrong_recipe", "recipe"),
    ("changed_reference_payload", "checksum"), ("wrong_expert", "projection"),
    ("missing_dense", "coverage"), ("missing_expert", "coverage"),
])
def test_calibrated_packed_cached_cli_refuses_mismatched_inputs(
        tmp_path, calibrated_wires, monkeypatch, problem, error):
    from safetensors.torch import save_file
    from tessera.serving_parts import source_identity

    case = _calibrated_packed_export(tmp_path, calibrated_wires, monkeypatch, "out_first_chunked")
    unit = case["units"][0]
    name = unit["tensor"].removesuffix(".weight")
    record = case["manifest"]["units"][name]
    if problem in {"missing_h", "physical_h", "changed_h", "wrong_h_shape", "wrong_h_owner", "changed_provenance"}:
        hessians, provenance = dict(case["hessians"]), dict(case["provenance"])
        if problem == "missing_h": del hessians[name]
        elif problem == "physical_h": hessians[unit["source_tensor"]] = hessians.pop(name)
        elif problem == "changed_h": hessians[name] = hessians[name] * 2
        elif problem == "wrong_h_shape": hessians[name] = torch.eye(16)
        elif problem == "wrong_h_owner":
            other = case["units"][1]["tensor"].removesuffix(".weight")
            hessians[name], hessians[other] = hessians[other], hessians[name]
        else: provenance["fit_ids_sha256"] = "c" * 64
        path = tmp_path / "changed.pt"
        torch.save(dict(H=hessians, provenance=provenance), path)
        case["argv"][case["argv"].index("--hessian") + 1] = str(path)
    elif problem == "missing_calibration":
        index = case["argv"].index("--hessian"); del case["argv"][index:index + 2]
    elif problem == "changed_settings": case["argv"].extend(["--refit-metric-trailing", "h^0.5"])
    elif problem == "changed_source":
        case["tensors"][unit["source_tensor"]][0, 0, 0] += 1
        save_file(case["tensors"], str(case["src"] / "model.safetensors"))
        # Refresh the outer source seal to exercise the actual slice-byte check.
        case["manifest"]["source"] = source_identity(case["src"])
    elif problem == "wrong_source_slice": record["identity"]["projection"]["source_slice"]["selector"] = "second_half"
    elif problem == "changed_blob":
        path = case["manifest_path"].parent / record["file"]
        raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
    elif problem == "changed_reference_payload":
        canonical = json.loads((case["handoff"].parent / "capture_manifest.json").read_text())
        path = case["handoff"].parent / canonical["entries"][name]["path"]
        raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
    elif problem == "wrong_expert": record["identity"]["projection"]["source_slice"]["expert"] = 1
    elif problem == "wrong_recipe": record["identity"]["recipe"]["q256"] = 1280
    elif problem == "missing_dense": del case["manifest"]["units"][next(iter(case["logical"]))]
    elif problem == "missing_expert": del case["manifest"]["units"][name]
    case["manifest_path"].write_text(json.dumps(case["manifest"]))
    from tessera.errors import GrammarError
    exception = GrammarError if problem == "changed_reference_payload" else ValueError
    with pytest.raises(exception, match=error): case["exporter"].main()
    assert not (case["out"] / "tessera_serving_manifest.json").exists()


@pytest.mark.parametrize("cache_mode", ["uncached", "expert_only"])
def test_calibrated_packed_cli_requires_complete_cached_mode(tmp_path, calibrated_wires, monkeypatch, cache_mode):
    case = _calibrated_packed_export(tmp_path, calibrated_wires, monkeypatch, "out_first_chunked")
    index = case["argv"].index("--cached-units")
    if cache_mode == "uncached": del case["argv"][index:index + 2]
    else: case["argv"][index] = "--cached-expert-units"
    with pytest.raises(SystemExit, match="--hessian.*packed expert"): case["exporter"].main()
    assert not case["out"].exists()
