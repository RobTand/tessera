"""Exact campaign blobs cross the producer boundary without another encode."""
from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import hashlib
from pathlib import Path
import shutil

import pytest
import torch

from tessera.alphabet import E4M3_GRID
from tessera.export import ActivationSource, encode_linear
from tessera.fused import parse_fused
from tessera.historical_producer import load_historical_producer
from tessera.moe_execution import ResearchSelectedMoeConfig

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


def _distinct_producer(tmp_path, *, escape=False):
    """A real package with a changed serving-only file and the same wire owner."""
    package = tmp_path / "src/tessera"
    shutil.copytree(ROOT / "src/tessera", package,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    route = package / "serving/scheme.py"
    route.write_bytes(route.read_bytes() +
                      f"\n# historical package source seal: {tmp_path.name}\n".encode())
    if escape:
        cached = package / "cached_unit.py"
        cached.write_bytes(cached.read_bytes() + b"\nimport tessera.serving.scheme\n")
    digest = hashlib.sha256()
    for path in sorted(p for p in package.rglob("*")
                       if p.suffix in {".py", ".cu", ".cuh", ".cpp", ".h"}):
        digest.update(path.relative_to(package).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return package, digest.hexdigest()


def test_sealed_historical_identity_accepts_original_without_relabelling(tmp_path, encoded):
    exporter = _exporter()
    package, source_sha256 = _distinct_producer(tmp_path)
    producer = load_historical_producer(package, source_sha256)
    identity = exporter.cached_input_identity(producer, encoded[0], TENSOR,
                                              _projection(), E4M3_GRID, 1024)
    record = {"file": "original.tessera", "identity": identity,
              "blob_bytes": len(encoded[1]),
              "blob_sha256": hashlib.sha256(encoded[1]).hexdigest()}
    assert identity["encoder_source_sha256"] == source_sha256
    assert identity["encoder_source_sha256"] != _api().encoder_source_sha256()
    assert producer.verify(encoded[1], record, identity).blob == encoded[1]
    accepted, packaged = exporter.pack_cached_expert_unit(encoded[1], record, identity)
    assert accepted.blob == parse_fused(packaged)[0].blob == encoded[1]
    assert producer.namespace in __import__("sys").modules
    assert producer.namespace + ".serving.scheme" not in __import__("sys").modules

    wrong = encoded[0].clone(); wrong[0, 0] += 1
    with pytest.raises(ValueError, match="source"):
        producer.verify(encoded[1], record, exporter.cached_input_identity(
            producer, wrong, TENSOR, _projection(), E4M3_GRID, 1024))
    provenance = {"text_sha256": "a" * 64, "fit_ids_sha256": "b" * 64, "fit_tokens": 32}
    first = ActivationSource({UNIT: torch.eye(32)}, provenance)
    for changed in (ActivationSource({UNIT: torch.eye(32) * 2}, provenance),
                    ActivationSource({UNIT: torch.eye(32)}, provenance,
                                     refit_objective_trailing="h^0.5")):
        historical = exporter.cached_input_identity(producer, encoded[0], TENSOR,
            _projection(), E4M3_GRID, 1024, activation=first)
        modified = exporter.cached_input_identity(producer, encoded[0], TENSOR,
            _projection(), E4M3_GRID, 1024, activation=changed)
        assert historical["calibration"] != modified["calibration"]
        with pytest.raises(ValueError, match="calibration"):
            producer.verify(encoded[1], {**record, "identity": historical}, modified)
    changed_wire = bytearray(encoded[1]); changed_wire[-1] ^= 1
    with pytest.raises(ValueError, match="sha256"):
        producer.verify(bytes(changed_wire), record, identity)
    (package / "serving/scheme.py").write_text("changed again")
    with pytest.raises(ValueError, match="SHA256"):
        load_historical_producer(package, source_sha256)

    current_gate = ResearchSelectedMoeConfig(max_experts_per_chunk=2)
    with pytest.raises(ValueError, match="no selected decoder"):
        current_gate.require_wire_recipe(grid="E2M1x2", q256=896,
            body="TCQ", plane="LUT", span=2, target=STACK)


def test_historical_namespace_refuses_serving_and_absolute_current_import(tmp_path):
    package, source_sha256 = _distinct_producer(tmp_path)
    producer = load_historical_producer(package, source_sha256)
    with pytest.raises(ImportError, match="cannot import serving"):
        importlib.import_module(producer.namespace + ".serving.scheme")
    escaped = tmp_path / "escaped"
    escaped.mkdir()
    altered_package, altered_sha256 = _distinct_producer(escaped, escape=True)
    with pytest.raises(ImportError, match="escape into the current producer"):
        load_historical_producer(altered_package, altered_sha256)


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


def test_a_partition_stamp_is_proved_against_the_bundle_whole_source(tmp_path, encoded):
    """tessera#495: a part hashed only its shards; the bundle vouches for each."""
    from safetensors.torch import save_file
    from tessera.serving_parts import source_identity, source_part_identity

    api = _api()
    src = tmp_path / "src"
    src.mkdir()
    first, second = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    save_file({TENSOR: encoded[0]}, str(src / first))
    save_file({"lm_head.weight": encoded[0]}, str(src / second))
    (src / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {TENSOR: first, "lm_head.weight": second}}))
    (src / "config.json").write_text(json.dumps({"architectures": ["Example"]}))
    record, _ = _record(encoded)
    manifest = {"schema": api.CACHE_SCHEMA, "source": source_identity(src), "units": {UNIT: record}}
    stamp = source_part_identity(src, [first])
    api.CachedUnitBundle(manifest, tmp_path, {UNIT}, stamp)
    cases = {"digest": lambda s: s["files"].update({first: "0" * 64}),
             "absent": lambda s: s["files"].update({"model-00009.safetensors": "0" * 64}),
             "tensors": lambda s: s["tensors"].pop("lm_head.weight"),
             "config": lambda s: s.update(config_sha256="0" * 64),
             "auxiliary": lambda s: s.update(auxiliary_sha256={})}
    for name, mutate in cases.items():
        bad = copy.deepcopy(stamp)
        mutate(bad)
        with pytest.raises(ValueError, match="cached unit bundle: source identity"):
            api.CachedUnitBundle(manifest, tmp_path, {UNIT}, bad)
    partial = dict(manifest, source={k: v for k, v in stamp.items() if k != "schema"})
    with pytest.raises(ValueError, match="not a whole-checkpoint identity"):
        api.CachedUnitBundle(partial, tmp_path, {UNIT}, stamp)
    with pytest.raises(ValueError, match="source checkpoint identity mismatch"):
        api.CachedUnitBundle(manifest, tmp_path, {UNIT}, source_identity(src) | {"files": {}})


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


@pytest.mark.parametrize("historical", [False, True])
def test_export_consumes_complete_bundle_without_encoder(tmp_path, encoded, monkeypatch, historical):
    from safetensors import safe_open
    from safetensors.torch import save_file
    from tessera.serving_parts import source_identity

    api, exporter = _api(), _exporter()
    old_flags = []
    if historical:
        producer_root = tmp_path / "original"
        producer_root.mkdir()
        package, source_sha256 = _distinct_producer(producer_root)
        api = load_historical_producer(package, source_sha256)
        old_flags = ["--cached-producer-package", str(package),
                     "--cached-producer-source-sha256", source_sha256]
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
        identity = exporter.cached_input_identity(api if historical else None,
            tensors[unit["source_tensor"]], unit["tensor"], unit, E4M3_GRID, 1024)
        filename = f"unit-{index}.tessera"
        (cache / filename).write_bytes(encoded[1])
        records[identity["unit"]] = ({"file": filename, "blob_bytes": len(encoded[1]),
            "blob_sha256": hashlib.sha256(encoded[1]).hexdigest(), "identity": identity}
            if historical else api.make_unit_record(encoded[1], identity, filename=filename))
    manifest = {"schema": _api().CACHE_SCHEMA, "source": source_identity(src), "units": records}
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
                                    "--allow-unrouted", "--allow-unserveable", *old_flags])
    exporter.main()
    with safe_open(str(out / "model.safetensors"), framework="pt") as handle:
        for name in tensors:
            packed = bytes(handle.get_tensor(name.removesuffix(".weight") + ".wire").tolist())
            assert parse_fused(packed)[0].blob == encoded[1]
    receipt = json.loads((out / "tessera_serving_manifest.json").read_text())
    assert receipt["cached_expert_units"]["planned_units"] == 3
    if historical:
        assert receipt["cached_expert_units"]["historical_producer"]["source_sha256"] == source_sha256
    assert len(receipt["modules"][STACK]["roles"]) == 3
    assert all(role["cached_blob_sha256"] for role in receipt["modules"][STACK]["roles"])


@pytest.mark.parametrize("partition", [False, True])
@pytest.mark.parametrize("include_experts", [False, True])
@pytest.mark.parametrize("omit_dense", [False, True])
def test_complete_cached_export_preserves_dense_and_expert_originals(
        tmp_path, encoded, monkeypatch, include_experts, omit_dense, partition):
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
        "--cached-units", str(manifest_path), "--device", "cpu", "--allow-unrouted", "--allow-unserveable",
        # A one-part partition stamps a source-part block the bundle must prove
        # against its whole-checkpoint source (tessera#495).
        *(["--partition", "0/1", "--partition-runtime-image", "test/image@sha256:" + "b" * 64]
          if partition else [])])
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


def _calibrated_packed_export(tmp_path, calibrated_wires, monkeypatch, layout, *, historical=False):
    from safetensors.torch import save_file
    from tessera.serving_parts import source_identity

    api, exporter = _api(), _exporter()
    provenance, triples = calibrated_wires
    producer, producer_flags = None, []
    if historical:
        # Receipts written by a distinct, source-sealed producer package: the
        # export must derive every identity field under that producer.
        # A package name unique to this call: the historical namespace is
        # keyed by the package seal, which ``_distinct_producer`` derives from
        # the directory name, and one process may load several.
        original = tmp_path / ("original-" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12])
        original.mkdir()
        package, source_sha256 = _distinct_producer(original)
        producer = load_historical_producer(package, source_sha256)
        producer_flags = ["--cached-producer-package", str(package),
                          "--cached-producer-source-sha256", source_sha256]
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
    identities = [exporter.cached_input_identity(producer, weight, name, None, E4M3_GRID, 1024,
                                                 activation=activation)
                  for name, weight in dense.items()]
    identities.extend(exporter.cached_input_identity(
        producer, exporter.packed_expert_weight(tensors[unit["source_tensor"]], unit),
        unit["tensor"], unit, E4M3_GRID, 1024, activation=activation) for unit in units)
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
            "--device", "cpu", "--allow-unrouted", "--allow-unserveable", *producer_flags]
    monkeypatch.setattr("sys.argv", argv)
    return dict(exporter=exporter, argv=argv, src=src, out=out, manifest=manifest,
                manifest_path=manifest_path, hessians=hessians, handoff=handoff,
                provenance=provenance, logical=logical, units=units, tensors=tensors,
                producer=producer)


def _observe_h_consumption(monkeypatch):
    """Record every H the export reads through the reference owner."""
    import weakref
    from tessera.hessian_capture import ReferenceHessians

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
    return consumed, returned


@pytest.mark.parametrize("identity_mode", ["digested", "committed"])
@pytest.mark.parametrize("layout", ["out_first_chunked", "in_first_interleaved"])
def test_calibrated_packed_cached_cli_preserves_originals(tmp_path, calibrated_wires, monkeypatch,
                                                          layout, identity_mode):
    from safetensors import safe_open

    case = _calibrated_packed_export(tmp_path, calibrated_wires, monkeypatch, layout)
    case["argv"].extend(["--cached-hessian-identity", identity_mode])
    consumed, returned = _observe_h_consumption(monkeypatch)
    case["exporter"].main()
    receipt = json.loads((case["out"] / "tessera_serving_manifest.json").read_text())
    established = receipt["cached_units"]["hessian_identity"]
    assert established["established"] == identity_mode
    if identity_mode == "digested":
        # Every unit's H read and digested: the path before tessera#497.
        assert sorted(consumed) == sorted(case["logical"])
        assert established["witness"] is None and established["committed_units_served"] is None
    else:
        # One witness H read; every unit's identity taken from the commitment.
        assert consumed == [established["witness"]["unit"]] and established["witness"]["agreed"]
        assert established["committed_units_served"] == len(case["logical"])
        assert established["reference"]["document_sha256"] == hashlib.sha256(
            case["handoff"].read_bytes()).hexdigest()
        assert established["reference"]["capture_sha256"] == receipt["activation_aware"]["hessian"]["capture_sha256"]
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
        # The digested path reads every H payload and refuses the flipped
        # byte; the committed path consumes no unwitnessed payload by design
        # (test_committed_intake_consumes_only_the_witness_payload).
        case["argv"].extend(["--cached-hessian-identity", "digested"])
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


def _identity_run(tmp_path, calibrated_wires, monkeypatch, label, flags, *, historical=False):
    """One calibrated cached export under ``flags``, with the H reads it made."""
    root = tmp_path / label
    root.mkdir()
    case = _calibrated_packed_export(root, calibrated_wires, monkeypatch, "out_first_chunked",
                                     historical=historical)
    case["argv"].extend(flags)
    consumed, _returned = _observe_h_consumption(monkeypatch)
    case["exporter"].main()
    # A snapshot: a later run's observer wraps this one and keeps appending.
    case["consumed"] = list(consumed)
    case["receipt"] = json.loads((case["out"] / "tessera_serving_manifest.json").read_text())
    return case


@pytest.mark.parametrize("historical", [False, True])
def test_committed_parallel_intake_is_byte_identical_to_digested_serial(
        tmp_path, calibrated_wires, monkeypatch, historical):
    """(a) Same bytes, same acceptance: threads and the H identity source change nothing written."""
    serial = _identity_run(tmp_path, calibrated_wires, monkeypatch, "serial",
                           ["--cached-hessian-identity", "digested", "--cached-intake-threads", "1"],
                           historical=historical)
    parallel = _identity_run(tmp_path, calibrated_wires, monkeypatch, "parallel",
                             ["--cached-hessian-identity", "committed", "--cached-intake-threads", "4"],
                             historical=historical)
    assert (serial["out"] / "model.safetensors").read_bytes() == (parallel["out"] / "model.safetensors").read_bytes()
    for key in ("modules", "totals", "plan", "default", "serving_gate"):
        assert serial["receipt"][key] == parallel["receipt"][key], key
    if historical:
        # Each run sealed its own distinct producer package, so the two cache
        # manifests differ in exactly that seal; the accepted bytes do not.
        for run in (serial, parallel):
            assert run["receipt"]["cached_units"]["historical_producer"]["source_sha256"] == \
                run["producer"].source_sha256
            assert all(record["identity"]["encoder_source_sha256"] == run["producer"].source_sha256
                       for record in run["manifest"]["units"].values())
    else:
        assert serial["receipt"]["cached_units"]["manifest_sha256"] == \
            parallel["receipt"]["cached_units"]["manifest_sha256"]
    # The serial digested run read every H; the parallel committed run read one.
    assert sorted(serial["consumed"]) == sorted(serial["logical"])
    witness = parallel["receipt"]["cached_units"]["hessian_identity"]["witness"]
    assert parallel["consumed"] == [witness["unit"]] and witness["agreed"] is True
    intake = parallel["receipt"]["cached_units"]["intake"]
    assert intake["threads"] == 4 and intake["units"] == len(parallel["units"])
    assert 1 <= intake["peak_in_flight_units"] <= intake["window_units"]
    assert serial["receipt"]["cached_units"]["intake"]["threads"] == 1


def test_committed_identity_equals_full_derivation_for_every_unit(tmp_path, calibrated_wires):
    """The deriver's committed form is the producer's own digested form, field for field."""
    from tessera.cached_unit import CachedUnitIdentity
    from tessera.errors import GrammarError

    provenance, triples = calibrated_wires
    exporter = _exporter()
    hessians = {f"unit{i}": triple[1] for i, triple in enumerate(triples)}
    handoff = _canonical_handoff(tmp_path / "capture", hessians, provenance)
    derive = lambda weight, unit_name, unit, grid, q256, *, activation: exporter.cached_input_identity(
        None, weight, unit_name, unit, grid, q256, activation=activation)
    digested = CachedUnitIdentity(derive, ActivationSource.from_capture(handoff), mode="digested")
    committed = CachedUnitIdentity(derive, ActivationSource.from_capture(handoff), mode="committed")
    assert (digested.established, committed.established) == ("digested", "committed")
    for i, (weight, _hessian, _blob) in enumerate(triples):
        assert committed(weight, f"unit{i}.weight", None, E4M3_GRID, 1024) == \
            digested(weight, f"unit{i}.weight", None, E4M3_GRID, 1024)
    assert digested.activation.hessians.receipt()["verified_units"] == sorted(hessians)
    assert committed.activation.hessians.receipt()["verified_units"] == ["unit0"]
    assert committed.record()["committed_units_served"] == len(triples)
    with pytest.raises(ValueError, match="no exact Hessian key"):
        committed(triples[0][0], "absent.weight", None, E4M3_GRID, 1024)
    with pytest.raises(ValueError, match="Hessian shape"):
        committed(torch.ones(32, 16, dtype=torch.bfloat16), "unit0.weight", None, E4M3_GRID, 1024)
    # A plain mapping has no commitments, whatever was asked; no activation, no calibration.
    plain = CachedUnitIdentity(derive, ActivationSource(hessians, provenance), mode="committed")
    assert plain.established == "digested" and plain.record()["reference"] is None
    none = CachedUnitIdentity(derive, None, mode="committed")
    assert none.established is None
    assert none(triples[0][0], "unit0.weight", None, E4M3_GRID, 1024)["calibration"] is None
    # The witness must agree or nothing is served from commitments.
    disagreeing = CachedUnitIdentity(derive, ActivationSource.from_capture(handoff), mode="committed")
    disagreeing._derive = lambda weight, unit_name, unit, grid, q256, *, activation: dict(
        derive(weight, unit_name, unit, grid, q256, activation=activation),
        **({"extra": True} if activation is not None else {}))
    with pytest.raises(GrammarError, match="committed Hessian identity disagrees"):
        disagreeing(triples[0][0], "unit0.weight", None, E4M3_GRID, 1024)
    assert disagreeing.record()["witness"] is None
    for deriver in (digested, committed, disagreeing):
        deriver.activation.hessians.close()


def test_cache_record_h_identity_disagreeing_with_commitment_refuses(tmp_path, calibrated_wires, monkeypatch):
    """(b) A receipt whose calibration.hessian is not the commitment refuses; the driver decides."""
    from tessera.hessian_capture import ReferenceHessians

    # Not the first task, so the witness derivation itself stays honest.
    tampered = case_unit = None
    for label in ("refuses", "driver_mutated"):
        root = tmp_path / label
        root.mkdir()
        case = _calibrated_packed_export(root, calibrated_wires, monkeypatch, "out_first_chunked")
        case_unit = case["units"][-1]["tensor"].removesuffix(".weight")
        record = case["manifest"]["units"][case_unit]
        record["identity"]["calibration"]["hessian"]["sha256"] = "0" * 64
        tampered = record["identity"]["calibration"]["hessian"]
        case["manifest_path"].write_text(json.dumps(case["manifest"]))
        if label == "refuses":
            with pytest.raises(ValueError, match="calibration"):
                case["exporter"].main()
            assert not (case["out"] / "tessera_serving_manifest.json").exists()
            continue
        # Mutate the DRIVER: a commitment that echoes the receipt makes the
        # check vacuous, and the export accepts -- proving the refusal above
        # is decided by ``ReferenceHessians.commitment``, not by the fixture.
        original = ReferenceHessians.commitment
        monkeypatch.setattr(ReferenceHessians, "commitment", lambda self, name:
                            dict(tampered) if name == case_unit else original(self, name))
        case["exporter"].main()
        receipt = json.loads((case["out"] / "tessera_serving_manifest.json").read_text())
        assert receipt["cached_units"]["hessian_identity"]["established"] == "committed"


@pytest.mark.parametrize("reseal", ["none", "capture_only", "capture_and_rows"])
def test_tampered_reference_commitment_refuses(tmp_path, calibrated_wires, monkeypatch, reseal):
    """(c) A commitment edited in the document refuses at load, or at the unit once resealed."""
    import tessera.hessian_capture as hessian_capture
    from tessera.errors import GrammarError
    from tessera.hessian_capture import capture_sha256_from_units

    case = _calibrated_packed_export(tmp_path, calibrated_wires, monkeypatch, "out_first_chunked")
    name = case["units"][-1]["tensor"].removesuffix(".weight")
    payload = json.loads(case["handoff"].read_text())
    payload["hessians"][name]["sha256"] = "0" * 64
    forged = capture_sha256_from_units(payload["provenance"],
                                       {n: v["sha256"] for n, v in payload["hessians"].items()})
    if reseal != "none":
        payload["capture_sha256"] = forged
    if reseal == "capture_and_rows":
        payload["rows"][0]["capture_sha256"] = forged
    case["handoff"].write_text(json.dumps(payload, sort_keys=True))
    if reseal == "none":
        with pytest.raises(GrammarError, match="disagree with the capture seal"):
            case["exporter"].main()
    elif reseal == "capture_only":
        with pytest.raises(GrammarError, match="row commitments disagree"):
            case["exporter"].main()
    else:
        # Fully resealed, the document loads; the receipt's H identity is
        # then not the (forged) commitment and the unit refuses.
        with pytest.raises(ValueError, match="calibration"):
            case["exporter"].main()
    assert not case["out"].exists() or not (case["out"] / "tessera_serving_manifest.json").exists()
    if reseal != "none":
        return
    # Mutate the DRIVER: a seal check that echoes the document lets the forged
    # commitment load -- so the load refusal above is that check's -- and the
    # unit still refuses against the receipt, the second line.
    monkeypatch.setattr(hessian_capture, "capture_sha256_from_units",
                        lambda provenance, units: payload["capture_sha256"])
    with pytest.raises(ValueError, match="calibration"):
        case["exporter"].main()


@pytest.mark.parametrize("flipped", ["witness", "other"])
def test_committed_intake_consumes_only_the_witness_payload(tmp_path, calibrated_wires, monkeypatch, flipped):
    """Committed intake reads one canonical payload; the rest are commitments, and the receipt says so."""
    from tessera.errors import GrammarError

    first = _identity_run(tmp_path, calibrated_wires, monkeypatch, "first", [])
    witness = first["receipt"]["cached_units"]["hessian_identity"]["witness"]["unit"]
    assert first["consumed"] == [witness]
    other = next(n for n in sorted(first["logical"]) if n != witness)
    root = tmp_path / "second"
    root.mkdir()
    case = _calibrated_packed_export(root, calibrated_wires, monkeypatch, "out_first_chunked")
    target = witness if flipped == "witness" else other
    canonical = json.loads((case["handoff"].parent / "capture_manifest.json").read_text())
    path = case["handoff"].parent / canonical["entries"][target]["path"]
    raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
    consumed, _returned = _observe_h_consumption(monkeypatch)
    if flipped == "witness":
        with pytest.raises(GrammarError, match="checksum"):
            case["exporter"].main()
        return
    case["exporter"].main()
    assert consumed == [witness]
    receipt = json.loads((case["out"] / "tessera_serving_manifest.json").read_text())
    established = receipt["cached_units"]["hessian_identity"]
    assert established["established"] == "committed"
    assert established["committed_units_served"] == len(case["logical"])
    assert (case["out"] / "model.safetensors").read_bytes() == (first["out"] / "model.safetensors").read_bytes()
