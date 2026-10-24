"""Acceptance for the retained issue-1388 measurement, NOT a GPU launcher.

The implementation supplies experiments/results/e2m1_dense_scalar_census/:
  census.json: unedited tools/tessera_route_census.py output
  config.json, tessera_serving_manifest.json: exact measured sidecars
  provenance.json: source/import, launch/build, export and reuse records below
  stdout.log, stderr.log: raw execution logs (stderr may be empty)
  checksums.json: relative filename -> SHA256, covering all these payloads and
                 any additional raw observations/build/export records referenced.
The note is docs/measurements/current-master-e2m1-dense-scalar-census.md.

An unavailable executor/refusal must be retained and reported, but must NOT make
these successful-census acceptance checks pass. No fake receipt, cell promotion,
package installation, historical reconstruction or repeated serving is a fix.
These CPU checks validate persisted evidence; they do not themselves attest GPU
execution. Review must inspect the raw run custody and the attributed claims.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "experiments/results/e2m1_dense_scalar_census"
NOTE = ROOT / "docs/measurements/current-master-e2m1-dense-scalar-census.md"
FAMILY = "TESSERA_E2M1_K2"
DECODER = "native_fused_window_dense_e2m1"
SYMBOL = "tessera::fused_window_dense_e2m1"
SOURCE_FILES = (
    "src/tessera/serving/nvfp4_route.py",
    "src/tessera/routed_fused_e2m1.py",
    "src/tessera/serving/csrc/routed_fused_window.cu",
    "src/tessera/serving/runtime_contract.json",
)


def _json(path):
    assert path.is_file(), (
        f"Missing retained issue-1388 evidence: {path.relative_to(ROOT)}; "
        "source reachability and historical receipts are not a served census"
    )
    return json.loads(path.read_text())


def _tool():
    spec = importlib.util.spec_from_file_location(
        "issue1388_route_census", ROOT / "tools/tessera_route_census.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evidence():
    return (_json(BUNDLE / "census.json"),
            _json(BUNDLE / "provenance.json"),
            _json(BUNDLE / "config.json"))


def test_retained_bundle_binds_source_build_artifact_and_execution():
    census, provenance, config = _evidence()
    hashes = _json(BUNDLE / "checksums.json")
    required = {"census.json", "provenance.json", "config.json",
                "tessera_serving_manifest.json", "stdout.log", "stderr.log"}
    assert required <= set(hashes)
    for name, digest in hashes.items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        payload = BUNDLE / relative
        assert payload.is_file(), f"Missing checksummed payload: {name}"
        assert hashlib.sha256(payload.read_bytes()).hexdigest() == digest, name
    assert (BUNDLE / "stdout.log").stat().st_size > 0

    source = provenance["source"]
    assert re.fullmatch(r"[0-9a-f]{40}", source["commit"])
    assert census["tessera_commit"] == source["commit"]
    assert source["identity"] and source["import_origin"]
    for path in SOURCE_FILES:
        assert source["files_sha256"][path] == hashlib.sha256(
            (ROOT / path).read_bytes()).hexdigest(), path
    # These are references to retained raw capture files, not free-form claims.
    for key in ("source_capture", "native_build_capture", "export_capture",
                "runtime_capture", "reuse_capture"):
        assert provenance[key] in hashes
        assert (BUNDLE / provenance[key]).stat().st_size > 0
    assert provenance["native_build"]["library_sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}",
                        provenance["native_build"]["library_sha256"])
    assert provenance["native_build"]["identity"] == census["kernel_build"]
    assert census["kernel_build"]
    for version in ("python", "torch", "cuda", "vllm"):
        assert provenance["runtime_versions"][version]
    assert census["device"] and census["platform"]
    assert re.fullmatch(r".+@sha256:[0-9a-f]{64}", census["runtime_image"])
    assert census["runtime_image_declaration"]
    assert provenance["observed_execution_mode"] == census["execution_mode"] == "eager"
    assert provenance["observed_residency"] == census["mode"] == "resident"
    assert census["tensor_parallel_size"] == 1
    assert census["topology"]["world_size"] == 1
    assert census["topology"]["all_ranks_reported"] is True
    assert census["served_config_fingerprint"]["config.json"] == hashes["config.json"]
    assert census["served_config_fingerprint"]["tessera_serving_manifest.json"] == hashes["tessera_serving_manifest.json"]
    assert provenance["artifact_files_sha256"]  # checkpoint tensor-byte inventory
    for name, digest in provenance["artifact_files_sha256"].items():
        assert name and re.fullmatch(r"[0-9a-f]{64}", digest)

    command = provenance["census_argv"]
    assert isinstance(command, list) and "--require-decoder" in command
    assert command[command.index("--require-decoder") + 1] == DECODER
    assert "--expect-modules" in command
    assert int(command[command.index("--expect-modules") + 1]) == len(census["declared_targets"])
    assert not {"--compiled", "--allow-fallback-decoder", "--draft-routes"} & set(command)
    assert "tessera_route_census.py" in " ".join(command)
    assert config["quantization_config"]["config_groups"]


def test_every_declared_dense_module_executes_replacement_pair_in_both_phases():
    census, provenance, config = _evidence()
    tool = _tool()
    # Independent declaration from the measured export, NOT len(observed records).
    declared = {}
    for group in config["quantization_config"]["config_groups"].values():
        scheme = group["scheme"]
        assert scheme["family"] == FAMILY
        assert scheme.get("structure", "dense") == "dense"
        rung = tool.declared_rung(scheme)
        assert rung is not None and 128 <= rung < 1024  # one ordinary, not terminal, rung
        for target in group["targets"]:
            assert target not in declared
            declared[target] = rung
    assert declared, "A zero-module census cannot decide whether the fusion executes"
    assert len(set(declared.values())) == 1
    # Source-to-runtime mapping is retained by the export/launch capture. Identity
    # mapping is explicit too. Legitimate packed projections may share a runtime
    # module; retain that mapping rather than deriving the roster from telemetry.
    mapping = provenance["source_to_runtime_modules"]
    assert set(mapping) == set(declared)
    assert all(isinstance(n, str) and n for n in mapping.values())
    assert set(census["declared_targets"]) == set(mapping.values())
    assert set(census["declared_targets"].values()) == {FAMILY}
    assert census["declared_rungs_q256"] == {mapping[n]: r for n, r in declared.items()}
    assert census["served"] is True and census["problems"] == []
    assert census["execution_mode"] == "eager"
    assert census["phase_regimes"] == {"prefill": "batch", "decode": "decode"}
    assert len(census["ranks"]) == 1
    rank = census["ranks"][0]
    assert rank["lane_refusals"] == {}
    assert rank["records"] == census["records"]
    records = {}
    for phase in ("prefill", "decode"):
        records[phase] = {n: r for n, r in census["records"][phase].items()
                          if str(r.get("policy", "")).startswith("TESSERA_")}
        assert set(records[phase]) == set(mapping.values())
        for name, record in records[phase].items():
            assert record["kind"] == "dense" and record["state"] == "served", (phase, name, record)
            assert record["policy"] == f"{FAMILY}:resident"
            assert (record["symbol"], record["decoder"]) == (SYMBOL, DECODER), (phase, name, record)
            m, n, k = tool.parse_eager_shape(record["shape"])
            assert n > 0 and k > 0
            assert m > 1 if phase == "prefill" else m == 1
    coverage, problems = tool.required_decoder_coverage(records, [DECODER])
    assert problems == []
    assert coverage["all_required_covered"] is True
    assert census["required_decoder_coverage"] == coverage


def test_recipe_and_measurement_note_do_not_promote_qualification():
    census, provenance, _ = _evidence()
    from tessera.alphabet import E2M1_GRID, tuple_grid
    from tessera.export import served_recipe

    rung = next(iter(census["declared_rungs_q256"].values()))
    expected = served_recipe(tuple_grid(E2M1_GRID, 2, "coset"), rung, "dense")
    assert provenance["served_recipe"] == expected.to_config()
    # Recompute cell agreement from the owner; unattested must not be edited to
    # covered merely because route telemetry saw a launch. Activation eligibility
    # is separate evidence and remains subject to the unchanged contract.
    from tessera.serving.contract import load_serving_contract, PAYLOAD_FAMILY_BY_ROUTE

    agreement, _ = _tool().all_structure_agreement(
        census["records"], cells=load_serving_contract()["lane_eligibility"]["cells"],
        phase_regimes=census["phase_regimes"], platform=census["platform"],
        declared_rungs=census["declared_rungs_q256"],
        record_owners={p: {n: n for n in census["declared_targets"]}
                       for p in ("prefill", "decode")},
        families_by_route=PAYLOAD_FAMILY_BY_ROUTE,
        runtime_image=census["runtime_image"], execution_mode="eager",
        kernel_build=census["kernel_build"])
    assert census["cell_launch_agreement"] == agreement
    assert NOTE.is_file(), "Publish a new measurement note, not an amended historical campaign"
    note = NOTE.read_text()
    for fact in (DECODER, SYMBOL, provenance["source"]["commit"],
                 "1380", "1367", "unattested", "activation", "prefill", "decode"):
        assert fact.lower() in note.lower(), fact
    for issue in ("1380", "1367"):
        assert provenance["reuse_check"][issue], "Record exact-source/exact-tuple reuse disposition"
    assert provenance["activation_attestation"]  # raw result + limitation, NOT a required pass
