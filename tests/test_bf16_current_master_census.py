"""Retained-evidence acceptance for the current raw-BF16 dense served census.

These CPU checks validate a *real*, unedited census packet, not synthetic dispatch.
Keep census.json, config.json, census.log and provenance.json in the result folder
below. provenance.json records source_commit, source_files (repo-relative path ->
sha256), retained_files (folder-relative path -> sha256), imports, native_build,
artifact, runtime and launch_configuration. Richer provenance belongs in the note.
A blocked or substituted run must be retained/reported but cannot pass these tests.
No CUDA execution, historical git objects or shared-mount reads are needed here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "experiments/results/bf16_dense_current_master_census"
NOTE = ROOT / "docs/measurements/bf16-dense-current-master-census.md"
PAIR = ("tessera::fused_window_dense", "native_fused_window_dense_bf16")
SOURCE_FILES = (
    "src/tessera/serving/bf16_route.py",
    "src/tessera/serving/native_window.py",
    "src/tessera/routed_fused.py",
    "src/tessera/window_gemm.py",
    "src/tessera/serving/csrc/routed_fused_window.cu",
    "src/tessera/serving/scheme.py",
    "tools/tessera_route_census.py",
)


def _json(name):
    path = PACKET / name
    assert path.is_file(), f"Missing current-source actual served evidence: {path.name}"
    return json.loads(path.read_text())


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_raw_bf16_fused_owner_in_both_eager_phases():
    receipt = _json("census.json")
    config = _json("config.json")
    # Use artifact declarations, not a caller's assertion that zero is complete.
    groups = config["quantization_config"]["config_groups"]
    groups = [g for g in groups.values()
              if g.get("scheme", {}).get("family", "").startswith("TESSERA_")]
    assert groups
    assert all(g["scheme"]["family"] == "TESSERA_BF16" for g in groups)
    targets = {t for g in groups for t in g["targets"]}
    assert targets
    mapping = receipt.get("declared_name_mapping") or {}
    expected = {mapping.get(t, t) for t in targets}
    assert len(expected) == len(targets)
    assert _json("provenance.json")["expect_modules"] == len(expected)
    assert receipt["verdict"] == "served", receipt.get("problems")
    assert receipt["problems"] == []
    assert receipt["compiled"] is False
    assert receipt["runtime"]["execution_mode"] == "eager"
    # The tool deliberately omits topology/ranks at its single-rank default.
    assert receipt.get("topology", {}).get("observed_world_size", 1) == 1
    assert not receipt.get("ranks")
    assert receipt["env"]["TESSERA_SERVE_MODE"] == "resident"
    assert PAIR[1] in receipt["decoder_coverage"]["required"]
    assert receipt["checkpoint_sidecars"]["config.json"] == _sha(PACKET / "config.json")
    for phase in ("prefill", "decode"):
        records = receipt["records"][phase]
        assert set(records) == expected, (phase, set(records), expected)
        assert receipt["histogram"][phase]["tessera_modules"] == len(expected)
        for name, record in records.items():
            assert record["state"] == "served", (phase, name, record)
            assert record["kind"] == "dense"
            assert (record["symbol"], record["decoder"]) == PAIR, (phase, name, record)
            assert record["reason"] is None
            assert record["shape"].startswith("M") and "M*" not in record["shape"]
    # A route census alone does not earn a previously withdrawn cell.
    assert receipt["cell_launch_agreement"]["agrees"] is None


def test_packet_is_source_bound_and_raw_evidence_is_checksummed():
    provenance = _json("provenance.json")
    receipt = _json("census.json")
    assert re.fullmatch(r"[0-9a-f]{40}", provenance["source_commit"])
    assert receipt["versions"]["tessera_commit"] == provenance["source_commit"]
    for path in SOURCE_FILES:
        assert provenance["source_files"][path] == _sha(ROOT / path), path
    for name in ("census.json", "config.json", "census.log"):
        path = PACKET / name
        assert path.is_file() and path.stat().st_size > 0, name
        assert provenance["retained_files"][name] == _sha(path), name
    # Presence/digest checks are intentionally not proof that these were observed;
    # the reviewer must join them to the raw execution/launcher/build records.
    for field in ("imports", "native_build", "artifact", "runtime", "launch_configuration"):
        assert provenance[field], f"Missing observed identity: {field}"
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["native_build"]["binary_sha256"])
    assert re.search(r"@sha256:[0-9a-f]{64}$", receipt["runtime"]["image"])
    assert NOTE.is_file(), "Missing source trace, reuse decision, GPU plan and measurement limits"
    assert NOTE.read_text().strip()
