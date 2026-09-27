"""Serving JSON layout changes no fields; artifact seals bind written bytes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.full_engine_artifact import read_tessera_artifact
from tessera import serving_parts


def test_compact_writer_round_trips_in_insertion_order_and_binds_disk_bytes(tmp_path):
    # Deliberately not alphabetic: layout must preserve the existing field order.
    manifest = {"totals": {"units": 1}, "modules": {
        "model.proj": {"roles": [{"tensor": "model.proj.weight"}],
                       "family": "TESSERA_FP8"}},
        "label": "spaces stay inside strings: café", "requires_lanes": []}
    compact = tmp_path / "compact"
    compact.mkdir()
    compact_path = compact / "tessera_serving_manifest.json"
    serving_parts.write_serving_manifest(compact_path, manifest)
    raw = compact_path.read_bytes()
    assert raw == json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    assert json.loads(raw) == manifest
    assert list(json.loads(raw)) == list(manifest)
    second = tmp_path / "second.json"
    serving_parts.write_serving_manifest(second, manifest)
    assert second.read_bytes() == raw

    # Each directory is a NEW test artifact. Old indented bytes stay untouched.
    old = tmp_path / "old"
    old.mkdir()
    pretty = json.dumps(manifest, indent=2).encode("utf-8")
    (old / "tessera_serving_manifest.json").write_bytes(pretty)
    assert len(raw) < len(pretty)
    digests = {}
    rosters = []
    for directory, written in ((old, pretty), (compact, raw)):
        (directory / "config.json").write_text(json.dumps({
            "quantization_config": {"quant_method": "tessera"}}))
        (directory / "model.safetensors").write_bytes(b"fixture payload")
        for attest in (False, True):
            result = read_tessera_artifact(directory, with_file_attestation=attest)
            source, roster, _assignment = result[:3]
            digest = source["files"]["tessera_serving_manifest.json"]
            assert digest == hashlib.sha256(written).hexdigest()
            digests[directory.name] = digest
            rosters.append(roster)
            if attest:
                assert result[3]["files"]["tessera_serving_manifest.json"]["size"] == len(written)
        assert (directory / "tessera_serving_manifest.json").read_bytes() == written
    assert all(roster == rosters[0] for roster in rosters)
    assert digests["old"] != digests["compact"]


def test_lane_display_reads_pretty_and_compact_json(tmp_path):
    # Execute ONLY the display command, never the surrounding GPU campaign.
    script = Path(__file__).parents[1] / "experiments" / "ts104_chain.sh"
    commands = [line for line in script.read_text().splitlines()
                if '"$NEW/tessera_serving_manifest.json"' in line]
    assert len(commands) == 1
    manifest = {"requires_lanes": ["tessera_window_gemv", "another_lane"]}
    env = {**os.environ, "PY": sys.executable, "NEW": str(tmp_path)}
    for options in ({"indent": 2}, {"separators": (",", ":")}):
        (tmp_path / "tessera_serving_manifest.json").write_text(json.dumps(manifest, **options))
        result = subprocess.run(["bash", "-c", commands[0]], env=env,
                                text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == manifest


def test_refresh_writes_compact_manifest_without_changing_source(tmp_path):
    from experiments.refresh_native_resident_manifest import refresh

    source = tmp_path / "source"
    source.mkdir()
    original = json.dumps({"modules": {}, "totals": {
        "passthrough_bytes": 7, "checkpoint_bytes": 7}}, indent=2).encode()
    (source / "tessera_serving_manifest.json").write_bytes(original)
    (source / "model.safetensors").write_bytes(b"fixture")
    output = tmp_path / "fresh"
    refresh(source, output)
    raw = (output / "tessera_serving_manifest.json").read_bytes()
    parsed = json.loads(raw)
    assert raw == json.dumps(parsed, separators=(",", ":")).encode("utf-8")
    assert parsed["native_resident_refresh"]["source_manifest_sha256"] == hashlib.sha256(original).hexdigest()
    assert (source / "tessera_serving_manifest.json").read_bytes() == original
    assert (source / "model.safetensors").read_bytes() == b"fixture"
    assert (output / "model.safetensors").read_bytes() == b"fixture"
