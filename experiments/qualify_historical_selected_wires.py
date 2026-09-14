"""Bounded CPU proof for two original selected GLM wires, never an encode.

The complete source/H/settings derivation was already qualified by the
241-cell prepare receipt; this independently checks original wire receipt
bytes through the sealed old verifier and current parser/decoder gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path
import stat

from tessera.cached_unit import verify_cached_unit
from tessera.historical_producer import load_historical_producer
from tessera.moe_execution import ResearchSelectedMoeConfig
from tessera.fused import parse_fused

NAME = "model.language_model.layers.3.mlp.experts.0.gate_proj"
FORMATS = ("TESSERA_E4M3_K1_R896", "TESSERA_BF16_K1_R1024")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--prior-results", type=Path, required=True)
    parser.add_argument("--producer-package", type=Path, required=True)
    parser.add_argument("--producer-source-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.fixture) == args.fixture_sha256
    with args.fixture.open("rb") as handle:
        fixture = pickle.load(handle)
    prior = json.loads(args.prior_results.read_text())
    assert prior["passed"] is True and prior["qualified_cells"] == 241
    assert prior["metadata_fixture_sha256"] == args.fixture_sha256
    assert fixture["schema"] == "prismaquant.bounded_joint_prepare_metadata.v3"
    producer = load_historical_producer(args.producer_package, args.producer_source_sha256)
    exporter_path = Path(__file__).resolve().with_name("export_tessera_serving.py")
    spec = importlib.util.spec_from_file_location("historical_export_proof", exporter_path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    reader = ResearchSelectedMoeConfig(max_experts_per_chunk=2,
                                       expected_tensor_parallel_size=2)
    rows = []
    for fmt in FORMATS:
        cell = fixture["cells"][(NAME, fmt)]
        record = cell["record"]
        path = Path(cell["wire"])
        assert stat.S_ISREG(path.lstat().st_mode) and path.stat().st_size == record["blob_bytes"]
        blob = path.read_bytes()
        assert hashlib.sha256(blob).hexdigest() == record["blob_sha256"]
        assert prior["records"][NAME + "@" + fmt]["wire_sha256"] == record["blob_sha256"]
        expected = record["identity"]  # source/H were rederived in prior verified qualification
        assert expected["encoder_source_sha256"] == args.producer_source_sha256
        old = producer.verify(blob, record, expected)
        current = verify_cached_unit(blob, record, expected)
        assert old.blob == current.blob == blob
        accepted, packaged = exporter.pack_cached_expert_unit(blob, record, expected)
        assert accepted.blob == parse_fused(packaged)[0].blob == blob
        recipe = expected["recipe"]
        family = reader.require_wire_recipe(grid=recipe["grid"], q256=recipe["q256"],
            body=current.manifest.body.name, plane=current.manifest.scale_plane.kind.name,
            span=current.manifest.span, target="model.language_model.layers.3.mlp.experts")
        def wire_fields(manifest):
            return (manifest.geometry.rows, manifest.geometry.columns,
                    manifest.branch.root_q256, manifest.encoder_profile_id.hex(),
                    manifest.body.name, manifest.scale_plane.kind.name, manifest.span,
                    manifest.planes[-1].element_count)
        same_wire = wire_fields(old.manifest) == wire_fields(current.manifest)
        rows.append({"format": fmt, "blob_sha256": record["blob_sha256"],
                     "blob_bytes": len(blob), "old_current_wire_fields_equal": same_wire,
                     "current_research_decoder": family, "exported_original_blob": True})
        assert same_wire
    result = {"passed": True, "scope": "two original selected receipts, not full-model export/serve",
              "producer_source_sha256": args.producer_source_sha256,
              "fixture_sha256": args.fixture_sha256, "prior_qualified_cells": prior["qualified_cells"],
              "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
