"""Offline acceptance of the artifact-refresh handoff, NOT a synthetic TP2 serve.

The implementer supplies docs/measurements/issue1358-tp2-readiness.json and
repo-local, digest-bound evidence. Large checkpoints remain in configured
artifact storage; these tests never open fleet paths or consult Git history.
A file reference is {"path": <repo-relative>, "sha256": <hex>}.
The receipt's authenticity and producer/pinned-consumer compatibility still
require review of the retained export/serve receipts, not just these checks.
"""
import hashlib
import json
from pathlib import Path
import re

from tessera.serving.scheme import validate_tessera_moe_scheme


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/measurements/issue1358-tp2-readiness.json"
RUNTIME = "4c0fd6aca04f1c3e95b094ae006c2479043c3372"


def _hex(value, size=64):
    assert isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % size, value), value


def _text(value):
    assert isinstance(value, str) and value.strip(), value


def _evidence(ref):
    path = Path(ref["path"])
    assert not path.is_absolute() and ".." not in path.parts, path
    resolved = (ROOT / path).resolve()
    assert resolved.is_relative_to(ROOT), path
    assert resolved.is_file(), f"Missing checked-in evidence: {path}"
    raw = resolved.read_bytes()
    assert raw, path
    _hex(ref["sha256"])
    assert hashlib.sha256(raw).hexdigest() == ref["sha256"], path
    return raw.decode("utf-8")


def _json(ref):
    return json.loads(_evidence(ref))


def _receipt():
    assert RECEIPT.is_file(), (
        "Missing issue1358 artifact-refresh/TP2 receipt: retain original failure, "
        "recover export inputs/reuse disposition, validate all routed stacks, "
        "and obtain pinned-runtime two-rank readiness evidence."
    )
    value = json.loads(RECEIPT.read_text())
    assert value["issue"] == "tessera#1358"
    return value


def _routed(config):
    result = {}
    for group in config["quantization_config"]["config_groups"].values():
        scheme = group["scheme"]
        if scheme.get("structure") == "routed_moe":
            assert len(group["targets"]) == 1, group["targets"]
            target = group["targets"][0]
            assert target not in result, target
            result[target] = scheme
    assert result, "An empty routed population cannot establish artifact repair"
    return result


def test_original_failure_and_preserved_export_bindings():
    receipt = _receipt()
    original = receipt["original"]
    assert original["artifact_id"] == "glm53-x-picks-full-6be66bc"
    _text(original["root"])
    config = _json(original["config"])
    old = _routed(config)
    assert len(old) == 42
    missing = sorted(target for target, scheme in old.items()
                     if scheme.get("expert_ids") is None)
    assert missing, "Retain the actual stale config, not a repaired replacement"
    assert original["missing_expert_ids"] == missing
    log = _evidence(original["failure_log"])
    assert "expert_ids is required" in log
    assert any(target in log for target in missing)
    # The issue's af7a86d43 report and recovered roster's 3725c1b1 are not
    # interchangeable identities. Retain the authorities and reconcile them.
    _evidence(original["producer_identity_evidence"])

    export = receipt["export"]
    assert export["disposition"] in {"accepted_reuse", "fresh_export"}
    assert export["root"] != original["root"]
    _text(export["root"])
    _hex(export["producer_commit"], 40)
    assert export["runtime_commit"] == RUNTIME
    _evidence(export["producer_consumer_compatibility"])
    _evidence(export["issue1225_recovery_and_reuse_decision"])
    _evidence(export["execution_receipt"])
    assert export["argv"] and all(isinstance(arg, str) for arg in export["argv"])
    assert "export_tessera_serving" in " ".join(export["argv"])
    assert export["original_unchanged"] is True
    assert export["metadata_hand_backfilled"] is False
    # Each authority includes a URI and digest. A bound evidence document
    # explains what was hashed (source inventory/config, plan, cache, etc.).
    for key in ("source", "allocation", "plan", "priced_inputs", "scales", "cache"):
        before = receipt["inputs_original"][key]
        after = receipt["inputs_repaired"][key]
        _text(before["uri"])
        _hex(before["sha256"])
        assert after == before, f"Refresh changed {key} instead of repairing the artifact"
    _evidence(receipt["binding_evidence"])


def test_every_routed_stack_matches_original_plan_in_storage_order():
    receipt = _receipt()
    old = _routed(_json(receipt["original"]["config"]))
    config = _json(receipt["export"]["config"])
    repaired = _routed(config)
    assert len(repaired) == 42 and repaired.keys() == old.keys()
    source = _json(receipt["source_config"])
    source = source.get("text_config", source)
    counts = [source[key] for key in ("n_routed_experts", "num_experts", "num_local_experts")
              if key in source]
    assert counts and all(type(n) is int and n > 0 for n in counts)
    assert len(set(counts)) == 1
    # Original/global-ordered rungs are extracted from the bound original plan,
    # not invented from the repaired map. Retain the extraction command/receipt.
    original_rungs = _json(receipt["original_plan_rungs"])
    _evidence(receipt["plan_projection_evidence"])
    assert original_rungs.keys() == repaired.keys()
    for target, scheme in repaired.items():
        declared = validate_tessera_moe_scheme(scheme, target)
        experts = declared["experts"]
        assert experts == counts[0] == old[target]["experts"], target
        ids = declared["expert_ids"]
        matrices = original_rungs[target]
        assert set(matrices) == {"w13", "w2"}
        for name, group in declared["groups"].items():
            rows = matrices[name]
            assert len(rows) == experts
            assert all(len(row) == (2 if name == "w13" else 1) for row in rows)
            assert all(type(q) is int and q > 0 for row in rows for q in row)
            stored = group.get("expert_role_q256", [group["role_q256"]] * experts)
            assert stored == [rows[global_id] for global_id in ids], (target, name)
    # Include the actual all-stack manifest/source/wire/scale reconciliation,
    # not only a layer-10 validation or a config parse.
    _evidence(receipt["all_stack_artifact_validation"])


def test_generated_artifact_inventory_is_bound_to_serve():
    receipt = _receipt()
    export = receipt["export"]
    inventory = _json(export["file_inventory"])
    assert {"config.json", "tessera_serving_manifest.json",
            "model.safetensors.index.json"} <= inventory.keys()
    for name, record in inventory.items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        _hex(record["sha256"])
        assert type(record["bytes"]) is int and record["bytes"] > 0
    assert inventory["config.json"]["sha256"] == export["config"]["sha256"]
    # Index membership is retained as a compact, digest-bound extraction so
    # offline checks do not read a 186 GB checkpoint or duplicate its index.
    shards = _json(export["index_shards"])
    assert shards and len(shards) == len(set(shards))
    assert set(shards) == {name for name in inventory if name.endswith(".safetensors")}
    _evidence(export["inventory_verification"])
    serve = receipt["tp2"]
    assert serve["artifact_root"] == export["root"]
    assert serve["file_inventory_sha256"] == export["file_inventory"]["sha256"]
    assert serve["runtime_commit"] == export["runtime_commit"] == RUNTIME


def test_pinned_tp2_execution_and_campaign_handoff():
    receipt = _receipt()
    serve = receipt["tp2"]
    preflight = serve["gpu_plan"]
    _text(preflight["smallest_deciding_case"])
    _text(preflight["why_cpu_cannot_decide"])
    assert preflight["expected_gpu_minutes"] > 0
    assert preflight["recorded_at"] < serve["started_at"]
    assert serve["elapsed_gpu_minutes"] > 0
    _evidence(serve["execution_receipt"])
    _evidence(serve["runtime_identity_evidence"])
    effective = _json(serve["effective_configuration"])
    assert effective["model"] == serve["artifact_root"]
    assert effective["tensor_parallel_size"] == 2
    assert serve["argv"] and all(isinstance(arg, str) for arg in serve["argv"])
    assert serve["artifact_root"] in " ".join(serve["argv"])
    _evidence(serve["configuration_reconciliation"])
    # Reconcile the actually executed configuration, rank logs and launcher;
    # the historical engine.json alone disagrees with the observed invocation.
    ranks = serve["rank_logs"]
    assert len(ranks) == 2 and {rank["rank"] for rank in ranks} == {0, 1}
    paths = set()
    for rank in ranks:
        text = _evidence(rank["log"])
        paths.add(rank["log"]["path"])
        _text(rank["loaded_excerpt"])
        assert rank["loaded_excerpt"] in text
        assert "expert_ids is required" not in text
    assert len(paths) == 2
    ready_log = _evidence(serve["ready_log"])
    _text(serve["ready_excerpt"])
    assert serve["ready_excerpt"] in ready_log
    # A successful nonempty completion demonstrates execution past construction.
    _json(serve["request"])
    response = _json(serve["response"])
    assert response["choices"] and response["usage"]["completion_tokens"] > 0
    _evidence(serve["request_execution_evidence"])
    handoff = receipt["campaign_handoff"]
    assert handoff["campaign"] == "tessera#1204"
    assert handoff["artifact_root"] == serve["artifact_root"]
    assert handoff["runtime_commit"] == RUNTIME
    assert handoff["file_inventory_sha256"] == serve["file_inventory_sha256"]
    assert handoff["scope"] == "artifact_refresh_and_tp2_readiness_only"
    _evidence(handoff["consumption_instructions"])
