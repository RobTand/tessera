"""CPU checks for ``tessera.endpoint_witness``: the runtime join and its refusals (tessera#1056).

These fixtures are not runtime evidence and qualify no consumer on their
own. What they establish is the contract: the join the producer publishes,
the refusals that keep an input manifest, alias, lease, launch argument or
publication receipt from certifying loaded state, and the fingerprint that
binds the joined body. A complete join still refuses as unverified until
:func:`stamp_byte_proof` proves each digest and size against one live
served directory; a CPU fixture directory supplies that proof of shape.
JSON-only agreement is structural validation, never loaded-state evidence:
``verify_witness`` needs the served directory to pass.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from tessera import endpoint_witness as ew

LISTENER = {"endpoint": "http://10.100.96.2:8142", "served_alias": "glm53-artifact",
            "lifetime_id": "serve-1", "observed_unix": 1000.0}
LAUNCH = {"attempt_id": "nonce-abc", "ranks": [0, 1],
          "lifetime_id": "serve-1", "observed_unix": 1001.0}
_AA = "a" * 64
_BB = "b" * 64
_CC = "c" * 64
_DD = "d" * 64
ARTIFACTS = [
    {"rank": 0, "files": {"config.json": _AA, "model.safetensors": _BB,
                          "tokenizer.json": _CC, "loaded:model.layers.0.mlp": _DD},
     "sizes": {"config.json": 10, "model.safetensors": 32, "tokenizer.json": 8,
               "loaded:model.layers.0.mlp": 32},
     "bytes": 82, "lifetime_id": "serve-1", "observed_unix": 1002.0},
    {"rank": 1, "files": {"config.json": _AA, "model.safetensors": _BB,
                          "tokenizer.json": _CC, "loaded:model.layers.0.mlp": _DD},
     "sizes": {"config.json": 10, "model.safetensors": 32, "tokenizer.json": 8,
               "loaded:model.layers.0.mlp": 32},
     "bytes": 82, "lifetime_id": "serve-1", "observed_unix": 1003.0},
]
TOKENIZER = {"vocab_size": 512, "vocab_source": "server-tokenizer",
             "files": {"tokenizer.json": _CC}, "sizes": {"tokenizer.json": 8},
             "lifetime_id": "serve-1", "observed_unix": 1004.0}


def _witness(**over):
    args = {"listener": LISTENER, "launch": LAUNCH,
            "artifacts": ARTIFACTS, "tokenizer": TOKENIZER}
    args.update(over)
    return ew.build_witness(**args)


def _served(tmp_path):
    root = tmp_path / "artifact"
    root.mkdir()
    return root


def _prove(witness, served):
    (served / "config.json").write_bytes(b"0" * 10)
    (served / "model.safetensors").write_bytes(b"1" * 32)
    (served / "tokenizer.json").write_bytes(b"2" * 8)
    witness = json.loads(json.dumps(witness))
    for rank in witness["artifacts"]:
        for name, blob in (("config.json", b"0" * 10), ("model.safetensors", b"1" * 32),
                           ("tokenizer.json", b"2" * 8)):
            rank["files"][name] = hashlib.sha256(blob).hexdigest()
    witness["tokenizer"]["files"]["tokenizer.json"] = hashlib.sha256(b"2" * 8).hexdigest()
    witness["fingerprint"] = ew.witness_fingerprint(witness)
    return ew.stamp_byte_proof(witness, served)


def test_a_complete_join_verifies_only_after_a_byte_proof(tmp_path):
    served = _served(tmp_path)
    witness = _witness()
    assert witness["schema"] == ew.SCHEMA
    assert "loaded:model.layers.0.mlp" in witness["byte_coverage"]["files"]
    assert witness["byte_coverage"]["ranks"] == [0, 1]
    assert "byte proof" in (ew.verify_witness(witness, served_dir=served) or "")
    proved = _prove(witness, served)
    assert ew.verify_witness(proved, served_dir=served) is None
    assert ew.verify_witness(proved, ranks=[0, 1], served_dir=served) is None


def test_json_agreement_without_bytes_is_not_loaded_evidence(tmp_path):
    served = _served(tmp_path)
    proved = _prove(_witness(), served)
    reason = ew.verify_witness(proved)
    assert reason is not None and "structural agreement" in reason


def test_a_fabricated_proof_block_never_verifies(tmp_path):
    served = _served(tmp_path)
    (served / "config.json").write_bytes(b"0" * 10)
    (served / "model.safetensors").write_bytes(b"1" * 32)
    (served / "tokenizer.json").write_bytes(b"2" * 8)
    witness = _witness()
    forged = json.loads(json.dumps(witness))
    live_names = ["config.json", "model.safetensors", "tokenizer.json"]
    sizes = {name: forged["artifacts"][0]["sizes"][name] for name in live_names}
    forged["byte_proof"] = {
        "served_files": live_names, "served_sizes": sizes,
        "served_bytes": sum(sizes.values()) + 32,
        "proof_input": hashlib.sha256(ew.canonical(
            {"files": {name: forged["artifacts"][0]["files"][name]
                       for name in live_names + ["loaded:model.layers.0.mlp"]},
             "sizes": {name: forged["artifacts"][0]["sizes"][name]
                       for name in live_names + ["loaded:model.layers.0.mlp"]},
             "served_sizes": sizes}).encode()).hexdigest(),
        "lifetime_id": "serve-1", "observed_unix": 1005.0,
    }
    reason = ew.verify_witness(forged, served_dir=served)
    assert reason is not None and "config.json" in reason


def test_the_fingerprint_is_the_joined_body_not_the_stamp(tmp_path):
    served = _served(tmp_path)
    witness = _witness()
    assert witness["fingerprint"] == ew.witness_fingerprint(witness)
    forged = dict(witness, fingerprint="0" * 64)
    assert "fingerprint" in ew.verify_witness(forged, served_dir=served)


def test_an_unknown_schema_is_refused(tmp_path):
    witness = dict(_witness(), schema="x")
    assert "schema" in ew.verify_witness(witness, served_dir=_served(tmp_path))


def test_a_serve_that_does_not_name_its_scope_is_refused():
    listener = {k: v for k, v in LISTENER.items() if k != "served_alias"}
    with pytest.raises(ValueError, match="misses"):
        _witness(listener=listener)


def test_a_launch_without_ranks_is_refused():
    launch = dict(LAUNCH, ranks=[])
    with pytest.raises(ValueError, match="ranks"):
        _witness(launch=launch)


def test_byte_observations_must_cover_the_launch_ranks():
    short = [ARTIFACTS[0]]
    with pytest.raises(ValueError, match="cover launch ranks"):
        _witness(artifacts=short)


def test_mixed_attempt_observations_are_refused():
    other = dict(ARTIFACTS[1], lifetime_id="serve-2")
    with pytest.raises(ValueError, match="lifetimes"):
        _witness(artifacts=[ARTIFACTS[0], other])


def test_a_launch_from_another_serve_is_refused():
    launch = dict(LAUNCH, lifetime_id="serve-2")
    with pytest.raises(ValueError, match="lifetime"):
        _witness(launch=launch)


def test_rank_byte_mismatch_is_refused():
    other = dict(ARTIFACTS[1], files=dict(ARTIFACTS[1]["files"], **{"model.safetensors": "d" * 64}))
    with pytest.raises(ValueError, match="differs across ranks"):
        _witness(artifacts=[ARTIFACTS[0], other])


def test_ranks_that_cover_different_files_are_refused():
    other = dict(ARTIFACTS[1], files={"config.json": "a" * 64, "tokenizer.json": _CC,
                                      "loaded:model.layers.0.mlp": _DD},
                 sizes={"config.json": 10, "tokenizer.json": 8,
                        "loaded:model.layers.0.mlp": 32}, bytes=50)
    with pytest.raises(ValueError, match="different file sets"):
        _witness(artifacts=[ARTIFACTS[0], other])


def test_rank_sizes_must_add_to_the_rank_bytes():
    other = dict(ARTIFACTS[1], bytes=51)
    with pytest.raises(ValueError, match="sizes do not add"):
        _witness(artifacts=[ARTIFACTS[0], other])


def test_rank_sizes_must_stay_positive():
    other = dict(ARTIFACTS[1],
                 sizes=dict(ARTIFACTS[1]["sizes"], **{"model.safetensors": 0}))
    with pytest.raises(ValueError, match="not a positive count"):
        _witness(artifacts=[ARTIFACTS[0], other])


def test_client_tokenizer_metadata_without_server_files_is_refused():
    tokenizer = {"vocab_size": 512, "lifetime_id": "serve-1", "observed_unix": 1004.0}
    with pytest.raises(ValueError, match="misses"):
        _witness(tokenizer=tokenizer)


def test_a_verifier_needs_no_tessera_serving_import():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(ew.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "torch" not in imported
    assert "vllm" not in imported
    source = Path(ew.__file__).read_text()
    assert "tessera.serving" not in source


def test_an_edited_body_is_refused_by_the_fingerprint(tmp_path):
    served = _served(tmp_path)
    witness = _prove(_witness(), served)
    forged = dict(witness, listener=dict(LISTENER, served_alias="other-alias"))
    assert "fingerprint" in ew.verify_witness(forged, served_dir=served)


def test_a_malformed_witness_is_a_refusal_not_an_exception(tmp_path):
    served = _served(tmp_path)
    assert isinstance(ew.verify_witness({"schema": ew.SCHEMA}, served_dir=served), str)
    assert "malformed" in ew.verify_witness({"schema": ew.SCHEMA}, served_dir=served)


def test_shared_partial_inventories_never_verify_without_a_live_proof(tmp_path):
    served = _served(tmp_path)
    short = [dict(item, files={"config.json": _AA, "tokenizer.json": _CC,
                               "loaded:model.layers.0.mlp": _DD},
                  sizes={"config.json": 10, "tokenizer.json": 8,
                         "loaded:model.layers.0.mlp": 32}, bytes=50)
             for item in ARTIFACTS]
    witness = _witness(artifacts=short)
    assert "byte proof" in (ew.verify_witness(witness, served_dir=served) or "")
    (served / "config.json").write_bytes(b"0" * 10)
    (served / "model.safetensors").write_bytes(b"1" * 32)
    (served / "tokenizer.json").write_bytes(b"2" * 8)
    small = json.loads(json.dumps(witness))
    for rank in small["artifacts"]:
        rank["files"] = {"config.json": hashlib.sha256(b"0" * 10).hexdigest(),
                         "tokenizer.json": hashlib.sha256(b"2" * 8).hexdigest(),
                         "loaded:model.layers.0.mlp": _DD}
    small["tokenizer"]["files"] = {"tokenizer.json": hashlib.sha256(b"2" * 8).hexdigest()}
    small["fingerprint"] = ew.witness_fingerprint(small)
    assert "model.safetensors" in (ew.prove_loaded_bytes(small, served) or "")


def test_contradictory_byte_coverage_is_refused(tmp_path):
    served = _served(tmp_path)
    witness = _prove(_witness(), served)
    assert ew.verify_witness(witness, served_dir=served) is None
    forged = dict(witness, byte_coverage=dict(witness["byte_coverage"], ranks=[0]))
    assert "byte_coverage" in ew.verify_witness(forged, served_dir=served)


def test_absent_byte_coverage_is_refused(tmp_path):
    served = _served(tmp_path)
    witness = _prove(_witness(), served)
    assert ew.verify_witness(witness, served_dir=served) is None
    forged = {key: value for key, value in witness.items() if key != "byte_coverage"}
    assert "byte_coverage" in ew.verify_witness(forged, served_dir=served)


def test_tampered_sizes_are_refused(tmp_path):
    served = _served(tmp_path)
    witness = _prove(_witness(), served)
    assert ew.verify_witness(witness, served_dir=served) is None
    forged = json.loads(json.dumps(witness))
    forged["artifacts"][0]["bytes"] += 1
    reason = ew.verify_witness(forged, served_dir=served)
    assert reason is not None and ("fingerprint" in reason or "malformed" in reason)


def test_a_missing_byte_proof_block_is_refused(tmp_path):
    served = _served(tmp_path)
    witness = _prove(_witness(), served)
    forged = {key: value for key, value in witness.items() if key != "byte_proof"}
    assert "byte_proof" in ew.verify_witness(forged, served_dir=served)


def test_an_empty_byte_proof_is_a_refusal_not_an_exception(tmp_path):
    served = _served(tmp_path)
    witness = _prove(_witness(), served)
    forged = dict(witness, byte_proof={})
    reason = ew.verify_witness(forged, served_dir=served)
    assert isinstance(reason, str) and "byte_proof" in reason


def test_a_partial_byte_proof_names_its_missing_field(tmp_path):
    served = _served(tmp_path)
    witness = _prove(_witness(), served)
    for missing in ew.BYTE_PROOF_KEYS:
        forged = dict(witness, byte_proof={key: value for key, value in witness["byte_proof"].items()
                                           if key != missing})
        reason = ew.verify_witness(forged, served_dir=served)
        assert isinstance(reason, str) and "byte_proof" in reason, missing


def test_a_tampered_byte_proof_is_refused(tmp_path):
    served = _served(tmp_path)
    witness = _prove(_witness(), served)
    forged = dict(witness, byte_proof=dict(witness["byte_proof"], served_bytes=1))
    assert "byte_proof" in ew.verify_witness(forged, served_dir=served)


def test_a_witness_without_a_loaded_entry_never_proves(tmp_path):
    served = _served(tmp_path)
    bare = [dict(item,
                 files={name: digest for name, digest in item["files"].items()
                        if not name.startswith("loaded:")},
                 sizes={name: size for name, size in item["sizes"].items()
                        if not name.startswith("loaded:")},
                 bytes=50)
            for item in ARTIFACTS]
    witness = _witness(artifacts=bare)
    (served / "config.json").write_bytes(b"0" * 10)
    (served / "model.safetensors").write_bytes(b"1" * 32)
    (served / "tokenizer.json").write_bytes(b"2" * 8)
    plain = json.loads(json.dumps(witness))
    for rank in plain["artifacts"]:
        for name, blob in (("config.json", b"0" * 10), ("model.safetensors", b"1" * 32),
                           ("tokenizer.json", b"2" * 8)):
            rank["files"][name] = hashlib.sha256(blob).hexdigest()
    plain["tokenizer"]["files"]["tokenizer.json"] = hashlib.sha256(b"2" * 8).hexdigest()
    plain["fingerprint"] = ew.witness_fingerprint(plain)
    assert "loaded wire" in (ew.prove_loaded_bytes(plain, served) or "")


def test_an_incorrect_digest_fails_the_live_proof(tmp_path):
    served = _served(tmp_path)
    (served / "config.json").write_bytes(b"0" * 10)
    (served / "model.safetensors").write_bytes(b"1" * 32)
    (served / "tokenizer.json").write_bytes(b"2" * 8)
    witness = _witness()
    reason = ew.prove_loaded_bytes(witness, served)
    assert reason is not None and "config.json" in reason


def test_a_removed_file_fails_the_live_proof(tmp_path):
    served = _served(tmp_path)
    witness = _prove(_witness(), served)
    (served / "model.safetensors").unlink()
    reason = ew.prove_loaded_bytes(witness, served)
    assert reason is not None and "model.safetensors" in reason


def test_an_added_file_fails_the_live_proof(tmp_path):
    served = _served(tmp_path)
    witness = _prove(_witness(), served)
    (served / "extra.bin").write_bytes(b"x")
    reason = ew.prove_loaded_bytes(witness, served)
    assert reason is not None and "extra.bin" in reason
