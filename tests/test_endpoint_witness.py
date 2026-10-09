"""CPU checks for ``tessera.endpoint_witness``: the runtime join and its refusals (tessera#1056).

These fixtures are not runtime evidence and qualify no consumer on their
own. What they establish is the contract: the join the producer publishes,
the refusals that keep an input manifest, alias, lease, launch argument or
publication receipt from certifying loaded state, and the fingerprint that
binds the joined body.
"""
from __future__ import annotations

import pytest

from tessera import endpoint_witness as ew

LISTENER = {"endpoint": "http://10.100.96.2:8142", "served_alias": "glm53-artifact",
            "lifetime_id": "serve-1", "observed_unix": 1000.0}
LAUNCH = {"attempt_id": "nonce-abc", "ranks": [0, 1],
          "lifetime_id": "serve-1", "observed_unix": 1001.0}
ARTIFACTS = [
    {"rank": 0, "files": {"config.json": "a" * 64, "model.safetensors": "b" * 64},
     "bytes": 42, "lifetime_id": "serve-1", "observed_unix": 1002.0},
    {"rank": 1, "files": {"config.json": "a" * 64, "model.safetensors": "b" * 64},
     "bytes": 42, "lifetime_id": "serve-1", "observed_unix": 1003.0},
]
TOKENIZER = {"vocab_size": 512, "files": {"tokenizer.json": "c" * 64},
             "lifetime_id": "serve-1", "observed_unix": 1004.0}


def _witness(**over):
    args = {"listener": LISTENER, "launch": LAUNCH,
            "artifacts": ARTIFACTS, "tokenizer": TOKENIZER}
    args.update(over)
    return ew.build_witness(**args)


def test_a_complete_join_verifies():
    witness = _witness()
    assert witness["schema"] == ew.SCHEMA
    assert witness["byte_coverage"]["files"] == ["config.json", "model.safetensors"]
    assert witness["byte_coverage"]["ranks"] == [0, 1]
    assert ew.verify_witness(witness) is None
    assert ew.verify_witness(witness, ranks=[0, 1]) is None


def test_the_fingerprint_is_the_joined_body_not_the_stamp():
    witness = _witness()
    assert witness["fingerprint"] == ew.witness_fingerprint(witness)
    forged = dict(witness, fingerprint="0" * 64)
    assert "fingerprint" in ew.verify_witness(forged)


def test_an_unknown_schema_is_refused():
    witness = dict(_witness(), schema="x")
    assert "schema" in ew.verify_witness(witness)


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
    other = dict(ARTIFACTS[1], files={"config.json": "a" * 64})
    with pytest.raises(ValueError, match="different file sets"):
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


def test_an_edited_body_is_refused_by_the_fingerprint():
    witness = _witness()
    forged = dict(witness, listener=dict(LISTENER, served_alias="other-alias"))
    assert "fingerprint" in ew.verify_witness(forged)


def test_a_malformed_witness_is_a_refusal_not_an_exception():
    assert isinstance(ew.verify_witness({"schema": ew.SCHEMA}), str)
    assert "malformed" in ew.verify_witness({"schema": ew.SCHEMA})
