"""Regressions for the four detached-input failures reproduced before the cutover."""
from __future__ import annotations

import copy

from tessera import endpoint_witness as ew
from _endpoint_fixture import receipt, resign


def test_directory_proof_cannot_certify_detached_loaded_digests(tmp_path):
    artifact = tmp_path / "artifact"
    value = receipt(artifact)
    assert "live runtime binding" in ew.verify_witness(value, served_dir=artifact)


def test_alias_agreement_does_not_prove_listener_ownership(tmp_path):
    artifact = tmp_path / "artifact"
    value = receipt(artifact)
    replacement = copy.deepcopy(value)
    replacement["listener"]["owner"]["start_ticks"] += 1
    replacement["launch"]["attempt_id"] = ew.attempt_id(replacement["listener"]["owner"])
    resign(replacement)
    assert "binding differs" in ew.verify_witness(value, served_dir=artifact, live=replacement)


def test_vocabulary_length_does_not_prove_loaded_tokenizer_mapping(tmp_path):
    artifact = tmp_path / "artifact"
    value = receipt(artifact)
    value["tokenizer"]["vocab"] = {"<s>": 1, "a": 0}
    resign(value)
    assert "tokenizer mapping" in ew.verify_witness(value, served_dir=artifact, live=value)


def test_probe_cannot_supply_the_actual_launch_attempt(tmp_path):
    artifact = tmp_path / "artifact"
    value = receipt(artifact)
    value["launch"]["attempt_id"] = "arbitrary-attempt"
    resign(value)
    assert "launch attempt" in ew.verify_witness(value, served_dir=artifact, live=value)
