"""Declaration replay is a public, standard-library-only producer owner."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "example/runtime@sha256:" + "e" * 64
OTHER = "example/runtime@sha256:" + "d" * 64


def api():
    return importlib.import_module("tessera.image_metadata")


def declaration(owner, **updates):
    record = {"schema": "tessera.runtime_image/1", "resolved_reference": IMAGE,
              "repo_digests": [IMAGE], "refused": False, "local_id": "local-only"}
    record.update(updates)
    return {owner.CENSUS_IMAGE_ENV: IMAGE,
            owner.CENSUS_DECLARATION_ENV: json.dumps(record)}


def test_public_replay_cold_import_blocks_serving_and_numerical_runtimes():
    script = '''
import importlib.abc, json, sys
sys.path.insert(0, sys.argv[1] + "/src")
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in
               ("tessera.serving", "torch", "transformers", "triton", "vllm", "numpy")):
            raise ImportError("replay crossed producer boundary: " + fullname)
sys.meta_path.insert(0, Block())
from tessera import image_metadata as owner
image = "example/runtime@sha256:" + "e" * 64
record = {"schema": "tessera.runtime_image/1", "resolved_reference": image,
          "repo_digests": [image], "refused": False}
env = {owner.CENSUS_IMAGE_ENV: image, owner.CENSUS_DECLARATION_ENV: json.dumps(record)}
assert owner.declared_reference(image, env=env)["record"] == record
assert not any(n == "tessera.serving" or n.startswith("tessera.serving.") for n in sys.modules)
'''
    result = subprocess.run([sys.executable, "-I", "-c", script, str(ROOT)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_valid_replay_preserves_launcher_record_and_declares_its_mechanism():
    owner = api()
    env = declaration(owner)
    result = owner.declared_reference(IMAGE, env=env)
    assert result == {"schema": owner.DECLARATION_SCHEMA,
                      "source": owner.DECLARATION_SOURCE, "image": IMAGE,
                      "env": [owner.CENSUS_IMAGE_ENV, owner.CENSUS_DECLARATION_ENV],
                      "record": json.loads(env[owner.CENSUS_DECLARATION_ENV])}
    result["record"]["refused"] = True
    assert json.loads(env[owner.CENSUS_DECLARATION_ENV])["refused"] is False


@pytest.mark.parametrize("raw", ["not json", "[]", "true", "false", "null", "12", '"text"',
                                  "{}", '{"schema":"other/1"}'])
def test_unreadable_records_refuse_by_name(raw):
    owner = api()
    env = declaration(owner)
    env[owner.CENSUS_DECLARATION_ENV] = raw
    with pytest.raises(owner.RuntimeImageError) as caught:
        owner.declared_reference(IMAGE, env=env)
    assert caught.value.payload == {"refused": True, "reason": "image_declaration_unreadable",
                                    "requested": IMAGE, "env": [owner.CENSUS_DECLARATION_ENV]}


@pytest.mark.parametrize("updates,requested,reason", [
    ({"refused": True, "resolved_reference": OTHER}, OTHER, "image_declaration_refused"),
    ({"refused": "yes"}, IMAGE, "image_declaration_refused"),
    ({"resolved_reference": OTHER}, OTHER, "image_declaration_inconsistent"),
    ({"repo_digests": None}, IMAGE, "image_declaration_inconsistent"),
    ({"repo_digests": []}, IMAGE, "image_declaration_inconsistent"),
    ({"repo_digests": [OTHER]}, IMAGE, "image_declaration_inconsistent"),
    ({}, OTHER, "image_declaration_mismatch"),
])
def test_refusal_order_remains_launcher_then_pair_then_request(updates, requested, reason):
    owner = api()
    with pytest.raises(owner.RuntimeImageError) as caught:
        owner.declared_reference(requested, env=declaration(owner, **updates))
    assert caught.value.payload == {"refused": True, "reason": reason,
                                    "requested": requested, "declared": IMAGE}


def test_explicit_empty_env_never_uses_host_declaration(monkeypatch):
    owner = api()
    for key, value in declaration(owner).items():
        monkeypatch.setenv(key, value)
    assert owner.declared_reference(IMAGE, env=None)["image"] == IMAGE
    with pytest.raises(owner.RuntimeImageError) as caught:
        owner.declared_reference(IMAGE, env={})
    assert caught.value.payload == {"refused": True, "reason": "image_declaration_missing",
                                    "requested": IMAGE,
                                    "env": [owner.CENSUS_IMAGE_ENV, owner.CENSUS_DECLARATION_ENV]}


def test_explicit_replay_ignores_contradictory_host_environment(monkeypatch):
    owner = api()
    monkeypatch.setenv(owner.CENSUS_IMAGE_ENV, OTHER)
    monkeypatch.setenv(owner.CENSUS_DECLARATION_ENV, "not json")
    assert owner.declared_reference(IMAGE, env=declaration(owner))["image"] == IMAGE


def test_error_is_runtime_error_with_a_shallow_payload_copy():
    owner = api()
    nested = []
    payload = {"reason": "original", "nested": nested}
    error = owner.RuntimeImageError("message", payload)
    payload["reason"] = "changed"
    assert isinstance(error, RuntimeError)
    assert str(error) == "message"
    assert error.payload == {"reason": "original", "nested": nested}
    assert error.payload is not payload
    assert error.payload["nested"] is nested


def test_serving_uses_the_same_public_error_and_replay_owners():
    owner = api()
    serving = importlib.import_module("tessera.serving.runtime_image")
    assert serving.RuntimeImageError is owner.RuntimeImageError
    assert serving.declared_reference is owner.declared_reference
    assert owner.RuntimeImageError.__module__ == "tessera.image_metadata"
    assert owner.declared_reference.__module__ == "tessera.image_metadata"
