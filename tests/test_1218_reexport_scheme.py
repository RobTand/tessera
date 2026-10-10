"""tessera#1218: head accepts the re-exported artifact.

The sealed export ``glm53-x-picks-full-6be66bc`` predates 299cc6d683,
which requires ``expert_ids`` and ``expert_classes`` in every routed_moe
scheme. ``TESSERA_1218_ARTIFACT`` selects the artifact under test; the
default is the re-export at head, and the test skips when it is absent.
Against the sealed export the census fails: the 42 routed schemes carry
neither field and the head refuses them with ``expert_ids is required``.
"""
import json
import os
from pathlib import Path

import pytest

from tessera.serving.scheme import validate_tessera_scheme

SEALED_ARTIFACT = Path(
    "/mnt/shared/tessera-runs/moe/glm53-x-picks-full-6be66bc/exported")
REEXPORT_ARTIFACT = Path(
    "/mnt/shared/tessera-runs/moe/glm53-x-picks-full-1218/exported")
EXPECTED_ROUTED = 42


def _artifact() -> Path:
    return Path(os.environ.get("TESSERA_1218_ARTIFACT", str(REEXPORT_ARTIFACT)))


def _routed_targets():
    config_path = _artifact() / "config.json"
    if not config_path.is_file():
        pytest.skip(f"artifact absent: {_artifact()}")
    groups = json.loads(config_path.read_bytes(
    ))["quantization_config"]["config_groups"]
    routed = [(targets[0], group["scheme"])
              for group in groups.values()
              for targets in [group.get("targets", [])]
              if group.get("scheme", {}).get("structure") == "routed_moe"]
    assert routed, "export holds no routed_moe scheme"
    return routed


def test_routed_schemes_carry_expert_metadata():
    routed = _routed_targets()
    assert len(routed) == EXPECTED_ROUTED, len(routed)
    for target, scheme in routed:
        declared = validate_tessera_scheme(scheme, target)
        assert declared["expert_ids"] == list(scheme["expert_ids"]), target
        assert declared["expert_classes"] == list(scheme["expert_classes"]), target


def test_from_config_accepts_artifact():
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from tessera.serving.config import TesseraConfig
    config_path = _artifact() / "config.json"
    if not config_path.is_file():
        pytest.skip(f"artifact absent: {_artifact()}")
    config = json.loads(config_path.read_bytes())["quantization_config"]
    parsed = TesseraConfig.from_config(config)
    assert len(parsed.target_scheme) == len(config["config_groups"])
