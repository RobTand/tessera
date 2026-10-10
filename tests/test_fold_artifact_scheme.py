"""tessera#1204: the served export predates the head's expert metadata rule.

The GLM-5.3 export at ``glm53-x-picks-full-6be66bc`` was built at pin
af7a86d43, before 299cc6d683 required ``expert_ids`` in every routed_moe
scheme. Head 4c0fd6ac refuses the export's 42 routed schemes at quant-config
parse, inside ``create_engine_config``, before speculation or model build.
Both probe variants (mtp, nospec) refuse the same way: the MTP draft reuses
the target quant_config object. This test pins that verdict from the real
export bytes so the rerun (re-export or backfill) must turn it green.
"""
import json
from pathlib import Path

import pytest

from tessera.serving.scheme import validate_tessera_scheme

ARTIFACT_CONFIG = Path(
    "/mnt/shared/tessera-runs/moe/glm53-x-picks-full-6be66bc/exported/config.json")


def _routed_targets():
    if not ARTIFACT_CONFIG.is_file():
        pytest.skip(f"export absent: {ARTIFACT_CONFIG}")
    groups = json.loads(ARTIFACT_CONFIG.read_bytes(
    ))["quantization_config"]["config_groups"]
    routed = [(targets[0], group["scheme"])
              for group in groups.values()
              for targets in [group.get("targets", [])]
              if group.get("scheme", {}).get("structure") == "routed_moe"]
    assert routed, "export holds no routed_moe scheme"
    return routed


def test_routed_schemes_carry_no_expert_ids():
    for target, scheme in _routed_targets():
        assert scheme.get("expert_ids") is None, target


def test_head_refuses_every_routed_scheme_without_expert_ids():
    refused = 0
    for target, scheme in _routed_targets():
        with pytest.raises(ValueError, match="expert_ids is required"):
            validate_tessera_scheme(scheme, target)
        refused += 1
    assert refused == len(_routed_targets())


def test_identity_metadata_is_the_only_missing_field():
    target, scheme = _routed_targets()[0]
    experts = int(scheme["experts"])
    w13 = scheme["groups"]["w13"]["q256"]
    w2 = scheme["groups"]["w2"]["q256"]
    fixed = dict(scheme)
    fixed["expert_ids"] = list(range(experts))
    fixed["expert_classes"] = [
        {"start": 0, "end": experts,
         "q256": {"w13": [w13, w13], "w2": [w2]}}]
    declared = validate_tessera_scheme(fixed, target)
    assert declared["experts"] == experts
    assert declared["expert_ids"] == list(range(experts))
