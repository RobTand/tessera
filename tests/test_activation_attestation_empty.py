"""An empty current collection makes no activation claim; old claims stay checked."""
import json
from pathlib import Path

import pytest

from tessera.serving.activation_attestation import (
    ACTIVATION_QUANTIZER_SCHEMA, validate_activation_quantizers)
from tessera.serving.contract import require_runtime_image

ARCHIVE = (Path(__file__).resolve().parents[1] / "experiments" / "results" /
           "t4_activation_quantizers_historical_20261007.json")


def test_empty_root_mapping_is_valid_without_a_current_attestation():
    block = {"schema": ACTIVATION_QUANTIZER_SCHEMA,
             "generator": "experiments/attest_activation_quantizer.py",
             "platforms": {}}
    validate_activation_quantizers(
        block, platforms=["sm_121"],
        cell_contracts={"sm_121": {"fp8_per_token_dynamic", "bf16_unquantized"}},
        require_image=require_runtime_image)
    block["platforms"] = json.loads(ARCHIVE.read_text())["platforms"]
    with pytest.raises(ValueError, match="no cell on this platform executes"):
        validate_activation_quantizers(
            block, platforms=["sm_121"],
            cell_contracts={"sm_121": {"fp8_per_token_dynamic", "bf16_unquantized"}},
            require_image=require_runtime_image)




@pytest.mark.parametrize("entries", [[], {}, None])
def test_empty_image_entries_are_still_refused(entries):
    block = {"schema": ACTIVATION_QUANTIZER_SCHEMA,
             "generator": "experiments/attest_activation_quantizer.py",
             "platforms": {"sm_121": entries}}
    with pytest.raises(ValueError, match="non-empty list"):
        validate_activation_quantizers(block, platforms=["sm_121"],
                                       cell_contracts={}, require_image=require_runtime_image)


def test_public_contract_keeps_required_empty_collection_without_t4_claims():
    from tessera.serving.contract import load_serving_contract

    current = load_serving_contract()
    assert current["activation_quantizers"]["platforms"] == {}
    assert not any(cell["activation_contract"] == "e2m1_group16_ue4m3_static"
                   for cell in current["lane_eligibility"]["cells"])


def test_archive_retains_the_exact_previous_receipt_bytes():
    import hashlib

    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == "ba0e4b07f1f51e392b07f51081648c2addc2781cdf11c1cf9de50a757e60acb7"


def test_empty_claimed_contracts_are_still_refused():
    block = json.loads(ARCHIVE.read_text())
    block["platforms"]["sm_121"][0]["contracts"] = {}
    with pytest.raises(ValueError, match="non-empty object"):
        validate_activation_quantizers(
            block, platforms=["sm_121"],
            cell_contracts={"sm_121": {"e2m1_group16_ue4m3_static"}},
            require_image=require_runtime_image)
