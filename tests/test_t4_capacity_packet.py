"""Real PB metadata contracts; no CUDA execution or numerical qualification."""
import copy
import json
from pathlib import Path

import pytest

from prismabuild import local_scratch


ROOT = Path(__file__).resolve().parents[1]
PACKETS = ROOT / "experiments/t4_code"
EXTRA_ENV = {
    "PRISMABUILD_LOCAL_SCRATCH_PAIRS": (
        "T4_HOST_TMP_ROOT:T4_HOST_TMP_MAX_BYTES,T4_DOCKER_ROOT:T4_DOCKER_MAX_BYTES"
    ),
    "T4_HOST_TMP_ROOT": "/home/rob/tmp",
    "T4_HOST_TMP_MAX_BYTES": "201326592",
    "T4_DOCKER_ROOT": "/var/lib/docker",
    "T4_DOCKER_MAX_BYTES": "872415232",
}


def load(name):
    return json.loads((PACKETS / name).read_text())


def test_capacity_only_delta_preserves_every_original_row():
    original = load("pb_875_finite_campaign.json")
    proposed = load("pb_875_sl_capacity_campaign.json")
    assert len(original) == len(proposed) == 15
    for old, new in zip(original, proposed):
        restored = copy.deepcopy(new)
        for key, value in EXTRA_ENV.items():
            assert key not in old["env"]
            assert restored["env"].pop(key) == value
        assert restored == old


@pytest.mark.parametrize("index", range(15))
def test_real_root_forecasts_derive_positive_native_reservation(index):
    row = load("pb_875_sl_capacity_campaign.json")[index]
    pairs = local_scratch.scratch_pairs(row["env"])
    assert [(p["root"], p["max_bytes"]) for p in pairs] == [
        ("/home/rob/tmp", 192 * 1024**2),
        ("/var/lib/docker", 832 * 1024**2),
    ]
    assert sum(p["max_bytes"] for p in pairs) == 1024**3
    # The existing API rounds each REAL root separately, not their sum.
    assert local_scratch.scratch_terms(row["env"]) == {"spool_gb": 2}
    assert "spool_gb" not in row["demand"]
    assert "tags" not in row and "here" not in row and "anywhere" not in row
    assert row["host_class"] == "gb10" and row["measurement"] is True


def test_proposal_requires_fresh_zero_not_the_observed_stale_zero():
    contract = load("pb_875_sl_capacity_contract.json")
    assert contract["scratch_forecast"]["derived_spool_gb"] == 2
    assert contract["scratch_forecast"]["forecast_total_bytes"] == 1024**3
    assert "fresh complete unsafeSP claim-ledger0" in contract["capacity_dependency"]["literal_parent_clause"]
    assert "not an unknown/stale offer assumption" in contract["capacity_dependency"]["literal_parent_clause"]
    assert contract["changed_fields"] == [
        "PRISMABUILD_LOCAL_SCRATCH_PAIRS and four REAL ROOT/MAX environment declarations"
    ]


def test_published_campaign_parser_accepts_complete_operational_packet():
    import pbcampaign

    rows = pbcampaign.load_manifest(
        PACKETS / "pb_875_sl_capacity_campaign.json",
        transport="pool",
        require_data_manifest=True,
    )
    assert len(rows) == 15
