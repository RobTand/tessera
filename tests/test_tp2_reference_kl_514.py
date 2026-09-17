"""The tessera#514 resolution is bound to the receipt's bytes and derives its ratios.

``experiments/results/glm53_a4_stub_tp2_reference_kl_514.json`` is the reading
that closed tessera#514: the A4 stub's distance to the BF16 stub at a world of
one and of two, from the SAME four ``kl_tool`` payloads the world-size receipt
(``glm53_a4_stub_tp_single_rank_kl_514.json``, contract v29) names.  Two
properties keep it honest, and they are the rule rather than the roster:

* it names the receipt's payloads by sha256 -- a resolution measured on other
  dumps would be about another serve;
* its headline ratio is derived from the means beside it, exactly as the
  contract derives ``excess_over_control`` from the arms, so a typed number
  that disagrees with its own table is caught here.

The measurement page that reads it must name the issue and the grade the
receipt keeps, because the resolution changes the READING of the receipt's
4.26x and not the receipt.
"""
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
RESOLUTION = _REPO / "experiments/results/glm53_a4_stub_tp2_reference_kl_514.json"
RECEIPT_TABLE = _REPO / "experiments/results/glm53_a4_stub_tp_single_rank_kl_514.json"
PAGE = _REPO / "docs/measurements/tessera-glm53-a4-stub-tp2-excess-resolved-2026-09-17.md"


@pytest.fixture(scope="module")
def resolution():
    return json.loads(RESOLUTION.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def receipt_table():
    return json.loads(RECEIPT_TABLE.read_text(encoding="utf-8"))


def test_the_resolution_reads_the_receipts_own_payloads(resolution, receipt_table):
    assert resolution["issue"] == receipt_table["issue"] == "tessera#514"
    assert resolution["resolves"] == RECEIPT_TABLE.relative_to(_REPO).as_posix()
    arms = receipt_table["arms"]
    expected = {
        "a4_tp1": arms["a4_tp2_vs_tp1"]["reference"]["npz_sha256"],
        "a4_tp2": arms["a4_tp2_vs_tp1"]["compared"]["npz_sha256"],
        "bf16_tp1": arms["bf16_tp2_vs_tp1"]["reference"]["npz_sha256"],
        "bf16_tp2": arms["bf16_tp2_vs_tp1"]["compared"]["npz_sha256"],
    }
    assert {k: v["npz_sha256"] for k, v in resolution["payloads"].items()} == expected
    assert resolution["positions"] == arms["a4_tp2_vs_tp1"]["positions"]


def test_the_headline_ratio_is_derived_from_its_own_means(resolution):
    ref = resolution["reference_kl"]
    assert ref["ratio_tp2_over_tp1"] == round(ref["tp2_mean"] / ref["tp1_mean"], 4)
    against = ref["against_bf16_tp2"]
    assert against["ratio_tp2_over_tp1"] == round(against["tp2_mean"] / against["tp1_mean"], 4)
    low, high = ref["ratio_ci95"]
    assert low <= ref["ratio_tp2_over_tp1"] <= high


def test_every_decomposition_value_sits_inside_its_own_interval(resolution):
    for key, entry in resolution["decomposition"].items():
        if not isinstance(entry, dict):
            continue
        low, high = entry["ci95"]
        assert low <= entry["value"] <= high, key


def test_the_page_names_the_issue_and_keeps_the_grade():
    text = PAGE.read_text(encoding="utf-8")
    assert "tessera#514" in text
    assert "route_only" in text
    assert RESOLUTION.name in text
