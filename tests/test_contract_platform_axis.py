"""The platform axis: what a platform executes, said before a cell exists.

Until schema v10 ``lane_eligibility.platforms`` was a set of KEYS, and the
only thing read of it was whether a cell's platform was declared. A cell is a
RECEIPT, so that made "has this been served here" the only sentence the table
could form -- and there is no way in v9 to say the sentence an AMD box needs
said: this family has **no** native route on this device. Silence had to
stand in for it, and a silence is not an attestation.

These tests pin the grammar of an entry, the three rules v10 adds, and the
one thing a schema bump must not do: move a byte of what was already
attested.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tessera.serving.contract import (
    LANE_ELIGIBILITY_SCHEMA,
    PLATFORM_ARCH_KEYS,
    PLATFORM_BACKENDS,
    _FAMILY_TO_ROUTE,
    load_serving_contract,
    validate_serving_contract,
)
from tessera.serving.scheme import ROUTES

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "lane_eligibility_cells_v22.json"
CONTRACT = ROOT / "src" / "tessera" / "serving" / "runtime_contract.json"


@pytest.fixture(scope="module")
def contract():
    return load_serving_contract()


def _mutated(contract, mutate):
    """A deep copy with one edit, so a refusal is attributable to that edit."""
    payload = copy.deepcopy(contract)
    mutate(payload)
    return payload


# --------------------------------------------------------------------------
# What the packaged document says


def test_the_packaged_contract_validates_at_v23(contract):
    assert int(contract["contract_version"]) == 23
    assert contract["lane_eligibility"]["schema"] == LANE_ELIGIBILITY_SCHEMA
    assert LANE_ELIGIBILITY_SCHEMA.endswith(".v10")


def test_every_platform_entry_carries_the_v10_shape(contract):
    platforms = contract["lane_eligibility"]["platforms"]
    assert set(platforms) == {"sm_121", "gfx1151", "gfx1201"}
    families = set(_FAMILY_TO_ROUTE)
    for key, entry in platforms.items():
        assert entry["backend"] in PLATFORM_BACKENDS, key
        assert PLATFORM_ARCH_KEYS[entry["backend"]] in entry, key
        assert set(entry["executes"]) == families, key
        assert "serve_image" in entry, key


def test_a_non_null_executes_value_is_its_route_s_own_constant(contract):
    """Derived, never transcribed: the value is the dispatch's or it is a
    claim about a runtime nobody read."""
    for key, entry in contract["lane_eligibility"]["platforms"].items():
        for family, value in entry["executes"].items():
            if value is None:
                continue
            assert value == ROUTES[_FAMILY_TO_ROUTE[family]]["activation_contract"], (key, family)


def test_tessera_16_is_the_amd_lane_and_the_quantized_families_are_unbacked(contract):
    """Rob's ruling, as the document now states it.

    RDNA3.5 and RDNA4 serve ``TESSERA_BF16_K1`` (W16A16) and nothing else:
    E4M3 and E2M1 have no native route on those devices under the pinned
    runtime, and ``null`` is how that is said. It is a claim somebody looked,
    which is what makes it different from a platform the table omits.
    """
    platforms = contract["lane_eligibility"]["platforms"]
    for key in ("gfx1151", "gfx1201"):
        entry = platforms[key]
        assert entry["backend"] == "hip", key
        assert entry["gcn_arch"] == key, key
        assert entry["executes"]["TESSERA_BF16_K1"] == "bf16_unquantized", key
        assert entry["executes"]["TESSERA_E4M3_K1"] is None, key
        assert entry["executes"]["TESSERA_E2M1_K2"] is None, key


def test_the_amd_platforms_carry_no_cells_and_no_serve_image(contract):
    """A cell is a device receipt and none has been taken (v23; v24 adds the
    gfx1201 cells). No ROCm image digest is a dependency of this release."""
    block = contract["lane_eligibility"]
    served = {cell["platform"] for cell in block["cells"]}
    assert served == {"sm_121"}
    for key in ("gfx1151", "gfx1201"):
        assert block["platforms"][key]["serve_image"] is None, key


# --------------------------------------------------------------------------
# Nothing already attested moved


def test_the_ten_sm121_cells_are_byte_identical_to_v22():
    """A schema bump may add a sentence; it may not edit a receipt.

    The fixture records the SHA-256 of the ``cells`` array as it was lifted
    out of the v22 document by exact offsets, so this compares BYTES --
    whitespace, key order and all -- not a re-serialization that could
    normalize away a real edit.  It is a digest and not a copy of the array
    on purpose: the array holds the runtime image pin, and
    ``tests/test_runtime_image_pin.py`` refuses a second copy of that digest
    in any file that acts.  A hash pins the same bytes without holding the
    pin, which is the whole argument that test is making.
    """
    raw = CONTRACT.read_text(encoding="utf-8")
    marker = '"cells": ['
    start = raw.index(marker) + len(marker) - 1
    value, end = json.JSONDecoder().raw_decode(raw, start)
    lifted = raw[start:end].encode("utf-8")
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert len(value) == recorded["cells"] == 10
    assert len(lifted) == recorded["bytes"], (
        "the v22 cells array changed length; a schema bump must not edit a receipt")
    assert hashlib.sha256(lifted).hexdigest() == recorded["sha256"], (
        "the v22 cells array changed bytes at the same length; regenerate the "
        "fixture only if a receipt was deliberately re-measured")


def test_no_shipped_cell_is_compile_only(contract):
    """The premise of the rule below: it cannot touch what is already here."""
    cells = contract["lane_eligibility"]["cells"]
    assert [c for c in cells if c["qualification"] == "compile_only"] == []
    assert {c["route_status"] for c in cells} == {"backed_with_serve_flag"}


# --------------------------------------------------------------------------
# The rules v10 adds, each shown refusing


def test_a_platform_entry_must_be_an_object(contract):
    payload = _mutated(contract, lambda c: c["lane_eligibility"]["platforms"].__setitem__(
        "gfx1151", {}))
    with pytest.raises(ValueError, match="backend"):
        validate_serving_contract(payload)


def test_a_platform_may_not_carry_two_architecture_spellings(contract):
    def mutate(c):
        c["lane_eligibility"]["platforms"]["gfx1151"]["compute_capability"] = [12, 0]
    with pytest.raises(ValueError, match="two devices sharing an identity"):
        validate_serving_contract(_mutated(contract, mutate))


def test_a_gcn_arch_may_not_carry_its_feature_suffixes(contract):
    def mutate(c):
        c["lane_eligibility"]["platforms"]["gfx1201"]["gcn_arch"] = "gfx1201:xnack-"
    with pytest.raises(ValueError, match="bare architecture name"):
        validate_serving_contract(_mutated(contract, mutate))


def test_executes_must_answer_for_every_family(contract):
    """A family left out is a question declined, not an unbacked route."""
    def mutate(c):
        del c["lane_eligibility"]["platforms"]["gfx1151"]["executes"]["TESSERA_E4M3_K1"]
    with pytest.raises(ValueError, match="null is the way to say unbacked"):
        validate_serving_contract(_mutated(contract, mutate))


def test_a_platform_may_not_name_a_contract_the_dispatch_does_not_run(contract):
    def mutate(c):
        c["lane_eligibility"]["platforms"]["gfx1151"]["executes"][
            "TESSERA_E4M3_K1"] = "fp8_per_token_dynamic_wna16"
    with pytest.raises(ValueError, match="does not get to name a contract"):
        validate_serving_contract(_mutated(contract, mutate))


def test_a_cell_on_a_null_entry_is_refused(contract):
    """The document contradicting itself: an unbacked platform states the
    fact in ``executes``, it does not mint cells."""
    def mutate(c):
        block = c["lane_eligibility"]
        block["platforms"]["sm_121"]["executes"]["TESSERA_E4M3_K1"] = None
    with pytest.raises(ValueError, match="the opposite claim"):
        validate_serving_contract(_mutated(contract, mutate))


def test_compile_only_cannot_carry_a_backed_route_status(contract):
    """A compile receipt proves a toolchain fact; backed needs a device."""
    def mutate(c):
        c["lane_eligibility"]["cells"][0]["qualification"] = "compile_only"
    with pytest.raises(ValueError, match="compile receipt proves a toolchain fact"):
        validate_serving_contract(_mutated(contract, mutate))


def test_compile_only_is_admissible_when_the_route_is_unbacked(contract):
    """The rule bounds the pairing; it does not ban the qualification."""
    def mutate(c):
        cell = c["lane_eligibility"]["cells"][0]
        cell["qualification"] = "compile_only"
        cell["route_status"] = "unbacked"
    validate_serving_contract(_mutated(contract, mutate))


def test_a_platform_with_cells_may_not_name_a_null_serve_image(contract):
    def mutate(c):
        c["lane_eligibility"]["platforms"]["sm_121"]["serve_image"] = None
    with pytest.raises(ValueError, match="nobody has served here"):
        validate_serving_contract(_mutated(contract, mutate))


def test_a_platform_without_cells_may_not_name_a_serve_image(contract):
    def mutate(c):
        c["lane_eligibility"]["platforms"]["gfx1201"]["serve_image"] = (
            c["versions"]["default_serve_image"])
    with pytest.raises(ValueError, match="no cell on this platform"):
        validate_serving_contract(_mutated(contract, mutate))


def test_a_serve_image_must_be_one_of_the_platform_s_own_receipts(contract):
    def mutate(c):
        c["lane_eligibility"]["platforms"]["sm_121"]["serve_image"] = (
            "vllm/vllm-openai@sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="which no cell on this platform"):
        validate_serving_contract(_mutated(contract, mutate))


def test_the_serve_image_rule_is_the_weaker_one_the_data_supports(contract):
    """Measured, not softened, and recorded here so nobody re-tightens it.

    The design asked that every cell's ``runtime.image`` equal its platform's
    ``serve_image``. The shipped document falsifies that: ``sm_121`` carries
    two attested images. Taken literally the stronger rule refuses the
    contract in this repository, so the rule is "attested by one of the
    platform's own cells" instead. This test is the measurement.
    """
    block = contract["lane_eligibility"]
    images = {cell["runtime"]["image"] for cell in block["cells"]
              if cell["platform"] == "sm_121"}
    assert len(images) == 2, images
    assert block["platforms"]["sm_121"]["serve_image"] in images
