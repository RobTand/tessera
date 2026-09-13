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


def test_the_packaged_contract_validates_at_v24(contract):
    """v24 adds cells; it does not move the lane schema.

    The rule the v24 changelog entry states, pinned here so the next bump has
    to decide the same question out loud: a contract_version bump that only
    ADDS cells -- or fills a ``serve_image`` a cell now attests -- is additive
    for a v10 reader and leaves ``lane_eligibility.schema`` alone.  A bump that
    changes what a field MEANS moves the schema string and fails a v10 reader
    closed, exactly as v10 did to v9.
    """
    assert int(contract["contract_version"]) == 24
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


def test_gfx1201_has_cells_and_gfx1151_still_has_none(contract):
    """A cell is a device receipt, so the two AMD entries differ by who ran.

    gfx1201 receipts were taken on an RX 9070 XT under WSL2 (#460), so that
    platform carries cells and names the image they were taken under.  Nobody
    here owns a Strix Halo part, so gfx1151 still carries none and its
    ``serve_image`` is still ``null`` -- which is the v10 state, not a gap: the
    entry already says BF16 is backed there and the quantized families are not.
    """
    block = contract["lane_eligibility"]
    served = {cell["platform"] for cell in block["cells"]}
    assert served == {"sm_121", "gfx1201"}
    assert block["platforms"]["gfx1151"]["serve_image"] is None
    assert block["platforms"]["gfx1201"]["serve_image"] is not None


def test_the_gfx1201_serve_image_is_one_its_own_cells_attest(contract):
    """v10's rule, on the platform this bump turns on."""
    block = contract["lane_eligibility"]
    attested = {cell["runtime"]["image"] for cell in block["cells"]
                if cell["platform"] == "gfx1201"}
    assert len(attested) == 1
    assert block["platforms"]["gfx1201"]["serve_image"] in attested


def test_the_gfx1201_cells_are_the_bf16_lane_only(contract):
    """Tessera-16 is the AMD lane, and the cells say only that.

    Two cells, one per regime, each covering both residencies through its
    serve flag: the residency is the axis that decides which launch a regime
    makes, and at this rung it decides nothing (rung 1792 is rate 7 and the
    window-GEMV lane serves rates 1, 2 and 4, so it refuses in both).  The
    per-residency KL receipts came back bit-identical, so there is no
    distinction here for a four-cell split to publish.
    """
    cells = [c for c in contract["lane_eligibility"]["cells"]
             if c["platform"] == "gfx1201"]
    assert [c["id"] for c in cells] == ["tessera_bf16_k1_dense_gfx1201_decode",
                                        "tessera_bf16_k1_dense_gfx1201_batch"]
    for cell in cells:
        assert cell["family"] == "TESSERA_BF16_K1"
        assert cell["structure"] == "dense"
        assert cell["rungs_q256"] == [1792]
        assert cell["activation_contract"] == "bf16_unquantized"
        assert cell["qualification"] == "device_qualified"
        assert cell["route_status"] == "backed_with_serve_flag"
        assert cell["requires_plugin"] == "tessera"
        assert cell["requires_serve_flags"] == ["TESSERA_SERVE_MODE=resident|streamed"]
        assert cell["executes"] == [{"symbol": "torch.mm", "decoder": "torch_window"}]


def test_the_gfx1201_cells_carry_a_bound_scored_in_their_own_regime(contract):
    """The grade is read off the entries, and each entry is this cell's regime.

    A prefill-scored bound written into a decode cell is what
    ``cell_evidence`` refuses by name, and it is the reason a decode-regime
    dump was taken rather than the prefill number being reused.
    """
    cells = [c for c in contract["lane_eligibility"]["cells"]
             if c["platform"] == "gfx1201"]
    for cell in cells:
        evidence = cell["evidence"]
        assert evidence["grade"] == "kl_lower_bound"
        assert len(evidence["kl"]) == 1
        entry = evidence["kl"][0]
        assert entry["kind"] == "topk_intersection_lower_bound"
        assert entry["top_k"] == 1024
        assert entry["regime"] == cell["regime"]
        assert entry["execution_modes"] == ["eager"]
        assert entry["receipt"].startswith("docs/measurements/tessera-gfx1201-")


# --------------------------------------------------------------------------
# Nothing already attested moved


def test_the_ten_sm121_cells_are_byte_identical_to_v22():
    """A version bump may add a cell; it may not edit a receipt.

    The fixture records the SHA-256 of the TEN sm_121 elements as they were
    lifted out of the v22 document by exact offsets, so this compares BYTES --
    whitespace, key order and all -- not a re-serialization that could
    normalize away a real edit.  It is the ten elements and no longer the
    whole array, because v24 appends cells on a second platform: hashing the
    array would make every later platform's arrival look like an edit to
    sm_121's receipts, which is the one thing this test exists to catch.  The
    span is taken by decoding exactly ten elements from the array's opening
    bracket, so the ten must also still be FIRST and contiguous.

    It is a digest and not a copy of the array on purpose: the array holds the
    runtime image pin, and ``tests/test_runtime_image_pin.py`` refuses a second
    copy of that digest in any file that acts.  A hash pins the same bytes
    without holding the pin, which is the whole argument that test is making.
    """
    raw = CONTRACT.read_text(encoding="utf-8")
    marker = '"cells": ['
    start = raw.index(marker) + len(marker)
    decoder = json.JSONDecoder()
    end, span = start, []
    for _ in range(10):
        while raw[end] in " \n\r\t,":
            end += 1
        value, end = decoder.raw_decode(raw, end)
        span.append(value)
    lifted = raw[start:end].encode("utf-8")
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert [cell["platform"] for cell in span] == ["sm_121"] * 10, (
        "the ten v22 cells are no longer the first ten elements of the array; "
        "this span is what the fixture's digest was taken over")
    assert len(span) == recorded["cells"] == 10
    assert len(lifted) == recorded["bytes"], (
        "the v22 sm_121 cells changed length; a version bump must not edit a receipt")
    assert hashlib.sha256(lifted).hexdigest() == recorded["sha256"], (
        "the v22 sm_121 cells changed bytes at the same length; regenerate the "
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
    """Driven on ``gfx1151``, which is the platform that still has none.

    It was driven on ``gfx1201`` until #460 minted that platform's cells.
    The mutation then hit the neighbouring rule instead -- a serve image no
    cell of the platform attests -- whose message contains this one's as a
    substring, so the test went on passing while testing something else.
    """
    def mutate(c):
        c["lane_eligibility"]["platforms"]["gfx1151"]["serve_image"] = (
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
