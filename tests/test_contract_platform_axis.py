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


def test_the_packaged_contract_validates_at_v33(contract):
    """v25 adds a top-level block; v26 and v27 do not move the lane schema either.

    The rule the v24 changelog entry states, pinned here so each bump has to
    decide the same question out loud: a contract_version bump that only ADDS
    -- cells at v24, a whole ``activation_quantizers`` table at v25, an
    optional per-format ``structures`` list at v27 -- is additive for a v10
    reader and leaves ``lane_eligibility.schema`` alone.  A bump that changes
    what a field MEANS moves the schema string and fails a v10 reader closed,
    exactly as v10 did to v9.

    v25 is the second kind of additive and the harder one to judge, so it is
    said here rather than only in the changelog: the new block is keyed by
    platform and validated against the CELLS, and it changes no value any
    existing field carries.  A v10 reader that has never heard of it resolves
    every cell with the code it already has.

    v26 (tessera#492) is the one bump in this run that moved a VALUE rather
    than adding a key -- ``tensor_parallel.units[TESSERA_E2M1_K2].loader_axes
    .row`` from refused to sharded -- and it still moved no lane-eligibility
    field: a v10 reader resolves every cell as before and only a TP planner
    reads the axis.  v27 adds ``structures`` (validated against
    ``scheme.MOE_BUILDERS``) and no cell.  v28 (#506) adds the two routed
    ``TESSERA_E2M1_K2`` cells and leaves ``tensor_parallel`` at a world of 1.
    v29 (#506, #514) raises ``tensor_parallel`` to a world of 2 on a named
    world-size receipt and publishes ``kv_head_replication`` (#330); no lane
    field moves, so both are additive for a v10 lane reader.  v30 (the A4
    whole-weight-expansion retirement) REMOVES one ``native_extensions``
    entry and moves no lane field: the extension list is a loadable-library
    table a fingerprint reads, every cell and rung stays byte for byte, and
    the retired decoder's historical receipts keep naming it.

    v31 (tessera#538) is the first bump in this run that is NOT additive and
    still does not move the lane schema: it WITHDRAWS eight dense cells whose
    ``executes`` named the retired window-GEMV dispatch, and nulls gfx1201's
    ``serve_image`` because a platform with no cells attests no image.  A v10
    reader resolves every remaining cell with the code it already has and reads
    the withdrawn combinations as absent, which is the answer the schema
    already defines for them; what changed is the ANSWER, not the grammar, so
    the schema string stays and the changelog carries the consumer warning.

    v33 (tessera#555) changes what a field MEANS outside the lane table and
    still does not move the lane schema: ``activation_quantizers`` platform
    entries go from one attestation object to a LIST of one attestation per
    image the platform publishes, because the fp4 rounding decision belongs to
    the runtime's compiled operator and two builds of one operator are two
    objects.  The activation-quantizer schema therefore moves v1 to v2 and a
    v1 reader fails closed on the name; a v10 lane reader resolves every cell
    with the code it already has, since no cell, rung, route, launch, grade,
    KL entry, qualification, TP/EP bound or served byte moves.  The consumer
    migration is to admit an fp4 cell only under an attestation whose image is
    the executing one.  NUMBERING NOTE: drafted as v32; PR #560 leg 2 landed
    its v32 first (routed reader widen), so this change is v33.

    v34 (tessera#545) is ADDITIVE for a lane reader and still does not move
    the lane schema: it MINTS four dense cells --
    ``tessera_{e4m3_k1,bf16_k1}_dense_sm121_{decode,batch}`` -- for the launch
    ``fp8_route.apply`` and ``bf16_route.apply`` have made since ``1b767a207``,
    ``tessera::window_gemm_dense`` / ``native_window_gemm``, on four served
    censuses taken on the platform's own ``serve_image``
    (``docs/measurements/tessera-window-gemm-census-2026-09-21.md``), and
    removes that pair from ``scheme.EXPERIMENTAL_LAUNCHES`` in the same change
    because ``_validate_cell_executes`` derives ``executes`` from
    ``route_launches`` with ``include_experimental=False``.  A v10 reader
    resolves the four new cells with the code it already has and reads four
    combinations that previously resolved ``unattested``.  It is NOT additive
    for a reader that derives ``executes`` from ``scheme.route_launches``
    itself.  It does not restore the gfx1201 cells v31 withdrew: no ROCm
    census of this launch exists.

    v35 (tessera#607) is ADDITIVE and moves no schema: a third sm_121
    ``activation_quantizers`` entry carries the fp4 table the routed E2M1_K2
    cells' own runtime image emitted, byte-identical to the two already
    published.  No cell, rung, route or launch moves.

    v36 (tessera#609) is ADDITIVE for a lane reader and moves no schema:
    ``scheme.MOE_BUILDERS`` gains the BF16 expert builder, so the
    ``TESSERA_BF16_K1`` format row's ``structures`` becomes ``[dense,
    routed_moe]``, and the stack's one launch (the compact adapter under the
    folded arithmetic) enters ``EXPERIMENTAL_LAUNCHES``.  No cell is minted:
    a routed BF16 stack still resolves unattested.

    v37 (tessera#614) is NOT additive for a lane reader and moves no schema:
    the dense BF16 route moves to the folded arithmetic and stamps
    ``native_window_gemm_folded``, which enters ``EXPERIMENTAL_LAUNCHES``, and
    the two ``tessera_bf16_k1_dense_sm121_{decode,batch}`` cells v34 minted on
    the epilogue kernel are withdrawn.  A v10 reader resolves dense BF16 at
    q256 1792 on sm_121 as unattested, as it did from v31 to v33.  The format
    row, its attested rung and wire stamp, and every other cell are unchanged.

    v38 (tessera#604) is NOT additive for a lane reader and moves no schema:
    eight resident, eager cells are minted on the GLM serving image for the
    dense epilogue and folded GEMMs and the compact window MoE adapter in both
    arithmetics, three pairs leave ``EXPERIMENTAL_LAUNCHES``, the FP8 routed
    materialising launch leaves ``ROUTE_LAUNCHES``, and the two routed E4M3
    cells that named it are withdrawn.  A v10 reader resolves every cell with
    the code it already has; what moved is which combinations resolve, and the
    E4M3/BF16 format rows' attested rungs.
    """
    assert int(contract["contract_version"]) == 38
    assert "activation_quantizers" in contract
    assert all("structures" in entry for entry in contract["formats"])
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


def test_neither_amd_platform_carries_a_cell_and_both_say_so(contract):
    """A cell is a device receipt, and gfx1201's was withdrawn at v31.

    gfx1201 carried two BF16 cells from #460 -- an RX 9070 XT under WSL2, the
    2026-09-13 receipt -- and they named ``torch.mm``/``torch_window``, the
    retired dense window-GEMV dispatch.  The receipt was taken three days
    before ``1b767a207`` retired it, so there is nothing on that platform that
    covers what the build launches today (tessera#538).  Nobody here owns a
    Strix Halo part, so gfx1151 never had one.

    Both entries therefore read the way v10 defines for a platform with no
    receipts -- ``serve_image: null`` -- which is not a gap: each entry still
    says BF16 is backed there and the quantized families are not.  That is the
    sentence the platform table exists to be able to form.
    """
    block = contract["lane_eligibility"]
    served = {cell["platform"] for cell in block["cells"]}
    assert served == {"sm_121"}
    for platform in ("gfx1151", "gfx1201"):
        assert block["platforms"][platform]["serve_image"] is None, platform
        assert block["platforms"][platform]["executes"]["TESSERA_BF16_K1"] == (
            "bf16_unquantized"), platform


def test_a_platform_with_no_cells_may_not_name_a_serve_image(contract):
    """v10's rule, driven on the platform the withdrawal emptied.

    It used to be checked the other way round -- gfx1201 had cells, so its
    ``serve_image`` had to be one of them.  With the cells gone the live half
    of the rule is the refusal, so it is driven rather than asserted about a
    state: put the ROCm digest back on a platform no cell attests and the
    document must refuse.
    """
    payload = _mutated(contract, lambda c: c["lane_eligibility"]["platforms"]["gfx1201"]
                       .__setitem__("serve_image", "example.invalid/rocm@sha256:" + "0" * 64))
    with pytest.raises(ValueError, match="no cell on this platform attests any image"):
        validate_serving_contract(payload)


def test_the_withdrawn_gfx1201_receipt_is_still_on_the_tree(contract):
    """The measurement outlives the cell, and is where the digest now lives.

    Withdrawing the two gfx1201 cells does not retract what was measured on
    2026-09-13; it retracts the claim that today's build executes it.  The
    receipt stays in the tree, carries the ROCm image digest the platform entry
    no longer names, and is what a re-attestation would be compared against.
    """
    receipt = ROOT / "docs/measurements/tessera-gfx1201-bf16-k1-served-2026-09-13.md"
    assert receipt.is_file()
    assert "sha256:0461258dfe253a3e0baca9c62804a4b41a21ab445b2624b6fca0d51711e14000" in (
        receipt.read_text(encoding="utf-8"))
    assert not [c for c in contract["lane_eligibility"]["cells"]
                if c["platform"] == "gfx1201"]


# --------------------------------------------------------------------------
# Nothing already attested moved


def test_the_surviving_v22_sm121_cells_are_byte_identical(contract):
    """A version bump may add a cell and may WITHDRAW one; it may not edit one.

    The fixture records the SHA-256 of the v22 ``sm_121`` elements that survive,
    lifted out of the document by exact offsets, so this compares BYTES --
    whitespace, key order and all -- not a re-serialization that could
    normalize away a real edit.  It is a span of the first elements rather than
    the whole array because later versions append cells on other scopes:
    hashing the array would make every arrival look like an edit, which is the
    one thing this test exists to catch.

    It was ten cells until contract v31, which withdrew six of them
    (tessera#538).  A withdrawal is neither an addition nor a re-measurement,
    so it is not allowed to arrive as a quietly regenerated digest: the fixture
    names the ids it dropped, and the assertions below check that every named
    id is really gone and that no other v22 cell went with them.  Regenerate
    this digest only when a receipt was deliberately re-measured; change the
    span only beside a ``withdrawn_at_v31``-style list saying what left and why.
    """
    raw = CONTRACT.read_text(encoding="utf-8")
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    marker = '"cells": ['
    start = raw.index(marker) + len(marker)
    decoder = json.JSONDecoder()
    end, span = start, []
    for _ in range(recorded["cells"]):
        while raw[end] in " \n\r\t,":
            end += 1
        value, end = decoder.raw_decode(raw, end)
        span.append(value)
    lifted = raw[start:end].encode("utf-8")
    assert [cell["platform"] for cell in span] == ["sm_121"] * recorded["cells"], (
        "the surviving v22 cells are no longer the first elements of the array; "
        "this span is what the fixture's digest was taken over")
    assert len(lifted) == recorded["bytes"], (
        "the surviving v22 sm_121 cells changed length; a version bump must not "
        "edit a receipt")
    assert hashlib.sha256(lifted).hexdigest() == recorded["sha256"], (
        "the surviving v22 sm_121 cells changed bytes at the same length; "
        "regenerate the fixture only if a receipt was deliberately re-measured")

    # The withdrawal itself, stated rather than absorbed into the digest -- and
    # the part of it contract v34 reversed.  Two of the six ids are shipped
    # again (tessera#545): a cell id is a SCOPE, so re-attesting a scope reuses
    # it, and the fixture records which two and on what receipt.  What may not
    # come back is the withdrawn CLAIM, so the launch is checked rather than
    # the spelling.
    #
    # Contract v37 (tessera#614) withdrew the re-earned pair again: the BF16
    # route serves the folded arithmetic under its own decoder, which the v34
    # receipt did not measure.
    #
    # Contract v38 (tessera#604) withdrew the two routed E4M3 cells from the
    # span, and re-earned four ids on the GLM-image census: the resident E4M3
    # dense pair v31 withdrew, and the routed pair v38 itself withdrew, each on
    # the launch the census recorded.
    present = {cell["id"] for cell in contract["lane_eligibility"]["cells"]}
    withdrawn = set(recorded["withdrawn_at_v31"]) | set(recorded["withdrawn_at_v38"])
    reearned = set(recorded["reearned_at_v34"])
    withdrawn_again = set(recorded["withdrawn_at_v37"])
    reearned_v38 = set(recorded["reearned_at_v38"])
    assert len(recorded["withdrawn_at_v31"]) == 6 and len(recorded["withdrawn_at_v38"]) == 2
    assert reearned < withdrawn and len(reearned) == 2
    assert withdrawn_again == reearned
    assert reearned_v38 < withdrawn and len(reearned_v38) == 4
    standing = (reearned - withdrawn_again) | reearned_v38
    assert not (present & (withdrawn - standing)), sorted(present & (withdrawn - standing))
    assert standing <= present
    launch = {"dense": ("tessera::window_gemm_dense", "native_window_gemm"),
              "routed_moe": ("tessera.native_window_moe.NativeWindowMoE.__call__",
                             "native_window_moe_compact")}
    for cell in contract["lane_eligibility"]["cells"]:
        if cell["id"] in standing:
            assert [(e["symbol"], e["decoder"]) for e in cell["executes"]] == [
                launch[cell["structure"]]], cell["id"]
    assert {cell["id"] for cell in span} <= present


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
    two attested images, and since v28 (#506) three: the routed E2M1_K2 cells
    name the NCCL 2.30 image their two-rank receipt ran. Taken literally the
    stronger rule refuses the contract in this repository, so the rule is
    "attested by one of the platform's own cells" instead. This test is the
    measurement.
    """
    block = contract["lane_eligibility"]
    images = {cell["runtime"]["image"] for cell in block["cells"]
              if cell["platform"] == "sm_121"}
    assert len(images) == 3, images
    assert block["platforms"]["sm_121"]["serve_image"] in images
