"""The fp4 quantizer table is READ from the kernel, and this proves it (#484).

``activation_contract`` publishes a name.  A name says the grid, the group and
the scale dtype; it does not say how a value becomes a code, and the route
hands the static global scale to vLLM's compiled ``scaled_fp4_quant``, so the
rounding decision is the runtime's.  The table exists so a consumer can read
that decision instead of re-implementing it.

Which means the tests here have an unusual shape, and it is the point:

* the INPUTS are the repository's, so an edited input is refused;
* the OUTPUTS are the runtime's, so a published code is checked for being a
  NEAREST code and never for being a PARTICULAR one.  The test that matters
  most is ``test_either_side_of_a_tie_is_accepted``: if flipping a tie to the
  other nearest code were refused here, this package would be asserting the
  tie-break, which is the defect the table was added to end.

Stdlib only -- the ``pure`` job has no torch, and the half of this table the
E2M1 and E4M3 formats themselves decide must be checkable there.
"""
from __future__ import annotations

import copy

import pytest

from tessera.alphabet import E2M1_VALUES
from tessera.serving.activation_attestation import (
    ACTIVATION_QUANTIZER_SCHEMA,
    BOUNDARIES,
    E2M1_MIDPOINTS,
    E4M3_MAX_FINITE,
    PROBES,
    bf16_bits,
    bf16_step,
    bf16_value,
    e4m3_byte_value,
    nearest_e2m1_codes,
    nearest_e4m3_bytes,
    probe_coverage,
    validate_activation_quantizers,
)
from tessera.serving.contract import (
    contract_path,
    load_serving_contract,
    require_runtime_image,
)
import json

CONTRACT_NAME = "e2m1_group16_ue4m3_static"


@pytest.fixture(scope="module")
def contract():
    return load_serving_contract()


@pytest.fixture(scope="module")
def lane(contract):
    served: dict = {}
    for cell in contract["lane_eligibility"]["cells"]:
        served.setdefault(cell["platform"], set()).add(cell["activation_contract"])
    return sorted(contract["lane_eligibility"]["platforms"]), served


@pytest.fixture(scope="module")
def block(contract):
    return contract["activation_quantizers"]


@pytest.fixture
def mutable(block):
    return copy.deepcopy(block)


def _check(candidate, lane):
    platforms, served = lane
    validate_activation_quantizers(candidate, platforms=platforms,
                                   cell_contracts=served,
                                   require_image=require_runtime_image)


def _vectors(candidate, platform="sm_121"):
    return candidate["platforms"][platform]["contracts"][CONTRACT_NAME]["vectors"]


# --- the probe spec --------------------------------------------------------
def test_the_spec_reaches_every_boundary_it_names():
    """Derived from the probes, not from a roster that would go stale."""
    assert probe_coverage(p.identifier for p in PROBES) == frozenset(BOUNDARIES)


def test_every_probe_input_is_exact_in_bf16():
    """No probe's input is itself rounded; only the arithmetic is under test."""
    for probe in PROBES:
        for bits, value in zip(probe.input_bits, probe.values):
            assert bf16_bits(value) == bits, probe.identifier


@pytest.mark.parametrize("midpoint", E2M1_MIDPOINTS)
def test_each_e2m1_midpoint_is_probed_at_two_used_scales(midpoint):
    """A dyadic used scale and a non-dyadic one.

    1.0 makes the midpoints the inputs themselves, so nothing but the
    tie-break can move the answer.  1.5 is exact in E4M3 and its reciprocal is
    not exact in F32, which is what the installed kernels' ``rcp.approx.ftz.f32``
    acts on -- so a difference between the two rows separates a tie-break from
    a reciprocal.
    """
    spec = {p.identifier: p for p in PROBES}
    dyadic = spec["midpoint_dyadic"]
    assert dyadic.implied_block_scale() == 1.0
    assert midpoint in dyadic.values and -midpoint in dyadic.values

    reciprocal = spec["midpoint_reciprocal"]
    assert reciprocal.implied_block_scale() == 1.5
    assert midpoint * 1.5 in reciprocal.values


@pytest.mark.parametrize("direction,identifier", (
    (-1, "midpoint_reciprocal_ulp_below"), (1, "midpoint_reciprocal_ulp_above")))
def test_the_ulp_probes_are_one_bf16_step_off_a_midpoint_and_are_not_ties(
        direction, identifier):
    """What separates a tie-break difference from a coarse intermediate.

    One BF16 ulp off a midpoint is unambiguous in F32 -- exactly one E2M1 code
    is nearest.  It becomes a tie only if the value is re-rounded to something
    coarser on the way in, so a code here that matches the midpoint row's
    answer is evidence about precision, not about tie-breaking.
    """
    spec = {p.identifier: p for p in PROBES}
    probe = spec[identifier]
    used = probe.implied_block_scale() / probe.global_scale
    for midpoint in E2M1_MIDPOINTS:
        nudged = bf16_value(bf16_step(bf16_bits(midpoint * used), direction))
        assert nudged in probe.values, midpoint
        # Distinct VALUES, not distinct codes: +0 and -0 are two encodings of
        # one value, so the 0.25 neighbourhood always has two nearest codes
        # and that is not the ambiguity this test is about.
        nearest = {E2M1_VALUES[c] for c in nearest_e2m1_codes(nudged / used)}
        assert len(nearest) == 1, (midpoint, nearest)


def test_the_block_scale_probes_sit_on_the_e4m3_boundaries():
    """The tie at 6*2**-10, the value above it, and the overflow past 448."""
    spec = {p.identifier: p for p in PROBES}
    assert len(nearest_e4m3_bytes(spec["block_scale_underflow_tie"]
                                  .implied_block_scale())) == 2
    assert len(nearest_e4m3_bytes(spec["block_scale_underflow_above"]
                                  .implied_block_scale())) == 1
    assert spec["block_scale_overflow"].implied_block_scale() > E4M3_MAX_FINITE


# --- the published table ---------------------------------------------------
def test_the_packaged_contract_carries_a_validated_table(block, lane):
    assert block["schema"] == ACTIVATION_QUANTIZER_SCHEMA
    _check(block, lane)


def test_the_published_inputs_are_this_package_s_probes(block):
    """Only the outputs came from the runtime; the inputs are the repository's."""
    published = {v["id"]: v for v in _vectors(block)}
    assert set(published) == {p.identifier for p in PROBES}
    for probe in PROBES:
        expected = probe.as_json()
        row = published[probe.identifier]
        assert row["input"] == expected["input"]
        assert row["global_scale"] == expected["global_scale"]
        assert row["boundary"] == expected["boundary"]


def test_the_generator_the_table_names_is_in_this_checkout(block):
    path = contract_path().resolve().parents[3] / block["generator"]
    assert path.is_file(), block["generator"]


def test_the_packaged_bytes_carry_the_block_too():
    """Read the file, not only the loader: the wheel ships these bytes."""
    raw = json.loads(contract_path().read_text())
    assert raw["activation_quantizers"]["schema"] == ACTIVATION_QUANTIZER_SCHEMA


# --- what the table is allowed to say --------------------------------------
def test_either_side_of_a_tie_is_accepted(mutable, lane):
    """THE test. The table publishes the tie-break; this package must not.

    Both nearest codes at a tie are accepted, so a kernel that changes which
    side it takes is a REGENERATION, not a refusal -- and nothing here decides
    the question the table was added to answer.
    """
    flipped = 0
    for vector in _vectors(mutable):
        probe = {p.identifier: p for p in PROBES}[vector["id"]]
        scale = e4m3_byte_value(vector["stored_scale"])
        if scale in (None, 0.0):
            continue
        used = scale / probe.global_scale
        for index, value in enumerate(probe.values):
            candidates = nearest_e2m1_codes(value / used)
            other = [c for c in candidates if c != vector["codes"][index]]
            if other:
                vector["codes"][index] = other[0]
                flipped += 1
    assert flipped, "no probe reached a tie; the midpoint rows are not doing their job"
    _check(mutable, lane)


def test_a_code_that_is_not_nearest_is_refused(mutable, lane):
    vector = _vectors(mutable)[0]
    vector["codes"][1] = (vector["codes"][1] + 4) % 8
    with pytest.raises(ValueError, match="whose nearest E2M1 code"):
        _check(mutable, lane)


def test_an_edited_input_is_refused(mutable, lane):
    _vectors(mutable)[0]["input"][1] = "0x4000"
    with pytest.raises(ValueError, match="is not the input this package's probe"):
        _check(mutable, lane)


def test_a_block_scale_that_is_not_nearest_is_refused(mutable, lane):
    for vector in _vectors(mutable):
        if vector["id"] == "midpoint_dyadic":
            vector["stored_scale"] += 2
    with pytest.raises(ValueError, match="nearest E4M3FN encoding"):
        _check(mutable, lane)


def test_a_dropped_boundary_is_refused(mutable, lane):
    contracts = mutable["platforms"]["sm_121"]["contracts"][CONTRACT_NAME]
    contracts["vectors"] = [v for v in contracts["vectors"]
                            if v["id"] != "block_scale_overflow"]
    with pytest.raises(ValueError, match="reaches no probe for"):
        _check(mutable, lane)


def test_an_unknown_probe_id_is_refused(mutable, lane):
    _vectors(mutable)[0]["id"] = "something_someone_measured_elsewhere"
    with pytest.raises(ValueError, match="not a probe this package defines"):
        _check(mutable, lane)


def test_a_contract_no_cell_on_the_platform_executes_is_refused(mutable, lane):
    entry = mutable["platforms"]["sm_121"]["contracts"]
    entry["bf16_unquantized"] = copy.deepcopy(entry[CONTRACT_NAME])
    platforms, served = lane
    served = {p: {c for c in names if c != "bf16_unquantized"}
              for p, names in served.items()}
    with pytest.raises(ValueError, match="no cell on this platform executes"):
        validate_activation_quantizers(mutable, platforms=platforms,
                                       cell_contracts=served,
                                       require_image=require_runtime_image)


def test_a_platform_the_lane_table_does_not_publish_is_refused(mutable, lane):
    mutable["platforms"]["sm_999"] = mutable["platforms"]["sm_121"]
    with pytest.raises(ValueError, match="a platform lane_eligibility does not publish"):
        _check(mutable, lane)


def test_an_unknown_field_is_refused_not_ignored(mutable, lane):
    mutable["platforms"]["sm_121"]["generated"]["measured_by"] = "someone"
    with pytest.raises(ValueError, match=r"carries unknown field\(s\)"):
        _check(mutable, lane)


# --- what the table actually answered --------------------------------------
def test_the_table_leaves_no_midpoint_undecided(block):
    """Every probed tie has a published code, which is the deliverable.

    Recorded rather than asserted against a rule: the point of the table is
    that the answer comes from the runtime, so this says the answer EXISTS and
    is one of the two admissible ones, and the values themselves are the diff a
    reviewer reads when the pinned runtime moves.
    """
    spec = {p.identifier: p for p in PROBES}
    decided = 0
    for vector in _vectors(block):
        probe = spec[vector["id"]]
        scale = e4m3_byte_value(vector["stored_scale"])
        if scale in (None, 0.0):
            continue
        used = scale / probe.global_scale
        for value, code in zip(probe.values, vector["codes"]):
            candidates = nearest_e2m1_codes(value / used)
            if len(candidates) > 1 and 0.0 not in (E2M1_VALUES[c] for c in candidates):
                assert code in candidates
                decided += 1
    assert decided >= len(E2M1_MIDPOINTS), (
        f"only {decided} genuine ties are published; the seven E2M1 midpoints "
        "should each be decided at least once")
