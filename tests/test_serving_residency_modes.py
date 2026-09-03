"""``formats[].residency_modes`` is validated against lane.MODES.

The formats[] validator required ``residency_modes`` but never examined it:
a contract claiming ``["resident"]`` -- or inventing a third mode -- passed
``validate_serving_contract`` with a green validator, while ``lane`` is what
the serve gates on (``serve_mode``, ``build_tessera_method``).  It was the
only cross-artifact claim of its kind with no document-vs-code validator;
``loader_axes`` vs ``ROUTE_TP_AXES`` is the pattern this follows.

Every expectation below derives from ``lane.MODES`` itself.  The mutation
tests fail on the unfixed validator (``DID NOT RAISE``) because the field had
no reader at all.
"""
from __future__ import annotations

import copy

import pytest

from tessera.serving.contract import (                             # noqa: E402
    load_serving_contract,
    validate_serving_contract,
)
from tessera.serving.lane import MODES                              # noqa: E402


@pytest.fixture(scope="module")
def contract():
    return load_serving_contract()


def _with_modes(contract, modes, row=0):
    bad = copy.deepcopy(contract)
    bad["formats"][row]["residency_modes"] = modes
    return bad


def _invented():
    """A mode string the lane does not serve, derived, not typed."""
    for candidate in ("cryogenic", "RESIDENT", "resident ", "", "streamed+resident"):
        if candidate not in MODES:
            return candidate
    raise AssertionError(f"no invented mode left beside {MODES}")


def test_packaged_rows_serve_exactly_the_modes_the_lane_gates_on(contract):
    """The shipped file, read off the code that serves it -- the same shape
    as the ``loader_axes IS ROUTE_TP_AXES`` check, not a restated roster."""
    assert contract["formats"], "the packaged contract publishes no formats[] rows"
    for entry in contract["formats"]:
        assert sorted(entry["residency_modes"]) == sorted(MODES), (
            f"{entry['family']}: the row claims {entry['residency_modes']} but the "
            f"lane serves {list(MODES)}")


def test_an_invented_residency_mode_is_refused(contract):
    """A producer inventing a third mode -- or a typo -- must not validate."""
    bad = _with_modes(contract, list(MODES) + [_invented()])
    with pytest.raises(ValueError, match="residency_modes"):
        validate_serving_contract(bad)


def test_every_row_gets_the_invented_mode_check(contract):
    """The mechanism, not the state: each formats[] row is checked."""
    seen = 0
    for i, entry in enumerate(contract["formats"]):
        bad = _with_modes(contract, list(MODES) + [_invented()], row=i)
        with pytest.raises(ValueError, match="residency_modes"):
            validate_serving_contract(bad)
        seen += 1
    assert seen, "the packaged contract publishes no formats[] rows"


def test_an_empty_residency_list_is_refused(contract):
    """A row that serves nowhere is a dead row, not a contract."""
    with pytest.raises(ValueError, match="residency_modes"):
        validate_serving_contract(_with_modes(contract, []))


def test_a_non_list_residency_is_refused(contract):
    with pytest.raises(ValueError, match="residency_modes"):
        validate_serving_contract(_with_modes(contract, "resident"))


def test_a_duplicated_residency_is_refused(contract):
    """Two names for one residency is a typo with a green validator."""
    with pytest.raises(ValueError, match="residency_modes"):
        validate_serving_contract(_with_modes(contract, list(MODES) + [MODES[0]]))


def test_a_subset_of_modes_still_validates(contract):
    """The rule is membership, not equality: a family the build serves in
    one residency only is an honest row, not a refusal."""
    if len(MODES) < 2:
        pytest.skip("the lane serves a single mode; there is no subset to admit")
    ok = copy.deepcopy(contract)
    ok["formats"][0]["residency_modes"] = [MODES[0]]
    validate_serving_contract(ok)
