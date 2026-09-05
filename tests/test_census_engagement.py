"""An arm that requested a route and got zero units on it is a failed census.

THE DEFECT THIS PINS (#104).  All four of #91's serve censuses logged 112 of
112 modules refusing the window-GEMV lane at load -- including the arm that was
supposed to be TAKING it -- and every receipt recorded one route, ``symbol
torch._scaled_mm``, ``decoder torch_window``, ``problems: []``.  The two arms
were one lane state wearing two names: the experiment compared a thing against
itself and reported agreement.

The per-module check cannot catch that, and not by oversight: the streamed FP8
route's decode regime legitimately admits both the GEMV pair and the
materialised one (a rate-1 unit and a box with no toolchain both fall back
inside that regime), so a serve in which the lane prepared for NOTHING passes
module by module.  What was missing is the question about the arm as a whole,
and it must be a value rather than a log line.

THE FAIL-BEFORE.  On the pre-#104 tree ``tessera.serving.census`` does not
exist, so every test here fails at import.
"""
from __future__ import annotations

import json

import pytest

from tessera.serving import bf16_route
from tessera.serving.census import LANE_ENGAGEMENT_SCHEMA, decoder_histogram, lane_engagement

LANE = "tessera_window_gemv"
DECODERS = {LANE: "window_gemv"}


def _records(n, decoder, *, symbol="torch._scaled_mm"):
    """One phase's records, in the shape ``read_route`` returns."""
    return {f"model.layers.{i}.mlp.down_proj": {
        "kind": "dense", "policy": "TESSERA_FP8:streamed", "symbol": symbol,
        "tile_m": 0, "shape": "M1:N1024:K1024", "contract": "fp8_per_token_dynamic",
        "state": "served", "reason": None, "decoder": decoder} for i in range(n)}


# --------------------------------------------------------------------------
# the void census
# --------------------------------------------------------------------------

def test_the_void_census_of_issue_104_is_now_a_refusal():
    """112 modules on ``torch_window``, the GEMV lane required: REFUSED.

    This is the exact receipt shape all four #91 censuses wrote.
    """
    phases = {"prefill": _records(112, "torch_window"),
              "decode": _records(112, "torch_window")}
    block, problems = lane_engagement(phases, required_lanes=[LANE], lane_decoders=DECODERS)
    assert block["schema"] == LANE_ENGAGEMENT_SCHEMA
    assert block["all_required_engaged"] is False
    assert block["engaged_modules_max"] == {LANE: 0}
    for phase in ("prefill", "decode"):
        assert block["phases"][phase]["engaged_modules"] == {LANE: 0}
        assert block["phases"][phase]["decoders"] == {"torch_window": 112}
        assert block["phases"][phase]["unengaged_required_lanes"] == [LANE]
    # ONE problem, over the census -- see the decode-only test below for why it
    # is not one per phase.
    assert len(problems) == 1
    assert "0 of 112" in problems[0] and LANE in problems[0]
    assert "in any phase" in problems[0]
    assert "prefill" in problems[0] and "decode" in problems[0]
    assert "measured the fallback, not the lane" in problems[0]


def test_the_refusal_carries_the_load_time_reason():
    """"The lane is unreached" and "nobody looked" must not read the same."""
    refusal = (f"{LANE}: GrammarError: rates [3] have no lane here "
               "(supported (1, 2, 4)); the materialised FP8 path serves this unit")
    phases = {"decode": _records(112, "torch_window")}
    refusals = {"decode": {name: refusal for name in phases["decode"]}}
    block, problems = lane_engagement(phases, required_lanes=[LANE], lane_decoders=DECODERS,
                                      refusals_by_phase=refusals)
    assert block["phases"]["decode"]["lane_refusals"] == {refusal: 112}
    assert len(problems) == 1
    assert "rates [3] have no lane here" in problems[0]
    assert "112 of 112 module(s) recorded a load-time refusal" in problems[0]


def test_a_load_time_refusal_is_not_multiplied_by_the_phase_count():
    """The tool hands the SAME load-time map to every phase (a refusal happens
    once, at load).  Summing them reported 224 of 112 modules refusing."""
    refusal = f"{LANE}: GrammarError: rates [3] have no lane here (supported (1, 2, 4))"
    records = _records(112, "torch_window")
    phases = {"prefill": records, "decode": records}
    refusals = {phase: {name: refusal for name in records} for phase in phases}
    _block, problems = lane_engagement(phases, required_lanes=[LANE], lane_decoders=DECODERS,
                                       refusals_by_phase=refusals)
    assert len(problems) == 1
    assert "112 of 112 module(s) recorded a load-time refusal" in problems[0]
    assert "224" not in problems[0]


def test_an_engaged_arm_passes():
    """The control: the same check on a serve that DID take the lane."""
    phases = {"prefill": _records(112, "window_gemv"),
              "decode": _records(112, "window_gemv", symbol="tessera_window_gemv::gemv")}
    block, problems = lane_engagement(phases, required_lanes=[LANE], lane_decoders=DECODERS)
    assert problems == []
    assert block["all_required_engaged"] is True
    assert block["engaged_modules_max"] == {LANE: 112}


def test_a_decode_only_lane_is_engaged_and_not_a_failed_census():
    """THE SHAPE OF EVERY CORRECT SERVE OF THIS LANE.  The window GEMV decodes
    M <= 8 (``kernel_window_gemv.GEMV_MAX_M``), so the census's prefill forward
    takes the torch decode BY DESIGN.  A per-phase verdict would refuse the one
    arm that works, which is the fastest way to teach a reader to ignore the
    field -- so the verdict is over the census and the per-phase counts are
    still in the block."""
    phases = {"prefill": _records(112, "torch_window"),
              "decode": _records(112, "window_gemv", symbol="tessera_window_gemv::gemv")}
    block, problems = lane_engagement(phases, required_lanes=[LANE], lane_decoders=DECODERS)
    assert problems == []
    assert block["all_required_engaged"] is True
    assert block["engaged_modules_max"] == {LANE: 112}
    assert block["phases"]["prefill"]["engaged_modules"] == {LANE: 0}
    assert block["phases"]["prefill"]["unengaged_required_lanes"] == [LANE]
    assert block["phases"]["decode"]["engaged_modules"] == {LANE: 112}


def test_a_partly_engaged_arm_passes_because_the_fallback_is_legitimate():
    """Rate-1 units and toolchain-less boxes fall back for real; the gate asks
    whether the lane took ANY units, not whether it took all of them."""
    phases = {"decode": {**_records(100, "window_gemv"),
                         **{f"other.{i}": r for i, r in
                            enumerate(_records(12, "torch_window").values())}}}
    block, problems = lane_engagement(phases, required_lanes=[LANE], lane_decoders=DECODERS)
    assert problems == []
    assert block["phases"]["decode"]["decoders"] == {"torch_window": 12, "window_gemv": 100}


def test_requiring_nothing_is_three_valued_and_not_a_pass():
    """A receipt that never said what to require must not read as "engaged"."""
    phases = {"decode": _records(112, "torch_window")}
    block, problems = lane_engagement(phases)
    assert problems == []
    assert block["all_required_engaged"] is None
    assert block["required_lanes"] == []
    # ... and it still carries the counts a gate would need.
    assert block["phases"]["decode"]["decoders"] == {"torch_window": 112}


def test_a_record_with_no_decoder_is_not_absorbed_into_a_total():
    phases = {"decode": _records(3, None)}
    assert decoder_histogram(phases["decode"]) == {"": 3}


def test_the_block_is_json_serialisable():
    """It is a receipt field; a gate reads it out of a file."""
    block, _ = lane_engagement({"decode": _records(2, "torch_window")},
                               required_lanes=[LANE], lane_decoders=DECODERS)
    assert json.loads(json.dumps(block))["all_required_engaged"] is False


def test_a_required_lane_with_no_decoder_is_a_refusal_not_a_zero():
    with pytest.raises(ValueError, match="no decoder given"):
        lane_engagement({"decode": _records(1, "torch_window")},
                        required_lanes=["tessera_unknown"], lane_decoders={})


def test_the_lane_decoder_defaults_to_the_packaged_contract():
    """The decoder a lane stamps is read from the contract, never restated by
    the caller: a lane rename fails at the contract rather than counting zero
    for a name nothing writes."""
    block, problems = lane_engagement({"decode": _records(4, "window_gemv")},
                                      required_lanes=[LANE])
    assert problems == []
    assert block["required_decoders"] == {LANE: "window_gemv"}


# --------------------------------------------------------------------------
# the load-time half: a lane refusal that leaves a trace
# --------------------------------------------------------------------------

class _Unit:
    """A parsed-unit stand-in carrying every fact the published lane
    predicate reads: since #264 the gate decides ALL of it (one home,
    ``scheme.decide_lane_requirements``), not just rates/window/state."""

    def __init__(self, rates, window_bits=14, initial_state=None):
        from types import SimpleNamespace

        import torch

        self.unit = SimpleNamespace(
            body="WINDOW", scale_plane="CHANNEL",
            release_index=torch.zeros(0, dtype=torch.int64), diagonals=None,
            rotation=None, rates=tuple(rates), window_bits=window_bits,
            initial_state=initial_state)
        self.grid = SimpleNamespace(arity=1)


def test_the_bf16_lane_gate_names_its_reason():
    """This route's lane did not RAISE when it did not apply -- it just took the
    fallback -- so there was nothing for a receipt to aggregate.  The verdict
    is unchanged; what is new is that it comes with the reason."""
    assert bf16_route.gemv_refusal_for_unit(_Unit((4, 4))) is None
    assert bf16_route.gemv_eligible_for_unit(_Unit((4, 4)))
    rate = bf16_route.gemv_refusal_for_unit(_Unit((3, 4)))
    assert rate is not None and "[3]" in rate
    assert not bf16_route.gemv_eligible_for_unit(_Unit((3, 4)))
    window = bf16_route.gemv_refusal_for_unit(_Unit((4, 4), window_bits=16))
    assert window is not None and "window_bits 16" in window
    shard = bf16_route.gemv_refusal_for_unit(_Unit((4, 4), initial_state=object()))
    assert shard is not None and "shard start state" in shard


def test_a_lane_refusal_is_a_value_on_the_layer_not_a_stderr_line():
    from tessera.serving import telemetry

    class _Layer:
        pass

    layer = _Layer()
    assert telemetry.read_lane_refusal(layer) is None
    telemetry.note_lane_refusal(layer, "tessera_window_gemv", "GrammarError: rates [3]")
    assert telemetry.read_lane_refusal(layer) == "tessera_window_gemv: GrammarError: rates [3]"
    # Cleared on a re-prepare, so no stale refusal survives into a second load.
    telemetry.note_lane_refusal(layer, "tessera_window_gemv", None)
    assert telemetry.read_lane_refusal(layer) is None


def test_the_routes_record_the_refusal_where_they_take_the_fallback():
    """The wiring, pinned by source: both streamed routes must note the refusal
    on the path that substitutes the torch decode."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "tessera" / "serving"
    for name in ("fp8_route.py", "bf16_route.py"):
        assert "note_lane_refusal(layer," in (root / name).read_text(), name


def test_the_census_requires_the_lane_the_artifact_declares():
    """The other end of the stamp: ``requires_lanes`` in the manifest is what
    the census reads, so a claim about a lane travels with the bytes."""
    from pathlib import Path

    tool = (Path(__file__).resolve().parents[1] / "tools" / "tessera_route_census.py").read_text()
    assert 'get("requires_lanes")' in tool
    assert "lane_engagement" in tool and "--require-lane" in tool
