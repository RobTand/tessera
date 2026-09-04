"""A lane an artifact could never reach is refused where the PLAN is made (#104).

THE DEFECT THIS PINS.  ``prepare_fp8_gemv`` requires every column rate of a
unit to be in ``kernel_window_gemv.SUPPORTED_RATES`` -- and a rung is a ROOT
rate that ``grammar.bresenham_rate_schedule`` realises by mixing the two rates
bracketing it, so q256 1006 (root 3.93) is columns at rate 3 and columns at
rate 4.  Every unit of such a checkpoint therefore refuses the window-GEMV
lane at LOAD, one module at a time, and the streamed FP8 route catches the
refusal and serves the same bytes through the torch window decode -- a
substitution the module reports as ``state: served``.  All six allocated
checkpoints under ``/mnt/shared/tessera-runs/allocated`` carried a rate outside
the set, so no artifact we held could exercise the lane at all, while two
issues were trying to measure it.

THE INVARIANT.  Reachability is a function of the RUNG alone, so it is
decidable before a shape is read -- and it is refused there.  The bound comes
from the packaged contract, and the contract's copy IS the kernel's constant.

THE FAIL-BEFORE.  On the pre-#104 tree ``refuse_unreachable_lane`` and
``lane.requires`` do not exist and ``--require-lane`` is not an argument, so
every test below fails at import or at the missing refusal.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from fractions import Fraction
from pathlib import Path

import pytest

from tessera import kernel_window_gemv as kg
from tessera.control import grid_for_name
from tessera.export import wire_recipe
from tessera.grammar import bresenham_rate_schedule, rate_set, root_from_q256
from tessera.serving import ext
from tessera.serving.contract import (
    extension_lane, lane_decoder, lane_requirements, load_serving_contract,
    validate_serving_contract)
from tessera.serving.scheme import lane_rate_report, refuse_unreachable_lane

ROOT = Path(__file__).resolve().parents[1]
LANE = ext.WINDOW_GEMV_MODULE_NAME
E4M3 = grid_for_name("E4M3")


def _exporter():
    spec = importlib.util.spec_from_file_location(
        "export_tessera_serving", ROOT / "experiments" / "export_tessera_serving.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# the contract's copy of the kernel's constants
# --------------------------------------------------------------------------

def test_the_published_lane_predicate_is_the_kernels_own_constants():
    """Not "agrees in prose": the same values, or this test is the failure.

    ``ext`` is read by a producer with no torch, so the numbers are literals
    there and ``kernel_window_gemv`` cannot be imported to build them.  That
    is exactly the drift ``loader_axes`` vs ``ROUTE_TP_AXES`` is tied against,
    and this is the same tie.
    """
    requires = lane_requirements(LANE)
    assert tuple(requires["column_rates"]) == tuple(kg.SUPPORTED_RATES)
    assert tuple(requires["window_bits"]) == tuple(kg.WINDOW_BITS_SUPPORTED)


def test_the_lane_decoder_is_a_decoder_the_telemetry_knows():
    from tessera.serving import telemetry

    assert lane_decoder(LANE) == telemetry.DECODER_WINDOW_GEMV
    for entry in ext.NATIVE_EXTENSIONS:
        assert entry["lane"]["decoder"] in telemetry.DECODERS


def test_the_packaged_json_publishes_the_lane_block():
    payload = json.loads(
        (ROOT / "src" / "tessera" / "serving" / "runtime_contract.json").read_text())
    assert payload["native_extensions"] == ext.NATIVE_EXTENSIONS
    assert int(payload["contract_version"]) >= 11


def _mutated_lane(monkeypatch, mutate):
    """A contract whose lane block is edited, with ``ext`` edited to match.

    The validator's FIRST check is authority -- the block must equal
    ``ext.NATIVE_EXTENSIONS`` -- so a bare mutation only ever exercises that
    check.  Patching ``ext`` to the same mutated list gets the payload past
    authority so the field checks below it are the ones under test.
    """
    payload = copy.deepcopy(load_serving_contract())
    entry = next(e for e in payload["native_extensions"]
                 if e["module_name_prefix"] == LANE)
    mutate(entry)
    monkeypatch.setattr(ext, "NATIVE_EXTENSIONS", payload["native_extensions"])
    return payload


def test_an_empty_requires_block_is_refused(monkeypatch):
    """"No constraint" and "nobody wrote the constraint down" must not read alike."""
    def _empty(entry):
        entry["lane"]["requires"] = {}

    payload = _mutated_lane(monkeypatch, _empty)
    with pytest.raises(ValueError, match="omits the block"):
        validate_serving_contract(payload)


def test_a_lane_predicate_a_gate_cannot_resolve_is_refused(monkeypatch):
    def _bad_body(entry):
        entry["lane"]["requires"]["body"] = "trellis"

    payload = _mutated_lane(monkeypatch, _bad_body)
    with pytest.raises(ValueError, match="not a body this package names"):
        validate_serving_contract(payload)


def test_the_authority_check_still_comes_first(monkeypatch):
    """Without the patch above, an edited block is refused as NOT WHAT THIS
    BUILD LOADS -- pinned so the helper is never mistaken for a loosening."""
    payload = copy.deepcopy(load_serving_contract())
    payload["native_extensions"][-1]["lane"]["requires"] = {}
    with pytest.raises(ValueError, match="not what this build loads"):
        validate_serving_contract(payload)


def test_an_unpublished_lane_raises_rather_than_reading_as_unconstrained():
    with pytest.raises(ValueError, match="publishes no native extension"):
        extension_lane("tessera_no_such_lane")


# --------------------------------------------------------------------------
# the rate set a rung implies
# --------------------------------------------------------------------------

@pytest.mark.parametrize("q256", [256, 300, 384, 512, 750, 768, 1006, 1024, 1044, 1262, 1792])
def test_rate_set_is_the_schedules_own_distinct_rates(q256):
    """``rate_set`` must be column-count-free AND right, or the gate is guessing."""
    root = root_from_q256(q256)
    want = rate_set(root, cap=15)
    for columns in (256, 512, 1024, 2048, 4096):
        try:
            schedule = bresenham_rate_schedule(root, columns, cap=15)
        except Exception:
            continue
        assert tuple(sorted(set(schedule))) == tuple(sorted(want)), (q256, columns)


def test_an_integral_root_is_one_rate_and_a_fractional_root_is_two():
    assert rate_set(Fraction(4), cap=7) == (4,)
    assert rate_set(Fraction(1006, 256), cap=7) == (3, 4)


# --------------------------------------------------------------------------
# the plan-time refusal
# --------------------------------------------------------------------------

def _refuse(q256):
    recipe = wire_recipe(E4M3, q256)
    return refuse_unreachable_lane(
        LANE, grid="E4M3", q256=q256, rate_cap=E4M3.rate_cap,
        body=recipe.body.name, plane=recipe.scale_plane.name,
        window_bits=int(recipe.window_bits), target=f"probe@q{q256}")


@pytest.mark.parametrize("q256,offending", [(750, 3), (1006, 3), (1262, 5), (768, 3)])
def test_the_rungs_every_allocated_checkpoint_carries_are_refused_by_name(q256, offending):
    """The four rungs of the #104 table, and the offending rate is NAMED.

    A refusal that does not say which rate is out of range leaves the producer
    to guess a rung, which is what "either widen the kernel or re-plan" turns
    into without it.
    """
    with pytest.raises(ValueError) as exc:
        _refuse(q256)
    message = str(exc.value)
    assert f"[{offending}]" in message
    assert "column_rates" in message and str(list(kg.SUPPORTED_RATES)) in message


@pytest.mark.parametrize("q256,rates", [(256, (1,)), (384, (1, 2)), (512, (2,)), (1024, (4,))])
def test_the_rungs_inside_the_lane_are_accepted(q256, rates):
    assert _refuse(q256) == rates


def test_the_refusal_is_a_property_of_the_rung_not_of_the_shape():
    """No column count anywhere in the call: that is what lets it run first."""
    import inspect

    assert "columns" not in inspect.signature(refuse_unreachable_lane).parameters


def test_a_window_the_lanes_table_cannot_hold_is_refused():
    with pytest.raises(ValueError, match="window bits"):
        refuse_unreachable_lane(LANE, grid="BF16", q256=1024, rate_cap=15,
                                body="WINDOW", plane="CHANNEL", window_bits=16,
                                target="probe")


def test_a_body_the_lane_does_not_read_is_refused():
    with pytest.raises(ValueError, match="WINDOW body"):
        refuse_unreachable_lane(LANE, grid="E2M1x2", q256=896, rate_cap=7,
                                body="TCQ", plane="LUT", window_bits=0, target="probe")


def test_lane_rate_report_is_the_value_both_sides_read():
    ok = lane_rate_report(LANE, (4, 4, 4))
    assert ok["reachable"] and ok["offending"] == []
    bad = lane_rate_report(LANE, (3, 4))
    assert not bad["reachable"] and bad["offending"] == [3]
    assert bad["supported"] == list(kg.SUPPORTED_RATES)


# --------------------------------------------------------------------------
# the exporter refuses BEFORE the encode
# --------------------------------------------------------------------------

def test_the_exporter_refuses_the_plan_before_it_reads_a_shape(tmp_path, monkeypatch):
    """``--require-lane`` on an unreachable rung must exit at argument time.

    Proven by making the encode itself explode: ``quantizable`` is the first
    thing ``main`` calls after the gates, and it is monkeypatched to raise.  A
    refusal that arrives after that call is not a refusal; it is a bill.
    """
    export = _exporter()

    def _never(*a, **k):
        raise AssertionError("the exporter read the checkpoint before refusing the plan")

    monkeypatch.setattr(export, "quantizable", _never)
    monkeypatch.setattr(
        "sys.argv",
        ["export_tessera_serving.py", str(tmp_path / "src"), str(tmp_path / "out"),
         "--grid", "E4M3", "--q256", "1006", "--require-lane", LANE])
    with pytest.raises(SystemExit) as exc:
        export.main()
    message = str(exc.value)
    assert "[3]" in message and LANE in message
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_the_exporter_accepts_the_plan_at_a_rung_the_lane_reads(tmp_path, monkeypatch):
    """The same call at q256 1024 must get PAST the gate -- else the test above
    would pass on a gate that refuses everything."""
    export = _exporter()
    reached = []

    def _mark(*a, **k):
        reached.append(True)
        raise SystemExit("reached the checkpoint read")

    monkeypatch.setattr(export, "quantizable", _mark)
    # A real (empty) config.json rather than a patched ``Path.read_text``: the
    # gate itself reads the packaged contract through that same method.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "config.json").write_text("{}")
    monkeypatch.setattr(
        "sys.argv",
        ["export_tessera_serving.py", str(tmp_path / "src"), str(tmp_path / "out"),
         "--grid", "E4M3", "--q256", "1024", "--require-lane", LANE])
    with pytest.raises(SystemExit):
        export.main()
    assert reached, "the q256=1024 plan was refused, so the gate is not rung-specific"


def test_require_lane_is_stamped_into_the_artifact():
    """The requirement must travel with the BYTES, not with a shell history."""
    source = (ROOT / "experiments" / "export_tessera_serving.py").read_text()
    assert '"requires_lanes": required_lanes,' in source


# --------------------------------------------------------------------------
# the CELL that names the lane, and the tie that stops it drifting (#111)
# --------------------------------------------------------------------------
#
# THE DEFECT THIS HALF PINS.  The E4M3 family's cells published
# ``scaled_mm_w8a8`` for both regimes -- in the cell ``id``, the only place a
# launch appeared -- and no cell named the window-GEMV lane at all.  That was
# accidentally true while the lane was unreachable (the first half of this
# file is why), and false the moment a rate-constrained artifact was served:
# ``/home/rob/tessera-runs/ts104/census-R1024-readable.json`` records
# ``tessera_window_gemv::gemv`` on 112 of 112 modules in the decode regime.
# So the contract answered "what does an E4M3 decode execute" with a value the
# runtime had stopped executing -- principle 14 from the other direction.
#
# Schema v4 puts the launch in ``executes``, and the validator DERIVES it from
# ``scheme.ROUTE_LAUNCHES`` narrowed by the lanes each rung reaches under
# ``lane.requires``.  Together with
# ``test_the_published_lane_predicate_is_the_kernels_own_constants`` above,
# that makes a chain the tests below break at each link: cell -> requires ->
# ``kernel_window_gemv``'s own constants.


def _gemv_cell(contract):
    """The one cell whose ``executes`` names the lane's own op."""
    cells = [c for c in contract["lane_eligibility"]["cells"]
             if any(e["decoder"] == telemetry_decoder() for e in c["executes"])
             and any(e["symbol"] == kg_symbol() for e in c["executes"])]
    assert len(cells) == 1, [c["id"] for c in cells]
    return cells[0]


def telemetry_decoder():
    from tessera.serving.contract import lane_decoder

    return lane_decoder(LANE)


def kg_symbol():
    from tessera.serving.scheme import WINDOW_GEMV_SYMBOL

    return WINDOW_GEMV_SYMBOL


def test_a_cell_names_the_lane_the_census_observed():
    """The shipped table says the E4M3 decode regime runs the GEMV, in streamed.

    The positive half.  Without it every refusal below would pass on a table
    that named the lane nowhere, which is the state #111 was filed on.
    """
    from tessera.serving.contract import cell_residency_modes

    cell = _gemv_cell(load_serving_contract())
    assert cell["family"] == "TESSERA_E4M3_K1"
    assert cell["regime"] == "decode"
    assert cell["rungs_q256"] == [1024]
    assert cell_residency_modes(cell) == ("streamed",)
    # The lane exists in streamed alone -- both window routes set
    # ``layer.tessera_gemv = None`` in resident -- so a cell claiming it in
    # both residencies would be claiming a launch the resident path cannot make.
    assert cell["requires_serve_flags"] == ["TESSERA_SERVE_MODE=streamed"]


def test_every_rung_a_gemv_cell_claims_is_one_the_lane_actually_reads():
    """Straight through ``refuse_unreachable_lane``, not through a restated set."""
    contract = load_serving_contract()
    cell = _gemv_cell(contract)
    wires = {int(w["q256"]): w
             for entry in contract["formats"] if entry["family"] == cell["family"]
             for w in entry["attested_wire"]}
    for rung in cell["rungs_q256"]:
        wire = wires[int(rung)]
        assert refuse_unreachable_lane(
            LANE, grid="E4M3", q256=int(rung), rate_cap=E4M3.rate_cap,
            body=wire_recipe(E4M3, int(rung)).body.name,
            plane=wire_recipe(E4M3, int(rung)).scale_plane.name,
            window_bits=int(wire["window_bits"]), target=f"cell {cell['id']}")


def test_a_lane_that_stops_reading_the_rung_invalidates_the_cell(monkeypatch):
    """The drift the issue asked for, broken at the kernel end.

    ``lane.requires.column_rates`` IS ``kernel_window_gemv.SUPPORTED_RATES``
    (pinned above).  Drop rate 4 from it -- what the kernel losing its 4-bit
    lane would do -- and the attested rung 1024 stops being readable, so the
    derivation says the decode regime executes the materialised pair while the
    cell still claims the GEMV.  The contract must refuse, not resolve.
    """
    def _no_rate_four(entry):
        entry["lane"]["requires"]["column_rates"] = [1, 2]

    payload = _mutated_lane(monkeypatch, _no_rate_four)
    with pytest.raises(ValueError, match="executes"):
        validate_serving_contract(payload)


def test_a_lane_that_stops_reading_the_window_invalidates_the_cell(monkeypatch):
    """The same tie on the other published constant, ``WINDOW_BITS_SUPPORTED``."""
    def _no_l14(entry):
        entry["lane"]["requires"]["window_bits"] = [12]

    payload = _mutated_lane(monkeypatch, _no_l14)
    with pytest.raises(ValueError, match="executes"):
        validate_serving_contract(payload)


def test_a_gemv_cell_cannot_acquire_an_unreadable_rung():
    """The drift broken at the CELL end: attest 1006 and claim the lane on it.

    1006 is the rung every allocated checkpoint carried -- root 3.93, columns
    at rates 3 and 4 -- so no unit of it can take the lane.  Attesting it (the
    family row and its wire stamp, as a receipt would) and adding it to the
    GEMV cell must be refused, because at that rung the route executes the
    materialised pair and the cell would say otherwise.
    """
    payload = copy.deepcopy(load_serving_contract())
    entry = next(e for e in payload["formats"] if e["family"] == "TESSERA_E4M3_K1")
    entry["attested_rungs_q256"] = [1006, 1024]
    entry["candidate_rungs_q256"] = [1006, 1024]
    stamp = dict(entry["attested_wire"][0])
    stamp["q256"] = 1006
    entry["attested_wire"] = [stamp] + list(entry["attested_wire"])
    cell = next(c for c in payload["lane_eligibility"]["cells"]
                if c["id"] == "tessera_e4m3_k1_dense_sm121_decode_streamed")
    cell["rungs_q256"] = [1006, 1024]
    with pytest.raises(ValueError, match="executes"):
        validate_serving_contract(payload)


def test_a_cell_may_not_claim_the_lane_in_the_residency_that_has_none(monkeypatch):
    """``resident`` sets ``tessera_gemv = None``; a cell claiming it is refused.

    The residency is the axis the two E4M3 decode cells are told apart by, so
    it has to be an axis the validator reads rather than a label.
    """
    payload = copy.deepcopy(load_serving_contract())
    cell = next(c for c in payload["lane_eligibility"]["cells"]
                if c["id"] == "tessera_e4m3_k1_dense_sm121_decode_streamed")
    cell["requires_serve_flags"] = ["TESSERA_SERVE_MODE=resident"]
    cell["id"] = "tessera_e4m3_k1_dense_sm121_decode_resident"
    with pytest.raises(ValueError, match=r"executes .* residency \['resident'\]"):
        validate_serving_contract(payload)


def test_two_cells_may_not_cover_one_residency_of_one_regime():
    """Otherwise a reader's answer depends on the order the cells were written.

    This is the hazard the residency split introduces and the rule that closes
    it: a consumer resolving ``(platform, family, structure, regime, rung)``
    picks among matching cells, and with two matches of equal status it picks
    whichever came first.  The publisher must not be able to write that table.
    """
    payload = copy.deepcopy(load_serving_contract())
    cell = next(c for c in payload["lane_eligibility"]["cells"]
                if c["id"] == "tessera_e4m3_k1_dense_sm121_decode_resident")
    twin = copy.deepcopy(cell)
    twin["id"] = "tessera_e4m3_k1_dense_sm121_decode_resident"
    payload["lane_eligibility"]["cells"].append(twin)
    with pytest.raises(ValueError, match="both cover"):
        validate_serving_contract(payload)


# --------------------------------------------------------------------------
# scope: the BF16 family's own attested rung
# --------------------------------------------------------------------------

def test_the_bf16_familys_own_attested_rung_cannot_reach_the_lane():
    """Scope, recorded rather than fixed: ``TESSERA_BF16_K1``'s attested rung is
    q256 1792 -- root 7 exactly -- so its streamed GEMV lane is as unreachable
    as the FP8 one was.  Pinned so the day someone attests a reachable BF16
    rung, this test says so."""
    rates = rate_set(root_from_q256(1792), cap=15)
    assert rates == (7,)
    assert not lane_rate_report(LANE, rates)["reachable"]
