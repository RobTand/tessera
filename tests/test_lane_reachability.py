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
    cell_runtime_id_suffix, extension_lane, lane_decoder, lane_requirements, load_serving_contract,
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
    with pytest.raises(ValueError, match="window_bits 16"):
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


def _gemv_cells(contract):
    """Every cell whose ``executes`` names the lane's own op.

    There are TWO, one per regime, and that is the correction #111's first
    pass needed: the lane serves the one-row forward and the 2-to-8-row tile
    alike, and the contract's ``batch`` regime is every M > 1 forward and not
    only a first prefill (``contract.CENSUS_PHASE_REGIMES``).  A helper that
    asserted one cell was reading the census's 64-row prefill as the whole of
    the batch regime.
    """
    cells = [c for c in contract["lane_eligibility"]["cells"]
             if any(e["decoder"] == telemetry_decoder() for e in c["executes"])
             and any(e["symbol"] == kg_symbol() for e in c["executes"])]
    assert {c["regime"] for c in cells} == {"decode", "batch"}, [c["id"] for c in cells]
    return cells


def _gemv_cell(contract, regime="decode"):
    """The one GEMV cell of ``regime``."""
    cells = [c for c in _gemv_cells(contract) if c["regime"] == regime]
    assert len(cells) == 1, [c["id"] for c in cells]
    return cells[0]


def telemetry_decoder():
    from tessera.serving.contract import lane_decoder

    return lane_decoder(LANE)


def kg_symbol():
    from tessera.serving.scheme import WINDOW_GEMV_SYMBOL

    return WINDOW_GEMV_SYMBOL


def test_a_cell_names_the_lane_the_census_observed():
    """The shipped table says the E4M3 streamed route runs the GEMV, in both regimes.

    The positive half.  Without it every refusal below would pass on a table
    that named the lane nowhere, which is the state #111 was filed on.  The
    decode cell executes the GEMV and NOTHING else -- one row is always the
    lane's own op -- while the batch cell holds both launches, because "every
    M > 1" spans the 2-to-8-row tiles the GEMV serves and the wider ones only
    the materialised path can.
    """
    from tessera.serving.contract import cell_executes, cell_residency_modes

    for cell in _gemv_cells(load_serving_contract()):
        assert cell["family"] == "TESSERA_E4M3_K1"
        assert cell["rungs_q256"] == [1024]
        assert cell_residency_modes(cell) == ("streamed",)
        # The lane exists in streamed alone -- both window routes set
        # ``layer.tessera_gemv = None`` in resident -- so a cell claiming it in
        # both residencies claims a launch the resident path cannot make.
        assert cell["requires_serve_flags"] == ["TESSERA_SERVE_MODE=streamed"]
    assert cell_executes(_gemv_cell(load_serving_contract(), "decode")) == {
        (kg_symbol(), telemetry_decoder())}
    assert cell_executes(_gemv_cell(load_serving_contract(), "batch")) == {
        (kg_symbol(), telemetry_decoder()), ("torch._scaled_mm", telemetry_decoder())}


def test_every_rung_a_gemv_cell_claims_is_one_the_lane_actually_reads():
    """Straight through ``refuse_unreachable_lane``, not through a restated set."""
    contract = load_serving_contract()
    for cell in _gemv_cells(contract):
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
    # Isolate this launch refusal from the separate unique-cell-ID gate.
    payload["lane_eligibility"]["cells"] = [cell]
    with pytest.raises(ValueError, match=r"executes .* residency \['resident'\]"):
        validate_serving_contract(payload)


def test_two_cells_may_not_cover_one_residency_of_one_regime():
    """Otherwise a reader's answer depends on the order the cells were written.

    This is the hazard the residency split introduces and the rule that closes
    it: a consumer resolving the same platform, family, structure, regime,
    rung, residency, runtime image and execution mode
    picks among matching cells, and with two matches of equal status it picks
    whichever came first.  The publisher must not be able to write that table.
    """
    payload = copy.deepcopy(load_serving_contract())
    cell = next(c for c in payload["lane_eligibility"]["cells"]
                if c["id"] == "tessera_e4m3_k1_dense_sm121_decode_resident")
    twin = copy.deepcopy(cell)
    # Distinct valid IDs must not bypass the semantic scope-overlap refusal.
    twin["id"] = cell["id"] + cell_runtime_id_suffix(twin)
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



# --------------------------------------------------------------------------
# the same question asked of the BYTES: tools/tessera_lane_preflight.py (#206)
# --------------------------------------------------------------------------
#
# THE DEFECT THIS HALF PINS.  The preflight parses every unit and then throws
# the parse away except its RATES, so a unit whose rates are inside the lane
# and whose window width, body or plane is not was published as
# ``units_readable``, ``reachable: true``, ``READABLE`` -- and the CLI exited
# 0 on an artifact that falls back on every module at load.  The producer-side
# checker (``refuse_unreachable_lane``, above) has always read all four
# published requirements, so the offline wire checker was strictly WEAKER than
# the plan checker it exists to double-check on bytes somebody else built.
#
# The fixtures are real committed wire, not a shape invented here
# (``tests/data/legacy``, written at master da2b371 and pinned byte for byte
# by ``tests/test_ladder_wire.py``).  ``e2m1-256-c1-lut-512c`` is the exact
# trap: rate 1, which the lane reads, over a TCQ body on a LUT plane with no
# window at all, none of which it reads.
#
# The second vacuity in the same verdict: a directory with no ``.wire_bytes``
# tensor read ``READABLE 0/0`` and exited 0.  Zero units is not a lane an
# artifact reaches; it is an artifact with no Tessera wire in it.

PREFLIGHT = ROOT / "tools" / "tessera_lane_preflight.py"
LEGACY_WIRE = ROOT / "tests" / "data" / "legacy"
#: Rate 1 -- inside the lane -- and nothing else about it is.
UNREADABLE_WIRE = LEGACY_WIRE / "e2m1-256-c1-lut-512c.tessera"
#: The wire this lane is for: rate 4, window body, channel plane, L=14.
READABLE_WIRE = LEGACY_WIRE / "e4m3-1024-window-channel-256c.tessera"


def _preflight():
    spec = importlib.util.spec_from_file_location("tessera_lane_preflight", PREFLIGHT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _facts(path):
    from tessera.serving.scheme import wire_facts_of_parsed
    from tessera.unit_artifact import parse_unit_artifact

    return wire_facts_of_parsed(parse_unit_artifact(path.read_bytes(), device="cpu"))


def _wire_checkpoint(directory, blobs):
    """A checkpoint directory holding one shard of ``{module: fused blob}``."""
    import torch
    from safetensors.torch import save_file

    from tessera.fused import pack_fused

    directory.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for module, blob in blobs.items():
        packed = pack_fused([("unit", 16, blob)])
        tensors[f"{module}.wire_bytes"] = torch.frombuffer(
            bytearray(packed), dtype=torch.uint8).clone()
    if not tensors:
        tensors["model.layers.0.mlp.down_proj.weight"] = torch.zeros(4, 4)
    save_file(tensors, str(directory / "model.safetensors"))
    return directory


def test_the_unreadable_fixture_is_one_whose_rates_alone_would_pass():
    """Anti-vacuity: the rate check -- all the tool used to do -- says readable."""
    facts = _facts(UNREADABLE_WIRE)
    assert lane_rate_report(LANE, facts["rates"])["reachable"], facts
    requires = lane_requirements(LANE)
    assert facts["window_bits"] not in requires["window_bits"]
    assert facts["body"] != requires["body"].upper()


def test_the_byte_side_predicate_refuses_what_the_plan_side_refuses():
    """One set of facts, both sides of the seam: the offline checker must not
    be weaker than the gate it double-checks."""
    from tessera.serving.scheme import lane_wire_report

    facts = _facts(UNREADABLE_WIRE)
    report = lane_wire_report(LANE, facts)
    assert not report["readable"]
    assert any("window_bits" in refusal for refusal in report["refusals"]), report
    with pytest.raises(ValueError):
        refuse_unreachable_lane(
            LANE, grid="E2M1", q256=256, rate_cap=7, body=facts["body"],
            plane=facts["plane"], window_bits=facts["window_bits"], target="probe")


def test_the_wire_the_lane_is_for_is_readable():
    """The positive control, off committed bytes rather than off a fixture dict."""
    from tessera.serving.scheme import lane_wire_report

    report = lane_wire_report(LANE, _facts(READABLE_WIRE))
    assert report["readable"], report


def test_every_published_requirement_is_decided_and_named():
    """Each one, flipped in turn: a checker that reads three of four is #206."""
    from tessera.serving.scheme import lane_wire_report

    facts = _facts(READABLE_WIRE)
    for field, value in (("window_bits", 12), ("body", "TCQ"), ("plane", "LUT"),
                         ("rates", (3, 4))):
        report = lane_wire_report(LANE, dict(facts, **{field: value}))
        assert not report["readable"], field
        assert any(field in refusal for refusal in report["refusals"]), (field, report)


def test_a_fact_the_unit_did_not_carry_is_refused_rather_than_skipped():
    from tessera.serving.scheme import lane_wire_report

    facts = _facts(READABLE_WIRE)
    facts.pop("window_bits")
    report = lane_wire_report(LANE, facts)
    assert not report["readable"]
    assert any("window_bits" in refusal for refusal in report["refusals"]), report


def test_a_published_requirement_this_reader_cannot_decide_is_refused(monkeypatch):
    """Pin the rule, not the roster: a requirement the contract grows and this
    reader has not learned must refuse, never pass as readable."""
    from tessera.serving import contract as contract_module
    from tessera.serving import scheme

    monkeypatch.setattr(
        contract_module, "lane_requirements",
        lambda lane, contract=None: {"column_rates": [4], "sparsity": "2:4"})
    with pytest.raises(ValueError, match="sparsity"):
        scheme.lane_wire_report(LANE, {"rates": (4,)})


def test_the_cli_refuses_a_checkpoint_the_lane_cannot_read(tmp_path, monkeypatch, capsys):
    tool = _preflight()
    path = _wire_checkpoint(tmp_path / "ckpt",
                            {"model.layers.0.mlp.down_proj": UNREADABLE_WIRE.read_bytes()})
    monkeypatch.setattr("sys.argv", ["tessera_lane_preflight.py", str(path), "--lane", LANE])
    assert tool.main() == 1
    out = capsys.readouterr().out
    assert "UNREACHABLE" in out and "window_bits" in out


def test_the_cli_reads_the_wire_the_lane_is_for(tmp_path, monkeypatch, capsys):
    """The positive control at the CLI, so the refusal above is not a gate that
    refuses everything."""
    tool = _preflight()
    path = _wire_checkpoint(tmp_path / "ckpt",
                            {"model.layers.0.mlp.down_proj": READABLE_WIRE.read_bytes()})
    monkeypatch.setattr("sys.argv", ["tessera_lane_preflight.py", str(path), "--lane", LANE])
    assert tool.main() == 0
    assert "READABLE" in capsys.readouterr().out


def test_the_cli_refuses_a_directory_with_no_tessera_wire(tmp_path, monkeypatch, capsys):
    """``READABLE 0/0`` on an ordinary safetensors directory is a vacuous pass."""
    tool = _preflight()
    path = _wire_checkpoint(tmp_path / "plain", {})
    monkeypatch.setattr("sys.argv", ["tessera_lane_preflight.py", str(path), "--lane", LANE])
    assert tool.main() == 1
    assert "READABLE" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# the decision core: ONE home for every requirement class (#264)
# --------------------------------------------------------------------------
#
# THE DEFECT THIS HALF PINS.  The published predicate named four conditions
# while ``kernel_window_gemv.prepare_from_parsed`` refuses nine on the same
# bytes -- RELEASE overrides, diagonals, rotation, a TP shard's start state
# and the grid's arity were conditions no gate outside the loader could see.
# A TP row shard (``layout.slice_unit`` stamps ``initial_state``) read
# READABLE, exit 0, at preflight, and every module fell back at load: an
# experiment built to measure the lane measured the fallback, which is #104
# and #206 one condition class over.
#
# The fix is one DECISION CORE (``scheme.decide_lane_requirements``): the
# byte-time report, the plan-time gate, the loader and the bf16 route's gate
# all decide a unit through it, over the requirements the contract publishes,
# so the predicate and the loader cannot drift.  These tests drive the core
# over the full requirement set explicitly; the published block itself is
# pinned to the same set where the contract grows it.

#: The window-GEMV lane's full predicate, spelled explicitly HERE so the core
#: can be tested against it independently of what the packaged contract
#: publishes.  ``test_the_published_predicate_is_the_full_requirement_set``
#: ties the two once the contract carries it.
FULL_REQUIRES = {
    "column_rates": [1, 2, 4],
    "window_bits": [14],
    "body": "window",
    "plane": "channel",
    "release_overrides": False,
    "diagonals": False,
    "rotation": ["none"],
    "start_state": False,
    "grid_arities": [1],
}

#: Facts of a unit the lane reads, in ``wire_facts_of_parsed``'s vocabulary.
READABLE_FACTS = {
    "rates": (4,), "window_bits": 14, "body": "WINDOW", "plane": "CHANNEL",
    "release_overrides": 0, "diagonals": False, "rotation": "NONE",
    "start_state": False, "grid_arity": 1,
}


def _shard_reparse():
    """A REAL TP row shard of the readable fixture, reparsed from its own bytes.

    ``layout.slice_unit`` stamps ``initial_state`` -- the issue's live case --
    and the facts must come off the shard's serialized wire, not off the
    in-memory slice, because the preflight reads bytes somebody else built.
    """
    import torch  # noqa: F401  (parse needs it; the file already imports kg)

    from tessera.layout import slice_unit
    from tessera.trellis import ConvCode
    from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact

    parsed = parse_unit_artifact(READABLE_WIRE.read_bytes(), device="cpu")
    rows = int(parsed.manifest.geometry.rows)
    shard = slice_unit(parsed, rows=(rows // 2, rows))
    manifest = parsed.manifest
    _m, _r, blob = build_unit_artifact(
        shard, "rank1", parsed.forests, int(manifest.branch.root_q256),
        parsed.code or ConvCode(),
        superblock=int(manifest.geometry.superblock_columns),
        container=manifest.branch.container)
    return parse_unit_artifact(blob, device="cpu"), blob


def test_the_core_reads_a_readable_unit_as_readable():
    from tessera.serving.scheme import decide_lane_requirements

    assert decide_lane_requirements(LANE, FULL_REQUIRES, READABLE_FACTS) == []


@pytest.mark.parametrize("field,value,named", [
    ("start_state", True, "start state"),
    ("rotation", "R_IN_ONLY", "rotation R_IN_ONLY"),
    ("diagonals", True, "diagonals"),
    ("release_overrides", 3, "3 RELEASE override"),
    ("grid_arity", 2, "grid_arities"),
])
def test_the_core_refuses_each_new_class_by_name(field, value, named):
    from tessera.serving.scheme import decide_lane_requirements

    refusals = decide_lane_requirements(
        LANE, FULL_REQUIRES, dict(READABLE_FACTS, **{field: value}))
    assert refusals, field
    assert any(named in refusal for refusal in refusals), (field, refusals)


def test_the_core_refuses_an_absent_new_fact_rather_than_skipping_it():
    """Absent evidence is not a pass, for the new classes exactly as for the
    four #206 taught the byte-side reader."""
    from tessera.serving.scheme import decide_lane_requirements

    for field in ("release_overrides", "rotation", "grid_arity"):
        facts = dict(READABLE_FACTS)
        facts[field] = None
        refusals = decide_lane_requirements(LANE, FULL_REQUIRES, facts)
        assert any(field in refusal and "not read" in refusal
                   for refusal in refusals), (field, refusals)


def test_wire_facts_carry_the_five_new_classes_off_real_bytes():
    """The facts travel: a whole readable unit reads pass values for all five,
    and a REAL row shard's serialized bytes read ``start_state`` True."""
    facts = _facts(READABLE_WIRE)
    assert facts["release_overrides"] == 0
    assert facts["diagonals"] is False
    assert facts["rotation"] == "NONE"
    assert facts["start_state"] is False
    assert facts["grid_arity"] == 1

    from tessera.serving.scheme import wire_facts_of_parsed

    reparsed, _blob = _shard_reparse()
    shard_facts = wire_facts_of_parsed(reparsed)
    assert shard_facts["start_state"] is True
    assert shard_facts["rates"] == facts["rates"]


def test_the_plan_gate_takes_the_plan_facts_and_refuses_them(monkeypatch):
    """A plan that says it will rotate, carry diagonals or slice into shards
    is refused BY NAME at plan time -- and the defaults, which state the
    exporter's own pinned plan, still pass."""
    from tessera.serving import contract as contract_module

    monkeypatch.setattr(contract_module, "lane_requirements",
                        lambda lane, contract=None: dict(FULL_REQUIRES))
    ok = dict(grid="E4M3", q256=1024, rate_cap=15, body="WINDOW",
              plane="CHANNEL", window_bits=14, target="probe")
    assert refuse_unreachable_lane(LANE, **ok) == (4,)
    with pytest.raises(ValueError, match="rotation R_IN_ONLY"):
        refuse_unreachable_lane(LANE, **ok, rotation="R_IN_ONLY")
    with pytest.raises(ValueError, match="start state"):
        refuse_unreachable_lane(LANE, **ok, start_state=True)
    with pytest.raises(ValueError, match="diagonals"):
        refuse_unreachable_lane(LANE, **ok, diagonals=True)
    with pytest.raises(ValueError, match="RELEASE override"):
        refuse_unreachable_lane(LANE, **ok, release_overrides=7)
    with pytest.raises(ValueError, match="grid_arities"):
        refuse_unreachable_lane(LANE, **ok, grid_arity=2)


def test_the_published_predicate_is_the_full_requirement_set():
    """The contract publishes EVERY class the loader refuses -- the roster IS
    the decision here (#264, decided and delegated), so it is pinned: four
    conditions published against nine refused is the defect itself.

    The one loader clause deliberately NOT here is ``prepare_from_parsed``'s
    scalar-256-native grid check: an entry-point fact of the E4M3 table
    build, not a lane fact -- the same extension reads BF16 window wire
    through ``prepare_value_unit``, whose grid is scalar with 65536 codes,
    so publishing size/native would call wire unreadable that the lane
    serves.  ``grid_arities`` is the lane-wide part (one code per
    position)."""
    assert lane_requirements(LANE) == FULL_REQUIRES


def test_the_live_case_a_tp_row_shard_is_refused_at_preflight(tmp_path, monkeypatch, capsys):
    """#264's live failure, end to end off serialized bytes: a TP row shard
    (``layout.slice_unit`` stamps ``initial_state``) must be UNREACHABLE at
    preflight, exit 1, with the start state named.

    PRE-FIX (b4e5231): ``lane tessera_window_gemv (requires {'column_rates':
    [1, 2, 4], 'window_bits': [14], 'body': 'window', 'plane': 'channel'}):
    READABLE -- 1/1 units readable`` and exit 0, while
    ``prepare_from_parsed`` refuses the same bytes by name -- so an
    experiment built to measure the lane measured the fallback."""
    _reparsed, blob = _shard_reparse()
    tool = _preflight()
    path = _wire_checkpoint(tmp_path / "shard", {"model.layers.0.mlp.down_proj": blob})
    monkeypatch.setattr("sys.argv", ["tessera_lane_preflight.py", str(path), "--lane", LANE])
    assert tool.main() == 1
    out = capsys.readouterr().out
    assert "UNREACHABLE" in out and "READABLE" not in out.replace("UNREACHABLE", "")
    assert "start_state" in out and "shard start state" in out


def _rotated_blob():
    """Real rotated wire: the readable recipe, R_IN_ONLY, off the encoder."""
    import torch

    from tessera.alphabet import E4M3_GRID
    from tessera.export import encode_linear_planes
    from tessera.manifest import RotationState

    torch.manual_seed(0)
    weight = torch.randn(32, 128) * 0.02
    exported, _unit, _forests = encode_linear_planes(
        weight, grid=E4M3_GRID, q256=1024, name="unit", verify=False,
        rotation=RotationState.R_IN_ONLY)
    return exported.blob


def test_a_rotated_unit_is_refused_at_preflight(tmp_path, monkeypatch, capsys):
    """A second missing class end to end: rotated wire at a readable rung was
    READABLE, exit 0, pre-fix (b4e5231) while the loader refuses 'the unit is
    rotated; not read here'."""
    tool = _preflight()
    path = _wire_checkpoint(tmp_path / "rotated",
                            {"model.layers.0.mlp.down_proj": _rotated_blob()})
    monkeypatch.setattr("sys.argv", ["tessera_lane_preflight.py", str(path), "--lane", LANE])
    assert tool.main() == 1
    out = capsys.readouterr().out
    assert "UNREACHABLE" in out and "rotation R_IN_ONLY" in out


def test_every_published_requirement_has_a_violating_case():
    """Rule 3 on the grown roster: derive the classes from the PUBLISHED
    block, flip one fact per class, and require a refusal naming it -- so a
    requirement added to the contract without a violating case here fails
    this test rather than passing vacuously."""
    from tessera.serving.scheme import lane_wire_report

    violate = {
        "column_rates": ("rates", (3, 4)),
        "window_bits": ("window_bits", 12),
        "body": ("body", "TCQ"),
        "plane": ("plane", "LUT"),
        "release_overrides": ("release_overrides", 2),
        "diagonals": ("diagonals", True),
        "start_state": ("start_state", True),
        "rotation": ("rotation", "R_IN_ONLY"),
        "grid_arities": ("grid_arity", 2),
    }
    published = lane_requirements(LANE)
    missing = sorted(set(published) - set(violate))
    assert not missing, f"published requirement(s) {missing} have no violating case here"
    facts = _facts(READABLE_WIRE)
    for name in published:
        fact, value = violate[name]
        report = lane_wire_report(LANE, dict(facts, **{fact: value}))
        assert not report["readable"], name
        assert any(name in refusal for refusal in report["refusals"]), (name, report)


def test_a_lane_predicate_with_an_unknown_rotation_state_is_refused(monkeypatch):
    def _bad_rotation(entry):
        entry["lane"]["requires"]["rotation"] = ["widdershins"]

    payload = _mutated_lane(monkeypatch, _bad_rotation)
    with pytest.raises(ValueError, match="rotation"):
        validate_serving_contract(payload)


def test_a_lane_predicate_with_a_non_boolean_carry_is_refused(monkeypatch):
    def _bad_start_state(entry):
        entry["lane"]["requires"]["start_state"] = "no"

    payload = _mutated_lane(monkeypatch, _bad_start_state)
    with pytest.raises(ValueError, match="start_state"):
        validate_serving_contract(payload)


def _violated_parse(name):
    """The readable fixture's REAL parse, one published class violated.

    The map is keyed by the requirement's published name, and the drift test
    below derives its roster from the contract, so a requirement added to
    the block without a violation here fails the test rather than passing
    vacuously.
    """
    import torch
    from types import SimpleNamespace

    from tessera.manifest import BodyKind, RotationState, ScalePlaneKind
    from tessera.unit_artifact import parse_unit_artifact

    parsed = parse_unit_artifact(READABLE_WIRE.read_bytes(), device="cpu")
    unit = parsed.unit
    if name == "column_rates":
        unit.rates = (3,) * len(tuple(unit.rates))
    elif name == "window_bits":
        unit.window_bits = 12
    elif name == "body":
        unit.body = BodyKind.TCQ
    elif name == "plane":
        unit.scale_plane = ScalePlaneKind.LUT
    elif name == "release_overrides":
        unit.release_index = torch.arange(2)
    elif name == "diagonals":
        unit.diagonals = object()
    elif name == "start_state":
        unit.initial_state = torch.zeros(256, dtype=torch.int64)
    elif name == "rotation":
        unit.rotation = RotationState.R_IN_ONLY
    elif name == "grid_arities":
        parsed.grid = SimpleNamespace(arity=2, native=None, size=256, name="probe")
    else:
        raise AssertionError(
            f"published requirement {name!r} has no violating parse here; "
            "teach _violated_parse to build one")
    return parsed


def test_the_loader_and_the_predicate_refuse_together_class_by_class():
    """THE DRIFT PIN (#264): for EVERY class the contract publishes, the
    loader (``prepare_from_parsed``) and the byte-side predicate
    (``lane_wire_report``) refuse the same parse, naming the same class --
    and the readable parse passes the predicate.  The roster is derived from
    the published block, so a grown requirement lands here by itself.

    The loader delegates to ``scheme.decide_lane_requirements`` over the
    same block, so this pins the delegation rather than restating nine
    conditions a third time."""
    from tessera.errors import GrammarError
    from tessera.kernel_window_gemv import lane_refusal_for_parsed, prepare_from_parsed
    from tessera.serving.scheme import lane_wire_report, wire_facts_of_parsed
    from tessera.unit_artifact import parse_unit_artifact

    readable = parse_unit_artifact(READABLE_WIRE.read_bytes(), device="cpu")
    assert lane_refusal_for_parsed(readable) is None

    for name in sorted(lane_requirements(LANE)):
        parsed = _violated_parse(name)
        report = lane_wire_report(LANE, wire_facts_of_parsed(parsed))
        assert not report["readable"], name
        assert any(name in refusal for refusal in report["refusals"]), (name, report)
        with pytest.raises(GrammarError) as exc:
            prepare_from_parsed(parsed)
        assert name in str(exc.value), (name, str(exc.value))


def test_the_bf16_gate_is_the_same_decision():
    """``bf16_route.gemv_refusal_for_unit`` was the third partial spelling
    (rates, window, start state; 'everything else is refused upstream').  It
    now delegates: a rotated parse -- a class the old spelling could not see
    -- is refused with the class named."""
    from tessera.serving.bf16_route import gemv_eligible_for_unit, gemv_refusal_for_unit
    from tessera.unit_artifact import parse_unit_artifact

    readable = parse_unit_artifact(READABLE_WIRE.read_bytes(), device="cpu")
    assert gemv_refusal_for_unit(readable) is None and gemv_eligible_for_unit(readable)
    rotated = _violated_parse("rotation")
    refusal = gemv_refusal_for_unit(rotated)
    assert refusal is not None and "rotation R_IN_ONLY" in refusal
    shard, _blob = _shard_reparse()
    refusal = gemv_refusal_for_unit(shard)
    assert refusal is not None and "start_state" in refusal


def test_on_the_device_the_loader_loads_what_the_predicate_calls_readable():
    """The agreement, on real bytes on a real device: the fixture the
    predicate calls READABLE prepares and decodes; the shard the predicate
    refuses raises the same class the report names."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device: the positive arm builds the extension")

    from tessera.errors import GrammarError
    from tessera.kernel_window_gemv import decode_fp8, prepare_from_parsed
    from tessera.serving.scheme import lane_wire_report, wire_facts_of_parsed
    from tessera.unit_artifact import parse_unit_artifact

    readable = parse_unit_artifact(READABLE_WIRE.read_bytes(), device="cpu")
    assert lane_wire_report(LANE, wire_facts_of_parsed(readable))["readable"]
    unit = prepare_from_parsed(readable)
    got, _scale = decode_fp8(unit)
    assert got.shape == (unit.rows, unit.cols)

    shard, _blob = _shard_reparse()
    report = lane_wire_report(LANE, wire_facts_of_parsed(shard))
    assert not report["readable"]
    with pytest.raises(GrammarError, match="start_state"):
        prepare_from_parsed(shard)


def test_the_plan_gate_refuses_a_requirement_it_has_not_learned(monkeypatch):
    """#206's rule, applied to the PLAN side too: the old gate silently skipped
    requirements it did not know, so a contract that grew one would have been
    quietly ignored where the plan is made."""
    from tessera.serving import contract as contract_module

    monkeypatch.setattr(contract_module, "lane_requirements",
                        lambda lane, contract=None: {"column_rates": [4], "sparsity": "2:4"})
    with pytest.raises(ValueError, match="sparsity"):
        refuse_unreachable_lane(LANE, grid="E4M3", q256=1024, rate_cap=15,
                                body="WINDOW", plane="CHANNEL", window_bits=14,
                                target="probe")
