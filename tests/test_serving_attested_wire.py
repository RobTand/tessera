"""Every attested rung names the wire it was attested on, not just the rung.

``attested_rungs_q256`` says a receipt covered rung 1792.  On its own that is
a claim about a rung, and two different encoders can write two different
byte strings at one rung: the window table the decoder reads lives on the
ALPHABET plane, so a reach term (``window_sigma``) that moves the table moves
the bytes while the rung, the route and the decoder all stand still.  A
producer preflight that compares only the rung cannot tell "attested on these
bytes" from "attested on this rung" (#55).

So each ``formats[]`` row carries ``attested_wire``: one ``wire.recipes``
entry per attested rung -- the checkpoint's own vocabulary
(``export.WireRecipe.to_config``), in the spelling a checkpoint's config
already records per rung -- saying which bytes the receipt was cut on.  The
validator checks it is exactly that: one entry per attested rung, and a
body/span/plane the route decodes.  The tripwire below checks it is the
CURRENT one: the stamp must equal what the serving exporter writes at that
rung today (``served_recipe``, for every structure a cell attests there),
so the day a bytes-moving encode change lands, this fails and forces a
re-cut or a re-stamp instead of letting the attestation silently describe
bytes no fresh export writes.
"""
from __future__ import annotations

import copy

import pytest

from tessera.serving.contract import load_serving_contract, validate_serving_contract


def _grid_for(entry_grid: str):
    """The payload grid a contract ``grid`` string names.

    The exporter's own resolution (``export.grid_from_config``: a base plus an
    arity-2 coset tuple for ``E2M1x2``); an unknown name fails closed here
    rather than resolving to another grid's numbers.
    """
    from tessera.alphabet import BF16_GRID, E2M1_GRID, E4M3_GRID, tuple_grid

    if entry_grid == "E2M1x2":
        return tuple_grid(E2M1_GRID, 2, "coset")
    try:
        return {"E4M3": E4M3_GRID, "BF16": BF16_GRID}[entry_grid]
    except KeyError:
        raise AssertionError(
            f"contract names grid {entry_grid!r}, which this test cannot resolve to the "
            "payload grid the exporter builds from that name; refusing rather than "
            "comparing against another grid's recipe") from None


@pytest.fixture(scope="module")
def contract():
    return load_serving_contract()


def _serving_exporter():
    """The serving exporter, loaded by path the way ``test_serving_export_gate`` loads it.

    ``experiments`` is not a package, and ``served_recipe`` -- the wire a
    served unit actually carries -- lives there, beside the ``main`` that
    writes it.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "export_tessera_serving.py"
    spec = importlib.util.spec_from_file_location("export_tessera_serving", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_attested_rung_stamps_the_wire_it_was_cut_on(contract):
    """One ``wire.recipes`` entry per attested rung, and it is today's SERVED wire.

    Derived, not restated: the expected entry is ``served_recipe`` at that
    rung -- the function the serving exporter calls for every unit it writes
    -- for each structure the contract's own ``lane_eligibility`` cells attest
    the family at on that rung.  A stamp has to be the served wire for every
    such structure, so a new attested rung or cell is covered by the loop
    rather than by a new line here.  If the exporter ever writes different
    bytes at an attested rung, this is the test that says the attestation no
    longer describes a fresh export.

    Not ``recipe_at(recipe_table(grid), q)``: that is the research table.
    Since contract v32 (tessera#506 leg 2) a ``routed_moe`` stack on the NVFP4
    route is served as span-2 TCQ at every reader rung, while the table keeps
    its WINDOW default below the cap for research encodes and dense modules
    (``served_recipe``'s docstring, D2b tessera#560).  Comparing the stamps
    against the table read a correct v32 attestation as stale at 128..768.

    Imports the exporter lazily: it needs torch, and the contract half of the
    suite must stay readable where torch is not installed.
    """
    pytest.importorskip("torch")
    exporter = _serving_exporter()

    cells = contract["lane_eligibility"]["cells"]
    seen = 0
    for entry in contract["formats"]:
        grid = _grid_for(entry["grid"])
        stamped = entry["attested_wire"]
        assert isinstance(stamped, list) and len(stamped) == len(entry["attested_rungs_q256"]), (
            f"{entry['family']}: attested_wire must carry one entry per attested rung, got "
            f"{stamped!r} for {entry['attested_rungs_q256']!r}")
        for item in stamped:
            seen += 1
            assert item["q256"] in entry["attested_rungs_q256"], (
                f"{entry['family']}: attested_wire stamps q256={item['q256']}, which the family "
                "does not attest")
            structures = sorted({cell["structure"] for cell in cells
                                 if cell["family"] == entry["family"]
                                 and item["q256"] in cell["rungs_q256"]})
            assert structures, (
                f"{entry['family']} q256={item['q256']}: no lane_eligibility cell attests this "
                "rung, so no structure says which wire a served unit carries there")
            for structure in structures:
                served = exporter.served_recipe(grid, item["q256"], structure=structure)
                assert item == {"q256": item["q256"], **served.to_config()}, (
                    f"{entry['family']} q256={item['q256']} ({structure}): the stamped wire is "
                    "not what the serving exporter writes at that rung today")
    assert seen, "no attested_wire entry was compared; the loop above never ran"


def test_the_bf16_attestation_is_cut_on_the_pinned_wire(contract):
    """The v5 receipt behind ``TESSERA_BF16_K1 @ 1792`` is the pinned wire.

    That serve ran before any reach term existed: ``window_sigma`` unset, reach
    ratio 1 at every rung.  Until the served A/B at R=7 (#48 gate) re-cuts the
    attestation on reach-term bytes, the honest reading of the BF16 row is
    "route attested at 1792; KL receipt on the pinned wire" -- and this pins
    the second half of that sentence to the stamped value, so a reach term
    that starts moving fresh-export bytes trips here first.
    """
    row = next(e for e in contract["formats"] if e["family"] == "TESSERA_BF16_K1")
    assert row["attested_rungs_q256"] == [1792]
    (stamped,) = row["attested_wire"]
    assert stamped["q256"] == 1792
    assert stamped["sigma"] is None, (
        f"TESSERA_BF16_K1 @ 1792 is stamped with sigma={stamped['sigma']!r}; the served "
        "receipt it cites was cut on the pinned wire (sigma unset)")


def _mutated(contract):
    return copy.deepcopy(contract)


def test_the_validator_speaks_both_dialects_it_compares():
    """The body's/plane's checkpoint spelling beside the route's spelling.

    ``contract._ATTESTED_WIRE_BODY`` / ``_ATTESTED_WIRE_PLANE`` map the
    ``wire.recipes`` vocabulary onto ``scheme.ROUTES``.  Both ends are owned
    elsewhere -- the checkpoint spelling by ``export``, the route fact by
    ``scheme`` -- so this compares the map against both sources rather than
    restating it.  A rename on either side fails here, not silently in a
    validator that stopped comparing what it claims to.
    """
    pytest.importorskip("torch")
    from tessera.export import _BODY_NAMES as RECIPE_BODY, _PLANE_NAMES as RECIPE_PLANE
    from tessera.serving.contract import _ATTESTED_WIRE_BODY, _ATTESTED_WIRE_PLANE
    from tessera.serving.scheme import ROUTES

    assert _ATTESTED_WIRE_BODY == {spelling: kind.name for kind, spelling in RECIPE_BODY.items()}
    assert _ATTESTED_WIRE_PLANE == {spelling: kind.name for kind, spelling in RECIPE_PLANE.items()}
    for route in ROUTES.values():
        assert route["body"] in set(_ATTESTED_WIRE_BODY.values())
        assert route["plane"] in set(_ATTESTED_WIRE_PLANE.values())


def test_a_row_without_its_wire_stamp_is_refused(contract):
    bad = _mutated(contract)
    del bad["formats"][0]["attested_wire"]
    with pytest.raises(ValueError, match=r"missing \['attested_wire'\]"):
        validate_serving_contract(bad)


def test_a_stamp_covering_fewer_rungs_than_attested_is_refused(contract):
    bad = _mutated(contract)
    bad["formats"][0]["attested_wire"] = bad["formats"][0]["attested_wire"][:-1] or []
    with pytest.raises(ValueError, match="one entry per attested rung"):
        validate_serving_contract(bad)


def test_a_stamp_naming_a_rung_the_family_does_not_attest_is_refused(contract):
    bad = _mutated(contract)
    stamped = bad["formats"][0]["attested_wire"]
    assert stamped, "test premise moved: the first family stamps nothing"
    stamped[0] = {**stamped[0], "q256": max(bad["formats"][0]["attested_rungs_q256"]) + 1}
    with pytest.raises(ValueError, match="does not attest"):
        validate_serving_contract(bad)


def test_a_stamp_whose_body_the_route_does_not_decode_is_refused(contract):
    bad = _mutated(contract)
    stamped = bad["formats"][2]["attested_wire"]
    assert stamped, "test premise moved: the BF16 family stamps nothing"
    stamped[0] = {**stamped[0], "body": "tcq", "span": 2}
    with pytest.raises(ValueError, match="does not decode"):
        validate_serving_contract(bad)


def test_a_stamp_with_an_unspellable_sigma_is_refused(contract):
    bad = _mutated(contract)
    stamped = bad["formats"][2]["attested_wire"]
    assert stamped, "test premise moved: the BF16 family stamps nothing"
    stamped[0] = {**stamped[0], "sigma": "wide"}
    with pytest.raises(ValueError, match="sigma"):
        validate_serving_contract(bad)
