"""Issue #51 (ambiguous grid resolves by dict order) and #53 (census guards
contract_for but not decoder_for).

#51: ``scheme.route_for_grid`` returns the first ROUTES entry holding a grid.
Today the grids are disjoint by accident of the table's contents, not by a
check, and ``refuse_unserveable_wire`` gates the export against whichever row
wins the iteration order. The tests below inject a second holder for E4M3
(the #42 option-(a) shape) and require the lookup to refuse the ambiguous
question -- naming both holders -- while the export gate takes the family
alongside the grid so the caller that knows the answer gets it.

#53: the census's family-map guard historically covered ``contract_for``
but not ``decoder_for``, allowing an incomplete decoder map through to a
mid-run KeyError. These tests require that guard to cover every hand-written
expectation map, independently of unrelated census population guards.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tessera.serving import scheme as scheme_module
from tessera.serving.scheme import ROUTES, refuse_unserveable_wire, route_for_grid

ROOT = Path(__file__).resolve().parents[1]
CENSUS = ROOT / "tools" / "tessera_route_census.py"

FAKE_FAMILY = "TESSERA_FP8_BF16A"


@pytest.fixture
def overlapping_e4m3():
    """A second ROUTES holder for the E4M3 grid (the #42 option-(a) shape).

    Derived from the live row rather than restated: whatever the FP8 row
    holds for plane/body/span, the shadow holds too, so only the holder set
    changes. Removed on teardown so no other test sees it.
    """
    assert "E4M3" in ROUTES["TESSERA_FP8"]["grids"]
    assert FAKE_FAMILY not in ROUTES
    fp8 = ROUTES["TESSERA_FP8"]
    ROUTES[FAKE_FAMILY] = {
        "grids": ("E4M3",), "plane": fp8["plane"],
        "body": fp8["body"], "span": fp8["span"],
        "short": "FP8-BF16A",
        "grid_kind": fp8["grid_kind"],
        "builder": fp8["builder"],
        "tile": fp8["tile"],
        "columns_multiple": fp8["columns_multiple"],
        "activation_contract": "bf16_unquantized",
        "gemm_symbol": fp8["gemm_symbol"],
    }
    try:
        yield FAKE_FAMILY
    finally:
        del ROUTES[FAKE_FAMILY]


def _holders(grid: str) -> list:
    """Who holds ``grid`` right now, read off the table that owns the fact."""
    return [fam for fam, route in ROUTES.items() if grid in route["grids"]]


def test_route_for_grid_refuses_an_ambiguous_grid_naming_both_holders(overlapping_e4m3):
    holders = _holders("E4M3")
    assert len(holders) == 2  # the fixture added the only overlap
    with pytest.raises(ValueError) as excinfo:
        route_for_grid("E4M3")
    message = str(excinfo.value)
    for holder in holders:
        assert holder in message


def test_export_gate_refuses_an_ambiguous_grid_without_a_family(overlapping_e4m3):
    holders = _holders("E4M3")
    assert len(holders) == 2
    with pytest.raises(ValueError) as excinfo:
        refuse_unserveable_wire("E4M3", 1024, "WINDOW", "CHANNEL", span=1, target="t")
    message = str(excinfo.value)
    for holder in holders:
        assert holder in message


def test_export_gate_reads_the_named_family_range(overlapping_e4m3, monkeypatch):
    """With two holders, family= selects which published range gates the wire."""
    from tessera.serving import contract as contract_module

    fp8 = ROUTES["TESSERA_FP8"]
    real = contract_module.reader_rate_grid

    def stub(route, grid, contract=None):
        if (route, grid) == (overlapping_e4m3, "E4M3"):
            return (overlapping_e4m3, 100, 200, 1)
        if (route, grid) == ("TESSERA_FP8", "E4M3"):
            return ("TESSERA_FP8", 1000, 1100, 1)
        return real(route, grid, contract)

    monkeypatch.setattr(contract_module, "reader_rate_grid", stub)
    kw = {"body": fp8["body"], "plane": fp8["plane"], "span": fp8["span"]}
    # 150 is inside the shadow's range and outside the FP8 range: the gate
    # must consult the NAMED family, not the first holder in dict order.
    assert refuse_unserveable_wire(
        "E4M3", 150, target="shadow", family=overlapping_e4m3, **kw) == overlapping_e4m3
    with pytest.raises(ValueError, match=r"\[1000, 1100\]"):
        refuse_unserveable_wire("E4M3", 150, target="fp8", family="TESSERA_FP8", **kw)


def test_export_gate_rejects_a_family_that_holds_no_such_grid():
    with pytest.raises(ValueError, match="holds"):
        refuse_unserveable_wire(
            "E4M3", 1024, "WINDOW", "CHANNEL", span=1,
            target="t", family="TESSERA_BF16")


def _hand_written_expectation_maps(source: str) -> list:
    """Names bound to a dict literal in the census (comprehensions excluded).

    Derived from the file rather than restated: whatever ``*_for`` maps the
    census hand-writes are the ones the guard must cover. ``symbol_for`` is a
    comprehension off ROUTES and cannot go short, so it is not required.
    """
    names = []
    for line in source.splitlines():
        m = re.match(r"^\s*(\w+_for)\s*=\s*\{", line)
        if m and (" for " not in line or " in " not in line):
            names.append(m.group(1))
    return names


def test_census_guard_covers_every_hand_written_expectation_map():
    source = CENSUS.read_text()
    guard = _expectation_guard(source)
    for name in _hand_written_expectation_maps(source):
        assert name in guard, (
            f"census guard {guard!r} does not cover {name}; "
            "a family added there but not here passes the guard and KeyErrors "
            "at the per-module lookup mid-run")
    assert "decoder_for" in guard


def test_census_guard_would_catch_a_decoder_short_map():
    """The guarded semantics: a family known to contract_for but missing from
    decoder_for is reported before any serve is stood up."""
    source = CENSUS.read_text()
    guard = _expectation_guard(source)
    assert "decoder_for" in guard
    families = ("TESSERA_NVFP4", "TESSERA_FP8", "TESSERA_BF16", "TESSERA_NEW")
    contract_for = {f: f"contract-{f}" for f in families}
    decoder_for = {f: f"decoder-{f}" for f in families if f != "TESSERA_NEW"}
    # The guard's own semantics, evaluated on a decoder-short pair of maps:
    # it must name the family the decoder map forgot.
    namespace = {"contract_for": contract_for, "decoder_for": decoder_for,
                 "TESSERA_FAMILIES": families}
    missing = eval(guard.split("=", 1)[1].strip(), dict(namespace))  # noqa: S307
    assert "TESSERA_NEW" in missing


def _expectation_guard(source):
    """Read the unique direct family guard in main, not another function's local."""
    main = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    guards = [node for node in main.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "missing"
                      for target in node.targets)]
    assert len(guards) == 1, "census main must expose one unambiguous family-map guard"
    return ast.unparse(guards[0])


def test_expectation_guard_ignores_an_unrelated_population_helper():
    source = '''def population_guard():
    missing = sorted(set(batch) ^ set(decode))

def main():
    missing = sorted(set(TESSERA_FAMILIES) - (set(contract_for) & set(decoder_for)))
'''
    assert "TESSERA_FAMILIES" in _expectation_guard(source)
