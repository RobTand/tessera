"""\"A scalar 256-code hardware grid\" has ONE home, on the grid (#277).

THE DEFECT THIS PINS.  The same three-legged predicate -- ``native`` table
present, 256 codes, arity 1 -- was written out by hand at four entry points
(``kernel_window_gemv.prepare_from_parsed``,
``serving.fp8_route.prepare_tessera_fp8_module``,
``kernel_window.window_code_table`` and ``decode.materialize_fp8``) and had
already drifted: the window GEMV's spelling omitted the arity leg and leaned
on the lane refusal above it.  Nothing was wrong on the wire, because E4M3 is
the only grid the FP8 route admits and every spelling agreed on it -- but a
fifth hardware-byte grid would have had to be taught to four call sites by
hand, which is AGENTS.md rule 4 exactly.

WHY IT IS NOT A CONTRACT FACT.  The clause stays OUT of the published
``lane.requires`` predicate on purpose (#264, contract v20): the same
extension reads BF16 window wire through ``prepare_value_unit``, whose scalar
grid has 65536 codes, so publishing it would call wire unreadable that the
lane serves.  That decision is unchanged here.  What changes is that the
excluded clause now has a home -- ``alphabet.PayloadGrid.hardware_byte`` and
``alphabet.require_hardware_byte_grid`` -- instead of four.

THE FAIL-BEFORE.  On master ``alphabet`` has neither name, so every test here
fails at the import or the attribute, and the drift guard finds the ``size !=
256`` spelling in four source files instead of one.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tessera.alphabet import E2M1_GRID, E4M3_GRID, PayloadGrid, tuple_grid
from tessera.errors import GrammarError

SRC = Path(__file__).resolve().parents[1] / "src" / "tessera"
HOME = SRC / "alphabet.py"

#: The spelling the clause was written in at all four sites.  Matched on the
#: SOURCE, because that is the only thing that can tell one home from four:
#: every spelling agreed on E4M3, so no behavioural test could see the drift.
SIZE_SPELLING = re.compile(r"\.size\s*[!=]=\s*256")


# --------------------------------------------------------------------------
# the grid says what a hardware byte is
# --------------------------------------------------------------------------

def _probe(name, *, size=256, native=True, arity=1):
    """A minimal grid failing exactly the legs the caller asks it to fail."""
    values = tuple(float(i) for i in range(size * arity))
    return PayloadGrid(
        name=name,
        values=values,
        native=tuple(range(size)) if native else None,
        arity=arity,
    )


def test_the_hardware_byte_predicate_is_the_three_legs():
    assert E4M3_GRID.hardware_byte
    assert not E2M1_GRID.hardware_byte          # 16 codes
    assert not tuple_grid(E2M1_GRID, 2).hardware_byte   # arity 2, no native
    assert not _probe("PROBE256", native=False).hardware_byte
    assert not _probe("PROBE128", size=128).hardware_byte


@pytest.mark.parametrize("probe,leg", [
    (_probe("PROBE256", native=False), "native"),
    (_probe("PROBE128", size=128), "128 codes"),
    (tuple_grid(E2M1_GRID, 2), "arity 2"),
])
def test_the_refusal_names_the_grid_and_the_leg(probe, leg):
    from tessera.alphabet import require_hardware_byte_grid

    with pytest.raises(GrammarError) as exc:
        require_hardware_byte_grid(probe, purpose="a probe")
    message = str(exc.value)
    assert probe.name in message, message
    assert leg in message, message
    assert "256-code hardware grid" in message, message


def test_the_refusal_keeps_the_callers_error_class():
    """``serving.fp8_route`` raises ``ValueError`` beside its ROUTES-derived
    refusals; the other three raise ``GrammarError``.  One home for the rule
    is not a licence to move any site's class (#277)."""
    from tessera.alphabet import require_hardware_byte_grid

    with pytest.raises(ValueError) as exc:
        require_hardware_byte_grid(
            _probe("PROBE128", size=128), purpose="a probe", error=ValueError)
    assert not isinstance(exc.value, GrammarError)
    assert require_hardware_byte_grid(E4M3_GRID, purpose="a probe") is E4M3_GRID


# --------------------------------------------------------------------------
# the drift guard: one home, in the source
# --------------------------------------------------------------------------

def test_only_the_grid_spells_the_256_code_clause():
    """``git grep -n 'size != 256' src/`` returned four files when #277 was
    filed.  It returns one, and that one is the home."""
    offenders = sorted(
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if SIZE_SPELLING.search(path.read_text(encoding="utf-8"))
    )
    assert offenders == ["alphabet.py"], offenders


def test_the_home_actually_holds_the_clause():
    """Anti-vacuity: the test above passes trivially if the clause is deleted
    or renamed out of the tree, so pin that it IS in ``alphabet.py``."""
    assert SIZE_SPELLING.search(HOME.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# every entry point reads it
# --------------------------------------------------------------------------

# The four entry points below are the torch half of the tree: this module's
# own import is torch-free (the clause lives in ``alphabet``), so the ``pure``
# job collects it, and a body that reaches torch must skip there rather than
# fail (tessera#309).


def test_materialize_fp8_refuses_a_non_hardware_byte_grid():
    pytest.importorskip("torch")
    from tessera.decode import materialize_fp8

    with pytest.raises(GrammarError) as exc:
        materialize_fp8(object(), _probe("PROBE128", size=128), None)
    assert "PROBE128" in str(exc.value) and "128 codes" in str(exc.value)


def test_the_window_code_table_refuses_a_non_hardware_byte_grid():
    torch = pytest.importorskip("torch")

    from tessera.kernel_window import window_code_table

    with pytest.raises(GrammarError) as exc:
        window_code_table(torch.zeros(4, dtype=torch.uint8), tuple_grid(E2M1_GRID, 2))
    assert "E2M1x2" in str(exc.value) and "arity 2" in str(exc.value)


def test_the_window_gemv_loader_refuses_a_non_hardware_byte_grid():
    """The parse is the lane's own readable fixture, so the lane refusal
    above the clause passes and the clause itself is what refuses.  The probe
    is arity 1 on purpose: an arity-2 grid is already refused by the
    published ``grid_arities`` requirement, which would make this vacuous."""
    pytest.importorskip("torch")
    from tessera.kernel_window_gemv import lane_refusal_for_parsed, prepare_from_parsed
    from tessera.unit_artifact import parse_unit_artifact

    wire = Path(__file__).resolve().parents[1] / "tests" / "data" / "legacy" / \
        "e4m3-1024-window-channel-256c.tessera"
    parsed = parse_unit_artifact(wire.read_bytes(), device="cpu")
    assert lane_refusal_for_parsed(parsed) is None
    parsed.grid = _probe("PROBE256", native=False)
    assert lane_refusal_for_parsed(parsed) is None, "the lane admits arity 1"
    with pytest.raises(GrammarError) as exc:
        prepare_from_parsed(parsed)
    assert "PROBE256" in str(exc.value) and "native" in str(exc.value)


def test_the_fp8_route_refuses_a_non_hardware_byte_grid():
    """A grid the ROUTES entry NAMES but whose codes are not hardware bytes:
    the route-membership leg is derived from ``scheme.ROUTES`` and is not the
    thing under test here, so the probe borrows E4M3's name."""
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from tessera.serving.fp8_route import prepare_tessera_fp8_module

    probe = _probe("E4M3", size=128)
    roles = [("q_proj", SimpleNamespace(unit=object(), grid=probe))]
    with pytest.raises(ValueError) as exc:
        prepare_tessera_fp8_module(roles, device="cpu")
    assert not isinstance(exc.value, GrammarError)
    assert "128 codes" in str(exc.value), str(exc.value)
