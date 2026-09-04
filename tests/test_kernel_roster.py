"""The window-GEMV lane's supported set has ONE home: the kernel that builds.

THE DEFECT THIS PINS (#145).  ``SUPPORTED_RATES = (1, 2, 4)`` and
``WINDOW_BITS_SUPPORTED = (14,)`` were literals in ``kernel_window_gemv``,
restated as literals in ``serving.ext.WINDOW_GEMV_LANE``, restated again in
``runtime_contract.json``, and restated a fourth time by the test that claimed
to derive them -- while ``csrc/window_gemv.cu``, the file that actually
instantiates the kernels, was tied to none of them.  Add ``case 3:`` to the
``.cu`` and the lane stays unreachable at rate 3 with nothing failing; delete
the rate-1 dispatch and ``SUPPORTED_RATES`` still advertises it.

The issue's own suggested fix showed the drift was already real.  It proposed
parsing the ``switch (it.rate)`` case labels, and at ``196c142`` that switch
reads ``case 4 / case 2 / default: if (RPL == 16) run_item<..., 1, ...>``:
parsing its labels yields ``(2, 4)``, not the ``(1, 2, 4)`` the roster
published.  Only the *decode* switch ever spelled rate 1 as a case label.

THE FIX THIS PINS.  The ``.cu`` declares its roster once, as preprocessor
definitions its OWN dispatch is generated from, and Python parses that
declaration (``tessera.kernel_roster``).  A rate is therefore in the kernel
and in the eligibility gate, or in neither: no edit puts it in one alone.
These tests pin the two halves -- that the declaration is the only spelling
inside the ``.cu``, and that every Python and JSON copy is the parse of it.

THE FAIL-BEFORE (this file alone, added to ``196c142``)::

    test_the_kernel_declares_its_roster_once
        AssertionError: window_gemv.cu declares no TESSERA_GEMV_RATES(X) block
    test_no_rate_dispatch_hand_writes_a_case[switch (it.rate)]
        AssertionError: switch (it.rate) at line 292 hand-writes case labels
        [4, 2]; the roster is declared once and every dispatch generated from it
    test_no_rate_dispatch_hand_writes_a_case[switch (rate)]
        AssertionError: switch (rate) at line 375 hand-writes case labels
        [4, 2, 1]; ...
    test_the_window_is_not_spelled_as_a_literal
        AssertionError: window_gemv.cu spells the window as a literal at lines
        [376, 377, 378, 441, 468, 469, 484]
    test_the_declaration_drives_the_dispatch
        AssertionError: switch (it.rate) at line 292 does not expand
        TESSERA_GEMV_RATES
    (the remaining nine) ModuleNotFoundError: No module named
        'tessera.kernel_roster'

Torch-free by construction: the roster is a property of a text file, a JSON
document and two torch-free modules, so it is decided without a device and
without an encoder.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "tessera" / "serving" / "csrc" / "window_gemv.cu"
TEXT = SOURCE.read_text(encoding="utf-8")

RATE_SWITCHES = ["switch (it.rate)", "switch (rate)"]


def _switch(header: str) -> "tuple[int, str]":
    """``(1-based line of the switch, its brace-matched body)``."""
    start = TEXT.find(header)
    assert start >= 0, f"{SOURCE.name} has no {header!r}"
    open_brace = TEXT.index("{", start)
    depth = 0
    for i in range(open_brace, len(TEXT)):
        if TEXT[i] == "{":
            depth += 1
        elif TEXT[i] == "}":
            depth -= 1
            if depth == 0:
                return TEXT.count("\n", 0, start) + 1, TEXT[open_brace:i]
    raise AssertionError(f"{header!r} is not brace-balanced")


# --------------------------------------------------------------------------
# the kernel states its roster once, and states it where a parser can read it
# --------------------------------------------------------------------------

def test_the_kernel_declares_its_roster_once():
    """One declaration of each, or the parse has a choice to make."""
    rates = re.findall(r"^#define\s+TESSERA_GEMV_RATES\(X\)", TEXT, re.M)
    window = re.findall(r"^#define\s+TESSERA_GEMV_WINDOW_BITS\b", TEXT, re.M)
    assert rates, f"{SOURCE.name} declares no TESSERA_GEMV_RATES(X) block"
    assert window, f"{SOURCE.name} declares no TESSERA_GEMV_WINDOW_BITS"
    assert len(rates) == 1, f"{SOURCE.name} declares TESSERA_GEMV_RATES {len(rates)} times"
    assert len(window) == 1, f"{SOURCE.name} declares TESSERA_GEMV_WINDOW_BITS {len(window)} times"


@pytest.mark.parametrize("header", RATE_SWITCHES)
def test_no_rate_dispatch_hand_writes_a_case(header):
    """Every rate dispatch is GENERATED from the declaration.

    Scoped to the two switches that dispatch on a rate: ``launch_mt``'s
    ``switch (mt)`` and ``launch_abl``'s ``switch (ablation)`` dispatch on
    other axes, and their literal labels are right.
    """
    line, body = _switch(header)
    labels = [int(m) for m in re.findall(r"\bcase\s+(\d+)\s*:", body)]
    assert not labels, (
        f"{header} at line {line} hand-writes case labels {labels}; the roster is "
        "declared once and every dispatch generated from it")


@pytest.mark.parametrize("header", RATE_SWITCHES)
def test_the_declaration_drives_the_dispatch(header):
    line, body = _switch(header)
    assert "TESSERA_GEMV_RATES(" in body, (
        f"{header} at line {line} does not expand TESSERA_GEMV_RATES")


def test_the_window_is_not_spelled_as_a_literal():
    """``TESSERA_GEMV_WINDOW_BITS`` in the template arguments, the check and
    its message alike.  A digit in any of those positions is a second roster,
    which is why this looks for a digit rather than for ``14``.  ``RPL=16``
    in the layout comments is not one -- hence the lookbehind."""
    offenders = sorted({
        TEXT.count("\n", 0, m.start()) + 1
        for m in re.finditer(
            r"(window_bits\s*==\s*\d|window_decode_kernel<\s*\d|launch_mt<\s*\d|(?<!\w)L=\d)", TEXT)})
    assert not offenders, (
        f"{SOURCE.name} spells the window as a literal at lines {offenders}")


# --------------------------------------------------------------------------
# the parse, and what it refuses
# --------------------------------------------------------------------------

def test_the_python_roster_is_the_parse_of_the_kernel_source():
    from tessera import kernel_roster

    assert Path(kernel_roster.WINDOW_GEMV_SOURCE) == SOURCE
    parsed = kernel_roster.parse_window_gemv_roster(TEXT, where=str(SOURCE))
    assert kernel_roster.SUPPORTED_RATES == parsed.rates
    assert kernel_roster.WINDOW_BITS_SUPPORTED == parsed.window_bits


def test_the_parsed_source_is_the_source_the_loader_builds():
    """The file parsed and the file handed to ``cpp_extension.load`` are one
    constant, so the roster cannot be read off a copy that does not build."""
    module = (ROOT / "src" / "tessera" / "kernel_window_gemv.py").read_text(encoding="utf-8")
    assert "sources=[WINDOW_GEMV_SOURCE]" in module
    assert "from .kernel_roster import" in module


def test_another_roster_gives_another_answer():
    """The derivation, shown moving: a source declaring a different set yields
    different constants.  A restatement could not do this."""
    from tessera import kernel_roster

    other = re.sub(r"#define TESSERA_GEMV_RATES\(X\).*",
                   "#define TESSERA_GEMV_RATES(X) X(1) X(2)", TEXT)
    other = re.sub(r"#define TESSERA_GEMV_WINDOW_BITS .*",
                   "#define TESSERA_GEMV_WINDOW_BITS 12", other)
    assert other != TEXT
    parsed = kernel_roster.parse_window_gemv_roster(other, where="synthetic")
    assert parsed.rates == (1, 2)
    assert parsed.window_bits == (12,)


@pytest.mark.parametrize("declaration, why", [
    ("", "no declaration"),
    ("#define TESSERA_GEMV_RATES(X)\n#define TESSERA_GEMV_WINDOW_BITS 14\n", "empty list"),
    ("#define TESSERA_GEMV_RATES(X) X(4) X(2) X(1)\n"
     "#define TESSERA_GEMV_WINDOW_BITS 14\n", "descending"),
    ("#define TESSERA_GEMV_RATES(X) X(1) X(2) X(2)\n"
     "#define TESSERA_GEMV_WINDOW_BITS 14\n", "repeated"),
    ("#define TESSERA_GEMV_RATES(X) X(0) X(2)\n"
     "#define TESSERA_GEMV_WINDOW_BITS 14\n", "rate 0"),
    ("#define TESSERA_GEMV_RATES(X) X(1) X(2)\n", "no window"),
    ("#define TESSERA_GEMV_WINDOW_BITS 14\n", "no rates"),
    ("#define TESSERA_GEMV_RATES(X) X(1)\n#define TESSERA_GEMV_RATES(X) X(2)\n"
     "#define TESSERA_GEMV_WINDOW_BITS 14\n", "two rate declarations"),
    ("#define TESSERA_GEMV_RATES(X) X(1)\n#define TESSERA_GEMV_WINDOW_BITS 14\n"
     "#define TESSERA_GEMV_WINDOW_BITS 12\n", "two window declarations"),
])
def test_the_parse_fails_closed(declaration, why):
    """An unreadable kernel is a named refusal, never an empty roster: an empty
    ``SUPPORTED_RATES`` would refuse every unit at load and read as "no
    checkpoint can take this lane" -- which is exactly what #104 spent four
    censuses failing to tell apart."""
    from tessera import kernel_roster
    from tessera.errors import KernelSourceError

    with pytest.raises(KernelSourceError):
        kernel_roster.parse_window_gemv_roster(declaration, where=why)


def test_the_refusal_names_the_file():
    from tessera import kernel_roster
    from tessera.errors import KernelSourceError

    with pytest.raises(KernelSourceError, match="some/path.cu"):
        kernel_roster.parse_window_gemv_roster("", where="some/path.cu")


# --------------------------------------------------------------------------
# every published copy IS the parse
# --------------------------------------------------------------------------

def test_the_lane_block_publishes_the_kernels_roster():
    """``serving.ext`` is read by a producer with no torch, so it cannot import
    the kernel module -- but it can read the kernel SOURCE, which is what turns
    the old tie between two literals into a derivation from one."""
    from tessera import kernel_roster
    from tessera.serving import ext

    requires = ext.WINDOW_GEMV_LANE["requires"]
    assert tuple(requires["column_rates"]) == kernel_roster.SUPPORTED_RATES
    assert tuple(requires["window_bits"]) == kernel_roster.WINDOW_BITS_SUPPORTED


def test_the_packaged_contract_publishes_the_kernels_roster():
    """The contract's copy, checked against the kernel directly rather than
    through the two Python spellings that sit between them."""
    from tessera import kernel_roster

    payload = json.loads(
        (ROOT / "src" / "tessera" / "serving" / "runtime_contract.json").read_text())
    lanes = [entry["lane"] for entry in payload["native_extensions"]
             if entry.get("lane", {}).get("decoder") == "window_gemv"]
    assert len(lanes) == 1, "exactly one native extension publishes the window-GEMV lane"
    requires = lanes[0]["requires"]
    assert tuple(requires["column_rates"]) == kernel_roster.SUPPORTED_RATES
    assert tuple(requires["window_bits"]) == kernel_roster.WINDOW_BITS_SUPPORTED
