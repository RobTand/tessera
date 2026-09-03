"""Issue #78: ``glm53_allocate.py`` must warn when it ranks rungs on the L1 currency.

``docs/ARCHITECTURE.md`` §4.9 records rung monotonicity in the additive-Fisher
L1 surrogate as measured false, and requires any cost path that ranks rungs in
that currency to refuse or warn rather than quietly rank. The driver builds
``predicted_dloss = mean(0.5 * h_trace * squared_error)`` per (unit, format)
and solves the DP on it, which is that currency.

The rule pinned here is derived from the menu, never from today's roster: a
family is the menu's own format stem (the part before the ``_R<rung>``
suffix), and the warning fires exactly when one family is ranked at more than
one rung. Synthetic family names prove the test does not restate the menu.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _driver():
    spec = importlib.util.spec_from_file_location(
        "glm53_allocate", ROOT / "experiments" / "glm53_allocate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_rungs_of_one_family_warns(capsys):
    driver = _driver()
    menu = [("PROBE_X_R10", 4.0), ("PROBE_X_R20", 4.5)]
    warning = driver.warn_l1_rung_ranking(menu)
    assert warning is not None
    assert "PROBE_X" in warning  # the menu's own stem, not a roster name
    assert "h_trace" in warning  # the currency it ranks in
    assert "ARCHITECTURE.md" in warning and "4.9" in warning  # the rule
    assert "tessera-allocated-served-2026-09-02" in warning  # the receipt
    assert "PROBE_X" in capsys.readouterr().out  # printed, not just returned


def test_each_family_at_two_rungs_names_each(capsys):
    driver = _driver()
    menu = [("PROBE_X_R10", 4.0), ("PROBE_X_R20", 4.5),
            ("PROBE_Y_R30", 5.0), ("PROBE_Y_R40", 5.5)]
    warning = driver.warn_l1_rung_ranking(menu)
    assert warning is not None
    assert "PROBE_X" in warning and "PROBE_Y" in warning
    assert "PROBE_X" in capsys.readouterr().out


def test_one_rung_per_family_is_silent(capsys):
    driver = _driver()
    assert driver.warn_l1_rung_ranking(
        [("PROBE_X_R10", 4.0), ("PROBE_Y_R10", 4.0)]) is None
    assert capsys.readouterr().out == ""


def test_a_single_format_menu_is_silent(capsys):
    driver = _driver()
    assert driver.warn_l1_rung_ranking([("PROBE_X_R10", 4.0)]) is None
    assert capsys.readouterr().out == ""


def test_formats_without_a_rung_suffix_are_silent(capsys):
    driver = _driver()
    assert driver.warn_l1_rung_ranking(
        [("BF16", 16.0), ("NVFP4", 4.5), ("FP8_E4M3", 8.0), ("PLAIN", 1.0)]) is None
    assert capsys.readouterr().out == ""


def test_real_spelling_two_rungs_of_one_family_warns():
    driver = _driver()
    warning = driver.warn_l1_rung_ranking(
        [("BF16", 16.0),
         ("TESSERA_E4M3_K1_R896", 4.0), ("TESSERA_E4M3_K1_R1024", 4.5)])
    assert warning is not None
    assert "TESSERA_E4M3_K1" in warning
