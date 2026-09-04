"""The controls in ``experiments/moe_wire_weight_error.py`` are the measurement.

A weight-space screen against a mis-implemented control is not a screen; it is
a number with a story attached, and this repo has paid for that twice
(``two-treatments-are-not-a-control``).  These pin the two RTN arms the wire is
compared against -- that each is the format it claims, at the residency it
claims -- so the ratio the screen reports is a ratio between two things that
exist.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest
import torch

SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
          / "experiments" / "moe_wire_weight_error.py")


def _script():
    spec = importlib.util.spec_from_file_location("moe_wire_weight_error", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_nvfp4_control_reproduces_its_own_alphabet_exactly():
    """A row made only of E2M1 magnitudes has no quantization error to find."""
    m = _script()
    exact = torch.tensor([list(m.E2M1_LEVELS) * 4], dtype=torch.float32)
    assert m.relative_error(exact, m.rtn_nvfp4(exact)) == 0.0


def test_the_nvfp4_control_scales_per_group_not_per_row():
    """One group scaled up must not degrade the group beside it.

    That is the whole point of a 16-wide block scale, and a per-row
    implementation would pass every other test in this file.
    """
    m = _script()
    torch.manual_seed(0)
    row = torch.randn(1, 64)
    quiet = m.relative_error(row[:, :16], m.rtn_nvfp4(row)[:, :16])
    loud = row.clone()
    loud[:, 16:32] *= 4096.0
    after = m.relative_error(row[:, :16], m.rtn_nvfp4(loud)[:, :16])
    assert after == pytest.approx(quiet, rel=1e-6)


def test_the_nvfp4_control_refuses_a_width_that_is_not_whole_groups():
    m = _script()
    with pytest.raises(ValueError, match="16-wide groups"):
        m.rtn_nvfp4(torch.zeros(2, 24))


def test_the_fp8_control_is_invariant_to_a_row_scale():
    """A per-channel scale means a row's magnitude cannot change its error."""
    m = _script()
    torch.manual_seed(0)
    row = torch.randn(1, 128)
    plain = m.relative_error(row, m.rtn_fp8_per_channel(row))
    scaled = m.relative_error(row * 1000.0, m.rtn_fp8_per_channel(row * 1000.0))
    assert scaled == pytest.approx(plain, rel=1e-6)


def test_fp8_at_8_bpp_beats_nvfp4_at_4_5_bpp():
    """FP8 at 8 bpp must beat NVFP4 at 4.5 bpp on the same rows.

    Not a tautology: an argmin over the wrong axis, or a missing sign, inverts
    this and nothing else in the file would notice.
    """
    m = _script()
    torch.manual_seed(0)
    w = torch.randn(32, 256)
    assert m.relative_error(w, m.rtn_fp8_per_channel(w)) < m.relative_error(w, m.rtn_nvfp4(w))


def test_a_sign_survives_the_nvfp4_snap():
    m = _script()
    w = torch.full((1, 16), -3.0)
    assert torch.all(m.rtn_nvfp4(w) < 0)


def test_the_geomean_is_a_geometric_mean():
    m = _script()
    assert m.geomean([1.0, 4.0]) == pytest.approx(2.0)
    assert m.geomean([2.0, 2.0]) == pytest.approx(2.0)
