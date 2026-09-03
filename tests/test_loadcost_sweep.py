"""Issue #44: the load harness must cover the trailing partial superblock.

``experiments/loadcost.py`` measured one width per invocation, so the three
widths the issue calls for -- a conforming baseline, a width one column past
a superblock multiple, and a width just under one -- could only be measured as
separate runs at different times.  Separate runs are not a matched pair (the
LDLQ lesson on #13): box contention alone was worth 15% on that measurement.
``--sweep`` measures all three in one process, each a column prefix of the
same tensor, with the shape line beside every figure.

Two facts shape the sweep, and both are pinned here rather than described:

* Every ``cols % 256 == 1`` probe is odd, and an odd width cannot pack nibble
  pairs -- ``decode.materialize_nvfp4`` refuses it.  The pack half of the
  load figure is undefined there by the format, so the sweep reports
  replay-only instead of ending the matched pair early.
* The block count in the shape line comes through
  ``grammar.superblock_count``, the one authority (#22), not a second inline
  ceiling that could floor what the layout ceilings.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from tessera.decode import materialize_nvfp4
from tessera.errors import GrammarError
from tessera.grammar import superblock_count, superblock_widths

HARNESS = Path(__file__).resolve().parents[1] / "experiments" / "loadcost.py"


def _load():
    """Import the harness without redirecting the rest of the session.

    ``loadcost.py`` inserts its own ``sys.path`` entry for direct execution.
    Importing it mid-session must not redirect later ``tessera`` imports, so
    the path is restored afterwards.  The cached modules are left alone: the
    names the harness needs (``tessera.errors``, ``tessera.grammar``) are
    already imported by ``conftest`` and the imports above, so the entry never
    fires here -- and purging the cache would re-initialise the native
    ``safetensors`` extension on the next import, which its runtime refuses.
    """
    saved_path = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("loadcost", HARNESS)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = saved_path


def test_sweep_measures_the_baseline_and_both_partial_extremes():
    module = _load()
    widths = module.sweep_widths(5120)
    # The shipping tensor itself, un-narrowed (#40), anchors the pair.
    assert widths[0] == 5120
    assert widths == (5120, 4865, 5119)
    assert len(set(widths)) == 3
    one_past, nearly_full = widths[1], widths[2]
    # The rule, not the roster: residues 1 and 255 mod 256, with the trailing
    # block holding exactly 1 and 255 columns.
    assert one_past % 256 == 1
    assert superblock_widths(one_past, 256)[-1] == 1
    assert nearly_full % 256 == 255
    assert superblock_widths(nearly_full, 256)[-1] == 255
    # Both probes span the same block count as the baseline: the ceiling (#22)
    # reaches the trailing block instead of flooring it away.
    assert superblock_count(one_past, 256) == superblock_count(5120, 256)
    assert superblock_count(nearly_full, 256) == superblock_count(5120, 256)


def test_sweep_keeps_the_shipping_shape_first_on_a_partial_tensor():
    module = _load()
    widths = module.sweep_widths(5119)
    assert widths[0] == 5119
    assert 4864 in widths  # the conforming baseline below it
    assert any(w % 256 == 1 for w in widths[1:])


def test_sweep_degrades_to_fewer_widths_rather_than_a_filler():
    module = _load()
    assert module.sweep_widths(256) == (256, 1, 255)
    assert module.sweep_widths(1) == (1,)


def test_sweep_refuses_a_non_positive_width():
    module = _load()
    with pytest.raises(module.GrammarError, match="positive column count"):
        module.sweep_widths(0)


def test_shape_note_counts_through_the_one_authority():
    module = _load()
    for width in module.sweep_widths(5120):
        note = module.shape_note(width)
        assert str(superblock_count(width, 256)) in note
        if width % 256:
            assert f"the last holding {width % 256} of 256" in note
            assert "whole superblocks" not in note
        else:
            assert "whole superblocks" in note
            assert "the last holding" not in note


def test_shape_note_labels_a_narrowed_width_not_shipping():
    module = _load()
    assert "not the shipping shape" in module.shape_note(5119, truncated=True)
    assert "not the shipping shape" not in module.shape_note(5119)
    assert "not the shipping shape" not in module.shape_note(5120, truncated=True)


def test_pack_branch_agrees_with_the_owning_guard_at_sweep_widths():
    """The replay-only probes are the format's doing, not the harness's.

    Every ``cols % 256 == 1`` probe is odd, and ``materialize_nvfp4`` itself
    refuses an odd column count -- probed here on the CPU at the sweep's own
    widths, with no encoder or GPU in the path -- while the even baseline
    packs.  The harness's ``pack_applies`` branch must agree with that guard
    width for width, or the sweep would either crash the matched pair or skip
    a pack it could have measured.
    """
    module = _load()
    widths = module.sweep_widths(5120)
    for width in widths:
        assert module.pack_applies(width) == (width % 2 == 0)
    rows = 2
    for probe in widths[1:]:
        assert probe % 2 == 1  # k * 256 +/- 1 is always odd
        codes = torch.zeros(rows, probe, dtype=torch.uint8)
        with pytest.raises(GrammarError, match="cannot pack 2 nibbles"):
            materialize_nvfp4(
                codes, torch.zeros(0, dtype=torch.uint8), torch.zeros(0, dtype=torch.uint8))
    baseline = widths[0]
    codes = torch.zeros(rows, baseline, dtype=torch.uint8)
    groups = rows * baseline // 32
    base = torch.full((groups,), 127, dtype=torch.uint8)
    refine = torch.zeros(2 * groups, dtype=torch.uint8)
    packed, e4m3, _global = materialize_nvfp4(codes, base, refine)
    assert packed.shape == (rows, baseline // 2)
    assert e4m3.shape == (rows, baseline // 16)


def test_truncate_and_sweep_refuse_together_before_touching_cuda():
    module = _load()
    with pytest.raises(SystemExit, match="disagree"):
        module.main(["--truncate", "--sweep"])
