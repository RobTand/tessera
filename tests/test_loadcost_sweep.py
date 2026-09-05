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
from tessera.grammar import (
    require_scale_groups,
    superblock_count,
    superblock_widths,
)

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
    assert widths == (5120, 4896, 5088)
    assert len(set(widths)) == 3
    one_group, nearly_full = widths[1], widths[2]
    # The rule, not the roster: the extremes of trailing fill are one scale
    # group and all-but-one, because the group is the granule a partial block
    # is made of.  Residues 1 and 255 are NOT the extremes -- they are not
    # wires (see the encodability test below).
    assert one_group % 256 == module.SCALE_GROUP
    assert superblock_widths(one_group, 256)[-1] == module.SCALE_GROUP
    assert nearly_full % 256 == 256 - module.SCALE_GROUP
    assert superblock_widths(nearly_full, 256)[-1] == 256 - module.SCALE_GROUP
    # Both probes span the same block count as the baseline: the ceiling (#22)
    # reaches the trailing block instead of flooring it away.
    assert superblock_count(one_group, 256) == superblock_count(5120, 256)
    assert superblock_count(nearly_full, 256) == superblock_count(5120, 256)


def test_sweep_keeps_the_shipping_shape_first_on_a_partial_tensor():
    module = _load()
    widths = module.sweep_widths(5119)
    assert widths[0] == 5119
    assert 4864 in widths  # the conforming baseline below it
    assert any(w % 256 == module.SCALE_GROUP for w in widths[1:])


def test_sweep_degrades_to_fewer_widths_rather_than_a_filler():
    module = _load()
    assert module.sweep_widths(256) == (256, 32, 224)
    # The shipping shape is unconditional even when it is not a wire: the
    # encoder is what says so, and an empty sweep would say nothing at all.
    assert module.sweep_widths(1) == (1,)


def test_every_derived_probe_is_a_width_the_encoder_can_actually_take():
    """The defect #44 sat behind for its whole life.

    The sweep asked for ``base + 1`` and ``base - 1`` columns -- residues 1 and
    255 mod 256 -- and neither is a wire: a width must be a whole number of
    32-weight scale groups, because a group's two halves share one base
    exponent (`grammar.require_scale_groups`, #57). So the run printed a
    shape line for 4865 columns and then died inside ``_pack_scales``, and the
    measurement this harness was fixed for (#40) had still never been taken.

    Checked against the grammar's own guard rather than against ``% 32``, so
    the harness cannot drift from the rule it has to satisfy -- and against
    the guard for the rule it *cites*: this called
    ``require_column_groups``, #56's ``half`` predicate, with the 32-weight
    group, which is the right arithmetic under the wrong name and drifts the
    moment either rule moves (tessera#260 gave #57 its own home).
    """
    module = _load()
    for cols in (5120, 5119, 4096, 2048, 256):
        widths = module.sweep_widths(cols)
        for width in widths[1:]:          # widths[0] is the tensor as given
            require_scale_groups(width, module.SCALE_GROUP)
    # And the old probes really are refused, so this is a fixed bug and not a
    # style preference.
    with pytest.raises(GrammarError):
        require_scale_groups(4865, module.SCALE_GROUP)
    with pytest.raises(GrammarError):
        require_scale_groups(5119, module.SCALE_GROUP)


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
    """The pack applies at every sweep width now, and that is the fix showing.

    ``materialize_nvfp4`` packs two nibbles to a byte and refuses an odd
    column count.  The sweep used to probe ``base +/- 1``, which is always
    odd, so the harness grew a replay-only branch to survive it -- a
    workaround for widths that turned out not to be encodable at all (#44).

    With the granule corrected to the scale group, every derived probe is a
    multiple of 32 and therefore even, so the guard passes at all of them and
    the matched triple carries a pack figure at every point instead of an
    ``n/a``.  ``pack_applies`` stays: it guards ``report``, which
    ``--truncate`` and a raw tensor can still reach at any width.  It is
    simply no longer reachable from the sweep.
    """
    module = _load()
    rows = 2
    widths = module.sweep_widths(5120)
    for width in widths:
        assert width % module.SCALE_GROUP == 0
        assert module.pack_applies(width) is True
        codes = torch.zeros(rows, width, dtype=torch.uint8)
        groups = rows * width // 32
        base = torch.full((groups,), 127, dtype=torch.uint8)
        refine = torch.zeros(2 * groups, dtype=torch.uint8)
        packed, e4m3, _global = materialize_nvfp4(codes, base, refine)
        assert packed.shape == (rows, width // 2)
        assert e4m3.shape == (rows, width // 16)
    # And the branch still tells the truth about a width the sweep no longer
    # produces but ``report`` can still be handed.
    assert module.pack_applies(4865) is False
    with pytest.raises(GrammarError, match="cannot pack 2 nibbles"):
        materialize_nvfp4(
            torch.zeros(rows, 4865, dtype=torch.uint8),
            torch.zeros(0, dtype=torch.uint8),
            torch.zeros(0, dtype=torch.uint8))


def test_truncate_and_sweep_refuse_together_before_touching_cuda():
    module = _load()
    with pytest.raises(SystemExit, match="disagree"):
        module.main(["--truncate", "--sweep"])
