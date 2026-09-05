"""Issue #57: an S6b scale group must come from one output row.

``_pack_scales`` cut groups from the flattened tensor, so at ``cols % 32 != 0``
a group spanned the tail of one output row and the head of the next.  S6b's own
semantics are what make that matter: a group's two halves share one base
exponent with ``d <= 1`` -- one octave -- and two unrelated rows' magnitudes
have no reason to be within an octave of each other.

A per-row plane cannot tile such a width at all: ``cols % 32 == 16`` is an odd
number of halves per row, and the leftover half has no within-row partner.  So
those widths are refused instead of mis-grouped -- the smaller change the issue
names, matching the ``cols % half`` refusal ``kernel._require_column_groups``
already performs.

Issue #260 is the same rule at the writer.  An ``EncodedUnit`` does
not have to come from ``encode_unit`` -- ``parse_unit_artifact`` rebuilds one
from planes, ``slice_unit`` returns one, a caller can restrict one by hand --
so the encoder's refusal is not the wire's.  The **writer** owes it too,
because that is where the bytes are decided.  All three stages call one
function (``grammar.require_scale_groups``) rather than restating one
sentence -- the encoder alone used to carry two copies of it.
"""
import sys
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.decode import reconstruct_unit  # noqa: E402
from tessera.encode import _pack_scales, _refit_scales  # noqa: E402
from tessera.errors import GrammarError  # noqa: E402
from tessera.unit_artifact import (  # noqa: E402
    build_unit_artifact,
    parse_unit_artifact,
)

#: The one committed S6b artifact: 16 x 512 on a whole number of groups.  It is
#: the source of the off-group units below because it is the only S6b unit this
#: tree holds that no test has to encode first.
S6B_FIXTURE = (
    Path(__file__).parent / "data" / "legacy" / "e2m1-256-cfull-s6b-512c.tessera"
)


def _row_banded(rows=4, cols=48):
    """Adjacent rows eight octaves apart: any straddling group's amax mixes
    magnitudes no one-octave pair can honestly cover."""
    w = torch.zeros(rows, cols)
    for r in range(rows):
        w[r] = (2.0 ** (8 * (r % 2))) * (0.5 + 0.01 * torch.arange(cols))
    return w


def test_pack_scales_refuses_a_width_that_is_not_a_whole_number_of_groups():
    w = _row_banded()
    assert w.shape[1] % 32 != 0
    with pytest.raises(GrammarError, match="whole number of 32"):
        _pack_scales(w, 32, 16)


def test_refit_scales_refuses_the_same_width():
    """The S6b refit groups halves the way the pack does; it must refuse the
    same widths rather than refitting straddled groups."""
    w = _row_banded()
    units = torch.ones_like(w)
    base = torch.zeros(6, dtype=torch.uint8)
    refine = torch.zeros(12, dtype=torch.uint8)
    effective = torch.ones(12)
    with pytest.raises(GrammarError, match="whole number of 32"):
        _refit_scales(w, units, 32, 16, base, refine, effective)


# --- #260: the writer and the reader owe the same rule -----------------------


def _fixture():
    return parse_unit_artifact(S6B_FIXTURE.read_bytes())


def _widths(unit):
    """``(off_group, whole_group)``, derived from the unit's own geometry.

    ``group + half`` is a whole number of halves (``group % half == 0``) and
    one half short of a whole number of groups: exactly the population #56's
    rule admits and #57's forbids, which is what makes it the width that tells
    the two rules apart.  ``2 * group`` is the control.  Neither number is
    written down here, because a test that spells "48" passes on the day the
    geometry moves.
    """
    group, half = int(unit.group), int(unit.half)
    off, whole = group + half, 2 * group
    assert off % half == 0 and off % group, (off, group, half)
    assert whole % group == 0
    return off, whole


def _s6b_unit_at(parsed, columns):
    """The fixture's unit restricted to ``columns`` -- what a hand restriction,
    a shard or a re-parse hands the writer.  ``encode_unit`` cannot produce
    this at an off-group width; every other producer of an ``EncodedUnit``
    can."""
    unit, rows = parsed.unit, parsed.manifest.geometry.rows
    fields = dict(
        rates=unit.rates[:columns],
        body_bits=unit.body_bits[:, :columns].contiguous(),
        scale_base=unit.scale_base[: rows * columns // unit.group].clone(),
        scale_refine=unit.scale_refine[: rows * columns // unit.half].clone(),
    )
    for key in ("completion_bits", "anchors", "codes"):
        plane = getattr(unit, key)
        if plane is not None and plane.ndim == 2 and plane.numel():
            fields[key] = plane[:, :columns].contiguous()
    return replace(unit, **fields)


def _build(parsed, columns, unit_id="s6b-260"):
    return build_unit_artifact(
        _s6b_unit_at(parsed, columns), unit_id, parsed.forests,
        parsed.manifest.branch.root_q256, parsed.code, fixture_id=None,
    )


def test_build_unit_artifact_refuses_an_off_group_s6b_width():
    """#260: the writer enforced only #56's weaker ``half`` rule, so a 48-column
    S6b unit assembled outside ``encode_unit`` wrote, parsed and decoded while
    ``slicing._slice_block_plane`` refused every cut of it -- the identity slice
    included -- and its base-scale groups paired two unrelated output rows under
    one exponent.  Refused at the writer, where the bytes are decided."""
    parsed = _fixture()
    off, _ = _widths(parsed.unit)
    with pytest.raises(
        GrammarError, match=rf"whole number of {parsed.unit.group}-weight"
    ):
        _build(parsed, off)


def test_whole_group_s6b_widths_still_write_and_decode():
    """The guard is not over-broad: the control width writes, parses and
    reconstructs, so the refusal is about the group and not about the
    restriction."""
    parsed = _fixture()
    _, whole = _widths(parsed.unit)
    _, _, blob = _build(parsed, whole)
    again = parse_unit_artifact(blob)
    assert again.manifest.geometry.columns == whole
    assert reconstruct_unit(again.unit, again.forests, again.code).shape == (
        parsed.manifest.geometry.rows, whole
    )


def test_the_32_weight_group_rule_has_one_home():
    """One rule, one place it is written; three stages that owe it.

    The pack, the refit and the writer all refuse the same widths, so the
    temptation is three copies of the same sentence -- and the encoder already
    carried two of them, character for character.  This pins that every
    stage *calls* ``grammar.require_scale_groups`` rather than restating it, by
    checking the message each raises against the owner's, and that the sentence
    is written in exactly one file.
    """
    import inspect

    from tessera import grammar as grammar_mod

    parsed = _fixture()
    group, half = int(parsed.unit.group), int(parsed.unit.half)
    off, _ = _widths(parsed.unit)

    with pytest.raises(GrammarError) as owner:
        grammar_mod.require_scale_groups(off, group)
    with pytest.raises(GrammarError) as packer:
        _pack_scales(torch.zeros(2, off), group, half)
    with pytest.raises(GrammarError) as refit:
        _refit_scales(
            torch.zeros(2, off), torch.ones(2, off), group, half,
            torch.zeros(0, dtype=torch.uint8), torch.zeros(0, dtype=torch.uint8),
            torch.zeros(0),
        )
    with pytest.raises(GrammarError) as writer:
        _build(parsed, off)
    assert str(packer.value) == str(owner.value)
    assert str(refit.value) == str(owner.value)
    assert str(writer.value) == str(owner.value)

    # The fragment is taken from the raise itself, not from prose about it, so
    # a module may explain the rule in a comment and still be caught if it
    # re-raises it.
    raised = "{group}-weight scale "
    homes = sorted(
        path.name
        for path in Path(inspect.getsourcefile(grammar_mod)).parent.glob("*.py")
        if raised in path.read_text()
    )
    assert homes == ["grammar.py"], homes
