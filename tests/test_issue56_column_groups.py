"""Issue #56: a .tessera unit can be WRITTEN at a width nothing can serve.

``encode_unit`` accepts an even column count that is not a whole number of
16-column scale groups (e.g. 264).  The artifact then round-trips exactly but
cannot be materialised to NVFP4 and cannot be fed to a Tessera kernel -- and
the failure at load is a bare ``RuntimeError`` from a tensor reshape, not a
``GrammarError`` naming the column-group rule that ``kernel.py`` already
states.

The write path (``build_unit_artifact``) must refuse such a width, and
``materialize_nvfp4`` must raise the same ``GrammarError`` instead of letting
the reshape fail.  So must the **read** path: an artifact already on disk at
such a width is byte-self-consistent -- every hash agrees with it -- so the
writer's gate cannot be the only one, and until tessera#260's audit of the
sibling S6b rule it was, so a nonconforming producer's bytes parsed and
decoded to weights whose halves straddle two output rows.
"""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tessera.alphabet import E2M1_GRID, build_forest
from tessera.decode import materialize_nvfp4, reconstruct_unit
from tessera.encode import encode_unit
from tessera.errors import GrammarError
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera.manifest import BodyKind, ScalePlaneKind
from tessera.trellis import ConvCode
from tessera.unit_artifact import (
    build_unit_artifact,
    parse_unit_artifact,
    read_unit_artifact,
)

CODE = ConvCode(memory=6)
FORESTS = {rate: build_forest(rate) for rate in (1, 2, 3)}

#: Even, but not a whole number of 16-column scale groups (264 % 16 == 8).
BAD_COLS = 264
#: Whole numbers of 16-groups that an S6b unit can also reach after #57:
#: 256 (a full superblock) and 288 (9 S6b groups, not a multiple of 256 --
#: the rule is mod 16 here and mod 32 there, neither is mod 256).
GOOD_COLS = [256, 288]
#: A whole number of 16-groups that S6b canNOT reach (272 % 32 == 16): the
#: width where the two rules visibly differ.
LUT_ONLY_COLS = 272
ROWS = 32

#: These write-side cases run on a LUT plane, and that is forced, not a
#: preference.  Issue #57's rule is strictly stronger on S6b -- a group there
#: is 32 weights, two halves sharing one base exponent -- so every off-16 width
#: an S6b unit could take is also off-32, and both ``_pack_scales`` at encode
#: and ``build_unit_artifact`` at write refuse it on that rule whether or not
#: the 16-rule is there (``grammar.require_scale_groups``, tessera#260; the
#: writer used to enforce this 16-rule alone, which is what let an S6b unit
#: assembled outside ``encode_unit`` be written at 48 columns).  A LUT plane's
#: group IS the 16-column half, so it is the plane on which this rule is the
#: binding one.
#: The materialise-side cases need no plane at all: ``materialize_nvfp4``'s
#: group is NVFP4's own 16.


def _unit(cols, seed=0, plane=ScalePlaneKind.LUT):
    torch.manual_seed(seed)
    weights = torch.randn(ROWS, cols) * 0.02
    rates = bresenham_rate_schedule(root_from_q256(640), cols)
    return encode_unit(
        weights, FORESTS, rates, CODE, released_positions=0, scale_refit=0,
        scale_plane=plane,
    )


#: The committed LUT artifact, 16 x 512 on a whole number of halves.  The
#: read-side cases restrict its planes instead of encoding, because what the
#: reader's rule is about is bytes that already exist -- and because an encode
#: at ``BAD_COLS`` costs two minutes that prove nothing extra here.
LEGACY_LUT = (
    Path(__file__).parent / "data" / "legacy" / "e2m1-256-cfull-lut-512c.tessera"
)


def _nonconforming_lut_blob(columns):
    """A LUT artifact at ``columns``, written with the write gate disarmed.

    That stands in for the nonconforming producer whose bytes the reader's gate
    exists to catch, the way ``tests/test_wire.py`` does for tessera#208's
    reserved SCALE_BASE word.  There is no other way to obtain one: since #56
    this writer refuses the width.
    """
    import tessera.unit_artifact as unit_artifact

    parsed = parse_unit_artifact(LEGACY_LUT.read_bytes())
    unit, rows = parsed.unit, parsed.manifest.geometry.rows
    fields = dict(
        rates=unit.rates[:columns],
        body_bits=unit.body_bits[:, :columns].contiguous(),
        scale_refine=unit.scale_refine[: rows * columns // unit.half].clone(),
    )
    for key in ("completion_bits", "anchors", "codes"):
        plane = getattr(unit, key)
        if plane is not None and plane.ndim == 2 and plane.numel():
            fields[key] = plane[:, :columns].contiguous()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(unit_artifact, "require_column_groups",
                      lambda *args, **kwargs: None)
        _, _, blob = build_unit_artifact(
            replace(unit, **fields), "unit0", parsed.forests,
            parsed.manifest.branch.root_q256, parsed.code, fixture_id=None,
        )
    return blob


def test_build_unit_artifact_refuses_a_partial_scale_group():
    """Today this returns bytes (4857 at 32x264); it must raise instead."""
    unit = _unit(BAD_COLS)
    with pytest.raises(GrammarError, match=r"whole number of 16"):
        build_unit_artifact(unit, "unit0", FORESTS, 640, CODE)


def test_materialize_nvfp4_refuses_a_partial_scale_group():
    """Today this dies in ``reshape``; it must refuse by name instead."""
    unit = _unit(BAD_COLS)
    with pytest.raises(GrammarError, match=r"whole number of 16"):
        materialize_nvfp4(unit.codes, unit.scale_base, unit.scale_refine)


def test_materialize_refusal_is_not_a_reshape_runtimeerror():
    """The load-side failure must name the cause, not the tensor shape."""
    unit = _unit(BAD_COLS)
    try:
        materialize_nvfp4(unit.codes, unit.scale_base, unit.scale_refine)
    except GrammarError:
        pass
    except RuntimeError as exc:  # pragma: no cover - the pre-fix behaviour
        raise AssertionError(f"bare reshape error survived: {exc}") from None
    else:  # pragma: no cover
        raise AssertionError("a partial scale group materialised without error")


def test_odd_width_still_refuses_nibble_packing_first():
    """257 is odd: the 2-nibbles-to-a-byte refusal keeps its existing message."""
    torch.manual_seed(0)
    weights = torch.randn(ROWS, 257) * 0.02
    unit = encode_unit(
        weights, FORESTS, (3,) * 257, CODE,
        released_positions=0, scale_refit=0, scale_plane=ScalePlaneKind.LUT,
    )
    with pytest.raises(GrammarError, match="cannot pack 2 nibbles to a byte"):
        materialize_nvfp4(unit.codes, unit.scale_base, unit.scale_refine)


@pytest.mark.parametrize("cols", GOOD_COLS)
def test_whole_scale_groups_still_write_and_materialise(cols):
    """The guards are not over-broad: legal widths are unaffected."""
    unit = _unit(cols, plane=ScalePlaneKind.S6B)
    _, _, blob = build_unit_artifact(unit, "unit0", FORESTS, 640, CODE)
    assert len(blob) > 0
    packed, scales, _ = materialize_nvfp4(
        unit.codes, unit.scale_base, unit.scale_refine
    )
    assert packed.shape == (ROWS, cols // 2)
    assert scales.shape == (ROWS, cols // 16)


def test_the_16_group_rule_stays_weaker_than_the_s6b_one():
    """Two rules, not one: 272 writes on a LUT plane and is refused on S6b.

    The LUT plane's group IS the 16-column half, so 17 of them tile a row
    exactly.  S6b pairs halves into 32-weight groups under one base exponent,
    so 17 halves leaves one without a partner and #57 refuses at encode.  A
    single rule would have to pick one of these answers and be wrong about
    the other plane.
    """
    _, _, blob = build_unit_artifact(
        _unit(LUT_ONLY_COLS), "unit0", FORESTS, 640, CODE)
    assert len(blob) > 0
    with pytest.raises(GrammarError, match=r"whole number of 32-weight"):
        _unit(LUT_ONLY_COLS, plane=ScalePlaneKind.S6B)


def test_channel_plane_still_writes_off_group_widths():
    """A CHANNEL plane carries one word per output row and no per-half
    plane, so the group rule is vacuous there: 40 columns (40 % 16 == 8)
    writes and round-trips exactly.  Refusing it would forbid servable
    artifacts -- ``materialize_fp8``/``materialize_bf16`` serve those units
    at any width."""
    torch.manual_seed(0)
    weights = torch.randn(ROWS, 40) * 0.02
    rates = bresenham_rate_schedule(
        root_from_q256(2 * 256), 40, cap=E2M1_GRID.payload_bits
    )
    unit = encode_unit(
        weights, E2M1_GRID, rates, CODE, body=BodyKind.WINDOW, window_bits=6,
        scale_plane=ScalePlaneKind.CHANNEL, scale_refit=1, completion=0,
    )
    _, _, blob = build_unit_artifact(unit, "unit0", E2M1_GRID, 2 * 256, CODE)
    assert torch.equal(
        read_unit_artifact(blob), reconstruct_unit(unit, E2M1_GRID, None)
    )


def test_the_rule_has_one_home_and_every_stage_calls_it():
    """One rule, one place it is written, four stages that owe it.

    The writer, the reader, the materialiser and the kernel lane all refuse
    the same widths, so the temptation is four copies of the same sentence --
    and four sentences drift. ``grammar.require_column_groups`` is the single
    definition; this pins that the other modules *call* it rather than
    restate it, by checking that the message they raise is character for
    character the one the owner raises, and that no module but the owner
    contains the sentence.
    """
    import inspect
    from pathlib import Path

    from tessera import decode as decode_mod
    from tessera import grammar as grammar_mod
    from tessera import unit_artifact as unit_mod

    with pytest.raises(GrammarError) as owner:
        grammar_mod.require_column_groups(BAD_COLS, 16)

    with pytest.raises(GrammarError) as writer:
        build_unit_artifact(_unit(BAD_COLS), "unit0", FORESTS, 640, CODE)
    with pytest.raises(GrammarError) as reader:
        parse_unit_artifact(_nonconforming_lut_blob(BAD_COLS))
    with pytest.raises(GrammarError) as materialiser:
        materialize_nvfp4(
            torch.zeros(ROWS, BAD_COLS, dtype=torch.int64),
            torch.ones(ROWS, BAD_COLS // 16), 1.0, half=16,
        )
    assert str(writer.value) == str(owner.value)
    assert str(reader.value) == str(owner.value)
    assert str(materialiser.value) == str(owner.value)

    # And the message itself is written in exactly one file. The fragment is
    # taken from the raise, not from prose about it, so a module may explain
    # the rule in a comment and still be caught if it re-raises it.
    raised = "-groups; the scale "
    homes = sorted(
        path.name
        for path in Path(inspect.getsourcefile(grammar_mod)).parent.glob("*.py")
        if raised in path.read_text()
    )
    assert homes == ["grammar.py"], homes


def test_parse_unit_artifact_refuses_a_partial_scale_group_on_disk():
    """The reader owes this rule for the same reason the writer does, and for
    one the writer cannot cover: these bytes were decided by somebody else.
    Before this, such an artifact parsed and reconstructed -- to weights whose
    16-column halves straddle two output rows, since the block planes are
    indexed ``(row * cols + col) // half`` -- and every cut of the unit it
    returned was refused by ``slicing._slice_block_plane``, the identity slice
    included.  Refused at acceptance, by name."""
    blob = _nonconforming_lut_blob(BAD_COLS)
    with pytest.raises(GrammarError, match=r"whole number of 16"):
        parse_unit_artifact(blob)


def test_the_reader_still_accepts_a_whole_group_width():
    """The read gate is not over-broad: the same restriction at a whole number
    of halves parses and reconstructs."""
    blob = _nonconforming_lut_blob(LUT_ONLY_COLS)
    parsed = parse_unit_artifact(blob)
    assert parsed.manifest.geometry.columns == LUT_ONLY_COLS
    assert reconstruct_unit(parsed.unit, parsed.forests, parsed.code).shape == (
        parsed.manifest.geometry.rows, LUT_ONLY_COLS
    )
