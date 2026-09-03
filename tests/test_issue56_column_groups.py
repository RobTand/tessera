"""Issue #56: a .tessera unit can be WRITTEN at a width nothing can serve.

``encode_unit`` accepts an even column count that is not a whole number of
16-column scale groups (e.g. 264).  The artifact then round-trips exactly but
cannot be materialised to NVFP4 and cannot be fed to a Tessera kernel -- and
the failure at load is a bare ``RuntimeError`` from a tensor reshape, not a
``GrammarError`` naming the column-group rule that ``kernel.py`` already
states.

The write path (``build_unit_artifact``) must refuse such a width, and
``materialize_nvfp4`` must raise the same ``GrammarError`` instead of letting
the reshape fail.
"""

import pytest
import torch

from tessera.alphabet import E2M1_GRID, build_forest
from tessera.decode import materialize_nvfp4, reconstruct_unit
from tessera.encode import encode_unit
from tessera.errors import GrammarError
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera.manifest import BodyKind, ScalePlaneKind
from tessera.trellis import ConvCode
from tessera.unit_artifact import build_unit_artifact, read_unit_artifact

CODE = ConvCode(memory=6)
FORESTS = {rate: build_forest(rate) for rate in (1, 2, 3)}

#: Even, but not a whole number of 16-column scale groups (264 % 16 == 8).
BAD_COLS = 264
#: Whole numbers of 16-groups: 256 (a full superblock) and 272 (17 groups,
#: not a multiple of 256 -- the rule is mod 16, not mod 256).
GOOD_COLS = [256, 272]
ROWS = 32


def _unit(cols, seed=0):
    torch.manual_seed(seed)
    weights = torch.randn(ROWS, cols) * 0.02
    rates = bresenham_rate_schedule(root_from_q256(640), cols)
    return encode_unit(
        weights, FORESTS, rates, CODE, released_positions=0, scale_refit=0,
    )


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
        released_positions=0, scale_refit=0,
    )
    with pytest.raises(GrammarError, match="cannot pack 2 nibbles to a byte"):
        materialize_nvfp4(unit.codes, unit.scale_base, unit.scale_refine)


@pytest.mark.parametrize("cols", GOOD_COLS)
def test_whole_scale_groups_still_write_and_materialise(cols):
    """The guards are not over-broad: multiples of 16 are unaffected."""
    unit = _unit(cols)
    _, _, blob = build_unit_artifact(unit, "unit0", FORESTS, 640, CODE)
    assert len(blob) > 0
    packed, scales, _ = materialize_nvfp4(
        unit.codes, unit.scale_base, unit.scale_refine
    )
    assert packed.shape == (ROWS, cols // 2)
    assert scales.shape == (ROWS, cols // 16)


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
