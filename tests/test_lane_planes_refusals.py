"""``lane_planes.pack_unit_for_kernel``: what the span-2 kernel lane refuses,
CPU-reachable on purpose -- every refusal here precedes any device touch, and
the CUDA-gated ``tests/test_lane_planes.py`` cannot run on the x86 arm.

The COMPLETION plane (tessera#296).  A TCQ column at body rate ``R`` under a
grid cap ``cap`` may spend up to ``cap - R`` further bits choosing among the
descendants its anchor reaches, and ``reconstruct_unit`` applies them.  The
span-2 planes the packer emits carry select, label and point bits and nothing
else; ``build_subset_values`` reads ``forest.blocks[*][0]`` -- the anchor --
for every position, and ``gemv_from_packed`` forwards no completion field.  So
a unit written at a nonzero completion depth packed to *exactly* the bytes of
the same unit with its completion plane zeroed, while the two reconstruct 319
weights apart on the issue's 32x32 fixture.  The packer now refuses the plane
by name.  The rule is the WRITTEN depth (``grammar.completion_widths`` over
``completion_limit``, the same arithmetic the serialiser sizes the plane by):
the plane is on the wire whatever its words hold, and a parsed full-depth
unit reads its limit back as ``None``, so a check on the limit alone would
pass exactly the unit the reader recovers.
"""
from dataclasses import replace

import pytest
import torch

from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.decode import reconstruct_unit
from tessera.encode import encode_unit
from tessera.errors import GrammarError
from tessera.export import DEFAULT_CODE, encode_linear_planes
from tessera.grammar import completion_widths
from tessera.lane_planes import pack_unit_for_kernel, prepare_span2_planes
from tessera.manifest import BodyKind, ScalePlaneKind
from tessera.unit_artifact import parse_unit_artifact

K2 = tuple_grid(E2M1_GRID, 2)
CODE = DEFAULT_CODE  # the exporter's code, which the parsed-path test encodes under


def _encode(forest, rows=32, cols=32, seed=1, **kw):
    """The issue's fixture: a span-2 LUT-plane unit at one rate, CPU."""
    w = torch.randn((rows, cols), generator=torch.Generator().manual_seed(seed)) * 0.02
    return encode_unit(
        w, forest, (forest.rate,) * cols, CODE, span=2,
        scale_plane=ScalePlaneKind.LUT, scale_refit=0, **kw,
    )


def test_the_span2_packer_refuses_a_completion_plane_it_does_not_read():
    """tessera#296's reproduction, held to a refusal: rate 3 under K2's cap
    of 7 leaves four completion levels; encoded one level deep the unit
    carries 319 nonzero completion words its reconstruction depends on, and
    the packer used to emit the same bytes for it as for the zeroed unit."""
    forest = build_forest(3, grid=K2)
    assert forest.cap - forest.rate == 4, "the fixture needs a completion axis"
    deep = _encode(forest, completion=1)
    assert deep.completion_limit == 1
    nonzero = int((deep.completion_bits != 0).sum())
    assert nonzero > 0, "a fixture with no completion words proves nothing"
    # the plane carries information: the two reconstructions differ
    zeroed = replace(
        deep, completion_bits=torch.zeros_like(deep.completion_bits), completion_limit=0,
    )
    differ = int((reconstruct_unit(deep, forest, CODE) != reconstruct_unit(zeroed, forest, CODE)).sum())
    assert differ == nonzero, (differ, nonzero)

    with pytest.raises(GrammarError, match="COMPLETION plane") as excinfo:
        pack_unit_for_kernel(deep, forest, CODE)
    # the refusal names the depth, the rate and the cap it was derived from
    assert "1 level" in str(excinfo.value) and "rate 3" in str(excinfo.value)
    assert "cap of 7" in str(excinfo.value)

    # the rule is the WRITTEN depth, not the words: the plane is on the wire
    # whatever it holds, so zero words at depth 1 are refused too ...
    hollow = replace(deep, completion_bits=torch.zeros_like(deep.completion_bits))
    with pytest.raises(GrammarError, match="COMPLETION plane"):
        pack_unit_for_kernel(hollow, forest, CODE)
    # ... and ``None`` -- as deep as the rate allows -- is depth 4 here
    full_depth = _encode(forest, completion=None)
    assert full_depth.completion_limit is None
    assert completion_widths(full_depth.rates, forest.cap, None) == (4,) * 32
    with pytest.raises(GrammarError, match="4 levels deep"):
        pack_unit_for_kernel(full_depth, forest, CODE)

    # controls: the same weight encoded at completion=0 still packs, and so
    # does the zeroed twin -- the unit whose reconstruction the packer used
    # to serve under ``deep``'s name is still servable when asked for by name
    shallow = _encode(forest, completion=0)
    assert shallow.completion_limit == 0 and int((shallow.completion_bits != 0).sum()) == 0
    for unit in (shallow, zeroed):
        packed = pack_unit_for_kernel(unit, forest, CODE)
        assert packed["kind"] == "span2" and packed["rate"] == 3
        assert packed["rows"] == 32 and packed["cols"] == 32


def test_a_full_rate_unit_packs_at_any_completion_limit():
    """R = cap leaves no completion axis: ``completion_widths`` is zero at
    every limit, so the shipping full-rate wire (and a parsed one, whose
    limit reads back as ``None``) packs exactly as before."""
    forest = build_forest(K2.rate_cap, grid=K2)
    assert forest.cap == forest.rate
    for completion in (None, 0, 3):
        unit = _encode(forest, completion=completion)
        assert int((unit.completion_bits != 0).sum()) == 0
        assert completion_widths(unit.rates, forest.cap, unit.completion_limit) == (0,) * 32
        packed = pack_unit_for_kernel(unit, forest, CODE)
        assert packed["kind"] == "span2" and packed["rate"] == forest.rate


@pytest.mark.parametrize(
    "q256,completion,refused",
    [
        (768, 1, True),      # rate 6 of cap 7: one level, parsed limit reads None
        (768, None, True),   # the same plane spelled as "as deep as allowed"
        (768, 0, False),     # rate 6, no plane written
        (896, None, False),  # rate 7 = cap: no completion axis at all
    ],
)
def test_the_reader_side_planes_refuse_the_same_plane(q256, completion, refused):
    """``prepare_span2_planes`` is the native decoder's entry
    (``serving/ops.py``) and it packs through the same function, so the
    refusal reaches a unit parsed back from artifact bytes -- where a
    full-depth plane's ``completion_limit`` is recovered as ``None``, which is
    why the rule reads the written width and not the limit."""
    torch.manual_seed(2)
    w = torch.randn(32, 64) * 0.02
    exported, unit, _forests = encode_linear_planes(
        w, grid=K2, q256=q256, name="u", verify=True, completion=completion,
        body=BodyKind.TCQ, span=2, scale_plane=ScalePlaneKind.LUT,
    )
    parsed = parse_unit_artifact(exported.blob, device="cpu")
    written = max(completion_widths(
        tuple(parsed.unit.rates), K2.rate_cap, parsed.unit.completion_limit,
    ))
    assert (written > 0) is refused
    if refused:
        with pytest.raises(GrammarError, match="COMPLETION plane"):
            prepare_span2_planes(parsed, device="cpu")
    else:
        packed = prepare_span2_planes(parsed, device="cpu")
        assert packed["kind"] == "span2"
        assert packed["rate"] == unit.rates[0]
        assert "subset_nibbles" in packed and "lut_bytes" in packed
