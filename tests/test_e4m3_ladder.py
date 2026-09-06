"""The 8-bit ladder: E4M3 is serialisable, and its absence was an omission.

`SERIALISABLE_GRIDS` gates both writing and reading, and its stated criterion
is reader-reconstructibility: a grid is admissible when a reader can rebuild
its values from an identifier, which is why the fitted Lloyd-Max grids are
excluded and E2M1 is not.  E4M3's values come from the byte pattern
(`_e4m3_value`), so it meets that criterion exactly as E2M1 does -- it was
simply never added.  The cost of the omission was a hole in the rate menu
between Tessera-4's 4.0 bpp ceiling and FP8's 8.0, so an allocator that wanted
five or six bits had to buy eight.

`E4M3^2` stays out, and that exclusion IS structural for the body it would
use: the TCQ forest planes are one byte per code, and 65536 anchors is the
count the encoder already refuses to score per step.  `BF16` reaches the same
code count and IS in, because the window body never scores the grid -- it
scores `2^window_bits` states -- and its code plane is sized from
`PayloadGrid.code_bytes` rather than assumed to be a byte.  The two questions
only looked like one while every grid fitted in a byte.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.alphabet import (  # noqa: E402
    E2M1_GRID,
    E4M3_GRID,
    SERIALISABLE_GRIDS,
    grid_digest,
    tuple_grid,
)
from tessera.errors import GrammarError  # noqa: E402
from tessera.container import parse  # noqa: E402
from tessera.export import encode_linear, wire_recipe  # noqa: E402
from tessera.manifest import BodyKind  # noqa: E402
from tessera.manifest import ScalePlaneKind  # noqa: E402
from tessera.manifest import RotationState  # noqa: E402
from tessera.planes import PlaneKind  # noqa: E402
from tessera.trellis import ConvCode  # noqa: E402
from tessera.unit_artifact import read_unit_artifact  # noqa: E402

CODE = ConvCode(memory=6)


def test_the_serialisable_set_is_the_grids_a_reader_can_rebuild():
    assert {g.name for g in SERIALISABLE_GRIDS.values()} == {
        "E2M1", "E2M1x2", "E4M3", "BF16"}
    assert grid_digest(E4M3_GRID) in SERIALISABLE_GRIDS
    # The criterion is reconstructibility, not width -- and the width every
    # grid does have is derived from it, never assumed.
    assert {g.name: g.code_bytes for g in SERIALISABLE_GRIDS.values()} == {
        "E2M1": 1, "E2M1x2": 1, "E4M3": 1, "BF16": 2}


def test_e4m3_squared_is_refused_by_the_wire_not_by_the_registry(monkeypatch):
    """Admitting E4M3^2 to the registry would move its failure later, not fix
    it: the ALPHABET/DESCENDANT planes name one code per byte, so 65536 codes
    are unwritable whatever the registry says.  Registering it here and
    watching the *wire* refuse is the difference between an exclusion that is
    structural and one that is merely an omission -- which is exactly the
    distinction that put E4M3 itself in."""
    import tessera.alphabet as alphabet

    grid = tuple_grid(E4M3_GRID, 2)
    assert grid.size == 65536 and grid.rate_cap == 15
    monkeypatch.setitem(alphabet.SERIALISABLE_GRIDS, grid_digest(grid), grid)
    torch.manual_seed(0)
    w = (torch.randn(16, 256) * 0.05).to(torch.bfloat16)
    with pytest.raises(GrammarError, match="one byte per code"):
        encode_linear(w, grid=grid, q256=1024, name="u", code=CODE,
                      rotation=RotationState.NONE, with_diagonals=False,
                      completion=0, verify=False)


RUNGS = [256, 512, 768, 1024, 1280, 1536, 1792]

#: The ladder's fixture.  Every invariant this file asserts about the wire --
#: the reader's reconstruction, and the plane-byte identity below -- is linear
#: in the tensor's own shape, so a wider tensor pins none of them harder.  It
#: costs a great deal more.  The E4M3 recipe's body is the window trellis at
#: ``L=14``, whose only CPU path is the reference Viterbi
#: (``encode.viterbi_window``), and that is ``O(rows * cols * 2^L)`` per pass
#: with ``scale_refit`` passes to a rung.  At 128x512 the seven rungs took
#: **987 s of a 1029 s parallel suite** and were one indivisible item, so no
#: number of workers could touch them (tessera#374); 32x128 is a sixteenth of
#: that, and is the shape the rest of this suite's E4M3 units already use.
#: What the larger tensor did buy is *encoder* coverage at scale -- how the
#: trellis behaves over more steps and more columns -- and that is a question
#: for the GPU path and its own measurement, not for a CPU serialisation test.
LADDER_ROWS, LADDER_COLS = 32, 128


def _plane_bytes(blob: bytes) -> "dict[PlaneKind, int]":
    """The artifact's own per-plane byte inventory, empty planes dropped.

    Read back from the bytes through ``container.parse``, so what this returns
    is what a reader sees, not what the writer meant.
    """
    manifest = parse(blob).manifest
    terminal = max(manifest.terminals, key=lambda t: t.exact_bytes)
    order = {kind: index for index, kind in enumerate(manifest.plane_order)}
    counted = {}
    for descriptor in manifest.planes:
        length = descriptor.byte_length(terminal.plane_elements[order[descriptor.kind]])
        if length:
            counted[descriptor.kind] = length
    return counted


def test_e4m3_ladder_serialises():
    """Every rung of the 8-bit ladder encodes, writes and reads back, and the
    size follows the rate exactly.

    The invariant asserted is the *increment*: one more payload bit per weight
    must cost one more bit per weight on the artifact.  Absolute bpp is not the
    thing to pin, because the alphabet/descendant planes are a fixed per-unit
    cost that a small tensor amortises badly -- 256 bytes of forest is 0.125
    bpp over a 64x256 unit and 0.008 over a 256x1024 one.  Pinning the
    increment tests the rate; pinning the total would test the test's tensor.

    The *decomposition*, though, is pinnable and is pinned here exactly, which
    is what lets the fixture shrink without the test learning less.  This
    recipe writes three planes and nothing else, and each one's size is a
    closed form in the wire's own terms: the ALPHABET plane is the per-unit
    ``2^L`` table at one byte per E4M3 code, the BODY is the rung's ``q256/256``
    bits per weight, and DIAG_SV is the CHANNEL plane's one fp16 per output
    row.  Asserting that identity against the parsed artifact says *which*
    plane every byte went to; the older reading -- total bytes within a tenth
    of a bit of the same three terms -- could only say that they summed.
    """
    torch.manual_seed(0)
    w = (torch.randn(LADDER_ROWS, LADDER_COLS) * 0.05).to(torch.bfloat16)
    sizes = []
    for q256 in RUNGS:
        built = encode_linear(w, grid=E4M3_GRID, q256=q256, name="e4m3",
                              code=CODE, rotation=RotationState.NONE,
                              with_diagonals=False, completion=0, verify=True)
        assert read_unit_artifact(built.blob, device=w.device).shape == w.shape
        # Exact, and per plane: the recipe's own width for this rung, the
        # rung's body rate over this tensor, and one fp16 per output row.
        planes = _plane_bytes(built.blob)
        assert planes == {
            PlaneKind.ALPHABET: 1 << wire_recipe(E4M3_GRID, q256).window_bits,
            PlaneKind.BODY: w.numel() * q256 // 256 // 8,
            PlaneKind.DIAG_SV: 2 * w.shape[0],
        }, (q256, planes)
        # And the plane region is those planes and nothing beside them.
        assert built.exact_bytes == sum(planes.values())
        sizes.append(built.exact_bytes * 8 / w.numel())
    for lower, upper in zip(sizes, sizes[1:]):
        # The window recipe's table has a fixed size across these rungs;
        # each step adds one body bit per weight, subject to wire alignment.
        assert 1.0 <= upper - lower <= 1.02, sizes   # window body: no forest to grow
    # The exporter's overhead above the rung's body rate follows the E4M3
    # recipe: the window body spends no code bit and grows no forest plane,
    # so the overhead is the per-unit 2^L table (8 * 2^L bits over the
    # tensor -- 32.0 bpp at L=14 on this small tensor, 0.016 at 2048x4096)
    # plus the CHANNEL plane (one fp16 per row).  ``exact_bytes`` is the plane
    # region, so the header and manifest are outside it and the residual is
    # exactly zero; the tenth of a bit is slack this reading has never needed,
    # and the per-plane identity above is what actually pins the three terms.
    recipe = wire_recipe(E4M3_GRID, RUNGS[0])
    assert recipe.body is BodyKind.WINDOW and recipe.scale_plane is ScalePlaneKind.CHANNEL
    table = 8 * (1 << recipe.window_bits) / w.numel()
    rows = 16 * w.shape[0] / w.numel()
    extra = table + rows
    assert all(extra <= size - rung / 256 < extra + 0.1
               for size, rung in zip(sizes, RUNGS)), sizes


def test_the_two_ladders_are_distinct_rate_bands():
    """E2M1 tops out at 3.0 payload bits per weight and E2M1^2 at 3.5; only the
    E4M3 ladder reaches above 4.0 bpp, which is the whole reason to have it."""
    assert E2M1_GRID.rate_cap / E2M1_GRID.arity == 3.0
    k2 = tuple_grid(E2M1_GRID, 2)
    assert k2.rate_cap / k2.arity == 3.5
    assert E4M3_GRID.rate_cap / E4M3_GRID.arity == 7.0
