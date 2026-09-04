"""``tessera.lane_planes``: the Triton-free packers a native decoder consumes.

The load-bearing check is the second one: the planes a *reader* derives from
the bytes on disk are the planes the *encoder* derives from its own unit, byte
for byte.  That is what lets a serving runtime decode the artifact it
verified instead of a copy the exporter kept.
"""
import os
import sys
from pathlib import Path

import pytest
import torch

from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.encode import encode_unit
from tessera.export import DEFAULT_CODE, encode_linear_planes
from tessera.lane_planes import (
    build_subset_nibbles, build_subset_values, lut_scale_bytes, lut_scale_table,
    pack_unit_for_kernel, prepare_span2_planes,
)
from tessera.manifest import ScalePlaneKind
from tessera.stock import _nvfp4_values, materialize_stock
from tessera.unit_artifact import parse_unit_artifact, read_unit_artifact

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
K2 = tuple_grid(E2M1_GRID, 2)
ROOT = Path(__file__).resolve().parents[1]


def test_lane_planes_never_imports_triton():
    """In a fresh interpreter: importing the packers must not pull Triton in
    (a serving runtime that forbids Triton imports exactly this module).

    The child is told where ``tessera`` is, the same way every other
    subprocess test here is.  Left to its own devices it resolves the name
    against whatever the venv has installed, and on this fleet that is an
    editable pin to one particular checkout: the test then reports on that
    tree no matter which one pytest is running from, and passes green on a
    box where the checkout under test has the regression.  On a box with no
    install at all it fails with ``ModuleNotFoundError``, which is the same
    bug arriving loudly.  ``CUDA_VISIBLE_DEVICES`` is deliberately *not*
    cleared -- a module that imports the kernel only when it sees a GPU is
    exactly the failure this asserts against.
    """
    import subprocess
    code = ("import sys; import tessera.lane_planes, tessera.fused, tessera.unit_artifact; "
            "print('triton' in sys.modules, 'tessera.kernel' in sys.modules)")
    env = dict(os.environ, TMPDIR="/home/rob/tmp", PYTHONPATH=str(ROOT / "src"))
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"], out.stdout


def test_subset_nibbles_spell_the_subset_values():
    forest = build_forest(K2.rate_cap, grid=K2)
    values = build_subset_values(forest, DEFAULT_CODE, "cuda")
    nibbles = build_subset_nibbles(forest, DEFAULT_CODE, "cuda")
    assert nibbles.dtype == torch.uint8 and nibbles.shape == values.shape
    spelled = _nvfp4_values(nibbles.long()).float()
    assert torch.equal(spelled, values)
    # E2M1 spells zero twice (+0.0 at 0, -0.0 at 8); equality cannot tell them
    # apart, the sign bit can.  The table carries the anchor's code, so a zero
    # anchor is nibble 0 exactly as ``materialize_stock`` writes it.
    assert torch.equal(torch.signbit(spelled), torch.signbit(values))
    from tessera.decode import _replay_tables
    subsets, _n, _s = _replay_tables(forest, DEFAULT_CODE, "cuda")
    digits = []
    for anchor in subsets.reshape(-1).tolist():
        c = int(forest.blocks[anchor][0])
        digits.extend([c // 16, c % 16])
    assert torch.equal(nibbles.cpu(), torch.tensor(digits, dtype=torch.uint8))


@pytest.mark.parametrize("rows,cols", [(256, 512), (192, 1024)])
def test_reader_planes_equal_encoder_planes(rows, cols):
    torch.manual_seed(3)
    w = (torch.randn(rows, cols, device="cuda") * 0.02).contiguous()
    exported, unit, forests = encode_linear_planes(w, grid=K2, q256=896, name="u", verify=True)
    parsed = parse_unit_artifact(exported.blob, device="cuda")
    from_reader = prepare_span2_planes(parsed, device="cuda")
    from tessera.decode import _grid_and_forests
    _g, fdict = _grid_and_forests(forests)
    from_encoder = pack_unit_for_kernel(unit, fdict[unit.rates[0]], DEFAULT_CODE)
    for k, v in from_encoder.items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(from_reader[k].cpu(), v.cpu()), k
        else:
            assert from_reader[k] == v, k
    assert torch.equal(from_reader["lut_bytes"][: unit.scale_lut.numel()].cpu(), unit.scale_lut.view(torch.uint8).cpu())
    assert torch.equal(lut_scale_table(from_reader["lut_bytes"], "cuda"), lut_scale_table(unit.scale_lut, "cuda"))
    # and the reader's own reconstruction is the stock materialisation
    assert torch.equal(read_unit_artifact(exported.blob, device="cuda").float(),
                       __import__("tessera.stock", fromlist=["stock_dequant"]).stock_dequant(
                           materialize_stock(unit, forests, DEFAULT_CODE)))


def test_lut_scale_bytes_pads_with_zero_and_refuses_overlong():
    lut = torch.tensor([0x38, 0x40], dtype=torch.uint8)
    out = lut_scale_bytes(lut, "cpu")
    assert out.tolist() == [0x38, 0x40] + [0] * 14
    with pytest.raises(Exception):
        lut_scale_bytes(torch.zeros(17, dtype=torch.uint8), "cpu")
