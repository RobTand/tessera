"""The scale-plane refit: a plane VALUE change, monotone, in today's bytes.

`_pack_scales` sets every half's scale from its amax; the trellis then chooses
codes for that plane.  `encode_unit` now alternates the two, and these tests
hold the alternation to what it promises:

  * weight-space squared error never rises from one refit to the next, and
    falls from the amax plane on Gaussian weights;
  * the refit plane is written in ordinary S6b words -- `scales_from_planes`
    reads them back exactly, and every group is canonical;
  * `scale_refit=0` is the amax plane byte for byte, so every artifact built
    before the refit existed is still reproducible from its source;
  * a column slice aligned to the scale group still encodes exactly as the
    same span of a whole-matrix encode, which is what keeps `compensate.py`
    a preprocessing step under the refit.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid  # noqa: E402
from tessera.decode import reconstruct_unit  # noqa: E402
from tessera.encode import _pack_scales, _refit_scales, encode_unit  # noqa: E402
from tessera.scale_codec import is_canonical_group, unpack_refinement_byte  # noqa: E402
from tessera.trellis import ConvCode  # noqa: E402
from tessera.wire import scales_from_planes  # noqa: E402

CODE = ConvCode(memory=6)


def _weights(rows=128, cols=256, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=generator) * 0.02


def _family(arity):
    grid = E2M1_GRID if arity == 1 else tuple_grid(E2M1_GRID, arity)
    rate = grid.rate_cap
    return grid, rate, {rate: build_forest(rate, grid=grid)}


def _sse(w, unit, forests):
    return float(((reconstruct_unit(unit, forests, CODE) - w) ** 2).sum())


@pytest.mark.parametrize("arity", [1, 2])
def test_refit_is_monotone_and_beats_the_amax_plane(arity):
    w = _weights()
    _, rate, forests = _family(arity)
    rates = (rate,) * w.shape[1]
    errors = [
        _sse(w, encode_unit(w, forests, rates, CODE, scale_refit=k), forests)
        for k in range(4)
    ]
    assert all(a >= b for a, b in zip(errors, errors[1:])), errors
    assert errors[3] < 0.97 * errors[0], errors


def test_refit_zero_is_the_amax_plane_byte_for_byte():
    w = _weights()
    _, rate, forests = _family(2)
    unit = encode_unit(w, forests, (rate,) * w.shape[1], CODE, scale_refit=0)
    base, refine, _ = _pack_scales(w, 32, 16)
    assert torch.equal(unit.scale_base, base)
    assert torch.equal(unit.scale_refine, refine)
    assert unit.scale_refit == 0


def test_refit_plane_is_ordinary_s6b_and_canonical():
    w = _weights()
    _, rate, forests = _family(2)
    unit = encode_unit(w, forests, (rate,) * w.shape[1], CODE)
    assert unit.scale_refit == 3
    base, refine, _ = _pack_scales(w, 32, 16)
    assert not torch.equal(unit.scale_refine, refine)      # it did move
    decoded = scales_from_planes(unit.scale_base, unit.scale_refine)
    assert torch.isfinite(decoded).all() and bool((decoded > 0).all())
    words = unit.scale_refine.reshape(-1, 2).tolist()
    for low, high in words:
        half0, half1 = unpack_refinement_byte((high << 4) | low)
        assert is_canonical_group(half0, half1)


def test_refit_step_never_raises_a_groups_error():
    """With the codes fixed, the step is exact arithmetic on <u,u> and <w,u>;
    a group that cannot improve keeps its word."""
    w = _weights(rows=64, cols=128, seed=3)
    _, rate, forests = _family(2)
    unit = encode_unit(w, forests, (rate,) * w.shape[1], CODE, scale_refit=0)
    base, refine, eff = _pack_scales(w, 32, 16)
    recon = reconstruct_unit(unit, forests, CODE)
    scale = torch.repeat_interleave(eff, 16).reshape(w.shape)
    units = recon / scale
    new_base, new_refine, new_eff = _refit_scales(w, units, 32, 16, base, refine, eff)
    assert torch.equal(scales_from_planes(new_base, new_refine), new_eff)
    before = ((w - units * scale) ** 2).reshape(-1, 16).sum(1)
    after = ((w - units * torch.repeat_interleave(new_eff, 16).reshape(w.shape)) ** 2
             ).reshape(-1, 16).sum(1)
    per_group = lambda v: v.reshape(-1, 2).sum(1)
    assert bool((per_group(after) <= per_group(before) + 1e-9).all())
    assert float(after.sum()) < float(before.sum())


def test_an_aligned_slice_encodes_as_the_whole_does():
    w = _weights(rows=64, cols=256, seed=5)
    _, rate, forests = _family(2)
    whole = encode_unit(w, forests, (rate,) * 256, CODE)
    start, stop = 64, 128                                   # two scale groups
    part = encode_unit(w[:, start:stop].contiguous(), forests, (rate,) * 64, CODE)
    assert torch.equal(part.codes, whole.codes[:, start:stop])
    groups = w.shape[0] * 256 // 32
    base_whole = whole.scale_base.reshape(w.shape[0], 256 // 32)[:, start // 32: stop // 32]
    assert torch.equal(part.scale_base.reshape(w.shape[0], -1), base_whole)
