"""The compensation module: the LDL identity, and the slice-equals-whole property.

The second is what makes compensation a *preprocessing* step rather than an
encoder change, so it is the property that has to hold exactly, not
approximately.
"""
from fractions import Fraction

import pytest
import torch

from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.compensate import block_ldl, compensated_targets, regularize_hessian
from tessera.decode import reconstruct_unit
from tessera.encode import encode_unit
from tessera.grammar import bresenham_rate_schedule
from tessera.manifest import RotationState
from tessera.trellis import ConvCode

CC = ConvCode(memory=6)

#: Two tests below encode through ``encode_unit``, which is a GPU job.  They
#: had no guard, so a host-safe run (``CUDA_VISIBLE_DEVICES=""``, which is how
#: this suite runs while a serve holds the box) reported four RED tests that
#: were only ever absent hardware -- a failure that says nothing about the code
#: and hides one that would.  Same spelling as test_merge_guard.py and
#: test_ldlq_lut_plane.py.
cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="the encoder is a GPU job")


def _hessian(n, tokens=512, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, n, generator=generator)
    return x.T @ x, tokens


def _plan(cols, arity):
    grid = tuple_grid(E2M1_GRID, arity, partition="coset") if arity > 1 else E2M1_GRID
    rates = bresenham_rate_schedule(Fraction(grid.rate_cap), cols, cap=grid.rate_cap)
    return grid, rates, {r: build_forest(r, grid=grid) for r in sorted(set(rates))}


def test_block_ldl_reproduces_the_hessian():
    """H = L D L^T with D block-diagonal, and L unit-block-lower."""
    raw, count = _hessian(64)
    H = regularize_hessian(raw, count=count)
    block = 16
    L = block_ldl(H.clone(), block)
    m = 64 // block
    blocked = L.view(m, block, m, block).permute(0, 2, 1, 3)
    for i in range(m):
        assert torch.equal(blocked[i, i], torch.eye(block))
        for j in range(i + 1, m):
            assert torch.equal(blocked[i, j], torch.zeros(block, block))
    # Recover D from L and check the factorisation.
    D = torch.linalg.solve(L, torch.linalg.solve(L, H).T).T
    assert torch.allclose(L @ D @ L.T, H, atol=1e-3, rtol=1e-3)
    off = D.clone()
    for i in range(m):
        off[i * block:(i + 1) * block, i * block:(i + 1) * block] = 0
    assert off.abs().max() < 1e-3 * D.abs().max()


def test_regularize_survives_a_dead_input_channel():
    raw, count = _hessian(32)
    raw[7, :] = 0
    raw[:, 7] = 0
    H = regularize_hessian(raw, count=count)
    torch.linalg.cholesky(H)          # raises if the damping did not take


def test_an_exact_encoder_compensates_to_nothing():
    """With a lossless encoder the residual is zero, so the target never moves."""
    weight = torch.randn(8, 64)
    raw, count = _hessian(64)
    L = block_ldl(regularize_hessian(raw, count=count), 16)
    target, recon = compensated_targets(
        weight, L, lambda slice_, start, stop: slice_.clone(), block=16
    )
    assert torch.equal(target, weight.float())
    assert torch.equal(recon, weight.float())


@cuda
@pytest.mark.parametrize("arity,rotation", [(1, RotationState.NONE),
                                            (2, RotationState.NONE),
                                            (2, RotationState.R_IN_ONLY)])
def test_encoding_a_slice_equals_the_span_of_a_whole_encode(arity, rotation):
    """The property that lets compensation be preprocessing rather than surgery.

    Columns are independent in ``viterbi_columns`` and scale groups are
    within-row spans, so an aligned slice must encode bit-identically.  If this
    ever fails, ``compensated_targets``' returned target no longer describes
    what was actually built and the arm is a confound.
    """
    cols, block = 512, 256
    weight = torch.randn(64, cols, device="cuda") * 0.02
    grid, rates, forests = _plan(cols, arity)

    def run(w, rate_slice):
        unit = encode_unit(w, forests, rate_slice, CC, rotation=rotation,
                           with_diagonals=False, completion=0, group=32, half=16)
        return reconstruct_unit(unit, forests, CC)

    whole = run(weight, rates)
    for start in range(0, cols, block):
        stop = start + block
        piece = run(weight[:, start:stop].contiguous(), rates[start:stop])
        assert torch.equal(piece, whole[:, start:stop])


@cuda
def test_compensation_lowers_the_hessian_weighted_error():
    """The mechanism does what it claims on real trellis output.

    Not a tolerance check on a magic number -- just that routing the residual
    of already-encoded input blocks into the ones still to come reduces the
    quantity the loss actually charges.
    """
    torch.manual_seed(0)
    cols, block = 512, 256
    weight = torch.randn(64, cols, device="cuda") * 0.02
    grid, rates, forests = _plan(cols, 2)
    raw, count = _hessian(cols, tokens=2048)
    H = regularize_hessian(raw, count=count).cuda()
    L = block_ldl(H.clone(), block)

    def encode(w, start, stop):
        unit = encode_unit(w, forests, rates[start:stop], CC,
                           rotation=RotationState.NONE, with_diagonals=False,
                           completion=0, group=32, half=16)
        return reconstruct_unit(unit, forests, CC)

    plain = encode(weight, 0, cols)
    _, compensated = compensated_targets(weight, L, encode, block=block)

    def charge(recon):
        d = (weight.float() - recon)
        return float((d @ H * d).sum())

    assert charge(compensated) < charge(plain)
