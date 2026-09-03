"""Activation-aware encoding: LDL-ordered error feedback along the input axis.

Tessera's trellis runs **down a column** -- over *output* features -- and
treats columns (*input* features) as independent.  For the loss that actually
matters that is backwards.  The functional error of a Linear is

    || (W - Wq) x ||^2   ->   sum_r  d_r^T H d_r,     H = E[x x^T]

where ``d_r`` is row ``r``'s error over input features.  Output features do not
couple at all -- the sum is over rows, with no cross-row term -- and input
features couple through every off-diagonal of ``H``.  So the encoder's own
sequential axis is the one axis on which the error is already independent, and
the axis the loss couples is the one it parallelises over.

Two consequences follow, and only the second is a lever.

**Diagonal-Hessian importance weighting is a provable no-op here.**  Weighting
column ``j``'s squared error by ``H[j,j]`` multiplies that column's entire cost
vector by one positive constant.  ``viterbi_columns`` keeps ``cost`` as
``[cols, states]`` and takes a per-column ``min`` over states; ``picked`` and
``choice`` are per-column; the completion ``argmin`` and the release order are
per-position.  A positive per-column scalar leaves every one of those argmins
unchanged, so the anchors, the completion bits, the scale bytes and the release
order come out **bit-identical**.  This is not "we measured no effect" -- it is
structural, and it is why a first probe built on importance weighting would
have returned a null that meant nothing.  Weighting cannot be the mechanism.

**Error feedback can be**, and it is the mechanism EXL3 actually uses: LDLQ,
block-LDL of the regularised Hessian, quantise input-feature blocks from last
to first, and push each block's residual into the blocks not yet quantised
(``exllamav3/modules/quant/exl3_lib/quantize.py::ldlq``).  It is also the
mechanism behind the **1.258x** that separates PrismaQuant's NVFP4 RTN render
from its GPTQ+JSO render on these same weights -- GPTQ is sequential residual
propagation, not weighting.

**No wire change, and no encoder change.**  Because Tessera's columns are
independent inside ``viterbi_columns`` and its scale groups are within-row
spans of ``group`` columns, encoding a column slice is bit-identical to the
corresponding span of a full encode -- provided the slice is aligned to both
the scale group and the rotation block.  So compensation is a *preprocessing*
step: it computes a modified target ``W + comp``, and the ordinary encoder runs
on that.  :func:`compensated_targets` returns the target so the caller can
re-encode it whole and check that it reproduces the stitched reconstruction.

**What this does not fix.**  The trellis still couples output features, which
the loss says are independent, and still leaves input features to a block
diagonal it never sees.  Compensation routes around that; it does not remove
it.  Moving the trellis onto the input axis is a wire change and is not
attempted here.
"""

from __future__ import annotations

import torch

from .errors import GrammarError

__all__ = ["regularize_hessian", "block_ldl", "compensated_targets"]


def regularize_hessian(
    H: torch.Tensor, *, count: int | None = None, sigma_reg: float = 0.025
) -> torch.Tensor:
    """Mean-normalise and damp a raw second-moment matrix.

    ``sigma_reg`` defaults to EXL3's own 0.025 so the two arms of a head-to-head
    are damped alike; a comparison that hands one side a different regulariser
    is measuring the regulariser.  Dead input channels (an exactly zero
    diagonal) are given unit weight rather than being left singular -- a zero
    row of ``H`` contributes nothing to the loss, so any finite value is
    equivalent and only the Cholesky cares.
    """
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise GrammarError(f"H must be square, got {tuple(H.shape)}")
    work = H.float().clone()
    if count:
        work /= float(count)
    diag = work.diagonal()
    dead = diag <= 0
    if bool(dead.any()):
        diag[dead] = float(diag[~dead].mean()) if bool((~dead).any()) else 1.0
    work.diagonal().add_(sigma_reg * float(diag.mean()))
    return work


def block_ldl(H: torch.Tensor, block: int) -> torch.Tensor:
    """Unit-block-lower ``L`` with ``H = L D L^T``, ``D`` block-diagonal.

    Ports ``exllamav3``'s ``block_ldl``.  The block diagonal comes out as the
    identity, so a compensation slice taken strictly *below* the current block
    never reads it -- which is what makes the diagonal block "uncompensated":
    the ``block`` columns quantised together see no correction from each other.
    Smaller blocks therefore compensate more; the floor is set by the scale
    group, since a group's scale is fit to whatever target the group ends up
    with and cannot be fit to half of one.
    """
    n = H.shape[0]
    if n % block:
        raise GrammarError(f"{n} inputs is not a multiple of the LDL block {block}")
    m = n // block
    try:
        L = torch.linalg.cholesky(H)
    except Exception as exc:                        # noqa: BLE001 - re-raised
        raise GrammarError(
            "Cholesky failed on the regularised Hessian: it is not positive "
            "definite. Raise sigma_reg rather than silently falling back to an "
            "uncompensated encode -- a silent fallback prices a compensated arm "
            "that never ran."
        ) from exc
    diag_blocks = torch.diagonal(
        L.reshape(m, block, m, block), dim1=0, dim2=2
    ).permute(2, 0, 1)                                        # [m, b, b]
    view = L.view(n, m, block)
    for index in range(m):
        # Right-divide by the unit-triangular diagonal block without forming
        # its inverse: ``X = A @ inv(D)`` is ``D^T X^T = A^T``, and ``D`` is
        # lower-triangular from the Cholesky factor, so ``D^T`` is upper.
        view[:, index, :] = torch.linalg.solve_triangular(
            diag_blocks[index].T, view[:, index, :].T, upper=True
        ).T
    L = view.reshape(n, n).contiguous()
    eye = torch.eye(block, device=L.device, dtype=L.dtype)
    blocked = L.view(m, block, m, block).permute(0, 2, 1, 3)
    index = torch.arange(m, device=L.device)
    blocked[index, index] = eye
    return L


def compensated_targets(
    weight: torch.Tensor,
    L: torch.Tensor,
    encode,
    *,
    block: int,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """LDLQ over input-feature blocks.  Returns ``(target, reconstruction)``.

    ``weight`` is ``[out_features, in_features]`` -- Tessera's layout, which is
    the transpose of the ``(k, n)`` LDLQ works in, so the compensation

        ``comp = L[after, here].T @ (W[after] - Wq[after])``

    is transposed into ``(W[:, after] - Wq[:, after]) @ L[after, here]``.

    ``encode(target_slice, start, stop) -> reconstruction_slice`` is supplied by
    the caller so this stays a scheduling routine and not a second encoder.  It
    must return the slice's reconstruction **in the original basis** -- whatever
    the encoder does internally with rotations is its own business, because the
    residual that propagates is measured against ``weight``.

    ``block`` must divide the input count, and must be aligned to both the
    encoder's scale group and its rotation block, or the slice encodes will not
    equal the corresponding spans of a whole-matrix encode.  The caller checks
    that by re-encoding the returned target; this function does not, because it
    does not know which encoder it was handed.
    """
    if weight.ndim != 2:
        raise GrammarError(f"expected [out, in], got {tuple(weight.shape)}")
    cols = weight.shape[1]
    if L.shape != (cols, cols):
        raise GrammarError(f"L is {tuple(L.shape)}, expected ({cols}, {cols})")
    if cols % block:
        raise GrammarError(f"{cols} inputs is not a multiple of the block {block}")

    target = weight.float().clone()
    recon = torch.zeros_like(target)
    for start in range(cols - block, -1, -block):
        stop = start + block
        if stop < cols:
            residual = weight[:, stop:].float() - recon[:, stop:]
            target[:, start:stop] = (
                weight[:, start:stop].float() + residual @ L[stop:, start:stop]
            )
        recon[:, start:stop] = encode(target[:, start:stop].contiguous(), start, stop)
    return target, recon
