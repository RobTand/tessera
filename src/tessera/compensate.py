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

That alignment requirement belongs to :func:`compensated_targets` and not to
LDLQ: ``encode.encode_unit(ldl=...)`` -- the path production takes -- runs the
same schedule *inside* one encode, reading the scale plane once per pass across
every block and refitting it after the loop, so it stitches nothing and no
scale group floors its block (tessera#95).

**What this does not fix.**  The trellis still couples output features, which
the loss says are independent, and still leaves input features to a block
diagonal it never sees.  Compensation routes around that; it does not remove
it.  Moving the trellis onto the input axis is a wire change and is not
attempted here.
"""

from __future__ import annotations

import torch

from .errors import GrammarError

__all__ = ["regularize_hessian", "block_ldl", "compensated_targets",
           "block_penalty", "choose_ldl_block"]


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
    Smaller blocks therefore compensate more.  How small a block may get is
    the *caller's* constraint and not this factorisation's, so the floor is
    stated where the path is known: :func:`compensated_targets` stitches
    independently-encoded slices and floors at the encoder's scale group and
    rotation block, since a group's scale is fit to whatever target the group
    ends up with and cannot be fit to half of one, while
    ``encode.encode_unit(ldl=...)`` reads one plane per pass across every
    block and refits it after the loop, so nothing there floors the schedule
    above a single column.
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


def block_penalty(H_reg: torch.Tensor, block: int) -> float:
    """Predicted LDLQ proxy loss at ``block``, relative to full feedback.

    ``block_ldl`` zeroes the strictly-lower part of each diagonal block, so a
    column sees earlier columns from earlier *blocks* only.  What that costs is
    not a taste and does not need a sweep -- it is a closed form in the Hessian
    the encoder already holds.

    With ``H = L D L^T`` (``L`` unit lower), LDLQ's proxy loss is
    ``sum_j D_jj ||eta_j||^2`` when every column sees every earlier column's
    error, and ``sum_j H_jj ||eta_j||^2`` when none does -- and
    ``H_jj = D_jj + sum_{k<j} L[j,k]^2 D_kk``.  Block-``b`` feedback keeps the
    cross-block terms and drops the within-block ones, so the loss interpolates
    exactly between those two ends:

        Loss(b) ~ tr(D) + S(b),
        S(b) = sum over diagonal blocks of sum_{j>k in the block} L[j,k]^2 D_kk

    taking ``||eta_j||^2`` as column-independent, which is what a fixed grid
    with per-column scales gives.  This function returns ``1 + S(b)/tr(D)``.

    Validated against the measured sweeps of tessera#60, both at 4.00 bpp:

    * dense Qwen 0.6B attention (layers 0-1 q/k/v, out-space geomean)
      ``b8/b32`` **measured 0.9273**, predicted 0.9268;
      ``b4/b32`` measured 0.9154, predicted 0.9214.
    * GLM-5.3 experts (L5/L20 gate/up, E2M1x2 TCQ q896 + LDLQ + refit)
      ``b16/b32`` measured 0.9983, predicted 0.9993;
      ``b256/b32`` measured 1.0051, predicted 1.0120.

    What it predicts well is the *material* effect: where the axis is worth
    taking, it lands to within a twentieth of a percent (Qwen ``b8/b32``, 7.3%
    measured).  Where the axis is nearly flat it over-predicts the excess by
    about 2x (GLM ``b256/b32``: 1.20% predicted against 0.51% measured), which
    is the constant-``||eta||`` assumption showing -- a coarser block also
    moves the errors it is pricing.  Read it as a screen that says *whether*
    the axis is worth spending encode time on, not as an out-space number: the
    absolute penalty at one block is a proxy-loss statement.  The same formula
    explains why the axis is worth 7% on one population and 0.2% on the other:
    at ``b=32`` it costs Qwen attention 9.8% of full feedback and GLM experts
    0.14%, a factor of 70.

    ``H_reg`` must already be regularised (see ``regularize_hessian``) -- the
    damping changes ``D`` and so changes the answer, and regularising twice
    would price a Hessian no encode uses.
    """
    n = H_reg.shape[0]
    if H_reg.ndim != 2 or H_reg.shape[1] != n:
        raise GrammarError(f"H must be square, got {tuple(H_reg.shape)}")
    if block < 1 or n % block:
        raise GrammarError(f"{n} inputs is not a multiple of the LDL block {block}")
    try:
        C = torch.linalg.cholesky(H_reg.float())
    except Exception as exc:                        # noqa: BLE001 - re-raised
        raise GrammarError(
            "Cholesky failed on the regularised Hessian: it is not positive "
            "definite. Raise sigma_reg rather than pricing a block size on a "
            "Hessian no encode could use."
        ) from exc
    d = torch.diagonal(C)
    L = C / d.unsqueeze(0)                 # unit lower: L[i, j] = C[i, j]/C[j, j]
    D = d ** 2
    skipped = 0.0
    for start in range(0, n, block):
        tri = torch.tril(L[start:start + block, start:start + block], diagonal=-1)
        skipped += float(((tri ** 2) * D[start:start + block].unsqueeze(0)).sum())
    return 1.0 + skipped / float(D.sum())


def choose_ldl_block(
    H_reg: torch.Tensor, *, max_penalty: float, floor: int
) -> int:
    """Largest power-of-two block whose predicted penalty is within budget.

    ``max_penalty`` is the caller's exchange rate, not a constant of the
    method: encode time goes as roughly ``1/block`` (measured on GLM experts,
    ``block * seconds`` is 5184 / 6100 / 6060 / 7347 across 16 / 32 / 128 /
    256), so halving the block roughly doubles the encode and buys whatever
    ``block_penalty`` says it buys.  Passing a budget states that trade once,
    where a constant hides it.

    ``floor`` is the smallest block **the caller's own path** allows, and it
    has no default on purpose.  The two paths that consume an LDL block have
    different floors, and the one that has a floor is not the one production
    takes:

    * :func:`compensated_targets` stitches independently-encoded slices, so a
      block must be aligned to the encoder's scale group *and* its rotation
      block or the slices stop equalling the spans of a whole-matrix encode.
      Its floor is a block both of those divide, read off the encoder the
      caller holds -- a narrower slice is refused by the encoder itself.
    * ``encode.encode_unit(ldl=..., ldl_block=...)`` -- the path every LDLQ arm
      in this repo is measured on -- has no such floor.  The scale plane is
      read once per pass *before* the block loop and refit once *after* it, so
      every block quantises against the same plane whatever its width and no
      scale is ever fit to part of a group.  Its floor is 1.

    What going wrong looks like: a 16 that is right for the stitching path,
    inherited by the production path, silently deletes every block below 16 --
    which is where the whole of the measured win lives.  ``1645c23`` validated
    ``block_penalty`` against measured dense-Qwen ``b8`` and ``b4`` arms and
    shipped a chooser that, floored at 16, could not return either of them,
    and nothing raised.  A floor is a property of the caller's path, so the
    caller states it and this function never guesses.

    A budget the floor itself cannot meet is refused rather than quietly served
    with the floor -- the caller asked for a quality its path cannot reach, and
    the refusal names what the floor does cost so a real budget can be set.  At
    ``floor=1`` that refusal is unreachable rather than dead: a block of one
    skips nothing, so ``block_penalty`` is exactly 1.0 there and every legal
    budget meets it.
    """
    if max_penalty < 1.0:
        raise GrammarError(
            f"max_penalty is a ratio against full feedback and is at least 1.0, "
            f"got {max_penalty}")
    n = H_reg.shape[0]
    at_floor = block_penalty(H_reg, floor)
    if at_floor > max_penalty:
        raise GrammarError(
            f"no legal block meets a budget of {max_penalty}: the smallest "
            f"block the caller's path allows is {floor}, and it already costs "
            f"{at_floor:.6f} of full feedback. Raise the budget, or state the "
            f"floor this path really has -- the stitching path's scale-group "
            f"floor is not the encode_unit path's.")
    best, b = floor, floor * 2
    while b <= n and n % b == 0 and block_penalty(H_reg, b) <= max_penalty:
        best, b = b, b * 2
    return best


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
