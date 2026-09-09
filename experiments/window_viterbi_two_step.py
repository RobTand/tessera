"""Two trellis steps as one front pass: the executable reference.

``encode.viterbi_window`` moves the ``[2^L, cols]`` front through memory once
per step.  The fused Triton step in ``window_viterbi`` already reads one front
and writes one front, so the step's floor is a front read plus a front write.
This module is the arithmetic that lowers *that* floor: two steps composed
into one pass, so a step-pair reads the old front once instead of twice and
writes the new front once instead of twice.

**The composition.**  Write ``F = 2^R``, ``LOW = 2^(L-R)`` and
``LOW2 = LOW / F = 2^(L-2R)``; the composition needs ``L >= 2R`` for ``LOW2``
to exist, which is the whole of its applicability bound.  One step is

    best[li]                = min over f of front[f*LOW + li]     (first argmin)
    front'[li*F + j]        = best[li] + branch(li*F + j)

so the second step reads ``front'[g*LOW + li2]``, whose class and new bits are

    li1 = (g*LOW + li2) >> R = g*LOW2 + (li2 >> R)
    j1  = (g*LOW + li2) & (F-1) = li2 & (F-1)

because ``g*LOW`` contributes nothing below bit ``R`` once ``L-R >= R``.
Index the output class as ``li2 = v*F + j`` with ``v = li2 >> R``, and the
pair collapses to

    best1[v, g]        = min over f of front[f*LOW + g*LOW2 + v]  (first argmin)
    mid[v, g, j]       = best1[v, g] + b1(g*LOW + v*F + j)
    best2[v, j]        = min over g of mid[v, g, j]               (first argmin)
    front''[(v*F+j)*F + j2] = best2[v, j] + b2((v*F+j)*F + j2)

``(f, g, v)`` ranges over ``F * F * LOW2 = 2^L`` values and the map
``f*LOW + g*LOW2 + v`` is a bijection onto the states, so the old front is
read exactly once for the pair.  ``mid`` is the *same float* the sequential
spelling would have written to ``front'``: ``best1[v,g]`` is
``best[g*LOW2 + v]`` and ``b1`` is the sequential branch, added in the same
association, so no value is recomputed by a different route.

**Why it is bit-exact and not merely close.**  Three things, and each of them
is a thing the flat spelling gets wrong:

  * the two minima stay *nested*.  A flat ``min`` over the ``F*F`` products
    ``front[...] + b1 + b2`` is the same real number and not the same float:
    fp32 addition creates ties the nested form never sees, and a tie is
    resolved by index order, so a flat scan can pick a different predecessor
    pair.  ``argmin_f`` runs on the raw old front; ``b1`` is added to the
    selected float only; ``argmin_g`` runs on the summed values.
  * both scans keep the reference's tie rule, the first minimal index --
    which is what ``torch.min(dim=0)`` returns and what the fused kernel's
    strict ``<`` scan returns.
  * step 0 pins the front at state 0 and leaves every other state ``+inf``,
    so most classes are an all-``inf`` scan.  ``inf`` compares equal to
    ``inf`` under ``<``, both spellings take index 0, and ``inf + finite``
    stays ``inf`` -- the case is exercised by construction here, not argued.

**What this module is not.**  It is not a kernel and makes no speed claim.
It halves front *traffic* on paper; whether a layout exists that realises it
on the board is a kernel A/B's question, and the intermediate ``mid`` is
``F`` floats per output class where the sequential step holds one, which is
the risk that A/B is measuring.
"""
from __future__ import annotations

import torch

__all__ = [
    "branch_costs",
    "step_sequential",
    "step_pair_nested",
    "viterbi_window_paired",
    "pair_is_applicable",
]


def pair_is_applicable(window_bits: int, rate: int) -> bool:
    """``LOW2 = 2^(L-2R)`` must exist for the index split to be a split."""
    return window_bits >= 2 * rate


def branch_costs(x_step, table, w_step):
    """``[size, n]`` branch cost, associated exactly as the reference does.

    ``(d*d)`` per coordinate, then the optional weight, then the arity sum.
    Transcribed from ``encode.viterbi_window`` so the pair and the sequence
    are adding the identical operand, not an equivalent one.
    """
    diff = x_step.t().unsqueeze(1) - table.unsqueeze(0)      # [n, size, arity]
    diff = diff * diff
    if w_step is not None:
        diff = diff * w_step.t().unsqueeze(1)
    return diff.sum(dim=2).t()                               # [size, n]


def step_sequential(cost, branch, fan, low):
    """One step: class minimum, then every state adds its own branch cost."""
    n = cost.shape[1]
    best, pred = cost.view(fan, low, n).min(dim=0)            # [low, n]
    return best.repeat_interleave(fan, dim=0) + branch, pred


def step_pair_nested(cost, b1, b2, fan, low):
    """Two steps in one pass over ``cost``.

    Returns ``(front, pred1, pred2)`` with both predecessor planes shaped
    ``[low, n]`` exactly as the sequential spelling writes them.
    """
    n = cost.shape[1]
    low2 = low // fan
    if low2 < 1:
        raise ValueError(
            f"the pair needs L >= 2R; low={low} cannot be split by fan={fan}"
        )
    # front[f*LOW + g*LOW2 + v] -> [f, g, v, n]; minimise over f.
    best1, arg_f = cost.view(fan, fan, low2, n).min(dim=0)    # [g, v, n]
    # class li1 = g*LOW2 + v is exactly this plane read row-major.
    pred1 = arg_f.reshape(low, n)
    # b1 state = li1*F + j = g*LOW + v*F + j -> [g, v, j, n].
    mid = best1.unsqueeze(2) + b1.view(fan, low2, fan, n)     # [g, v, j, n]
    best2, arg_g = mid.min(dim=0)                             # [v, j, n]
    # class li2 = v*F + j is that plane read row-major.
    pred2 = arg_g.reshape(low, n)
    # b2 state = li2*F + j2 = v*F*F + j*F + j2 -> [v, j, j2, n].
    out = best2.unsqueeze(2) + b2.view(low2, fan, fan, n)     # [v, j, j2, n]
    return out.reshape(fan * low, n), pred1, pred2


def viterbi_window_paired(targets, vectors, window_bits, rate,
                          weights=None, chunk=512):
    """``encode.viterbi_window``'s reference loop with the steps paired.

    Same arguments, same returns.  An odd step count leaves one trailing
    sequential step, which is the honest way to handle it: the pair is an
    arithmetic identity on two steps and inventing a third changes the
    answer.
    """
    device = targets.device
    rows, cols = targets.shape
    size, arity = vectors.shape
    steps = rows // arity
    fan = 1 << rate
    low = size >> rate
    if not pair_is_applicable(window_bits, rate):
        raise ValueError(f"the pair needs L >= 2R; L={window_bits} R={rate}")

    tuples = targets.float().reshape(steps, arity, cols)
    wrows = None if weights is None else weights.float().reshape(steps, arity, cols)
    table = vectors.float().to(device)
    states = torch.empty(steps, cols, dtype=torch.long, device=device)
    sse = 0.0
    for start in range(0, cols, chunk):
        x = tuples[:, :, start : start + chunk]
        n = x.shape[2]
        w = None if wrows is None else wrows[:, :, start : start + chunk]
        cost = torch.full((size, n), float("inf"), device=device)
        cost[0] = 0.0
        back = torch.empty(steps, low, n,
                           dtype=torch.uint8 if fan <= 256 else torch.int32,
                           device=device)
        step = 0
        while step + 1 < steps:
            b1 = branch_costs(x[step], table, None if w is None else w[step])
            b2 = branch_costs(x[step + 1], table,
                              None if w is None else w[step + 1])
            cost, p1, p2 = step_pair_nested(cost, b1, b2, fan, low)
            back[step] = p1.to(back.dtype)
            back[step + 1] = p2.to(back.dtype)
            step += 2
        if step < steps:
            b = branch_costs(x[step], table, None if w is None else w[step])
            cost, p = step_sequential(cost, b, fan, low)
            back[step] = p.to(back.dtype)
        final, state = cost.min(dim=0)
        sse += float(final.sum())
        column = torch.empty(steps, n, dtype=torch.long, device=device)
        for s in range(steps - 1, -1, -1):
            column[s] = state
            lowbits = state >> rate
            pred = back[s].gather(0, lowbits.unsqueeze(0)).squeeze(0).long()
            state = (pred << (window_bits - rate)) | lowbits
        states[:, start : start + chunk] = column
    return states, sse


# ---------------------------------------------------------------------------
# The larger lever on the same axis: carry the class minimum, not the front.
# ---------------------------------------------------------------------------
#
# The recurrence closes in ``best`` alone.  The reference writes a full
# ``[2^L, n]`` front every step only so the next step can minimise it away
# again, and the fused kernel inherits that: one front in, one front out.  But
# substituting ``cost_s[f*LOW + c] = best_{s-1}[(f*LOW + c) >> R] +
# B_{s-1}(f*LOW + c)`` into the class minimum removes the front entirely:
#
#     best_s[c]  = min over f of ( best_{s-1}[(f*LOW + c) >> R]
#                                  + B_{s-1}(f*LOW + c) )    (first argmin)
#     back[s][c] = that argmin
#
# The scan now compares sums rather than raw fronts -- but those sums are the
# *same floats* the reference stores to ``cost_s`` and then scans, in the same
# order, under the same strict ``<``.  Nothing about the tie rule moves, which
# is why this needs no new exactness argument at all; the pair above needed
# one only because it moved a minimum across an addition.
#
# Resident state per column falls from ``2^L`` floats to ``2^(L-R)``, and the
# stores fall by the same factor: at L=14, R=3 that is 18,432 B a step against
# 133,120 B.  The last step is left in the front-producing form so the epilogue
# min and the ``sse`` sum see the reference's tensor.
#
# This module is still arithmetic, not a kernel.  The byte counts above are
# what the recurrence permits; what a kernel achieves is a kernel A/B's answer.


def step_best_form(best_prev, branch, fan, low, size):
    """``best_{s}`` and its argmin straight from ``best_{s-1}``, no front.

    Written as the scan a kernel would run: one ``[low, n]`` candidate live
    at a time, never the ``[size, n]`` front.
    """
    device = best_prev.device
    c = torch.arange(low, device=device)
    best = None
    pred = None
    for f in range(fan):
        states = f * low + c                                  # [low]
        cand = best_prev[states // fan] + branch[states]       # [low, n]
        if f == 0:
            best = cand
            pred = torch.zeros(low, best_prev.shape[1], dtype=torch.long,
                               device=device)
            continue
        take = cand < best
        best = torch.where(take, cand, best)
        pred = torch.where(take, f, pred)
    return best, pred


def front_from_best(best, branch, fan):
    """The reference's ``best.repeat_interleave(fan) + branch``."""
    return best.repeat_interleave(fan, dim=0) + branch


def viterbi_window_best_form(targets, vectors, window_bits, rate,
                             weights=None, chunk=512):
    """``encode.viterbi_window``'s reference loop carrying ``best``."""
    device = targets.device
    rows, cols = targets.shape
    size, arity = vectors.shape
    steps = rows // arity
    fan = 1 << rate
    low = size >> rate

    tuples = targets.float().reshape(steps, arity, cols)
    wrows = None if weights is None else weights.float().reshape(steps, arity, cols)
    table = vectors.float().to(device)
    states = torch.empty(steps, cols, dtype=torch.long, device=device)
    sse = 0.0
    for start in range(0, cols, chunk):
        x = tuples[:, :, start : start + chunk]
        n = x.shape[2]
        w = None if wrows is None else wrows[:, :, start : start + chunk]
        back = torch.empty(steps, low, n,
                           dtype=torch.uint8 if fan <= 256 else torch.int32,
                           device=device)
        # Step 0 reads the pinned front, whose class minimum is closed form:
        # state 0 is the only finite entry, so class 0 takes 0.0 and every
        # other class ties at inf -- both resolving to predecessor 0.
        best = torch.full((low, n), float("inf"), device=device)
        best[0] = 0.0
        back[0] = 0
        for step in range(steps - 1):
            b = branch_costs(x[step], table, None if w is None else w[step])
            best, pred = step_best_form(best, b, fan, low, size)
            back[step + 1] = pred.to(back.dtype)
        # The last step materialises the front the epilogue is defined on.
        b = branch_costs(x[steps - 1], table,
                         None if w is None else w[steps - 1])
        cost = front_from_best(best, b, fan)
        final, state = cost.min(dim=0)
        sse += float(final.sum())
        column = torch.empty(steps, n, dtype=torch.long, device=device)
        for s in range(steps - 1, -1, -1):
            column[s] = state
            lowbits = state >> rate
            pred = back[s].gather(0, lowbits.unsqueeze(0)).squeeze(0).long()
            state = (pred << (window_bits - rate)) | lowbits
        states[:, start : start + chunk] = column
    return states, sse
