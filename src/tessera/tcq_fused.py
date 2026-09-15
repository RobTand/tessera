"""The coset TCQ trellis, fused, on the GPU it is meant to run on.

``encode._TCQPlan.run`` is the definition: an exact Viterbi down every column
over a rate-1/2 convolutional code, where one super-step scores every anchor
of every subset at ``span`` positions, takes the best point per subset, folds
the per-position subset costs with a min-plus convolution over Z/4, and then
adds the folded cost onto a ``2^memory``-state front.  Written as torch ops it
is about forty generic kernels a super-step plus a traceback of the same
length, so a call is tens of thousands of launches whatever its width
(tessera#486: 82 % of a live GLM census row's host samples waited on them).

This module is the same recurrence in three launches:

  * ``_minima`` -- the best point per subset, for every (step, column,
    subset) at once.  It depends on the targets and the tables and on
    nothing the trellis decides, so it is computed for all steps in one
    launch, before the recurrence runs.  This is where the arithmetic is:
    ``4 * points * 2^completion * arity`` squared distances per position.
  * ``_forward`` -- the fold and the state recurrence, one column per lane,
    super-steps looped inside the program.  It writes the fold's argument
    and the branch choice, which are all the traceback reads.
  * ``_traceback`` -- the state walk backwards, one column per lane.

**Bit-exactness is the contract, not an aspiration.**  The anchors and the
body field this returns are identical to the reference's, and the ``sse`` is
the same float, which takes the same deliberate choices ``window_viterbi``
documents:

  * ``enable_fp_fusion=False`` on every launch, and every multiply written as
    the inline-asm ``mul.f32`` (``_mul``), because on Blackwell the packed
    NVPTX pair is contracted into an FMA even with fusion off.  The
    reference is separate torch kernels, which never contract.
  * the squared distance is associated exactly as the reference associates
    it: ``(t - d) * (t - d)`` (torch's CUDA ``pow`` at exponent 2 is that
    product -- checked, not assumed, in ``tests/test_tcq_fused.py``), then
    ``* w`` when weighted, then the sum over the ``arity`` coordinates.  At
    arity 1 and 2 that sum has one spelling (IEEE addition commutes), which
    is why the fused path is admitted at those two arities only.
  * every minimum that returns an index is a scan in index order with a
    strict ``<`` from an ``inf`` seed, so the winner is the FIRST minimal
    index: the point per subset, the fold's label per super-label and the
    branch's predecessor per state.  That is what ``torch.min(dim)`` returns
    on CUDA, ties included (checked on this box in the tests, including the
    all-``inf`` states of the first super-steps, where both branches tie).
    The descendant minimum returns a value only, and a minimum's value does
    not depend on its order.
  * the epilogue -- ``argmin`` over the final front, the gather and
    ``float(sum)`` -- is left to torch on a front of the reference's shape
    and layout, so the ``sse`` is summed in the reference's order.

The fused path takes float32, float16 and bfloat16 targets and weights,
whose front is float32 in the reference too (``promote_types(dtype,
float32)``), and converts them exactly with ``.float()``.  Float64 promotes
the reference's front to float64 and is refused here, as are arities above
2: ``tcq_fused_refusal`` names the reason, ``encode.viterbi_columns``'s
``auto`` falls back to the captured graph, and an explicit ``impl="fused"``
raises.  NaN targets are outside the contract, as they are for the window
kernels: the strict scans do not propagate NaN the way torch's reductions do.
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import torch

from .alphabet import SUBSET_COUNT
from .window_viterbi import fused_available

__all__ = ["fused_available", "tcq_fused_refusal", "viterbi_columns_fused"]

#: The dtypes whose front the reference keeps in float32.
_ADMITTED = (torch.float32, torch.float16, torch.bfloat16)

#: How many super-steps one ``_forward`` launch runs.  ``0`` is all of them --
#: one launch a call.  A positive value launches the same kernel over
#: consecutive ranges, which is the per-step spelling an A/B measures against.
#: A measurement knob, never a correctness one: the front ping-pongs on the
#: super-step's parity whatever the ranges are, so every value writes the same
#: bytes and the same ``sse`` float.  Read per call.
_SUPERS_ENV = "TESSERA_TCQ_FUSED_SUPERS"
#: ``BS,BC,W;BC,W;BC,W``: the minima tile (steps x columns per program, warps),
#: the forward's columns per program and warps, and the traceback's.  Unset
#: takes ``_tiles``.  Every tile writes identical bytes -- each kernel masks
#: both of its axes -- so a screen that cannot show identical anchors, bits
#: and ``sse`` for every configuration has measured something else.
_TILE_ENV = "TESSERA_TCQ_FUSED_TILE"
#: The ``loop_unroll_factor`` of the minima's point and descendant loops.  The
#: window scan's measured policy (``window_viterbi._scan_unroll``) is the
#: default; the knob exists so it can be re-measured on this kernel's loads.
_UNROLL_ENV = "TESSERA_TCQ_FUSED_UNROLL"


def tcq_fused_refusal(targets: torch.Tensor, weights: "torch.Tensor | None",
                      arity: int) -> "str | None":
    """Why the fused trellis cannot serve this call, or ``None`` when it can."""
    if not targets.is_cuda:
        return f"targets are on {targets.device}"
    if not fused_available():
        return "triton is absent, or this torch is a HIP build the NVPTX kernel cannot target"
    if targets.dim() != 2:
        return f"targets are {targets.dim()}-dimensional"
    if targets.dtype not in _ADMITTED:
        return (f"targets are {targets.dtype}, whose reference front is not float32; "
                "the fused kernel is float32 end to end")
    if weights is not None:
        if weights.dtype not in _ADMITTED:
            return (f"weights are {weights.dtype}, which promotes the reference's "
                    "branch metric past float32")
        if weights.device != targets.device:
            return f"weights are on {weights.device}, targets on {targets.device}"
        if weights.shape != targets.shape:
            return f"weights are {tuple(weights.shape)} for {tuple(targets.shape)} targets"
    if arity not in (1, 2):
        return (f"arity {arity}: the coordinate sum has one IEEE spelling at arity 1 "
                "and 2 only")
    return None


@dataclass(frozen=True)
class _Tables:
    dsub: torch.Tensor        # [4P, 2^c, arity] fp32: descendant values, subset order
    anchor: torch.Tensor      # [4P] int32: anchor index per (subset, point)
    prev: torch.Tensor        # [2S] int32: predecessor per (side, state)
    subset: torch.Tensor      # [2S] int32: super-label per (side, state)
    points: int
    descendants: int


@functools.lru_cache(maxsize=32)
def _tables(forest, completion: int, code, device) -> _Tables:
    """The reference's own tables, laid out for the kernels.

    Built FROM ``encode``'s memoised tables, so the two paths cannot read
    different trellises.  ``dsub`` is a gather of float values, which is
    exact; the integer tables are narrowed to int32, which every index here
    fits.
    """
    from .encode import _descendant_values, _subset_table, _transition_tables
    from .trellis import TCQ

    dvals = _descendant_values(forest, completion, device)          # [A, 2^c, k]
    subsets = _subset_table(TCQ(forest, code), device)              # [4, P]
    prev, subset_of = _transition_tables(code, device)              # [2, S] each
    points = subsets.shape[1]
    if subsets.shape[0] != SUBSET_COUNT or dvals.shape[0] != SUBSET_COUNT * points:
        raise ValueError(
            f"{dvals.shape[0]} anchors do not split into {SUBSET_COUNT} subsets of {points}")
    return _Tables(
        dsub=dvals[subsets.reshape(-1)].contiguous(),
        anchor=subsets.reshape(-1).to(torch.int32).contiguous(),
        prev=prev.reshape(-1).to(torch.int32).contiguous(),
        subset=subset_of.reshape(-1).to(torch.int32).contiguous(),
        points=points, descendants=dvals.shape[1],
    )


def _build():
    import triton
    import triton.language as tl

    @triton.jit
    def _mul(a, b):
        """A multiply no compiler may fold into the add that consumes it.

        ``window_viterbi._mul`` records the measurement: on Blackwell the
        packed ``mul.rn.f32x2`` / ``add.rn.f32x2`` pair is contracted into an
        FMA even under ``enable_fp_fusion=False``.  Inline asm is opaque.
        """
        return tl.inline_asm_elementwise("mul.f32 $0, $1, $2;", "=f,f,f", [a, b],
                                         dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit(do_not_specialize=["cols", "steps"])
    def _minima(xptr, wptr, dsub, best, point, cols, steps,
                ARITY: tl.constexpr, P: tl.constexpr, J: tl.constexpr,
                HAS_W: tl.constexpr, POINT_U8: tl.constexpr,
                BS: tl.constexpr, BC: tl.constexpr, UNROLL: tl.constexpr):
        """Best point per subset for a tile of ``BS`` steps x ``BC`` columns.

        The reference's ``err = ((t - d)**2 [* w]).sum(k).amin(2^c)`` and
        ``by_subset.min(dim=2)``, per lane ``(step, column, subset)``: the
        points are a strict-``<`` scan in point order from an ``inf`` seed,
        so ``point`` is the first minimal index and ``best`` its value.
        ``best`` and ``point`` are ``[cols, steps, 4]``, a column's steps
        contiguous, because the two kernels after this walk one column a lane.
        """
        si = tl.program_id(0) * BS + tl.arange(0, BS)            # steps
        ci = tl.program_id(1) * BC + tl.arange(0, BC)            # columns
        m2 = (si < steps)[:, None] & (ci < cols)[None, :]
        m3 = tl.broadcast_to(m2[:, :, None], (BS, BC, 4))
        s64 = si.to(tl.int64)
        c64 = ci.to(tl.int64)
        lab = tl.arange(0, 4)
        # Coordinate k of step s is row s*ARITY + k: a tuple is ARITY
        # consecutive rows of one column (``_TCQPlan.bind``).
        xo = s64[:, None] * ARITY * cols + c64[None, :]
        x0 = tl.load(xptr + xo, mask=m2, other=0.0)[:, :, None]
        if HAS_W:
            w0 = tl.load(wptr + xo, mask=m2, other=0.0)[:, :, None]
        if ARITY == 2:
            x1 = tl.load(xptr + xo + cols, mask=m2, other=0.0)[:, :, None]
            if HAS_W:
                w1 = tl.load(wptr + xo + cols, mask=m2, other=0.0)[:, :, None]
        bst = tl.full([BS, BC, 4], float("inf"), tl.float32)
        pt = tl.zeros([BS, BC, 4], dtype=tl.int32)
        for p in tl.range(0, P, loop_unroll_factor=UNROLL):
            if J == 1:
                off = (lab * P + p) * ARITY
                d = x0 - tl.load(dsub + off)[None, None, :]
                e = _mul(d, d)
                if HAS_W:
                    e = _mul(e, w0)
                if ARITY == 2:
                    d = x1 - tl.load(dsub + off + 1)[None, None, :]
                    f = _mul(d, d)
                    if HAS_W:
                        f = _mul(f, w1)
                    e = e + f
            else:
                # The descendant minimum returns a value, so its order is free;
                # the strict ``<`` keeps it from depending on the lane.
                e = tl.full([BS, BC, 4], float("inf"), tl.float32)
                for j in tl.range(0, J, loop_unroll_factor=UNROLL):
                    off = ((lab * P + p) * J + j) * ARITY
                    d = x0 - tl.load(dsub + off)[None, None, :]
                    g = _mul(d, d)
                    if HAS_W:
                        g = _mul(g, w0)
                    if ARITY == 2:
                        d = x1 - tl.load(dsub + off + 1)[None, None, :]
                        f = _mul(d, d)
                        if HAS_W:
                            f = _mul(f, w1)
                        g = g + f
                    e = tl.where(g < e, g, e)
            take = e < bst
            bst = tl.where(take, e, bst)
            pt = tl.where(take, p, pt)
        boff = ((c64[None, :] * steps + s64[:, None]) * 4)[:, :, None] + lab[None, None, :]
        tl.store(best + boff, bst, mask=m3)
        if POINT_U8:
            tl.store(point + boff, pt.to(tl.uint8), mask=m3)
        else:
            tl.store(point + boff, pt, mask=m3)

    @triton.jit(do_not_specialize=["cols", "steps", "supers", "sup0", "sup1"])
    def _forward(best, front, accbuf, fold, choice, prev_tab, sub_tab,
                 cols, steps, supers, sup0, sup1,
                 S: tl.constexpr, SPAN: tl.constexpr, NFOLD: tl.constexpr,
                 BC: tl.constexpr):
        """The fold and the state recurrence for super-steps ``[sup0, sup1)``.

        ``front`` is ``[2, cols, S]`` and ping-pongs on the super-step's
        parity, so the front after super-step ``sup`` is half ``(sup+1) & 1``
        however the super-steps are split across launches.  The fold is the
        reference's ``terms[l, v] = acc[(l - v) mod 4] + best[v]`` scanned in
        ``v`` order with a strict ``<``; the branch is ``cost[prev] +
        acc[subset]`` on both sides and side 1 wins only when strictly less.
        """
        ci = tl.program_id(0) * BC + tl.arange(0, BC)
        mc = ci < cols
        c64 = ci.to(tl.int64)
        m4 = tl.broadcast_to(mc[:, None], (BC, 4))
        ms = tl.broadcast_to(mc[:, None], (BC, S))
        lab = tl.arange(0, 4)
        si = tl.arange(0, S)
        p0 = tl.load(prev_tab + si)
        p1 = tl.load(prev_tab + S + si)
        u0 = tl.load(sub_tab + si)
        u1 = tl.load(sub_tab + S + si)
        for sup in tl.range(sup0, sup1):
            h_in = sup & 1
            fin = front + h_in * cols * S + c64[:, None] * S          # [BC, 1]
            fout = front + (1 - h_in) * cols * S + c64[:, None] * S
            step0 = sup * SPAN
            if SPAN == 1:
                abase = best + (c64 * steps + step0) * 4                # [BC]
            else:
                for oo in tl.static_range(1, SPAN):
                    if oo == 1:
                        src = best + (c64 * steps + step0) * 4
                    else:
                        src = accbuf + c64 * 4
                    bstep = best + (c64 * steps + step0 + oo) * 4
                    cur = tl.full([BC, 4], float("inf"), tl.float32)
                    arg = tl.zeros([BC, 4], dtype=tl.int32)
                    for v in tl.static_range(0, 4):
                        a = tl.load(src[:, None] + ((lab[None, :] - v) & 3), mask=m4,
                                    other=0.0)
                        b = tl.load(bstep + v, mask=mc, other=0.0)
                        t = a + b[:, None]
                        tk = t < cur
                        cur = tl.where(tk, t, cur)
                        arg = tl.where(tk, v, arg)
                    tl.store(accbuf + c64[:, None] * 4 + lab[None, :], cur, mask=m4)
                    tl.store(fold + ((c64[:, None] * supers + sup) * NFOLD + (oo - 1)) * 4
                             + lab[None, :], arg.to(tl.uint8), mask=m4)
                abase = accbuf + c64 * 4
            c0 = tl.load(fin + p0[None, :], mask=ms, other=0.0) \
                + tl.load(abase[:, None] + u0[None, :], mask=ms, other=0.0)
            c1 = tl.load(fin + p1[None, :], mask=ms, other=0.0) \
                + tl.load(abase[:, None] + u1[None, :], mask=ms, other=0.0)
            take = c1 < c0
            tl.store(fout + si[None, :], tl.where(take, c1, c0), mask=ms)
            tl.store(choice + (c64[:, None] * supers + sup) * S + si[None, :],
                     take.to(tl.uint8), mask=ms)

    @triton.jit(do_not_specialize=["cols", "steps", "supers"])
    def _traceback(end, choice, fold, point, prev_tab, sub_tab, anchor_tab,
                   anchors, bits, cols, steps, supers,
                   S: tl.constexpr, SPAN: tl.constexpr, NFOLD: tl.constexpr,
                   P: tl.constexpr, MEMSHIFT: tl.constexpr, SHIFT: tl.constexpr,
                   BC: tl.constexpr):
        """The reference's traceback, one column a lane, super-steps backwards."""
        ci = tl.program_id(0) * BC + tl.arange(0, BC)
        mask = ci < cols
        c64 = ci.to(tl.int64)
        state = tl.load(end + ci, mask=mask, other=0).to(tl.int32)
        for i in range(0, supers):
            sup = supers - 1 - i
            srow = c64 * supers + sup
            side = tl.load(choice + srow * S + state, mask=mask, other=0).to(tl.int32)
            label = tl.load(sub_tab + side * S + state, mask=mask, other=0)
            # The input bit is read off the state, not off ``side``: both
            # predecessors of a state share it (``_TCQPlan.run``).
            select = (state >> MEMSHIFT) & 1
            for o in tl.static_range(1, SPAN):
                oo = SPAN - o
                v = tl.load(fold + (srow * NFOLD + (oo - 1)) * 4 + label, mask=mask,
                            other=0).to(tl.int32)
                step = c64 * 0 + sup * SPAN + oo
                pt = tl.load(point + (c64 * steps + step) * 4 + v, mask=mask,
                             other=0).to(tl.int32)
                anc = tl.load(anchor_tab + v * P + pt, mask=mask, other=0)
                tl.store(anchors + step * cols + c64, anc.to(tl.int64), mask=mask)
                tl.store(bits + step * cols + c64, ((v << SHIFT) | pt).to(tl.int64),
                         mask=mask)
                label = (label - v) & 3
            step = c64 * 0 + sup * SPAN
            pt = tl.load(point + (c64 * steps + step) * 4 + label, mask=mask,
                         other=0).to(tl.int32)
            anc = tl.load(anchor_tab + label * P + pt, mask=mask, other=0)
            tl.store(anchors + step * cols + c64, anc.to(tl.int64), mask=mask)
            tl.store(bits + step * cols + c64, ((select << SHIFT) | pt).to(tl.int64),
                     mask=mask)
            state = tl.load(prev_tab + side * S + state, mask=mask, other=0)

    return _minima, _forward, _traceback


_CACHE: dict = {}


def _kernels():
    if "k" not in _CACHE:
        _CACHE["k"] = _build()
    return _CACHE["k"]


def _pow2(n: int, cap: int) -> int:
    b = 1
    while b < n and b < cap:
        b *= 2
    return b


def _tiles(steps: int, cols: int, states: int):
    """``((BS, BC, W), (BC, W), (BC, W))`` for the three kernels.

    The minima's lanes are ``BS * BC * 4``, sized at about 256 a program with
    the point loop inside, the shape ``window_viterbi._tile_best`` measured
    for a scan carried in a loop; the forward's are one column's ``S``
    states times ``BC`` columns, about two thousand; the traceback's are
    columns.  Starting points, not measurements: ``_TILE_ENV`` sweeps them.
    """
    bc = _pow2(cols, 4)
    bs = _pow2(steps, max(1, 64 // bc))
    minima = (bs, bc, max(1, min(8, bs * bc * 4 // 64)))
    fbc = _pow2(cols, max(1, 2048 // states))
    forward = (fbc, max(1, min(8, fbc * states // 512)))
    tbc = _pow2(cols, 128)
    traceback = (tbc, max(1, min(4, tbc // 32)))
    return minima, forward, traceback


def _resolve_tiles(steps: int, cols: int, states: int):
    raw = os.environ.get(_TILE_ENV, "")
    if raw == "":
        return _tiles(steps, cols, states)
    try:
        parts = [[int(x) for x in part.split(",")] for part in raw.split(";")]
    except ValueError as exc:
        raise ValueError(f"{_TILE_ENV}={raw!r} is not BS,BC,W;BC,W;BC,W") from exc
    if [len(p) for p in parts] != [3, 2, 2]:
        raise ValueError(f"{_TILE_ENV}={raw!r} is not BS,BC,W;BC,W;BC,W")
    for value in (v for p in parts for v in p):
        if value < 1 or value > (1 << 16) or value & (value - 1):
            raise ValueError(f"{_TILE_ENV}={raw!r}: {value} is not a power of two")
    return tuple(parts[0]), tuple(parts[1]), tuple(parts[2])


def _resolve_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not an integer") from exc
    if value < 0:
        raise ValueError(f"{name}={raw!r} is negative")
    return value


def viterbi_columns_fused(targets: torch.Tensor, weights: "torch.Tensor | None",
                          forest, code, completion: int, span: int):
    """The fused CUDA path behind ``encode.viterbi_columns``.

    Same arguments, same returns -- ``(anchor_index[steps, cols],
    body_field[steps, cols], sse)`` -- identical tensors and the identical
    ``sse`` float.  The caller has validated the shape (``rows`` a multiple
    of the arity, ``steps`` of the span) and ``tcq_fused_refusal`` has
    admitted the call; callers go through ``viterbi_columns``, and this entry
    point exists so the tests can pin the implementation.
    """
    import triton

    minima, forward, traceback = _kernels()
    device = targets.device
    rows, cols = targets.shape
    arity = forest.grid.arity
    steps = rows // arity
    supers = steps // span
    if steps == 0 or cols == 0:
        # The reference runs no step and no traceback: its anchors are an
        # empty ``[steps, cols]`` and its sse sums the pinned start's zeros
        # (or nothing), exactly ``0.0``.  Launching nothing answers the same.
        empty = torch.empty(steps, cols, dtype=torch.long, device=device)
        return empty, torch.empty_like(empty), 0.0

    tabs = _tables(forest, completion, code, device)
    states = code.states
    points = tabs.points
    point_u8 = points <= 256
    # ``.float()`` is exact from every admitted dtype and a no-op on float32;
    # it is the conversion the reference's ``target - dvals`` performs.
    x = targets.float().contiguous()
    w = None if weights is None else weights.float().contiguous()
    (bs, bc, warps), (fbc, fwarps), (tbc, twarps) = _resolve_tiles(steps, cols, states)
    unroll = _resolve_int(_UNROLL_ENV, 32)

    best = torch.empty(cols, steps, SUBSET_COUNT, dtype=torch.float32, device=device)
    point = torch.empty(cols, steps, SUBSET_COUNT,
                        dtype=torch.uint8 if point_u8 else torch.int32, device=device)
    minima[(triton.cdiv(steps, bs), triton.cdiv(cols, bc))](
        x, x if w is None else w, tabs.dsub, best, point, cols, steps,
        ARITY=arity, P=points, J=tabs.descendants, HAS_W=w is not None,
        POINT_U8=point_u8, BS=bs, BC=bc, UNROLL=unroll,
        num_warps=warps, enable_fp_fusion=False,
    )

    nfold = max(span - 1, 1)
    front = torch.empty(2, cols, states, dtype=torch.float32, device=device)
    front[0].fill_(float("inf"))
    front[0, :, 0] = 0.0
    accbuf = (torch.empty(cols, SUBSET_COUNT, dtype=torch.float32, device=device)
              if span > 1 else best)
    fold = torch.empty(cols, supers, nfold, SUBSET_COUNT, dtype=torch.uint8, device=device)
    choice = torch.empty(cols, supers, states, dtype=torch.uint8, device=device)
    per_launch = _resolve_int(_SUPERS_ENV, 0) or supers
    for sup0 in range(0, supers, per_launch):
        forward[(triton.cdiv(cols, fbc),)](
            best, front, accbuf, fold, choice, tabs.prev, tabs.subset,
            cols, steps, supers, sup0, min(supers, sup0 + per_launch),
            S=states, SPAN=span, NFOLD=nfold, BC=fbc,
            num_warps=fwarps, enable_fp_fusion=False,
        )

    # The reference's epilogue on a front of its shape: ``[cols, states]``,
    # contiguous, so the gather and the sum run the reference's kernels over
    # the reference's layout and the float is the same float.
    cost = front[supers & 1]
    end = torch.argmin(cost, dim=1)
    anchors = torch.empty(steps, cols, dtype=torch.long, device=device)
    bits = torch.empty(steps, cols, dtype=torch.long, device=device)
    traceback[(triton.cdiv(cols, tbc),)](
        end, choice, fold, point, tabs.prev, tabs.subset, tabs.anchor,
        anchors, bits, cols, steps, supers,
        S=states, SPAN=span, NFOLD=nfold, P=points, MEMSHIFT=code.memory - 1,
        SHIFT=points.bit_length() - 1, BC=tbc,
        num_warps=twarps, enable_fp_fusion=False,
    )
    return anchors, bits, float(cost.gather(1, end.unsqueeze(1)).sum())
