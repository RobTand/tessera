"""The window body's Viterbi, fused, on the GPU it is meant to run on.

``encode.viterbi_window`` is the definition: an exact Viterbi over ``2^L``
states for a column stream of ``R`` bits per position, one step being a
``[2^R, 2^(L-R)]`` minimum followed by a per-state branch cost.  Written as a
chain of torch ops it is a chain of *materialised* ``[2^L, cols]`` fp32
intermediates -- a view, a min, a repeat_interleave, a subtract, a square, a
sum, an add -- so a step moves eight fronts through memory to do one front's
worth of arithmetic.  That is why the reference reads 96% "utilisation" at a
third of this board's power envelope: it is bandwidth-bound on scratch.

This module is the same recurrence with the scratch removed:

  * one Triton kernel per step does the class minimum, the traceback write
    and the branch cost together -- one front in, one front out;
  * the column batch is sized so both fronts stay in L2 across the whole step
    loop, which is what turns the front traffic from DRAM into cache -- at
    L=12 the step measures 398 GB/s of front traffic against this board's
    273 GB/s of DRAM, so the cache is doing the work;
  * the traceback is one kernel down the stored predecessor bytes, instead of
    a gather launched per step.

**Bit-exactness is the contract, not an aspiration.**  The states this
returns are identical to the reference's and the sse is the same float, which
takes four things the kernels do deliberately:

  * ``enable_fp_fusion=False`` on every launch.  Triton contracts ``a*b + c``
    into an FFMA by default -- measured here, 246017 differing results in a
    2^20-element probe -- while the reference, being separate torch kernels,
    never contracts.
  * that flag is *not sufficient* on Blackwell, and the multiply that feeds
    an add is written as inline asm.  See ``_mul``.
  * the class minimum is an unrolled scan over the ``2^R`` predecessors with
    a strict ``<``, so the winner is the *first* minimal index.  That is what
    ``torch.min(dim=0)`` returns (verified on this box for CPU and CUDA,
    including the all-``inf`` classes at step 0, where every candidate ties).
  * the branch cost is associated exactly as the reference associates it:
    ``(d*d) * w`` per coordinate, then the arity sum, then ``best + branch``.

The epilogue -- the final ``min`` over states and ``sse += float(final.sum())``
-- is left to torch on the assembled front, so the summation order over a
chunk is the reference's summation order and the sse is the same float, not
merely the same number to a tolerance.
"""
from __future__ import annotations

import os

import torch

__all__ = ["fused_available", "viterbi_window_fused"]

#: The step loop's working set is two fronts of ``width`` columns; this is the
#: byte budget that sets ``width``.  GB10's L2 is 24 MB and a sixth of it is
#: the knee, with a cliff after it: at L=16, 5.34 s at 4 MB, 6.23 s at 8 MB,
#: 14.59 s at 16 MB (L=14: 1.31 / 1.59 / 3.78 s).  It is a measurement knob,
#: never a correctness one -- every value returns the same bytes, `sse`
#: included.
_L2_BUDGET = int(os.environ.get("TESSERA_WINDOW_L2_BYTES", "0")) or None
# One alternative was built, measured and dropped; the number is here so
# nobody pays for it twice.  A *register-resident column kernel* -- one block
# per column holding the whole 2^L front in registers for every step, so the
# only traffic is the traceback byte -- is exact and lost: at L=12, R=4,
# 2048x4096 it ran 1.156 s against this module's 0.980 s, identical states and
# identical sse.  The front never leaves the block, but tl.min(axis=0,
# return_indices) and the reshape that shifts the front are block-wide
# barriers twice a step, where the tiled kernel's scan is thread-local --
# holding the front costs more than moving it through L2.

#: Whether the batch is captured as a CUDA graph: unset lets the break-even in
#: ``viterbi_window_fused`` decide, 0 forces the eager loop, 1 forces capture.
#: Principle 10 asks for graphs where they apply, and here they do: the
#: profiler puts a step at 4.8-4.9 us of kernel while the eager loop takes
#: 9.9 us of wall for it, and replay takes 6.3 us.  Whole-tensor, that is 2x
#: at L=14 and L=16 (2.74 -> 1.34 s, 10.88 -> 5.30 s).
#:
#: An earlier reading here said graphs did not help.  It was wrong and the
#: mistake is worth keeping: it compared a ``cudaEvent`` span against wall,
#: and an event span measures the *stream*, launch gaps included, so it read
#: 19.1 ms of "pure board time" for 21.7 ms of wall and hid exactly the gap
#: the graph closes.  Only the per-kernel profiler number exposes it.
_GRAPH = ({"0": False, "1": True}.get(os.environ.get("TESSERA_WINDOW_GRAPH", ""))
          if os.environ.get("TESSERA_WINDOW_GRAPH") else None)


def fused_available() -> bool:
    """Whether the fused CUDA path can run at all."""
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except Exception:                                        # pragma: no cover
        return False
    return True


def _build():
    import triton
    import triton.language as tl

    @triton.jit
    def _mul(a, b):
        """A multiply no compiler may fold into the add that consumes it.

        ``enable_fp_fusion=False`` reaches ptxas as ``--fmad=false``, and for
        *scalar* fp32 that is enough.  It is not enough here: the NVPTX
        backend packs this arithmetic into Blackwell's ``mul.rn.f32x2`` /
        ``add.rn.f32x2``, and the packed pair is contracted anyway --
        measured, the fused path returned exactly ``fma(d0, d0, e1)`` where
        the reference returns ``(d0*d0) + e1``, one ulp apart on a third of
        the elements at arity 2.  An inline-asm multiply is opaque, so every
        add downstream of it keeps its own rounding, which is the
        reference's rounding.
        """
        return tl.inline_asm_elementwise("mul.f32 $0, $1, $2;", "=f,f,f", [a, b],
                                         dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _init(front, ctl, SIZE: tl.constexpr, BS: tl.constexpr, BC: tl.constexpr):
        """The pinned start: state 0 free, every other state unreachable."""
        m = tl.load(ctl + 2)
        ci = tl.program_id(1) * BC + tl.arange(0, BC)
        si = tl.program_id(0) * BS + tl.arange(0, BS)
        val = tl.broadcast_to(tl.where(si[None, :] == 0, 0.0, float("inf")), (BC, BS))
        tl.store(front + ci[:, None] * SIZE + si[None, :], val,
                 mask=(ci < m)[:, None] & (si < SIZE)[None, :])

    @triton.jit
    def _copy_front(src, dst, ctl, SIZE: tl.constexpr, BS: tl.constexpr,
                    BC: tl.constexpr):
        """The batch's final front, into the chunk-wide front the epilogue reads."""
        loff = tl.load(ctl + 1)
        m = tl.load(ctl + 2)
        ci = tl.program_id(1) * BC + tl.arange(0, BC)
        si = tl.program_id(0) * BS + tl.arange(0, BS)
        mask = (ci < m)[:, None] & (si < SIZE)[None, :]
        v = tl.load(src + ci[:, None] * SIZE + si[None, :], mask=mask)
        tl.store(dst + (loff + ci)[:, None] * SIZE + si[None, :], v, mask=mask)

    @triton.jit(do_not_specialize=["step"])
    def _step(front_in, front_out, back, xptr, wptr, table, ctl,
              cols, steps, step, low,
              ARITY: tl.constexpr, FAN: tl.constexpr, SIZE: tl.constexpr,
              HAS_W: tl.constexpr, BACK_U8: tl.constexpr,
              BL: tl.constexpr, BC: tl.constexpr):
        """One trellis step over a tile of ``BL`` classes x ``BC`` columns.

        The front is ``[width, SIZE]`` -- a column's states contiguous -- so
        the class minimum reads a run of ``BL`` floats per predecessor and
        the new front is written as a run of ``BL * FAN``.  ``low`` is
        ``2^(L-R)``; the traceback is ``[n, steps, low]``.  ``goff`` is the
        batch's first column in the tensor, ``loff`` its first column inside
        the chunk and ``m`` its width -- all three read from ``ctl`` on the
        device, so one captured graph replays for every batch.
        """
        goff = tl.load(ctl + 0)
        loff = tl.load(ctl + 1)
        m = tl.load(ctl + 2)
        li = tl.program_id(0) * BL + tl.arange(0, BL)            # classes
        ci = tl.program_id(1) * BC + tl.arange(0, BC)            # columns
        mask_l = li < low
        mask_c = ci < m
        m2 = mask_c[:, None] & mask_l[None, :]
        base = ci[:, None] * SIZE                                # [BC, 1]

        # --- the class minimum over 2^R predecessors, first minimal index.
        best = tl.load(front_in + base + li[None, :], mask=m2, other=float("inf"))
        pred = tl.zeros([BC, BL], dtype=tl.int32)
        for f in tl.static_range(1, FAN):
            v = tl.load(front_in + base + (f * low + li)[None, :], mask=m2,
                        other=float("inf"))
            take = v < best
            best = tl.where(take, v, best)
            pred = tl.where(take, f, pred)

        # --- the traceback: the winning predecessor's top R bits, per class.
        # int64: at L=16, R=4 the per-column traceback stride is steps*low =
        # 8388608 bytes, so column 256 is exactly 2^31 and an i32 offset wraps
        # to before the allocation.
        slot = back + (loff + ci)[:, None].to(tl.int64) * (steps * low) \
            + step * low + li[None, :]
        if BACK_U8:
            tl.store(slot, pred.to(tl.uint8), mask=m2)
        else:
            tl.store(slot, pred, mask=m2)

        # --- the branch cost per state, then the new front.
        state = li[None, :, None] * FAN + tl.arange(0, FAN)[None, None, :]
        ml3 = mask_l[None, :, None]
        mc3 = mask_c[:, None, None]
        xoff = step * (ARITY * cols) + goff + ci
        d = tl.load(xptr + xoff, mask=mask_c, other=0.0)[:, None, None] \
            - tl.load(table + state * ARITY, mask=ml3, other=0.0)
        cost = _mul(d, d)
        if HAS_W:
            cost = _mul(cost, tl.load(wptr + xoff, mask=mask_c, other=0.0)[:, None, None])
        for k in tl.static_range(1, ARITY):
            d = tl.load(xptr + xoff + k * cols, mask=mask_c, other=0.0)[:, None, None] \
                - tl.load(table + state * ARITY + k, mask=ml3, other=0.0)
            e = _mul(d, d)
            if HAS_W:
                e = _mul(e, tl.load(wptr + xoff + k * cols, mask=mask_c,
                                    other=0.0)[:, None, None])
            cost = cost + e
        cost = best[:, :, None] + cost
        tl.store(front_out + base[:, :, None] + state, cost, mask=mc3 & ml3)

    @triton.jit
    def _traceback(back, final_state, states, n, cols, steps, low, col0,
                   RATE: tl.constexpr, SHIFT: tl.constexpr, BC: tl.constexpr):
        """The shift register run backwards: ``s = (pred << (L-R)) | (s >> R)``."""
        ci = tl.program_id(0) * BC + tl.arange(0, BC)
        mask = ci < n
        s = tl.load(final_state + ci, mask=mask, other=0).to(tl.int32)
        cbase = ci.to(tl.int64) * (steps * low)          # 2^31 at L=16, R=4
        for i in range(0, steps):
            t = steps - 1 - i
            tl.store(states + t * cols + col0 + ci, s.to(tl.int64), mask=mask)
            lowbits = s >> RATE
            p = tl.load(back + cbase + t * low + lowbits, mask=mask, other=0).to(tl.int32)
            s = (p << SHIFT) | lowbits

    return _step, _traceback, _init, _copy_front


_CACHE: dict = {}


def _kernels():
    if "k" not in _CACHE:
        _CACHE["k"] = _build()
    return _CACHE["k"]


def _tile(fan: int, low: int, n: int):
    """Class/column tile and warp count: about two thousand elements a
    program, never wider than the problem.  Every choice writes identical
    bytes; this one is the fast one."""
    bl = 1
    while bl < low and bl * fan < 1024:
        bl *= 2
    bl = min(bl, low)
    bc = 1
    while bc < n and bl * fan * bc < 2048:
        bc *= 2
    warps = max(1, min(8, (bl * fan * bc) // 512))
    return bl, bc, warps


def viterbi_window_fused(targets, vectors, window_bits: int, rate: int,
                         weights=None, chunk: int = 512):
    """The fused CUDA path behind ``encode.viterbi_window``.

    Same arguments, same returns, identical states and identical sse.
    Callers go through ``viterbi_window``; this entry point exists so the
    tests can pin the implementation.
    """
    import triton

    step_kernel, traceback_kernel, init_kernel, copy_kernel = _kernels()
    device = targets.device
    rows, cols = targets.shape
    size, arity = vectors.shape
    steps = rows // arity
    fan = 1 << rate
    low = size >> rate
    back_u8 = fan <= 256

    tuples = targets.float().reshape(steps, arity, cols).contiguous()
    wrows = None if weights is None else \
        weights.float().reshape(steps, arity, cols).contiguous()
    table = vectors.float().to(device).contiguous()

    states = torch.empty(steps, cols, dtype=torch.long, device=device)
    sse = 0.0

    budget = _L2_BUDGET
    if budget is None:
        budget = getattr(torch.cuda.get_device_properties(device),
                         "L2_cache_size", 24 << 20) // 6
    nmax = min(chunk, cols)
    width = max(1, min(budget // (2 * size * 4), nmax))

    front_all = torch.empty(nmax, size, dtype=torch.float32, device=device)
    back = torch.empty(nmax, steps, low,
                       dtype=torch.uint8 if back_u8 else torch.int32, device=device)
    cur = torch.empty(width, size, dtype=torch.float32, device=device)
    nxt = torch.empty(width, size, dtype=torch.float32, device=device)

    bl, bc, warps = _tile(fan, low, width)
    grid = (triton.cdiv(low, bl), triton.cdiv(width, bc))
    bs = min(size, 1024)
    cgrid = (triton.cdiv(size, bs), triton.cdiv(width, max(1, 2048 // bs)))
    cbc = max(1, 2048 // bs)

    # Every batch is the same 2 + steps launches with the same pointers; only
    # the descriptor moves.  So the whole batch is captured once and replayed,
    # which is what principle 10 asks for and what the clock agrees with:
    # measured on this board, a step is 5.0-5.5 us of kernel (profiler,
    # ``_step`` self CUDA) against 9.9 us of wall eager and 6.3 us replayed.
    # Capture costs about 17 us a node and replay saves about 3.6 us, so the
    # break-even is five batches; below that the eager loop is cheaper.
    ctl = torch.empty(3, dtype=torch.int32, device=device)
    descs = []
    for start in range(0, cols, chunk):
        n = min(chunk, cols - start)
        for c0 in range(0, n, width):
            descs.append((start + c0, c0, min(width, n - c0)))
    desc = torch.tensor(descs, dtype=torch.int32, device=device)

    def one_batch():
        init_kernel[cgrid](cur, ctl, SIZE=size, BS=bs, BC=cbc, num_warps=4)
        a, b = cur, nxt
        for step in range(steps):
            step_kernel[grid](
                a, b, back, tuples, tuples if wrows is None else wrows, table,
                ctl, cols, steps, step, low,
                ARITY=arity, FAN=fan, SIZE=size, HAS_W=wrows is not None,
                BACK_U8=back_u8, BL=bl, BC=bc,
                num_warps=warps, enable_fp_fusion=False,
            )
            a, b = b, a
        copy_kernel[cgrid](a, front_all, ctl, SIZE=size, BS=bs, BC=cbc,
                           num_warps=4)

    graph = None
    if _GRAPH if _GRAPH is not None else len(descs) >= 6:
        ctl.copy_(desc[0])
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            one_batch()                      # compiles and warms; capture runs nothing
        torch.cuda.current_stream().wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            one_batch()

    b = 0
    for start in range(0, cols, chunk):
        n = min(chunk, cols - start)
        for _ in range(0, n, width):
            ctl.copy_(desc[b])
            b += 1
            if graph is None:
                one_batch()
            else:
                graph.replay()

        cost = front_all[:n].t().contiguous()                 # [size, n]
        final, state = cost.min(dim=0)                        # [n]
        sse += float(final.sum())
        tb = 128
        traceback_kernel[(triton.cdiv(n, tb),)](
            back, state.to(torch.int32), states, n, cols, steps, low, start,
            RATE=rate, SHIFT=window_bits - rate, BC=tb,
            num_warps=4, enable_fp_fusion=False,
        )
    return states, sse
