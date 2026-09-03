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
  * the class minimum is a scan over the ``2^R`` predecessors with a strict
    ``<``, so the winner is the *first* minimal index.  That is what
    ``torch.min(dim=0)`` returns (verified on this box for CPU and CUDA,
    including the all-``inf`` classes at step 0, where every candidate ties).
    The scan is spelled as a runtime loop rather than a flat unroll
    (``_scan_unroll``) -- which changes how many loads are in flight and not
    the order the selects run in, so it changes the clock and not the bytes.
  * the branch cost is associated exactly as the reference associates it:
    ``(d*d) * w`` per coordinate, then the arity sum, then ``best + branch``.

The epilogue -- the final ``min`` over states and ``sse += float(final.sum())``
-- is left to torch on the assembled front, so the summation order over a
chunk is the reference's summation order and the sse is the same float, not
merely the same number to a tolerance.
"""
from __future__ import annotations

import collections
import functools
import os
import threading

import torch

__all__ = ["fused_available", "viterbi_window_fused", "window_plan_cache_clear"]

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

#: Whether the batch is captured as a CUDA graph: unset lets the rules in
#: ``_plan_for_call`` decide, 0 forces the eager loop, 1 captures as soon as a
#: shape is asked for.  Principle 10 asks for graphs where they apply, and here
#: they do: the profiler puts a step at 4.8-4.9 us of kernel while the eager
#: loop takes 9.9 us of wall for it, and replay takes 6.3 us.  Whole-tensor,
#: that is 2x at L=14 and L=16 (2.74 -> 1.34 s, 10.88 -> 5.30 s).
#:
#: An earlier reading here said graphs did not help.  It was wrong and the
#: mistake is worth keeping: it compared a ``cudaEvent`` span against wall,
#: and an event span measures the *stream*, launch gaps included, so it read
#: 19.1 ms of "pure board time" for 21.7 ms of wall and hid exactly the gap
#: the graph closes.  Only the per-kernel profiler number exposes it.
#:
#: Read per CALL, not once at import.  It used to be a module constant, which
#: made it unsweepable from inside one process -- a matched pair could not put
#: the two spellings on one tensor seconds apart, which is the only way to
#: measure either of them on a shared box.  ``encode``'s own graph knob was
#: already per call.
_GRAPH_ENV = "TESSERA_WINDOW_GRAPH"
#: Captures are serialised per process; see ``_WindowPlan.capture``.
_CAPTURE_LOCK = threading.Lock()
#: How many batches one call must carry before capturing inside that call pays
#: for itself.  Capture costs about 17 us a node and replay saves about 3.6 us
#: a node, so a call that runs the loop five times over cannot get its capture
#: back and a call that runs it six times can.  This is the break-even the
#: module has always used; what changed in issue #94 is that falling below it
#: no longer means "run eager forever" (see ``_WINDOW_GRAPH_MIN_CALLS``).
_GRAPH_MIN_BATCHES = 6
#: How many times one shape must be asked for before a PERSISTENT plan is
#: built for it and captured.  A capture is paid once per shape and saves on
#: every call after it, so the only question is whether a shape recurs at all;
#: one is the case that must not pay, since a shape seen once would buy a
#: capture it never replays.  This is the half issue #94 was about: LDLQ hands
#: this function ``ldl_block`` columns at a time, which is ONE batch, so
#: ``_GRAPH_MIN_BATCHES`` refused the capture on every one of the hundreds of
#: identical calls a pass makes -- while the same tensor encoded in one call
#: captured and replayed 96 times.  The eager and captured spellings were
#: split by whether the caller was compensating, which is not a property the
#: launch stream has any business reading.
_WINDOW_GRAPH_MIN_CALLS = 2
#: How many shapes hold buffers at once, per thread.  Plans are only kept for
#: calls below ``_GRAPH_MIN_BATCHES``, which bounds what one holds: its front
#: and traceback cover fewer than six batches of columns.  A full-width call
#: keeps today's behaviour -- per-call buffers, captured and dropped inside the
#: call -- because persisting one would pin the whole tensor's traceback (at
#: L=14, R=4 over 3072 columns, 545 MiB) to save the ~17 ms its own capture
#: costs, which is 3% of that call for half a gigabyte.
_WINDOW_PLAN_CACHE = 4
#: Plans are **per thread**, and that is a correctness requirement rather than
#: a tuning choice: a plan owns the fronts and the traceback its Viterbi
#: writes, so two threads sharing one would overwrite each other's states, and
#: PrismaBuild's workers encode units concurrently in one process.  The kernel
#: cache above is shared because it is read-only; these are not.
_WINDOW_LOCAL = threading.local()


def _resolve_graph():
    """``True`` to capture, ``False`` to stay eager, ``None`` to let the rules decide."""
    raw = os.environ.get(_GRAPH_ENV, "")
    if raw == "":
        return None
    if raw in ("0", "1"):
        return raw == "1"
    raise ValueError(
        f"{_GRAPH_ENV}={raw!r} is not 0 (eager), 1 (capture) or unset (auto)"
    )


def _window_maps():
    plans = getattr(_WINDOW_LOCAL, "plans", None)
    if plans is None:
        plans = _WINDOW_LOCAL.plans = collections.OrderedDict()
        _WINDOW_LOCAL.seen = collections.OrderedDict()
    return plans, _WINDOW_LOCAL.seen


def window_plan_cache_clear() -> None:
    """Drop this thread's cached plans and their graphs.

    For tests, and for a caller that wants the residency back.  It clears the
    calling thread's maps only, which is the same scope they are built in.
    """
    plans, seen = _window_maps()
    plans.clear()
    seen.clear()


def _l2_budget(device) -> int:
    """The byte budget the column width is sized to."""
    return _L2_BUDGET if _L2_BUDGET is not None else _device_l2_budget(device)


@functools.lru_cache(maxsize=8)
def _device_l2_budget(device) -> int:
    """A sixth of this board's L2, memoised.

    ``torch.cuda.get_device_properties`` is a real query and this module was
    making one per call -- which under LDLQ is once per column block per refit
    pass, for a number that is a property of the board and nothing else.  The
    override above is deliberately left OUT of the memo so a test that rebinds
    it is not answered from a cache keyed on the device alone.
    """
    return getattr(torch.cuda.get_device_properties(device),
                   "L2_cache_size", 24 << 20) // 6


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
              BL: tl.constexpr, BC: tl.constexpr, SCAN_UNROLL: tl.constexpr):
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
        # Two spellings of ONE scan.  The body is the same six lines either
        # way -- load, compare strictly, two selects -- so the answer is the
        # same answer; what differs is how many of the ``FAN - 1`` loads the
        # compiler is allowed to have in flight, which is the whole of the
        # R = 8 cliff (see ``_scan_unroll``).
        best = tl.load(front_in + base + li[None, :], mask=m2, other=float("inf"))
        pred = tl.zeros([BC, BL], dtype=tl.int32)
        if SCAN_UNROLL <= 0:
            for f in tl.static_range(1, FAN):
                v = tl.load(front_in + base + (f * low + li)[None, :], mask=m2,
                            other=float("inf"))
                take = v < best
                best = tl.where(take, v, best)
                pred = tl.where(take, f, pred)
        else:
            for f in tl.range(1, FAN, loop_unroll_factor=SCAN_UNROLL):
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


#: How many iterations of the class scan the IR unroller may fuse.  ``<= 0``
#: is ``tl.static_range`` -- the flat unroll every rate ran until 2026-09-02,
#: kept reachable so the measurement below can be reproduced and so a box
#: whose scheduler behaves differently can have it back.  A measurement knob,
#: never a correctness one: both spellings run the same six-line body in the
#: same order and return the same states and the same ``sse`` float, which
#: ``test_window_viterbi_fast`` pins at every rate.
_SCAN_UNROLL_ENV = "TESSERA_WINDOW_SCAN_UNROLL"


def _scan_unroll(fan: int, bl: int, bc: int, warps: int) -> int:
    """How the class-minimum scan is spelled, and the measurement that fixed it.

    The class minimum is ``FAN - 1`` INDEPENDENT masked loads feeding a
    dependent select chain.  Flat-unrolled, the scheduler hoists the loads to
    cover their latency and holds one live value per hoisted load per element
    a thread owns; the tile ``[BC, BL]`` shrinks as ``2048 / FAN`` while the
    iteration count grows as ``FAN``, so the live set grows like ``FAN``
    however the tile is sized.  Reading ``n_regs``/``n_spills`` off the
    launched kernel on GB10 at L = 14
    (``experiments/window_viterbi_r8_diagnosis.py``):

        R          4     5     6     7      8
        n_regs    40    64    96   128     40
        n_spills   0     0     0     0    690    <- ptxas gives up and spills

    690 bytes of local memory inside a serial chain is issue #11's cliff:
    38.2 s against the reference's 4.2 s, where R = 7 took 0.93 s.  It is a
    step function, not a trend -- R = 7's live set fits in 128 registers and
    R = 8's does not fit in 255.  So the fix is not a wider tile (issue #11's
    candidate, which puts MORE elements under each hoisted load): it is to
    stop asking for the flat unroll.

    A runtime ``tl.range`` holds one load live whatever ``FAN`` is, and the
    sweep says it is not merely a rescue at R = 8 but faster at every rate --
    the flat unroll was costing 2-3x wherever the live set was large enough to
    crowd the file and not yet large enough to spill
    (``experiments/window_viterbi_scan_unroll_sweep.py``, 1024x1024 at L = 14,
    seconds, states and ``sse`` identical to the reference in every cell):

        R              4      6      7       8
        flat unroll  0.173  0.498  0.927  38.190
        loop x32     0.161  0.193  0.290   0.490
                     1.07x  2.58x  3.20x   77.9x

    which is why the policy is uniform rather than a threshold: there is no
    rate at which the flat unroll wins, so there is no rate that needs it.
    It holds off L = 14 too -- L = 12 (R = 4/6/8: 2.7x / 2.2x / 74x) and
    L = 16 (R = 4/8: 1.04x / 81x), and at arity 2 (R = 7/8: 1.54x / 2.4x).

    THE FACTOR IS MEASURED, NOT DERIVED.  1, 2, 4, 8, 16, 32, 64 and 128 were
    timed at R = 8 (1.49, 0.84, 0.55, 0.57, 0.50, 0.49, 0.48, 0.48 s) and
    8/16/32 across R = 4..8; 32 is at or within 3% of the best in every cell
    and spills nowhere, at L = 12, 14 and 16.  The tile arguments are taken
    because a future ``_tile`` may make the right factor depend on them; today
    it does not, and saying so is cheaper than pretending it does.
    """
    return 32


def _resolve_scan_unroll(fan: int, bl: int, bc: int, warps: int) -> int:
    raw = os.environ.get(_SCAN_UNROLL_ENV)
    if raw is None or raw == "":
        return _scan_unroll(fan, bl, bc, warps)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{_SCAN_UNROLL_ENV}={raw!r} is not an integer unroll factor") from exc


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


def _layout(device, size: int, cols: int, chunk: int):
    """``(nmax, width, descs)`` for one shape: how the columns are batched.

    ``width`` is set so both fronts stay in L2 across the step loop, which is
    what turns the front traffic from DRAM into cache; ``descs`` is one triple
    per batch -- its first column in the tensor, its first column inside the
    chunk, and its width -- which the kernels read from the device so that one
    captured graph replays for every batch.
    """
    budget = _l2_budget(device)
    nmax = min(chunk, cols)
    width = max(1, min(budget // (2 * size * 4), nmax))
    descs = []
    for start in range(0, cols, chunk):
        n = min(chunk, cols - start)
        for c0 in range(0, n, width):
            descs.append((start + c0, c0, min(width, n - c0)))
    return nmax, width, descs


class _WindowPlan:
    """One window Viterbi's tensors, and the graph that replays its batch.

    ``one_batch`` is **the definition** of the fused window trellis -- the
    per-call path and the persistent path both execute this method, so they
    cannot drift.  What differs between them is only where the tensors come
    from: a per-call plan points straight at the caller's targets and throws
    its scratch away, and a persistent plan owns both so the addresses are
    stable enough to capture once and replay for every later call.

    Why a persistent plan at all (issue #94): every batch is the same
    ``2 + steps`` launches at the same pointers, with only the three-integer
    descriptor moving, and the descriptor lives on the device so one capture
    covers every batch.  A call wide enough to run six batches therefore
    captures inside itself and always did.  LDLQ makes the calls narrow
    instead of few -- ``ldl_block`` columns is ONE batch -- so the same tensor,
    same table, same rate ran captured when it was encoded in one call and
    eager when it was encoded in ``cols / ldl_block`` of them, hundreds of
    times a pass, plus a fresh allocate and a fresh board query each time.
    """

    __slots__ = ("device", "rows", "cols", "arity", "size", "rate", "chunk",
                 "steps", "fan", "low", "back_u8", "nmax", "width",
                 "bl", "bc", "warps", "scan_unroll", "grid", "cgrid", "bs",
                 "cbc", "owns_input", "tuples", "wrows", "table", "front_all",
                 "back", "cur", "nxt", "ctl", "desc", "batches", "graph")

    def __init__(self, *, device, rows, cols, arity, size, rate, chunk,
                 has_weights, owns_input):
        import triton

        self.device, self.rows, self.cols = device, rows, cols
        self.arity, self.size, self.rate, self.chunk = arity, size, rate, chunk
        self.owns_input = owns_input
        self.steps = steps = rows // arity
        self.fan = fan = 1 << rate
        self.low = low = size >> rate
        self.back_u8 = back_u8 = fan <= 256

        nmax, width, descs = _layout(device, size, cols, chunk)
        self.nmax, self.width = nmax, width

        self.front_all = torch.empty(nmax, size, dtype=torch.float32, device=device)
        self.back = torch.empty(nmax, steps, low,
                                dtype=torch.uint8 if back_u8 else torch.int32,
                                device=device)
        self.cur = torch.empty(width, size, dtype=torch.float32, device=device)
        self.nxt = torch.empty(width, size, dtype=torch.float32, device=device)

        self.bl, self.bc, self.warps = _tile(fan, low, width)
        self.scan_unroll = _resolve_scan_unroll(fan, self.bl, self.bc, self.warps)
        self.grid = (triton.cdiv(low, self.bl), triton.cdiv(width, self.bc))
        self.bs = bs = min(size, 1024)
        self.cbc = cbc = max(1, 2048 // bs)
        self.cgrid = (triton.cdiv(size, bs), triton.cdiv(width, cbc))

        # A persistent plan ships the descriptors to the device once instead
        # of once per call: ``torch.tensor(descs, device=...)`` is a host
        # allocation, a walk over a Python list and a copy over the bus, and
        # under LDLQ it was one of those per column block per refit pass.
        self.batches = len(descs)
        self.ctl = torch.empty(3, dtype=torch.int32, device=device)
        self.desc = torch.tensor(descs, dtype=torch.int32, device=device)

        # A captured graph reads its inputs from fixed addresses, so a
        # persistent plan owns them and ``bind`` copies in; a per-call plan
        # points at the caller's tensors and copies nothing beyond the
        # ``.float().contiguous()`` the kernels need anyway.
        if owns_input:
            self.tuples = torch.empty(steps, arity, cols, dtype=torch.float32,
                                      device=device)
            self.wrows = (torch.empty(steps, arity, cols, dtype=torch.float32,
                                      device=device) if has_weights else None)
            self.table = torch.empty(size, arity, dtype=torch.float32, device=device)
        else:
            self.tuples = self.wrows = self.table = None
        self.graph = None

    def bind(self, targets, vectors, weights):
        """Point the plan at this call's inputs, copying only if it owns them.

        The copy is exact: ``copy_`` into an fp32 buffer rounds a wider source
        the way ``.float()`` does, and an fp32 source is moved unchanged.  So
        both spellings hand the kernels the identical bits, which is what lets
        ``tests/test_window_graph`` demand byte equality rather than a
        tolerance.
        """
        steps, arity, cols = self.steps, self.arity, self.cols
        if self.owns_input:
            self.tuples.copy_(targets.reshape(steps, arity, cols))
            if self.wrows is not None:
                self.wrows.copy_(weights.reshape(steps, arity, cols))
            self.table.copy_(vectors)
        else:
            self.tuples = targets.float().reshape(steps, arity, cols).contiguous()
            self.wrows = (None if weights is None else
                          weights.float().reshape(steps, arity, cols).contiguous())
            self.table = vectors.float().to(self.device).contiguous()

    def one_batch(self):
        """The exact batch: init the front, run the step loop, keep the last front."""
        step_kernel, _, init_kernel, copy_kernel = _kernels()
        init_kernel[self.cgrid](self.cur, self.ctl, SIZE=self.size, BS=self.bs,
                                BC=self.cbc, num_warps=4)
        a, b = self.cur, self.nxt
        for step in range(self.steps):
            step_kernel[self.grid](
                a, b, self.back, self.tuples,
                self.tuples if self.wrows is None else self.wrows, self.table,
                self.ctl, self.cols, self.steps, step, self.low,
                ARITY=self.arity, FAN=self.fan, SIZE=self.size,
                HAS_W=self.wrows is not None, BACK_U8=self.back_u8,
                BL=self.bl, BC=self.bc, SCAN_UNROLL=self.scan_unroll,
                num_warps=self.warps, enable_fp_fusion=False,
            )
            a, b = b, a
        copy_kernel[self.cgrid](a, self.front_all, self.ctl, SIZE=self.size,
                                BS=self.bs, BC=self.cbc, num_warps=4)

    def capture(self):
        """Warm on a side stream, then capture ``one_batch`` as one graph.

        One capture at a time, per process.  A capture is a few milliseconds
        and the replays are the long pole, so serialising the captures costs
        nothing; running them concurrently costs the graph.  Two things break
        a concurrent capture: under CUDA's default ``global`` error mode any
        CUDA call from *another* thread while a capture is open faults it (an
        allocator call from a second worker encoding its own unit is enough),
        and even ``thread_local`` mode leaves the two captures' begin/end
        bookkeeping -- ``torch.cuda.graph`` synchronises the device and empties
        the cache on entry -- to invalidate each other.  PrismaBuild's workers
        encode units concurrently in one process, so the capture takes the lock
        and stays thread-local; replays, eager launches, allocations and
        stream-level syncs (``.item()``, ``.cpu()``) on the other threads
        proceed.  The lock is shared with ``encode``'s coset-trellis capture so
        a window capture and a TCQ capture cannot overlap either.  The one call
        no mode permits while a capture is open anywhere is a device-wide
        ``torch.cuda.synchronize()``: nothing in this package makes one, and a
        threaded caller must not either.
        """
        with _CAPTURE_LOCK:
            self.ctl.copy_(self.desc[0])
            warm = torch.cuda.Stream()
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                self.one_batch()             # compiles and warms; capture runs nothing
            torch.cuda.current_stream().wait_stream(warm)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                self.one_batch()
        self.graph = graph

    def replay(self):
        if self.graph is None:
            self.one_batch()
        else:
            self.graph.replay()


def _plan_for_call(*, device, rows, cols, arity, size, rate, chunk, has_weights):
    """The plan this call runs on, and whether it should be captured.

    Three rules, and the first two are the ones the module has always had:

    * ``TESSERA_WINDOW_GRAPH=0`` is the eager loop on a per-call plan -- the
      spelling that owns nothing, so it is also the control an A/B measures
      against.
    * a call carrying ``_GRAPH_MIN_BATCHES`` batches or more captures inside
      itself, on a per-call plan, exactly as before.  It amortises its own
      capture and persisting its buffers would pin the whole tensor's
      traceback to save a capture worth 3% of the call.
    * anything narrower is the issue #94 case.  It gets a PERSISTENT plan the
      second time its shape is asked for, so the capture is paid once and
      replayed by every call after -- which is what turns LDLQ's hundreds of
      one-batch calls from hundreds of eager step loops into hundreds of
      replays.  The first call of a shape stays eager on a per-call plan, so
      a shape seen once never buys a capture it cannot use.
    """
    forced = _resolve_graph()
    plans, seen = _window_maps()
    key = (device, rows, cols, arity, size, rate, chunk, has_weights,
           _L2_BUDGET, os.environ.get(_SCAN_UNROLL_ENV, ""))
    plan = plans.get(key)
    if plan is not None:
        plans.move_to_end(key)
        return plan, True

    count = seen.get(key, 0) + 1
    seen[key] = count
    seen.move_to_end(key)
    while len(seen) > 64:
        seen.popitem(last=False)

    def fresh(owns):
        return _WindowPlan(device=device, rows=rows, cols=cols, arity=arity,
                           size=size, rate=rate, chunk=chunk,
                           has_weights=has_weights, owns_input=owns)

    if forced is False:
        return fresh(False), False
    batches = len(_layout(device, size, cols, chunk)[2])
    if batches >= _GRAPH_MIN_BATCHES:
        return fresh(False), True                         # captures inside this call
    if forced is None and count < _WINDOW_GRAPH_MIN_CALLS:
        return fresh(False), False
    plan = fresh(True)
    plans[key] = plan
    while len(plans) > _WINDOW_PLAN_CACHE:
        plans.popitem(last=False)
    return plan, True


def viterbi_window_fused(targets, vectors, window_bits: int, rate: int,
                         weights=None, chunk: int = 512):
    """The fused CUDA path behind ``encode.viterbi_window``.

    Same arguments, same returns, identical states and identical sse.
    Callers go through ``viterbi_window``; this entry point exists so the
    tests can pin the implementation.

    The machine is picked by ``_plan_for_call`` and is never the answer: a
    per-call plan and a persistent one run the same ``one_batch`` over the same
    values in the same order, and the epilogue below -- the final ``min`` over
    states, ``sse += float(final.sum())`` and the traceback -- runs on the host
    stream either way, so the ``sse`` float is summed in the reference's order
    whichever plan produced the front.
    """
    import triton

    _, traceback_kernel, _, _ = _kernels()
    device = targets.device
    rows, cols = targets.shape
    size, arity = vectors.shape
    steps = rows // arity
    low = size >> rate

    plan, wants_graph = _plan_for_call(
        device=device, rows=rows, cols=cols, arity=arity, size=size, rate=rate,
        chunk=chunk, has_weights=weights is not None)
    plan.bind(targets, vectors, weights)
    if wants_graph and plan.graph is None:
        plan.capture()

    states = torch.empty(steps, cols, dtype=torch.long, device=device)
    sse = 0.0
    b = 0
    for start in range(0, cols, chunk):
        n = min(chunk, cols - start)
        for _ in range(0, n, plan.width):
            plan.ctl.copy_(plan.desc[b])
            b += 1
            plan.replay()

        cost = plan.front_all[:n].t().contiguous()            # [size, n]
        final, state = cost.min(dim=0)                        # [n]
        sse += float(final.sum())
        tb = 128
        traceback_kernel[(triton.cdiv(n, tb),)](
            plan.back, state.to(torch.int32), states, n, cols, steps, low, start,
            RATE=rate, SHIFT=window_bits - rate, BC=tb,
            num_warps=4, enable_fp_fusion=False,
        )
    return states, sse
