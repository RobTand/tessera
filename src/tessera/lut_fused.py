"""The LUT fit's swap passes, fused, deciding exactly what the reference decides.

``encode._fit_lut`` chooses the sixteen distinct E4M3 scales of a LUT plane: a
candidate bracket, a greedy elimination, then swap passes that try every table
entry against every unused grid value at full cost and take a trial when it
lowers the running cost by more than one float32 ulp.
``encode._lut_swap_passes_reference`` is those passes, and it is the
definition.  Written as torch ops it is a clone, an entry write, six kernels
and a host sync per trial, so one fit is thousands of launches and hundreds of
syncs (tessera#486: once the trellis was fused, 47 % of an E2M1_K2 encode's
host samples waited on that sync).

This module runs the same passes with one host sync a pass:

  * ``_suffix`` -- once a pass, every target's distance to its nearest entry
    after each position, ``S[i] = min_{j > i} |s - t_j|``.
  * ``_prefix`` -- once a position, the nearest entry before it (the entries
    this pass has settled) and the distance with entry ``i`` taken out,
    ``m = min(min_{j < i} |s - t_j|, S[i])``.
  * ``_blocks``, and ``_stage`` when torch splits the sum across blocks --
    every trial's cost at the position, in torch's own summation order.
  * ``_accept`` -- the reference's accept scan over those costs in trial
    order, the table and byte update, and the next position's unused list.

**Identity is the contract.**  The table and the bytes are the reference's
because every decision is:

  * A trial at position ``i`` differs from the running table at entry ``i``
    only: the pass has settled the entries before ``i`` and not yet touched
    those after it, and an accepted trial changes entry ``i`` only.  So a
    trial's gap is ``min(m, |s - v|)``.  The value of a minimum does not
    depend on its order, so that is the float ``amin`` returns.
  * The term is ``(w * g) * g``, as torch evaluates ``weights * gap * gap``:
    two correctly rounded multiplies, each the inline-asm ``mul.f32``
    (``_mul``) under ``enable_fp_fusion=False``, because on Blackwell a packed
    multiply-add pair is contracted even with fusion off.
  * The sum is torch's CUDA float32 ``sum`` of a contiguous ``[n]``, in torch's
    order.  ``sum_plan`` is ``setReduceConfig`` from
    ``ATen/native/cuda/Reduce.cuh`` for that reduction on a non-ROCm build,
    fed the device properties it reads.  Every thread keeps four accumulators,
    one per element of a four-element load, fed its loads in order; the
    ``n % 4`` tail goes to accumulator 0 of the first lanes of block 0; the
    accumulators combine left to right; a block's lanes reduce as a halving
    tree; and when the input is split across blocks, each lane sums its staged
    block values from zero before the same tree reduces the lanes.  Torch sums
    fewer than 128 elements without vectorising, in another order, so those
    fits keep the reference.  Torch also reduces the elements ahead of the
    input's first 16-byte boundary apart; ``_lut_cost`` sums a product fresh
    from the native caching allocator, whose blocks start on 512-byte
    boundaries, so there are none, and another allocator keeps the reference.
  * The accept test ``cost < base * (1 - eps)`` runs in float64 on the device.
    ``base`` is a float32 value and ``1 - 2^-23`` has 23 significant bits, so
    the product is exact and the comparison is the host's.
  * The trials at a position run over the bracket's bytes that are not in the
    table, ascending, as they stand when the position starts: the list
    ``all_bytes[~isin(all_bytes, candidate_bytes)].tolist()`` yields there.

A tripwire holds the replica to torch on every fit: the first trial's cost,
and each improving pass's final running cost against torch's ``_lut_cost`` of
the same table (the next pass's base).  Any difference, or a non-finite target
or weight, which the strict comparisons would not propagate the way ``amin``
does, discards the fused passes, and the caller runs the reference from the
arguments, which this does not modify.  Every launch masks its own axes, so
the tile knob moves launches and never floats.
"""
from __future__ import annotations

import functools
import os
import warnings
from dataclasses import dataclass

import torch

from .window_viterbi import fused_available

__all__ = ["STATS", "SumPlan", "fused_available", "lut_swap_refusal", "position_costs",
           "sum_plan", "swap_passes_fused"]

#: The torch releases whose CUDA float32 ``sum`` order this module reproduces:
#: read from torch 2.11.0's shipped ``Reduce.cuh`` and held to ``torch.sum``
#: bitwise (PB ``197cbc785234``: 906 of 906 cases over 151 sizes from 128 to
#: 33,333,331 elements, 593 of them order-sensitive).  Another release may
#: reorder the reduction, so it keeps the reference until it is checked.
_VERIFIED_TORCH = ("2.11.",)
#: ``setReduceConfig`` vectorises a reduction of at least this many elements.
_MIN_VECTORISED = 128
#: The largest fit admitted: the longest sum the replica is held to torch on
#: (PB ``197cbc785234``).  Longer sums reach layouts it is not checked on -- a
#: lane summing more than one staged row, and past 2^31 index bits torch's
#: split into sub-reductions -- and LUT fits are far shorter.
_MAX_TARGETS = 33_333_331
#: ``T`` or ``T,W``: trials per ``_blocks`` program and its warps, both powers
#: of two.  Unset takes ``_TILE_DEFAULT``.  A measurement knob, never a
#: correctness one: every program masks its trials and lanes, so every tile
#: writes the same floats.  Read per fit.
_TILE_ENV = "TESSERA_LUT_FUSED_TILE"
_TILE_DEFAULT = (16, 4)
#: Targets per ``_suffix`` and ``_prefix`` program.
_ELEMENTS = 4096

#: Per-process counters: fits the fused passes answered, fits sent back to the
#: reference because the replica disagreed with torch, and fits sent back for a
#: non-finite target or weight.
STATS = {"fused": 0, "tripped": 0, "nonfinite": 0}


def _last_pow2(v: int) -> int:
    return 1 << (v.bit_length() - 1)


def _div_up(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class SumPlan:
    """Where torch's CUDA reduction puts each element of a full float32 ``sum``."""

    n: int
    lanes: int      # block width: the threads of one block, a power of two
    blocks: int     # ctas_per_output: the blocks the input is split across (1: none)
    step: int       # the vectors between one thread's successive loads
    vectors: int    # the most loads any thread makes

    @property
    def tail(self) -> int:
        """Elements past the last whole four-element load."""
        return self.n % 4

    @property
    def rows(self) -> int:
        """The most staged block values one lane sums when ``blocks > 1``."""
        return _div_up(self.blocks, self.lanes)


@functools.lru_cache(maxsize=512)
def sum_plan(n: int, multi_processor_count: int, max_threads_per_multi_processor: int,
             warp_size: int = 32) -> SumPlan:
    """``setReduceConfig`` for a contiguous 1-D float32 full reduction of ``n`` elements.

    This is the reduction ``_lut_cost``'s ``sum`` launches: one output, the
    reduced dimension the only one, a one-element stride, so ``dim0 = n``,
    ``dim1 = 1`` and the input is vectorised four elements a load.  It follows
    torch 2.11's ``Reduce.cuh`` line for line where ``USE_ROCM`` is not defined
    (``min_values_per_thread`` 16, no 64-block floor).  The device properties
    are the ones the reduction reads.
    """
    if n < _MIN_VECTORISED:
        raise ValueError(f"torch does not vectorise a reduction of {n} elements")
    max_num_threads = 512                        # mnt_wrapper<float>::MAX_NUM_THREADS
    dim0, dim1 = n // 4, 1                       # vectorize_input: dim0 /= input_vec_size
    # set_block_dimension
    dim0_pow2 = _last_pow2(dim0) if dim0 < max_num_threads else max_num_threads
    dim1_pow2 = _last_pow2(dim1) if dim1 < max_num_threads else max_num_threads
    block_width = min(dim0_pow2, warp_size)
    block_height = min(dim1_pow2, max_num_threads // block_width)
    block_width = min(dim0_pow2, max_num_threads // block_height)
    num_threads = block_width * block_height
    if block_height != 1:                        # one output: one warp per block
        raise AssertionError(f"block_height {block_height} for a single-output reduction")
    step_input = block_width                     # input_mult[0] = split_input(block_width)
    step_output = 1
    split_across_warps = _div_up(n, step_input) >= min(block_height * 16, 256)
    if split_across_warps:                       # input_mult[1] = split_input(block_height)
        step_input *= block_height
    else:                                        # output_mult[1] = split_output(block_height)
        step_output *= block_height
    blocks_per_sm = max_threads_per_multi_processor // num_threads
    target_grid_size = multi_processor_count * blocks_per_sm
    grid = _div_up(1, step_output)               # grid().x for one output
    values_per_thread = _div_up(n, step_input)
    blocks = 1
    if split_across_warps and values_per_thread >= 256 and grid <= target_grid_size:
        blocks = max(min(_div_up(target_grid_size, grid), _div_up(values_per_thread, 16)),
                     _div_up(values_per_thread, 256))
        if blocks > 1:                           # input_mult[2] = split_input(ctas)
            step_input *= blocks
        else:
            blocks = 1
    return SumPlan(n=n, lanes=block_width, blocks=blocks, step=step_input,
                   vectors=_div_up(dim0, step_input))


def _plan(targets: torch.Tensor) -> SumPlan:
    props = torch.cuda.get_device_properties(targets.device)
    return sum_plan(targets.numel(), int(props.multi_processor_count),
                    int(props.max_threads_per_multi_processor), int(props.warp_size))


def lut_swap_refusal(targets: torch.Tensor, weights: torch.Tensor, table: torch.Tensor,
                     grid_values: torch.Tensor) -> "str | None":
    """Why the fused swap passes cannot take this fit, or ``None`` when they can.

    ``targets`` and ``weights`` are the live ``[n]`` the passes score, and
    ``table`` and ``grid_values`` the running table and the grid it is drawn
    from, as ``_fit_lut`` holds them.
    """
    if not targets.is_cuda:
        return f"targets are on {targets.device}"
    if not fused_available():
        return "triton is absent, or this torch is a HIP build the NVPTX kernel cannot target"
    if not torch.__version__.startswith(_VERIFIED_TORCH):
        return (f"torch {torch.__version__}: the replicated sum order is checked on "
                f"{', '.join(v + 'x' for v in _VERIFIED_TORCH)} only")
    backend = torch.cuda.get_allocator_backend()
    if backend != "native":
        return (f"the {backend} CUDA allocator: torch reduces the elements ahead of a "
                "product's first 16-byte boundary apart, and only the native caching "
                "allocator is known to leave none")
    for name, tensor in (("weights", weights), ("table", table), ("grid", grid_values)):
        if tensor.device != targets.device:
            return f"{name} are on {tensor.device}, targets on {targets.device}"
    for name, tensor in (("targets", targets), ("weights", weights), ("table", table),
                         ("grid", grid_values)):
        if tensor.dtype != torch.float32:
            return f"{name} are {tensor.dtype}; the replicated cost is float32 end to end"
    if targets.dim() != 1 or weights.shape != targets.shape:
        return (f"targets {tuple(targets.shape)} and weights {tuple(weights.shape)} "
                "are not one [n]")
    n = targets.numel()
    if n < _MIN_VECTORISED:
        return (f"{n} targets: torch sums fewer than {_MIN_VECTORISED} elements "
                "without vectorising, in an order this path does not replicate")
    if n > _MAX_TARGETS:
        return (f"{n} targets: longer than any sum the replica is checked on "
                f"({_MAX_TARGETS} elements)")
    plan = _plan(targets)
    if plan.rows > 1:
        return (f"{n} targets: on this device a lane sums {plan.rows} staged rows "
                "of blocks, a layout the replica is not checked on")
    return None


def _resolve_tile() -> "tuple[int, int]":
    raw = os.environ.get(_TILE_ENV, "")
    if raw == "":
        return _TILE_DEFAULT
    try:
        values = [int(part) for part in raw.split(",")]
    except ValueError as exc:
        raise ValueError(f"{_TILE_ENV}={raw!r} is not T or T,W") from exc
    if len(values) not in (1, 2) or any(v < 1 or v & (v - 1) for v in values):
        raise ValueError(f"{_TILE_ENV}={raw!r}: T and W are powers of two")
    return values[0], values[1] if len(values) == 2 else _TILE_DEFAULT[1]


@dataclass(frozen=True)
class _Kernels:
    suffix: object
    prefix: object
    blocks: object
    stage: object
    accept: object


def _build() -> _Kernels:
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

    @triton.jit
    def _halve(h, TRIALS: tl.constexpr, HALF: tl.constexpr):
        """One level of the lane tree: ``h[x] + h[x + HALF]`` for ``x < HALF``."""
        lo, hi = tl.split(tl.permute(tl.reshape(h, (TRIALS, 2, HALF)), (0, 2, 1)))
        return lo + hi

    @triton.jit
    def _tree(h, TRIALS, LANES):
        """``block_x_reduce`` over each trial row: ``h[x] += h[x + off]``, ``off = LANES/2 .. 1``.

        Spelled level by level with literal widths: a shape must be a
        compile-time int, and Triton does not fold one computed in a loop.
        ``LANES`` is a power of two from 32 to 512 (``max_num_threads``).
        No ``tl.constexpr`` annotations: a nested call takes constexpr-ness
        from its arguments, and Triton resolves an annotation in the helper's
        own scope, where a body that never names ``tl`` has no ``tl``.
        """
        if LANES >= 512:
            h = _halve(h, TRIALS, 256)
        if LANES >= 256:
            h = _halve(h, TRIALS, 128)
        if LANES >= 128:
            h = _halve(h, TRIALS, 64)
        if LANES >= 64:
            h = _halve(h, TRIALS, 32)
        if LANES >= 32:
            h = _halve(h, TRIALS, 16)
        if LANES >= 16:
            h = _halve(h, TRIALS, 8)
        if LANES >= 8:
            h = _halve(h, TRIALS, 4)
        if LANES >= 4:
            h = _halve(h, TRIALS, 2)
        if LANES >= 2:
            h = _halve(h, TRIALS, 1)
        return h

    @triton.jit
    def _term(sp, wp, mp, e, ok, v, TRIALS: tl.constexpr, LANES: tl.constexpr):
        """``weights * gap * gap`` at elements ``e``, for every trial value ``v``."""
        s = tl.load(sp + e, mask=ok, other=0.0)[None, :]
        w = tl.broadcast_to(tl.load(wp + e, mask=ok, other=0.0)[None, :], (TRIALS, LANES))
        m = tl.broadcast_to(tl.load(mp + e, mask=ok, other=0.0)[None, :], (TRIALS, LANES))
        d = tl.abs(s - v)
        g = tl.where(d < m, d, m)
        return _mul(_mul(w, g), g)

    @triton.jit(do_not_specialize=["n", "entries"])
    def _suffix(sp, tab, suf, n, entries, BLOCK: tl.constexpr):
        """``suf[i * n + e] = min_{j > i} |s[e] - tab[j]|``, ``inf`` past the last entry."""
        e = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        ok = e < n
        s = tl.load(sp + e, mask=ok, other=0.0)
        cur = tl.full([BLOCK], float("inf"), tl.float32)
        for jj in range(0, entries):
            j = entries - 1 - jj
            tl.store(suf + j.to(tl.int64) * n + e, cur, mask=ok)
            d = tl.abs(s - tl.load(tab + j))
            cur = tl.where(d < cur, d, cur)

    @triton.jit(do_not_specialize=["n", "i"])
    def _prefix(sp, tab, suf, pre, dist, n, i, BLOCK: tl.constexpr):
        """Fold entry ``i - 1`` into the prefix minimum; ``dist`` = entry ``i`` taken out."""
        e = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        ok = e < n
        s = tl.load(sp + e, mask=ok, other=0.0)
        p = tl.load(pre + e, mask=ok, other=float("inf"))
        if i > 0:
            d = tl.abs(s - tl.load(tab + i - 1))
            p = tl.where(d < p, d, p)
            tl.store(pre + e, p, mask=ok)
        r = tl.load(suf + i.to(tl.int64) * n + e, mask=ok, other=float("inf"))
        tl.store(dist + e, tl.where(r < p, r, p), mask=ok)

    @triton.jit(do_not_specialize=["q", "tail", "step", "vectors", "ntrial", "blocks"])
    def _blocks(sp, wp, mp, vals, out, q, tail, step, vectors, ntrial, blocks,
                LANES: tl.constexpr, TRIALS: tl.constexpr):
        """Block ``c`` of every trial's sum in a tile: the thread reduce, then the lane tree.

        Thread ``t = c * LANES + lane`` loads the four-element vectors ``t``,
        ``t + step``, ... below ``q``, one accumulator per element position;
        block 0's first ``tail`` lanes then add element ``4q + lane`` to
        accumulator 0; the accumulators combine left to right.  Writes
        ``out[u * blocks + c]``.
        """
        c = tl.program_id(0)
        u = tl.program_id(1) * TRIALS + tl.arange(0, TRIALS)
        uok = u < ntrial
        v = tl.load(vals + u, mask=uok, other=0.0)[:, None]
        lane = tl.arange(0, LANES)
        t = c.to(tl.int64) * LANES + lane
        a0 = tl.zeros([TRIALS, LANES], dtype=tl.float32)
        a1 = tl.zeros([TRIALS, LANES], dtype=tl.float32)
        a2 = tl.zeros([TRIALS, LANES], dtype=tl.float32)
        a3 = tl.zeros([TRIALS, LANES], dtype=tl.float32)
        for k in range(0, vectors):
            idx = t + k.to(tl.int64) * step
            ok = idx < q
            wk = ok[None, :]
            e = idx * 4
            a0 = tl.where(wk, a0 + _term(sp, wp, mp, e, ok, v, TRIALS, LANES), a0)
            a1 = tl.where(wk, a1 + _term(sp, wp, mp, e + 1, ok, v, TRIALS, LANES), a1)
            a2 = tl.where(wk, a2 + _term(sp, wp, mp, e + 2, ok, v, TRIALS, LANES), a2)
            a3 = tl.where(wk, a3 + _term(sp, wp, mp, e + 3, ok, v, TRIALS, LANES), a3)
        tm = (lane < tail) & (c == 0)
        et = q.to(tl.int64) * 4 + lane
        a0 = tl.where(tm[None, :], a0 + _term(sp, wp, mp, et, tm, v, TRIALS, LANES), a0)
        h = _tree(((a0 + a1) + a2) + a3, TRIALS, LANES)
        tl.store(out + u.to(tl.int64) * blocks + c, tl.reshape(h, (TRIALS,)), mask=uok)

    @triton.jit(do_not_specialize=["ntrial", "blocks", "rows"])
    def _stage(blk, out, ntrial, blocks, rows,
               LANES: tl.constexpr, TRIALS: tl.constexpr):
        """``global_reduce``: lane ``x`` sums staged blocks ``x, x + LANES, ...`` from zero."""
        u = tl.program_id(0) * TRIALS + tl.arange(0, TRIALS)
        uok = u < ntrial
        lane = tl.arange(0, LANES)
        row0 = u.to(tl.int64) * blocks
        h = tl.zeros([TRIALS, LANES], dtype=tl.float32)
        for r in range(0, rows):
            off = r * LANES + lane
            ok = off < blocks
            b = tl.load(blk + row0[:, None] + off[None, :], mask=uok[:, None] & ok[None, :],
                        other=0.0)
            h = tl.where(ok[None, :], h + b, h)
        h = _tree(h, TRIALS, LANES)
        tl.store(out + u, tl.reshape(h, (TRIALS,)), mask=uok)

    @triton.jit(do_not_specialize=["i", "ntrial", "bracket"])
    def _accept(cost, base, tab, cand, inuse, unused, vals, grid, flags, i, ntrial, bracket):
        """Position ``i``'s accept scan, then the unused list the next position iterates.

        ``cost[u] < base * (1 - 2^-23)`` in float64, trial order, the running
        cost moving on every accept -- the reference's loop.  An accept writes
        the trial's grid index and value at entry ``i``.  ``ntrial = 0`` only
        rebuilds the unused list.
        """
        b = tl.load(base)
        acc = tl.full([], -1, tl.int32)
        for u in range(0, ntrial):
            cu = tl.load(cost + u).to(tl.float64)
            take = cu < b * 0.99999988079071044921875
            b = tl.where(take, cu, b)
            acc = tl.where(take, u, acc)
        tl.store(base, b)
        took = acc >= 0
        # Both loads are masked.  The launch before the first pass runs before any launch
        # has written ``unused``, and an unmasked load of ``grid`` at an index left in that
        # memory reads wherever the index points, whether or not a trial was taken.
        new = tl.load(unused + tl.where(took, acc, 0), mask=took, other=0)
        old = tl.load(cand + i)
        tl.store(inuse + old, tl.zeros([], tl.int8), mask=took)
        tl.store(inuse + new, tl.full([], 1, tl.int8), mask=took)
        tl.store(cand + i, new, mask=took)
        tl.store(tab + i, tl.load(grid + new, mask=took, other=0.0), mask=took)
        tl.store(flags, tl.full([], 1, tl.int32), mask=took)
        k = tl.zeros([], tl.int32)
        for bb in range(0, bracket):
            free = tl.load(inuse + bb) == 0
            tl.store(unused + k, bb, mask=free)
            tl.store(vals + k, tl.load(grid + bb), mask=free)
            k = tl.where(free, k + 1, k)

    return _Kernels(suffix=_suffix, prefix=_prefix, blocks=_blocks, stage=_stage,
                    accept=_accept)


@functools.lru_cache(maxsize=1)
def _kernels() -> _Kernels:
    return _build()


def _launch_costs(kernels: _Kernels, plan: SumPlan, s, w, dist, vals, cost, blk,
                  trials: int, tile: "tuple[int, int]") -> None:
    import triton

    per, warps = tile
    lanes = plan.lanes
    kernels.blocks[(plan.blocks, triton.cdiv(trials, per))](
        s, w, dist, vals, blk, plan.n // 4, plan.tail, plan.step, plan.vectors, trials,
        plan.blocks, LANES=lanes, TRIALS=per,
        num_warps=warps, enable_fp_fusion=False,
    )
    if plan.blocks > 1:
        kernels.stage[(triton.cdiv(trials, per),)](
            blk, cost, trials, plan.blocks, plan.rows, LANES=lanes, TRIALS=per,
            num_warps=warps, enable_fp_fusion=False,
        )


def position_costs(targets: torch.Tensor, weights: torch.Tensor, table: torch.Tensor,
                   values: torch.Tensor, position: int) -> torch.Tensor:
    """Every trial's fused ``_lut_cost`` at one position.

    Trial ``u`` is ``table`` with entry ``position`` set to ``values[u]``.  An
    instrument: ``tests/test_lut_fused.py`` holds it to ``encode._lut_cost`` of
    each trial table, bit for bit, which is the claim the swap passes stand on.
    ``lut_swap_refusal(targets, weights, table, values)`` must admit the call.
    """
    import triton

    kernels = _kernels()
    s, w = targets.contiguous(), weights.contiguous()
    tab, vals = table.contiguous(), values.contiguous()
    n, entries, trials = s.numel(), tab.numel(), vals.numel()
    if not 0 <= position < entries:
        raise ValueError(f"position {position} is not an entry of a {entries}-entry table")
    device = s.device
    cost = torch.empty(trials, dtype=torch.float32, device=device)
    if trials == 0:
        return cost
    plan = _plan(s)
    suf = torch.empty(entries * n, dtype=torch.float32, device=device)
    pre = torch.full((n,), float("inf"), dtype=torch.float32, device=device)
    dist = torch.empty(n, dtype=torch.float32, device=device)
    grid = (triton.cdiv(n, _ELEMENTS),)
    kernels.suffix[grid](s, tab, suf, n, entries, BLOCK=_ELEMENTS, enable_fp_fusion=False)
    for i in range(position + 1):
        kernels.prefix[grid](s, tab, suf, pre, dist, n, i, BLOCK=_ELEMENTS,
                             enable_fp_fusion=False)
    blk = (torch.empty(trials * plan.blocks, dtype=torch.float32, device=device)
           if plan.blocks > 1 else cost)
    _launch_costs(kernels, plan, s, w, dist, vals, cost, blk, trials, _resolve_tile())
    return cost


def _fall_back(nonfinite: bool, plan: SumPlan) -> None:
    if nonfinite:
        STATS["nonfinite"] += 1
        return None
    STATS["tripped"] += 1
    if STATS["tripped"] == 1:
        warnings.warn(
            f"tessera.lut_fused: a fused LUT cost differed from torch's own sum ({plan}); "
            "the fit ran the reference swap passes instead.  The replicated reduction "
            "order does not hold on this torch or device, so every such fit pays the "
            "fused passes and then the reference.", RuntimeWarning, stacklevel=3)
    return None


def swap_passes_fused(targets: torch.Tensor, weights: torch.Tensor, table: torch.Tensor,
                      candidate_bytes: torch.Tensor, grid_values: torch.Tensor,
                      first: int, last: int, swaps: int):
    """The fused CUDA path behind ``encode._lut_swap_passes``.

    Same arguments and returns as ``encode._lut_swap_passes_reference`` --
    ``(table, candidate_bytes)`` as the passes leave them, with identical
    values -- or ``None`` when the tripwire fires, leaving the caller to run the
    reference from the same arguments, which this does not modify.
    ``lut_swap_refusal`` must have admitted the call.
    """
    import triton

    from .encode import E4M3_NORMAL_BYTES, _lut_cost

    entries = table.numel()
    bracket = last - first
    trials = bracket - entries
    if swaps <= 0 or trials <= 0:
        # No pass has a trial to take: each reads the running cost and stops.
        return table, candidate_bytes

    kernels = _kernels()
    device = targets.device
    s, w = targets.contiguous(), weights.contiguous()
    n = s.numel()
    plan = _plan(s)
    tile = _resolve_tile()
    first_byte = E4M3_NORMAL_BYTES[0]

    grid = grid_values[first:last].contiguous()
    tab = table.contiguous().clone()
    cand = (candidate_bytes - (first + first_byte)).to(torch.int32)
    inuse = torch.zeros(bracket, dtype=torch.int8, device=device)
    inuse.index_fill_(0, cand.to(torch.long), 1)
    unused = torch.empty(trials, dtype=torch.int32, device=device)
    vals = torch.empty(trials, dtype=torch.float32, device=device)
    suf = torch.empty(entries * n, dtype=torch.float32, device=device)
    pre = torch.empty(n, dtype=torch.float32, device=device)
    dist = torch.empty(n, dtype=torch.float32, device=device)
    cost = torch.empty(trials, dtype=torch.float32, device=device)
    blk = (torch.empty(trials * plan.blocks, dtype=torch.float32, device=device)
           if plan.blocks > 1 else cost)
    base = torch.empty((), dtype=torch.float64, device=device)
    # [a trial was taken this pass, the replica differed from torch, non-finite input]
    flags = torch.zeros(3, dtype=torch.int32, device=device)
    flags[2] = ~(torch.isfinite(s).all() & torch.isfinite(w).all())

    egrid = (triton.cdiv(n, _ELEMENTS),)
    kernels.accept[(1,)](cost, base, tab, cand, inuse, unused, vals, grid, flags,
                         0, 0, bracket, num_warps=1, enable_fp_fusion=False)
    improved = 0
    for p in range(swaps):
        running = _lut_cost(s, w, tab).to(torch.float64)
        if p:
            # The pass before improved, so its last running cost is a replica
            # sum of exactly this table, which torch has just summed.
            flags[1:2].add_((base != running).to(torch.int32))
        base.copy_(running)
        flags[0].zero_()
        kernels.suffix[egrid](s, tab, suf, n, entries, BLOCK=_ELEMENTS,
                              enable_fp_fusion=False)
        pre.fill_(float("inf"))
        for i in range(entries):
            kernels.prefix[egrid](s, tab, suf, pre, dist, n, i, BLOCK=_ELEMENTS,
                                  enable_fp_fusion=False)
            _launch_costs(kernels, plan, s, w, dist, vals, cost, blk, trials, tile)
            if p == 0 and i == 0:
                trial = tab.clone()
                trial[0] = vals[0]
                flags[1:2].add_((_lut_cost(s, w, trial) != cost[0]).to(torch.int32))
            kernels.accept[(1,)](cost, base, tab, cand, inuse, unused, vals, grid, flags,
                                 i, trials, bracket, num_warps=1, enable_fp_fusion=False)
        improved, differed, nonfinite = flags.tolist()
        if nonfinite or differed:
            return _fall_back(bool(nonfinite), plan)
        if not improved:
            break
    if improved:
        # ``swaps`` ran out on an improving pass: hold its running cost to torch too.
        if bool(_lut_cost(s, w, tab).to(torch.float64) != base):
            return _fall_back(False, plan)
    STATS["fused"] += 1
    return tab, cand.to(torch.long) + (first + first_byte)
