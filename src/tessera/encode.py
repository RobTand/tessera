"""The Tessera encoder: scales, trellis body, completion, release (doc S5-S9).

Vectorised over columns, which is the only axis that parallelises: the trellis
runs *down* a column (S5's per-column integer rates), so positions within a
column are sequentially dependent while columns are independent.  A GLM expert
is 4096 columns wide, so that is 4096-way parallelism at every step --
principle 7's GPU-first requirement met by the structure of the problem rather
than by a kernel.

The pass order is forced by what each stage needs:

1. **Group scales first.**  The trellis quantises ``w / scale`` against the
   E2M1 grid, so the scale has to exist before the body does.
2. **Viterbi body**, with S9's anticipated-completion metric.
3. **Completion bits**, which refine within the anchor's own tree.
4. **Release**, last, because S9's canonical placement orders positions by
   "descending decoded |value| within the superblock" -- the *pre-release*
   decoded value.  Ordering on the post-release value would be circular: the
   decoder cannot reproduce an order that depends on the overrides it has not
   read yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .alphabet import E2M1_VALUES, AnchorForest, build_forest
from .diagonals import (
    Diagonals,
    apply_diagonals,
    apply_rotation,
    fit_diagonals,
)
from .manifest import RotationState
from .errors import GrammarError
from .trellis import SUBSET_COUNT, ConvCode, TCQ

__all__ = ["EncodedUnit", "encode_unit", "e2m1_value_table"]

#: E4M3 scale grid, used for the segment-2b refinement (S6b).
_E4M3_MAX = 448.0


def e2m1_value_table(device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(E2M1_VALUES, device=device, dtype=dtype)


@dataclass
class EncodedUnit:
    """One encoded Linear: every plane, plus what the accountant needs."""

    rates: "tuple[int, ...]"
    anchors: torch.Tensor        # [rows, cols] int64, index into forest anchors
    codes: torch.Tensor          # [rows, cols] int64, E2M1 nibble after Stage C
    body_bits: torch.Tensor      # [rows, cols] int64, the R input bits per position
    completion_bits: torch.Tensor  # [rows, cols] int64, the c completion bits
    scale_base: torch.Tensor     # [groups] uint8, E8M0 exponent byte
    scale_refine: torch.Tensor   # [halves] uint8, 4-bit refinement word
    release_index: torch.Tensor  # [n_released] int64, flat position indices
    release_code: torch.Tensor   # [n_released] int64, the override nibble
    sse: float
    rotation: RotationState = RotationState.NONE
    rotation_block: int = 1
    diagonals: "Diagonals | None" = None   # segment 2a, None when not fitted

    @property
    def released_positions(self) -> int:
        return int(self.release_index.numel())


def _transition_tables(code: ConvCode, device):
    """For each next state, its two predecessors and the subsets they emit.

    A rate-1/2 code has exactly two branches into every state, so these are
    dense tables and the Viterbi step becomes two gathers and a min.
    """
    states = code.states
    prev = torch.zeros(2, states, dtype=torch.long, device=device)
    subset = torch.zeros(2, states, dtype=torch.long, device=device)
    filled = [0] * states
    for state in range(states):
        for bit in (0, 1):
            nxt, sub = code.step(state, bit)
            slot = filled[nxt]
            if slot > 1:
                raise GrammarError(
                    f"state {nxt} has more than two predecessors; the code is "
                    "not rate-1/2"
                )
            prev[slot, nxt] = state
            subset[slot, nxt] = sub
            filled[nxt] = slot + 1
    if any(count != 2 for count in filled):
        raise GrammarError("the trellis is not regular: some state has != 2 branches")
    return prev, subset


def _descendant_values(forest: AnchorForest, completion: int, device):
    """``[n_anchors, 2^c]`` of the values reachable at this completion level."""
    table = [
        [E2M1_VALUES[code] for code in forest.reachable(anchor, completion)]
        for anchor in range(len(forest.blocks))
    ]
    return torch.tensor(table, device=device, dtype=torch.float32)


def _subset_table(tcq: TCQ, device):
    """``[4, 2^(R-1)]`` of anchor indices, by subset."""
    return torch.tensor(tcq.subsets, device=device, dtype=torch.long)


def viterbi_columns(
    targets: torch.Tensor,
    forest: AnchorForest,
    code: ConvCode,
    completion: int,
) -> "tuple[torch.Tensor, torch.Tensor, float]":
    """Exact Viterbi down every column at once.

    ``targets`` is ``[rows, cols]`` already divided by its group scale.
    Returns ``(anchor_index[rows, cols], point_bits[rows, cols], sse)``.
    """
    device = targets.device
    rows, cols = targets.shape
    tcq = TCQ(forest, code)
    states = code.states
    prev, subset_of = _transition_tables(code, device)
    dvals = _descendant_values(forest, completion, device)      # [A, 2^c]
    subsets = _subset_table(tcq, device)                        # [4, P]
    points = subsets.shape[1]

    cost = torch.full((cols, states), float("inf"), device=device)
    cost[:, 0] = 0.0
    choice = torch.zeros(rows, cols, states, dtype=torch.bool, device=device)
    picked = torch.zeros(rows, cols, SUBSET_COUNT, dtype=torch.uint8, device=device)

    for row in range(rows):
        target = targets[row].unsqueeze(1).unsqueeze(2)          # [cols,1,1]
        # Anticipated-completion metric: score the best reachable descendant.
        err = ((target - dvals.unsqueeze(0)) ** 2).amin(dim=2)   # [cols, A]
        by_subset = err[:, subsets.reshape(-1)].reshape(cols, SUBSET_COUNT, points)
        best, point = by_subset.min(dim=2)                       # [cols, 4]
        picked[row] = point.to(torch.uint8)

        branch = torch.stack(
            [cost[:, prev[side]] + best[:, subset_of[side]] for side in (0, 1)]
        )                                                        # [2, cols, states]
        cost, taken = branch.min(dim=0)
        choice[row] = taken.bool()

    end = cost.argmin(dim=1)                                     # [cols]
    sse = float(cost.gather(1, end.unsqueeze(1)).sum())

    anchors = torch.zeros(rows, cols, dtype=torch.long, device=device)
    bits = torch.zeros(rows, cols, dtype=torch.long, device=device)
    column = torch.arange(cols, device=device)
    state = end
    for row in range(rows - 1, -1, -1):
        side = choice[row][column, state].long()                 # [cols]
        sub = subset_of[side, state]
        pt = picked[row][column, sub].long()
        anchors[row] = subsets[sub, pt]
        # ``side`` says which *predecessor* won, which is not the input bit.
        # For this code ``next = (bit << m-1) | (state >> 1)``, so both
        # predecessors of a state share one input bit and it is read off the
        # state itself.  Emitting ``side`` instead would produce a stream that
        # replays to different anchors -- the round-trip test catches it.
        select = (state >> (code.memory - 1)) & 1
        bits[row] = (select << (subsets.shape[1].bit_length() - 1)) | pt
        state = prev[side, state]
    return anchors, bits, sse


def _pack_scales(weights: torch.Tensor, group: int, half: int):
    """S6b: one E8M0 base byte per group, one 4-bit refinement per half.

    The refinement word is ``d`` (one exponent-delta bit) and ``m`` (three
    mantissa bits), giving the half's scale ``2^(E-127+d) * (1 + m/8)``.  Both
    halves share the base and ``d <= 1``, so the two half-exponents lie within
    one octave -- S6b is explicit that arbitrary legal E4M3 pairs are therefore
    *not* representable, and that the cost of that restriction is arm 5's to
    measure.
    """
    flat = weights.reshape(-1)
    groups = flat.reshape(-1, group)
    halves = flat.reshape(-1, half)
    amax_group = groups.abs().amax(dim=1).clamp_min(1e-30)
    amax_half = halves.abs().amax(dim=1).clamp_min(1e-30)

    # Base: the po2 that puts the group's amax at the top of the E2M1 range.
    target = amax_group / 6.0
    exponent = torch.floor(torch.log2(target)).clamp(-127, 128)
    base_byte = (exponent + 127).clamp(0, 255).to(torch.uint8)

    per_half = amax_half / 6.0
    base_for_half = torch.repeat_interleave(exponent, group // half)
    ratio = per_half / torch.exp2(base_for_half)
    # ratio in [1, 4); d picks the octave, m the mantissa within it.
    delta = (ratio >= 2.0).to(torch.long)
    mantissa = torch.floor((ratio / torch.exp2(delta.float()) - 1.0) * 8.0)
    mantissa = mantissa.clamp(0, 7).to(torch.long)
    refine = ((delta << 3) | mantissa).to(torch.uint8)
    effective = torch.exp2(base_for_half + delta.float()) * (1.0 + mantissa.float() / 8.0)
    return base_byte, refine, effective


def encode_unit(
    weights: torch.Tensor,
    forest: "AnchorForest | dict[int, AnchorForest]",
    rates: "tuple[int, ...]",
    code: ConvCode = ConvCode(),
    rotation: RotationState = RotationState.NONE,
    with_diagonals: bool = False,
    completion: int | None = None,
    released_positions: int = 0,
    group: int = 32,
    half: int = 16,
    superblock: int = 256,
) -> EncodedUnit:
    """Encode one Linear.  ``weights`` is ``[rows, cols]`` in the source dtype."""
    if weights.ndim != 2:
        raise GrammarError(f"expected a 2-D weight, got shape {tuple(weights.shape)}")
    rows, cols = weights.shape
    if len(rates) != cols:
        raise GrammarError(f"{len(rates)} rates for {cols} columns")
    forests = forest if isinstance(forest, dict) else {forest.rate: forest}
    for present in sorted(set(rates)):
        if present not in forests:
            raise GrammarError(
                f"the schedule uses rate {present} but no forest was supplied "
                f"for it; got forests for {sorted(forests)}"
            )
    device = weights.device

    # S5 transforms, outermost first: rotate the input basis, then remove the
    # rank-1 magnitude field, then set scales on what is left.  The order is
    # forced -- fitting diagonals before rotating would fit the rotation's
    # own structure, and setting scales first would price a matrix the body
    # never sees.
    rotated, rotation_block = apply_rotation(weights, rotation)
    fitted = fit_diagonals(rotated) if with_diagonals else None
    work = apply_diagonals(rotated, fitted) if fitted else rotated

    base_byte, refine, effective = _pack_scales(work, group, half)
    scale = torch.repeat_interleave(effective, half).reshape(rows, cols)
    targets = work / scale

    anchors = torch.zeros(rows, cols, dtype=torch.long, device=device)
    body_bits = torch.zeros(rows, cols, dtype=torch.long, device=device)
    completion_bits = torch.zeros(rows, cols, dtype=torch.long, device=device)
    codes = torch.zeros(rows, cols, dtype=torch.long, device=device)
    values = e2m1_value_table(device)
    rate_vector = torch.tensor(rates, device=device)
    sse = 0.0

    # One Viterbi per rate: columns are independent, so a mixed-rate schedule
    # is a partition of columns and not a harder problem.
    for present in sorted(set(rates)):
        picked = forests[present]
        depth = 3 - present
        level = depth if completion is None else min(completion, depth)
        which = torch.nonzero(rate_vector == present).squeeze(1)
        sub = targets[:, which].contiguous()
        a, b, s_ = viterbi_columns(sub, picked, code, level)
        sse += s_
        blocks = torch.tensor(picked.blocks, device=device, dtype=torch.long)
        reachable = blocks[:, :: 1 << (depth - level)]
        per_pos = values[reachable][a]
        c_bits = ((sub.unsqueeze(2) - per_pos) ** 2).argmin(dim=2)
        anchors[:, which] = a
        body_bits[:, which] = b
        completion_bits[:, which] = c_bits
        codes[:, which] = reachable[a, c_bits]

    # Stage B: release, in S9's canonical order -- descending |decoded value|
    # within the superblock, on the PRE-release decode so the decoder can
    # reproduce the order from bytes it already has.
    release_index = torch.zeros(0, dtype=torch.long, device=device)
    release_code = torch.zeros(0, dtype=torch.long, device=device)
    if released_positions:
        decoded = values[codes] * scale
        release_index = _canonical_release_order(
            decoded, cols, superblock, released_positions
        )
        flat_t = (work.reshape(-1))[release_index]
        flat_s = (scale.reshape(-1))[release_index]
        best = ((flat_t / flat_s).unsqueeze(1) - values.unsqueeze(0)) ** 2
        release_code = best.argmin(dim=1)
        codes.reshape(-1)[release_index] = release_code

    return EncodedUnit(
        rates=rates,
        anchors=anchors,
        codes=codes,
        body_bits=body_bits,
        completion_bits=completion_bits,
        scale_base=base_byte,
        scale_refine=refine,
        release_index=release_index,
        release_code=release_code,
        sse=sse,
        rotation=rotation,
        rotation_block=rotation_block,
        diagonals=fitted,
    )


def _canonical_release_order(
    decoded: torch.Tensor, cols: int, superblock: int, total: int
) -> torch.Tensor:
    """S9's release placement: which positions get a 4-bit override.

    S9 fixes the order *within* a superblock -- "descending decoded |value|
    within the superblock, positional tie-break" -- and puts the **count** for
    each superblock in the manifest ("each plane's per-superblock count
    vector").  So placement is free but the counts are charged, and this
    returns the flat indices in the order the decoder will reconstruct them.

    Counts are spread across superblocks by the same exact Bresenham quota the
    rate schedule uses, so the total is met exactly with no superblock more
    than one release away from any other.  A quality-driven allocation is S9's
    lambda-greedy pass; this is the uniform baseline it has to beat.
    """
    device = decoded.device
    rows = decoded.shape[0]
    blocks = max(1, cols // superblock)
    per, remainder = divmod(total, blocks)
    counts = [per + (1 if index < remainder else 0) for index in range(blocks)]

    flat = decoded.abs().reshape(-1)
    position = torch.arange(flat.numel(), device=device)
    block_of = (position % cols) // superblock

    chosen = []
    for index, count in enumerate(counts):
        if not count:
            continue
        members = position[block_of == index]
        magnitude = flat[members]
        # Stable descending sort: ties fall back to ascending position, which
        # makes the order total and so reproducible by the decoder.
        order = torch.argsort(magnitude, descending=True, stable=True)
        chosen.append(members[order[:count]])
    if not chosen:
        return torch.zeros(0, dtype=torch.long, device=device)
    return torch.cat(chosen)
