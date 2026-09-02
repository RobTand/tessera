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

import functools

from .alphabet import (
    E2M1_GRID,
    E2M1_VALUES,
    GAUSSIAN_SOURCE,
    GROUP_SCALED_SOURCE,
    AnchorForest,
    PayloadGrid,
    build_forest,
)
from .diagonals import (
    Diagonals,
    apply_diagonals,
    apply_rotation,
    fit_diagonals,
)
from .manifest import WINDOW_BITS_MAX, BodyKind, RotationState, ScalePlaneKind
from .errors import GrammarError
from .trellis import SUBSET_COUNT, ConvCode, TCQ

__all__ = [
    "EncodedUnit",
    "encode_unit",
    "e2m1_value_table",
    "grid_value_table",
    "LUT_ENTRIES",
]

#: Entries in the per-unit scale table of a ``ScalePlaneKind.LUT`` plane.  The
#: SCALE_REFINE plane is four bits per half, so sixteen is what a nibble
#: indexes; fewer is legal on the wire and wastes index bits.
LUT_ENTRIES = 16

#: E4M3 scale grid, used for the segment-2b refinement (S6b).
_E4M3_MAX = 448.0


def e2m1_value_table(device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(E2M1_VALUES, device=device, dtype=dtype)


def grid_value_table(
    grid: PayloadGrid = E2M1_GRID, device=None, dtype=torch.float32
) -> torch.Tensor:
    """``slot -> value`` for any payload grid.

    TESSERA-4 and TESSERA-8 are one construction at two grid widths, so every
    place that used to reach for the E2M1 table takes the forest's own grid
    instead.  ``e2m1_value_table`` survives as the TESSERA-4 spelling because
    the kernel lane and the NVFP4 materialiser are E2M1 by definition.
    """
    if grid.arity != 1:
        raise GrammarError(
            f"grid {grid.name} has arity {grid.arity}: a code decodes to "
            f"{grid.arity} values, not one. Use grid_vector_table."
        )
    return torch.tensor(grid.values, device=device, dtype=dtype)


def grid_vector_table(
    grid: PayloadGrid = E2M1_GRID, device=None, dtype=torch.float32
) -> torch.Tensor:
    """``[size, arity]`` -- every code's reconstruction, whatever its width.

    The scalar table is the ``arity == 1`` column of this one, so the encoder
    and decoder run the arity-1 path through the same expression: a sum over a
    length-1 axis is exact in floating point, which is what makes "arity 1 is
    byte-identical" a property rather than a hope.
    """
    return torch.tensor(grid.values, device=device, dtype=dtype).reshape(
        grid.size, grid.arity
    )


@dataclass
class EncodedUnit:
    """One encoded Linear: every plane, plus what the accountant needs."""

    rates: "tuple[int, ...]"
    anchors: torch.Tensor        # [rows, cols] int64, index into forest anchors
    codes: torch.Tensor          # [rows, cols] int64, E2M1 nibble after Stage C
    body_bits: torch.Tensor      # [rows, cols] uint8, the R input bits per position
    completion_bits: torch.Tensor  # [rows, cols] int64, the c completion bits
    scale_base: torch.Tensor     # [groups] uint8, E8M0 exponent byte
    scale_refine: torch.Tensor   # [halves] uint8, 4-bit refinement word
    release_index: torch.Tensor  # [n_released] int64, flat position indices
    release_code: torch.Tensor   # [n_released] int64, the override nibble
    sse: float
    rotation: RotationState = RotationState.NONE
    rotation_block: int = 1
    diagonals: "Diagonals | None" = None   # segment 2a, None when not fitted
    # The segment-2b block geometry.  Stored because a decoder holding only the
    # planes cannot otherwise turn ``scale_base``/``scale_refine`` back into a
    # per-position scale: the byte counts alone do not say which axis they run
    # along.  These travel in the manifest geometry on the wire.
    group: int = 32
    half: int = 16
    # The completion depth this unit was *encoded* at, or None for "as deep as
    # each rate allows".  It is not the same as ``3 - R``: the encoder truncates
    # to ``min(completion, depth)`` (see ``encode_unit``), so a unit written at
    # completion=0 carries no completion information at all.  The serialiser
    # needs it to size the COMPLETION plane at the depth that was used rather
    # than the depth the rate leaves room for -- writing the wider plane is what
    # made every rung of a family weigh the same.
    completion_limit: "int | None" = None
    # How many scale-plane refits the encoder ran (``encode_unit``): that many
    # trellis passes and that many refits, the last refit trailing.  Not wire:
    # the bytes decode identically at any value.  Recorded because two units
    # built at different settings are different renderings of one weight, and
    # a merge that mixes them should be able to see that it did.
    scale_refit: int = 0
    # The super-symbol length L of the trellis (``trellis.py``): one select
    # bit per L positions, ``L - 1`` stored two-bit labels, ``LR + L - 1``
    # body bits per super-symbol.  Wire: the reader needs it to slice the body
    # and to replay the code, so it travels in the manifest and is bound into
    # the encoder profile id.  ``1`` is the per-position trellis, byte for byte.
    span: int = 1
    # The scale plane's *kind*.  ``S6B`` is ``scale_base`` + ``scale_refine``
    # read by ``scales_from_planes``.  ``LUT`` drops the base plane: the
    # SCALE_REFINE nibble indexes ``scale_lut`` -- up to sixteen distinct E4M3
    # bytes chosen per unit -- times ``scale_global``.  Same 4 bits per half,
    # half the plane's bytes, and the half's scale is still one E4M3 behind an
    # fp32 global, which is exactly the NVFP4 tile's two-level scale.
    scale_plane: ScalePlaneKind = ScalePlaneKind.S6B
    scale_lut: "torch.Tensor | None" = None   # [<=16] uint8 E4M3FN bytes, ascending
    scale_global: float = 1.0
    # The BODY plane's *kind* (schema minor 2).  Under ``WINDOW`` the trellis
    # is the bitshift trellis: ``anchors`` holds the per-position STATE (the
    # last ``window_bits`` bits of the column's stream), ``body_bits`` the R
    # new bits per position, ``codes`` is ``window_codes[state]``, and the
    # completion plane is empty.  ``window_codes`` is the table -- one grid
    # code per state -- and it travels on the ALPHABET plane.  Wire, all
    # three: bound into the encoder profile id and read off the manifest.
    body: BodyKind = BodyKind.TCQ
    window_bits: int = 0
    window_codes: "torch.Tensor | None" = None   # [2^window_bits] uint8 grid codes
    # A CHANNEL scale plane (schema minor 3): one fp16 word per output row,
    # times ``scale_global``; ``scale_base`` and ``scale_refine`` are empty.
    # Travels on the DIAG_SV plane (``scale_channel.py``).
    scale_rows: "torch.Tensor | None" = None     # [rows] fp16

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
    """``[n_anchors, 2^c, arity]`` of the reconstructions reachable at level c."""
    grid = forest.grid
    table = [
        [grid.vector(code) for code in forest.reachable(anchor, completion)]
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
    span: int = 1,
    weights: "torch.Tensor | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor, float]":
    """Exact Viterbi down every column at once.

    ``targets`` is ``[rows, cols]`` already divided by its group scale.
    Returns ``(anchor_index[steps, cols], body_field[steps, cols], sse)`` where
    ``steps = rows // grid.arity`` -- one trellis position per *code*, which is
    one row only when the grid is scalar.

    ``weights`` is an optional ``[rows, cols]`` positive weight per POSITION
    on the branch metric.  The targets are normalised per half, so an
    unweighted trellis minimises ``sum (w/c - q)^2`` -- every position's error
    divided by its own scale squared, which over-serves the quiet groups a
    column passes through.  With ``weights = c^2`` the path minimises the
    true ``sum (w - c q)^2``.  The weight is applied per row inside a code's
    Euclidean sum: a half is sixteen consecutive columns of one row, so the
    ``arity`` rows of a code sit in different halves and carry different
    scales.  ``None`` is bit-identical to the unweighted encoder.

    ``span`` is the super-symbol length L (``trellis.py``).  One trellis step
    then covers L consecutive positions: each position's best point per subset
    is found independently, the L per-subset cost vectors are folded with a
    min-plus convolution over Z/4 -- ``acc[l] = min_v acc[(l - v) mod 4] +
    best[v]`` -- so the trellis branch sees one four-entry cost vector per
    super-symbol exactly as it sees one per position at L = 1, and the
    traceback descends the fold to recover every position's label.  The fold
    is exact: it is the same minimisation the scalar oracle does by exhausting
    ``4^(L-1)`` label assignments, in ``L`` steps of a 4x4 minimum.

    The body field per position is ``[select | point]`` at position 0 of a
    super-symbol and ``[label | point]`` at the others (``R + 1`` bits); at
    ``L = 1`` every position is ``[select | point]`` and this function is
    bit-identical to the per-position encoder it replaces.
    """
    device = targets.device
    rows, cols = targets.shape
    grid = forest.grid
    arity = grid.arity
    if rows % arity:
        raise GrammarError(
            f"{rows} rows is not a whole number of arity-{arity} tuples; a "
            "k-tuple code spans k consecutive rows and cannot straddle the edge"
        )
    steps = rows // arity
    if span < 1 or steps % span:
        raise GrammarError(
            f"{steps} trellis positions is not a whole number of span-{span} "
            "super-symbols; the multidimensional trellis needs the column "
            "length to be a multiple of its span"
        )
    supers = steps // span
    tcq = TCQ(forest, code)
    states = code.states
    prev, subset_of = _transition_tables(code, device)
    dvals = _descendant_values(forest, completion, device)      # [A, 2^c, arity]
    subsets = _subset_table(tcq, device)                        # [4, P]
    points = subsets.shape[1]
    # The traceback stores the winning point index per (step, column, subset).
    # A byte holds it up to rate 9; above that the index must widen or it wraps
    # silently -- the same "a code is a nibble" assumption that corrupted the
    # body plane at rate 9.  Width follows the rate; it is never assumed.
    point_dtype = torch.uint8 if points <= 256 else torch.int32

    # [steps, arity, cols]: a tuple is ``arity`` CONSECUTIVE ROWS of one
    # column, because the trellis runs down columns and the k positions of a
    # code have to share one branch decision.
    tuples = targets.reshape(steps, arity, cols)
    wrows = None if weights is None else weights.reshape(steps, arity, cols)

    cost = torch.full((cols, states), float("inf"), device=device)
    cost[:, 0] = 0.0
    choice = torch.zeros(supers, cols, states, dtype=torch.bool, device=device)
    picked = torch.zeros(steps, cols, SUBSET_COUNT, dtype=point_dtype, device=device)
    # The fold's argument per stored position: which label ``v`` position i
    # took, indexed by the accumulated label after it.  Empty at L = 1.
    fold = torch.zeros(
        supers, max(span - 1, 0), cols, SUBSET_COUNT, dtype=torch.uint8, device=device
    )
    roll = torch.arange(SUBSET_COUNT, device=device)

    for sup in range(supers):
        acc = None
        for offset in range(span):
            step = sup * span + offset
            target = tuples[step].t().reshape(cols, 1, 1, arity)     # [cols,1,1,k]
            # Anticipated-completion metric, in k dimensions: score the best
            # reachable descendant under squared Euclidean distance.  At k=1
            # the sum is over one term and this is bit-identical to the
            # scalar form.
            sq = (target - dvals.unsqueeze(0)) ** 2                       # [cols,A,2^c,k]
            if weights is not None:
                sq = sq * wrows[step].t().reshape(cols, 1, 1, arity)
            err = sq.sum(dim=3).amin(dim=2)                              # [cols,A]
            by_subset = err[:, subsets.reshape(-1)].reshape(cols, SUBSET_COUNT, points)
            best, point = by_subset.min(dim=2)                       # [cols, 4]
            picked[step] = point.to(point_dtype)
            if acc is None:
                acc = best
                continue
            terms = torch.stack(
                [acc[:, (roll - v) % SUBSET_COUNT] + best[:, v : v + 1]
                 for v in range(SUBSET_COUNT)],
                dim=2,
            )                                                        # [cols, 4, 4]
            acc, arg = terms.min(dim=2)
            fold[sup, offset - 1] = arg.to(torch.uint8)

        branch = torch.stack(
            [cost[:, prev[side]] + acc[:, subset_of[side]] for side in (0, 1)]
        )                                                        # [2, cols, states]
        cost, taken = branch.min(dim=0)
        choice[sup] = taken.bool()

    end = cost.argmin(dim=1)                                     # [cols]
    sse = float(cost.gather(1, end.unsqueeze(1)).sum())

    anchors = torch.zeros(steps, cols, dtype=torch.long, device=device)
    bits = torch.zeros(steps, cols, dtype=torch.long, device=device)
    column = torch.arange(cols, device=device)
    shift = points.bit_length() - 1
    state = end
    for sup in range(supers - 1, -1, -1):
        side = choice[sup][column, state].long()                 # [cols]
        label = subset_of[side, state]                           # super-label
        # ``side`` says which *predecessor* won, which is not the input bit.
        # For this code ``next = (bit << m-1) | (state >> 1)``, so both
        # predecessors of a state share one input bit and it is read off the
        # state itself.  Emitting ``side`` instead would produce a stream that
        # replays to different anchors -- the round-trip test catches it.
        select = (state >> (code.memory - 1)) & 1
        labels = [None] * span
        for offset in range(span - 1, 0, -1):
            v = fold[sup, offset - 1][column, label].long()
            labels[offset] = v
            label = (label - v) % SUBSET_COUNT
        labels[0] = label
        for offset in range(span):
            step = sup * span + offset
            pt = picked[step][column, labels[offset]].long()
            anchors[step] = subsets[labels[offset], pt]
            head = select if offset == 0 else labels[offset]
            bits[step] = (head << shift) | pt
        state = prev[side, state]
    return anchors, bits, sse


@functools.lru_cache(maxsize=64)
def _window_table_cpu(
    grid: PayloadGrid, window_bits: int, sigma: "float | None", seed: int, half: int,
) -> torch.Tensor:
    size = 1 << window_bits
    peak = max(abs(v) for v in grid.values)
    # Equal-mass quantiles of the modelled source, one set per coordinate.
    # ``sigma=None`` models what the per-half scale plane actually delivers
    # to the grid -- a Gaussian normalised by its own half's maximum, bounded
    # at the peak (``GROUP_SCALED_SOURCE``); a number is a plain Gaussian in
    # grid units for a plane that does not bound (a per-channel scale).
    if sigma is None:
        # The group-scaled source is built in whole halves, so a table
        # narrower than a half takes order statistics of one half's worth.
        count = max(size, half)
        sample = torch.tensor(GROUP_SCALED_SOURCE(peak, half, count=count))
        if sample.numel() != count:
            raise GrammarError(
                f"the group-scaled source yielded {sample.numel()} values for "
                f"{count}; half {half} must divide the table size"
            )
        picks = ((torch.arange(size, dtype=torch.float64) + 0.5) * count / size).long()
        quantiles = sample[picks]
    else:
        quantiles = torch.tensor(GAUSSIAN_SOURCE(size, float(sigma)))
    generator = torch.Generator().manual_seed(int(seed))
    points = torch.stack(
        [quantiles[torch.randperm(size, generator=generator)] for _ in range(grid.arity)],
        dim=1,
    ).float()                                                   # [size, arity]
    vectors = grid_vector_table(grid).float()                   # [codes, arity]
    # Nearest grid vector, ties to the lower code: E4M3's duplicate slots
    # (the two former NaNs, the negative zero) sit above the legal byte.
    if grid.arity == 1 and grid.size > 256:
        codes = _nearest_scalar_code(points[:, 0], grid)
    else:
        codes = torch.cdist(points, vectors).argmin(dim=1)
    return codes.to(torch.uint8) if grid.size <= 256 else codes.to(torch.int32)


def _nearest_scalar_code(points: torch.Tensor, grid: PayloadGrid) -> torch.Tensor:
    """``argmin_c |points - value(c)|``, ties to the lower code, exactly.

    The same answer ``cdist(...).argmin`` gives, computed the way a sorted
    one-dimensional grid allows -- and computed in float64, which matters at
    this width for two separate reasons.  BF16 is 65536 codes: the pairwise
    matrix is 4 GB at L=14, and ``cdist``'s ``x^2 + y^2 - 2xy`` expansion is
    not the exact ``|x - y|``, so a quantile a hair from a midpoint can land
    on the wrong side of it in float32.  Neither is a problem a byte-wide
    grid has, which is why the narrow path is left exactly as it was.

    "Ties to the lower code" is a statement about *codes*, not values, so
    duplicate values are collapsed to the lowest code carrying them before
    the search and the two candidates are compared as codes at an exact
    midpoint.  On BF16 that makes the snap round-half-toward-zero, next to
    bf16 hardware's round-half-to-even; they differ only on exact midpoints
    of the table's Gaussian quantiles, which the receipt counts.
    """
    values = torch.tensor(grid.values, dtype=torch.float64)
    order = torch.argsort(values, stable=True)
    ordered = values[order]
    # First code of each distinct value, in ascending value order.
    keep = torch.ones(ordered.numel(), dtype=torch.bool)
    keep[1:] = ordered[1:] != ordered[:-1]
    uniq, ucode = ordered[keep], torch.zeros(int(keep.sum()), dtype=torch.long)
    # ``order`` is a stable sort, so among equal values the lowest code comes
    # first -- but only within the sort's own tie order, which is code order.
    ucode.scatter_reduce_(
        0, torch.cumsum(keep.long(), 0) - 1, order, reduce="amin", include_self=False
    )
    p = points.double()
    right = torch.searchsorted(uniq, p).clamp(0, uniq.numel() - 1)
    left = (right - 1).clamp_min(0)
    dl, dr = (p - uniq[left]).abs(), (p - uniq[right]).abs()
    take_left = torch.where(
        dl == dr, ucode[left] < ucode[right], dl < dr
    )
    return torch.where(take_left, ucode[left], ucode[right])


def window_table(
    grid: PayloadGrid,
    window_bits: int,
    *,
    sigma: "float | None" = None,
    seed: int = 0,
    half: int = 16,
    device=None,
) -> torch.Tensor:
    """The window body's table: ``2^window_bits`` grid codes, one per state.

    A state is the last ``window_bits`` bits of a column's stream, so the
    table is what turns shared history into shaped reconstruction: the
    trellis picks a path whose states index good values, and the shaping
    gain comes from the ``2^(window_bits - R)`` states every R-bit choice can
    land in.  The entries are a seeded permutation of equal-mass quantiles of
    the modelled source, snapped to the grid (Tseng et al.'s "random Gaussian
    codebook", on the tile); a computed (hash) table was measured worse than
    the stored one at every width, so the table is stored, on the ALPHABET
    plane, and priced there.

    Deterministic in ``(grid, window_bits, sigma, seed, half)`` and cached:
    every unit on one grid at one width shares the table, and a table that
    changed run to run would make artifacts irreproducible.  The table is
    wire regardless -- a reader takes it off the plane, never rebuilds it.
    """
    if not 1 <= window_bits <= WINDOW_BITS_MAX:
        raise GrammarError(f"window_bits {window_bits} outside 1..{WINDOW_BITS_MAX}")
    if sigma is not None and not sigma > 0:
        raise GrammarError(f"the window source sigma must be positive, got {sigma}")
    table = _window_table_cpu(grid, int(window_bits), sigma, int(seed), int(half))
    return table.to(device) if device is not None else table.clone()


def viterbi_window(
    targets: torch.Tensor,
    vectors: torch.Tensor,
    window_bits: int,
    rate: int,
    weights: "torch.Tensor | None" = None,
    chunk: int = 512,
    impl: str = "auto",
) -> "tuple[torch.Tensor, float]":
    """Exact Viterbi over the bitshift trellis, down every column at once.

    ``targets`` is ``[rows, cols]`` already divided by its scale; ``vectors``
    is ``[2^window_bits, arity]`` -- the table's reconstruction per state.
    Returns ``(state[steps, cols] int64, sse)``, ``steps = rows // arity``.

    The trellis: ``state_t = ((state_{t-1} << R) | bits_t) mod 2^L`` from
    ``state_{-1} = 0``, so a state's ``2^R`` predecessors share its low
    ``L - R`` bits and differ in the ``R`` bits that fall off the top.  One
    step is a minimum over those predecessors per low class -- a ``[2^R,
    2^(L-R)]`` reduction -- then every state adds its own branch cost.  The
    start is **pinned** at state 0, exactly as the decoder assumes: a free
    start would encode information the reader cannot recover.

    ``weights`` is the same per-POSITION branch-metric weight as
    ``viterbi_columns`` takes; ``chunk`` bounds the column batch, since the
    cost front is ``2^L`` floats per column and the traceback ``2^(L-R)``
    bytes per position per column.

    ``impl`` picks the machine, never the answer.  ``"reference"`` is the
    torch chain below -- the definition, and the only path on CPU;
    ``"fused"`` is the Triton step kernel in ``window_viterbi``, which
    returns identical states and the identical sse float (see that module for
    why that is a contract and not a hope); ``"auto"`` takes the fused path
    on CUDA inputs when Triton is present and the reference otherwise.
    """
    if impl not in ("auto", "reference", "fused"):
        raise GrammarError(f"unknown viterbi_window impl {impl!r}")
    device = targets.device
    rows, cols = targets.shape
    size, arity = vectors.shape
    if size != 1 << window_bits:
        raise GrammarError(
            f"the table holds {size} states, window_bits {window_bits} needs {1 << window_bits}"
        )
    if not 1 <= rate <= window_bits:
        raise GrammarError(f"rate {rate} does not fit a {window_bits}-bit window")
    if rows % arity:
        raise GrammarError(
            f"{rows} rows is not a whole number of arity-{arity} tuples; a "
            "k-tuple code spans k consecutive rows and cannot straddle the edge"
        )
    steps = rows // arity
    fan = 1 << rate                                  # predecessors per state
    low = size >> rate                               # low classes
    if impl != "reference":
        from .window_viterbi import fused_available, viterbi_window_fused

        if targets.is_cuda and fused_available():
            return viterbi_window_fused(targets, vectors, window_bits, rate,
                                        weights=weights, chunk=chunk)
        if impl == "fused":
            raise GrammarError(
                "the fused window Viterbi is a CUDA path and needs triton; "
                f"targets are on {device} and triton is "
                f"{'present' if fused_available() else 'absent'}"
            )
    tuples = targets.float().reshape(steps, arity, cols)
    wrows = None if weights is None else weights.float().reshape(steps, arity, cols)
    table = vectors.float().to(device)
    states = torch.empty(steps, cols, dtype=torch.long, device=device)
    sse = 0.0
    for start in range(0, cols, chunk):
        x = tuples[:, :, start : start + chunk]                  # [steps, arity, n]
        n = x.shape[2]
        cost = torch.full((size, n), float("inf"), device=device)
        cost[0] = 0.0
        # The traceback stores the winning predecessor's top R bits per
        # (step, low class, column).  A byte holds it up to rate 8.
        back = torch.empty(steps, low, n, dtype=torch.uint8 if fan <= 256 else torch.int32,
                           device=device)
        for step in range(steps):
            best, pred = cost.view(fan, low, n).min(dim=0)      # [low, n]
            back[step] = pred.to(back.dtype)
            diff = x[step].t().unsqueeze(1) - table.unsqueeze(0)  # [n, size, arity]
            diff = diff * diff
            if wrows is not None:
                diff = diff * wrows[step, :, start : start + chunk].t().unsqueeze(1)
            branch = diff.sum(dim=2).t()                          # [size, n]
            # new state = (low class << R) | new bits: consecutive states
            # share one predecessor class.
            cost = best.repeat_interleave(fan, dim=0) + branch
        final, state = cost.min(dim=0)                           # [n]
        sse += float(final.sum())
        column = torch.empty(steps, n, dtype=torch.long, device=device)
        for step in range(steps - 1, -1, -1):
            column[step] = state
            lowbits = state >> rate
            pred = back[step].gather(0, lowbits.unsqueeze(0)).squeeze(0).long()
            state = (pred << (window_bits - rate)) | lowbits
        states[:, start : start + chunk] = column
    return states, sse


def _pack_scales(
    weights: torch.Tensor, group: int, half: int, peak: float = 6.0,
    headroom: float = 1.0,
):
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

    # Base: the po2 that puts the group's amax at the top of the payload grid's
    # range -- 6.0 for E2M1, 448.0 for E4M3.  Scaling to the wrong peak wastes
    # binades at one end and clips at the other.
    #
    # ``headroom`` scales that landing point.  ``amax / peak`` is a *heuristic*:
    # it guarantees nothing is clipped, which is not the same as minimising
    # error.  Below 1.0 the extremes clip and everything else is coded finer;
    # above 1.0 the reverse.  Which is better is a property of the weight
    # distribution and the grid, so it is a question for the objective, not for
    # a rule -- principle 2.  1.0 is exactly today's behaviour, byte for byte.
    target = amax_group / (peak * headroom)
    exponent = torch.floor(torch.log2(target)).clamp(-127, 128)
    base_byte = (exponent + 127).clamp(0, 255).to(torch.uint8)

    per_half = amax_half / (peak * headroom)
    base_for_half = torch.repeat_interleave(exponent, group // half)
    ratio = per_half / torch.exp2(base_for_half)
    # ratio in [1, 4); d picks the octave, m the mantissa within it.
    delta = (ratio >= 2.0).to(torch.long)
    mantissa = torch.floor((ratio / torch.exp2(delta.float()) - 1.0) * 8.0)
    mantissa = mantissa.clamp(0, 7).to(torch.long)
    refine = ((delta << 3) | mantissa).to(torch.uint8)
    effective = torch.exp2(base_for_half + delta.float()) * (1.0 + mantissa.float() / 8.0)
    return base_byte, refine, effective


def _refit_scales(
    work: torch.Tensor,
    units: torch.Tensor,
    group: int,
    half: int,
    base_byte: torch.Tensor,
    refine: torch.Tensor,
    effective: torch.Tensor,
):
    """One least-squares step on the scale plane, landed on S6b words.

    ``_pack_scales`` sets a half's scale from its amax so that nothing clips.
    That is a legality rule, not a minimiser: once the trellis has chosen its
    codes, the scale that minimises the half's squared error is
    ``<w, u> / <u, u>`` over the half's weights ``w`` and their unscaled grid
    values ``u`` -- and it is lower, because amax/peak spends the half's whole
    range on one element.  This step moves every half toward that optimum.

    The optimum is a float; the wire stores one E8M0 base per group and a
    ``(d, m)`` refinement per half.  With the codes fixed a half's error is a
    parabola in its scale, ``A s^2 - 2 B s + C`` with ``A = <u, u>`` and
    ``B = <w, u>``, so the choice between candidate words is exact arithmetic
    on ``A`` and ``B``: three base candidates are tried, each half rounds to its
    nearest ``(d, m)`` under that base, and a group keeps the word it had
    wherever no candidate lowers its error.  No group ends worse than it began,
    and the trellis pass that follows is optimal for the new plane, so the
    alternation in ``encode_unit`` is monotone in weight-space squared error.

    Words are emitted canonical: a group whose halves both carry ``d = 1`` is
    written one base higher with ``d = 0``, the form ``scale_codec`` names.
    The bytes' meaning is unchanged -- ``scales_from_planes`` reads them exactly
    as it reads the amax plane -- which is what makes this an encoder choice
    and not a wire change.
    """
    per_group = group // half
    W = work.float().reshape(-1, half)
    U = units.float().reshape(-1, half)
    A = (U * U).sum(dim=1)
    B = (W * U).sum(dim=1)
    # A half whose codes are all zero, or whose fit points the wrong way, has
    # no least-squares scale; it keeps the one it has.
    desired = torch.where(A > 0, B / A.clamp_min(1e-30), effective)
    desired = torch.where(desired > 0, desired, effective)

    g = desired.reshape(-1, per_group)
    Ag, Bg = A.reshape(-1, per_group), B.reshape(-1, per_group)
    prev = effective.reshape(-1, per_group)
    best_cost = (Ag * prev * prev - 2.0 * Bg * prev).sum(dim=1)
    best_E = base_byte.to(torch.float32) - 127.0
    word = refine.to(torch.long).reshape(-1, per_group)
    best_d = ((word >> 3) & 1).to(torch.float32)
    best_m = (word & 7).to(torch.float32)

    lo = torch.floor(torch.log2(g.min(dim=1).values))
    hi = torch.floor(torch.log2(g.max(dim=1).values)) - 1.0
    for E in (lo, lo + 1.0, hi):
        E = E.clamp(-127.0, 126.0)
        ratio = (g / torch.exp2(E)[:, None]).clamp(1.0, 4.0)
        d = (ratio >= 2.0).to(torch.float32)
        m = torch.round((ratio / torch.exp2(d) - 1.0) * 8.0)
        carry = (m >= 8.0) & (d == 0.0)
        m = torch.where(carry, torch.zeros_like(m), m)
        d = torch.where(carry, torch.ones_like(d), d)
        m = m.clamp(max=7.0)
        both = d.min(dim=1).values == 1.0
        E = torch.where(both, E + 1.0, E)
        d = torch.where(both[:, None], torch.zeros_like(d), d)
        eff = torch.exp2(E[:, None] + d) * (1.0 + m / 8.0)
        cost = (Ag * eff * eff - 2.0 * Bg * eff).sum(dim=1)
        take = cost < best_cost
        best_cost = torch.where(take, cost, best_cost)
        best_E = torch.where(take, E, best_E)
        best_d = torch.where(take[:, None], d, best_d)
        best_m = torch.where(take[:, None], m, best_m)

    new_base = (best_E + 127.0).clamp(0.0, 255.0).to(torch.uint8)
    new_refine = ((best_d.to(torch.long) << 3) | best_m.to(torch.long)).reshape(-1).to(torch.uint8)
    new_effective = torch.exp2(
        torch.repeat_interleave(best_E, per_group) + best_d.reshape(-1)
    ) * (1.0 + best_m.reshape(-1) / 8.0)
    return new_base, new_refine, new_effective


E4M3_NORMAL_BYTES = (0x08, 0x7E)   # inclusive: 2^-6 .. 448


def e4m3_positive_values(device=None) -> torch.Tensor:
    """``[119]`` -- the positive *normal* E4M3FN values, ascending; entry ``i``
    is byte ``i + 8``.

    Subnormals (bytes 0x01..0x07, ``2^-6 * m/8``) are excluded on purpose: the
    kernel lane decodes a scale byte by field arithmetic, ``2^(e-7) (1+m/8)``,
    which is the S6b relabelling's contract (``wire.nvfp4_scale_bytes`` only
    ever emits exponent fields 1..15) and is wrong at exponent field 0.  The
    LUT plane is materialised to the same bytes, so it holds to the same range.
    Seventeen binades remain; a unit whose scales span more is not a unit.
    """
    lo, hi = E4M3_NORMAL_BYTES
    return (
        torch.arange(lo, hi + 1, dtype=torch.uint8, device=device)
        .view(torch.float8_e4m3fn)
        .float()
    )


def _lut_cost(targets: torch.Tensor, weights: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """``sum_h A_h (s_h - nearest(table, s_h))^2`` -- the plane's weighted error."""
    gap = (targets[:, None] - table[None, :]).abs().amin(dim=1)
    return (weights * gap * gap).sum()


def _nearest(targets: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Index of the nearest table entry, in the linear domain.

    With the codes fixed a half's error is ``A s^2 - 2 B s + C``, a parabola
    with its minimum at ``s* = B/A``, so among candidate scales the nearest to
    ``s*`` in *linear* distance is the exact minimiser -- not the nearest in
    log distance, which is what an E4M3 rounder would do.
    """
    return (targets[:, None] - table[None, :]).abs().argmin(dim=1)


def _fit_lut(
    targets: torch.Tensor,
    weights: torch.Tensor,
    global_scale: float,
    entries: int = LUT_ENTRIES,
    swaps: int = 2,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Choose ``entries`` DISTINCT E4M3 scales minimising the weighted error.

    Returns ``(bytes[entries] uint8 ascending, values[entries] float32)``
    where ``values = e4m3(bytes) * global_scale``.

    Two things this deliberately is not.  It is not k-means: continuous
    centroids snapped to E4M3 after each Lloyd step collapse onto one another
    (sixteen centroids became eleven distinct scales on a GLM expert, a 3.4%
    loss), because the snap is not part of the objective Lloyd minimises.  And
    it is not a rounder: the objective is exact per assignment, on the finite
    grid the wire can actually store.

    Greedy backward elimination from every in-range E4M3 value: each round
    removes the entry whose loss is smallest.  Removing entry ``i`` moves
    exactly the targets assigned to it, each to the nearer of its two
    neighbours in the sorted table, so the loss of every candidate removal is
    one ``index_add`` over the current assignment rather than a full
    re-evaluation per candidate.  A few swap passes then try each table entry
    against each unused grid value at full cost.
    """
    device = targets.device
    grid_values = e4m3_positive_values(device) * global_scale          # [119]
    live = weights > 0
    if not bool(live.any()):
        # Nothing to fit: a unit of all-zero halves.  Any table decodes it.
        first_byte = E4M3_NORMAL_BYTES[0]
        return (
            torch.arange(first_byte, first_byte + entries, dtype=torch.uint8, device=device),
            grid_values[:entries],
        )
    s, w = targets[live], weights[live]
    lo, hi = float(s.min()), float(s.max())
    # The grid values bracketing [lo, hi], one step wider each side, and never
    # fewer than ``entries`` candidates: a unit whose targets span less than
    # two octaves has fewer in-range E4M3 values than the table holds.
    first = max(int((grid_values < lo).sum()) - 1, 0)
    last = min(int((grid_values <= hi).sum()) + 1, grid_values.numel())
    while last - first < entries:
        if first > 0:
            first -= 1
        if last - first < entries and last < grid_values.numel():
            last += 1
    # Grid index -> byte: entry ``i`` of ``e4m3_positive_values`` is byte
    # ``i + FIRST``; the swap loop below inverts it the same way.
    FIRST = E4M3_NORMAL_BYTES[0]
    candidate_bytes = torch.arange(first + FIRST, last + FIRST, dtype=torch.long, device=device)
    table = grid_values[first:last]

    while table.numel() > entries:
        assign = _nearest(s, table)
        left = table[(assign - 1).clamp_min(0)]
        right = table[(assign + 1).clamp_max(table.numel() - 1)]
        left_gap = torch.where(assign > 0, (s - left).abs(), torch.full_like(s, float("inf")))
        right_gap = torch.where(
            assign < table.numel() - 1, (s - right).abs(), torch.full_like(s, float("inf"))
        )
        here = (s - table[assign]).abs()
        alt = torch.minimum(left_gap, right_gap)
        loss = torch.zeros(table.numel(), device=device, dtype=s.dtype).index_add_(
            0, assign, w * (alt * alt - here * here)
        )
        drop = int(loss.argmin())
        keep = torch.ones(table.numel(), dtype=torch.bool, device=device)
        keep[drop] = False
        table, candidate_bytes = table[keep], candidate_bytes[keep]

    for _ in range(swaps):
        improved = False
        base = float(_lut_cost(s, w, table))
        all_bytes = torch.arange(first + FIRST, last + FIRST, dtype=torch.long, device=device)
        unused = all_bytes[~torch.isin(all_bytes, candidate_bytes)]
        for i in range(table.numel()):
            for byte in unused.tolist():
                trial = table.clone()
                trial[i] = grid_values[byte - FIRST]
                cost = float(_lut_cost(s, w, trial))
                if cost < base * (1.0 - 1e-9):
                    table, base, improved = trial, cost, True
                    candidate_bytes = candidate_bytes.clone()
                    candidate_bytes[i] = byte
                    unused = all_bytes[~torch.isin(all_bytes, candidate_bytes)]
        if not improved:
            break
    order = torch.argsort(candidate_bytes)
    return candidate_bytes[order].to(torch.uint8), table[order]


def _lut_values(table_bytes: torch.Tensor, global_scale: float) -> torch.Tensor:
    """E4M3 bytes -> scales, exactly as ``scales_from_lut`` reads them."""
    return table_bytes.view(torch.float8_e4m3fn).float() * global_scale


def _pack_scales_lut(
    weights: torch.Tensor, half: int, peak: float = 6.0, headroom: float = 1.0,
    entries: int = LUT_ENTRIES,
):
    """The LUT plane's starting point: amax targets, energy weights.

    Returns ``(table_bytes[entries], index[halves] uint8, effective[halves],
    global_scale)``.  The global is a power of two placing the largest target
    in E4M3's seventh binade from the top, so the table has headroom above
    (the least-squares refit can raise a scale past its amax) and seventeen
    binades below.  Weighting each half by the energy of its normalised
    weights approximates the ``<u, u>`` the refit will use once codes exist.
    """
    flat = weights.reshape(-1)
    halves = flat.reshape(-1, half)
    amax_half = halves.abs().amax(dim=1).clamp_min(1e-30)
    target = amax_half / (peak * headroom)
    energy = ((halves / target[:, None]) ** 2).sum(dim=1)
    global_scale = float(2.0 ** (torch.floor(torch.log2(target.max())).item() - 6.0))
    table_bytes, table = _fit_lut(target, energy, global_scale, entries)
    index = _nearest(target, table)
    return table_bytes, index.to(torch.uint8), table[index], global_scale


def _refit_scales_lut(
    work: torch.Tensor,
    units: torch.Tensor,
    half: int,
    table_bytes: torch.Tensor,
    index: torch.Tensor,
    effective: torch.Tensor,
    global_scale: float,
):
    """One least-squares step on the LUT plane, monotone by construction.

    The per-half optimum ``s* = <w, u> / <u, u>`` is the same as in
    ``_refit_scales``; what differs is where it lands.  Two tables are tried:
    the one the unit has, re-assigned nearest-in-linear (which cannot cost
    more than the current assignment), and a fresh ``_fit_lut`` on the new
    targets.  The lower weighted cost wins, so a greedy fit that happens to be
    worse than the table it would replace is never taken -- without this the
    alternation with the trellis could oscillate.
    """
    W = work.float().reshape(-1, half)
    U = units.float().reshape(-1, half)
    A = (U * U).sum(dim=1)
    B = (W * U).sum(dim=1)
    # The assignment rule.  A half with ``B <= 0`` -- codes that anti-correlate
    # with its weights -- has its exact minimiser at the smallest positive
    # scale, but handing it a collapsed scale is not free: the trellis runs on
    # ``work / scale`` (a per-half normalised objective, see ``encode_unit``),
    # so the next pass would be forced to spend the column's shared path on
    # that half's enormous normalised residual.  Measured: the alternation
    # stops being monotone in true SSE.  Such a half keeps its scale.
    valid = (A > 0) & (B > 0)
    targets = torch.where(valid, B / A.clamp_min(1e-30), effective)
    weights = torch.where(valid, A, torch.zeros_like(A))

    # The accept rule, exact.  ``_lut_cost`` is what the fit optimises -- a
    # weighted distance that equals the per-half parabola ``A c^2 - 2 B c`` up
    # to a constant only where ``valid``.  The decision between the two tables
    # is made on the parabola itself, over every half, so a re-assignment of
    # a held half under the new table is charged at its true cost and the
    # step is monotone with no hole: under the old table each valid half
    # lands on its exact minimiser and each held half on the entry it holds.
    def exact_cost(table: torch.Tensor) -> "tuple[torch.Tensor, float]":
        index = _nearest(targets, table)
        c = table[index]
        return index, float((A * c * c - 2.0 * B * c).sum())

    old_table = _lut_values(table_bytes, global_scale)
    new_bytes, new_table = _fit_lut(targets, weights, global_scale, table_bytes.numel())
    old_index, old_cost = exact_cost(old_table)
    new_index, new_cost = exact_cost(new_table)
    if new_cost < old_cost:
        table_bytes, table, index = new_bytes, new_table, new_index
    else:
        table, index = old_table, old_index
    return table_bytes, index.to(torch.uint8), table[index]


def encode_unit(
    weights: torch.Tensor,
    forest: "AnchorForest | dict[int, AnchorForest]",
    rates: "tuple[int, ...]",
    code: ConvCode = ConvCode(),
    rotation: RotationState = RotationState.NONE,
    with_diagonals: bool = False,
    diagonals: "Diagonals | None" = None,
    completion: int | None = None,
    released_positions: int = 0,
    group: int = 32,
    half: int = 16,
    scale_headroom: float = 1.0,
    superblock: int = 256,
    scale_refit: int = 4,
    span: int = 1,
    scale_plane: ScalePlaneKind = ScalePlaneKind.S6B,
    trellis_weighting: str = "none",
    body: BodyKind = BodyKind.TCQ,
    window_bits: int = 0,
    window_seed: int = 0,
    window_sigma: "float | None" = None,
    channel_sigma: "float | None" = None,
    ldl: "torch.Tensor | None" = None,
    ldl_block: int = 32,   # DEFAULT_LDLQ_BLOCK in export.py; kept literal to avoid a cycle
    refit_metric: "torch.Tensor | None" = None,
    refit_reach_floor: bool = False,
) -> EncodedUnit:
    """Encode one Linear.  ``weights`` is ``[rows, cols]`` in the source dtype.

    ``scale_plane=CHANNEL`` (schema minor 3) sets one scale per output row
    instead of a block plane: rows start at ``channel_sigma`` grid units of
    RMS (``scale_channel.default_channel_sigma`` when ``None``), the window
    table -- or the TCQ forest the caller built -- models that Gaussian, and
    ``scale_refit`` least-squares refits every row to its codes.  Segment 2a
    cannot be fitted under it: the row field *is* the plane.

    ``body`` selects the trellis (schema minor 2): ``TCQ`` is the shaped
    convolutional trellis over the anchor forests; ``WINDOW`` is the bitshift
    trellis over a ``2^window_bits``-entry table (``window_table``,
    ``viterbi_window``), which needs no forest, no completion axis and span
    1.  ``window_seed``/``window_sigma`` parameterise the table and are
    recorded by the exporter so a replay rebuilds the same one.

    ``trellis_weighting`` is the branch-metric weight the Viterbi runs under:
    ``"none"`` minimises the per-half normalised error (the encoder as first
    built), ``"scale"`` weights each code by its half's scale squared so the
    path minimises the true squared error (``viterbi_columns``).  An encoder
    setting, not wire: any decoder reads either.

    ``ldl`` turns the pass into **LDLQ**: the unit-block-lower factor of the
    regularised input Hessian (``compensate.block_ldl``) over this unit's
    columns, under which the encoder quantises column blocks of ``ldl_block``
    from last to first and pushes each block's residual into the blocks not
    yet quantised.  Columns are independent inside the Viterbi, so a block is
    exactly the columns of a whole-matrix pass restricted to that range --
    which is what makes this a scheduling change and not a second encoder.
    The scale plane is shared by every block and refit between passes, so the
    slice-equals-whole property the standalone ``compensate.compensated_targets``
    needs does not arise.  Encoder-side only: no byte of the wire changes, and
    the same decoder reads the result.

    ``refit_metric`` is the error the row-scale refit minimises: ``None`` the
    plain squared error, ``[cols]`` a per-input-column weight (a diagonal
    Hessian, or a power of one), ``[cols, cols]`` the full Hessian's exact
    quadratic (``scale_channel.refit_channel_scale``).  ``refit_reach_floor``
    keeps every row's refit scale high enough that the pass's own target stays
    inside the body's reach.  Both are encoder settings; neither is wire.

    ``span`` is the trellis super-symbol length (``viterbi_columns``) and
    ``scale_plane`` how segment 2b is written; both are wire and both default
    to the per-position trellis over the S6b plane so that every artifact
    built before they existed is reproducible from its source.  The exporter
    sets the shipping defaults (``export.DEFAULT_SPAN``,
    ``export.DEFAULT_SCALE_PLANE``).
    """
    if weights.ndim != 2:
        raise GrammarError(f"expected a 2-D weight, got shape {tuple(weights.shape)}")
    rows, cols = weights.shape
    if len(rates) != cols:
        raise GrammarError(f"{len(rates)} rates for {cols} columns")
    body = BodyKind(body)
    if isinstance(forest, PayloadGrid):
        # A window body has no forests; the grid is all it needs.
        if body is not BodyKind.WINDOW:
            raise GrammarError("a TCQ body needs its anchor forests, not a bare grid")
        grid, forests = forest, {}
    else:
        forests = forest if isinstance(forest, dict) else {forest.rate: forest}
        grid = next(iter(forests.values())).grid
        if any(f.grid != grid for f in forests.values()):
            raise GrammarError("a unit's rate schedule must share one payload grid")
        for present in sorted(set(rates)):
            if present not in forests and body is BodyKind.TCQ:
                raise GrammarError(
                    f"the schedule uses rate {present} but no forest was supplied "
                    f"for it; got forests for {sorted(forests)}"
                )
    device = weights.device
    if body is BodyKind.WINDOW:
        if span != 1:
            raise GrammarError(f"a window body has no super-symbols; span must be 1, got {span}")
        if completion not in (None, 0):
            raise GrammarError(
                "a window body has no completion axis: its table is flat, not a "
                f"forest; got completion={completion}"
            )
        if window_bits < max(rates):
            raise GrammarError(
                f"window_bits {window_bits} cannot hold a rate-{max(rates)} position's bits"
            )
        completion = 0
    elif window_bits:
        raise GrammarError("window_bits is only meaningful under a window body")

    # S5 transforms, outermost first: rotate the input basis, then remove the
    # rank-1 magnitude field, then set scales on what is left.  The order is
    # forced -- fitting diagonals before rotating would fit the rotation's
    # own structure, and setting scales first would price a matrix the body
    # never sees.
    arity = grid.arity
    if rows % arity:
        raise GrammarError(
            f"{rows} rows is not a whole number of arity-{arity} tuples; a "
            "k-tuple code spans k consecutive rows and cannot straddle the edge"
        )

    rotated, rotation_block = apply_rotation(weights, rotation)
    # A caller may supply a fit made on a WIDER matrix than this call sees.
    # ``sv`` is per output row and a row spans every column, so a fit made on a
    # column slice is not the fit a whole-matrix encode would make -- which
    # silently breaks the slice-equals-whole property ``compensate.py`` relies
    # on to be preprocessing rather than surgery.  Passing the whole matrix's
    # fit in is how a compensated encode stays reproducible from its target.
    if diagonals is not None and with_diagonals:
        raise GrammarError(
            "with_diagonals=True fits its own; pass one or the other, not both"
        )
    fitted = diagonals if diagonals is not None else (
        fit_diagonals(rotated) if with_diagonals else None
    )
    work = apply_diagonals(rotated, fitted) if fitted else rotated

    peak = max(abs(v) for v in grid.values)
    scale_plane = ScalePlaneKind(scale_plane)
    table_bytes, global_scale = None, 1.0
    channel_rows = effective_rows = None
    body_reach = None
    if scale_plane is ScalePlaneKind.CHANNEL:
        from .scale_channel import default_channel_sigma, initial_channel_scale

        if fitted is not None:
            raise GrammarError(
                "a CHANNEL scale plane carries its row scale on the DIAG_SV field; "
                "segment 2a diagonals cannot also be fitted under it"
            )
        if channel_sigma is None:
            channel_sigma = default_channel_sigma(grid)
        # The body's reach in grid units, so the initial plane keeps every
        # row's largest weight inside what the trellis can emit
        # (``initial_channel_scale``): the window table's extreme entry, or
        # the largest anchor the forests reach.
        if body is BodyKind.WINDOW:
            reach_sigma = channel_sigma if window_sigma is None else window_sigma
            reach_codes = window_table(
                grid, window_bits, sigma=reach_sigma, seed=window_seed, half=half, device=device,
            )
            reach = float(grid_vector_table(grid, device)[reach_codes.long()].abs().max())
        else:
            reach = max(
                float(grid_vector_table(grid, device)[
                    torch.as_tensor(f.blocks, device=device, dtype=torch.long)].abs().max())
                for f in forests.values()
            )
        body_reach = reach
        channel_rows, effective_rows, global_scale = initial_channel_scale(
            work, channel_sigma, reach=reach,
        )
        base_byte = torch.zeros(0, dtype=torch.uint8, device=device)
        refine = torch.zeros(0, dtype=torch.uint8, device=device)
        effective = None
    elif scale_plane is ScalePlaneKind.LUT:
        table_bytes, refine, effective, global_scale = _pack_scales_lut(
            work, half, peak=peak, headroom=scale_headroom,
        )
        base_byte = torch.zeros(0, dtype=torch.uint8, device=device)
    else:
        base_byte, refine, effective = _pack_scales(
            work, group, half, peak=peak, headroom=scale_headroom,
        )

    # A code covers ``arity`` consecutive rows, so every per-code plane is
    # ``steps`` tall, not ``rows``.  The scale planes stay per-position.
    steps = rows // arity
    if span < 1 or steps % span:
        raise GrammarError(
            f"{steps} trellis positions per column is not a whole number of "
            f"span-{span} super-symbols; pass span=1 for this shape"
        )
    # One code costs R bits -- R + 1 at positions carrying a stored label when
    # span > 1 -- so the body plane is a uint8 only while that fits.  A
    # 1024-code k-tuple grid runs at R=9 and wrapped silently here, decoding
    # to weights worse than zero (rel_err 1.55) with nothing raising.
    body_dtype = torch.uint8 if max(rates) + (1 if span > 1 else 0) <= 8 else torch.int32
    anchors = torch.zeros(steps, cols, dtype=torch.long, device=device)
    body_bits = torch.zeros(steps, cols, dtype=body_dtype, device=device)
    completion_bits = torch.zeros(steps, cols, dtype=torch.long, device=device)
    codes = torch.zeros(steps, cols, dtype=torch.long, device=device)
    vectors = grid_vector_table(grid, device)
    rate_vector = torch.tensor(rates, device=device)
    window_codes = window_vectors = None
    if body is BodyKind.WINDOW:
        # Under a CHANNEL plane the table models the Gaussian the rows were
        # scaled to; under a block plane ``None`` models the amax-bounded
        # source the half's scale delivers.
        table_sigma = window_sigma
        if table_sigma is None and scale_plane is ScalePlaneKind.CHANNEL:
            table_sigma = channel_sigma
        window_codes = window_table(
            grid, window_bits, sigma=table_sigma, seed=window_seed, half=half, device=device,
        )
        window_vectors = vectors[window_codes.long()]           # [2^L, arity]

    def current_scale() -> torch.Tensor:
        # The per-position scale the trellis quantises against, ``[rows,
        # cols]``: a block plane's halves repeated along the row, or a
        # CHANNEL plane's row word broadcast along it.
        if scale_plane is ScalePlaneKind.CHANNEL:
            from .scale_channel import channel_scale_field

            return channel_scale_field(channel_rows, global_scale, rows, cols)
        return torch.repeat_interleave(effective, half).reshape(rows, cols)

    def trellis_pass(
        targets: torch.Tensor,
        weights: "torch.Tensor | None" = None,
        span_cols: "tuple[int, int] | None" = None,
    ) -> float:
        # One Viterbi per rate: columns are independent, so a mixed-rate
        # schedule is a partition of columns and not a harder problem.
        # ``span_cols`` restricts the pass to a half-open range of columns and
        # is the whole of what LDLQ needs from the trellis: because the Viterbi
        # carries no state across columns, encoding a range is bit-identical to
        # the same columns of a full pass over the same targets and scale.
        total = 0.0
        in_range = None
        if span_cols is not None:
            lo, hi = span_cols
            column = torch.arange(cols, device=device)
            in_range = (column >= lo) & (column < hi)
        if body is BodyKind.WINDOW:
            # One table for every rate: a state indexes the same entry
            # whatever width the column's new bits have.
            for present in sorted(set(rates)):
                which = torch.nonzero(
                    (rate_vector == present) if in_range is None
                    else ((rate_vector == present) & in_range)
                ).squeeze(1)
                if which.numel() == 0:
                    continue
                sub = targets[:, which].contiguous()
                sub_w = None if weights is None else weights[:, which].contiguous()
                state, s_ = viterbi_window(sub, window_vectors, window_bits, present, weights=sub_w)
                total += s_
                anchors[:, which] = state
                body_bits[:, which] = (state & ((1 << present) - 1)).to(body_dtype)
                codes[:, which] = window_codes.long()[state]
            return total
        for present in sorted(set(rates)):
            picked = forests[present]
            depth = picked.cap - present
            level = depth if completion is None else min(completion, depth)
            which = torch.nonzero(
                (rate_vector == present) if in_range is None
                else ((rate_vector == present) & in_range)
            ).squeeze(1)
            if which.numel() == 0:
                continue
            sub = targets[:, which].contiguous()
            sub_w = None if weights is None else weights[:, which].contiguous()
            a, b, s_ = viterbi_columns(sub, picked, code, level, span=span, weights=sub_w)
            total += s_
            blocks = torch.tensor(picked.blocks, device=device, dtype=torch.long)
            reachable = blocks[:, :: 1 << (depth - level)]
            per_pos = vectors[reachable][a]              # [steps, n, D, arity]
            want = sub.reshape(steps, arity, -1).permute(0, 2, 1).unsqueeze(2)
            c_bits = ((want - per_pos) ** 2).sum(dim=3).argmin(dim=2)
            anchors[:, which] = a
            body_bits[:, which] = b.to(body_dtype)
            completion_bits[:, which] = c_bits
            codes[:, which] = reachable[a, c_bits]
        return total

    # The amax plane and the trellis are each set without knowledge of the
    # other.  ``scale_refit`` alternates them: re-fit every half's scale to
    # the codes just chosen (``_refit_scales``), then let the trellis choose
    # again for the new plane.  The schedule ENDS on a refit: a refit after
    # the last trellis pass costs no Viterbi and is monotone, and at equal
    # pass count it always beats ending on the trellis (six GLM experts,
    # held-out: T 1.000, TR 1.044, TRTR 1.072, TRTRTRTR 1.084 vs TRTRTRT
    # 1.082).  So ``scale_refit=k`` runs k trellis passes and k refits, and
    # ``scale_refit=0`` is the amax plane, byte for byte.  Each refit is a
    # plane VALUE written in the same S6b bytes: the decoder, the kernel and
    # the profile id are untouched.
    if trellis_weighting not in ("none", "scale"):
        raise GrammarError(f"trellis_weighting must be 'none' or 'scale', got {trellis_weighting!r}")
    # The refit metric and the reach floor are read ONLY by the CHANNEL
    # branch of the refit below, and only when a refit runs at all.  Given
    # either under a block plane or at ``scale_refit=0`` they would be
    # silently dropped and the unit would encode as if the caller had passed
    # nothing -- an activation-aware export would then ship weights-only bytes
    # and raise nothing, which is the whole failure this plumbing exists to
    # prevent.  Refuse instead of ignoring.
    if refit_metric is not None or refit_reach_floor:
        named = "refit_metric" if refit_metric is not None else "refit_reach_floor"
        if scale_plane is not ScalePlaneKind.CHANNEL:
            raise GrammarError(
                f"{named} is read only by the CHANNEL plane's refit; a block "
                f"plane fits its scales to within-row column spans and has no "
                f"row-scale to weight, so this would be silently ignored"
            )
        if scale_refit == 0:
            raise GrammarError(
                f"{named} shapes the scale refit, and scale_refit=0 runs none: "
                f"the amax plane is written byte for byte and the argument "
                f"would be silently ignored"
            )
    if ldl is not None:
        if ldl.shape != (cols, cols):
            raise GrammarError(
                f"the LDL factor is {tuple(ldl.shape)}, expected ({cols}, {cols}) "
                "-- one row and column per input feature of THIS unit"
            )
        if ldl_block < 1:
            raise GrammarError(f"the LDLQ block must be at least one column, got {ldl_block}")
        if cols % ldl_block:
            raise GrammarError(
                f"{cols} input features is not a multiple of the LDLQ block {ldl_block}; "
                "block_ldl refuses the same shape, so the factor and the schedule "
                "cannot both be right"
            )
        # The factor and the schedule must agree on the block size.  block_ldl
        # leaves the identity on its own diagonal blocks, so this catches the
        # dangerous direction: a factor built FINER than the schedule has spent
        # compensation on columns the schedule then quantises together, and the
        # arithmetic stays well-formed while pricing an arm that is neither
        # block size.  (The reverse -- a coarser factor read at a finer block --
        # keeps the identity here and merely compensates less, so it passes.)
        _m = cols // ldl_block
        _diag = torch.diagonal(
            ldl.reshape(_m, ldl_block, _m, ldl_block), dim1=0, dim2=2
        ).permute(2, 0, 1)
        if not torch.allclose(
            _diag,
            torch.eye(ldl_block, dtype=ldl.dtype, device=ldl.device).expand(_m, -1, -1),
            atol=1e-5,
        ):
            raise GrammarError(
                f"the LDL factor's {ldl_block}-column diagonal blocks are not the "
                f"identity: it was not produced by block_ldl at block {ldl_block}. "
                "Pass the ldl_block the factor was built with."
            )
        if scale_plane is not ScalePlaneKind.CHANNEL:
            # A block plane's scale is fit to a within-row span of columns, so
            # a block-sequential pass would fit each span to a target the next
            # block has not produced yet.  The CHANNEL plane has no column
            # axis, which is why LDLQ lands there first.
            raise GrammarError(
                "LDLQ is implemented for the CHANNEL scale plane; a block plane's "
                "per-column-span scales would have to be scheduled with it"
            )
        ldl_factor = ldl.to(device=device, dtype=torch.float32)
        # Descending block starts; only the lowest-index block may be short.
        block_spans = [
            (max(stop - ldl_block, 0), stop) for stop in range(cols, 0, -ldl_block)
        ]
    ldlq_target = None
    for _ in range(max(scale_refit, 1)):
        scale = current_scale()
        weights = None
        if trellis_weighting == "scale":
            # One weight per POSITION (a half is sixteen columns of one row,
            # so every position of a trellis column has its own scale).
            # Normalised to the column's loudest position so the fp32 path
            # costs stay O(1); a per-column constant moves no argmin.
            weights = (scale / scale.amax(dim=0, keepdim=True)) ** 2
        if ldl is None:
            targets = work / scale
            sse = trellis_pass(targets, weights)
        else:
            # LDLQ: quantise column blocks last to first, and push each
            # block's reconstruction residual into the blocks still to come.
            base = work.float()
            ldlq_target = base.clone()
            recon = torch.zeros_like(base)
            targets = torch.empty_like(base)
            sse = 0.0
            for start, stop in block_spans:
                if stop < cols:
                    residual = base[:, stop:] - recon[:, stop:]
                    ldlq_target[:, start:stop] = (
                        base[:, start:stop] + residual @ ldl_factor[stop:, start:stop]
                    )
                # Only this block's columns are read; scaling the whole matrix
                # once per block would cost the pass a factor of cols/block.
                targets[:, start:stop] = ldlq_target[:, start:stop] / scale[:, start:stop]
                sse += trellis_pass(targets, weights, span_cols=(start, stop))
                block_units = (
                    vectors[codes[:, start:stop]].permute(0, 2, 1).reshape(rows, stop - start)
                )
                recon[:, start:stop] = block_units * scale[:, start:stop]
            del base, recon, targets
        if scale_refit == 0:
            break
        units = vectors[codes].permute(0, 2, 1).reshape(rows, cols)
        if scale_plane is ScalePlaneKind.CHANNEL:
            from .scale_channel import refit_channel_scale

            floor = None
            if refit_reach_floor:
                if body_reach is None:
                    raise GrammarError("a reach floor needs the body's reach; none was computed")
                # The target the trellis actually saw this pass -- W without
                # LDLQ, the compensated target with it, which is the one that
                # can walk out past the table's last entry.
                seen = work.float() if ldlq_target is None else ldlq_target
                floor = seen.abs().amax(dim=1) / float(body_reach)
            channel_rows, effective_rows = refit_channel_scale(
                work, units, channel_rows, global_scale,
                metric=refit_metric, floor=floor,
            )
        elif scale_plane is ScalePlaneKind.LUT:
            table_bytes, refine, effective = _refit_scales_lut(
                work, units, half, table_bytes, refine, effective, global_scale
            )
        else:
            base_byte, refine, effective = _refit_scales(
                work, units, group, half, base_byte, refine, effective
            )
    if scale_refit or ldl is not None:
        # The plane moved after the last pass; the codes are unchanged and
        # decode against the new plane.  Report the error in ITS target units
        # so ``sse`` means one thing at every setting -- and under LDLQ that
        # must be the error against the WEIGHT, not against the compensated
        # target the last block was measured on, or the number would flatter
        # every compensated arm by construction.
        scale = current_scale()
        units = vectors[codes].permute(0, 2, 1).reshape(rows, cols)
        sse = float(((work / scale - units) ** 2).sum())

    # Stage B: release, in S9's canonical order -- descending |decoded value|
    # within the superblock, on the PRE-release decode so the decoder can
    # reproduce the order from bytes it already has.
    release_index = torch.zeros(0, dtype=torch.long, device=device)
    release_code = torch.zeros(0, dtype=torch.long, device=device)
    if released_positions and arity > 1:
        raise GrammarError(
            "release is not defined at arity > 1: an override replaces one "
            "position's code, and a k-tuple code has no per-position code to "
            "replace. The k-tuple trellis is what release was the alternative "
            "to -- see docs/measurements/release-vs-tuple-trellis.md."
        )
    if released_positions:
        values = grid_value_table(grid, device)
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
        group=group,
        half=half,
        completion_limit=completion,
        scale_refit=scale_refit,
        span=span,
        scale_plane=scale_plane,
        scale_lut=table_bytes,
        scale_global=global_scale,
        body=body,
        window_bits=window_bits,
        window_codes=window_codes,
        scale_rows=channel_rows,
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
