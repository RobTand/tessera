"""The CPU window Viterbi returns the reference's bytes (tessera#795).

``encode.viterbi_window`` runs CPU inputs through ``_viterbi_window_cpu``, a
copy-free spelling of the reference torch chain.  The contract is identity:
the same state per position and the same ``sse`` float, for every input.  A
trellis is a chain of decisions, so one ulp in one branch cost or one tie
broken the other way can move a state and every state after it.

The definition is held here as a **verbatim copy** of ``viterbi_window`` as
it stood on master at 4bf7f55cda, before the CPU path existed (it is only ever
called with ``impl="reference"``, so its fused branch never runs).  The new
path is compared with it over a seeded matrix -- arity 1 and 2, window bits
12 and 14, rates 1, 4 and 8, weighted and not, column counts no chunk divides
-- on integer-valued problems, where exactly equal path costs are everywhere
and the first-minimal-index rule decides the bytes, and on continuous ones.
The edge cases drive the inputs on which the tournament minimum must stand
down for ``torch.min`` (NaN, infinities, negative weights, overflow), the
arity above 2, the rates above 8, and the degenerate shapes.
"""
from __future__ import annotations

import math

import pytest
import torch

from tessera import encode
from tessera.errors import GrammarError


# --- verbatim: src/tessera/encode.py viterbi_window at master 4bf7f55cda ----
def viterbi_window(
    targets: torch.Tensor,
    vectors: torch.Tensor,
    window_bits: int,
    rate: int,
    weights: "torch.Tensor | None" = None,
    chunk: int = 512,
    impl: str = "auto",
    want_sse: bool = True,
) -> "tuple[torch.Tensor, float | None]":
    """Exact Viterbi over the bitshift trellis, down every column at once.

    ``targets`` is ``[rows, cols]`` already divided by its scale; ``vectors``
    is ``[2^window_bits, arity]`` -- the table's reconstruction per state.
    Returns ``(state[steps, cols] int64, sse)``, ``steps = rows // arity``.
    ``want_sse=False`` returns ``(state, None)``: the states are the same
    tensor, and the fused path then makes no host round trip at all --
    reading the float is the one sync a Viterbi call has, and the encoder's
    batch driver, which discards the cost, is what asks for this.

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

    ``chunk`` is exact in the states and approximate in the float.  Columns
    are independent, so the returned states are bit-identical at every chunk
    size; ``sse``, however, is accumulated one chunk at a time in fp32 and so
    depends on how the columns were partitioned.  Measured on 32x96 CPU
    targets at L=8, R=2: identical states at chunk 7/16/32/96/512, and
    ``sse`` spread over 3.05e-05 on a total of 910.09 -- about one ulp per
    partial sum.  A figure anyone quotes is recomputed from the codes, not
    read off this return; the equality tests that do read it hold ``chunk``
    fixed, which is also the sense in which the fused path below returns "the
    identical sse float".

    ``impl`` picks the machine, never the answer.  ``"reference"`` is the
    torch chain below -- the definition, and the only path on CPU;
    ``"fused"`` is the Triton step kernel in ``window_viterbi``, which
    returns identical states and the identical sse float (see that module for
    why that is a contract and not a hope); ``"auto"`` takes the fused path
    on CUDA inputs when Triton is present **and the rate is at or below
    ``WINDOW_FUSED_MAX_RATE``**, and the reference otherwise.  ``"fused"``
    asked for explicitly is still honoured above the crossover: the crossover
    governs the choice ``auto`` makes, not what the caller may demand.
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

        wanted = impl == "fused" or rate <= _fused_max_rate()
        if targets.is_cuda and fused_available() and wanted:
            return viterbi_window_fused(targets, vectors, window_bits, rate,
                                        weights=weights, chunk=chunk,
                                        want_sse=want_sse)
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
    return states, (sse if want_sse else None)
# --- end verbatim -------------------------------------------------------------


def _problem(window_bits, arity, steps, cols, weighted, tied, seed):
    g = torch.Generator().manual_seed(seed)
    size = 1 << window_bits
    rows = steps * arity
    if tied:
        # Few distinct integers: every path cost is an exact integer, so
        # equal costs, and therefore ties in the class minimum, are common.
        vectors = torch.randint(-3, 4, (size, arity), generator=g).float()
        targets = torch.randint(-4, 5, (rows, cols), generator=g).float()
        weights = torch.randint(0, 3, (rows, cols), generator=g).float()
    else:
        vectors = torch.randn(size, arity, generator=g)
        targets = torch.randn(rows, cols, generator=g) * 1.5
        weights = torch.rand(rows, cols, generator=g) * 2.0
    return targets, vectors, (weights if weighted else None)


def _same(new, old):
    (new_states, new_sse), (old_states, old_sse) = new, old
    assert new_states.dtype == old_states.dtype == torch.long
    assert torch.equal(new_states, old_states)
    if old_sse is None or new_sse is None:
        assert old_sse is None and new_sse is None
    elif math.isnan(old_sse):
        assert math.isnan(new_sse)
    else:
        assert new_sse == old_sse


def _both(targets, vectors, window_bits, rate, weights=None, chunk=512,
          want_sse=True):
    new = encode.viterbi_window(targets, vectors, window_bits, rate,
                                weights=weights, chunk=chunk, impl="reference",
                                want_sse=want_sse)
    old = viterbi_window(targets, vectors, window_bits, rate, weights=weights,
                         chunk=chunk, impl="reference", want_sse=want_sse)
    return new, old


# Eighteen steps: past L / R for every rate here except rate 1 at L = 14 and
# 12, where the trellis fills at step 14 and 12 and still runs four and six
# full steps beyond it.
STEPS = 18


@pytest.mark.parametrize("window_bits", [12, 14])
@pytest.mark.parametrize("rate", [1, 4, 8])
@pytest.mark.parametrize("arity", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("chunk,cols", [(512, 700), (32, 75)])
@pytest.mark.parametrize("tied", [True, False])
def test_the_cpu_path_is_the_reference_byte_for_byte(window_bits, rate, arity,
                                                     weighted, chunk, cols, tied):
    seed = (window_bits * 1000 + rate * 100 + arity * 10 + weighted
            + 2 * (chunk == 32) + 4 * tied)
    targets, vectors, weights = _problem(window_bits, arity, STEPS, cols,
                                         weighted, tied, seed)
    tuples = targets.float().reshape(STEPS, arity, cols)
    wrows = None if weights is None else weights.reshape(STEPS, arity, cols)
    # These are the inputs the tournament minimum runs on; the edge cases
    # below are the ones it stands down on.
    assert encode._branch_cannot_be_nan(tuples, vectors.float(), wrows)
    _same(*_both(targets, vectors, window_bits, rate, weights, chunk))


@pytest.mark.parametrize("rate", [2, 3, 5, 6])
@pytest.mark.parametrize("arity", [1, 2])
def test_every_tournament_depth_and_the_rate_above_it(rate, arity):
    # Rates 1 to 5 run the tournament (one to five rounds) and 6 is the first
    # rate that hands the minimum back to torch.min.
    targets, vectors, weights = _problem(12, arity, STEPS, 70, True, True,
                                         100 + rate * 10 + arity)
    _same(*_both(targets, vectors, 12, rate, weights, 32))
    _same(*_both(targets, vectors, 12, rate, None, 512))


def test_the_tied_problems_really_tie():
    # The tie matrix above proves the tie-breaking rule only if the class
    # minimum actually meets equal candidates.  Run the reference's first
    # steps by hand and count them.
    window_bits, rate, arity, cols = 12, 4, 1, 64
    targets, vectors, _ = _problem(window_bits, arity, STEPS, cols, False, True, 7)
    size, fan = 1 << window_bits, 1 << rate
    low = size >> rate
    x = targets.reshape(STEPS, arity, cols)
    cost = torch.full((size, cols), float("inf"))
    cost[0] = 0.0
    tied_classes = 0
    for step in range(STEPS):
        view = cost.view(fan, low, cols)
        best = view.min(dim=0).values
        tied_classes += int(((view == best) & torch.isfinite(best)).sum(0).gt(1).sum())
        branch = ((x[step].t().unsqueeze(1) - vectors.unsqueeze(0)) ** 2).sum(2).t()
        cost = best.repeat_interleave(fan, dim=0) + branch
    assert tied_classes > 1000


def _edge(window_bits, arity, steps, cols, seed=11):
    g = torch.Generator().manual_seed(seed)
    targets = torch.randn(steps * arity, cols, generator=g)
    vectors = torch.randn(1 << window_bits, arity, generator=g)
    weights = torch.rand(steps * arity, cols, generator=g) + 0.25
    return targets, vectors, weights


def _poison_target(value):
    def poison(targets, vectors, weights):
        targets[3, 5] = value
        return targets, vectors, weights
    return poison


def _poison_weight(value):
    def poison(targets, vectors, weights):
        weights[2, 1] = value
        weights[7, 4] = value
        return targets, vectors, weights
    return poison


def _negative_weights(targets, vectors, weights):
    return targets, vectors, weights - 0.75


def _signed_zero_weights(targets, vectors, weights):
    weights[::2] = -0.0
    weights[1::4] = 0.0
    return targets, vectors, weights


def _overflowing_targets(targets, vectors, weights):
    # (1e20)^2 overflows float32, and a zero weight then makes inf * 0 = NaN.
    targets[4, 2] = 1e20
    weights[4, 2] = 0.0
    targets[6, 3] = -3e19
    return targets, vectors, weights


def _inf_table_row(targets, vectors, weights):
    vectors[5] = float("inf")
    return targets, vectors, weights


# (name, mutate, expect_safe)
POISONS = [
    ("nan-target", _poison_target(float("nan")), False),
    ("inf-target", _poison_target(float("inf")), False),
    ("-inf-target", _poison_target(float("-inf")), False),
    ("nan-weight", _poison_weight(float("nan")), False),
    ("inf-weight", _poison_weight(float("inf")), False),
    ("negative-weights", _negative_weights, False),
    ("overflow", _overflowing_targets, False),
    ("inf-table", _inf_table_row, False),
    ("signed-zero-weights", _signed_zero_weights, True),
]


@pytest.mark.parametrize("name,mutate,expect_safe", POISONS,
                         ids=[p[0] for p in POISONS])
@pytest.mark.parametrize("arity", [1, 2])
@pytest.mark.parametrize("rate", [1, 3])
def test_non_finite_and_signed_inputs_match_the_reference(name, mutate, expect_safe,
                                                          arity, rate):
    window_bits, steps, cols = 8, 12, 9
    targets, vectors, weights = mutate(*_edge(window_bits, arity, steps, cols))
    wrows = weights.reshape(steps, arity, cols)
    assert encode._branch_cannot_be_nan(
        targets.reshape(steps, arity, cols), vectors, wrows) is expect_safe
    _same(*_both(targets, vectors, window_bits, rate, weights, chunk=4))
    _same(*_both(targets, vectors, window_bits, rate, None, chunk=4))


@pytest.mark.parametrize("window_bits,rate,arity,steps,cols,chunk", [
    (8, 2, 3, 10, 13, 5),        # arity above 2 keeps the reference's sum
    (8, 3, 4, 8, 6, 512),
    (10, 9, 1, 12, 7, 3),        # rates above 8: an int32 traceback, torch.min
    (10, 10, 1, 12, 7, 512),     # R = L above 8
    (6, 6, 1, 10, 11, 4),        # R = L: one low class
    (6, 6, 2, 10, 11, 512),
    (8, 2, 1, 10, 1, 512),       # one column
    (8, 2, 1, 10, 5, 512),       # a chunk wider than the matrix
    (8, 2, 1, 10, 33, 1),        # one column per chunk
])
@pytest.mark.parametrize("weighted", [False, True])
def test_odd_shapes_match_the_reference(window_bits, rate, arity, steps, cols,
                                        chunk, weighted):
    targets, vectors, weights = _edge(window_bits, arity, steps, cols, seed=rate)
    _same(*_both(targets, vectors, window_bits, rate,
                 weights if weighted else None, chunk))


def test_zero_rows_match_the_reference():
    targets = torch.empty(0, 6)
    vectors = torch.randn(256, 1)
    new, old = _both(targets, vectors, 8, 2)
    _same(new, old)
    assert new[0].shape == (0, 6)


def test_without_sse_the_states_match_and_the_float_is_absent():
    targets, vectors, weights = _problem(12, 1, STEPS, 40, True, True, 3)
    new, old = _both(targets, vectors, 12, 4, weights, 16, want_sse=False)
    assert new[1] is None
    _same(new, old)


def test_dtype_and_layout_of_the_inputs_do_not_change_the_bytes():
    # float64 targets and weights, and a non-contiguous target view: both
    # paths cast and reshape the same way.
    targets, vectors, weights = _problem(12, 2, STEPS, 50, True, False, 5)
    wide = torch.empty(targets.shape[0], 2 * targets.shape[1])
    wide[:, ::2] = targets
    _same(*_both(wide[:, ::2], vectors.double(), 12, 4, weights.double(), 32))
    _same(*_both(targets.double(), vectors, 12, 4, weights, 32))


def test_the_refusals_are_the_references():
    vectors = torch.randn(256, 2)
    for args in [(torch.randn(5, 4), vectors, 8, 2),     # odd rows at arity 2
                 (torch.randn(4, 4), vectors, 9, 2),     # wrong table size
                 (torch.randn(4, 4), vectors, 8, 9)]:    # rate past the window
        with pytest.raises(GrammarError):
            encode.viterbi_window(*args, impl="reference")
        with pytest.raises(GrammarError):
            viterbi_window(*args, impl="reference")


@pytest.mark.parametrize("arity", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("impl", ["auto", "reference"])
def test_cpu_inputs_keep_their_device_under_a_meta_default(arity, weighted, impl):
    # Inputs already on CPU must keep the reference's explicit placement,
    # even when a caller constructs unrelated tensors on meta by default.
    targets, vectors, weights = _problem(8, arity, 18, 7, weighted, False, 91)
    with torch.device("meta"):
        old = viterbi_window(targets, vectors, 8, 2, weights, 4,
                             impl="reference")
        new = encode.viterbi_window(targets, vectors, 8, 2, weights, 4,
                                    impl=impl)
        assert old[0].device.type == new[0].device.type == "cpu"
        _same(new, old)


@pytest.mark.parametrize("default_device", ["cpu", "meta"])
@pytest.mark.parametrize("dtype", [
    torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("arity", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
def test_global_float_dtype_preserves_the_reference(
        default_device, dtype, arity, weighted):
    # The old front inherits the default dtype, but each branch cost is
    # computed in float32 before the addition. Widening/narrowing in-place
    # branch buffers changes that arithmetic, so compare the exact SSE too.
    targets, vectors, weights = _problem(8, arity, 18, 7, weighted, False, 91)
    initial = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device(default_device):
            _same(*_both(targets, vectors, 8, 2, weights, 4))
    finally:
        torch.set_default_dtype(initial)
