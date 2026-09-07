"""The encoder's per-block path, counted in host synchronisations.

A trellis pass is a chain of small dependent kernels.  Every host read inside
one -- ``.item()``, ``float(tensor)``, ``.tolist()``, a boolean mask's
``nonzero`` -- stops the CPU until the device drains, so the CPU cannot run
ahead and enqueue the next block's work.  The #283 profile measured that as
92 % of self CPU in ``cudaStreamSynchronize`` on a shipping anchor row.

These tests count those reads with torch's own instrument
(``torch.cuda.set_sync_debug_mode``) and distinguish per-trial/per-block reads
from the intended per-pass or per-sweep decisions:

* ``_fit_lut``'s swap refinement decides one accept per *index*, not one per
  *trial*, so its host reads must not scale with the trial count.
* ``_coupled_landing``'s sweep decides one stop test and one move count per
  *sweep*, not per block.
* ``viterbi_window_fused`` returns one ``sse`` float per call, not per chunk,
  regardless of the chunk count.

The bytes are pinned separately, by oracles that re-implement the sequential
semantics these rewrites had to preserve.
"""
import warnings

import pytest
import torch

from tessera import encode
from tessera.encode import _fit_lut, _lut_cost, e4m3_positive_values

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA path")


def sync_ops(fn):
    """``(result, [warning text])`` -- instrumented synchronizing ops ``fn`` performed.

    ``set_sync_debug_mode("warn")`` is torch's own accounting of
    instrumented call sites that synchronize. The prototype does not promise
    exhaustive coverage of CUDA runtime synchronization; full runtime counts
    belong to the profiler evidence, rather than this warning counter.
    """
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            out = fn()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    # ``set_sync_debug_mode`` announces itself once per call ("a prototype
    # feature"); the synchronisations are the messages that name one.
    return out, [str(w.message) for w in seen
                 if "synchronizing CUDA operation" in str(w.message)]


@cuda
def test_the_counter_counts():
    """The instrument before the measurement: a host read must register and a
    pure device op must not, or every bound below is vacuous."""
    _, none = sync_ops(lambda: torch.ones(8, device="cuda").sum() * 2)
    assert none == [], none
    _, one = sync_ops(lambda: float(torch.ones(8, device="cuda").sum()))
    assert len(one) >= 1, "set_sync_debug_mode saw no sync for float(cuda tensor)"


# ---------------------------------------------------------------- _fit_lut

def _fit_case(device, rows=4096, seed=7):
    g = torch.Generator().manual_seed(seed)
    # Targets spanning several octaves, so the candidate bracket is the whole
    # in-range E4M3 grid and the swap loop has ~100 unused bytes per index.
    t = torch.exp(torch.randn(rows, generator=g) * 1.5) * 0.01
    w = torch.rand(rows, generator=g) + 0.25
    return t.to(device), w.to(device)


@cuda
def test_fit_lut_host_reads_do_not_scale_with_the_trials():
    """The refinement scores ``entries x unused`` trials per pass and decides
    at most one accept per index.  A per-trial host read makes the two counts
    equal; the batched fit reads once per pass plus once per accepting index.
    """
    t, w = _fit_case("cuda")
    gs = 2.0 ** -6
    trials = 0
    real = encode._lut_cost

    def counting(*a, **k):
        nonlocal trials
        trials += 1
        return real(*a, **k)

    encode._lut_cost = counting
    try:
        _, syncs = sync_ops(lambda: _fit_lut(t, w, gs))
    finally:
        encode._lut_cost = real
    assert trials > 200, f"the case is too small to separate the two counts: {trials}"
    assert len(syncs) * 4 <= trials, (
        f"{len(syncs)} host syncs for {trials} trial evaluations: the fit is "
        f"still reading the device once per trial")


def _fit_lut_oracle(targets, weights, global_scale, entries=16, swaps=32):
    """``_fit_lut``'s sequential semantics, written out.

    This is the algorithm as it stood before the trials were batched: every
    trial scored on its own and read back to the host, every accept changing
    the base the trials after it are judged against, ``unused`` recomputed
    after each accept and re-read at the top of each index.  It is the
    oracle, not the implementation: if batching the trials changed one
    decision, the tables differ here.
    """
    device = targets.device
    grid_values = e4m3_positive_values(device) * global_scale
    live = weights > 0
    if not bool(live.any()):
        first_byte = int(encode.E4M3_NORMAL_BYTES[0])
        return (torch.arange(first_byte, first_byte + entries, dtype=torch.uint8,
                             device=device), grid_values[:entries])
    s, w = targets[live], weights[live]
    lo, hi = float(s.min()), float(s.max())
    first = max(int((grid_values < lo).sum()) - 1, 0)
    last = min(int((grid_values <= hi).sum()) + 1, grid_values.numel())
    while last - first < entries:
        if first > 0:
            first -= 1
        if last - first < entries and last < grid_values.numel():
            last += 1
    FIRST = int(encode.E4M3_NORMAL_BYTES[0])
    candidate_bytes = torch.arange(first + FIRST, last + FIRST, dtype=torch.long,
                                   device=device)
    table = grid_values[first:last]
    while table.numel() > entries:
        assign = encode._nearest(s, table)
        left = table[(assign - 1).clamp_min(0)]
        right = table[(assign + 1).clamp_max(table.numel() - 1)]
        left_gap = torch.where(assign > 0, (s - left).abs(), torch.full_like(s, float("inf")))
        right_gap = torch.where(assign < table.numel() - 1, (s - right).abs(),
                                torch.full_like(s, float("inf")))
        here = (s - table[assign]).abs()
        alt = torch.minimum(left_gap, right_gap)
        loss = torch.zeros(table.numel(), device=device, dtype=s.dtype).index_add_(
            0, assign, w * (alt * alt - here * here))
        drop = int(loss.argmin())
        keep = torch.ones(table.numel(), dtype=torch.bool, device=device)
        keep[drop] = False
        table, candidate_bytes = table[keep], candidate_bytes[keep]
    for _ in range(swaps):
        improved = False
        base_cost = _lut_cost(s, w, table)
        base = float(base_cost)
        step = (torch.finfo(base_cost.dtype).eps
                if base_cost.is_floating_point() else 0.0)
        all_bytes = torch.arange(first + FIRST, last + FIRST, dtype=torch.long,
                                 device=device)
        unused = all_bytes[~torch.isin(all_bytes, candidate_bytes)]
        for i in range(table.numel()):
            for byte in unused.tolist():
                trial = table.clone()
                trial[i] = grid_values[byte - FIRST]
                cost = float(_lut_cost(s, w, trial))
                if cost < base * (1.0 - step):
                    table, base, improved = trial, cost, True
                    candidate_bytes = candidate_bytes.clone()
                    candidate_bytes[i] = byte
                    unused = all_bytes[~torch.isin(all_bytes, candidate_bytes)]
        if not improved:
            break
    order = torch.argsort(candidate_bytes)
    return candidate_bytes[order].to(torch.uint8), table[order]


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fit_lut_is_the_sequential_oracle(seed, device):
    """Byte for byte and value for value, on both devices the fixtures and the
    campaign use."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("a CUDA path")
    t, w = _fit_case(device, rows=1024, seed=seed)
    gs = 2.0 ** -6
    got_b, got_v = _fit_lut(t, w, gs)
    want_b, want_v = _fit_lut_oracle(t, w, gs)
    assert torch.equal(got_b, want_b)
    assert torch.equal(got_v, want_v)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_the_accept_walk_is_not_an_argmin(device):
    """The refinement's accept test is sequential and its threshold is a
    relative one, so the trial it lands on is NOT the cheapest trial: a second
    trial below the first by less than one ulp of the running cost is rejected
    where a plain ``argmin`` would take it.  The batched fit keeps the walk;
    this is the witness that the two answers really differ."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("a CUDA path")
    costs = torch.tensor([5.0, 5.0 * (1.0 - 1e-8), 9.0], dtype=torch.float64,
                         device=device)
    step = float(torch.finfo(torch.float32).eps)
    idx, base = encode._greedy_accept(costs.tolist(), 100.0, step)
    assert idx == 0 and base == 5.0
    assert int(torch.argmin(costs)) == 1


# ------------------------------------------------------- viterbi_window_fused

def _window_case(L, R, arity, rows, cols, seed=17):
    g = torch.Generator().manual_seed(seed)
    targets = torch.randn(rows, cols, generator=g).cuda()
    vectors = torch.randn(1 << L, arity, generator=g).cuda()
    weights = (torch.rand(rows, cols, generator=g) + 0.5).cuda()
    return targets, vectors, weights


@cuda
def test_the_window_call_syncs_once_whatever_the_chunk_count():
    """``sse`` is one float per call.  Reading it per chunk makes the count a
    function of the column width, which is exactly what a wide batched call
    was supposed to buy back."""
    from tessera.window_viterbi import fused_available, viterbi_window_fused
    if not fused_available():
        pytest.skip("the fused window Viterbi needs triton")
    targets, vectors, weights = _window_case(12, 4, 1, 64, 300)
    # warm the plan and its capture: the first call builds them.
    viterbi_window_fused(targets, vectors, 12, 4, weights=weights, chunk=64)
    _, syncs = sync_ops(lambda: viterbi_window_fused(
        targets, vectors, 12, 4, weights=weights, chunk=64))
    # One for the ``sse`` float this call still returns, and one the plan
    # machinery itself performs -- neither is per chunk, which is the claim.
    assert len(syncs) <= 2, f"{len(syncs)} host syncs over 5 chunks of 64 columns"


@cuda
@pytest.mark.parametrize("cols,chunk", [(300, 64), (129, 32), (64, 512)])
def test_the_window_sse_is_the_reference_float(cols, chunk):
    """The float64 device accumulator adds the same fp32 chunk sums in the
    same order the host accumulator did, so the float is the same float."""
    from tessera.encode import viterbi_window
    from tessera.window_viterbi import fused_available, viterbi_window_fused
    if not fused_available():
        pytest.skip("the fused window Viterbi needs triton")
    targets, vectors, weights = _window_case(12, 4, 1, 64, cols)
    ref, sse_ref = viterbi_window(targets, vectors, 12, 4, weights=weights,
                                  impl="reference", chunk=chunk)
    got, sse = viterbi_window_fused(targets, vectors, 12, 4, weights=weights,
                                    chunk=chunk)
    assert torch.equal(got, ref)
    assert sse == sse_ref


# -------------------------------------------------------- _coupled_landing

def _coupled_case(device, rows=8, nb=4, half=16, seed=3):
    g = torch.Generator().manual_seed(seed)
    cols = nb * half
    W = torch.randn(rows, cols, generator=g).to(device)
    U = (torch.randn(rows, cols, generator=g) * 0.5).to(device)
    Ub = U.reshape(rows, nb, half)
    M = torch.randn(cols, cols, generator=g)
    H = (M @ M.t() / cols + torch.eye(cols)).to(device)
    A = torch.einsum("rbi,bij,rbj->rb", Ub, torch.diagonal(
        H.reshape(nb, half, nb, half), dim1=0, dim2=2).permute(2, 0, 1), Ub)
    table = (e4m3_positive_values(device) * (2.0 ** -6))[40:56]
    C = table[torch.randint(0, table.numel(), (rows, nb), generator=g)].to(device)
    I = torch.zeros(rows, nb, dtype=torch.long, device=device)
    E = W - C.repeat_interleave(half, dim=1) * U
    start = float(((E @ H) * E).sum())
    return W, U, Ub, H, A, C, I, table, half, start


@cuda
def test_coupled_landing_host_reads_are_per_sweep():
    """The sweep decides two host questions -- has the recomputed cost stopped
    improving, and did anything move -- once per sweep.  Asking them per block
    multiplies both by ``nb``."""
    args = _coupled_case("cuda", nb=8)
    out, syncs = sync_ops(lambda: encode._coupled_landing(*args))
    sweeps = out[3]["sweeps"]
    assert sweeps >= 1
    assert len(syncs) <= 2 * sweeps + 2, (
        f"{len(syncs)} host syncs over {sweeps} sweeps of 8 blocks: the sweep "
        f"is still reading the device once per block")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("seed", [3, 11])
def test_coupled_landing_matches_the_branching_oracle(device, seed):
    """The masked unconditional update returns the branch's plane exactly,
    including the sweeps in which some block moves nothing at all."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("a CUDA path")
    args = _coupled_case(device, nb=6, seed=seed)
    C, I, cost, rec = encode._coupled_landing(*args)
    oC, oI, ocost, orec = _coupled_landing_oracle(*args)
    assert torch.equal(C, oC)
    assert torch.equal(I, oI)
    assert cost == ocost
    assert rec == orec


def _coupled_landing_oracle(W, U, Ub, H, A, C, I, table, half, start_cost,
                            row_weight=None):
    """``_coupled_landing`` as it stood: the update guarded by ``take.any()``
    and the move count read per block."""
    rows, nb = C.shape
    C, I = C.clone(), I.clone()
    eps = torch.finfo(torch.float32).eps
    prev = start_cost
    sweeps = moves = 0
    rw = None if row_weight is None else row_weight.reshape(rows, 1)
    kept_C = kept_I = None
    last_moved = 0
    while True:
        E = W - C.repeat_interleave(half, dim=1) * U
        G = E @ H
        now = float(((G * E) if rw is None else (G * E * rw)).sum())
        if sweeps:
            if now >= prev:
                C, I = kept_C, kept_I
                moves -= last_moved
                break
            if prev - now <= eps * prev:
                break
        kept_C, kept_I = C.clone(), I.clone()
        prev = now
        moved = 0
        for b in range(nb):
            lo, hi = b * half, (b + 1) * half
            Ubb = Ub[:, b, :]
            Ab = A[:, b]
            s = C[:, b] + (G[:, lo:hi] * Ubb).sum(dim=1) / Ab.clamp_min(1e-30)
            j = (s[:, None] - table[None, :]).abs().argmin(dim=1)
            new = table[j]
            gain = Ab * ((C[:, b] - s) ** 2 - (new - s) ** 2)
            take = (Ab > 0) & (s > 0) & (gain > 0)
            if bool(take.any()):
                d = torch.where(take, new - C[:, b], torch.zeros_like(new))
                G = G - (d.unsqueeze(1) * Ubb) @ H[lo:hi, :]
                C[:, b] = torch.where(take, new, C[:, b])
                I[:, b] = torch.where(take, j, I[:, b])
                moved += int(take.sum())
        sweeps += 1
        moves += moved
        last_moved = moved
        if moved == 0:
            break
    E = W - C.repeat_interleave(half, dim=1) * U
    q = (E @ H) * E
    final = float(q.sum() if rw is None else (q * rw).sum())
    return C, I, final, {"cost": final, "sweeps": sweeps, "moves": moves}
