"""The fused LUT swap passes return the reference's table and bytes, not merely a good table.

``lut_fused.swap_passes_fused`` replaces ``encode._lut_swap_passes_reference``'s
host loop -- a clone, six kernels and a host sync per trial -- with one host
sync a pass (tessera#486 stage 2).  The contract between the two is identity:
the same bytes and the same table floats.  The LUT plane is on the wire, and
every census row's reseal proof re-encodes against stored wires byte for byte.
A swap pass is a chain of decisions, so one trial cost a ulp off near the
accept threshold, or a tie taken by the other trial, changes every decision
after it.

So these hold that line where it can break:

- Every trial cost against torch's own ``_lut_cost``, bit for bit, at sizes on
  both sides of each change in torch's reduction layout: the vectorised minimum,
  tails of 0-3, the warp split and the block split.
- Whole fits against the reference, with exact ties, dead halves, narrow and wide
  brackets, short tables and swap caps.
- A tie the test proves, which must go to the first trial in byte order.
- The dispatch rules, the refusals by name, the tripwire and the tile knob.
"""
import dataclasses
import struct
import threading

import pytest
import torch

import tessera.encode as enc
import tessera.lut_fused as lf
from tessera.errors import GrammarError
from tessera.lut_fused import (SumPlan, fused_available, lut_swap_refusal, position_costs,
                               sum_plan, swap_passes_fused)

cuda = pytest.mark.skipif(not fused_available(),
                          reason="the fused LUT swap passes are a CUDA path and need triton")

FIRST = enc.E4M3_NORMAL_BYTES[0]


def _bits(x) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _unit(n, dist, seed):
    """``(targets, weights)`` of a unit's ``n`` halves, on the GPU."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    if dist == "halves":        # per-half LUT targets and energies over a few octaves
        t = torch.exp(torch.randn(n, device="cuda", generator=g) * 0.7)
        w = torch.exp(torch.randn(n, device="cuda", generator=g) * 1.2)
    elif dist == "wide":        # a bracket as wide as the grid allows
        t = torch.exp(torch.randn(n, device="cuda", generator=g) * 2.0)
        w = torch.rand(n, device="cuda", generator=g) * 50 + 1e-3
    elif dist == "lattice":     # targets on grid-commensurate points, few weight levels
        t = torch.randint(1, 200, (n,), device="cuda", generator=g).float() / 32
        w = torch.randint(1, 4, (n,), device="cuda", generator=g).float()
    else:                       # "dead": a quarter of the halves carry no energy
        t = torch.exp(torch.randn(n, device="cuda", generator=g) * 0.7)
        w = torch.exp(torch.randn(n, device="cuda", generator=g))
        w = torch.where(torch.rand(n, device="cuda", generator=g) < 0.25,
                        torch.zeros_like(w), w)
    return t.contiguous(), w.contiguous()


def _global_scale(t):
    """``_pack_scales_lut``'s global: the binade of the largest target, less 6."""
    return float(2.0 ** (torch.floor(torch.log2(t.max())).item() - 6.0))


def _bracket(t, w, gs, entries):
    """``_fit_lut``'s candidate bracket, ``(first, last, live)``."""
    grid = enc.e4m3_positive_values(t.device) * gs
    s = t[w > 0]
    lo, hi = float(s.min()), float(s.max())
    first = max(int((grid < lo).sum()) - 1, 0)
    last = min(int((grid <= hi).sum()) + 1, grid.numel())
    while last - first < entries:
        if first > 0:
            first -= 1
        if last - first < entries and last < grid.numel():
            last += 1
    return first, last, s.numel()


def assert_identical(ref, got, what=""):
    (b0, t0), (b1, t1) = ref, got
    assert b1.dtype == b0.dtype == torch.uint8, what
    assert t1.dtype == t0.dtype == torch.float32, what
    assert torch.equal(b0, b1), f"{what}: bytes {b0.tolist()} != {b1.tolist()}"
    assert torch.equal(t0.view(torch.int32), t1.view(torch.int32)), f"{what}: table bits differ"


# ---- torch's reduction layout ------------------------------------------------------------

@pytest.mark.parametrize("n,lanes,blocks,step,vectors", [
    (128, 32, 1, 32, 1),            # the smallest vectorised reduction
    (131, 32, 1, 32, 1),            # a three-element tail
    (1000, 128, 1, 128, 2),         # too few values a thread to split across warps
    (130560, 512, 1, 512, 64),      # split across warps, one block
    (130561, 512, 16, 8192, 4),     # the first block split
    (229376, 512, 28, 14336, 4),    # an LFM L18 expert's halves
    (524288, 512, 64, 32768, 4),    # a GLM-5.3 routed expert's halves
    (4194304, 512, 144, 73728, 15),  # the target grid binds
])
def test_the_sum_plan_is_torchs_layout_on_gb10(n, lanes, blocks, step, vectors):
    """``setReduceConfig`` at GB10's properties: 48 multiprocessors, 1536 threads each."""
    assert sum_plan(n, 48, 1536, 32) == SumPlan(n, lanes, blocks, step, vectors)


def test_torch_does_not_vectorise_below_128():
    with pytest.raises(ValueError):
        sum_plan(127, 48, 1536, 32)


# ---- every trial cost is torch's float ---------------------------------------------------

TRIAL_SIZES = [128, 129, 130, 131, 1000, 2051, 130560, 130561, 131072, 229375, 229376,
               229377, 524288, 1048577]


@cuda
@pytest.mark.parametrize("dist", ["halves", "lattice"])
@pytest.mark.parametrize("n", TRIAL_SIZES)
def test_every_trial_cost_is_torchs_sum(n, dist):
    s, w = _unit(n, dist, seed=n)
    grid = enc.e4m3_positive_values("cuda") * _global_scale(s)
    pick = torch.randperm(grid.numel(), generator=torch.Generator().manual_seed(n)).cuda()
    table, values = grid[pick[:16]], grid[pick[16:40]].sort().values
    assert lut_swap_refusal(s, w, table, grid) is None
    sensitive = 0
    for position in (0, 5, 15):
        got = position_costs(s, w, table, values, position)
        for u in range(values.numel()):
            trial = table.clone()
            trial[position] = values[u]
            want = enc._lut_cost(s, w, trial)
            assert _bits(got[u]) == _bits(want), (
                f"position {position} trial {u}: fused {float(got[u])!r} != torch "
                f"{float(want)!r} under {lf._plan(s)}")
            gap = (s[:, None] - trial[None, :]).abs().amin(dim=1)
            sensitive += _bits(torch.cumsum(w * gap * gap, 0)[-1]) != _bits(want)
    if dist == "halves" and n >= 1000:
        assert sensitive, "no cost here depends on the summation order, so the case proves nothing"


# ---- whole fits --------------------------------------------------------------------------

def _fit_matrix():
    for n in (128, 131, 2051, 130561, 229376, 229377, 524288):
        for dist in ("halves", "wide", "lattice", "dead"):
            yield n, dist, 16, 32
    for n in (131, 229376):
        for dist in ("halves", "lattice"):
            for entries, swaps in ((16, 1), (8, 32), (4, 3)):
                yield n, dist, entries, swaps


@cuda
@pytest.mark.parametrize("n,dist,entries,swaps", list(_fit_matrix()))
def test_fused_fit_matches_reference(n, dist, entries, swaps, monkeypatch):
    t, w = _unit(n, dist, seed=n * 5 + entries + swaps)
    gs = _global_scale(t)
    monkeypatch.setenv("TESSERA_LUT_FUSED", "0")
    before = dict(lf.STATS)
    ref = enc._fit_lut(t, w, gs, entries, swaps=swaps)
    assert lf.STATS == before
    monkeypatch.setenv("TESSERA_LUT_FUSED", "1")
    got = enc._fit_lut(t, w, gs, entries, swaps=swaps)
    assert_identical(ref, got, f"n={n} {dist} entries={entries} swaps={swaps}")
    first, last, live = _bracket(t, w, gs, entries)
    answered = live >= 128 and last - first > entries
    assert lf.STATS["fused"] == before["fused"] + int(answered)
    assert lf.STATS["tripped"] == before["tripped"]


@cuda
def test_an_exact_tie_goes_to_the_first_trial_in_byte_order():
    """Two unused values equally near every target: the first in byte order is taken.

    Every target sits at 3.125, midway between the grid's 3.0 (byte 68) and
    3.25 (byte 69), and the running table holds sixteen values far below.  At
    position 0 the trials walk up the bracket, each nearer than the last and
    accepted, to 3.0; then 3.25 costs exactly what 3.0 costs, which is not an
    improvement.  A scan that took ties would end on 3.25.
    """
    grid = enc.e4m3_positive_values("cuda")
    lo = 68 - FIRST
    v1, v2 = grid[lo], grid[lo + 1]
    assert (float(v1), float(v2)) == (3.0, 3.25)
    n = 1000
    s = torch.full((n,), 3.125, device="cuda")
    w = torch.ones(n, device="cuda")
    first, last = lo - 30, lo + 2
    table = grid[first:first + 16].clone()
    cand = torch.arange(first + FIRST, first + 16 + FIRST, device="cuda")
    values = grid[first + 16:last]
    costs = position_costs(s, w, table, values, 0)
    assert _bits(costs[-2]) == _bits(costs[-1]), "the tie is not exact, so it proves nothing"
    ref = enc._lut_swap_passes_reference(s, w, table, cand, grid, first, last, 32)
    got = swap_passes_fused(s, w, table, cand, grid, first, last, 32)
    assert got is not None
    assert torch.equal(ref[1], got[1]) and torch.equal(ref[0].view(torch.int32),
                                                      got[0].view(torch.int32))
    assert 68 in got[1].tolist() and 69 not in got[1].tolist()


@cuda
def test_no_trial_and_no_pass_return_the_arguments():
    s, w = _unit(4096, "halves", seed=1)
    grid = enc.e4m3_positive_values("cuda")
    table = grid[40:56]
    cand = torch.arange(40 + FIRST, 56 + FIRST, device="cuda")
    got = swap_passes_fused(s, w, table, cand, grid, 40, 56, 32)   # bracket == entries
    assert got[0] is table and got[1] is cand
    got = swap_passes_fused(s, w, table, cand, grid, 30, 70, 0)    # no pass
    assert got[0] is table and got[1] is cand


@cuda
def test_concurrent_fits_are_each_the_reference(monkeypatch):
    units = [_unit(229376, "halves", seed=90 + k) for k in range(4)]
    monkeypatch.setenv("TESSERA_LUT_FUSED", "0")
    refs = [enc._fit_lut(t, w, _global_scale(t)) for t, w in units]
    monkeypatch.setenv("TESSERA_LUT_FUSED", "1")
    got = [None] * len(units)

    def run(k):
        with torch.cuda.device(0):
            t, w = units[k]
            got[k] = enc._fit_lut(t, w, _global_scale(t))

    threads = [threading.Thread(target=run, args=(k,)) for k in range(len(units))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    for k, (ref, g) in enumerate(zip(refs, got)):
        assert_identical(ref, g, f"unit {k}")


# ---- dispatch, refusals, tripwire, knobs -------------------------------------------------

@cuda
def test_the_env_rule(monkeypatch):
    t, w = _unit(229376, "halves", seed=3)
    gs = _global_scale(t)
    monkeypatch.setenv("TESSERA_LUT_FUSED", "0")
    before = dict(lf.STATS)
    enc._fit_lut(t, w, gs)
    assert lf.STATS == before
    monkeypatch.setenv("TESSERA_LUT_FUSED", "yes")
    with pytest.raises(GrammarError, match="TESSERA_LUT_FUSED"):
        enc._fit_lut(t, w, gs)


@cuda
def test_a_scripted_cost_drives_the_reference_loop(monkeypatch):
    """The fused passes reproduce ``_lut_cost``; a replaced cost is not theirs to reproduce."""
    t, w = _unit(229376, "halves", seed=4)
    calls = []
    real = enc._lut_cost

    def counting(*args):
        calls.append(1)
        return real(*args)

    monkeypatch.setattr(enc, "_lut_cost", counting)
    before = dict(lf.STATS)
    enc._fit_lut(t, w, _global_scale(t))
    assert lf.STATS == before and calls


def test_refusals_name_their_reason():
    cpu = torch.ones(200)
    assert lut_swap_refusal(cpu, cpu, cpu[:16], cpu[:119]) == "targets are on cpu"
    if not fused_available():
        return
    s, w = _unit(4096, "halves", seed=5)
    grid = enc.e4m3_positive_values("cuda")
    table = grid[40:56]
    assert "127 targets" in lut_swap_refusal(s[:127], w[:127], table, grid)
    assert "float64" in lut_swap_refusal(s.double(), w.double(), table, grid)
    assert "not one [n]" in lut_swap_refusal(s, w[:-1], table, grid)
    assert "grid are on cpu" in lut_swap_refusal(s, w, table, grid.cpu())
    assert lut_swap_refusal(s, w, table, grid) is None


@cuda
def test_another_allocator_backend_is_refused(monkeypatch):
    """Torch reduces a product's unaligned head apart; the native allocator leaves none."""
    s, w = _unit(4096, "halves", seed=5)
    grid = enc.e4m3_positive_values("cuda")
    assert torch.cuda.get_allocator_backend() == "native"
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "cudaMallocAsync")
    assert "cudaMallocAsync" in lut_swap_refusal(s, w, grid[40:56], grid)


@cuda
def test_the_tripwire_hands_a_wrong_replica_back_to_the_reference(monkeypatch):
    """A replica summing in another order must not answer a fit, only cost one."""
    t, w = _unit(229376, "halves", seed=6)
    gs = _global_scale(t)
    monkeypatch.setenv("TESSERA_LUT_FUSED", "0")
    ref = enc._fit_lut(t, w, gs)
    right = lf._plan(t)
    # The same sum without the block split: every element present, another order.
    wrong = dataclasses.replace(right, blocks=1, step=right.lanes,
                                vectors=-(-(right.n // 4) // right.lanes))
    monkeypatch.setattr(lf, "_plan", lambda targets: wrong)
    monkeypatch.setitem(lf.STATS, "tripped", 0)
    monkeypatch.setenv("TESSERA_LUT_FUSED", "1")
    fused_before = lf.STATS["fused"]
    with pytest.warns(RuntimeWarning, match="differed from torch"):
        got = enc._fit_lut(t, w, gs)
    assert_identical(ref, got, "after the tripwire")
    assert lf.STATS["tripped"] == 1 and lf.STATS["fused"] == fused_before


@cuda
def test_a_non_finite_weight_takes_the_reference(monkeypatch):
    t, w = _unit(229376, "halves", seed=7)
    w[12345] = float("inf")
    gs = _global_scale(t)
    monkeypatch.setenv("TESSERA_LUT_FUSED", "0")
    ref = enc._fit_lut(t, w, gs)
    monkeypatch.setenv("TESSERA_LUT_FUSED", "1")
    before = dict(lf.STATS)
    got = enc._fit_lut(t, w, gs)
    assert_identical(ref, got, "inf weight")
    assert lf.STATS["nonfinite"] == before["nonfinite"] + 1
    assert lf.STATS["fused"] == before["fused"]


@cuda
@pytest.mark.parametrize("tile", ["1", "2,1", "8,2", "64,8"])
def test_every_tile_writes_the_same_costs(tile, monkeypatch):
    s, w = _unit(229377, "halves", seed=8)
    grid = enc.e4m3_positive_values("cuda") * _global_scale(s)
    pick = torch.randperm(grid.numel(), generator=torch.Generator().manual_seed(8)).cuda()
    table, values = grid[pick[:16]], grid[pick[16:59]].sort().values
    monkeypatch.delenv("TESSERA_LUT_FUSED_TILE", raising=False)
    want = position_costs(s, w, table, values, 9)
    monkeypatch.setenv("TESSERA_LUT_FUSED_TILE", tile)
    got = position_costs(s, w, table, values, 9)
    assert torch.equal(want.view(torch.int32), got.view(torch.int32))


def test_a_junk_tile_is_refused(monkeypatch):
    monkeypatch.setenv("TESSERA_LUT_FUSED_TILE", "3")
    with pytest.raises(ValueError, match="powers of two"):
        lf._resolve_tile()
