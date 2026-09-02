"""The fused window Viterbi returns the reference's bytes, not merely its answer.

``encode.viterbi_window`` is the definition of the window body's encoder and
``window_viterbi.viterbi_window_fused`` is the machine that runs it fast.  The
contract between them is *identity*, not tolerance: the same state per
position and the same ``sse`` float.  A trellis is a chain of decisions, so a
last-ulp difference in one branch cost is not a rounding detail -- it can flip
a state, and every state after it, and change the bytes an artifact ships.

These tests hold that line on the axes that can break it:

  * the window ``L`` and rate ``R``, including ``R = L`` (one predecessor
    class) and ``L = 9, R = 7`` (four classes, 128 predecessors), the shape
    ``tests/test_window_body.py`` already drives through ``encode_unit``;
  * arity 1 and 2 -- the arity sum is where a fused multiply-add hides, and
    where the first implementation of this module was one ulp out;
  * the per-position weights, the chunk width, and the odd shapes (one
    column, seven columns, a column count no tile divides);
  * **ties**: a table of few distinct values makes exactly equal path costs
    common, and the reference's ``torch.min(dim=0)`` takes the first minimal
    index.  A kernel that takes the last one agrees on the cost and disagrees
    on the bytes.
"""
import pytest
import torch

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the fused window Viterbi is a CUDA path")

from tessera.encode import viterbi_window                     # noqa: E402
from tessera.errors import GrammarError                       # noqa: E402
from tessera.window_viterbi import fused_available            # noqa: E402

E2M1 = [-6., -4., -3., -2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2., 3., 4., 6., 0.]


def _case(L, R, arity, rows, cols, weighted, chunk, seed, quantised=False):
    """Targets and a table of the shape the encoder really hands the Viterbi."""
    dev = "cuda"
    torch.manual_seed(seed)
    size = 1 << L
    if quantised:                                    # few values: ties are common
        vals = torch.arange(-2.0, 3.0, device=dev)
        targets = torch.randint(-2, 3, (rows, cols), device=dev).float()
        weights = torch.randint(1, 4, (rows, cols), device=dev).float() if weighted else None
    else:
        vals = (torch.linspace(-448.0, 448.0, 256, device=dev) if arity == 1
                else torch.tensor(E2M1, device=dev))
        targets = torch.randn(rows, cols, device=dev) * (vals.abs().max() / 4)
        weights = (torch.rand(rows, cols, device=dev) + 0.1) if weighted else None
    vectors = vals[torch.randint(0, vals.numel(), (size, arity), device=dev)].contiguous()
    return targets, vectors, weights


#: (L, R, arity, rows, cols, weighted, chunk) -- every axis the task names is
#: covered, and the big shapes ride the small windows so the suite stays a
#: suite.  ``rows`` is always a whole number of arity tuples.
CASES = [
    (6, 2, 1, 16, 7, False, 4),
    (6, 2, 2, 16, 512, True, 128),
    (6, 6, 1, 33, 64, False, 16),                    # R = L: one class
    (6, 3, 1, 2048, 4096, False, 512),               # the encoder's own shape
    (6, 4, 2, 128, 4096, False, 512),
    (7, 2, 1, 33, 7, True, 2),
    (8, 3, 1, 33, 512, True, 64),
    (8, 4, 2, 128, 512, False, 512),
    (8, 7, 1, 16, 4096, True, 1024),                 # two classes, 128 fan
    (9, 7, 1, 64, 512, False, 128),                  # the shape test_window_body drives
    (10, 2, 1, 16, 1, False, 512),                   # one column
    (10, 4, 1, 128, 512, False, 128),
    (10, 4, 2, 2048, 16, True, 8),
    (10, 5, 2, 128, 64, True, 7),                    # a chunk no tile divides
    (10, 7, 1, 64, 33, True, 16),
    (12, 3, 1, 16, 4096, False, 1024),
    (12, 4, 1, 128, 512, False, 512),
    (12, 4, 1, 2048, 64, False, 64),                 # 2048 rows
    (12, 5, 2, 64, 128, True, 32),
    (14, 4, 1, 128, 512, False, 256),
    (14, 5, 2, 32, 64, True, 16),
    (14, 7, 1, 16, 7, False, 4),
    (16, 2, 1, 16, 7, True, 3),
    (16, 4, 1, 16, 512, False, 128),
    (16, 5, 2, 16, 33, True, 16),
    (16, 7, 1, 16, 4, False, 1),                     # one column a chunk
    (5, 5, 2, 32, 64, True, 8),
]


@pytest.mark.parametrize("L,R,arity,rows,cols,weighted,chunk", CASES)
def test_fused_is_the_reference_byte_for_byte(L, R, arity, rows, cols, weighted, chunk):
    targets, vectors, weights = _case(L, R, arity, rows, cols, weighted, chunk,
                                      seed=L * 131 + R * 7 + arity)
    ref, sse_ref = viterbi_window(targets, vectors, L, R, weights=weights,
                                  chunk=chunk, impl="reference")
    got, sse = viterbi_window(targets, vectors, L, R, weights=weights,
                              chunk=chunk, impl="fused")
    assert got.shape == (rows // arity, cols) and got.dtype == ref.dtype
    assert torch.equal(got, ref), "the fused path chose a different path"
    assert sse == sse_ref, f"sse {sse!r} is not the reference's {sse_ref!r}"


@pytest.mark.parametrize("L,R,arity,rows,cols,weighted,chunk", [
    (6, 3, 1, 128, 512, True, 128),
    (8, 4, 1, 64, 128, True, 64),
    (10, 5, 2, 64, 64, False, 32),
    (12, 4, 2, 32, 40, True, 7),
])
def test_fused_breaks_ties_where_the_reference_breaks_them(L, R, arity, rows, cols,
                                                           weighted, chunk):
    """Five values and integer targets: equal path costs, in quantity.

    The test first proves the ties are there -- more minimal predecessors than
    classes, in the front the reference itself builds -- so that the identity
    below is evidence about tie-breaking and not about arithmetic that never
    tied.
    """
    targets, vectors, weights = _case(L, R, arity, rows, cols, weighted, chunk,
                                      seed=L + R, quantised=True)
    size, fan, low = 1 << L, 1 << R, (1 << L) >> R
    steps = rows // arity
    x = targets.float().reshape(steps, arity, cols)
    wr = None if weights is None else weights.float().reshape(steps, arity, cols)
    table = vectors.float()
    cost = torch.full((size, cols), float("inf"), device=targets.device)
    cost[0] = 0.0
    ties = 0
    for step in range(min(steps, 6)):
        view = cost.view(fan, low, cols)
        best = view.min(dim=0).values
        finite = torch.isfinite(best)
        ties += int(((view == best.unsqueeze(0)) & finite.unsqueeze(0)).sum()
                    - finite.sum())
        diff = x[step].t().unsqueeze(1) - table.unsqueeze(0)
        diff = diff * diff
        if wr is not None:
            diff = diff * wr[step].t().unsqueeze(1)
        cost = best.repeat_interleave(fan, dim=0) + diff.sum(dim=2).t()
    assert ties > 0, "this case does not tie, so it tests nothing about ties"

    ref, sse_ref = viterbi_window(targets, vectors, L, R, weights=weights,
                                  chunk=chunk, impl="reference")
    got, sse = viterbi_window(targets, vectors, L, R, weights=weights,
                              chunk=chunk, impl="fused")
    assert torch.equal(got, ref) and sse == sse_ref


def test_auto_is_the_fused_path_on_cuda_and_the_reference_on_cpu():
    targets, vectors, weights = _case(8, 3, 1, 64, 96, True, 32, seed=1)
    auto, sse_auto = viterbi_window(targets, vectors, 8, 3, weights=weights, chunk=32)
    fused, sse_fused = viterbi_window(targets, vectors, 8, 3, weights=weights,
                                      chunk=32, impl="fused")
    assert torch.equal(auto, fused) and sse_auto == sse_fused
    assert fused_available()
    # the CPU tensors take the reference whichever way they are asked, and say
    # so when asked for the machine that is not there
    cpu = (targets.cpu(), vectors.cpu(), weights.cpu())
    a, sa = viterbi_window(cpu[0], cpu[1], 8, 3, weights=cpu[2], chunk=32)
    b, sb = viterbi_window(cpu[0], cpu[1], 8, 3, weights=cpu[2], chunk=32,
                           impl="reference")
    assert torch.equal(a, b) and sa == sb
    with pytest.raises(GrammarError, match="CUDA path"):
        viterbi_window(cpu[0], cpu[1], 8, 3, weights=cpu[2], impl="fused")
    with pytest.raises(GrammarError, match="impl"):
        viterbi_window(targets, vectors, 8, 3, impl="triton")


def test_the_fused_path_refuses_what_the_reference_refuses():
    """The grammar is checked before the machine is chosen."""
    z = torch.zeros(4, 1, device="cuda")
    with pytest.raises(GrammarError, match="does not fit"):
        viterbi_window(z, torch.zeros(8, 1, device="cuda"), 3, 4, impl="fused")
    with pytest.raises(GrammarError, match="states"):
        viterbi_window(z, torch.zeros(8, 1, device="cuda"), 4, 2, impl="fused")
    with pytest.raises(GrammarError, match="tuples"):
        viterbi_window(torch.zeros(5, 1, device="cuda"),
                       torch.zeros(8, 2, device="cuda"), 3, 2, impl="fused")


def test_the_fused_path_is_chunk_invariant_and_starts_pinned():
    """Chunking is a memory decision; it may not be a numerical one."""
    targets, vectors, _ = _case(10, 4, 1, 96, 40, False, 7, seed=5)
    a, sa = viterbi_window(targets, vectors, 10, 4, chunk=7, impl="fused")
    b, sb = viterbi_window(targets, vectors, 10, 4, chunk=512, impl="fused")
    assert torch.equal(a, b)
    assert sa == pytest.approx(sb)
    # the start is state 0: the first state is its own last L bits, so its top
    # L-R bits are zero
    assert int(a[0].max()) < (1 << 4)


def test_the_traceback_offset_survives_a_two_gigabyte_predecessor_buffer():
    """The per-column traceback stride is ``steps * 2^(L-R)`` bytes, and a
    chunk of 512 columns multiplies it by 511.

    At the encoder's own working shape -- L=16, R=4, 2048 rows, chunk 512 --
    that product is 4.3e9, so an int32 offset wraps negative somewhere around
    column 256 and the kernel writes before its own allocation.  This case is
    the cheapest shape that crosses 2^31 (511 * 1032 * 4096 = 2.16e9): it
    costs a 2.2 GB buffer and a few seconds, and it fails loudly on an int32
    offset while every other case here passes.
    """
    L, R, arity, rows, cols, chunk = 14, 2, 1, 1032, 512, 512
    assert (chunk - 1) * (rows // arity) * ((1 << L) >> R) > 2 ** 31
    targets, vectors, weights = _case(L, R, arity, rows, cols, False, chunk, seed=17)
    ref, sse_ref = viterbi_window(targets, vectors, L, R, chunk=chunk,
                                  impl="reference")
    got, sse = viterbi_window(targets, vectors, L, R, chunk=chunk, impl="fused")
    assert torch.equal(got, ref) and sse == sse_ref


# ------------------------------------------------- through the real encoder


@pytest.mark.parametrize("grid_name, rate, window, plane", [
    ("E4M3", 4, 10, "CHANNEL"),
    ("E4M3", 5, 12, "CHANNEL"),
    ("E2M1x2", 7, 10, "LUT"),
    ("E2M1x2", 6, 12, "S6B"),
])
def test_the_fused_path_is_the_reference_under_every_plane(grid_name, rate, window, plane,
                                                             monkeypatch):
    """The unit tests above compare the two machines on raw targets; the
    wire runs them under a scale plane, with the CHANNEL plane's rows scaled
    to a modelled sigma and weighted by (s/s.max)^2.  The artifact bytes
    must be the same bytes either way."""
    import tessera.encode as encode_mod
    from tessera.alphabet import E2M1_GRID, E4M3_GRID, tuple_grid
    from tessera.encode import encode_unit
    from tessera.export import build_unit_artifact
    from tessera.manifest import BodyKind, ScalePlaneKind
    from tessera.trellis import ConvCode

    grid = E4M3_GRID if grid_name == "E4M3" else tuple_grid(E2M1_GRID, 2, "coset")
    kind = ScalePlaneKind[plane]
    code = ConvCode(memory=6)
    g = torch.Generator().manual_seed(11)
    w = torch.randn(256, 512, generator=g).cuda()
    rates = tuple([rate] * 512)

    def run(impl):
        original = encode_mod.viterbi_window

        def pinned(*args, **kwargs):
            kwargs["impl"] = impl
            return original(*args, **kwargs)

        monkeypatch.setattr(encode_mod, "viterbi_window", pinned)
        try:
            unit = encode_unit(w, grid, rates, code, body=BodyKind.WINDOW, window_bits=window,
                               scale_plane=kind, scale_refit=1, completion=0)
        finally:
            monkeypatch.setattr(encode_mod, "viterbi_window", original)
        return build_unit_artifact(unit, "unit0", grid, rate * 256, code)[2]

    reference, fused = run("reference"), run("fused")
    assert reference == fused


def test_graph_capture_survives_other_threads_encoding(monkeypatch):
    """Three threads each capture and replay their own graph at once.

    The capture used CUDA's default ``global`` error mode, under which any
    CUDA call from *another* thread while a capture is open -- a second
    worker's allocator call is enough -- faults the capture.  PrismaBuild's
    workers encode units concurrently in one process, so the capture has to
    be ``thread_local`` and serialised: each thread guards its own stream and
    the others run.  The result must still be the reference's bytes, per
    thread.  The workers here compare through ``.cpu()`` (a stream sync) and
    never call ``torch.cuda.synchronize()``: a device-wide sync is refused
    while any capture is open, in every capture mode, and that is the
    contract a threaded caller keeps.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from tessera import window_viterbi

    monkeypatch.setattr(window_viterbi, "_GRAPH", True)   # capture even for small batches
    cases = [(8, 3, 1, 64, 160, True, 32, seed) for seed in range(6)]
    expected = {}
    for L, R, arity, rows, cols, weighted, chunk, seed in cases:
        targets, vectors, weights = _case(L, R, arity, rows, cols, weighted, chunk, seed=seed)
        expected[seed] = (targets, vectors, weights,
                          viterbi_window(targets, vectors, L, R, weights=weights,
                                         chunk=chunk, impl="reference"))
    barrier = threading.Barrier(3)

    def work(seed):
        targets, vectors, weights, (ref, sse_ref) = expected[seed]
        barrier.wait(timeout=60)              # all three capture at the same moment
        for _ in range(3):
            got, sse = viterbi_window(targets, vectors, 8, 3, weights=weights,
                                      chunk=32, impl="fused")
            assert torch.equal(got.cpu(), ref.cpu()) and sse == sse_ref, \
                f"seed {seed} diverged under threads"
        return seed

    with ThreadPoolExecutor(max_workers=3) as pool:
        for batch in (cases[:3], cases[3:]):
            done = list(pool.map(work, [c[-1] for c in batch]))
            assert sorted(done) == sorted(c[-1] for c in batch)


# ---------------------------------------------------------------------------
# The crossover: past it the fast path is slower than the definition, so
# ``auto`` must stop taking it -- while an explicit ``impl="fused"`` still gets
# the machine it asked for.  See ``encode.WINDOW_FUSED_MAX_RATE`` for the
# measurement that fixes the constant.


def test_auto_stops_at_the_crossover_and_fused_asked_for_is_still_honoured(monkeypatch):
    from tessera import encode as enc
    from tessera import window_viterbi

    calls = []
    real = window_viterbi.viterbi_window_fused

    def counted(*a, **k):
        calls.append(a[3])                                  # the rate
        return real(*a, **k)

    monkeypatch.setattr(window_viterbi, "viterbi_window_fused", counted)

    torch.manual_seed(0)
    targets = torch.randn(64, 32, device="cuda")
    vectors = torch.randn(1 << 10, 1, device="cuda")

    below = enc.viterbi_window(targets, vectors, 10, enc.WINDOW_FUSED_MAX_RATE)
    assert calls == [enc.WINDOW_FUSED_MAX_RATE], "auto must take the fused path at the crossover"

    calls.clear()
    above = enc.viterbi_window(targets, vectors, 10, enc.WINDOW_FUSED_MAX_RATE + 1)
    assert calls == [], "auto must not take the fused path above the crossover"

    calls.clear()
    asked = enc.viterbi_window(targets, vectors, 10, enc.WINDOW_FUSED_MAX_RATE + 1,
                               impl="fused")
    assert calls == [enc.WINDOW_FUSED_MAX_RATE + 1], "an explicit fused request is honoured"

    # The dispatch picks the machine, never the answer.
    assert torch.equal(asked[0], above[0])
    assert asked[1] == above[1]
    assert below[0].shape == above[0].shape



def test_the_crossover_is_movable_for_a_box_whose_crossover_differs(monkeypatch):
    from tessera import encode as enc
    from tessera import window_viterbi

    calls = []
    real = window_viterbi.viterbi_window_fused
    monkeypatch.setattr(window_viterbi, "viterbi_window_fused",
                        lambda *a, **k: (calls.append(a[3]), real(*a, **k))[1])
    monkeypatch.setenv("TESSERA_WINDOW_FUSED_MAX_RATE", str(enc.WINDOW_FUSED_MAX_RATE + 1))

    torch.manual_seed(0)
    targets = torch.randn(64, 32, device="cuda")
    vectors = torch.randn(1 << 10, 1, device="cuda")
    enc.viterbi_window(targets, vectors, 10, enc.WINDOW_FUSED_MAX_RATE + 1)
    assert calls == [enc.WINDOW_FUSED_MAX_RATE + 1]

    monkeypatch.setenv("TESSERA_WINDOW_FUSED_MAX_RATE", "eight")
    with pytest.raises(GrammarError):
        enc.viterbi_window(targets, vectors, 10, 4)
