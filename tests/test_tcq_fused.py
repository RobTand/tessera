"""The fused TCQ trellis returns the reference's bytes, not merely its answer.

``tcq_fused.viterbi_columns_fused`` replaces ``_TCQPlan.run``'s forty-odd
kernels a super-step with three Triton launches a call (tessera#486).  The
contract between the two is *identity*: the anchors, the body field and the
``sse`` float.  A trellis is a chain of decisions, so a last-ulp difference in
one branch cost, or the other index winning one exact tie, flips a state and
every state after it and changes the bytes an artifact ships -- and every
census row's re-seal proof re-encodes against stored wires byte for byte.

So these hold that line on every axis that can break it: every rate and
completion depth of both E2M1-family grids, span 1/2/3, arity 1 and 2, the
weighted branch metric, every admitted dtype, odd column counts, exact ties
that are *proved* to occur, the launch and tile knobs, and the torch
primitives whose semantics the kernels reproduce.
"""
import threading

import pytest
import torch

from tessera.alphabet import SERIALISABLE_GRIDS
from tessera.encode import (_descendant_values, _subset_table, _tcq_maps,
                            _transition_tables, build_forest, tcq_plan_cache_clear,
                            viterbi_columns)
from tessera.errors import GrammarError
from tessera.tcq_fused import fused_available, tcq_fused_refusal
from tessera.trellis import TCQ, ConvCode

pytestmark = pytest.mark.skipif(
    not fused_available(), reason="the fused TCQ trellis is a CUDA path and needs triton")

CODE = ConvCode()


def grid(name):
    return [g for g in SERIALISABLE_GRIDS.values() if g.name == name][0]


def forest_for(name, rate):
    return build_forest(rate, grid=grid(name))


def both(targets, forest, completion, span, weights=None):
    tcq_plan_cache_clear()
    ref = viterbi_columns(targets, forest, CODE, completion, span=span, weights=weights,
                          impl="reference")
    got = viterbi_columns(targets, forest, CODE, completion, span=span, weights=weights,
                          impl="fused")
    return ref, got


def assert_identical(ref, got, what=""):
    (a0, b0, s0), (a1, b1, s1) = ref, got
    assert a1.dtype == torch.long and b1.dtype == torch.long, what
    assert a1.shape == a0.shape and b1.shape == b0.shape, what
    assert torch.equal(a0, a1), f"{what}: anchors differ at {int((a0 != a1).sum())} positions"
    assert torch.equal(b0, b1), f"{what}: body field differs at {int((b0 != b1).sum())} positions"
    assert type(s1) is float, what
    assert s0 == s1, f"{what}: sse {s0!r} != {s1!r}"


def _matrix():
    """Every (grid, rate, completion) the E2M1 family can ask for.

    E2M1 is arity 1 with a rate cap of 3; E2M1x2 is arity 2 with a cap of 7.
    Every rate is taken, and at each rate the shallowest and the deepest
    completion -- the depth changes the descendant loop's length, which is
    the kernel's only completion-dependent shape.
    """
    cases = []
    for name, cap in (("E2M1", 3), ("E2M1x2", 7)):
        for rate in range(1, cap + 1):
            for completion in sorted({0, cap - rate, (cap - rate) // 2}):
                cases.append((name, rate, completion))
    return cases


@pytest.mark.parametrize("weighted", [False, True], ids=["plain", "weighted"])
@pytest.mark.parametrize("span", [1, 2, 3])
@pytest.mark.parametrize("name,rate,completion", _matrix())
def test_fused_matches_reference(name, rate, completion, span, weighted):
    forest = forest_for(name, rate)
    arity = grid(name).arity
    steps, cols = 36, 37                     # past the pinned start's inf phase; odd width
    g = torch.Generator().manual_seed(rate * 1000 + completion * 10 + span)
    targets = (torch.randn(steps * arity, cols, generator=g) * 2.0).cuda()
    weights = (torch.rand(steps * arity, cols, generator=g) + 0.5).cuda() if weighted else None
    ref, got = both(targets, forest, completion, span, weights)
    assert_identical(ref, got, f"{name} R{rate} c{completion} L{span} w={weighted}")


@pytest.mark.parametrize("cols", [1, 5, 64, 255])
@pytest.mark.parametrize("name,rate,span", [("E2M1x2", 7, 2), ("E2M1", 3, 1)])
def test_fused_matches_reference_at_every_width(name, rate, span, cols):
    forest = forest_for(name, rate)
    arity = grid(name).arity
    torch.manual_seed(cols)
    targets = torch.randn(48 * arity, cols, device="cuda")
    weights = torch.rand(48 * arity, cols, device="cuda") + 0.5
    ref, got = both(targets, forest, 0, span, weights)
    assert_identical(ref, got, f"{name} cols={cols}")


@pytest.mark.parametrize("wdtype", [None, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tdtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("name,rate,span,completion", [("E2M1x2", 7, 2, 0), ("E2M1", 2, 3, 1)])
def test_fused_matches_reference_in_every_admitted_dtype(name, rate, span, completion,
                                                         tdtype, wdtype):
    """float16 and bfloat16 inputs are converted exactly and meet a float32
    front on both paths; the reference's own promotion is what is matched."""
    forest = forest_for(name, rate)
    arity = grid(name).arity
    g = torch.Generator().manual_seed(17)
    targets = (torch.randn(36 * arity, 29, generator=g) * 2.0).to(tdtype).cuda()
    weights = (None if wdtype is None else
               (torch.rand(36 * arity, 29, generator=g) + 0.5).to(wdtype).cuda())
    ref, got = both(targets, forest, completion, span, weights)
    assert_identical(ref, got, f"{tdtype} targets, {wdtype} weights")


def test_the_campaign_call_shape_matches_reference_and_graph():
    """One GLM routed ``down_proj`` LDLQ call: 4096 rows at arity 2 is 1024
    super-steps of span 2, over eight units' 32-column blocks joined."""
    forest = forest_for("E2M1x2", 7)
    torch.manual_seed(486)
    targets = torch.randn(4096, 256, device="cuda") * 3.0
    weights = torch.rand(4096, 256, device="cuda") + 0.5
    ref, got = both(targets, forest, 0, 2, weights)
    assert_identical(ref, got, "campaign shape vs reference")
    tcq_plan_cache_clear()
    for _ in range(2):                                   # capture, then replay
        graph = viterbi_columns(targets, forest, CODE, 0, span=2, weights=weights,
                                impl="graph")
    assert_identical(graph, got, "campaign shape vs graph")


def _tie_counts(targets, forest, completion, span, weights=None):
    """How many of the reference forward pass's decisions were exact ties.

    A tie test that never ties proves nothing, so this re-runs the forward
    recurrence with the reference's ops and counts the minima whose minimal
    value occurs at more than one index: points per subset, fold labels, and
    branches -- finite ties separately from the all-``inf`` states of the
    pinned start.  It decides nothing the assertions compare; it only proves
    the inputs exercise the tie rule.
    """
    device = targets.device
    arity = forest.grid.arity
    rows, cols = targets.shape
    steps = rows // arity
    dvals = _descendant_values(forest, completion, device)
    subsets = _subset_table(TCQ(forest, CODE), device)
    prev, subset_of = _transition_tables(CODE, device)
    points = subsets.shape[1]
    roll = torch.arange(4, device=device)
    tuples = targets.float().reshape(steps, arity, cols)
    wrows = None if weights is None else weights.float().reshape(steps, arity, cols)
    cost = torch.full((cols, CODE.states), float("inf"), device=device)
    cost[:, 0] = 0.0
    counts = dict(point=0, fold=0, branch_finite=0, branch_inf=0)

    def ties(values, dim):
        low = values.min(dim=dim, keepdim=True).values
        return int(((values == low).sum(dim=dim) > 1).sum())

    for sup in range(steps // span):
        acc = None
        for offset in range(span):
            step = sup * span + offset
            sq = (tuples[step].t().reshape(cols, 1, 1, arity) - dvals.unsqueeze(0)) ** 2
            if wrows is not None:
                sq = sq * wrows[step].t().reshape(cols, 1, 1, arity)
            by_subset = sq.sum(dim=3).amin(dim=2)[:, subsets.reshape(-1)].reshape(cols, 4, points)
            counts["point"] += ties(by_subset, 2)
            best = by_subset.min(dim=2).values
            if acc is None:
                acc = best
                continue
            terms = torch.stack([acc[:, (roll - v) % 4] + best[:, v:v + 1] for v in range(4)], dim=2)
            counts["fold"] += ties(terms, 2)
            acc = terms.min(dim=2).values
        branch = torch.stack([cost[:, prev[side]] + acc[:, subset_of[side]] for side in (0, 1)])
        equal = branch[0] == branch[1]
        counts["branch_finite"] += int((equal & torch.isfinite(branch[0])).sum())
        counts["branch_inf"] += int((equal & torch.isinf(branch[0])).sum())
        cost = branch.min(dim=0).values
    return counts


def _tie_targets(name, rows, cols, seed):
    """Targets on the grid's own values and their midpoints, and a zero band.

    E2M1's magnitudes are multiples of a half, so every midpoint is exact in
    float32 and sits at exactly equal distance from two grid values -- and the
    grid carries +0 and -0 as two codes of one value.  Exact equal costs are
    therefore everywhere, which is the point.
    """
    values = sorted({abs(v) for v in grid("E2M1").values})
    levels = sorted({s * (a + b) / 2 for a in values for b in values for s in (1, -1)})
    g = torch.Generator().manual_seed(seed)
    pick = torch.randint(0, len(levels), (rows, cols), generator=g)
    targets = torch.tensor(levels)[pick]
    targets[:, : max(1, cols // 8)] = 0.0
    return targets.cuda()


@pytest.mark.parametrize("weighted", [False, True], ids=["plain", "weighted"])
@pytest.mark.parametrize("name,rate,span,completion", [
    ("E2M1x2", 7, 2, 0), ("E2M1x2", 5, 3, 2), ("E2M1x2", 3, 1, 4),
    ("E2M1", 3, 1, 0), ("E2M1", 2, 2, 1),
])
def test_exact_ties_break_to_the_first_index(name, rate, span, completion, weighted):
    forest = forest_for(name, rate)
    arity = grid(name).arity
    rows, cols = 48 * arity, 64
    targets = _tie_targets(name, rows, cols, seed=rate + span)
    # Weights of a power of two keep every product exact, so the ties survive.
    weights = (torch.full((rows, cols), 4.0, device="cuda") if weighted else None)
    counts = _tie_counts(targets, forest, completion, span, weights)
    assert counts["point"] > 0, counts
    assert counts["branch_finite"] > 0, counts
    assert counts["branch_inf"] > 0, counts
    if span > 1:
        assert counts["fold"] > 0, counts
    ref, got = both(targets, forest, completion, span, weights)
    assert_identical(ref, got, f"ties {name} R{rate} L{span} {counts}")


@pytest.mark.parametrize("env", [
    {"TESSERA_TCQ_FUSED_SUPERS": "1"},
    {"TESSERA_TCQ_FUSED_SUPERS": "5"},
    {"TESSERA_TCQ_FUSED_TILE": "1,1,1;1,1;1,1"},
    {"TESSERA_TCQ_FUSED_TILE": "64,16,8;64,8;256,4"},
    {"TESSERA_TCQ_FUSED_UNROLL": "1"},
], ids=lambda env: ",".join(f"{k.rsplit('_', 1)[-1]}={v}" for k, v in env.items()))
def test_launch_and_tile_knobs_never_move_a_byte(monkeypatch, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    for name, rate, span in (("E2M1x2", 7, 2), ("E2M1", 2, 3)):
        forest = forest_for(name, rate)
        arity = grid(name).arity
        torch.manual_seed(rate)
        targets = torch.randn(36 * arity, 21, device="cuda")
        weights = torch.rand(36 * arity, 21, device="cuda") + 0.5
        ref, got = both(targets, forest, 0, span, weights)
        assert_identical(ref, got, f"{env} {name}")


@pytest.mark.parametrize("rows,cols", [(0, 5), (8, 0), (0, 0)])
def test_empty_calls_match_the_reference(rows, cols):
    forest = forest_for("E2M1x2", 7)
    targets = torch.randn(rows, cols, device="cuda")
    ref, got = both(targets, forest, 0, 2)
    assert_identical(ref, got, f"{rows}x{cols}")


def test_auto_takes_the_fused_path_and_zero_restores_the_graph_rule(monkeypatch):
    from tessera import tcq_fused

    monkeypatch.delenv("TESSERA_TCQ_FUSED", raising=False)
    monkeypatch.delenv("TESSERA_TCQ_GRAPH", raising=False)
    calls = []
    real = tcq_fused.viterbi_columns_fused

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(tcq_fused, "viterbi_columns_fused", spy)
    forest = forest_for("E2M1x2", 7)
    torch.manual_seed(5)
    targets = torch.randn(64, 32, device="cuda")
    tcq_plan_cache_clear()
    ref = viterbi_columns(targets, forest, CODE, 0, span=2, impl="reference")
    for _ in range(3):
        assert_identical(ref, viterbi_columns(targets, forest, CODE, 0, span=2), "auto")
    assert len(calls) == 3 and not _tcq_maps()[0]
    monkeypatch.setenv("TESSERA_TCQ_FUSED", "0")
    for _ in range(3):
        assert_identical(ref, viterbi_columns(targets, forest, CODE, 0, span=2), "auto, 0")
    assert len(calls) == 3, "TESSERA_TCQ_FUSED=0 still took the fused path"
    assert _tcq_maps()[0], "TESSERA_TCQ_FUSED=0 should leave auto on the graph rule"
    monkeypatch.setenv("TESSERA_TCQ_FUSED", "yes")
    with pytest.raises(GrammarError):
        viterbi_columns(targets, forest, CODE, 0, span=2)


def test_what_the_fused_path_cannot_take_is_refused_by_name():
    forest = forest_for("E2M1x2", 7)
    torch.manual_seed(9)
    wide = torch.randn(64, 32, device="cuda", dtype=torch.float64)
    with pytest.raises(GrammarError, match="float64"):
        viterbi_columns(wide, forest, CODE, 0, span=2, impl="fused")
    # ``auto`` serves it anyway, on the machine whose front it promotes.
    tcq_plan_cache_clear()
    ref = viterbi_columns(wide, forest, CODE, 0, span=2, impl="reference")
    assert_identical(ref, viterbi_columns(wide, forest, CODE, 0, span=2), "float64 auto")
    narrow = wide.float()
    with pytest.raises(GrammarError, match="float64"):
        viterbi_columns(narrow, forest, CODE, 0, span=2, weights=wide.abs() + 0.5,
                        impl="fused")
    with pytest.raises(GrammarError, match="cpu"):
        viterbi_columns(narrow.cpu(), forest, CODE, 0, span=2, impl="fused")
    assert tcq_fused_refusal(narrow, None, 4) is not None
    assert tcq_fused_refusal(narrow, narrow[:, :7], 2) is not None
    assert tcq_fused_refusal(narrow, None, 2) is None


def test_concurrent_calls_share_no_buffers():
    """PrismaBuild's workers encode units concurrently in one process."""
    forest = forest_for("E2M1x2", 7)
    torch.manual_seed(13)
    inputs = {tag: torch.randn(128, 48, device="cuda") for tag in "ab"}
    want = {tag: viterbi_columns(t, forest, CODE, 0, span=2, impl="reference")
            for tag, t in inputs.items()}
    got, errors = {}, []

    def work(tag):
        try:
            for _ in range(4):
                got[tag] = viterbi_columns(inputs[tag], forest, CODE, 0, span=2, impl="fused")
        except Exception as error:                        # pragma: no cover
            errors.append(error)

    threads = [threading.Thread(target=work, args=(tag,)) for tag in inputs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    for tag in inputs:
        assert_identical(want[tag], got[tag], tag)


def test_the_reference_primitives_are_what_the_kernels_reproduce():
    """The three torch facts the fused path is built on, checked on CUDA.

    ``(x)**2`` is ``x * x`` bit for bit; ``min(dim)`` returns the first
    minimal index, finite and ``inf`` ties alike, including through the
    ``out=`` spelling the reference's branch uses; so does ``argmin``.
    """
    g = torch.Generator().manual_seed(0)
    scale = torch.logspace(-20, 20, 1 << 20, dtype=torch.float64).float()
    x = (torch.randn(1 << 20, generator=g) * scale).cuda()
    assert torch.equal(x ** 2, x * x)

    ties = torch.randint(0, 3, (4096, 64), generator=g).float()
    ties[:64] = float("inf")
    for dim in (0, 1):
        assert torch.equal(ties.cuda().min(dim=dim).indices.cpu(), ties.min(dim=dim).indices)
        assert torch.equal(torch.argmin(ties.cuda(), dim=dim).cpu(), torch.argmin(ties, dim=dim))
    branch = torch.tensor([[3.0, float("inf"), 2.0, 1.0], [3.0, float("inf"), 1.0, 2.0]],
                          device="cuda")
    values = torch.empty(4, device="cuda")
    taken = torch.empty(4, dtype=torch.long, device="cuda")
    torch.min(branch, dim=0, out=(values, taken))
    assert taken.tolist() == [0, 0, 1, 0]
    assert values.tolist() == [3.0, float("inf"), 1.0, 1.0]


class _Spy:
    def __init__(self, jit):
        self.jit, self.compiled = jit, []

    def __getitem__(self, grid_):
        inner = self.jit[grid_]

        def call(*args, **kwargs):
            out = inner(*args, **kwargs)
            self.compiled.append(out)
            return out

        return call


@pytest.mark.parametrize("name,rate,span,completion,weighted", [
    ("E2M1x2", 7, 2, 0, True), ("E2M1x2", 7, 2, 0, False), ("E2M1x2", 1, 2, 6, True),
    ("E2M1", 3, 1, 0, True), ("E2M1x2", 4, 3, 3, True),
])
def test_no_kernel_spills_on_the_shipping_shapes(name, rate, span, completion, weighted):
    """A spill is a local-memory round trip inside a dependent chain -- the
    mechanism behind the window scan's R=8 cliff -- and a property of the
    compiled kernel, so it is pinned here rather than a wall clock."""
    from tessera import tcq_fused

    forest = forest_for(name, rate)
    arity = grid(name).arity
    torch.manual_seed(1)
    targets = torch.randn(96 * arity, 128, device="cuda")
    weights = torch.rand(96 * arity, 128, device="cuda") + 0.5 if weighted else None
    kernels = tcq_fused._kernels()
    spies = tuple(_Spy(k) for k in kernels)
    tcq_fused._CACHE["k"] = spies
    try:
        viterbi_columns(targets, forest, CODE, completion, span=span, weights=weights,
                        impl="fused")
    finally:
        tcq_fused._CACHE["k"] = kernels
    for which, spy in zip(("minima", "forward", "traceback"), spies):
        assert spy.compiled, f"{which} never launched"
        ck = spy.compiled[-1]
        spills = getattr(ck, "n_spills", None)
        assert spills == 0, (
            f"the fused TCQ {which} kernel spills {spills} bytes (n_regs="
            f"{getattr(ck, 'n_regs', None)}) at {name} R{rate} L{span} c{completion}")
