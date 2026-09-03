"""The captured TCQ trellis returns the reference's bytes, not merely its answer.

``encode.viterbi_columns`` is the coset trellis every E2M1 and every E2M1x2
cap-rung artifact is built with.  Its step loop is ``supers`` Python iterations
of about forty small kernels, so its wall cost is fixed per CALL and nearly
independent of how many columns the call carries -- which is why LDLQ, whose
schedule turns one call into ``cols / ldl_block`` calls, cost 19.2x on a
matched pair (issue #13).  ``impl="graph"`` captures that loop once per shape
and replays it.

The contract between the two is *identity*, not tolerance.  A trellis is a
chain of decisions, so a last-ulp difference in one branch cost flips a state,
and every state after it, and changes the bytes an artifact ships.  These
tests hold that line on the axes that can break it: rate, span, completion
depth, arity, the weighted branch metric, odd column counts, and the narrow
shapes LDLQ actually asks for.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the captured TCQ trellis is a CUDA path")

from tessera.alphabet import SERIALISABLE_GRIDS
from tessera.encode import build_forest, tcq_plan_cache_clear, viterbi_columns
from tessera.errors import GrammarError
from tessera.trellis import ConvCode


def grid(name):
    return [g for g in SERIALISABLE_GRIDS.values() if g.name == name][0]


def both(targets, forest, completion, span, weights=None):
    tcq_plan_cache_clear()
    ref = viterbi_columns(targets, forest, ConvCode(), completion, span=span,
                          weights=weights, impl="reference")
    got = viterbi_columns(targets, forest, ConvCode(), completion, span=span,
                          weights=weights, impl="graph")
    return ref, got


@pytest.mark.parametrize("name,rate,span,completion", [
    ("E2M1x2", 7, 2, 0),      # the cap rung, the shipping E2M1x2 TCQ wire
    ("E2M1x2", 6, 2, 1),
    ("E2M1x2", 5, 2, 2),
    ("E2M1", 3, 1, 0),        # arity 1, span 1
    ("E2M1", 2, 1, 1),
])
@pytest.mark.parametrize("cols", [1, 5, 32, 96])
def test_graph_matches_reference(name, rate, span, completion, cols):
    g = grid(name)
    forest = build_forest(rate, grid=g)
    rows = 64
    torch.manual_seed(rate * 100 + cols)
    targets = torch.randn(rows, cols, device="cuda")
    (a0, b0, s0), (a1, b1, s1) = both(targets, forest, completion, span)
    assert torch.equal(a0, a1)
    assert torch.equal(b0, b1)
    assert s0 == s1


@pytest.mark.parametrize("name,rate,span,completion", [
    ("E2M1x2", 7, 2, 0), ("E2M1x2", 6, 2, 1), ("E2M1", 3, 1, 0),
])
def test_graph_matches_reference_weighted(name, rate, span, completion):
    g = grid(name)
    forest = build_forest(rate, grid=g)
    torch.manual_seed(7)
    targets = torch.randn(64, 32, device="cuda")
    weights = torch.rand(64, 32, device="cuda") + 0.5
    (a0, b0, s0), (a1, b1, s1) = both(targets, forest, completion, span, weights)
    assert torch.equal(a0, a1)
    assert torch.equal(b0, b1)
    assert s0 == s1


def test_ties_agree():
    """Few distinct targets make exactly equal path costs common; the winner
    must be the same index on both paths or the bytes differ at equal cost."""
    forest = build_forest(7, grid=grid("E2M1x2"))
    torch.manual_seed(3)
    targets = torch.randint(0, 2, (64, 48), device="cuda").float()
    (a0, b0, s0), (a1, b1, s1) = both(targets, forest, 0, 2)
    assert torch.equal(a0, a1) and torch.equal(b0, b1) and s0 == s1


def test_replay_is_stable_across_calls():
    """A cached plan is reused, so a second call must not read the first's
    buffers: same targets give the same bytes, different targets give the
    reference's bytes for THEM."""
    forest = build_forest(7, grid=grid("E2M1x2"))
    torch.manual_seed(11)
    first = torch.randn(64, 32, device="cuda")
    second = torch.randn(64, 32, device="cuda")
    tcq_plan_cache_clear()
    ref_first = viterbi_columns(first, forest, ConvCode(), 0, span=2, impl="reference")
    ref_second = viterbi_columns(second, forest, ConvCode(), 0, span=2, impl="reference")
    tcq_plan_cache_clear()
    got_first = viterbi_columns(first, forest, ConvCode(), 0, span=2, impl="graph")
    got_second = viterbi_columns(second, forest, ConvCode(), 0, span=2, impl="graph")
    got_again = viterbi_columns(first, forest, ConvCode(), 0, span=2, impl="graph")
    for ref, got in ((ref_first, got_first), (ref_second, got_second),
                     (ref_first, got_again)):
        assert torch.equal(ref[0], got[0])
        assert torch.equal(ref[1], got[1])
        assert ref[2] == got[2]


def test_auto_takes_the_graph_only_on_a_repeat(monkeypatch):
    """``auto`` is the reference on a shape's first call and the graph after,
    and both spellings return the same bytes -- so the policy is a cost
    decision and never an answer decision."""
    monkeypatch.delenv("TESSERA_TCQ_GRAPH", raising=False)
    forest = build_forest(7, grid=grid("E2M1x2"))
    torch.manual_seed(5)
    targets = torch.randn(64, 32, device="cuda")
    tcq_plan_cache_clear()
    ref = viterbi_columns(targets, forest, ConvCode(), 0, span=2, impl="reference")
    tcq_plan_cache_clear()
    first = viterbi_columns(targets, forest, ConvCode(), 0, span=2)
    second = viterbi_columns(targets, forest, ConvCode(), 0, span=2)
    from tessera.encode import _tcq_maps
    assert _tcq_maps()[0], "the repeat should have built a plan"
    for got in (first, second):
        assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])
        assert ref[2] == got[2]


def test_env_zero_forces_the_eager_loop(monkeypatch):
    monkeypatch.setenv("TESSERA_TCQ_GRAPH", "0")
    forest = build_forest(7, grid=grid("E2M1x2"))
    torch.manual_seed(9)
    targets = torch.randn(64, 32, device="cuda")
    tcq_plan_cache_clear()
    for _ in range(3):
        viterbi_columns(targets, forest, ConvCode(), 0, span=2)
    from tessera.encode import _tcq_maps
    assert not _tcq_maps()[0]


def test_plans_do_not_cross_threads():
    """Two threads encoding the same shape must not share one plan's buffers:
    a plan owns the tensors its Viterbi writes, and PrismaBuild's workers
    encode units concurrently in one process."""
    import threading

    forest = build_forest(7, grid=grid("E2M1x2"))
    torch.manual_seed(13)
    a = torch.randn(64, 32, device="cuda")
    b = torch.randn(64, 32, device="cuda")
    tcq_plan_cache_clear()
    want_a = viterbi_columns(a, forest, ConvCode(), 0, span=2, impl="reference")
    want_b = viterbi_columns(b, forest, ConvCode(), 0, span=2, impl="reference")
    got = {}
    seen = {}

    def work(tag, targets):
        tcq_plan_cache_clear()
        for _ in range(3):
            got[tag] = viterbi_columns(targets, forest, ConvCode(), 0, span=2,
                                       impl="graph")
        from tessera.encode import _tcq_maps
        seen[tag] = id(_tcq_maps()[0])

    threads = [threading.Thread(target=work, args=("a", a)),
               threading.Thread(target=work, args=("b", b))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen["a"] != seen["b"], "the two threads shared one plan map"
    for want, tag in ((want_a, "a"), (want_b, "b")):
        assert torch.equal(want[0], got[tag][0])
        assert torch.equal(want[1], got[tag][1])
        assert want[2] == got[tag][2]


def test_unknown_impl_is_refused():
    forest = build_forest(7, grid=grid("E2M1x2"))
    targets = torch.randn(64, 32, device="cuda")
    with pytest.raises(GrammarError):
        viterbi_columns(targets, forest, ConvCode(), 0, span=2, impl="fast")
