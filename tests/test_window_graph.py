"""The persistent window plan returns the reference's bytes, not merely its answer.

``window_viterbi.viterbi_window_fused`` is the machine behind the WINDOW body,
which is the default E4M3 wire -- the shipping per-channel recipe.  Its batch is
``2 + steps`` Triton launches at fixed pointers with a three-integer descriptor
on the device, so one capture replays for every batch; a call wide enough to run
six batches has always captured inside itself.

LDLQ makes the calls narrow instead of few.  ``ldl_block`` columns is ONE batch,
so the same tensor, the same table and the same rate ran captured when it was
encoded in one call and eager when it was encoded in ``cols / ldl_block`` of
them -- hundreds of times a pass (issue #94).  A plan kept per shape fixes that
by paying the capture once.

The contract between the two spellings is *identity*, not tolerance.  A trellis
is a chain of decisions, so a last-ulp difference in one branch cost flips a
state, and every state after it, and changes the bytes an artifact ships.  These
tests hold that line on the axes that can break it: rate, window width, arity,
the weighted branch metric, odd column counts, a plan reused across calls with
different data, and the whole encoder under an LDLQ schedule.
"""
import pytest
import torch

from tessera.encode import viterbi_window
from tessera.window_viterbi import (_window_maps, fused_available,
                                    window_plan_cache_clear)

pytestmark = pytest.mark.skipif(
    not fused_available(),
    reason="the fused window Viterbi is a CUDA path and needs triton")


def _case(L, R, arity, rows, cols, weighted, seed):
    g = torch.Generator().manual_seed(seed)
    targets = torch.randn(rows, cols, generator=g).cuda()
    vectors = torch.randn(1 << L, arity, generator=g).cuda()
    weights = (torch.rand(rows, cols, generator=g).cuda() + 0.5) if weighted else None
    return targets, vectors, weights


@pytest.mark.parametrize("L,R,arity,rows,cols,weighted", [
    (14, 4, 1, 128, 16, False),    # the shipping E4M3 wire's L and rung
    (14, 4, 1, 128, 16, True),     # ... under the CHANNEL plane's (s/s.max)^2
    (14, 5, 1, 128, 30, True),     # the odd width a Bresenham schedule leaves
    (12, 4, 1, 64, 3, False),      # the two-or-three column remainder call
    (10, 2, 2, 64, 16, True),      # arity 2
    (10, 8, 1, 64, 16, False),     # the wide fan the runtime scan exists for
])
def test_the_persistent_plan_is_the_reference(L, R, arity, rows, cols, weighted):
    """Every call on a repeated shape returns the reference's states and its
    ``sse`` float -- the first, which is eager, and the ones after, which
    replay a captured graph."""
    targets, vectors, weights = _case(L, R, arity, rows, cols, weighted, seed=L * 31 + R)
    ref, sse_ref = viterbi_window(targets, vectors, L, R, weights=weights,
                                  impl="reference")
    window_plan_cache_clear()
    for _ in range(3):
        got, sse = viterbi_window(targets, vectors, L, R, weights=weights,
                                  impl="fused")
        assert torch.equal(got, ref)
        assert sse == sse_ref


@pytest.mark.parametrize("lever", ["0", "1", None])
def test_every_capture_lever_returns_one_answer(lever, monkeypatch):
    """``0`` is the eager loop, ``1`` captures on first sight, unset lets the
    rules decide.  The knob picks the machine and never the answer."""
    if lever is None:
        monkeypatch.delenv("TESSERA_WINDOW_GRAPH", raising=False)
    else:
        monkeypatch.setenv("TESSERA_WINDOW_GRAPH", lever)
    targets, vectors, weights = _case(14, 4, 1, 128, 16, True, seed=101)
    ref, sse_ref = viterbi_window(targets, vectors, 14, 4, weights=weights,
                                  impl="reference")
    window_plan_cache_clear()
    for _ in range(3):
        got, sse = viterbi_window(targets, vectors, 14, 4, weights=weights,
                                  impl="fused")
        assert torch.equal(got, ref) and sse == sse_ref


def test_a_reused_plan_does_not_leak_the_previous_call():
    """A plan owns its fronts and its traceback, so a second call must not read
    the first's: different targets through one plan give the reference's bytes
    for THEM, and coming back to the first gives the first's again."""
    first, vectors, _ = _case(14, 4, 1, 128, 16, False, seed=5)
    second, _, _ = _case(14, 4, 1, 128, 16, False, seed=6)
    want_first = viterbi_window(first, vectors, 14, 4, impl="reference")
    want_second = viterbi_window(second, vectors, 14, 4, impl="reference")
    window_plan_cache_clear()
    order = [first, second, first, second, first]
    wants = [want_first, want_second, want_first, want_second, want_first]
    for targets, want in zip(order, wants):
        got, sse = viterbi_window(targets, vectors, 14, 4, impl="fused")
        assert torch.equal(got, want[0]) and sse == want[1]


def test_a_reused_plan_reads_this_call_s_table():
    """The reconstruction table is bound per call, not baked into the plan: two
    tables through one shape must give each table's own answer."""
    targets, table_a, _ = _case(12, 4, 1, 64, 16, False, seed=21)
    _, table_b, _ = _case(12, 4, 1, 64, 16, False, seed=22)
    want_a = viterbi_window(targets, table_a, 12, 4, impl="reference")
    want_b = viterbi_window(targets, table_b, 12, 4, impl="reference")
    assert not torch.equal(want_a[0], want_b[0]), "the two tables must differ"
    window_plan_cache_clear()
    for table, want in ((table_a, want_a), (table_b, want_b),
                        (table_a, want_a), (table_b, want_b)):
        got, sse = viterbi_window(targets, table, 12, 4, impl="fused")
        assert torch.equal(got, want[0]) and sse == want[1]


def test_auto_keeps_a_plan_only_on_a_repeat(monkeypatch):
    """A shape seen once must not buy a capture it never replays; a shape seen
    twice must."""
    monkeypatch.delenv("TESSERA_WINDOW_GRAPH", raising=False)
    targets, vectors, _ = _case(12, 4, 1, 64, 16, False, seed=31)
    window_plan_cache_clear()
    viterbi_window(targets, vectors, 12, 4, impl="fused")
    assert not _window_maps()[0], "one call should not have kept a plan"
    viterbi_window(targets, vectors, 12, 4, impl="fused")
    assert _window_maps()[0], "the repeat should have kept a plan"


def test_env_zero_keeps_no_plan(monkeypatch):
    monkeypatch.setenv("TESSERA_WINDOW_GRAPH", "0")
    targets, vectors, _ = _case(12, 4, 1, 64, 16, False, seed=41)
    window_plan_cache_clear()
    for _ in range(3):
        viterbi_window(targets, vectors, 12, 4, impl="fused")
    assert not _window_maps()[0]


def test_a_wide_call_keeps_no_plan(monkeypatch):
    """A call that runs six batches or more amortises its own capture, so it
    keeps today's per-call buffers rather than pinning a whole tensor's
    traceback for the life of the thread."""
    monkeypatch.delenv("TESSERA_WINDOW_GRAPH", raising=False)
    from tessera.window_viterbi import _GRAPH_MIN_BATCHES, _layout

    targets, vectors, _ = _case(14, 4, 1, 64, 512, False, seed=51)
    assert len(_layout(targets.device, 1 << 14, 512, 512)[2]) >= _GRAPH_MIN_BATCHES
    window_plan_cache_clear()
    ref, sse_ref = viterbi_window(targets, vectors, 14, 4, impl="reference")
    for _ in range(3):
        got, sse = viterbi_window(targets, vectors, 14, 4, impl="fused")
        assert torch.equal(got, ref) and sse == sse_ref
    assert not _window_maps()[0]


def test_an_unparseable_lever_is_refused(monkeypatch):
    monkeypatch.setenv("TESSERA_WINDOW_GRAPH", "yes")
    targets, vectors, _ = _case(12, 4, 1, 64, 16, False, seed=61)
    with pytest.raises(ValueError):
        viterbi_window(targets, vectors, 12, 4, impl="fused")


def test_plans_do_not_cross_threads():
    """A plan owns the tensors its Viterbi writes and PrismaBuild's workers
    encode units concurrently in one process, so two threads must not share
    one."""
    import threading

    a, vectors, _ = _case(12, 4, 1, 64, 16, False, seed=71)
    b, _, _ = _case(12, 4, 1, 64, 16, False, seed=72)
    want_a = viterbi_window(a, vectors, 12, 4, impl="reference")
    want_b = viterbi_window(b, vectors, 12, 4, impl="reference")
    got, maps = {}, {}

    def work(tag, targets):
        window_plan_cache_clear()
        for _ in range(3):
            got[tag] = viterbi_window(targets, vectors, 12, 4, impl="fused")
        maps[tag] = id(_window_maps()[0])

    threads = [threading.Thread(target=work, args=("a", a)),
               threading.Thread(target=work, args=("b", b))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert maps["a"] != maps["b"], "the two threads shared one plan map"
    for want, tag in ((want_a, "a"), (want_b, "b")):
        assert torch.equal(want[0], got[tag][0]) and want[1] == got[tag][1]


# ------------------------------------------------- through the real encoder


@pytest.mark.parametrize("lever", ["0", "1", None])
def test_an_ldlq_encode_ships_the_same_bytes_under_every_lever(lever, monkeypatch):
    """The shape issue #94 is about, end to end.

    ``ldl_block`` columns at a time is one batch a call, so this is the
    schedule that never captured; the bytes it writes must not depend on
    whether it captures now.  The reference arm pins ``impl="reference"``, so
    the comparison is against the torch chain that defines the trellis and not
    merely against another run of the same kernels.
    """
    import tessera.encode as encode_mod
    from tessera.alphabet import E4M3_GRID
    from tessera.compensate import block_ldl, regularize_hessian
    from tessera.encode import encode_unit
    from tessera.export import build_unit_artifact
    from tessera.manifest import BodyKind, ScalePlaneKind
    from tessera.trellis import ConvCode

    rows, cols, block, rate, window = 128, 64, 16, 4, 10
    g = torch.Generator().manual_seed(19)
    w = torch.randn(rows, cols, generator=g).cuda()
    x = torch.randn(512, cols, generator=g).cuda()
    hessian = (x.t() @ x).float()
    ldl = block_ldl(regularize_hessian(hessian, sigma_reg=1.0), block)
    metric = (hessian.diagonal() / hessian.diagonal().mean()).clone()
    code = ConvCode(memory=6)
    rates = tuple([rate] * cols)

    def run(impl):
        original = encode_mod.viterbi_window

        def pinned(*args, **kwargs):
            kwargs["impl"] = impl
            return original(*args, **kwargs)

        window_plan_cache_clear()
        monkeypatch.setattr(encode_mod, "viterbi_window", pinned)
        try:
            unit = encode_unit(w, E4M3_GRID, rates, code, body=BodyKind.WINDOW,
                               window_bits=window, scale_plane=ScalePlaneKind.CHANNEL,
                               scale_refit=2, completion=0, ldl=ldl,
                               ldl_block=block, refit_metric=metric)
        finally:
            monkeypatch.setattr(encode_mod, "viterbi_window", original)
        return build_unit_artifact(unit, "unit0", E4M3_GRID, rate * 256, code)[2]

    monkeypatch.delenv("TESSERA_WINDOW_GRAPH", raising=False)
    reference = run("reference")
    if lever is None:
        monkeypatch.delenv("TESSERA_WINDOW_GRAPH", raising=False)
    else:
        monkeypatch.setenv("TESSERA_WINDOW_GRAPH", lever)
    assert run("fused") == reference
