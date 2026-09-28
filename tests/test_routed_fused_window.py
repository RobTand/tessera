"""The fused routed window MoE lane (``tessera.routed_fused``, tessera#640).

Parity against the same per-expert definition oracle the grouped GEMM tests
use (``test_window_gemm_grouped.Expert``: each expert's weights ARE the
definition, routing applied on the host), for both window families, with a
TP row cut (per-expert start state), M = 1 and M past one 64-route
superblock, empty and repeated experts, the route-preserving ``gate_up`` and
the reduced ``down_routes`` stages, the SwiGLU clamp, the top-k = 1
weight-on-input placement; CUDA-graph replay (twice, against eager); two-run
bitwise equality of the deterministic reduction; the support predicate's
refusals by name; and the adapter dispatch (fused where admitted, compact
otherwise, each recording its own launch pair).

The lane is a CUDA kernel JIT-built on first use; every GPU case here runs
through PrismaBuild inside the pinned serving image
(``experiments/routed_fused_tests.sh``).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera import routed_fused as rf                    # noqa: E402
from tessera import window_gemm_grouped as wgg            # noqa: E402
from tessera.errors import GrammarError                   # noqa: E402
from tessera.native_window_moe import (NativeWindowMoE, PackedWindowMoeBundles,  # noqa: E402
                                       _silu_and_mul, native_window_moe_from_bundles)

import fused_bound as fb                                   # noqa: E402
from test_window_gemm_grouped import Expert, _quant, _tol  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the lane is a CUDA kernel")

#: The rungs the mixed-rate routed tests read (tessera#694): the GLM E4M3
#: rungs q256 832 (rates 3/4), 928, 1088 (4/5), 1152, the one-rate 768 and
#: 1280 (rate 5, the largest the two-table gate/up launch fits in the sm_121
#: shared-memory block), and the low extremes 256 (rate 1) and 384 (1/2); 256
#: and 576 columns realise each exactly.  Rates 6..8 on gate/up are refused
#: by name (``test_support_predicate_refuses_the_gate_up_slot_the_device_cannot_hold``).
Q256_CASES = [256, 384, 768, 832, 928, 1088, 1152, 1280]

L = 14
# gate/up: [INTER, HIDDEN] at rate 4 (two 512-row tiles: INTER > 512);
# down: [HIDDEN, INTER] (two tiles again; HIDDEN a multiple of 128).
HIDDEN, INTER, EXPERTS, TOP_K = 256, 576, 5, 3


def _init(cols, seed):
    return torch.randint(0, 1 << L, (cols,), generator=torch.Generator().manual_seed(seed),
                         dtype=torch.int32)


def _sched(cols, q256):
    """The grammar's Bresenham schedule for ``q256`` over ``cols`` columns
    (cap 8: the packer's whole range, so the value family's rungs are read too)."""
    from fractions import Fraction

    from tessera.grammar import bresenham_rate_schedule

    return bresenham_rate_schedule(Fraction(q256, 256), cols, cap=8)


def _stacks(family, *, hidden=HIDDEN, inter=INTER, experts=EXPERTS, seed=300, cut=True, q256=1024):
    """gate, up, down Expert lists at the ``q256`` rung's schedule (rate 4
    everywhere by default); experts 1 and 3 carry a start state (the TP
    row-cut case) when ``cut``."""
    r_h, r_i = _sched(hidden, q256), _sched(inter, q256)
    gate = [Expert(inter, hidden, r_h, seed + i, family=family,
                   init=_init(hidden, seed + 40 + i) if (cut and i % 2) else None)
            for i in range(experts)]
    up = [Expert(inter, hidden, r_h, seed + 10 + i, family=family,
                 init=_init(hidden, seed + 50 + i) if (cut and i % 2) else None)
          for i in range(experts)]
    down = [Expert(hidden, inter, r_i, seed + 20 + i, family=family,
                   init=_init(inter, seed + 60 + i) if (cut and i == 3) else None)
            for i in range(experts)]
    return gate, up, down


def _bundles(family, stacks):
    arithmetic = "folded" if family == "value" else "epilogue"
    gate, up, down = (wgg.prepare_grouped_window_gemm([e.unit for e in s], block_m=32,
                                                      block_n=64, block_k=64,
                                                      arithmetic=arithmetic)
                      for s in stacks)
    return PackedWindowMoeBundles(gate=gate, up=up, down=down, family=family)


def _fused(bundles):
    return rf.FusedRoutedWindowMoE.from_bundles(bundles.gate, bundles.up, bundles.down)


def _legacy(bundles):
    return native_window_moe_from_bundles(bundles.down, gate=bundles.gate, up=bundles.up)


def _routes(t, k, seed, experts=EXPERTS):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, experts, (t, k), generator=g, dtype=torch.int32).cuda()
    rw = torch.rand(t, k, generator=g).cuda()
    return ids, rw


def _reference(stacks, x, ids, rw, family, *, limit=None, weight_input=False):
    """vLLM's placement, per route on the host: bf16 gate/up, SwiGLU (clamp)
    in fp32 to one bf16, the family's A quant, down per route, weighted and
    rounded to bf16 per route, summed in fp32, rounded once."""
    gate_stack, up_stack, down_stack = stacks
    folded = family == "value"
    t, k = ids.shape
    inter, hidden = gate_stack[0].rows, down_stack[0].rows
    sel = torch.arange(t, device="cuda")[:, None].expand_as(ids)
    if weight_input:
        x = x * rw.reshape(-1, 1).to(x.dtype)
    if family == "e4m3":
        xq, a1 = _quant(x)
        xq1 = xq.float()
    else:
        xq1, a1 = x, None
    g = torch.stack([e.reference(xq1, family, folded=folded) for e in gate_stack])   # [E,T,I]
    u = torch.stack([e.reference(xq1, family, folded=folded) for e in up_stack])
    if family == "e4m3":
        g = g * a1.reshape(1, t, 1)
        u = u * a1.reshape(1, t, 1)
    gate = g[ids.long(), sel].bfloat16().float()                                      # [T,K,I]
    up = u[ids.long(), sel].bfloat16().float()
    if limit is not None:
        gate = torch.clamp(gate, max=limit)
        up = torch.clamp(up, min=-limit, max=limit)
    act = (torch.nn.functional.silu(gate) * up).bfloat16()
    flat = act.reshape(t * k, inter)
    if family == "e4m3":
        aq, a2 = _quant(flat)
        flat_q = aq.float()
    else:
        flat_q, a2 = flat, None
    d = torch.stack([e.reference(flat_q, family, folded=folded) for e in down_stack])  # [E,T*K,H]
    routes = d[ids.long().reshape(-1), torch.arange(t * k, device="cuda")].reshape(t, k, hidden)
    if family == "e4m3":
        routes = routes * a2.reshape(t, k, 1)
    if not weight_input:
        routes = routes * rw[..., None]
    routes = routes.bfloat16()
    return routes.float().sum(1).bfloat16()


def _close(out, ref, what):
    err = float((out.float() - ref.float()).abs().max())
    assert err < _tol(ref), f"{what}: max abs err {err} vs tol {_tol(ref)}"
    return err


# --- parity ------------------------------------------------------------------

@cuda
@pytest.mark.parametrize("family", ["value", "e4m3"])
@pytest.mark.parametrize("t", [1, 7, 71])
def test_fused_forward_matches_the_per_expert_oracle_and_the_compact_adapter(family, t):
    """M = 1 (decode), a short batch, and 71 x 3 = 213 routes so at least one
    expert spans two 64-route superblocks; experts 1 and 3 carry a start
    state.  The compact adapter computes the same function of the wire, so
    the two lanes agree to accumulation order; the oracle is the definition."""
    stacks = _stacks(family)
    bundles = _bundles(family, stacks)
    fused, legacy = _fused(bundles), _legacy(bundles)
    x = torch.randn(t, HIDDEN, device="cuda").bfloat16()
    ids, rw = _routes(t, TOP_K, 900 + t)
    out = fused(x, ids, rw)
    assert out.shape == (t, HIDDEN) and out.dtype == torch.bfloat16
    ref = _reference(stacks, x, ids, rw, family)
    _close(out, ref, f"{family} fused vs oracle")
    _close(legacy(x, ids, rw), ref, f"{family} compact vs oracle")
    _close(out, legacy(x, ids, rw), f"{family} fused vs compact")


@cuda
@pytest.mark.parametrize("family", ["value", "e4m3"])
def test_fused_swiglu_clamp_is_the_stock_placement(family):
    stacks = _stacks(family, cut=False)
    bundles = _bundles(family, stacks)
    fused, legacy = _fused(bundles), _legacy(bundles)
    x = (torch.randn(24, HIDDEN, device="cuda") * 3).bfloat16()
    ids, rw = _routes(24, TOP_K, 77)
    limit = 0.5
    out = fused(x, ids, rw, swiglu_limit=limit)
    ref = _reference(stacks, x, ids, rw, family, limit=limit)
    _close(out, ref, f"{family} clamp vs oracle")
    _close(out, legacy(x, ids, rw, swiglu_limit=limit), f"{family} clamp vs compact")
    # the clamp changes the answer, so a lane that ignored it could not pass
    assert not torch.equal(out, fused(x, ids, rw))


@cuda
@pytest.mark.parametrize("family", ["value", "e4m3"])
def test_fused_empty_and_repeated_experts(family):
    """Every route to one expert (the others empty) and a batch whose experts
    are all hit: the device work list is sized by route counts, never by E."""
    stacks = _stacks(family, cut=False)
    bundles = _bundles(family, stacks)
    fused = _fused(bundles)
    x = torch.randn(40, HIDDEN, device="cuda").bfloat16()
    rw = torch.rand(40, TOP_K, device="cuda")
    ids = torch.full((40, TOP_K), 2, dtype=torch.int32, device="cuda")
    _close(fused(x, ids, rw), _reference(stacks, x, ids, rw, family), f"{family} one expert")
    ids = torch.arange(40 * TOP_K, device="cuda", dtype=torch.int32).reshape(40, TOP_K) % EXPERTS
    _close(fused(x, ids, rw), _reference(stacks, x, ids, rw, family), f"{family} all experts")
    empty = fused(x[:0], ids[:0], rw[:0])
    assert empty.shape == (0, HIDDEN) and empty.dtype == torch.bfloat16


@cuda
@pytest.mark.parametrize("family", ["value", "e4m3"])
def test_fused_staged_interfaces_match_the_compact_adapters(family):
    """``gate_up`` (route-preserving) and ``down_routes`` (reduced) are the
    stages the routed pair oracle teacher-forces; they must be the compact
    adapter's to accumulation order, and their composition the forward."""
    stacks = _stacks(family)
    bundles = _bundles(family, stacks)
    fused, legacy = _fused(bundles), _legacy(bundles)
    t = 33
    x = torch.randn(t, HIDDEN, device="cuda").bfloat16()
    ids, rw = _routes(t, TOP_K, 5)
    gu = fused.gate_up(x, ids, rw, preserve=True, apply_router_weight_on_input=False)
    assert gu.shape == (t, TOP_K, 2 * INTER) and gu.dtype == torch.bfloat16
    g_l = legacy.gate(x, ids, rw, preserve=True, apply_router_weight_on_input=False)
    u_l = legacy.up(x, ids, rw, preserve=True, apply_router_weight_on_input=False)
    _close(gu[..., :INTER], g_l, f"{family} gate")
    _close(gu[..., INTER:], u_l, f"{family} up")
    act = _silu_and_mul(gu[..., :INTER].reshape(t * TOP_K, INTER),
                        gu[..., INTER:].reshape(t * TOP_K, INTER), clamp_limit=None)
    down = fused.down_routes(act, ids, rw, route_input=True,
                             apply_router_weight_on_input=False, round_routes=True)
    down_l = legacy.down(act, ids, rw, route_input=True,
                         apply_router_weight_on_input=False, round_routes=True)
    _close(down, down_l, f"{family} down")
    # the staged composition IS the forward's arithmetic (same rounding points)
    _close(down, fused(x, ids, rw), f"{family} staged vs forward")
    if family == "e4m3":
        aq, a2 = _quant(act)
        pre = fused.down_routes(aq, ids, rw, a_scale=a2, route_input=True,
                                apply_router_weight_on_input=False, round_routes=True)
        assert torch.equal(pre, down), "the prequantized path must be the internal quantizer's"
    with pytest.raises(GrammarError):
        fused.gate_up(x, ids, rw, preserve=False)
    with pytest.raises(GrammarError):
        fused.down_routes(act, ids, rw, round_routes=False)


@cuda
@pytest.mark.parametrize("family", ["value", "e4m3"])
def test_fused_weight_on_input_is_the_modular_prepare_placement(family):
    stacks = _stacks(family, cut=False)
    bundles = _bundles(family, stacks)
    fused, legacy = _fused(bundles), _legacy(bundles)
    t = 19
    x = torch.randn(t, HIDDEN, device="cuda").bfloat16()
    ids, rw = _routes(t, 1, 11)
    out = fused(x, ids, rw, apply_router_weight_on_input=True)
    ref = _reference(stacks, x, ids, rw, family, weight_input=True)
    _close(out, ref, f"{family} weight on input vs oracle")
    _close(out, legacy(x, ids, rw, apply_router_weight_on_input=True),
           f"{family} weight on input vs compact")
    ids3, rw3 = _routes(t, 3, 12)
    with pytest.raises(GrammarError, match="topk=1"):
        fused(x, ids3, rw3, apply_router_weight_on_input=True)


# --- determinism and graphs ------------------------------------------------------

@cuda
@pytest.mark.parametrize("family", ["value", "e4m3"])
def test_fused_two_runs_are_bitwise_equal(family):
    """The down reduction is a fixed-order per-token sum over route-sorted
    bf16 rows: no atomics, so two runs of the same forward are one tensor."""
    stacks = _stacks(family)
    fused = _fused(_bundles(family, stacks))
    x = torch.randn(71, HIDDEN, device="cuda").bfloat16()
    ids, rw = _routes(71, TOP_K, 21)
    first = fused(x, ids, rw)
    for _ in range(3):
        assert torch.equal(fused(x, ids, rw), first)


@cuda
@pytest.mark.parametrize("family", ["value", "e4m3"])
def test_fused_forward_captures_and_replays_twice_against_eager(family):
    """The device work counter is zeroed INSIDE the captured region, so a
    replay starts a fresh work list; two replays must equal the eager forward
    on the same (static) inputs -- bitwise, the lane being deterministic."""
    stacks = _stacks(family)
    fused = _fused(_bundles(family, stacks))
    t = 40
    x = torch.randn(t, HIDDEN, device="cuda").bfloat16()
    ids, rw = _routes(t, TOP_K, 31)
    eager = fused(x, ids, rw)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            fused(x, ids, rw)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fused(x, ids, rw)
    for _ in range(2):
        captured.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, eager)
    # new routing under the same static buffers replays to the new answer
    ids2, rw2 = _routes(t, TOP_K, 32)
    ids.copy_(ids2)
    rw.copy_(rw2)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, fused(x, ids2, rw2))


# --- the support predicate and the adapter dispatch ------------------------------

@cuda
def test_support_predicate_refuses_by_name():
    value = _stacks("value", cut=False)
    ok = _bundles("value", value)
    assert rf.fused_routed_window_supported(ok.gate, ok.up, ok.down) is None

    def grouped(experts):
        return wgg.prepare_grouped_window_gemm([e.unit for e in experts], block_m=32, block_n=64,
                                               block_k=64, arithmetic="folded")
    # three rates: three runs per expert; the kernel reads the two bracketing the root
    three = grouped([Expert(INTER, HIDDEN, tuple((2, 3, 4)[c % 3] for c in range(HIDDEN)), 700 + i)
                     for i in range(EXPERTS)])
    reason = rf.fused_routed_window_supported(three, ok.up, ok.down)
    assert reason is not None and "runs" in reason
    # two rates (a permuted column order) ARE read since contract v45 (#694) --
    # but gate and up must share the tile stride the one launch reads
    mixed_rates = tuple(2 if c % 2 else 4 for c in range(HIDDEN))
    mixed = grouped([Expert(INTER, HIDDEN, mixed_rates, 710 + i) for i in range(EXPERTS)])
    reason = rf.fused_routed_window_supported(mixed, ok.up, ok.down)
    assert reason is not None and "tile_words" in reason
    mixed_up = grouped([Expert(INTER, HIDDEN, mixed_rates, 720 + i) for i in range(EXPERTS)])
    assert rf.fused_routed_window_supported(mixed, mixed_up, ok.down) is None
    # one rate other than 4 everywhere is one run the kernel reads
    rate2 = grouped([Expert(INTER, HIDDEN, (2,) * HIDDEN, 730 + i) for i in range(EXPERTS)])
    rate2_up = grouped([Expert(INTER, HIDDEN, (2,) * HIDDEN, 740 + i) for i in range(EXPERTS)])
    assert rf.fused_routed_window_supported(rate2, rate2_up, ok.down) is None
    # experts that disagree on their schedule: the kernel reads one run pair per
    # stack.  Half the columns at 3 and half at 5 carry the same words as rate 4
    # everywhere, so the wire is [E, W] and the run tables are what differ.
    three_five = tuple(3 if c % 2 else 5 for c in range(HIDDEN))
    uneven = grouped([Expert(INTER, HIDDEN, three_five if i % 2 else (4,) * HIDDEN, 750 + i)
                      for i in range(EXPERTS)])
    reason = rf.fused_routed_window_supported(uneven, ok.up, ok.down)
    assert reason is not None and "run tables" in reason
    # the epilogue arithmetic on the value family is not the published contract
    epi = wgg.prepare_grouped_window_gemm([e.unit for e in value[0]], block_m=32, block_n=64,
                                          block_k=64, arithmetic="epilogue")
    reason = rf.fused_routed_window_supported(epi, ok.up, ok.down)
    assert reason is not None and "arithmetic" in reason
    # geometry: an intermediate size the 64-wide half tile cannot cover
    small = _stacks("value", inter=96, hidden=256, cut=False)
    small_b = _bundles("value", small)
    reason = rf.fused_routed_window_supported(small_b.gate, small_b.up, small_b.down)
    # the down bundle's 96 columns fail the per-bundle column rule (a multiple
    # of 32, at least 128) before the cross-bundle geometry check is reached
    assert reason is not None and "down has 96 columns" in reason
    narrow = _stacks("value", inter=128, hidden=192, cut=False)
    narrow_b = _bundles("value", narrow)
    reason = rf.fused_routed_window_supported(narrow_b.gate, narrow_b.up, narrow_b.down)
    assert reason is not None and "hidden" in reason


def test_support_predicate_honours_the_opt_out_and_the_device(monkeypatch):
    class B:  # the fields the predicate reads first
        family = "value"
        arithmetic = "folded"
        device = torch.device("cpu")
        experts = 1
    monkeypatch.setenv(rf.ENV_TOGGLE, "0")
    assert "disabled" in rf.fused_routed_window_supported(B, B, B)
    monkeypatch.delenv(rf.ENV_TOGGLE)
    assert "cpu" in rf.fused_routed_window_supported(B, B, B)


def test_the_word_stage_slot_and_shared_memory_are_the_kernels_layout():
    """The host restates the kernel's shared-memory layout (tessera#694): a
    column's slot is its ``2 * rate`` words plus the one word the re-aligned
    window over-reads at rates whose ``8 * rate`` is not a multiple of 32; a
    launch's slot is the larger of its pair's, rounded to a multiple of 4; the
    word stages sit after the fixed part (two tables for gate/up, one for
    down/dense) and their size follows the slot."""
    assert [rf.slot_words_for_rate(r) for r in rf.RATES] == [3, 5, 7, 8, 11, 13, 15, 16]

    def pair(r_lo, r_hi=0, n_hi=0):
        return torch.tensor([r_lo, 0, 64, 0, r_hi, 64, n_hi, 0], dtype=torch.int32)

    assert [rf.slot_words_for_pair(pair(r)) for r in rf.RATES] == [4, 8, 8, 8, 12, 16, 16, 16]
    assert [rf.slot_words_for_pair(pair(lo, lo + 1, 32)) for lo in range(1, 8)] == [8, 8, 8, 12, 16, 16, 16]
    assert rf.slot_words_for_pair(pair(1, 8, 32)) == 16
    # a pair whose high run is empty is the low rate's slot alone
    assert rf.slot_words_for_pair(pair(3, 8, 0)) == 8
    assert rf.smem_bytes(0, 8) == rf.smem_bytes(1, 8) == 91_216 + 3 * 2 * rf.BK * 8 * 4 == 97_360
    assert rf.smem_bytes(0, 12) == 100_432
    assert rf.smem_bytes(0, 16) == 103_504          # over sm_121's 101,376: gate/up refuses rates 6..8
    assert rf.smem_bytes(2, 16) == 70_736           # the one-table down/dense launch fits every rate
    assert rf.SLOT_WORDS_MAX == rf.slot_words_for_rate(rf.RATE_MAX) == 16


@cuda
def test_support_predicate_refuses_the_gate_up_slot_the_device_cannot_hold():
    """A gate/up stack whose slot does not fit the device's opt-in shared
    memory is refused by name (the compact adapter serves it); one that fits
    is admitted.  Rate 6 (q256 1536) is the first the two-table launch cannot
    hold on sm_121; rate 5 (q256 1280) is the largest it can."""
    lib = rf._ext("value")
    have = int(lib.max_dynamic_smem_bytes(torch.cuda.current_device()))
    assert int(lib.smem_bytes(0, 16)) == rf.smem_bytes(0, 16) and int(lib.smem_bytes(2, 16)) == rf.smem_bytes(2, 16)
    for q256, slot in ((1536, 16), (1280, 12)):
        b = _bundles("value", _stacks("value", q256=q256, cut=False))
        reason = rf.fused_routed_window_supported(b.gate, b.up, b.down)
        if rf.smem_bytes(0, slot) <= have:
            assert reason is None, reason
        else:
            assert reason is not None and "shared memory" in reason and "gate/up" in reason \
                and str(rf.smem_bytes(0, slot)) in reason and str(have) in reason
    # the same rates on the down/dense launch fit: one table, not two
    assert rf.smem_bytes(2, 16) <= have


@cuda
@pytest.mark.parametrize("family", ["value", "e4m3"])
def test_bundles_adapter_dispatches_and_names_its_own_launch(family, monkeypatch):
    from tessera.serving.scheme import ROUTED_FUSED_WINDOW_SYMBOL, WINDOW_MOE_COMPACT_SYMBOL
    from tessera.serving.telemetry import DECODERS

    stacks = _stacks(family)
    monkeypatch.delenv(rf.ENV_TOGGLE, raising=False)
    bundles = _bundles(family, stacks)
    adapter = bundles.adapter()
    assert isinstance(adapter, rf.FusedRoutedWindowMoE)
    assert adapter is bundles.adapter(), "built once"
    symbol, decoder = adapter.launch_pair
    assert symbol == ROUTED_FUSED_WINDOW_SYMBOL and decoder in DECODERS
    assert decoder == ("native_routed_fused_window_folded" if family == "value"
                       else "native_routed_fused_window")
    names = dict(bundles.named_tensors())
    assert {"routed_fused.table_gate", "routed_fused.table_up", "routed_fused.table_down"} <= set(names)
    assert bundles.resident_bytes() >= adapter.resident_bytes()
    monkeypatch.setenv(rf.ENV_TOGGLE, "0")
    compact = _bundles(family, stacks).adapter()
    assert isinstance(compact, NativeWindowMoE)
    symbol, decoder = compact.launch_pair
    assert symbol == WINDOW_MOE_COMPACT_SYMBOL and decoder in DECODERS
    assert decoder == ("native_window_moe_compact_folded" if family == "value"
                       else "native_window_moe_compact")
    # the mixed-rate stack keeps the compact adapter whatever the toggle says
    monkeypatch.delenv(rf.ENV_TOGGLE)
    gate, up, down = stacks
    mixed = [Expert(INTER, HIDDEN, tuple(2 if c % 2 else 4 for c in range(HIDDEN)), 800 + i,
                    family=family) for i in range(EXPERTS)]
    mixed_bundles = _bundles(family, (mixed, up, down))
    assert isinstance(mixed_bundles.adapter(), NativeWindowMoE)


@cuda
@pytest.mark.parametrize("family", ["value", "e4m3"])
def test_a_build_failure_substitutes_the_compact_adapter(family, monkeypatch, caplog):
    """``native_extensions[].when_unavailable`` publishes the compact adapter as
    the substitute; ``adapter()`` makes that substitution at construction, not
    on the first forward, and says why."""
    import logging

    def no_toolchain(_family):
        raise RuntimeError("Error building extension: nvcc not found")

    monkeypatch.delenv(rf.ENV_TOGGLE, raising=False)
    monkeypatch.setattr(rf, "_ext", no_toolchain)
    bundles = _bundles(family, _stacks(family))
    with caplog.at_level(logging.WARNING, logger="tessera.native_window_moe"):
        adapter = bundles.adapter()
    assert isinstance(adapter, NativeWindowMoE)
    assert adapter is bundles.adapter()
    assert "_fused_adapter" not in bundles.__dict__
    assert any("native build unavailable" in r.getMessage() and "nvcc not found" in r.getMessage()
               for r in caplog.records)
    # a grammar refusal is not a build failure and is never swallowed
    monkeypatch.setattr(rf, "_ext", lambda _f: (_ for _ in ()).throw(GrammarError("grammar")))
    with pytest.raises(GrammarError):
        _bundles(family, _stacks(family)).adapter()


def test_the_published_lane_predicate_is_the_kernels_shape():
    """``ext.ROUTED_FUSED_LANE_REQUIRES`` and the load-time predicate read one
    kernel: every rate 1..8 (contract v45), window bits 14, and the two module
    names are the loader's literals."""
    from tessera.serving import ext

    assert ext.ROUTED_FUSED_LANE_REQUIRES["column_rates"] == list(rf.RATES) == list(range(1, 9))
    assert ext.ROUTED_FUSED_LANE_REQUIRES["window_bits"] == [rf.WINDOW_BITS]
    assert "start_state" not in ext.ROUTED_FUSED_LANE_REQUIRES   # reads a cut and a whole alike
    assert ext.ROUTED_FUSED_E4M3_MODULE_NAME == rf.MODULE_NAME_E4M3
    assert ext.ROUTED_FUSED_VALUE_MODULE_NAME == rf.MODULE_NAME_VALUE
    assert ext.ROUTED_FUSED_SOURCE == rf.SOURCE
    source = (Path(__file__).resolve().parents[1] / "src" / "tessera" / "routed_fused.py").read_text()
    assert 'name="tessera_routed_fused_e4m3"' in source
    assert 'name="tessera_routed_fused_value"' in source
    prefixes = {e["module_name_prefix"] for e in ext.NATIVE_EXTENSIONS}
    assert {rf.MODULE_NAME_E4M3, rf.MODULE_NAME_VALUE} <= prefixes


# --- every rung's schedule: one-hot decode oracle, derived bounds (tessera#694) ----

def _w64(stack, family):
    return [fb.fp64_weight(e, family) for e in stack]


def _a64(family, x):
    """The scaled fp64 A operand the kernel multiplies: the route's own E4M3
    quantisation times its scale, or the bf16 x."""
    if family == "e4m3":
        xq, a = _quant(x)
        return xq.double() * a.double().reshape(-1, 1)
    return x.double()


def _per_route(values, ids):
    """``[E, T, N]`` gathered to ``[T, top_k, N]`` by the routing."""
    t, k = ids.shape
    sel = torch.arange(t, device=ids.device)[:, None].expand(t, k)
    return values[ids.long(), sel]


@cuda
@pytest.mark.parametrize("family", ["value", "e4m3"])
@pytest.mark.parametrize("q256", Q256_CASES)
def test_fused_stages_decode_every_rate_exactly(family, q256):
    """One-hot rows through the real kernel, both launches: each output
    element is one decoded weight through the family's epilogue, so
    ``gate_up`` (both halves, two run tables and two block descriptors per
    item) and ``down_routes`` (top_k = 1: the one-route sum is the route) must
    equal the torch restatement of that arithmetic BITWISE for every (expert,
    column, row) -- a wrong column map, run offset or field position at any
    rate, in either run, with or without a start state, shows here."""
    stacks = _stacks(family, q256=q256)
    gate, up, down = stacks
    assert set(gate[0].rates) <= set(rf.RATES) and len(set(gate[0].rates)) in (1, 2)
    fused = _fused(_bundles(family, stacks))
    assert fused.tile_words_gate_up == 16 * sum(gate[0].rates)
    assert fused.tile_words_down == 16 * sum(down[0].rates)
    # gate/up: HIDDEN one-hot tokens, every one routed to every expert
    _x, xq, a, hot = fb.one_hot_inputs(family, HIDDEN, _quant)
    ids = torch.arange(EXPERTS, device="cuda", dtype=torch.int32).expand(HIDDEN, EXPERTS).contiguous()
    rw = torch.ones(HIDDEN, EXPERTS, device="cuda")
    gu = fused.gate_up(xq, ids, rw, a_scale=a, preserve=True)
    assert gu.shape == (HIDDEN, EXPERTS, 2 * INTER)
    for e in range(EXPERTS):
        for name, stack, half in (("gate", gate, gu[:, e, :INTER]), ("up", up, gu[:, e, INTER:])):
            want = fb.one_hot_expected(stack[e], family, hot, a)
            bad = half != want
            assert not bool(bad.any()), (
                f"{family} q256={q256} {name} expert {e}: {int(bad.sum())} of {bad.numel()} "
                f"one-hot products differ; first at {bad.nonzero()[0].tolist()} (column, row)")
    # down: INTER one-hot routes per expert, top_k = 1, unit weight
    _x, xq, a, hot = fb.one_hot_inputs(family, INTER, _quant)
    xq_all = xq.repeat(EXPERTS, 1).contiguous()
    a_all = a.repeat(EXPERTS).contiguous() if a is not None else None
    ids = torch.arange(EXPERTS, device="cuda", dtype=torch.int32).repeat_interleave(INTER).reshape(-1, 1)
    rw = torch.ones(EXPERTS * INTER, 1, device="cuda")
    out = fused.down_routes(xq_all, ids, rw, a_scale=a_all, route_input=True, round_routes=True)
    assert out.shape == (EXPERTS * INTER, HIDDEN)
    for e in range(EXPERTS):
        want = fb.one_hot_expected(down[e], family, hot, a)
        got = out[e * INTER:(e + 1) * INTER]
        bad = got != want
        assert not bool(bad.any()), (
            f"{family} q256={q256} down expert {e}: {int(bad.sum())} of {bad.numel()} one-hot "
            f"products differ; first at {bad.nonzero()[0].tolist()} (column, row)")


@cuda
@pytest.mark.parametrize("family", ["value", "e4m3"])
@pytest.mark.parametrize("q256", Q256_CASES)
def test_fused_stages_at_every_rate_are_within_the_derived_bounds(family, q256):
    """Random tokens at the rung's schedule: ``gate_up`` is within
    ``fused_bound.dense_bound`` (S = 1) of the fp64 reference per route and
    element; ``down_routes`` on the SAME quantised intermediate is within the
    per-route bound (one more multiply, the routing weight) summed by
    ``route_sum_bound``; and the forward IS the composition of the two stages
    through the host SwiGLU, bitwise (the kernel's fused epilogue rounds gate
    and up to bf16, widens, clamps and activates in fp32 as ``_silu_and_mul``
    does), so the end-to-end forward is held by the stage bounds without a
    limit of its own.  Both lanes sit inside the bound, so the compact adapter
    is within twice it of the fused lane."""
    stacks = _stacks(family, q256=q256)
    gate, up, down = stacks
    bundles = _bundles(family, stacks)
    fused, legacy = _fused(bundles), _legacy(bundles)
    t = 71
    x = torch.randn(t, HIDDEN, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(8000 + q256)).bfloat16()
    ids, rw = _routes(t, TOP_K, 8100 + q256)
    # stage 1
    a64 = _a64(family, x)
    refs, bounds = zip(*(fb.dense_bound(family, a64, w, HIDDEN, 1) for w in _w64(gate, family)))
    r_g, b_g = _per_route(torch.stack(refs), ids), _per_route(torch.stack(bounds), ids)
    refs, bounds = zip(*(fb.dense_bound(family, a64, w, HIDDEN, 1) for w in _w64(up, family)))
    r_u, b_u = _per_route(torch.stack(refs), ids), _per_route(torch.stack(bounds), ids)
    gu = fused.gate_up(x, ids, rw, preserve=True)
    what = f"{family} q256={q256}"
    fb.check_within(gu[..., :INTER], r_g, b_g, f"{what}: gate vs the fp64 reference")
    fb.check_within(gu[..., INTER:], r_u, b_u, f"{what}: up vs the fp64 reference")
    g_l = legacy.gate(x, ids, rw, preserve=True, apply_router_weight_on_input=False)
    u_l = legacy.up(x, ids, rw, preserve=True, apply_router_weight_on_input=False)
    fb.check_within(gu[..., :INTER], g_l.double(), b_g, f"{what}: gate fused vs compact", scale=2.0)
    fb.check_within(gu[..., INTER:], u_l.double(), b_u, f"{what}: up fused vs compact", scale=2.0)
    # stage 2 on the same intermediate
    act = _silu_and_mul(gu[..., :INTER].reshape(t * TOP_K, INTER),
                        gu[..., INTER:].reshape(t * TOP_K, INTER), clamp_limit=None)
    h64 = _a64(family, act)                                                # [T*K, I]
    rw64 = rw.double().reshape(t * TOP_K, 1)
    refs, bounds = zip(*(fb.dense_bound(family, h64, w, INTER, 1, weight=rw64)
                         for w in _w64(down, family)))
    flat_ids = ids.reshape(-1, 1)
    r_d = _per_route(torch.stack(refs), flat_ids).reshape(t, TOP_K, HIDDEN)
    b_d = _per_route(torch.stack(bounds), flat_ids).reshape(t, TOP_K, HIDDEN)
    # each route is rounded to bf16 before the sum: its bound already ends in
    # half a bf16 ulp; the token sum adds gamma(top_k) and one more rounding
    r_tok, b_tok = fb.route_sum_bound(r_d, b_d, top_k_dim=1)
    dn = fused.down_routes(act, ids, rw, route_input=True, round_routes=True)
    ratio = fb.check_within(dn, r_tok, b_tok, f"{what}: down vs the fp64 reference")
    dn_l = legacy.down(act, ids, rw, route_input=True, apply_router_weight_on_input=False,
                       round_routes=True)
    fb.check_within(dn, dn_l.double(), b_tok, f"{what}: down fused vs compact", scale=2.0)
    # the forward is the composition, bitwise
    out = fused(x, ids, rw)
    diff = int((out != dn).sum())
    assert diff == 0, f"{what}: the forward differs from its staged composition in {diff} elements"
    print(f"ROUTED-RATE-BOUND {what} down/bound={ratio:.4f}")
