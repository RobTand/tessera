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

from test_window_gemm_grouped import Expert, _quant, _tol  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the lane is a CUDA kernel")

L = 14
# gate/up: [INTER, HIDDEN] at rate 4 (two 512-row tiles: INTER > 512);
# down: [HIDDEN, INTER] (two tiles again; HIDDEN a multiple of 128).
HIDDEN, INTER, EXPERTS, TOP_K = 256, 576, 5, 3


def _init(cols, seed):
    return torch.randint(0, 1 << L, (cols,), generator=torch.Generator().manual_seed(seed),
                         dtype=torch.int32)


def _stacks(family, *, hidden=HIDDEN, inter=INTER, experts=EXPERTS, seed=300, cut=True):
    """gate, up, down Expert lists at rate 4 everywhere; experts 1 and 3 carry
    a start state (the TP row-cut case) when ``cut``."""
    gate = [Expert(inter, hidden, (4,) * hidden, seed + i, family=family,
                   init=_init(hidden, seed + 40 + i) if (cut and i % 2) else None)
            for i in range(experts)]
    up = [Expert(inter, hidden, (4,) * hidden, seed + 10 + i, family=family,
                 init=_init(hidden, seed + 50 + i) if (cut and i % 2) else None)
          for i in range(experts)]
    down = [Expert(hidden, inter, (4,) * inter, seed + 20 + i, family=family,
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
    # mixed rates: two runs per expert (and a permuted column order)
    mixed = [Expert(INTER, HIDDEN, tuple(2 if c % 2 else 4 for c in range(HIDDEN)), 700 + i)
             for i in range(EXPERTS)]
    mixed_b = wgg.prepare_grouped_window_gemm([e.unit for e in mixed], block_m=32, block_n=64,
                                              block_k=64, arithmetic="folded")
    reason = rf.fused_routed_window_supported(mixed_b, ok.up, ok.down)
    assert reason is not None and "run" in reason
    # a rate other than 4 everywhere: one run, but not the kernel's
    rate2 = [Expert(INTER, HIDDEN, (2,) * HIDDEN, 720 + i) for i in range(EXPERTS)]
    rate2_b = wgg.prepare_grouped_window_gemm([e.unit for e in rate2], block_m=32, block_n=64,
                                              block_k=64, arithmetic="folded")
    reason = rf.fused_routed_window_supported(rate2_b, ok.up, ok.down)
    assert reason is not None and "[[4, 0" in reason
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
    kernel: rate 4, window bits 14, and the two module names are the loader's
    literals."""
    from tessera.serving import ext

    assert ext.ROUTED_FUSED_LANE_REQUIRES["column_rates"] == [rf.RATE]
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
