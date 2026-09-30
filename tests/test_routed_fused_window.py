"""The fused routed window MoE lane (``tessera.routed_fused``, tessera#640).

Parity against the same per-expert definition the grouped GEMM tests use
(``test_window_gemm_grouped.Expert``: each expert's weights ARE the
definition, routing applied on the host), held per element to bounds derived
from the dtypes and operation counts (``fused_bound``), stage by stage on the
exact input each stage consumes, for both window families, with a
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
from test_window_gemm_grouped import Expert, _quant  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the lane is a CUDA kernel")

#: The three libraries every kernel test runs on: the value family, and the
#: E4M3 family on each tensor-core instruction -- ``e4m3`` widens the bytes to
#: f16 (``m16n8k16``), ``e4m3mma`` runs ``m16n8k32.e4m3`` on them.  The
#: ``family`` fixture turns the id into the family plus the process's
#: ``TESSERA_FUSED_E4M3_MMA`` choice.
LIBRARY_IDS = ["value", "e4m3", "e4m3mma"]


@pytest.fixture
def family(request, monkeypatch):
    """The window family of the ``LIBRARY_IDS`` id, with the E4M3 instruction
    set for the adapters the test builds."""
    lib = request.param
    monkeypatch.setenv(rf.ENV_E4M3_MMA, "e4m3" if lib == "e4m3mma" else "f16")
    return "e4m3" if lib == "e4m3mma" else lib

#: The rungs the mixed-rate routed tests read (tessera#694): the GLM E4M3
#: rungs q256 832 (rates 3/4), 928, 960 (3/4), 1088 (4/5), 1152, the one-rate
#: 768 and 1280, 1408 (5/6) and 1536 (rate 6, the largest slot the 16-bit
#: libraries' two-table gate/up launch holds at three word stages), the low
#: extremes 256 (rate 1) and 384 (1/2), and the 16-word slot above it: 1600
#: (6/7), 1792 (rate 7), 1920 (7/8) and 2048 (rate 8), which the 16-bit
#: gate/up launch runs at two word stages (``routed_fused.word_stages``) and
#: the E4M3 instruction's at three -- with 512 (rate 2), 640 (2/3) and 1024
#: (rate 4), every one-run rate 1..8 and every adjacent pair is a case, so
#: every (pair, launch) instantiation meets the oracle.  256 and 576 columns
#: realise each exactly.
Q256_CASES = [256, 384, 512, 640, 768, 832, 928, 960, 1024, 1088, 1152, 1280, 1408, 1536, 1600, 1792,
              1920, 2048]
#: The rungs the CUDA-graph capture test replays: the #640 rate-4 rung, every
#: other one-rate rung the routed lane reaches -- 256 (rate 1), 512 (rate 2),
#: 768 (rate 3: every odd half takes the aligned-pair copy), 1280 (rate 5,
#: odd at slot 12) and 1536 (rate 6, slot 12 on the two-table launch) -- and
#: the GLM two-run tables 832 and 960 (3/4), 1088 (4/5) and 1152 (4/5, half
#: and half), and the two-word-stage slot: 1792 (rate 7), 2048 (rate 8) and
#: 1600 (6/7), 1920 (7/8) -- so every rate in ``ROUTED_LANE_RATES`` replays in
#: a graph.
CAPTURE_Q256 = [1024, 256, 512, 768, 1280, 1536, 832, 960, 1088, 1152, 1792, 2048, 1600, 1920]

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


def _axis_bundles(family, stacks):
    """The same stacks through the serving loader's path: every expert's unit
    placed on a ``WindowUnitAxis`` (the start state on the unit itself, in
    original column order, as ``compact_prep.prepare_window_compact`` hands
    it over), ``finish`` and ``prepare_grouped_window_gemm_from_soa`` -- what
    ``moe_route._RankLocalPackedIntake`` builds on a TP rank > 0."""
    import dataclasses

    from tessera.native_window_moe import WindowUnitAxis

    arithmetic = "folded" if family == "value" else "epilogue"
    experts = len(stacks[0])
    parts = {"gate": ("w13", "gate_proj"), "up": ("w13", "up_proj"), "down": ("w2", "down_proj")}
    axes = {"w13": WindowUnitAxis(experts, ("gate_proj", "up_proj"), family=family),
            "w2": WindowUnitAxis(experts, ("down_proj",), family=family)}
    for name, stack in zip(("gate", "up", "down"), stacks):
        group, part = parts[name]
        for e, ex in enumerate(stack):
            state = None if ex.init is None else ex.init.to(device="cuda", dtype=torch.int32)
            axes[group].put(part, e, dataclasses.replace(ex.unit, initial_state=state))
    soa = {g: axes[g].finish() for g in axes}

    def bundle(name):
        group, part = parts[name]
        slot = soa[group][part]
        return wgg.prepare_grouped_window_gemm_from_soa(
            words_all=slot["words"], table_all=slot["table"], codes_all=slot["codes"],
            native_all=slot["native"], scale_all=slot["scale"], runs_all=slot["runs"],
            init_all=slot["init"], has_init=slot["has_init"], word_off=slot["word_off"],
            tile_words=slot["tile_words"], total_words=slot["total_words"],
            run_off=slot["run_off"], perm_all=slot["perm"], rows=slot["rows"],
            cols=slot["cols"], experts=experts, window_bits=slot["window_bits"],
            family=family, block_m=32, block_n=64, block_k=64, arithmetic=arithmetic)

    return PackedWindowMoeBundles(gate=bundle("gate"), up=bundle("up"), down=bundle("down"),
                                  family=family)


def _fused(bundles):
    return rf.FusedRoutedWindowMoE.from_bundles(bundles.gate, bundles.up, bundles.down)


def _legacy(bundles):
    return native_window_moe_from_bundles(bundles.down, gate=bundles.gate, up=bundles.up)


def _routes(t, k, seed, experts=EXPERTS):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, experts, (t, k), generator=g, dtype=torch.int32).cuda()
    rw = torch.rand(t, k, generator=g).cuda()
    return ids, rw


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


def _gate_up_bounds(stacks, family, x, ids):
    """Stage 1 per route: ``(r_gate, b_gate, r_up, b_up)``, each ``[T, top_k,
    I]`` fp64 -- ``fused_bound.dense_bound`` with ``S = 1`` (the fused lane
    runs one pass over K; the Triton lane's chunked depth is at most K), the
    family's epilogue multiplies, no routing weight (the route-preserving
    projection is unweighted), one bf16 rounding."""
    gate, up, _down = stacks
    a64 = _a64(family, x)
    out = []
    for stack in (gate, up):
        refs, bounds = zip(*(fb.dense_bound(family, a64, w, stack[0].cols, 1)
                             for w in _w64(stack, family)))
        out += [_per_route(torch.stack(refs), ids), _per_route(torch.stack(bounds), ids)]
    return out


def _down_bounds(stacks, family, act, ids, rw, *, weight_input=False):
    """Stage 2 on the route-indexed activation ``act`` ``[T * top_k, I]`` (the
    exact bf16 tensor the kernel consumes; E4M3 re-quantises it with the
    route's own quantiser): per route ``dense_bound`` with ``S = 1`` and the
    routing weight as one more multiply unless it was applied on input, each
    route rounded to bf16, then ``route_sum_bound`` (the fp32 sum of top_k
    bf16 routes -- fixed order in the fused ``token_sum``, atomics in the
    compact lane; gamma(top_k) covers either -- and one rounding)."""
    down = stacks[2]
    t, k = ids.shape
    h64 = _a64(family, act)
    weight = None if weight_input else rw.double().reshape(t * k, 1)
    refs, bounds = zip(*(fb.dense_bound(family, h64, w, down[0].cols, 1, weight=weight)
                         for w in _w64(down, family)))
    flat_ids = ids.reshape(-1, 1)
    hidden = down[0].rows
    r_d = _per_route(torch.stack(refs), flat_ids).reshape(t, k, hidden)
    b_d = _per_route(torch.stack(bounds), flat_ids).reshape(t, k, hidden)
    return fb.route_sum_bound(r_d, b_d, top_k_dim=1)


def _staged_check(stacks, bundles, x, ids, rw, family, what, *, limit=None,
                  weight_input=False, compact=True):
    """Hold the routed lanes to derived bounds, stage by stage, on the exact
    input each stage consumes; returns ``(gate_up, act, down, forward)`` of
    the fused lane.

    * ``gate_up`` per route and element within :func:`_gate_up_bounds` of the
      fp64 reference;
    * ``down_routes`` on the SwiGLU of that output (``_silu_and_mul``, clamp
      included) within :func:`_down_bounds`;
    * the fused forward EQUAL, bitwise, to that staged composition: its
      epilogue rounds gate and up to bf16, widens, clamps and activates in
      fp32 with one rounding, as ``_silu_and_mul`` does, and ``token_sum`` is
      fixed-order -- so the end-to-end forward is held by the stage bounds
      without a limit of its own;
    * ``compact``: the compact adapter's gate and up within the stage-1 bound
      (and within twice it of the fused lane: both sit inside it); its own
      forward within the stage-2 bound of ITS OWN staged activation (its
      preserve stages store each route once -- no atomics -- so the forward
      consumes that activation); and its down on the fused lane's activation
      within twice the fused down's bound.

    With ``weight_input`` (top_k = 1) the weight scales x in bf16 before the
    quantiser, as both lanes' forwards do, and no stage applies it again.
    """
    fused = _fused(bundles)
    t, k = ids.shape
    inter = stacks[0][0].rows
    x_in = x * rw.reshape(-1, 1).to(x.dtype) if weight_input else x
    r_g, b_g, r_u, b_u = _gate_up_bounds(stacks, family, x_in, ids)
    gu = fused.gate_up(x_in, ids, rw, preserve=True)
    ratios = {"gate": fb.check_within(gu[..., :inter], r_g, b_g,
                                      f"{what}: fused gate vs the fp64 reference"),
              "up": fb.check_within(gu[..., inter:], r_u, b_u,
                                    f"{what}: fused up vs the fp64 reference")}
    act = _silu_and_mul(gu[..., :inter].reshape(t * k, inter),
                        gu[..., inter:].reshape(t * k, inter), clamp_limit=limit)
    r_tok, b_tok = _down_bounds(stacks, family, act, ids, rw, weight_input=weight_input)
    dn = fused.down_routes(act, ids, rw, route_input=True,
                           apply_router_weight_on_input=weight_input, round_routes=True)
    ratios["down"] = fb.check_within(dn, r_tok, b_tok, f"{what}: fused down vs the fp64 reference")
    out = fused(x, ids, rw, apply_router_weight_on_input=weight_input, swiglu_limit=limit)
    diff = int((out != dn).sum())
    assert diff == 0, f"{what}: the forward differs from its staged composition in {diff} elements"
    if compact:
        legacy = _legacy(bundles)
        g_l = legacy.gate(x_in, ids, rw, preserve=True, apply_router_weight_on_input=False)
        u_l = legacy.up(x_in, ids, rw, preserve=True, apply_router_weight_on_input=False)
        ratios["compact gate"] = fb.check_within(g_l, r_g, b_g,
                                                 f"{what}: compact gate vs the fp64 reference")
        ratios["compact up"] = fb.check_within(u_l, r_u, b_u,
                                               f"{what}: compact up vs the fp64 reference")
        ratios["gate pair"] = fb.check_within(gu[..., :inter], g_l.double(), b_g,
                                              f"{what}: gate fused vs compact", scale=2.0)
        ratios["up pair"] = fb.check_within(gu[..., inter:], u_l.double(), b_u,
                                            f"{what}: up fused vs compact", scale=2.0)
        act_l = _silu_and_mul(g_l.reshape(t * k, inter), u_l.reshape(t * k, inter),
                              clamp_limit=limit)
        r_l, b_l = _down_bounds(stacks, family, act_l, ids, rw, weight_input=weight_input)
        out_l = legacy(x, ids, rw, apply_router_weight_on_input=weight_input, swiglu_limit=limit)
        ratios["compact forward"] = fb.check_within(
            out_l, r_l, b_l, f"{what}: compact forward vs the fp64 reference of its own stages")
        dn_l = legacy.down(act, ids, rw, route_input=True,
                           apply_router_weight_on_input=weight_input, round_routes=True)
        ratios["down pair"] = fb.check_within(dn, dn_l.double(), b_tok,
                                              f"{what}: down fused vs compact", scale=2.0)
    print(f"ROUTED-BOUND {what} " + " ".join(f"{name.replace(' ', '_')}={v:.4f}"
                                             for name, v in ratios.items()))
    return gu, act, dn, out


# --- parity ------------------------------------------------------------------

@cuda
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
@pytest.mark.parametrize("t", [1, 7, 71])
def test_fused_forward_matches_the_per_expert_oracle_and_the_compact_adapter(family, t):
    """M = 1 (decode), a short batch, and 71 x 3 = 213 routes so at least one
    expert spans two 64-route superblocks; experts 1 and 3 carry a start
    state.  Both lanes are held stage by stage to the derived bounds of the
    fp64 definition (``_staged_check``); the fused forward is its staged
    composition bitwise, and the two lanes agree within twice the bound."""
    stacks = _stacks(family)
    bundles = _bundles(family, stacks)
    x = torch.randn(t, HIDDEN, device="cuda").bfloat16()
    ids, rw = _routes(t, TOP_K, 900 + t)
    *_stages, out = _staged_check(stacks, bundles, x, ids, rw, family, f"{family} T={t}")
    assert out.shape == (t, HIDDEN) and out.dtype == torch.bfloat16


@cuda
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
def test_fused_swiglu_clamp_is_the_stock_placement(family):
    stacks = _stacks(family, cut=False)
    bundles = _bundles(family, stacks)
    fused = _fused(bundles)
    x = (torch.randn(24, HIDDEN, device="cuda") * 3).bfloat16()
    ids, rw = _routes(24, TOP_K, 77)
    limit = 0.5
    *_stages, out = _staged_check(stacks, bundles, x, ids, rw, family, f"{family} clamp",
                                  limit=limit)
    # the clamp changes the answer, so a lane that ignored it could not pass
    assert not torch.equal(out, fused(x, ids, rw))


@cuda
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
def test_fused_empty_and_repeated_experts(family):
    """Every route to one expert (the others empty) and a batch whose experts
    are all hit: the device work list is sized by route counts, never by E."""
    stacks = _stacks(family, cut=False)
    bundles = _bundles(family, stacks)
    fused = _fused(bundles)
    x = torch.randn(40, HIDDEN, device="cuda").bfloat16()
    rw = torch.rand(40, TOP_K, device="cuda")
    ids = torch.full((40, TOP_K), 2, dtype=torch.int32, device="cuda")
    _staged_check(stacks, bundles, x, ids, rw, family, f"{family} one expert", compact=False)
    ids = torch.arange(40 * TOP_K, device="cuda", dtype=torch.int32).reshape(40, TOP_K) % EXPERTS
    _staged_check(stacks, bundles, x, ids, rw, family, f"{family} all experts", compact=False)
    empty = fused(x[:0], ids[:0], rw[:0])
    assert empty.shape == (0, HIDDEN) and empty.dtype == torch.bfloat16


@cuda
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
def test_fused_staged_interfaces_match_the_compact_adapters(family):
    """``gate_up`` (route-preserving) and ``down_routes`` (reduced) are the
    stages the routed pair oracle teacher-forces; each lane's stage sits
    inside the derived bound, so the fused and compact stages agree within
    twice it, and their composition is the forward bitwise
    (``_staged_check``)."""
    stacks = _stacks(family)
    bundles = _bundles(family, stacks)
    fused = _fused(bundles)
    t = 33
    x = torch.randn(t, HIDDEN, device="cuda").bfloat16()
    ids, rw = _routes(t, TOP_K, 5)
    gu, act, down, _out = _staged_check(stacks, bundles, x, ids, rw, family, f"{family} staged")
    assert gu.shape == (t, TOP_K, 2 * INTER) and gu.dtype == torch.bfloat16
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
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
def test_fused_weight_on_input_is_the_modular_prepare_placement(family):
    stacks = _stacks(family, cut=False)
    bundles = _bundles(family, stacks)
    fused = _fused(bundles)
    t = 19
    x = torch.randn(t, HIDDEN, device="cuda").bfloat16()
    ids, rw = _routes(t, 1, 11)
    _staged_check(stacks, bundles, x, ids, rw, family, f"{family} weight on input",
                  weight_input=True)
    ids3, rw3 = _routes(t, 3, 12)
    with pytest.raises(GrammarError, match="topk=1"):
        fused(x, ids3, rw3, apply_router_weight_on_input=True)


# --- determinism and graphs ------------------------------------------------------

@cuda
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
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
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
@pytest.mark.parametrize("q256", CAPTURE_Q256)
def test_fused_forward_captures_and_replays_twice_against_eager(family, q256):
    """The device work counter is zeroed INSIDE the captured region, so a
    replay starts a fresh work list; two replays must equal the eager forward
    on the same (static) inputs -- bitwise, the lane being deterministic.  At
    every rung of ``CAPTURE_Q256``: the run pairs, block descriptors and the
    per-launch slot are launch arguments and device tensors the graph holds,
    so a mixed-rate stack replays exactly like the rate-4 one."""
    stacks = _stacks(family, q256=q256)
    fused = _fused(_bundles(family, stacks))
    rates = set(stacks[0][0].rates) | set(stacks[2][0].rates)
    assert rates <= set(rf.ROUTED_LANE_RATES), (q256, sorted(rates))
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
    mixed_rates = tuple(3 if c % 2 else 4 for c in range(HIDDEN))
    mixed = grouped([Expert(INTER, HIDDEN, mixed_rates, 710 + i) for i in range(EXPERTS)])
    reason = rf.fused_routed_window_supported(mixed, ok.up, ok.down)
    assert reason is not None and "tile_words" in reason
    mixed_up = grouped([Expert(INTER, HIDDEN, mixed_rates, 720 + i) for i in range(EXPERTS)])
    assert rf.fused_routed_window_supported(mixed, mixed_up, ok.down) is None
    # ... when the two rates are ADJACENT, the pair bracketing a root that every
    # grammar schedule emits: the kernel is instantiated per (low rate, one
    # or two runs), so a wider pair is refused by name, not decoded
    wide_rates = tuple(2 if c % 2 else 4 for c in range(HIDDEN))
    wide = grouped([Expert(INTER, HIDDEN, wide_rates, 760 + i) for i in range(EXPERTS)])
    wide_up = grouped([Expert(INTER, HIDDEN, wide_rates, 770 + i) for i in range(EXPERTS)])
    reason = rf.fused_routed_window_supported(wide, wide_up, ok.down)
    assert reason is not None and "not adjacent" in reason, reason
    # one rate other than 4 everywhere is one run the kernel reads
    rate2 = grouped([Expert(INTER, HIDDEN, (2,) * HIDDEN, 730 + i) for i in range(EXPERTS)])
    rate2_up = grouped([Expert(INTER, HIDDEN, (2,) * HIDDEN, 740 + i) for i in range(EXPERTS)])
    assert rf.fused_routed_window_supported(rate2, rate2_up, ok.down) is None
    # experts that disagree on their schedule: the kernel reads one run pair per
    # stack.  Half the columns at 3 and half at 5, or half at 2 and half at 6,
    # carry the same words (two runs each), so the wire is [E, W] with two runs
    # per expert and the run tables are what differ.
    three_five = tuple(3 if c % 2 else 5 for c in range(HIDDEN))
    two_six = tuple(2 if c % 2 else 6 for c in range(HIDDEN))
    uneven = grouped([Expert(INTER, HIDDEN, three_five if i % 2 else two_six, 750 + i)
                      for i in range(EXPERTS)])
    reason = rf.fused_routed_window_supported(uneven, ok.up, ok.down)
    assert reason is not None and "disagree on their run tables" in reason
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
    column's slot is its ``2 * rate`` words, plus the two words the odd-rate
    copies start early by (an odd rate's half is 8-byte aligned at odd half
    indices, and the copies are 16-byte pieces from the aligned pair before
    it); a launch's slot is the larger of its pair's, rounded to a multiple of
    4; the word stages sit after the fixed part (two tables for gate/up, one
    for down/dense) and their size follows the slot -- three stages where
    they fit sm_121's block, two where they do not (the 16-bit gate/up
    launch at the 16-word slot of rates 7 and 8)."""
    assert [rf.slot_words_for_rate(r) for r in rf.RATES] == [4, 4, 8, 8, 12, 12, 16, 16]

    def pair(r_lo, r_hi=0, n_hi=0):
        return torch.tensor([r_lo, 0, 64, 0, r_hi, 64, n_hi, 0], dtype=torch.int32)

    assert [rf.slot_words_for_pair(pair(r)) for r in rf.RATES] == [4, 4, 8, 8, 12, 12, 16, 16]
    assert [rf.slot_words_for_pair(pair(lo, lo + 1, 32)) for lo in range(1, 8)] == [4, 8, 8, 12, 12, 16, 16]
    assert rf.slot_words_for_pair(pair(1, 8, 32)) == 16
    # a pair whose high run is empty is the low rate's slot alone
    assert rf.slot_words_for_pair(pair(3, 8, 0)) == 8
    assert rf.smem_bytes(0, 8) == rf.smem_bytes(1, 8) == 91_600 + 3 * 2 * rf.BK * 8 * 4 == 97_744
    assert rf.smem_bytes(0, 12) == 100_816          # rates 5 and 6: 560 B under sm_121's 101,376
    # three stages of the 16-word slot would take 103,888 B, over sm_121's
    # 101,376: the 16-bit gate/up launch runs rates 7 and 8 at two
    assert rf.SMEM_FIXED[0] + rf.WORD_STAGES * 2 * rf.BK * 16 * 4 == 103_888 > rf.SM121_MAX_DYNAMIC_SMEM
    assert [rf.word_stages(0, sw) for sw in (4, 8, 12, 16)] == [3, 3, 3, 2] == \
        [rf.word_stages(1, sw) for sw in (4, 8, 12, 16)]
    assert rf.smem_bytes(0, 16) == rf.smem_bytes(1, 16) == 91_600 + 2 * 2 * rf.BK * 16 * 4 == 99_792
    assert rf.smem_bytes(2, 16) == 70_928           # the one-table down/dense launch: three stages at every rate
    assert all(rf.word_stages(2, sw) == rf.WORD_STAGES for sw in (4, 8, 12, 16))
    assert all(rf.word_stages(m, 16, mma8=True) == rf.WORD_STAGES for m in (0, 1, 2))
    assert rf.WORD_STAGES_MIN == 2 < rf.WORD_STAGES == 3
    # the fixed parts differ by a table and the gate/up launch's second
    # projection in the descriptor ring: 4 chunks x 12 int32 x 4 B
    assert rf.SMEM_FIXED[0] - rf.SMEM_FIXED[2] == 32_768 + rf.DRING_STAGES * rf.BDESC_INTS * 4
    assert rf.SLOT_WORDS_MAX == rf.slot_words_for_rate(rf.RATE_MAX) == 16


class _SmallerDevice:
    """A built library whose device reports ``have`` bytes of opt-in shared
    memory: the refusal path of a part with less than sm_121's."""

    def __init__(self, lib, have):
        self._lib, self._have = lib, int(have)

    def max_dynamic_smem_bytes(self, _index):
        return self._have

    def __getattr__(self, name):
        return getattr(self._lib, name)


@cuda
def test_support_predicate_admits_the_gate_up_slots_the_device_holds(monkeypatch):
    """Every gate/up slot fits sm_121's opt-in shared memory at its word
    stages -- the 16-word slot of rates 7 and 8 at two -- so a rate-7 stack
    (q256 1792) is admitted on the value library, and the library's own
    layout functions are the host's.  A device that holds less is refused by
    name, the launch and both byte counts in the reason (the compact adapter
    serves it)."""
    lib = rf._ext("value")
    have = int(lib.max_dynamic_smem_bytes(torch.cuda.current_device()))
    for mode, sw in ((0, 12), (0, 16), (2, 16)):
        assert int(lib.smem_bytes(mode, sw)) == rf.smem_bytes(mode, sw)
        assert int(lib.word_stages(mode, sw)) == rf.word_stages(mode, sw)
    # predicate == loader on the target: the rates whose one-rate gate/up slot
    # this device holds are exactly the published column_rates_routed_moe
    admitted = tuple(r for r in rf.RATES
                     if rf.smem_bytes(0, rf._round_up_4(rf.slot_words_for_rate(r))) <= have)
    if have == rf.SM121_MAX_DYNAMIC_SMEM:
        assert admitted == rf.ROUTED_LANE_RATES == rf.RATES
    else:  # another device: the published set is sm_121's, and this test says which device it ran on
        assert admitted, have
    stacks = {q256: _bundles("value", _stacks("value", q256=q256, cut=False)) for q256 in (1792, 1536, 1024)}
    for q256, b in stacks.items():
        slot = rf.slot_words_for_rate(q256 // 256)
        if rf.smem_bytes(0, slot) <= have:
            assert rf.fused_routed_window_supported(b.gate, b.up, b.down) is None, q256
    # a device with the 8-word slot's three stages and no more: rates 5..8
    # refused by name, rate 4 admitted
    small = rf.smem_bytes(0, 8)
    real = rf._ext
    monkeypatch.setattr(rf, "_ext", lambda library, *a, **k: _SmallerDevice(real(library, *a, **k), small))
    for q256, slot in ((1792, 16), (1536, 12)):
        b = stacks[q256]
        reason = rf.fused_routed_window_supported(b.gate, b.up, b.down)
        assert reason is not None and "shared memory" in reason and "gate/up" in reason \
            and str(rf.smem_bytes(0, slot)) in reason and str(small) in reason, reason
    b = stacks[1024]
    assert rf.fused_routed_window_supported(b.gate, b.up, b.down) is None


@cuda
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
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
    assert adapter.library == rf.library_for(family)
    assert decoder == {"value": "native_routed_fused_window_folded",
                       "e4m3": "native_routed_fused_window",
                       "e4m3mma": "native_routed_fused_window_e4m3mma"}[adapter.library]
    # the tables are the library's: 16-bit, or the E4M3 bytes on the E4M3 instruction
    want_dtype = torch.uint8 if adapter.library == "e4m3mma" else torch.int16
    assert adapter.table_gate.dtype == adapter.table_down.dtype == want_dtype
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
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
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
    # the routed-expert launch's set is DERIVED from the kernel's shared-memory
    # layout at the target's opt-in limit, and published equal to it
    assert ext.ROUTED_FUSED_LANE_REQUIRES["column_rates_routed_moe"] == list(rf.ROUTED_LANE_RATES) \
        == list(range(1, 9))
    assert rf.ROUTED_LANE_RATES == tuple(
        r for r in rf.RATES
        if rf.smem_bytes(0, rf.slot_words_for_pair(torch.tensor([r, 0, 64, 0, 0, 64, 0, 0], dtype=torch.int32)))
        <= rf.SM121_MAX_DYNAMIC_SMEM)
    assert all(rf.smem_bytes(2, rf.slot_words_for_rate(r)) <= rf.SM121_MAX_DYNAMIC_SMEM for r in rf.RATES)
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

@cuda
@pytest.mark.parametrize("family", ["value", "e4m3"])
@pytest.mark.parametrize("q256", Q256_CASES)
def test_the_loader_axis_stores_the_start_state_the_kernels_read(family, q256):
    """The serving loader's ``WindowUnitAxis`` stack and the per-unit
    ``prepare_grouped_window_gemm`` stack are one kernel input: both kernels
    (the fused lane's ``init[p]`` and the grouped GEMM's ``init_all[e, kglob]``)
    read the start state at the REPACKED column, so the axis must hold
    ``permuted_start_state()``, not the unit's original-order state.  On a
    mixed-rate rung the repack permutes columns, and an original-order state
    corrupts every row of a TP row cut whose window still holds pre-cut bits:
    rows t with (t + 1) * rate < L, i.e. ceil(L / rate) - 1 of them
    (tessera#729)."""
    stacks = _stacks(family, q256=q256)
    ref, axis = _bundles(family, stacks), _axis_bundles(family, stacks)
    for name in ("gate", "up", "down"):
        a, b = getattr(ref, name), getattr(axis, name)
        assert torch.equal(a.perm_all, b.perm_all), f"q256={q256} {name}: perm differs"
        assert torch.equal(a.has_init, b.has_init), f"q256={q256} {name}: has_init differs"
        bad = a.init_all != b.init_all
        assert not bool(bad.any()), (
            f"{family} q256={q256} {name}: the axis start state differs from the prepared "
            f"stack's in {int(bad.sum())} of {bad.numel()} (expert, repacked column) entries")


@cuda
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
@pytest.mark.parametrize("q256", Q256_CASES)
@pytest.mark.parametrize("build", ["prepare", "axis"])
def test_fused_stages_decode_every_rate_exactly(family, q256, build):
    """One-hot rows through the real kernel, both launches: each output
    element is one decoded weight through the family's epilogue, so
    ``gate_up`` (both halves, two run tables and two block descriptors per
    item) and ``down_routes`` (top_k = 1: the one-route sum is the route) must
    equal the torch restatement of that arithmetic BITWISE for every (expert,
    column, row) -- a wrong column map, run offset or field position at any
    rate, in either run, with or without a start state, shows here.  ``axis``
    builds the stack the way the serving loader does (``_axis_bundles``).  On
    the E4M3 instruction it is also the operand-layout oracle: a byte of the
    B or A tile in the wrong k or n slot of a fragment moves a one-hot
    product to another output element."""
    _decode_exact(family, q256, build)


def _decode_exact(family, q256, build):
    stacks = _stacks(family, q256=q256)
    gate, up, down = stacks
    assert set(gate[0].rates) <= set(rf.RATES) and len(set(gate[0].rates)) in (1, 2)
    fused = _fused((_bundles if build == "prepare" else _axis_bundles)(family, stacks))
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
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
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
    _rate_bound(family, q256)


def _rate_bound(family, q256):
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


# --- the E4M3 instruction (``tessera_routed_fused_mma_e4m3``) ---------------------

#: The rungs at the 16-word gate/up slot: q256 1792 is rate 7, 1920 the 7/8
#: pair, 2048 rate 8.  The E4M3 instruction's 16 KB byte tables hold three
#: word stages of it (58,832 B against sm_121's 101,376 B); the 16-bit
#: libraries' 32 KB tables hold two (99,792 B; three would be 103,888 B).
SLOT16_Q256 = [1792, 1920, 2048]


def test_library_for_reads_the_instruction_choice(monkeypatch):
    monkeypatch.delenv(rf.ENV_E4M3_MMA, raising=False)
    # The E4M3 instruction is the default: unset reads as ``e4m3``.
    assert rf.library_for("e4m3") == "e4m3mma" and rf.library_for("value") == "value"
    monkeypatch.setenv(rf.ENV_E4M3_MMA, "f16")
    assert rf.library_for("e4m3") == "e4m3"
    monkeypatch.setenv(rf.ENV_E4M3_MMA, "e4m3")
    assert rf.library_for("e4m3") == "e4m3mma" and rf.library_for("value") == "value"
    monkeypatch.setenv(rf.ENV_E4M3_MMA, "fp8")
    with pytest.raises(GrammarError, match=rf.ENV_E4M3_MMA):
        rf.library_for("e4m3")
    assert rf.library_for("value") == "value"      # the value family has one library


def test_the_e4m3_instructions_layout_and_rates():
    """The E4M3 instruction's library halves the table and operand-tile bytes:
    two 16 KB tables for gate/up where the 16-bit libraries hold 32 KB, and
    8-bit A and B stages.  Its gate/up launch therefore fits every rate, and
    the published ``column_rates_routed_moe`` of its extension entry is
    derived from that layout like the 16-bit entries' is."""
    from tessera.serving import ext

    # one byte less per table entry and per operand element: 16 KB per table,
    # and the two operand stages' B (BK x BN) and A (BM x BK) tiles
    stages = 2 * (rf.BK * rf.BN + rf.BM * rf.BK)
    assert rf.SMEM_FIXED[0] - rf.SMEM_FIXED_MMA8[0] == 2 * 16_384 + stages
    assert rf.SMEM_FIXED[2] - rf.SMEM_FIXED_MMA8[2] == 16_384 + stages
    assert rf.smem_bytes(0, 16, mma8=True) == 58_832 <= rf.SM121_MAX_DYNAMIC_SMEM
    assert rf.routed_lane_rates("e4m3mma") == rf.RATES
    assert rf.routed_lane_rates("e4m3") == rf.routed_lane_rates("value") == rf.ROUTED_LANE_RATES
    assert ext.ROUTED_FUSED_MMA_E4M3_LANE_REQUIRES["column_rates_routed_moe"] \
        == list(rf.routed_lane_rates("e4m3mma"))
    assert {k: v for k, v in ext.ROUTED_FUSED_MMA_E4M3_LANE_REQUIRES.items()
            if k != "column_rates_routed_moe"} == {k: v for k, v in ext.ROUTED_FUSED_LANE_REQUIRES.items()
                                                   if k != "column_rates_routed_moe"}
    assert ext.ROUTED_FUSED_MMA_E4M3_MODULE_NAME == rf.MODULE_NAME_E4M3MMA
    source = (Path(__file__).resolve().parents[1] / "src" / "tessera" / "routed_fused.py").read_text()
    assert 'name="tessera_routed_fused_mma_e4m3"' in source
    # no other entry's glob matches this library's file, nor this one's theirs
    import fnmatch
    entries = {e["module_name_prefix"]: e["filename_glob"] for e in ext.NATIVE_EXTENSIONS}
    for name in entries:
        hits = [p for p, g in entries.items() if fnmatch.fnmatch(f"{name}.so", g)]
        assert hits == [name], (name, hits)


@cuda
@pytest.mark.parametrize("q256", SLOT16_Q256)
def test_every_library_serves_gate_up_at_rates_7_and_8(q256, monkeypatch):
    """Every library admits the 16-word gate/up slot: the 16-bit ones at two
    word stages, the E4M3 instruction's at three, each as its own layout
    says.  The one-hot oracle, the derived bounds and graph replay at these
    rungs are ``Q256_CASES`` / ``CAPTURE_Q256`` on every library."""
    for library, family, choice in (("value", "value", None), ("e4m3", "e4m3", "f16"),
                                    ("e4m3mma", "e4m3", "e4m3")):
        if choice is not None:
            monkeypatch.setenv(rf.ENV_E4M3_MMA, choice)
        b = _bundles(family, _stacks(family, q256=q256, cut=False))
        assert rf.fused_routed_window_supported(b.gate, b.up, b.down) is None, library
        fused = _fused(b)
        assert fused.library == library
        mma8 = rf.library_mma8(library)
        lib = rf._ext(library)
        stages = int(lib.word_stages(0, fused.slot_words_gate_up))
        assert stages == rf.word_stages(0, fused.slot_words_gate_up, mma8=mma8) \
            == (rf.WORD_STAGES if mma8 else rf.WORD_STAGES_MIN), library
        assert int(lib.launch_smem_bytes(0, fused.slot_words_gate_up, rf.BM)) \
            == rf.launch_smem_bytes(0, fused.slot_words_gate_up, mma8=mma8) <= rf.SM121_MAX_DYNAMIC_SMEM


@cuda
@pytest.mark.parametrize("q256", [1024, 832, 1088, 1536])
def test_the_two_e4m3_instructions_differ_by_accumulation_order_only(q256, monkeypatch):
    """The same stack on both E4M3 libraries, the same inputs: the products
    are the same exact values (an E4M3 byte widens to f16 exactly, and a
    product of two E4M3 values is exact in fp32), so the two answers differ
    only by where the fp32 sums round -- 16 products per ``m16n8k16``, 32 per
    ``m16n8k32``.  Each is within the derived bound of the fp64 definition
    (``fused_bound.dense_bound``, S = 1, which covers any order of K fp32
    additions), so they are within twice it of each other; the ratio is
    printed as the measured size of the difference."""
    stacks = _stacks("e4m3", q256=q256)
    monkeypatch.setenv(rf.ENV_E4M3_MMA, "f16")
    f16 = _fused(_bundles("e4m3", stacks))
    monkeypatch.setenv(rf.ENV_E4M3_MMA, "e4m3")
    mma8 = _fused(_bundles("e4m3", stacks))
    assert (f16.library, mma8.library) == ("e4m3", "e4m3mma")
    assert f16.launch_pair[1] != mma8.launch_pair[1]
    t = 71
    x = torch.randn(t, HIDDEN, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(9000 + q256)).bfloat16()
    ids, rw = _routes(t, TOP_K, 9100 + q256)
    gate, up, down = stacks
    a64 = _a64("e4m3", x)
    what = f"e4m3 q256={q256}"
    ratios = {}
    gu = {}
    for name, lane in (("f16", f16), ("mma8", mma8)):
        gu[name] = lane.gate_up(x, ids, rw, preserve=True)
    for half, stack, sl in (("gate", gate, slice(0, INTER)), ("up", up, slice(INTER, 2 * INTER))):
        refs, bounds = zip(*(fb.dense_bound("e4m3", a64, w, HIDDEN, 1) for w in _w64(stack, "e4m3")))
        r, b = _per_route(torch.stack(refs), ids), _per_route(torch.stack(bounds), ids)
        for name in gu:
            ratios[f"{half} {name}"] = fb.check_within(gu[name][..., sl], r, b, f"{what}: {half} {name}")
        ratios[f"{half} pair"] = fb.check_within(gu["mma8"][..., sl], gu["f16"][..., sl].double(), b,
                                                 f"{what}: {half} mma8 vs f16", scale=2.0)
        ratios[f"{half} equal"] = float((gu["mma8"][..., sl] == gu["f16"][..., sl]).double().mean())
    # stage 2 on ONE intermediate (the f16 lane's), both libraries
    act = _silu_and_mul(gu["f16"][..., :INTER].reshape(t * TOP_K, INTER),
                        gu["f16"][..., INTER:].reshape(t * TOP_K, INTER), clamp_limit=None)
    r_tok, b_tok = _down_bounds(stacks, "e4m3", act, ids, rw)
    dn = {name: lane.down_routes(act, ids, rw, route_input=True, round_routes=True)
          for name, lane in (("f16", f16), ("mma8", mma8))}
    for name in dn:
        ratios[f"down {name}"] = fb.check_within(dn[name], r_tok, b_tok, f"{what}: down {name}")
    ratios["down pair"] = fb.check_within(dn["mma8"], dn["f16"].double(), b_tok,
                                          f"{what}: down mma8 vs f16", scale=2.0)
    ratios["down equal"] = float((dn["mma8"] == dn["f16"]).double().mean())
    print(f"E4M3-MMA-PAIR {what} " + " ".join(f"{k.replace(' ', '_')}={v:.4f}" for k, v in ratios.items()))


# --- the wide superblock (tessera#741) ------------------------------------------

def test_the_superblock_width_is_a_host_choice_of_the_launch(monkeypatch):
    """``superblock_rows`` reads host integers and the environment only (so a
    captured forward records the width with its shapes).  128 routes exist on
    the E4M3 family's one-table launch in both libraries and on its gate/up
    launch on the E4M3 instruction; ``auto`` takes them from
    ``WIDE_MIN_ROWS`` routed tokens on the E4M3 instruction and from
    ``WIDE_UNMEASURED`` elsewhere (dense rows, the f16 instruction), ``1``
    always, ``0`` never.  The wide A region is one
    more A tile per stage (8 KB on 16-bit tiles, 4 KB on 8-bit ones), which
    every launch that has the width holds at every slot it decodes; the
    published formula and the lanes' rates do not move."""
    wide_modes = {"value": (), "e4m3": (2,), "e4m3mma": (0, 1, 2)}
    monkeypatch.delenv(rf.ENV_WIDE, raising=False)
    for library, modes in wide_modes.items():
        for mode in (0, 1, 2):
            has = mode in modes
            assert rf.has_width(library, mode, rf.BM) and rf.has_width(library, mode, rf.BM_WIDE) == has
            assert not rf.has_width(library, mode, 96)
            floor = rf.WIDE_MIN_ROWS if rf.library_mma8(library) else rf.WIDE_UNMEASURED
            assert rf.superblock_rows(library, mode, floor) == (rf.BM_WIDE if has else rf.BM)
            assert rf.superblock_rows(library, mode, floor - 1) == rf.BM
            assert rf.superblock_rows(library, mode, rf.WIDE_UNMEASURED - 1, dense=True) == rf.BM
            assert rf.superblock_rows(library, mode, rf.WIDE_UNMEASURED, dense=True) \
                == (rf.BM_WIDE if has else rf.BM)
    monkeypatch.setenv(rf.ENV_WIDE, "1")
    for library, modes in wide_modes.items():
        assert [rf.superblock_rows(library, mode, 1) for mode in (0, 1, 2)] \
            == [rf.BM_WIDE if mode in modes else rf.BM for mode in (0, 1, 2)], library
    monkeypatch.setenv(rf.ENV_WIDE, "0")
    assert rf.superblock_rows("e4m3mma", 0, rf.WIDE_UNMEASURED) == rf.BM
    monkeypatch.setenv(rf.ENV_WIDE, "yes")
    with pytest.raises(GrammarError, match=rf.ENV_WIDE):
        rf.superblock_rows("e4m3", 2, 1)
    for library, modes in wide_modes.items():
        mma8 = rf.library_mma8(library)
        extra = rf.BM * rf.BK * 2 * (1 if mma8 else 2)
        assert rf.a_region_bytes(rf.BM_WIDE, mma8=mma8) - rf.a_region_bytes(rf.BM, mma8=mma8) == extra
        for mode in modes:
            rates = rf.RATES if mode == 2 else rf.routed_lane_rates(library)
            for sw in sorted({rf._round_up_4(max(rf.slot_words_for_rate(r), 4)) for r in rates}):
                narrow = rf.launch_smem_bytes(mode, sw, mma8=mma8)
                wide = rf.launch_smem_bytes(mode, sw, mma8=mma8, bm=rf.BM_WIDE)
                assert wide == narrow + extra <= rf.SM121_MAX_DYNAMIC_SMEM, (library, mode, sw)
    # the 16-bit gate/up layout cannot hold it even at the smallest slot
    assert rf.smem_bytes(0, 4) + 8192 > rf.SM121_MAX_DYNAMIC_SMEM
    assert rf.smem_bytes(2, 16) == 70_928
    assert rf.ROUTED_LANE_RATES == rf.RATES


def _skewed(ids):
    """Two thirds of the routes onto expert 0: superblocks of every fill."""
    return torch.where(ids < 3, torch.zeros_like(ids), ids)


#: (library id, q256): the GLM rungs on both E4M3 libraries -- rate 4 and the
#: two-run 3/4 and 4/5 tables -- and rates 7 and 8 on the E4M3 instruction,
#: whose wide gate/up launch runs the largest slot.
WIDE_CASES = ([(lib, q) for lib in ("e4m3", "e4m3mma") for q in (1024, 832, 1088)]
              + [("e4m3mma", q) for q in SLOT16_Q256])


@cuda
@pytest.mark.parametrize("family,q256", WIDE_CASES, indirect=["family"])
def test_wide_superblocks_are_bitwise_the_64_route_launch(family, q256, monkeypatch):
    """At 128-route superblocks every route's row sees the same MMAs in the
    same K order as at 64, and its own epilogue, so the forward (gate/up then
    down) and the teacher-forced ``gate_up`` and ``down_routes`` stages are
    the 64-route launch's bits: from one route to superblocks past 128 routes
    (300 x 3 routes, two thirds on one expert), so the last wide superblock
    of an expert holds fewer than 64 routes in some cases and more in
    others.  A forward captured at 128 replays to the eager
    64-route answer.  The oracle and bound tests hold the 64-route launch, so
    this carries them over."""
    stacks = _stacks(family, q256=q256)
    fused = _fused(_bundles(family, stacks))
    fills = set()
    for t, skew in ((1, False), (40, False), (71, True), (150, False), (300, True)):
        x = torch.randn(t, HIDDEN, device="cuda").bfloat16()
        ids, rw = _routes(t, TOP_K, 4100 + t)
        if skew:
            ids = _skewed(ids)
        fills |= {c % rf.BM_WIDE or rf.BM_WIDE for c in torch.bincount(ids.reshape(-1).long()).tolist() if c}
        act = torch.randn(t * TOP_K, INTER, device="cuda").bfloat16()
        got = {}
        for setting in ("0", "1"):
            monkeypatch.setenv(rf.ENV_WIDE, setting)
            got[setting] = (fused(x, ids, rw), fused.gate_up(x, ids, rw, preserve=True),
                            fused.down_routes(act, ids, rw))
        for name, narrow, wide in zip(("forward", "gate_up", "down_routes"), got["0"], got["1"]):
            assert torch.equal(wide, narrow), (family, q256, t, skew, name)
    assert min(fills) < rf.BM < max(fills), fills
    t = 300
    x = torch.randn(t, HIDDEN, device="cuda").bfloat16()
    ids, rw = _routes(t, TOP_K, 4500)
    ids = _skewed(ids)
    monkeypatch.setenv(rf.ENV_WIDE, "0")
    eager = fused(x, ids, rw)
    monkeypatch.setenv(rf.ENV_WIDE, "1")
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
        assert torch.equal(captured, eager), (family, q256)


@cuda
@pytest.mark.parametrize("family", LIBRARY_IDS, indirect=True)
def test_the_routed_width_reaches_the_launches_that_have_it(family, monkeypatch):
    """The ``bm`` each launch receives under ``TESSERA_ROUTED_FUSED_WIDE``:
    128 under ``1`` on the launches ``has_width`` names (down in the E4M3
    family, gate/up too on the E4M3 instruction), 64 everywhere else and
    under ``0``; and the ``item_off`` a launch receives counts superblocks of
    that width."""
    fused = _fused(_bundles(family, _stacks(family, cut=False)))
    lib = rf._ext(fused.library)
    seen = []

    class Spy:
        def __getattr__(self, name):
            return getattr(lib, name)

        def routed_fused_forward(self, *args):
            mode, item_off, bm = int(args[0]), args[-9], int(args[-1])
            seen.append((mode, bm, [int(v) for v in item_off.tolist()]))
            return lib.routed_fused_forward(*args)

    monkeypatch.setattr(rf, "_ext", lambda _library, spy=Spy(): spy)
    t = 300
    x = torch.randn(t, HIDDEN, device="cuda").bfloat16()
    ids, rw = _routes(t, TOP_K, 4600)
    ids = _skewed(ids)
    counts = torch.bincount(ids.reshape(-1).long(), minlength=EXPERTS).tolist()

    def off(bm):
        out = [0]
        for c in counts:
            out.append(out[-1] + -(-c // bm))
        return out

    for setting in ("1", "0"):
        monkeypatch.setenv(rf.ENV_WIDE, setting)
        want = [(mode, bm, off(bm)) for mode in (0, 2)
                for bm in [rf.BM_WIDE if setting == "1" and rf.has_width(fused.library, mode, rf.BM_WIDE)
                           else rf.BM]]
        seen.clear()
        fused(x, ids, rw)
        assert seen == want, (fused.library, setting)
