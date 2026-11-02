"""Shared routed-window fixtures, builders and stage-by-stage oracles."""
import pytest
import torch

from tessera import routed_fused as rf
from tessera import window_gemm_grouped as wgg
from tessera.native_window_moe import (
    PackedWindowMoeBundles, _silu_and_mul, native_window_moe_from_bundles,
)

import fused_bound as fb
from window_gemm_grouped_support import Expert, _quant

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


def _sched(cols, q256, cap=8):
    """The grammar's Bresenham schedule for ``q256`` over ``cols`` columns
    (cap 8 by default: the routed launches' whole range, so the value family's
    rungs are read too; 14 for a value rung above 8, which only the dense
    launch reads)."""
    from fractions import Fraction

    from tessera.grammar import bresenham_rate_schedule

    return bresenham_rate_schedule(Fraction(q256, 256), cols, cap=cap)


def _stacks(family, *, hidden=HIDDEN, inter=INTER, experts=EXPERTS, seed=300, cut=True, q256=1024,
            place=None, cap=8):
    """gate, up, down Expert lists at the ``q256`` rung's schedule (rate 4
    everywhere by default); experts 1 and 3 carry a start state (the TP
    row-cut case) when ``cut``.  ``place(schedule, seed)``, when given,
    rearranges each expert's schedule (its own seed per expert and
    projection): the rate counts, so the run table, stay the stack's.
    ``cap`` is the schedule's rate bound (see ``_sched``)."""
    r_h, r_i = _sched(hidden, q256, cap), _sched(inter, q256, cap)

    def rates(schedule, s):
        return schedule if place is None else place(schedule, s)

    gate = [Expert(inter, hidden, rates(r_h, seed + i), seed + i, family=family,
                   init=_init(hidden, seed + 40 + i) if (cut and i % 2) else None)
            for i in range(experts)]
    up = [Expert(inter, hidden, rates(r_h, seed + 10 + i), seed + 10 + i, family=family,
                 init=_init(hidden, seed + 50 + i) if (cut and i % 2) else None)
          for i in range(experts)]
    down = [Expert(hidden, inter, rates(r_i, seed + 20 + i), seed + 20 + i, family=family,
                   init=_init(inter, seed + 60 + i) if (cut and i == 3) else None)
            for i in range(experts)]
    return gate, up, down


def _uniform_class_metadata(gate):
    runs = gate.runs_all.view(gate.experts, -1, 4)[0].tolist()
    q256 = sum(rate * count for rate, _, count, _ in runs) * 256 // gate.cols
    return [{"start": 0, "end": gate.experts, "q256": {"w13": [q256, q256], "w2": [q256]}}]



def _bundles(family, stacks):
    gate, up, down = (wgg.prepare_grouped_window_gemm([e.unit for e in s], block_m=32,
                                                      block_n=64, block_k=64)
                      for s in stacks)
    return PackedWindowMoeBundles(gate=gate, up=up, down=down, family=family, expert_classes=_uniform_class_metadata(gate))


def _axis_bundles(family, stacks):
    """The same stacks through the serving loader's path: every expert's unit
    placed on a ``WindowUnitAxis`` (the start state on the unit itself, in
    original column order, as ``compact_prep.prepare_window_compact`` hands
    it over), ``finish`` and ``prepare_grouped_window_gemm_from_soa`` -- what
    ``moe_route._RankLocalPackedIntake`` builds on a TP rank > 0."""
    import dataclasses

    from tessera.native_window_moe import WindowUnitAxis

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
        return wgg.prepare_grouped_window_gemm_from_soa(words_all=slot["words"], table_all=slot["table"], codes_all=slot["codes"],
        native_all=slot["native"], scale_all=slot["scale"], runs_all=slot["runs"],
        init_all=slot["init"], has_init=slot["has_init"], word_off=slot["word_off"],
        tile_words=slot["tile_words"], total_words=slot["total_words"],
        run_off=slot["run_off"], perm_all=slot["perm"], rows=slot["rows"],
        cols=slot["cols"], experts=experts, window_bits=slot["window_bits"],
        family=family, block_m=32, block_n=64, block_k=64)

    return PackedWindowMoeBundles(gate=(g := bundle("gate")), up=bundle("up"), down=bundle("down"), family=family, expert_classes=_uniform_class_metadata(g))


def _fused(bundles):
    return rf.FusedRoutedWindowMoE.from_bundles(bundles.gate, bundles.up, bundles.down, expert_classes=bundles.expert_classes)


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
