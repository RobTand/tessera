"""Saved stitched pure-schedule BF16 bits, including changed graph populations.

Parent invocation: pytest -q -rA -n 2 --dist worksteal --durations=20
 tests/test_routed_window_classes_cuda.py (GB10 serving image, native threads one).
The same file's CPU fixture control is the dry run before GPU submission.
"""
import dataclasses
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tessera import routed_fused as rf
from tessera.native_window_moe import PackedWindowMoeBundles, WindowUnitAxis
from tessera.window_gemm_grouped import prepare_grouped_window_gemm_from_soa
from test_routed_fused_window import _sched
from test_window_gemm_grouped import Expert

HIDDEN, INTER, EXPERTS = 256, 128, 8
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="native class dispatch needs CUDA")


def _profiles(three=False):
    # Down may use a distinct schedule, while gate and up share one.
    return ([(0, 2, 512, 768), (2, 5, 768, 896), (5, 8, 1024, 1024)] if three else
            [(0, 4, 768, 896), (4, 8, 1024, 1024)])


def _packed(family, *, three=False, device="cuda"):
    profiles = _profiles(three)
    qs = [(g, d) for start, end, g, d in profiles for _ in range(start, end)]
    stacks = []
    for role, rows, cols in ((0, INTER, HIDDEN), (1, INTER, HIDDEN), (2, HIDDEN, INTER)):
        stack = []
        for e, (gate, down) in enumerate(qs):
            init = (torch.arange(cols, dtype=torch.int32) + e) if e % 2 else None
            ex = Expert(rows, cols, _sched(cols, down if role == 2 else gate),
                        5200 + 100 * role + e, family=family, init=init, device=device)
            stack.append(dataclasses.replace(ex.unit,
                initial_state=None if init is None else init.to(device)))
        stacks.append(stack)
    axis = WindowUnitAxis(EXPERTS, ("gate", "up", "down"), family=family,
        word_runs={role: [(unit.rep.words.numel(), unit.rep.runs.shape[0]) for unit in stack]
                   for role, stack in zip(("gate", "up", "down"), stacks)})
    for role, stack in zip(("gate", "up", "down"), stacks):
        for e, unit in enumerate(stack):
            axis.put(role, e, unit)
    soa = axis.finish()
    arithmetic = "folded" if family == "value" else "epilogue"
    def bundle(role):
        s = soa[role]
        return prepare_grouped_window_gemm_from_soa(
            words_all=s["words"], table_all=s["table"], codes_all=s["codes"], native_all=s["native"],
            scale_all=s["scale"], runs_all=s["runs"], init_all=s["init"], has_init=s["has_init"],
            word_off=s["word_off"], tile_words=s["tile_words"], total_words=s["total_words"],
            run_off=s["run_off"], perm_all=s["perm"], rows=s["rows"], cols=s["cols"],
            experts=EXPERTS, window_bits=s["window_bits"], family=family, block_m=32, block_n=64,
            block_k=64, arithmetic=arithmetic)
    return PackedWindowMoeBundles(*(bundle(role) for role in ("gate", "up", "down")), family=family,
        expert_classes=[{"start": start, "end": end, "q256": {"w13": [g, g], "w2": [d]}}
                        for start, end, g, d in profiles])


def test_cpu_same_entry_fixture_shapes_and_storage_reads():
    for family in ("value", "e4m3"):
        packed = _packed(family, three=True, device="cpu")
        assert packed.experts == EXPERTS
        for role in (packed.gate, packed.up, packed.down):
            assert role.words_all.numel() == sum(role.total_words.tolist())
            assert role.words_all[:8].numel() == 8
            assert role.runs_all[:1].shape[-1] == 4
        assert packed.expert_classes[1]["q256"]["w2"] == [896]
    for counts in ([1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 1, 1, 1, 1, 0, 0],
                   [3, 2, 1, 1, 1, 0, 0, 0]):
        ids = _population(1, 8, counts, device="cpu")
        assert ids.shape == (1, 8)
        assert torch.bincount(ids.reshape(-1).long(), minlength=8).tolist() == counts


def _pure_launch(adapter, mode, x, scale, routing, out, *, weight):
    """The previous pure-stack entry, not the class dispatcher or its counter code."""
    cls = adapter.classes[0]
    down = mode == 2
    b0, b1 = (cls.down, cls.down) if down else (cls.gate, cls.up)
    w0, w1 = (cls.words_down, cls.words_down) if down else (cls.words_gate, cls.words_up)
    t0, t1 = (cls.table_down, cls.table_down) if down else (cls.table_gate, cls.table_up)
    r0, r1 = (cls.runs_down, cls.runs_down) if down else (cls.runs_gate, cls.runs_up)
    d0, d1 = (cls.bdesc_down, cls.bdesc_down) if down else (cls.bdesc_gate, cls.bdesc_up)
    bm = rf.superblock_rows(adapter.library, mode, routing.tokens)
    counter = torch.zeros(1, dtype=torch.int32, device=x.device)
    empty = torch.empty(0, dtype=torch.float32, device=x.device)
    rf._ext(adapter.library).routed_fused_forward(
        mode, adapter.fp8, x, scale if scale is not None else empty,
        w0, w1, t0, t1, b0.init_all, b1.init_all, b0.has_init, b1.has_init,
        b0.scale_all, b1.scale_all, r0, r1, d0, d1,
        cls.tile_words_down if down else cls.tile_words_gate_up,
        cls.slot_words_down if down else cls.slot_words_gate_up, adapter.piece_major,
        routing.offsets, routing.flat_sorted, routing.rw_sorted, routing.superblocks(bm), counter,
        routing.top_k, 1 if down else 0, weight, 2.0,
        out, torch.cuda.get_device_properties(x.device).multi_processor_count, bm)


def _saved_reference(packed, x, ids, weights, *, input_weight=False, shared=None):
    """Stitch pure-class route outputs, then use the saved fixed-order sum."""
    routed = torch.empty(ids.numel(), HIDDEN, device=x.device, dtype=torch.bfloat16)
    flat_ids = ids.reshape(-1)
    for desc in packed.expert_classes:
        start, end = desc["start"], desc["end"]
        take = torch.where((flat_ids >= start) & (flat_ids < end))[0]
        if take.numel() == 0:
            continue
        gate, up, down = (rf.grouped_class_view(role, start, end)
                          for role in (packed.gate, packed.up, packed.down))
        pure_desc = [{"start": 0, "end": end - start, "q256": desc["q256"]}]
        pure = rf.FusedRoutedWindowMoE.from_bundles(gate, up, down, expert_classes=pure_desc)
        local_ids = (flat_ids[take] - start).reshape(-1, 1)
        rw = weights.reshape(-1)[take].reshape(-1, 1)
        xin = x[take // ids.shape[1]]
        if input_weight:
            xin = xin * rw.to(xin.dtype)
        routing = pure._routing(local_ids, rw)
        xq, a1 = pure._quantized(xin, None, len(take))
        act = torch.empty(len(take), INTER, device=x.device, dtype=torch.bfloat16)
        _pure_launch(pure, 0, xq, a1, routing, act, weight=False)
        aq, a2 = pure._quantized(act, None, len(take))
        output = torch.empty(len(take), HIDDEN, device=x.device, dtype=torch.bfloat16)
        _pure_launch(pure, 2, aq, a2, routing, output, weight=not input_weight)
        routed[take] = output
    out = torch.empty_like(x)
    lib = rf._ext(rf.library_for(packed.family))
    if shared is None:
        lib.token_sum(routed, out, ids.shape[1])
    else:
        lib.token_sum_shared(routed, shared, out, ids.shape[1])
    return out.clone()


def _population(tokens, top_k, counts, *, device="cuda"):
    cycle = torch.tensor([e for e, count in enumerate(counts) for _ in range(count)], dtype=torch.int32)
    flat = cycle.repeat((tokens * top_k + len(cycle) - 1) // len(cycle))[:tokens * top_k]
    # Put experts in non-storage order, without moving top-k weights/positions.
    return flat.flip(0).reshape(tokens, top_k).to(device)


@cuda
@pytest.mark.parametrize("family", ["value", "e4m3"])
@pytest.mark.parametrize("tokens", [1, 16, 2048, 4096])
@pytest.mark.parametrize("reverse", [False, True])
def test_saved_pure_bits_eager_and_changed_population_replay(family, tokens, reverse, monkeypatch):
    monkeypatch.setenv(rf.ENV_E4M3_MMA, "e4m3")
    for three in (False, True):
        packed = _packed(family, three=three)
        adapter = packed.adapter()
        if reverse:
            adapter = dataclasses.replace(adapter, class_issue_order=tuple(reversed(range(len(adapter.classes)))))
        k = 8  # All population ratios are exact, also at M1.
        x = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.05
        rw = torch.linspace(0.1, 0.9, tokens * k, device="cuda").reshape(tokens, k)
        populations = ([8, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 8],
                       [1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 1, 1, 1, 1, 0, 0],
                       [3, 2, 1, 1, 1, 0, 0, 0])
        routes = [_population(tokens, k, counts) for counts in populations]
        saved = [_saved_reference(packed, x, ids, rw) for ids in routes]
        for ids, expected in zip(routes, saved):
            assert torch.equal(adapter(x, ids, rw, swiglu_limit=2.0), expected)
        ids = routes[0].clone()
        adapter(x, ids, rw, swiglu_limit=2.0)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = adapter(x, ids, rw, swiglu_limit=2.0)
        for route, expected in zip(routes + list(reversed(routes)), saved + list(reversed(saved))):
            ids.copy_(route)
            captured.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(captured, expected)


@cuda
@pytest.mark.parametrize("family", ["value", "e4m3"])
def test_uniform_identity_matches_old_pure_and_input_weight_shared(family, monkeypatch):
    monkeypatch.setenv(rf.ENV_E4M3_MMA, "e4m3")
    mixed = _packed(family)
    roles = [rf.grouped_class_view(role, 0, 4) for role in (mixed.gate, mixed.up, mixed.down)]
    desc = [{"start": 0, "end": 4, "q256": mixed.expert_classes[0]["q256"]}]
    packed = PackedWindowMoeBundles(*roles, family=family, expert_classes=desc)
    adapter = packed.adapter()
    x = torch.randn(16, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.05
    ids = torch.arange(16, device="cuda", dtype=torch.int32).remainder(4).reshape(16, 1)
    rw = torch.linspace(0.1, 0.9, 16, device="cuda").reshape(16, 1)
    shared = torch.randn_like(x)
    for input_weight in (False, True):
        expected = _saved_reference(packed, x, ids, rw, input_weight=input_weight, shared=shared)
        actual = adapter(x, ids, rw, swiglu_limit=2.0, shared=shared,
                         apply_router_weight_on_input=input_weight)
        assert torch.equal(actual, expected)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = adapter(x, ids, rw, swiglu_limit=2.0, shared=shared,
                               apply_router_weight_on_input=input_weight)
        for _ in range(2):
            captured.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(captured, expected)
