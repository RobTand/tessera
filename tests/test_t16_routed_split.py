"""CPU source/ownership regressions, not CUDA qualification or timing evidence."""
import importlib
import inspect
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

from tessera import routed_fused as rf
from tessera.errors import GrammarError


def split_module():
    return importlib.import_module("tessera.t16_routed_split")


def projection(rate, *, high_columns=0, rows=256, cols=256, experts=2):
    low_columns = cols - high_columns
    runs = [[rate, 0, low_columns, 0]]
    if high_columns:
        runs.append([rate + 1, low_columns, high_columns, 16 * rate * low_columns])
    tile_words = 16 * (rate * cols + high_columns)
    words = ((rows + 511) // 512) * tile_words
    return SimpleNamespace(
        family="value", device=torch.device("cuda"), experts=experts,
        rows=rows, cols=cols, window_bits=rf.WINDOW_BITS, word_layout="legacy",
        words_all=torch.zeros(experts, words, dtype=torch.int32),
        word_off=torch.arange(experts, dtype=torch.int32) * words,
        total_words=torch.full((experts,), words, dtype=torch.int32),
        runs_all=torch.tensor([runs] * experts, dtype=torch.int32),
        run_off=torch.arange(experts + 1, dtype=torch.int64) * len(runs),
        perm_all=torch.arange(cols, dtype=torch.int32).expand(experts, -1).clone(),
        tile_words=torch.full((experts,), tile_words, dtype=torch.int32),
        init_all=torch.zeros(experts, cols, dtype=torch.int32),
        has_init=torch.zeros(experts, dtype=torch.int32),
        scale_all=torch.ones(experts, rows))


@pytest.fixture
def tp2_cpu_intake(monkeypatch):
    import test_routed_fused_window as window
    from tessera import compact_prep, export, kernel_window_gemv, unit_artifact

    calls = []
    default_ceiling = inspect.signature(compact_prep.prepare_window_compact).parameters["rate_max"].default
    randn = torch.randn
    monkeypatch.setattr(torch, "randn", lambda *shape, **kwargs: randn(*shape))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    monkeypatch.setattr(window, "HIDDEN", 4)
    monkeypatch.setattr(window, "TP2_INTER", 2)

    def encode(weight, *, q256, **kwargs):
        wire = SimpleNamespace(rows=weight.shape[0], cols=weight.shape[1],
                               rates=[q256 // 256] * weight.shape[1])
        return SimpleNamespace(blob=wire), None, None

    def prepare(wire, *, rows=None, cols=None, rate_max=default_ceiling, **kwargs):
        calls.append((rows, cols, rate_max))
        if max(wire.rates) > rate_max:
            raise GrammarError("test intake exceeds its explicit ceiling")
        return SimpleNamespace(table=torch.tensor([0.0, 1.0]), scale=torch.ones(wire.rows),
                               row_offset=rows[0] if rows else 0,
                               initial_state=torch.tensor([int(rows is not None)]))

    monkeypatch.setattr(export, "encode_linear_planes", encode)
    monkeypatch.setattr(compact_prep, "parse_compact_wire", lambda blob, **kwargs: blob)
    monkeypatch.setattr(compact_prep, "prepare_window_compact", prepare)
    monkeypatch.setattr(unit_artifact, "parse_unit_artifact", lambda blob, **kwargs: SimpleNamespace(
        unit=SimpleNamespace(window_bits=rf.WINDOW_BITS, rates=blob.rates,
                             body_bits=torch.ones(blob.rows, blob.cols, dtype=torch.int64))))
    monkeypatch.setattr(kernel_window_gemv, "reference_states", lambda bits, *args: bits)
    return window, calls, default_ceiling


def test_tp2_explicit_ceiling_reaches_cut_and_whole_preparations(tp2_cpu_intake):
    window, calls, default_ceiling = tp2_cpu_intake
    ceiling = rf.DENSE_RATE_MAX["value"]
    window._tp2_rank1_stacks("value", ceiling * 256, rate_max=ceiling)
    assert len(calls) == 2 * 3 * window.TP2_EXPERTS
    assert all(rate_max == ceiling for _rows, _cols, rate_max in calls)
    assert sum(rows is not None for rows, _cols, _rate in calls) == 2 * window.TP2_EXPERTS
    assert sum(cols is not None for _rows, cols, _rate in calls) == window.TP2_EXPERTS
    calls.clear()
    window._tp2_rank1_stacks("value", default_ceiling * 256)
    assert all(rate_max == default_ceiling for _rows, _cols, rate_max in calls)


def test_tp2_truth_owns_rank_local_dimensions_through_down_bounds(tp2_cpu_intake):
    window, _calls, default_ceiling = tp2_cpu_intake
    _units, truths, _rates = window._tp2_rank1_stacks("value", default_ceiling * 256)
    ids = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
    weights = torch.ones_like(ids, dtype=torch.float32)
    activation = torch.ones(ids.numel(), window.TP2_INTER, dtype=torch.bfloat16)
    reference, bound = window._down_bounds(truths, "value", activation, ids, weights)
    assert reference.shape == bound.shape == (ids.shape[0], window.HIDDEN)
    assert bool((reference == ids.shape[1] * window.TP2_INTER).all())
    for stack in truths:
        for truth in stack:
            assert (truth.rows, truth.cols) == tuple(truth.states.shape)


@pytest.mark.parametrize("mutation", ["zero", "identity_gather", "no_clamp", "no_sigmoid", "no_mul"])
def test_stage_oracle_rejects_activation_mutations(monkeypatch, mutation):
    import fused_bound as fb

    from tessera.native_window_moe import _silu_and_mul
    from test_t16_routed_split_cuda import check_stages

    split = split_module()
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    monkeypatch.setattr(rf, "_sm_count", lambda device: 1)
    matrices = (
        [[[1, 0], [0, 1]], [[2, 0], [0, -1]]],
        [[[0, 2], [1, 0]], [[1, 1], [-1, 0]]],
        [[[1, 1], [1, 2]], [[2, 1], [1, 1]]],
    )
    values = torch.tensor([-1, 0, 1, 2], dtype=torch.bfloat16)
    truths = tuple([SimpleNamespace(
        states=torch.tensor([[int((values == value).nonzero()[0]) for value in row]
                             for row in matrix]), values=values, scale=torch.ones(2),
        rows=2, cols=2) for matrix in role] for role in matrices)
    empty = torch.empty(0, dtype=torch.int32)
    roles = [split._Projection(words=torch.full((2, 1), index, dtype=torch.int32),
                               table=values, init=empty, has_init=empty, scale=empty,
                               runs=empty, bdesc=empty, tile_words=0, slot_words=0,
                               rows=2, cols=2) for index in range(len(truths))]

    class Native:
        mutation = None

        def routed_single_forward(self, *args):
            inputs, words = args[:2]
            offsets, flat_sorted, rw_sorted = args[10:13]
            top_k, a_row_mode, mul_weight, out = args[15:19]
            for expert, truth in enumerate(truths[int(words[0, 0])]):
                start, end = int(offsets[expert]), int(offsets[expert + 1])
                flat = flat_sorted[start:end].long()
                activation = inputs[flat // top_k] if a_row_mode == 0 else inputs[start:end]
                weight = truth.values[truth.states].float() * truth.scale[:, None]
                result = activation.float() @ weight.t()
                if mul_weight:
                    result *= rw_sorted[start:end, None]
                out[flat] = result.bfloat16()

        def split_swiglu(self, gate, up, flat, limit, out):
            if self.mutation == "zero":
                out.zero_()
                return
            clamp = None if self.mutation == "no_clamp" else limit
            activation = _silu_and_mul(gate, up, clamp_limit=clamp)
            if self.mutation == "no_sigmoid":
                activation = (gate.float().clamp(max=limit)
                              * up.float().clamp(min=-limit, max=limit)).bfloat16()
            if self.mutation == "no_mul":
                activation = _silu_and_mul(gate, torch.ones_like(up), clamp_limit=limit)
            out.copy_(activation if self.mutation == "identity_gather" else activation[flat.long()])

        def token_sum(self, routes, out, top_k):
            out.copy_(routes.float().reshape(out.shape[0], top_k, out.shape[1]).sum(1).bfloat16())

    native = Native()
    owner = split.SplitT16RoutedMoE(native, *roles, torch.zeros(1, dtype=torch.int32))
    inputs = torch.tensor([[4, -4], [2, 1], [-1, 3]], dtype=torch.bfloat16)
    ids = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.int32)
    weights = torch.tensor([[1.0, 0.5], [0.5, 1.0], [1.0, 0.5]])
    check_stages(owner, truths, inputs, ids, weights)
    native.mutation = mutation
    with pytest.raises(AssertionError, match="split activation vs independent SwiGLU"):
        check_stages(owner, truths, inputs, ids, weights)
    check_within = fb.check_within

    def check_only_linear_stages(out, reference, bound, what, **kwargs):
        if what == "split activation vs independent SwiGLU":
            return
        return check_within(out, reference, bound, what, **kwargs)

    monkeypatch.setattr(fb, "check_within", check_only_linear_stages)
    with pytest.raises(AssertionError, match="split down real wire"):
        check_stages(owner, truths, inputs, ids, weights)


@pytest.mark.parametrize("rate", range(rf.RATE_MAX + 1, rf.DENSE_RATE_MAX["value"] + 1))
@pytest.mark.parametrize("mixed", [False, True])
def test_split_tables_resources_and_production_exclusion(monkeypatch, rate, mixed):
    split = split_module()
    if mixed:
        rate -= 1
    roles = [projection(rate, high_columns=128 if mixed else 0) for _ in range(3)]
    monkeypatch.setattr(rf, "smem_reason", lambda *args: None)
    assert split.supported(*roles) is None
    assert rf.fused_routed_window_supported(*roles) is not None
    for role in roles:
        runs, descriptors, tile_words, slot = rf.projection_tables(
            role, rate_max=rf.DENSE_RATE_MAX["value"])
        assert runs.shape == (role.experts, 8)
        assert descriptors.shape == (role.experts, role.cols // rf.BK, rf.BDESC_INTS)
        assert tile_words == int(role.tile_words[0])
        assert rf.smem_bytes(2, slot) <= rf.SM121_MAX_DYNAMIC_SMEM


@pytest.mark.parametrize("rate", [rf.WINDOW_BITS + 1, rf.WINDOW_BITS + 2])
def test_split_refuses_terminal_rates(monkeypatch, rate):
    split = split_module()
    monkeypatch.setattr(rf, "smem_reason", lambda *args: None)
    reason = split.supported(*(projection(rate) for _ in range(3)))
    assert reason is not None and "rate" in reason


def test_split_independent_strides_geometry_and_family(monkeypatch):
    split = split_module()
    calls = []
    monkeypatch.setattr(rf, "smem_reason", lambda *args: calls.append(args) or None)
    roles = [projection(9), projection(10), projection(11, high_columns=128)]
    assert split.supported(*roles) is None
    assert [args[0] for args in calls] == [2, 2, 2]
    roles[0].rows = roles[1].rows = 320
    roles[2] = projection(9, cols=320)
    roles[0].scale_all = roles[1].scale_all = torch.ones(2, 320)
    assert "128" in split.supported(*roles)
    roles[0].family = "e4m3"
    assert "value" in split.supported(*roles)


def test_shared_stack_validation_is_not_bypassed(monkeypatch):
    split = split_module()
    monkeypatch.setattr(rf, "smem_reason", lambda *args: None)
    roles = [projection(9) for _ in range(3)]
    roles[1].init_all = roles[1].init_all.to(torch.int64)
    assert "up init_all" in split.supported(*roles)


def test_native_single_projection_policy_executes_source(tmp_path):
    source = (Path(rf.__file__).parent / "serving" / rf.SOURCE).read_text()
    marker = "__host__ __device__ constexpr bool single_projection_decodes("
    assert marker in source, "the native single-table routed policy is missing"
    end = source.index("\n}", source.index(marker)) + 2
    constants = source[source.index("constexpr int THREADS ="):end]
    prefix = """
#include <type_traits>
#include <cstdint>
#include <initializer_list>
#define __host__
#define __device__
#define TESSERA_ROUTED_FUSED_A_PREFETCH 4
constexpr bool FAMILY_FP8=false, FAMILY_MMA8=false, FAMILY_FP4=false;
constexpr bool PAIRED_K32_BUILD=false;
"""
    main = """
int main() {
    for (int rate = RATE_MIN; rate <= RATE_MAX + 2; ++rate) {
        for (bool two : {false, true}) {
            bool expected = rate + int(two) <= RATE_MAX;
            if (single_projection_decodes(rate, two) != expected) return 1;
            if (launch_decodes(2, rate, two) != (rate + int(two) <= ROUTED_RATE_MAX)) return 2;
            if (expected && smem_bytes(2, pair_slot_words(rate, two)) > SM121_SMEM_OPTIN) return 3;
        }
    }
}
"""
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    assert compiler, "a C++ compiler is required for the native policy regression"
    path = tmp_path / "single.cpp"
    path.write_text(prefix + constants + main)
    binary = tmp_path / "single"
    result = subprocess.run([compiler, "-std=c++17", str(path), "-o", str(binary)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    subprocess.run([str(binary)], check=True)


def test_owner_routes_single_tables_and_sums_deterministically(monkeypatch):
    split = split_module()
    events = []
    roles = [projection(rate) for rate in (9, 10, 11)]
    for role in roles:
        role.device = torch.device("cpu")
        role.table_all = torch.empty(role.experts, rf.TABLE_ENTRIES, dtype=torch.bfloat16)
    monkeypatch.setattr(split, "supported", lambda *args: None)
    monkeypatch.setattr(rf, "compose_table16", lambda role: role.table_all.view(torch.int16))
    monkeypatch.setattr(rf, "_sm_count", lambda device: 1)

    class Native:
        def single_projection_decodes(self, rate, two):
            return rate + int(two) <= rf.DENSE_RATE_MAX["value"]

        def routed_single_forward(self, *args):
            events.append(("project", args[8], args[-5], args[-4], args[0].shape))
            args[-3].fill_(1)

        def split_swiglu(self, gate, up, flat, limit, out):
            events.append(("activation", limit))
            out.fill_(1)

        def token_sum(self, routes, out, top_k):
            events.append(("sum", top_k))
            out.fill_(2)

    monkeypatch.setattr(rf, "_ext", lambda library: Native())
    owner = split.SplitT16RoutedMoE.from_bundles(*roles)
    ids = torch.tensor([[1, 0], [1, 1]], dtype=torch.int32)
    weights = torch.ones_like(ids, dtype=torch.float32)
    x = torch.ones(2, 256, dtype=torch.bfloat16)
    result = owner(x, ids, weights, swiglu_limit=3.0)
    assert result.shape == x.shape and bool((result == 2).all())
    projects = [event for event in events if event[0] == "project"]
    assert [event[1] for event in projects] == [int(role.tile_words[0]) for role in roles]
    assert [(event[2], event[3]) for event in projects] == [(0, False), (0, False), (1, True)]
    assert events[-1] == ("sum", 2)
    assert ("activation", 3.0) in events
    assert owner.resident_bytes == sum(t.numel() * t.element_size() for _, t in owner.named_tensors())
    with pytest.raises(GrammarError, match="top_k"):
        owner(x, ids, weights, apply_router_weight_on_input=True)
    events.clear()
    owner(x, ids[:, :1], weights[:, :1], apply_router_weight_on_input=True)
    assert [(event[2], event[3]) for event in events if event[0] == "project"] == [
        (0, False), (0, False), (1, False)]
