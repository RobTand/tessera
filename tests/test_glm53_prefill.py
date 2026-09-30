"""CPU checks for ``tessera.serving.glm53_prefill``.

The KDA output-norm hook and the sequence-parallel mHC rebind, without vLLM:
config stand-ins for the decline rules, file digests for the source-identity
guard, and two threads standing in for the two TP ranks (a barrier-backed
all-reduce, all-gather and reduce-scatter) for the forward itself.  The
forward check runs a toy decoder stack whose mHC, attention and MLP are
per-token and row-parallel the way the model's are, and holds the rebound
forward to the stock one exactly, with and without the SP branch, at an even
and an odd token count.
"""
from __future__ import annotations

import hashlib
import math
import threading
from types import SimpleNamespace as NS

import pytest
import torch

from tessera.serving import glm53_prefill as gp


# ------------------------------------------------------------------ config stand-ins


def _config(**over):
    par = dict(tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1,
               enable_expert_parallel=False, use_sequence_parallel_moe=False,
               decode_context_parallel_size=1, prefill_context_parallel_size=1)
    par.update(over.pop("parallel", {}))
    arch = over.pop("architectures", ["Glm5NextForConditionalGeneration"])
    text = NS(mhc=over.pop("mhc", True))
    cfg = NS(model_config=NS(hf_config=NS(architectures=arch, text_config=text)),
             parallel_config=NS(**par),
             compilation_config=NS(custom_ops=over.pop("custom_ops", ["none"])),
             scheduler_config=NS(max_num_batched_tokens=2048),
             speculative_config=over.pop("speculative", None))
    assert not over
    return cfg


def test_onorm_hook_appends_for_glm5next_only(monkeypatch):
    monkeypatch.delenv("TESSERA_GLM53_ONORM_CUDA", raising=False)
    cfg = _config()
    assert gp.enable_onorm_cuda(cfg) is True
    assert cfg.compilation_config.custom_ops == ["none", "+fused_rms_norm_gated"]
    assert gp.enable_onorm_cuda(cfg) is True  # idempotent: no second append
    assert cfg.compilation_config.custom_ops.count("+fused_rms_norm_gated") == 1

    other = _config(architectures=["Qwen3ForCausalLM"])
    assert gp.enable_onorm_cuda(other) is False
    assert other.compilation_config.custom_ops == ["none"]


@pytest.mark.parametrize("ops", [["none", "-fused_rms_norm_gated"], ["all", "+fused_rms_norm_gated"]])
def test_onorm_hook_respects_the_serve_either_way(monkeypatch, ops):
    monkeypatch.delenv("TESSERA_GLM53_ONORM_CUDA", raising=False)
    cfg = _config(custom_ops=list(ops))
    assert gp.enable_onorm_cuda(cfg) is False
    assert cfg.compilation_config.custom_ops == ops


def test_onorm_hook_opt_out(monkeypatch):
    monkeypatch.setenv("TESSERA_GLM53_ONORM_CUDA", "0")
    cfg = _config()
    assert gp.enable_onorm_cuda(cfg) is False
    assert cfg.compilation_config.custom_ops == ["none"]


def test_sp_eligible_config_has_no_decline_reason(monkeypatch):
    monkeypatch.delenv("TESSERA_GLM53_SP_MHC_SPEC", raising=False)
    assert gp.sp_decline_reasons(_config()) == []


@pytest.mark.parametrize("over, needle", [
    ({"parallel": {"tensor_parallel_size": 1}}, "tensor_parallel_size 1"),
    ({"parallel": {"tensor_parallel_size": 4}}, "tensor_parallel_size 4"),
    ({"parallel": {"pipeline_parallel_size": 2}}, "pipeline_parallel_size 2"),
    ({"parallel": {"data_parallel_size": 2}}, "data_parallel_size 2"),
    ({"parallel": {"enable_expert_parallel": True}}, "expert parallelism"),
    ({"parallel": {"use_sequence_parallel_moe": True}}, "sequence-parallel MoE"),
    ({"parallel": {"decode_context_parallel_size": 2}}, "decode_context_parallel_size"),
    ({"parallel": {"prefill_context_parallel_size": 2}}, "prefill_context_parallel_size"),
    ({"mhc": False}, "mHC off"),
    ({"architectures": ["DeepseekV4ForCausalLM"]}, "not a Glm5Next"),
])
def test_sp_declines(monkeypatch, over, needle):
    monkeypatch.delenv("TESSERA_GLM53_SP_MHC_SPEC", raising=False)
    reasons = gp.sp_decline_reasons(_config(**over))
    assert any(needle in r for r in reasons), reasons


def test_sp_speculative_needs_the_opt_in(monkeypatch):
    monkeypatch.delenv("TESSERA_GLM53_SP_MHC_SPEC", raising=False)
    cfg = _config(speculative=NS(method="mtp"))
    assert any("speculative" in r for r in gp.sp_decline_reasons(cfg))
    monkeypatch.setenv("TESSERA_GLM53_SP_MHC_SPEC", "1")
    assert gp.sp_decline_reasons(cfg) == []


def test_sp_mode_rejects_unknown(monkeypatch):
    monkeypatch.setenv("TESSERA_GLM53_SP_MHC", "sometimes")
    with pytest.raises(ValueError):
        gp.sp_mode()


def test_mode_off_installs_nothing(monkeypatch):
    monkeypatch.setenv("TESSERA_GLM53_SP_MHC", "off")
    assert gp.install_sp_mhc(_config()) is False


# ------------------------------------------------------------------- source identity


def test_source_identity_matches_only_inspected_bytes(tmp_path, monkeypatch):
    files = []
    for i, name in enumerate(gp.SP_MODULES):
        f = tmp_path / f"m{i}.py"
        f.write_bytes(f"# {name}\n".encode())
        files.append(f)
    mods = tuple(NS(__name__=n, __file__=str(f)) for n, f in zip(gp.SP_MODULES, files))
    interface, why = gp._match_interface(mods)
    assert interface is None and "no inspected interface matches" in why
    digests = tuple(hashlib.sha256(f.read_bytes()).hexdigest() for f in files)
    monkeypatch.setattr(gp, "_INTERFACES", (gp._Interface("test", digests),))
    interface, why = gp._match_interface(mods)
    assert interface is not None and interface.name == "test" and why == ""
    files[2].write_bytes(b"# edited\n")
    assert gp._match_interface(mods)[0] is None


def test_pinned_interface_names_every_module():
    for interface in gp._INTERFACES:
        assert len(interface.digests) == len(gp.SP_MODULES)
        assert all(len(d) == 64 for d in interface.digests)


# ----------------------------------------------------------------------- threshold


def _row(t, saving, cost):
    return {"tokens": t, "saving_ms_per_layer": saving, "cost_ms_per_layer": cost}


def test_t_star_is_the_start_of_the_winning_tail():
    rows = [_row(16, 0.01, 0.05), _row(64, 0.04, 0.05), _row(256, 0.2, 0.06),
            _row(1024, 0.8, 0.07), _row(2048, 1.6, 0.08)]
    assert gp.choose_t_star(rows) == 256


def test_t_star_ignores_an_isolated_early_win():
    rows = [_row(16, 0.2, 0.05), _row(64, 0.01, 0.05), _row(256, 0.2, 0.06)]
    assert gp.choose_t_star(rows) == 256


def test_t_star_infinite_when_sp_never_pays():
    assert math.isinf(gp.choose_t_star([_row(64, 0.0, 0.1), _row(2048, 0.05, 0.1)]))
    assert math.isinf(gp.choose_t_star([]))


def test_state_decisions():
    s = gp.SpState("auto", 2048, 2)
    assert not s.use_sp(4096) and s.wants_measurement()
    s.t_star = 256
    assert s.use_sp(256) and not s.use_sp(255) and not s.wants_measurement()
    f = gp.SpState("force", 2048, 2)
    assert f.use_sp(2) and not f.use_sp(1) and not f.wants_measurement()
    # SP is armed only by a completed pass, and never under capture.
    assert not f.begin_pass(8, capturing=False) and not f.ready  # pass 1: never SP
    assert f.begin_pass(8, capturing=False) and f.ready          # pass 2
    assert not f.begin_pass(8, capturing=True)                   # a capture: stock sequence


# --------------------------------------------------------------- two-rank stand-in


class TwoRanks:
    """Barrier-backed collectives for two threads, summing in rank order."""

    def __init__(self):
        self.barrier = threading.Barrier(2)
        self.slots = [None, None]
        self.local = threading.local()
        self.calls = {"all_reduce": 0, "all_gather": 0, "reduce_scatter": 0}

    @property
    def rank(self):
        return self.local.rank

    def _exchange(self, x):
        self.slots[self.rank] = x
        self.barrier.wait()
        a, b = self.slots
        self.barrier.wait()
        return a, b

    def all_reduce(self, x):
        if self.rank == 0:
            self.calls["all_reduce"] += 1
        a, b = self._exchange(x)
        return a + b

    def sp_all_gather(self, x):
        if self.rank == 0:
            self.calls["all_gather"] += 1
        a, b = self._exchange(x)
        return torch.cat([a, b], 0)

    def sp_reduce_scatter(self, x):
        if self.rank == 0:
            self.calls["reduce_scatter"] += 1
        pad = (-x.shape[0]) % 2
        if pad:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        a, b = self._exchange(x)
        s = a + b
        chunk = s.shape[0] // 2
        return s[self.rank * chunk:(self.rank + 1) * chunk]

    def sp_shard(self, x):
        pad = (-x.shape[0]) % 2
        if pad:
            x = torch.nn.functional.pad(x, (0, 0) * (x.ndim - 1) + (0, pad))
        chunk = x.shape[0] // 2
        return x[self.rank * chunk:(self.rank + 1) * chunk]

    def max_across_tp(self, v):
        a, b = self._exchange(v)
        return max(a, b)


H, N = 8, 4


def _hc_expand(x, n):
    return x.unsqueeze(1).expand(-1, n, -1).contiguous()


def _hc_contract(x, n):
    return x.sum(1) / n


class RowParallel(torch.nn.Module):
    """Rank-sharded input columns; reduces inside when ``reduce_results``."""

    def __init__(self, full, ranks):
        super().__init__()
        self.full, self.ranks, self.reduce_results = full, ranks, True

    def forward(self, x_full_cols):
        r = self.ranks.rank
        k = self.full.shape[1] // 2
        part = x_full_cols[:, r * k:(r + 1) * k] @ self.full[:, r * k:(r + 1) * k].T
        return self.ranks.all_reduce(part) if self.reduce_results else part


class Attn(torch.nn.Module):
    def __init__(self, w_in, w_out, ranks):
        super().__init__()
        self.w_in = w_in
        self.o_proj = RowParallel(w_out, ranks)

    def forward(self, hidden_states, positions):
        h = torch.tanh(hidden_states @ self.w_in.T + positions[:, None].double() * 1e-3)
        return self.o_proj(h)


class MoeRunner:
    def __init__(self, w_up, w_down, ranks):
        self.moe_config = NS(skip_final_all_reduce=False, is_sequence_parallel=False,
                             moe_parallel_config=NS(use_all2all_kernels=False))
        self._fused_output_is_reduced = False
        self.zero_expert_type = None
        self.down = RowParallel(w_down, ranks)
        self.w_up = w_up

    def __call__(self, x):
        self.down.reduce_results = not self.moe_config.skip_final_all_reduce
        return self.down(torch.relu(x @ self.w_up.T))


class Moe(torch.nn.Module):
    def __init__(self, runner):
        super().__init__()
        self.experts = runner

    def forward(self, x):
        return self.experts(x)


class Dense(torch.nn.Module):
    def __init__(self, w_up, w_down, ranks):
        super().__init__()
        self.w_up = w_up
        self.down_proj = RowParallel(w_down, ranks)

    def forward(self, x):
        return self.down_proj(torch.relu(x @ self.w_up.T))


class Layer(torch.nn.Module):
    """The attributes and hc methods the rebound forward reads, per token."""

    def __init__(self, idx, n_layers, ranks, moe, gen):
        super().__init__()
        rnd = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64) * 0.3  # noqa: E731
        self.layer_idx, self.num_hidden_layers, self.n, self.hidden_size = idx, n_layers, N, H
        self.mhc, self.is_mtp_layer = True, False
        self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base = rnd(N, N * H), rnd(3), rnd(N)
        self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base = rnd(N, N * H), rnd(3), rnd(N)
        self.input_layernorm = NS(weight=NS(data=1 + rnd(H)), variance_epsilon=1e-6)
        self.post_attention_layernorm = NS(weight=NS(data=1 + rnd(H)), variance_epsilon=1e-6)
        self.self_attn = Attn(rnd(2 * H, H), rnd(H, 2 * H), ranks)
        if moe:
            self.mlp = Moe(MoeRunner(rnd(2 * H, H), rnd(H, 2 * H), ranks))
        else:
            self.mlp = Dense(rnd(2 * H, H), rnd(H, 2 * H), ranks)
        self._mlp_is_moe = moe

    # Per-token stand-ins with the shapes of the real ops.
    def _pre(self, residual, fn, scale, base, norm_weight, norm_eps):
        flat = residual.reshape(residual.shape[0], -1)
        mix = torch.sigmoid(flat @ fn.T * scale[0] + base)            # (T, N)
        comb = torch.softmax(torch.einsum("ti,tj->tij", mix, mix), -1)  # (T, N, N)
        x = (mix.unsqueeze(-1) * residual).sum(1)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm_eps) * norm_weight
        return mix.unsqueeze(-1), comb, x

    def hc_pre(self, x, fn, scale, base, norm_weight=None, norm_eps=0.0):
        post, comb, x = self._pre(x, fn, scale, base, norm_weight, norm_eps)
        return post, comb, x

    def hc_post(self, x, residual, post, comb):
        return torch.einsum("tij,tjh->tih", comb, residual) + post * x.unsqueeze(1)

    def hc_fused_post_pre(self, x, residual, post, comb, fn, scale, base, norm_weight=None,
                          norm_eps=0.0):
        res = self.hc_post(x, residual, post, comb)
        post2, comb2, x2 = self._pre(res, fn, scale, base, norm_weight, norm_eps)
        return res, post2, comb2, x2


def stock_forward(self, positions, hidden_states, residual=None, post=None, comb=None):
    """The stock non-SP mHC forward: reductions happen inside the modules."""
    x = hidden_states
    if post is None:
        if self.layer_idx == 0:
            x = _hc_expand(x, self.n)
        residual = x
        post, comb, x = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
                                    norm_weight=self.input_layernorm.weight.data,
                                    norm_eps=self.input_layernorm.variance_epsilon)
    else:
        residual, post, comb, x = self.hc_fused_post_pre(
            x, residual, post, comb, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
            norm_weight=self.input_layernorm.weight.data,
            norm_eps=self.input_layernorm.variance_epsilon)
    x = self.self_attn(hidden_states=x, positions=positions)
    residual, post, comb, x = self.hc_fused_post_pre(
        x, residual, post, comb, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
        norm_weight=self.post_attention_layernorm.weight.data,
        norm_eps=self.post_attention_layernorm.variance_epsilon)
    x = self.mlp(x)
    if self.layer_idx == self.num_hidden_layers - 1:
        x = self.hc_post(x, residual, post, comb)
        return _hc_contract(x, self.n), None, None, None
    return x, residual, post, comb


def _stack(ranks, seed=0):
    gen = torch.Generator().manual_seed(seed)
    kinds = [False, True, True]  # a dense layer, then MoE layers
    return [Layer(i, len(kinds), ranks, moe, gen) for i, moe in enumerate(kinds)]


def _run(ranks, forward, t, passes=1, stacks=None):
    """Both ranks through the stack ``passes`` times; each rank's output of the last pass.

    ``forward`` is one callable for both ranks or a per-rank pair (each rank
    process holds its own SP state).
    """
    gen = torch.Generator().manual_seed(7)
    hidden = torch.randn(t, H, generator=gen, dtype=torch.float64)
    positions = torch.arange(t)
    out, errors = [None, None], []
    forwards = forward if isinstance(forward, (list, tuple)) else (forward, forward)

    def worker(rank, layers):
        ranks.local.rank = rank
        try:
            for _ in range(passes):
                x, res, post, comb = hidden.clone(), None, None, None
                for layer in layers:
                    x, res, post, comb = forwards[rank](layer, positions, x, res, post, comb)
            out[rank] = x
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            ranks.barrier.abort()

    # Each rank owns its own module objects (the reduce flags are per rank).
    stacks = stacks or [_stack(ranks), _stack(ranks)]
    threads = [threading.Thread(target=worker, args=(r, stacks[r])) for r in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=60)
    if errors:
        raise errors[0]
    return out, stacks


def _ops(ranks):
    return NS(sp_shard=ranks.sp_shard, sp_all_gather=ranks.sp_all_gather,
              sp_reduce_scatter=ranks.sp_reduce_scatter, all_reduce=ranks.all_reduce,
              hc_expand=_hc_expand, hc_contract=_hc_contract, max_across_tp=ranks.max_across_tp)


class _NoCuda:
    class cuda:  # noqa: N801
        @staticmethod
        def is_current_stream_capturing():
            return False


class _Capturing:
    class cuda:  # noqa: N801
        @staticmethod
        def is_current_stream_capturing():
            return True


@pytest.mark.parametrize("tokens", [8, 7])
@pytest.mark.parametrize("mode", ["force", "below", "capture"])
def test_rebound_forward_equals_stock(tokens, mode):
    ref_ranks = TwoRanks()
    ref, _ = _run(ref_ranks, stock_forward, tokens)
    assert torch.equal(ref[0], ref[1])

    ranks = TwoRanks()
    states = [gp.SpState("force", 2048, 2) for _ in range(2)]
    for state in states:
        if mode == "below":
            state.t_star = tokens + 1  # every batch is below T*: the non-SP branch
    # Under capture the forced threshold is ignored: a graph holds the stock sequence.
    fwds = [gp.make_forward(stock_forward, _ops(ranks), state,
                            _Capturing if mode == "capture" else _NoCuda) for state in states]
    # Pass 1 is the profile run (prepares every layer, never SP); pass 2 is checked.
    got, stacks = _run(ranks, fwds, tokens, passes=2)
    for r in range(2):
        assert got[r].shape == ref[r].shape
        assert torch.equal(got[r], ref[r]), (mode, tokens, r, (got[r] - ref[r]).abs().max())
    n_layers = len(stacks[0])
    assert ref_ranks.calls["all_reduce"] == 2 * n_layers
    if mode == "force":
        # Pass 1 as stock; pass 2 per layer two all-gathers and two reduce-scatters, plus the final gather.
        assert ranks.calls == {"all_reduce": 2 * n_layers, "all_gather": 2 * n_layers + 1,
                               "reduce_scatter": 2 * n_layers}
        assert all(s.ready and s.pass_sp for s in states)
    else:
        assert ranks.calls == {"all_reduce": 4 * n_layers, "all_gather": 0, "reduce_scatter": 0}
        assert not any(s.pass_sp for s in states)
    for layer in stacks[0]:
        assert layer._tessera_sp_ready and layer.self_attn.o_proj.reduce_results is False
        if layer._mlp_is_moe:
            assert layer.mlp.experts.moe_config.skip_final_all_reduce is True
        else:
            assert layer.mlp.down_proj.reduce_results is False


def test_first_pass_is_never_sp_and_an_unpreparable_layer_declines_the_serve():
    """One layer fails its checks: it runs stock, the serve never takes SP, nothing raises."""
    ref, _ = _run(TwoRanks(), stock_forward, 8)
    ranks = TwoRanks()
    stacks = [_stack(ranks), _stack(ranks)]
    for layers in stacks:
        layers[2].mlp.experts._fused_output_is_reduced = True  # not the inspected MoE
    states = [gp.SpState("force", 2048, 2) for _ in range(2)]
    fwds = [gp.make_forward(stock_forward, _ops(ranks), st, _NoCuda) for st in states]
    got, _ = _run(ranks, fwds, 8, passes=3, stacks=stacks)
    for r in range(2):
        assert torch.equal(got[r], ref[r])
    n_layers = len(stacks[0])
    assert ranks.calls == {"all_reduce": 3 * 2 * n_layers, "all_gather": 0, "reduce_scatter": 0}
    for st, layers in zip(states, stacks):
        assert st.declined and "reduces its own output" in st.declined
        assert not st.ready and not st.pass_sp
        assert not layers[2].__dict__.get("_tessera_sp_ready")
        assert layers[2].mlp.experts.moe_config.skip_final_all_reduce is False  # untouched
        assert layers[2].self_attn.o_proj.reduce_results is True


def test_mtp_and_non_mhc_layers_take_the_stock_forward():
    seen = []
    state = gp.SpState("force", 2048, 2)
    fwd = gp.make_forward(lambda *a: seen.append(a) or "stock", NS(), state, _NoCuda)
    assert fwd(NS(mhc=True, is_mtp_layer=True), torch.arange(4), None) == "stock"
    assert fwd(NS(mhc=False, is_mtp_layer=False), torch.arange(4), None) == "stock"
    assert len(seen) == 2


def _layer_for_prepare(moe=True):
    ranks = TwoRanks()
    return _stack(ranks)[1 if moe else 0]


def test_prepare_fails_closed_on_unexpected_objects():
    layer = _layer_for_prepare()
    layer.self_attn.o_proj.reduce_results = False
    with pytest.raises(RuntimeError, match="o_proj.reduce_results"):
        gp.prepare_layer(layer)

    layer = _layer_for_prepare()
    layer.mlp.experts._fused_output_is_reduced = True
    with pytest.raises(RuntimeError, match="reduces its own output"):
        gp.prepare_layer(layer)

    layer = _layer_for_prepare()
    layer.mlp.experts.moe_config.skip_final_all_reduce = True
    with pytest.raises(RuntimeError, match="already skipped"):
        gp.prepare_layer(layer)

    layer = _layer_for_prepare()
    layer.mlp.experts.zero_expert_type = "identity"
    with pytest.raises(RuntimeError, match="parallel layout"):
        gp.prepare_layer(layer)

    layer = _layer_for_prepare(moe=False)
    layer.mlp.down_proj.reduce_results = False
    with pytest.raises(RuntimeError, match="down_proj"):
        gp.prepare_layer(layer)


def test_measurement_picks_and_agrees_on_t_star(monkeypatch):
    """The measurement path on stand-in timings: both ranks end on the MAX."""
    ranks = TwoRanks()
    layer = _layer_for_prepare()
    monkeypatch.setattr(gp, "T_GRID", (8, 16))
    # Per grid count: mhc(T), mhc(T/2), all_reduce, all_gather, reduce_scatter.
    per_rank = {0: [1.0, 0.5, 0.1, 0.1, 0.1, 2.0, 1.0, 0.1, 0.1, 0.1],   # wins at 8 and 16
                1: [1.0, 0.99, 0.1, 0.1, 0.1, 2.0, 1.0, 0.1, 0.1, 0.1]}  # wins at 16 only
    states, errors = [None, None], []

    class _T:
        bfloat16 = torch.float64
        float32 = torch.float64

        @staticmethod
        def randn(*shape, device=None, dtype=None):
            return torch.zeros(*shape, dtype=torch.float64)

        @staticmethod
        def full(shape, value, device=None, dtype=None):
            return torch.full(shape, value, dtype=torch.float64)

    def worker(rank):
        ranks.local.rank = rank
        local_iter = iter(per_rank[rank])
        try:
            state = gp.SpState("auto", 16, 2)
            nonlocal_timings[rank] = local_iter
            gp.measure_t_star(layer, state, _ops(ranks), _T, "cpu")
            states[rank] = state
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            ranks.barrier.abort()

    nonlocal_timings = {}

    def fake_median_rank(fn, torch_mod):
        return next(nonlocal_timings[ranks.rank])

    monkeypatch.setattr(gp, "_median_ms", fake_median_rank)
    threads = [threading.Thread(target=worker, args=(r,)) for r in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=30)
    assert not errors, errors
    assert [len(s.table) for s in states] == [2, 2]
    assert states[0].t_star == states[1].t_star == 16
