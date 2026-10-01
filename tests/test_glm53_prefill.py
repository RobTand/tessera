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

import contextlib
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
    monkeypatch.setenv("TESSERA_GLM53_ONORM_CUDA", "1")
    cfg = _config()
    assert gp.enable_onorm_cuda(cfg) is True
    assert cfg.compilation_config.custom_ops == ["none", "+fused_rms_norm_gated"]
    assert gp.enable_onorm_cuda(cfg) is True  # idempotent: no second append
    assert cfg.compilation_config.custom_ops.count("+fused_rms_norm_gated") == 1

    other = _config(architectures=["Qwen3ForCausalLM"])
    assert gp.enable_onorm_cuda(other) is False
    assert other.compilation_config.custom_ops == ["none"]


def test_onorm_hook_is_not_fooled_by_a_reused_id(monkeypatch):
    """CPython hands a dead object's id to a new one.  A config this hook enabled can die and
    its id go to a config whose serve named the op itself, which is the serve's decision."""
    monkeypatch.setenv("TESSERA_GLM53_ONORM_CUDA", "1")
    monkeypatch.setattr(gp, "_ONORM_DONE", type(gp._ONORM_DONE)())
    monkeypatch.setattr(gp, "id", lambda obj: 42, raising=False)  # every config gets one id
    assert gp.enable_onorm_cuda(_config()) is True
    assert gp.enable_onorm_cuda(_config(custom_ops=["all", "+fused_rms_norm_gated"])) is False


@pytest.mark.parametrize("ops", [["none", "-fused_rms_norm_gated"], ["all", "+fused_rms_norm_gated"]])
def test_onorm_hook_respects_the_serve_either_way(monkeypatch, ops):
    monkeypatch.setenv("TESSERA_GLM53_ONORM_CUDA", "1")
    cfg = _config(custom_ops=list(ops))
    assert gp.enable_onorm_cuda(cfg) is False
    assert cfg.compilation_config.custom_ops == ops


@pytest.mark.parametrize("value", [None, "0"])
def test_onorm_hook_is_off_unless_the_serve_asks(monkeypatch, value):
    # Off by default: outside mode NONE the hook changes a stock default.
    if value is None:
        monkeypatch.delenv("TESSERA_GLM53_ONORM_CUDA", raising=False)
    else:
        monkeypatch.setenv("TESSERA_GLM53_ONORM_CUDA", value)
    cfg = _config()
    assert gp.enable_onorm_cuda(cfg) is False
    assert cfg.compilation_config.custom_ops == ["none"]


def test_onorm_flag_refuses_an_unknown_value(monkeypatch):
    monkeypatch.setenv("TESSERA_GLM53_ONORM_CUDA", "yes")
    with pytest.raises(ValueError, match="TESSERA_GLM53_ONORM_CUDA"):
        gp.enable_onorm_cuda(_config())


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


def test_sp_mode_defaults_off(monkeypatch):
    # SP is not bit-identical to stock (the mHC kernels are not token-count
    # invariant), so a serve that does not ask for it runs the stock forward.
    monkeypatch.delenv("TESSERA_GLM53_SP_MHC", raising=False)
    assert gp.sp_mode() == "off"
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
    assert not f.begin_pass(8, capturing=False, exact=True) and not f.ready  # pass 1: never SP
    assert f.begin_pass(8, capturing=False, exact=True) and f.ready          # pass 2
    assert not f.begin_pass(8, capturing=True, exact=True)    # a capture: stock sequence
    assert not f.begin_pass(8, capturing=False, exact=False)  # not exact: stock, whatever T*


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


def _ops(ranks, exact=lambda t, hidden, n: True, tile=None, tile_exact=lambda t, hidden, n: True,
         tiled=None):
    """The forward's ops on the two-rank stand-in.  ``full_split`` records, per rank,
    the token count it forces while it is open (``ranks.local.forced``).  ``tiled_post_pre``
    runs the layer's per-token ``hc_fused_post_pre`` on ``tile``-token slices and joins
    them, and appends ``(rank, call tokens, forced)`` to ``tiled``."""

    def tiled_post_pre(layer, x, residual, post, comb, fn, scale, base, norm_weight, norm_eps):
        if tiled is not None:
            tiled.append((ranks.rank, x.shape[0], getattr(ranks.local, "forced", None)))
        parts = [layer.hc_fused_post_pre(x[a:a + tile], residual[a:a + tile], post[a:a + tile],
                                         comb[a:a + tile], fn, scale, base,
                                         norm_weight=norm_weight, norm_eps=norm_eps)
                 for a in range(0, x.shape[0], tile)]
        return tuple(torch.cat([p[j] for p in parts], 0) for j in range(4))

    @contextlib.contextmanager
    def full_split(tokens):
        previous = getattr(ranks.local, "forced", None)
        ranks.local.forced = tokens
        try:
            yield
        finally:
            ranks.local.forced = previous

    return NS(sp_shard=ranks.sp_shard, sp_all_gather=ranks.sp_all_gather,
              sp_reduce_scatter=ranks.sp_reduce_scatter, all_reduce=ranks.all_reduce,
              hc_expand=_hc_expand, hc_contract=_hc_contract, max_across_tp=ranks.max_across_tp,
              sp_exact=exact, full_split=full_split, tile_exact=tile_exact,
              tiled_post_pre=tiled_post_pre)


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


class _CpuTorch:
    """The torch calls ``measure_t_star`` makes, on CPU float64."""
    bfloat16 = torch.float64
    float32 = torch.float64

    @staticmethod
    def randn(*shape, device=None, dtype=None):
        return torch.zeros(*shape, dtype=torch.float64)

    @staticmethod
    def full(shape, value, device=None, dtype=None):
        return torch.full(shape, value, dtype=torch.float64)


def test_measurement_picks_and_agrees_on_t_star(monkeypatch):
    """The measurement path on stand-in timings: both ranks end on the MAX."""
    ranks = TwoRanks()
    layer = _layer_for_prepare()
    monkeypatch.setattr(gp, "T_GRID", (8, 16))
    # Per grid count: mhc(T), mhc(T/2), all_reduce, all_gather, reduce_scatter.
    per_rank = {0: [1.0, 0.5, 0.1, 0.1, 0.1, 2.0, 1.0, 0.1, 0.1, 0.1],   # wins at 8 and 16
                1: [1.0, 0.99, 0.1, 0.1, 0.1, 2.0, 1.0, 0.1, 0.1, 0.1]}  # wins at 16 only
    states, errors = [None, None], []

    def worker(rank):
        ranks.local.rank = rank
        local_iter = iter(per_rank[rank])
        try:
            state = gp.SpState("auto", 16, 2)
            nonlocal_timings[rank] = local_iter
            gp.measure_t_star(layer, state, _ops(ranks), _CpuTorch, "cpu")
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


# ------------------------------------------------------------------------ exact SP


def _vllm_split_rule(block_k, k, grid_size, n_sms=48):
    """vLLM's ``compute_num_split`` on a 48-SM device (GB10), for checking real values."""
    split = n_sms // grid_size
    if k is not None:
        split = min(split, (-(-k // block_k)) // 4)
    return max(split, 1)


def test_split_forcer_is_stock_outside_and_the_full_batch_inside():
    f = gp.SplitForcer(_vllm_split_rule)
    k = 4 * 4096
    # The case the probe found: a 1024-token shard of a 2048-token batch splits 3, the batch 1.
    assert f(64, k, 1024 // 64) == 3
    with f.full_batch(2048):
        assert f(64, k, 1024 // 64) == _vllm_split_rule(64, k, 2048 // 64) == 1
        with f.full_batch(100):  # nested: cdiv(100, 64) = 2
            assert f(64, k, 1) == _vllm_split_rule(64, k, 2)
        assert f(64, k, 1024 // 64) == 1
        other = []
        th = threading.Thread(target=lambda: other.append(f(64, k, 1024 // 64)))
        th.start()
        th.join()
        assert other == [3]  # another thread keeps the stock rule
    assert f(64, k, 1024 // 64) == 3


def test_install_split_forcer_wraps_the_module_attribute_once():
    kernels = NS(compute_num_split=_vllm_split_rule)
    a = gp.install_split_forcer(kernels)
    b = gp.install_split_forcer(kernels)
    assert a is b and kernels.compute_num_split is a and a.stock is _vllm_split_rule


def test_shard_split_exact_follows_vllm_dispatch():
    asked = []

    def fused_config(tokens, hidden, n):  # vLLM's rule: the fused kernel up to 32 tokens
        asked.append((tokens, hidden, n))
        return None if tokens > 32 else (6, 8, 128)

    kernels = NS(mhc_fused_post_pre_split_config=fused_config)
    dg = NS(is_deep_gemm_supported=lambda: True)
    assert gp.shard_split_exact(kernels, dg, 2, 66, 4096, 4)        # shard 33
    assert asked == [(66, 4096, 4), (33, 4096, 4)]
    assert gp.shard_split_exact(kernels, dg, 2, 65, 4096, 4)        # padded shard 33
    assert not gp.shard_split_exact(kernels, dg, 2, 64, 4096, 4)    # shard 32: fused kernel
    assert not gp.shard_split_exact(kernels, dg, 2, 8, 4096, 4)
    no_dg = NS(is_deep_gemm_supported=lambda: False)
    assert not gp.shard_split_exact(kernels, no_dg, 2, 4096, 4096, 4)  # no split-k at all


def _record(obj, name, log, ranks, tag):
    fn = getattr(obj, name)

    def wrapped(*args, **kwargs):
        log.append((ranks.rank, tag, getattr(ranks.local, "forced", None)))
        return fn(*args, **kwargs)
    setattr(obj, name, wrapped)


@pytest.mark.parametrize("tokens", [8, 7])
def test_sp_pass_runs_only_the_mhc_calls_at_the_full_batch_split(tokens):
    ref, _ = _run(TwoRanks(), stock_forward, tokens)
    ranks = TwoRanks()
    stacks = [_stack(ranks), _stack(ranks)]
    log = []
    for layers in stacks:
        for layer in layers:
            _record(layer, "hc_pre", log, ranks, "mhc")
            _record(layer, "hc_fused_post_pre", log, ranks, "mhc")
            _record(layer.self_attn, "forward", log, ranks, "attn")
            _record(layer.mlp, "forward", log, ranks, "mlp")
    states = [gp.SpState("force", 2048, 2) for _ in range(2)]
    fwds = [gp.make_forward(stock_forward, _ops(ranks), st, _NoCuda) for st in states]
    got, _ = _run(ranks, fwds, tokens, passes=2, stacks=stacks)
    for r in range(2):
        assert torch.equal(got[r], ref[r])
        mine = [(tag, forced) for rank, tag, forced in log if rank == r]
        first, second = mine[:len(mine) // 2], mine[len(mine) // 2:]
        assert all(forced is None for _, forced in first)  # pass 1: stock, nothing forced
        # Pass 2 (SP): every mHC call at the full batch's token count, attention and MLP untouched.
        assert {(tag, forced) for tag, forced in second} == {("mhc", tokens), ("attn", None),
                                                            ("mlp", None)}
        assert sum(tag == "mhc" for tag, _ in second) == 2 * len(stacks[r])


def test_a_pass_that_would_not_be_exact_runs_stock():
    ref, _ = _run(TwoRanks(), stock_forward, 8)
    ranks = TwoRanks()
    states = [gp.SpState("force", 2048, 2) for _ in range(2)]
    ops = _ops(ranks, exact=lambda t, hidden, n: False)
    fwds = [gp.make_forward(stock_forward, ops, st, _NoCuda) for st in states]
    got, stacks = _run(ranks, fwds, 8, passes=2)
    for r in range(2):
        assert torch.equal(got[r], ref[r])
    n_layers = len(stacks[0])
    assert ranks.calls == {"all_reduce": 4 * n_layers, "all_gather": 0, "reduce_scatter": 0}
    assert all(s.ready and not s.pass_sp for s in states)


def test_measurement_skips_inexact_sizes_and_times_the_shard_at_the_full_split(monkeypatch):
    ranks = TwoRanks()
    layer = _layer_for_prepare()
    monkeypatch.setattr(gp, "T_GRID", (8, 16, 32))
    timed = []

    def fake_median(fn, torch_mod):
        timed.append(getattr(ranks.local, "forced", None))
        return 1.0

    monkeypatch.setattr(gp, "_median_ms", fake_median)
    ranks.local.rank = 0
    ops = _ops(ranks, exact=lambda t, hidden, n: t >= 16)
    ops.max_across_tp = lambda v: v
    state = gp.SpState("auto", 32, 2)
    gp.measure_t_star(layer, state, ops, _CpuTorch, "cpu")
    assert [row["tokens"] for row in state.table] == [16, 32]
    # Per row: mhc(T), mhc(T/2) inside full_split(T), all_reduce, all_gather, reduce_scatter.
    assert timed == [None, 16, None, None, None, None, 32, None, None, None]


# ------------------------------------------------------------------ KDA conv per slice


def _ref_conv(x, weight, bias, activation="silu", conv_states=None, has_initial_state=None,
              cache_indices=None, query_start_loc=None, metadata=None):
    """vLLM's causal_conv1d_fn contract on CPU: x (dim, T) channel-last, varlen, conv_states
    (cache, dim, width-1) updated in place, output allocated with empty_like(x)."""
    assert x.stride(0) == 1 and x.stride(1) > 1, "channel-last only, like the kernel's wrapper"
    out = torch.empty_like(x)
    width = weight.shape[1]
    for b in range(query_start_loc.numel() - 1):
        s, e = int(query_start_loc[b]), int(query_start_loc[b + 1])
        ci = int(cache_indices[b])
        hist = conv_states[ci].clone() if bool(has_initial_state[b]) else \
            torch.zeros(x.shape[0], width - 1, dtype=x.dtype)
        xs = torch.cat([hist, x[:, s:e]], dim=1)
        for t in range(e - s):
            acc = torch.zeros(x.shape[0], dtype=torch.float32)
            for k in range(width):
                acc = acc + weight[:, k].float() * xs[:, t + k].float()
            if bias is not None:
                acc = acc + bias.float()
            out[:, s + t] = torch.nn.functional.silu(acc).to(x.dtype)
        conv_states[ci].copy_(xs[:, -(width - 1):])
    return out


def _conv_inputs(p=8, width=4, lens=(5, 2, 9), seed=0):
    g = torch.Generator().manual_seed(seed)
    t = sum(lens)
    qkv = torch.randn(t, 3 * p, generator=g).to(torch.bfloat16)
    weight = torch.randn(3 * p, width, generator=g).to(torch.bfloat16)
    states = torch.randn(4, 3 * p, width - 1, generator=g).to(torch.bfloat16)
    qsl = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
    has = torch.tensor([True, False, True][:len(lens)])
    idx = torch.tensor([2, 0, 3][:len(lens)], dtype=torch.int32)
    return qkv, weight, states, has, idx, qsl


def test_kda_conv_split_equals_the_merged_conv_and_is_dense():
    p = 8
    qkv, weight, states, has, idx, qsl = _conv_inputs(p=p, lens=(5, 2, 9))  # one sequence shorter than width
    s_stock, s_split = states.clone(), states.clone()
    merged = _ref_conv(qkv.transpose(0, 1), weight, None, conv_states=s_stock, has_initial_state=has,
                       cache_indices=idx, query_start_loc=qsl).transpose(0, 1)
    q0, k0, v0 = merged.split(p, dim=-1)
    q, k, v = gp.conv_split(_ref_conv, qkv, weight, None, s_split, has, idx, qsl, None, p)
    for a, b in ((q, q0), (k, k0), (v, v0)):
        assert torch.equal(a, b)
        assert a.is_contiguous()  # FlashKDA's .contiguous() is then a no-op
    assert not q0.is_contiguous()  # the stock split is row-strided: what FlashKDA copies
    assert torch.equal(s_split, s_stock)


def test_kda_conv_split_keeps_the_merged_call_with_a_bias():
    p = 8
    qkv, weight, states, has, idx, qsl = _conv_inputs(p=p)
    bias = torch.randn(3 * p).to(torch.bfloat16)
    calls = []

    def conv(x, *a, **kw):
        calls.append(tuple(x.shape))
        return _ref_conv(x, *a, **kw)

    q, k, v = gp.conv_split(conv, qkv, weight, bias, states, has, idx, qsl, None, p)
    assert calls == [(3 * p, qkv.shape[0])]
    assert not q.is_contiguous()


_KDA_SRC = '''
def eager_break_during_capture(fn):
    fn._decorated = True
    return fn


class Glm5NextLinearAttention:
    @eager_break_during_capture
    def _forward(self, qkv_ns, conv_weights, conv_bias, conv_state, has_initial_state,
                 non_spec_state_indices_tensor, non_spec_query_start_loc, attn_metadata_narrowed):
        if True:
%s
        return q_ns, k_ns, v_ns
'''


def _kda_module(tmp_path, body, name="kda_fake"):
    import importlib.util
    f = tmp_path / f"{name}.py"
    f.write_text(_KDA_SRC % body)
    spec = importlib.util.spec_from_file_location(name, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.causal_conv1d_fn = _ref_conv
    return mod


def test_kda_recompiled_forward_takes_the_split_and_keeps_the_decorator(tmp_path):
    mod = _kda_module(tmp_path, gp.KDA_STOCK_CONV_BLOCK.rstrip("\n"))
    stock = mod.Glm5NextLinearAttention._forward
    new, why = gp.recompile_kda_forward(mod)
    assert new is not None, why
    assert getattr(new, "_decorated", False) and new._tessera_kda_conv_split
    assert "_tessera_glm53_conv_split" in new.__code__.co_names
    assert "_tessera_glm53_conv_split" not in vars(mod)  # vLLM's namespace untouched
    p = 8
    qkv, weight, states, has, idx, qsl = _conv_inputs(p=p)
    self_ = NS(local_projection_size=p)
    s1, s2 = states.clone(), states.clone()
    want = stock(self_, qkv, weight, None, s1, has, idx, qsl, None)
    got = new(self_, qkv, weight, None, s2, has, idx, qsl, None)
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert all(t.is_contiguous() for t in got) and torch.equal(s1, s2)


@pytest.mark.parametrize("body, needle", [
    ("            q_ns = k_ns = v_ns = None", "occurs 0 times"),
    (gp.KDA_STOCK_CONV_BLOCK.rstrip("\n") + "\n" + gp.KDA_STOCK_CONV_BLOCK.rstrip("\n"), "occurs 2 times"),
])
def test_kda_recompile_declines_unless_the_block_occurs_once(tmp_path, body, needle):
    new, why = gp.recompile_kda_forward(_kda_module(tmp_path, body))
    assert new is None and needle in why


def test_kda_recompile_declines_on_other_decorators(tmp_path):
    mod = _kda_module(tmp_path, gp.KDA_STOCK_CONV_BLOCK.rstrip("\n"))
    src = (tmp_path / "kda_fake.py").read_text().replace("    @eager_break_during_capture\n", "")
    (tmp_path / "kda_fake.py").write_text(src)
    new, why = gp.recompile_kda_forward(mod)
    assert new is None and "decorators" in why


def test_kda_recompile_compiles_exactly_the_edited_stock_method(tmp_path):
    """The served ``_forward`` is the stock method's own lines with the one block swapped,
    dedented and compiled under the override's file name, in a copy of the module's
    namespace.  This pins what runs, whichever module does the parse and the exec."""
    import ast
    import textwrap

    mod = _kda_module(tmp_path, gp.KDA_STOCK_CONV_BLOCK.rstrip("\n"))
    new, why = gp.recompile_kda_forward(mod)
    assert new is not None, why
    src = (tmp_path / "kda_fake.py").read_text()
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == gp.KDA_CLASS)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == gp.KDA_METHOD)
    text = "".join(src.splitlines(keepends=True)[fn.lineno - 1:fn.end_lineno])
    text = textwrap.dedent(text.replace(gp.KDA_STOCK_CONV_BLOCK, gp.KDA_SPLIT_CONV_BLOCK))
    ns: dict = {}
    exec(compile(text, f"<tessera glm53_prefill: {gp.KDA_CLASS}.{gp.KDA_METHOD} conv split>", "exec"), ns)
    want, got = ns[gp.KDA_METHOD].__code__, new.__code__
    for attr in ("co_code", "co_consts", "co_names", "co_varnames", "co_filename", "co_firstlineno",
                 "co_argcount"):
        assert getattr(got, attr) == getattr(want, attr), attr
    assert set(new.__globals__) == set(vars(mod)) | {"_tessera_glm53_conv_split", gp.KDA_METHOD}
    assert new.__globals__["_tessera_glm53_conv_split"] is gp.conv_split


def test_serve_start_imports_and_install_order(monkeypatch):
    """What ``install_for_current_config`` does at serve start with every flag on and no
    inspected interface matching: the ONORM append, then SP mHC (importing ``SP_MODULES``
    in order), then the KDA conv split (importing ``KDA_MODULES``), and nothing else."""
    import sys
    import types

    cfg = _config()
    vllm = types.ModuleType("vllm")
    vllm_config = types.ModuleType("vllm.config")
    vllm_config.get_current_vllm_config_or_none = lambda: cfg
    vllm.config = vllm_config
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", vllm_config)
    for name, value in (("TESSERA_GLM53_ONORM_CUDA", "1"), ("TESSERA_GLM53_SP_MHC", "force"),
                        ("TESSERA_GLM53_KDA_CONV_SPLIT", "on")):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(gp, "_INSTALLED", {})
    monkeypatch.setattr(gp, "_ONORM_DONE", type(gp._ONORM_DONE)())
    events = []

    def fake_import(name):
        events.append(("import", name))
        return types.ModuleType(name)  # no __file__, so no digest and no interface matches

    monkeypatch.setattr(gp, "importlib", NS(import_module=fake_import))
    for fn_name in ("enable_onorm_cuda", "install_sp_mhc", "install_kda_conv_split"):
        def wrapped(config, orig=getattr(gp, fn_name), fn_name=fn_name):
            events.append(("install", fn_name))
            return orig(config)
        monkeypatch.setattr(gp, fn_name, wrapped)

    gp.install_for_current_config()

    assert events == ([("install", "enable_onorm_cuda"), ("install", "install_sp_mhc")]
                      + [("import", n) for n in gp.SP_MODULES]
                      + [("install", "install_kda_conv_split")]
                      + [("import", n) for n in gp.KDA_MODULES])
    assert cfg.compilation_config.custom_ops == ["none", "+fused_rms_norm_gated"]


def test_kda_install_off_by_default_and_declines_on_digest_mismatch(monkeypatch, tmp_path):
    monkeypatch.delenv("TESSERA_GLM53_KDA_CONV_SPLIT", raising=False)
    assert gp.kda_conv_split_mode() == "off" and gp.install_kda_conv_split(_config()) is False
    monkeypatch.setenv("TESSERA_GLM53_KDA_CONV_SPLIT", "sometimes")
    with pytest.raises(ValueError):
        gp.kda_conv_split_mode()
    monkeypatch.setenv("TESSERA_GLM53_KDA_CONV_SPLIT", "on")
    mod = _kda_module(tmp_path, gp.KDA_STOCK_CONV_BLOCK.rstrip("\n"))
    conv = NS(__name__=gp.KDA_MODULES[1], __file__=str(tmp_path / "kda_fake.py"))
    monkeypatch.setattr(gp, "_import_all", lambda names=gp.SP_MODULES: ((mod, conv), ""))
    stock = mod.Glm5NextLinearAttention._forward
    assert gp._install_kda_conv_split(_config()) is False  # digests are not the inspected ones
    assert mod.Glm5NextLinearAttention._forward is stock
    digest = hashlib.sha256((tmp_path / "kda_fake.py").read_bytes()).hexdigest()
    monkeypatch.setattr(gp, "_KDA_INTERFACES", (gp._Interface("test", (digest, digest)),))
    assert gp._install_kda_conv_split(_config()) is True
    assert mod.Glm5NextLinearAttention._forward._tessera_kda_conv_split
    assert gp._install_kda_conv_split(_config()) is True  # idempotent
    assert gp._install_kda_conv_split(_config(architectures=["Other"])) is False


def test_kda_pinned_interface_names_every_module():
    for interface in gp._KDA_INTERFACES:
        assert len(interface.digests) == len(gp.KDA_MODULES)
        assert all(len(d) == 64 for d in interface.digests)


# ------------------------------------------------------------------- mHC token tiles


def test_mhc_tile_env(monkeypatch):
    for off in ("", "off", "OFF", "0"):
        monkeypatch.setenv("TESSERA_GLM53_MHC_TILE", off)
        assert gp.mhc_tile() is None
    monkeypatch.delenv("TESSERA_GLM53_MHC_TILE")
    assert gp.mhc_tile() is None
    monkeypatch.setenv("TESSERA_GLM53_MHC_TILE", " 512 ")
    assert gp.mhc_tile() == 512
    for bad in ("-128", "abc", "1.5"):
        monkeypatch.setenv("TESSERA_GLM53_MHC_TILE", bad)
        with pytest.raises(ValueError, match="TESSERA_GLM53_MHC_TILE"):
            gp.mhc_tile()


def test_tile_only_installs_the_rebind_with_sp_off(monkeypatch):
    monkeypatch.setenv("TESSERA_GLM53_SP_MHC", "off")
    monkeypatch.setenv("TESSERA_GLM53_MHC_TILE", "512")
    seen = []
    monkeypatch.setattr(gp, "_install_sp_mhc", lambda cfg, mode, tile=None: seen.append((mode, tile)) or True)
    monkeypatch.setattr(gp, "_INSTALLED", {})
    assert gp.install_sp_mhc(_config()) and seen == [("off", 512)]
    monkeypatch.setenv("TESSERA_GLM53_MHC_TILE", "off")
    monkeypatch.setattr(gp, "_INSTALLED", {})
    assert not gp.install_sp_mhc(_config()) and seen == [("off", 512)]


def test_state_tile_decisions():
    s = gp.SpState("off", 2048, 2, tile=4)
    assert not s.sp_on
    assert not s.begin_pass(8, capturing=False, exact=True, tile_exact=True) and s.pass_tile
    s.begin_pass(8, capturing=True, exact=True, tile_exact=True)
    assert not s.pass_tile                               # a capture: stock sequence
    s.begin_pass(8, capturing=False, exact=True, tile_exact=False)
    assert not s.pass_tile                               # full batch off the split-k path
    s.begin_pass(4, capturing=False, exact=True, tile_exact=True)
    assert not s.pass_tile                               # one tile: nothing to split
    assert not s.pass_sp
    f = gp.SpState("force", 2048, 2, tile=4)
    assert f.sp_on
    f.begin_pass(10, capturing=False, exact=True, tile_exact=True)
    assert not f.pass_sp and f.pass_tile                 # pass 1: not SP, the 10-token call tiles
    assert f.begin_pass(10, capturing=False, exact=True, tile_exact=True) and f.pass_tile  # shard 5
    assert f.begin_pass(8, capturing=False, exact=True, tile_exact=True) and not f.pass_tile  # shard 4
    assert not gp.SpState("off", 2048, 2).begin_pass(8, False, exact=True, tile_exact=True)
    assert not gp.SpState("force", 2048, 2).pass_tile


@pytest.mark.parametrize("tokens", [8, 7])
def test_tiled_pass_equals_stock_and_keeps_the_module_reductions(tokens):
    ref_ranks = TwoRanks()
    ref, _ = _run(ref_ranks, stock_forward, tokens)
    ranks = TwoRanks()
    tiled = []
    states = [gp.SpState("off", 2048, 2, tile=3) for _ in range(2)]
    ops = _ops(ranks, tile=3, tiled=tiled)
    fwds = [gp.make_forward(stock_forward, ops, st, _NoCuda) for st in states]
    got, stacks = _run(ranks, fwds, tokens, passes=2)
    for r in range(2):
        assert torch.equal(got[r], ref[r]), (tokens, r, (got[r] - ref[r]).abs().max())
    n_layers = len(stacks[0])
    # No SP: the modules reduce, as stock does, twice per layer per pass.
    assert ranks.calls == {"all_reduce": 2 * 2 * n_layers, "all_gather": 0, "reduce_scatter": 0}
    for layer in stacks[0]:
        assert "_tessera_sp_ready" not in layer.__dict__
        assert layer.self_attn.o_proj.reduce_results is True
    # Every hc_fused_post_pre call tiles (layer 0's first site is hc_pre), at the full batch's split.
    assert len(tiled) == 2 * 2 * (2 * n_layers - 1)
    assert set(tiled) == {(0, tokens, tokens), (1, tokens, tokens)}
    assert all(s.pass_tile and not s.pass_sp for s in states)


def test_tiles_off_the_split_k_path_run_stock():
    ref, _ = _run(TwoRanks(), stock_forward, 8)
    ranks = TwoRanks()
    tiled = []
    states = [gp.SpState("off", 2048, 2, tile=3) for _ in range(2)]
    ops = _ops(ranks, tile=3, tiled=tiled, tile_exact=lambda t, hidden, n: False)
    got, _ = _run(ranks, [gp.make_forward(stock_forward, ops, st, _NoCuda) for st in states], 8,
                  passes=2)
    for r in range(2):
        assert torch.equal(got[r], ref[r])
    assert tiled == [] and not any(s.pass_tile for s in states)


@pytest.mark.parametrize("tokens", [8, 7])
def test_sp_and_tiles_compose(tokens):
    ref, _ = _run(TwoRanks(), stock_forward, tokens)
    ranks = TwoRanks()
    tiled = []
    states = [gp.SpState("force", 2048, 2, tile=2) for _ in range(2)]
    ops = _ops(ranks, tile=2, tiled=tiled)
    got, stacks = _run(ranks, [gp.make_forward(stock_forward, ops, st, _NoCuda) for st in states],
                       tokens, passes=2)
    for r in range(2):
        assert torch.equal(got[r], ref[r]), (tokens, r)
    n_layers = len(stacks[0])
    per_pass = 2 * (2 * n_layers - 1)
    shard = -(-tokens // 2)
    # Pass 1 (never SP) tiles the whole batch; pass 2 (SP) tiles each rank's shard, both at the
    # full batch's split.
    assert sorted(tiled[:0]) == []
    assert sorted(set(tiled)) == sorted({(0, tokens, tokens), (1, tokens, tokens),
                                         (0, shard, tokens), (1, shard, tokens)})
    assert len(tiled) == 2 * per_pass
    assert all(s.pass_sp and s.pass_tile for s in states)


class _TileKernels:
    """Per-token CPU stand-ins with the stock kernels' call signatures, which record each call."""

    def __init__(self):
        self.calls = []
        self.torch = torch

    def post(self, comb, residual, post_mix, x, out, hc_mult, hidden):
        self.calls.append(("post", residual.shape[0], out.data_ptr(), out.is_contiguous()))
        out.copy_((torch.einsum("tij,tjh->tih", comb, residual.float())
                   + post_mix.unsqueeze(-1) * x.float().unsqueeze(1)).to(out.dtype))

    def gemm(self, x2d, fn, *, hidden_size, hc_mult):
        self.calls.append(("gemm", x2d.shape[0], x2d.data_ptr(), x2d.is_contiguous()))
        xf = x2d.float()
        return (xf @ fn.T).unsqueeze(0), xf.square().sum(-1).unsqueeze(0)

    def pre(self, mul, sqrsum, scale, base, residual, post_mix, comb_mix, layer_input, rms_eps,
            hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, *, norm_weight,
            norm_eps):
        self.calls.append(("pre", residual.shape[0], layer_input.data_ptr(), post_mix.is_contiguous()))
        rms = torch.rsqrt(sqrsum[0] / mul.shape[-1] + rms_eps).unsqueeze(-1)
        mix = mul.sum(0) * rms * scale[0] + base
        n = residual.shape[1]
        post_mix.copy_(torch.sigmoid(mix[:, :n]) * hc_post_mult_value)
        comb_mix.copy_(torch.softmax(mix[:, 2 * n:], -1))
        li = (torch.sigmoid(mix[:, n:2 * n]).unsqueeze(-1) * residual.float()).sum(1)
        li = li * torch.rsqrt(li.pow(2).mean(-1, keepdim=True) + norm_eps) * norm_weight.float()
        layer_input.copy_(li.to(layer_input.dtype))


def _tile_inputs(t, n=4, h=8, seed=3):
    g = torch.Generator().manual_seed(seed)
    mix = n * (n + 2)
    return dict(x=torch.randn(t, h, generator=g).bfloat16(),
                residual=torch.randn(t, n, h, generator=g).bfloat16(),
                post_layer_mix=torch.rand(t, n, 1, generator=g),
                comb_res_mix=torch.softmax(torch.randn(t, n, n, generator=g), -1),
                fn=torch.randn(mix, n * h, generator=g), hc_scale=torch.rand(3, generator=g),
                hc_base=torch.randn(mix, generator=g), rms_eps=1e-5, hc_pre_eps=1e-6,
                hc_sinkhorn_eps=1e-6, hc_post_mult_value=2.0, sinkhorn_repeat=20,
                norm_weight=(1 + torch.randn(h, generator=g)).float(), norm_eps=1e-5)


@pytest.mark.parametrize("t, tile", [(8, 3), (7, 2), (8, 8), (5, 64)])
def test_tiled_body_equals_one_call_and_writes_in_place(t, tile):
    inp = _tile_inputs(t)
    one = gp.tiled_fused_post_pre(_TileKernels(), t, **inp)
    k = _TileKernels()
    got = gp.tiled_fused_post_pre(k, tile, **inp)
    for a, b in zip(got, one):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    assert [o.shape for o in got] == [(t, 4, 8), (t, 4, 1), (t, 4, 4), (t, 8)]
    sizes = [min(tile, t - a) for a in range(0, t, tile)]
    assert [c[:2] for c in k.calls] == [(kind, n) for n in sizes for kind in ("post", "gemm", "pre")]
    # Each tile's post writes, and its GEMM reads, the tile's slice of the one residual output;
    # its pre writes the tile's slice of the one layer input.  Nothing is joined afterwards.
    res, li = got[0], got[3]
    starts = [a for a in range(0, t, tile)]
    posts = [c for c in k.calls if c[0] == "post"]
    gemms = [c for c in k.calls if c[0] == "gemm"]
    pres = [c for c in k.calls if c[0] == "pre"]
    assert [c[2] for c in posts] == [res[a:].data_ptr() for a in starts] == [c[2] for c in gemms]
    assert [c[2] for c in pres] == [li[a:].data_ptr() for a in starts]
    assert all(c[3] for c in k.calls)


def test_tiled_body_converts_the_norm_weight_as_stock_does():
    inp = _tile_inputs(6)
    k = _TileKernels()
    seen = []
    pre = k.pre

    def spy(*a, norm_weight, norm_eps):
        seen.append((norm_weight.dtype, norm_weight.is_contiguous()))
        return pre(*a, norm_weight=norm_weight, norm_eps=norm_eps)
    k.pre = spy
    inp["norm_weight"] = torch.randn(16).float()[::2]  # float32 and strided
    gp.tiled_fused_post_pre(k, 4, **inp)
    assert seen == [(torch.bfloat16, True)] * 2


def test_dispatches_cuda_reads_the_resolved_forward():
    class Op:
        def forward_cuda(self):
            pass

        def forward_native(self):
            pass

    op = Op()
    assert not gp._dispatches_cuda(op)
    op._forward_method = op.forward_native
    assert not gp._dispatches_cuda(op)
    op._forward_method = op.forward_cuda
    assert gp._dispatches_cuda(op)
