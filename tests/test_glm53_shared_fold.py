"""CPU stand-in for folding the MoE shared-expert add into the routed token sum.

The bitwise claim on the real kernel is a GPU receipt
(``tests/test_glm53_shared_fold_cuda.py`` and
``experiments/t8r_speed/shared_fold_check.py``). These tests pin the host-side
contract:

- the routed call folds only when every condition holds, and otherwise calls
  the adapter exactly as before;
- the runner skips its add only for the fold that was just made, and refuses a
  record that doesn't match;
- the install runs from the production quant-config entry, only when opted in,
  only on inspected sources, and only in compilation mode ``NONE``.
"""
from __future__ import annotations

import enum
import hashlib
import logging
import sys
import types

import pytest
import torch

import test_serving_dispatch as dispatch
from test_serving_dispatch import TARGET, TESSERA_MODE_ENV, _config, _layer
from test_serving_dispatch import runtime_modules as _runtime_modules
from test_serving_dispatch import _the_platform_these_tests_are_about  # noqa: F401  (autouse)

runtime_modules = _runtime_modules  # pytest discovers the isolated runtime fixture

RUNNER = "vllm.model_executor.layers.fused_moe.runner.moe_runner"
SHARED = "vllm.model_executor.layers.fused_moe.runner.shared_experts"
H = 64


class SharedExpertsOrder(enum.IntEnum):
    NONE = 0
    NO_OVERLAP = 1
    MK_INTERNAL_OVERLAPPED = 2
    MULTI_STREAM_OVERLAPPED = 3


class SharedExperts:
    """The stock coordinator's slots, as the fold reads them."""

    def __init__(self, order=SharedExpertsOrder.NO_OVERLAP, enable_dbo=False):
        self.order = order
        self.enable_dbo = enable_dbo
        self._output = [None, None]

    @property
    def _output_idx(self):
        return 0

    def _determine_shared_experts_order(self, hidden_states):
        return self.order

    def run(self, shared_input):
        """The NO_OVERLAP call before the routed experts."""
        self._output[0] = (shared_input.float() * 0.5 + 1).bfloat16()

    @property
    def output(self):
        out, self._output[0] = self._output[0], None
        return out


class MoERunner:
    """The stock runner's combine tail (``moe_runner.py:778-786``)."""

    def __init__(self, shared_experts, **overrides):
        self._shared_experts = shared_experts
        self.routed_scaling_factor = 1.0
        self._fused_output_is_reduced = False
        self.routed_input_transform = None
        self.routed_output_transform = None
        self._quant_method = types.SimpleNamespace(has_unpadded_output=False)
        self.do_naive_dispatch_combine = False
        self.moe_config = types.SimpleNamespace(pcp_size=1)
        for key, value in overrides.items():
            setattr(self, key, value)

    def _maybe_apply_routed_scale_to_output(self, shared_output, fused_output):
        if self.routed_scaling_factor != 1.0:
            fused_output *= self.routed_scaling_factor
        return shared_output, fused_output

    def combine(self, shared_output, fused_output):
        shared_output, fused_output = self._maybe_apply_routed_scale_to_output(
            shared_output, fused_output)
        if shared_output is not None:
            return shared_output + fused_output
        return fused_output


class Adapter:
    """A routed adapter with the fused lane's ``shared=`` contract."""

    supports_shared_fold = True

    def __init__(self):
        self.calls = []

    def __call__(self, x, expert_ids, routing_weights, *, swiglu_limit=None,
                 apply_router_weight_on_input=False, **extra):
        self.calls.append(extra)
        routed = (x.float() * routing_weights.sum(-1, keepdim=True)).bfloat16()
        shared = extra.get("shared")
        return routed if shared is None else (shared.float() + routed.float()).bfloat16()


class PlainAdapter(Adapter):
    supports_shared_fold = False


def _forward(fold, runner, adapter, x, shared_input=None):
    """One MoE layer as the stock runner runs it: shared first, routed, combine."""
    shared_input = x if shared_input is None else shared_input
    runner._shared_experts.run(shared_input)
    ids = torch.zeros((x.shape[0], 2), dtype=torch.int32)
    weights = torch.full((x.shape[0], 2), 0.75)
    fused = fold.native_call(adapter, x, ids, weights, shared_experts=runner._shared_experts,
                             shared_experts_input=shared_input, swiglu_limit=None,
                             apply_router_weight_on_input=False)
    return runner.combine(runner._shared_experts.output, fused)


def _stock(x):
    """The same layer with no fold: shared + routed, two roundings."""
    shared = (x.float() * 0.5 + 1).bfloat16()
    routed = (x.float() * 1.5).bfloat16()
    return shared + routed


def _x(tokens=8):
    return torch.randn(tokens, H).bfloat16()


def _module(monkeypatch, tmp_path, name, **attrs):
    source = tmp_path / (name.rsplit(".", 1)[1] + ".py")
    source.write_text(f"# stand-in for the inspected {name}\n")
    module = types.ModuleType(name)
    module.__file__ = str(source)
    for key, value in attrs.items():
        setattr(module, key, value)
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    monkeypatch.setitem(sys.modules, name, module)
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _compilation(monkeypatch, mode_name):
    config = sys.modules.get("vllm.config")
    if config is None:
        config = types.ModuleType("vllm.config")
        monkeypatch.setitem(sys.modules, "vllm.config", config)
    mode = types.SimpleNamespace(name=mode_name)
    context = types.SimpleNamespace(speculative_config=None,
                                    compilation_config=types.SimpleNamespace(mode=mode))
    monkeypatch.setattr(config, "get_current_vllm_config_or_none", lambda: context,
                        raising=False)


@pytest.fixture
def shared_fold(monkeypatch, tmp_path):
    """The module under test with a fresh flag latch and stand-in stock modules."""
    from tessera.serving import flags, glm53_shared_fold as mod

    flags.reset_for_tests(mod.FLAG)
    monkeypatch.setattr(mod, "_REPORTED", [])
    monkeypatch.setattr(mod, "_FIRST", set())
    monkeypatch.setattr(mod, "_STATE", {})
    monkeypatch.setattr(mod, "_PENDING", types.SimpleNamespace())

    class Runner(MoERunner):
        pass

    original = Runner._maybe_apply_routed_scale_to_output
    runner_sha = _module(monkeypatch, tmp_path, RUNNER, MoERunner=Runner)
    shared_sha = _module(monkeypatch, tmp_path, SHARED, SharedExpertsOrder=SharedExpertsOrder,
                         SharedExperts=SharedExperts)
    monkeypatch.setattr(mod, "_INSPECTED_SHA256", {RUNNER: frozenset({runner_sha}),
                                                   SHARED: frozenset({shared_sha})})
    _compilation(monkeypatch, "NONE")
    monkeypatch.delenv("TESSERA_CENSUS_RUNTIME_IMAGE", raising=False)
    yield types.SimpleNamespace(mod=mod, runner=Runner, original=original)
    flags.reset_for_tests(mod.FLAG)


def _install(shared_fold, monkeypatch, caplog=None):
    monkeypatch.setenv(shared_fold.mod.FLAG, "1")
    assert shared_fold.mod.install_for_current_config() is True
    if caplog is not None:
        caplog.clear()  # the install line is another test's subject


def _lines(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == "tessera.serving.glm53_shared_fold"]


def test_off_by_default_logs_one_line_and_calls_the_adapter_unchanged(shared_fold, monkeypatch,
                                                                      caplog):
    monkeypatch.delenv(shared_fold.mod.FLAG, raising=False)
    with caplog.at_level(logging.WARNING):
        assert shared_fold.mod.install_for_current_config() is False
        assert shared_fold.mod.install_for_current_config() is False
    assert _lines(caplog) == [
        "tessera.glm53_shared_fold: shared add fold off (TESSERA_GLM53_FOLD_SHARED_ADD unset or 0)"]
    assert shared_fold.runner._maybe_apply_routed_scale_to_output is shared_fold.original
    adapter, x = Adapter(), _x()
    runner = shared_fold.runner(SharedExperts())
    for _ in range(2):
        assert torch.equal(_forward(shared_fold.mod, runner, adapter, x), _stock(x))
    assert adapter.calls == [{}, {}]


def test_installs_from_the_production_quant_config_entry(runtime_modules, shared_fold,
                                                         monkeypatch, caplog):
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    monkeypatch.setenv(shared_fold.mod.FLAG, "1")
    config = dispatch.TesseraConfig.from_config(_config())
    with caplog.at_level(logging.WARNING):
        config.get_quant_method(_layer(), TARGET)
        config.get_quant_method(_layer(), TARGET)
    installed = [line for line in _lines(caplog) if "shared add fold" in line]
    assert len(installed) == 1 and " installed (stock source sha256 " in installed[0]
    assert getattr(shared_fold.runner._maybe_apply_routed_scale_to_output,
                   shared_fold.mod._MARK, False)


def test_the_fold_matches_the_stock_layer_bitwise_after_the_runner_is_checked(
        shared_fold, monkeypatch, caplog):
    _install(shared_fold, monkeypatch, caplog)
    adapter, runner = Adapter(), shared_fold.runner(SharedExperts())
    x = _x()
    with caplog.at_level(logging.WARNING):
        # The runner's first forward (vLLM's profile run): checked after the routed call.
        first = _forward(shared_fold.mod, runner, adapter, x)
        assert adapter.calls == [{}]
        assert runner._shared_experts._tessera_shared_fold is True
        for _ in range(2):
            out = _forward(shared_fold.mod, runner, adapter, x)
    assert torch.equal(first.view(torch.int16), _stock(x).view(torch.int16))
    assert torch.equal(out.view(torch.int16), _stock(x).view(torch.int16))
    assert [sorted(c) for c in adapter.calls[1:]] == [["shared"], ["shared"]]
    assert getattr(shared_fold.mod._PENDING, "fold", None) is None
    assert _lines(caplog) == ["tessera.glm53_shared_fold: shared add fold: first folded call "
                              "(8, 64): shared add folded into token_sum"]


def _checked(runner):
    runner._shared_experts._tessera_shared_fold = True
    return runner


@pytest.mark.parametrize("case,reason", [
    ("adapter", "the routed adapter PlainAdapter has no shared fold"),
    ("order", "shared experts order is MULTI_STREAM_OVERLAPPED, not NO_OVERLAP"),
    ("padded", "the routed input is padded or transformed"),
    ("empty", "the routed input is not a non-empty [T, H] tensor"),
    ("runner", "dual-batch overlap is on"),
])
def test_a_declined_call_reaches_the_adapter_unchanged_and_the_runner_adds(
        shared_fold, monkeypatch, caplog, case, reason):
    _install(shared_fold, monkeypatch, caplog)
    adapter = PlainAdapter() if case == "adapter" else Adapter()
    order = (SharedExpertsOrder.MULTI_STREAM_OVERLAPPED if case == "order"
             else SharedExpertsOrder.NO_OVERLAP)
    runner = _checked(shared_fold.runner(SharedExperts(order=order)))
    if case == "runner":
        runner._shared_experts._tessera_shared_fold = reason
    x = _x(0 if case == "empty" else 8)
    shared_input = torch.randn(x.shape[0], H + 8).bfloat16() if case == "padded" else None
    with caplog.at_level(logging.WARNING):
        if case == "padded":
            runner._shared_experts.run(shared_input)
            runner._shared_experts._output[0] = runner._shared_experts._output[0][:, :H].contiguous()
            ids = torch.zeros((8, 2), dtype=torch.int32)
            out = shared_fold.mod.native_call(adapter, x, ids, torch.full((8, 2), 0.75),
                                              shared_experts=runner._shared_experts,
                                              shared_experts_input=shared_input)
            assert adapter.calls == [{}]
            assert torch.equal(out, (x.float() * 1.5).bfloat16())
        else:
            out = _forward(shared_fold.mod, runner, adapter, x)
            assert adapter.calls == [{}]
            assert torch.equal(out, _stock(x))
    assert getattr(shared_fold.mod._PENDING, "fold", None) is None
    assert _lines(caplog) == [f"tessera.glm53_shared_fold: shared add fold: first kept call "
                              f"{tuple(x.shape)}: shared add kept ({reason})"]


@pytest.mark.parametrize("bad", ["missing", "dtype", "strided", "shape"])
def test_a_shared_output_that_is_not_contiguous_bf16_t_h_is_kept(shared_fold, monkeypatch, bad):
    _install(shared_fold, monkeypatch)
    adapter = Adapter()
    runner = _checked(shared_fold.runner(SharedExperts()))
    x = _x()
    shared = (x.float() * 0.5 + 1).bfloat16()
    runner._shared_experts._output[0] = {
        "missing": None,
        "dtype": shared.float(),
        "strided": torch.cat([shared, shared], dim=1)[:, ::2],
        "shape": shared[:4],
    }[bad]
    ids = torch.zeros((8, 2), dtype=torch.int32)
    shared_fold.mod.native_call(adapter, x, ids, torch.full((8, 2), 0.75),
                                shared_experts=runner._shared_experts, shared_experts_input=x)
    assert adapter.calls == [{}]
    assert getattr(shared_fold.mod._PENDING, "fold", None) is None


@pytest.mark.parametrize("override,reason", [
    ({"routed_scaling_factor": 2.5}, "routed_scaling_factor is 2.5, not 1.0"),
    ({"_fused_output_is_reduced": True}, "the routed kernel reduces its own output"),
    ({"routed_output_transform": object()}, "a routed input or output transform is set"),
    ({"routed_input_transform": object()}, "a routed input or output transform is set"),
    ({"_quant_method": types.SimpleNamespace(has_unpadded_output=True)},
     "the quant method may return unpadded output"),
    ({"do_naive_dispatch_combine": True}, "naive dispatch/combine is on"),
    ({"moe_config": types.SimpleNamespace(pcp_size=2)}, "prefill context parallel size is 2"),
    ({"_shared_experts": None}, "the runner has no shared experts"),
])
def test_a_runner_whose_combine_is_not_the_plain_add_declines(shared_fold, override, reason):
    runner = shared_fold.runner(SharedExperts(), **override)
    assert shared_fold.mod.runner_decline_reason(runner).startswith(reason)
    assert shared_fold.mod.runner_decline_reason(shared_fold.runner(SharedExperts())) is None
    dbo = shared_fold.runner(SharedExperts(enable_dbo=True))
    assert shared_fold.mod.runner_decline_reason(dbo) == "dual-batch overlap is on"


def test_the_runner_check_runs_once_and_a_declined_runner_keeps_adding(shared_fold, monkeypatch):
    _install(shared_fold, monkeypatch)
    adapter = Adapter()
    runner = shared_fold.runner(SharedExperts(), routed_scaling_factor=2.5)
    x = _x()
    for _ in range(3):
        shared_fold.mod.native_call(adapter, x, torch.zeros((8, 2), dtype=torch.int32),
                                    torch.full((8, 2), 0.75),
                                    shared_experts=runner._shared_experts, shared_experts_input=x)
        runner._shared_experts.run(x)
        runner.combine(runner._shared_experts.output, x.clone())
    assert runner._shared_experts._tessera_shared_fold == "routed_scaling_factor is 2.5, not 1.0"
    assert adapter.calls == [{}, {}, {}]


def test_a_record_that_does_not_match_the_runner_tensors_raises(shared_fold, monkeypatch):
    _install(shared_fold, monkeypatch)
    adapter = Adapter()
    runner = _checked(shared_fold.runner(SharedExperts()))
    x = _x()
    runner._shared_experts.run(x)
    fused = shared_fold.mod.native_call(adapter, x, torch.zeros((8, 2), dtype=torch.int32),
                                        torch.full((8, 2), 0.75),
                                        shared_experts=runner._shared_experts,
                                        shared_experts_input=x)
    with pytest.raises(RuntimeError, match="refusing to add the shared output twice"):
        runner.combine(runner._shared_experts.output, fused.clone())
    assert getattr(shared_fold.mod._PENDING, "fold", None) is None


def test_a_fold_never_consumed_refuses_the_next_routed_call(shared_fold, monkeypatch):
    _install(shared_fold, monkeypatch)
    adapter = Adapter()
    runner = _checked(shared_fold.runner(SharedExperts()))
    x = _x()
    kwargs = dict(shared_experts=runner._shared_experts, shared_experts_input=x)
    ids, weights = torch.zeros((8, 2), dtype=torch.int32), torch.full((8, 2), 0.75)
    runner._shared_experts.run(x)
    shared_fold.mod.native_call(adapter, x, ids, weights, **kwargs)
    with pytest.raises(RuntimeError, match="never consumed"):
        shared_fold.mod.native_call(adapter, x, ids, weights, **kwargs)


def test_install_is_idempotent_and_logs_one_line(shared_fold, monkeypatch, caplog):
    monkeypatch.setenv(shared_fold.mod.FLAG, "1")
    with caplog.at_level(logging.WARNING):
        assert shared_fold.mod.install_for_current_config() is True
        wrapped = shared_fold.runner._maybe_apply_routed_scale_to_output
        assert shared_fold.mod.install_for_current_config() is True
    assert shared_fold.runner._maybe_apply_routed_scale_to_output is wrapped
    assert wrapped.__wrapped_stock__ is shared_fold.original
    assert len(_lines(caplog)) == 1
    assert _lines(caplog)[0].startswith(
        "tessera.glm53_shared_fold: shared add fold installed (stock source sha256 ")
    assert _lines(caplog)[0].endswith(
        "image sha unstated): MoERunner skips the shared add when token_sum already made it")


def _declined(shared_fold, monkeypatch, caplog):
    monkeypatch.setenv(shared_fold.mod.FLAG, "1")
    with caplog.at_level(logging.WARNING):
        assert shared_fold.mod.install_for_current_config() is False
        assert shared_fold.mod.install_for_current_config() is False
    assert shared_fold.runner._maybe_apply_routed_scale_to_output is shared_fold.original
    assert len(_lines(caplog)) == 1
    prefix = ("tessera.glm53_shared_fold: shared add fold declined, stock "
              "MoERunner._maybe_apply_routed_scale_to_output: ")
    assert _lines(caplog)[0].startswith(prefix)
    return _lines(caplog)[0][len(prefix):]


def test_an_uninspected_source_declines(shared_fold, monkeypatch, caplog):
    monkeypatch.setitem(shared_fold.mod._INSPECTED_SHA256, RUNNER, frozenset({"0" * 64}))
    assert _declined(shared_fold, monkeypatch, caplog).endswith("is not an inspected source")


def test_a_changed_signature_declines(shared_fold, monkeypatch, caplog):
    def changed(self, shared_output, fused_output, scale):
        return shared_output, fused_output

    monkeypatch.setattr(shared_fold.runner, "_maybe_apply_routed_scale_to_output", changed)
    shared_fold.original = changed
    assert "has parameters" in _declined(shared_fold, monkeypatch, caplog)


@pytest.mark.parametrize("mode", ["VLLM_COMPILE", "STOCK_TORCH_COMPILE"])
def test_a_compiled_runner_forward_declines(shared_fold, monkeypatch, caplog, mode):
    _compilation(monkeypatch, mode)
    assert _declined(shared_fold, monkeypatch, caplog).startswith(f"compilation mode is {mode}")


def test_an_absent_runner_declines(shared_fold, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, RUNNER, None)
    assert "is not importable" in _declined(shared_fold, monkeypatch, caplog)


@pytest.mark.parametrize('fault',['missing_source','directory_source','invalid_path','missing_method','bad_signature','import_failure'])
def test_uninspectable_stock_declines_without_rebinding(shared_fold,monkeypatch,fault):
    from pathlib import Path
    mod=shared_fold.mod;monkeypatch.setenv(mod.FLAG,'1')
    stock=sys.modules[RUNNER]
    if fault=='missing_source':Path(stock.__file__).unlink()
    elif fault=='directory_source':monkeypatch.setattr(stock,'__file__',str(Path(stock.__file__).parent))
    elif fault=='invalid_path':monkeypatch.setattr(stock,'__file__',object())
    elif fault=='missing_method':monkeypatch.setattr(stock,'MoERunner',type('MissingRunner',(),{}))
    elif fault=='bad_signature':
        class BadSignature:
            def __call__(self,*args):pass
            @property
            def __signature__(self):raise ValueError('uninspectable stock method')
        monkeypatch.setattr(shared_fold.runner,'_maybe_apply_routed_scale_to_output',BadSignature())
    else:
        def unavailable(_):raise RuntimeError('stock dependency failed during import')
        monkeypatch.setattr(mod.importlib,'import_module',unavailable)
    before=shared_fold.runner._maybe_apply_routed_scale_to_output
    assert mod.install_for_current_config() is False
    assert shared_fold.runner._maybe_apply_routed_scale_to_output is before
    assert not mod._STATE.get('installed')
