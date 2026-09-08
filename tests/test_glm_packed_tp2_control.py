"""CPU instrumentation contract only; native FP8/collectives remain a separate gate."""
import contextlib
import json
import sys
import types

import pytest
import torch

from experiments.glm_packed_tp2_control import check_runner


@pytest.fixture
def runtime(monkeypatch):
    names = ['vllm', 'vllm.model_executor', 'vllm.model_executor.layers',
             'vllm.model_executor.layers.fused_moe',
             'vllm.model_executor.layers.fused_moe.runner',
             'vllm.model_executor.layers.fused_moe.runner.moe_runner',
             'vllm.forward_context', 'vllm.distributed']
    modules = {}
    for name in names:
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        if '.' in name:
            parent, child = name.rsplit('.', 1)
            setattr(modules[parent], child, module)
        modules[name] = module
    runner_module = modules[names[5]]
    runner_module.tensor_model_parallel_all_reduce = lambda x: x * 2
    modules['vllm.forward_context'].set_forward_context = lambda *a: contextlib.nullcontext()
    modules['vllm.distributed'].get_tp_group = lambda: types.SimpleNamespace(
        all_gather=lambda x, dim: torch.cat([x, x], dim=dim))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.e_score_correction_bias = torch.tensor([0., .2, -.3, .5])
        def forward(self, x):
            return torch.cat([x, -x], dim=1), None

    class Shared(torch.nn.Module):
        def forward(self, x):
            return x + 1

    def arithmetic(x, weights, ids):
        return x * (weights * (ids + 1)).sum(-1, keepdim=True)

    class Router:
        num_expert_group = topk_group = 1
        scoring_func, renormalize, routed_scaling_factor = 'sigmoid', False, 2.5
        def select_experts(self, *, hidden_states, router_logits, topk_indices_dtype, input_ids):
            scores = router_logits.sigmoid()
            ids = (scores + gate.e_score_correction_bias).topk(2, dim=1).indices
            return scores.gather(1, ids) * 2.5, ids.to(topk_indices_dtype)

    class Runner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router = Router()
            self.routed_experts = types.SimpleNamespace(global_num_experts=4,
                quant_method=types.SimpleNamespace(apply=lambda layer,x,w,i,se,si: arithmetic(x,w,i)))
            self.extra_gate = False
        def forward(self, x, router_logits):
            # Mirrors the pinned source's internal-gate contract, not FP8 kernels.
            router_logits, _ = gate(x)
            if self.extra_gate:
                gate(x)
            weights, ids = self.router.select_experts(hidden_states=x,
                router_logits=router_logits, topk_indices_dtype=torch.int32, input_ids=None)
            partial = self.routed_experts.quant_method.apply(
                self.routed_experts,x,weights,ids,None,None)
            return runner_module.tensor_model_parallel_all_reduce(shared(x) + partial)

    class Moe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate, self.shared_experts, self.experts = gate, shared, Runner()
        def forward(self, x):
            logits, _ = self.gate(x)
            return self.experts(x, router_logits=logits)

    gate, shared = Gate(), Shared()
    return Moe(), arithmetic, runner_module


@pytest.mark.parametrize('entrypoint,gate_calls', [('runner',1), ('glm',2)])
def test_internal_gate_observations_and_arithmetic(runtime, tmp_path, entrypoint, gate_calls):
    moe, oracle, _ = runtime
    path = tmp_path/'routing.json'
    x = torch.tensor([[.25, -.5], [1., 2.]])
    hints = torch.tensor([[0,1],[0,1]])
    result = check_runner(moe,None,x,hints,oracle,diagnostic_path=path,entrypoint=entrypoint)
    diagnostic = json.loads(path.read_text())
    assert diagnostic['counts']['gate'] == gate_calls
    assert diagnostic['effective_router_logits'][0] == diagnostic['gate_logits'][-1]
    if entrypoint == 'runner':
        assert diagnostic['supplied_router_logits'] != diagnostic['effective_router_logits'][0]
        assert diagnostic['actual_topk_ids'][0] != hints.tolist()
    assert result['rank_partial_vs_stock_tp2']['max_abs'] == 0
    assert result['combined_vs_stock_tp2']['max_abs'] == 0
    assert result['observed_final_all_reduce_calls'] == 1


def test_failure_persists_counts_and_restores_observers(runtime, tmp_path):
    moe, oracle, runner_module = runtime
    moe.experts.extra_gate = True
    method = moe.experts.routed_experts.quant_method
    original_apply, original_select = method.apply, moe.experts.router.select_experts
    original_reduce = runner_module.tensor_model_parallel_all_reduce
    path = tmp_path/'failed.json'
    with pytest.raises(AssertionError):
        check_runner(moe,None,torch.tensor([[1.,2.]]),torch.tensor([[0,1]]),oracle,
                     diagnostic_path=path,entrypoint='runner')
    assert json.loads(path.read_text())['counts']['gate'] == 2
    assert method.apply == original_apply
    assert moe.experts.router.select_experts == original_select
    assert runner_module.tensor_model_parallel_all_reduce == original_reduce
    assert not moe.gate._forward_hooks and not moe.shared_experts._forward_hooks
