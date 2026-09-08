"""Unexecuted-until-cleared native TP2 controls used by glm_packed_moe_control.

Each explicitly configured rank runs inside the same pinned stock vLLM image.
This module supplies no scheduler and changes no stock source files. Its oracle
is stock TP2 on independently sliced full reference weights, including the
stock runner's shared expert combination and exactly one final all-reduce.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def build_model(model_path, device, max_model_len, *, quant_config, distributed):
    import torch
    from datetime import timedelta
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import initialize_model
    from tools.tessera_construction_census import _set_default_torch_dtype

    assert distributed['world_size'] == 2 and distributed['rank'] in (0,1)
    assert distributed['init_method'].startswith('tcp://')
    torch.cuda.set_device(0)  # one visible GPU per explicitly configured host
    config = EngineArgs(model=model_path,load_format='dummy',enforce_eager=True,
        max_model_len=max_model_len,trust_remote_code=True,tensor_parallel_size=2,
        disable_custom_all_reduce=True).create_engine_config()
    config.quant_config = quant_config
    timeout = distributed['timeout_seconds']
    assert type(timeout) is int and 1 <= timeout <= 180
    config.parallel_config.distributed_timeout_seconds = timeout
    config.parallel_config.cpu_distributed_timeout_seconds = timeout
    with set_current_vllm_config(config,check_compile=False):
        init_distributed_environment(world_size=2,rank=distributed['rank'],local_rank=0,
            distributed_init_method=distributed['init_method'],backend='nccl',timeout=timedelta(seconds=timeout))
        initialize_model_parallel(2,1)
        with _set_default_torch_dtype()(config.model_config.dtype),torch.device(device):
            model = initialize_model(vllm_config=config)
    return model,quant_config,config


def verify_common_request(outer):
    import copy
    import torch
    from vllm.distributed import get_tp_group
    common = copy.deepcopy(outer)
    del common['distributed']['rank']
    digest = hashlib.sha256(json.dumps(common,sort_keys=True,separators=(',',':')).encode()).digest()
    identity = torch.tensor(list(digest),device='cuda',dtype=torch.uint8)
    both = get_tp_group().all_gather(identity,dim=0)
    assert torch.equal(both[:32],both[32:]), 'TP2 ranks have different common requests'
    return digest.hex()


def load_shared_and_gate(moe, source_root, prefix, *, index_sha256):
    """Use native dense weight loaders with original full source tensors."""
    import torch
    from safetensors import safe_open
    root = Path(source_root)
    index_path = root/'model.safetensors.index.json'
    assert hashlib.sha256(index_path.read_bytes()).hexdigest() == index_sha256
    index = json.loads(index_path.read_text())['weight_map']
    records = []
    def read(suffix):
        key = prefix + '.' + suffix
        path = root/index[key]
        with safe_open(path,framework='pt',device='cpu') as handle:
            value = handle.get_tensor(key)
        records.append({'key':key,'file':str(path),'shape':list(value.shape),
            'dtype':str(value.dtype),'tensor_sha256':hashlib.sha256(
                value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()})
        return value
    for shard,role in enumerate(('gate_proj','up_proj')):
        param = moe.shared_experts.gate_up_proj.weight
        param.weight_loader(param,read(f'shared_experts.{role}.weight'),shard)
    param = moe.shared_experts.down_proj.weight
    param.weight_loader(param,read('shared_experts.down_proj.weight'))
    param = moe.gate.weight
    param.weight_loader(param,read('gate.weight'))
    moe.gate.e_score_correction_bias.copy_(read('gate.e_score_correction_bias'))
    assert moe.shared_experts.down_proj.reduce_results is False
    return records


def check_runner(moe, config, x, desired_ids, oracle_partial, *, trained_router=False):
    """Observe actual stock execution; compare outside the observed invocation.

    Observation wrappers only count and copy results; the original functions
    execute. The independent oracle uses the original stock all-reduce after
    the observed invocation. That extra comparator collective is excluded from
    the exactly-one assertion and from any performance window.
    """
    import torch
    import vllm.model_executor.layers.fused_moe.runner.moe_runner as runner_module
    from vllm.forward_context import set_forward_context
    from experiments.glm_packed_moe_control import compare
    runner = moe.experts
    layer = runner.routed_experts
    method = layer.quant_method
    apply = method.apply
    reduce = runner_module.tensor_model_parallel_all_reduce
    observed = {'apply':[],'shared':[],'reduce':[]}
    def apply_observer(layer,x,topk_weights,topk_ids,shared_experts,shared_experts_input):
        out = apply(layer,x,topk_weights,topk_ids,shared_experts,shared_experts_input)
        observed['apply'].append((out.clone(),topk_weights.clone(),topk_ids.clone()))
        return out
    def reduce_observer(value):
        observed['reduce'].append(value.clone())
        return reduce(value)
    hook = moe.shared_experts.register_forward_hook(
        lambda module,arguments,out:observed['shared'].append(out.clone()))
    method.apply = apply_observer
    runner_module.tensor_model_parallel_all_reduce = reduce_observer
    try:
        with set_forward_context(None,config):
            if trained_router:
                got = moe(x)
            else:
                # The real stock grouped router consumes these logits. The
                # bias is retained; the large separation fixes the chosen set.
                logits = torch.full((x.shape[0],layer.global_num_experts),-100.,
                                    device=x.device,dtype=torch.float32)
                logits.scatter_(1,desired_ids.long(),100.)
                got = runner(x,router_logits=logits)
        torch.cuda.synchronize()
    finally:
        hook.remove()
        method.apply = apply
        runner_module.tensor_model_parallel_all_reduce = reduce
    assert {key:len(value) for key,value in observed.items()} == {'apply':1,'shared':1,'reduce':1}
    partial,weights,ids = observed['apply'][0]
    if not trained_router:
        assert torch.equal(ids.sort(-1).values,desired_ids.sort(-1).values)
    # Every rank must use exactly the same global routing choices and weights.
    from vllm.distributed import get_tp_group
    all_ids = get_tp_group().all_gather(ids,dim=0)
    all_weights = get_tp_group().all_gather(weights,dim=0)
    assert torch.equal(all_ids[:x.shape[0]],all_ids[x.shape[0]:])
    assert torch.equal(all_weights[:x.shape[0]],all_weights[x.shape[0]:])
    expected_partial = oracle_partial(x,weights,ids)
    partial_error = compare(partial,expected_partial)
    assert partial_error['max_abs'] == 0 and partial_error['finite'],partial_error
    # Stock's BF16 combine and collective order are themselves part of the
    # comparator: adding after reducing each term need not round identically.
    expected_combined = observed['shared'][0] + expected_partial
    assert torch.equal(observed['reduce'][0],expected_combined)
    expected = reduce(expected_combined)
    total_error = compare(got,expected)
    assert total_error['max_abs'] == 0 and total_error['finite'],total_error
    assert bool(torch.count_nonzero(observed['shared'][0]))
    result = {'stock_runner_class':type(runner).__module__+'.'+type(runner).__qualname__,
        'trained_gate_executed':trained_router,'stock_shared_expert_executed':True,
        'observed_method_calls':1,'observed_shared_calls':1,'observed_final_all_reduce_calls':1,
        'rank_partial_vs_stock_tp2':partial_error,'combined_vs_stock_tp2':total_error,
        'routing_identical_across_ranks':True,'selected_global_experts':torch.unique(ids).numel(),
        'shared_partial_nonzero':True,'oracle_collective_outside_observed_invocation':True}
    del observed,got,expected,expected_combined,expected_partial,partial,weights,ids
    return result
