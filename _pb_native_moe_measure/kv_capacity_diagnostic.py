"""CPU capacity projection through exact stock helpers; no engine admission."""
import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, KVCacheGroupSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.core.kv_cache_utils import (get_kv_cache_config_from_groups,
    get_max_concurrency_for_kv_cache_config, _pool_bytes_per_block,
    _max_memory_usage_bytes_from_groups)
p=argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
a=p.parse_args()
r=Path('/mnt/shared/tessera-native376-resource/full-engine-startup-r3')
observations_path=r/'capture/worker-108/worker-observations.json'
observed=json.loads(observations_path.read_text())['kv_configuration']
config_path=Path('/mnt/shared/tessera-native376-resource/configs/lfm25_first_model_clean_20260907.json')
config=json.loads(config_path.read_text())
model_path=Path('/mnt/shared/models/LFM2.5-8B-A1B-BF16/config.json')
model=json.loads(model_path.read_text())
assert model['hidden_size']==2048 and model['num_attention_heads']==32 and model['num_key_value_heads']==8
assert model['conv_L_cache']==3 and model['dtype']=='bfloat16'
assert len(observed['tensors'])==4 and all(len(t['layers'])==6 for t in observed['tensors'])
assert all(t['block_stride']==32768 and t['offset']==0 for t in observed['tensors'])
# The actual outer strides and model K/V geometry determine 16 attention tokens
# per 32,768-byte page. In align mode recurrent block_size does not enter the
# maximum resident-page calculation; the actual resolved value must be captured
# before admission. 16 below is only a capacity-equivalent projection.
block_size=observed['tensors'][-1]['block_stride']//(2*model['num_key_value_heads']*(model['hidden_size']//model['num_attention_heads'])*2)
assert block_size==16
attention=FullAttentionSpec(block_size=block_size,num_kv_heads=8,head_size=64,dtype=torch.bfloat16)
mamba=MambaSpec(block_size=block_size,shapes=((2048,2),),dtypes=(torch.bfloat16,),page_size_padded=32768,mamba_cache_mode='align',num_speculative_blocks=0,num_prefill_checkpoint_blocks=0)
groups=[]
for t in observed['tensors']:
 recurrent=all(name.endswith('.short_conv') for name in t['layers'])
 assert recurrent or all(name.endswith('.self_attn.attn') for name in t['layers'])
 groups.append(KVCacheGroupSpec(t['layers'],mamba if recurrent else attention))
assert sum(isinstance(g.kv_cache_spec,MambaSpec) for g in groups)==3
ns=SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096),parallel_config=SimpleNamespace(decode_context_parallel_size=1),cache_config=SimpleNamespace(mamba_cache_mode='align',num_gpu_blocks_override=None,prefix_cache_retention_interval=None,get_resolved_kv_cache_layout=lambda:KVCacheLayout.LBHNC))
bytes_per_block=_pool_bytes_per_block(groups)
per_request_bytes=_max_memory_usage_bytes_from_groups(ns,groups)
per_request_blocks=per_request_bytes//bytes_per_block
assert per_request_blocks==262 and bytes_per_block==196608
pool_blocks=config['engine_args']['max_num_seqs']*per_request_blocks+1
pool_bytes=pool_blocks*bytes_per_block
old=get_kv_cache_config_from_groups(ns,groups,observed['num_blocks']*bytes_per_block)
old_tensors=[asdict(t) for t in old.kv_cache_tensors]
assert old_tensors==observed['tensors'],'Capacity projection does not reproduce actual r3 outer placements'
new=get_kv_cache_config_from_groups(ns,groups,pool_bytes)
ns.cache_config.num_gpu_blocks_override=pool_blocks
override=get_kv_cache_config_from_groups(ns,groups,observed['num_blocks']*bytes_per_block)
assert asdict(new)==asdict(override)
assert new.num_blocks-1==8*per_request_blocks
result={'schema':'tessera.stock_kv_capacity_projection.v1','status':'proposed_explicit_capacity_not_engine_admission','inputs':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (observations_path,config_path,model_path)},'actual_r3_descriptors_reproduced':True,'workload':config['engine_args'],'projection':{'attention_block_size':block_size,'attention_page_bytes':attention.page_size_bytes,'recurrent_content_bytes':mamba.state_content_size_bytes,'recurrent_padded_page_bytes':mamba.page_size_bytes,'group_layer_counts':[len(g.layer_names) for g in groups],'resident_pages_per_request_by_group':[g.kv_cache_spec.max_memory_usage_bytes(ns)//g.kv_cache_spec.page_size_bytes for g in groups],'pool_bytes_per_block':bytes_per_block,'blocks_per_request':per_request_blocks,'max_simultaneous_requests':8,'null_blocks':1,'proposed_num_gpu_blocks_override':pool_blocks,'proposed_kv_cache_memory_bytes':pool_bytes},'stock_capacity_helper':{'r3_max_concurrency':get_max_concurrency_for_kv_cache_config(ns,old),'proposed_reported_max_concurrency_including_null':get_max_concurrency_for_kv_cache_config(ns,new),'usable_max_concurrency_after_null':(new.num_blocks-1)/per_request_blocks},'proposed_placements':[asdict(t) for t in new.kv_cache_tensors],'scope_limits':['Capacity projection reconstructs actual r3 outer placement descriptors and model geometry; no GPU engine or concurrent workload was run.','Actual resolved recurrent block size, group spec fields, align/speculative/checkpoint settings and unique physical storage must be captured and asserted on next engine run.','GPU memory and prefix-cache hit performance are not qualified; replacing automatic KV capacity changes configuration identity.'],'stock_helper_files':{str(Path(inspect.getfile(f))):hashlib.sha256(Path(inspect.getfile(f)).read_bytes()).hexdigest() for f in (get_kv_cache_config_from_groups,FullAttentionSpec,MambaSpec)}}
a.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'output':str(a.out),'sha256':hashlib.sha256(a.out.read_bytes()).hexdigest(),'projection':result['projection']}),flush=True)
