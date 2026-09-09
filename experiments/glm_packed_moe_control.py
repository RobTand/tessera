"""Explicit TP1/TP2 eager lifecycle control on unchanged stock vLLM.

Original q512 wires and saved inputs are common to the resident/packed arms.
An additional source-derived diagnostic wire fixture can bind distinct expert
identities; it is not the set of trained GLM experts. Neither arm runs a model
quality probe or promotes a serving cell.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback

from experiments.glm_native_construction import digest, install, plain, scheme_for, write
from experiments.glm_repeated_expert_control import error


def checked(row):
    path = Path(row['path'])
    if digest(path) != row['sha256']:
        raise ValueError(f'control identity mismatch: {path}')
    return path



def checkpoint_quant_config(binding, expected_quantization):
    """Reconstruct the ordinary registered plugin from byte-bound JSON only."""
    path = Path(binding['path'])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding['sha256']:
        raise ValueError(f'checkpoint config identity mismatch: {path}')
    checkpoint = json.loads(raw)
    quantization = checkpoint.get('quantization_config')
    if quantization != expected_quantization or 'research_selected_moe' not in quantization:
        raise ValueError('checkpoint quantization differs from the bound control fixture')
    from tessera.serving import register
    from tessera.serving.config import TesseraConfig
    from vllm.model_executor.layers.quantization import get_quantization_config
    register()
    registered = get_quantization_config(quantization['quant_method'])
    if registered is not TesseraConfig:
        raise ValueError('checkpoint dispatch did not resolve ordinary TesseraConfig')
    quant = registered.from_config(quantization)
    assert type(quant) is TesseraConfig
    assert quant._research_selected_moe.as_checkpoint() == quantization['research_selected_moe']
    return quant, {'checkpoint_config': binding,
        'config_class': type(quant).__module__ + '.' + type(quant).__qualname__,
        'registry_key': quantization['quant_method'],
        'research_selected_moe': quantization['research_selected_moe'],
        'quantization_sha256': hashlib.sha256(json.dumps(quantization,
            sort_keys=True,separators=(',',':')).encode()).hexdigest()}


def compare(got, expected):
    if got.numel() == expected.numel() == 0:
        assert got.shape == expected.shape and got.dtype == expected.dtype
        return {'finite':True,'max_abs':0.,'relative_l2':0.}
    return error(got,expected)


def memory():
    import torch
    torch.cuda.synchronize()
    return {'allocated_bytes':torch.cuda.memory_allocated(),
            'reserved_bytes':torch.cuda.memory_reserved(),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'epoch':time.time()}


def diagnostic_down_wires(template, experts):
    """Canonical scale-only diagnostic; original wire BODY/ALPHABET stay exact.

    The positive integer binade scale signatures are the same diagnostic rule
    as #417, now carried by bytes the actual loader parses. No encoder is run.
    No encoder fixture is claimed for this deliberately changed scale plane.
    """
    import torch
    from tessera.fused import parse_fused, pack_fused
    from tessera.unit_artifact import parse_unit_artifact, build_unit_artifact
    from tessera.container import parse
    from tessera.planes import PlaneKind
    member, = parse_fused(template)
    original = parse_unit_artifact(member.blob,device='cpu')
    before = parse(member.blob)
    rows = original.unit.scale_rows.numel()
    bit = torch.arange(rows) % (experts - 1).bit_length()
    for expert in range(experts):
        signature = (1 + ((expert >> bit) & 1)).to(torch.float16)
        unit = dataclasses.replace(original.unit,scale_rows=original.unit.scale_rows * signature)
        manifest, region, raw = build_unit_artifact(unit,
            unit_id=f'glm-diagnostic-expert-{expert}.down_proj', forests=original.forests,
            q256=original.manifest.branch.root_q256, code=original.code,
            superblock=original.manifest.geometry.superblock_columns,
            alignment_bytes=max(p.alignment_bytes for p in original.manifest.planes),
            container=original.manifest.branch.container,
            fixture_id=None, layout=original.manifest.layout)
        # Every non-scale payload is the original payload. Its descriptor hash
        # is a byte identity; unrelated metadata/provenance may change size.
        old = {p.kind:p.content_digest for p in before.manifest.planes}
        for plane in manifest.planes:
            if plane.kind != PlaneKind.DIAG_SV:
                assert plane.content_digest == old[plane.kind], plane.kind
        yield expert, pack_fused([(member.name,member.rows,raw)]), signature


def run(args, request, outer):
    import torch
    from safetensors.torch import load_file, save_file
    from tools import tessera_construction_census as census
    from tessera.serving.config import TesseraConfig
    from vllm.config import set_current_vllm_config
    from vllm.v1.worker.workspace import init_workspace_manager
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import make_fp8_moe_kernel, make_fp8_moe_quant_config

    baseline = json.loads(checked(outer['construction']).read_text())
    assert baseline['request_sha256'] == digest(args.request)
    control_path = checked(outer['control_request'])
    control = json.loads(control_path.read_text())
    producer = json.loads(checked(outer['producer_manifest']).read_text())
    assert producer['control_request_sha256'] == digest(control_path)
    target = baseline['owners'][0]['prefix']
    text = json.loads((Path(request['bounded_config'])/'config.json').read_text())['text_config']
    e,h,n = text['n_routed_experts'],text['hidden_size'],text['moe_intermediate_size']
    assert (e,h,n,text['num_experts_per_tok']) == (288,4096,2048,8)
    templates, stock, input_receipts = {},{},{}
    source_prefixes = set()
    for role,row in producer['projections'].items():
        root = Path(row['directory'])
        assert digest(root/'receipt.json') == row['receipt_sha256']
        receipt = json.loads((root/'receipt.json').read_text())
        assert receipt['status'] == 'encoded_source_projection' and receipt['q256'] == outer['q256']
        assert receipt['control_request_sha256'] == digest(control_path)
        assert digest(root/'projection.wire') == receipt['wire_sha256']
        assert digest(root/'independent-stock.safetensors') == receipt['stock_sha256']
        source_name = receipt['source_tensor']['name']
        assert source_name.endswith(f'.experts.0.{role}.weight')
        source_prefixes.add(source_name.rsplit('.experts.',1)[0])
        templates[role] = (root/'projection.wire').read_bytes()
        stock[role] = load_file(root/'independent-stock.safetensors',device='cpu')
        input_receipts[role] = {'path':str(root/'receipt.json'),'sha256':digest(root/'receipt.json')}

    assert len(source_prefixes) == 1
    source_prefix = source_prefixes.pop()
    diagnostics = None
    diagnostic_rows = []
    if outer['fixture'] == 'diagnostic':
        diagnostics = {}
        for expert,blob,signature in diagnostic_down_wires(templates['down_proj'],e):
            diagnostics[expert] = blob
            diagnostic_rows.append({'expert':expert,'wire_sha256':hashlib.sha256(blob).hexdigest(),
                                    'wire_bytes':len(blob)})
        write(args.out,'diagnostic-wire-roster.json',diagnostic_rows)
    scheme = scheme_for(text)
    for group in scheme['groups'].values():
        group['q256'] = outer['q256']
        group['wire_stride'] = max(len(templates[role]) for role,_ in group['roles'])
    if diagnostics is not None:
        scheme['groups']['w2']['wire_stride'] = max(map(len,diagnostics.values()))

    quant_dict = {'quant_method':'tessera','format':'tessera',
        'config_groups':{'glm_experts':{'format':'TESSERA','targets':[source_prefix+'.experts'],'scheme':scheme}},
        'ignore':sorted({p for p,_ in baseline['offered_modules'] if p != target})}
    tp_size = outer.get('tp_size',1)
    assert type(tp_size) is int and tp_size in (1,2) and (tp_size == 1 or outer['arm'] == 'packed')
    tp_rank = 0 if tp_size == 1 else outer['distributed']['rank']
    quant_cls = TesseraConfig
    config_source = outer.get('configuration_source', 'python_control')
    if config_source not in ('python_control', 'checkpoint_json'):
        raise ValueError('unknown packed control configuration_source')
    if config_source == 'checkpoint_json' and outer['arm'] != 'packed':
        raise ValueError('checkpoint_json control requires the packed arm')
    checkpoint_evidence = None
    if outer['arm'] == 'packed':
        from tessera.serving.moe_route import ResearchSelectedMoeConfig, build_tessera_moe_method
        research = ResearchSelectedMoeConfig(max_experts_per_chunk=outer['max_experts_per_chunk'],
            decode_backend=outer.get('decode_backend', 'torch'),
            expected_tensor_parallel_size=tp_size)
        if config_source == 'checkpoint_json':
            quant_dict['research_selected_moe'] = research.as_checkpoint()
        else:
            class ResearchConfig(TesseraConfig):
                def get_quant_method(self,layer,prefix):
                    declaration = self.target_scheme.get(prefix)
                    if declaration is not None and declaration.get('structure') == 'routed_moe':
                        return build_tessera_moe_method(declaration,prefix,self._mode,layer,research_selected=research)
                    return super().get_quant_method(layer,prefix)
            quant_cls = ResearchConfig
    if config_source == 'checkpoint_json':
        quant, checkpoint_evidence = checkpoint_quant_config(outer['checkpoint_config'], quant_dict)
        write(args.out, 'checkpoint-reconstruction.json', checkpoint_evidence)
    else:
        quant = quant_cls.from_config(quant_dict)
    if tp_size == 1:
        model,mapped,config = census.build_model(request['bounded_config'],'meta',512,quant_config=quant)
    else:
        from experiments.glm_packed_tp2_control import build_model
        model,mapped,config = build_model(request['bounded_config'],'meta',512,quant_config=quant,
                                          distributed=outer['distributed'])
    paired_request_sha256 = None
    if tp_size == 2:
        from experiments.glm_packed_tp2_control import verify_common_request
        paired_request_sha256 = verify_common_request(outer)
    assert target in mapped.target_scheme and config.model_config.enforce_eager
    if checkpoint_evidence is not None:
        assert type(mapped) is TesseraConfig and config.quant_config is mapped
        assert mapped._research_selected_moe.as_checkpoint() == checkpoint_evidence['research_selected_moe']
    owner_name = baseline['owners'][0]['name']
    moe_type = type(model.get_submodule(owner_name.rsplit('.experts.',1)[0]))
    meta_owner = model.get_submodule(owner_name)
    meta_parameters = {name:plain(value) for name,value in meta_owner.named_parameters(recurse=False)}
    registered = config.compilation_config.static_forward_context.pop(target)
    assert registered is model.get_submodule(owner_name.rsplit('.',1)[0])
    del registered,meta_owner,model
    gc.collect()

    from experiments.glm_packed_intake_observer import IntakeObservation

    with set_current_vllm_config(config,check_compile=False), torch.no_grad():
        init_workspace_manager(torch.device('cuda'))
        torch.cuda.reset_peak_memory_stats()
        stages = {'before_create':memory()}
        with IntakeObservation(args.out, outer.get('intake_observation')) as intake_observer:
            with census._set_default_torch_dtype()(torch.bfloat16),torch.device('cuda'):
                moe = moe_type(config.model_config.hf_text_config,config.parallel_config,mapped,
                               prefix=target.removesuffix('.experts'))
            layer,method = moe.experts.routed_experts,moe.experts.routed_experts.quant_method
            stages['after_create'] = memory()
            intake_observer.after_create(layer, method)
            initial_parameters = {name:plain(value) for name,value in layer.named_parameters(recurse=False)}
            if outer['arm'] == 'packed':
                assert set(initial_parameters) == {'e_score_correction_bias','w13_wire','w2_wire'}
                assert layer.tessera_mode == 'research_selected'
                assert method._research_phase == 'loading'
            def weights():
                for expert in range(e):
                    for role,template in templates.items():
                        blob = diagnostics[expert] if diagnostics is not None and role == 'down_proj' else template
                        yield f'{expert}.{role}.wire',torch.frombuffer(bytearray(blob),dtype=torch.uint8)
                        intake_observer.after_callback(method)
            loaded = sorted(layer.load_weights(weights()))
            stages['after_wire_load'] = memory()
            intake_observer.after_load(method, e * 3)
            write(args.out,'load-progress.json',{'loaded_names':loaded,'supplied_projections':e*3,
                'meta_parameters':meta_parameters,'initial_parameters':initial_parameters,'stages':stages})
            with torch.profiler.record_function('tessera_control_prepare_owner'):
                method.process_weights_after_loading(layer)
            stages['after_prepare'] = memory()
            parameters = dict(layer.named_parameters(recurse=False))
            if outer['arm'] == 'packed':
                assert set(parameters) == {'e_score_correction_bias'}
                assert method._research_phase == 'ready'
                assert method.moe_kernel is None and method.moe_quant_config is None
                resident_bytes = method.research_resident_bytes()
                intake_observer.after_finalize(method)
            else:
                resident_bytes = sum(p.numel()*p.element_size() for p in parameters.values())
            write(args.out,'prepared-owner.json',{'stages':stages,'owner_resident_bytes':resident_bytes,
                'parameters':{name:plain(p) for name,p in parameters.items()},
                'no_persistent_full_fp8':outer['arm']=='packed'})

        # Allocate the independent stock oracle only after measuring the load.
        # Stock TP2 slices each role of the full independent source; local
        # post-SwiGLU activation quantization makes TP1 FP8 a different oracle.
        local_n = n // tp_size
        lo,hi = tp_rank * local_n,(tp_rank + 1) * local_n
        first = torch.cat([stock[r]['weight'][lo:hi] for r in ('gate_proj','up_proj')]).cuda().unsqueeze(0).repeat(e,1,1)
        second = stock['down_proj']['weight'][:,lo:hi].contiguous().cuda().unsqueeze(0).repeat(e,1,1)
        s13 = torch.cat([stock[r]['weight_scale'].flatten()[lo:hi] for r in ('gate_proj','up_proj')]).cuda().view(1,2*local_n,1).repeat(e,1,1)
        s2 = stock['down_proj']['weight_scale'].cuda().view(1,h,1).repeat(e,1,1)
        if diagnostics is not None:
            bits = (torch.arange(h,device='cuda') % (e-1).bit_length()).view(1,h,1)
            signature = 1 + ((torch.arange(e,device='cuda').view(e,1,1) >> bits) & 1)
            s2 *= signature
        def oracle(scales1,scales2):
            qc = make_fp8_moe_quant_config(fp8_backend=method.fp8_backend,w1_scale=scales1,w2_scale=scales2,
                a1_scale=None,a2_scale=None,per_act_token_quant=True,per_out_ch_quant=True,block_shape=None,
                gemm1_alpha=layer.swiglu_alpha,gemm1_beta=layer.swiglu_beta,swiglu_limit=layer.swiglu_limit,layer=layer)
            return make_fp8_moe_kernel(moe_quant_config=qc,moe_config=method.moe,
                fp8_backend=method.fp8_backend,experts_cls=method.experts_cls,routing_tables=layer._expert_routing_tables())
        full_oracle = oracle(s13,s2)
        def apply_oracle(kernel,x,w13,w2,weights,ids,expert_map=None):
            return kernel.apply(x,w13,w2,weights,ids,activation=layer.activation,
                global_num_experts=e,expert_map=expert_map,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                shared_experts=None,shared_experts_input=None)

        dense_source_records = None
        if tp_size == 2:
            from experiments.glm_packed_tp2_control import load_shared_and_gate
            dense_source_records = load_shared_and_gate(moe,request['source_model'],
                source_prefix,index_sha256=outer['source_index_sha256'])
            write(args.out,'shared-and-gate-source.json',dense_source_records)
        saved_inputs = load_file(checked(outer['inputs']),device='cuda') if outer.get('inputs') else None
        generated_inputs = {}
        results = []
        for case in outer['cases']:
            name,tokens = case['name'],case['tokens']
            if saved_inputs is None:
                generator = torch.Generator(device='cuda').manual_seed(case['seed'])
                x = torch.randn(tokens,h,generator=generator,device='cuda',dtype=torch.bfloat16)*case['input_multiplier']
                ids = ((torch.arange(tokens*8,device='cuda')*37+19)%e).reshape(tokens,8).int()
                weights = torch.rand(tokens,8,generator=generator,device='cuda')
                weights *= 2.5 / weights.sum(-1,keepdim=True)
            else:
                x,ids,weights = (saved_inputs[f'{name}.{field}'] for field in ('x','ids','weights'))
            generated_inputs.update({f'{name}.{field}':value.cpu() for field,value in [('x',x),('ids',ids),('weights',weights)]})
            expected = (apply_oracle(full_oracle,x,first,second,weights,ids)
                        if tokens else x.new_empty((0,h)))
            got = method.apply(layer,x,weights,ids,None,None)
            parity = compare(got,expected)
            assert parity['finite'] and parity['max_abs'] == 0, (name,parity)
            del got
            row = {'case':case,'output_vs_independent_stock':parity}
            write(args.out,f'{name}-direct-parity.json',row)
            if tp_size == 2 and tokens:
                from experiments.glm_packed_tp2_control import check_runner
                stock_partial = lambda xx,ww,ii:apply_oracle(full_oracle,xx,first,second,ww,ii)
                row['stock_runner_internal_gate'] = check_runner(moe,config,x,ids,stock_partial,
                    diagnostic_path=args.out/f'{name}-runner-routing.json',entrypoint='runner')
                row['stock_runner_trained_gate'] = check_runner(moe,config,x,ids,stock_partial,
                    diagnostic_path=args.out/f'{name}-trained-gate-routing.json',entrypoint='glm')
            if outer['arm'] == 'packed':
                selected_ids = torch.unique(ids).flip(0)
                decoded = method._packed.decode(selected_ids,max_experts_per_chunk=outer['max_experts_per_chunk'],
                                                backend=outer.get('decode_backend', 'torch'))
                assert torch.equal(decoded.w13_weight.view(torch.uint8),first.index_select(0,selected_ids.long()).view(torch.uint8))
                assert torch.equal(decoded.w2_weight.view(torch.uint8),second.index_select(0,selected_ids.long()).view(torch.uint8))
                assert torch.equal(decoded.w13_weight_scale,s13.index_select(0,selected_ids.long()))
                assert torch.equal(decoded.w2_weight_scale,s2.index_select(0,selected_ids.long()))
                row['all_selected_tiles_and_scales_exact'] = True
                if diagnostics is not None and tokens:
                    expert_map = torch.full((e,),-1,dtype=torch.int32,device='cuda')
                    expert_map.scatter_(0,selected_ids.long(),torch.arange(selected_ids.numel(),device='cuda',dtype=torch.int32))
                    expert_map[selected_ids[:2].long()] = expert_map[selected_ids[:2].flip(0).long()]
                    wrong = apply_oracle(oracle(decoded.w13_weight_scale,decoded.w2_weight_scale),x,
                        decoded.w13_weight,decoded.w2_weight,weights,ids,expert_map)
                    wrong_error = compare(wrong,expected)
                    assert wrong_error['finite'] and wrong_error['max_abs'] > 0, wrong_error
                    row['deliberately_wrong_map'] = wrong_error
                    del wrong
                del decoded
            del expected
            if tokens and outer['measure']:
                for _ in range(2):
                    out = method.apply(layer,x,weights,ids,None,None)
                    del out
                memory_before = memory()
                torch.cuda.reset_peak_memory_stats()
                start = time.time()
                wall_start = time.perf_counter()
                samples = []
                while len(samples) < outer['max_timing_samples']:
                    a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record()
                    out = method.apply(layer,x,weights,ids,None,None)
                    b.record(); b.synchronize()
                    samples.append(a.elapsed_time(b))
                    del out
                    if time.perf_counter()-wall_start >= outer['min_timing_seconds']:
                        break
                end = time.time()
                memory_after = memory()
                row['measurement'] = {'start_epoch':start,'end_epoch':end,'samples_ms':samples,
                    'median_ms':statistics.median(samples),'invocations':len(samples),
                    'memory_before':memory_before,'memory_after':memory_after,
                    'incremental_peak_bytes':memory_after['peak_allocated_bytes']-memory_before['allocated_bytes']}
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True,profile_memory=True) as prof:
                    with torch.profiler.record_function(f'glm_{outer["arm"]}_{name}_whole_apply'):
                        out = method.apply(layer,x,weights,ids,None,None)
                    torch.cuda.synchronize()
                trace = args.out/f'{name}-trace.json'
                prof.export_chrome_trace(str(trace))
                row['profile'] = {'path':str(trace),'sha256':digest(trace),
                    'events':[{'key':item.key,'calls':item.count,'cpu_time_us':item.cpu_time_total,
                               'device_time_us':item.device_time_total,'self_device_memory_bytes':item.self_device_memory_usage}
                              for item in prof.key_averages()]}
                del out,prof
                gc.collect()
                row['after_profile_cleanup'] = memory()
            if outer['arm']=='packed':
                assert method.research_resident_bytes() == resident_bytes
                assert method.moe_kernel is None and method.moe_quant_config is None
            results.append(row)
            write(args.out,'case-progress.json',results)
        save_file(generated_inputs,args.out/'inputs.safetensors')
        return {'status':'packed_lifecycle_control_passed','arm':outer['arm'],'fixture':outer['fixture'],
            'decode_backend':outer.get('decode_backend', 'torch'),
            'configuration_source':config_source,'checkpoint_reconstruction':checkpoint_evidence,
            'scheme':scheme,'source_input_receipts':input_receipts,'owner_resident_bytes':resident_bytes,
            'load_stages':stages,'cases':results,'input_file_sha256':digest(args.out/'inputs.safetensors'),
            'backend':plain(method.fp8_backend),'actual_moe_class':type(moe).__module__+'.'+type(moe).__qualname__,
            'actual_owner_class':type(layer).__module__+'.'+type(layer).__qualname__,
            'trained_diverse_experts':False,'trained_router_executed':tp_size==2,'shared_experts_executed':tp_size==2,
            'shared_and_gate_source_records':dense_source_records,'tp_rank':tp_rank,
            'paired_request_sha256':paired_request_sha256,
            'model_quality_probe':False,'runtime_cell_promoted':False,'tp_size':tp_size,'ep_size':1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--packed-request',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage',choices=['control'],required=True)
    args = parser.parse_args()
    outer = json.loads(args.packed_request.read_text())
    assert outer['schema']=='tessera.glm_packed_lifecycle_request.v1'
    assert outer['arm'] in ('resident','packed') and outer['fixture'] in ('original','diagnostic')
    assert outer['cases'] and all(x['tokens']>=0 for x in outer['cases'])
    assert type(outer['max_timing_samples']) is int and outer['max_timing_samples']>0
    assert outer['min_timing_seconds']>0
    assert checked(outer['original_request']) == args.request
    request = json.loads(args.request.read_text())
    checked(outer['source_archive']); checked(request['core_manifest'])
    for row in (request['source_config'],*request['config_files'].values()):
        checked(row)
    installation_request = {**request,'source_archive':outer['source_archive'],'source_commit':outer['source_commit']}
    core,expected = install(installation_request,args.out)
    from _pb_native_moe_measure.per_job_install import files
    from experiments.original_wire_generation import observed
    record = {'schema':'tessera.glm_packed_lifecycle_control.v1',
              'packed_request_sha256':digest(args.packed_request),'installed_source_commit':outer['source_commit']}
    rc = 0
    try:
        record.update(run(args,request,outer))
    except Exception as exc:
        rc = 1
        record.update(status='failed',reason=str(exc),traceback=traceback.format_exc())
    finally:
        empty = type('Empty',(),{'named_modules':lambda self:[]})()
        identity = observed(empty)['package_identity']
        assert identity['loaded_module_origins_verified']
        write(args.out,'package-identity.json',identity)
        assert files(core)==expected,'Stock vLLM core changed'
        record.update(stock_core_unchanged=True,package_identity_sha256=digest(args.out/'package-identity.json'))
        write(args.out,'receipt.json',record)
    return rc


if __name__=='__main__':
    raise SystemExit(main())
