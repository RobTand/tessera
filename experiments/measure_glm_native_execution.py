"""Original-reference whole GLM operator acquisition, run as an admitted child.

This producer consumes immutable PQ artifacts; it imports no PrismaQuant code.
Raw panels name actual execution, never an unmeasured joint distortion table.
"""
import argparse,copy,hashlib,io,json,os,stat
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def dump(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def freeze(inputs,prepared,source_sha256):
    from experiments import bench_native_moe_operator as moe
    dense=moe.dense;operator=prepared['operator']
    if inputs['schema']!='prismaquant.native_moe_inputs.v1':raise ValueError('native input schema differs')
    for key in ('shape','routing','profile_role_order','routing_capture_sha256','serving_config_sha256'):
        if operator[key]!=inputs[key]:raise ValueError('prepared '+key+' differs from independent inputs')
    if prepared['runtime']['image']!=inputs['runtime_image']:raise ValueError('prepared runtime image differs')
    members=inputs['members'];expected=[]
    for member in members:
        value={key:member[key] for key in ('unit','expert','role','format','shape','source_weight','rendered_weight')}
        value.update(wire_sha256=member['wire']['blob_sha256'],wire_record_sha256=dense.identity_sha256(member['wire']['record']))
        if member['activation'].get('input_global_scale') is not None:value['input_global_scale']=member['activation']['input_global_scale']
        expected.append(value)
    if operator['members']!=expected:raise ValueError('prepared complete member identities differ')
    route=operator['declared_route']
    if not route['decoder'].startswith('native_'):raise ValueError('raw acquisition requires actual compact native route')
    panel={'schema':moe.RAW_PANEL_SCHEMA,'unit':inputs['unit'],'format':inputs['format'],
        'shape':inputs['shape'],'members':members,'profile_role_order':inputs['profile_role_order'],
        'routing':inputs['routing'],'routing_capture_sha256':inputs['routing_capture_sha256'],
        'source_sha256':source_sha256,'calibration_sha256':inputs['calibration']['calibration_sha256'],
        'source_execution':inputs['routing_capture']['source_execution'],
        'source_execution_qualification_sha256':None,'probe_scope':None,
        'runtime_binding':{'member_formats':{m['unit']:m['format'] for m in members},
            'member_execution_identity_sha256':{m['unit']:dense.identity_sha256({'qname':m['unit'],
                **{key:m[key] for key in ('format','source_weight','rendered_weight','activation')}}) for m in members},
            'member_shapes':{m['unit']:moe._member_shape(inputs['shape'],m['role']) for m in members},
            'operator_route':json.dumps(route,sort_keys=True,separators=(',',':'),allow_nan=False)},
        'execution':inputs['execution'],'runtime':prepared['runtime'],
        'native_tensors_sha256':dense.identity_sha256(operator['native_tensors']),
        'scheme_sha256':operator['scheme_sha256'],'config_sha256':operator['config_sha256'],
        'serving_config_sha256':inputs['serving_config_sha256'],'workspace':prepared['workspace'],
        'workspace_sha256':dense.identity_sha256(prepared['workspace']),'numerics':inputs['numerics'],
        'phases':{phase:{**inputs['phases'][phase],'expected_route':route} for phase in moe.PHASES}}
    if 'reference_served_quantizer' in inputs:panel['reference_served_quantizer']=inputs['reference_served_quantizer']
    if 'source_acquisition' in inputs:panel['source_acquisition']=copy.deepcopy(inputs['source_acquisition'])
    return moe.validate_panel(copy.deepcopy(panel))


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs-root',type=Path,required=True)
    p.add_argument('--origin-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resource-library',type=Path,required=True);p.add_argument('--device-bytes',type=int,required=True);args=p.parse_args()
    root=args.inputs_root
    if sha(root/'origin.json')!=args.origin_sha256:raise ValueError('independent preparation changed')
    origin=json.loads((root/'origin.json').read_text())
    for name,digest in origin['artifacts'].items():
        if sha(root/name)!=digest:raise ValueError('independent artifact changed: '+name)
    args.out.mkdir(parents=True,exist_ok=False)
    from experiments.native_operator_resources import NativeMemoryCollector
    collector=NativeMemoryCollector(args.resource_library);finished=False
    from experiments import bench_native_moe_operator as moe
    import torch
    if args.device_bytes != 48<<30:raise ValueError('native pilot device envelope differs')
    total=int(torch.cuda.get_device_properties(0).total_memory)
    if args.device_bytes>=total:raise ValueError('native device envelope does not bound this device')
    fraction=args.device_bytes/total
    torch.cuda.set_per_process_memory_fraction(fraction,0)
    dump(args.out/'device-envelope.json',{'enforced':True,'device_bytes':args.device_bytes,
        'device_total_bytes':total,'fraction':fraction,'scope':'Torch caching allocator; native/CUDA context allocations additional'})
    from safetensors.torch import load_file
    inputs=json.loads((root/'inputs.json').read_text());refs=json.loads((root/'weight-references.json').read_text())
    ref_by_name={r['unit']:r for r in refs}
    from experiments.native_local_inputs import LocalNativeInputs
    local_inputs=LocalNativeInputs(origin['local_bundle'],os.environ.get('PRISMABUILD_ACTION_KEY'))
    read_local=local_inputs.read
    serving=origin['local_serving_config']
    if sha(serving['path'])!=serving['sha256']:raise ValueError('serving configuration changed')
    config,serving_record=moe.resolve_serving_config(serving['path'],inputs['runtime_image'],tensor_parallel=1)
    distributed=moe.owner_distributed(inputs['shape'],None)
    # A BF16 owner is priced on its production builder, the served owner (#613).
    selected=moe.owner_research_selected(inputs['shape'],moe.owner_wire(inputs['shape']),None)
    tracepath=args.out/'memory.json'
    try:
        with moe.native_runtime_context(config,distributed=distributed):
            tensors=load_file(str(root/'phase-tensors.safetensors'),device='cuda')
            phases={phase:{key:tensors[f'{phase}.{key}'] for key in moe.TENSOR_KEYS} for phase in moe.PHASES}
            member_inputs=[]
            for ref in refs:
                raw=read_local(ref['render'],ref['render_file_sha256'],64<<20)
                if hashlib.sha256(raw).hexdigest()!=ref['render_file_sha256']:raise ValueError('original PWC render changed')
                value=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
                if not isinstance(value,torch.Tensor):raise ValueError('original PWC payload is not a tensor')
                tensors['rendered_weight/'+ref['unit']]=value.to('cuda')
                member_inputs.append({key:ref[key] for key in ('unit','expert','role','format','record')})
                member_inputs[-1]['blob']=read_local(ref['wire'],ref['record']['blob_sha256'],64<<20)
                if 'input_global_scale' in ref:member_inputs[-1]['input_global_scale']=ref['input_global_scale']
            def source_reader(unit):
                ref=ref_by_name[unit]
                bound=ref['source_local']
                raw=read_local(bound['path'],bound['sha256'],64<<20)
                return torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
            weights={key:value for key,value in tensors.items() if key.startswith('rendered_weight/')}
            prepared=moe.prepare_native_moe_operator(member_inputs,weights,phases,
                unit=inputs['unit'],shape=inputs['shape'],routing=inputs['routing'],
                runtime_image=inputs['runtime_image'],execution=inputs['execution'],
                profile_role_order=inputs['profile_role_order'],serving_config=serving_record,
                routing_capture_sha256=inputs['routing_capture_sha256'],
                phase_transport={phase:inputs['phases'][phase]['transport'] for phase in moe.PHASES},
                routing_bias=tensors['routing_bias'],warmup_iterations=2,selected=selected,
                distributed=distributed,source_reader=source_reader)
            moe.release_verification_tensors(tensors,inputs['members'])
            weights.clear()
            prepared['runtime']['resource_collector']={'library_sha256':collector.library_sha256,
                'analysis_source_sha256':sha(Path(moe.__file__).with_name('native_operator_resources.py'))}
            preflight={'schema':'tessera.native_moe_preflight.v1','status':'untimed_preparation',
                'operator':prepared['operator'],'runtime':prepared['runtime'],
                'runtime_sha256':moe.dense.identity_sha256(prepared['runtime']),
                'native_tensors_sha256':moe.dense.identity_sha256(prepared['operator']['native_tensors']),
                'scheme_sha256':prepared['operator']['scheme_sha256'],'workspace':prepared['workspace'],
                'workspace_sha256':moe.dense.identity_sha256(prepared['workspace'])}
            panel=freeze(inputs,prepared,origin['source_sha256'])
            dump(args.out/'preflight.json',preflight);dump(args.out/'execution-panel.json',panel)
            receipt=moe.measure_prepared_operator(prepared,panel,phases,warmup_iterations=8,iterations=32,
                resource_collector=collector)
            trace=collector.finish(tracepath);finished=True;moe.attach_resource_trace(receipt,trace)
            if any(receipt['phases'][phase]['qdq_numerics']['max_abs_error']!=0 for phase in moe.PHASES):
                receipt['status']='reference_qdq_refused'
            if receipt['status']=='resources_observed' and receipt['resources']['status']=='complete_operator_bound':
                receipt['resources'].update(moe.per_rank_resource_identity(receipt,distributed))
                moe.time_after_resource_collection(prepared,panel,phases,receipt,collector=collector,
                    warmup_iterations=8,iterations=32)
            dump(args.out/'execution-receipt.json',receipt)
            print(json.dumps({'status':receipt['status'],'resources':receipt['resources']['status'],
                'receipt_path':str(args.out/'execution-receipt.json'),'receipt_sha256':sha(args.out/'execution-receipt.json'),
                'memory_sha256':sha(tracepath),'panel_sha256':sha(args.out/'execution-panel.json')}),flush=True)
            return 0 if receipt['status']=='timing_admissible' else 2
    finally:
        try:
            if not finished:collector.finish(tracepath)
        finally:local_inputs.close()


if __name__=='__main__':raise SystemExit(main())
