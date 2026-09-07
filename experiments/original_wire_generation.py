"""Complete-answer proof for the original 96-wire checkpoint on the stock engine."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT: Path
CHECKPOINT=Path('/mnt/shared/tessera-clean-runtime-20260907/original-wire-layer2/checkpoint')
CONFIG=Path('/mnt/shared/tessera-native376-resource/configs/lfm25_first_model_fixed_kv_20260907.json')
CONFIG_SHA='f5064609d62a3e61ef1d9bb87b2b62ea10b31b71759db3dce666543d7585233e'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(name,value):
    path=ROOT/name;path.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'artifact':str(path),'sha256':digest(path),'bytes':path.stat().st_size}),flush=True)

def observed(model):
    from tessera.serving.telemetry import read_route, route_trace_snapshot
    routes={}; methods={}
    for name,layer in model.named_modules():
        if getattr(layer,'tessera_structure',None)=='routed_moe':
            method=layer.quant_method
            methods[name]={'class':type(method).__module__+'.'+type(method).__qualname__,
                'backend':layer.tessera_backend,'structure':layer.tessera_structure,
                'family':layer.tessera_family,'mode':layer.tessera_mode,
                'weight_dtype':str(layer.w13_weight.dtype),'weight_shape':list(layer.w13_weight.shape),
                'scale_dtype':str(layer.w13_weight_scale.dtype),'scale_shape':list(layer.w13_weight_scale.shape),
                'kernel_class':type(method.moe_kernel).__module__+'.'+type(method.moe_kernel).__qualname__}
            routes[name]=read_route(layer)
    import hashlib
    import sys
    from pathlib import Path
    import tessera
    import tessera.cached_unit as cached_unit
    package_root=Path(tessera.__file__).resolve().parent
    source_files={}; combined=hashlib.sha256()
    for path in sorted(p for p in package_root.rglob('*') if p.suffix in {'.py','.cu','.cuh','.cpp','.h'}):
        relative=path.relative_to(package_root).as_posix(); raw=path.read_bytes()
        source_files[relative]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
        combined.update(relative.encode()+b'\0');combined.update(raw);combined.update(b'\0')
    cached_unit.encoder_source_sha256.cache_clear()
    fresh=cached_unit.encoder_source_sha256()
    modules={}
    for name,module in sorted(sys.modules.items()):
        if name!='tessera' and not name.startswith('tessera.'):
            continue
        row={'file':getattr(module,'__file__',None),'spec_origin':getattr(getattr(module,'__spec__',None),'origin',None)}
        origins={}
        for field in ('file','spec_origin'):
            origin=row[field]; check={'canonical_source':False}
            if isinstance(origin,str):
                path=Path(origin).resolve()
                check['resolved_path']=str(path)
                try:
                    relative=path.relative_to(package_root).as_posix()
                except ValueError:
                    relative=None
                check['package_relative_path']=relative
                if relative in source_files:
                    raw=path.read_bytes(); actual={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
                    check.update(actual,canonical_source=actual==source_files[relative])
            origins[field]=check
        row['origin_checks']=origins
        row['canonical_source']=all(check['canonical_source'] for check in origins.values()) and origins['file'].get('resolved_path')==origins['spec_origin'].get('resolved_path')
        modules[name]=row
    identity={'tessera_file':tessera.__file__,'cached_unit_file':cached_unit.__file__,'sys_path':list(sys.path),
              'loaded_tessera_modules':modules,'package_source_files':source_files,
              'loaded_module_origins_verified':bool(modules) and all(row['canonical_source'] for row in modules.values()),
              'native_or_foreign_module_origin_exceptions':[],
              'fresh_encoder_source_sha256':fresh,'independently_recomputed_source_sha256':combined.hexdigest()}
    return {'methods':methods,'routes':routes,'trace':route_trace_snapshot(),'package_identity':identity}

def main():
    global ROOT
    ROOT=Path(sys.argv[1])
    started=time.time()
    assert digest(CONFIG)==CONFIG_SHA
    config=json.loads(CONFIG.read_text()); (ROOT/'serving-config.json').write_bytes(CONFIG.read_bytes())
    export_proof_path=CHECKPOINT.parent/'export-proof.json'; export_proof=json.loads(export_proof_path.read_text())
    assert export_proof['status']=='passed' and len(export_proof['wires'])==96
    for name,value in export_proof['checkpoint_files'].items():
        assert digest(CHECKPOINT/name)==value['sha256'],name
    from vllm import LLM, SamplingParams
    from tessera.cached_unit import encoder_source_sha256
    package_proof_path=CHECKPOINT.parent/'controls/package-identity-proof.json'
    assert digest(package_proof_path)=='26ab4829797e81b96ec8f6e4bf4257757c25905324df5d1f6baba687e692a4f7'
    package_proof=json.loads(package_proof_path.read_text())
    assert encoder_source_sha256()==package_proof['installed_package_source_sha256']
    assert os.environ['TESSERA_SERVE_MODE']=='resident'
    helper=CHECKPOINT.parent/'controls/full_engine_kv.py'
    assert digest(helper)=='3320b77dd392e71bd07c967a3e2aa2e85707f64b4992ec185256936ec0986651'
    from full_engine_kv import inspect_worker_kv
    llm=LLM(model=str(CHECKPOINT),seed=0,**config['engine_args'])
    try:
        before=llm.apply_model(observed)[0]
        write('loaded-runtime.json',before)
        kv_workers=llm.collective_rpc(inspect_worker_kv,timeout=60,args=(config['capacity_assertions'],))
        write('actual-kv-capacity.json',{'helper_sha256':digest(helper),'workers':kv_workers})
        assert len(kv_workers)==1 and kv_workers[0]['capacity_assertions']['passed'], 'Actual KV capacity differs from selected configuration'
        messages=[{'role':'user','content':'Return exactly the word blue.'}]
        outputs=llm.chat(messages,SamplingParams(temperature=0.0,top_p=1.0,seed=0,max_tokens=512),use_tqdm=False)
        after=llm.apply_model(observed)[0]
        write('served-runtime.json',after)
        def counts(trace):
            return {json.dumps({k:v for k,v in entry.items() if k not in ('launches','modules')},sort_keys=True):entry['launches'] for entry in trace['entries']}
        old=counts(before['trace']); new=counts(after['trace'])
        delta=[dict(json.loads(key),launches=count-old.get(key,0)) for key,count in new.items() if count>old.get(key,0)]
        assert len(outputs)==1 and len(outputs[0].outputs)==1
        answer=outputs[0].outputs[0]
        final=answer.text.strip();closed=True
        if '<think>' in final:
            closed=final.count('<think>')==1 and final.count('</think>')==1
            final=final.rsplit('</think>',1)[-1].strip() if closed else ''
        expected_source=package_proof['installed_package_source_sha256']
        checks={'loaded_module_origins_match_canonical_sources':all(record['package_identity']['loaded_module_origins_verified'] for record in (before,after)),
            'post_load_package_identity':all(record['package_identity']['fresh_encoder_source_sha256']==record['package_identity']['independently_recomputed_source_sha256']==expected_source and record['package_identity']['package_source_files']==package_proof['expected_installed_source_files'] for record in (before,after)),
            'complete_stop':answer.finish_reason=='stop','exact_final_blue':final=='blue','closed_reasoning':closed,
            'actual_kv_capacity':len(kv_workers)==1 and kv_workers[0]['capacity_assertions']['passed'],
            'one_native_routed_stack':len(after['methods'])==1,
            'native_plugin_method':all(m['class'].startswith('tessera.serving.moe_route.') for m in after['methods'].values()),
            'actual_post_request_native_launches':bool(delta) and all(e['symbol'].startswith('vllm.fused_moe.modular_kernel:') and e['kind']=='moe' for e in delta),
            'served_route_state':all(r and r['state']=='served' for r in after['routes'].values())}
        # All official core files, including generated/bundled files, after generation.
        sys.path.insert(0,'/mnt/shared/tessera-clean-runtime-20260907/control-854e672')
        from per_job_install import files
        baseline=Path('/mnt/shared/tessera-clean-runtime-20260907/official-primary/runtime-inventory.json')
        actual=files(Path(importlib.util.find_spec('vllm').origin).parent)
        checks['stock_core_unchanged']=actual==json.loads(baseline.read_text())['files']
        write('core-after-generation.json',{'status':'passed' if checks['stock_core_unchanged'] else 'failed','files':actual,'baseline_sha256':digest(baseline)})
        record={'schema':'tessera.original_wire_complete_generation.v1','status':'passed' if all(checks.values()) else 'failed','checks':checks,
            'request':{'messages':messages,'max_tokens':512,'temperature':0.0,'top_p':1.0,'seed':0},
            'response':{'text':answer.text,'finish_reason':answer.finish_reason,'stop_reason':answer.stop_reason,'token_ids':answer.token_ids,'prompt_token_ids':outputs[0].prompt_token_ids},
            'final_answer':final,'post_request_route_delta':delta,'export_proof_sha256':digest(export_proof_path),'serving_config_sha256':CONFIG_SHA,
            'plugin_receipt_sha256':digest(ROOT/'plugin-install/per-job-runtime.json'),'actual_kv_capacity_sha256':digest(ROOT/'actual-kv-capacity.json'),'package_identity_proof_sha256':digest(package_proof_path),'started_epoch':started,'finished_epoch':time.time(),
            'scope':'One complete deterministic answer from the original 96-wire layer-2 checkpoint with all other tensors at source precision. No statistical quality, performance, eight-request concurrency, release, or serving-cell promotion.'}
        write('generation-proof.json',record)
        assert all(checks.values()),checks
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=30)

if __name__=='__main__':main()
