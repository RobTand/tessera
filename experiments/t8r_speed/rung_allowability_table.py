"""Harvest the existing geometry/quality harness into versioned D41 tables.

Run through PrismaBuild on CPU. Unfinished jobs and missing quality stay pending.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

from tessera import routed_fused as rf
from tessera.control import grid_for_name, unit_wire_bits
from tessera.rung_allowability import validate_index, validate_table

MS=(1,16,2048,4096)
SHAPES=(("routed","gate_up",1024,4096,0),("routed","down",4096,1024,2),("dense","o_proj",4096,4096,2),("dense","q_b",8192,1536,2))
FORMAT="TESSERA_E4M3_K1"


def roster():
    return [{"cell_id":f"{kind}:{name}:M{m}","kernel_kind":kind,"shape_id":name,"M":m} for kind,name,_r,_c,_mode in SHAPES for m in MS]


def resources(usage,head,cell):
    want=f"routed_fused_kernel<true, {head.get('mode',2)}, {'true' if head['kind']=='dense' else 'false'}, {'true' if cell.get('k_split',1)>1 else 'false'}, {head['r_lo']}, {'true' if head['n_hi'] else 'false'}, {cell['bm']}, false, false>"
    matches=[(name,res) for name,res in usage.get("kernels",{}).items() if want in name]
    return matches[0] if len(matches)==1 else (None,None)


def measurement(path,data,head,cell,key,build_id,cols,rows):
    meta=data['meta']
    f,r=cell.get('F',{}),cell.get('R',{})
    timer=f.get('timer') if f.get('timer')==r.get('timer') else None
    name,res=resources(meta.get('resource_usage',{}),head,cell) if 'bm' in cell else (None,None)
    mode=head.get('mode',2)
    geo=head.get('geometry',{})
    slot=geo.get('slot_words')
    requested=rf.launch_smem_bytes(mode,slot,mma8=True,bm=cell['bm']) if slot is not None and 'bm' in cell else None
    available=meta.get('shared_memory_available')
    bits=Fraction(unit_wire_bits(grid_for_name('E4M3'),head['q256'],rows,cols))*256/(rows*cols)
    error=f.get('error') or r.get('error')
    good=not error and cell.get('ms',0)>0 and timer and name and res and requested is not None and available is not None
    evidence={'geometry_file':str(path),'action_key':meta.get('pb_action'),'host':meta.get('host'),
              'comparison_id':meta.get('pb_action'),'paired_seed_contract':meta.get('paired_seed_contract'),
              'timing_statistic':meta.get('statistic'),'timer':timer,'F':f,'R':r,'profile':cell.get('profile'),
              'power':cell.get('power'),'loaded_library_sha256':meta.get('library_sha256'),'kernel_source_sha256':meta.get('kernel_sha'),
              'rows':rows,'columns':cols,'mode':mode,'routing':'balanced' if key['kernel_kind']=='routed' else 'none',
              'input_distribution':meta.get('activation_contract'),'epilogue':'SwiGLU clipped at 10' if mode==0 else ('route-weighted BF16 down' if key['kernel_kind']=='routed' else 'BF16 linear output')}
    if error: evidence['reason']=error
    elif not good: evidence['reason']='missing paired timer, actual compiler resource or launch geometry evidence'
    geometry={'bits_per_256_weight_tile':{'numerator':bits.numerator,'denominator':bits.denominator},
              'alignment':{k:v for k,v in geo.items() if k in ('lane_bits','lane_ends_on_word','half_bytes','half_copy','slot_words')},
              'shared_memory':{'requested_bytes':requested,'available_bytes':available,'fits':bool(requested is not None and available is not None and requested<=available)},
              'register_pressure':res,'decode_width':{'window_bits':rf.WINDOW_BITS,'value_bits':8,'run_widths':geo.get('rates'),'word_stages':geo.get('word_stages'),'superblock_rows':cell.get('bm'),'k_split':cell.get('k_split',1)},
              'raw':{'owner':'tessera.routed_fused.launch_smem_bytes and control.unit_wire_bits','packed_tile_words':head.get('tile_words'),'payload_bits_per_256_weights':{'numerator':head.get('tile_words',0)*16,'denominator':cols}}}
    return {**key,'measurement_status':'measured' if good else ('failed' if error else 'pending'),'kernel_time_us':cell['ms']*1000 if good else None,
            'kernel_path':name,'geometry':geometry,'evidence':evidence,'pass_times_us':[f['median_ms']*1000,r['median_ms']*1000] if good else [],'measurement_build_id':build_id}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--schema',required=True)
    ap.add_argument('--index-schema',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--version',required=True,type=int)
    args=ap.parse_args()
    root=Path(args.root)
    completed=[]
    for path in sorted(root.glob('gpu/*/bench_geometry_all.json')):
        try:data=json.loads(path.read_text())
        except (json.JSONDecodeError,OSError):continue
        if data.get('meta',{}).get('end_unix'): completed.append((path,data))
    if not completed: raise ValueError('No completed geometry quantum; no observed build to publish')
    meta=completed[0][1]['meta']
    # Identity derives from observed code/architecture/variant, not a refusal seal.
    signature={k:meta[k] for k in ('kernel_sha','architecture','library','activation_contract','image','torch')}
    build_id='e4m3mma-'+meta['architecture']+'-'+hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:16]
    build={'id':build_id,'source_commit':'f0ec9bb35dcf1c21925b1ac6c7b3364d2e99403e','library_variant':meta['library'],'architecture':meta['architecture'],'activation_contract':meta['activation_contract'],
           'metadata':{'observed_signature':signature,'loaded_libraries':sorted({d['meta']['library_sha256'] for _,d in completed}),'actual_source_snapshots':sorted({d['meta']['tessera_head'] for _,d in completed}),'flags':[]}}
    for path,_ in completed:
        for ninja in path.parent.glob('home/torch_extensions/*/build.ninja'):
            build['metadata']['flags'].append({'file':str(ninja),'cuda_cflags':[l for l in ninja.read_text().splitlines() if l.startswith('cuda_cflags =')]})
    required=roster()
    by_rung={q:{'rung':q,'measurement_status':'pending','supported':None,'anomaly_flags':[],'observations':[], 'excluded':False,'dominating_rung':None,'measurements':[],'quality':{},'dominance_evidence':[],'lineage':{}} for q in range(768,1153)}
    quality={}
    if (root/'quality.json').exists():
        try:quality=json.loads((root/'quality.json').read_text())
        except json.JSONDecodeError:pass
    # Prefer the quantum where the rung and its next neighbor share one F/R run.
    candidates={}
    for path,data in completed:
        if any(data['meta'].get(k)!=v for k,v in signature.items()):
            continue  # Different actual build belongs in a different table, not mixed numerics.
        qs=[int(v[1:]) for v in data['meta']['cases'].split(',')]
        for group in data['groups'].values():
            q=group.get('q256')
            if q not in by_rung or group.get('kind') not in ('routed','dense'):continue
            kind=group['kind']; name=group.get('shape') if kind=='dense' else ('gate_up' if group['mode']==0 else 'down')
            spec=next((s for s in SHAPES if s[:2]==(kind,name)),None)
            if not spec:continue
            for m in MS:
                key=next(k for k in required if k['kernel_kind']==kind and k['shape_id']==name and k['M']==m)
                cell=group.get('cells',{}).get(str(m)+(':balanced' if kind=='routed' else ''),{})
                if not cell:continue
                candidate=measurement(path,data,group,cell,key,build_id,spec[3],spec[2])
                preference=(int(q+1 in qs),int(candidate['measurement_status']=='measured'))
                ckey=(q,key['cell_id'])
                if ckey not in candidates or preference>candidates[ckey][0]:candidates[ckey]=(preference,candidate)
    for q,row in by_rung.items():
        row['measurements']=[candidates[(q,k['cell_id'])][1] for k in required if (q,k['cell_id']) in candidates]
        row['quality']=quality.get('rungs',{}).get(str(q),{'measurement_status':'pending'})
        row['anomaly_flags']=row['quality'].get('anomaly_flags',[])
        row['lineage']={'quality_file':str(root/'quality.json'),'geometry_files':sorted({m['evidence']['geometry_file'] for m in row['measurements']})}
        if len(row['measurements'])==len(required) and all(m['measurement_status']=='measured' for m in row['measurements']) and row['quality'].get('measurement_status')=='measured':
            row['measurement_status']='measured';row['supported']=True
        elif any(m['measurement_status']=='failed' for m in row['measurements']) or row['quality'].get('measurement_status')=='failed':
            row['measurement_status']='failed'; row['supported']=None
        if q in (880,912):row['observations'].append({'kind':'missing_census','issue':689,'url':'https://github.com/RobTand/tessera/issues/689','blocking':False,'exclusion_basis':False})
    for q,row in by_rung.items():
        high=by_rung.get(q+1)
        if row['measurement_status']!='measured' or not high or high['measurement_status']!='measured' or high['supported'] is not True or high['anomaly_flags']:continue
        proof=[]
        for low_m,high_m in zip(row['measurements'],high['measurements']):
            a,b=low_m['evidence'],high_m['evidence']
            paired=all(a.get(k) and a.get(k)==b.get(k) for k in ('comparison_id','paired_seed_contract','timing_statistic','timer'))
            if not paired or high_m['kernel_time_us']>low_m['kernel_time_us']:break
            proof.append({'cell_id':low_m['cell_id'],'lower_time_us':low_m['kernel_time_us'],'higher_time_us':high_m['kernel_time_us'],'comparison_id':a['comparison_id']})
        if len(proof)==len(required):row.update(excluded=True,dominating_rung=q+1,dominance_evidence=proof)
    baseline=by_rung[1024]
    for q,row in by_rung.items():
        if q>1024:
            ratios=[]
            for m in row['measurements']:
                base=next((x for x in baseline['measurements'] if x['cell_id']==m['cell_id'] and x['measurement_status']=='measured'),None)
                if base and m['measurement_status']=='measured':ratios.append({'cell_id':m['cell_id'],'ratio_to_1024':m['kernel_time_us']/base['kernel_time_us'],'receipt':m['evidence']['action_key'],'baseline_receipt':base['evidence']['action_key'],'paired':m['evidence']['comparison_id']==base['evidence']['comparison_id']})
            row['observations'].append({'kind':'beyond_1024_slow_lane','issue':690,'url':'https://github.com/RobTand/tessera/issues/690','blocking':False,'exclusion_basis':False,'ratios':ratios,'missing_baseline':not bool(ratios)})
    rows=list(by_rung.values())
    table={'schema':'fleet.rung_allowability.v1','table_version':args.version,'table_status':'complete' if all(r['measurement_status']!='pending' for r in rows) else 'partial','format':FORMAT,'kernel_build':build,'generated_at':datetime.now(timezone.utc).isoformat(),
           'scope':{'rung_min':768,'rung_max':1152,'grid_step_q256':1,'grid_owner':'prismaquant.tessera_formats.realisable_rungs(step_q256=1), lines 1145-1159','required_cells':required,'shapes':[{'shape_id':n,'kernel_kind':k,'rows':r,'columns':c,'mode':mode} for k,n,r,c,mode in SHAPES],'timing_statistic':meta['statistic'],'shape_owner':'bench_rates TP2 shapes; actual GLM config hidden4096, routed inter2048/2, experts288, topk8'},'rungs':rows,'evidence':{'summary':dict(Counter(r['measurement_status'] for r in rows)),'completed_quanta':len(completed),'quality_scope':'fixed actual expert 0 layer3 gate/up/down 32x256 sample; unweighted weight-space SSE, not served KL','exclusion_review_status':'pending independent review; no defaults promoted'}}
    validate_table(table)
    import jsonschema
    jsonschema.Draft202012Validator(json.loads(Path(args.schema).read_text())).validate(table)
    relative=f'{FORMAT}/{build_id}/v{args.version:04d}.json'
    index={'schema':'fleet.rung_allowability.index.v1','formats':{FORMAT:{'kernel_builds':{build_id:{'current_version':args.version,'versions':{str(args.version):{'path':relative,'table_schema':table['schema'],'table_status':table['table_status']}}}}}}}
    validate_index(index)
    jsonschema.Draft202012Validator(json.loads(Path(args.index_schema).read_text())).validate(index)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with (out/'table.json').open('x') as h:json.dump(table,h,indent=1,allow_nan=False)
    with (out/'index.json').open('x') as h:json.dump(index,h,indent=2,allow_nan=False)
    report={'status':'schema_and_semantic_validation_passed','table_path':relative,'table_status':table['table_status'],'kernel_build_id':build_id,'summary':table['evidence']['summary'],'excluded':[r['rung'] for r in rows if r['excluded']],'action_key':__import__('os').environ.get('PRISMABUILD_ACTION_KEY')}
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
