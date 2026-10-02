"""Closed-world admission and raw-output binding controls; CPU-only evidence."""
import ast
import copy
import hashlib
import importlib.util
import json
from dataclasses import fields, MISSING
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'experiments/t8r_speed/paired_k32_qualification.py'
spec=importlib.util.spec_from_file_location('paired_k32_qualification',PATH)
q=importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def options():
    return SimpleNamespace(paired_k32_numerics=True,artifact=q.REPLAY_ARTIFACT,groups=q.GROUP,
        ms='1,512,2048',no_graph=True,ncu=False,power_s=0,warmup=0,iters=0,
        input_manifest='immutable-inputs.json',routing=None,single_routing_file=None,
        profile_native_file='/held/native.so',paired_k32_source_sha256='a'*64)


def test_exact_scope_admits():
    q.require_options(options(),stubbed=False)


def test_fixed_timing_scope_requires_exact_event_and_power_counts():
    args=options();args.paired_k32_timing=True;args.warmup=10;args.iters=30;args.power_s=3.0
    q.require_options(args,stubbed=False)
    args.iters=31
    with pytest.raises(ValueError):q.require_options(args,stubbed=False)


@pytest.mark.parametrize('key,value',[
    ('artifact','/arbitrary/model'),('groups','experts.R1024.L11'),('ms','1,2048'),
    ('no_graph',False),('ncu',True),('power_s',30),('warmup',1),('iters',1),
    ('input_manifest',None),('routing','/new/routes'),('single_routing_file','/history'),
    ('profile_native_file',None),('paired_k32_source_sha256',None),
    ('paired_k32_source_sha256','z'*64),
])
def test_closed_scope_refuses_overrides(key,value):
    args=options();setattr(args,key,value)
    with pytest.raises(ValueError,match='exact A8SE'):
        q.require_options(args,stubbed=False)


def test_stubbed_vllm_refuses():
    with pytest.raises(ValueError,match='stubbed vLLM'):
        q.require_options(options(),stubbed=True)


def report(tmp_path,build):
    cells={}
    for key in ('1','512','2048'):
        outputs={}
        for role in ('gate_up','down_routes','out'):
            path=tmp_path/(key+'-'+role)
            path.parent.mkdir(parents=True,exist_ok=True)
            raw=b'actual raw words'+key.encode()+role.encode()
            path.write_bytes(raw)
            outputs[role]={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),
                'bytes':len(raw),'shape':[1,len(raw)//2],'dtype':'torch.bfloat16'}
        cells[key]={'outputs':outputs,'input_hashes':{'x':'owned'},'native_paired_build':build,
            'repeat_bits_equal':True,'independent_token_sum_equal':True,
            'independent_reference':{'bounded_numeric_pass':True},
            'geometry':{mode:{'paired':bool(build and key!='1'),'bm':128} for mode in ('0','2')}}
    synthetic={}
    for K in ('128','192'):
        control=copy.deepcopy(cells['512']);control['diagnostic_grid']=1
        control['geometry']={mode:{'paired':bool(build and (mode=='0' or K=='192')),'bm':128} for mode in ('0','2')}
        synthetic[K]=control
    return {'meta':{'paired_k32':{'input_manifest_sha256':'c'*64},
        'direct_input_bindings':{'transport':'direct-vllm-held-original-fds','manifest_sha256':'c'*64,
            'before':{'owned':'fingerprint'},'after':{'owned':'fingerprint'}},
        'native_code_artifact':{k:'d'*64 for k in ['before_load_sha256','after_load_sha256','after_profile_sha256']}},
        'results':[{'ok':True,'cells':cells,'synthetic_controls':synthetic}]}


def test_comparator_reads_exact_words(tmp_path):
    a=report(tmp_path/'baseline',False);b=report(tmp_path/'candidate',True)
    result=q.compare_reports(a,b)
    assert result['exact_words_equal'] and len(result['compared'])==9


@pytest.mark.parametrize('fault', ['source','native','population','input','output','bytes','build','reference','dispatch','ownership','ownership_between_arms'])
def test_comparator_refuses_unqualified_or_changed_evidence(tmp_path,fault):
    a=report(tmp_path/'baseline',False);b=report(tmp_path/'candidate',True)
    cell=b['results'][0]['cells']['512']
    if fault=='source':b['meta']['paired_k32']['input_manifest_sha256']='e'*64
    elif fault=='native':b['meta']['native_code_artifact']['after_profile_sha256']='e'*64
    elif fault=='population':b['results'][0]['cells'].pop('1')
    elif fault=='input':cell['input_hashes']['x']='unowned'
    elif fault=='output':cell['outputs']['gate_up']['sha256']='e'*64
    elif fault=='bytes':Path(cell['outputs']['gate_up']['path']).write_bytes(b'changed after report')
    elif fault=='build':cell['native_paired_build']=False
    elif fault=='reference':cell['independent_reference']['bounded_numeric_pass']=False
    elif fault=='dispatch':cell['geometry']['0']['paired']=False
    elif fault=='ownership':b['meta']['direct_input_bindings']['after']={'changed':'fingerprint'}
    else:b['meta']['direct_input_bindings'].update(before={'other':'fingerprint'},after={'other':'fingerprint'})
    with pytest.raises(ValueError):q.compare_reports(a,b)


def driver_gate(path):
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='require_single_replay_options')
    namespace={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['require_single_replay_options']


def test_driver_admits_only_new_numeric_mode(monkeypatch):
    monkeypatch.syspath_prepend(str(PATH.parent))
    gate=driver_gate(ROOT/'experiments/t8r_speed/bench_t8r.py')
    args=options()
    gate(args,stubbed=False)
    args.paired_k32_numerics=False
    with pytest.raises(ValueError,match='counter-only replay'):
        gate(args,stubbed=False)


@pytest.mark.parametrize('numeric,stubbed,expected_calls', [
    (True,False,0), (False,False,1), (False,True,0),
])
def test_driver_world_startup_is_only_for_existing_vllm_modes(numeric,stubbed,expected_calls):
    path=ROOT/'experiments/t8r_speed/bench_t8r.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='ctx' for t in n.targets))
    calls=[]
    context=object()
    def initialize(out):
        calls.append(out)
        return context
    namespace={'args':SimpleNamespace(paired_k32_numerics=numeric,out='/owned/output'),
               'VLLM_STUBBED':stubbed,'_init_vllm_world1':initialize}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    assert len(calls)==expected_calls
    assert namespace['ctx'] is (context if expected_calls else None)


@pytest.mark.parametrize('fault', [None,'repeat','missing_role','reduction','profile','foreign'])
def test_actual_numeric_observer_and_refusals(tmp_path,monkeypatch,fault):
    import torch
    from tessera import routed_fused as rf
    lib=SimpleNamespace(PAIRED_K32_BUILD=False,paired_k32_scope=lambda *args:False,
        launch_smem_bytes=lambda *args:40976,max_dynamic_smem_bytes=lambda index:101376)
    monkeypatch.setattr(rf,'_ext',lambda library:lib)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    # Use the actual frozen production adapter; only its CUDA launch seam is
    # replaced by CPU tensor writes in this CPU-only control.
    FusedRoutedWindowMoE=rf.FusedRoutedWindowMoE
    values={field.name:None for field in fields(FusedRoutedWindowMoE) if field.default is MISSING}
    values.update(library='e4m3mma',gate=SimpleNamespace(cols=128,rows=128),
        down=SimpleNamespace(cols=128,rows=4),slot_words_gate_up=8,slot_words_down=8)
    def cpu_launch(self,mode,*args,**kwargs):kwargs['out'].fill_(mode+1)
    monkeypatch.setattr(FusedRoutedWindowMoE,'_launch',cpu_launch)
    native=FusedRoutedWindowMoE(**values);calls=0
    foreign=FusedRoutedWindowMoE(**values)
    def call(x,ids,w):
        nonlocal calls
        calls+=1
        act=torch.empty((4,128),dtype=torch.bfloat16,device='cpu')
        down=torch.empty((4,4),dtype=torch.bfloat16,device='cpu')
        if fault=='foreign':foreign._launch(0,out=torch.empty_like(act))
        native._launch(0,out=act)
        if fault!='missing_role':native._launch(2,out=down)
        else:down.fill_(3)
        out=down.reshape(2,2,4).float().sum(1).bfloat16()
        if fault=='repeat' and calls==2:out[0,0]+=1
        if fault=='reduction':out[0,0]+=1
        return out
    call.native_adapter=native
    def profile(fn,**kwargs):
        return {'top':{f'routed_fused_kernel<true, {mode}, false, false, 4, false, 64, false>':
            {'count_per_call':2 if fault=='profile' else 1} for mode in (0,2)}}
    args=(call,None,torch.ones(2,128,dtype=torch.bfloat16,device='cpu'),torch.zeros(2,2,dtype=torch.int32,device='cpu'),
          torch.ones(2,2,device='cpu'),tmp_path/'words')
    if fault in (None,'foreign'):
        result=q.numeric_cell(*args,kernel_profile=profile,independent_reference=False)
        assert result['repeat_bits_equal'] and result['independent_token_sum_equal']
    else:
        with pytest.raises(ValueError):
            q.numeric_cell(*args,kernel_profile=profile,independent_reference=False)
    # Observation must always restore the real owner, including errors.
    assert native._launch.__func__ is FusedRoutedWindowMoE._launch


@pytest.mark.parametrize('fault',[None,'output','stored'])
def test_timing_guards_accepted_words_before_unprofiled_events(tmp_path,monkeypatch,fault):
    import torch
    from tessera import routed_fused as rf
    values={f.name:None for f in fields(rf.FusedRoutedWindowMoE) if f.default is MISSING}
    values.update(library='e4m3mma',gate=SimpleNamespace(cols=128,rows=128),
                  down=SimpleNamespace(cols=128,rows=4),slot_words_gate_up=8,slot_words_down=8)
    native=rf.FusedRoutedWindowMoE(**values)
    original=torch.ones((1,4),dtype=torch.bfloat16,device='cpu')
    raw=q._words(original);digest=hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(q,'NUMERIC_RECEIPT',str(tmp_path/'RESULT.json'))
    stored=tmp_path/'baseline/numeric-words/1/out.bin';stored.parent.mkdir(parents=True)
    stored.write_bytes(raw if fault!='stored' else b'changed')
    def fn(*args):return original+1 if fault=='output' else original
    fn.native_adapter=native
    lib=SimpleNamespace(PAIRED_K32_BUILD=False,paired_k32_scope=lambda *a:False,
        launch_smem_bytes=lambda *a:40976,max_dynamic_smem_bytes=lambda *a:101376)
    monkeypatch.setattr(rf,'_ext',lambda *a:lib)
    phases=[]
    def events(call,warm,iters):
        assert (warm,iters)==(10,30);phases.append('events');return [1.0]*30
    def profile(call,**kwargs):
        assert kwargs=={'reps':3,'full_names':True};phases.append('profile')
        return {'top':{f'routed_fused_kernel<true, {m}, false, false, 4, false, 64, false>':
                       {'count_per_call':1} for m in (0,2)}}
    def sample(call,seconds,**kwargs):
        assert seconds==3 and kwargs['capture_series'];phases.append('power')
        return {'calls_per_j':123,'source':'control','power_series_unix_w':[]}
    args=(fn,torch.ones(1,128,device='cpu'),torch.zeros(1,2,dtype=torch.int32,device='cpu'),torch.ones(1,2,device='cpu'))
    kwargs=dict(certificate={'compared':[{'M':1,'role':'out','bytes':len(raw),'sha256':digest}]},
                time_events=events,summarize=lambda x:{'median_ms':1},kernel_profile=profile,
                power=SimpleNamespace(sample_during=sample))
    if fault:
        with pytest.raises(ValueError,match='stored numeric words'):q.timing_cell(*args,**kwargs)
        assert phases==[]
    else:
        result=q.timing_cell(*args,**kwargs)
        assert phases==['events','profile','power'] and len(result['raw_events_ms'])==30
        assert result['power']['calls_per_j'] is None
