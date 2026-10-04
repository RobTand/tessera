"""Finite admitted-PB batch bounds and exact owned cleanup, no real Docker."""
import importlib.util
import json
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH=Path(__file__).resolve().parents[1]/'experiments/t8r_speed/paired_k32_action.py'


def action(monkeypatch):
    monkeypatch.syspath_prepend(str(PATH.parent))
    spec=importlib.util.spec_from_file_location('paired_numeric_action_control',PATH)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("admitted,staged", [(False, False), (True, False), (True, True)])
def test_paired_batch_requires_admitted_pb_without_mixing_input_transports(monkeypatch, admitted, staged):
    module = action(monkeypatch)
    if admitted:
        monkeypatch.setenv("PRISMABUILD_ACTION_KEY", "a" * 64)
    else:
        monkeypatch.delenv("PRISMABUILD_ACTION_KEY", raising=False)
    if staged:
        monkeypatch.setenv("BENCH_STRICT_STAGED", "1")
    else:
        monkeypatch.delenv("BENCH_STRICT_STAGED", raising=False)
    if not admitted:
        with pytest.raises(ValueError, match="admitted PrismaBuild"):
            module.require_pb_execution()
    elif staged:
        with pytest.raises(ValueError, match="cannot also use staged"):
            module.require_pb_execution()
    else:
        module.require_pb_execution()



@pytest.mark.parametrize('state',['owned','foreign','absent','absent_lowercase','absent_wrong_cid','daemon_error'])
def test_cleanup_checks_actual_cid_and_owner_before_removing(tmp_path,monkeypatch,state):
    module=action(monkeypatch)
    cid='c'*64;token='a'*32
    (tmp_path/'owned.cid').write_text(cid)
    (tmp_path/'owner-token.txt').write_text(token)
    calls=[]
    def run(argv,**kwargs):
        calls.append(argv)
        assert kwargs['timeout'] in (5,15)
        if argv[1]=='inspect':
            if state=='absent':return SimpleNamespace(returncode=1,stderr='Error: No such object: '+cid)
            if state=='absent_lowercase':return SimpleNamespace(returncode=1,stderr='error: no such object: '+cid)
            if state=='absent_wrong_cid':return SimpleNamespace(returncode=1,stderr='error: no such object: '+'f'*64)
            if state=='daemon_error':return SimpleNamespace(returncode=1,stderr='Cannot connect to daemon')
            return SimpleNamespace(returncode=0,stdout=json.dumps([{'Id':cid,'Config':{'Labels':
                {'tessera.paired_numeric_owner':token if state=='owned' else 'f'*32}}}]))
        return SimpleNamespace(returncode=0,stderr='')
    if state in ('foreign','daemon_error','absent_wrong_cid'):
        with pytest.raises((ValueError,RuntimeError)):module.owned_cleanup(tmp_path,run=run)
        assert len(calls)==1
    else:
        result=module.owned_cleanup(tmp_path,run=run)
        assert result['cid']==cid
        assert calls==[['docker','inspect',cid]]+([['docker','rm','-f',cid]] if state=='owned' else [])


@pytest.mark.parametrize('pressure',[(39*(1<<30),0),(80*(1<<30),20)])
def test_launch_headroom_refuses_before_starting_child(tmp_path,monkeypatch,pressure):
    module=action(monkeypatch)
    with pytest.raises(RuntimeError,match='headroom'):
        module.run_direct_arm([],{},None,tmp_path/'arm',tmp_path/'guard',pressure=lambda:pressure,
            popen=lambda *a,**k:pytest.fail('must not launch'))


@pytest.mark.parametrize('reason',['memory','psi','timeout','kill_race'])
def test_active_bound_stops_only_its_owned_process_group_and_cleans_up(tmp_path,monkeypatch,reason):
    module=action(monkeypatch)
    pressures=iter([(80*(1<<30),0),((23 if reason in ('memory','kill_race') else 80)*(1<<30),20 if reason=='psi' else 0)])
    clock=iter([0,241 if reason=='timeout' else 1])
    monkeypatch.setattr(module.time,'monotonic',lambda:next(clock))
    killed=[]
    def kill(pid,sig):
        killed.append((pid,sig))
        if reason=='kill_race':raise ProcessLookupError('owned group already exited')
    monkeypatch.setattr(module.os,'killpg',kill)
    class Child:
        pid=424242
        returncode=None
        def poll(self):return self.returncode
        def wait(self,timeout):self.returncode=-15;return self.returncode
    child=Child()
    cleanup=[]
    monkeypatch.setattr(module,'owned_cleanup',lambda out:cleanup.append(out) or {'state':'controlled'})
    with pytest.raises(RuntimeError,match='bound reached'):
        module.run_direct_arm([],{},None,tmp_path/'arm',tmp_path/'guard',pressure=lambda:next(pressures),
            popen=lambda *a,**k:child)
    assert killed==[(child.pid,module.signal.SIGTERM)]
    assert cleanup==[tmp_path/'arm']
    assert json.loads((tmp_path/'arm-cleanup.json').read_text())['state']=='controlled'


@pytest.mark.parametrize('fault',[None,'population','event_count','dispatch','native','owner','output',
    'advertised_median','sample_nan','sample_inf','sample_zero','sample_negative','sample_bool','median_bool'])
def test_abba_summary_checks_qualified_population(monkeypatch,fault):
    module=action(monkeypatch);reports={}
    cert={'compared':[{'M':int(m),'role':'out','sha256':m} for m in ('1','512','2048')]}
    for arm in ('A1','B1','B2','A2'):
        reports[arm]={'meta':{'direct_input_bindings':{'before':{'owned':1},'after':{'owned':1}},
            'native_code_artifact':{k:'sha' for k in ('before_load_sha256','after_load_sha256','after_profile_sha256')}},
            'results':[{'ok':True,'cells':{m:{'raw_events_ms':([0.5,1.5] if arm.startswith('B') else [1,3])*15,'out_sha256':m,
                'numeric_receipt_sha256':module.NUMERIC_RECEIPT_SHA,'native_paired_build':arm.startswith('B'),
                'wall':{'median_ms':1 if arm.startswith('B') else 2},
                'geometry':{mode:{'paired':arm.startswith('B') and m!='1'} for mode in ('0','2')}}
                for m in ('1','512','2048')}}]}
    r=reports['B1'];cell=r['results'][0]['cells']['512']
    if fault=='population':reports.pop('A2')
    elif fault=='event_count':cell['raw_events_ms'].pop()
    elif fault=='dispatch':cell['geometry']['0']['paired']=False
    elif fault=='native':r['meta']['native_code_artifact']['after_profile_sha256']='other'
    elif fault=='owner':r['meta']['direct_input_bindings']['after']={'changed':1}
    elif fault=='output':cell['out_sha256']='wrong'
    elif fault=='advertised_median':cell['wall']['median_ms']=0.25
    elif fault=='median_bool':cell['wall']['median_ms']=True
    elif fault in ('sample_nan','sample_inf','sample_zero','sample_negative','sample_bool'):
        cell['raw_events_ms'][0]={'sample_nan':float('nan'),'sample_inf':float('inf'),
            'sample_zero':0,'sample_negative':-1,'sample_bool':True}[fault]
    if fault:
        with pytest.raises(ValueError):module.timing_summary(reports,cert)
    else:assert module.timing_summary(reports,cert)['cells']['512']['operator_speedup']==2
