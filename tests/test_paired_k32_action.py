"""Finite direct-vLLM bounds and exact owned-container cleanup, no real Docker."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH=Path(__file__).resolve().parents[1]/'experiments/t8r_speed/paired_k32_action.py'


def action(monkeypatch):
    monkeypatch.syspath_prepend(str(PATH.parent))
    spec=importlib.util.spec_from_file_location('paired_numeric_action_control',PATH)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


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
