"""CPU controls for the native app; public plugin calls are explicit stand-ins."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest
import box_artifacts
import torch

from test_native_timing_panel import panel, canonical_wire
from tools import tessera_shape_time_panel as app, tessera_shape_time_worker as worker
from tessera.serving import contract, scheme, telemetry, backend, source_identity, runtime_image


def request_file(panel, monkeypatch):
    monkeypatch.setattr(contract, 'contract_path', lambda: Path(panel['evidence']['contract']['path']))
    value = {'schema': app.REQUEST_SCHEMA, 'expected_runtime': copy.deepcopy(panel['runtime']),
             'scope': copy.deepcopy(panel['plan']['rows'][0]['scope']), 'prefix': 'test.dense',
             'scheme': copy.deepcopy(panel['rows'][0]['scheme']), 'wire': panel['evidence']['wire'],
             'contract': panel['evidence']['contract'], 'record_verifier': app.tp.file_binding(box_artifacts.skip_now('prismabuild_tools','pbtest_pins.py')), 'runtime_python': app.tp.file_binding(__import__('sys').executable), 'worker_timeout_s': 120,
             'sampling': {'samples': 4, 'warmup_iterations': 1, 'steady_s': 20.0, 'seed': 688},
             'netdata_hosts': {'sparky': 'sparky', 'sparklina': 'sparklina'}}
    source = {'schema': app.PRODUCER_SCHEMA, 'commit': 'a'*40, 'commit_source': 'sealed_checkout', **app.producer_source_identity()}
    value['producer_identity'] = app.publish_json(source, Path(panel['evidence']['wire']['path']).parent / 'producer-input.json')
    path = Path(panel['evidence']['wire']['path']).parent / 'request.json'
    path.write_bytes(app.tp.canonical(value))
    return path, value


def test_preflight_is_pure_and_positive(panel, monkeypatch):
    path, value = request_file(panel, monkeypatch)
    got, scope, wire = app.read_request(path)
    assert got == value and wire == app.tp.read_bound(value['wire'])
    assert scope == value['scope']
    assert app.main(['check-request', str(path)]) == 0


@pytest.mark.parametrize('fault', ['contract', 'cell_source', 'tp2', 'samples_bool', 'empty_hosts', 'extra'])
def test_request_refuses_before_native_preparation(panel, monkeypatch, fault):
    path, value = request_file(panel, monkeypatch)
    if fault == 'contract': value['expected_runtime']['contract_sha256'] = '0' * 64
    elif fault == 'cell_source': value['expected_runtime']['serving_source_sha256'] = '0' * 64
    elif fault == 'tp2': value['scope']['tp_degree'] = 2
    elif fault == 'samples_bool': value['sampling']['samples'] = True
    elif fault == 'empty_hosts': value['netdata_hosts']['sparklina'] = ''
    else: value['scope']['implicit_rank'] = 0
    path.write_bytes(app.tp.canonical(value))
    with pytest.raises(ValueError): app.read_request(path)


def test_preparation_uses_public_exact_wire_path(panel, monkeypatch):
    from tessera.serving import lane
    events = []
    class Method:
        def create_weights(self, layer, **kw):
            events.append(('create', kw))
            layer.wire_bytes = torch.nn.Parameter(torch.empty(len(wire), dtype=torch.uint8), requires_grad=False)
        def process_weights_after_loading(self, layer):
            events.append(('finalize', bytes(layer.wire_bytes.detach().numpy())))
            layer.tessera_shard_plan = SimpleNamespace(tp_rank=0, tp_size=1)
            from tessera.serving import native_window
            published=native_window.DENSE_LANES[native_window.LANE_FUSED]
            layer.tessera_native = SimpleNamespace(rows=16, columns=128, packed_bytes=lambda: 123, lane=native_window.LANE_FUSED, launch_pair=(published[0],published[1]["epilogue"]))
    method = Method()
    monkeypatch.setattr(lane, 'build_tessera_method', lambda *a: (events.append(('build', a)) or method))
    path, request = request_file(panel, monkeypatch);wire = app.tp.read_bound(request['wire'])
    layer, got, prep = worker.prepare_dense(dict(request,_wire_roles=app.tp.wire_facts(wire,request["scheme"])[1]), wire)
    assert got is method and [e[0] for e in events] == ['build', 'create', 'finalize']
    assert events[-1][1] == wire and events[1][1]['input_size_per_partition'] == 128
    assert events[1][1]['output_partition_sizes'] == [16]
    assert prep['wire_sha256'] == hashlib.sha256(wire).hexdigest()
    assert prep['native_packed_bytes'] == 123
    assert (layer.tp_rank, layer.tp_size) == (0, 1)


@pytest.mark.parametrize('fresh', [False, True])
def test_route_must_be_emitted_by_this_apply(monkeypatch, fresh):
    layer = SimpleNamespace()
    setattr(layer, telemetry.ATTR_PREFIX + 'state', {'state': 'served', 'symbol': 'old'})
    monkeypatch.setattr(telemetry, 'read_route', lambda obj: getattr(obj, telemetry.ATTR_PREFIX + 'state'))
    def apply(obj, x):
        if fresh: setattr(obj, telemetry.ATTR_PREFIX + 'state', {'state': 'served', 'symbol': 'actual'})
        return x
    method = SimpleNamespace(apply=apply)
    if fresh:
        _, record = worker.fresh_call(layer, method, object());assert record['symbol'] == 'actual'
    else:
        with pytest.raises(ValueError, match='fresh'): worker.fresh_call(layer, method, object())


@pytest.mark.parametrize('failed', [False, True])
def test_publish_file_then_directory_sync(tmp_path, monkeypatch, failed):
    events = []; real_fsync, real_replace = app.os.fsync, app.os.replace
    monkeypatch.setattr(app.os, 'fsync', lambda fd: (events.append('file_sync'), real_fsync(fd))[-1])
    monkeypatch.setattr(app.os, 'replace', lambda a, b: (events.append('rename'), real_replace(a,b))[-1])
    def directory(path):
        assert Path(path) == tmp_path;events.append('directory_sync')
        if failed: raise OSError('directory sync failure')
    monkeypatch.setattr(app, 'fsync_path', directory)
    if failed:
        with pytest.raises(OSError): app.publish_json({'committed': True}, tmp_path/'receipt.json')
    else:
        bound = app.publish_json({'committed': True}, tmp_path/'receipt.json')
        assert app.tp.json_bytes(app.tp.read_bound(bound)) == {'committed': True}
    assert events == ['file_sync', 'rename', 'directory_sync']


def test_runtime_commit_comes_from_actual_imported_checkout(tmp_path, monkeypatch):
    package = SimpleNamespace(__file__=str(tmp_path/'imported'/'__init__.py'))
    (tmp_path/'imported').mkdir();Path(package.__file__).write_text('')
    seen=[]
    def run(argv, **kw):
        seen.append(argv)
        if '--show-toplevel' in argv: out=str(tmp_path)
        elif 'status' in argv: out=''
        elif 'ls-files' in argv: out='imported/__init__.py'
        else: out='a'*40
        return SimpleNamespace(returncode=0,stdout=out)
    monkeypatch.setattr(worker.subprocess,'run',run)
    assert worker.observed_commit(package) == 'a'*40
    assert seen[0][2] == str(Path(package.__file__).parent)


def test_dirty_runtime_and_missing_vcs_identity_refuse(tmp_path,monkeypatch):
    package=SimpleNamespace(__file__=str(tmp_path/'__init__.py'));Path(package.__file__).write_text('')
    def run(argv,**kw):
        return SimpleNamespace(returncode=0,stdout=str(tmp_path) if '--show-toplevel' in argv else ' M source.py')
    monkeypatch.setattr(worker.subprocess,'run',run)
    with pytest.raises(ValueError,match='dirty'):worker.observed_commit(package)
    monkeypatch.setattr(worker.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stdout=''))
    monkeypatch.setattr(worker.importlib.metadata,'distribution',lambda *a:SimpleNamespace(read_text=lambda *a:None,files=[Path('__init__.py')],locate_file=lambda p:tmp_path/p))
    with pytest.raises(ValueError,match='immutable'):worker.observed_commit(package)


def test_foreign_installed_metadata_cannot_label_actual_runtime(tmp_path,monkeypatch):
    package=SimpleNamespace(__file__=str(tmp_path/'foreign'/'__init__.py'));Path(package.__file__).parent.mkdir();Path(package.__file__).write_text('')
    monkeypatch.setattr(worker.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stdout=''))
    dist=SimpleNamespace(read_text=lambda *a:json.dumps({'vcs_info':{'commit_id':'a'*40}}),
                         files=[Path('tessera/__init__.py')],locate_file=lambda p:tmp_path/p)
    monkeypatch.setattr(worker.importlib.metadata,'distribution',lambda *a:dist)
    with pytest.raises(ValueError,match='imported'):worker.observed_commit(package)


@pytest.fixture
def installed_origins(tmp_path,monkeypatch):
    import sys,importlib
    root=tmp_path/'installed'/'tessera';root.mkdir(parents=True)
    modules={}
    for name in worker.MODULES:
        path=root/('__init__.py' if name=='tessera' else name.removeprefix('tessera.').replace('.','/')+'.py')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# installed runtime fixture\n')
        modules[name]=SimpleNamespace(__file__=str(path))
    monkeypatch.delenv('PYTHONPATH',raising=False)
    monkeypatch.setattr(sys,'path',[p for p in sys.path if not Path(p or '.').resolve().is_relative_to(app.ROOT/'src')])
    monkeypatch.setattr(importlib,'import_module',lambda name:modules[name])
    # Test module cache ownership with explicit synthetic installed owners.
    for name in list(sys.modules):
        if name=='tessera' or name.startswith('tessera.'):
            monkeypatch.delitem(sys.modules,name)
    return root,modules


def test_runtime_origins_accept_installed_roster(installed_origins):
    root,_=installed_origins
    proof=worker.runtime_origins(str(root))
    assert set(proof['modules'])==set(app.tp.RUNTIME_MODULES)
    assert proof['package_root']==str(root)


@pytest.mark.parametrize('fault',['pythonpath','producer_src','foreign_module','foreign_cache','producer_root'])
def test_origin_boundary_refuses_before_native_setup(installed_origins,monkeypatch,fault):
    import sys
    root,modules=installed_origins
    if fault=='pythonpath':monkeypatch.setenv('PYTHONPATH','/foreign/source')
    elif fault=='producer_src':monkeypatch.setattr(sys,'path',[str(app.ROOT/'src'),*sys.path])
    elif fault=='foreign_module':modules['tessera.serving.lane'].__file__=str(root.parent/'foreign.py')
    elif fault=='foreign_cache':monkeypatch.setitem(sys.modules,'tessera.foreign',SimpleNamespace(__file__='/foreign/root.py'))
    else:root=app.ROOT/'src'/'tessera'
    with pytest.raises(ValueError):worker.runtime_origins(str(root))


def test_changed_runtime_contract_refuses_before_device_query(panel,monkeypatch):
    import sys
    expected=panel['runtime']
    monkeypatch.setattr(worker,'runtime_origins',lambda p:{'package_root':p,'modules':{}})
    monkeypatch.setattr(worker,'observed_commit',lambda p:expected['tessera_commit'])
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(__version__=expected['vllm']))
    monkeypatch.setattr(torch,'__version__',expected['torch'])
    monkeypatch.setattr(source_identity,'serving_source_sha256',lambda:expected['serving_source_sha256'])
    monkeypatch.setattr(runtime_image,'declared_reference',lambda p:{'image':p})
    path=Path(panel['evidence']['contract']['path']);path.write_bytes(b'changed contract')
    monkeypatch.setattr(contract,'contract_path',lambda:path)
    monkeypatch.setattr(backend,'platform_of_this_process',lambda *a:pytest.fail('device query reached after changed contract'))
    with pytest.raises(ValueError,match='before device'):worker.observe_runtime(expected)


def test_phase_argv_is_isolated_and_preserves_admission(panel,monkeypatch,tmp_path):
    from experiments import step4_capture_launch
    path,_=request_file(panel,monkeypatch)
    seen=[]
    def phase(label,command,log,timeout):
        seen.append((label,command,timeout));return {'returncode':2}
    monkeypatch.setattr(step4_capture_launch,'run_phase',phase)
    # Preserve the real helper origin while observing its exact sealed argv.
    original=Path(app.ROOT/'experiments/step4_capture_launch.py')
    phase.__code__=phase.__code__.replace(co_filename=str(original))
    with pytest.raises(ValueError,match='CPU preflight'):app.measure(path,tmp_path/'owned-output',expected_request_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    label,command,timeout=seen[0]
    assert command[:3]==['env','-u','PYTHONPATH']
    assert '-I' in command and '-B' in command and 'OMP_NUM_THREADS=1' in command
    assert str(app.ROOT/'src') not in command
    assert timeout==120 and label=='runtime-preflight'
    assert 'CUDA_VISIBLE_DEVICES=' in command and '--preflight' in command and '--job-sha256' in command
    assert '-I' in command and not any('pbrun' in arg for arg in command)


def test_prepared_triton_lane_refuses_before_timing(panel,monkeypatch):
    from tessera.serving import lane
    path,request=request_file(panel,monkeypatch);wire=app.tp.read_bound(request["wire"])
    class Method:
        def create_weights(self,layer,**kw):layer.wire_bytes=torch.nn.Parameter(torch.empty(len(wire),dtype=torch.uint8),requires_grad=False)
        def process_weights_after_loading(self,layer):
            layer.tessera_shard_plan=SimpleNamespace(tp_rank=0,tp_size=1)
            layer.tessera_native=SimpleNamespace(rows=16,columns=128,packed_bytes=lambda:1,lane="triton",launch_pair=("tessera::window_gemm_dense","native_window_gemm"))
    monkeypatch.setattr(lane,'build_tessera_method',lambda *a:Method())
    with pytest.raises(ValueError,match="ELF-backed fused"):
        worker.prepare_dense(dict(request,_wire_roles=app.tp.wire_facts(wire,request["scheme"])[1]),wire)


def test_binary_observation_uses_existing_maps_and_never_loader(tmp_path,monkeypatch):
    from tessera import routed_fused
    from tessera.serving import native_window
    from experiments import bench_native_operator
    published=native_window.DENSE_LANES[native_window.LANE_FUSED]
    native=SimpleNamespace(lane=native_window.LANE_FUSED,launch_pair=(published[0],published[1]["epilogue"]))
    binary=tmp_path/(routed_fused.MODULE_NAME_E4M3+'.so');binary.write_bytes(b'\x7fELF CPU fixture')
    monkeypatch.setattr(routed_fused,'_ext',lambda *a:pytest.fail('observer invoked unused native loader'))
    monkeypatch.setattr(bench_native_operator,'_mapped_shared_libraries',lambda:{binary})
    raw=worker.canonical({'native_extensions':[{'module_name_prefix':routed_fused.MODULE_NAME_E4M3,'filename_glob':routed_fused.MODULE_NAME_E4M3+'*.so'}]})
    assert worker.loaded_fused_binary(native,'TESSERA_FP8',raw)==worker.file_binding(binary)
    monkeypatch.setattr(bench_native_operator,'_mapped_shared_libraries',lambda:set())
    with pytest.raises(ValueError,match='already-loaded'):worker.loaded_fused_binary(native,'TESSERA_FP8',raw)


def test_sealed_request_hash_is_checked_before_json(panel,monkeypatch):
    path,_=request_file(panel,monkeypatch)
    monkeypatch.setattr(app.tp,'json_bytes',lambda *a:pytest.fail('unmatched request reached JSON parser'))
    with pytest.raises(ValueError,match='owned request bytes'):
        app.read_request(path,expected_sha256='0'*64)


def test_request_hash_and_json_share_the_same_owned_bytes(panel,monkeypatch):
    path,value=request_file(panel,monkeypatch);raw=path.read_bytes();expected=hashlib.sha256(raw).hexdigest()
    real_json=app.tp.json_bytes
    def parse(owned):
        if owned==raw:path.write_bytes(b'concurrent replacement')
        return real_json(owned)
    monkeypatch.setattr(app.tp,'json_bytes',parse)
    got,_,_=app.read_request(path,expected_sha256=expected)
    assert got==value and path.read_bytes()==b'concurrent replacement'


def test_external_request_defers_contract_validation_to_installed_owner(panel, monkeypatch):
    path, value = request_file(panel, monkeypatch)
    def producer_validator(_):
        raise ValueError('producer roster differs from installed runtime')
    monkeypatch.setattr(app.census_plan, 'validate_serving_contract', producer_validator)
    request, scope, wire = app.read_request(path)
    assert request == value and scope == value['scope']
    assert wire == app.tp.read_bound(value['wire'])


def test_software_observation_does_not_query_device(panel, monkeypatch):
    import sys
    expected = panel['runtime']
    monkeypatch.setenv('TESSERA_SERVE_MODE','resident')
    origins = {'package_root': expected['package_root'], 'modules': {}}
    monkeypatch.setattr(worker, 'runtime_origins', lambda _: origins)
    monkeypatch.setattr(worker, 'observed_commit', lambda _: expected['tessera_commit'])
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(__version__=expected['vllm']))
    monkeypatch.setattr(torch, '__version__', expected['torch'])
    monkeypatch.setattr(source_identity, 'serving_source_sha256', lambda: expected['serving_source_sha256'])
    monkeypatch.setattr(runtime_image, 'declared_reference', lambda _: {'image': expected['image']})
    monkeypatch.setattr(contract, 'contract_path', lambda: Path(panel['evidence']['contract']['path']))
    monkeypatch.setattr(backend, 'platform_of_this_process', lambda *_: pytest.fail('CPU preflight queried CUDA'))
    verifier = app.tp.file_binding(box_artifacts.skip_now('prismabuild_tools','pbtest_pins.py'))
    monkeypatch.setattr(worker, 'record_verifier_bytes', lambda _: b'def verify_install(*args): return {"verified_files": 1}')
    got, _, raw = worker.observe_software_runtime(expected, verifier)
    assert 'platform' not in got
    assert got == {k: v for k, v in expected.items() if k != 'platform'}
    assert raw == app.tp.read_bound(panel['evidence']['contract'])


def preflight_inputs(tmp_path, panel):
    request_source=app.publish_json({'independent':'request'},tmp_path/'original.json')
    worker_source=app.tp.file_binding(worker.__file__)
    origins=copy.deepcopy(app.tp.json_bytes(app.tp.read_bound(panel['evidence']['runtime_origins'])))
    root=tmp_path/'installed'/'tessera';root.mkdir(parents=True)
    runtime=copy.deepcopy(panel['runtime']);runtime['package_root']=str(root)
    origins['package_root']=str(root)
    for name in worker.MODULES:
        suffix='__init__.py' if name=='tessera' else name.removeprefix('tessera.').replace('.','/')+'.py'
        path=root/suffix;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# CPU owner fixture\n')
        origins['modules'][name]=app.tp.file_binding(path)
    origins['installation']['origin']=str(root/'__init__.py')
    verifier=app.tp.file_binding(box_artifacts.skip_now('prismabuild_tools','pbtest_pins.py'))
    origins['record_verifier']=verifier
    request={'record_verifier':verifier}
    request_source=app.publish_json(request,tmp_path/'request-source.json')
    job_source=app.publish_json({'request':request, 'request_source':request_source,
                                'worker_source':worker_source, 'wire_roles':[],
                                'producer':app.tp.json_bytes(app.tp.read_bound(panel['evidence']['producer']))},
                               tmp_path/'job.json')
    command=['env','-u','PYTHONPATH','CUDA_VISIBLE_DEVICES=',str(worker.__file__),'--job',job_source['path'],
             '--job-sha256',job_source['sha256'],'--preflight']
    phase={'phase':'runtime-preflight','returncode':0,'command':command}
    result={'schema':'tessera.installed_contract_preflight.v1','software':{k:v for k,v in runtime.items() if k!='platform'},
            'runtime_origins':origins,'validator':{'module':'tessera.serving.contract','function':'validate_serving_contract',
                                                  'source':origins['modules']['tessera.serving.contract']},
            'contract_sha256':runtime['contract_sha256'],'gpu_executed':False,
            'worker_source':worker_source,'job_source':job_source,'request_source':request_source}
    kwargs=dict(raw_contract=app.tp.read_bound(panel['evidence']['contract']),expected_runtime=runtime,
                job_source=job_source,worker_source=worker_source,request_source=request_source,command=command,phase=phase)
    return result,kwargs


@pytest.mark.parametrize('fault',['rc','bool_rc','argv','job','request','worker','raw','root','validator','module_bytes','copied_roster'])
def test_preflight_refuses_unbound_or_failed_runtime_proof(panel,tmp_path,fault):
    result,kwargs=preflight_inputs(tmp_path,panel)
    if fault=='rc':kwargs['phase']['returncode']=2
    elif fault=='bool_rc':kwargs['phase']['returncode']=False
    elif fault=='argv':kwargs['phase']['command']=['unsealed']
    elif fault in ('job','request','worker'):result[fault+'_source']['sha256']='0'*64
    elif fault=='raw':kwargs['raw_contract']=b'changed owned contract'
    elif fault=='root':result['runtime_origins']['package_root']='/foreign/root'
    elif fault=='validator':result['validator']['function']='caller_roster_attestation'
    elif fault=='module_bytes':Path(result['validator']['source']['path']).write_text('# changed installation\n')
    else:result['native_extensions']=app.tp.json_bytes(kwargs['raw_contract'])['native_extensions']
    with pytest.raises(ValueError):app.tp._verify_runtime_preflight(result,**kwargs)


def test_verified_preflight_is_immutable_and_external_panel_requires_it(panel,tmp_path):
    result,kwargs=preflight_inputs(tmp_path,panel)
    verified=app.tp._verify_runtime_preflight(result,**kwargs)
    result['gpu_executed']=True
    assert verified.result['gpu_executed'] is False
    for attestation in (None,True,False,{'validated':True}):
        with pytest.raises(ValueError,match='verified installed preflight'):
            app.tp.validate_external_panel(panel,expected_runtime=panel['runtime'],runtime_validation=attestation)


def test_local_public_plan_keeps_strict_validator(panel,monkeypatch):
    def strict(_):raise ValueError('strict local owner refusal')
    monkeypatch.setattr(app.census_plan,'validate_serving_contract',strict)
    with pytest.raises(ValueError,match='strict local owner refusal'):
        app.census_plan.build_census_plan([panel['plan']['rows'][0]['scope']],raw_contract=app.tp.read_bound(panel['evidence']['contract']))


def test_external_final_panel_replay_uses_verified_owner_and_refuses_drift(panel,tmp_path,monkeypatch):
    result,kwargs=preflight_inputs(tmp_path,panel)
    verified=app.tp._verify_runtime_preflight(result,**kwargs)
    panel['runtime']=kwargs['expected_runtime']
    root=Path(panel['evidence']['runtime']['path']).parent
    panel['evidence']['runtime']=app.publish_json(panel['runtime'],root/'external-runtime.json')
    panel['evidence']['runtime_origins']=app.publish_json(result['runtime_origins'],root/'external-origins.json')
    panel['preflight']={'result':app.publish_json(result,root/'external-preflight.json'),
                        'phase':app.publish_json(kwargs['phase'],root/'external-phase.json')}
    def wrong_owner(_):pytest.fail('external replay reached producer strict validator')
    monkeypatch.setattr(app.tp,'validate_serving_contract',wrong_owner)
    monkeypatch.setattr(app.census_plan,'validate_serving_contract',wrong_owner)
    replay=app.tp.validate_external_panel(panel,expected_runtime=panel['runtime'],runtime_validation=verified)
    assert replay['energy_status']=='hold' and replay['timing']['n']==4
    changed=copy.deepcopy(result);changed['contract_sha256']='0'*64
    panel['preflight']['result']=app.publish_json(changed,root/'changed-preflight.json')
    with pytest.raises(ValueError,match='preflight differs'):
        app.tp.validate_external_panel(panel,expected_runtime=panel['runtime'],runtime_validation=verified)


def test_local_public_plan_rejects_external_b40_contract(panel):
    raw=box_artifacts.skip_now('b40_runtime','serving','runtime_contract.json').read_bytes()
    # This is the unchanged historical document, not a producer-schema upgrade.
    # New required fields may refuse it before the loader-roster comparison.
    with pytest.raises(ValueError,match='cannot read valid packaged runtime contract'):
        app.census_plan.build_census_plan([panel['plan']['rows'][0]['scope']],raw_contract=raw)


def test_local_public_plan_rejects_external_b40_loader_roster(panel):
    raw=box_artifacts.skip_now('b40_runtime','serving','runtime_contract.json').read_bytes()
    # Isolate the historical roster in a current-schema CPU fixture so this
    # assertion proves roster strictness independently of schema precedence.
    doc=copy.deepcopy(contract.load_serving_contract())
    doc['native_extensions']=json.loads(raw)['native_extensions']
    with pytest.raises(ValueError,match='native_extensions'):
        app.census_plan.build_census_plan([panel['plan']['rows'][0]['scope']],raw_contract=app.tp.canonical(doc))


def test_job_hash_is_checked_before_json_or_imports(tmp_path,monkeypatch):
    path=tmp_path/'job.json';path.write_bytes(b'{not-json')
    monkeypatch.setattr(worker,'json_bytes',lambda _:pytest.fail('unbound job reached JSON parser'))
    with pytest.raises(ValueError,match='owned native job'):
        worker.read_job(path,'0'*64)


def test_repeated_runtime_observation_inserts_overlay_once(installed_origins):
    import sys
    root,_=installed_origins
    worker.runtime_origins(str(root))
    worker.runtime_origins(str(root))
    assert sys.path.count(str(root.parent))==1, 'duplicate overlay roots make RECORD owner discovery ambiguous'


@pytest.mark.parametrize('fault', ['none', 'bytes', 'writable', 'symlink', 'read_drift'])
def test_historical_verifier_requires_the_same_immutable_published_bytes(tmp_path, monkeypatch, fault):
    generations = tmp_path / 'generations'
    old = generations / 'old' / 'tools' / 'pbtest_pins.py'
    current = generations / 'current' / 'tools' / 'pbtest_pins.py'
    old.parent.mkdir(parents=True); current.parent.mkdir(parents=True)
    old.write_bytes(b'published helper bytes'); current.write_bytes(old.read_bytes())
    if fault == 'bytes': old.write_bytes(b'other helper bytes')
    old.chmod(0o444); current.chmod(0o444)
    if fault == 'writable': old.chmod(0o644)
    if fault == 'symlink':
        old.unlink(); old.symlink_to(current)
    binding = worker.file_binding(old)
    if fault == 'symlink': binding['path'] = str(old)
    monkeypatch.setattr(worker, 'PB_RECORD_VERIFIER', current)
    monkeypatch.setattr(worker, 'PB_IMMUTABLE_GENERATIONS', generations)
    if fault == 'read_drift':
        real_fstat = worker.os.fstat
        calls = []
        def drifting(fd):
            value = real_fstat(fd); calls.append(fd)
            if len(calls) == 2:
                return SimpleNamespace(**{name: getattr(value, name) + (1 if name == 'st_mtime_ns' else 0)
                                          for name in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')})
            return value
        monkeypatch.setattr(worker.os, 'fstat', drifting)
    if fault == 'none':
        assert worker.record_verifier_bytes(binding) == current.read_bytes()
    else:
        with pytest.raises(ValueError): worker.record_verifier_bytes(binding)


def external_observation_case(tmp_path, panel, monkeypatch):
    """A genuinely verified external context for the bound-observation path.

    The request is rewritten so its expected runtime names the synthetic
    installed root, then the job, the recorded preflight result and phase are
    rebuilt exactly as the in-container producer would own them. The returned
    ``validation`` is the same private object the real check path derives.
    """
    path, request = request_file(panel, monkeypatch)
    root = tmp_path / 'installed' / 'tessera'; root.mkdir(parents=True)
    runtime = copy.deepcopy(panel['runtime']); runtime['package_root'] = str(root)
    request['expected_runtime'] = runtime
    path.write_bytes(app.tp.canonical(request))
    request_source = app.tp.file_binding(path)
    origins = copy.deepcopy(app.tp.json_bytes(app.tp.read_bound(panel['evidence']['runtime_origins'])))
    origins['package_root'] = str(root)
    for name in worker.MODULES:
        suffix = '__init__.py' if name == 'tessera' else name.removeprefix('tessera.').replace('.', '/') + '.py'
        target = root / suffix; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('# CPU owner fixture\n')
        origins['modules'][name] = app.tp.file_binding(target)
    origins['installation']['origin'] = str(root / '__init__.py')
    verifier = app.tp.file_binding(box_artifacts.skip_now('prismabuild_tools', 'pbtest_pins.py'))
    origins['record_verifier'] = verifier
    worker_source = app.tp.file_binding(app.ROOT / 'tools/tessera_shape_time_worker.py')
    job = {'schema': 'tessera.native_shape_worker_job.v1', 'request': request,
           'request_source': request_source,
           'wire_roles': app.tp.wire_facts(app.tp.read_bound(request['wire']), request['scheme'])[1],
           'producer': app.producer_identity(request['producer_identity']),
           'worker_source': worker_source}
    job_source = app.publish_json(job, tmp_path / 'worker-job.json')
    unit = tmp_path / 'measure'; (unit / 'onecell').mkdir(parents=True)
    phase_dir = unit / 'preflight'
    command = app.phase_command(request, job_source, phase_dir, preflight=True)
    phase = {'phase': 'runtime-preflight', 'returncode': 0, 'command': command}
    result = {'schema': 'tessera.installed_contract_preflight.v1',
              'software': {k: v for k, v in runtime.items() if k != 'platform'},
              'runtime_origins': origins,
              'validator': {'module': 'tessera.serving.contract', 'function': 'validate_serving_contract',
                            'source': origins['modules']['tessera.serving.contract']},
              'contract_sha256': runtime['contract_sha256'], 'gpu_executed': False,
              'worker_source': worker_source, 'job_source': job_source, 'request_source': request_source}
    validation = app.tp._verify_runtime_preflight(result, raw_contract=app.tp.read_bound(request['contract']),
        expected_runtime=runtime, job_source=job_source, worker_source=worker_source,
        request_source=request_source, command=command, phase=phase)
    panel['runtime'] = runtime
    panel['evidence']['producer'] = app.publish_json(job['producer'], unit / 'onecell' / 'original-producer.json')
    panel['evidence']['runtime'] = app.publish_json(runtime, unit / 'onecell' / 'external-runtime.json')
    panel['evidence']['runtime_origins'] = app.publish_json(origins, unit / 'onecell' / 'external-origins.json')
    panel['preflight'] = {'result': app.publish_json(result, unit / 'onecell' / 'external-preflight.json'),
                          'phase': app.publish_json(phase, unit / 'onecell' / 'external-phase.json')}
    panel_path = unit / 'onecell' / 'panel.json'; panel_path.write_bytes(app.tp.canonical(panel))
    runtime_path = unit / 'onecell' / 'expected-runtime.json'; runtime_path.write_bytes(app.tp.canonical(runtime))
    return SimpleNamespace(path=path, request=request, panel=panel, panel_path=panel_path,
                           runtime_path=runtime_path, result=result, origins=origins, unit=unit,
                           job_source=job_source, worker_source=worker_source,
                           request_source=request_source, validation=validation, command=command, phase=phase,
                           panel_binding=app.tp.file_binding(panel_path))


def bound_observation(case, **overrides):
    kwargs = dict(panel_binding=case.panel_binding, expected_panel_sha256=case.panel_binding['sha256'],
                  request_binding=app.tp.file_binding(case.path), request=case.request,
                  expected_runtime_binding=app.tp.file_binding(case.runtime_path),
                  expected_runtime=case.request['expected_runtime'], runtime_validation=case.validation,
                  replay={**app.producer_source_identity(), 'tool': app.tp.file_binding(__file__)})
    kwargs.update(overrides)
    return app.tp.observation(case.panel, **kwargs)


@pytest.mark.parametrize('fault', ['none', 'producer', 'wire_roles', 'worker'])
def test_current_cpu_job_joins_the_unchanged_original_measurement(tmp_path, panel, monkeypatch, fault):
    case = external_observation_case(tmp_path, panel, monkeypatch)
    replay_job = app.tp.json_bytes(app.tp.read_bound(case.job_source))
    if fault == 'producer': replay_job['producer']['tool_source_sha256'] = '0' * 64
    elif fault == 'wire_roles': replay_job['wire_roles'] = []
    replay_source = app.publish_json(replay_job, tmp_path / 'replay-job.json')
    result = copy.deepcopy(case.result); result['job_source'] = replay_source
    command = app.phase_command(case.request, replay_source, tmp_path / 'replay', preflight=True)
    if fault == 'worker':
        command[command.index(case.worker_source['path'])] = '/foreign/worker.py'
    phase = {'phase': 'runtime-preflight', 'returncode': 0, 'command': command}
    def validate():
        validation = app.tp._verify_runtime_preflight(
            result, raw_contract=app.tp.read_bound(case.request['contract']),
            expected_runtime=case.request['expected_runtime'], job_source=replay_source,
            worker_source=case.worker_source, request_source=case.request_source,
            command=command, phase=phase)
        return app.tp.validate_external_panel(case.panel, expected_runtime=case.request['expected_runtime'],
                                               runtime_validation=validation)
    if fault == 'none':
        assert validate()['timing']['n'] == 4
    else:
        with pytest.raises(ValueError): validate()


def test_bound_observation_records_distinct_identities_and_sampling(tmp_path, panel, monkeypatch):
    case = external_observation_case(tmp_path, panel, monkeypatch)
    doc = bound_observation(case)
    assert doc['schema'] == app.tp.OBSERVATION_SCHEMA and doc['status'] == 'validated'
    assert doc['gpu_executed'] is False and doc['energy_status'] == 'hold'
    assert doc['family'] == contract.PAYLOAD_FAMILY_BY_ROUTE[scheme.TESSERA_FP8]
    assert doc['rank_local_shape'] == '16x128' and doc['payload'] == {
        'route': 'TESSERA_FP8', 'grid': 'E4M3', 'q256': 1024, 'rows': 16, 'columns': 128}
    assert doc['timing']['median_ms'] == 2.5
    assert doc['sampling'] == {'method': 'cuda_events', 'sample_unit': 'single_apply',
                               'warmup_iterations': 1, 'n': 4, 'samples_ms': [1.0, 2.0, 3.0, 4.0],
                               'interval_unix': [10.0, 11.0]}
    assert doc['operator_projection']['batch_size'] == 1 and doc['operator_projection']['rows'] == 512
    assert doc['producer'] == case.request['producer_identity']
    producer_identity = app.tp.json_bytes(app.tp.read_bound(case.request['producer_identity']))
    # The producer and replay tools live in the same source tree here, so their
    # tree-derived digests coincide; the two identities stay separately bound.
    assert producer_identity['tool_source_sha256'] == doc['replay']['tool_source_sha256']
    assert doc['replay']['tool']['sha256'] == app.tp.file_binding(__file__)['sha256']
    assert doc['replay']['tool']['path'] != doc['producer']['path']
    assert doc['evidence'] == case.panel['evidence'] and doc['preflight'] == case.panel['preflight']


@pytest.mark.parametrize('fault', ['digest', 'token', 'panel_bytes', 'runtime_bytes'])
def test_bound_observation_refuses_unbound_or_unverified_context(tmp_path, panel, monkeypatch, fault):
    case = external_observation_case(tmp_path, panel, monkeypatch)
    overrides = {}
    if fault == 'digest':
        overrides['expected_panel_sha256'] = '0' * 64
        match = 'externally supplied digest'
    elif fault == 'token':
        overrides['runtime_validation'] = {'validated': True}
        match = 'verified installed preflight'
    elif fault == 'panel_bytes':
        Path(case.panel_binding['path']).write_bytes(b'changed panel bytes')
        match = 'file evidence'
    else:
        # An extra JSON key changes the expected runtime the caller passes,
        # while the observation still binds the original bytes, so the
        # bound-bytes cross-check must refuse rather than trust the caller.
        altered = copy.deepcopy(case.request['expected_runtime'])
        altered['implicit_rank'] = 0
        overrides['expected_runtime'] = altered
        match = 'bound bytes'
    with pytest.raises(ValueError, match=match):
        bound_observation(case, **overrides)


def test_check_emits_bound_observation_after_verified_preflight(tmp_path, panel, monkeypatch):
    case = external_observation_case(tmp_path, panel, monkeypatch)
    from experiments import step4_capture_launch
    seen = []
    def phase(label, command, log, timeout):
        seen.append(command)
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        result = copy.deepcopy(case.result)
        job_path = Path(command[command.index('--job') + 1])
        result['job_source'] = app.tp.file_binding(job_path)
        result['worker_source'] = app.tp.json_bytes(job_path.read_bytes())['worker_source']
        (Path(log).parent / 'runtime-preflight.json').write_bytes(app.tp.canonical(result))
        return {'phase': label, 'returncode': 0, 'command': command}
    phase.__code__ = phase.__code__.replace(co_filename=str(Path(app.ROOT / 'experiments/step4_capture_launch.py')))
    monkeypatch.setattr(step4_capture_launch, 'run_phase', phase)
    observation_path = case.unit / 'observation.json'
    rc = app.main(['check', str(case.panel_path), '--expected-runtime', str(case.runtime_path),
                   '--request', str(case.path),
                   '--request-sha256', hashlib.sha256(case.path.read_bytes()).hexdigest(),
                   '--preflight-output', str(case.unit / 'replay-preflight'),
                   '--expected-panel-sha256', hashlib.sha256(case.panel_path.read_bytes()).hexdigest(),
                   '--observation-out', str(observation_path)])
    assert rc == 0 and observation_path.exists()
    assert seen and 'CUDA_VISIBLE_DEVICES=' in seen[0] and '--preflight' in seen[0]
    doc = app.tp.json_bytes(observation_path.read_bytes())
    assert doc['schema'] == app.tp.OBSERVATION_SCHEMA and doc['status'] == 'validated'
    assert doc['sampling']['samples_ms'] == [1.0, 2.0, 3.0, 4.0]
    assert doc['panel']['sha256'] == hashlib.sha256(case.panel_path.read_bytes()).hexdigest()


@pytest.mark.parametrize('fault', ['digest', 'no_digest', 'local_panel'])
def test_check_writes_no_observation_on_a_refused_bound_request(tmp_path, panel, monkeypatch, fault):
    case = external_observation_case(tmp_path, panel, monkeypatch)
    from experiments import step4_capture_launch
    def phase(label, command, log, timeout):
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        (Path(log).parent / 'runtime-preflight.json').write_bytes(app.tp.canonical(case.result))
        return {'phase': label, 'returncode': 0, 'command': command}
    phase.__code__ = phase.__code__.replace(co_filename=str(Path(app.ROOT / 'experiments/step4_capture_launch.py')))
    monkeypatch.setattr(step4_capture_launch, 'run_phase', phase)
    panel_path = case.panel_path
    if fault == 'local_panel':
        # The observation path re-binds the expected runtime against the bytes
        # the caller names; point `--expected-runtime` at different bytes so the
        # bound-observation cross-check refuses even though the panel is valid.
        bound = case.unit / 'bound-panel.json'
        bound.write_bytes(app.tp.canonical(case.panel))
        binding = app.tp.file_binding(bound)
        other_runtime = case.unit / 'other-runtime.json'
        other_runtime.write_bytes(app.tp.canonical(dict(case.request['expected_runtime'], implicit_rank=0)))
        assert app.main(['check', str(bound), '--expected-runtime', str(other_runtime),
                         '--request', str(case.path),
                         '--request-sha256', hashlib.sha256(case.path.read_bytes()).hexdigest(),
                         '--preflight-output', str(case.unit / 'refused-local_panel'),
                         '--expected-panel-sha256', binding['sha256'],
                         '--observation-out', str(case.unit / 'refused-observation.json')]) == 2
        assert not (case.unit / 'refused-observation.json').exists()
        return
    argv = ['check', str(panel_path), '--expected-runtime', str(case.runtime_path),
            '--request', str(case.path),
            '--request-sha256', hashlib.sha256(case.path.read_bytes()).hexdigest(),
            '--preflight-output', str(case.unit / f'refused-{fault}'),
            '--observation-out', str(case.unit / 'refused-observation.json')]
    if fault == 'digest':
        argv += ['--expected-panel-sha256', '0' * 64]
    elif fault != 'no_digest':
        argv += ['--expected-panel-sha256', hashlib.sha256(panel_path.read_bytes()).hexdigest()]
    assert app.main(argv) == 2
    assert not (case.unit / 'refused-observation.json').exists()


def _observe_with_source(panel, monkeypatch, source_digest):
    import sys
    expected = panel['runtime']
    monkeypatch.setenv('TESSERA_SERVE_MODE', 'resident')
    monkeypatch.setattr(worker, 'runtime_origins', lambda _: {'package_root': expected['package_root'], 'modules': {}})
    monkeypatch.setattr(worker, 'observed_commit', lambda _: expected['tessera_commit'])
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(__version__=expected['vllm']))
    monkeypatch.setattr(torch, '__version__', expected['torch'])
    monkeypatch.setattr(source_identity, 'serving_source_sha256', lambda: source_digest)
    monkeypatch.setattr(runtime_image, 'declared_reference', lambda _: {'image': expected['image']})
    monkeypatch.setattr(contract, 'contract_path', lambda: Path(panel['evidence']['contract']['path']))
    monkeypatch.setattr(backend, 'platform_of_this_process', lambda *_: pytest.fail('CPU observation queried CUDA'))
    monkeypatch.setattr(worker, 'record_verifier_bytes', lambda _: b'def verify_install(*args): return {"verified_files": 1}')
    return expected, app.tp.file_binding(box_artifacts.skip_now('prismabuild_tools', 'pbtest_pins.py'))


def test_source_pin_difference_stamps_and_continues(panel, monkeypatch, capsys):
    """The reported old-source-pin refusal: the worker stamps and proceeds."""
    monkeypatch.delenv('PRISMAQUANT_DEV_MODE', raising=False)
    expected, verifier = _observe_with_source(panel, monkeypatch, 'e' * 64)
    got, origins, raw = worker.observe_software_runtime(expected, verifier)
    assert got['serving_source_sha256'] == 'e' * 64
    # One [DEV-MODE] stamp per seal site: the code identity and the frozen
    # software context each stamp once; the record keeps the actual digest.
    assert capsys.readouterr().out.count('[DEV-MODE]') == 2
    assert raw == app.tp.read_bound(panel['evidence']['contract'])


def test_certified_zero_keeps_before_device_refusal(panel, monkeypatch):
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', '0')
    expected, verifier = _observe_with_source(panel, monkeypatch, 'e' * 64)
    with pytest.raises(ValueError, match='runtime source/contract/version differs before device setup'):
        worker.observe_software_runtime(expected, verifier)


def test_installation_proof_verifies_the_observed_commit_in_dev_mode(panel, monkeypatch):
    """Dev mode proves the installation at the commit the running tree claims."""
    monkeypatch.delenv('PRISMAQUANT_DEV_MODE', raising=False)
    expected, verifier = _observe_with_source(panel, monkeypatch, 'e' * 64)
    staged = 'f' * 40
    monkeypatch.setattr(worker, 'observed_commit', lambda _: staged)
    code = (f"def verify_install(module, commit):\n"
            f"    assert commit == '{staged}', commit\n"
            f"    return {{'verified_files': 1}}\n")
    monkeypatch.setattr(worker, 'record_verifier_bytes', lambda _: code.encode())
    got, origins, _ = worker.observe_software_runtime(expected, verifier)
    assert origins['installation'] == {'verified_files': 1}


def test_changed_runtime_contract_still_refuses_in_dev_mode(panel, monkeypatch):
    """Contract bytes against their own pinned digest are integrity, not a seal."""
    monkeypatch.delenv('PRISMAQUANT_DEV_MODE', raising=False)
    expected, verifier = _observe_with_source(
        panel, monkeypatch, panel['runtime']['serving_source_sha256'])
    Path(panel['evidence']['contract']['path']).write_bytes(b'changed contract')
    with pytest.raises(ValueError, match='runtime source/contract/version differs before device setup'):
        worker.observe_software_runtime(expected, verifier)


def test_preflight_software_difference_stamps_and_continues(tmp_path, panel, monkeypatch, capsys):
    monkeypatch.delenv('PRISMAQUANT_DEV_MODE', raising=False)
    result, kwargs = preflight_inputs(tmp_path, panel)
    result['software']['serving_source_sha256'] = '0' * 64
    app.tp._verify_runtime_preflight(result, **kwargs)
    assert capsys.readouterr().out.count('[DEV-MODE]') == 1


def test_certified_zero_keeps_preflight_software_refusal(tmp_path, panel, monkeypatch):
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', '0')
    result, kwargs = preflight_inputs(tmp_path, panel)
    result['software']['serving_source_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='preflight software/contract differs from independent context'):
        app.tp._verify_runtime_preflight(result, **kwargs)


def test_preflight_installation_commit_difference_stamps(tmp_path, panel, monkeypatch, capsys):
    monkeypatch.delenv('PRISMAQUANT_DEV_MODE', raising=False)
    result, kwargs = preflight_inputs(tmp_path, panel)
    result['runtime_origins']['installation']['expected_commit'] = 'f' * 40
    result['runtime_origins']['installation']['installed_commit'] = 'f' * 40
    app.tp._verify_runtime_preflight(result, **kwargs)
    assert capsys.readouterr().out.count('[DEV-MODE]') == 1


def test_preflight_tampered_raw_contract_still_refuses_in_dev_mode(tmp_path, panel, monkeypatch):
    monkeypatch.delenv('PRISMAQUANT_DEV_MODE', raising=False)
    result, kwargs = preflight_inputs(tmp_path, panel)
    kwargs['raw_contract'] = b'changed owned contract'
    with pytest.raises(ValueError, match='preflight software/contract differs'):
        app.tp._verify_runtime_preflight(result, **kwargs)
