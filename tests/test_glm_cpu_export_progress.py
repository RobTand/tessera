"""The campaign wrapper advances only after owning helpers finish durable work."""
import hashlib, importlib.util, json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tessera.source_digest_cache import SourceDigestCache


def driver():
    spec=importlib.util.spec_from_file_location('glm_cpu_export_runner',Path(__file__).resolve().parents[1]/'tools/run_glm_cached_cpu_export.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_cached_source_and_atomic_output_advance_durable_cumulative_units(tmp_path):
    module=driver();source=tmp_path/'source';source.mkdir();shard=source/'model.safetensors';shard.write_bytes(b'unchanged source')
    cache_dir=tmp_path/'cache';cache_dir.mkdir();cache=SourceDigestCache(cache_dir,source=source,quiescent_seconds=0)
    output=tmp_path/'output';output.mkdir();events=[]
    progress=module.DurableExportProgress(tmp_path/'events',output,{shard.name:hashlib.sha256(shard.read_bytes()).hexdigest()},lambda n,p:events.append((n,p)))
    def save(payload,path):path.write_bytes(payload)
    exporter=SimpleNamespace(save_serving_shard=save)
    with progress.install(exporter):
        cache.sha256(shard);cache.sha256(shard)
        exporter.save_serving_shard(b'produced',(output/'model.safetensors'))
        for name in ('config.json','model.safetensors.index.json','tessera_serving_manifest.json'):(output/name).write_text('{}')
        progress.finish()
    assert events==[(1,'source_verify'),(2,'export_shards'),(3,'publish')]
    assert len(progress.events.read_text().splitlines())==3
    assert exporter.save_serving_shard is save


def test_differing_source_or_failed_publish_cannot_advance(tmp_path):
    module=driver();source=tmp_path/'source';source.mkdir();shard=source/'model.safetensors';shard.write_bytes(b'changed source')
    cache_dir=tmp_path/'cache';cache_dir.mkdir();cache=SourceDigestCache(cache_dir,source=source,quiescent_seconds=0)
    output=tmp_path/'output';output.mkdir();events=[]
    progress=module.DurableExportProgress(tmp_path/'events',output,{shard.name:'0'*64},lambda n,p:events.append((n,p)))
    def fail(*a):raise OSError('write failed')
    exporter=SimpleNamespace(save_serving_shard=fail)
    with progress.install(exporter):
        with pytest.raises(ValueError,match='current source'):cache.sha256(shard)
        with pytest.raises(OSError,match='write failed'):exporter.save_serving_shard(b'x',output/'model.safetensors')
    assert events==[]
    assert not progress.events.exists()


def test_progress_line_follows_the_fsync_of_the_unit_it_reports(tmp_path,monkeypatch):
    """Order is the obligation: the durable record is on disk before the progress line is printed."""
    import os
    import stat
    module=driver();order=[];real=module.os.fsync
    def fsync(fd):
        order.append('directory-fsync' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file-fsync')
        return real(fd)
    monkeypatch.setattr(module.os, 'fsync', fsync)
    output=tmp_path/'output';output.mkdir()
    progress=module.DurableExportProgress(tmp_path/'events',output,{},lambda n,p:order.append(('commit',n,p)))
    progress.completed('export_shards','a',{'bytes':1})
    assert order[-1]==('commit',1,'export_shards')
    assert order.index('file-fsync') < order.index('directory-fsync') < len(order) - 1
    assert (tmp_path/'events'/'durable-events.jsonl').read_text().count('\n')==1


def test_the_shared_progress_helper_prints_one_schema_line_per_report(capsys):
    import sys
    tools=Path(__file__).resolve().parents[1]
    if str(tools) not in sys.path:sys.path.insert(0,str(tools))
    from tessera._dev.durable_progress import emit
    emit(3,'construct',unit='durable checkpoints');emit(4,'publish')
    lines=capsys.readouterr().out.splitlines()
    assert [json.loads(line) for line in lines]==[
        {'schema':'tessera.durable_progress.v1','units_completed':3,'phase':'construct','unit':'durable checkpoints'},
        {'schema':'tessera.durable_progress.v1','units_completed':4,'phase':'publish'}]


def test_incomplete_input_bindings_cannot_enter_export(tmp_path):
    module=driver();path=tmp_path/'bindings.json';path.write_text(json.dumps({'schema':'prismaquant.glm_cached_cpu_export_bindings.v1','inputs':{}}))
    with pytest.raises(ValueError,match='incomplete'):
        module.main(['--bindings',str(path),'--bindings-sha256',hashlib.sha256(path.read_bytes()).hexdigest()])


def renderer():
    import sys
    tools=Path(__file__).resolve().parents[1]/'tools'
    if str(tools) not in sys.path:sys.path.insert(0,str(tools))
    spec=importlib.util.spec_from_file_location('render_glm_cpu_export_command',tools/'render_glm_cpu_export_command.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def _bound(path,payload):
    path.write_bytes(payload);return {'path':str(path),'sha256':hashlib.sha256(payload).hexdigest()}


def _bindings(tmp_path,inputs):
    bound={}
    for name in inputs:
        payload=(json.dumps({'schema':'tessera.cached_units.v2','units':{'u':{'blob_bytes':5}}})
                 if name=='selected_manifest' else name).encode()
        bound[name]=_bound(tmp_path/f'{name}.json',payload)
    doc={'schema':'prismaquant.glm_cached_cpu_export_bindings.v1','inputs':bound,'source':'/src',
         'output':'/out','required_free_bytes':1,'intake_threads':7,'intake_window_bytes':8<<30}
    return _bound(tmp_path/'bindings.json',json.dumps(doc).encode())


def _repo(path):
    import subprocess
    path.mkdir()
    run=lambda *a:subprocess.run(['git','-C',str(path),*a],check=True,capture_output=True,text=True).stdout
    run('init','-q');(path/'f').write_text('x');run('add','f')
    run('-c','user.name=t','-c','user.email=t@t','commit','-qm','x')
    return run('rev-parse','HEAD').strip()


def test_the_rendered_row_is_a_stock_jobspec_on_the_venv_of_its_commit(tmp_path):
    module=renderer();binding=_bindings(tmp_path,module.INPUTS);commit=_repo(tmp_path/'checkout')
    result=module.render(binding['path'],binding['sha256'],tmp_path/'checkout', walltime='20m', fleet_master_approved='fleet-master fixture decision')
    spec=result['jobspec']
    assert result['schema']=='prismaquant.reviewable_cpu_export_submission.v2' and 'argv' not in result
    assert spec['resources_shape']=='slot/[core=8;memory=48]' and spec['requires']=='host:dl380g10'
    python=str(module.VENV_ROOT/f'pq-cpu312-tessera-{commit[:8]}'/'bin'/'python')
    assert spec['command'][0]==python and spec['command'][1]=='tools/run_glm_cached_cpu_export.py'
    assert result['tessera_commit']==commit and result['interpreter']['path']==python
    assert result['submitted'] is False and spec['walltime_s'] == 1200
    assert spec['environment']['PYTHONPATH']=='src' and spec['environment']['OMP_NUM_THREADS']=='1'
    text=json.dumps(result)
    assert 'pbrun' not in text and 'prismabuild' not in text.lower() and '--priority' not in text


def test_a_dirty_checkout_is_refused_rather_than_named_after_its_head(tmp_path):
    module=renderer();binding=_bindings(tmp_path,module.INPUTS);_repo(tmp_path/'checkout')
    (tmp_path/'checkout'/'f').write_text('changed')
    with pytest.raises(ValueError,match='uncommitted'):
        module.render(binding['path'],binding['sha256'],tmp_path/'checkout', walltime='20m', fleet_master_approved='fleet-master fixture decision')


def test_a_producer_authority_binding_is_optional_and_nothing_else_is(tmp_path):
    module=renderer();_repo(tmp_path/'checkout')
    binding=_bindings(tmp_path,(*module.INPUTS,'producer_authority'))
    assert module.render(binding['path'],binding['sha256'],tmp_path/'checkout', walltime='20m', fleet_master_approved='fleet-master fixture decision')['submitted'] is False
    other=tmp_path/'other';other.mkdir();binding=_bindings(other,(*module.INPUTS,'unreviewed'))
    with pytest.raises(ValueError,match='incomplete'):
        module.render(binding['path'],binding['sha256'],tmp_path/'checkout', walltime='20m', fleet_master_approved='fleet-master fixture decision')


@pytest.mark.parametrize('walltime', ['0', '0s', '12', '-1m', 'nanm'])
def test_cpu_export_renderer_requires_positive_explicit_walltime(walltime):
    with pytest.raises(ValueError, match='positive|walltime'):
        renderer().render('unused', 'unused', 'unused', walltime=walltime,
                          fleet_master_approved='fleet-master fixture decision')


def test_cpu_export_oversize_requires_named_approval(tmp_path):
    module=renderer();binding=_bindings(tmp_path,module.INPUTS);_repo(tmp_path/'checkout')
    args=(binding['path'],binding['sha256'],tmp_path/'checkout')
    with pytest.raises(ValueError, match='--fleet-master-approved'):
        module.render(*args, walltime='20m')
    result=module.render(*args, walltime='20m', fleet_master_approved='fleet-master fixture decision')
    assert result['jobspec']['fleet_master_approved']=='fleet-master fixture decision'
    assert result['jobspec']['walltime_s']==1200


@pytest.mark.parametrize('cores,memory,placement', [(11, 8, 'sparky'), (8, 25, 'dl380g10')])
def test_cpu_export_plain_limits_are_checked_before_bindings(monkeypatch, cores, memory, placement):
    module=renderer()
    monkeypatch.setattr(module, 'CORES', cores)
    monkeypatch.setattr(module, 'MEMORY_GIB', memory)
    monkeypatch.setattr(module, 'PLACEMENT_TAG', placement)
    with pytest.raises(ValueError, match='--fleet-master-approved'):
        module.render('unused', 'unused', 'unused', walltime='20m')
