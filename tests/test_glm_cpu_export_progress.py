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


def test_incomplete_input_bindings_cannot_enter_export(tmp_path,monkeypatch):
    module=driver();path=tmp_path/'bindings.json';path.write_text(json.dumps({'schema':'prismaquant.glm_cached_cpu_export_bindings.v1','inputs':{}}))
    monkeypatch.setenv('PRISMABUILD_ACTION_PROGRESS_HELPER','/unused')
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


def test_the_rendered_row_takes_the_agent_band_and_the_venv_of_its_commit(tmp_path):
    module=renderer();binding=_bindings(tmp_path,module.INPUTS);commit=_repo(tmp_path/'checkout')
    result=module.render(binding['path'],binding['sha256'],tmp_path/'checkout')
    argv=result['argv']
    assert argv[argv.index('--priority')+1]=='-10'
    python=f'/home/rob/venvs/pq-cpu312-tessera-{commit[:8]}/bin/python'
    assert argv[argv.index('--')+1]==python
    assert result['tessera_commit']==commit and result['interpreter']['path']==python
    assert result['submitted'] is False
    assert not [a for a in argv if '4c384e60' in a]


def test_a_dirty_checkout_is_refused_rather_than_named_after_its_head(tmp_path):
    module=renderer();binding=_bindings(tmp_path,module.INPUTS);_repo(tmp_path/'checkout')
    (tmp_path/'checkout'/'f').write_text('changed')
    with pytest.raises(ValueError,match='uncommitted'):
        module.render(binding['path'],binding['sha256'],tmp_path/'checkout')
