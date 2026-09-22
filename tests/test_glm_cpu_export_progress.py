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
