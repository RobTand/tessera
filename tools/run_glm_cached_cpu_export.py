"""Run the existing cached CPU exporter with bound inputs and durable PB progress."""
from __future__ import annotations
import argparse, contextlib, hashlib, importlib.util, json, os, runpy, shutil, sys, threading
from pathlib import Path

INPUTS = ('assignment', 'pact_result', 'plan', 'selected_manifest', 'selected_manifest_receipt',
          'hessian', 'input_scales', 'priced_inputs', 'research_selected_moe')


def read_bound(bound):
    path = Path(bound['path'])
    if not path.is_absolute() or path.is_symlink():
        raise ValueError('export inputs must be absolute regular files')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != bound['sha256']:
        raise ValueError('export input digest differs: '+str(path))
    return raw


def fsync_path(path):
    fd=os.open(path,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


class DurableExportProgress:
    """Observation only: existing cache and exporter own every byte and check."""
    def __init__(self, directory, output, expected_source, commit):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        self.output=Path(output).resolve();self.expected_source=expected_source
        self.commit=commit;self.lock=threading.Lock();self.seen=set();self.count=0
        self.events=self.directory/'durable-events.jsonl'

    def completed(self, phase, key, evidence):
        with self.lock:
            if (phase,key) in self.seen:return
            row={'phase':phase,'key':key,'units_completed':self.count+1,**evidence}
            with self.events.open('a') as handle:
                handle.write(json.dumps(row,sort_keys=True)+'\n');handle.flush();os.fsync(handle.fileno())
            fsync_path(self.directory)
            self.seen.add((phase,key));self.count+=1
            self.commit(self.count,phase)

    @contextlib.contextmanager
    def install(self, exporter):
        from tessera.source_digest_cache import SourceDigestCache
        original_hash=SourceDigestCache.sha256;original_save=exporter.save_serving_shard
        def digest(cache,path):
            value=original_hash(cache,path)
            if self.expected_source.get(Path(path).name)!=value:
                raise ValueError('current source differs from selected manifest: '+str(path))
            # The owner has returned either an authenticated fenced digest or
            # a complete before/after-checked read. Preserve its truthful mode.
            fsync_path(cache.directory)
            self.completed('source_verify',Path(path).name,{'sha256':value,'cache':cache.receipt()})
            return value
        def save(payload,path):
            path=Path(path)
            if path.resolve().parent!=self.output:
                raise ValueError('export attempted to publish outside its owned output')
            original_save(payload,path)
            fsync_path(path);fsync_path(path.parent)
            self.completed('export_shards',path.name,{'bytes':path.stat().st_size})
        SourceDigestCache.sha256=digest;exporter.save_serving_shard=save
        try:yield
        finally:SourceDigestCache.sha256=original_hash;exporter.save_serving_shard=original_save

    def finish(self):
        paths=[self.output/name for name in ('config.json','model.safetensors.index.json','tessera_serving_manifest.json')]
        for path in paths:fsync_path(path)
        fsync_path(self.output)
        self.completed('publish','complete-metadata',{'metadata':{
            path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings',required=True);parser.add_argument('--bindings-sha256',required=True)
    args=parser.parse_args(argv)
    if not os.environ.get('PRISMABUILD_ACTION_PROGRESS_HELPER'):
        raise RuntimeError('full export requires an admitted PB semantic-progress action')
    doc=json.loads(read_bound({'path':args.bindings,'sha256':args.bindings_sha256}))
    if doc.get('schema')!='prismaquant.glm_cached_cpu_export_bindings.v1':raise ValueError('unknown export bindings')
    if set(doc['inputs'])!=set(INPUTS):raise ValueError('actual allocation/PACT/export bindings are incomplete')
    inputs=doc['inputs'];raw={name:read_bound(value) for name,value in inputs.items()}
    selected=json.loads(raw['selected_manifest']);receipt=json.loads(raw['selected_manifest_receipt'])
    if (receipt.get('schema')!='prismaquant.tessera_selected_cache_handoff.v1'
        or receipt.get('manifest_sha256')!=inputs['selected_manifest']['sha256']
        or receipt.get('assignment_sha256')!=inputs['assignment']['sha256']):
        raise ValueError('selected manifest is not bound to the actual allocator assignment')
    if selected.get('schema')!='tessera.cached_units.v2':raise ValueError('full mixed export requires rooted selected manifest')
    if selected.get('served_activation_policy') is None:raise ValueError('full A4 selection lacks served policy')
    expected_source=selected['source']['files'];del raw,selected
    source=Path(doc['source']);output=Path(doc['output']);cache=Path(doc['source_digest_cache'])
    if output.exists():raise ValueError('full export requires a fresh output directory')
    output.parent.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output.parent).free < int(doc['required_free_bytes']):
        raise ValueError('output filesystem lacks the reviewed artifact plus staging space')
    if len(os.sched_getaffinity(0))<doc['intake_threads']+1:raise ValueError('intake exceeds admitted aggregate CPU affinity')
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('glm_cpu_exporter',root/'experiments/export_tessera_serving.py')
    exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
    def forbidden(*a,**k):raise RuntimeError('reuse-only export attempted encoding')
    exporter.encode_linear_planes=forbidden
    commit=runpy.run_path(os.environ['PRISMABUILD_ACTION_PROGRESS_HELPER'])['commit']
    progress=DurableExportProgress(doc['progress_directory'],output,expected_source,commit)
    sys.argv=['export',str(source),str(output),'--device','cpu','--cached-units',inputs['selected_manifest']['path'],
        '--plan-json',inputs['plan']['path'],'--hessian',inputs['hessian']['path'],
        '--input-scales',inputs['input_scales']['path'],'--priced-inputs',inputs['priced_inputs']['path'],
        '--priced-inputs-sha256',inputs['priced_inputs']['sha256'],
        '--research-selected-moe-json',inputs['research_selected_moe']['path'],
        '--source-digest-cache',str(cache),'--cached-hessian-identity','committed',
        '--cached-intake-threads',str(doc['intake_threads']),
        '--cached-intake-window-bytes',str(doc['intake_window_bytes'])]
    with progress.install(exporter):exporter.main()
    for binding in inputs.values():read_bound(binding)  # refuse input drift before final receipt
    progress.finish()
    print(json.dumps({'status':'exported','output':str(output),'durable_units':progress.count,
        'assignment':inputs['assignment'],'pact_result':inputs['pact_result'],
        'device':'cpu','encoded_units':0,'serving_qualified':False},sort_keys=True))
    return 0

if __name__=='__main__':raise SystemExit(main())
