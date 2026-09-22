"""Read-only geometry and host inventory; no payload hashes or model export."""
import argparse, collections, glob, json, math, os, platform, shutil, struct, subprocess
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
source=Path('/mnt/shared/models/GLM-5.3-Flash-BF16')
census_path=Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/activation-runtime-allocation-20260911/extension-r1024-02/workspace/census.json')
census=json.loads(census_path.read_text()); index=json.loads((source/'model.safetensors.index.json').read_text())
shards=[];tensors=[]
for name in sorted(set(index['weight_map'].values())):
 path=source/name
 with path.open('rb') as handle:
  n=struct.unpack('<Q',handle.read(8))[0];header=json.loads(handle.read(n))
 shards.append({'name':name,'file_bytes':path.stat().st_size,'payload_bytes':sum(v['data_offsets'][1]-v['data_offsets'][0] for k,v in header.items() if k!='__metadata__')})
 for key,value in header.items():
  if key=='__metadata__':continue
  tensors.append({'name':key,'shard':name,'shape':value['shape'],'dtype':value['dtype'],'bytes':value['data_offsets'][1]-value['data_offsets'][0]})
units=[{'name':name,'shape':shape,'bf16_bytes':math.prod(shape)*2,'hessian_f32_bytes':shape[1]**2*4} for name,shape in census['unit_shapes'].items()]
href=Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/activation-runtime-allocation-20260911/extension-r1024-02/workspace/merged/cache/hessian_capture.references.json')
hdoc=json.loads(href.read_text())
filesystems={}
for spelling in ['/mnt/shared','/storage_pool/shared','/home/rob','/tmp','/mnt/ssd','/mnt/nvme','/mnt/local']:
 path=Path(spelling)
 if path.exists():
  usage=shutil.disk_usage(path);filesystems[spelling]={'total':usage.total,'used':usage.used,'free':usage.free,'resolved':str(path.resolve())}
venvs=[]
for pattern in ['/home/rob/venvs/*/bin/python','/home/rob/*venv*/bin/python','/mnt/shared/venvs/*/bin/python']:
 venvs.extend(glob.glob(pattern))
result={'schema':'prismaquant.glm_cpu_export_inventory.v1','hostname':platform.node(),'machine':platform.machine(),
 'cpu_count':os.cpu_count(),'affinity':sorted(os.sched_getaffinity(0)),'meminfo':Path('/proc/meminfo').read_text(),
 'mounts':Path('/proc/mounts').read_text(),'filesystems':filesystems,'venv_python_paths':sorted(set(venvs)),
 'lsblk':subprocess.run(['lsblk','-b','-J','-o','NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS'],capture_output=True,text=True).stdout,
 'source':str(source),'source_index_total_bytes':index.get('metadata',{}).get('total_size'),
 'source_file_bytes':sum(row['file_bytes'] for row in shards),'source_shards':shards,
 'source_tensors':len(tensors),'largest_source_tensors':sorted(tensors,key=lambda x:x['bytes'],reverse=True)[:24],
 'source_dtype_counts':dict(collections.Counter(row['dtype'] for row in tensors)),
 'census':str(census_path),'census_nsamples':census['nsamples'],'census_seqlen':census['seqlen'],
 'units':len(units),'unit_shape_counts':dict(collections.Counter(str(row['shape']) for row in units)),
 'largest_units':sorted(units,key=lambda x:x['bf16_bytes'],reverse=True)[:12],
 'largest_hessian_units':sorted(units,key=lambda x:x['hessian_f32_bytes'],reverse=True)[:12],
 'hessian_reference':str(href),'hessian_reference_file_bytes':href.stat().st_size,
 'hessian_reference_keys':list(hdoc),'hessian_reference_preview':{k:v for k,v in hdoc.items() if k not in ('hessians','counts')},
 'hessian_entry_count':len(hdoc.get('hessians',{})),
 'hessian_entry_example':next(iter(hdoc.get('hessians',{}).items()),None)}
Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'read_only_geometry_collected','output':args.out,'host':result['hostname'],
 'source_bytes':result['source_file_bytes'],'max_shard_bytes':max(x['file_bytes'] for x in shards),
 'units':len(units),'hessian_entries':result['hessian_entry_count'],'filesystems':filesystems}))
