"""Bounded real-wire/source/H CPU intake proof; no encode or full export."""
import argparse, concurrent.futures, hashlib, importlib.util, json, os, pickle, platform, resource, shutil, time
from pathlib import Path
import torch
from safetensors import safe_open
from tessera.cached_unit import CachedUnitIdentity, verify_cached_unit
from tessera.control import grid_for_name
from tessera.export import ActivationSource
from tessera.fused import pack_fused, parse_fused
from tessera.historical_producer import load_historical_producer
from tessera.unit_artifact import parse_unit_artifact

parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('glm_cpu_pack_exporter',root/'experiments/export_tessera_serving.py');exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
def forbidden(*a,**k):raise AssertionError('reuse-only CPU proof attempted encoding')
exporter.encode_linear_planes=forbidden
source=Path('/mnt/shared/models/GLM-5.3-Flash-BF16');index=json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
base=Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908')
merged=base/'activation-runtime-allocation-20260911/extension-r1024-02/workspace/merged'
hpath=merged/'cache/hessian_capture.references.json'
activation=ActivationSource.from_capture(hpath,ldlq_sigma=1.0,ldlq_block=32,refit_reach_floor=False)
packages={'old':(base/'identity-reseal-20260911/producer-source-d403cc5a31/src/tessera','0833671bbddbc3fb7186bdbed0a905ef248c23ffdf1aeee0c083322680d20f6b'),
          'added':(base/'identity-reseal-20260915/producer-source-c92826fa4/src/tessera','a4c9209437c7601d4f8cd3ab8ac1e7a2d2db33461a4245e0fbbbdf74a9d8de83')}
producers={key:load_historical_producer(path,seal) for key,(path,seal) in packages.items()}
identities={key:CachedUnitIdentity(lambda *a,_producer=p,**kw:exporter.cached_input_identity(_producer,*a,**kw),activation,mode='committed') for key,p in producers.items()}
pilot=Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/t4-reuse-20260922/pilot24-batch.json');tasks=[]
for q,fmt in [('model.language_model.layers.0.mlp.down_proj','TESSERA_BF16_K1_R1088'),('model.language_model.layers.10.mlp.experts.0.down_proj','TESSERA_E4M3_K1_R1024')]:
 part=merged/'cost.anchors.json.parts/units'/(hashlib.sha256(q.encode()).hexdigest()+'.pkl')
 envelope=pickle.loads(part.read_bytes());assert envelope['qname']==q and hashlib.sha256(envelope['payload']).hexdigest()==envelope['payload_sha256']
 record=pickle.loads(envelope['payload'])['wire_records'][fmt]
 tasks.append(('old',record,merged/'cache/wire'/record['file']))
for task in json.loads(pilot.read_text())['tasks']:
 cell=task['payload']['cell'];tasks.append(('added',cell['record'],Path(cell['wire'])))

def run(task):
 owner,record,path=task;identity=record['identity'];name=identity['unit'];unit=identity.get('projection');tensor=unit['source_tensor'] if unit else name+'.weight';before=path.stat();started=time.monotonic()
 with safe_open(str(source/index[tensor]),framework='pt') as handle:
  weight=handle.get_tensor(tensor)
  if unit:weight=exporter.packed_expert_weight(weight,unit)
  expected=identities[owner](weight,tensor,unit,grid_for_name(identity['recipe']['grid']),identity['recipe']['q256'])
 raw=path.read_bytes();producers[owner].verify(raw,record,expected)
 if unit:accepted,packed=exporter.pack_cached_expert_unit(raw,record,expected)
 else:
  accepted=verify_cached_unit(raw,record,expected);parse_unit_artifact(accepted.blob,device='cpu')
  packed=pack_fused([('down_proj',identity['source']['shape'][0],accepted.blob)])
 assert parse_fused(packed)[0].blob==raw
 after=path.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
 return {'unit':name,'producer':identity['encoder_source_sha256'],'wire_bytes':len(raw),'packed_bytes':len(packed),'wire_sha256':record['blob_sha256'],'elapsed_s':time.monotonic()-started,'source_shape':identity['source']['shape'],'device':str(weight.device)}
start=time.monotonic();cpu=time.process_time();results=[]
# Each producer establishes its actual source/H witness before parallel reuse.
results.append(run(tasks[0]));results.append(run(tasks[2]))
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
 results.extend(pool.map(run,[tasks[1],*tasks[3:]]))
fs={p:dict(zip(('total','used','free'),shutil.disk_usage(p))) for p in ('/stage','/stage/prewarm','/mnt/shared') if Path(p).exists()}
result={'schema':'prismaquant.real_glm_cpu_cached_pack_probe.v1','hostname':platform.node(),'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'affinity':sorted(os.sched_getaffinity(0)),
 'status':'passed','units':results,'total_wire_bytes':sum(r['wire_bytes'] for r in results),'wall_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 'hessian_identity':{key:value.record() for key,value in identities.items()},'storage':fs,'full_model_exported':False,'encoded_units':0}
Path(args.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('units','hessian_identity')}))
