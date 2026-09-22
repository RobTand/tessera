"""Validate actual export bindings and emit a reviewable PB argv; never submit."""
import argparse, hashlib, json
from pathlib import Path
from run_glm_cached_cpu_export import INPUTS, read_bound

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bindings',required=True);parser.add_argument('--bindings-sha256',required=True)
parser.add_argument('--checkout',required=True);parser.add_argument('--out',required=True)
args=parser.parse_args()
bound={'path':args.bindings,'sha256':args.bindings_sha256};doc=json.loads(read_bound(bound))
if doc.get('schema')!='prismaquant.glm_cached_cpu_export_bindings.v1' or set(doc.get('inputs',{}))!=set(INPUTS):
 raise ValueError('actual allocation/PACT/export bindings are incomplete')
for value in doc['inputs'].values():read_bound(value)
selected=json.loads(read_bound(doc['inputs']['selected_manifest']))
if selected.get('schema')!='tessera.cached_units.v2':raise ValueError('mixed selected manifest required')
wire_bytes=sum(record['blob_bytes'] for record in selected['units'].values())
if doc['intake_threads']!=7 or doc['intake_window_bytes']!=8<<30:raise ValueError('resource settings differ from reviewed row')
argv=['python3','/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py','--cwd',str(Path(args.checkout).resolve()),
 '--tag','dl380g10','--cpus','8','--demand','mem_gb=48','--priority','0',
 '--env','OMP_NUM_THREADS=1','--env','MKL_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1',
 '--env','PYTHONPATH=src','--env','PYTHONDONTWRITEBYTECODE=1',
 '--progress','source_verify=900','--progress','export_shards=600','--progress','publish=600','--',
 '/home/rob/venvs/pq-cpu312-tessera-4c384e60/bin/python','tools/run_glm_cached_cpu_export.py',
 '--bindings',args.bindings,'--bindings-sha256',args.bindings_sha256]
result={'schema':'prismaquant.reviewable_cpu_export_submission.v1','submitted':False,'argv':argv,
 'source':doc['source'],'output':doc['output'],'selected_wire_bytes':wire_bytes,
 'required_free_bytes':doc['required_free_bytes'],'binding':bound,
 'remaining_gate':'root review of actual PACT result, selected assignment and complete scientific scope'}
with Path(args.out).open('x') as handle:handle.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,sort_keys=True))
