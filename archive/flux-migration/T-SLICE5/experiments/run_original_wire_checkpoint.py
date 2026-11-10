"""PB-owned CPU export with canonical plugin and actual image attestation."""
import json
import os
from pathlib import Path
import subprocess
import uuid

ROOT=Path('/mnt/shared/tessera-clean-runtime-20260907/original-wire-layer2')
BASE='vllm/vllm-openai@sha256:4e31c581716a5cb9ef31eddb0a425842b75cab07d5cd63fb9572e69ae8794c33'
ROOT.mkdir(exist_ok=False)
inspected=json.loads(subprocess.check_output(['docker','image','inspect',BASE]))[0]
assert BASE in inspected.get('RepoDigests',[])
image_path=ROOT/'launcher-image-inspect.json';image_path.write_text(json.dumps(inspected,indent=2))
cidfile=ROOT/'container.cid'
cmd=['docker','run','--cidfile',str(cidfile),'--name','tessera-original-wire-export-'+uuid.uuid4().hex,'--network','none','--volume',f'{Path.cwd()}:/control:ro','--volume','/mnt/shared:/mnt/shared','--env','OMP_NUM_THREADS','--env','MKL_NUM_THREADS','--env','OPENBLAS_NUM_THREADS','--entrypoint','python3',inspected['Id'],'/mnt/shared/tessera-clean-runtime-20260907/control-854e672/per_job_install.py','--evidence-dir',str(ROOT/'plugin-install'),'--launcher-image-id',inspected['Id'],'--launcher-image-inspect',str(image_path),'--','python3','/control/experiments/original_wire_checkpoint.py']
run=subprocess.run(cmd)
cid=cidfile.read_text().strip();container=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert container['Id']==cid and container['Config']['Labels']['prismabuild.action']==os.environ['PRISMABUILD_CONTAINER_OWNER']
assert container['Image']==inspected['Id'] and not container['State']['Running']
(ROOT/'container-inspect.json').write_text(json.dumps(container,indent=2))
assert run.returncode==0 and container['State']['ExitCode']==0
