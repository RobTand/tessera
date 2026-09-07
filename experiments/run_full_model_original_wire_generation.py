"""Finite PB-owned all-original reference service, with trusted local observers."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

BASE = 'vllm/vllm-openai@sha256:4e31c581716a5cb9ef31eddb0a425842b75cab07d5cd63fb9572e69ae8794c33'
ROOT = Path('/mnt/shared/tessera-clean-runtime-20260907/full-model-original-r1024/functional-01')
ROOT.mkdir(exist_ok=False)
image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', BASE]))[0]
assert BASE in image.get('RepoDigests', [])
image_path = ROOT / 'launcher-image-inspect.json'
image_path.write_text(json.dumps(image, indent=2))
cidfile = ROOT / 'container.cid'
environment = {'TESSERA_SERVE_MODE': 'resident', 'TESSERA_ROUTE_TRACE': str(ROOT / 'route-trace.json'),
    'VLLM_ALLOW_INSECURE_SERIALIZATION': '1',  # observer-only trusted local RPC transport
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'VLLM_NO_USAGE_STATS': '1',
    'TRITON_CACHE_DIR': '/tmp/tessera-triton', 'FLASHINFER_WORKSPACE_BASE': '/tmp/tessera-flashinfer',
    'XDG_CACHE_HOME': '/tmp/tessera-cache', 'TORCH_EXTENSIONS_DIR': '/tmp/tessera-extensions',
    'VLLM_CONFIG_ROOT': '/tmp/tessera-vllm-config', 'VLLM_CACHE_ROOT': '/tmp/tessera-vllm-cache',
    'PYTHONPATH': '/control/experiments:/mnt/shared/tessera-clean-runtime-20260907/original-wire-layer2/controls'}
cmd = ['docker', 'run', '--gpus', 'all', '--ipc', 'host', '--cidfile', str(cidfile),
       '--name', 'tessera-full-original-functional-' + uuid.uuid4().hex, '--network', 'none',
       '--volume', f'{Path.cwd()}:/control:ro', '--volume', '/mnt/shared:/mnt/shared', '--workdir', '/control',
       '--env', 'OMP_NUM_THREADS', '--env', 'MKL_NUM_THREADS', '--env', 'OPENBLAS_NUM_THREADS']
for key, value in environment.items():
    cmd += ['--env', key + '=' + value]
cmd += ['--entrypoint', 'python3', image['Id'],
        '/mnt/shared/tessera-clean-runtime-20260907/control-854e672/per_job_install.py',
        '--evidence-dir', str(ROOT / 'plugin-install'), '--launcher-image-id', image['Id'],
        '--launcher-image-inspect', str(image_path), '--', 'python3',
        '/control/experiments/full_model_original_wire_generation.py', str(ROOT)]
started = time.time()
(ROOT / 'launch.json').write_text(json.dumps({'argv': cmd, 'environment': environment,
    'started_epoch': started, 'affinity': sorted(os.sched_getaffinity(0))}, indent=2))
run = subprocess.run(cmd)
cid = cidfile.read_text().strip()
container = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
assert container['Id'] == cid and container['Image'] == image['Id']
assert container['Config']['Labels']['prismabuild.action'] == os.environ['PRISMABUILD_CONTAINER_OWNER']
assert not container['State']['Running']
(ROOT / 'container-inspect.json').write_text(json.dumps(container, indent=2))
subprocess.run(['docker', 'rm', cid], check=True)
record = {'started_epoch': started, 'finished_epoch': time.time(), 'returncode': run.returncode,
          'container_exit_code': container['State']['ExitCode'], 'container_id': cid,
          'container_state': container['State'], 'container_removed': True,
          'generation_proof_exists': (ROOT / 'generation-proof.json').is_file()}
path = ROOT / 'launcher-result.json'
path.write_text(json.dumps(record, indent=2))
for path in (ROOT / 'launch.json', ROOT / 'container-inspect.json', path):
    print(json.dumps({'artifact': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'bytes': path.stat().st_size}), flush=True)
assert run.returncode == 0 and record['container_exit_code'] == 0 and record['generation_proof_exists']
