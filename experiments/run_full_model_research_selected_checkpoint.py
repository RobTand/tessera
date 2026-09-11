"""PB-owned research-selected full-model export in the canonical stock image."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import uuid

BASE = 'vllm/vllm-openai@sha256:4e31c581716a5cb9ef31eddb0a425842b75cab07d5cd63fb9572e69ae8794c33'
# The 09-07 launcher ran the frozen per-job installer, which pip-installs Tessera
# from an archive pinned at 382a1a97 -- a tree with no tessera.moe_execution, so
# this branch's exporter cannot import under it. This job installs nothing and
# imports the checkout instead; checkout_runtime_identity records that identity
# and refuses if an installed distribution could shadow it.
IDENTITY = '/control/experiments/checkout_runtime_identity.py'

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--execution-json', required=True,
                help='path under /control, e.g. /control/experiments/research_selected_moe_lfm_tp1.json')
ap.add_argument('--layers', type=int, default=None,
                help='forwarded to the driver: plan only the first N body layers (smoke)')
ap.add_argument('--jit-root', type=Path, default=Path.home() / 'tessera-runs' / 'jit',
                help='box-local parent for this job JIT caches; must not be on the shared mount')
args = ap.parse_args()

ROOT = args.out.resolve()
ROOT.mkdir(parents=True, exist_ok=False)
inspected = json.loads(subprocess.check_output(['docker', 'image', 'inspect', BASE]))[0]
assert BASE in inspected.get('RepoDigests', [])
image_path = ROOT / 'launcher-image-inspect.json'
image_path.write_text(json.dumps(inspected, indent=2))
cidfile = ROOT / 'container.cid'
# The recorder drops to uid 1000 before exec, and the stock image leaves HOME=/root.
# Every JIT cache this encode fills expands a home-relative default: Triton writes
# $TRITON_CACHE_DIR or $HOME/.triton, and kernel_window_gemv reads
# $TORCH_EXTENSIONS_DIR or ~/tmp/torch-ext-gemv (never /tmp, which its docstring
# states). Unset, the first kernel compile is a mkdir denial under /root, which
# surfaces as a build failure rather than as a permission message.
#
# They are box-local and not under --out, which is the shared mount. A JIT cache on
# NFS is a separate recorded failure: a compile that takes a file-lock baton there
# can hang, and the denial it reports reads as an inspection failure rather than as
# a cache problem. Weights are shared; caches are local. The assertion is the check,
# because the natural thing to write is ROOT / name and that is the wrong answer.
jit_run = args.jit_root.resolve() / ('research-selected-' + uuid.uuid4().hex)
assert not str(jit_run).startswith('/mnt/shared'), (
    f'{jit_run} is on the shared mount. Pass --jit-root a box-local directory: a JIT '
    'cache on NFS can wedge on a build baton, and the kernels it holds are valid only '
    'for the box that compiled them.')
jit_run.mkdir(parents=True)
caches = {name: jit_run / name for name in ('home', 'triton-cache', 'torch-extensions')}
for path in caches.values():
    path.mkdir()
cmd = ['docker', 'run', '--gpus', 'all', '--ipc', 'host', '--cidfile', str(cidfile),
       '--name', 'tessera-research-selected-export-' + uuid.uuid4().hex, '--network', 'none',
       '--volume', f'{Path.cwd()}:/control:ro', '--volume', '/mnt/shared:/mnt/shared',
       '--volume', f'{jit_run}:{jit_run}',
       '--env', 'OMP_NUM_THREADS', '--env', 'MKL_NUM_THREADS', '--env', 'OPENBLAS_NUM_THREADS',
       '--env', f'HOME={caches["home"]}',
       '--env', f'TRITON_CACHE_DIR={caches["triton-cache"]}',
       '--env', f'TORCH_EXTENSIONS_DIR={caches["torch-extensions"]}',
       '--entrypoint', 'python3', inspected['Id'],
       IDENTITY, '--evidence-dir', str(ROOT / 'runtime-identity'), '--checkout', '/control',
       '--launcher-image-id', inspected['Id'], '--launcher-image-inspect', str(image_path),
       '--', 'python3', '/control/experiments/full_model_research_selected_checkpoint.py',
       '--out', str(ROOT / 'export'), '--execution-json', args.execution_json]
if args.layers is not None:
    cmd += ['--layers', str(args.layers)]
(ROOT / 'launch.json').write_text(json.dumps(
    {'argv': cmd, 'affinity': sorted(os.sched_getaffinity(0)),
     'jit_cache_dirs': {name: str(path) for name, path in caches.items()},
     'jit_root': str(args.jit_root.resolve())}, indent=2))
run = subprocess.run(cmd)
cid = cidfile.read_text().strip()
container = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
assert container['Id'] == cid and container['Config']['Labels']['prismabuild.action'] == os.environ['PRISMABUILD_CONTAINER_OWNER']
assert container['Image'] == inspected['Id'] and not container['State']['Running']
(ROOT / 'container-inspect.json').write_text(json.dumps(container, indent=2))
subprocess.run(['docker', 'rm', cid], check=True)
assert run.returncode == 0 and container['State']['ExitCode'] == 0
