"""Direct finite vLLM qualification under Rob's explicit vLLM exemption."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('construction', 'resident', 'streamed', 'encode', 'control'), required=True)
    parser.add_argument('--construction', type=Path)
    parser.add_argument('--control-request', type=Path)
    parser.add_argument('--producer-manifest', type=Path)
    parser.add_argument('--selected-request', type=Path)
    parser.add_argument('--role', choices=('gate_proj', 'up_proj', 'down_proj'))
    parser.add_argument('--q256', type=int)
    parser.add_argument('--cpus', type=int, default=4)
    parser.add_argument('--memory-gib', type=int, default=12)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    root = args.out
    root.mkdir(parents=True, exist_ok=False)
    image = request['runtime_image']
    inspected = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image]))[0]
    assert image in inspected.get('RepoDigests', [])
    (root / 'image-inspect.json').write_text(json.dumps(inspected, indent=2) + '\n')
    affinity = sorted(os.sched_getaffinity(0))[:args.cpus]
    assert len(affinity) == args.cpus and args.cpus > 0
    cidfile = root / 'container.cid'
    env = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'MAX_JOBS': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': '/work',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'VLLM_NO_USAGE_STATS': '1',
        'TESSERA_SERVE_MODE': 'streamed' if args.stage == 'streamed' else 'resident',
        'TRITON_CACHE_DIR': '/tmp/glm-triton', 'TORCH_EXTENSIONS_DIR': '/tmp/glm-extensions',
        'XDG_CACHE_HOME': '/tmp/glm-cache', 'VLLM_CONFIG_ROOT': '/tmp/glm-config',
        'VLLM_CACHE_ROOT': '/tmp/glm-vllm-cache', 'FLASHINFER_WORKSPACE_BASE': '/tmp/glm-flashinfer'}
    if args.stage == 'encode':
        env['PRISMABUILD_CONTAINER_OWNER'] = os.environ['PRISMABUILD_CONTAINER_OWNER']
    command = ['docker', 'run', '--gpus', 'all', '--network', 'none',
        '--cpuset-cpus', ','.join(map(str, affinity)), '--memory', f'{args.memory_gib}g',
        '--memory-swap', f'{args.memory_gib}g', '--shm-size', '1g', '--cidfile', str(cidfile),
        '--name', 'tessera-glm-construction-' + uuid.uuid4().hex,
        '--volume', f'{Path.cwd()}:/work:ro', '--volume', '/mnt/shared:/mnt/shared',
        '--workdir', '/work']
    for key, value in env.items():
        command += ['--env', key + '=' + value]
    entry = ('glm_native_construction.py' if args.stage in ('construction', 'resident', 'streamed')
             else 'glm_repeated_expert_control.py')
    command += ['--entrypoint', 'python3', inspected['Id'],
        '/work/experiments/' + entry, '--request', str(args.request),
        '--out', str(root), '--stage', args.stage]
    if args.construction:
        command += ['--construction', str(args.construction)]
    for name in ('control_request', 'producer_manifest', 'selected_request', 'role', 'q256'):
        value = getattr(args, name)
        if value is not None:
            command += ['--' + name.replace('_', '-'), str(value)]
    controls = [Path('experiments/glm_native_construction.py'), Path(__file__),
                Path('tools/tessera_construction_census.py')]
    if entry != 'glm_native_construction.py':
        controls.append(Path('experiments') / entry)
    if args.selected_request:
        controls.append(Path('experiments/glm_selected_expert_control.py'))
    before = {str(p): digest(p) for p in controls}
    start = time.time()
    (root / 'launch.json').write_text(json.dumps({'command': command, 'environment': env,
        'request_sha256': digest(args.request), 'control_sha256': before,
        'affinity': affinity, 'memory_limit_gib': args.memory_gib,
        'execution_authority': ('PrismaBuild admitted producer' if args.stage == 'encode'
                                else 'explicit user vLLM exemption 2026-09-07'),
        'started_epoch': start}, indent=2) + '\n')
    with (root / 'container.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    cid = cidfile.read_text().strip()
    container = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
    assert container['Id'] == cid and container['Image'] == inspected['Id']
    assert not container['State']['Running']
    (root / 'container-inspect.json').write_text(json.dumps(container, indent=2) + '\n')
    subprocess.run(['docker', 'rm', cid], check=True)
    after = {str(p): digest(p) for p in controls}
    record = {'started_epoch': start, 'finished_epoch': time.time(),
        'returncode': result.returncode, 'container_exit_code': container['State']['ExitCode'],
        'oom_killed': container['State']['OOMKilled'], 'container_removed': True,
        'controls_unchanged': before == after}
    (root / 'launcher-result.json').write_text(json.dumps(record, indent=2) + '\n')
    for path in sorted(root.iterdir()):
        if path.is_file():
            print(json.dumps({'artifact': str(path), 'sha256': digest(path),
                              'bytes': path.stat().st_size}), flush=True)
    print(json.dumps(record), flush=True)
    assert before == after, 'Qualification controls changed during execution'
    return result.returncode or container['State']['ExitCode']


if __name__ == '__main__':
    raise SystemExit(main())
