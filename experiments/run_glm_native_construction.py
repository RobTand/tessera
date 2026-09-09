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
    parser.add_argument('--packed-request', type=Path)
    parser.add_argument('--role', choices=('gate_proj', 'up_proj', 'down_proj'))
    parser.add_argument('--q256', type=int)
    parser.add_argument('--timeout-seconds',type=int)
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
    if args.packed_request:
        assert args.stage == 'control'
        env['VLLM_DISABLE_SHARED_EXPERTS_STREAM'] = '1'
    if args.stage == 'encode':
        env['PRISMABUILD_CONTAINER_OWNER'] = os.environ['PRISMABUILD_CONTAINER_OWNER']
    packed = json.loads(args.packed_request.read_text()) if args.packed_request else {}
    network = 'host' if packed.get('tp_size',1) == 2 else 'none'
    if network == 'host':
        assert args.timeout_seconds is not None and 1 <= args.timeout_seconds <= 600
        env['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
        fabric = packed['distributed'].get('ipv4_tcp_fabric')
        if fabric is not None:
            import ipaddress
            assert len(fabric['rank_addresses']) == 2 and fabric['interface']
            rank = packed['distributed']['rank']
            addresses = [str(ipaddress.IPv4Address(value)) for value in fabric['rank_addresses']]
            observed = json.loads(subprocess.check_output(['ip','-j','-4','address','show','dev',fabric['interface']]))
            assert addresses[rank] in {row['local'] for dev in observed for row in dev['addr_info']}
            env.update(GLOO_SOCKET_IFNAME=fabric['interface'],
                NCCL_SOCKET_IFNAME='='+fabric['interface'],NCCL_SOCKET_FAMILY='AF_INET',
                NCCL_IB_DISABLE='1',VLLM_HOST_IP=addresses[rank])
    command = ['docker', 'run', '--gpus', 'all', '--network', network,
        '--cpuset-cpus', ','.join(map(str, affinity)), '--memory', f'{args.memory_gib}g',
        '--memory-swap', f'{args.memory_gib}g', '--shm-size', '1g', '--cidfile', str(cidfile),
        '--name', 'tessera-glm-construction-' + uuid.uuid4().hex,
        '--volume', f'{Path.cwd()}:/work:ro', '--volume', '/mnt/shared:/mnt/shared',
        '--workdir', '/work']
    for key, value in env.items():
        command += ['--env', key + '=' + value]
    entry = ('glm_native_construction.py' if args.stage in ('construction', 'resident', 'streamed')
             else 'glm_repeated_expert_control.py')
    if args.packed_request:
        entry = 'glm_packed_moe_control.py'
    command += ['--entrypoint', 'python3', inspected['Id'],
        '/work/experiments/' + entry, '--request', str(args.request),
        '--out', str(root), '--stage', args.stage]
    if args.construction:
        command += ['--construction', str(args.construction)]
    for name in ('control_request', 'producer_manifest', 'selected_request', 'packed_request', 'role', 'q256'):
        value = getattr(args, name)
        if value is not None:
            command += ['--' + name.replace('_', '-'), str(value)]
    controls = [Path('experiments/glm_native_construction.py'), Path(__file__),
                Path('tools/tessera_construction_census.py')]
    if entry != 'glm_native_construction.py':
        controls.append(Path('experiments') / entry)
    if args.selected_request:
        controls.append(Path('experiments/glm_selected_expert_control.py'))
    if args.packed_request:
        controls.append(args.packed_request)
        controls.append(Path('experiments/glm_packed_intake_observer.py'))
        if packed.get('tp_size',1) == 2:
            controls.append(Path('experiments/glm_packed_tp2_control.py'))
    before = {str(p): digest(p) for p in controls}
    start = time.time()
    (root / 'launch.json').write_text(json.dumps({'command': command, 'environment': env,
        'request_sha256': digest(args.request), 'control_sha256': before,
        'affinity': affinity, 'memory_limit_gib': args.memory_gib,
        'execution_authority': ('PrismaBuild admitted producer' if args.stage == 'encode'
                                else 'explicit user vLLM exemption 2026-09-07'),
        'timeout_seconds':args.timeout_seconds,'started_epoch': start}, indent=2) + '\n')
    timed_out = False
    with (root / 'container.log').open('w') as log:
        try:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                                    timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            result = subprocess.CompletedProcess(command,124)
    if not cidfile.exists():
        (root/'launcher-result.json').write_text(json.dumps({'returncode':result.returncode,
            'timed_out':timed_out,'container_id_recorded':False,'finished_epoch':time.time()},indent=2)+'\n')
        return result.returncode or 125
    cid = cidfile.read_text().strip()
    if timed_out:
        # The CID was recorded by this launch. Stop only that owned container;
        # killing the docker client at timeout does not stop its payload.
        try:
            subprocess.run(['docker','stop','--time','10',cid],check=True,timeout=20)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError):
            subprocess.run(['docker','kill',cid],check=True,timeout=10)
    container = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
    assert container['Id'] == cid and container['Image'] == inspected['Id']
    assert not container['State']['Running']
    (root / 'container-inspect.json').write_text(json.dumps(container, indent=2) + '\n')
    subprocess.run(['docker', 'rm', cid], check=True)
    after = {str(p): digest(p) for p in controls}
    record = {'started_epoch': start, 'finished_epoch': time.time(),
        'returncode': result.returncode,'timed_out':timed_out,'timeout_seconds':args.timeout_seconds,
        'container_exit_code': container['State']['ExitCode'],
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
