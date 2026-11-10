"""Controlled runtime inputs for the GPU launcher and foreground wrappers."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'tools/_flux_gpu_jobspec.py'
IMAGE = 'example.invalid/cuda@sha256:' + 'a' * 64


def controlled_runtime(tmp_path, *, hold=False):
    binary = tmp_path / 'bin'
    binary.mkdir()
    podman = binary / 'podman'
    podman.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys, threading
p=pathlib.Path(os.environ['CONTROL'])
a=sys.argv[1:]
with (p/'calls').open('a') as f: f.write(json.dumps(a)+'\\n')
if a[:2]==['image','inspect']: print('sha256:'+'c'*64+'\\t'+json.dumps([a[2]]))
elif a[0]=='info': print(os.environ.get('ROOTLESS', 'true'))
elif a[0]=='create':
 (p/'created').touch()
 pathlib.Path(a[a.index('--cidfile')+1]).write_text('b'*64)
 print('b'*64)
elif a[0]=='start':
 (p/'ready').touch()
 if os.environ.get('HOLD')=='1': threading.Event().wait()
 # The fake drives a known observable container affinity, not a verdict.
 mount=next(json.loads(l) for l in reversed((p/'calls').read_text().splitlines()) if json.loads(l)[0]=='create')
 scratch=mount[mount.index('--volume')+1].split(':')[0]
 pathlib.Path(scratch,'container-status').write_text('Cpus_allowed_list:\\t'+','.join(map(str,sorted(os.sched_getaffinity(0))))+'\\n')
elif a[0]=='rm': (p/'created').unlink(missing_ok=True)
elif a[:2]==['container','exists']: sys.exit(int(not (p/'created').exists()))
''')
    podman.chmod(0o755)
    flux = binary / 'flux'
    flux.write_text('#!/bin/sh\nprintf \"%s\\n\" \"$*\" > \"$CONTROL/flux-id-call\"\necho f123\n')
    flux.chmod(0o755)
    findmnt = binary / 'findmnt'
    findmnt.write_text('#!/bin/sh\nprintf "%s\\n" "${FILESYSTEM:-ext4}"\n')
    findmnt.chmod(0o755)
    env = {**os.environ, 'PATH': f'{binary}:{os.environ["PATH"]}',
           'CONTROL': str(tmp_path), 'FLUX_JOB_ID': 'f123', 'CUDA_VISIBLE_DEVICES': '0',
           'HOLD': str(int(hold))}
    out = tmp_path / 'out'
    command = [sys.executable, str(LAUNCHER), 'run', '--out', str(out),
               '--scratch-root', str(tmp_path / 'scratch'), '--',
               '--entrypoint', '/bin/sh', IMAGE, '-c', 'exit 0']
    return command, env, out
