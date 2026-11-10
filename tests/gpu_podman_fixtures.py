"""Controlled runtime inputs for the GPU launcher and foreground wrappers."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'tools/_flux_gpu_jobspec.py'
IMAGE = 'example.invalid/cuda@sha256:' + 'a' * 64


def controlled_runtime(tmp_path, *, hold=False, affinity=None, survives_rm=False, create_failure=False):
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
 if os.environ.get('CREATE_FAILURE')=='1': sys.exit(17)
 (p/'created').touch()
 pathlib.Path(a[a.index('--cidfile')+1]).write_text('b'*64)
 print('b'*64)
elif a[0]=='start':
 (p/'ready').touch()
 if os.environ.get('HOLD')=='1': threading.Event().wait()
 # The fake drives a known observable container affinity, not a verdict.
 mount=next(json.loads(l) for l in reversed((p/'calls').read_text().splitlines()) if json.loads(l)[0]=='create')
 scratch=mount[mount.index('--volume')+1].split(':')[0]
 mask=os.environ.get('CONTAINER_AFFINITY', ','.join(map(str,sorted(os.sched_getaffinity(0)))))
 pathlib.Path(scratch,'container-status').write_text('Cpus_allowed_list:\\t'+mask+'\\n')
elif a[0]=='rm':
 if os.environ.get('SURVIVES_RM')!='1': (p/'created').unlink(missing_ok=True)
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
           'HOLD': str(int(hold)), 'SURVIVES_RM': str(int(survives_rm)),
           'CREATE_FAILURE': str(int(create_failure))}
    if affinity is not None:
        env['CONTAINER_AFFINITY'] = affinity
    out = tmp_path / 'out'
    command = [sys.executable, str(LAUNCHER), 'run', '--out', str(out),
               '--scratch-root', str(tmp_path / 'scratch'), '--',
               '--entrypoint', '/bin/sh', IMAGE, '-c', 'exit 0']
    return command, env, out


def controlled_submission(tmp_path, monkeypatch, *, evidence='ok'):
    import runpy
    module = runpy.run_path(str(LAUNCHER))
    _, env, out = controlled_runtime(tmp_path)
    binary = tmp_path / 'bin'
    flux = binary / 'flux'
    flux.write_text('''#!/usr/bin/env python3
import json,os,pathlib,subprocess,sys
a=sys.argv[1:]; p=pathlib.Path(os.environ['CONTROL']); case=os.environ['EVIDENCE']
with (p/'flux-calls').open('a') as f: f.write(json.dumps(a)+'\\n')
if a[:2]==['resource','list']: print('free sparky sparklina dl380g10')
elif a[0]=='submit':
 if case=='unknown': print('unrecognized submission'); sys.exit(1)
 if case=='payload':
  rc=subprocess.run(a[a.index('--')+1:]).returncode
  (p/'payload-exit').write_text(str(rc))
 print('f123')
elif a[:2]==['job','status']:
 status=int((p/'payload-exit').read_text())*256 if case=='payload' else (256 if case=='waitstatus' else 0)
 print(json.dumps({'waitstatus':status,'exception_occurred':case=='exception'}))
elif a[:2]==['job','eventlog']:
 print(json.dumps({'name':'finish','context':{'status':0}}))
 if case!='clean': print(json.dumps({'name':'clean'}))
elif a[:2]==['job','info']:
 if case=='unreadable': print('not JSON'); sys.exit(1)
 print(json.dumps({'execution':{'R_lite':[{'rank':'1','children':{'core':'0-1' if case=='cores' else '0','gpu':'0-1' if case=='gpu' else '0'}}],'nodelist':['sparky']}}))
''')
    ssh = binary / 'ssh'
    ssh.write_text('#!/bin/sh\ncase "$*" in *query-gpu*) echo 0;; esac\n')
    ssh.chmod(0o755)
    for key in ('PATH', 'CONTROL'):
        monkeypatch.setenv(key, env[key])
    monkeypatch.setenv('EVIDENCE', evidence)
    argv = ['flux', 'submit', '--env=-*', '--resources-shape=slot/[core=1;memory=8;gpu=1]',
            '-n1', '-t', '10m', '--', '/bin/true']
    return module, argv, out
