"""Bind each phase to one live Flux instance. Run the CLI only after review."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

TOOLS = '/mnt/shared/flux-migration/lead-1114/tools'


def client(name):
    if name == 'fv1':
        return [TOOLS + '/fx.sh']
    if name not in ('fv1e-sparky', 'fv1e-sparklina'):
        raise ValueError('unknown phase instance')
    return [TOOLS + '/fxe.sh', name.removeprefix('fv1e-')]


def query(command, *args):
    return subprocess.check_output([*command, *args], text=True, timeout=30).strip()


def observe(name, command=None):
    command = command or client(name)
    rpc = ['python', '-c', 'import flux,json;f=flux.Flux();print(json.dumps({k:f.rpc("attr.get",{"name":k},nodeid=0).get()["value"] for k in ("broker.starttime","local-uri")}))']
    before = json.loads(query(command, *rpc))
    config = json.dumps(json.loads(query(command, 'config', 'get')), sort_keys=True, separators=(',', ':'))
    if json.loads(query(command, *rpc)) != before or not all(before.values()):
        raise ValueError('phase instance changed during observation')
    return {'name': name, 'config_sha256': hashlib.sha256(config.encode()).hexdigest(),
            'broker_starttime': before['broker.starttime'], 'uri': before['local-uri']}


def save(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, sort_keys=True) + '\n')
    temporary.replace(path)


def submit(job, args):
    selection = job / 'instance-name.txt'
    name = selection.read_text().strip() if selection.exists() else 'fv1'
    if name != 'fv1' and (job.name != 'prepare' or json.loads((job.parent/'campaign.json').read_text())['canary']):
        raise ValueError('only counted prepare can use fv1e')
    if args[0] != 'submit' or (job/'jobid').exists():
        raise ValueError('phase submission already exists or command is not submit')
    bound, command = observe(name), client(name)
    urgency = next((a.split('=')[1] for a in args if a.startswith('--urgency=')), '16')
    args = [a for a in args if not a.startswith('--urgency=')]
    jobid = query(command, 'job', 'id', '--to=f58plain', query(command, args[0], '--urgency=0', *args[1:]))
    (job/'jobid').write_text(jobid + '\n')
    save(job/'instance.json', {**bound, 'job_id': jobid})
    if observe(name) != bound:
        raise ValueError('phase instance changed at submit; job remains held')
    query(command, 'job', 'urgency', jobid, urgency)


def check(job):
    prefix = Path('/mnt/shared/flux-migration/INSTANCE').read_text().split()[0]
    command = ['env', f'PATH={prefix}/install/bin:{prefix}/deps/bin:/usr/bin:/bin',
               f'LD_LIBRARY_PATH={prefix}/install/lib:{prefix}/deps/lib', f'{prefix}/install/bin/flux']
    bound = json.loads((job/'instance.json').read_text())
    live = {**observe(bound['name'], command),
            'job_id': query(command, 'job', 'id', '--to=f58plain', os.environ['FLUX_JOB_ID'])}
    if live != bound:
        raise ValueError('live phase instance or job differs from submission')
    return bound


def bind(row, phase, identity, canary=False):
    jobs = row.setdefault('phase_jobs', {})
    if (phase in jobs and jobs[phase] != identity) or ((canary or phase != 'prepare') and identity['name'] != 'fv1'):
        raise ValueError('one phase cannot mix instances or use this placement')
    jobs[phase] = identity


if __name__ == '__main__':
    action, directory, *arguments = sys.argv[1:]
    if action == 'submit':
        submit(Path(directory), arguments)
    elif action == 'check':
        print(json.dumps(check(Path(directory))))
    else:
        raise ValueError('unknown instance action')
