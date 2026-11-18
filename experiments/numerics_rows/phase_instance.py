"""Bind each phase to one live Flux instance. Run the CLI only after review."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

TOOLS = '/mnt/shared/flux-migration/lead-1114/tools'
INSTANCE = Path('/mnt/shared/flux-migration/INSTANCE')


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
    config = json.dumps(json.loads(query(command, 'python', '-c', 'import flux,json;print(json.dumps(flux.Flux().rpc("config.get",nodeid=0).get()))')), sort_keys=True, separators=(',', ':'))
    if json.loads(query(command, *rpc)) != before or not all(before.values()):
        raise ValueError('phase instance changed during observation')
    return {'name': name, 'config_sha256': hashlib.sha256(config.encode()).hexdigest(),
            'broker_starttime': before['broker.starttime'], 'uri': before['local-uri']}


def save(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, sort_keys=True) + '\n')
    temporary.replace(path)


def campaign(job):
    return next(parent / 'campaign.json' for parent in (job, *job.parents) if (parent / 'campaign.json').is_file())


def complete(directory):
    document = json.loads((directory / 'phase.json').read_text())
    if (directory.resolve() != directory or document.get('completion') is not True or document.get('phase') != 'prepare'
            or document.get('schema') != 'tessera.dense_row_phase.v2' or not (directory / 'completion.sha256').is_file()):
        raise ValueError('prepare is not complete or hashes differ')


def submit(job, args, phase=None):
    prefix = phase + '-' if phase else ''
    selection = job / (prefix + 'instance-name.txt')
    name = selection.read_text().strip() if selection.exists() else 'fv1'
    if name != 'fv1' and ((phase or job.name) not in ('prepare', 'compare') or json.loads(campaign(job).read_text())['canary']):
        raise ValueError('only counted prepare or compare can use fv1e')
    if args[0] != 'submit' or (job/(prefix + 'jobid')).exists():
        raise ValueError('phase submission already exists or command is not submit')
    bound, command = observe(name), client(name)
    urgency = next((a.split('=')[1] for a in args if a.startswith('--urgency=')), '16')
    args = [a for a in args if not a.startswith('--urgency=')]
    jobid = query(command, 'job', 'id', '--to=f58plain', query(command, args[0], '--urgency=0', *args[1:]))
    (job/(prefix + 'jobid')).write_text(jobid + '\n')
    save(job/(prefix + 'instance.json'), {**bound, 'job_id': jobid})
    if observe(name) != bound:
        raise ValueError('phase instance changed at submit; job remains held')
    query(command, 'job', 'urgency', jobid, urgency)


def check(job, phase=None):
    prefix = INSTANCE.read_text().split()[0]
    command = ['env', f'PATH={prefix}/install/bin:{prefix}/deps/bin:/usr/bin:/bin',
               f'LD_LIBRARY_PATH={prefix}/install/lib:{prefix}/deps/lib', f'{prefix}/install/bin/flux']
    bound = json.loads((job / ((phase + '-' if phase else '') + 'instance.json')).read_text())
    live = {**observe(bound['name'], command),
            'job_id': query(command, 'job', 'id', '--to=f58plain', os.environ['FLUX_JOB_ID'])}
    if live != bound:
        raise ValueError('live phase instance or job differs from submission')
    return bound


def bind(row, phase, identity, canary=False):
    jobs = row.setdefault('phase_jobs', {})
    if (phase in jobs and jobs[phase] != identity) or ((canary or phase not in ('prepare', 'compare')) and identity['name'] != 'fv1'):
        raise ValueError('one phase cannot mix instances or use this placement')
    jobs[phase] = identity


if __name__ == '__main__':
    action, directory, *arguments = sys.argv[1:]
    try:
        if action == 'submit':
            submit(Path(directory), arguments)
        elif action == 'complete':
            complete(Path(directory))
        elif action == 'check':
            print(json.dumps(check(Path(directory), *arguments)))
        else:
            raise ValueError('unknown instance action')
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(3) from error
