"""Render GPU submissions and own rootless Podman containers inside Flux jobs.

The default CLI renders a stock-shape request. ``run --out DIR -- OPTIONS
IMAGE COMMAND`` executes inside an admitted job and records lifetime evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import uuid
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _flux_shape import THREADS, duration_seconds, resource_shape  # noqa: E402

BASE_ENV = {"PATH": "/usr/bin:/bin", "HOME": "/home/rob", "FLUX_F58_FORCE_ASCII": "1", **THREADS}


def render(command, *, cores, memory_gib, walltime, env=(), hosts=None, directory=None, cwd=None, fleet_master_approved=None):
    """Return the flux submit argv for ``command``. Nothing is executed."""
    duration_seconds(walltime)
    if (cores > 10 or memory_gib > 32) and not (fleet_master_approved or "").strip():
        raise ValueError("GPU request above 10 cores / 32 GiB requires --fleet-master-approved naming the approval")
    if not command or cores < 1 or memory_gib < 1:
        raise ValueError("a GPU jobspec needs a command, a walltime, cores >= 1 and memory >= 1")
    if directory is None or not Path(directory).is_absolute() or not Path(directory).resolve().is_relative_to("/mnt/shared") or Path(directory).resolve() == Path("/mnt/shared"):
        raise ValueError("GPU output directory must be an owned unique path under /mnt/shared")
    directory = Path(directory)
    cwd = str(Path.cwd() if cwd is None else Path(cwd))
    allowlist = [*[f"{key}={value}" for key, value in BASE_ENV.items()], *env]
    for entry in allowlist:
        if "=" not in entry or entry.startswith("="):
            raise ValueError(f"environment allowlist entry is not NAME=VALUE: {entry!r}")
    return ["flux", "submit", "--env=-*", *[f"--env={entry}" for entry in allowlist],
            f"--resources-shape={resource_shape(cores, memory_gib, 1)}",
            *([f"--requires=host:{hosts}"] if hosts else []), "-t", walltime,
            "-n1", f"--job-name={directory.name}", "--cwd", cwd, "--input=/dev/null", f"--output={directory}/stdout", f"--error={directory}/stderr",
            "--", *command]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["run"]:
        return container_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submit", action="store_true", help="submit one bounded job and collect terminal evidence")
    parser.add_argument("--cores", type=int, required=True)
    parser.add_argument("--memory-gib", type=int, required=True)
    parser.add_argument("--walltime", required=True, help="explicit walltime, e.g. 45m")
    parser.add_argument("--out", required=True, help="unique owned shared output directory; create it before using the line")
    parser.add_argument("--fleet-master-approved", help="fleet master approval reference for an oversized request")
    parser.add_argument("--cwd", default=str(Path.cwd()), help="payload working directory")
    parser.add_argument("--env", action="append", default=[], help="NAME=VALUE allowed into the job")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    rendered = render(command, cores=args.cores, memory_gib=args.memory_gib,
                      walltime=args.walltime, env=args.env, directory=args.out, cwd=args.cwd,
                      fleet_master_approved=args.fleet_master_approved)
    if args.submit:
        try:
            return submit_and_collect(rendered, args.out, cores=args.cores,
                                      memory_gib=args.memory_gib, walltime=args.walltime)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    print(shlex.join(rendered))
    print("RENDERED, NOT SUBMITTED", file=sys.stderr)
    return 0
def flux_client():
    client = shutil.which('flux')
    client_environment = None
    if client is None:
        prefix = Path('/mnt/shared/flux-migration/INSTANCE').read_text().split()[0]
        client = prefix + '/install/bin/flux'
        client_environment = {'PATH': prefix + '/install/bin:' + prefix + '/deps/bin:/usr/bin:/bin',
                              'LD_LIBRARY_PATH': prefix + '/install/lib:' + prefix + '/deps/lib',
                              'HOME': '/home/rob'}
        client_environment['FLUX_URI'] = subprocess.check_output(
            ['/mnt/shared/flux-migration/flux-uri.sh'], env=client_environment, text=True).strip()
    return client, client_environment


def plain_job_id(value):
    client, environment = flux_client()
    plain = subprocess.check_output([client, 'job', 'id', '--to=f58plain', value],
                                    env=environment, text=True).strip()
    if not re.fullmatch(r'f[1-9A-HJ-NP-Za-km-z]+', plain):
        raise ValueError('Flux did not return a plain f58 job ID')
    return plain


def submit_and_collect(argv, directory, *, cores, memory_gib, walltime):
    """Submit one request and wait for its native terminal records, never retry."""
    if duration_seconds(walltime) > 600:
        raise ValueError('GPU runs over 10 minutes need a fleet-master duration decision; not submitted')
    if cores > 10 or memory_gib > 32:
        raise ValueError('GPU request exceeds 10 cores / 32 GiB; needs a size decision; not submitted')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    client, client_environment = flux_client()
    def native(arguments, name):
        result = subprocess.run([client, *arguments], env=client_environment, capture_output=True, text=True)
        (directory / (name + '.stdout')).write_text(result.stdout)
        (directory / (name + '.stderr')).write_text(result.stderr)
        return result
    resources = native(['resource', 'list'], 'resources-before')
    if resources.returncode:
        raise ValueError('Flux resource inspection failed; not submitted')
    explicit = [arg.removeprefix('--requires=host:') for arg in argv if arg.startswith('--requires=host:')]
    for host in explicit or ['sparky', 'sparklina']:
        for field in ('--query-gpu=utilization.gpu', '--query-compute-apps=process_name'):
            result = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, 'nvidia-smi', field,
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True)
            (directory / (host + ('.utilization' if 'query-gpu' in field else '.process-names'))).write_text(result.stdout)
            if result.returncode or (result.stdout.strip() != '0' if 'query-gpu' in field else bool(result.stdout.strip())):
                raise ValueError('external GPU load or unreadable preflight on ' + host + '; not submitted')
    # The last payload step samples its own sdexec unit, including Podman.
    payload = directory / 'payload.sh'
    payload.write_text('''#!/bin/bash
out=$1; memory=$2; shift 2
finish(){
 rc=$?; trap - EXIT
 peak_path=/sys/fs/cgroup$(cut -d: -f3- /proc/self/cgroup)/memory.peak
 python3 - "$out/memory.json" "$memory" "$peak_path" <<'MEMORY'
import json,pathlib,sys
r={'declared_memory_gib':int(sys.argv[2]),'memory_peak_bytes':None,'memory_peak_path':sys.argv[3]}
try: r['memory_peak_bytes']=int(pathlib.Path(sys.argv[3]).read_text())
except OSError as e: r['memory_peak_read_error']=str(e)
pathlib.Path(sys.argv[1]).write_text(json.dumps(r))
MEMORY
 exit "$rc"
}
trap finish EXIT
"$@"
''')
    split = argv.index('--')
    args = [*argv[2:split], '--env=XDG_RUNTIME_DIR=/run/user/1000',
            '--env=CUDA_VISIBLE_DEVICES=0', f'--env=FLUX_MEMORY_GIB={memory_gib}',
            '--', '/bin/bash', str(payload), str(directory), str(memory_gib), *argv[split + 1:]]
    (directory / 'submission.json').write_text(json.dumps({'state': 'unknown'}))
    submitted = native(['submit', *args], 'submit')
    job = submitted.stdout.strip()
    if submitted.returncode or not re.fullmatch(r'[fƒ][1-9A-HJ-NP-Za-km-z]+|[0-9]+', job):
        raise ValueError('submission unknown; reconcile this directory, never resubmit blindly')
    (directory / 'jobid').write_text(job + '\n')
    (directory / 'submission.json').write_text(json.dumps({'state': 'submitted', 'job_id': job}))
    status = native(['job', 'status', '--json', '--exception-exit-code=125', job], 'status')
    native(['job', 'wait-event', job, 'clean'], 'wait-clean')
    events = native(['job', 'eventlog', '--format=json', job], 'eventlog')
    allocation = native(['job', 'info', job, 'R'], 'R')
    (directory / 'status.json').write_text(status.stdout)
    (directory / 'eventlog.jsonl').write_text(events.stdout)
    (directory / 'R.json').write_text(allocation.stdout)
    (directory / 'flux-exit-code.txt').write_text(str(status.returncode) + '\n')
    errors = []
    try:
        terminal = json.loads(status.stdout)
        if terminal.get('exception_occurred') or terminal.get('waitstatus') != 0:
            errors.append('native job failed or raised an exception')
        if not any(json.loads(line).get('name') == 'clean' for line in events.stdout.splitlines()):
            errors.append('native clean event missing')
        assigned = json.loads(allocation.stdout)['execution']['R_lite']
        if len(assigned) != 1 or assigned[0]['children'].get('gpu') != '0':
            errors.append('allocation is not one whole Spark GPU')
        else:
            allocated_cores = set()
            for part in assigned[0]['children']['core'].split(','):
                low, _, high = part.partition('-')
                allocated_cores.update(range(int(low), int(high or low) + 1))
            if len(allocated_cores) != cores:
                errors.append('allocated core count differs from request')
    except (ValueError, KeyError, TypeError):
        errors.append('unreadable native terminal/allocation evidence')
    if events.returncode or allocation.returncode:
        errors.append('native evidence query failed')
    receipt = {'job_id': job, 'returncode': status.returncode,
               'declared_memory_gib': memory_gib, 'memory_peak_bytes': None,
               'evidence_errors': errors}
    try:
        receipt.update(json.loads((directory / 'memory.json').read_text()))
    except (OSError, ValueError) as exc:
        receipt['memory_peak_read_error'] = str(exc)
    (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('Flux job ' + job + ': ' + str(directory / 'receipt.json'))
    return status.returncode or (2 if errors else 0)


def container_options(arguments):
    """Parse only payload options; containment flags belong to this launcher."""
    paired = {'-v', '--volume', '--mount', '-e', '--env', '-w', '--workdir',
              '--entrypoint', '--user', '--label', '--cidfile', '--tmpfs'}
    options, entrypoint, external_cid = [], None, None
    index = 0
    while index < len(arguments) and arguments[index].startswith('-'):
        key, equal, value = arguments[index].partition('=')
        if key not in paired:
            raise ValueError('container option is owned by launcher or unsupported: ' + key)
        if not equal:
            index += 1
            if index == len(arguments):
                raise ValueError('container option needs a value: ' + key)
            value = arguments[index]
        if key == '--entrypoint':
            if entrypoint is not None:
                raise ValueError('duplicate entrypoint')
            entrypoint = value
        elif key == '--cidfile':
            external_cid = Path(value)
            if external_cid.exists():
                raise ValueError('external cidfile already exists')
        else:
            options.extend([key, value])
        index += 1
    if index == len(arguments) or not entrypoint:
        raise ValueError('container requires a pinned image and explicit entrypoint')
    image = arguments[index]
    if not re.fullmatch(r'(?:[^\s]+@)?sha256:[0-9a-f]{64}', image):
        raise ValueError('container image must be digest-pinned')
    return options, entrypoint, image, arguments[index + 1:], external_cid


def run_container(arguments, directory, scratch_root, *, gpu=True):
    """Own a foreground rootless container inside the calling Flux job unit."""
    options, entrypoint, image, command, external_cid = container_options(arguments)
    # Flux injects these values. They are context, not an admission token.
    if not os.environ.get('FLUX_JOB_ID'):
        raise ValueError('GPU container requires Flux job context')
    job = plain_job_id(os.environ['FLUX_JOB_ID'])
    if gpu and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('GPU container requires Flux job and allocated CUDA_VISIBLE_DEVICES context')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(scratch_root).resolve() / job
    root.parent.mkdir(parents=True, exist_ok=True)
    filesystem = subprocess.check_output(['findmnt', '-n', '-o', 'FSTYPE', '-T', str(root.parent)], text=True).strip()
    if (gpu and filesystem != 'ext4') or (not gpu and filesystem not in {'ext4', 'xfs', 'btrfs'}):
        raise ValueError('GPU scratch requires ext4, found ' + filesystem)
    rootless = subprocess.check_output(['podman', 'info', '--format', '{{.Host.Security.Rootless}}'], text=True).strip()
    if rootless != 'true':
        raise ValueError('GPU container requires rootless Podman')
    name = 'tessera-' + job + '-' + uuid.uuid4().hex
    scratch = root / name
    cidfile = scratch / 'container.cid'
    affinity = sorted(os.sched_getaffinity(0))
    record = {'job_id': job, 'name': name, 'image': image, 'host_affinity': affinity,
              'scratch': str(scratch), 'scratch_filesystem': filesystem,
              'rootless': True, 'removed': False, 'scratch_removed': False,
              'container_affinity': None, 'returncode': None, 'cleanup_errors': []}
    path = directory / ('container-' + name + '.json')
    def save():
        path.write_text(json.dumps(record, indent=2) + '\n')
    save()  # Register the unique owned name before create can start.
    process = None
    class Interrupted(BaseException):
        pass
    def interrupt(signum, frame):
        record['signal'] = signum
        raise Interrupted
    handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        root.mkdir(exist_ok=True)
        scratch.mkdir()
        (scratch / 'tmp').mkdir()
        identity = ['--userns=keep-id'] if '--user' in options and options[options.index('--user') + 1] != '0:0' else []
        create = ['podman', 'create', '--pull=never', '--cgroups=disabled', '--network=none', '--init',
                  '--name', name, '--cidfile', str(cidfile),
                  *(['--device', 'nvidia.com/gpu=all'] if gpu else []),
                  '--cpuset-cpus', ','.join(map(str, affinity)), '--volume', str(scratch) + ':/tessera-scratch',
                  *identity, *options,
                  '--env', 'TMPDIR=/tessera-scratch/tmp',
                  '--env', 'CUDA_VISIBLE_DEVICES=' + (os.environ['CUDA_VISIBLE_DEVICES'] if gpu else ''),
                  *[arg for key, value in THREADS.items() for arg in ('--env', key + '=' + value)],
                  '--entrypoint', '/bin/sh', image, '-ec',
                  'cat /proc/self/status > /tessera-scratch/container-status; exec "$@"',
                  'tessera-gpu', entrypoint, *command]
        process = subprocess.Popen(create, stdout=subprocess.PIPE, text=True)
        stdout, _ = process.communicate()
        record['create_returncode'] = process.returncode
        if process.returncode:
            raise ValueError('rootless Podman create failed')
        cid = cidfile.read_text().strip()
        if not re.fullmatch(r'[0-9a-f]{64}', cid):
            raise ValueError('Podman did not publish a container ID')
        record['cid'] = cid
        if external_cid:
            with external_cid.open('x') as stream:
                stream.write(cid + '\n')
        process = subprocess.Popen(['podman', 'start', '--attach', name])
        record['returncode'] = process.wait()
        status = (scratch / 'container-status').read_text()
        mask = next(line.split(':', 1)[1].strip() for line in status.splitlines() if line.startswith('Cpus_allowed_list:'))
        actual = set()
        for part in mask.split(','):
            low, _, high = part.partition('-')
            actual.update(range(int(low), int(high or low) + 1))
        record['container_affinity'] = sorted(actual)
        if sorted(actual) != affinity:
            raise ValueError('container changed the Flux CPU affinity')
    except Interrupted:
        record['returncode'] = 128 + record['signal']
    finally:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            removed = subprocess.run(['podman', 'rm', '--force', '--time=2', name], capture_output=True, text=True, timeout=15)
            absent = subprocess.run(['podman', 'container', 'exists', name], timeout=15).returncode
            record['removed'] = absent == 1
            if not record['removed']:
                raise ValueError('owned Podman container remains: ' + removed.stderr)
            if scratch.exists():
                shutil.rmtree(scratch)
            record['scratch_removed'] = not scratch.exists()
            try:
                root.rmdir()
            except OSError:
                pass  # A parent payload may own other scratch in this job.
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            record['cleanup_errors'].append(str(exc))
        finally:
            memory_path = Path('/sys/fs/cgroup' + Path('/proc/self/cgroup').read_text().strip().split(':', 2)[2]) / 'memory.peak'
            record['declared_memory_gib'] = os.environ.get('FLUX_MEMORY_GIB')
            record['memory_peak_bytes'] = None
            try:
                record['memory_peak_bytes'] = int(memory_path.read_text())
            except OSError as exc:
                record['memory_peak_read_error'] = str(exc)
            save()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    if record['cleanup_errors']:
        raise ValueError('GPU container cleanup failed: ' + '; '.join(record['cleanup_errors']))
    return record['returncode']


def container_main(argv):
    parser = argparse.ArgumentParser(description='Run inside an admitted Flux GPU job with rootless Podman')
    parser.add_argument('--out', required=True)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--scratch-root', default='/tmp/flux-vanilla')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    try:
        return run_container(arguments, args.out, args.scratch_root, gpu=not args.cpu_only)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
