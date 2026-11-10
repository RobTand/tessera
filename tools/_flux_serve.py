"""One scored serve per Flux GPU job, using the reviewed root Docker pattern.

Only this module may start root Docker inside a job. The shell adapter keeps
scoring in the existing wrapper; a foreground supervisor owns its container.
No real serving run is qualified until fleet F-6 housekeeping is live.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import uuid
import time
import urllib.request

CLIENT_MEMORY_GIB = 4
SPARK_MEMORY_GIB = 108
SPARK_PHYSICAL_GIB = Decimal('121.6')
DIGEST = re.compile(r'[^\s@]+@sha256:[0-9a-f]{64}')
WRAPPERS = (
    'serve_and_dump_kl.sh', 'tessera_plugin_served.sh', 'decode_regime_kl.sh',
    'gridbook_lane_served.sh', 'window_gemv_latency.sh', 'serve_smoke_graph.sh',
    'lfm_bound_served.sh',
    'moe_greedy_smoke_pair.sh',
)
SERVE_ENV = (
    'TS', 'GB', 'RUNS', 'EXT', 'VLLM_CACHE', 'TRACEDIR', 'PY', 'KL', 'PORT',
    'TAG', 'PROF', 'OUT', 'TEACHER_DECODE', 'TEACHER_PREFILL',
    'TESSERA_KL_PORT', 'TESSERA_KL_LOGDIR', 'TESSERA_KL_LOG', 'TESSERA_KL_DUMP',
    'TESSERA_KL_CORPUS', 'TESSERA_KL_TOPK', 'TESSERA_KL_EAGER', 'TESSERA_KL_REGIME',
    'TESSERA_KL_DECODE_STRIDE', 'TESSERA_KL_VLLM_EXTRA', 'TESSERA_KL_DOCKER_EXTRA',
    'TESSERA_KL_IMAGE_COMMAND', 'TESSERA_KL_REQUIRE_IN_LOG', 'TESSERA_KL_VLLM_CACHE',
    'TESSERA_KL_TEACHER', 'TESSERA_KL_DUMP_PREFIX', 'TESSERA_KL_DIR',
    'TESSERA_KL_PROFILE_DIR', 'TESSERA_LANE_EAGER', 'TESSERA_LANE_DOCKER_EXTRA',
    'TESSERA_SERVE_DETERMINISTIC', 'TESSERA_KL_PYTHON', 'TESSERA_KL_TOOL',
    'TESSERA_PROFILE_MIN_AVAILABLE_BYTES', 'TESSERA_PROFILE_SUMMARY_TIMEOUT_S',
    'PROMPTS',
)
ROOT = Path(__file__).resolve().parents[1]


class Refused(RuntimeError):
    """A named failure; never a request for a fallback runtime."""


class Interrupted(BaseException):
    pass


def require(condition, reason):
    if not condition:
        raise Refused(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def spec_id(spec):
    return hashlib.sha256(canonical(spec)).hexdigest()


def content_sha256(directory):
    """Hash relative names and file bytes, following checkpoint shard symlinks."""
    root = Path(directory).resolve(strict=True)
    require(root.is_dir(), 'MODEL_DIRECTORY_REQUIRED')
    entries = []
    for path in sorted(root.rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            entries.append([path.relative_to(root).as_posix(), path.stat().st_size, digest])
    require(entries, 'MODEL_DIRECTORY_EMPTY')
    return hashlib.sha256(canonical(entries)).hexdigest()


def bounds(util):
    value = Decimal(str(util))
    require(value.is_finite() and 0 < value < 1, 'GPU_MEMORY_UTILIZATION')
    serve = int((SPARK_PHYSICAL_GIB * value).to_integral_value(rounding=ROUND_CEILING))
    require(serve + CLIENT_MEMORY_GIB <= SPARK_MEMORY_GIB, 'SPARK_MEMORY_BOUND')
    return serve, serve + CLIENT_MEMORY_GIB


def payload_options(options):
    """Accept payload mounts, environment and entrypoint, never containment overrides."""
    result = []
    index = 0
    while index < len(options):
        key, equal, value = options[index].partition('=')
        require(key in {'-v', '--volume', '-e', '--env', '-w', '--workdir', '--entrypoint'},
                'CONTAINER_OPTION: ' + key)
        if not equal:
            index += 1
            require(index < len(options), 'CONTAINER_OPTION_VALUE: ' + key)
            value = options[index]
        result.extend([key, value])
        index += 1
    return result


def plugin_identity(directory):
    def git(*args):
        return subprocess.check_output(['git', '-C', directory, *args], text=True).strip()
    require(not git('status', '--porcelain', '--untracked-files=normal'), 'PLUGIN_TREE_DIRTY')
    return {'commit': git('rev-parse', 'HEAD'), 'tree': git('rev-parse', 'HEAD^{tree}')}


def plugin_roots(request):
    return [Path(path).resolve() for path in
            ([request['plugin']] if request.get('plugin') else []) + request.get('supporting_plugins', [])]


def make_spec(request):
    require(DIGEST.fullmatch(request['image']), 'IMAGE_DIGEST_REQUIRED')
    options = payload_options(request['options'])
    bounds(request['util'])
    plugin = request.get('plugin')
    identity = plugin_identity(plugin) if plugin else None
    supporting = request.get('supporting_plugins', [])
    aliases = [(str(Path(path).resolve()), '${SUPPORTING_PLUGIN_' + str(index) + '}')
               for index, path in enumerate(supporting)]
    if plugin:
        aliases.append((str(Path(plugin).resolve()), '${PLUGIN}'))
    # Paths and arguments retain their meaning without naming a requesting repo.
    # Host plugin locations are replaced by their mounted project identity.
    def normalize(value):
        for path, alias in sorted(aliases, key=lambda pair: len(pair[0]), reverse=True):
            value = value.replace(path, alias)
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return value
        return canonical(parsed).decode() if isinstance(parsed, (list, dict)) else value
    return {'image_digest': request['image'], 'model_dir': str(Path(request['model']).resolve()),
            'model_sha256': content_sha256(request['model']), 'plugin': identity,
            'supporting_plugins': [plugin_identity(path) for path in supporting],
            'vllm_args': [normalize(arg) for arg in request['command']],
            'container_options': [normalize(arg) for arg in options],
            'gpu_memory_utilization': str(Decimal(request['util']).normalize())}


def docker_command(*args, check=True):
    result = subprocess.run(['docker', *args], capture_output=True, text=True, timeout=60)
    if check and result.returncode:
        raise Refused('DOCKER_COMMAND_FAILED: ' + args[0] + ': ' + result.stderr.strip())
    return result.stdout.strip()


def decimal_job_id(value):
    # Client paths are isolated from the job environment and its subprocesses.
    sys.path.insert(0, str(ROOT / 'tools'))
    from _flux_gpu_jobspec import flux_client
    client, environment = flux_client()
    value = subprocess.check_output([client, 'job', 'id', '--to=dec', value],
                                    env=environment, text=True, timeout=30).strip()
    require(re.fullmatch(r'[0-9]+', value), 'DECIMAL_JOB_ID')
    return value


def cpu_list():
    return next(line.split(':', 1)[1].strip() for line in Path('/proc/self/status').read_text().splitlines()
                if line.startswith('Cpus_allowed_list:'))


def fetch_models(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.load(response)
    require(isinstance(payload, dict) and isinstance(payload.get('data'), list), 'HEALTH_RESPONSE')
    return payload


def container_peak_path(inspection, *, proc_root=Path('/proc'), cgroup_root=Path('/sys/fs/cgroup')):
    pid = inspection['State']['Pid']
    require(isinstance(pid, int) and pid > 0, 'CONTAINER_MEMORY_PEAK: missing container PID')
    lines = (proc_root / str(pid) / 'cgroup').read_text().splitlines()
    unified = next(line[3:] for line in lines if line.startswith('0::'))
    path = Path(unified)
    require(path.is_absolute() and '..' not in path.parts and
            ('docker-' + inspection['Id'] + '.scope') in path.parts,
            'CONTAINER_MEMORY_PEAK: not the inspected Docker scope')
    return cgroup_root / path.relative_to('/') / 'memory.peak'


class Serve:
    """Own one container. Inject only Docker, HTTP and observed host boundaries."""

    def __init__(self, spec, receipt, *, docker=None, decimal_id=None, cpus=None,
                 models=None, peak_path=None):
        self.spec = spec
        self.receipt = Path(receipt)
        self.docker = docker or docker_command
        self.decimal_id = decimal_id or decimal_job_id
        self.cpus = cpus or cpu_list
        self.models = models or fetch_models
        self.peak_path = peak_path or container_peak_path
        self.name = None
        self.cid = None
        self.peak = None
        self.started = False
        self.record = {'spec_id': spec_id(spec), 'spec': spec, 'removed': False,
                       'container_memory_peak_bytes': None}

    def save(self):
        self.receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.receipt.with_suffix('.tmp')
        temporary.write_bytes(canonical(self.record) + b'\n')
        temporary.replace(self.receipt)

    def inspect(self):
        return json.loads(self.docker('container', 'inspect', self.cid or self.name))[0]

    def check_running(self):
        require(self.inspect()['State']['Running'], 'CONTAINER_EXITED')

    def sample_peak(self):
        try:
            value = int(self.peak.read_text())
            self.record['container_memory_peak_bytes'] = max(
                value, self.record['container_memory_peak_bytes'] or 0)
        except (OSError, ValueError, AttributeError) as exc:
            raise Refused('CONTAINER_MEMORY_PEAK: ' + str(exc)) from exc

    def start(self, request, *, health_seconds=2400):
        require(os.environ.get('FLUX_JOB_ID'), 'OUTSIDE_FLUX_JOB')
        require(re.fullmatch(r'[0-9]+', os.environ.get('CUDA_VISIBLE_DEVICES', '')), 'GPU_ALLOCATION_MISSING')
        actual = make_spec(request)
        require(spec_id(actual) == spec_id(self.spec), 'SPEC_ID_MISMATCH')
        serve, total = bounds(request['util'])
        require(int(os.environ.get('FLUX_MEMORY_GIB', '0')) == total, 'JOB_MEMORY_BOUND')
        job = self.decimal_id(os.environ['FLUX_JOB_ID'])
        self.name = 'flux-' + job
        cpus = self.cpus()
        require(re.fullmatch(r'[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*', cpus), 'CPUSET_MISSING')
        self.record.update(job_id=job, name=self.name, cpus_allowed_list=cpus,
                           declared_serve_memory_gib=serve, declared_job_memory_gib=total,
                           log_path=request.get('log') or str(self.receipt.with_suffix('.log')))
        try:
            local = json.loads(self.docker('image', 'inspect', request['image']))[0]
        except (RuntimeError, ValueError, IndexError) as exc:
            raise Refused('IMAGE_DIGEST_ABSENT: ' + request['image']) from exc
        require(request['image'] in local.get('RepoDigests', []), 'RUNNING_DIGEST_MISMATCH')
        # A pre-existing container is never ours to remove, even with our name.
        require(not self.docker('ps', '-aq', '--filter', 'name=^/' + self.name + '$'), 'CONTAINER_NAME_TAKEN')
        supplied = payload_options(request['options'])
        options = []
        for index in range(0, len(supplied), 2):
            key, value = supplied[index:index + 2]
            if key in {'-v', '--volume'}:
                parts = value.split(':')
                require(len(parts) in {2, 3} and Path(parts[0]).is_absolute(), 'CONTAINER_MOUNT')
                source = Path(parts[0]).resolve()
                # Existing sources are required: Docker must never create root-owned directories.
                require(source.exists(), 'CONTAINER_MOUNT_MISSING: ' + str(source))
                readonly = len(parts) == 3 and 'ro' in parts[2].split(',')
                model_source = Path(request['model']).resolve()
                if source.is_relative_to(model_source) or model_source.is_relative_to(source):
                    require(readonly, 'MODEL_MOUNT_READ_ONLY')
                if any(source.is_relative_to(plugin) or plugin.is_relative_to(source)
                       for plugin in plugin_roots(request)):
                    require(readonly, 'PLUGIN_MOUNT_READ_ONLY')
                if not readonly:
                    require(source.stat().st_uid == os.getuid(), 'CONTAINER_MOUNT_OWNER: ' + str(source))
                if parts[1] == str(Path(request['model']).resolve()):
                    continue  # This destination is emitted once, read-only below.
            options.extend([key, value])
        # The model always has a more specific, read-only mount than any parent.
        model = str(Path(request['model']).resolve())
        options += ['-v', model + ':' + model + ':ro']
        self.save()
        self.started = True  # Own the registered name before Docker can create it.
        self.cid = self.docker(
            'run', '-d', '--init', '--pull=never', '--ipc=host', '--name=' + self.name,
            '--label=flux.jobid=' + job, '--cpuset-cpus=' + cpus,
            f'--memory={serve}g', f'--memory-swap={serve}g',
            '--gpus=device=' + os.environ['CUDA_VISIBLE_DEVICES'],
            '--publish=127.0.0.1:' + str(request['port']) + ':8000',
            f'--user={os.getuid()}:{os.getgid()}',
            '-e', 'HOME=/tmp', '-e', 'OMP_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=1',
            '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'MAX_JOBS=1',
            *options, request['image'], *request['command']).strip()
        require(re.fullmatch(r'[0-9a-f]{64}', self.cid), 'CONTAINER_ID_MISSING')
        inspection = self.inspect()
        running_image = json.loads(self.docker('image', 'inspect', inspection['Image']))[0]
        require(inspection['Image'] == local['Id'] and request['image'] in running_image.get('RepoDigests', []),
                'RUNNING_DIGEST_MISMATCH')
        self.record.update(container_id=self.cid, image_id=inspection['Image'],
                           repo_digest=request['image'], started_at=inspection['State']['StartedAt'])
        self.check_running()
        try:
            self.peak = self.peak_path(inspection)
            self.record['container_memory_peak_path'] = str(self.peak)
            self.sample_peak()
        except (OSError, StopIteration, KeyError) as exc:
            raise Refused('CONTAINER_MEMORY_PEAK: ' + str(exc)) from exc
        deadline = time.monotonic() + health_seconds
        while True:
            self.check_running()
            try:
                self.record['models'] = self.models('http://127.0.0.1:' + str(request['port']) + '/v1/models',
                                                    min(5, max(.001, deadline - time.monotonic())))
                break
            except (OSError, ValueError, Refused):
                require(time.monotonic() < deadline, 'HEALTH_DEADLINE')
                time.sleep(min(.2, max(0, deadline - time.monotonic())))
        self.save()
        return self.cid

    def stop(self):
        if not self.started:
            return
        errors = []
        try:
            Path(self.record['log_path']).write_text(self.docker('logs', self.cid or self.name, check=False) + '\n')
        except OSError as exc:
            self.record['log_error'] = str(exc)
        if self.peak:
            try:
                self.sample_peak()
            except Refused as exc:
                # A container that died may already have lost its cgroup. Retain
                # the highest observed peak and name the incomplete final sample.
                self.record['final_peak_error'] = str(exc)
        try:
            self.docker('stop', '--time=10', self.cid or self.name, check=False)
            self.docker('rm', '-f', self.cid or self.name, check=False)
            remaining = self.docker('ps', '-aq', '--filter', 'name=^/' + self.name + '$')
            require(not remaining, 'CLEANUP_UNVERIFIED')
            self.record['removed'] = True
            self.started = False
        except (RuntimeError, OSError) as exc:
            errors.append(str(exc))
        self.record['cleanup_errors'] = errors
        self.save()
        require(not errors, 'CLEANUP_UNVERIFIED: ' + '; '.join(errors))


def request_from_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--plugin')
    p.add_argument('--supporting-plugin', action='append', default=[])
    p.add_argument('--log')
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--util', required=True)
    p.add_argument('payload', nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    args = a.payload[1:] if a.payload[:1] == ['--'] else a.payload
    index = 0
    while index < len(args) and args[index].startswith('-'):
        key, equal, value = args[index].partition('=')
        require(key in {'-v', '--volume', '-e', '--env', '-w', '--workdir', '--entrypoint'}, 'CONTAINER_OPTION: ' + key)
        index += 1 if equal else 2
    require(index < len(args), 'IMAGE_DIGEST_REQUIRED')
    return {'model': a.model, 'plugin': a.plugin, 'supporting_plugins': a.supporting_plugin,
            'port': a.port, 'util': a.util,
            'log': a.log, 'options': args[:index], 'image': args[index], 'command': args[index + 1:]}


def rpc(action, arguments):
    require(os.environ.get('TESSERA_SERVE_SOCKET'), 'SERVE_SUPERVISOR_REQUIRED')
    with socket.socket(socket.AF_UNIX) as client:
        client.connect(os.environ['TESSERA_SERVE_SOCKET'].replace('@', '\0', 1))
        client.sendall(canonical({'action': action, 'arguments': arguments}) + b'\n')
        response = json.loads(client.makefile('rb').readline())
    require(response['ok'], response.get('error', 'SERVE_RPC_FAILED'))
    if response.get('value'):
        print(response['value'])


def enter_scoring_stage(command):
    """Enter the caller's scoring command through its foreground supervisor."""
    if os.environ.get('TESSERA_SERVE_SOCKET'):
        return
    os.execvp(command[0], command)


def cleanup_scoring_stage(name, *, launched, completed, stop, monitor):
    """Stop telemetry and ask the sole container owner to verify cleanup."""
    result = {'owner': 'reviewed-serving-supervisor', 'measurement_completed': completed,
              'launched': launched, 'safe_to_release': False}
    stop.set()
    monitor.join(timeout=2)
    try:
        if launched:
            rpc('stop', [])
        result['safe_to_release'] = True
    except (RuntimeError, OSError) as exc:
        result['error'] = str(exc)
    return result


def supervise(spec, receipt, command):
    require(os.environ.get('FLUX_JOB_ID'), 'OUTSIDE_FLUX_JOB')
    serve = Serve(spec, receipt)
    child = None
    interrupted = []
    handlers = {}
    def on_signal(signum, frame):
        if not interrupted:
            interrupted.append(signum)
            raise Interrupted
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        handlers[signum] = signal.signal(signum, on_signal)
    # An abstract Unix socket avoids filesystem path limits and leaves no file.
    require(re.fullmatch(r'(?:[fƒ][1-9A-HJ-NP-Za-km-z]+|[0-9]+)', os.environ['FLUX_JOB_ID']), 'FLUX_JOB_ID_INVALID')
    scratch = Path(os.environ.get('FLUX_SCRATCH', '/tmp/flux-vanilla')) / os.environ['FLUX_JOB_ID']
    scratch.mkdir(parents=True, exist_ok=False)
    code = 2
    used = False
    try:
        with socket.socket(socket.AF_UNIX) as server, selectors.DefaultSelector() as selector:
            address = '@tessera-serve-' + uuid.uuid4().hex
            server.bind('\0' + address[1:])
            server.listen()
            selector.register(server, selectors.EVENT_READ)
            child = subprocess.Popen(command, env={**os.environ, 'TESSERA_SERVE_SOCKET': address,
                                                  'IMAGE': spec['image_digest'],
                                                  'TS': os.environ.get('TS', str(ROOT)),
                                                  'TESSERA_KL_IMAGE': spec['image_digest'],
                                                  'TESSERA_GPU_MEM_UTIL': spec['gpu_memory_utilization']},
                                     start_new_session=True)
            while child.poll() is None:
                if interrupted:
                    raise Refused('SIGNAL_' + str(interrupted[0]))
                for key, mask in selector.select(.1):
                    with server.accept()[0] as client:
                        message = json.loads(client.makefile('rb').readline())
                        try:
                            if message['action'] == 'start':
                                require(not used, 'ONE_SCORED_SERVE_PER_JOB')
                                used = True
                                value = serve.start(request_from_args(message['arguments']))
                            elif message['action'] == 'stop':
                                serve.stop()
                                value = None
                            else:
                                raise Refused('SERVE_RPC_ACTION')
                            client.sendall(canonical({'ok': True, 'value': value}) + b'\n')
                        except (RuntimeError, OSError, ValueError) as exc:
                            client.sendall(canonical({'ok': False, 'error': str(exc)}) + b'\n')
                            raise
                if serve.started:
                    serve.check_running()
                    serve.sample_peak()
            code = child.returncode
            require(used, 'NO_SCORED_SERVE')
    except (RuntimeError, OSError, ValueError, Interrupted) as exc:
        serve.record['refusal'] = str(exc)
        print('REFUSED: ' + str(exc), file=sys.stderr)
        code = 128 + interrupted[0] if interrupted else 2
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        try:
            serve.stop()
        except (RuntimeError, OSError) as exc:
            serve.record['cleanup_refusal'] = str(exc)
            code = code or 2
        shutil.rmtree(scratch)
        serve.record['returncode'] = code
        serve.record['scratch_removed'] = not scratch.exists()
        serve.save()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return code


def render_serving(command, spec, *, out, cores=4, walltime='10m', image_host=None, spec_path,
                   duration_approval=None):
    require(1 <= cores <= 10, 'SERVE_CORE_BOUND')
    entry = Path(command[0]).resolve() if command else None
    require(entry is not None and entry.parent == ROOT / 'experiments' and entry.name in WRAPPERS,
            'REVIEWED_SERVING_WRAPPER_REQUIRED')
    sys.path.insert(0, str(ROOT / 'tools'))
    from _flux_gpu_jobspec import render
    from _flux_shape import duration_seconds
    require(duration_seconds(walltime) <= 600 or (duration_approval or '').strip(),
            'SERVE_DURATION_APPROVAL_REQUIRED: GPU runs over 10 minutes need the fleet master')
    _, total = bounds(spec['gpu_memory_utilization'])
    return render(command, cores=cores, memory_gib=total, walltime=walltime,
                  env=[*[key + '=' + os.environ[key] for key in SERVE_ENV if key in os.environ],
                       'TESSERA_SERVE_SPEC=' + str(Path(spec_path).resolve()),
                       'TESSERA_SERVE_EXPECTED_SPEC_ID=' + spec_id(spec),
                       'TESSERA_SERVE_OUT=' + str(out), 'FLUX_MEMORY_GIB=' + str(total)],
                  hosts=image_host, directory=out,
                  fleet_master_approved='CAPS-v7 serving contract 2026-11-10, <=108 GiB')


def submit_and_collect(argv, out):
    """Submit once and collect the native verdict, clean event and allocation.

    Rendering stays in Tessera's existing GPU jobspec path. This collector has
    the serving contract's <=108 GiB bound rather than the Podman trial's 32.
    """
    sys.path.insert(0, str(ROOT / 'tools'))
    from _flux_gpu_jobspec import flux_client
    client, environment = flux_client()
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=False)
    def native(arguments, name):
        result = subprocess.run([client, *arguments], env=environment, capture_output=True, text=True)
        (directory / (name + '.stdout')).write_text(result.stdout)
        (directory / (name + '.stderr')).write_text(result.stderr)
        return result
    # Flux cannot see external serves or the BuildBuddy GPU executor. Check
    # every eligible Spark without pinning an otherwise portable request.
    hosts = [arg.split(':', 1)[1] for arg in argv if arg.startswith('--requires=host:')]
    for host in hosts or ['sparky', 'sparklina']:
        load = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, 'nvidia-smi',
                               '--query-compute-apps=process_name', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=30)
        (directory / (host + '.external-gpu.stdout')).write_text(load.stdout)
        (directory / (host + '.external-gpu.stderr')).write_text(load.stderr)
        require(load.returncode == 0 and not load.stdout.strip(), 'EXTERNAL_GPU_LOAD: ' + host)
    (directory / 'submission.json').write_bytes(canonical({'state': 'unknown'}))
    result = native(argv[1:], 'submit')
    job = result.stdout.strip()
    require(not result.returncode and re.fullmatch(r'(?:[fƒ][1-9A-HJ-NP-Za-km-z]+|[0-9]+)', job),
            'SUBMISSION_UNKNOWN: reconcile this directory; never retry blindly')
    (directory / 'jobid').write_text(job + '\n')
    (directory / 'submission.json').write_bytes(canonical({'state': 'submitted', 'job_id': job}))
    status = native(['job', 'status', '--json', '--exception-exit-code=125', job], 'status')
    clean = native(['job', 'wait-event', job, 'clean'], 'clean')
    events = native(['job', 'eventlog', '--format=json', job], 'eventlog')
    allocation = native(['job', 'info', job, 'R'], 'R')
    terminal = json.loads(status.stdout)
    # A native failure stays red even if later evidence is missing.
    if status.returncode or terminal.get('exception_occurred') or terminal.get('waitstatus') != 0:
        return status.returncode or 2
    require(clean.returncode == events.returncode == allocation.returncode == 0 and
            any(json.loads(line).get('name') == 'clean' for line in events.stdout.splitlines()),
            'NATIVE_TERMINAL_EVIDENCE_MISSING')
    assigned = json.loads(allocation.stdout)['execution']['R_lite']
    require(len(assigned) == 1 and assigned[0]['children'].get('gpu') == '0', 'GPU_ALLOCATION_MISSING')
    receipt = json.loads((directory / 'serve-receipt.json').read_text())
    expected_spec = next(arg.split('=', 2)[2] for arg in argv if arg.startswith('--env=TESSERA_SERVE_EXPECTED_SPEC_ID='))
    require(receipt.get('spec_id') == expected_spec, 'SPEC_ID_MISMATCH')
    require(receipt['returncode'] == 0 and receipt['removed'] and receipt['scratch_removed'] and
            receipt['container_memory_peak_bytes'] is not None, 'SERVE_RECEIPT_INCOMPLETE')
    try:
        peak = json.loads((directory / 'job-memory.json').read_text())['memory_peak_bytes']
        require(isinstance(peak, int) and peak >= 0, 'JOB_MEMORY_PEAK_MISSING')
    except (OSError, ValueError, KeyError) as exc:
        raise Refused('JOB_MEMORY_PEAK_MISSING') from exc
    return 0


def dispatch(command):
    require(not os.environ.get('FLUX_JOB_ID'), 'CAMPAIGN_INSIDE_FLUX_JOB')
    require(os.environ.get('TESSERA_SERVE_PLANS'), 'SERVE_PLAN_MISSING: set TESSERA_SERVE_PLANS')
    plans = json.loads(Path(os.environ['TESSERA_SERVE_PLANS']).read_text())
    matches = [plan for plan in plans if plan['wrapper'] == Path(command[0]).name and
               plan['arguments'] == command[1:]]
    require(len(matches) == 1, 'SERVE_PLAN_MISSING: exactly one reviewed plan must match wrapper and arguments')
    plan = matches[0]
    spec = json.loads(Path(plan['spec']).read_text())
    # Campaigns may change cwd to a historical checkout; only our reviewed
    # wrapper is submitted, regardless of the caller's checkout location.
    command = [str(ROOT / 'experiments' / Path(command[0]).name), *command[1:]]
    argv = render_serving(command, spec, out=plan['out'], spec_path=plan['spec'],
                          cores=plan.get('cores', 4), walltime=plan['walltime'],
                          image_host=plan.get('image_host'), duration_approval=plan.get('duration_approval'))
    return submit_and_collect(argv, plan['out'])


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    action = args.pop(0)
    if action in {'start', 'stop'}:
        rpc(action, args)
        return 0
    if action == 'dispatch':
        return dispatch(args)
    if action == 'spec':
        print(json.dumps(make_spec(request_from_args(args)), indent=2, sort_keys=True))
        return 0
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--cores', type=int, default=4)
    p.add_argument('--walltime', default='10m')
    p.add_argument('--image-host', choices=['sparky', 'sparklina'])
    p.add_argument('--duration-approval', help='fleet-master decision for a GPU run over 10 minutes')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args(args)
    spec = json.loads(Path(a.spec).read_text())
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    require(command, 'WRAPPER_REQUIRED')
    if action == 'wrapper':
        try:
            require(spec_id(spec) == os.environ.get('TESSERA_SERVE_EXPECTED_SPEC_ID'), 'SPEC_ID_MISMATCH')
            return supervise(spec, Path(a.out) / 'serve-receipt.json', command)
        finally:
            # Docker's scope is separate; this is the client's own final peak.
            group = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
            peak = int((Path('/sys/fs/cgroup') / group.lstrip('/') / 'memory.peak').read_text())
            (Path(a.out) / 'job-memory.json').write_bytes(canonical({'memory_peak_bytes': peak}) + b'\n')
    require(action == 'render', 'SERVE_ACTION')
    rendered = render_serving(command, spec, out=a.out, spec_path=a.spec, cores=a.cores,
                              walltime=a.walltime, image_host=a.image_host, duration_approval=a.duration_approval)
    print(shlex.join(rendered))
    print('RENDERED, NOT SUBMITTED; unqualified until F-6 housekeeping', file=sys.stderr)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        print('REFUSED: ' + str(exc), file=sys.stderr)
        sys.exit(2)
