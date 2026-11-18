"""Run every counted module through Flux. Resume saved IDs without resubmission."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

if __package__:
    from .phase_instance import client, observe, save
else:
    from instance import client, observe, save

FXE = '/mnt/shared/flux-migration/lead-1114/tools/fxe.sh'
SETTINGS = Path(__file__).with_name('feeder-settings.json')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


class FeederError(ValueError):
    pass


def attempt_jobs(root, key):
    """Read the attempt inventory. Never infer a retry from missing metadata."""
    parent = root / 'units' / key / 'attempts'
    if parent.is_symlink() or parent.resolve() != parent:
        raise FeederError('ATTEMPT_PATH_INVALID')
    entries = list(parent.iterdir()) if parent.exists() else []
    if any(p.is_symlink() or not p.is_dir() or not re.fullmatch(r'[1-9][0-9]*', p.name) for p in entries):
        raise FeederError('ATTEMPT_INVENTORY_INVALID')
    entries.sort(key=lambda p: int(p.name))
    if [int(p.name) for p in entries] != list(range(1, len(entries) + 1)):
        raise FeederError('ATTEMPT_IDS_HAVE_GAPS')
    return entries


def dry_run(template, arguments, name, host):
    """Build a jobspec from Fleet's reviewed table. Only the gate submits it."""
    table = read(template / 'table.json')
    placement = table['instances'][name]
    constraint = next(c for c in placement['constraints'] if c == {'hostlist': [host]})
    resource, = placement['shapes'][0]
    shape = 'slot=' + str(resource['count']) + '/[' + ';'.join(
        item['type'] + '=' + str(item['count']) for item in resource['with']) + ']'
    command = [*client(name), 'submit', '--dry-run', '--dependency=singleton',
               '--job-name=' + table['job_name'], '--cwd=' + table['cwd'],
               '--requires=host:' + constraint['hostlist'][0], '--resources-shape=' + shape,
               '-t' + str(placement['max_duration_s']) + 's', '--env=-*']
    environment = dict(PATH='/usr/bin:/bin', HOME='/home/rob', XDG_RUNTIME_DIR='/run/user/1000',
                       FLUX_F58_FORCE_ASCII='1', FLUX_SCRATCH='/var/tmp',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    command += ['--env=' + field + '=' + environment[field] for field in table['env_names']]
    command += ['-o' + key + '=' + json.dumps(value) for key, value in table['shell_options'].items()]
    return subprocess.check_output([*command, str(template / 'job.sh'), *map(str, arguments)], text=True)


def excludes_sparks(constraints):
    """Only a known x86 restriction proves a pending job cannot use a Spark."""
    if 'and' in constraints:
        return any(excludes_sparks(c) for c in constraints['and'])
    if 'or' in constraints:
        return bool(constraints['or']) and all(excludes_sparks(c) for c in constraints['or'])
    return any(p in ('amd64', 'x86_64', 'host:dl380g10') for p in constraints.get('properties', []))


def wait_dense_slot():
    settings = read(SETTINGS)
    started, reported = time.monotonic(), False
    while True:
        jobs = json.loads(subprocess.check_output([*client('fv1'), 'jobs', '--json', '--filter=pending'], text=True))['jobs']
        blocking = []
        for job in jobs:
            if job.get('urgency') == 0:
                continue
            spec = json.loads(subprocess.check_output([*client('fv1'), 'job', 'info', str(job['id']), 'jobspec'], text=True))
            if not excludes_sparks(spec.get('attributes', {}).get('system', {}).get('constraints', {})):
                blocking.append(str(job['id']))
        if not blocking:
            return
        if not reported and time.monotonic() - started >= settings['stall_seconds']:
            print('STALL: pending aarch64 or portable jobs ' + ','.join(blocking), flush=True)
            reported = True
        time.sleep(settings['pending_poll_seconds'])


def drain_ecores_before_submit(memory_gib, submit, reason):
    settings = read(SETTINGS)
    if memory_gib <= settings['fv1_memory_without_fv1e_drain_gib']:
        return None
    hosts = re.findall(r'--requires=host:([\w-]+)', submit)
    if len(hosts) != 1 or hosts[0] not in ('sparky', 'sparklina'):
        raise FeederError('FV1E_REQUIRES_ONE_SPARK_HOST')
    host = hosts[0]
    subprocess.run([FXE, host, 'resource', 'drain', '0', reason], check=True, timeout=settings['client_timeout_seconds'])
    try:
        deadline = time.monotonic() + settings['drain_timeout_seconds']
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FeederError('FV1E_DRAIN_TIMEOUT')
            try:
                jobs = json.loads(subprocess.check_output([FXE, host, 'jobs', '--json'], text=True, timeout=min(remaining, settings['client_timeout_seconds'])))['jobs']
            except subprocess.TimeoutExpired:
                raise FeederError('FV1E_DRAIN_TIMEOUT') from None
            if not jobs:
                return {'host': host, 'reason': reason}
            time.sleep(min(max(0, deadline - time.monotonic()), settings['drain_poll_seconds']))
    except BaseException:
        undrain_ecores(host, reason)
        raise


def undrain_ecores(host, reason):
    timeout = read(SETTINGS)['client_timeout_seconds']
    current = subprocess.check_output([FXE, host, 'resource', 'drain', '--no-header', '--include=0', '--format={reason}'], text=True, timeout=timeout).rstrip('\n')
    if current == reason:
        subprocess.run([FXE, host, 'resource', 'undrain', '0', reason], check=True, timeout=timeout)


def anon_peak(jobid, name, settings):
    """Return the largest valid sampled anon peak, or no admission evidence."""
    try:
        runtime = float(subprocess.check_output([*client(name), 'jobs', '-n', '-o', '{runtime:.6f}', jobid], text=True))
        if not 0 < runtime < float('inf'):
            return None
        prefix = '' if name == 'fv1' else name + '-'
        rows = []
        for directory in settings['fv1e_jobmem_row_dirs']:
            rows = list(Path(directory).glob(prefix + jobid + '-*.tsv'))
            if rows:
                break
        peaks = []
        for row in rows:
            fields = row.read_text().split()
            if len(fields) < 11 or fields[10] != 'ok':
                return None
            peak = int(fields[settings['fv1e_anon_peak_column'] - 1])
            if not 0 <= peak <= int(fields[5]) or int(fields[9]) < 0.5 * runtime:
                return None
            peaks.append(peak)
        return max(peaks) if peaks else None
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def run(canary, bookings, retries=None):
    canary = Path(canary)
    settings = read(SETTINGS)
    lock = (canary / 'counted-feeder.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = read(canary / 'counted-rows.json')
    requested = read(bookings)
    if set(requested) != set(rows):
        raise FeederError('BOOKING_MODULE_SET_DIFFERS')
    for phases in requested.values():
        if set(phases) != {'prepare', 'capture', 'compare'}:
            raise FeederError('BOOKING_PHASE_SET_DIFFERS')
        for shape in phases.values():
            if (set(shape) != {'memory_gib', 'minutes'} or any(type(v) is not int for v in shape.values())
                    or shape['memory_gib'] <= 0 or shape['minutes'] <= 1):
                raise FeederError('BOOKING_SHAPE_INVALID')
        if phases['prepare']['memory_gib'] != settings['fv1_prepare_memory_gib']:
            raise FeederError(f"DENSE_PREPARE_REQUIRES_{settings['fv1_prepare_memory_gib']}_GIB")
        if phases['capture']['minutes'] != settings['capture_minutes']:
            raise FeederError(f"DENSE_CAPTURE_REQUIRES_{settings['capture_minutes']}_MINUTES")
    state_path = canary / 'counted-feeder-state.json'
    state = read(state_path) if state_path.exists() else {'bookings_sha256': sha(bookings), 'settings_sha256': sha(SETTINGS), 'jobs': {}}
    if state['bookings_sha256'] != sha(bookings) or state['settings_sha256'] != sha(SETTINGS):
        raise FeederError('RESUME_INPUTS_DIFFER')
    def event(kind, **fields):
        record = {'utc': subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(), 'event': kind, **fields}
        with (canary / 'counted-feeder.jsonl').open('a') as stream:
            stream.write(json.dumps(record, sort_keys=True) + '\n')
        temporary = state_path.with_suffix('.pending')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_path)
        print(json.dumps(record), flush=True)
    inputs = read(canary / 'counted-chain-inputs.json')
    revision = read(canary / 'campaign.json')['tool_commit']
    from experiments.numerics_rows.prepare_split import check_canary
    check_canary(canary, revision, inputs[next(iter(rows))])
    def fv1e_limit(host):
        reason = None
        try:
            value = (canary / ('fv1e-limit-' + host)).read_text().strip()
            if not re.fullmatch(r'[+-]?[0-9]+', value):
                raise ValueError('non-integer')
            limit = int(value)
            if limit < 0:
                raise ValueError('negative')
        except (OSError, ValueError) as failure:
            limit, reason = 0, str(failure) if isinstance(failure, ValueError) else type(failure).__name__
        limit = min(limit, settings['fv1e_pool_memory_gib'] // settings['fv1e_prepare_memory_gib'])
        limits = state.setdefault('fv1e_limits', {})
        previous = limits.get(host, {})
        current = {'limit': limit, 'reason': reason}
        if previous != current:
            limits[host] = current
            event('FV1E_LIMIT_INVALID' if reason else 'FV1E_LIMIT', spark=host,
                  old=previous.get('limit'), new=limit, reason=reason,
                  effect='new fv1e prepares only; running prepares continue')
        return limit

    accepted, attempts, templates = set(), {}, {}
    for module, directory in rows.items():
        directory = Path(directory)
        config = read(directory / 'campaign.json')
        if not config.get('bound'):
            raise FeederError('COUNTED_ROW_NOT_BOUND')
        templates[module] = config['templates']
        for unit in config['templates']:
            members = unit.get('unit_keys', [])
            selected = set(members).intersection(retries or {})
            if selected and selected != set(members):
                raise FeederError('RETRY_GROUP_REQUIRES_ALL_MEMBERS')
        for key in [*config['unit_keys'], *(t['key'] for t in config['templates'] if t['phase'] == 'compare')]:
            inventory = attempt_jobs(directory, key)
            target = int((retries or {}).get(key, len(inventory) or 1))
            pairs = {member: str(attempts[module, member]) for member in config['unit_keys']} if key.startswith('compare-') else None
            if pairs is not None and inventory and (inventory[-1] / 'compare-inputs.json').exists():
                if read(inventory[-1] / 'compare-inputs.json') != pairs:
                    target = len(inventory) + 1
            if target != (len(inventory) or 1):
                if target != len(inventory) + 1 or not inventory:
                    raise FeederError('RETRY_MUST_BE_NEXT_ATTEMPT')
                previous = inventory[-1]
                phases = ['compare'] if key.startswith('compare-') else ['prepare', 'capture']
                ended = False
                for phase in phases:
                    metadata = previous / (phase + '-instance.json')
                    stored = next((v for v in state['jobs'].values() if v['directory'] == str(previous / phase)), None)
                    if metadata.exists() or stored:
                        identity = read(metadata) if metadata.exists() else stored['instance']
                        if observe(identity['name']) != {k: v for k, v in identity.items() if k != 'job_id'}:
                            raise FeederError('RESUME_PHASE_INSTANCE_DIFFERS')
                        result = subprocess.run([*client(identity['name']), 'job', 'status', '--exception-exit-code=125', identity['job_id']], check=False)
                        native = int(subprocess.check_output([*client(identity['name']), 'jobs', '-n', '-o', '{returncode}', identity['job_id']], text=True))
                        output = subprocess.check_output([*client(identity['name']), 'job', 'info', identity['job_id'], 'guest.output'])
                        event('RETRY_END', unit=key, phase=phase, id=identity['job_id'], status=result.returncode,
                              native_exit=native, logs={'flux-kvs': hashlib.sha256(output).hexdigest()})
                        ended = True
                    elif not (previous / (phase + '-refused.json')).exists() and (
                            (previous / phase).exists() or (previous / (phase + '-intent.json')).exists()):
                        raise FeederError('SUBMISSION_AMBIGUOUS_AUDIT_REQUIRED')
                if not ended:
                    raise FeederError('RETRY_REQUIRES_ENDED_JOB')
            attempt = directory / 'units' / key / 'attempts' / str(target)
            attempt.mkdir(parents=True, exist_ok=True)
            attempts[module, key] = attempt
            if pairs is not None:
                (attempt / 'compare-inputs.json').write_text(json.dumps(pairs) + '\n')

    if set(retries or {}) - {key for _, key in attempts}:
        raise FeederError('RETRY_UNIT_UNKNOWN')

    def phase_job(module, unit, name='fv1', host='sparklina', collect=True):
        phase, key = unit['phase'], unit['key']
        template = Path(rows[module]) / 'templates' / key / phase
        arguments = [attempts[module, member] for member in unit.get('unit_keys', [key])]
        jobs = [attempt / phase for attempt in arguments]
        ids = [attempt / (phase + '-jobid') for attempt in arguments]
        identities = [attempt / (phase + '-instance.json') for attempt in arguments]
        intents = [attempt / (phase + '-intent.json') for attempt in arguments]
        refusals = [attempt / (phase + '-refused.json') for attempt in arguments]
        if any(path.exists() for path in refusals):
            return False
        saved = [v for v in state['jobs'].values() if v['directory'] in map(str, jobs)]
        known = {path.read_text().strip() for path in ids if path.exists()} | {v['id'] for v in saved}
        drain = next((v.get('fv1e_drain') for v in saved), None)
        if known:
            if len(known) != 1:
                raise FeederError('SUBMISSION_ID_AMBIGUOUS')
            jobid, = known
            bound = next((read(path) for path in identities if path.exists()), saved[0]['instance'] if saved else None)
            if bound is None:
                raise FeederError('SUBMISSION_AMBIGUOUS_AUDIT_REQUIRED')
            name = bound['name']
            if drain is None and intents[0].exists():
                drain = read(intents[0]).get('fv1e_drain')
        else:
            if any(path.exists() for path in [*intents, *identities, *jobs]):
                event('AUDIT_HOLD', module=module, phase=phase, unit=key)
                raise FeederError('SUBMISSION_AMBIGUOUS_AUDIT_REQUIRED')
            if name == 'fv1':
                wait_dense_slot()
            else:
                active = json.loads(subprocess.check_output(
                    [*client(name), 'jobs', '--json', '--filter=pending,running'], text=True,
                    timeout=settings['client_timeout_seconds']))['jobs']
                limit = fv1e_limit(host)
                if (state['fv1e_admission'] != 'admitted' or len(active) >= limit
                        or (canary / ('pause-fv1e-' + host)).exists()):
                    return False
            table = read(template / 'table.json')
            spec = dry_run(template, arguments, name, host)
            jobspec = arguments[0] / (phase + '-jobspec.json')
            jobspec.write_text(spec)
            memory = sum(r['count'] for r in table['instances'][name]['shapes'][0][0]['with'] if r['type'] == 'memory')
            drain = drain_ecores_before_submit(memory, '--requires=host:' + host, 'tessera-1717:' + str(jobs[0]))
            bound = observe(name)
            intent = {'instance': bound, 'arguments': list(map(str, arguments)), 'fv1e_drain': drain}
            with ExitStack() as submission:
                for path in intents:
                    stream = submission.enter_context(path.open('w'))
                    fcntl.flock(stream, fcntl.LOCK_EX)
                    stream.write(json.dumps(intent) + '\n')
                    stream.flush()
                event('SUBMIT_INTENT', module=module, phase=phase, unit=key)
                result = subprocess.run([settings['fleet_submit_gate'], name, str(template), str(table['urgency']), str(jobspec)],
                                        text=True, capture_output=True, check=False)
                if result.returncode == 3:
                    for path in refusals:
                        path.write_text(json.dumps({'reason': result.stdout.strip()}) + '\n')
                    event('REFUSED', module=module, phase=phase, unit=key, reason=result.stdout.strip())
                    if drain:
                        undrain_ecores(**drain)
                    return False
                if result.returncode != 0:
                    raise FeederError('SUBMISSION_AMBIGUOUS_AUDIT_REQUIRED')
                jobid = result.stdout.strip()
                bound = {**bound, 'job_id': jobid}
                if not re.fullmatch(r'f[1-9A-HJ-NP-Za-km-z]+', jobid) or observe(name) != intent['instance']:
                    raise FeederError('SUBMISSION_AMBIGUOUS_AUDIT_REQUIRED')
                for id_path, identity_path in zip(ids, identities):
                    id_path.write_text(jobid + '\n')
                    save(identity_path, bound)
        if not re.fullmatch(r'f[1-9A-HJ-NP-Za-km-z]+', jobid):
            raise FeederError('SUBMISSION_ID_AMBIGUOUS')
        if (any(read(path) != bound for path in identities if path.exists())
                or any(v['instance'] != bound for v in saved) or bound['job_id'] != jobid
                or observe(name) != {k: v for k, v in bound.items() if k != 'job_id'}):
            raise FeederError('RESUME_PHASE_INSTANCE_DIFFERS')
        for job, id_path, identity_path in zip(jobs, ids, identities):
            id_path.write_text(jobid + '\n')
            save(identity_path, bound)
            state['jobs'][str(job)] = {'id': jobid, 'instance': bound, 'directory': str(job), 'fv1e_drain': drain}
        token = (name, jobid)
        if token not in accepted:
            event('WAIT', module=module, phase=phase, id=jobid)
        if not collect or token in accepted:
            return True
        status = subprocess.run([*client(name), 'job', 'status', '--exception-exit-code=125', jobid], check=False)
        native = int(subprocess.check_output([*client(name), 'jobs', '-n', '-o', '{returncode}', jobid], text=True).strip())
        kvs = subprocess.check_output([*client(name), 'job', 'info', jobid, 'guest.output'])
        logs = {'flux-kvs': hashlib.sha256(kvs).hexdigest()}
        for job in jobs:
            exits = {label: (job / (label + '-exit.txt')).read_text().strip() for label in ('tool', 'payload', 'podman')}
            logs.update({str(p): sha(p) for p in (job / 'payload.stdout', job / 'payload.stderr') if p.is_file()})
            event('END', module=module, phase=phase, id=jobid, native_exit=native, exits=exits, logs=logs)
            if status.returncode != 0 or native != 0 or any(value != '0' for value in exits.values()):
                raise FeederError('COUNTED_PHASE_FAILED')
            if read(job / 'phase.json').get('completion') is not True:
                raise FeederError('COUNTED_PHASE_UNFINISHED')
            manifest = job / 'completion.sha256'
            lines = manifest.read_text().splitlines()
            if not lines or not (jobs[0] / 'payload.stdout').is_file() or not (jobs[0] / 'payload.stderr').is_file():
                raise FeederError('COUNTED_PHASE_LOGS_MISSING')
            if (job / 'instance.json').exists() and read(job / 'instance.json') != bound:
                raise FeederError('RESUME_PHASE_INSTANCE_DIFFERS')
            for line in lines:
                digest, relative = line.split(None, 1)
                path = job / relative.lstrip('*')
                if not path.resolve().is_relative_to(job) or path.is_symlink() or sha(path) != digest:
                    raise FeederError('COUNTED_PHASE_LOG_HASH_DIFFERS')
        accepted.add(token)
        if drain:
            undrain_ecores(**drain)
        if phase == 'prepare':
            review_peak(jobid, name)
        return True

    first = read(canary/'campaign.json')['module']
    modules = [first, *(m for m in rows if m != first)]
    def review_peak(jobid, name):
        if state['fv1e_admission'] == 'stopped':
            return
        peak = anon_peak(jobid, name, settings)
        if peak is not None and peak > settings['fv1e_prepare_memory_gib'] * 2**30 * 2 / 3:
            state['fv1e_admission'] = 'stopped'
            event('STOP', instance=name, id=jobid, anon_peak_bytes=peak, scope='new fv1e prepares only')
        elif peak is not None and state['fv1e_admission'] == 'pending':
            lanes = []
            for host in ('sparklina', 'sparky'):
                try:
                    observe('fv1e-' + host)
                except (OSError, subprocess.SubprocessError):
                    continue
                lanes.append(('fv1e-' + host, host))
            if lanes:
                state['lanes'] = lanes + fv1_lanes
                state['fv1e_admission'] = 'admitted'
                event('PLACEMENT', instance=name, id=jobid, anon_peak_bytes=peak, policy=policy)

    fv1_lanes = [('fv1', host) for host in ('sparklina', 'sparky') for _ in range(settings['fv1_max_prepares_per_spark'])]
    policy = ('Anon peak is a sampled LOWER bound, admitted with a margin. Prepare page cache is reclaimable and excluded. '
              'Per-job MemoryMax and the 108 GiB slice cap are the backstops.')
    if 'lanes' not in state:
        state.update(lanes=fv1_lanes, fv1e_admission='pending')
        event('PLACEMENT', policy=policy)
    canary_config = read(canary / 'campaign.json')
    if canary_config.get('templates'):
        for unit in canary_config['templates']:
            if unit['phase'] == 'prepare':
                latest = attempt_jobs(canary, unit['key'])[-1]
                identity = read(latest / 'prepare-instance.json')
                review_peak(identity['job_id'], identity['name'])
    else:
        review_peak((canary / 'prepare/jobid').read_text().strip(), 'fv1')
    pending = [(module, unit) for index in range(max(len(templates[m]) for m in modules))
               for module in modules for unit in templates[module][index:index + 1] if unit['phase'] == 'prepare']
    finished, refused, captures_running, launched = set(), set(), [], set()
    while pending or captures_running:
        previous_captures = list(captures_running)
        if state['fv1e_admission'] == 'pending' and all(m == first for m, _ in pending):
            state.update(fv1e_admission='stopped', fv1e_stop_reason='no valid anon sample in this wave')
            event('STOP', reason=state['fv1e_stop_reason'], scope='new fv1e prepares only')
        limits = {host: fv1e_limit(host) for host in ('sparklina', 'sparky')}
        e_lanes = [lane for lane in dict.fromkeys(map(tuple, state['lanes'])) if lane[0] != 'fv1'
                   and not (canary / ('pause-fv1e-' + lane[1])).exists() for _ in range(limits[lane[1]])]
        if state['fv1e_admission'] == 'admitted' and 'ep_tensor_check' not in state and e_lanes:
            probe = next(unit for module, unit in pending if module == first)
            if phase_job(first, probe, *e_lanes[0]):
                from experiments.numerics_rows.phase_artifact import tensor_subset_equal
                reference = canary / 'prepare/results/prepare'
                if canary_config.get('templates'):
                    original = next(u for u in canary_config['templates'] if u['phase'] == 'prepare'
                                    and (u['size'], u['case_set']) == (probe['size'], probe['case_set']))
                    reference = attempt_jobs(canary, original['key'])[-1] / 'prepare'
                state['ep_tensor_check'] = tensor_subset_equal(reference, attempts[first, probe['key']] / 'prepare')
                event('EP_TENSOR_EQUAL', **state['ep_tensor_check'])
            elif (attempts[first, probe['key']] / 'prepare-refused.json').exists():
                refused.add(first)
                state.update(fv1e_admission='stopped', fv1e_stop_reason='gate refused counted canary unit')
                event('STOP', reason=state['fv1e_stop_reason'], scope='new fv1e prepares only')
                state['ep_tensor_check'] = {'ran': False, 'reason': 'gate refused counted canary unit'}
                event('EP_NOT_RUN', **state['ep_tensor_check'])
        if state['fv1e_admission'] == 'stopped' and 'ep_tensor_check' not in state:
            state['ep_tensor_check'] = {'ran': False, 'reason': state.get('fv1e_stop_reason', 'no fv1e in this wave')}
            event('EP_NOT_RUN', **state['ep_tensor_check'])
        eligible = [(m, u) for m, u in pending if m != first]
        if 'ep_tensor_check' in state:
            held = [(m, u) for m, u in pending if m == first]
            eligible = eligible + held if state['fv1e_admission'] == 'stopped' else held + eligible
        lanes = (e_lanes if state['fv1e_admission'] == 'admitted' and 'ep_tensor_check' in state else []) + fv1_lanes
        started = []
        batch = eligible[:len(fv1_lanes)] if previous_captures else eligible
        for (module, unit), lane in zip(batch, lanes):
            submitted = phase_job(module, unit, *lane, collect=False)
            if not submitted and lane[0] != 'fv1' and not (attempts[module, unit['key']] / 'prepare-refused.json').exists():
                submitted = phase_job(module, unit, *fv1_lanes[0], collect=False)
            if submitted:
                started.append((module, unit))
            elif (attempts[module, unit['key']] / 'prepare-refused.json').exists():
                refused.add(module)
                pending.remove((module, unit))
        for module, unit in started:
            phase_job(module, unit)
            finished.add((module, unit['key'], 'prepare'))
            pending.remove((module, unit))
        for module in modules:
            for unit in templates[module]:
                members = unit.get('unit_keys', [unit['key']])
                if unit['phase'] != 'capture' or (module, unit['key']) in launched:
                    continue
                if all((module, member, 'prepare') in finished for member in members):
                    launched.add((module, unit['key']))
                    if phase_job(module, unit, collect=False):
                        captures_running.append((module, unit))
                    else:
                        refused.add(module)
        for module, unit in previous_captures:
            phase_job(module, unit)
            finished.add((module, unit['key'], 'capture'))
            captures_running.remove((module, unit))
        for module in modules:
            captures = [u for u in templates[module] if u['phase'] == 'capture']
            if module not in refused and (module, 'compare') not in finished and all(
                    (module, u['key'], 'capture') in finished for u in captures):
                compare = next(u for u in templates[module] if u['phase'] == 'compare')
                if phase_job(module, compare):
                    receipt = read(attempts[module, compare['key']] / 'compare/numerics_row.json')
                    event('ROW', module=module, outcome=receipt['row']['outcome'])
                    if not receipt['row'].get('finalization_finished') or receipt['row']['outcome'] != 'PASS':
                        raise FeederError('COUNTED_COMPARE_NOT_PASS')
                else:
                    refused.add(module)
                finished.add((module, 'compare'))
        if not started and pending and not any(item in eligible for item in pending):
            event('HOLD', reason='canary counted row awaits fv1e admission or pause release')
            time.sleep(settings['pending_poll_seconds'])
    if refused:
        event('UNITS_REFUSED', modules=sorted(refused))
        raise FeederError('COUNTED_UNITS_REFUSED')
    event('COMPLETE', modules=len(rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-dense-slot', action='store_true')
    parser.add_argument('--canary', type=Path)
    parser.add_argument('--bookings', type=Path)
    parser.add_argument('--result-directory', type=Path)
    parser.add_argument('--retry', action='append', default=[], metavar='UNIT=ATTEMPT')
    args = parser.parse_args()
    if args.wait_dense_slot:
        wait_dense_slot()
        raise SystemExit(0)
    if not all((args.canary, args.bookings, args.result_directory)):
        parser.error('provide --canary, --bookings and --result-directory')
    try:
        run(args.canary, args.bookings, dict(item.split('=', 1) for item in args.retry))
        (args.result_directory / 'last-error.txt').write_text('last_error=none\n')
    except Exception as error:
        failure_reason = str(error) if isinstance(error, FeederError) else type(error).__name__
        (args.result_directory / 'last-error.txt').write_text('last_error=' + failure_reason + '\n')
        print('Counted feeder stopped: ' + failure_reason, file=sys.stderr)
        raise SystemExit(3) from None
