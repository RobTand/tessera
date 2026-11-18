"""Run every counted module through Flux. Resume saved IDs without resubmission."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

if __package__:
    from .phase_instance import client, observe
else:
    from instance import client, observe

FXE = '/mnt/shared/flux-migration/lead-1114/tools/fxe.sh'
SETTINGS = Path(__file__).with_name('feeder-settings.json')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


class FeederError(ValueError):
    pass


def replace_once(pattern, value, text):
    text, count = re.subn(pattern, value, text)
    if count != 1:
        raise FeederError("BOOKING_REQUIRES_EXACTLY_ONE_MATCH")
    return text


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


def run(canary, bookings):
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

    def phase_job(module, phase, name='fv1', host='sparklina', collect=True):
        directory = rows[module]
        directory = Path(directory)
        config = read(directory / 'campaign.json')
        if not config.get('bound'):
            paths = directory / 'chain-inputs.json'
            paths.write_text(json.dumps(inputs[module], indent=2) + '\n')
            subprocess.run(['python3', str(directory / 'bind-inputs.py'), str(paths)], check=True)
        job = directory / phase
        drain = next((v.get('fv1e_drain') for v in state['jobs'].values() if v['directory'] == str(job)), None)
        if (job / 'jobid').exists():
            jobid = (job / 'jobid').read_text().strip()
            name = read(job/'instance.json')['name']
            if not re.fullmatch(r'f[1-9A-HJ-NP-Za-km-z]+', jobid):
                raise FeederError('SUBMISSION_ID_AMBIGUOUS')
        else:
            shape = dict(requested[module][phase])
            if name != 'fv1':
                shape['memory_gib'] = settings['fv1e_prepare_memory_gib']
            (job/'instance-name.txt').write_text(name + '\n')
            script = (job / 'job.sh').read_text()
            script = replace_once(r'--kill-after=20s \d+s ', f'--kill-after=20s {shape["minutes"] * 60 - 60}s ', script)
            submit = (job / 'submit-after-review.sh').read_text()
            submit = replace_once(r'--requires=host:[\w-]+', '--requires=host:' + host, submit)
            submit = replace_once(r' -t\d+m ', f' -t{shape["minutes"]}m ', submit)
            submit = replace_once(r'slot/\[core=1;memory=\d+(;gpu=1)?\]',
                'slot/[core=1;memory=' + str(shape['memory_gib']) + (';gpu=1]' if phase == 'capture' else ']'), submit)
            urgency = '--urgency=' + str(settings['urgency'])
            submit = replace_once(r'--urgency=\d+', urgency, submit) if '--urgency=' in submit else replace_once(r'flux submit\b', 'flux submit ' + urgency, submit)
            (job / 'job.sh').write_text(script)
            (job / 'submit-after-review.sh').write_text(submit)
            files = [*directory.glob('*.py'), *directory.glob('*/*.sh')]
            (directory / 'transport-preparation.sha256').write_text(''.join(f'{sha(p)}  {p}\n' for p in sorted(files)))
            if name == 'fv1':
                wait_dense_slot()
            drain = drain_ecores_before_submit(shape['memory_gib'], submit, 'tessera-1717:' + str(job))
            if name != 'fv1':
                active = json.loads(subprocess.check_output(
                    [*client(name), 'jobs', '--json', '--filter=pending,running'], text=True,
                    timeout=settings['client_timeout_seconds']))['jobs']
                limit = fv1e_limit(host)
                if (state['fv1e_admission'] != 'admitted' or len(active) >= limit
                        or (canary/('pause-fv1e-' + host)).exists()):
                    return False
            subprocess.run(['bash', str(job / 'submit-after-review.sh')], check=True)
            jobid = (job / 'jobid').read_text().strip()
        bound = read(job/'instance.json')
        if bound['name'] != name or bound['job_id'] != jobid or observe(name) != {k: v for k, v in bound.items() if k != 'job_id'}:
            raise FeederError('RESUME_PHASE_INSTANCE_DIFFERS')
        key = name + ':' + jobid
        state['jobs'][key] = {'id': jobid, 'instance': bound, 'directory': str(job), 'fv1e_drain': drain}
        event('WAIT', module=module, phase=phase, id=jobid)
        if not collect:
            return True
        subprocess.run([*client(name), 'job', 'wait-event', jobid, 'clean'], check=True, stdout=subprocess.DEVNULL)
        native = int(subprocess.check_output([*client(name), 'jobs', '-n', '-o', '{returncode}', jobid], text=True).strip())
        exits = {label: (job / path).read_text().strip() for label, path in (
            ('tool', 'tool-exit.txt'), ('payload', 'results/payload-exit.txt'), ('podman', 'results/podman-exit.txt'))}
        logs = {str(p): sha(p) for p in (job / 'stdout', job / 'stderr', job / 'results/payload.stdout', job / 'results/payload.stderr') if p.is_file()}
        event('END', module=module, phase=phase, id=jobid, native_exit=native, exits=exits, logs=logs)
        if native != 0 or any(value != '0' for value in exits.values()):
            raise FeederError('COUNTED_PHASE_FAILED')
        artifact = read(job / 'results' / phase / 'phase.json')
        if artifact.get('completion') is not True:
            raise FeederError('COUNTED_PHASE_UNFINISHED')
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
    review_peak((canary/'prepare/jobid').read_text().strip(), 'fv1')
    pending = list(modules)
    while pending:
        # A module leaves pending only after its prepare was collected and checked.
        if state['fv1e_admission'] == 'pending' and pending == [first]:
            state.update(fv1e_admission='stopped', fv1e_stop_reason='no valid anon sample in this wave')
            event('STOP', reason=state['fv1e_stop_reason'], scope='new fv1e prepares only')
        limits = {host: fv1e_limit(host) for host in ('sparklina', 'sparky')}
        e_lanes = [lane for lane in dict.fromkeys(map(tuple, state['lanes'])) if lane[0] != 'fv1'
                   and not (canary/('pause-fv1e-' + lane[1])).exists() for _ in range(limits[lane[1]])]
        if state['fv1e_admission'] == 'admitted' and 'ep_tensor_check' not in state and e_lanes:
            if not phase_job(first, 'prepare', *e_lanes[0]):
                continue
            from experiments.numerics_rows.phase_artifact import tensor_subset_equal
            state['ep_tensor_check'] = tensor_subset_equal(canary/'prepare/results/prepare', Path(rows[first])/'prepare/results/prepare')
            event('EP_TENSOR_EQUAL', **state['ep_tensor_check'])
        if state['fv1e_admission'] == 'stopped' and 'ep_tensor_check' not in state:
            state['ep_tensor_check'] = {'ran': False, 'reason': state.get('fv1e_stop_reason', 'no fv1e in this wave')}
            event('EP_NOT_RUN', **state['ep_tensor_check'])
        eligible = [m for m in pending if m != first]
        if 'ep_tensor_check' in state and first in pending:
            eligible = eligible + [first] if state['fv1e_admission'] == 'stopped' else [first] + eligible
        # Collect saved jobs before new submits. A resume keeps their bound instances.
        saved = [m for m in eligible if (Path(rows[m])/'prepare/jobid').exists()]
        lanes = (e_lanes if state['fv1e_admission'] == 'admitted' and 'ep_tensor_check' in state else []) + fv1_lanes
        wave = [(m, ('fv1', 'sparklina')) for m in saved] if saved else list(zip(eligible, lanes))
        if not wave:
            event('HOLD', reason='canary counted row awaits fv1e admission or pause release')
            time.sleep(settings['pending_poll_seconds'])
            review_peak((canary/'prepare/jobid').read_text().strip(), 'fv1')
            for module in modules:
                job = Path(rows[module])/'prepare'
                if (job/'jobid').exists() and read(job/'instance.json')['name'] == 'fv1':
                    review_peak((job/'jobid').read_text().strip(), 'fv1')
            continue
        started = [(module, lane) for module, lane in wave if phase_job(module, 'prepare', *lane, collect=False)]
        for module, (name, host) in started:
            # The first fv1e prepare was collected for the E/P check above.
            if module != first or state['ep_tensor_check'].get('ran') is False:
                phase_job(module, 'prepare', name, host)
            for phase in ('capture', 'compare'):
                phase_job(module, phase)
            receipt = read(Path(rows[module])/'compare/results/numerics_row.json')
            event('ROW', module=module, outcome=receipt['row']['outcome'])
            if not receipt['row'].get('finalization_finished') or receipt['row']['outcome'] != 'PASS':
                raise FeederError('COUNTED_COMPARE_NOT_PASS')
            pending.remove(module)
    event('COMPLETE', modules=len(rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-dense-slot', action='store_true')
    parser.add_argument('--canary', type=Path)
    parser.add_argument('--bookings', type=Path)
    parser.add_argument('--result-directory', type=Path)
    args = parser.parse_args()
    if args.wait_dense_slot:
        wait_dense_slot()
        raise SystemExit(0)
    if not all((args.canary, args.bookings, args.result_directory)):
        parser.error('provide --canary, --bookings and --result-directory')
    try:
        run(args.canary, args.bookings)
        (args.result_directory / 'last-error.txt').write_text('last_error=none\n')
    except Exception as error:
        failure_reason = str(error) if isinstance(error, FeederError) else type(error).__name__
        (args.result_directory / 'last-error.txt').write_text('last_error=' + failure_reason + '\n')
        print('Counted feeder stopped: ' + failure_reason, file=sys.stderr)
        raise SystemExit(3) from None
