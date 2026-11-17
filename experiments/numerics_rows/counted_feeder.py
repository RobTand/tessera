"""Run every counted module through Flux. Resume saved IDs without resubmission."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time


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
    while True:
        jobs = json.loads(subprocess.check_output(['flux', 'jobs', '--json', '--filter=pending'], text=True))['jobs']
        specs = [json.loads(subprocess.check_output(['flux', 'job', 'info', str(j['id']), 'jobspec'], text=True)) for j in jobs]
        if all(excludes_sparks(s.get('attributes', {}).get('system', {}).get('constraints', {})) for s in specs):
            return
        print('DEFER_DENSE_SUBMIT: pending aarch64 or portable job', flush=True)
        time.sleep(15)


def run(canary, bookings):
    canary = Path(canary)
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
        if phases['prepare']['memory_gib'] != 8:
            raise FeederError('DENSE_PREPARE_REQUIRES_8_GIB')
        if phases['capture']['minutes'] != 15:
            raise FeederError('DENSE_CAPTURE_REQUIRES_15_MINUTES')
    state_path = canary / 'counted-feeder-state.json'
    state = read(state_path) if state_path.exists() else {'bookings_sha256': sha(bookings), 'instance_sha256': sha('/mnt/shared/flux-migration/INSTANCE'), 'jobs': {}}
    if state['bookings_sha256'] != sha(bookings) or state['instance_sha256'] != sha('/mnt/shared/flux-migration/INSTANCE'):
        raise FeederError('RESUME_INPUTS_DIFFER')
    def event(kind, **fields):
        record = {'utc': datetime.now(timezone.utc).isoformat(), 'event': kind, **fields}
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
    for module, directory in rows.items():
        directory = Path(directory)
        settings = read(directory / 'campaign.json')
        if not settings.get('bound'):
            paths = directory / 'chain-inputs.json'
            paths.write_text(json.dumps(inputs[module], indent=2) + '\n')
            subprocess.run(['python3', str(directory / 'bind-inputs.py'), str(paths)], check=True)
        for phase in ('prepare', 'capture', 'compare'):
            job = directory / phase
            key = module + ':' + phase
            if (job / 'jobid').exists():
                jobid = (job / 'jobid').read_text().strip()
                if not re.fullmatch(r'f[1-9A-HJ-NP-Za-km-z]+', jobid):
                    raise FeederError('SUBMISSION_ID_AMBIGUOUS')
            else:
                shape = requested[module][phase]
                script = (job / 'job.sh').read_text()
                script = replace_once(r'--kill-after=20s \d+s ', f'--kill-after=20s {shape["minutes"] * 60 - 60}s ', script)
                submit = (job / 'submit-after-review.sh').read_text()
                submit = replace_once(r' -t\d+m ', f' -t{shape["minutes"]}m ', submit)
                submit = replace_once(r'slot/\[core=1;memory=\d+(;gpu=1)?\]',
                    'slot/[core=1;memory=' + str(shape['memory_gib']) + (';gpu=1]' if phase == 'capture' else ']'), submit)
                submit = replace_once(r'--urgency=\d+', '--urgency=8', submit) if '--urgency=' in submit else replace_once(r'flux submit\b', 'flux submit --urgency=8', submit)
                (job / 'job.sh').write_text(script)
                (job / 'submit-after-review.sh').write_text(submit)
                files = [*directory.glob('*.py'), *directory.glob('*/*.sh')]
                (directory / 'transport-preparation.sha256').write_text(''.join(f'{sha(p)}  {p}\n' for p in sorted(files)))
                wait_dense_slot()
                subprocess.run(['bash', str(job / 'submit-after-review.sh')], check=True)
                jobid = (job / 'jobid').read_text().strip()
            state['jobs'][key] = {'id': jobid, 'directory': str(job)}
            event('WAIT', module=module, phase=phase, id=jobid)
            subprocess.run(['flux', 'job', 'wait-event', jobid, 'clean'], check=True, stdout=subprocess.DEVNULL)
            native = int(subprocess.check_output(['flux', 'jobs', '-n', '-o', '{returncode}', jobid], text=True).strip())
            exits = {label: (job / path).read_text().strip() for label, path in (
                ('tool', 'tool-exit.txt'), ('payload', 'results/payload-exit.txt'), ('podman', 'results/podman-exit.txt'))}
            logs = {str(p): sha(p) for p in (job / 'stdout', job / 'stderr', job / 'results/payload.stdout', job / 'results/payload.stderr') if p.is_file()}
            event('END', module=module, phase=phase, id=jobid, native_exit=native, exits=exits, logs=logs)
            if native != 0 or any(value != '0' for value in exits.values()):
                raise FeederError('COUNTED_PHASE_FAILED')
            artifact = read(job / 'results' / phase / 'phase.json')
            if artifact.get('completion') is not True:
                raise FeederError('COUNTED_PHASE_UNFINISHED')
        receipt = read(directory / 'compare/results/numerics_row.json')
        event('ROW', module=module, outcome=receipt['row']['outcome'])
        if not receipt['row'].get('finalization_finished') or receipt['row']['outcome'] != 'PASS':
            raise FeederError('COUNTED_COMPARE_NOT_PASS')
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
        reason = str(error) if isinstance(error, FeederError) else type(error).__name__
        (args.result_directory / 'last-error.txt').write_text('last_error=' + reason + '\n')
        print('Counted feeder stopped: ' + reason, file=sys.stderr)
        raise SystemExit(3) from None
