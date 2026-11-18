"""Check the feeder with a stub gate and fake Flux on PATH."""
import hashlib
import json
import os
from pathlib import Path
import signal
import select
import fcntl
import subprocess
import sys

import pytest

from experiments.numerics_rows import counted_feeder as api, phase_instance, prepare_split


FAKE = r'''#!/usr/bin/env python3
import hashlib, json, os, pathlib, sys, time
root = pathlib.Path(os.environ['FAKE_ROOT'])
args = sys.argv[1:]
state_path = root / 'fake-state.json'
state = json.loads(state_path.read_text()) if state_path.exists() else {'jobs': {}}
def record(value):
    with (root / 'calls.jsonl').open('a') as stream:
        stream.write(json.dumps(value) + '\n')
def save():
    state_path.write_text(json.dumps(state))
if pathlib.Path(sys.argv[0]).name == 'stub-gate':
    name, template, urgency, spec_path = args
    table = json.loads((pathlib.Path(template) / 'table.json').read_text())
    spec = json.loads(pathlib.Path(spec_path).read_text())
    system = spec['attributes']['system']
    placement = table['instances'].get(name)
    reason = None
    if not placement: reason = 'instance'
    elif int(urgency) != table['urgency']: reason = 'urgency'
    elif spec['resources'] not in placement['shapes']: reason = 'shape'
    elif system['constraints'] not in placement['constraints']: reason = 'host'
    elif system['duration'] > placement['max_duration_s']: reason = 'duration'
    elif system['job']['name'] != table['job_name']: reason = 'name'
    elif system['dependencies'] != [{'scheme': 'singleton', 'value': ''}]: reason = 'dependencies'
    elif system['cwd'] != table['cwd']: reason = 'cwd'
    elif set(system['environment']) != set(table['env_names']): reason = 'environment'
    elif system['shell']['options'] != table['shell_options']: reason = 'shell'
    elif spec['tasks'][0]['command'][0] != str(pathlib.Path(template) / 'job.sh'): reason = 'command'
    elif len(spec['tasks'][0]['command']) != table['args'] + 1: reason = 'arguments'
    if (root / 'refuse').exists() and (root / 'refuse').read_text() in template:
        reason = 'review record'
    if reason:
        record({'action': 'refuse', 'template': template, 'reason': reason})
        print('REFUSED: ' + reason)
        sys.exit(3)
    record({'action': 'gate', 'instance': name, 'template': template, 'spec': spec, 'urgency': urgency})
    import subprocess
    result = subprocess.run(['flux', name, 'job', 'submit', spec_path], text=True, capture_output=True)
    print(result.stdout.strip())
    sys.exit(result.returncode)
name, *args = args
if args[:2] == ['submit', '--dry-run']:
    def option(prefix): return next(a[len(prefix):] for a in args if a.startswith(prefix))
    shape = option('--resources-shape=')
    count, fields = shape.removeprefix('slot=').split('/[')
    resources = [{'type': 'slot', 'count': int(count), 'label': 'task', 'with':
                  [{'type': key, 'count': int(value)} for key, value in (p.split('=') for p in fields.rstrip(']').split(';'))]}]
    command_index = next(i for i, a in enumerate(args) if a.endswith('/job.sh'))
    spec = {'version': 1, 'resources': resources, 'tasks': [{'command': args[command_index:], 'slot': 'task', 'count': {'per_slot': 1}}],
            'attributes': {'system': {'constraints': {'hostlist': [option('--requires=host:')]},
              'cwd': option('--cwd='), 'dependencies': [{'scheme': 'singleton', 'value': ''}],
              'duration': int(option('-t').rstrip('s')), 'job': {'name': option('--job-name=')},
              'environment': {a[6:].split('=', 1)[0]: '' for a in args if a.startswith('--env=') and a != '--env=-*'},
              'shell': {'options': {}}}}}
    if (root / 'mutate').exists():
        field = (root / 'mutate').read_text()
        if field == 'shape': spec['resources'][0]['with'][1]['count'] += 1
        elif field == 'urgency': pass
        elif field == 'dependencies': spec['attributes']['system'][field].append({'scheme': 'afterok', 'value': 'foreign'})
        elif field == 'duration': spec['attributes']['system'][field] += 1
        elif field == 'job': spec['attributes']['system'][field] = {'name': 'changed'}
        else: spec['attributes']['system'][field] = {} if field == 'environment' else 'changed'
    print(json.dumps(spec))
elif args[:2] == ['job', 'submit']:
    spec = json.loads(pathlib.Path(args[2]).read_text())
    jobid = 'fAAA' + 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'[len(state['jobs'])]
    command = spec['tasks'][0]['command']
    phase = pathlib.Path(command[0]).parent.name
    state['jobs'][jobid] = {'instance': name, 'phase': phase, 'arguments': command[1:]}
    save()
    for argument in command[1:]:
        job = pathlib.Path(argument) / phase
        job.mkdir()
        for label in ('tool', 'payload', 'podman'): (job / (label + '-exit.txt')).write_text('0')
        (job / 'phase.json').write_text(json.dumps({'completion': True}))
        (job / 'payload.stdout').write_text('payload log')
        (job / 'payload.stderr').write_text('')
        if phase == 'compare':
            (job / 'numerics_row.json').write_text(json.dumps({'row': {'outcome': 'PASS', 'finalization_finished': True}}))
        files = sorted(job.iterdir())
        (job / 'completion.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name + '\n' for p in files))
    record({'action': 'submit', 'instance': name, 'id': jobid, 'phase': phase, 'arguments': command[1:]})
    print(jobid)
elif args[:2] == ['job', 'status']:
    record({'action': 'status', 'instance': name, 'id': args[-1]})
    if (root / 'block').exists():
        with (root / 'wait-fifo').open('w') as stream: stream.write(args[-1] + '\n')
        while not (root / 'release').exists(): time.sleep(0.01)
    sys.exit(0)
elif args[:2] == ['job', 'info']:
    print('KVS stdout and stderr')
elif '--json' in args:
    print(json.dumps({'jobs': []}))
elif '{runtime:.6f}' in args:
    print('100')
elif '{returncode}' in args:
    print('0')
else:
    sys.exit('unexpected fake Flux call')
'''


def make_graph(tmp_path, monkeypatch, modules=('module-A', 'module-B')):
    from tests.test_numerics_unit_templates import templates
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name in ('flux', 'stub-gate'):
        executable = bin_dir / name
        executable.write_text(FAKE)
        executable.chmod(0o700)
    monkeypatch.setenv('PATH', str(bin_dir) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('FAKE_ROOT', str(tmp_path))
    canary = tmp_path / 'canary'
    (canary / 'prepare').mkdir(parents=True)
    rows, configs = {}, {}
    for module in modules:
        directory = tmp_path / module
        config, _ = templates(directory, module=module)
        config.update(module=module, bound=True)
        (directory / 'campaign.json').write_text(json.dumps(config))
        rows[module], configs[module] = str(directory), config
    settings = api.read(api.SETTINGS)
    settings.update(fleet_submit_gate=str(bin_dir / 'stub-gate'), fv1e_jobmem_row_dirs=[str(tmp_path / 'samples')])
    settings_path = tmp_path / 'settings.json'
    settings_path.write_text(json.dumps(settings))
    monkeypatch.setattr(api, 'SETTINGS', settings_path)
    monkeypatch.setattr(api, 'client', lambda name: ['flux', name])
    monkeypatch.setattr(api, 'observe', lambda name: {'name': name, 'generation': 1})
    monkeypatch.setattr(api, 'wait_dense_slot', lambda: None)
    monkeypatch.setattr(prepare_split, 'check_canary', lambda *args: None)
    (canary / 'campaign.json').write_text(json.dumps({'module': modules[0], 'tool_commit': 'tool'}))
    (canary / 'prepare/jobid').write_text('f111')
    (canary / 'counted-rows.json').write_text(json.dumps(rows))
    (canary / 'counted-chain-inputs.json').write_text(json.dumps(dict.fromkeys(rows, {})))
    bookings = tmp_path / 'bookings.json'
    bookings.write_text(json.dumps({module: {phase: {'memory_gib': 8 if phase == 'prepare' else 32,
        'minutes': 15 if phase == 'capture' else 360} for phase in ('prepare', 'capture', 'compare')} for module in rows}))
    saved = {'bookings_sha256': api.sha(bookings), 'settings_sha256': api.sha(settings_path), 'jobs': {},
             'lanes': [['fv1', 'sparklina'], ['fv1', 'sparky']], 'fv1e_admission': 'stopped',
             'ep_tensor_check': {'ran': False, 'reason': 'fixture'}}
    (canary / 'counted-feeder-state.json').write_text(json.dumps(saved))
    return canary, bookings, configs


def calls(graph):
    return [json.loads(line) for line in (graph[0].parent / 'calls.jsonl').read_text().splitlines()]


@pytest.mark.parametrize('ending', ('submit', 'refused', 'error'))
def test_feeder_keeps_environment_values_out_of_attempt_records(tmp_path, monkeypatch, ending):
    assert hasattr(api, 'NamedTemporaryFile'), 'feature absent: no environment values in attempt records'
    canary, bookings, _ = make_graph(tmp_path, monkeypatch, ('module-A',))
    original_dry_run, original_run = api.dry_run, api.subprocess.run
    checked, private_values = [], set()

    def private_spec(*args):
        document = json.loads(original_dry_run(*args))
        environment = document['attributes']['system']['environment']
        for field in environment:
            environment[field] = 'private-' + hashlib.sha256(field.encode()).hexdigest()
        private_values.update(value.encode() for value in environment.values())
        return json.dumps(document)

    def check_gate(command, **kwargs):
        if command[0] == api.read(api.SETTINGS)['fleet_submit_gate']:
            path = Path(command[4])
            raw = path.read_bytes()
            document = json.loads(raw)
            arguments = list(map(Path, document['tasks'][0]['command'][1:]))
            assert path.stat().st_mode & 0o777 == 0o600
            assert path.parent.stat().st_mode & 0o777 == 0o700
            assert path.parent.parent == canary
            assert all(not path.is_relative_to(attempt) for attempt in arguments)
            checked.append((path, arguments, Path(command[2]).name, hashlib.sha256(raw).hexdigest(),
                            sorted(document['attributes']['system']['environment'])))
            result = original_run(command, **kwargs)
            if ending == 'error':
                raise OSError('gate transport failed')
            return result
        return original_run(command, **kwargs)

    monkeypatch.setattr(api, 'dry_run', private_spec)
    monkeypatch.setattr(api.subprocess, 'run', check_gate)
    if ending == 'refused':
        (tmp_path / 'refuse').write_text('module-A/templates')
    if ending == 'submit':
        api.run(canary, bookings)
        assert any(row['action'] == 'submit' for row in calls((canary,)))
    else:
        with pytest.raises(OSError if ending == 'error' else api.FeederError,
                           match='gate transport failed' if ending == 'error' else 'COUNTED_UNITS_REFUSED'):
            api.run(canary, bookings)
    assert checked and private_values
    for path, arguments, phase, digest, names in checked:
        assert not path.exists() and not path.parent.exists()
        for attempt in arguments:
            intent = api.read(attempt / (phase + '-intent.json'))
            assert intent['jobspec_sha256'] == digest
            assert intent['env_names'] == names
    files = [path for path in (tmp_path / 'module-A/units').rglob('*') if path.is_file()]
    assert files
    assert all(value not in path.read_bytes() for path in files for value in private_values)


def test_feeder_uses_gate_table_and_producer_status(tmp_path, monkeypatch):
    assert hasattr(api, 'dry_run'), 'feature absent: gate table and producer status on every edge'
    canary, bookings, _ = graph = make_graph(tmp_path, monkeypatch)
    api.run(canary, bookings)
    trace = calls(graph)
    submits = [row for row in trace if row['action'] == 'submit']
    assert len(submits) == 18
    for index, row in enumerate(trace):
        if row['action'] != 'gate':
            continue
        table = api.read(Path(row['template']) / 'table.json')
        system = row['spec']['attributes']['system']
        assert system['dependencies'] == [{'scheme': 'singleton', 'value': ''}]
        assert int(row['urgency']) == table['urgency'] == 8
        assert system['cwd'] == table['cwd']
        assert system['job']['name'] == table['job_name']
        assert set(system['environment']) == set(table['env_names'])
        command = row['spec']['tasks'][0]['command']
        if Path(row['template']).name == 'prepare':
            continue
        producers = [p for p in submits if p['phase'] == ('prepare' if Path(row['template']).name == 'capture' else 'capture')
                     and (set(p['arguments']) & set(command[1:]) or Path(row['template']).name == 'compare'
                          and Path(p['arguments'][0]).parents[3] == Path(command[1]).parents[3])]
        assert producers
        for producer in producers:
            checks = [c for c in trace[:index] if c['action'] == 'status' and c['id'] == producer['id']]
            assert checks == [{'action': 'status', 'instance': producer['instance'], 'id': producer['id']}]


def test_feeder_prepares_other_modules_before_capture_wait(tmp_path, monkeypatch):
    assert hasattr(api, 'dry_run'), 'feature absent: module overlap without worker threads'
    graph = make_graph(tmp_path, monkeypatch, ('module-canary', 'module-A', 'module-B'))
    api.run(graph[0], graph[1])
    trace = calls(graph)
    capture = next(row for row in trace if row['action'] == 'submit' and row['phase'] == 'capture')
    wait = next(i for i, row in enumerate(trace) if row['action'] == 'status' and row['id'] == capture['id'])
    assert {Path(row['arguments'][0]).parents[3].name for row in trace[:wait]
            if row['action'] == 'submit' and row['phase'] == 'prepare'} == {'module-A', 'module-B'}


@pytest.mark.parametrize('field', ['shape', 'dependencies', 'duration', 'job', 'cwd', 'environment', 'urgency'])
def test_feeder_gate_refusal_stops_unit_and_keeps_other_units(tmp_path, monkeypatch, field):
    assert hasattr(api, 'dry_run'), 'feature absent: gate refusal stops a unit and records its reason'
    canary, bookings, configs = graph = make_graph(tmp_path, monkeypatch)
    # Refuse one reviewed template. Other units must still complete their captures.
    key = configs['module-A']['unit_keys'][0]
    (canary.parent / 'refuse').write_text('module-A/templates/' + key + '/prepare')
    with pytest.raises(api.FeederError, match='COUNTED_UNITS_REFUSED'):
        api.run(canary, bookings)
    trace = calls(graph)
    assert any(c['action'] == 'submit' and c['phase'] == 'compare' and 'module-B' in c['arguments'][0] for c in trace)
    assert not any(c['action'] == 'submit' and key in c['arguments'][0] and 'module-A' in c['arguments'][0] for c in trace)
    events = [json.loads(line) for line in (canary / 'counted-feeder.jsonl').read_text().splitlines()]
    assert any(e['event'] == 'REFUSED' and e['reason'] == 'REFUSED: review record' for e in events)
    # A changed jobspec also fails the same stub gate before a submit.
    template = canary.parent / 'module-B/templates' / configs['module-B']['unit_keys'][0] / 'prepare'
    argument = canary.parent / 'unclaimed'
    spec_path = canary.parent / 'changed-jobspec.json'
    (canary.parent / 'mutate').write_text(field)
    spec_path.write_text(api.dry_run(template, [argument], 'fv1', 'sparklina'))
    result = subprocess.run([api.read(api.SETTINGS)['fleet_submit_gate'], 'fv1', str(template), '9' if field == 'urgency' else '8', str(spec_path)], capture_output=True, text=True)
    assert result.returncode == 3 and result.stdout.startswith('REFUSED: ')
    assert not argument.exists()


def test_feeder_restart_reuses_finished_attempts_and_saved_ids(tmp_path, monkeypatch):
    assert hasattr(api, 'attempt_jobs'), 'feature absent: finished attempts and saved ids are reused'
    canary, bookings, _ = graph = make_graph(tmp_path, monkeypatch)
    api.run(canary, bookings)
    before = [c for c in calls(graph) if c['action'] == 'submit']
    # Recover one lost job-id file from saved state, with the original instance.
    next(canary.parent.glob('module-A/units/*/attempts/1/prepare-jobid')).unlink()
    api.run(canary, bookings)
    assert [c for c in calls(graph) if c['action'] == 'submit'] == before
    assert len(list(canary.parent.glob('module-*/units/*/attempts/1/prepare-jobid'))) == 8


@pytest.mark.parametrize('claim', ['intent', 'phase'])
def test_feeder_holds_ambiguous_attempt_for_audit(tmp_path, monkeypatch, claim):
    assert hasattr(api, 'attempt_jobs'), 'feature absent: claimed attempt without saved id requires audit'
    canary, bookings, configs = graph = make_graph(tmp_path, monkeypatch)
    key = configs['module-B']['unit_keys'][0]
    attempt = canary.parent / 'module-B/units' / key / 'attempts/1'
    attempt.mkdir(parents=True)
    if claim == 'intent':
        (attempt / 'prepare-intent.json').write_text('{}')
    else:
        (attempt / 'prepare').mkdir()
    with pytest.raises(api.FeederError, match='SUBMISSION_AMBIGUOUS_AUDIT_REQUIRED'):
        api.run(canary, bookings)
    assert not (attempt / 'prepare-jobid').exists()
    assert not attempt.with_name('2').exists()


def test_feeder_explicit_retry_is_next_attempt_and_keeps_fail(tmp_path, monkeypatch):
    assert hasattr(api, 'attempt_jobs'), 'feature absent: explicit retry is contiguous and keeps FAIL'
    canary, bookings, configs = graph = make_graph(tmp_path, monkeypatch)
    api.run(canary, bookings)
    key = configs['module-B']['unit_keys'][0]
    first = canary.parent / 'module-B/units' / key / 'attempts/1'
    fail = first / 'capture/numerics_row.json'
    fail.write_text('{"row":{"outcome":"FAIL"}}')
    digest = hashlib.sha256(fail.read_bytes()).hexdigest()
    api.run(canary, bookings, {key: 2})
    assert (first.with_name('2') / 'capture/phase.json').is_file()
    assert hashlib.sha256(fail.read_bytes()).hexdigest() == digest
    assert not first.with_name('3').exists()
    with pytest.raises(api.FeederError, match='RETRY_MUST_BE_NEXT_ATTEMPT'):
        api.run(canary, bookings, {key: 4})
    first.with_name('2').rename(first.with_name('3'))
    with pytest.raises(api.FeederError, match='ATTEMPT_IDS_HAVE_GAPS'):
        api.run(canary, bookings)


def test_feeder_rechecks_instance_and_bind_refuses_wrong_identity(tmp_path, monkeypatch):
    assert hasattr(api, 'dry_run'), 'feature absent: saved producer instance identity is checked'
    canary, bookings, _ = graph = make_graph(tmp_path, monkeypatch)
    api.run(canary, bookings)
    count = len([c for c in calls(graph) if c['action'] == 'submit'])
    monkeypatch.setattr(api, 'observe', lambda name: {'name': name, 'generation': 2})
    with pytest.raises(api.FeederError, match='RESUME_PHASE_INSTANCE_DIFFERS'):
        api.run(canary, bookings)
    assert len([c for c in calls(graph) if c['action'] == 'submit']) == count
    row = {}
    phase_instance.bind(row, 'prepare', {'name': 'fv1e-sparklina'})
    with pytest.raises(ValueError, match='cannot mix'):
        phase_instance.bind(row, 'prepare', {'name': 'fv1'})
    with pytest.raises(ValueError, match='cannot mix'):
        phase_instance.bind({}, 'capture', {'name': 'fv1e-sparklina'})


@pytest.mark.parametrize('name', ['fv1e-sparky', 'fv1e-sparklina'])
def test_canary_bind_keeps_fv1_placement(tmp_path, name):
    assert not hasattr(phase_instance, 'submit'), 'feature absent: canary bind stays separate from gate submit'
    from tests.test_numerics_unit_templates import templates
    _, directories = templates(tmp_path / 'canary', canary=True)
    assert all(list(api.read(directory / 'table.json')['instances']) == ['fv1'] for directory in directories)
    with pytest.raises(ValueError, match='cannot mix'):
        phase_instance.bind({}, 'prepare', {'name': name}, canary=True)


def test_generator_keeps_nine_gib_and_exact_dense_urgency(tmp_path):
    assert hasattr(api, 'dry_run'), 'feature absent: reviewed prepare table keeps nine GiB and urgency eight'
    from tests.test_numerics_unit_templates import templates
    _, directories = templates(tmp_path / 'module')
    prepares = [api.read(directory / 'table.json') for directory in directories if directory.name == 'prepare']
    assert len(prepares) == 4
    for table in prepares:
        assert table['urgency'] == 8
        for name in ('fv1e-sparky', 'fv1e-sparklina'):
            shape, = table['instances'][name]['shapes']
            assert next(r['count'] for r in shape[0]['with'] if r['type'] == 'memory') == 9
    assert all(api.read(d / 'table.json')['instances']['fv1']['max_duration_s'] == (36000 if '-gate-' in d.parent.name else 18000)
               for d in directories if d.name == 'capture')


def test_feeder_gate_keeps_booking_checks_and_dense_admission(tmp_path, monkeypatch):
    assert hasattr(api, 'dry_run'), 'feature absent: gate path keeps booking and dense admission checks'
    canary, bookings, _ = graph = make_graph(tmp_path, monkeypatch)
    waits = []
    monkeypatch.setattr(api, 'wait_dense_slot', lambda: waits.append(True))
    api.run(canary, bookings)
    assert len(waits) == 18
    values = api.read(bookings)
    values['module-A']['prepare']['memory_gib'] = 32
    bookings.write_text(json.dumps(values))
    with pytest.raises(ValueError, match='REQUIRES_8_GIB'):
        api.run(canary, bookings)
    values['module-A']['prepare']['memory_gib'] = 8
    values['module-A']['capture']['minutes'] = 60
    bookings.write_text(json.dumps(values))
    with pytest.raises(ValueError, match='REQUIRES_15_MINUTES'):
        api.run(canary, bookings)


def test_feeder_restart_in_blocking_wait_submits_no_duplicate(tmp_path, monkeypatch):
    assert hasattr(api, 'attempt_jobs'), 'feature absent: restart in native wait reuses the saved job'
    canary, bookings, _ = graph = make_graph(tmp_path, monkeypatch)
    root = canary.parent
    fifo = root / 'wait-fifo'
    os.mkfifo(fifo)
    (root / 'block').touch()
    driver = root / 'driver.py'
    driver.write_text('import sys\nfrom pathlib import Path\n'
        + 'sys.path.insert(0, ' + repr(str(Path(__file__).resolve().parents[1])) + ')\n'
        + 'from experiments.numerics_rows import counted_feeder as a, prepare_split as p\n'
        + 'a.SETTINGS = Path(' + repr(str(api.SETTINGS)) + ')\n'
        + "a.client = lambda name: ['flux', name]\na.observe = lambda name: {'name': name, 'generation': 1}\n"
        + 'a.wait_dense_slot = lambda: None\np.check_canary = lambda *args: None\n'
        + 'a.run(Path(' + repr(str(canary)) + '), Path(' + repr(str(bookings)) + '))\n')
    log = root / 'driver.log'
    with log.open('w') as output:
        process = subprocess.Popen([sys.executable, str(driver)], stdout=output, stderr=output, start_new_session=True)
        try:
            with fifo.open() as stream:
                saved_id = stream.readline().strip()
            before = [c for c in calls(graph) if c['action'] == 'submit']
            assert saved_id in {c['id'] for c in before}
            os.killpg(process.pid, signal.SIGKILL)
            assert process.wait(timeout=10) == -signal.SIGKILL
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    (root / 'release').touch()
    (root / 'block').unlink()
    result = subprocess.run([sys.executable, str(driver)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    submits = [c for c in calls(graph) if c['action'] == 'submit']
    assert submits[:len(before)] == before
    assert len({c['id'] for c in submits}) == len(submits) == 18
    assert len([c for c in calls(graph) if c['action'] == 'status' and c['id'] == saved_id]) == 2


def test_feeder_crossing_uses_saved_producer_instance(tmp_path, monkeypatch):
    assert hasattr(api, 'dry_run'), 'feature absent: crossing waits on the producing instance'
    canary, bookings, _ = graph = make_graph(tmp_path, monkeypatch)
    state = api.read(canary / 'counted-feeder-state.json')
    state.update(fv1e_admission='admitted', lanes=[['fv1e-sparklina', 'sparklina']])
    (canary / 'counted-feeder-state.json').write_text(json.dumps(state))
    (canary / 'fv1e-limit-sparklina').write_text('2')
    api.run(canary, bookings)
    trace = calls(graph)
    producers = [c for c in trace if c['action'] == 'submit' and c['phase'] == 'prepare' and c['instance'] == 'fv1e-sparklina']
    assert producers
    for row in trace:
        if row['action'] == 'gate' and row['instance'] == 'fv1e-sparklina':
            assert next(r['count'] for r in row['spec']['resources'][0]['with'] if r['type'] == 'memory') == 9
    for producer in producers:
        consumer = next(i for i, c in enumerate(trace) if c['action'] == 'gate' and Path(c['template']).name == 'capture'
                        and set(c['spec']['tasks'][0]['command'][1:]) & set(producer['arguments']))
        assert [c for c in trace[:consumer] if c['action'] == 'status' and c['id'] == producer['id']] == [
            {'action': 'status', 'instance': 'fv1e-sparklina', 'id': producer['id']}]
        assert trace[consumer]['spec']['attributes']['system']['dependencies'] == [{'scheme': 'singleton', 'value': ''}]


def test_feeder_capture_group_uses_ordered_arguments_and_recovers_id(tmp_path, monkeypatch):
    assert hasattr(api, 'attempt_jobs'), 'feature absent: grouped capture saves and reuses every member job id'
    import shutil
    canary, bookings, configs = graph = make_graph(tmp_path, monkeypatch)
    config = configs['module-A']
    directory = canary.parent / 'module-A'
    shutil.rmtree(directory / 'templates')
    settings = api.read(directory / 'feeder-settings.json')
    settings['capture_group_canary'] = {'cores_per_capture': 1, 'anon_peak_gib_per_capture': 2, 'margin_gib': 1, 'max_k': 2}
    (directory / 'feeder-settings.json').write_text(json.dumps(settings))
    keys = config['unit_keys'][:2]
    config['capture_groups'] = [{'key': 'group-A', 'unit_keys': keys}]
    prepare_split.write_templates(directory, config, directory / 'repository.git')
    prepare_split.bind_templates(directory, config)
    api.run(canary, bookings)
    grouped, = [c for c in calls(graph) if c['action'] == 'gate' and '/group-A/' in c['template']]
    arguments = grouped['spec']['tasks'][0]['command'][1:]
    assert arguments == [str(directory / 'units' / key / 'attempts/1') for key in keys]
    ids = [Path(argument) / 'capture-jobid' for argument in arguments]
    assert len({p.read_text() for p in ids}) == 1
    ids[0].unlink()
    before = [c for c in calls(graph) if c['action'] == 'submit']
    api.run(canary, bookings)
    assert [c for c in calls(graph) if c['action'] == 'submit'] == before
    assert ids[0].read_text() == ids[1].read_text()


@pytest.mark.parametrize('fault', ['native', 'tool', 'payload', 'podman', 'log', 'artifact', 'instance'])
def test_feeder_keeps_predecessor_evidence_checks(tmp_path, monkeypatch, fault):
    assert hasattr(api, 'dry_run'), 'feature absent: native exits and log hashes gate dependent submit'
    canary, bookings, _ = graph = make_graph(tmp_path, monkeypatch)
    api.run(canary, bookings)
    job = next(canary.parent.glob('module-B/units/*/attempts/1/prepare'))
    if fault in ('tool', 'payload', 'podman'):
        (job / (fault + '-exit.txt')).write_text('1')
    elif fault == 'log':
        (job / 'payload.stdout').write_text('changed log')
    elif fault == 'artifact':
        (job / 'phase.json').write_text('{"completion":false}')
    elif fault == 'instance':
        (job / 'instance.json').write_text('{"name":"wrong"}')
    else:
        original = subprocess.run
        def failed_status(command, **kwargs):
            result = original(command, **kwargs)
            if 'status' in command:
                result.returncode = 125
            return result
        monkeypatch.setattr(api.subprocess, 'run', failed_status)
    before = [c for c in calls(graph) if c['action'] == 'submit']
    with pytest.raises(api.FeederError, match='COUNTED_PHASE_|RESUME_PHASE_INSTANCE_DIFFERS'):
        api.run(canary, bookings)
    assert [c for c in calls(graph) if c['action'] == 'submit'] == before
    assert not list(canary.parent.rglob('lead-cause.json'))


def test_worker_blocks_until_submit_metadata_is_complete(tmp_path, monkeypatch):
    assert not hasattr(phase_instance, 'submit'), 'feature absent: worker waits for submit metadata under intent lock'
    job = tmp_path / '1'
    job.mkdir()
    intent = job / 'prepare-intent.json'
    intent.write_text('{}')
    driver = tmp_path / 'worker.py'
    driver.write_text('import sys, fcntl, json\nfrom pathlib import Path\n'
        + 'sys.path.insert(0, ' + repr(str(Path(__file__).resolve().parents[1])) + ')\n'
        + 'from experiments.numerics_rows import phase_instance as a\n'
        + 'a.INSTANCE = Path(' + repr(str(tmp_path / 'INSTANCE')) + ')\n'
        + "a.observe = lambda *args: {'name': 'fv1', 'generation': 1}\n"
        + "a.query = lambda *args: 'f123'\noriginal = fcntl.flock\n"
        + "def wait(stream, mode):\n    print('waiting', flush=True)\n    original(stream, mode)\n"
        + 'a.fcntl.flock = wait\n'
        + 'print(json.dumps(a.check(Path(' + repr(str(job)) + "), 'prepare')))\n")
    (tmp_path / 'INSTANCE').write_text(str(tmp_path / 'native'))
    monkeypatch.setenv('FLUX_JOB_ID', 'f123')
    identity = {'name': 'fv1', 'generation': 1, 'job_id': 'f123'}
    with intent.open() as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        process = subprocess.Popen([sys.executable, str(driver)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            assert select.select([process.stdout], [], [], 10)[0], 'worker did not reach the blocking lock'
            assert process.stdout.readline().strip() == 'waiting'
            phase_instance.save(job / 'prepare-instance.json', identity)
            fcntl.flock(stream, fcntl.LOCK_UN)
            output, errors = process.communicate(timeout=10)
            assert process.returncode == 0, errors
            assert json.loads(output) == identity
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)


def test_feeder_canary_gate_refusal_keeps_ecore_admission_closed(tmp_path, monkeypatch):
    assert hasattr(api, 'dry_run'), 'feature absent: canary gate refusal keeps E-core admission closed'
    canary, bookings, configs = graph = make_graph(tmp_path, monkeypatch)
    state = api.read(canary / 'counted-feeder-state.json')
    state.pop('ep_tensor_check')
    state.update(fv1e_admission='admitted', lanes=[['fv1e-sparklina', 'sparklina']])
    (canary / 'counted-feeder-state.json').write_text(json.dumps(state))
    (canary / 'fv1e-limit-sparklina').write_text('2')
    key = configs['module-A']['unit_keys'][0]
    (canary.parent / 'refuse').write_text('module-A/templates/' + key + '/prepare')
    with pytest.raises(api.FeederError, match='COUNTED_UNITS_REFUSED'):
        api.run(canary, bookings)
    assert api.read(canary / 'counted-feeder-state.json')['fv1e_admission'] == 'stopped'
    assert all(c['instance'] == 'fv1' for c in calls(graph) if c['action'] == 'submit')
    assert any(c['action'] == 'submit' and c['phase'] == 'compare' for c in calls(graph))
