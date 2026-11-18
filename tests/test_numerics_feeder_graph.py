"""Check the feeder with a stub gate and fake Flux."""
import hashlib
import inspect
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
from experiments.numerics_rows.launch_state import json_digest


def fake_command(root, command):
    executable, name, *args = command
    state_path = root / 'fake-state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    def record(value):
        with (root / 'calls.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n')
    if Path(executable).name == 'stub-gate':
        template, urgency, spec_path = args
        table = json.loads((Path(template) / 'table.json').read_text())
        spec = json.loads(Path(spec_path).read_text())
        system = spec['attributes']['system']
        placement = table['instances'].get(name)
        reason = None
        if (set(table) != {'schema', 'instances', 'job_name', 'args', 'max_attempts', 'urgency', 'cwd', 'env_names'}
                or not set(table['instances']) <= {'fv1', 'fv1e-sparky', 'fv1e-sparklina'}
                or any(set(entry) != {'constraints', 'shapes', 'max_duration_s'} for entry in table['instances'].values())):
            reason = 'table'
        elif not placement: reason = 'instance'
        elif int(urgency) != table['urgency']: reason = 'urgency'
        elif spec['resources'] not in placement['shapes']: reason = 'shape'
        elif system['constraints'] not in placement['constraints']: reason = 'host'
        elif system['duration'] > placement['max_duration_s']: reason = 'duration'
        elif system['job']['name'] != table['job_name']: reason = 'name'
        elif system['dependencies'] != [{'scheme': 'singleton', 'value': ''}]: reason = 'dependencies'
        elif system['cwd'] != table['cwd']: reason = 'cwd'
        elif set(system['environment']) != set(table['env_names']): reason = 'environment'
        elif set(system) != {'constraints', 'cwd', 'dependencies', 'duration', 'environment', 'job'}: reason = 'system keys'
        elif spec['tasks'][0]['command'][0] != str(Path(template) / 'job.sh'): reason = 'command'
        elif len(spec['tasks'][0]['command']) != table['args'] + 1: reason = 'arguments'
        if (root / 'refuse').exists() and (root / 'refuse').read_text() in template:
            reason = 'review record'
        if reason:
            record({'action': 'refuse', 'template': template, 'reason': reason})
            return 3, 'REFUSED: ' + reason + '\n'
        record({'action': 'gate', 'instance': name, 'template': template, 'spec': spec, 'urgency': urgency})
        jobid = 'fAAA' + 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'[len(state)]
        command = spec['tasks'][0]['command']
        phase = Path(command[0]).parent.name
        state[jobid] = 0
        rules = json.loads((root / 'job-outcomes.json').read_text()) if (root / 'job-outcomes.json').exists() else []
        empty_digest = hashlib.sha256(b'').hexdigest()
        for attempt in map(Path, command[1:]):
            job, module, key = attempt / phase, attempt.parents[3].name, attempt.parents[1].name
            rule = next((r for r in rules if r['module'] == module and r['phase'] == phase
                         and int(attempt.name) in r.get('attempts', [1]) and r.get('key', key) == key), {})
            rc = rule.get('exit', 0)
            state[jobid] = max(state[jobid], rc)
            job.mkdir()
            for label in ('tool', 'payload', 'podman'): (job / (label + '-exit.txt')).write_text(str(rc))
            document = {'schema': 'tessera.dense_row_phase.v2', 'phase': phase, 'completion': rule.get('completion', True),
                        'identities': {}, 'scopes': [{'module': module, 'unit_key': key}],
                        'case_set': 'diagnostics' if '-diagnostics-' in str(attempt) else 'gate', 'ordinal_start': 0,
                        'streams': {'cases': {'count': 0, 'sha256': empty_digest}}, 'case_ids_sha256': empty_digest}
            (job / 'phase.json').write_text(json.dumps(document))
            if phase == 'capture' and not rule.get('missing_member_exit'):
                (job / 'exit.json').write_text(json.dumps({'exit_status': rc}))
            (job / 'cases.jsonl').touch()
            (job / 'payload.stdout').write_text('payload log')
            (job / 'payload.stderr').write_text('')
            if phase == 'compare' or 'outcome' in rule:
                (job / 'numerics_row.json').write_text(json.dumps({'row': {'outcome': rule.get('outcome', 'PASS'),
                    'finalization_finished': rule.get('finalized', phase == 'compare')}}))
            if rule.get('completion') is not False:
                (job / 'completion.sha256').write_text(''.join(
                    hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name + '\n' for p in sorted(job.iterdir())))
        state_path.write_text(json.dumps(state))
        record({'action': 'submit', 'instance': name, 'id': jobid, 'phase': phase, 'arguments': command[1:]})
        return 0, jobid + '\n'
    elif args[:2] == ['submit', '--dry-run']:
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
                  'environment': {a[6:].split('=', 1)[0]: '' for a in args if a.startswith('--env=') and a != '--env=-*'}}}}
        if '--rlimit=-*' not in args or any(a.startswith('-o') for a in args):
            spec['attributes']['system']['shell'] = {'options': {'rlimit': {'cpu': 1}}}
        if (root / 'mutate').exists():
            field = (root / 'mutate').read_text()
            if field == 'shape': spec['resources'][0]['with'][1]['count'] += 1
            elif field == 'dependencies': spec['attributes']['system'][field].append({'scheme': 'afterok', 'value': 'foreign'})
            elif field == 'duration': spec['attributes']['system'][field] += 1
            elif field == 'job': spec['attributes']['system'][field] = {'name': 'changed'}
            elif field != 'urgency': spec['attributes']['system'][field] = {} if field == 'environment' else 'changed'
        return 0, json.dumps(spec) + '\n'
    elif args[:2] == ['job', 'status']:
        record({'action': 'status', 'instance': name, 'id': args[-1]})
        if (root / 'block').exists():
            with (root / 'wait-fifo').open('w') as stream: stream.write(args[-1] + '\n')
            signal.pause()
        return state[args[-1]], ''
    elif args[:2] == ['job', 'info']:
        return 0, 'KVS stdout and stderr\n'
    elif '--json' in args:
        return 0, '{"jobs": []}\n'
    elif '{runtime:.6f}' in args:
        return 0, '100\n'
    elif '{returncode}' in args:
        return 0, str(state[args[-1]]) + '\n'
    raise ValueError('unexpected fake Flux call')


def make_graph(tmp_path, monkeypatch, modules=('module-A', 'module-B'), *, external=False):
    from tests.test_numerics_unit_templates import templates
    gate = str(tmp_path / 'stub-gate')
    if external:
        script = ('#!/usr/bin/env python3\nimport hashlib, json, signal, sys\nfrom pathlib import Path\n'
                  + inspect.getsource(fake_command)
                  + "code, output = fake_command(Path(__file__).parent, sys.argv)\n"
                  + "sys.stdout.write(output)\nsys.exit(code)\n")
        for executable in (tmp_path / 'flux', Path(gate)):
            executable.write_text(script)
            executable.chmod(0o700)
        monkeypatch.setenv('PATH', str(tmp_path), prepend=os.pathsep)
    else:
        run = subprocess.run

        def fake_run(command, **kwargs):
            if command[0] not in ('flux', gate):
                return run(command, **kwargs)
            code, output = fake_command(tmp_path, command)
            captured = kwargs.get('capture_output') or kwargs.get('stdout') == subprocess.PIPE
            stdout = (output if kwargs.get('text') else output.encode()) if captured else None
            stderr = ('' if kwargs.get('text') else b'') if kwargs.get('capture_output') else None
            result = subprocess.CompletedProcess(command, code, stdout, stderr)
            if kwargs.get('check'):
                result.check_returncode()
            return result

        monkeypatch.setattr(subprocess, 'run', fake_run)
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
    settings.update(fleet_submit_gate=gate, fv1e_jobmem_row_dirs=[str(tmp_path / 'samples')], pending_poll_seconds=0)
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


def feeder_events(canary):
    return [json.loads(line) for line in (canary / 'counted-feeder.jsonl').read_text().splitlines()]


@pytest.mark.parametrize('outcome,finalized', [('FAIL', True), ('STOPPED', True), ('FAIL', False)])
def test_feeder_continues_after_compare_exit_three(tmp_path, monkeypatch, outcome, finalized):
    assert hasattr(api, 'PhaseJobFailed'), 'feature absent: finalized compare exit 3 lets the wave continue'
    canary, bookings, _ = graph = make_graph(tmp_path, monkeypatch, ('module-A', 'module-B', 'module-C'))
    (tmp_path / 'job-outcomes.json').write_text(json.dumps([
        {'module': module, 'phase': 'compare', 'exit': 3, 'outcome': outcome, 'finalized': finalized}
        for module in ('module-B', 'module-C')]))
    with pytest.raises(api.FeederError, match='COUNTED_PHASE_.*module-B.*module-C'):
        api.run(canary, bookings)
    trace = feeder_events(canary)
    rows = [e for e in trace if e['event'] == 'ROW']
    assert {e['module'] for e in rows} == ({'module-A', 'module-B', 'module-C'} if finalized else {'module-A'})
    assert all(e['outcome'] == (outcome if e['module'] != 'module-A' else 'PASS') for e in rows)
    first = [e for e in trace if e['event'] == 'FIRST_FAIL_ROW']
    assert len(first) == int(finalized and outcome == 'FAIL')
    if first:
        assert trace.index(first[0]) + 1 == next(i for i, e in enumerate(trace) if e['event'] == 'ROW' and e['outcome'] == 'FAIL')
        assert any(e['event'] == 'SUBMIT_INTENT' for e in trace[trace.index(first[0]) + 1:])
    before = [c for c in calls(graph) if c['action'] == 'submit']
    with pytest.raises(api.FeederError, match='COUNTED_PHASE_'):
        api.run(canary, bookings)
    assert [c for c in calls(graph) if c['action'] == 'submit'] == before
    assert sum(e['event'] == 'FIRST_FAIL_ROW' for e in feeder_events(canary)) == len(first)


@pytest.mark.parametrize('phase', ['prepare', 'capture'])
@pytest.mark.parametrize('failures', [1, 2])
def test_feeder_retries_a_job_once_and_holds_its_module(tmp_path, monkeypatch, phase, failures):
    assert hasattr(api, 'PhaseJobFailed'), 'feature absent: one automatic job retry then a module hold'
    canary, bookings, configs = graph = make_graph(tmp_path, monkeypatch)
    key = configs['module-B']['unit_keys'][0]
    (tmp_path / 'job-outcomes.json').write_text(json.dumps([
        {'module': 'module-B', 'key': key, 'phase': phase, 'exit': 1,
         'completion': False, 'attempts': list(range(1, failures + 1))}]))
    with pytest.raises(api.FeederError, match='COUNTED_PHASE_.*module-B'):
        api.run(canary, bookings)
    trace = calls(graph)
    unit_submits = [c for c in trace if c['action'] == 'submit' and c['phase'] == phase and key in c['arguments'][0]]
    assert [Path(c['arguments'][0]).name for c in unit_submits] == ['1', '2']
    assert any(e['event'] == 'ROW' and e['module'] == 'module-A' for e in feeder_events(canary))
    assert sum(e['event'] == 'AUTO_RETRY' for e in feeder_events(canary)) == 1
    assert any(e['event'] == 'HOLD_FOR_LEAD' and e['phase'] == (phase if failures == 2 else 'compare') for e in feeder_events(canary))
    assert not any(c['action'] == 'submit' and c['phase'] == 'compare' and 'module-B' in c['arguments'][0] for c in trace)
    assert not list((tmp_path / 'module-B/units').glob('compare-*/attempts/*'))
    before = [c for c in trace if c['action'] == 'submit']
    with pytest.raises(api.FeederError, match='COUNTED_PHASE_'):
        api.run(canary, bookings)
    assert [c for c in calls(graph) if c['action'] == 'submit'] == before
    attempt = tmp_path / 'module-B/units' / key / 'attempts/1'
    for number in range(1, failures + 1):
        (attempt.with_name(str(number)) / 'lead-cause.json').write_text(json.dumps(
            {'unit': key, 'attempt': number, 'cause': 'The lead confirmed job termination.'}))
    api.run(canary, bookings, {key: 3} if failures == 2 else None)
    compare = next((tmp_path / 'module-B/units').glob('compare-*/attempts/1'))
    pairs = api.read(compare / 'compare-inputs.json')
    assert pairs[key] == str(attempt.with_name('3' if failures == 2 else '2'))
    assert (compare / 'compare/numerics_row.json').is_file()
    assert not compare.with_name('2').exists()


@pytest.mark.parametrize('phase', ['prepare', 'capture'])
def test_feeder_does_not_retry_a_finding(tmp_path, monkeypatch, phase):
    assert hasattr(api, 'PhaseJobFailed'), 'feature absent: a finding holds its module without a job retry'
    canary, bookings, configs = graph = make_graph(tmp_path, monkeypatch)
    key = configs['module-B']['unit_keys'][0]
    (tmp_path / 'job-outcomes.json').write_text(json.dumps([
        {'module': 'module-B', 'key': key, 'phase': phase, 'exit': 3, 'outcome': 'FAIL', 'finalized': False}]))
    with pytest.raises(api.FeederError, match='COUNTED_PHASE_FINDING'):
        api.run(canary, bookings)
    assert not any(e['event'] == 'AUTO_RETRY' for e in feeder_events(canary))
    assert any(e['event'] == 'ROW' and e['module'] == 'module-A' for e in feeder_events(canary))
    assert not (tmp_path / 'module-B/units' / key / 'attempts/2').exists()
    assert len([c for c in calls(graph) if c['action'] == 'submit' and c['phase'] == phase and key in c['arguments'][0]]) == 1


def test_feeder_reports_failed_and_refused_modules_together(tmp_path, monkeypatch):
    assert hasattr(api, 'PhaseJobFailed'), 'feature absent: one final error names failed and refused modules'
    canary, bookings, configs = make_graph(tmp_path, monkeypatch, ('module-A', 'module-B', 'module-C'))
    key = configs['module-B']['unit_keys'][0]
    (tmp_path / 'job-outcomes.json').write_text(json.dumps([
        {'module': 'module-B', 'key': key, 'phase': 'prepare', 'exit': 1, 'completion': False, 'attempts': [1, 2]}]))
    (tmp_path / 'refuse').write_text('module-C/templates/' + configs['module-C']['unit_keys'][0] + '/prepare')
    with pytest.raises(api.FeederError, match='COUNTED_PHASE_FAILED.*module-B.*COUNTED_UNITS_REFUSED.*module-C'):
        api.run(canary, bookings)
    summary, = [e for e in feeder_events(canary) if e['event'] == 'UNITS_REFUSED']
    assert summary['modules'] == ['module-C'] and summary['failed'] == {'module-B': 'COUNTED_PHASE_FAILED'}
    assert any(e['event'] == 'ROW' and e['module'] == 'module-A' for e in feeder_events(canary))


@pytest.mark.parametrize('prior', ['refused', 'unused'])
def test_feeder_explicit_retry_accepts_a_phase_that_never_ran(tmp_path, monkeypatch, prior):
    assert hasattr(api, 'PhaseJobFailed'), 'feature absent: refused and unused phases permit an explicit retry'
    canary, bookings, configs = graph = make_graph(tmp_path, monkeypatch)
    key = configs['module-B']['unit_keys'][0]
    attempt = tmp_path / 'module-B/units' / key / 'attempts/1'
    attempt.mkdir(parents=True)
    if prior == 'refused':
        (tmp_path / 'refuse').write_text('module-B/templates/' + key + '/prepare')
        with pytest.raises(api.FeederError, match='COUNTED_UNITS_REFUSED'):
            api.run(canary, bookings)
        (tmp_path / 'refuse').unlink()
    with pytest.raises(api.FeederError, match='COUNTED_PHASE_STOPPED'):
        api.run(canary, bookings, {key: 2})
    assert (attempt.with_name('2') / 'capture/phase.json').is_file()
    assert not list((tmp_path / 'module-B/units').glob('compare-*/attempts/*'))
    (attempt / 'lead-cause.json').write_text(json.dumps({'unit': key, 'attempt': 1, 'cause': 'The lead confirmed no job ran.'}))
    api.run(canary, bookings)
    assert any(c['action'] == 'submit' and c['phase'] == 'compare' and 'module-B' in c['arguments'][0] for c in calls(graph))


@pytest.mark.parametrize('refusal', ['unknown', 'unbound', 'bad-next', 'claimed'])
def test_feeder_checks_all_startup_inputs_before_creating_attempts(tmp_path, monkeypatch, refusal):
    assert hasattr(api, 'PhaseJobFailed'), 'feature absent: startup checks precede all attempt creation'
    canary, bookings, configs = graph = make_graph(tmp_path, monkeypatch)
    api.run(canary, bookings)
    retries = {configs['module-A']['unit_keys'][0]: 2}
    if refusal == 'unknown':
        retries['typo'] = 2
    elif refusal == 'unbound':
        config = configs['module-B']
        config['bound'] = False
        (tmp_path / 'module-B/campaign.json').write_text(json.dumps(config))
    else:
        key = configs['module-B']['unit_keys'][0]
        retries[key] = 4 if refusal == 'bad-next' else 2
        if refusal == 'claimed':
            (tmp_path / 'module-B/units' / key / 'attempts/1/prepare-instance.json').unlink()
            state_path = canary / 'counted-feeder-state.json'
            state = api.read(state_path)
            state['jobs'] = {k: v for k, v in state['jobs'].items() if key not in k}
            state_path.write_text(json.dumps(state))
    before = {str(p): p.read_bytes() for p in tmp_path.glob('module-*/units/compare-*/attempts/*/compare-inputs.json')}
    submits = [c for c in calls(graph) if c['action'] == 'submit']
    with pytest.raises(api.FeederError, match={'unknown': 'RETRY_UNIT_UNKNOWN', 'unbound': 'COUNTED_ROW_NOT_BOUND',
                                             'bad-next': 'RETRY_MUST_BE_NEXT_ATTEMPT', 'claimed': 'SUBMISSION_AMBIGUOUS_AUDIT_REQUIRED'}[refusal]):
        api.run(canary, bookings, retries)
    assert not list(tmp_path.glob('module-*/units/*/attempts/2'))
    assert {str(p): p.read_bytes() for p in tmp_path.glob('module-*/units/compare-*/attempts/*/compare-inputs.json')} == before
    assert [c for c in calls(graph) if c['action'] == 'submit'] == submits


def test_feeder_removes_stale_private_scratch_under_its_lock(tmp_path, monkeypatch):
    assert hasattr(api, 'rmtree'), 'feature absent: feeder start removes stale private jobspec scratch'
    canary, bookings, _ = make_graph(tmp_path, monkeypatch)
    stale = canary / '.counted-feeder-stale'
    stale.mkdir(mode=0o700)
    (stale / 'spec.json').write_text('private jobspec bytes')
    original = api.dry_run
    def checked_submit(*args):
        assert not stale.exists()
        with (canary / 'counted-feeder.lock').open() as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return original(*args)
    monkeypatch.setattr(api, 'dry_run', checked_submit)
    api.run(canary, bookings)
    assert not stale.exists()
    assert not list(canary.glob('.counted-feeder-*'))


@pytest.mark.parametrize('ending', ('submit', 'refused', 'error'))
def test_feeder_keeps_environment_values_out_of_attempt_records(tmp_path, monkeypatch, ending):
    assert hasattr(api, 'NamedTemporaryFile'), 'feature absent: no environment values in attempt records'
    canary, bookings, _ = make_graph(tmp_path, monkeypatch, ('module-A',), external=True)
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
    canary, bookings, _ = graph = make_graph(tmp_path, monkeypatch, external=True)
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
    events = feeder_events(canary)
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


def test_dry_run_has_no_shell_key_and_table_keys_are_exact(tmp_path, monkeypatch):
    assert "'--rlimit=-*'" in inspect.getsource(api.dry_run), 'feature absent: dry-run turns off copied shell limits'
    canary, _, configs = make_graph(tmp_path, monkeypatch)
    template = canary.parent / 'module-B/templates' / configs['module-B']['unit_keys'][0] / 'prepare'
    spec = json.loads(api.dry_run(template, [canary.parent / 'unclaimed'], 'fv1', 'sparklina'))
    assert set(spec['attributes']['system']) == {'constraints', 'cwd', 'dependencies', 'duration', 'environment', 'job'}
    table = json.loads((template / 'table.json').read_text())
    assert set(table) == {'schema', 'instances', 'job_name', 'args', 'max_attempts', 'urgency', 'cwd', 'env_names'}
    assert all(type(table[key]) is int for key in ('args', 'max_attempts', 'urgency')) and table['max_attempts'] == 5
    assert set(table['instances']) <= {'fv1', 'fv1e-sparky', 'fv1e-sparklina'}
    assert all(set(entry) == {'constraints', 'shapes', 'max_duration_s'} for entry in table['instances'].values())
    assert all(type(name) is str for name in table['env_names'])

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
    capture, digest_of = first / 'capture', lambda data: hashlib.sha256(data).hexdigest()
    document = json.loads((capture / 'phase.json').read_text())
    scope = document['scopes'][0]
    case = {'ordinal': 0, 'kind': 'dense', 'scope': scope, 'identities': document['identities'], 'configuration': scope['unit_key'][4],
            'status': {'findings': ['planted FAIL']}, 'id': json_digest([scope, 'dense', 0])}
    fail = capture / 'cases.jsonl'
    fail.write_text(json.dumps(case) + '\n')
    document['streams']['cases'] = {'count': 1, 'sha256': digest_of(fail.read_bytes())}
    document['case_ids_sha256'] = digest_of((case['id'] + '\n').encode())
    (capture / 'phase.json').write_text(json.dumps(document))
    (capture / 'completion.sha256').write_text(''.join(digest_of(p.read_bytes()) + '  ' + p.name + '\n'
                                                       for p in sorted(capture.iterdir()) if p.name != 'completion.sha256'))
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
    canary, bookings, _ = graph = make_graph(tmp_path, monkeypatch, external=True)
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


def group_graph(tmp_path, monkeypatch, k=2):
    import shutil
    canary, bookings, configs = graph = make_graph(tmp_path, monkeypatch)
    config = configs['module-A']
    directory = canary.parent / 'module-A'
    shutil.rmtree(directory / 'templates')
    settings = api.read(directory / 'feeder-settings.json')
    settings['capture_group_canary'] = {'cores_per_capture': 1, 'anon_peak_gib_per_capture': 2, 'margin_gib': 1, 'max_k': k}
    (directory / 'feeder-settings.json').write_text(json.dumps(settings))
    keys = config['unit_keys'][:k]
    config['capture_groups'] = [{'key': 'group-A', 'unit_keys': keys}]
    prepare_split.write_templates(directory, config, directory / 'repository.git')
    prepare_split.bind_templates(directory, config)
    return graph, directory, keys


def test_feeder_capture_group_uses_ordered_arguments_and_recovers_id(tmp_path, monkeypatch):
    assert hasattr(api, 'attempt_jobs'), 'feature absent: grouped capture saves and reuses every member job id'
    graph, directory, keys = group_graph(tmp_path, monkeypatch)
    canary, bookings, _ = graph
    api.run(canary, bookings)
    key = 'group-' + json_digest(['module-A', keys])
    grouped, = [c for c in calls(graph) if c['action'] == 'gate' and f'/{key}/' in c['template']]
    arguments = grouped['spec']['tasks'][0]['command'][1:]
    assert arguments == [str(directory / 'units' / key / 'attempts/1') for key in keys]
    ids = [Path(argument) / 'capture-jobid' for argument in arguments]
    assert len({p.read_text() for p in ids}) == 1
    ids[0].unlink()
    before = [c for c in calls(graph) if c['action'] == 'submit']
    api.run(canary, bookings)
    assert [c for c in calls(graph) if c['action'] == 'submit'] == before
    assert ids[0].read_text() == ids[1].read_text()


@pytest.mark.parametrize('failed_member', (0, 2))
@pytest.mark.parametrize('missing', (False, True))
def test_feeder_end_records_each_capture_member_exit(tmp_path, monkeypatch, failed_member, missing):
    assert 'member_exits' in inspect.getsource(api.run), 'feature absent: END records every capture member exit'
    graph, directory, keys = group_graph(tmp_path, monkeypatch, k=3)
    canary, bookings, _ = graph
    (tmp_path / 'job-outcomes.json').write_text(json.dumps([
        {'module': 'module-A', 'key': keys[failed_member], 'phase': 'capture', 'exit': 7,
         'missing_member_exit': missing, 'attempts': [1, 2]}]))
    with pytest.raises(api.FeederError, match='COUNTED_PHASE_.*module-A'):
        api.run(canary, bookings)
    endings = [event for event in feeder_events(canary) if event['event'] == 'END' and event['member_exits']]
    assert endings
    for event in endings:
        assert event['native_exit'] == 7
        members = event['member_exits']
        assert {Path(attempt).parents[1].name for attempt in members} == set(keys)
        for attempt, status in members.items():
            assert status == ((None if missing else 7) if Path(attempt).parents[1].name == keys[failed_member] else 0)
            if status is not None:
                assert status == api.read(Path(attempt) / 'capture/exit.json')['exit_status']


@pytest.mark.parametrize('interrupt', (False, True))
def test_feeder_records_drain_intent_before_drain(tmp_path, monkeypatch, interrupt):
    assert inspect.getsource(api.run).index("event('SUBMIT_INTENT'") < inspect.getsource(api.run).index('drain_ecores_before_submit(memory'), 'feature absent: E-core drain follows its saved intent'
    canary, bookings, _ = graph = make_graph(tmp_path, monkeypatch)
    for table_path in tmp_path.glob('module-*/templates/*/capture/table.json'):
        table = api.read(table_path)
        next(resource for resource in table['instances']['fv1']['shapes'][0][0]['with'] if resource['type'] == 'memory')['count'] = 37
        table_path.write_text(json.dumps(table))
    drains = []
    def drain(memory, submit, reason):
        if memory <= 36:
            return None
        target = Path(reason.removeprefix('tessera-1717:'))
        intent = api.read(target.parent / 'capture-intent.json')
        assert intent['fv1e_drain'] == {'host': 'sparklina', 'reason': reason}
        assert intent['arguments'] == [str(target.parent)]
        assert feeder_events(canary)[-1]['event'] == 'SUBMIT_INTENT'
        assert not any(call['action'] == 'gate' and str(target.parent) in call['spec']['tasks'][0]['command'][1:]
                       and Path(call['template']).name == 'capture' for call in calls(graph))
        drains.append(intent['fv1e_drain'])
        if interrupt:
            raise RuntimeError('drain interrupted')
        return intent['fv1e_drain']
    monkeypatch.setattr(api, 'drain_ecores_before_submit', drain)
    monkeypatch.setattr(api, 'undrain_ecores', lambda **kwargs: None)
    if interrupt:
        with pytest.raises(RuntimeError, match='drain interrupted'):
            api.run(canary, bookings)
        assert not any(call['action'] == 'gate' and Path(call['template']).name == 'capture' for call in calls(graph))
    else:
        api.run(canary, bookings)
    assert drains


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
    if fault in ('tool', 'payload', 'podman'):
        (tmp_path / 'job-outcomes.json').write_text(json.dumps([
            {'module': 'module-B', 'key': job.parents[2].name, 'phase': 'prepare', 'exit': 1,
             'completion': False, 'attempts': [2]}]))
    with pytest.raises(api.FeederError, match='COUNTED_PHASE_|RESUME_PHASE_INSTANCE_DIFFERS'):
        api.run(canary, bookings)
    after = [c for c in calls(graph) if c['action'] == 'submit']
    if fault in ('native', 'tool', 'payload', 'podman'):
        assert after[:len(before)] == before and len(after) > len(before)
        assert all(c['phase'] == 'prepare' and Path(c['arguments'][0]).name == '2' for c in after[len(before):])
        assert not list(tmp_path.glob('module-*/units/*/attempts/3'))
    else:
        assert after == before
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
