"""Exercise actual finite controller terminal retention with inert owner calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1]/'experiments/t8r_speed/piece_major_pb_action.py'
TEST_SOURCE = Path(__file__).resolve()


@pytest.fixture
def prior_sigterm():
    """Own the test's handler even when the controller's restoration is broken."""
    def prior(signum, frame):
        pass
    original = signal.signal(signal.SIGTERM, prior)
    try:
        yield prior
    finally:
        signal.signal(signal.SIGTERM, original)


def _inert_controller(tmp_path, monkeypatch):
    source = SOURCE
    spec = importlib.util.spec_from_file_location('unit_finite_controller', source)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    def emit(path, value):
        raw = json.dumps(value).encode()
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()
    payload = tmp_path/'input.json'
    payload_sha = emit(payload, {})
    protocol = tmp_path/'protocol.json'
    protocol_sha = emit(protocol, {'harness': {}, 'input_manifest': {'path':str(payload), 'sha256':payload_sha}})
    evidence = tmp_path/'evidence'
    evidence.mkdir()
    window = tmp_path/'window.json'
    window_sha = emit(window, {'protocol':str(protocol), 'protocol_sha256':protocol_sha,
        'worktree':str(tmp_path), 'output':str(tmp_path/'never-executed-benchmark'),
        'numeric_qualification':{'receipt':str(payload), 'receipt_sha256':payload_sha},
        'execution_owner':{'source_bindings':{}, 'namespace_paths':[]}, 'env':{}, 'argv':['unit-owner-call-only']})
    packet = tmp_path/'packet.json'
    packet_sha = emit(packet, {'controller_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'window':str(window), 'window_sha256':window_sha, 'evidence_root':str(evidence)})
    go = tmp_path/'unit-go.json'
    go_sha = emit(go, {'gpu_go':True, 'packet_sha256':packet_sha})
    monkeypatch.setenv('PRISMABUILD_ACTION_KEY','unit-context-not-a-GPU-claim')
    monkeypatch.setattr(sys,'argv',['controller','--packet',str(packet),'--packet-sha256',packet_sha,
                                  '--go-record',str(go),'--go-sha256',go_sha])
    calls = SimpleNamespace(run_direct_arm=lambda *a, **k: 1,
        owned_cleanup=lambda out:{'state':'unit-no-container-was-launched'})
    monkeypatch.setitem(sys.modules, 'paired_k32_action', calls)
    monkeypatch.chdir(tmp_path)
    return owner, evidence, tmp_path/'never-executed-benchmark', calls


@pytest.mark.parametrize('failure', ['nonzero', 'exception'])
def test_controller_reports_unpublished_failed_measurements(tmp_path, monkeypatch, failure, prior_sigterm):
    owner, evidence, out, calls = _inert_controller(tmp_path, monkeypatch)
    error = RuntimeError('injected contained-owner failure')
    def run(*args, **kwargs):
        assert signal.getsignal(signal.SIGTERM) is not prior_sigterm
        if failure == 'exception':
            raise error
        return 1
    calls.run_direct_arm = run
    if failure == 'exception':
        with pytest.raises(RuntimeError,match='injected contained-owner failure') as caught:
            owner.main()
        assert caught.value is error
    else:
        assert owner.main() == 1
    terminal = json.loads((evidence/'terminal.json').read_text())
    assert terminal['measurement_retention'] == 'benchmark_not_published_partial_inprocess_events_not_retained'
    assert terminal['cleanup'] == {'state':'unit-no-container-was-launched'}
    assert terminal['returncode'] == (None if failure == 'exception' else 1)
    if failure == 'exception':
        assert terminal['error'] == 'RuntimeError: injected contained-owner failure'
    assert not out.exists()
    assert signal.getsignal(signal.SIGTERM) is prior_sigterm


@pytest.mark.parametrize('exit_path', [
    'success', 'setup', 'interruption', 'cleanup', 'terminal-evidence',
    'postprocess', 'evidence-hash', 'evidence-write',
])
def test_controller_restores_previous_sigterm_on_every_exit(tmp_path, monkeypatch, prior_sigterm, exit_path):
    owner, evidence, out, calls = _inert_controller(tmp_path, monkeypatch)
    error = RuntimeError('injected ' + exit_path + ' failure')
    events = []
    installed = []

    def assert_owned():
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler) and handler is not prior_sigterm
        if not installed:
            installed.append(handler)
        assert handler is installed[0]

    def run(*args, **kwargs):
        assert_owned()
        events.append('run')
        if exit_path == 'interruption':
            # Exercise the actual installed callback, not a synthetic exception.
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            pytest.fail('SIGTERM callback did not interrupt the owned execution')
        out.mkdir()
        (out/'bench_t8r.json').write_text('{}')
        return 0

    def cleanup(path):
        assert_owned()
        assert path == out
        events.append('cleanup')
        if exit_path == 'cleanup':
            raise error
        return {'state':'unit-no-container-was-launched'}

    def postprocess(*args, **kwargs):
        assert_owned()
        events.append('postprocess')
        assert kwargs['check'] is True
        if exit_path == 'postprocess':
            raise error
        (evidence/'netdata.json').write_text('{}')

    calls.run_direct_arm = run
    calls.owned_cleanup = cleanup
    monkeypatch.setattr(owner.subprocess, 'run', postprocess)
    if exit_path == 'setup':
        def fail_affinity(pid):
            assert_owned()
            raise error
        monkeypatch.setattr(owner.os, 'sched_getaffinity', fail_affinity)
    if exit_path in ('terminal-evidence', 'evidence-write'):
        write_text = Path.write_text
        target = evidence/('terminal.json' if exit_path == 'terminal-evidence' else 'evidence-sha256.json')
        def fail_write(path, *args, **kwargs):
            if path == target:
                assert_owned()
                raise error
            return write_text(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'write_text', fail_write)
    if exit_path == 'evidence-hash':
        digest = owner.digest
        def fail_digest(path):
            if Path(path) == evidence/'netdata.json':
                assert_owned()
                raise error
            return digest(path)
        monkeypatch.setattr(owner, 'digest', fail_digest)

    if exit_path == 'success':
        assert owner.main() == 0
    elif exit_path == 'interruption':
        with pytest.raises(KeyboardInterrupt, match='PM timing interrupted: ' + str(signal.SIGTERM)):
            owner.main()
    else:
        with pytest.raises(RuntimeError, match=str(error)) as caught:
            owner.main()
        assert caught.value is error

    if exit_path == 'setup':
        assert events == []
    else:
        assert events[:2] == ['run', 'cleanup']
        assert events.count('cleanup') == 1
    if exit_path not in ('setup', 'cleanup', 'terminal-evidence'):
        terminal = json.loads((evidence/'terminal.json').read_text())
        assert terminal['cleanup'] == {'state':'unit-no-container-was-launched'}
        assert terminal['returncode'] == (None if exit_path == 'interruption' else 0)
        assert (evidence/'action-window.txt').is_file()
        if exit_path == 'interruption':
            assert terminal['error'] == 'KeyboardInterrupt: PM timing interrupted: ' + str(signal.SIGTERM)
            assert terminal['measurement_retention'] == 'benchmark_not_published_partial_inprocess_events_not_retained'
            assert events == ['run', 'cleanup']
        else:
            assert terminal['measurement_retention'] == 'benchmark_receipt_present'
    if exit_path == 'success':
        assert events == ['run', 'cleanup', 'postprocess']
        assert (evidence/'timing').resolve() == out
        hashes = json.loads((evidence/'evidence-sha256.json').read_text())
        assert hashes == {str(path): owner.digest(path) for path in (out/'bench_t8r.json', evidence/'netdata.json')}
    assert signal.getsignal(signal.SIGTERM) is prior_sigterm


def test_subsequent_sigterm_reaches_previous_handler_in_subprocess(tmp_path):
    # A real signal must never risk interrupting the pytest runner on regression.
    code = r'''
import os
from pathlib import Path
import runpy
import signal
import sys
import pytest

helpers = runpy.run_path(sys.argv[1])
patch = pytest.MonkeyPatch()
seen = []
def prior(signum, frame):
    seen.append(signum)
original = signal.signal(signal.SIGTERM, prior)
try:
    owner, evidence, out, calls = helpers['_inert_controller'](Path(sys.argv[2]), patch)
    assert owner.main() == 1
    os.kill(os.getpid(), signal.SIGTERM)
    assert seen == [signal.SIGTERM], seen
    assert signal.getsignal(signal.SIGTERM) is prior
    print('previous SIGTERM handler received signal', flush=True)
finally:
    signal.signal(signal.SIGTERM, original)
    patch.undo()
'''
    result = subprocess.run([sys.executable, '-c', code, str(TEST_SOURCE), str(tmp_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'previous SIGTERM handler received signal' in result.stdout
