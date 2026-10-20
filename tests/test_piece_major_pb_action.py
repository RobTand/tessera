"""Exercise actual finite controller terminal retention with inert owner calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def previous_sigterm():
    original = signal.getsignal(signal.SIGTERM)
    def sentinel(signum, frame):
        raise AssertionError('caller SIGTERM handler must not run inside the controller')
    signal.signal(signal.SIGTERM, sentinel)
    try:
        yield sentinel
    finally:
        # Failed restoration assertions must not contaminate later pytest tests.
        signal.signal(signal.SIGTERM, original)


@pytest.fixture
def sealed_controller(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1]/'experiments/t8r_speed/piece_major_pb_action.py'
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
    monkeypatch.chdir(tmp_path)
    return SimpleNamespace(owner=owner, evidence=evidence, out=tmp_path/'never-executed-benchmark')


@pytest.mark.parametrize('failure', ['nonzero', 'exception'])
def test_controller_reports_unpublished_failed_measurements(sealed_controller, monkeypatch, failure):
    owner, evidence, out = sealed_controller.owner, sealed_controller.evidence, sealed_controller.out
    def run(*args, **kwargs):
        if failure == 'exception':
            raise RuntimeError('injected contained-owner failure')
        return 1
    monkeypatch.setitem(sys.modules,'paired_k32_action',SimpleNamespace(
        run_direct_arm=run, owned_cleanup=lambda out:{'state':'unit-no-container-was-launched'}))
    if failure == 'exception':
        with pytest.raises(RuntimeError,match='injected contained-owner failure'):
            owner.main()
    else:
        assert owner.main() == 1
    terminal = json.loads((evidence/'terminal.json').read_text())
    assert terminal['measurement_retention'] == 'benchmark_not_published_partial_inprocess_events_not_retained'
    assert not out.exists()


@pytest.mark.parametrize('exit_path', [
    'success', 'nonzero', 'exception', 'interrupted', 'setup_failure',
    'cleanup_failure', 'terminal_failure', 'postprocess_failure', 'evidence_failure',
])
def test_controller_restores_previous_sigterm_on_every_exit(
        sealed_controller, monkeypatch, previous_sigterm, exit_path):
    owner, evidence, out = sealed_controller.owner, sealed_controller.evidence, sealed_controller.out
    error = RuntimeError('injected ' + exit_path)
    calls = []

    def assert_controller_handler():
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler) and handler is not previous_sigterm
        return handler

    def run(argv, env, log, output, pressure, *, deadline_s):
        calls.append('run')
        handler = assert_controller_handler()
        if exit_path == 'exception':
            raise error
        if exit_path == 'interrupted':
            # Invoke the real installed callback without signalling pytest itself.
            handler(signal.SIGTERM, None)
        if exit_path == 'nonzero':
            return 23
        output.mkdir()
        (output/'bench_t8r.json').write_text('{}')
        return 0

    cleanup = {'state': 'unit-no-container-was-launched'}
    def owned_cleanup(output):
        assert output == out
        calls.append('cleanup')
        assert_controller_handler()
        if exit_path == 'cleanup_failure':
            raise error
        return cleanup

    def postprocess(argv, *, check):
        calls.append('postprocess')
        assert check is True
        assert_controller_handler()
        if exit_path == 'postprocess_failure':
            raise error
        (evidence/'netdata.json').write_text('{}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setitem(sys.modules, 'paired_k32_action', SimpleNamespace(
        run_direct_arm=run, owned_cleanup=owned_cleanup))
    monkeypatch.setattr(owner.subprocess, 'run', postprocess)
    if exit_path == 'setup_failure':
        def fail_setup(pid):
            assert_controller_handler()
            raise error
        monkeypatch.setattr(owner.os, 'sched_getaffinity', fail_setup)
    if exit_path in ('terminal_failure', 'evidence_failure'):
        original_write = Path.write_text
        target = evidence/('terminal.json' if exit_path == 'terminal_failure' else 'evidence-sha256.json')
        def fail_write(path, *args, **kwargs):
            if path == target:
                assert_controller_handler()
                raise error
            return original_write(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'write_text', fail_write)

    if exit_path in ('success', 'nonzero'):
        assert owner.main() == (23 if exit_path == 'nonzero' else 0)
    elif exit_path == 'interrupted':
        with pytest.raises(KeyboardInterrupt, match=f'^PM timing interrupted: {signal.SIGTERM}$'):
            owner.main()
    else:
        with pytest.raises(RuntimeError) as raised:
            owner.main()
        assert raised.value is error

    if exit_path == 'setup_failure':
        assert calls == []
    else:
        assert calls[:2] == ['run', 'cleanup']
        assert calls.count('cleanup') == 1
        if exit_path not in ('cleanup_failure', 'terminal_failure'):
            terminal = json.loads((evidence/'terminal.json').read_text())
            assert terminal['cleanup'] == cleanup
            assert terminal['end_unix'] >= terminal['start_unix']
            expected_code = None if exit_path in ('exception', 'interrupted') else 23 if exit_path == 'nonzero' else 0
            assert terminal['returncode'] == expected_code
            expected_retention = ('benchmark_receipt_present' if (out/'bench_t8r.json').is_file()
                else 'benchmark_not_published_partial_inprocess_events_not_retained')
            assert terminal['measurement_retention'] == expected_retention
            if exit_path in ('exception', 'interrupted'):
                expected_error = ('RuntimeError: ' + str(error) if exit_path == 'exception'
                    else f'KeyboardInterrupt: PM timing interrupted: {signal.SIGTERM}')
                assert terminal['error'] == expected_error
    if exit_path == 'success':
        assert calls == ['run', 'cleanup', 'postprocess']
        assert (evidence/'timing').resolve() == out
        hashes = json.loads((evidence/'evidence-sha256.json').read_text())
        assert hashes[str(out/'bench_t8r.json')] == hashlib.sha256((out/'bench_t8r.json').read_bytes()).hexdigest()
        assert hashes[str(evidence/'netdata.json')] == hashlib.sha256((evidence/'netdata.json').read_bytes()).hexdigest()
    assert signal.getsignal(signal.SIGTERM) is previous_sigterm
