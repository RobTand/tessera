"""Exercise actual finite controller terminal retention with inert owner calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('failure', ['nonzero', 'exception'])
def test_controller_reports_unpublished_failed_measurements(tmp_path, monkeypatch, failure):
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
    def run(*args, **kwargs):
        if failure == 'exception':
            raise RuntimeError('injected contained-owner failure')
        return 1
    monkeypatch.setitem(sys.modules,'paired_k32_action',SimpleNamespace(
        run_direct_arm=run, owned_cleanup=lambda out:{'state':'unit-no-container-was-launched'}))
    monkeypatch.chdir(tmp_path)
    if failure == 'exception':
        with pytest.raises(RuntimeError,match='injected contained-owner failure'):
            owner.main()
    else:
        assert owner.main() == 1
    terminal = json.loads((evidence/'terminal.json').read_text())
    assert terminal['measurement_retention'] == 'benchmark_not_published_partial_inprocess_events_not_retained'
    assert not (tmp_path/'never-executed-benchmark').exists()
