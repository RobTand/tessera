"""The real teacher wrapper retains its producer and cleanup integration checks."""
from serve_wrapper_fakes import _run_wrapper
def test_teacher_wrapper_reaps_after_unexpected_exit(tmp_path):
    result, events = _run_wrapper(tmp_path)
    assert result.returncode == 2, result.stderr
    assert 'HEALTH_DEADLINE' in result.stderr
    assert not (tmp_path / 'live').exists(), 'health refusal left the serve live'
    assert not (tmp_path / 'serve.lock').exists()
    assert events == ['logs', 'remove']
    assert (tmp_path / 'serve_dump.log').read_text() == 'preserved serve log\n'


def test_teacher_wrapper_retains_ownership_when_reap_fails(tmp_path):
    import json
    result, events = _run_wrapper(tmp_path, fail_remove=True)
    assert result.returncode == 2, result.stderr
    assert (tmp_path / 'live').exists()
    receipt = json.loads((tmp_path / 'receipt.json').read_text())
    assert receipt['removed'] is False
    assert receipt['name'] == 'flux-12345'
    assert events == ['logs', 'remove']
    assert 'CLEANUP_UNVERIFIED' in receipt['cleanup_refusal']


def test_teacher_wrapper_refusal_cleanup_preserves_logs_once(tmp_path):
    result, events = _run_wrapper(tmp_path, ready=True, metrics='spec_one')
    assert result.returncode == 2, result.stderr
    assert not (tmp_path / 'live').exists()
    assert events == ['logs', 'remove']
    assert (tmp_path / 'serve_dump.log').read_text() == 'preserved serve log\n'


def test_a_marker_followed_by_a_large_body_still_refuses(tmp_path):
    result, _ = _run_wrapper(tmp_path, ready=True, metrics='spec_bulk')
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'spec-decode active' in result.stderr
    assert not (tmp_path / 'live').exists(), 'refusal left the serve live'


def test_an_unreachable_metrics_endpoint_refuses(tmp_path):
    result, _ = _run_wrapper(tmp_path, ready=True, metrics='unreachable')
    assert result.returncode == 2, result.stdout + result.stderr
    assert "could not read the serve's /metrics" in result.stderr


def test_an_http_error_from_metrics_refuses(tmp_path):
    result, _ = _run_wrapper(tmp_path, ready=True, metrics='http_500')
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'answered HTTP 500' in result.stderr


def test_an_empty_metrics_body_refuses(tmp_path):
    result, _ = _run_wrapper(tmp_path, ready=True, metrics='empty')
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'empty body' in result.stderr


def test_a_clean_metrics_response_passes_the_gate(tmp_path):
    result, _ = _run_wrapper(tmp_path, ready=True, metrics='clean')
    assert result.returncode == 3, result.stdout + result.stderr
    assert 'dump FAILED' in result.stdout
    metrics = tmp_path / 'serve_dump.metrics.txt'
    assert metrics.exists() and 'vllm:spec_decode' not in metrics.read_text()
