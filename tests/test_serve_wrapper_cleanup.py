"""Exercise the real teacher wrapper's exit paths without launching a serve.

The speculative-decoding gate is here too (tessera#247).  It used to be
``curl -s .../metrics | grep -q 'vllm:spec_decode'``, which cannot detect the
condition it owns: ``grep -q`` exits at the first match and closes the pipe, so
on a real metrics response -- the marker followed by thousands of further lines
-- curl fails its write, ``set -o pipefail`` makes the pipeline non-zero, the
``if`` reads false and the wrapper dumps anyway.  A failed fetch or an HTTP
error took the same accepting branch.  The stub below is therefore a curl that
honours ``-o`` and ``-w '%{http_code}'`` and can answer with a body larger than
a pipe buffer, because a stub that emits one marker line and exits cannot
exercise either failure.
"""
from pathlib import Path
import os
import shutil
import subprocess

import pytest

WRAPPER = Path(__file__).resolve().parents[1] / "experiments" / "serve_and_dump_kl.sh"

#: A curl faithful enough to fail the way the real one does: it writes the
#: whole body where ``-o`` says, prints the code ``-w`` asks for, and can be
#: told to fail at the transport or answer an HTTP error.
CURL_STUB = r'''#!/bin/bash
out=""; wfmt=""; url=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    -w) wfmt="$2"; shift 2 ;;
    -*) shift ;;
    *) url="$1"; shift ;;
  esac
done
emit() { if [ -n "$out" ]; then cat > "$out"; else cat; fi; }
bulk() { seq 1 20000 | sed 's/^/vllm:num_requests_running /'; }
case "$url" in
  */v1/models)
    [ "$READY" = 1 ] || exit 7
    echo '{"data":[{"id":"kl-target"}]}' | emit
    [ -z "$wfmt" ] || printf '200'
    exit 0 ;;
  */metrics)
    case "$METRICS" in
      unreachable) exit 7 ;;
      http_500)    echo 'internal error' | emit
                   [ -z "$wfmt" ] || printf '500'
                   exit 0 ;;
      empty)       : | emit
                   [ -z "$wfmt" ] || printf '200'
                   exit 0 ;;
      clean)       bulk | emit ;;
      spec_bulk)   { echo 'vllm:spec_decode_num_accepted_tokens_total 3'; bulk; } | emit ;;
      spec_one)    echo 'vllm:spec_decode' | emit ;;
      *)           echo "unknown METRICS=$METRICS" >&2; exit 90 ;;
    esac
    # curl reports a failed write, and dies on SIGPIPE, exactly as it does
    # when a downstream `grep -q` exits at the first match: that status is the
    # whole of tessera#247, so the stub must carry it rather than swallow it.
    st=$?
    [ "$st" -eq 0 ] || exit "$st"
    [ -z "$wfmt" ] || printf '200'
    exit 0 ;;
esac
exit 1
'''


def _run_wrapper(tmp_path, *, fail_remove=False, ready=False, metrics="clean"):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    wrapper = scripts / WRAPPER.name
    shutil.copyfile(WRAPPER, wrapper)
    shutil.copyfile(WRAPPER.parent / "serve_metrics.sh", scripts / "serve_metrics.sh")
    (scripts / "runtime_image.sh").write_text("runtime_image_require() { :; }\n")
    (scripts / "build_identity.sh").write_text(
        "build_identity_docker_env() { :; }\n"
    )
    (scripts / "serve_lock.sh").write_text(
        'SERVE_LOCK="$TEST_ROOT/serve.lock"\n'
        'serve_lock_acquire() { touch "$SERVE_LOCK"; }\n'
        'serve_lock_release() { echo release >> "$TEST_ROOT/events"; '
        'rm "$SERVE_LOCK"; }\n'
    )
    commands = {
        "docker": r'''#!/bin/bash
case "$1" in
    run) touch "$TEST_ROOT/live"; echo container ;;
    ps) [ ! -f "$TEST_ROOT/live" ] || echo container ;;
    logs)
        echo logs >> "$TEST_ROOT/events"
        if [ -f "$TEST_ROOT/live" ]; then echo 'preserved serve log';
        else echo 'missing container' >&2; exit 1; fi ;;
    rm)
        [ -f "$TEST_ROOT/live" ] || exit 0
        echo remove >> "$TEST_ROOT/events"
        [ "$FAIL_REMOVE" = 0 ] || exit 1
        rm "$TEST_ROOT/live" ;;
    *) exit 90 ;;
esac
''',
        "curl": CURL_STUB,
        "sleep": "#!/bin/bash\nexit 23\n",
    }
    for name, body in commands.items():
        path = scripts / name
        path.write_text(body)
        path.chmod(0o755)
    env = os.environ | {
        "PATH": f"{scripts}:{os.environ['PATH']}",
        "TEST_ROOT": str(tmp_path),
        "FAIL_REMOVE": str(int(fail_remove)),
        "READY": str(int(ready)),
        "METRICS": metrics,
        "TESSERA_KL_LOGDIR": str(tmp_path),
        # The dump is the stage after the gate; point it at a corpus that is
        # not there so a run that gets past the gate fails there, fast.
        "TESSERA_KL_CORPUS": str(tmp_path / "no-such-corpus.json"),
    }
    result = subprocess.run(
        ["bash", str(wrapper), str(tmp_path / "model"),
         str(tmp_path / "dump.json"), "teacher"],
        env=env, text=True, capture_output=True,
    )
    events = tmp_path / "events"
    return result, events.read_text().splitlines() if events.exists() else []


def test_teacher_wrapper_reaps_after_unexpected_exit(tmp_path):
    result, events = _run_wrapper(tmp_path)
    assert result.returncode == 23, result.stderr
    assert not (tmp_path / "live").exists(), "unexpected exit left the serve live"
    assert not (tmp_path / "serve.lock").exists()
    assert events == ["logs", "remove", "release"]
    assert (tmp_path / "serve_dump.log").read_text() == "preserved serve log\n"


def test_teacher_wrapper_retains_ownership_when_reap_fails(tmp_path):
    result, events = _run_wrapper(tmp_path, fail_remove=True)
    assert result.returncode == 23, result.stderr
    assert (tmp_path / "live").exists()
    assert (tmp_path / "serve.lock").exists(), "live serve lost its ownership fence"
    assert events == ["logs", "remove"]
    assert "cleanup unverified" in result.stderr


def test_teacher_wrapper_refusal_cleanup_preserves_logs_once(tmp_path):
    result, events = _run_wrapper(tmp_path, ready=True, metrics="spec_one")
    assert result.returncode == 2, result.stderr
    assert not (tmp_path / "live").exists()
    assert events == ["logs", "remove", "release"]
    assert (tmp_path / "serve_dump.log").read_text() == "preserved serve log\n"


def test_a_marker_followed_by_a_large_body_still_refuses(tmp_path):
    """The audited case: a real metrics response is the marker plus thousands
    of further lines, and the old pipeline read that as 'not speculative'."""
    result, _ = _run_wrapper(tmp_path, ready=True, metrics="spec_bulk")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "spec-decode active" in result.stderr
    assert not (tmp_path / "live").exists(), "refusal left the serve live"


def test_an_unreachable_metrics_endpoint_refuses(tmp_path):
    """A serve that cannot be asked is not a serve without spec-decode."""
    result, _ = _run_wrapper(tmp_path, ready=True, metrics="unreachable")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "could not read the serve's /metrics" in result.stderr


def test_an_http_error_from_metrics_refuses(tmp_path):
    """``curl -s`` does not reject an HTTP error response, so the gate must."""
    result, _ = _run_wrapper(tmp_path, ready=True, metrics="http_500")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "answered HTTP 500" in result.stderr


def test_an_empty_metrics_body_refuses(tmp_path):
    result, _ = _run_wrapper(tmp_path, ready=True, metrics="empty")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "empty body" in result.stderr


def test_a_clean_metrics_response_passes_the_gate(tmp_path):
    """A marker-free response of realistic size proceeds to the dump, and the
    evidence the gate read is kept beside the serve log."""
    result, _ = _run_wrapper(tmp_path, ready=True, metrics="clean")
    assert result.returncode == 3, result.stdout + result.stderr
    assert "dump FAILED" in result.stdout
    metrics = tmp_path / "serve_dump.metrics.txt"
    assert metrics.exists() and "vllm:spec_decode" not in metrics.read_text()
