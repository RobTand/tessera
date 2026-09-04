"""Exercise the real teacher wrapper's exit paths without launching a serve."""
from pathlib import Path
import os
import shutil
import subprocess

WRAPPER = Path(__file__).resolve().parents[1] / "experiments" / "serve_and_dump_kl.sh"


def _run_wrapper(tmp_path, *, fail_remove=False, speculative=False):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    wrapper = scripts / WRAPPER.name
    shutil.copyfile(WRAPPER, wrapper)
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
        "curl": r'''#!/bin/bash
[ "$SPECULATIVE" = 1 ] || exit 1
case "$*" in *metrics*) echo vllm:spec_decode ;; esac
''',
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
        "SPECULATIVE": str(int(speculative)),
        "TESSERA_KL_LOGDIR": str(tmp_path),
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
    result, events = _run_wrapper(tmp_path, speculative=True)
    assert result.returncode == 2, result.stderr
    assert not (tmp_path / "live").exists()
    assert events == ["logs", "remove", "release"]
    assert (tmp_path / "serve_dump.log").read_text() == "preserved serve log\n"
