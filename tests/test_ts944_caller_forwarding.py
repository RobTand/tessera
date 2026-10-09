"""The migrated GB and D1 caller forwards the explicit producer source.

tessera#944: the historical caller outside the repository submits both
census wrappers but names only the producer interpreter. Each wrapper
already requires the explicit genuine source reference and passes it
unchanged to the actual exporter, so a submission without it can never
authenticate. The repository caller
``experiments/submit_gb_d1_producer.sh`` is the migrated entry: it
requires both selector variables by name, authenticates the selected
producer through the existing exporter owner before anything submits,
and forwards both values unchanged into each submission action.

The submission test uses a stub producer and a stub PrismaBuild client.
The refusal tests also call the real authentication owner.
These CPU control tests make no native byte claim.
"""
from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path

import pytest

CHECKOUT = Path(__file__).resolve().parents[1]
CALLER = CHECKOUT / "experiments" / "submit_gb_d1_producer.sh"
IMAGE = "localhost/prismaquant/spark-vllm-nccl230@sha256:" + "ab" * 32


def _caller_env(tmp_path, **over):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("TESSERA_PRODUCER")
           and k not in ("INPUT_SCALES", "PART_BOUND_S", "ENCODE_BATCH",
                         "BEST_FORM", "PROFILE", "PRODUCER_AUTHORITY",
                         "PART_IMAGE", "CENSUS_ROOT", "SUBMIT", "PBRUN",
                         "TESSERA_ENV_DUMP", "PRISMAQUANT_DEV_MODE")}
    authority = tmp_path / "producer_authority.py"
    authority.write_text("authority-placeholder\n")
    env["PRODUCER_AUTHORITY"] = str(authority)
    env["PART_IMAGE"] = IMAGE
    env["CENSUS_ROOT"] = str(tmp_path / "census")
    env["SUBMIT"] = "0"
    env["PRISMAQUANT_DEV_MODE"] = os.environ["PRISMAQUANT_DEV_MODE"]
    env.update({k: str(v) for k, v in over.items()})
    return env


@pytest.fixture(autouse=True, params=["1", "0"], ids=["dev", "certified"])
def caller_mode(request, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", request.param)


def _run_caller(tmp_path, env):
    return subprocess.run(
        ["bash", str(CALLER)],
        env=env, cwd=CHECKOUT, capture_output=True, text=True, timeout=300)


def test_caller_refuses_missing_producer_before_any_work(tmp_path):
    source = tmp_path / "qualified" / "src" / "tessera"
    source.mkdir(parents=True)
    done = _run_caller(tmp_path, _caller_env(
        tmp_path, TESSERA_PRODUCER_SOURCE=source))
    text = done.stdout + done.stderr
    assert done.returncode == 2, f"expected refusal, got rc={done.returncode}\n{text}"
    assert "TESSERA_PRODUCER_PYTHON" in text


def test_caller_refuses_missing_source_reference(tmp_path):
    done = _run_caller(tmp_path, _caller_env(
        tmp_path, TESSERA_PRODUCER_PYTHON=sys.executable))
    text = done.stdout + done.stderr
    assert done.returncode == 2, f"expected refusal, got rc={done.returncode}\n{text}"
    assert "TESSERA_PRODUCER_SOURCE" in text


def test_caller_refuses_relative_source_reference(tmp_path):
    done = _run_caller(tmp_path, _caller_env(
        tmp_path, TESSERA_PRODUCER_PYTHON=sys.executable,
        TESSERA_PRODUCER_SOURCE="src/tessera"))
    text = done.stdout + done.stderr
    assert done.returncode == 2, f"expected refusal, got rc={done.returncode}\n{text}"
    assert "TESSERA_PRODUCER_SOURCE" in text


def test_caller_refuses_nonexecutable_producer(tmp_path):
    absent = tmp_path / "absent" / "bin" / "python"
    source = tmp_path / "qualified" / "src" / "tessera"
    source.mkdir(parents=True)
    done = _run_caller(tmp_path, _caller_env(
        tmp_path, TESSERA_PRODUCER_PYTHON=absent,
        TESSERA_PRODUCER_SOURCE=source))
    text = done.stdout + done.stderr
    assert done.returncode == 2, f"expected refusal, got rc={done.returncode}\n{text}"
    assert "TESSERA_PRODUCER_PYTHON" in text
    assert str(absent) in text


def test_caller_authenticates_before_any_submission(tmp_path):
    """Authentication gates the submission path: an unqualified producer
    refuses before any action submits, even with submission enabled."""
    source = tmp_path / "qualified" / "src" / "tessera"
    source.mkdir(parents=True)
    done = _run_caller(tmp_path, _caller_env(
        tmp_path, TESSERA_PRODUCER_PYTHON=sys.executable,
        TESSERA_PRODUCER_SOURCE=source, SUBMIT=1))
    text = done.stdout + done.stderr
    assert done.returncode == 2, f"expected refusal, got rc={done.returncode}\n{text}"
    assert "failed producer authentication" in text
    assert "TESSERA_PRODUCER_PYTHON" in text


def test_caller_refuses_real_unqualified_interpreter(tmp_path):
    """The running test interpreter is not a qualified producer install.

    Through the caller, the actual authentication owner must refuse it --
    whichever check fires first (no installed distribution behind the
    import, unclean reference, ancestry) -- by selector variable name.
    """
    source = CHECKOUT / "src" / "tessera"
    done = _run_caller(tmp_path, _caller_env(
        tmp_path, TESSERA_PRODUCER_PYTHON=sys.executable,
        TESSERA_PRODUCER_SOURCE=source))
    text = done.stdout + done.stderr
    assert done.returncode == 2, f"expected refusal, got rc={done.returncode}\n{text}"
    assert "TESSERA_PRODUCER_PYTHON" in text or "TESSERA_PRODUCER_SOURCE" in text, text


def test_caller_preserves_selectors_in_all_submissions(tmp_path):
    """Capture the real caller's three actions without GPU work."""
    capture = tmp_path / "events.jsonl"
    producer = tmp_path / "selected producer"
    source = tmp_path / "qualified source" / "src" / "tessera"
    source.mkdir(parents=True)
    producer.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "assert sys.argv[1] == '-c'\n"
        "assert 'from tessera.export_serving import authenticate_producer_python' in sys.argv[2]\n"
        "assert 'authenticate_producer_python()' in sys.argv[2]\n"
        "with open(os.environ['CALLER_CAPTURE'], 'a') as stream:\n"
        "    stream.write(json.dumps({'event': 'authenticate', "
        "'python': os.environ['TESSERA_PRODUCER_PYTHON'], "
        "'source': os.environ['TESSERA_PRODUCER_SOURCE']}) + '\\n')\n"
        "print(json.dumps({'package_sha256': 'ab' * 32}))\n"
    )
    producer.chmod(0o755)
    pbrun = tmp_path / "stub pbrun.py"
    pbrun.write_text(
        "import json, os, sys\n"
        "with open(os.environ['CALLER_CAPTURE'], 'a') as stream:\n"
        "    stream.write(json.dumps({'event': 'submit', 'argv': sys.argv[1:]}) + '\\n')\n"
    )
    # Refuse any real PB client, even when the caller ignores the override.
    # Other python3 calls still execute their real receipt parser.
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    python3 = shim_dir / "python3"
    python3.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "if sys.argv[1] not in ('-c', '-') and sys.argv[1] != os.environ['PBRUN']:\n"
        "    raise SystemExit('unexpected submission client: ' + sys.argv[1])\n"
        f"os.execv({sys.executable!r}, [{sys.executable!r}] + sys.argv[1:])\n"
    )
    python3.chmod(0o755)
    scales = tmp_path / "input scales.safetensors"
    scales.write_bytes(b"CPU control fixture")
    env = _caller_env(
        tmp_path, TESSERA_PRODUCER_PYTHON=producer,
        TESSERA_PRODUCER_SOURCE=source, SUBMIT=1, PBRUN=pbrun,
        INPUT_SCALES=scales, CALLER_CAPTURE=capture,
        PATH=str(shim_dir) + os.pathsep + os.environ["PATH"])
    done = _run_caller(tmp_path, env)
    assert done.returncode == 0, done.stdout + done.stderr
    events = [json.loads(line) for line in capture.read_text().splitlines()]
    assert events[0] == {
        "event": "authenticate", "python": str(producer), "source": str(source)}
    assert [event["event"] for event in events] == [
        "authenticate", "submit", "submit", "submit"]
    expected_commands = [
        ["bash", "experiments/t8_census/export_routed_part.sh", "GB", "3", "8"],
        ["bash", "experiments/t8_census/export_routed_part.sh", "GB", "4", "8"],
        ["bash", "experiments/t16_census/export_census_stub.sh", "D1"],
    ]
    for event, command in zip(events[1:], expected_commands):
        argv = event["argv"]
        assert argv[argv.index("--") + 1:] == command
        declared_env = [argv[index + 1] for index, arg in enumerate(argv) if arg == "--env"]
        assert declared_env.count(f"TESSERA_PRODUCER_PYTHON={producer}") == 1
        assert declared_env.count(f"TESSERA_PRODUCER_SOURCE={source}") == 1
