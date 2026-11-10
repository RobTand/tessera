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
import shlex
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
                         "PART_IMAGE", "CENSUS_ROOT", "SUBMIT",
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
        "import json, os, sys, types\n"
        "assert sys.argv[1] == '-c'\n"
        "def authenticate_producer_python():\n"
        "    with open(os.environ['CALLER_CAPTURE'], 'a') as stream:\n"
        "        stream.write(json.dumps({'event': 'authenticate', "
        "'python': os.environ['TESSERA_PRODUCER_PYTHON'], "
        "'source': os.environ['TESSERA_PRODUCER_SOURCE']}) + '\\n')\n"
        "    return {'package_sha256': 'ab' * 32}\n"
        "sys.modules['tessera'] = types.ModuleType('tessera')\n"
        "owner = types.ModuleType('tessera.export_serving')\n"
        "owner.authenticate_producer_python = authenticate_producer_python\n"
        "sys.modules[owner.__name__] = owner\n"
        "exec(sys.argv[2])\n"
    )
    producer.chmod(0o755)
    scales = tmp_path / "input scales.safetensors"
    scales.write_bytes(b"CPU control fixture")
    env = _caller_env(
        tmp_path, TESSERA_PRODUCER_PYTHON=producer, FLUX_OUT="/mnt/shared/producer-render-fixture",
        TESSERA_PRODUCER_SOURCE=source, SUBMIT=1,
        INPUT_SCALES=scales, CALLER_CAPTURE=capture)
    done = _run_caller(tmp_path, env)
    assert done.returncode == 0, done.stdout + done.stderr
    events = [json.loads(line) for line in capture.read_text().splitlines()]
    assert events == [{"event": "authenticate", "python": str(producer), "source": str(source)}]
    lines = [shlex.split(line) for line in done.stdout.splitlines() if line.startswith("flux submit ")]
    expected_commands = [
        ["bash", "experiments/t8_census/export_routed_part.sh", "GB", "3", "8"],
        ["bash", "experiments/t8_census/export_routed_part.sh", "GB", "4", "8"],
        ["bash", "experiments/t16_census/export_census_stub.sh", "D1"],
    ]
    assert len(lines) == 3, done.stdout
    for argv, command in zip(lines, expected_commands):
        assert argv[argv.index("--") + 1:] == command
        assert "--env=-*" in argv and not any(token.startswith("--requires=") for token in argv)
        assert "--cwd" in argv and "--input=/dev/null" in argv
        assert any(token.startswith("--output=/mnt/shared/producer-render-fixture/") for token in argv)
        assert any(token.startswith("--error=/mnt/shared/producer-render-fixture/") for token in argv)
        assert any(part.startswith("--resources-shape=slot/[core=") and part.endswith(";gpu=1]") for part in argv)
        assert argv[argv.index("-t") + 1] == "45m"
        assert argv.count(f"--env=TESSERA_PRODUCER_PYTHON={producer}") == 1
        assert argv.count(f"--env=TESSERA_PRODUCER_SOURCE={source}") == 1


def test_ts91_chain_renders_a_stock_shape_flux_gpu_line_and_runs_nothing(tmp_path):
    """The chain's former pbrun --gpu exec is a rendered, never-run flux submit line."""
    env = {"PATH": os.environ["PATH"], "IMG": "inert-image", "RUNS": str(tmp_path / "runs"), "FLUX_OUT": "/mnt/shared/chain-render-fixture"}
    done = subprocess.run(["bash", str(CHECKOUT / "experiments/ts91_chain.sh"), "before"], env=env,
                          cwd=CHECKOUT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    (line,) = [x for x in done.stdout.splitlines() if x.startswith("flux submit ")]
    argv = shlex.split(line)
    assert "--resources-shape=slot/[core=4;memory=32;gpu=1]" in argv
    assert "--env=-*" in argv and not any(token.startswith("--requires=") for token in argv)
    assert "--input=/dev/null" in argv and "--cwd" in argv
    assert "--output=/mnt/shared/chain-render-fixture/stdout" in argv
    assert "--error=/mnt/shared/chain-render-fixture/stderr" in argv
    assert argv[argv.index("-t") + 1] == "60m"
    command = argv[argv.index("--") + 1:]
    assert command[0] == "bash" and command[1].endswith("experiments/ts91_chain_body.sh")
    assert command[-1] == "inert-image"
    assert not (tmp_path / "runs").exists(), "rendering must not create run directories"
