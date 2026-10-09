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

``SUBMIT=0`` is the explicit processor entry used here: it checks the
inputs and runs the actual authentication only. No submission, no
graphics work, no encode. These processor control tests make no native
byte claim.
"""
from __future__ import annotations

import os
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
                         "TESSERA_ENV_DUMP")}
    authority = tmp_path / "producer_authority.py"
    authority.write_text("authority-placeholder\n")
    env["PRODUCER_AUTHORITY"] = str(authority)
    env["PART_IMAGE"] = IMAGE
    env["CENSUS_ROOT"] = str(tmp_path / "census")
    env["SUBMIT"] = "0"
    env.update({k: str(v) for k, v in over.items()})
    return env


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
