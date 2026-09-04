"""A run must say which population it covered (tessera#112).

Master was red on three CUDA-gated tests while GitHub Actions, the x86 pool
suite and a local CPU run all read green, because none of those three could
collect or run them and none of them said so.  Two mechanisms answer that, and
the tests below are the two questions they answer.

Both drive a **child** pytest rather than pytest's own ``pytester`` fixture:
``pytester`` needs ``-p pytester`` enabled from a rootdir conftest, and this
repo's conftest is ``tests/conftest.py``, which is not one.  A subprocess also
exercises the thing a merge check will actually run -- an interpreter, an
argv and an exit code -- rather than an in-process approximation of it.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _child_env(**extra):
    """An environment a nested pytest can run in from anywhere.

    ``src`` so ``tessera`` imports and ``tests`` so ``-p conftest`` resolves;
    the two cache directories because the fleet forbids ``/tmp`` and a
    root-owned Triton cache fails every kernel test on this box.
    """

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "tests"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env.setdefault("TMPDIR", "/home/rob/tmp")
    env.setdefault("TRITON_CACHE_DIR", str(Path.home() / ".triton-cache"))
    env.update(extra)
    return env


def _run(args, **env_extra):
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=str(ROOT),
        env=_child_env(**env_extra),
        capture_output=True,
        text=True,
        timeout=600,
    )


# A file that collects with nothing but the standard library, so the refusal
# below is provable in the torch-free `pure` CI job too.  Named as a property
# ("collects without torch"), not as a roster entry: if it ever grows a torch
# import the assertion that it collected at all will say so.
STDLIB_ONLY_TEST = "tests/test_alphabet.py"

SYNTHETIC = '''
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ts112 synthetic gate")
def test_gated():
    assert torch.cuda.is_available()


def test_plain():
    assert True
'''


def _write_synthetic(tmp_path):
    path = tmp_path / "test_ts112_probe.py"
    path.write_text(SYNTHETIC)
    return path


# --- the gate --------------------------------------------------------------


@pytest.mark.parametrize(
    "flag, env",
    [
        (["--strict-cuda"], {}),
        ([], {"TESSERA_STRICT_CUDA": "1"}),
    ],
    ids=["flag", "env"],
)
def test_strict_cuda_refuses_an_interpreter_with_no_device(flag, env):
    """The exact misreading in #112: hide the device, read "passed", believe it.

    Both spellings must refuse.  The environment variable is not a convenience:
    a pool action is a sealed command line assembled by a wrapper, and an
    environment variable is what such a wrapper can set without rewriting argv.
    """

    result = _run([STDLIB_ONLY_TEST, "-q", *flag],
                  CUDA_VISIBLE_DEVICES="", **env)
    output = result.stdout + result.stderr
    assert result.returncode != 0, (
        "a run that cannot see a CUDA device reported success:\n" + output
    )
    assert "--strict-cuda" in output and "Refusing rather than skipping" in output, output
    # Refused, not merely failed: nothing ran.
    assert " passed" not in result.stdout, output


def test_without_the_flag_the_same_run_is_green():
    """The gate is opt-in.  Without it a CPU box still runs what it can."""

    result = _run([STDLIB_ONLY_TEST, "-q"], CUDA_VISIBLE_DEVICES="")
    assert result.returncode == 0, result.stdout + result.stderr
    assert " passed" in result.stdout, result.stdout


# --- the diagnostic --------------------------------------------------------


def _skipped_in_tail(stdout):
    """What pytest's own -q summary says was skipped."""

    match = re.search(r"(\d+) skipped", stdout)
    return int(match.group(1)) if match else 0


def test_the_summary_names_what_a_deviceless_run_skipped(tmp_path):
    """"41 passed, 4 skipped" must stop being readable as coverage.

    Driven on a file written here rather than on one of the repo's own
    CUDA-gated modules: the reason string asserted below is then this test's
    own, so the assertion pins the mechanism and not a roster of other files'
    wording.
    """

    pytest.importorskip("torch")
    probe = _write_synthetic(tmp_path)
    result = _run([str(probe), "-q", "-p", "conftest"], CUDA_VISIBLE_DEVICES="")
    out = result.stdout + result.stderr
    assert result.returncode == 0, out

    assert "tessera surface: NO CUDA" in out, out
    assert "did not exercise the CUDA-gated surface" in out, out
    assert "skip reasons, verbatim --" in out, out
    assert re.search(r"^\s+1\s+ts112 synthetic gate$", out, re.M), out

    # The population the line reports is the population pytest counted.
    reported = re.search(r"tessera surface: (\d+) test\(s\) skipped", out)
    assert reported, out
    assert int(reported.group(1)) == _skipped_in_tail(result.stdout) == 1, out


def test_the_reason_histogram_is_verbatim_not_classified(tmp_path):
    """No regex over "cuda|gpu|triton" decides what is reported.

    A gate whose reason names no device must still be counted and printed, or
    the count is a guess that silently undercounts -- the same blindness one
    level down.
    """

    pytest.importorskip("torch")
    probe = tmp_path / "test_ts112_opaque.py"
    probe.write_text(
        "import pytest\n"
        "pytestmark = pytest.mark.skip(reason='an entirely unrelated sentence')\n"
        "def test_one():\n    pass\n"
    )
    result = _run([str(probe), "-q", "-p", "conftest"], CUDA_VISIBLE_DEVICES="")
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    assert re.search(r"^\s+1\s+an entirely unrelated sentence$", out, re.M), out


def _cuda_here() -> bool:
    """Deliberately not ``pytest.importorskip``: a torch-free interpreter and a
    torch-without-a-device one are the same answer for this file, and the
    module must stay importable in the ``pure`` job either way."""

    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _cuda_here(), reason="needs a CUDA device")
def test_a_device_visible_run_says_so_and_runs_the_gated_test(tmp_path):
    """The other half of the population, stated by the same line.

    This test is itself CUDA-gated, so on a CPU box it appears in the very
    histogram it is about -- which is the demonstration, not an irony.
    """

    probe = _write_synthetic(tmp_path)
    result = _run([str(probe), "-q", "-p", "conftest", "--strict-cuda"])
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    assert "tessera surface: CUDA --" in out, out
    assert "did not exercise" not in out, out
    assert _skipped_in_tail(result.stdout) == 0, out


def test_the_population_is_published_as_a_table_not_scraped(tmp_path):
    """A receipt reads the run's own table, never a rendering of it.

    "1404 passed / 487 skipped" quoted out of a scrollback is how a population
    gets separated from its device in the first place.  ``--surface-json`` puts
    the two in one object so they cannot be quoted apart.
    """

    pytest.importorskip("torch")
    probe = _write_synthetic(tmp_path)
    surface = tmp_path / "surface.json"
    result = _run(
        [str(probe), "-q", "-p", "conftest", "--surface-json", str(surface)],
        CUDA_VISIBLE_DEVICES="",
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0, out

    import json

    payload = json.loads(surface.read_text())
    assert payload["schema"] == "tessera.test_surface.v1"
    assert payload["cuda"] is False
    assert payload["strict_cuda"] is False
    assert payload["counts"]["skipped"] == 1
    assert payload["counts"]["passed"] == 1
    assert payload["skip_reasons"] == {"ts112 synthetic gate": 1}
