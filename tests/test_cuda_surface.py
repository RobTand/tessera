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
    assert payload["schema"] == "tessera.test_surface.v2"
    assert payload["cuda"] is False
    assert payload["strict_cuda"] is False
    assert payload["counts"]["skipped"] == 1
    assert payload["counts"]["passed"] == 1
    assert payload["skip_reasons"] == {"ts112 synthetic gate": 1}
    # A serial run is its own controller, so this file is the whole run.  The
    # field is what makes that a stated fact rather than an assumption a
    # reader has to make about how the run was launched.
    assert payload["role"] == "population"
    assert payload["worker_id"] is None


def test_both_mechanisms_survive_a_parallel_run(tmp_path):
    """`--cpus N` fans pytest out; neither the gate nor the report may break.

    `tools/merge_suite.py` passes `-n N` whenever the submission reserves more
    than one core, so both mechanisms have to hold under xdist -- and both have
    a way to go wrong there that a serial run cannot show. The gate lives in
    `pytest_sessionstart`, which every worker also runs, so it must refuse once
    on the controller rather than N times or, worse, from a worker whose
    failure reads as a test error. The report lives in
    `pytest_terminal_summary`, and its counts have to be the aggregate the
    controller collected, not one worker's share.

    Skipped where xdist is absent.  That is not "everywhere": the x86 arm's
    interpreter on dl380g10 has it, which is where the ``-n 8`` run that
    exposed the shard bug below came from.  On a venv without it this test
    appears in the very histogram it is about.
    """

    pytest.importorskip("xdist", reason="parallel run needs pytest-xdist")

    refused = _run([STDLIB_ONLY_TEST, "-q", "-n", "2", "--dist", "loadfile",
                    "--strict-cuda"], CUDA_VISIBLE_DEVICES="")
    out = refused.stdout + refused.stderr
    assert refused.returncode != 0, out
    assert "Refusing rather than skipping" in out, out
    # Once, on the controller -- not once per worker.
    assert out.count("Refusing rather than skipping") == 1, out
    assert " passed" not in refused.stdout, out

    surface = tmp_path / "surface.json"
    ran = _run([STDLIB_ONLY_TEST, "-q", "-n", "2", "--dist", "loadfile",
                "--surface-json", str(surface)], CUDA_VISIBLE_DEVICES="")
    assert ran.returncode == 0, ran.stdout + ran.stderr

    import json

    payload = json.loads(surface.read_text())
    # The aggregate the controller collected, not a single worker's share.
    assert payload["counts"]["passed"] == _passed_in_tail(ran.stdout)
    assert payload["counts"]["passed"] > 1


def _passed_in_tail(stdout):
    match = re.search(r"(\d+) passed", stdout)
    return int(match.group(1)) if match else 0


def test_a_parallel_run_writes_one_population_and_named_worker_shares(tmp_path):
    """Eight shards on the population's path, filed as eight retries.

    ``pytest_terminal_summary`` runs in every xdist worker, not only in the
    controller, and each worker's ``stats`` are that worker's SHARE of the run.
    The first spelling of ``--surface-json`` sent all of them to the arm's one
    canonical path, so a single ``-n 8`` run wrote eight shards over each other
    before the controller's aggregate landed on top.  ``_keep_any_previous``
    then renamed the eight to ``superseded-<mtime>`` -- a name whose meaning is
    "an earlier run wrote this path".  Receipt ``20260904T040432`` on
    ``/mnt/shared`` therefore records eight retries of a run that ran ONCE, and
    its shards' passed counts sum to exactly the aggregate (1406) with 520
    skips against the aggregate's 499 -- 21 = 7x3 duplicated collection skips.
    False provenance, in the one artefact this branch exists to make
    trustworthy.  And had that run been killed between a worker's write and the
    controller's, the canonical path would have held a shard -- 206 passed /
    108 skipped, in that receipt -- and ``--resume`` would have recorded it in
    ``docs/status/suite-populations.md`` as the arm's population.

    Before this test, on dl380g10 under ``/home/rob/venvs/pb-cpu`` -- the
    interpreter that has xdist, and the one the ``-n 8`` run used::

        AssertionError: a run that ran once left retry-named files:
        ['surface.x86.superseded-20260904T084454Z.json',
         'surface.x86.superseded-20260904T084456Z.json']

    Two workers, two shards, one run.  Eight of them at ``-n 8``.

    ``--dist load`` rather than production's ``loadfile`` for one reason: it
    guarantees both workers get tests, so the sum-equals-aggregate assertion
    discriminates.  Which worker gets which test is not what is under test; the
    path each worker writes is, and that is chosen the same way in both modes.
    """

    pytest.importorskip("xdist", reason="parallel run needs pytest-xdist")

    surface = tmp_path / "surface.x86.json"
    ran = _run([STDLIB_ONLY_TEST, "-q", "-n", "2", "--dist", "load",
                "--surface-json", str(surface)], CUDA_VISIBLE_DEVICES="")
    out = ran.stdout + ran.stderr
    assert ran.returncode == 0, out

    import json

    # A run that ran ONCE leaves no trace of a retry.  This is the assertion
    # the receipt on /mnt/shared fails.
    kept = sorted(p.name for p in tmp_path.glob("*superseded*"))
    assert kept == [], f"a run that ran once left retry-named files: {kept}"

    payload = json.loads(surface.read_text())
    assert payload["role"] == "population", payload
    assert payload["worker_id"] is None, payload
    assert payload["xdist_workers"] == 2, payload

    shares = sorted(tmp_path.glob("surface.x86.gw*.json"))
    assert len(shares) == 2, sorted(p.name for p in tmp_path.iterdir())
    total = 0
    seen = set()
    for share in shares:
        slice_ = json.loads(share.read_text())
        assert slice_["role"] == "worker-share", slice_
        # The worker names itself, and its name is in its filename: a reader
        # of the directory can tell the shards apart and tell them from the
        # population without opening anything.
        assert slice_["worker_id"], slice_
        assert slice_["worker_id"] in share.name, share.name
        assert slice_["worker_id"] not in seen, "two workers, one path"
        seen.add(slice_["worker_id"])
        assert slice_["counts"]["passed"] > 0, "a share with no work proves nothing"
        total += slice_["counts"]["passed"]

    # The relation that identified the shards in the first place: the shares
    # partition the run.  Asserting it here is what makes the split provable
    # rather than plausible.
    assert total == payload["counts"]["passed"] == _passed_in_tail(ran.stdout), (
        f"shares sum to {total}, population says "
        f"{payload['counts']['passed']}, pytest said {_passed_in_tail(ran.stdout)}"
    )


def test_the_population_names_the_tree_it_was_measured_on(tmp_path):
    """A population without its commit is half a receipt.

    The arms of a merge run are separate processes on separate boxes, and
    nothing makes them start together: an x86 arm can publish and finish while
    the GPU arm is still queued behind a held reservation, and the clone the
    queued arm will run in can be fast-forwarded while it waits.  A receipt
    that asks the checkout which commit it is at *assembly* time then stamps
    one commit on two arms that ran two trees.

    So the run states its own commit, at the moment it runs, in the same
    object as its counts -- the same reason the device is in there.  Before
    this, ``payload["commit"]`` raised ``KeyError: 'commit'``.
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
    import subprocess as sp

    payload = json.loads(surface.read_text())
    head = sp.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                  capture_output=True, text=True).stdout.strip()
    assert payload["commit"] == head, payload["commit"]
    # Not a guess dressed as a fact: an interpreter that cannot answer says so.
    assert payload["commit"] is None or len(payload["commit"]) == 40


def test_a_second_run_keeps_the_population_the_first_one_published(tmp_path):
    """A retry must not erase the measurement it is retrying.

    ``--surface-json`` names one path per arm, and the pool retries an action
    in place: an expired lease or a dead worker sends the same action back
    through a worker, which runs the suite again and writes the same filename.
    That happened here -- the population at
    ``20260904T025044/surface.x86.json`` (1389 passed / 1 failed) was replaced
    at 07:28:49Z by a retry reporting 1388/2, from a checkout that had moved in
    between.  Two populations, possibly of two trees, and the first was gone.

    Before this the last assertion read
    ``AssertionError: assert 0 == 1`` -- no superseded file, because the first
    one had been written over.
    """

    pytest.importorskip("torch")
    probe = _write_synthetic(tmp_path)
    surface = tmp_path / "surface.json"
    for _ in range(2):
        result = _run(
            [str(probe), "-q", "-p", "conftest", "--surface-json", str(surface)],
            CUDA_VISIBLE_DEVICES="",
        )
        assert result.returncode == 0, result.stdout + result.stderr

    import json

    # The plain name is always the newest, so nothing that reads it changes.
    assert json.loads(surface.read_text())["schema"] == "tessera.test_surface.v2"
    kept = sorted(tmp_path.glob("surface.superseded-*.json"))
    assert len(kept) == 1, sorted(p.name for p in tmp_path.iterdir())
    assert json.loads(kept[0].read_text())["counts"]["passed"] == 1
    assert "kept at" in (result.stdout + result.stderr)


def test_a_child_run_is_not_the_worker_that_launched_it(tmp_path):
    """``PYTEST_XDIST_WORKER`` is inherited; it never says who THIS run is.

    Under ``-n``, xdist sets that variable in each worker's environment, and
    every process a test starts from there inherits it.  A nested pytest is
    its own controller -- it has no ``workerinput`` -- so reading the variable
    made it file its whole run as ``surface.gw1.json`` and leave the path it
    was asked for empty.  That is the converse of the bug the shard fix
    closed, and it shipped inside the fix: the ``-n 8`` x86 population of
    ``82f0047`` on dl380g10 was 1536 passed / **5 failed** / 503 skipped, all
    five in this file, and the same commit run serially was green.

    Serial on purpose, and with no xdist needed: the point is a process that
    is NOT a worker but is told it is.

    Before this test::

        >       assert surface.exists(), sorted(p.name for p in tmp_path.iterdir())
        E       AssertionError: ['surface.gw1.json']
    """

    import json

    surface = tmp_path / "surface.json"
    ran = _run([STDLIB_ONLY_TEST, "-q", "--surface-json", str(surface)],
               CUDA_VISIBLE_DEVICES="", PYTEST_XDIST_WORKER="gw1")
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert surface.exists(), sorted(p.name for p in tmp_path.iterdir())
    payload = json.loads(surface.read_text())
    assert payload["role"] == "population", payload
    assert payload["worker_id"] is None, payload
