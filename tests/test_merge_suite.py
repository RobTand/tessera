"""The merge suite composes submissions; these pin what it composes.

``tools/merge_suite.py`` never runs a suite itself -- it builds ``pbrun``
command lines and reads back what they publish.  So the things worth pinning
are the properties of those command lines and of the verdict it derives, not a
placement, which belongs to the pool.
"""

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "merge_suite.py"


def _module():
    spec = importlib.util.spec_from_file_location("_merge_suite", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_gpu_arm_carries_the_gate_and_the_cpu_arm_does_not():
    """``--strict-cuda`` on the arm that exists to cover the CUDA surface.

    Without it the GPU arm placed on a device-less box returns a green tick for
    a population it never touched -- the exact shape of tessera#112.  The x86
    arm must NOT carry it: that box has no torch by design, and refusing there
    would make an honest population unreportable.
    """

    merge_suite = _module()
    surface = Path("/dev/null")
    gpu = merge_suite._command(merge_suite.ARMS["gpu"], surface, [])
    x86 = merge_suite._command(merge_suite.ARMS["x86"], surface, [])
    assert "--strict-cuda" in gpu
    assert "--strict-cuda" not in x86
    for name, command in (("gpu", gpu), ("x86", x86)):
        # Both arms must publish a population, or the receipt has nothing to
        # put side by side.
        assert "--surface-json" in command
        # The interpreter is NAMED -- taken from the arm table -- never
        # inherited from whatever launched this test, because a pool action
        # runs in a sealed environment on a box that is not this one.
        #
        # ``command[0] != sys.executable`` was the first spelling and it is a
        # different claim wearing the same words: it asks whether the named
        # interpreter happens to differ from the running one, which is a fact
        # about WHERE the test runs.  It passed on sparky and failed on
        # dl380g10, whose python IS the x86 arm's named interpreter, and it
        # would fail on the GPU arm too for the same reason.  A test whose
        # verdict depends on its box is the blindness tessera#112 is about,
        # and this branch's own first cross-population run is what caught it.
        assert command[0] == merge_suite.ARMS[name]["python"]
        assert command[0].startswith("/")



def test_the_declared_cores_are_the_cores_the_command_uses():
    """A reservation the command does not spend is an over-declaration.

    ``--cpus`` reserves that many cores on the chosen box for the life of the
    action.  Submitting a serial pytest under ``--cpus 8`` idles seven of them
    -- the pool's ledger says the box is busy while it is not, which is the
    accounting failure the pool exists to prevent.  One number now feeds both
    the reservation and pytest's ``-n``, so they cannot drift apart.

    Serial stays the default on purpose: ``-n`` needs pytest-xdist in the
    TARGET venv, and sparky's CUDA venv inherits a system interpreter that has
    pytest 9.0.3 and no xdist, so a hardcoded ``-n`` would abort that arm.
    """

    merge_suite = _module()
    surface = Path("/dev/null")
    for name in ("gpu", "x86"):
        serial = merge_suite._command(merge_suite.ARMS[name], surface, [], 1)
        assert "-n" not in serial, "one declared core must not fan out"

        parallel = merge_suite._command(merge_suite.ARMS[name], surface, [], 6)
        assert parallel[parallel.index("-n") + 1] == "6"
        # A module's tests share fixtures, and on the GPU arm device state.
        assert parallel[parallel.index("--dist") + 1] == "loadfile"
        # The gate survives the fan-out: it is asserted on the controller,
        # which is the process that has -- or has not -- the device.
        assert ("--strict-cuda" in parallel) == merge_suite.ARMS[name]["strict_cuda"]


def test_the_default_submission_declares_one_core(tmp_path):
    """The default is honest with no operator knowledge of the target box.

    Whoever submits this may not know whether the target venv has xdist. The
    default must therefore be the one that is true everywhere -- and it must
    be the SUBMITTED line that says so, not just the composed command, since
    the reservation is what the pool acts on.
    """

    out = tmp_path / "receipt.json"
    subprocess.run(
        [sys.executable, str(TOOL), "--arm", "gpu", "--dry-run",
         "--checkout", str(ROOT), "--out", str(out)],
        capture_output=True, text=True, timeout=120, check=False,
    )
    pbrun = json.loads(out.read_text())["arms"][0]["pbrun"]
    assert "--cpus 1" in pbrun, pbrun
    assert " -n " not in pbrun, pbrun


def test_a_missing_surface_is_reported_as_absent_not_as_a_pass():
    """An arm that was never placed is not a green arm."""

    merge_suite = _module()
    assert merge_suite._verdict(
        [{"arm": "gpu", "surface": None, "returncode": 0}]
    ) == "incomplete: an arm published no population"
    assert merge_suite._verdict([
        {"arm": "gpu", "surface": {}, "returncode": 0},
        {"arm": "x86", "surface": {}, "returncode": 1},
    ]) == "red on one of: gpu, x86"


def test_a_green_verdict_names_the_populations_it_is_green_on():
    """One arm run must never report a verdict about two."""

    merge_suite = _module()
    one = merge_suite._verdict([{"arm": "gpu", "surface": {}, "returncode": 0}])
    two = merge_suite._verdict([
        {"arm": "gpu", "surface": {}, "returncode": 0},
        {"arm": "x86", "surface": {}, "returncode": 0},
    ])
    assert one == "green on 1 population(s): gpu"
    assert two == "green on 2 population(s): gpu, x86"
    assert "x86" not in one


def test_the_x86_arm_refuses_a_checkout_only_one_box_can_see():
    """pbrun pins a local checkout to the submitting box; say so, do not route around it."""

    result = subprocess.run(
        [sys.executable, str(TOOL), "--arm", "x86", "--dry-run",
         "--checkout", str(ROOT)],
        capture_output=True, text=True, timeout=120,
    )
    if str(ROOT).startswith("/mnt/shared"):
        pytest.skip("this checkout IS shared, so the refusal cannot fire here")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "/mnt/shared" in result.stderr


def test_the_receipt_states_which_tree_it_is_about(tmp_path):
    """A branch receipt is not a merge receipt, and must not read as one."""

    out = tmp_path / "receipt.json"
    result = subprocess.run(
        [sys.executable, str(TOOL), "--arm", "gpu", "--dry-run",
         "--checkout", str(ROOT), "--out", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode != 0, "a dry run has covered no population"
    receipt = json.loads(out.read_text())
    population = receipt["population"]
    assert population["commit"]
    assert "is_master_head" in population
    # A clone made for a pool run carries only ``origin/master``; the ref that
    # actually answered is recorded so an unresolved comparison reads as
    # "not established" rather than as "not master".
    assert population["master_ref_used"] != "none resolved"
    assert population["is_master_head"] is not None
    assert receipt["verdict"] == "not run"
    # Both arms' numbers live under one key, so quoting one without its device
    # means quoting it out of this object rather than out of a scrollback.
    assert "reading_note" in receipt


def test_the_ledger_puts_the_two_arms_next_to_each_other(tmp_path):
    """A row without its device is the misreading; there is no such row.

    The markdown ledger is the "somewhere a reader checks" half of #112 item 1.
    Its shape is the whole argument: one row per arm, the arms of a run
    adjacent, and the device in the same row as the counts.
    """

    merge_suite = _module()
    ledger = tmp_path / "suite-populations.md"
    receipt = {
        "generated_utc": "2026-09-04T00:00:00Z",
        "population": {"commit": "0123456789abcdef", "is_master_head": True},
        "arms": [
            {"arm": "gpu", "surface": {
                "device": "torch 2.11, 1 CUDA device(s), device 0 = NVIDIA GB10",
                "counts": {"passed": 1827, "failed": 0, "skipped": 10},
                "not_collected": []}},
            {"arm": "x86", "surface": {
                "device": "torch 2.10.0+cpu reports no CUDA device",
                "counts": {"passed": 1381, "failed": 0, "skipped": 497},
                "not_collected": []}},
        ],
    }
    merge_suite._record_markdown(ledger, receipt)
    text = ledger.read_text()
    assert "| measured (UTC) |" in text
    gpu_row = [line for line in text.splitlines() if "| gpu |" in line]
    x86_row = [line for line in text.splitlines() if "| x86 |" in line]
    assert len(gpu_row) == len(x86_row) == 1
    assert "NVIDIA GB10" in gpu_row[0] and "1827" in gpu_row[0]
    assert "no CUDA device" in x86_row[0] and "1381" in x86_row[0]

    # A second run appends rather than replacing: the header is written once.
    merge_suite._record_markdown(ledger, receipt)
    assert ledger.read_text().count("| measured (UTC) |") == 1
    assert ledger.read_text().count("| gpu |") == 2


def test_an_arm_with_no_population_says_so_in_the_ledger(tmp_path):
    """Never a blank cell that reads as zero failures."""

    merge_suite = _module()
    ledger = tmp_path / "ledger.md"
    merge_suite._record_markdown(ledger, {
        "generated_utc": "2026-09-04T00:00:00Z",
        "population": {"commit": "deadbeefcafe", "is_master_head": None},
        "arms": [{"arm": "gpu", "surface": None}],
    })
    row = [line for line in ledger.read_text().splitlines() if "| gpu |" in line][0]
    assert "no population published" in row
    assert "| -- | -- | -- |" in row
    assert "| unknown |" in row


def test_a_resumed_receipt_reports_failures_but_never_declares_green(tmp_path):
    """The asymmetry is the point: red is provable from a surface, green is not.

    A run whose submitting process died still finished on the pool and still
    published its population. Assembling the receipt from those files is what
    makes the result survive the terminal (#112 item 1); pretending the exit
    status was observed is not. A suite can exit non-zero after a clean
    summary -- a crash in teardown, an internal error, a timeout kill -- so
    failures in the surface prove red while their absence proves nothing.
    """

    merge_suite = _module()
    clean = {"counts": {"passed": 10, "failed": 0, "error": 0, "skipped": 1},
             "device": "torch 2.11, 1 CUDA device(s), device 0 = NVIDIA GB10"}
    broken = {"counts": {"passed": 9, "failed": 1, "error": 0, "skipped": 1},
              "device": "torch 2.11.0+cpu reports no CUDA device"}

    (tmp_path / "surface.gpu.json").write_text(json.dumps(clean))
    record = merge_suite._resume("gpu", merge_suite.ARMS["gpu"], tmp_path)
    assert record["exit_status_observed"] is False
    assert record["returncode"] is None
    verdict = merge_suite._verdict([record])
    assert "green" not in verdict
    assert "exit status not observed" in verdict and "gpu" in verdict

    # A failure the run published is conclusive even unobserved.
    (tmp_path / "surface.x86.json").write_text(json.dumps(broken))
    both = [record, merge_suite._resume("x86", merge_suite.ARMS["x86"], tmp_path)]
    assert merge_suite._verdict(both) == "red on one of: gpu, x86"

    # An arm that published nothing is still an absent measurement, not a pass.
    assert merge_suite._verdict(
        [merge_suite._resume("gpu", merge_suite.ARMS["gpu"], tmp_path / "nope")]
    ) == "incomplete: an arm published no population"


def test_resume_submits_nothing_and_needs_no_shared_checkout(tmp_path):
    """The refusal that guards a submission must not block reading a result.

    The x86 arm refuses a checkout only one box can see, because pbrun would
    pin the action to this box. A resume submits nothing, so there is no
    placement to constrain -- and the run it reads already happened.
    """

    surfaces = tmp_path / "surfaces"
    surfaces.mkdir()
    for arm, device in (("gpu", "1 CUDA device(s), device 0 = NVIDIA GB10"),
                        ("x86", "torch 2.11.0+cpu reports no CUDA device")):
        (surfaces / f"surface.{arm}.json").write_text(json.dumps(
            {"device": device,
             "counts": {"passed": 5, "failed": 0, "skipped": 2},
             "not_collected": []}))
    out = tmp_path / "receipt.json"
    ledger = tmp_path / "ledger.md"
    result = subprocess.run(
        [sys.executable, str(TOOL), "--resume", str(surfaces),
         "--checkout", str(ROOT), "--out", str(out), "--record", str(ledger)],
        capture_output=True, text=True, timeout=120,
    )
    # Not green: nobody watched either exit status.
    assert result.returncode == 1, result.stdout + result.stderr
    receipt = json.loads(out.read_text())
    assert receipt["assembled_by"] == "resume"
    assert [arm["arm"] for arm in receipt["arms"]] == ["gpu", "x86"]
    assert "exit status not observed" in receipt["verdict"]
    # Both populations still land side by side in the ledger, which is the
    # whole reason to assemble a receipt at all.
    text = ledger.read_text()
    assert "NVIDIA GB10" in text and "no CUDA device" in text


def test_a_resumed_row_is_dated_by_the_run_and_never_looks_watched(tmp_path):
    """Two ways a resumed row could quietly overclaim; neither is allowed.

    Found by reading the first real ledger this tool wrote. The row carried
    the time the row was WRITTEN, not the time the suite ran -- for a resume
    that is hours off, and it is the measurement a reader is dating. And its
    exit cell would have read like any watched row, when nobody watched it.

    A resumed row therefore takes its date from the surface file the run
    published, and says `not observed` where a watched row says a status.
    """

    merge_suite = _module()
    surfaces = tmp_path / "s"
    surfaces.mkdir()
    surface = surfaces / "surface.gpu.json"
    surface.write_text(json.dumps({
        "device": "torch 2.11, 1 CUDA device(s), device 0 = NVIDIA GB10",
        "counts": {"passed": 1827, "failed": 0, "skipped": 10},
        "not_collected": []}))
    # A run that finished well before anyone came back for its receipt.
    ran_at = 1788504000
    os.utime(surface, (ran_at, ran_at))

    record = merge_suite._resume("gpu", merge_suite.ARMS["gpu"], surfaces)
    expected = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ran_at))
    assert record["measured_utc"] == expected

    ledger = tmp_path / "ledger.md"
    merge_suite._record_markdown(ledger, {
        "generated_utc": "2099-01-01T00:00:00Z",       # bookkeeping time
        "population": {"commit": "abcdef012345", "is_master_head": False},
        "arms": [record],
    })
    row = [line for line in ledger.read_text().splitlines() if "| gpu |" in line][0]
    assert expected in row, row
    assert "2099" not in row, "the row was dated by the bookkeeping, not the run"
    # Zero failures on an unwatched run is not a green row, and must not read
    # as one.
    assert row.rstrip().endswith("| not observed |"), row

    # A watched arm still records the status that was actually seen.
    merge_suite._record_markdown(ledger, {
        "generated_utc": "2026-09-04T00:00:00Z",
        "population": {"commit": "abcdef012345", "is_master_head": True},
        "arms": [{"arm": "x86", "returncode": 0, "surface": {
            "device": "torch 2.11.0+cpu reports no CUDA device",
            "counts": {"passed": 1381, "failed": 0, "skipped": 497},
            "not_collected": []}}],
    })
    watched = [line for line in ledger.read_text().splitlines() if "| x86 |" in line][0]
    assert watched.rstrip().endswith("| 0 |"), watched
