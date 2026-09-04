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
    arm must NOT carry it: that box has no CUDA device, and refusing there
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


def test_the_x86_arm_refuses_a_checkout_only_one_box_can_see(tmp_path):
    """pbrun pins a local checkout to the submitting box; say so, do not route around it.

    The skip is decided BEFORE the tool runs, not after.  Deciding it after
    still ran a dry run, and a dry run with no ``--out`` writes its receipt
    under ``DEFAULT_RECEIPT_ROOT`` -- so every full suite run on a shared
    checkout left a directory in the store that holds the real ones.  Sixteen
    of them were there when this was found, fourteen written by pool runs on
    dl380g10, each holding one arm with ``"status": "not submitted
    (--dry-run)"``.  Nothing read them, but a reader of the store cannot tell
    a run that measured nothing from one that has not finished, which is the
    reading tessera#112 is about.
    """

    if str(ROOT).startswith("/mnt/shared"):
        pytest.skip("this checkout IS shared, so the refusal cannot fire here")
    result = subprocess.run(
        [sys.executable, str(TOOL), "--arm", "x86", "--dry-run",
         "--checkout", str(ROOT), "--out", str(tmp_path / "receipt.json")],
        capture_output=True, text=True, timeout=120,
    )
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
    # ``cuda``/``strict_cuda`` are in every surface a run publishes, and the
    # verdict now reads them for an arm submitted to cover the CUDA surface
    # (see the second-leg test below).  A fixture without them is not a
    # smaller version of a real population; it is a different one.
    clean = {"counts": {"passed": 10, "failed": 0, "error": 0, "skipped": 1},
             "cuda": True, "strict_cuda": True,
             "device": "torch 2.11, 1 CUDA device(s), device 0 = NVIDIA GB10"}
    broken = {"counts": {"passed": 9, "failed": 1, "error": 0, "skipped": 1},
              "cuda": False, "strict_cuda": False,
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
    for arm, cuda, device in (
            ("gpu", True, "1 CUDA device(s), device 0 = NVIDIA GB10"),
            ("x86", False, "torch 2.11.0+cpu reports no CUDA device")):
        (surfaces / f"surface.{arm}.json").write_text(json.dumps(
            {"device": device, "cuda": cuda, "strict_cuda": cuda,
             "role": "population",
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
    # ``[-1]``, not ``[0]``: the gpu-only receipt above now also writes an x86
    # row saying that arm was not submitted, so the first x86 line in the file
    # is that absence and the watched row is the one just appended.
    x86_rows = [line for line in ledger.read_text().splitlines() if "| x86 |" in line]
    assert "not submitted in this run" in x86_rows[0], x86_rows
    watched = x86_rows[-1]
    assert watched.rstrip().endswith("| 0 |"), watched


def test_each_row_names_the_tree_its_own_arm_ran(tmp_path):
    """Two arms that ran two commits must not be stamped with one.

    The arms are separate processes on separate boxes and nothing makes them
    start together.  The real case: the x86 arm ran and published on
    ``e61974c``; the GPU arm sat in the queue behind a held reservation, and
    the clone it would run in was fast-forwarded while it waited.  Assembling
    that receipt reads the checkout ONCE, at assembly time, so both rows would
    have carried the later commit -- and the x86 row would have been attributed
    to a tree it never saw.

    That is this branch's own thesis violated by this branch's own tool: a
    measurement separated from the context that gives it meaning.  Before the
    fix the second assertion below read
    ``AssertionError: '| `bbbbbbbbbbbb`' not in '| 2026-... | `aaaaaaaaaaaa` (assumed) | ...'``
    -- the x86 row wearing the GPU arm's commit.
    """

    merge_suite = _module()
    ledger = tmp_path / "ledger.md"
    merge_suite._record_markdown(ledger, {
        "generated_utc": "2026-09-04T09:00:00Z",
        "population": {"commit": "a" * 40, "is_master_head": True},
        "arms": [
            {"arm": "x86", "returncode": 0, "measured_utc": "2026-09-04T07:04:59Z",
             "surface": {"commit": "b" * 40,
                         "device": "torch 2.11.0+cpu reports no CUDA device",
                         "counts": {"passed": 1389, "failed": 0, "skipped": 499},
                         "not_collected": []}},
            {"arm": "gpu", "returncode": 0, "measured_utc": "2026-09-04T09:00:00Z",
             "surface": {"commit": "a" * 40,
                         "device": "torch 2.11, 1 CUDA device(s), device 0 = NVIDIA GB10",
                         "counts": {"passed": 1827, "failed": 0, "skipped": 10},
                         "not_collected": []}},
        ],
    })
    text = ledger.read_text()
    x86 = [line for line in text.splitlines() if "| x86 |" in line][0]
    gpu = [line for line in text.splitlines() if "| gpu |" in line][0]

    assert "`" + "a" * 12 + "`" in gpu, gpu
    assert "`" + "b" * 12 + "`" in x86, x86
    assert "a" * 12 not in x86, "the x86 row was stamped with the other arm's tree"
    # ``master head?`` was answered about the population commit. The x86 arm
    # did not run that commit, so that answer is not about its row.
    assert "| unknown |" in x86, x86
    assert "| yes |" in gpu, gpu


def test_a_population_with_no_commit_is_labelled_a_guess(tmp_path):
    """An unstamped surface leaves the question open; it does not answer it.

    Surfaces written before the field existed cannot say which tree they ran.
    The receipt's own commit is then the best available guess, and a guess
    that does not say so is indistinguishable from a measurement.
    """

    merge_suite = _module()
    ledger = tmp_path / "ledger.md"
    merge_suite._record_markdown(ledger, {
        "generated_utc": "2026-09-04T09:00:00Z",
        "population": {"commit": "c" * 40, "is_master_head": False},
        "arms": [{"arm": "x86", "returncode": 1, "surface": {
            "device": "torch 2.11.0+cpu reports no CUDA device",
            "counts": {"passed": 1389, "failed": 1, "skipped": 499},
            "not_collected": []}}],
    })
    row = [line for line in ledger.read_text().splitlines() if "| x86 |" in line][0]
    assert "`" + "c" * 12 + "` (assumed)" in row, row
    assert "(assumed)" in merge_suite.LEDGER_HEADER


def test_the_receipt_says_whether_the_arms_ran_one_tree():
    """A reader must not have to diff the rows to learn the arms disagreed."""

    merge_suite = _module()

    agree = merge_suite._commits_measured([
        {"arm": "gpu", "surface": {"commit": "a" * 40}},
        {"arm": "x86", "surface": {"commit": "a" * 40}}])
    assert agree["agree"] is True
    assert agree["unstamped_arms"] == []

    split = merge_suite._commits_measured([
        {"arm": "gpu", "surface": {"commit": "a" * 40}},
        {"arm": "x86", "surface": {"commit": "b" * 40}}])
    assert split["agree"] is False
    assert split["by_arm"]["x86"] == "b" * 40

    # Nothing stamped: not agreement, and not disagreement either.
    silent = merge_suite._commits_measured([
        {"arm": "gpu", "surface": {}}, {"arm": "x86", "surface": None}])
    assert silent["agree"] is None
    assert silent["unstamped_arms"] == ["gpu", "x86"]

    # One arm stamped and one silent is NOT agreement. Reporting `True` here
    # -- one commit in the set, so they "match" -- would be this file's own
    # error in miniature: an unanswered question rendered as an answer.
    half = merge_suite._commits_measured([
        {"arm": "gpu", "surface": {"commit": "a" * 40}},
        {"arm": "x86", "surface": {}}])
    assert half["agree"] is None, "a silent arm cannot agree with anything"
    assert half["unstamped_arms"] == ["x86"]


def test_the_recorded_ledger_is_written_in_the_tools_current_dialect():
    """The ledger in the repo must be readable by its own header.

    `_record_markdown` writes the header only when the file does not exist, so
    a ledger created once keeps whatever header it was born with while the
    tool's columns and their meanings move on.  That is how the committed
    ledger came to render two *assumed* commit attributions as established
    ones -- under a header that had no word for the difference.

    Before this test the last assertion read
    ``AssertionError: assert False`` on
    ``ledger.startswith(merge_suite.LEDGER_HEADER)``.
    """

    merge_suite = _module()
    ledger = Path(__file__).resolve().parents[1] / "docs/status/suite-populations.md"
    if not ledger.exists():                      # no run has recorded one yet
        pytest.skip("no ledger recorded in this checkout")
    text = ledger.read_text()
    assert text.startswith(merge_suite.LEDGER_HEADER), (
        "docs/status/suite-populations.md was written by an older dialect of "
        "tools/merge_suite.py; its rows may not mean what its header says")
    # Every row is a row of the table the header declares.
    columns = merge_suite.LEDGER_HEADER.strip().splitlines()[-2].count("|")
    for line in text.splitlines():
        if line.startswith("| 2026-") or line.startswith("| 20"):
            assert line.count("|") == columns, line


def test_an_arm_that_measured_nothing_carries_no_measurement_time(tmp_path):
    """No population means no measurement, so no date for one.

    The cell fell through to the receipt's own clock, which put a plausible
    timestamp beside `no population published` -- a row that had measured
    nothing, wearing the time something was measured. Before this the last
    assertion read
    ``AssertionError: assert '2026-09-04T09:00:00Z' == '--'``.

    ``tmp_path``, not ``tempfile.mkdtemp(dir="/home/rob/tmp")``: the hardcoded
    path is a fact about this fleet, and a test whose verdict depends on its
    box is the blindness tessera#112 is about. Commit 05777a8 on this branch
    fixed one of those in this same file; this was the other.
    """

    merge_suite = _module()
    ledger = tmp_path / "l.md"
    merge_suite._record_markdown(ledger, {
        "generated_utc": "2026-09-04T09:00:00Z",
        "population": {"commit": "e" * 40, "is_master_head": False},
        "arms": [{"arm": "gpu", "surface": None,
                  "exit_status_observed": False, "returncode": None}],
    })
    row = [l for l in ledger.read_text().splitlines() if "| gpu |" in l][0]
    assert "no population published" in row, row
    assert "2026-09-04T09:00:00Z" not in row, row
    assert row.split("|")[1].strip() == "--", row


def test_a_worker_share_is_never_read_as_this_arms_population(tmp_path):
    """A shard on the population's path is an absent measurement, not a result.

    Under `-n 8` every xdist worker used to write the arm's canonical
    `--surface-json` path, so that path held one worker's SHARE until the
    controller's final write. A `--timeout-s` kill in that window, followed by
    `--resume`, would have recorded a fraction of a suite -- 206 passed / 0
    failed / 108 skipped, in receipt `20260904T040432` -- in
    `docs/status/suite-populations.md` as the x86 arm's population. Not a false
    green (`_verdict` withholds green when nobody observed an exit status) but
    a wrong number in the permanent artefact, which is the thing #112 item 1
    asked for.

    `tests/conftest.py` now gives each worker its own path, so this cannot
    happen by accident. This is the reader's own leg: a file that says it is a
    share is refused as a population however it got there.

    Before this::

        AssertionError: assert {'counts': {'error': 0, 'failed': 0,
        'passed': 206, 'skipped': 108}, 'cuda': False, ...} is None
    """

    merge_suite = _module()
    (tmp_path / "surface.x86.json").write_text(json.dumps({
        "schema": "tessera.test_surface.v2",
        "role": "worker-share", "worker_id": "gw6", "xdist_workers": 8,
        "cuda": False, "strict_cuda": False,
        "device": "torch 2.11.0+cpu reports no CUDA device",
        "counts": {"passed": 206, "failed": 0, "error": 0, "skipped": 108},
        "not_collected": []}))

    record = merge_suite._resume("x86", merge_suite.ARMS["x86"], tmp_path)
    assert record["surface"] is None
    assert "gw6" in record["no_surface_means"]
    assert "206" not in json.dumps(record), "a share's counts reached the record"
    assert merge_suite._verdict([record]) == \
        "incomplete: an arm published no population"

    # And the row a reader sees says nothing was measured, not 206.
    ledger = tmp_path / "ledger.md"
    merge_suite._record_markdown(ledger, {
        "generated_utc": "2026-09-04T09:00:00Z",
        "population": {"commit": "f" * 40, "is_master_head": True},
        "arms": [record]})
    row = [l for l in ledger.read_text().splitlines() if "| x86 |" in l][0]
    assert "no population published" in row, row
    assert "206" not in row, row


def test_a_population_that_states_its_role_is_read_and_a_silent_one_is_flagged(tmp_path):
    """A pre-v2 surface is read, and the open question is written down.

    Refusing every file without a `role` would refuse the population the queued
    GPU arm will publish if it places on a tree that predates the field, which
    would be throwing away the measurement everyone is waiting for. Reading it
    silently would be pretending v1 answered a question it cannot. So: read it,
    and record that it did not say.

    Before this: `KeyError: 'surface_role'`.
    """

    merge_suite = _module()
    (tmp_path / "surface.gpu.json").write_text(json.dumps({
        "schema": "tessera.test_surface.v1",
        "cuda": True, "strict_cuda": True,
        "device": "torch 2.11, 1 CUDA device(s), device 0 = NVIDIA GB10",
        "counts": {"passed": 1827, "failed": 0, "skipped": 10},
        "not_collected": []}))
    record = merge_suite._resume("gpu", merge_suite.ARMS["gpu"], tmp_path)
    assert record["surface"]["counts"]["passed"] == 1827
    assert "unstated" in record["surface_role"], record["surface_role"]

    (tmp_path / "surface.x86.json").write_text(json.dumps({
        "schema": "tessera.test_surface.v2", "role": "population",
        "worker_id": None, "cuda": False, "strict_cuda": False,
        "device": "torch 2.11.0+cpu reports no CUDA device",
        "counts": {"passed": 1406, "failed": 0, "skipped": 499},
        "not_collected": []}))
    stated = merge_suite._resume("x86", merge_suite.ARMS["x86"], tmp_path)
    assert stated["surface_role"] == "population"


def test_the_gpu_arms_green_has_two_legs_not_one(tmp_path):
    """A pass count is not coverage; the population has to say it saw a device.

    The GPU arm's entire claim to have covered the CUDA-gated surface rested on
    `--strict-cuda` having refused a device-less session -- one code path, and
    one that has never executed its ACCEPT branch on a real device. If it were
    ever mis-wired, mis-spelled or dropped from the submitted command line, a
    placement on a box with no GPU would skip the whole surface, publish 1406
    passed / 0 failed, and this tool would call it green. That is tessera#112
    reproduced inside the tool written to prevent it.

    The surface already publishes `cuda` and `strict_cuda`, both derived
    in-process from torch on the box that ran -- an attestation, not a claim
    about another runtime (principle 14) -- so the verdict reads them as a
    second, independent leg.

    Before this test::

        AssertionError: green on 1 population(s): gpu
        assert 'green' not in 'green on 1 population(s): gpu'

    -- for the first case below, a GPU arm that saw no device at all.
    """

    merge_suite = _module()
    cpu_population = {
        "cuda": False, "strict_cuda": False,
        "device": "torch 2.11.0+cpu reports no CUDA device",
        "counts": {"passed": 1406, "failed": 0, "error": 0, "skipped": 499}}

    # Submitted to cover the CUDA surface, ran on a box with no device.
    landed_wrong = {"arm": "gpu", "requires_cuda": True, "returncode": 0,
                    "surface": cpu_population}
    verdict = merge_suite._verdict([landed_wrong])
    assert "green" not in verdict, verdict
    assert "no device" in verdict and "gpu" in verdict, verdict

    # A device, but the gate was not armed: the run passed because nothing
    # made it refuse, which is the same coverage question one step earlier.
    unarmed = {"arm": "gpu", "requires_cuda": True, "returncode": 0,
               "surface": dict(cpu_population, cuda=True, strict_cuda=False,
                               device="torch 2.11, 1 CUDA device(s)")}
    assert "not armed" in merge_suite._verdict([unarmed])

    # Both legs: a device, and the gate that would have refused without one.
    covered = {"arm": "gpu", "requires_cuda": True, "returncode": 0,
               "surface": dict(cpu_population, cuda=True, strict_cuda=True,
                               device="torch 2.11, 1 CUDA device(s)")}
    assert merge_suite._verdict([covered]) == "green on 1 population(s): gpu"

    # The x86 arm was never submitted to cover that surface, and must not be
    # held to it -- that box has no torch by design.
    x86 = {"arm": "x86", "requires_cuda": False, "returncode": 0,
           "surface": cpu_population}
    assert merge_suite._verdict([x86]) == "green on 1 population(s): x86"

    # The arm records the tool builds carry the flag, so this is the verdict
    # the tool actually reaches -- not one only this test can construct.
    assert merge_suite._resume("gpu", merge_suite.ARMS["gpu"],
                              tmp_path)["requires_cuda"] is True
    assert merge_suite._resume("x86", merge_suite.ARMS["x86"],
                              tmp_path)["requires_cuda"] is False


def test_an_arm_the_run_did_not_submit_is_named_in_the_ledger(tmp_path):
    """The header promises two populations per run; the artefact must keep it.

    Three of the first four rows this tool wrote were lone `--arm x86` rows
    under a header that says "the two arms of a run are adjacent on purpose".
    A reader of `docs/status/suite-populations.md` saw a suite result with no
    second population beside it and no way to tell whether the other arm was
    elsewhere in the file, absent, or forgotten -- so the guarantee was true of
    the prose and false of the bytes.

    Three absences, three different sentences: `not submitted in this run`,
    `no population published`, and a device string.

    Before this test::

        AssertionError: ['# Suite populations', '', 'One row per arm per ...']
        assert 0 == 1
         +  where 0 = len([])

    -- the run that submitted only the GPU arm said nothing at all about the
    x86 population.
    """

    merge_suite = _module()
    ledger = tmp_path / "ledger.md"
    merge_suite._record_markdown(ledger, {
        "generated_utc": "2026-09-04T09:00:00Z",
        "population": {"commit": "a" * 40, "is_master_head": True},
        "arms": [{"arm": "gpu", "returncode": 0,
                  "measured_utc": "2026-09-04T09:00:00Z",
                  "surface": {"commit": "a" * 40, "cuda": True,
                              "device": "torch 2.11, 1 CUDA device(s)",
                              "counts": {"passed": 1827, "failed": 0,
                                         "skipped": 10},
                              "not_collected": []}}]})
    lines = ledger.read_text().splitlines()
    x86 = [l for l in lines if "| x86 |" in l]
    assert len(x86) == 1, lines
    assert "not submitted in this run" in x86[0], x86[0]
    # Never a number, and never the other absence's words.
    assert "no population published" not in x86[0], x86[0]
    assert "| -- | -- | -- | -- | -- |" in x86[0], x86[0]
    # Same dialect as every other row.
    columns = merge_suite.LEDGER_HEADER.strip().splitlines()[-2].count("|")
    assert x86[0].count("|") == columns, x86[0]

    # A run that submitted both arms adds no such row -- both were asked for,
    # and "published nothing" is the other sentence.
    before = ledger.read_text()
    merge_suite._record_markdown(ledger, {
        "generated_utc": "2026-09-04T10:00:00Z",
        "population": {"commit": "a" * 40, "is_master_head": True},
        "arms": [{"arm": "gpu", "returncode": 0, "surface": None},
                 {"arm": "x86", "returncode": 0, "surface": None}]})
    appended = ledger.read_text()[len(before):]
    assert "not submitted in this run" not in appended, appended
    assert appended.count("no population published") == 2, appended


def _fake_pool(root, surface_json, actions):
    """A pb-queue and a CAS request tree holding exactly these actions.

    The layout is the fleet's, not an invention: an outcome record per action
    in ``pb-queue/<state>/<key>.json`` carrying ``detail.returncode``, and the
    command that action ran in ``cas/requests/<key[:2]>/<key>.json``. Built
    here rather than read from ``/mnt/shared`` so the test states its own
    population -- a test whose verdict depends on what the live fleet happens
    to hold is the box-dependence this file already fixed twice.
    """

    queue, requests = root / "pb-queue", root / "cas" / "requests"
    for state in ("done", "failed"):
        (queue / state).mkdir(parents=True, exist_ok=True)
    for key, state, returncode, host in actions:
        (requests / key[:2]).mkdir(parents=True, exist_ok=True)
        (requests / key[:2] / f"{key}.json").write_text(json.dumps({
            "params": {"command": ["python", "-m", "pytest", "tests",
                                   "--surface-json", str(surface_json),
                                   "--strict-cuda"]}}))
        (queue / state / f"{key}.json").write_text(json.dumps({
            "attempts": 1, "claimed_host": host, "status": state,
            "detail": {"returncode": returncode, "status": "executed",
                       "elapsed_s": 511.4}}))
    return queue, requests


def _gpu_population(path, commit="a" * 40):
    path.write_text(json.dumps({
        "schema": "tessera.test_surface.v2",
        "role": "population",
        "worker_id": None,
        "commit": commit,
        "cuda": True,
        "device": "torch 2.11.0+cu130, 1 CUDA device(s), device 0 = GB10",
        "strict_cuda": True,
        "counts": {"passed": 1827, "failed": 0, "error": 0, "skipped": 10,
                   "xfailed": 0, "xpassed": 0},
        "skip_reasons": {},
        "not_collected": [],
    }))


def test_a_resumed_row_reads_the_exit_status_the_pool_recorded(tmp_path):
    """The run nobody here watched was watched by the worker that ran it.

    A resumed receipt declined to state an exit status at all, on the correct
    ground that this process never saw one -- and that ground stops being
    correct one directory over. PrismaBuild's worker waits on the child and
    writes the status it saw into the action's outcome record; the CAS request
    beside it holds the command, so "the action that wrote this population" is
    answerable by the ``--surface-json`` path rather than by matching prose.
    Reading that is a derivation from a table the pool publishes about its own
    execution, not an inference from a clean summary -- which is the thing
    ``_verdict`` refuses, and still refuses.

    It matters because it is the only shape the GPU arm has: both real GPU
    submissions on this branch outlived their submitting session, so a receipt
    that can only be resumed can only ever say ``not observed``, and an arm
    that can never be green is a gate that can never pass.

    Before this test, on this box under ``/usr/bin/python3`` (torch
    2.10.0+cpu, no device)::

        >       assert record["returncode"] == 0, record
        E       AssertionError: {'arm': 'gpu', 'exit_status_note': ...}
        E       assert None == 0
    """

    merge_suite = _module()
    receipt_dir = tmp_path / "receipt"
    receipt_dir.mkdir()
    surface = receipt_dir / "surface.gpu.json"
    _gpu_population(surface)
    merge_suite.POOL_QUEUE, merge_suite.POOL_CAS_REQUESTS = _fake_pool(
        tmp_path, surface, [("beef" + "0" * 60, "done", 0, "sparky")])

    record = merge_suite._resume("gpu", merge_suite.ARMS["gpu"], receipt_dir)
    assert record["returncode"] == 0, record
    assert record["exit_status_observed"] is True, record
    # Whose observation it is, on the record and on the row.
    assert record["exit_status_source"] == "pool", record
    assert record["pool_action"]["host"] == "sparky", record
    assert "PrismaBuild" in record["exit_status_note"], record

    # An arm that ran the surface it was submitted to cover, with a status
    # somebody saw, is the one thing this tool exists to be able to say.
    assert merge_suite._verdict([record]).startswith("green on"), \
        merge_suite._verdict([record])

    ledger = tmp_path / "l.md"
    merge_suite._record_markdown(ledger, {
        "generated_utc": "2026-09-04T09:00:00Z",
        "population": {"commit": "a" * 40, "is_master_head": True},
        "arms": [record],
    })
    row = [l for l in ledger.read_text().splitlines() if "| gpu |" in l][0]
    assert "0 (pool)" in row, row
    assert "1827" in row, row


def test_two_actions_on_one_population_leave_the_row_unobserved(tmp_path):
    """Several exit statuses is not one exit status.

    A retried action, or a receipt directory two runs shared -- and
    ``20260904T025044`` on this branch is one -- has no single status behind
    the file. Picking the newest, or the zero, would be the overclaim the rest
    of this file refuses, so the row stays unobserved and says how many wrote
    it.

    Before this test::

        >       assert len(record["pool_actions_matching"]) == 2, record
        E       KeyError: 'pool_actions_matching'
    """

    merge_suite = _module()
    receipt_dir = tmp_path / "receipt"
    receipt_dir.mkdir()
    surface = receipt_dir / "surface.gpu.json"
    _gpu_population(surface)
    merge_suite.POOL_QUEUE, merge_suite.POOL_CAS_REQUESTS = _fake_pool(
        tmp_path, surface, [("beef" + "0" * 60, "done", 0, "sparky"),
                            ("cafe" + "0" * 60, "failed", 1, "sparky")])

    record = merge_suite._resume("gpu", merge_suite.ARMS["gpu"], receipt_dir)
    assert record["returncode"] is None, record
    assert record["exit_status_observed"] is False, record
    assert len(record["pool_actions_matching"]) == 2, record
    assert "no single exit status" in record["exit_status_note"], record
    assert not merge_suite._verdict([record]).startswith("green on")


def test_a_resume_keeps_the_receipt_the_original_run_wrote(tmp_path):
    """The file that holds the two arms together was the one left unprotected.

    ``--resume`` reassembles into the directory the original run wrote, and its
    default output name is that run's own ``receipt.json``. So resuming one arm
    wrote over the receipt recording the other -- the same loss the surface
    files have been protected from since ``ab4867e``, in the file whose whole
    purpose is to hold both populations side by side. On this branch that would
    have landed on ``20260904T025044``, whose ``receipt.json`` is the only
    record of that run's x86 submission and whose GPU arm sat queued behind a
    held reservation for nine hours before anyone could resume it.

    Before this test::

        >       assert kept, sorted(p.name for p in receipt_dir.iterdir())
        E       AssertionError: ['receipt.json', 'surface.gpu.json']
        E       assert []
    """

    merge_suite = _module()
    receipt_dir = tmp_path / "receipt"
    receipt_dir.mkdir()
    (receipt_dir / "receipt.json").write_text(
        json.dumps({"schema": "tessera.merge_suite.v1", "arms": [{"arm": "x86"}]}))
    os.utime(receipt_dir / "receipt.json", (1788500000, 1788500000))
    _gpu_population(receipt_dir / "surface.gpu.json")

    rc = subprocess.run(
        [sys.executable, str(TOOL), "--arm", "gpu",
         "--resume", str(receipt_dir), "--checkout", str(ROOT)],
        capture_output=True, text=True, timeout=300)
    assert rc.returncode in (0, 1), rc.stdout + rc.stderr

    kept = sorted(receipt_dir.glob("receipt.superseded-*.json"))
    assert kept, sorted(p.name for p in receipt_dir.iterdir())
    # The first run's record survives intact...
    assert json.loads(kept[0].read_text())["arms"][0]["arm"] == "x86"
    # ...and the plain name is the newest, so nothing that reads it changes.
    assert json.loads((receipt_dir / "receipt.json").read_text())["arms"][0]["arm"] == "gpu"
    assert "kept at" in rc.stdout, rc.stdout


def test_no_test_here_can_write_into_the_receipt_store_the_real_runs_use():
    """Every invocation in this file names its own ``--out``.

    ``--out`` defaults to a timestamped directory under
    ``DEFAULT_RECEIPT_ROOT`` -- the shared store the real receipts live in.  A
    test that omits it publishes into that store from every box that runs the
    suite, which is how sixteen one-arm ``not run`` directories came to sit
    beside the four real ones.  Read statically rather than by watching the
    store: watching it would be a race against every other run on the fleet,
    and the property being pinned is about this file, not about a placement.

    Before this test, run over ``1cdeee0~1:tests/test_merge_suite.py``::

        invocations checked: 5
        would fail at lines: [146]

    which is ``test_the_x86_arm_refuses_a_checkout_only_one_box_can_see``,
    the one this commit also fixes.
    """

    import ast

    source = Path(__file__).read_text()
    tree = ast.parse(source)
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.List):
            continue
        flags = [e.value for e in node.elts
                 if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        names_tool = any(
            isinstance(e, ast.Call)
            and isinstance(e.func, ast.Name) and e.func.id == "str"
            and e.args and isinstance(e.args[0], ast.Name) and e.args[0].id == "TOOL"
            for e in node.elts)
        if not names_tool:
            continue
        checked += 1
        # ``--resume`` writes into the directory it was handed, never into the
        # default store, so it is the other honest way to name a destination.
        assert "--out" in flags or "--resume" in flags, (
            f"{Path(__file__).name}:{node.lineno} runs merge_suite.py with no --out"
        )
    # A guard that matched nothing would pass forever.
    assert checked >= 4, checked
