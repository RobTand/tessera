"""The merge suite composes submissions; these pin what it composes.

``tools/merge_suite.py`` submits native Flux jobs and binds their populations.  So the things worth pinning
are the properties of those command lines and of the verdict it derives, not a
placement, which belongs to the pool.
"""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tessera._dev import surface_publication

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

    And an arm that cannot fan out spends one core whatever ``--cpus`` says,
    which is why this asks the arm rather than assuming the number.  Before
    that existed::

        >       assert merge_suite._arm_cpus(merge_suite.ARMS[name], 6) == 6
        E       AttributeError: module '_merge_suite' has no attribute '_arm_cpus'
    """

    merge_suite = _module()
    surface = Path("/dev/null")
    for name in ("gpu", "x86"):
        serial = merge_suite._command(merge_suite.ARMS[name], surface, [], 1)
        assert "-n" not in serial, "one declared core must not fan out"

        parallel = merge_suite._command(merge_suite.ARMS[name], surface, [], 6)
        # The gate survives the fan-out: it is asserted on the controller,
        # which is the process that has -- or has not -- the device.
        assert ("--strict-cuda" in parallel) == merge_suite.ARMS[name]["strict_cuda"]
        if not merge_suite.ARMS[name].get("fans_out", True):
            # An arm that cannot fan out spends one core whatever is asked.
            assert "-n" not in parallel, parallel
            assert merge_suite._arm_cpus(merge_suite.ARMS[name], 6) == 1
            continue
        assert parallel[parallel.index("-n") + 1] == "6"
        # A module's tests share fixtures, and on the GPU arm device state.
        assert parallel[parallel.index("--dist") + 1] == "worksteal"
        assert merge_suite._arm_cpus(merge_suite.ARMS[name], 6) == 6


@pytest.mark.parametrize("targets", [None, ["tests/test_a.py", "tests/test_b.py"]])
def test_the_targets_an_arm_runs_are_the_targets_read_back_from_its_sealed_command(targets):
    """An arm may run named test files, and the pool's record says which.

    A tool that submits arms of selected files has to check afterwards that the
    sealed commands cover the selection, so composing the targets and reading
    them back are tested as one pair.  The default stays the whole tree.
    """

    merge_suite = _module()
    arm = {**merge_suite.ARMS["x86"], **({"targets": targets} if targets else {})}
    command = merge_suite._command(arm, Path("/tmp/surface.json"), ["--durations=5"], 2)
    sealed = merge_suite._timed_command(command, 600.0)
    argv, why = merge_suite._pytest_argv(sealed)
    assert why is None
    assert merge_suite._targets_of(argv) == (targets or ["tests"])


def test_one_receipt_owner_writes_the_verdict_for_any_set_of_arms():
    """``main`` and tools that submit arms of their own share one receipt shape."""

    merge_suite = _module()
    receipt = merge_suite._assemble_receipt(
        [_population("x86")], ROOT, assembled_by="test")
    assert receipt["schema"] == "tessera.merge_suite.flux.v2"
    assert receipt["assembled_by"] == "test"
    assert receipt["verdict"] == merge_suite._verdict(receipt["arms"], receipt["commits_measured"])
    assert receipt["arm_results"] == merge_suite._arm_results(receipt["arms"])












@pytest.mark.parametrize("duration", ["0", "-1", "nan", "inf", "-inf"])
def test_attempt_timeout_refuses_disabled_or_nonfinite_deadlines(tmp_path, duration):
    out = tmp_path / "receipt.json"
    result = subprocess.run(
        [sys.executable, str(TOOL), "--arm", "gpu", "--dry-run",
         "--checkout", str(ROOT), "--out", str(out), f"--timeout-s={duration}"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 2
    assert "positive finite" in result.stderr
    assert not out.exists(), "invalid deadlines must refuse before receipt/submission writes"


@pytest.mark.parametrize("status", [0, 7])
def test_attempt_timeout_preserves_ordinary_command_status(status):
    merge_suite = _module()
    command = merge_suite._timed_command([sys.executable, "-c", f"raise SystemExit({status})"], 2.0)
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == status


def test_attempt_timeout_owns_child_reaping_despite_inherited_sigchld_ignore():
    merge_suite = _module()
    command = merge_suite._timed_command([sys.executable, "-c", "raise SystemExit(7)"], 2.0)
    script = ("import os,signal; signal.signal(signal.SIGCHLD,signal.SIG_IGN); "
              f"os.execv({command[0]!r}, {command!r})")
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 7, "auto-reaping must not convert a failed command to status0"


def test_attempt_timeout_term_handler_cannot_turn_deadline_into_success():
    merge_suite = _module()
    script = ("import signal,time,sys; "
              "signal.signal(signal.SIGTERM, lambda *args: sys.exit(0)); "
              "print('ready', flush=True); time.sleep(60)")
    result = subprocess.run(merge_suite._timed_command([sys.executable, "-c", script], 2.0),
                            capture_output=True, text=True, timeout=10)
    assert "ready" in result.stdout
    assert result.returncode == 124, "timeout must remain non-green even if TERM cleanup exits0"


@pytest.mark.parametrize("leader_exits", [False, True])
def test_attempt_timeout_kills_a_term_resistant_process_group(monkeypatch, leader_exits):
    import signal

    merge_suite = _module()
    monkeypatch.setattr(merge_suite, "TIMEOUT_KILL_AFTER_S", 0.2)
    leader_handler = ("signal.signal(signal.SIGTERM, lambda *args: sys.exit(0)); "
                      if leader_exits else "")
    script = ("import json,os,signal,subprocess,sys,time; "
              "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
              "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], "
              "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
              + leader_handler +
              "print(json.dumps([os.getpid(),child.pid]), flush=True); time.sleep(60)")
    command = merge_suite._timed_command([sys.executable, "-c", script], 2.0)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    pids = []

    def running(pid):
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            # The entry can vanish mid-read (ESRCH) while the task is reaped.
            return False
        return state != "Z"  # a reparented zombie is dead, not retained work

    try:
        stdout, stderr = process.communicate(timeout=10)
        pids = json.loads(stdout)
        expected = {124} if leader_exits else {-signal.SIGKILL, 137}
        assert process.returncode in expected, stderr
        deadline = time.monotonic() + 2.0
        while any(running(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not any(running(pid) for pid in pids), "the managed parent and child must stop"
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        for pid in pids:
            if running(pid):
                os.kill(pid, signal.SIGKILL)


def test_attempt_timeout_supervisor_interrupt_cleans_its_owned_group():
    import selectors
    import signal

    merge_suite = _module()
    script = ("import json,os,signal,subprocess,sys,time; "
              "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
              "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], "
              "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
              "print(json.dumps([os.getpid(),child.pid]), flush=True); time.sleep(60)")
    process = subprocess.Popen(merge_suite._timed_command([sys.executable, "-c", script], 60.0),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    pids = []

    def running(pid):
        try:
            return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0] != "Z"
        except (FileNotFoundError, ProcessLookupError):
            # The entry can vanish mid-read (ESRCH) while the task is reaped.
            return False

    try:
        with selectors.DefaultSelector() as ready:
            ready.register(process.stdout, selectors.EVENT_READ)
            assert ready.select(timeout=5), "owned child must announce readiness"
        pids = json.loads(process.stdout.readline())
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)
        assert process.returncode in (-signal.SIGTERM, 143)
        deadline = time.monotonic() + 2.0
        while any(running(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not any(running(pid) for pid in pids), "interrupt must not abandon the owned group"
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        for pid in pids:
            if running(pid):
                os.kill(pid, signal.SIGKILL)




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
    one = merge_suite._verdict([_population("gpu")])
    two = merge_suite._verdict([_population("gpu"), _population("x86")])
    assert one == "green on 1 population(s): gpu"
    assert two == "green on 2 population(s): gpu, x86"
    assert "x86" not in one






def _master_history(checkout, master_ref):
    """Two real master tips, with measured HEAD fixed at the first one.

    May extend an existing fixture checkout. No remote, live pool, or runner's
    default git branch/configuration is needed.
    """

    checkout.mkdir(exist_ok=True)

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(checkout), "-c", "user.name=Test",
             "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
             *args], text=True).strip()

    if not (checkout / ".git").exists():
        git("init", "-q", "--initial-branch=measured")
        git("commit", "-q", "--allow-empty", "-m", "measured source")
    else:
        git("checkout", "-q", "-B", "measured")
    git("update-ref", "-d", "refs/heads/master")
    git("update-ref", "-d", "refs/remotes/origin/master")
    submitted = git("rev-parse", "HEAD")
    advanced = git("commit-tree", "HEAD^{tree}", "-p", submitted, "-m", "master advances")
    ref = "refs/heads/master" if master_ref == "master" else "refs/remotes/origin/master"
    git("update-ref", ref, submitted)
    return git, ref, submitted, advanced






def test_pr_ci_fetches_the_base_ref_that_population_identity_reads():
    """The PR checkout must contain master; absence is not a branch verdict."""
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    pure_job = workflow.split("  pure:\n", 1)[1].split("\n  publish:\n", 1)[0]
    checkout = pure_job.split("- uses: actions/checkout@", 1)[1].split("\n      - ", 1)[0]
    assert "fetch-depth: 0" in checkout, (
        "the pure PR job uses merge_suite's population test but its shallow checkout "
        "contains neither master nor origin/master")


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


def test_snapshot_commit_agreement_is_separate_from_effective_source():
    merge_suite = _module()
    arms = [{"arm": arm, "surface": {"commit": commit, "source_identity": {
        "schema": "tessera.suite_source.v1", "verification": "verified",
        "snapshot_commit": commit, "sha256": "c" * 64}}}
        for arm, commit in (("gpu", "a" * 40), ("x86", "b" * 40))]
    comparison = merge_suite._commits_measured(arms)
    assert comparison["agree"] is False
    assert comparison["effective_source"]["agree"] is True
    arms[1]["surface"]["source_identity"]["sha256"] = "d" * 64
    assert merge_suite._commits_measured(arms)["effective_source"]["agree"] is False
    arms[1]["surface"]["source_identity"]["verification"] = "unknown"
    assert merge_suite._commits_measured(arms)["effective_source"]["agree"] is None
    del arms[1]["surface"]["source_identity"]
    assert merge_suite._commits_measured(arms)["effective_source"]["agree"] is None


def test_source_identity_cannot_be_borrowed_from_another_snapshot():
    merge_suite = _module()
    record = {"arm": "gpu", "surface": {"commit": "a" * 40, "source_identity": {
        "schema": "tessera.suite_source.v1", "verification": "verified",
        "snapshot_commit": "b" * 40, "sha256": "c" * 64}}}
    assert merge_suite._commits_measured([record])["effective_source"]["agree"] is None






def test_a_resumed_row_reads_the_run_mode_out_of_the_pools_command(tmp_path):
    """The mode is in the command the pool ran, which outlives the submitter.

    A resumed receipt is assembled after the process that chose `-n` is gone,
    so it cannot know the mode from itself. It can read it from the same table
    it already reads the exit status out of: the action's own command in the
    CAS request, found by the `--surface-json` path.

    Before this test::

        >       assert record["cpus_used"] == 8
        E       KeyError: 'cpus_used'
    """

    merge_suite = _module()
    assert merge_suite._cpus_of_command(
        ["python", "-m", "pytest", "tests", "-n", "8", "--dist", "loadfile"]) == 8
    assert merge_suite._cpus_of_command(
        ["python", "-m", "pytest", "tests", "--strict-cuda"]) == 1
    # Not recorded is not serial.
    assert merge_suite._cpus_of_command([]) is None
    assert merge_suite._cpus_of_command(["python", "-m", "pytest", "-n"]) is None


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










def _attempt_stdout(surface_json, counts=None, published=True, digest=None):
    """What the attempt's pytest printed, as PrismaBuild's worker captured it.

    ``detail.stdout`` on a real outcome record is the attempt's whole stdout
    (the worker ``communicate()``s, nothing is truncated), and two lines of it
    are the attempt's own account of what it published: the conftest's
    publication line and pytest's summary line with the counts the population
    also carries. A test that wants an attempt which died before publishing
    says ``published=False`` and gets neither line, which is what such an
    attempt leaves behind.

    The publication line is **not spelled here**. It is built by the module
    the conftest writes it from, so this fixture can no longer agree with the
    reader by restating the reader's own string -- which is the shape #331
    was: four tests of the join, each feeding the parser a sentence the test
    wrote to match it, all green with the binding dead. What still cannot be
    built here is a real run's stdout, so one test does not build it:
    ``test_a_real_runs_stdout_binds_to_the_population_it_wrote`` runs pytest.

    ``digest=None`` is the pre-#331 line, which is what the captured stdout of
    every population under ``/mnt/shared/tessera-suite-receipts`` holds; the
    negative cases below are all of that vintage and stay there deliberately,
    since a reader that stopped binding them would lose the three historical
    populations that bind today.
    """

    counts = {"passed": 1827, "skipped": 10, **(counts or {})}
    summary = ", ".join(f"{n} {word}" for word, n in counts.items() if n)
    lines = ["....s...."]
    if published:
        lines += [surface_publication.publication_line(
                      surface_publication.POPULATION, surface_json, digest),
                  f"{summary}, 14 warnings in 511.40s (0:08:31)"]
    else:
        lines.append("Fatal Python error: Aborted")
    return "\n".join(lines) + "\n"


def _with(base, overrides):
    """``base`` updated by ``overrides``; a ``None`` value DROPS the field.

    The fleet's records lose fields rather than null them -- a requeue strips
    ``claimed_unix`` and an old request has no ``checkout_snapshot`` -- so a
    test that wants that shape must be able to state absence, not just a
    different value.
    """

    merged = dict(base)
    for field, value in (overrides or {}).items():
        if value is None:
            merged.pop(field, None)
        else:
            merged[field] = value
    return merged




















def test_a_probe_that_did_not_answer_is_not_a_clean_tree(tmp_path):
    """``working_tree_dirty`` has three states, and ``false`` is not the default.

    ``_git`` returned ``""`` for "git said nothing" and for "git failed", and
    ``bool("")`` made both of them ``working_tree_dirty: false`` -- a receipt
    asserting a clean tree it had not established.  Worse, the 30 s timeout
    propagated: a ``--resume`` of a real x86 population died on
    ``TimeoutExpired`` from ``git status --porcelain`` against a /mnt/shared
    checkout that had a suite running in it, and wrote neither receipt nor
    ledger row.  A provenance field nothing gates on must not be able to
    destroy the measurement it annotates.

    Both legs are exercised: a directory that is not a repository (git exits
    non-zero) and a timeout short enough to be certain (the except branch).

    Before this test::

        >       assert population["working_tree_dirty"] is None, population
        E       AssertionError: {'checkout': '.../plain', 'commit': '',
        E       'describe': '', 'is_master_head': None, ...}
        E       assert False is None
    """

    merge_suite = _module()

    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    population = merge_suite._population_of(not_a_repo)
    assert population["working_tree_dirty"] is None, population
    assert population["commit"] is None, population

    original = merge_suite.GIT_PROBE_TIMEOUT_S
    try:
        merge_suite.GIT_PROBE_TIMEOUT_S = 0.000001
        # ROOT is a real repository, so this can only end in the timeout.
        timed_out = merge_suite._population_of(ROOT)
    finally:
        merge_suite.GIT_PROBE_TIMEOUT_S = original
    assert timed_out["working_tree_dirty"] is None, timed_out
    assert timed_out["commit"] is None, timed_out


def test_an_arm_that_ran_nothing_is_not_a_green_population():
    """The other door onto the green lie tessera#114 is about.

    #114's x86 arm published no population, and the merger refused.  The
    cheapest way to make that refusal stop firing is to give the arm a
    population it can always produce: collect the suite, skip all of it, exit
    0.  Nothing above this check would object -- no failures, a clean exit, a
    surface on disk with its skip reasons -- and the receipt would read green
    on a population that executed not one test.

    So a population is only evidence for green when something in it ran.  The
    check is on ``passed`` rather than on the skip *reasons*, because a reason
    is prose and this is a gate: an arm that legitimately cannot run part of
    the surface still has to run the rest of it.
    """

    merge_suite = _module()
    gpu = _population("gpu", counts={"passed": 2059, "failed": 0, "error": 0,
                                     "skipped": 13})
    all_skipped = merge_suite._verdict([gpu, _population(
        "x86",
        counts={"passed": 0, "failed": 0, "error": 0, "skipped": 2072},
        skip_reasons={"no CUDA device on this box": 2072})])
    assert all_skipped.startswith("incomplete: the x86 arm"), all_skipped
    assert "0 passed" in all_skipped and "2072 skipped" in all_skipped

    # And the same two arms, with the x86 one having actually executed the
    # device-less surface, is the verdict this issue is trying to reach.
    ran = merge_suite._verdict([gpu, _population(
        "x86",
        counts={"passed": 1600, "failed": 0, "error": 0, "skipped": 472},
        skip_reasons={"no CUDA device on this box": 467})])
    assert ran == "green on 2 population(s): gpu, x86", ran


def _population(arm="gpu", *, source="c" * 64, commit="a" * 40, **overrides):
    """An arm record holding a population with every field green requires.

    A hand-made surface with three fields in it is not a smaller version of a
    published population; it is a different object, and a verdict tested only
    against it is a verdict tested against nothing the suite writes.  This is
    what ``tests/conftest.py`` actually publishes, so a test that wants a gap
    states the gap as an override.
    """

    cuda = arm == "gpu"
    surface = {
        "schema": "tessera.test_surface.v3",
        "role": "population",
        "worker_id": None,
        "commit": commit,
        "source_identity": {"schema": "tessera.suite_source.v1",
                            "verification": "verified",
                            "snapshot_commit": commit, "sha256": source},
        "cuda": cuda,
        "strict_cuda": cuda,
        "device": ("torch 2.11.0+cu130, 1 CUDA device(s), device 0 = NVIDIA GB10"
                   if cuda else "torch 2.11.0+cpu reports no CUDA device"),
        "counts": {"passed": 1827, "failed": 0, "error": 0, "skipped": 10,
                   "xfailed": 0, "xpassed": 0},
        "skip_reasons": {},
        "cuda_surface": {"executed": 311 if cuda else 0,
                         "is_a_floor": True,
                         "box_artifact_skips": {}},
        "not_collected": [],
    }
    surface.update(overrides)
    return {"arm": arm, "requires_cuda": cuda, "returncode": 0,
            "surface": surface}


def test_a_complete_matching_source_pair_is_the_only_shape_of_green():
    """#217: the receipt's own fields said the arms measured two trees."""

    merge_suite = _module()
    arms = [_population("gpu"), _population("x86")]
    assert merge_suite._verdict(arms) == "green on 2 population(s): gpu, x86"


def test_arms_that_measured_different_source_are_not_a_merge_success():
    """A merge check compares two runs of ONE tree, or it compares nothing.

    ``commits_measured.effective_source.agree`` already answered this on the
    receipt while ``_verdict`` never read it, so a receipt whose own fields
    said the arms ran different source still exited 0.
    """

    merge_suite = _module()
    arms = [_population("gpu", source="c" * 64),
            _population("x86", source="d" * 64)]

    assert merge_suite._commits_measured(arms)["effective_source"]["agree"] is False
    verdict = merge_suite._verdict(arms)
    assert not verdict.startswith("green on"), verdict
    assert "source" in verdict, verdict


@pytest.mark.parametrize("gap", [
    pytest.param({"source_identity": None}, id="no-source-identity"),
    pytest.param({"source_identity": {"schema": "tessera.suite_source.v1",
                                      "verification": "unknown",
                                      "snapshot_commit": "a" * 40,
                                      "sha256": None}}, id="unverified"),
    pytest.param({"source_identity": {"schema": "tessera.suite_source.v1",
                                      "verification": "verified",
                                      "snapshot_commit": "b" * 40,
                                      "sha256": "c" * 64}},
                 id="identity-from-another-snapshot"),
])
def test_an_unestablished_source_identity_is_not_a_merge_success(gap):
    """Unknown provenance is not equivalence; it is the absence of it."""

    merge_suite = _module()
    arms = [_population("gpu", **gap), _population("x86")]
    verdict = merge_suite._verdict(arms)
    assert not verdict.startswith("green on"), verdict
    assert "source" in verdict, verdict


@pytest.mark.parametrize(("gap", "expected"), [
    pytest.param({"schema": None}, "schema", id="no-schema"),
    pytest.param({"schema": "tessera.test_surface.v99"}, "schema",
                 id="unrecognised-schema"),
    pytest.param({"role": None}, "role", id="no-role"),
    pytest.param({"role": "worker-share"}, "role", id="worker-share"),
    pytest.param({"counts": None}, "counts", id="no-counts"),
    pytest.param({"counts": {"passed": "many", "failed": 0}}, "counts",
                 id="counts-are-not-numbers"),
])
def test_a_population_that_cannot_be_read_is_not_green(gap, expected):
    """#217: the executed-test check was skipped entirely when counts were absent.

    ``_verdict`` walked past a surface with no ``counts`` -- ``continue`` --
    and then had nothing left to object to, so a record with no population
    evidence in it at all reached ``green``.
    """

    merge_suite = _module()
    verdict = merge_suite._verdict([_population("x86", **gap)])
    assert not verdict.startswith("green on"), verdict
    assert expected in verdict, verdict


@pytest.mark.parametrize(("gap", "expected"), [
    pytest.param({"cuda_surface": {"executed": 0, "box_artifact_skips": {}}},
                 "allocated", id="nothing-executed-on-the-device"),
    pytest.param({"cuda_surface": {"executed": 311, "box_artifact_skips": {
        "TESSERA_SHIPPED_CHECKPOINT is unset": 4}}},
        "does not hold", id="box-artifact-skips"),
    pytest.param({"schema": "tessera.test_surface.v2", "cuda_surface": None},
                 "cannot say", id="pre-v3-cannot-answer"),
])
def test_a_gpu_arm_that_did_not_cover_the_surface_is_not_green(gap, expected):
    """The third leg of tessera#152, read by the tool that quotes the receipt.

    ``tests/conftest.py`` publishes ``cuda_surface.executed`` and the
    box-artifact skip reasons, and ``--strict-cuda`` refuses on them in the
    run's own process.  ``_verdict`` read neither, so a receipt could carry a
    population that says nothing ran on the device and still exit 0.
    """

    merge_suite = _module()
    verdict = merge_suite._verdict([_population("gpu", **gap)])
    assert not verdict.startswith("green on"), verdict
    assert expected in verdict, verdict
    # The x86 arm is not submitted to cover that surface and is not held to it.
    assert merge_suite._verdict([_population("x86")]).startswith("green on")


def test_each_arms_own_result_is_readable_without_being_the_merge_verdict():
    """Per-population success, kept where automation cannot read it as merge success."""

    merge_suite = _module()
    arms = [_population("gpu"), _population("x86", source="d" * 64)]

    statuses = merge_suite._arm_results(arms)
    assert statuses == {"gpu": "green", "x86": "green"}, statuses
    assert not merge_suite._verdict(arms).startswith("green on")

    red = _population("x86", counts={"passed": 1, "failed": 2, "error": 0,
                                     "skipped": 0})
    assert merge_suite._arm_results([red])["x86"].startswith("red"), \
        merge_suite._arm_results([red])






























#: A small, self-contained file for the child run below. It is named for the
#: property that makes it usable -- it collects and runs in a few seconds
#: without a device or an out-of-tree artifact -- and nothing here reads its
#: pass count, which is free to drift.
CHILD_SUITE = "tests/test_collection_probe.py"


@pytest.fixture(scope="module")
def real_publication(tmp_path_factory):
    """One real ``pytest --surface-json`` run: its stdout and its population.

    Everything else in this file builds the attempt's stdout, which is how
    #331 happened: ``_attempt_stdout`` wrote the parser the string the parser
    was looking for, so all four tests of the join were green with the join
    dead. This fixture is the one place a real run's terminal output reaches
    the reader, and it is deliberately a subprocess -- an interpreter, an
    argv, a conftest and an exit code -- rather than an in-process call of
    ``_write_surface_json``, because the thing under test is what pytest
    actually prints.

    Serial on purpose (``PYTEST_ADDOPTS`` cleared): under ``-n`` the workers
    write shares and only the controller writes the population, and this
    fixture wants the population's own line.
    """

    root = tmp_path_factory.mktemp("real-publication")
    surface = root / "surface.gpu.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "tests"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["PYTEST_ADDOPTS"] = ""
    env["TMPDIR"] = str(root)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q",
         CHILD_SUITE, "--surface-json", str(surface)],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600,
    )
    assert surface.is_file(), (result.returncode, result.stdout[-4000:],
                               result.stderr[-2000:])
    population = json.loads(surface.read_text())
    assert population.get("counts"), (population, result.stdout[-4000:])
    return {"surface": surface, "stdout": result.stdout,
            "returncode": result.returncode, "population": population}


def _reannounce(stdout, path, digest):
    """The run's own stdout with its publication line re-announced.

    Rebuilt through the same module ``tests/conftest.py`` writes it from, so
    the FORMAT stays the producer's and only the digest value moves. The
    assertion is half the point: the reader's own parser must find exactly one
    publication line in a real run's terminal output.
    """

    line = surface_publication.publication_line(
        surface_publication.POPULATION, path, digest)
    out, replaced = [], 0
    for raw in stdout.splitlines():
        if surface_publication.published_digests(raw, path):
            out.append(line)
            replaced += 1
        else:
            out.append(raw)
    assert replaced == 1, (replaced, stdout)
    return "\n".join(out) + "\n"


















@pytest.mark.parametrize("line, expected", [
    ("516 passed, 70 skipped, 28 subtests passed in 115.84s (0:01:55)",
     {"passed": 516, "skipped": 70}),
    ("1 failed, 515 passed, 3 subtests passed in 9.00s", {"failed": 1, "passed": 515}),
    ("516 passed, 70 skipped in 115.84s (0:01:55)", {"passed": 516, "skipped": 70}),
])
def test_a_summary_line_that_counts_subtests_is_still_a_terminal_summary(line, expected):
    """pytest prints ``N subtests passed`` beside the test buckets, and the line is whole.

    The term has two words, and a parser that read only ``N word`` refused the
    entire line, so an attempt whose tests used subtests printed "no terminal
    summary" and its exit status could never be bound.  Subtests are not a
    population bucket, so they are read and not compared, as warnings are.
    """

    merge_suite = _module()
    counts, read = merge_suite._summary_counts("....\n" + line + "\n")
    assert read == line
    assert {key: value for key, value in counts.items() if value} == expected




def _native_jobs(module, monkeypatch):
    fixture = importlib.util.spec_from_file_location("native_suite_fixtures", ROOT / "tests/suite_flux_fixtures.py")
    loaded = importlib.util.module_from_spec(fixture)
    fixture.loader.exec_module(loaded)
    return loaded.NativeJobs(module, monkeypatch)


@pytest.mark.parametrize("mutation,phrase", [
    ("reader", "command"), ("override", "command"), ("source", "launch digest"),
    ("no-announcement", "announce"), ("counts", "counts"), ("replacement", "digest"),
    ("missing-payload", "incomplete"), ("query", "incomplete"), ("jobspec", "incomplete")])
def test_native_resume_binds_saved_id_command_source_and_publication(tmp_path, monkeypatch, mutation, phrase):
    merge = _module()
    jobs = _native_jobs(merge, monkeypatch)
    world = jobs.add(tmp_path, "x86", _population("x86")["surface"])
    assert merge._verdict([merge._resume("x86", merge.ARMS["x86"], tmp_path)]).startswith("green on")
    launch = json.loads((world.directory / "launch.json").read_bytes())
    if mutation == "reader":
        launch["command"] = ["cat", str(world.surface)]
    elif mutation == "override":
        launch["command"] += ["--surface-json", str(tmp_path / "elsewhere.json")]
    elif mutation == "source":
        launch["source"]["sha256"] = "f" * 64
    elif mutation == "no-announcement":
        (world.directory / "stdout").write_text("1827 passed, 10 skipped in 0.10s\n")
    elif mutation == "counts":
        path = world.directory / "stdout"
        path.write_text(path.read_text().replace("1827 passed", "1826 passed"))
    elif mutation == "replacement":
        body = json.loads(world.surface.read_bytes()); body["device"] = "different device"
        world.surface.write_text(json.dumps(body))
    elif mutation == "missing-payload":
        (world.directory / "payload.json").unlink()
    elif mutation == "query":
        world.query_error = "status"
    elif mutation == "jobspec":
        world.spec["tasks"][0]["command"] = ["echo", "unrelated"]
    if mutation in ("reader", "override", "source"):
        merge.suite_flux.save(world.directory / "launch.json", launch)
        sent = json.loads((world.directory / "submission.json").read_bytes())
        sent["launch_sha256"] = hashlib.sha256(merge.suite_flux.encoded(launch)).hexdigest()
        merge.suite_flux.save(world.directory / "submission.json", sent)
        body = json.loads((world.directory / "payload.json").read_bytes())
        body.update(source=launch["source"], launch_sha256=sent["launch_sha256"])
        merge.suite_flux.save(world.directory / "payload.json", body)
    record = merge._resume("x86", merge.ARMS["x86"], tmp_path)
    assert not record["exit_status_observed"], record
    assert phrase in record["binding_refusal"]
    assert not merge._verdict([record]).startswith("green")
    assert not any("submit" in call for call in jobs.calls)


@pytest.mark.parametrize("mutation", ["replacement", "unlink"])
def test_native_resume_authenticates_one_read_snapshot(tmp_path, monkeypatch, mutation):
    merge = _module(); jobs = _native_jobs(merge, monkeypatch)
    world = jobs.add(tmp_path, "x86", _population("x86")["surface"], code=7)
    held = world.surface.read_bytes()
    original = merge.suite_flux.collect
    def collect(directory):
        if mutation == "unlink":
            world.surface.unlink()
        else:
            world.surface.write_text('{"counts":{"passed":100}}')
        return original(directory)
    monkeypatch.setattr(merge.suite_flux, "collect", collect)
    record = merge._resume("x86", merge.ARMS["x86"], tmp_path)
    assert record["surface"] == json.loads(held)
    assert record["surface_sha256"] == hashlib.sha256(held).hexdigest()
    assert record["exit_status_observed"] and record["returncode"] == 7
    assert merge._verdict([record]).startswith("red")


def test_real_producer_publication_binds_without_rewriting_its_output(real_publication):
    merge = _module()
    published = merge._read_publication(real_publication["surface"])
    native = {"command": [sys.executable, "-m", "pytest", CHILD_SUITE, "--surface-json", str(published.path)],
              "source": real_publication["population"]["source_identity"],
              "exit_status_observed": True, "returncode": real_publication["returncode"]}
    assert surface_publication.published_digests(real_publication["stdout"], published.path) == [published.digest]
    assert merge._native_binding_refusal(native, published, real_publication["stdout"]) is None
    published.path.write_text(json.dumps({**published.payload, "device": "changed"}))
    assert merge._native_binding_refusal(native, merge._read_publication(published.path), real_publication["stdout"])


@pytest.mark.parametrize("cores", [1, 4])
def test_native_dry_run_declares_exact_workers_deadline_and_creates_nothing(tmp_path, cores):
    out = tmp_path / "fresh" / "receipt.json"
    result = subprocess.run([sys.executable, str(TOOL), "--arm", "x86", "--dry-run",
        "--cpus", str(cores), "--target", CHILD_SUITE, "--checkout", str(ROOT), "--out", str(out)],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    arm = receipt["arms"][0]
    inner, why = _module()._pytest_argv(arm["command"])
    assert why is None and _module()._targets_of(inner) == [CHILD_SUITE]
    assert arm["cpus_used"] == cores
    assert (inner[inner.index("-n") + 1] if "-n" in inner else "1") == str(cores)
    assert "tools/suite_deadline.py" in arm["command"]
    assert not out.parent.exists()


@pytest.mark.parametrize("arm,extra", [("gpu", []), ("x86", ["--gpu-image", "sha256:" + "a" * 64,
    "--gpu-deps-site", "/deps", "--gpu-deps-sha256", "b" * 64, "--gpu-cache-dir", "/tmp"])])
def test_unqualified_native_gpu_container_refuses_without_writing(tmp_path, arm, extra):
    out = tmp_path / "fresh" / "receipt.json"
    result = subprocess.run([sys.executable, str(TOOL), "--arm", arm, "--dry-run", "--checkout", str(ROOT),
                             "--out", str(out), *extra], capture_output=True, text=True, timeout=60)
    assert result.returncode == 2
    assert "GPU/container" in result.stderr
    assert not out.parent.exists()


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


def test_the_ledger_names_the_run_mode_of_each_row(tmp_path):
    """Two rows of one commit can differ by the mode and not by the device.

    The five failures on `82f0047` were an `-n`-only defect: the same commit,
    the same box and the same two files were green run serially. A ledger that
    sets a GPU population beside an x86 one and omits how each ran invites the
    reader to attribute that difference to the device -- the misreading #112 is
    about, one column over. The header used to explain the absence in prose,
    which is a footnote a tired reader has to remember to apply.

    `--` stays available and means "nobody recorded it": a resumed row can name
    a mode only when exactly one finished pool action wrote its population, and
    every row written before this column existed has none.

    Before this test::

        >       assert row.count("|") == columns, row
        E       AssertionError: | ... | gpu | serial | ... |
        (and, on the header, `mode` absent from LEDGER_HEADER)
    """

    merge_suite = _module()
    ledger = tmp_path / "l.md"
    merge_suite._record_markdown(ledger, {
        "generated_utc": "2026-09-04T10:00:00Z",
        "population": {"commit": "a" * 40, "is_master_head": True},
        "arms": [
            {"arm": "gpu", "cpus_used": 1, "returncode": 0,
             "measured_utc": "2026-09-04T10:00:00Z",
             "surface": {"commit": "a" * 40, "device": "a GB10",
                         "counts": {"passed": 3}}},
            {"arm": "x86", "cpus_used": 8, "returncode": 0,
             "measured_utc": "2026-09-04T10:00:00Z",
             "surface": {"commit": "a" * 40, "device": "no CUDA device",
                         "counts": {"passed": 3}}},
        ],
    })
    text = ledger.read_text()
    assert "| mode |" in text, text
    gpu = [l for l in text.splitlines() if "| gpu |" in l][0]
    x86 = [l for l in text.splitlines() if "| x86 |" in l][0]
    assert "| serial |" in gpu, gpu
    assert "| -n 8 |" in x86, x86

    # An arm with no recorded mode says so rather than defaulting to serial.
    silent = tmp_path / "s.md"
    merge_suite._record_markdown(silent, {
        "generated_utc": "2026-09-04T10:00:00Z",
        "population": {"commit": "a" * 40, "is_master_head": True},
        "arms": [{"arm": "gpu", "surface": None,
                  "exit_status_observed": False, "returncode": None}],
    })
    row = [l for l in silent.read_text().splitlines() if "| gpu |" in l][0]
    assert row.split("|")[5].strip() == "--", row

    # Every row this tool writes is a row of the header it writes.
    columns = merge_suite.LEDGER_HEADER.strip().splitlines()[-2].count("|")
    for line in text.splitlines():
        if line.startswith("| 2026-") or line.startswith("| -- |"):
            assert line.count("|") == columns, line


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
    # tessera#672: fixture pool, never the live fleet queue (see above).
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
    # The record also carries the tmp path, and pytest's session counter can spell 206 there
    # (pytest-20673 did); strip the path, then hold the bare number as before.
    assert "206" not in json.dumps(record).replace(str(tmp_path), ""), \
        "a share's counts reached the record"
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
    # tessera#672: fixture pool, never the live fleet queue (see above).
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
    # tessera#672: fixture pool, never the live fleet queue (see above).
    device_less = {"cuda": False, "strict_cuda": False,
                   "device": "torch 2.11.0+cpu reports no CUDA device"}

    # Submitted to cover the CUDA surface, ran on a box with no device.
    landed_wrong = _population("gpu", **device_less)
    verdict = merge_suite._verdict([landed_wrong])
    assert "green" not in verdict, verdict
    assert "no device" in verdict and "gpu" in verdict, verdict

    # A device, but the gate was not armed: the run passed because nothing
    # made it refuse, which is the same coverage question one step earlier.
    unarmed = _population("gpu", cuda=True, strict_cuda=False,
                          device="torch 2.11, 1 CUDA device(s)")
    assert "not armed" in merge_suite._verdict([unarmed])

    # Both legs: a device, and the gate that would have refused without one.
    covered = _population("gpu")
    assert merge_suite._verdict([covered]) == "green on 1 population(s): gpu"

    # The x86 arm was never submitted to cover that surface, and must not be
    # held to it -- that box has no torch by design.
    x86 = _population("x86")
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


@pytest.mark.parametrize("master_ref", ["master", "origin/master"])
@pytest.mark.parametrize("lifecycle,failed", [("submit", False), ("resume", True)])
def test_submit_time_master_survives_a_ref_advance(
        tmp_path, monkeypatch, master_ref, lifecycle, failed):
    """#1227: master moves after dispatch, not the measured source.

    The resume case stops the submitter before it can assemble anything and
    resumes with a fresh module, so an in-memory snapshot or one saved only
    after clients return cannot satisfy the submit-time claim.

    Pre-fix: AssertionError: master_head_at_submit must retain the
    pre-dispatch observation, not assembly-time master (all four cases).
    """

    merge = _module()
    checkout = tmp_path / "checkout"
    git, ref, submitted, advanced = _master_history(checkout, master_ref)
    arm = _population("x86", commit=submitted)
    if failed:
        arm["returncode"] = 1
        arm["surface"]["counts"]["failed"] = 1
    expected = merge._assemble_receipt([arm], checkout, assembled_by="submit")
    monkeypatch.setattr(merge, "SHARED_ROOT", tmp_path)
    monkeypatch.setattr(merge, "DEFAULT_RECEIPT_ROOT", tmp_path / "runs")
    monkeypatch.setattr(merge.suite_flux, "source", lambda path: merge.suite_flux.measured_source(path, verifier=None))
    observed = []

    def client(name, config, args, receipt_dir):
        observed.append(receipt_dir)
        git("update-ref", ref, advanced)
        if lifecycle == "resume":
            raise RuntimeError("submitter stopped before receipt assembly")
        return arm

    monkeypatch.setattr(merge, "_submit", client)
    argv = ["merge_suite", "--arm", "x86", "--checkout", str(checkout)]
    monkeypatch.setattr(sys, "argv", argv)
    if lifecycle == "resume":
        with pytest.raises(RuntimeError, match="submitter stopped"):
            merge.main()
        merge = _module()  # discard all in-memory state of the submitter
        monkeypatch.setattr(merge, "_resume", lambda *args: arm)
        monkeypatch.setattr(merge, "_submit", lambda *args: pytest.fail("resume submitted an arm"))
        monkeypatch.setattr(sys, "argv", [*argv, "--resume", str(observed[0])])
    status = merge.main()
    receipt = json.loads((observed[0] / "receipt.json").read_text())

    assert git("rev-parse", "HEAD") == submitted
    assert git("rev-parse", ref) == advanced
    assert status == (1 if failed else 0)
    for key in ("commits_measured", "verdict", "arm_results", "arms"):
        assert receipt[key] == expected[key], key
    population = receipt["population"]
    assert population["master_head_at_submit"] == submitted, (
        "master_head_at_submit must retain the pre-dispatch observation, not assembly-time master")
    assert population["master_ref_used"] == master_ref
    assert population["is_master_head"] is True
    assert population["commit"] == submitted
    ledger = tmp_path / "ledger.md"
    merge._record_markdown(ledger, receipt)
    row = next(line for line in ledger.read_text().splitlines() if "| x86 |" in line)
    assert "| yes | x86 |" in row


@pytest.mark.parametrize("old_receipt", [False, True])
def test_submit_time_master_is_unknown_for_a_legacy_resume(tmp_path, monkeypatch, old_receipt):
    """A prior submit-labelled receipt is not evidence of a genuine snapshot.

    Pre-fix (both cases): AssertionError: legacy runs have no submit-time
    evidence; do not backfill from current master or an old receipt.
    """

    merge = _module()
    checkout = tmp_path / "checkout"
    _, _, submitted, _ = _master_history(checkout, "master")
    arm = _population("x86", commit=submitted)
    receipt_dir = tmp_path / "legacy"
    receipt_dir.mkdir()
    if old_receipt:
        prior = merge._assemble_receipt([arm], checkout, assembled_by="submit")
        prior["population"].update(master_head_at_submit=submitted,
                                   master_ref_used="master", is_master_head=True)
        (receipt_dir / "receipt.json").write_text(json.dumps(prior))
    monkeypatch.setattr(merge, "_resume", lambda *args: arm)
    monkeypatch.setattr(merge, "_submit", lambda *args: pytest.fail("resume submitted an arm"))
    monkeypatch.setattr(sys, "argv", ["merge_suite", "--arm", "x86", "--checkout", str(checkout),
                                     "--resume", str(receipt_dir)])
    assert merge.main() == 0
    receipt = json.loads((receipt_dir / "receipt.json").read_text())
    assert receipt["population"]["master_head_at_submit"] is None, (
        "legacy runs have no submit-time evidence; do not backfill from current master or an old receipt")
    assert receipt["population"]["is_master_head"] is None
    ledger = tmp_path / "ledger.md"
    merge._record_markdown(ledger, receipt)
    row = next(line for line in ledger.read_text().splitlines() if "| x86 |" in line)
    assert "| unknown | x86 |" in row


@pytest.mark.parametrize("mode", ["python", "deadline", "console"])
def test_native_saved_command_keeps_the_existing_pytest_grammar(tmp_path, monkeypatch, mode):
    merge = _module(); jobs = _native_jobs(merge, monkeypatch)
    path = tmp_path / "surface.x86.json"
    command = ["/named/python", "-m", "pytest", "tests/test_a.py", "-n", "2", "--surface-json", str(path)]
    if mode == "deadline": command = merge._timed_command(command, 60)
    if mode == "console": command = ["/named/pytest", *command[3:]]
    jobs.add(tmp_path, "x86", _population("x86")["surface"], command=command)
    record = merge._resume("x86", merge.ARMS["x86"], tmp_path)
    assert record["exit_status_observed"] and record["returncode"] == 0
    assert record["cpus_used"] == 2
    assert merge._targets_of(record["pytest_command"]) == ["tests/test_a.py"]


def test_native_binding_keeps_subtest_summary_coverage(tmp_path, monkeypatch):
    merge = _module(); jobs = _native_jobs(merge, monkeypatch)
    world = jobs.add(tmp_path, "x86", _population("x86")["surface"])
    path = world.directory / "stdout"
    path.write_text(path.read_text().replace(" in 0.10s", ", 28 subtests passed in 0.10s"))
    record = merge._resume("x86", merge.ARMS["x86"], tmp_path)
    assert record["exit_status_observed"] and record["returncode"] == 0
