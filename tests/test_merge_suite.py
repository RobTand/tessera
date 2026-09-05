"""The merge suite composes submissions; these pin what it composes.

``tools/merge_suite.py`` never runs a suite itself -- it builds ``pbrun``
command lines and reads back what they publish.  So the things worth pinning
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
        assert parallel[parallel.index("--dist") + 1] == "loadfile"
        assert merge_suite._arm_cpus(merge_suite.ARMS[name], 6) == 6


def test_one_submission_fans_out_the_x86_arm_and_keeps_the_gpu_arm_serial(tmp_path):
    """One ``--cpus`` for two arms that cannot spend it the same way.

    ``--cpus 8`` composed ``-n 8`` for BOTH arms, and the GPU arm's
    interpreter has no pytest-xdist: it would have exited on an unrecognised
    argument before collecting a test.  So the tool could submit its two arms
    together only at ``--cpus 1``, and every ``-n`` run this branch made was a
    lone ``--arm x86`` submission -- a row with `not submitted in this run`
    beside it, which is the half-a-result its own ledger header warns about.

    Running both arms serially would keep them in one invocation and lose the
    thing that matters: the five failures on ``82f0047`` were an ``-n``-only
    defect in the suite's own conftest, invisible to a serial run of the same
    commit.  A merge check that cannot fan out cannot see that class.

    So the clamp is per arm, and it reaches the pool as well as pytest: the
    GPU arm must not RESERVE eight cores it will not spend.

    Before this test::

        >       assert " -n " not in gpu, (
        E       AssertionError: the gpu arm was submitted with -n, and its
                interpreter has no xdist: ... pbrun.py --gpu --cpus 8 ...
                -- .../prismaquant-cu130/bin/python -m pytest tests -q ...
                -n 8 --dist loadfile --strict-cuda
    """

    merge_suite = _module()
    # A checkout under the shared root, because the x86 arm refuses one only a
    # single box can see; named from the module's own constant rather than
    # spelled out, and never created -- a dry run composes paths and writes
    # nothing but the receipt named by --out.
    checkout = merge_suite.SHARED_ROOT / "ts112-no-such-checkout"
    assert not checkout.exists(), checkout

    out = tmp_path / "receipt.json"
    subprocess.run(
        [sys.executable, str(TOOL), "--dry-run", "--cpus", "8",
         "--checkout", str(checkout), "--out", str(out)],
        capture_output=True, text=True, timeout=180, check=False,
    )
    arms = {record["arm"]: record for record in json.loads(out.read_text())["arms"]}
    assert set(arms) == {"gpu", "x86"}, sorted(arms)

    gpu = arms["gpu"]["pbrun"]
    assert " -n " not in gpu, (
        "the gpu arm was submitted with -n, and its interpreter has no "
        "xdist: " + gpu)
    assert "--cpus 1" in gpu, ("a serial arm must not reserve cores it will "
                               "not spend: " + gpu)
    assert arms["gpu"]["cpus_requested"] == 8
    assert arms["gpu"]["cpus_used"] == 1
    # Why it was clamped travels with the record, so a reader of the receipt
    # does not have to know the arm table.
    assert "xdist" in arms["gpu"]["cpus_note"], arms["gpu"]

    x86 = arms["x86"]["pbrun"]
    assert " -n 8 " in x86, x86
    assert "--cpus 8" in x86, x86
    assert arms["x86"]["cpus_used"] == 8


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


@pytest.mark.parametrize("gpu_tag", ["sparky", "sparklina"])
def test_gpu_submission_delegates_physical_exclusion_to_pbrun(tmp_path, gpu_tag):
    """One declared slot does not exclude a box offering two or three."""
    import shlex

    out = tmp_path / "receipt.json"
    result = subprocess.run(
        [sys.executable, str(TOOL), "--arm", "gpu", "--dry-run",
         "--gpu-tag", gpu_tag, "--cpus", "8",
         "--checkout", str(ROOT), "--out", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert out.exists(), result.stdout + result.stderr
    record = json.loads(out.read_text())["arms"][0]
    invocation = shlex.split(record["pbrun"])
    options = invocation[:invocation.index("--")]
    assert "--exclusive" in options
    assert options[options.index("--tag") + 1] == gpu_tag
    assert "gpu=" not in options[options.index("--demand") + 1]
    assert "--gpu-capacity" not in options, "capacity belongs to PB's live offers"
    assert options[options.index("--cpus") + 1] == "1"
    assert "-n" not in invocation


@pytest.mark.parametrize(("name", "requested"), [("gpu", 8), ("x86", 8), ("x86", 1)])
def test_submission_caps_native_threads_and_compiler_jobs_per_process(
    tmp_path, monkeypatch, name, requested,
):
    import shlex
    from types import SimpleNamespace

    merge_suite = _module()
    limits = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MAX_JOBS")
    for variable in limits:
        monkeypatch.setenv(variable, "64")
    args = SimpleNamespace(cpus=requested, pytest_arg=[], gpu_tag="sparky", mem_gb=4,
                           checkout=ROOT, timeout_s=300, wait_s=300, dry_run=True)
    record = merge_suite._submit(name, merge_suite.ARMS[name], args, tmp_path)
    invocation = shlex.split(record["pbrun"])
    options = invocation[:invocation.index("--")]
    environment = [options[index + 1] for index, option in enumerate(options) if option == "--env"]
    for variable in limits:
        assert [value for value in environment if value.startswith(variable + "=")] == [variable + "=1"], (
            "every pytest process must override ambient/pool native and compiler thread defaults"
        )
    reserved = int(options[options.index("--cpus") + 1])
    command = invocation[invocation.index("--") + 1:]
    processes = int(command[command.index("-n") + 1]) if "-n" in command else 1
    assert processes == reserved
    assert record["process_thread_limits"] == dict.fromkeys(limits, "1")


@pytest.mark.parametrize("name", ["gpu", "x86"])
def test_attempt_timeout_is_inside_the_sealed_submission(tmp_path, name):
    import shlex
    from types import SimpleNamespace

    merge_suite = _module()
    args = SimpleNamespace(cpus=8, pytest_arg=[], gpu_tag="sparky", mem_gb=4,
                           checkout=ROOT, timeout_s=12.5, wait_s=300, dry_run=True)
    record = merge_suite._submit(name, merge_suite.ARMS[name], args, tmp_path)
    invocation = shlex.split(record["pbrun"])
    command = invocation[invocation.index("--") + 1:]
    assert command[:7] == [merge_suite.ARMS[name]["python"], "tools/suite_deadline.py",
                           "--timeout-s", "12.5", "--kill-after-s", "5.0", "--"]
    assert command[7] == merge_suite.ARMS[name]["python"]
    assert "--foreground" not in command and "--preserve-status" not in command
    assert record["timeout_s"] == 12.5
    assert record["timeout_kill_after_s"] == 5.0


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
        except FileNotFoundError:
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
        except FileNotFoundError:
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


def test_live_gpu_submission_requires_an_explicit_placement_tag(monkeypatch, tmp_path, capsys):
    merge_suite = _module()
    monkeypatch.setattr(merge_suite, "DEFAULT_RECEIPT_ROOT", tmp_path / "receipts")
    monkeypatch.setattr(sys, "argv", [str(TOOL), "--arm", "gpu",
                                     "--checkout", str(ROOT),
                                     "--out", str(tmp_path / "receipt.json")])

    def must_not_submit(*args, **kwargs):
        raise AssertionError("a GPU action was submitted without explicit placement")

    monkeypatch.setattr(merge_suite, "_submit", must_not_submit)
    assert merge_suite.main() == 2
    assert "--gpu-tag" in capsys.readouterr().err


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


def test_the_x86_arm_refuses_a_checkout_only_one_box_can_see(tmp_path):
    """pbrun pins a local checkout to the submitting box; say so, do not route around it.

    The skip is decided BEFORE the tool runs, not after.  Deciding it after
    still ran a dry run, and a dry run with no ``--out`` writes its receipt
    under ``DEFAULT_RECEIPT_ROOT`` -- so every full suite run on a shared
    checkout left a directory in the store that holds the real ones, each
    holding one arm with ``"status": "not submitted (--dry-run)"``.  Seventeen
    were there when this was found and eighteen twenty-five minutes later;
    the count is not a constant, it grows with the suite, which is the point.
    Of the eighteen, sixteen were written by pool runs on dl380g10, one by the
    GPU arm's own suite on sparky, and one is a deliberate ``--dry-run`` from a
    terminal.  Nothing read them, but a reader of the store cannot tell a run
    that measured nothing from one that has not finished, which is the reading
    tessera#112 is about.

    Counted by opening every ``receipt.json`` under the store and taking those
    whose single arm carries that status; the first count said "fourteen on
    dl380g10", which was wrong before it was stale.
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


@pytest.mark.parametrize("master_ref,at_master", [
    ("master", True), ("master", False),
    ("origin/master", True), ("origin/master", False), (None, None),
])
def test_the_receipt_states_which_tree_it_is_about(tmp_path, master_ref, at_master):
    """Exercise the ref states, not whichever refs the test runner inherited."""

    checkout = tmp_path / "checkout"
    checkout.mkdir()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(checkout), "-c", "user.name=Test",
             "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
             *args], text=True).strip()

    git("init", "-q", "--initial-branch=measured")
    git("commit", "-q", "--allow-empty", "-m", "fixture base")
    base = git("rev-parse", "HEAD")
    if master_ref:
        ref = ("refs/heads/master" if master_ref == "master"
               else "refs/remotes/origin/master")
        git("update-ref", ref, base)
    if at_master is False:
        git("commit", "-q", "--allow-empty", "-m", "fixture branch")
    head = git("rev-parse", "HEAD")

    out = tmp_path / "receipt.json"
    result = subprocess.run(
        [sys.executable, str(TOOL), "--arm", "gpu", "--dry-run",
         "--checkout", str(checkout), "--out", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode != 0, "a dry run has covered no population"
    receipt = json.loads(out.read_text())
    population = receipt["population"]
    assert population["commit"] == head
    assert population["master_ref_used"] == (master_ref or "none resolved")
    assert population["master_head_at_submit"] == (base if master_ref else None)
    assert population["is_master_head"] is at_master
    # A parentless PB snapshot has neither ref. That is an unknown comparison,
    # not a failed receipt and not a reason to manufacture a master ref.
    assert receipt["verdict"] == "not run"
    # Both arms' numbers live under one key, so quoting one without its device
    # means quoting it out of this object rather than out of a scrollback.
    assert "reading_note" in receipt


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
    # Every field a published population carries, because the verdict reads
    # them: a fixture with three keys in it is not a smaller version of a real
    # population, it is a different object (see ``_population``).
    clean = _population("gpu")["surface"]
    broken = _population("x86", counts={"passed": 9, "failed": 1, "error": 0,
                                        "skipped": 1})["surface"]

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
    for arm in ("gpu", "x86"):
        (surfaces / f"surface.{arm}.json").write_text(
            json.dumps(_population(arm)["surface"]))
    out = tmp_path / "receipt.json"
    ledger = tmp_path / "ledger.md"
    result = subprocess.run(
        [sys.executable, str(TOOL), "--resume", str(surfaces),
         "--checkout", str(ROOT), "--out", str(out), "--record", str(ledger),
         "--pool-root", str(tmp_path / "pool")],
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


def test_a_resume_reads_the_pool_it_is_pointed_at_not_the_live_one(tmp_path):
    """A test must not read the fleet's live queue, and could not help it.

    `_pool_actions_that_wrote` opens every outcome record in `pb-queue/done`
    and `pb-queue/failed` and the CAS request beside each, over NFS. That is
    fine for a real resume -- it is how the exit status is derived from a
    table the pool publishes -- and it is not fine inside a test: the cost is
    one NFS read per finished action, it grows with the fleet's history, and
    nothing about the test's verdict depends on what the live pool happens to
    hold.

    It stopped being theoretical. With 547 finished actions in the queue, the
    two resume tests that run this tool as a subprocess went past their own
    timeouts on sparky under load. Before `--pool-root`::

        E       subprocess.TimeoutExpired: Command '[... merge_suite.py
                --resume ... --record ...]' timed out after 120 seconds

    So the tool names the pool it reads, defaulting to the live one, and the
    tests point it at a queue they built. The default is asserted here too: a
    flag that quietly changed where a real resume looks would be worse than
    the slow scan.
    """

    merge_suite = _module()
    assert merge_suite.POOL_QUEUE == merge_suite.SHARED_ROOT / \
        "prismabuild-fleet" / "pb-queue"
    assert merge_suite.POOL_CAS_REQUESTS == merge_suite.SHARED_ROOT / \
        "prismabuild-fleet" / "cas" / "requests"

    receipt_dir = tmp_path / "receipt"
    receipt_dir.mkdir()
    surface = receipt_dir / "surface.gpu.json"
    key = "cafe" + "0" * 60
    _, requests = _fake_pool(tmp_path / "pool", surface,
                             [(key, "done", 0, "sparky")])
    _gpu_population(surface, producer=_request_of(requests, key))

    out = tmp_path / "receipt.json"
    started = time.monotonic()
    rc = subprocess.run(
        [sys.executable, str(TOOL), "--arm", "gpu", "--resume",
         str(receipt_dir), "--checkout", str(ROOT), "--out", str(out),
         "--pool-root", str(tmp_path / "pool")],
        capture_output=True, text=True, timeout=120)
    elapsed = time.monotonic() - started
    assert rc.returncode in (0, 1), rc.stdout + rc.stderr

    record = json.loads(out.read_text())["arms"][0]
    # The status came from the fake pool, which is the proof the flag is read.
    assert record["returncode"] == 0, record
    assert record["exit_status_source"] == "pool", record
    assert record["pool_action"]["action_key"].startswith("cafe"), record
    # And it did not walk the live queue on the way: a scan of that took the
    # same call past 120 s on this box.
    assert elapsed < 60, elapsed


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


def _attempt_stdout(surface_json, counts=None, published=True):
    """What the attempt's pytest printed, as PrismaBuild's worker captured it.

    ``detail.stdout`` on a real outcome record is the attempt's whole stdout
    (the worker ``communicate()``s, nothing is truncated), and two lines of it
    are the attempt's own account of what it published: the conftest's
    ``tessera surface: population written to <path>`` and pytest's summary
    line with the counts the population also carries. A test that wants an
    attempt which died before publishing says ``published=False`` and gets
    neither line, which is what such an attempt leaves behind.
    """

    counts = {"passed": 1827, "skipped": 10, **(counts or {})}
    summary = ", ".join(f"{n} {word}" for word, n in counts.items() if n)
    lines = ["....s...."]
    if published:
        lines += [f"tessera surface: population written to {surface_json}",
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


def _fake_pool(root, surface_json, actions, command=None, params=None,
               outcome=None, detail=None):
    """A pb-queue and a CAS request tree holding exactly these actions.

    The layout is the fleet's, not an invention: an outcome record per action
    in ``pb-queue/<state>/<key>.json`` carrying its top-level ``status`` and
    the final attempt's ``detail`` (return code and captured stdout), and the
    command that action ran in ``cas/requests/<key[:2]>/<key>.json`` under a
    ``checkout_snapshot`` naming the tree. The default command is the one this
    tool seals for a GPU submission -- the deadline wrapper around ``python -m
    pytest`` -- composed by the tool rather than copied, so the fixture is the
    tool's own shape. Built here rather than read from ``/mnt/shared`` so the
    test states its own population -- a test whose verdict depends on what the
    live fleet happens to hold is the box-dependence this file already fixed
    twice.
    """

    queue, requests = root / "pb-queue", root / "cas" / "requests"
    for state in ("done", "failed"):
        (queue / state).mkdir(parents=True, exist_ok=True)
    if command is None:
        merge_suite = _module()
        command = merge_suite._timed_command(
            merge_suite._command(merge_suite.ARMS["gpu"], surface_json, []),
            7200.0)
    snapshot = {"schema": "prismaquant.prismabuild.pbrun_checkout_snapshot.v1",
                "commit": "a" * 40, "subdirectory": "."}
    for key, state, returncode, host in actions:
        (requests / key[:2]).mkdir(parents=True, exist_ok=True)
        (requests / key[:2] / f"{key}.json").write_text(json.dumps({
            "params": _with({"command": command,
                             "checkout_snapshot": snapshot}, params)}))
        (queue / state / f"{key}.json").write_text(json.dumps(_with({
            "attempts": 1, "claimed_host": host,
            "claimed_unix": 1_756_990_000.0,
            "status": "executed" if state == "done" else "failed",
            "detail": _with({"returncode": returncode, "status": "executed",
                             "elapsed_s": 511.4,
                             "stdout": _attempt_stdout(surface_json)},
                            detail)}, outcome)))
    return queue, requests


def _gpu_population(path, commit="a" * 40, producer=None, **overrides):
    """A GPU population at ``path``; ``producer`` is the CAS request it names.

    A population the suite publishes from a PrismaBuild snapshot carries the
    sealed action it ran under in ``source_identity.excluded_metadata``:
    ``suite_source._verified_stamp`` writes the action key and the digest of
    the request's bytes there once it has verified the closure member against
    that request. ``producer=None`` is the shape with no such stamp -- a
    population from before it existed, or one whose source came out
    ``unknown`` -- which names no action at all.
    """

    surface = _population("gpu", commit=commit)["surface"]
    if producer is not None:
        producer = Path(producer)
        surface["source_identity"]["excluded_metadata"] = [{
            "path": f".pbrun-closure.{producer.stem[:16]}.json",
            "bytes": 153, "sha256": "d" * 64,
            "action_key": producer.stem,
            "request_sha256": hashlib.sha256(
                producer.read_bytes()).hexdigest()}]
    surface.update(overrides)
    path.write_text(json.dumps(surface))


def _request_of(requests, key):
    return requests / key[:2] / f"{key}.json"


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
    key = "beef" + "0" * 60
    merge_suite.POOL_QUEUE, merge_suite.POOL_CAS_REQUESTS = _fake_pool(
        tmp_path, surface, [(key, "done", 0, "sparky")])
    _gpu_population(surface,
                    producer=_request_of(merge_suite.POOL_CAS_REQUESTS, key))

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

    Since #294 the population names its producer, so the only two records
    that can both bind to it are two records of THAT action -- one in ``done``
    and one in ``failed`` is the pool disagreeing with itself about a single
    action, and still not one status. A second action on the same path is not
    ambiguity; it is refused by name, which the last assertion pins.
    """

    merge_suite = _module()
    receipt_dir = tmp_path / "receipt"
    receipt_dir.mkdir()
    surface = receipt_dir / "surface.gpu.json"
    key, other = "beef" + "0" * 60, "cafe" + "0" * 60
    merge_suite.POOL_QUEUE, merge_suite.POOL_CAS_REQUESTS = _fake_pool(
        tmp_path, surface, [(key, "done", 0, "sparky"),
                            (key, "failed", 1, "sparky"),
                            (other, "failed", 1, "sparky")])
    _gpu_population(surface,
                    producer=_request_of(merge_suite.POOL_CAS_REQUESTS, key))

    record = merge_suite._resume("gpu", merge_suite.ARMS["gpu"], receipt_dir)
    assert record["returncode"] is None, record
    assert record["exit_status_observed"] is False, record
    assert len(record["pool_actions_matching"]) == 2, record
    assert "no single exit status" in record["exit_status_note"], record
    assert not merge_suite._verdict([record]).startswith("green on")
    assert [r for r in record["pool_actions_refused"]
            if r.startswith(other[:12])], record


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
         "--resume", str(receipt_dir), "--checkout", str(ROOT),
         "--pool-root", str(tmp_path / "pool")],
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
    suite, which is how eighteen one-arm ``not run`` directories came to sit
    beside the four real ones -- a count that rises with every suite run, not
    a fixed number.  Read statically rather than by watching the
    store: watching it would be a race against every other run on the fleet,
    and the property being pinned is about this file, not about a placement.
    (The store's count of them is not stable enough to assert on for the same
    reason -- it rises whenever any box runs the suite.)

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


def _resumed_with(tmp_path, producer=True, population=None, **pool):
    """A resumed GPU record whose pool holds exactly one finished action.

    The population names that action as its producer unless the test says
    ``producer=False``; ``population`` overrides fields of the population and
    the rest is the pool fixture's.
    """

    merge_suite = _module()
    receipt_dir = tmp_path / "receipt"
    receipt_dir.mkdir(exist_ok=True)
    surface = receipt_dir / "surface.gpu.json"
    key = "beef" + "0" * 60
    merge_suite.POOL_QUEUE, merge_suite.POOL_CAS_REQUESTS = _fake_pool(
        tmp_path, surface, [(key, "done", 0, "sparky")], **pool)
    _gpu_population(
        surface, **(population or {}),
        producer=(_request_of(merge_suite.POOL_CAS_REQUESTS, key)
                  if producer else None))
    return merge_suite, merge_suite._resume("gpu", merge_suite.ARMS["gpu"],
                                            receipt_dir)


def _unobserved(merge_suite, record, *named):
    """The row adopted nothing, and every reason in ``named`` was given."""

    assert record["exit_status_observed"] is False, record
    assert record["returncode"] is None, record
    assert "pool_action" not in record, record
    assert not merge_suite._verdict([record]).startswith("green on"), record
    reasons = record.get("pool_actions_refused", [])
    for phrase in named:
        assert any(phrase in reason for reason in reasons), (phrase, record)


def test_an_action_that_only_read_the_population_is_not_its_producer(tmp_path):
    """#218: any command containing the path was treated as the writer.

    A successful ``cat`` of a published summary says nothing about whether the
    suite that wrote it crashed, timed out or failed after publication -- and
    it was enough to turn a resumed arm green, because the join was mere argv
    membership of the path string.
    """

    merge_suite, record = _resumed_with(
        tmp_path, command=["cat", str(tmp_path / "receipt/surface.gpu.json")])
    _unobserved(merge_suite, record, "`cat`")


def test_a_command_whose_effective_output_is_elsewhere_cannot_claim_this_path(
        tmp_path):
    """pytest takes the last ``--surface-json``; so does the reader of it."""

    ours = str(tmp_path / "receipt/surface.gpu.json")
    theirs = str(tmp_path / "somewhere-else.json")

    _, overridden = _resumed_with(tmp_path, command=[
        "python", "-m", "pytest", "tests",
        "--surface-json", ours, "--surface-json", theirs])
    assert overridden["exit_status_observed"] is False, overridden

    _, effective = _resumed_with(tmp_path, command=[
        "python", "-m", "pytest", "tests",
        "--surface-json", theirs, "--surface-json", ours])
    assert effective["exit_status_observed"] is True, effective
    assert effective["returncode"] == 0, effective


def test_a_status_recorded_for_another_source_tree_is_not_borrowed(tmp_path):
    """The population names the tree it measured; so does the sealed action."""

    merge_suite, record = _resumed_with(tmp_path, params={
        "checkout_snapshot": {
            "schema": "prismaquant.prismabuild.pbrun_checkout_snapshot.v1",
            "commit": "b" * 40, "subdirectory": "."}})
    _unobserved(merge_suite, record, "commit")


def test_a_retry_that_never_published_cannot_claim_the_earlier_population(
        tmp_path):
    """The pool requeues on any non-zero exit, and a retry may never publish.

    An attempt that died before its terminal summary leaves the previous
    attempt's population at the path and its own status in the outcome record.
    Reading the two together attributes one attempt's bytes to another
    attempt's exit. #218 refused this by clock: a population more than
    ``ATTEMPT_CLOCK_SLACK_S`` older than the claim was another attempt's. That
    is a bound on how fast a retry may follow, asserted and not measured, and
    a retry five minutes behind -- the shape codex replayed for #294 -- was
    inside it and bound. What distinguishes the attempts is not the clock but
    the record: the worker captured the attempt's whole stdout, and an attempt
    that published says so on it. This one did not.

    Before the fix (the retry was inside the 600 s allowance)::

        >       assert record["exit_status_observed"] is False, record
        E       AssertionError: {'arm': 'gpu', ..., 'exit_status_observed': True, ...}
        E       assert True is False
    """

    surface = tmp_path / "receipt" / "surface.gpu.json"
    merge_suite, record = _resumed_with(
        tmp_path, outcome={"attempts": 2, "claimed_unix": time.time() + 300},
        detail={"stdout": _attempt_stdout(surface, published=False)})
    _unobserved(merge_suite, record, "never said it published")


def test_the_clock_is_not_the_evidence_of_which_attempt_published(tmp_path):
    """A day between publication and claim binds if the attempt says it wrote.

    The converse of the retry test: #218's clock rule would have refused this
    record, and the rule it stood in for -- the attempt's own account of what
    it published -- accepts it. The 600 s constant and its "asserted, not
    measured" comment are gone with the inference.

    Before the fix::

        >       assert record["exit_status_observed"] is True, record
        E       AssertionError: {'arm': 'gpu', ..., 'exit_status_observed': False, ...}
        E       assert False is True
    """

    merge_suite, record = _resumed_with(
        tmp_path, outcome={"claimed_unix": time.time() + 86400})
    assert record["exit_status_observed"] is True, record
    assert record["returncode"] == 0, record
    assert not hasattr(merge_suite, "ATTEMPT_CLOCK_SLACK_S")


def test_a_wrapper_the_tool_does_not_know_is_refused_by_name(tmp_path):
    """#294: a token spelled ``pytest`` anywhere in argv made a producer.

    ``echo pytest --surface-json <path>`` exits 0 having written nothing, and
    the #218 join accepted it because ``_runs_pytest`` looked for the word.
    The reader now parses the command shapes this tool seals -- ``pytest``,
    ``python -m pytest`` and the ``tools/suite_deadline.py ... --`` wrapper
    around them -- and refuses any other program by name.

    Before the fix::

        >       assert record["exit_status_observed"] is False, record
        E       AssertionError: {'arm': 'gpu', ..., 'exit_status_observed': True, ...}
        E       assert True is False
    """

    ours = str(tmp_path / "receipt" / "surface.gpu.json")
    merge_suite, record = _resumed_with(
        tmp_path, command=["echo", "pytest", "--surface-json", ours])
    _unobserved(merge_suite, record, "`echo`")

    # And a wrapper that is not the one this tool seals is refused as such,
    # even though the real invocation is inside it.
    merge_suite, record = _resumed_with(tmp_path, command=[
        "timeout", "7200", "python", "-m", "pytest", "tests",
        "--surface-json", ours])
    _unobserved(merge_suite, record, "`timeout`")


def test_missing_binding_evidence_is_unobserved_not_established(tmp_path):
    """#294: absent identity was read as agreement.

    ``_binding_refusal`` compared snapshot commit to population commit only if
    both were present and checked the clock only if ``claimed_unix`` was a
    number, so a request with no ``checkout_snapshot`` and an outcome with no
    claim time passed every check by having nothing to check. A binding is
    established by evidence, and a record without the evidence has not
    established it.

    Before the fix::

        >       assert record["exit_status_observed"] is False, record
        E       AssertionError: {'arm': 'gpu', ..., 'exit_status_observed': True, ...}
        E       assert True is False
    """

    merge_suite, record = _resumed_with(
        tmp_path, params={"checkout_snapshot": None},
        outcome={"claimed_unix": None})
    _unobserved(merge_suite, record, "no checkout_snapshot")


def test_a_population_that_names_no_producer_adopts_no_status(tmp_path):
    """A population's stamp is the identifier that binds it to an action.

    ``suite_source._verified_stamp`` writes the sealed action's key and the
    digest of its request into the population once the closure member is
    verified against that request; a population without it -- one from before
    the stamp, or one whose source came out ``unknown`` -- names no action, so
    no action's status is its status. The receipt is still read and the
    counts still shown; only the exit status is withheld, by name.

    Before the fix::

        >       assert record["exit_status_observed"] is False, record
        E       AssertionError: {'arm': 'gpu', ..., 'exit_status_observed': True, ...}
        E       assert True is False
    """

    merge_suite, record = _resumed_with(tmp_path, producer=False)
    _unobserved(merge_suite, record, "names no sealed action")
    assert record["surface"]["counts"]["passed"] == 1827, record

    # Naming a different action is the same refusal with both keys in it.
    merge_suite, record = _resumed_with(
        tmp_path, population={"source_identity": {
            **_population("gpu")["surface"]["source_identity"],
            "excluded_metadata": [{"action_key": "dead" + "0" * 60,
                                   "request_sha256": "e" * 64}]}})
    _unobserved(merge_suite, record, "dead00000000", "beef00000000")


def test_a_request_the_population_did_not_see_is_not_its_producer(tmp_path):
    """The stamp digests the request's bytes; a changed request is another.

    The action key is a digest of the request body, but the reader has only
    the pool's word for which body sits under which key. The population
    carries its own digest of the bytes it verified against, and a request
    file whose bytes differ is not the request that produced it.

    Before the fix::

        >       assert record["exit_status_observed"] is False, record
        E       AssertionError: {'arm': 'gpu', ..., 'exit_status_observed': True, ...}
        E       assert True is False
    """

    merge_suite = _module()
    receipt_dir = tmp_path / "receipt"
    receipt_dir.mkdir()
    surface = receipt_dir / "surface.gpu.json"
    key = "beef" + "0" * 60
    merge_suite.POOL_QUEUE, merge_suite.POOL_CAS_REQUESTS = _fake_pool(
        tmp_path, surface, [(key, "done", 0, "sparky")])
    request = _request_of(merge_suite.POOL_CAS_REQUESTS, key)
    _gpu_population(surface, producer=request)
    # The request is rewritten -- same command, a byte of whitespace more.
    request.write_text(request.read_text() + " ")

    record = merge_suite._resume("gpu", merge_suite.ARMS["gpu"], receipt_dir)
    _unobserved(merge_suite, record, "request_sha256")


def test_a_lease_lost_record_carries_an_earlier_attempts_detail(tmp_path):
    """Only ``executed`` and ``failed`` records describe their final attempt.

    The pool's lease reaper moves an action to ``failed`` as
    ``lease_lost_max_attempts`` without touching ``detail``, so the detail on
    such a record is whatever the last attempt to FINISH wrote -- observed on
    ``dbd91b92`` in the live queue, where the detail was an earlier attempt's.
    Its return code is real and it is not the final attempt's, so it is
    refused by the status that says so.

    Before the fix::

        >       assert record["exit_status_observed"] is False, record
        E       AssertionError: {'arm': 'gpu', ..., 'exit_status_observed': True, ...}
        E       assert True is False
    """

    merge_suite, record = _resumed_with(
        tmp_path, outcome={"status": "lease_lost_max_attempts", "attempts": 3})
    _unobserved(merge_suite, record, "lease_lost_max_attempts")


def test_an_attempt_whose_summary_disagrees_with_the_population_is_not_bound(
        tmp_path):
    """The attempt's counts and the file's counts come from one table.

    ``tests/conftest.py`` writes ``counts`` from ``terminalreporter.stats``,
    the same table pytest's summary line is built from, so an attempt that
    published this population printed these numbers. One that printed others
    published a different population -- an earlier attempt's file is at the
    path, or a later one overwrote it -- and its status is not this file's.

    Before the fix::

        >       assert record["exit_status_observed"] is False, record
        E       AssertionError: {'arm': 'gpu', ..., 'exit_status_observed': True, ...}
        E       assert True is False
    """

    surface = tmp_path / "receipt" / "surface.gpu.json"
    merge_suite, record = _resumed_with(
        tmp_path, detail={"stdout": _attempt_stdout(
            surface, counts={"passed": 1826, "failed": 1})})
    _unobserved(merge_suite, record, "1826 passed")


def test_a_later_attempt_that_could_not_publish_cannot_inherit(tmp_path):
    """The acceptance shape for #294: a producer binds, its successor does not.

    The same population, the same path, two records of two actions on the
    same tree. The first is the producer: its request is the one the
    population stamps, its stdout says it wrote this path with these counts,
    and its status is the row's. The second ran five minutes later, exited 0,
    and its stdout has no publication line -- it could not publish (the suite
    aborted, the path was unwritable, whatever) -- and it is refused by name
    rather than inheriting the file the first one left. Under #218 it was
    inside the clock allowance, so it counted as a second writer and the
    genuine producer's status was lost to "no single exit status".

    Before the fix::

        >       assert record["exit_status_observed"] is True, record
        E       AssertionError: {'arm': 'gpu', ..., 'exit_status_observed': False, ...}
        E       assert False is True
    """

    merge_suite = _module()
    receipt_dir = tmp_path / "receipt"
    receipt_dir.mkdir()
    surface = receipt_dir / "surface.gpu.json"
    producer, later = "beef" + "0" * 60, "cafe" + "0" * 60
    _fake_pool(tmp_path, surface, [(producer, "done", 0, "sparky")])
    merge_suite.POOL_QUEUE, merge_suite.POOL_CAS_REQUESTS = _fake_pool(
        tmp_path, surface, [(later, "done", 0, "sparklina")],
        outcome={"attempts": 1, "claimed_unix": time.time() + 300},
        detail={"stdout": _attempt_stdout(surface, published=False)})
    _gpu_population(surface,
                    producer=_request_of(merge_suite.POOL_CAS_REQUESTS,
                                         producer))

    record = merge_suite._resume("gpu", merge_suite.ARMS["gpu"], receipt_dir)
    assert record["exit_status_observed"] is True, record
    assert record["returncode"] == 0, record
    assert record["pool_action"]["action_key"] == producer, record
    assert record["pool_action"]["host"] == "sparky", record
    assert merge_suite._verdict([record]).startswith("green on")
    refused = record["pool_actions_refused"]
    assert len(refused) == 1 and refused[0].startswith(later[:12]), refused
