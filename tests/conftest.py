"""Shared fixtures: small, fully-determined artifacts built from bytes."""

import hashlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import box_artifacts  # noqa: E402 -- the one home for out-of-tree roots

from tessera.container import serialize  # noqa: E402
from tessera.grammar import bresenham_rate_schedule, root_from_q256  # noqa: E402
from tessera.layout import (  # noqa: E402
    TerminalSpec,
    build_plane_region,
    build_planes,
    build_terminal,
)
from tessera.planes import PlaneKind  # noqa: E402
from tessera.manifest import (  # noqa: E402
    ArrangementMode,
    BranchIdentity,
    ContainerClass,
    Geometry,
    Manifest,
    RotationState,
)

ALPHABET_BLOB = bytes(range(16))
DESCENDANT_BLOB = bytes(range(32))


def make_geometry(rows=8, columns=32, superblock_columns=8):
    return Geometry(
        rows=rows,
        columns=columns,
        superblock_columns=superblock_columns,
        group_weights=32,
        half_weights=16,
        quantizable_params=rows * columns,
    )


def make_artifact(q256=512, rows=8, columns=32, superblock_columns=8):
    """Build a complete, self-consistent artifact plus its terminal ladder."""
    geometry = make_geometry(rows, columns, superblock_columns)
    root = root_from_q256(q256)
    rates = bresenham_rate_schedule(root, columns)
    payloads = {
        PlaneKind.ALPHABET: ALPHABET_BLOB,
        PlaneKind.DESCENDANT: DESCENDANT_BLOB,
    }
    planes = build_planes(
        geometry,
        rates,
        ALPHABET_BLOB,
        DESCENDANT_BLOB,
        max_released=4,
        payloads=payloads,
    )
    plane_region = build_plane_region(planes, payloads)

    specs = [
        TerminalSpec("t-po2", (0,) * columns, with_scale_base=True),
        TerminalSpec(
            "t-c3",
            tuple(3 - rate for rate in rates),
            with_scale_base=True,
        ),
        TerminalSpec(
            "t-nvfp4",
            tuple(3 - rate for rate in rates),
            released_positions=4,
            with_scale_base=True,
            with_scale_refine=True,
            with_diagonals=True,
        ),
    ]
    terminals = tuple(
        build_terminal(
            geometry,
            rates,
            spec,
            planes,
            len(ALPHABET_BLOB),
            len(DESCENDANT_BLOB),
            plane_region=plane_region,
        )
        for spec in specs
    )

    manifest = Manifest(
        encoder_profile_id=hashlib.sha256(b"profile").digest(),
        branch=BranchIdentity(
            unit_id="layers.0.mlp.down_proj",
            root_q256=q256,
            rotation=RotationState.NONE,
            container=ContainerClass.GRIDBOOK,
        ),
        geometry=geometry,
        arrangement=ArrangementMode.BRESENHAM,
        rates=rates,
        planes=planes,
        terminals=terminals,
        payload_digest=hashlib.sha256(plane_region).digest(),
    )
    return manifest, plane_region, serialize(manifest, plane_region)


@pytest.fixture
def artifact():
    return make_artifact()


# --- what a torch-free interpreter can collect -----------------------------
#
# The `pure` CI job installs pytest and nothing else, to prove structurally
# that the byte layer needs no torch.  Most test modules DO need torch, and
# they cannot be collected there.
#
# Naming them is the obvious move and the wrong one: a roster of forty-odd
# filenames is a second place to remember, and the failure mode is silent --
# a new torch-needing test file turns the job red, someone appends it, and
# nothing checks that the entries still describe reality.  The roster is not
# the decision here; "this module cannot import without its dependency" is,
# and that is a question the module answers itself.
#
# So ask it.  In an interpreter that HAS the dependency this costs nothing:
# the loop does not run at all.  In one that does not, each failing import
# fails on its own import line, before any work.
def _own_import_roots() -> frozenset:
    """The top-level names this tree provides on its own import roots.

    Read off the filesystem (``repo/``, ``repo/src``, ``repo/tests``: the
    roots pytest and this conftest put on ``sys.path``) rather than listed, so
    a package added to the tree is "ours" the moment it exists.  The limit,
    stated: a top-level package deleted outright is no longer on any root, so
    a test importing it is classified as a dependency skip, not a failure.
    A moved or deleted SUBmodule (``tessera.decode``) is still caught, and
    that is the breakage this job exists to catch (tessera#154).
    """
    repo = Path(__file__).resolve().parents[1]
    names = set()
    for base in (repo, repo / "src", repo / "tests"):
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if child.name.startswith((".", "_")):
                continue
            if child.is_dir() and (child / "__init__.py").is_file():
                names.add(child.name)
            elif child.suffix == ".py":
                names.add(child.stem)
    return frozenset(names)


def _third_party_import_failure(exc: BaseException, own=None) -> bool:
    """Did this fail for a missing dependency, or for a real bug?

    Only the former is a reason to skip a file.  A module that raises
    ``ImportError`` because one of ITS OWN names moved must still fail the
    run -- that is the breakage this whole job exists to catch.  "Its own"
    is the set ``_own_import_roots`` reads off the tree, never a spelled-out
    roster.
    """
    missing = getattr(exc, "name", None) or ""
    root = missing.split(".", 1)[0]
    if not root:
        return False
    return root not in (_own_import_roots() if own is None else own)


# The probe executes every test module.  It does so in ONE child interpreter
# (one, not one per module: the cost is a single startup), so the process that
# goes on to collect and run the suite has executed none of them first -- no
# module-level side effect, monkeypatch or ``sys.modules`` entry leaks from
# the probe into the run.  The child is given the same import roots this
# conftest gives the run, and the own-name set, and answers with the files
# that failed on a missing dependency.
_PROBE = r"""
import importlib.util, json, sys
from pathlib import Path
here, own = Path(sys.argv[1]), set(json.loads(sys.argv[2]))
sys.path[:0] = json.loads(sys.argv[3])
skip = []
for path in sorted(here.glob("test_*.py")):
    spec = importlib.util.spec_from_file_location("_probe_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (ImportError, ModuleNotFoundError) as exc:
        root = (getattr(exc, "name", None) or "").split(".", 1)[0]
        if root and root not in own:
            skip.append(path.name)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        # Not ``Exception``: a module that already says
        # ``pytest.importorskip("torch")`` raises pytest's own ``Skipped``,
        # which derives from ``BaseException``.  That file is handling
        # itself correctly and must NOT be ignored -- pytest will report
        # it as a skip, which is the truth.  Anything else here is the
        # file's own problem and belongs in the run as the failure it is.
        pass
print(json.dumps(skip))
"""


def _probe_uncollectable(here: Path, own) -> list:
    import json
    import subprocess

    tests = Path(__file__).resolve().parent
    roots = [str(tests.parent / "src"), str(tests)]   # the two roots lines 10-11 give the run
    # The child is marked so that a test module importing this conftest does
    # not start a probe of its own from inside the probe.
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(here), json.dumps(sorted(own)), json.dumps(roots)],
        capture_output=True, text=True, check=True,
        env={**os.environ, _PROBE_MARK: "1"})
    return json.loads(result.stdout.strip().splitlines()[-1])


_PROBE_MARK = "TESSERA_COLLECTION_PROBE"


def _uncollectable() -> list:
    import importlib.util

    if importlib.util.find_spec("torch") is not None or os.environ.get(_PROBE_MARK):
        return []
    return _probe_uncollectable(Path(__file__).resolve().parent, _own_import_roots())


collect_ignore = _uncollectable()


# --- what this run did NOT exercise ---------------------------------------
#
# tessera#112: master was red on three CUDA-gated tests while GitHub Actions,
# the x86 pool suite and a local CPU run all read green.  None of the three
# could have seen it.  The failure was not any one signal; it was that a run
# does not say which population it covered, so "41 passed, 4 skipped" reads as
# coverage when three of those four skips are the tests being asked about.
#
# Two things fix that, and they are different in kind.
#
# The *diagnostic* is unconditional: every run prints the device it had, how
# many tests it skipped, how many modules it never collected, and -- verbatim,
# never pattern-matched -- what the skip reasons said.  A reason histogram is
# the exact answer where a regex over "cuda|gpu|triton" would be a guess, and
# a guess that undercounts is the same blindness one level down.  There is no
# threshold: "more than N" would be a constant nobody derived, and the honest
# statement is the count itself (principle 2).
#
# The *gate* is ``--strict-cuda`` (or ``TESSERA_STRICT_CUDA=1``, for a wrapper
# that seals a command line), and it has three legs, because device presence
# alone was never the claim it was read as (tessera#152).
#
#   1. It refuses the session outright when this interpreter has no CUDA
#      device.  Exact, and checkable before anything runs.
#   2. At the end it refuses a run that touched no device.  A device the
#      session never used is not coverage of the surface, and every other
#      check is satisfied by a run that collected the suite and skipped it.
#      "Touched" is measured, not declared: torch's own allocator counts the
#      allocations a test made, so this is a floor rather than a roster, and a
#      floor cannot manufacture a false green.
#   3. At the end it refuses a run that skipped for a MISSING BOX ARTIFACT --
#      the checkpoints and serve logs ``tests/box_artifacts.py`` resolves.
#      Those are the gates of tessera#146: they hold the line between a
#      captured encode and an eager one producing different bytes, and on a
#      box without the directory they skipped quietly while the arm reported
#      the surface covered.  An arm submitted to cover the surface must say so
#      when it cannot, which means giving that box the roots (the skip reason
#      names the variable) or not claiming the surface.
#
# Legs 2 and 3 read the population this file already publishes; there is no
# second source of truth and no reason string is classified by pattern.
#
# Nothing here imports torch at module scope: the ``pure`` CI job imports this
# file and asserts torch stayed out of ``sys.modules``.

_STRICT_CUDA_HELP = (
    "refuse the session unless a CUDA device is present, so a run submitted to "
    "cover the CUDA-gated surface cannot pass by skipping it"
)


def pytest_addoption(parser):
    parser.addoption(
        "--strict-cuda",
        action="store_true",
        default=False,
        help=_STRICT_CUDA_HELP,
    )
    parser.addoption(
        "--surface-json",
        action="store",
        default=os.environ.get("TESSERA_SURFACE_JSON", ""),
        metavar="PATH",
        help=(
            "write this run's population -- device, counts, verbatim skip "
            "reasons, uncollected modules -- as JSON, so a receipt is derived "
            "from a table this run publishes rather than scraped from its prose"
        ),
    )


#: The two ways to spell the environment switch, and nothing else.  The first
#: reading of it treated only ``("", "0", "false")`` as off, so ``False``,
#: ``FALSE`` and ``no`` all armed the gate and so did every typo -- a gate that
#: arms itself on a misspelling is one that will one day disarm itself on one
#: (tessera#152).
_STRICT_ENV = "TESSERA_STRICT_CUDA"
_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("", "0", "false", "no", "off")


def _strict_cuda(config) -> bool:
    if config.getoption("--strict-cuda"):
        return True
    raw = os.environ.get(_STRICT_ENV, "")
    spelling = raw.strip().lower()
    if spelling in _TRUTHY:
        return True
    if spelling in _FALSY:
        return False
    raise pytest.UsageError(
        f"{_STRICT_ENV}={raw!r} is neither on nor off. Write one of "
        f"{list(_TRUTHY)} or {list(_FALSY)}. A coverage gate may not guess "
        "what an unrecognised spelling meant: guessing 'on' turns a typo into "
        "a refusal, and guessing 'off' turns one into an unmeasured green "
        "(tessera#152)."
    )


#: Recorded on an item whose call made the CUDA allocator hand out memory.
#: ``user_properties`` rather than an attribute because xdist serialises it,
#: so a worker's observation survives into the controller's aggregate.
_CUDA_EXECUTED = "tessera_cuda_executed"


def _cuda_allocations():
    """How many device allocations this process has made, or ``None``.

    Read from ``sys.modules`` rather than imported: this file must stay
    torch-free for the ``pure`` job, and a test that has not imported torch
    cannot have allocated on a device either.  ``None`` means the question is
    unanswerable here and no claim is made from it.
    """

    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_initialized():
            return 0
        return int(torch.cuda.memory_stats().get("allocation.all.allocated", 0))
    except Exception:  # noqa: BLE001 -- an unanswerable probe is not an answer
        return None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Did this test actually run on the device?

    Declaring it with a marker would be a roster of forty-odd files that
    passes on the day it is wrong (AGENTS.md rule 3), and classifying skip
    reasons by regex is the guess this file already refuses to make.  The
    allocator's own counter is neither: a test that ran on the device
    allocated on it.  The converse does not hold -- a test that only reads
    device properties allocates nothing -- so this UNDERCOUNTS, deliberately,
    because the gate that reads it refuses at zero and an undercount can only
    make that refusal easier to trip, never harder to.
    """

    before = _cuda_allocations()
    try:
        return (yield)
    finally:
        after = _cuda_allocations()
        if before is not None and after is not None and after > before:
            item.user_properties.append((_CUDA_EXECUTED, True))


def _cuda_executed(terminalreporter) -> int:
    """Tests whose call allocated on the device, over the whole run."""

    total = 0
    for outcome in ("passed", "failed"):
        for report in terminalreporter.stats.get(outcome, []):
            if getattr(report, "when", None) != "call":
                continue
            properties = getattr(report, "user_properties", ()) or ()
            if any(key == _CUDA_EXECUTED for key, _ in properties):
                total += 1
    return total


def _box_artifact_skips(reasons) -> dict:
    """The skips that mean "this box has no copy of the evidence".

    Not a pattern over prose: ``box_artifacts`` writes every one of these
    sentences and stamps its own prefix on them, so this reads a declaration.
    """

    prefix = box_artifacts.ABSENT + ":"
    return {reason: count for reason, count in reasons.items()
            if reason.startswith(prefix)}


def _cuda_device() -> tuple[bool, str]:
    """(has a device, one line naming it or naming its absence).

    Imported here rather than at module scope, and tolerant of every way this
    can go wrong: an interpreter with no torch, a torch built without CUDA, and
    a driver that raises on probe are three different absences and all three
    mean the same thing for coverage.
    """
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 -- any import failure is an absence
        return False, f"no torch in this interpreter ({type(exc).__name__})"
    try:
        if not torch.cuda.is_available():
            return False, f"torch {torch.__version__} reports no CUDA device"
        count = torch.cuda.device_count()
        return True, (
            f"torch {torch.__version__}, {count} CUDA device(s), "
            f"device 0 = {torch.cuda.get_device_name(0)}"
        )
    except Exception as exc:  # noqa: BLE001 -- a probe that raises is an absence
        return False, f"CUDA probe raised {type(exc).__name__}: {exc}"


def pytest_report_header(config):
    present, detail = _cuda_device()
    lines = [f"tessera surface: {'CUDA' if present else 'NO CUDA'} -- {detail}"]
    if collect_ignore:
        lines.append(
            f"tessera surface: {len(collect_ignore)} test module(s) not "
            "collected at all (their third-party imports are missing)"
        )
    return lines


def pytest_sessionstart(session):
    # xdist workers inherit the controller's verdict; one refusal, not one per
    # worker.
    if hasattr(session.config, "workerinput"):
        return
    if not _strict_cuda(session.config):
        return
    present, detail = _cuda_device()
    if present:
        return
    raise pytest.UsageError(
        "--strict-cuda: " + detail + ". This interpreter cannot exercise the "
        "CUDA-gated test surface, so a pass here would certify a population it "
        "never ran. Refusing rather than skipping (tessera#112)."
    )


def _skip_reason(report) -> str:
    """The reason text pytest recorded, verbatim."""

    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        reason = str(longrepr[2])
    else:
        reason = str(longrepr)
    # pytest prefixes a bare ``Skipped: ``; the rest is the author's sentence.
    return reason[len("Skipped: "):] if reason.startswith("Skipped: ") else reason


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    from collections import Counter

    present, detail = _cuda_device()
    write = terminalreporter.write_line

    skipped = terminalreporter.stats.get("skipped", [])
    write("")
    write(f"tessera surface: {'CUDA' if present else 'NO CUDA'} -- {detail}")
    write(
        f"tessera surface: {len(skipped)} test(s) skipped, "
        f"{len(collect_ignore)} module(s) not collected"
    )
    if not present:
        write(
            "tessera surface: this run did not exercise the CUDA-gated "
            "surface. Its pass count is not coverage of it."
        )
    counts = Counter(_skip_reason(report) for report in skipped)
    if skipped:
        write("tessera surface: skip reasons, verbatim --")
        for reason, count in counts.most_common():
            write(f"    {count:5d}  {reason}")
    if collect_ignore:
        import textwrap

        write("tessera surface: modules not collected --")
        for line in textwrap.wrap(" ".join(collect_ignore), width=72):
            write(f"    {line}")

    executed = _cuda_executed(terminalreporter)
    gated = _box_artifact_skips(counts)
    write(f"tessera surface: {executed} test(s) allocated on the device")
    if gated:
        write("tessera surface: skipped for evidence this box does not hold --")
        for reason, count in sorted(gated.items()):
            write(f"    {count:5d}  {reason}")

    destination = config.getoption("--surface-json")
    if destination:
        _write_surface_json(Path(destination), config, terminalreporter,
                            present, detail, counts, executed, gated)


def _coverage_refusals(executed: int, gated: dict) -> list:
    """Why this run may not claim the CUDA-gated surface, if it may not."""

    problems = []
    if executed == 0:
        problems.append(
            "no test in this run allocated on the CUDA device. A device this "
            "session never used is not coverage of the surface it was "
            "submitted to cover -- every other check here is satisfied by a "
            "run that collected the suite and skipped all of it (tessera#152)."
        )
    for reason, count in sorted(gated.items()):
        problems.append(
            f"{count} test(s) skipped for evidence this box does not hold: "
            f"{reason}. Those are the bit-exactness gates of tessera#146; an "
            "arm that skips them is not covering the surface it claims. Give "
            "this box the root named above, or do not run with --strict-cuda."
        )
    return problems


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Legs 2 and 3 of the gate, read off the population above.

    They can only be evaluated once the run is over, so this is a refusal
    after the fact rather than before it -- the same verdict, one summary
    later.  Only the controller speaks, and only when nothing else already
    made the run red: a coverage refusal must not overwrite a test failure.
    """

    if hasattr(session.config, "workerinput"):
        return
    if not _strict_cuda(session.config):
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    from collections import Counter

    counts = Counter(_skip_reason(report)
                     for report in reporter.stats.get("skipped", []))
    problems = _coverage_refusals(_cuda_executed(reporter),
                                  _box_artifact_skips(counts))
    if not problems:
        return
    reporter.write_line("")
    for problem in problems:
        reporter.write_line("--strict-cuda: " + problem)
    reporter.write_line(
        "--strict-cuda: refusing rather than reporting coverage this run "
        "does not have.")
    if exitstatus == 0:
        session.exitstatus = int(pytest.ExitCode.USAGE_ERROR)


def _measured_commit():
    """Which tree did this population come from?

    A population without its commit is half a receipt.  The arms of a merge
    run are separate processes on separate boxes, and nothing makes them
    start at the same instant: an x86 arm can finish, a GPU slot can free an
    hour later, and the clone the pool action runs in can have moved between
    the two.  A reader that asks the checkout at *read* time gets one commit
    and stamps it on both -- which is exactly the mistake this whole file
    exists to stop, one level up.  So the run says which tree it ran, at the
    moment it ran it, in its own receipt.

    ``None`` when the answer is not knowable (no git, no repository, a source
    tarball).  An absent commit is honest; a guessed one is not.
    """

    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent),
             "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    commit = out.stdout.strip()
    return commit or None


def _worker_id(config):
    """Which xdist worker is speaking, or ``None`` for a whole run.

    ``pytest_terminal_summary`` runs in every xdist worker as well as in the
    controller, and a worker's ``stats`` hold that worker's SHARE of the run.
    ``workerinput`` is the same attribute ``pytest_sessionstart`` above already
    uses to tell the two apart, and it is the ONLY answer read here: xdist
    sets it on this process's own config, so it cannot describe another one.

    ``PYTEST_XDIST_WORKER`` was kept beside it as "a second answer to the same
    question" and is not one -- it is inherited.  A pytest launched from
    inside a worker sees ``gw1`` in its environment and has no ``workerinput``
    of its own, so it filed its whole run as that worker's share: the
    population never appeared on the path it was asked for.  Five tests in
    ``tests/test_cuda_surface.py`` failed exactly that way in the ``-n 8`` x86
    population of ``82f0047`` (1536 passed / 5 failed / 503 skipped,
    dl380g10), and passed when the same commit ran serially.  The fix that
    stopped a share being read as a population had introduced the converse.

    Bracketed on the same box, same ``-n 8``: ``dee1aa9`` -- before this
    function existed -- was 1406 / 0 / 499; ``d11dc01`` and ``82f0047``, with
    the environment fallback, are 1406 / **5** / 499 and 1536 / **5** / 503;
    and with it gone the two files that hold all five are 36 passed / 0 failed
    under ``-n 8`` and under ``-n 4``.
    """

    workerinput = getattr(config, "workerinput", None)
    if workerinput:
        return str(workerinput.get("workerid") or "") or None
    return None


def _worker_count(config):
    """How many workers the controller fanned out to, when it can say."""

    count = getattr(config.option, "numprocesses", None)
    return count if isinstance(count, int) else None


def _write_surface_json(path, config, terminalreporter, present, detail,
                        reasons, executed=0, gated=None):
    """The same population, as a table rather than as prose.

    A receipt that scrapes "1404 passed / 487 skipped" out of a terminal is
    reading a rendering; this is the run stating its own population, which is
    what a downstream gate or a stored receipt should read.

    **A worker's share is not a population, and never lands on the population's
    path.**  Under ``-n N`` every worker reaches this function with its own
    slice of the run, and the first spelling of this file gave all of them the
    same filename: one ``-n 8`` run wrote eight shards over each other and then
    the controller's aggregate, and ``_keep_any_previous`` renamed the eight to
    ``superseded-<mtime>`` -- a name that means "a retry wrote over this".  A
    reader of ``20260904T040432`` therefore sees eight retries of a run that
    ran once, and would have read a shard as the arm's population had the run
    been killed before the controller wrote.  Both are false provenance, and
    false provenance in the artefact built to make a suite result trustworthy
    is worse than the gap it closes.

    So the worker writes ``surface.<arm>.<workerid>.json`` and says ``role:
    worker-share``; only the controller -- or a serial run, which is its own
    controller -- writes the plain name, and only that file says ``role:
    population``.  The two can no longer be confused by a reader, by a resume,
    or by a kill.
    """

    import json
    from tessera._dev.suite_source import measured_source

    worker = _worker_id(config)
    if worker:
        path = path.with_name(f"{path.stem}.{worker}{path.suffix}")

    stats = terminalreporter.stats
    payload = {
        # v2: a file at the population's own path is now GUARANTEED to be a
        # whole run.  Under v1 it could be one worker's share, so the version
        # is what tells a reader which guarantee this file carries.
        # v3: the population now states what it EXECUTED on the device, not
        # only which device it saw.  A v2 reader may read a v3 file -- every
        # field it knows is unchanged and in place -- so receipts stay
        # backward-readable; what it cannot do is answer the coverage
        # question, and merge_suite says so rather than assuming green.
        "schema": "tessera.test_surface.v3",
        "role": "worker-share" if worker else "population",
        "worker_id": worker,
        "xdist_workers": _worker_count(config),
        "commit": _measured_commit(),
        "source_identity": measured_source(Path(__file__).resolve().parents[1]),
        "cuda": present,
        "device": detail,
        "strict_cuda": _strict_cuda(config),
        "counts": {
            outcome: len(stats.get(outcome, []))
            for outcome in ("passed", "failed", "error", "skipped",
                            "xfailed", "xpassed")
        },
        "skip_reasons": dict(sorted(reasons.items(),
                                    key=lambda kv: (-kv[1], kv[0]))),
        "cuda_surface": {
            "executed": executed,
            "measured_by": ("tests during whose call torch's CUDA allocator "
                            "recorded a new allocation"),
            "is_a_floor": True,
            "box_artifact_skips": dict(sorted((gated or {}).items())),
        },
        "not_collected": list(collect_ignore),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    superseded = _keep_any_previous(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    what = "worker share" if worker else "population"
    terminalreporter.write_line(f"tessera surface: {what} written to {path}")
    if superseded:
        terminalreporter.write_line(
            f"tessera surface: a previous {what} was here; kept at {superseded}")


def _keep_any_previous(path):
    """Move an earlier file aside instead of writing over it.

    ``superseded-<mtime>`` means exactly one thing: **an earlier run wrote this
    same path, and this run is writing it again.**  The path is fixed per arm
    (and, since the shard fix above, per worker), so a pool retry -- a lease
    that expired, a worker that died, an action on its second attempt -- runs
    the suite again and lands on the same filename.  That happened: the
    population at ``20260904T025044/surface.x86.json`` (1389 passed / 1 failed,
    on e61974c) was overwritten at 07:28:49Z by a retry that ran a different
    tree and reported 1388/2.  The first measurement survived only because
    someone happened to have copied it.

    A measurement is evidence, and the second one does not disprove the first
    -- they are two populations, possibly of two trees.  Keep both.  The reader
    is unchanged: it still opens the plain name, which is always the newest.

    Read the eight ``surface.x86.superseded-*.json`` files in receipt
    ``20260904T040432`` against that sentence and they contradict it: that run
    ran ONCE, under ``-n 8``, and the eight are its workers' shares, which the
    first spelling of ``_write_surface_json`` aimed at the population's path.
    They are shards misfiled as retries.  The bytes are left where they are --
    a measurement is not edited after the fact -- and this docstring is the
    place that says what they actually are.
    """

    import time

    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(path.stat().st_mtime))
    kept = path.with_name(f"{path.stem}.superseded-{stamp}{path.suffix}")
    # Two retries within one second, or a re-run of an already-kept file: do
    # not clobber the thing we are here to preserve.
    index = 0
    while kept.exists():
        index += 1
        kept = path.with_name(
            f"{path.stem}.superseded-{stamp}-{index}{path.suffix}")
    path.rename(kept)
    return kept
