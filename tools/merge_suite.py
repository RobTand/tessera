#!/usr/bin/env python3
"""Run the suite on the merge result, on both device populations, once.

tessera#112: master was red on three CUDA-gated tests while GitHub Actions,
the x86 pool suite and a local CPU run all read green.  Not one of those three
signals could have seen the failure, and none of them said so.  The CUDA-gated
surface is exercised by nothing automatic.

*How big is that surface?*  **467 tests**, on ``d11dc01``, measured rather
than estimated.  The device-less x86 arm's own skip histogram on that commit
names 467 skips whose verbatim reason is a CUDA or GPU path; the GPU arm on
the same commit skipped 13, none of them device-shaped.  The issue's "450-480"
was a subtraction between two runs of two different commits on two different
boxes, which is an estimate; this is one commit, two arms, each arm publishing
its own population.

Read the two arms with their run modes attached, because they are not
otherwise comparable.  The GPU arm ran serially -- the cu130 venv has no xdist
-- under ``--strict-cuda`` on an NVIDIA GB10, and published 1910 passed / 0
failed / 13 skipped.  The x86 arm ran ``-n 8`` on dl380g10 and published 1406
passed / **5 failed** / 499 skipped.  All five failures are in
``tests/test_cuda_surface.py`` -- exactly that file's five tests that read back
a surface a child pytest wrote -- and they failed because
``PYTEST_XDIST_WORKER`` is inherited by a nested run (see
``tests/conftest.py::_worker_id``).  Green against red here is a run-mode
difference, not a device one.  The arms also collected 1923 and 1910 items;
that 13-item gap is not chased here and is not the surface figure.

Two facts shape what this tool is.

*A suite result is meaningless without its population.*  "1404 passed / 487
skipped" and "1381 passed / 497 skipped" are not two measurements of the same
thing; they are two different populations, and quoting either alone is how a
red commit reads green.  So this submits **both** arms and writes **one**
receipt holding both, side by side.  There is no way to quote one from it
without the other being on the same screen.

*The GPU arm must not be able to lie.*  It runs with ``--strict-cuda``, which
refuses a session that has no CUDA device instead of skipping the surface it
was submitted to cover.  A placement that lands the GPU arm on a box without a
device now fails loudly; before, it would have returned a green tick.

*The receipt must outlive the terminal.*  ``--record`` appends one row per arm
to ``docs/status/suite-populations.md``, which is where a reader of the repo
looks; and if the submitting session dies while the pool carries on -- which is
how the first real run went -- ``--resume <receipt dir>`` rebuilds the receipt
from the populations the runs published, and takes the exit status from the
pool's own outcome record for the action that wrote that population -- shown
as ``0 (pool)``, so a status nobody here watched is not mistaken for one this
process saw.  "Wrote" is the action's **effective** pytest ``--surface-json``
argument, read out of a command shape this tool seals (``pytest``, ``python -m
pytest``, or those under ``tools/suite_deadline.py ... --``; any other program
is refused by name), and bound to the population by evidence the population
and the record each carry: the sealed snapshot's commit is the population's,
the population's verified source stamp names that action and digests its
request, and the recorded attempt's own stdout says it published this path,
with the digest of the bytes it wrote and with these counts.  That
publication line has one home -- ``tessera._dev.surface_publication``, which
``tests/conftest.py`` writes it from and this file reads it with -- because a
sentence spelled once in the producer and once in the consumer is a contract
neither knows it is in, and it was one until #331.

A command that merely mentions the path, one that
overrides the option later, ``echo pytest``, a request with no snapshot, a
population with no stamp, a lease-lost record, and a retry that never
published are none of them this file's producer, and absent evidence is
absent, not agreement.  Where no single record is bound -- still in flight,
refused, or two records of the producer -- the row stays ``not observed`` and
nothing is borrowed: published failures prove red, their absence does not
prove green.

Everything about scheduling is PrismaBuild's: this composes ``pbrun``
invocations and reads what they return.  It never runs a suite itself, never
sshes anywhere, and a refused placement comes back as the refusal it is.
``--cpus N`` is spent as well as declared: above 1 it becomes pytest's ``-n``,
so the reservation and the command cannot disagree.  It is clamped **per
arm**: the GPU arm runs serially whatever is asked, because its workers would
share one device and its interpreter has no xdist.  Without that clamp the two
arms could only be submitted together at ``--cpus 1`` -- every ``-n`` run on
this branch was a lone ``--arm x86`` submission for exactly that reason, which
is the half-a-result the ledger header warns about, produced by the tool that
warns about it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tessera._dev.suite_deadline import positive_seconds as _positive_seconds  # noqa: E402
from tessera._dev.surface_publication import (  # noqa: E402
    POPULATION,
    digest_bytes,
    published_digests,
)

PBRUN = Path("/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py")
SHARED_ROOT = Path("/mnt/shared")
#: Surface reports are written OUTSIDE the checkout on purpose.  pbrun binds a
#: command's identity to the checkout's git delta, so an artefact dropped into
#: the tree moves the action key of every later submission from it -- a cache
#: miss dressed up as a different action.
DEFAULT_RECEIPT_ROOT = SHARED_ROOT / "tessera-suite-receipts"

# Each arm has one reserved CPU per pytest process: the GPU arm is serial,
# while xdist spends the x86 reservation as one process per core. Native math
# threads and per-process extension builds must not multiply that reservation.
PROCESS_THREAD_LIMITS = dict.fromkeys(
    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MAX_JOBS"), "1"
)
# Cleanup backstop, matching the deployed PB worker's TERM grace. The inner
# command needs its own deadline: deployed pbrun parses but ignores --timeout-s.
TIMEOUT_KILL_AFTER_S = 5.0

#: Where the pool publishes what it did.  A finished action's outcome record
#: carries the exit status the worker actually saw; the CAS request beside it
#: carries the command that action ran.  Reading those two is how a resumed
#: receipt can state an exit status instead of declining to -- derived from a
#: table PrismaBuild publishes, never inferred from the population's contents.
POOL_ROOT = SHARED_ROOT / "prismabuild-fleet"
POOL_QUEUE = POOL_ROOT / "pb-queue"
POOL_CAS_REQUESTS = POOL_ROOT / "cas" / "requests"

#: The two arms, and why each is spelled the way it is.  The interpreter is
#: named rather than inherited: a pool action runs in a sealed environment, so
#: ``sys.executable`` here is a fact about the submitting box, not the target.
#:
#: ``fans_out`` is a property of the ARM, not of the submission.  ``--cpus`` is
#: one number for a run and the two arms cannot spend it the same way, so an
#: arm that must run serially says so here and clamps its own share.
ARMS = {
    "gpu": {
        "why": "the CUDA-gated surface; nothing else exercises it",
        "python": "/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python",
        "pbrun_flags": ["--gpu", "--exclusive"],
        "strict_cuda": True,
        "fans_out": False,
        "serial_because": (
            "this arm runs serially whatever --cpus says: its workers would "
            "share one device, and pytest-xdist is absent from the CUDA venv's "
            "interpreter, so -n would abort the run on an unrecognised "
            "argument rather than fan it out"
        ),
    },
    "x86": {
        "why": "the device-less population: torch present, no CUDA device, "
               "which is what dl380g10 actually reports",
        "python": "/home/rob/venvs/pb-cpu/bin/python",
        "pbrun_flags": ["--tag", "x86"],
        "strict_cuda": False,
        "needs_shared_checkout": True,
        "fans_out": True,
    },
}


def _arm_cpus(arm: dict, requested: int) -> int:
    """The cores THIS arm will spend, which is not always the cores asked for.

    ``--cpus`` is one number and a run has two arms, and until this existed
    both got it: ``--cpus 8`` composed ``-n 8`` for the GPU arm too, whose
    interpreter has no xdist and aborts on the argument.  So the tool could
    submit both arms together only at ``--cpus 1``, and every ``-n`` run on
    this branch was therefore a lone ``--arm x86`` submission -- which is
    exactly the half-a-result the ledger's own header warns about, produced by
    the tool that warns about it.  The five failures on ``82f0047`` were an
    ``-n``-only defect, so "run both arms serially to keep them in one
    invocation" would have been a merge check blind to the class of bug this
    branch's own worst regression belonged to.

    Clamping here rather than at the call sites is deliberate: the number that
    reaches pytest as ``-n`` and the number declared to the pool as ``--cpus``
    must be the same one, and one function is how they cannot drift.
    """

    if arm.get("fans_out", True):
        return max(1, requested)
    return 1


# A checkout on /mnt/shared is on NFS, and `git status --porcelain` there
# while a suite is running in it took longer than the 30 s this used to allow.
GIT_PROBE_TIMEOUT_S = 180


def _git(checkout: Path, *args: str) -> str | None:
    """Git's answer, or ``None`` when git did not give one.

    ``None`` is not ``""``.  An empty ``git status --porcelain`` means the tree
    is clean; a probe that timed out or errored means nothing was established,
    and the two used to collapse into ``working_tree_dirty: false`` -- a
    receipt claiming a clean tree it never looked at.

    The timeout also used to propagate.  ``--resume`` of a real population died
    on ``TimeoutExpired`` from this call, on a checkout that had a suite
    running in it, and wrote no receipt and no ledger row: a measurement lost
    to a probe that was never load-bearing.  A receipt is worth more than one
    of its provenance fields.
    """

    try:
        out = subprocess.run(["git", "-C", str(checkout), *args],
                             capture_output=True, text=True,
                             timeout=GIT_PROBE_TIMEOUT_S)
    except (subprocess.TimeoutExpired, OSError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _population_of(checkout: Path) -> dict:
    """What tree this receipt is about, stated so it cannot be mistaken.

    A receipt from a feature branch is not a merge receipt.  Recording the head
    of ``master`` alongside the commit under test makes that difference
    readable rather than assumed.
    """

    head = _git(checkout, "rev-parse", "HEAD")
    # A clone made for a pool run has no local ``master`` -- only
    # ``origin/master`` -- and a bare ``rev-parse master`` there returns
    # nothing, which would silently make ``is_master_head`` false for the very
    # commit that IS master.  Say which ref answered, so a null comparison
    # reads as "not established" and never as "not master".
    master, master_ref = "", ""
    for ref in ("master", "origin/master", "refs/remotes/origin/master"):
        master = _git(checkout, "rev-parse", ref)
        if master:
            master_ref = ref
            break
    dirty = _git(checkout, "status", "--porcelain")
    return {
        "checkout": str(checkout),
        "commit": head,
        "describe": _git(checkout, "log", "-1", "--pretty=%s"),
        "master_ref_used": master_ref or "none resolved",
        "master_head_at_submit": master or None,
        "is_master_head": (head == master) if (head and master) else None,
        # None where the probe did not answer: "not established", never "clean".
        "working_tree_dirty": (None if dirty is None else bool(dirty)),
    }


def _command(arm: dict, surface_json: Path, extra: list[str],
             cpus: int = 1) -> list[str]:
    """The command, with the core count it is submitted under actually used.

    ``--cpus`` is a reservation: the pool holds that many cores on the chosen
    box for the life of the action.  A serial pytest submitted under ``--cpus
    8`` idles seven of them, which is the over-declaration the pool exists to
    prevent -- it is how a box ends up oversubscribed on paper and idle in
    fact, and the mirror of the under-declaration that swapped sparklina on
    2026-09-03.  So the declaration and the command are derived from one
    number rather than chosen separately.

    Serial is the default because parallelism is not free here and not always
    available.  ``-n`` needs ``pytest-xdist`` in the TARGET venv, which is a
    fact about another box: the CUDA venv on sparky inherits the system
    interpreter, which has pytest 9.0.3 and no xdist, so a hardcoded ``-n``
    would abort that arm on an unrecognised argument.  And on the GPU arm the
    workers share one device, so fanning out is a memory risk, not a speedup.
    An operator who knows the target venv passes ``--cpus N`` and gets both
    halves at once.

    Which is why the number is passed through ``_arm_cpus`` here rather than
    used as given: an arm that cannot fan out clamps to serial, so one
    ``--cpus 8`` can submit an ``-n 8`` x86 arm and a serial GPU arm in the
    same run.  Before that, ``--cpus 8`` composed ``-n 8`` for both.
    """

    cpus = _arm_cpus(arm, cpus)
    command = [arm["python"], "-m", "pytest", "tests", "-q",
               "-p", "no:cacheprovider",
               "--surface-json", str(surface_json)]
    if cpus > 1:
        # loadfile, not the default: a module's tests share fixtures and, on
        # the GPU arm, device state, and splitting one across workers is how a
        # suite goes flaky in a way no population report would explain.
        command += ["-n", str(cpus), "--dist", "loadfile"]
    if arm["strict_cuda"]:
        command.append("--strict-cuda")
    return command + extra


def _timed_command(command: list[str], timeout_s: float) -> list[str]:
    """Per-attempt managed-process-group deadline, not a queue/descendant cap."""
    deadline = _positive_seconds(timeout_s)
    grace = _positive_seconds(TIMEOUT_KILL_AFTER_S)
    # Use the arm's named interpreter, not a host's unverified timeout binary.
    return [command[0], "tools/suite_deadline.py", "--timeout-s", str(deadline),
            "--kill-after-s", str(grace), "--", *command]


def _submit(name: str, arm: dict, args, receipt_dir: Path) -> dict:
    surface_json = receipt_dir / f"surface.{name}.json"
    # The reservation is this ARM's, not the run's: an arm clamped to serial
    # must not hold the cores it was told to spend, or the ledger says the box
    # is busy while seven of its cores idle.
    cpus = _arm_cpus(arm, args.cpus)
    command = _timed_command(_command(arm, surface_json, args.pytest_arg, cpus),
                             args.timeout_s)
    flags = list(arm["pbrun_flags"])
    if name == "gpu" and args.gpu_tag:
        flags += ["--tag", args.gpu_tag]
    invocation = [
        sys.executable, str(PBRUN),
        *flags,
        "--cpus", str(cpus),
        # --exclusive derives full GPU capacity from the selected worker's
        # live offer through pbrun.exclusive_gpu_demand, never from a slot guess.
        "--demand", f"mem_gb={args.mem_gb}",
        "--cwd", str(args.checkout),
        "--timeout-s", str(args.timeout_s),
        "--wait-s", str(args.wait_s),
        *[part for key, value in PROCESS_THREAD_LIMITS.items()
          for part in ("--env", f"{key}={value}")],
        "--", *command,
    ]
    record = {
        "arm": name,
        "why": arm["why"],
        "python": arm["python"],
        # What this arm was SUBMITTED to cover, carried on the record so the
        # verdict can check the population against it without reaching back
        # into a table the receipt does not contain.
        "requires_cuda": bool(arm["strict_cuda"]),
        # The run mode, on the record, because a pass count of one arm read
        # against the other's differs by more than the device: the five
        # failures on ``82f0047`` were an ``-n``-only defect.
        "cpus_requested": args.cpus,
        "cpus_used": cpus,
        "process_thread_limits": dict(PROCESS_THREAD_LIMITS),
        "timeout_s": args.timeout_s,
        "timeout_kill_after_s": TIMEOUT_KILL_AFTER_S,
        "timeout_scope": "per attempt; excludes queue time, retries and detached descendants",
        "pbrun": " ".join(shlex.quote(part) for part in invocation),
    }
    if cpus != args.cpus:
        record["cpus_note"] = arm.get("serial_because", "")
    if args.dry_run:
        record["status"] = "not submitted (--dry-run)"
        return record

    started = time.monotonic()
    proc = subprocess.run(invocation, capture_output=True, text=True)
    record["returncode"] = proc.returncode
    record["elapsed_s"] = round(time.monotonic() - started, 1)
    record["stderr_tail"] = proc.stderr.strip().splitlines()[-6:]
    record["stdout_tail"] = proc.stdout.strip().splitlines()[-6:]
    _attach_surface(record, surface_json)
    return record


#: What a surface file has to say about itself before this tool will read it
#: as an arm's population.  ``tessera.test_surface.v1`` had no such field, and
#: a v1 file at this path could be one xdist worker's share: under ``-n 8``
#: every worker wrote the arm's canonical path, so eight shards landed there
#: before the controller's aggregate did.  v2 files say which they are.
_SHARD_ROLE = "worker-share"


def _attach_surface(record: dict, surface_json: Path) -> None:
    """Put this arm's published population on the record, or say it has none.

    A worker's share is refused rather than read.  It is a real measurement of
    a real slice, and it is not the arm's population -- reading one as the
    population is how ``docs/status/suite-populations.md`` would come to carry
    206 passed / 108 skipped as an x86 suite result.  An absent measurement is
    honest; a partial one wearing the whole one's name is not.
    """

    if surface_json.exists():
        published = json.loads(surface_json.read_text())
        role = published.get("role")
        if role == _SHARD_ROLE:
            record["surface"] = None
            record["shard_at_population_path"] = str(surface_json)
            record["no_surface_means"] = (
                f"the file at {surface_json.name} says it is one xdist "
                f"worker's share ({published.get('worker_id')}), not this "
                "arm's population. A share is a slice of the run; reporting "
                "it as the population would put a fraction of a suite in the "
                "ledger as the whole of it. This is an absent measurement."
            )
            return
        record["surface"] = published
        record["surface_path"] = str(surface_json)
        # Pre-v2 files cannot answer, and the honest reading of one is that the
        # question is open -- not that the answer is "population".
        record["surface_role"] = role or (
            "unstated: written before the role field existed, so under -n it "
            "could be a worker's share rather than the whole run")
        # WHEN the population was measured, which on a resumed receipt is not
        # when the receipt was assembled -- the run may have finished hours
        # before anyone came back for it.  The ledger dates the measurement,
        # not the bookkeeping.
        record["measured_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(surface_json.stat().st_mtime))
        return
    # The distinction that matters: a suite that ran and failed published a
    # surface; one that was never placed, or was refused before collection,
    # did not.  Saying which is the whole value of a receipt.
    record["surface"] = None
    record["no_surface_means"] = (
        "the suite published no population: it was refused, never placed, "
        "or died before the terminal summary. This is not a pass and not a "
        "fail; it is an absent measurement."
    )


#: Interpreter basenames the sealed commands run pytest under.  The arms name
#: their venv's ``python``; ``python3`` and ``python3.12`` are the same program
#: under the names a venv gives it.  Anything else is not an interpreter this
#: tool recognises, so a command starting with it is not a pytest invocation
#: this tool parses.
_PYTHON_NAME = re.compile(r"^python(3(\.\d+)?)?$")

#: The deadline wrapper ``_timed_command`` composes: its own options up to
#: ``--``, then the command it supervises.
_DEADLINE_WRAPPER = "suite_deadline.py"


def _pytest_argv(command: list) -> tuple[list[str] | None, str | None]:
    """The pytest argv inside this sealed command, or why there is none.

    The only shapes read are the ones this tool seals, and their unwrapped
    core: ``pytest ...``, ``<python> -m pytest ...``, and ``<python>
    tools/suite_deadline.py <options> -- <either of those>``.  The wrapper is
    parsed by name and recursively -- what follows its ``--`` is the command
    it supervised, and that is what has to be the pytest invocation.  Every
    other program is refused with its name: ``echo``, ``cat``, a ``timeout``
    binary this tool does not trust (see ``_timed_command``), a wrapper sealed
    without its ``--``.  A token spelled ``pytest`` somewhere in argv is not a
    pytest invocation -- ``echo pytest --surface-json <path>`` exits 0 having
    written nothing, and #218's word-search accepted it as the producer
    (#294).  A program string passed to ``-c`` is deliberately not read:
    guessing at a shell or Python fragment is how a reader starts inventing
    producers.
    """

    parts = [str(part) for part in command]
    if not parts:
        return None, "the request holds no command"
    program = Path(parts[0]).name
    if program in ("pytest", "py.test"):
        return parts, None
    if not _PYTHON_NAME.match(program):
        return None, f"`{program}` is not a pytest invocation this tool parses"
    if len(parts) >= 3 and parts[1] == "-m" and parts[2] == "pytest":
        return parts, None
    if len(parts) >= 2 and Path(parts[1]).name == _DEADLINE_WRAPPER:
        if "--" not in parts[2:]:
            return None, (f"`{_DEADLINE_WRAPPER}` was sealed without the `--` "
                          "that separates the command it supervises")
        return _pytest_argv(parts[parts.index("--", 2) + 1:])
    ran = " ".join([program, *parts[1:3]])
    return None, f"`{ran}` is not a pytest invocation this tool parses"


def _effective_surface_json(command: list) -> tuple[str | None, str | None]:
    """The population path this command actually writes, or why none is read.

    ``--surface-json`` is an ordinary ``store`` option, so pytest keeps the
    LAST one; a command naming this path and then overriding it writes
    somewhere else entirely.  And a command that merely mentions the path --
    ``cat``, an inspection, a copy -- writes nothing at all.  Membership of
    the path string in argv answered neither question, which is how a
    successful reader could supply a suite's exit status (#218).

    Returns ``(path, None)`` for a pytest command that names one, ``(None,
    None)`` for a pytest command that does not, and ``(None, reason)`` for a
    command that is not a pytest invocation this tool parses.
    """

    argv, refusal = _pytest_argv(command)
    if argv is None:
        return None, refusal
    destination = None
    for index, part in enumerate(argv):
        if part == "--surface-json":
            destination = argv[index + 1] if index + 1 < len(argv) else None
        elif part.startswith("--surface-json="):
            destination = part.split("=", 1)[1]
    return destination, None


#: The population's ``counts`` buckets, and the words pytest's summary line
#: spells the same ``terminalreporter.stats`` buckets with.  ``warnings`` and
#: ``deselected`` are on the line and not in the population, so they are not
#: compared.
_COMPARED_COUNTS = ("passed", "failed", "error", "skipped", "xfailed", "xpassed")
_SUMMARY_WORDS = {"passed": "passed", "failed": "failed", "error": "error",
                  "errors": "error", "skipped": "skipped",
                  "xfailed": "xfailed", "xpassed": "xpassed"}
_SUMMARY_TERM = re.compile(r"^(\d+) ([a-z]+)$")
_SUMMARY_TAIL = re.compile(r" in \d+\.\d+s( \(\d+:\d\d:\d\d\))?$")


def _summary_counts(stdout: str) -> tuple[dict | None, str | None]:
    """pytest's terminal summary counts out of an attempt's captured stdout.

    The last line of the form ``N passed, M skipped, ... in 12.34s`` -- bare
    under ``-q``, ``=``-padded otherwise, ``no tests ran`` when nothing did.
    A bucket the line does not mention is zero, which is how pytest prints
    it.  Returns the counts and the line they were read from, or ``(None,
    None)`` where the stdout holds no such line: an attempt that died before
    its summary printed none.
    """

    for raw in reversed(stdout.splitlines()):
        line = raw.strip().strip("=").strip()
        if not _SUMMARY_TAIL.search(line):
            continue
        head = _SUMMARY_TAIL.sub("", line)
        counts = dict.fromkeys(_COMPARED_COUNTS, 0)
        if head == "no tests ran":
            return counts, line
        for term in head.split(", "):
            match = _SUMMARY_TERM.match(term)
            if not match:
                break
            bucket = _SUMMARY_WORDS.get(match.group(2))
            if bucket:
                counts[bucket] = int(match.group(1))
        else:
            return counts, line
    return None, None


#: The outcome statuses PrismaBuild's worker writes together with the attempt
#: they describe: ``finish()`` replaces ``detail`` and sets one of these.  The
#: lease reaper's ``lease_lost_max_attempts`` moves the record without
#: touching ``detail``, so that record's detail is whatever attempt last
#: finished -- an earlier one's, as observed on ``dbd91b92``.
_FINAL_ATTEMPT_STATUSES = ("executed", "failed")


def _binding_refusal(key: str, payload: dict, request_bytes: bytes,
                     outcome: dict, surface_json: Path,
                     population: dict | None) -> str | None:
    """Why this finished action's status is not this population's, or ``None``.

    A status is adopted only from the action the population itself names, for
    the attempt that itself says it published this file.  Each leg is a
    required piece of evidence and a record without it is refused for the
    absence: #218 compared identities only where both sides were present,
    which let a request with no snapshot and an outcome with no claim time
    pass every check by having nothing to check (#294).

    * the **tree**: the sealed request's ``checkout_snapshot.commit`` and the
      population's ``commit`` are both present and equal;
    * the **producer**: the population carries ``suite_source``'s verified
      stamp -- ``source_identity.excluded_metadata[].action_key`` -- naming
      this action, and the stamp's ``request_sha256`` is the digest of the
      request bytes read here, so the population and the pool agree on which
      request this is.  A population without the stamp (pre-stamp, or
      ``unknown`` source) names no producer and adopts no status; its counts
      are still read;
    * the **attempt**: the outcome's top-level status is one the worker
      writes with the final attempt's own ``detail``, and that attempt's
      captured stdout says it published a population at this path -- with the
      **digest** of the bytes it wrote, which must be the digest of the file
      that is there now -- and printed the counts this population holds.
      The pool requeues on any non-zero exit and a retry may die before
      publishing; what tells the attempts apart is what each said it wrote.
      #218 told them apart by a 600 s clock allowance between the file's
      mtime and the claim, which bounded how quickly a retry may follow -- a
      thing this tool does not know -- and admitted a retry five minutes
      behind.

      The publication line is not spelled here.  It is
      ``tessera._dev.surface_publication``'s, which ``tests/conftest.py``
      writes it from: a contract two modules each state is two rules that
      drift, and this one drifted silently by construction -- the only test
      of the join wrote the parser's own string back to it, so a reworded
      producer left every arm of every resumed receipt ``not observed`` with
      nothing red anywhere (#331).  The digest is why the join is now
      evidence and not a matched sentence: an attempt that overwrote this
      path with a population of *identical counts* is refused on bytes, which
      the counts leg below cannot see.  A line with no digest -- every run
      captured before #331, whose stdout is what a resume of an old receipt
      reads -- is bound on path and counts alone, because that is all such an
      attempt ever said.
    """

    snapshot = (payload.get("params") or {}).get("checkout_snapshot")
    stamped = snapshot.get("commit") if isinstance(snapshot, dict) else None
    measured = (population or {}).get("commit")
    if not stamped:
        return ("the request has no checkout_snapshot.commit, so the tree it "
                "ran is not established")
    if not measured:
        return ("the population names no commit, so the tree it measured is "
                "not established")
    if stamped != measured:
        return (f"the action ran snapshot commit {stamped[:12]} while the "
                f"population says it measured {measured[:12]}")

    identity = (population or {}).get("source_identity") or {}
    stamps = [entry for entry in (identity.get("excluded_metadata") or [])
              if isinstance(entry, dict) and entry.get("action_key")]
    if not stamps:
        return ("the population names no sealed action as its producer (no "
                "source_identity.excluded_metadata action_key), so no "
                "action's status is its status")
    named = [entry for entry in stamps if entry.get("action_key") == key]
    if not named:
        others = ", ".join(str(entry["action_key"])[:12] for entry in stamps)
        return (f"the population names action {others} as its producer, "
                f"not {key[:12]}")
    digest = hashlib.sha256(request_bytes).hexdigest()
    for entry in named:
        expected = entry.get("request_sha256")
        if expected != digest:
            return ("the population's request_sha256 "
                    f"{str(expected)[:12]} is not the digest of this request "
                    f"({digest[:12]}), so the request it verified against is "
                    "not the one read here")

    status = outcome.get("status")
    attempt = outcome.get("attempts")
    if status not in _FINAL_ATTEMPT_STATUSES:
        return (f"outcome status is {status!r}, whose detail is not the final "
                "attempt's own -- only executed/failed records carry the "
                "attempt they describe")
    stdout = (outcome.get("detail") or {}).get("stdout")
    if not isinstance(stdout, str):
        stdout = ""
    # The sentence is not restated here: `tessera._dev.surface_publication` is
    # the one home the conftest writes it from, so a reword moves both sides
    # or neither (#331).
    announced = published_digests(stdout, surface_json, POPULATION)
    if not announced:
        return (f"the attempt whose status is recorded (attempt {attempt}) "
                "never said it published a population at this path")
    digests = [value for value in announced if value]
    if digests:
        try:
            held = digest_bytes(Path(surface_json).read_bytes())
        except OSError as error:
            return (f"the population at this path cannot be read ({error}), "
                    f"so the digest attempt {attempt} published is not "
                    "checkable")
        if held not in digests:
            return (f"the attempt whose status is recorded (attempt {attempt}) "
                    f"published sha256 {digests[0][:12]} at this path and the "
                    f"file here is {held[:12]}, so the population at this path "
                    "is not the one that attempt wrote")
    counts, line = _summary_counts(stdout)
    if counts is None:
        return (f"the attempt whose status is recorded (attempt {attempt}) "
                "printed no terminal summary, so what it published is not "
                "established")
    held = (population or {}).get("counts") or {}
    differing = [bucket for bucket in _COMPARED_COUNTS
                 if counts[bucket] != int(held.get(bucket) or 0)]
    if differing:
        return (f"the attempt's summary line `{line}` disagrees with the "
                "population's counts ("
                + ", ".join(f"{held.get(b) or 0} {b}" for b in differing)
                + "), so the population at this path is not what that "
                "attempt published")
    return None


def _pool_actions_that_wrote(surface_json: Path,
                             population: dict | None = None) -> tuple[list, list]:
    """The finished pool actions whose command wrote this population.

    A resumed receipt used to have no exit status at all: the submitting
    process died, so nobody in this program watched the run.  But somebody
    did.  PrismaBuild's worker waits on the child, records the status it saw
    in the action's outcome record, and leaves that record in
    ``pb-queue/done`` or ``pb-queue/failed`` -- a machine-readable table the
    pool publishes about its own execution.  Reading it is not guessing, and
    it is not the forbidden move either: the number comes from the runtime
    that ran the thing, not from this program's opinion about what a clean
    summary implies.

    The join is the action's **effective** ``--surface-json`` argument, parsed
    out of a pytest command this tool recognises in its CAS request
    (``_pytest_argv``) -- not the presence of the path somewhere in argv,
    which made every reader of the file a candidate writer, and not the
    presence of the word ``pytest``, which made ``echo`` one.  A command of a
    shape this tool does not parse that nevertheless names the path is listed
    among the refusals with its program's name; one that does not name it is
    simply not about this population.

    A match is then bound to the population it claims to have written, or
    refused with the reason (``_binding_refusal``): the population must name
    that action as its producer and the recorded attempt must say it
    published this file.  Returns the bound producers and the refusals,
    because the caller must be able to tell one from several: two records of
    the producer (the pool disagreeing with itself about one action) have no
    single exit status, and picking one would be exactly the overclaim the
    rest of this file refuses.
    """

    wanted = str(Path(surface_json))
    found: list[dict] = []
    refused: list[str] = []
    for state in ("done", "failed"):
        folder = POOL_QUEUE / state
        if not folder.is_dir():
            continue
        for outcome_path in sorted(folder.glob("*.json")):
            key = outcome_path.stem
            request = POOL_CAS_REQUESTS / key[:2] / f"{key}.json"
            try:
                request_bytes = request.read_bytes()
                request_payload = json.loads(request_bytes)
            except (OSError, ValueError):
                continue
            if not isinstance(request_payload, dict):
                continue
            command = [str(part) for part in
                       (request_payload.get("params") or {}).get("command") or []]
            destination, unparsed = _effective_surface_json(command)
            if unparsed is not None:
                # Named only where it claimed this path: an unrelated build
                # never mentions it, and listing every action in the pool
                # would bury the one refusal a reader needs to see.
                if any(wanted in part for part in command):
                    refused.append(f"{key[:12]}: {unparsed}")
                continue
            if destination is None or str(Path(destination)) != wanted:
                continue
            try:
                outcome = json.loads(outcome_path.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(outcome, dict):
                continue
            refusal = _binding_refusal(key, request_payload, request_bytes,
                                       outcome, surface_json, population)
            if refusal:
                refused.append(f"{key[:12]}: {refusal}")
                continue
            detail = outcome.get("detail") or {}
            found.append({
                "action_key": key,
                "queue_state": state,
                "returncode": detail.get("returncode"),
                "worker_status": detail.get("status"),
                "attempts": outcome.get("attempts"),
                "host": outcome.get("claimed_host"),
                "elapsed_s": detail.get("elapsed_s"),
                # The command is what says whether that run fanned out, which
                # a resumed receipt otherwise cannot know: the process that
                # chose the mode is gone.
                "command": command,
            })
    return found, refused


def _cpus_of_command(command: list) -> int | None:
    """The run mode a recorded command actually ran under.

    ``-n N`` or nothing.  ``None`` where the command is unavailable -- a
    resumed row with no single pool action behind it -- because "not recorded"
    and "serial" are different answers and collapsing them is how a row comes
    to claim a mode nobody observed.
    """

    if not command:
        return None
    parts = [str(part) for part in command]
    if "-n" not in parts:
        return 1
    index = parts.index("-n")
    if index + 1 >= len(parts):
        return None
    try:
        return int(parts[index + 1])
    except ValueError:
        return None


def _attach_pool_exit_status(record: dict, surface_json: Path) -> None:
    """Let the pool answer the question this process cannot.

    Three outcomes, and the two that are not "one action" both leave the row
    unobserved rather than borrowing a number:

    * exactly one finished record is bound to this population -- the action it
      names as producer, for an attempt that says it published this file --
      and carries an integer status: that status is the exit status, and the
      row says where it came from;
    * none is -- the run may still be in flight, its record may have been
      reaped, the population may name no producer, or every record naming the
      path was refused (listed on the record, by reason) -- and nobody here
      saw an exit;
    * several are -- two records of the producing action.  There is no single
      status to report, so none is reported and the row says how many.
    """

    matches, refused = _pool_actions_that_wrote(surface_json,
                                                record.get("surface"))
    if refused:
        record["pool_actions_refused"] = refused
        record["exit_status_note"] += (
            " A finished pool action named this population's path and was not "
            "bound to it: " + "; ".join(refused) + ".")
    if len(matches) == 1 and isinstance(matches[0].get("returncode"), int):
        pool = matches[0]
        record["returncode"] = pool["returncode"]
        record["exit_status_observed"] = True
        record["exit_status_source"] = "pool"
        record["pool_action"] = pool
        record["exit_status_note"] = (
            "the submitting process did not survive to watch this run, so the "
            "exit status is the one PrismaBuild's worker recorded for action "
            f"{pool['action_key'][:12]} on {pool['host']} -- read from the "
            "pool's own outcome record, not inferred from the population."
        )
        # Same source, same reason: the run mode is in the command that action
        # ran, so a resumed row can name it instead of leaving it blank.
        mode = _cpus_of_command(pool.get("command") or [])
        if mode is not None:
            record["cpus_used"] = mode
        return
    if matches:
        record["pool_actions_matching"] = [m["action_key"] for m in matches]
        record["exit_status_note"] += (
            f" {len(matches)} finished pool actions wrote this population's "
            "path, so there is no single exit status to read and none is "
            "borrowed."
        )


def _resume(name: str, arm: dict, receipt_dir: Path) -> dict:
    """Assemble an arm's record from the population it already published.

    The receipt was the one part of this tool that only existed while the
    submitting terminal did.  When that process dies -- an interrupted session,
    a dropped connection -- the suite keeps running on the pool and writes its
    surface to shared storage, and the receipt that was supposed to hold the
    two populations together is simply never written.  A result that survives
    only in a live scrollback is what #112 item 1 asks us to stop producing, so
    the tool should not have that shape itself.

    The exit status this process never watched is read from the one runtime
    that did: PrismaBuild's worker records it in the action's outcome record,
    and the action is found by the effective ``--surface-json`` output of the
    pytest command in its own sealed request, bound to this population's
    source and attempt.
    It is never reconstructed from ``counts.failed`` -- a run can exit non-zero
    after a clean summary (a crash in teardown, an internal error, a timeout
    kill), so failures in the surface prove red while their absence does not
    prove green.  Where no single finished action wrote the path, the record
    says ``not observed`` and borrows nothing; a requeued action is one such
    case, and the pool requeues on any non-zero exit, so a red suite spends a
    while with no status at all.
    """

    record = {
        "arm": name,
        "why": arm["why"],
        "python": arm["python"],
        "requires_cuda": bool(arm["strict_cuda"]),
        "resumed": True,
        "exit_status_observed": False,
        "returncode": None,
        "exit_status_note": (
            "assembled after the fact from the population this run published; "
            "the submitting process did not survive to observe an exit status. "
            "Failures in the surface below are conclusive; their absence is "
            "not, so this arm cannot be called green from this receipt alone."
        ),
    }
    surface_json = receipt_dir / f"surface.{name}.json"
    _attach_surface(record, surface_json)
    _attach_pool_exit_status(record, surface_json)
    return record


#: The surface schemas this tool knows how to read.  An unrecognised string is
#: not a population it can reason about, and a population it cannot reason
#: about is not evidence for green.
_SURFACE_SCHEMAS = ("tessera.test_surface.v1", "tessera.test_surface.v2",
                    "tessera.test_surface.v3")
#: The first schema that publishes what the run EXECUTED on the device rather
#: than only which device it saw.  Older files cannot answer tessera#152's
#: question, and "cannot answer" is not "yes".
_EXECUTION_SCHEMA = "tessera.test_surface.v3"


def _evidence_problem(record: dict) -> str | None:
    """Why this arm's population cannot be read as evidence, or ``None``.

    A verdict is only as good as the fields it checked, and the ones it did
    not check were the ones a defect would land in: a surface with no
    ``counts`` skipped the executed-test leg entirely (``continue``), and a
    surface that never said which schema it was written in, or whether it was
    one xdist worker's share, was read as a population anyway.  Each of those
    is a green verdict resting on a file that never claimed to be the thing it
    was read as (#217).
    """

    arm = record["arm"]
    surface = record.get("surface") or {}
    schema = surface.get("schema")
    if schema not in _SURFACE_SCHEMAS:
        return (f"the {arm} arm's population states no schema this tool "
                f"recognises ({schema!r}), so what its fields mean is unknown")
    role = surface.get("role")
    if role != "population":
        return (f"the {arm} arm's population does not state the role "
                f"'population' ({role!r}); a worker's share and a file written "
                "before the role field existed are both slices as far as this "
                "tool can tell")
    counts = surface.get("counts")
    if not isinstance(counts, dict) or not all(
            isinstance(counts.get(key), int)
            for key in ("passed", "failed", "skipped")):
        return (f"the {arm} arm's population publishes no readable counts "
                f"({counts!r}), so nothing establishes that it ran")
    if not counts.get("passed"):
        skipped = counts.get("skipped") or 0
        return (f"the {arm} arm published a population in which nothing ran "
                f"(0 passed, {skipped} skipped)")
    return None


def _coverage_problem(record: dict) -> str | None:
    """Why an arm submitted to cover the CUDA-gated surface did not, or ``None``.

    Four legs, and the last two are the ones ``--strict-cuda`` already refuses
    on inside the run itself (tessera#146, tessera#152).  A receipt that quotes
    a population must refuse on the same evidence, or the gate holds only for
    whoever was watching the terminal.
    """

    if not record.get("requires_cuda"):
        return None
    arm = record["arm"]
    surface = record.get("surface") or {}
    if not surface.get("cuda"):
        device = surface.get("device") or "device not stated"
        return (f"the {arm} arm was submitted to cover the CUDA-gated surface "
                f"and published a population that saw no device ({device})")
    if not surface.get("strict_cuda"):
        return (f"the {arm} arm's population says the --strict-cuda gate was "
                "not armed, so a device-less placement would have skipped the "
                "surface and passed")
    if surface.get("schema") != _EXECUTION_SCHEMA:
        return (f"the {arm} arm published a {surface.get('schema')} "
                "population, which cannot say whether any test executed on "
                f"the device; {_EXECUTION_SCHEMA} is the schema that does")
    coverage = surface.get("cuda_surface") or {}
    executed = coverage.get("executed")
    if not isinstance(executed, int) or executed <= 0:
        return (f"the {arm} arm's population says {executed!r} test(s) "
                "allocated on the device, so the surface it was submitted to "
                "cover was collected and skipped rather than run")
    gated = coverage.get("box_artifact_skips") or {}
    if gated:
        named = "; ".join(f"{count} for {reason}"
                          for reason, count in sorted(gated.items()))
        return (f"the {arm} arm skipped tests for evidence this box does not "
                f"hold ({named}), so it did not cover the surface it claims")
    return None


def _source_problem(commits: dict) -> str | None:
    """Why these arms did not establish that they measured one tree, or ``None``.

    The receipt already computed this and the verdict did not read it, so a
    merge check could exit 0 over two populations of two different source
    trees -- which is not a merge check, it is two measurements.  Snapshot
    commit IDs may legitimately differ (PB stamps an action-specific closure
    member), so the field that decides is the independently verified effective
    source, never the raw commit.
    """

    source = commits.get("effective_source") or {}
    if source.get("agree") is True:
        return None
    unverified = source.get("unverified_arms") or []
    if source.get("agree") is False:
        pairs = ", ".join(f"{arm} {digest[:12]}"
                          for arm, digest in sorted(
                              (source.get("by_arm") or {}).items()) if digest)
        return ("refused: the arms measured different verified source trees "
                f"({pairs}); two source trees are two measurements, not a "
                "merge receipt")
    return ("incomplete: no verified effective source was established for: "
            + (", ".join(unverified) or "these arms")
            + " -- a population of unknown provenance cannot be shown to be "
              "the same source as its counterpart")


def _arm_results(arms: list[dict]) -> dict:
    """Each arm's own result, which is not the merge's.

    Per-population success stays available and stays *separate*: a reader (or
    a script) that wants to know whether the x86 arm passed can have that
    answer without it ever being spelled the way the merge verdict is spelled,
    because those are different claims and the exit status belongs to the
    second one.
    """

    results = {}
    for record in arms:
        counts = (record.get("surface") or {}).get("counts") or {}
        if record.get("status"):
            results[record["arm"]] = record["status"]
        elif record.get("surface") is None:
            results[record["arm"]] = "no population published"
        elif counts.get("failed") or counts.get("error"):
            results[record["arm"]] = (
                f"red: {counts.get('failed', 0)} failed, "
                f"{counts.get('error', 0)} error")
        elif record.get("returncode") not in (0, None):
            results[record["arm"]] = f"red: exit {record['returncode']}"
        elif _evidence_problem(record) or _coverage_problem(record):
            results[record["arm"]] = "incomplete: " + (
                _evidence_problem(record) or _coverage_problem(record))
        elif not record.get("exit_status_observed", True):
            results[record["arm"]] = (
                "published no failure; exit status not observed")
        else:
            results[record["arm"]] = "green"
    return results


def _verdict(arms: list[dict], commits: dict | None = None) -> str:
    """Green is never stated without naming the populations it is green on.

    "green on both populations" was the first spelling here and it lies the
    moment somebody passes ``--arm gpu``: one arm, and a verdict claiming two.
    That is the same sentence shape as the one #112 is about -- a result quoted
    without the population it was measured on -- so the arms are named.
    """

    names = ", ".join(record["arm"] for record in arms) or "no arms"
    if any(record.get("status") for record in arms):
        return "not run"
    if any(record.get("surface") is None for record in arms):
        return "incomplete: an arm published no population"
    # Evidence for red and evidence for green are not symmetric, and a resumed
    # receipt is where that stops being pedantry.  A failure the run itself
    # published is conclusive whether or not anybody watched the exit status; a
    # clean summary is not, because a run can still exit non-zero after it (a
    # crash in teardown, an internal error, a timeout kill).  So: count
    # failures from the surface, and refuse "green" for any arm whose exit
    # status nobody observed.
    for record in arms:
        counts = (record.get("surface") or {}).get("counts") or {}
        if counts.get("failed") or counts.get("error"):
            return f"red on one of: {names}"
        if record.get("returncode") not in (0, None):
            return f"red on one of: {names}"
    # A population in which nothing ran is not a green population, and neither
    # is a file that never said what it was.  Every check above is satisfied by
    # an arm that collected the suite and skipped all of it -- no failures,
    # exit 0, a surface on disk -- which is the shape this tool would take if
    # the fix for tessera#114 were "make the x86 arm skip whatever it cannot
    # import".  That is the green lie the refusal above exists to prevent,
    # arriving through the other door.  A verdict is allowed to say green only
    # about tests it can see were actually executed.
    for record in arms:
        problem = _evidence_problem(record)
        if problem:
            return "incomplete: " + problem
    # Second leg, and the reason it exists: everything above is satisfied by a
    # run that skipped the entire surface it was submitted to cover.  The GPU
    # arm's whole claim rests on ``--strict-cuda`` having refused a device-less
    # session -- one unexercised code path between a green tick and an
    # unmeasured population, which is tessera#112 verbatim.  The surface
    # already publishes those facts, derived in-process from torch rather than
    # asserted about another runtime, so the verdict reads them too.
    for record in arms:
        problem = _coverage_problem(record)
        if problem:
            return "incomplete: " + problem
    # Third: the arms have to have measured one tree.  Two clean populations
    # of two different source trees are two measurements; calling that a merge
    # success is the same category error as quoting a pass count without its
    # device.
    problem = _source_problem(commits if commits is not None
                              else _commits_measured(arms))
    if problem:
        return problem
    unobserved = [record["arm"] for record in arms
                  if not record.get("exit_status_observed", True)]
    if unobserved:
        return (f"published {len(arms)} population(s) with no failure: {names} "
                f"-- exit status not observed for: {', '.join(unobserved)}")
    return f"green on {len(arms)} population(s): {names}"


def _commits_measured(arms: list[dict]) -> dict:
    """The trees the arms actually ran, and whether they agree.

    Snapshot IDs remain verbatim. PB embeds action-specific closure metadata,
    so effective source agreement is a separate, independently verified fact.

    ``agree`` is ``True`` only when every arm said which tree it ran and they
    all said the same one.  One arm that cannot answer makes agreement
    *unestablished*, not true: a single stamped commit next to a silent arm is
    exactly the shape of the mistake this field exists to catch.
    """

    by_arm = {r["arm"]: (r.get("surface") or {}).get("commit") for r in arms}
    unstamped = sorted(a for a, c in by_arm.items() if not c)
    stamped = {c for c in by_arm.values() if c}
    sources = {}
    for record in arms:
        surface = record.get("surface") or {}
        source = surface.get("source_identity") or {}
        digest = source.get("sha256")
        verified = (source.get("schema") == "tessera.suite_source.v1"
                    and source.get("verification") == "verified"
                    and source.get("snapshot_commit") == surface.get("commit")
                    and isinstance(digest, str) and len(digest) == 64
                    and all(c in "0123456789abcdef" for c in digest))
        sources[record["arm"]] = digest if verified else None
    unknown_sources = sorted(arm for arm, digest in sources.items() if not digest)
    return {
        "by_arm": by_arm,
        "agree": None if (unstamped or not stamped) else (len(stamped) == 1),
        "unstamped_arms": unstamped,
        "effective_source": {
            "by_arm": sources,
            "agree": (None if unknown_sources or not sources
                      else len(set(sources.values())) == 1),
            "unverified_arms": unknown_sources,
        },
    }


def _arm_commit(record: dict, population: dict) -> tuple[str, bool]:
    """Which tree this arm ran, and whether that is established or assumed.

    The arms of a run are separate processes on separate boxes and they do not
    start together: an x86 arm can finish while the GPU arm is still queued
    behind a held reservation, and the clone a pool action runs in can be
    fast-forwarded in between.  Reading the checkout once, at receipt-assembly
    time, and stamping that commit on every row is the same error this file is
    about -- a number separated from the context that gives it meaning -- just
    one level up.

    So the arm's own published population answers first.  A surface written
    before this field existed cannot answer, and then the population commit is
    the best available guess and is labelled as one.
    """

    stamped = (record.get("surface") or {}).get("commit")
    if stamped:
        return stamped, True
    return population["commit"], False


LEDGER_HEADER = """# Suite populations

One row per arm per `tools/merge_suite.py` run, and **every** arm gets a row:
an arm the run did not submit is written as `not submitted in this run` rather
than left out. A pass count means nothing without the device population it was
measured on, and a lone row is a result quoted without its counterpart --
exactly the misreading tessera#112 is about. So the rows of a run always name
both populations, even when only one was measured.

Rows above 2026-09-04T08:11 predate that rule and can be lone: a run submitted
with `--arm x86` wrote one row and said nothing about the GPU population. Read
a lone row there as "the other arm was not recorded", not as a whole result.

`master head?` is whether the commit under test was master's tip at submit
time. `yes` is a merge receipt; `no` is a branch's own run; `unknown` means no
master ref resolved in that checkout and the question was not answered.

`commit` is the tree that arm's own run reported measuring, which is not
always the tree the receipt was assembled against: the arms are separate
processes on separate boxes, and a queued arm can place after the checkout has
moved. `(assumed)` marks a row whose run predates that field, where the
receipt's own commit is the best available guess. PrismaBuild's parentless
snapshot commits also differ when only its verified action-specific closure
stamp differs. New populations retain that raw commit and independently hash
the effective source; the JSON receipt's
`commits_measured.effective_source.agree` distinguishes equivalent source from
different source, and is unknown for legacy or unverifiable populations.
`source <hash>` beside a row's snapshot commit names that verified source.
Pass counts alone do not establish a same-source merge check.

`exit` is the status the submitting process observed. `0 (pool)` is a status
this program did not watch and did not guess: the run was resumed, and the
number is the one PrismaBuild's own worker recorded for the action the
population names as its producer, for the attempt whose own output says it
published that population. `not observed` is the remaining case -- a resumed
row with no single finished pool record bound to it, including a population
that names no producer (pre-stamp, or `unknown` source) -- and there the
failure count is still a fact while a zero in it does not make the row green,
because a suite can exit non-zero after a clean summary.

`mode` is how that arm ran, and it changes what the row means as much as the
device does. The GPU arm is always `serial` -- its workers would share one
device and its CUDA venv has no xdist -- while the x86 arm runs `-n <cpus>`, so
two rows of one commit can differ by more than the box. On `d11dc01` the gpu
row is green and the x86 row is red at 5 failed, and those five are an
`-n`-only defect in the suite's own conftest, not a CUDA one. Match a pair by
`commit`, then read `mode` and the failures before attributing the difference
to the device. `--` is a row whose mode was not recorded: rows above
2026-09-04T10:30 predate the column, and a resumed row can only name a mode
when exactly one finished pool action wrote its population, since the mode is
read out of that action's own command.

`device` distinguishes three absences that are not the same thing. `not
submitted in this run` is an arm nobody asked for. `no population published`
is an arm that was submitted and returned nothing -- refused, never placed, or
dead before its summary. A device string is a measurement.

| measured (UTC) | commit | master head? | arm | mode | device | passed | failed | skipped | not collected | exit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
"""


def _mode_cell(record: dict) -> str:
    """How this arm ran, or `--` where nobody recorded it.

    The five failures on ``82f0047`` were an ``-n``-only defect: the same
    commit, the same box and the same files were green run serially.  A ledger
    that puts two populations side by side and omits the mode therefore invites
    the reader to attribute a difference to the device that the device did not
    cause -- which is the misreading tessera#112 is about, one column over.
    Its own header used to say so in prose and leave the column out.
    """

    cpus = record.get("cpus_used")
    if not isinstance(cpus, int):
        return "--"
    return "serial" if cpus <= 1 else f"-n {cpus}"


def _record_markdown(path: Path, receipt: dict) -> None:
    """Append this run's arms to a ledger a reader of the repo can check.

    A receipt on shared storage is not a scrollback, but it is also not
    somewhere anybody looks. This is.
    """

    population = receipt["population"]
    head = population["is_master_head"]
    head_text = "unknown" if head is None else ("yes" if head else "no")
    rows = []
    for record in receipt["arms"]:
        surface = record.get("surface") or {}
        counts = surface.get("counts") or {}
        cell = lambda key: str(counts.get(key, "--"))          # noqa: E731
        if record.get("exit_status_observed", True):
            exit_text = str(record.get("returncode", "--"))
            # Whose observation it is stays on the row.  A status this process
            # watched and a status read out of the pool's outcome record are
            # both facts, and they are not the same fact -- one of them is
            # about a run nobody here was present for.
            if record.get("exit_status_source") == "pool":
                exit_text += " (pool)"
        else:
            # Never a bare number here: this process did not watch the run, and
            # a row that looks watched when it was not is the same overclaim
            # the whole file exists to prevent.
            exit_text = "not observed"
        commit, established = _arm_commit(record, population)
        commit_text = f"`{commit[:12]}`" if established \
            else f"`{commit[:12]}` (assumed)"
        source = _commits_measured([record])["effective_source"]["by_arm"][record["arm"]]
        if source:
            commit_text += f"<br>source `{source[:12]}`"
        # ``master head?`` was answered against the population commit. If this
        # arm ran a different tree, that answer is not about this row.
        row_head = head_text if commit == population["commit"] else "unknown"
        # An arm that published nothing measured nothing, so it has no
        # measurement time. Falling through to the receipt's own clock put a
        # timestamp in that cell that dated the bookkeeping and looked exactly
        # like a measurement -- the same confusion as the resumed row, in the
        # one case where there is no measurement at all.
        when = record.get("measured_utc") or (
            receipt["generated_utc"] if surface else "--")
        rows.append(
            f"| {when} | "
            f"{commit_text} | "
            f"{row_head} | {record['arm']} | {_mode_cell(record)} | "
            f"{surface.get('device', 'no population published')} | "
            f"{cell('passed')} | {cell('failed')} | {cell('skipped')} | "
            f"{len(surface.get('not_collected', []))} | {exit_text} |"
        )
    # An arm this run did not submit still gets a row, saying so.  The header
    # promises the two populations side by side; three of the first four rows
    # this tool ever wrote were lone `--arm x86` rows with no GPU counterpart,
    # so the promise was true of the prose and false of the artefact.  A reader
    # of a lone row cannot tell "the other arm was not asked for" from "the
    # other arm is somewhere else in this file", and a whole result is what a
    # lone row looks like.  Naming the absence costs one line and removes the
    # question.
    covered = {record["arm"] for record in receipt["arms"]}
    for name in sorted(ARMS):
        if name in covered:
            continue
        rows.append(
            f"| -- | -- | -- | {name} | -- | not submitted in this run "
            "| -- | -- | -- | -- | -- |")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(LEDGER_HEADER)
    with path.open("a") as handle:
        handle.write("\n".join(rows) + "\n")


def _keep_any_previous(path: Path):
    """Move an earlier receipt aside instead of writing over it.

    ``--resume`` reassembles a receipt in the directory the original run wrote
    one into, and its default output name is that run's own ``receipt.json``.
    So resuming an arm silently destroyed the receipt that recorded the other
    arm -- the exact loss the surface files are already protected from, in the
    file that holds them together.  ``20260904T025044`` is where this would
    have landed: its ``receipt.json`` is the only record of that run's x86
    submission, and the GPU arm queued behind a held reservation for nine
    hours before anyone could resume it.

    Both receipts are evidence and the second does not disprove the first, so
    keep both.  The reader is unchanged: ``receipt.json`` is always the newest.
    """

    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(path.stat().st_mtime))
    kept = path.with_name(f"{path.stem}.superseded-{stamp}{path.suffix}")
    index = 0
    while kept.exists():
        index += 1
        kept = path.with_name(
            f"{path.stem}.superseded-{stamp}-{index}{path.suffix}")
    path.rename(kept)
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkout", default=str(Path(__file__).resolve().parents[1]),
                    help="tree to test; the x86 arm needs it under /mnt/shared")
    ap.add_argument("--arm", action="append", choices=sorted(ARMS),
                    default=[], help="repeatable; default is both")
    ap.add_argument("--gpu-tag", default="",
                    help="explicit GPU worker tag (e.g. sparky or sparklina); "
                         "required for live GPU submissions. pbrun --exclusive "
                         "reserves that worker's advertised full GPU capacity")
    ap.add_argument("--cpus", type=int, default=1,
                    help="cores each arm that can use them will ACTUALLY use: "
                         "declared to the pool and, above 1, passed to pytest "
                         "as -n so the reservation and the command cannot "
                         "disagree. Clamped to 1 for any arm that cannot fan "
                         "out -- the GPU arm always, since its workers share "
                         "one device and the CUDA venv has no pytest-xdist -- "
                         "so one number can submit an -n x86 arm and a serial "
                         "GPU arm in the same run. Default 1")
    ap.add_argument("--mem-gb", type=int, default=16)
    ap.add_argument("--timeout-s", type=_positive_seconds, default=3600.0,
                    help="positive finite per-attempt inner deadline; TERM then KILL "
                         "after 5s, not a queue/retry lifetime limit")
    ap.add_argument("--wait-s", type=float, default=5400.0)
    ap.add_argument("--out", default="",
                    help="receipt path; default is a timestamped file under "
                         f"{DEFAULT_RECEIPT_ROOT}")
    ap.add_argument("--pytest-arg", action="append", default=[],
                    help="repeatable, passed through to pytest")
    ap.add_argument("--record", default="",
                    help="also append one row per arm to this markdown ledger; "
                         "docs/status/suite-populations.md is the one a reader "
                         "of the repo checks")
    ap.add_argument("--resume", default="",
                    help="submit nothing; assemble the receipt from the "
                         "surface.<arm>.json files already in this directory. "
                         "For a run whose submitting process died while the "
                         "pool kept going -- the exit status is recorded as "
                         "unobserved rather than guessed")
    ap.add_argument("--pool-root", default=str(POOL_ROOT),
                    help="where PrismaBuild publishes what it did: "
                         "<root>/pb-queue holds the outcome records a resumed "
                         "receipt reads its exit status from, and "
                         "<root>/cas/requests the commands that identify which "
                         "action wrote which population. Naming it is what "
                         "lets a test read a pool it built rather than the "
                         "live one -- a scan of the real queue is an NFS read "
                         "per finished action and grows with the fleet's "
                         "history, which took two tests past their timeout "
                         "here")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the pbrun invocations and submit nothing")
    args = ap.parse_args()

    # Module-level because that is where the readers look, and because a test
    # that monkeypatches them is doing the same thing this flag does.
    global POOL_QUEUE, POOL_CAS_REQUESTS
    pool_root = Path(args.pool_root).resolve()
    POOL_QUEUE = pool_root / "pb-queue"
    POOL_CAS_REQUESTS = pool_root / "cas" / "requests"

    args.checkout = Path(args.checkout).resolve()
    wanted = args.arm or sorted(ARMS)
    if "gpu" in wanted and not args.gpu_tag and not (args.resume or args.dry_run):
        print("merge_suite: live GPU submission requires --gpu-tag to select "
              "the worker whose full GPU capacity pbrun reserves", file=sys.stderr)
        return 2

    shared = str(args.checkout).startswith(str(SHARED_ROOT))
    for name in wanted:
        # A resume submits nothing, so a placement constraint has nothing to
        # constrain: the run it is reading already happened somewhere.
        if args.resume:
            break
        if ARMS[name].get("needs_shared_checkout") and not shared:
            print(
                f"merge_suite: the {name} arm needs a checkout under "
                f"{SHARED_ROOT} -- pbrun pins an action to the submitting box "
                f"when its tree exists on one box only, and {args.checkout} "
                "does. Copy the tree there, or drop this arm and say in the "
                "receipt that only one population was covered.",
                file=sys.stderr)
            return 2

    stamp = time.strftime("%Y%m%dT%H%M%S")
    receipt_dir = Path(args.resume).resolve() if args.resume \
        else DEFAULT_RECEIPT_ROOT / stamp
    if args.resume and not receipt_dir.is_dir():
        print(f"merge_suite: --resume {receipt_dir} is not a directory",
              file=sys.stderr)
        return 2
    # A dry run composes paths and creates nothing: littering shared storage
    # with an empty directory per invocation is how a receipt root stops being
    # readable.
    if not args.dry_run and not args.resume:
        receipt_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else receipt_dir / "receipt.json"

    if args.resume:
        arms = [_resume(name, ARMS[name], receipt_dir) for name in wanted]
    else:
        with ThreadPoolExecutor(max_workers=len(wanted)) as pool:
            futures = {name: pool.submit(_submit, name, ARMS[name], args,
                                         receipt_dir)
                       for name in wanted}
            arms = [futures[name].result() for name in wanted]

    commits = _commits_measured(arms)
    receipt = {
        "schema": "tessera.merge_suite.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "submitted_from": os.uname().nodename,
        "assembled_by": "resume" if args.resume else "submit",
        "population": _population_of(args.checkout),
        "commits_measured": commits,
        "verdict": _verdict(arms, commits),
        # Each arm's own result, kept separate from the merge verdict on
        # purpose: an arm can be green while the run is not a merge success,
        # and only the verdict decides the exit status.
        "arm_results": _arm_results(arms),
        "arms": arms,
        "reading_note": (
            "Each arm's counts belong to that arm's device population and to "
            "no other. A pass count quoted without the device beside it is the "
            "misreading tessera#112 is about. Snapshot commit IDs are preserved; "
            "commits_measured.effective_source.agree separately establishes "
            "whether the populations exercised equivalent verified source."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = _keep_any_previous(out)
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    if kept:
        print(f"merge_suite: a previous receipt was here; kept at {kept}")
    if args.record:
        _record_markdown(Path(args.record), receipt)

    print(f"merge_suite: {receipt['verdict']}")
    for record in arms:
        surface = record.get("surface") or {}
        counts = surface.get("counts") or {}
        print(f"  {record['arm']:4s} rc={record.get('returncode', '-')} "
              f"cuda={surface.get('cuda', '?')} "
              f"passed={counts.get('passed', '?')} "
              f"failed={counts.get('failed', '?')} "
              f"skipped={counts.get('skipped', '?')} "
              f"not_collected={len(surface.get('not_collected', []))}")
    print(f"merge_suite: receipt {out}")
    return 0 if receipt["verdict"].startswith("green on") else 1


if __name__ == "__main__":
    raise SystemExit(main())
