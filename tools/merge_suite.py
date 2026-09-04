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
process saw.  Where no single finished action wrote the path -- still in
flight, requeued after a non-zero exit, or two of them did -- the row stays
``not observed`` and nothing is borrowed: published failures prove red, their
absence does not prove green.

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
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PBRUN = Path("/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py")
SHARED_ROOT = Path("/mnt/shared")
#: Surface reports are written OUTSIDE the checkout on purpose.  pbrun binds a
#: command's identity to the checkout's git delta, so an artefact dropped into
#: the tree moves the action key of every later submission from it -- a cache
#: miss dressed up as a different action.
DEFAULT_RECEIPT_ROOT = SHARED_ROOT / "tessera-suite-receipts"

#: Where the pool publishes what it did.  A finished action's outcome record
#: carries the exit status the worker actually saw; the CAS request beside it
#: carries the command that action ran.  Reading those two is how a resumed
#: receipt can state an exit status instead of declining to -- derived from a
#: table PrismaBuild publishes, never inferred from the population's contents.
POOL_QUEUE = SHARED_ROOT / "prismabuild-fleet" / "pb-queue"
POOL_CAS_REQUESTS = SHARED_ROOT / "prismabuild-fleet" / "cas" / "requests"

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
        "pbrun_flags": ["--gpu"],
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


def _submit(name: str, arm: dict, args, receipt_dir: Path) -> dict:
    surface_json = receipt_dir / f"surface.{name}.json"
    # The reservation is this ARM's, not the run's: an arm clamped to serial
    # must not hold the cores it was told to spend, or the ledger says the box
    # is busy while seven of its cores idle.
    cpus = _arm_cpus(arm, args.cpus)
    command = _command(arm, surface_json, args.pytest_arg, cpus)
    invocation = [
        sys.executable, str(PBRUN),
        *arm["pbrun_flags"],
        "--cpus", str(cpus),
        "--demand", f"{'gpu=1,' if arm['pbrun_flags'][0] == '--gpu' else ''}"
                    f"mem_gb={args.mem_gb}",
        "--cwd", str(args.checkout),
        "--timeout-s", str(args.timeout_s),
        "--wait-s", str(args.wait_s),
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


def _pool_actions_that_wrote(surface_json: Path) -> list[dict]:
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

    The join is the ``--surface-json`` path.  It appears verbatim in the
    action's command in the CAS request, so "the action that wrote this
    population" is answerable exactly rather than by matching prose.  The
    stdout in the outcome record would also contain it, but stdout can be
    truncated and a command cannot.

    Returns every match, because the caller must be able to tell one from
    several: a receipt directory that two actions wrote to (a retry, or the
    polluted ``20260904T025044``) has no single exit status, and picking one
    would be exactly the overclaim the rest of this file refuses.
    """

    wanted = str(surface_json)
    found = []
    for state in ("done", "failed"):
        folder = POOL_QUEUE / state
        if not folder.is_dir():
            continue
        for outcome_path in sorted(folder.glob("*.json")):
            key = outcome_path.stem
            request = POOL_CAS_REQUESTS / key[:2] / f"{key}.json"
            try:
                request_payload = json.loads(request.read_text())
            except (OSError, ValueError):
                continue
            command = (request_payload.get("params") or {}).get("command") or []
            if wanted not in [str(part) for part in command]:
                continue
            try:
                outcome = json.loads(outcome_path.read_text())
            except (OSError, ValueError):
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
            })
    return found


def _attach_pool_exit_status(record: dict, surface_json: Path) -> None:
    """Let the pool answer the question this process cannot.

    Three outcomes, and the two that are not "one action" both leave the row
    unobserved rather than borrowing a number:

    * exactly one finished action wrote this path, and its record carries an
      integer status -- that status is the exit status, and the row says where
      it came from;
    * no finished action wrote it -- the run may still be in flight, or its
      record may have been reaped, and either way nobody here saw an exit;
    * several did -- a retried action or a receipt directory two runs shared.
      There is no single status to report, so none is reported and the row
      says how many wrote it.
    """

    matches = _pool_actions_that_wrote(surface_json)
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
    and the action is found by the ``--surface-json`` path in its own command.
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


def _verdict(arms: list[dict]) -> str:
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
    # Second leg, and the reason it exists: everything above is satisfied by a
    # run that skipped the entire surface it was submitted to cover.  The GPU
    # arm's whole claim rests on ``--strict-cuda`` having refused a device-less
    # session -- one unexercised code path between a green tick and an
    # unmeasured population, which is tessera#112 verbatim.  The surface
    # already publishes both facts, derived in-process from torch rather than
    # asserted about another runtime, so the verdict reads them too.  Two
    # independent legs, not one.
    for record in arms:
        if not record.get("requires_cuda"):
            continue
        surface = record.get("surface") or {}
        if not surface.get("cuda"):
            device = surface.get("device") or "device not stated"
            return ("incomplete: the " + record["arm"] + " arm was submitted "
                    "to cover the CUDA-gated surface and published a "
                    "population that saw no device (" + device + ")")
        if not surface.get("strict_cuda"):
            return ("incomplete: the " + record["arm"] + " arm's population "
                    "says the --strict-cuda gate was not armed, so a "
                    "device-less placement would have skipped the surface "
                    "and passed")
    unobserved = [record["arm"] for record in arms
                  if not record.get("exit_status_observed", True)]
    if unobserved:
        return (f"published {len(arms)} population(s) with no failure: {names} "
                f"-- exit status not observed for: {', '.join(unobserved)}")
    return f"green on {len(arms)} population(s): {names}"


def _commits_measured(arms: list[dict]) -> dict:
    """The trees the arms actually ran, and whether they agree.

    Two arms that ran different commits are two measurements, not one merge
    receipt, and a reader must be told that without having to diff the rows.

    ``agree`` is ``True`` only when every arm said which tree it ran and they
    all said the same one.  One arm that cannot answer makes agreement
    *unestablished*, not true: a single stamped commit next to a silent arm is
    exactly the shape of the mistake this field exists to catch.
    """

    by_arm = {r["arm"]: (r.get("surface") or {}).get("commit") for r in arms}
    unstamped = sorted(a for a, c in by_arm.items() if not c)
    stamped = {c for c in by_arm.values() if c}
    return {
        "by_arm": by_arm,
        "agree": None if (unstamped or not stamped) else (len(stamped) == 1),
        "unstamped_arms": unstamped,
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
receipt's own commit is the best available guess. Rows of one run with two
commits are two measurements, not one merge receipt.

`exit` is the status the submitting process observed. `0 (pool)` is a status
this program did not watch and did not guess: the run was resumed, and the
number is the one PrismaBuild's own worker recorded for the action that wrote
that population. `not observed` is the remaining case -- a resumed row with no
single finished pool action behind it -- and there the failure count is still a
fact while a zero in it does not make the row green, because a suite can exit
non-zero after a clean summary.

The run mode is not in the table and changes what a row means. The GPU arm
runs serially -- its CUDA venv has no xdist -- while the x86 arm runs `-n
<cpus>`, so two rows of one commit can differ by more than the device. On
`d11dc01` the gpu row is green and the x86 row is red at 5 failed, and those
five are a `-n`-only defect in the suite's own conftest, not a CUDA one. Match
a pair by `commit`, then read the failures before attributing the difference to
the device.

`device` distinguishes three absences that are not the same thing. `not
submitted in this run` is an arm nobody asked for. `no population published`
is an arm that was submitted and returned nothing -- refused, never placed, or
dead before its summary. A device string is a measurement.

| measured (UTC) | commit | master head? | arm | device | passed | failed | skipped | not collected | exit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
"""


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
            f"{row_head} | {record['arm']} | "
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
            f"| -- | -- | -- | {name} | not submitted in this run "
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
    ap.add_argument("--timeout-s", type=float, default=3600.0)
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
    ap.add_argument("--dry-run", action="store_true",
                    help="print the pbrun invocations and submit nothing")
    args = ap.parse_args()

    args.checkout = Path(args.checkout).resolve()
    wanted = args.arm or sorted(ARMS)

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

    receipt = {
        "schema": "tessera.merge_suite.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "submitted_from": os.uname().nodename,
        "assembled_by": "resume" if args.resume else "submit",
        "population": _population_of(args.checkout),
        "commits_measured": _commits_measured(arms),
        "verdict": _verdict(arms),
        "arms": arms,
        "reading_note": (
            "Each arm's counts belong to that arm's device population and to "
            "no other. A pass count quoted without the device beside it is the "
            "misreading tessera#112 is about."
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
