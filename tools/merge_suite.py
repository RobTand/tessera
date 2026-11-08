#!/usr/bin/env python3
"""Run explicit suite populations through native Flux and retain bound receipts.

Each arm saves its immutable source, command, resources and exact native ID.
Resume submits nothing. One population snapshot is bound to that job's original
jobspec, verified tracked source, publication digest, summary and final status.
CPU evidence does not qualify the CUDA or container population.
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
from pathlib import PurePosixPath
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tessera._dev.suite_deadline import positive_seconds as _positive_seconds  # noqa: E402
import _suite_flux as suite_flux  # noqa: E402
import _suite_container as suite_container  # noqa: E402
from tessera._dev.surface_publication import (  # noqa: E402
    POPULATION,
    digest_bytes,
    published_digests,
)

SHARED_ROOT = Path("/mnt/shared")
DEFAULT_RECEIPT_ROOT = SHARED_ROOT / "tessera-suite-receipts"
PROCESS_THREAD_LIMITS = suite_flux.THREADS
CHILD_ENV = {"PYTHONPATH": "src"}
TIMEOUT_KILL_AFTER_S = 5.0

ARMS = {
    "gpu": {
        "why": "the CUDA-gated surface; nothing else exercises it",
        "python": "/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python",
        "strict_cuda": True,
        "fans_out": False,
        "serial_because": (
            "this legacy wrapper selects serial GPU execution whatever "
            "--cpus says; direct PrismaBuild submissions may use xdist "
            "when the tests have independent process state"
        ),
    },
    "x86": {
        "why": "the device-less population: torch present, no CUDA device, "
               "which is what dl380g10 actually reports",
        "python": "/home/rob/venvs/pb-cpu/bin/python",
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


SUBMISSION_METADATA = "submission-master.json"
_SUBMISSION_SCHEMA = "tessera.submission_master.v1"


def _submission_master_of(checkout: Path) -> dict:
    """Observe the checkout/master comparison once, before dispatch.

    Ref precedence is retained; this does not fetch or establish freshness.
    Compare against the same resolved head that is recorded, not a second
    lookup of a potentially moving ref.
    """

    head = _git(checkout, "rev-parse", "HEAD")
    master, master_ref = None, "none resolved"
    for ref in ("master", "origin/master", "refs/remotes/origin/master"):
        master = _git(checkout, "rev-parse", ref)
        if master:
            master_ref = ref
            break
    return {
        "schema": _SUBMISSION_SCHEMA,
        "checkout_commit_at_submit": head,
        "master_ref_used": master_ref,
        "master_head_at_submit": master or None,
        "is_master_head": (head == master) if (head and master) else None,
    }


def _write_submission_master(checkout: Path, receipt_dir: Path) -> None:
    """Persist before launching any client that can outlive this process."""

    observation = _submission_master_of(checkout)
    (receipt_dir / SUBMISSION_METADATA).write_text(
        json.dumps(observation, indent=2) + "\n")


def _read_submission_master(receipt_dir: Path) -> dict | None:
    """Only the dedicated pre-dispatch observation is submission evidence.

    Old receipts used the same labels for late probes; never adopt them or
    backfill missing evidence from today's refs. This is provenance, no seal.
    """

    try:
        observation = json.loads((receipt_dir / SUBMISSION_METADATA).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(observation, dict) or observation.get("schema") != _SUBMISSION_SCHEMA:
        return None
    return observation


def _population_of(checkout: Path, submission_master: dict | None = None) -> dict:
    """Current checkout context, with submit-time fields only from evidence."""

    observation = submission_master or {}
    dirty = _git(checkout, "status", "--porcelain")
    return {
        "checkout": str(checkout),
        "commit": _git(checkout, "rev-parse", "HEAD"),
        "describe": _git(checkout, "log", "-1", "--pretty=%s"),
        "checkout_commit_at_submit": observation.get("checkout_commit_at_submit"),
        "master_ref_used": observation.get("master_ref_used", "none resolved"),
        "master_head_at_submit": observation.get("master_head_at_submit"),
        "is_master_head": observation.get("is_master_head"),
        # None where the probe did not answer: "not established", never "clean".
        "working_tree_dirty": (None if dirty is None else bool(dirty)),
    }


def _command(arm: dict, surface_json: Path, extra: list[str],
             cpus: int = 1) -> list[str]:
    """Compose owned targets, process count and population publication options.

    The native CPU reservation and pytest worker count come from the same
    value. Retained GPU arm metadata stays serial and cannot authorize native
    admission before that transport is separately qualified.
    """

    _validate_pytest_extra(extra)
    targets = arm.get("targets", ["tests"])
    if (not targets or len(targets) > 1024 or len(set(targets)) != len(targets)
            or any(not isinstance(target, str) or not target or "\0" in target
                   or target.startswith("-") or PurePosixPath(target.split("::")[0]).is_absolute()
                   or ".." in PurePosixPath(target.split("::")[0]).parts
                   or not (target == "tests" or target.startswith("tests/"))
                   for target in targets)):
        raise ValueError("suite targets require unique canonical paths under tests/")
    cpus = _arm_cpus(arm, cpus)
    command = [arm["python"], "-m", "pytest", *targets, "-q",
               "-o", "addopts=", "-p", "no:cacheprovider",
               "--surface-json", str(surface_json)]
    if cpus > 1:
        # The retained native mode is loadfile. The explicit container arm
        # uses independent process state and worksteal under its own contract.
        command += ["-n", str(cpus), "--dist", arm.get("dist", "worksteal")]
        if arm.get("container"):
            command += ["-p", "xdist.plugin"]
    if arm["strict_cuda"]:
        command.append("--strict-cuda")
    return command + extra


def _validate_pytest_extra(extra):
    """Own targets, config, process count, plugins and output destinations."""
    flags = {"-q", "-v", "-s", "-x", "--disable-warnings", "--no-header",
             "--full-trace", "--showlocals"}
    valued = {"-k", "-m", "--maxfail", "--durations", "--tb"}
    index = 0
    while index < len(extra):
        token = extra[index]
        key, equal, value = token.partition("=")
        if token in flags:
            index += 1
        elif key in valued:
            if not equal:
                index += 1
                if index == len(extra):
                    raise ValueError("pytest option requires a value: " + key)
            index += 1
        else:
            raise ValueError("pytest targets and execution controls are tool-owned: " + token)


def _timed_command(command: list[str], timeout_s: float) -> list[str]:
    """Per-attempt managed-process-group deadline, not a queue/descendant cap."""
    deadline = _positive_seconds(timeout_s)
    grace = _positive_seconds(TIMEOUT_KILL_AFTER_S)
    # Use the arm's named interpreter, not a host's unverified timeout binary.
    return [command[0], "tools/suite_deadline.py", "--timeout-s", str(deadline),
            "--kill-after-s", str(grace), "--", *command]


def _preflight(arm: dict, args) -> dict:
    if arm["strict_cuda"] or getattr(args, "gpu_image", "") or arm.get("container"):
        raise ValueError("native GPU/container suite admission is not qualified; preserve the requested arm")
    roots = {}
    for item in getattr(args, "artifact_root", []):
        key, separator, value = item.partition("=")
        if not separator or key in roots:
            raise ValueError("artifact roots require unique ENV=PATH declarations")
        roots[key] = value
    suite_flux.validate(_arm_cpus(arm, args.cpus), args.mem_gb, args.timeout_s, roots)
    _validate_pytest_extra(getattr(args, "pytest_arg", []))
    _positive_seconds(args.wait_s)
    _command(arm, Path("/preflight/population.json"), getattr(args, "pytest_arg", []),
             _arm_cpus(arm, args.cpus))
    return roots


def _submit(name: str, arm: dict, args, receipt_dir: Path) -> dict:
    roots = _preflight(arm, args)
    surface_json = receipt_dir / f"surface.{name}.json"
    cpus = _arm_cpus(arm, args.cpus)
    command = _command(arm, surface_json, args.pytest_arg, cpus)
    command += ["--junitxml", str(receipt_dir / f"junit.{name}.xml"), "--durations=10"]
    destination, refusal = _effective_surface_json(command)
    if refusal or destination != str(surface_json):
        raise ValueError("pytest arguments change the declared population path")
    command = _timed_command(command, args.timeout_s)
    if args.dry_run:
        return {"arm": name, "why": arm["why"], "python": arm["python"],
                "requires_cuda": arm["strict_cuda"], "command": command,
                "cpus_used": cpus, "mem_gb": args.mem_gb,
                "status": "not submitted (--dry-run)",
                "flux": shlex.join(command)}
    directory = receipt_dir / f"flux.{name}"
    suite_flux.submit(directory, args.checkout, command, cpus, args.mem_gb,
                      args.timeout_s, roots,
                      application=getattr(args, "selection_binding", None))
    try:
        suite_flux.wait(directory, args.wait_s)
    except subprocess.TimeoutExpired:
        pass  # A client wait limit does not cancel or resubmit the native ID.
    return _resume(name, arm, receipt_dir)


#: What a surface file has to say about itself before this tool will read it
#: as an arm's population.  ``tessera.test_surface.v1`` had no such field, and
#: a v1 file at this path could be one xdist worker's share: under ``-n 8``
#: every worker wrote the arm's canonical path, so eight shards landed there
#: before the controller's aggregate did.  v2 files say which they are.
_SHARD_ROLE = "worker-share"


class _Publication(NamedTuple):
    """ONE read of a surface file: its bytes, their digest, and their parse.

    A pathname is not a publication.  A pathname is a name that a pool retry
    can point at different bytes at two different instants, and this tool read
    it twice: ``_attach_surface`` parsed the population that the verdict then
    consumed, and the digest leg of ``_binding_refusal`` re-opened the same
    name to authenticate it.  Nothing established that the two reads saw one
    file.  The reachable sequence is a requeue (#340): attempt A publishes a
    verified population and crashes after its terminal summary; a resume loads
    A; retry B replaces the file with a population of *equal counts* whose
    source came out ``unknown`` (a dirty checkout) and exits 0; the resume
    then hashes B's bytes, accepts B's announced digest, keeps **A's** verified
    source and producer stamp on the record, adopts B's exit 0, and returns
    green.  A source-verification bypass assembled out of two honest reads.

    So the bytes are read once and everything downstream comes off that one
    buffer: ``digest`` is of ``raw`` and ``payload`` is the parse of ``raw``,
    both computed in ``_read_publication`` and never separately.  There is no
    constructor that takes a payload someone else parsed, which is the point --
    the invariant is structural rather than remembered, and a caller cannot
    hand the reader metadata from one read and a digest from another even by
    mistake.  ``path`` travels with them because "which publication" is part of
    the same fact.

    The file is free to change afterwards; that is the pool's business, not a
    reason to look again.  What this tool then says is a true statement about
    the publication it read, and ``record["surface_sha256"]`` names which one
    so a later reader can check for itself.
    """

    path: Path
    raw: bytes
    digest: str
    payload: dict


def _read_publication(surface_json: Path) -> "_Publication | None":
    """The publication at this path as one snapshot, or ``None`` if absent.

    The only place a surface file is read.  A malformed one still raises, as
    it always has: a receipt assembled around an unreadable population would
    be a worse answer than a traceback.
    """

    if not surface_json.exists():
        return None
    raw = surface_json.read_bytes()
    return _Publication(Path(surface_json), raw, digest_bytes(raw),
                        json.loads(raw))


def _attach_surface(record: dict, surface_json: Path) -> "_Publication | None":
    """Put this arm's published population on the record, or say it has none.

    A worker's share is refused rather than read.  It is a real measurement of
    a real slice, and it is not the arm's population -- reading one as the
    population is how ``docs/status/suite-populations.md`` would come to carry
    206 passed / 108 skipped as an x86 suite result.  An absent measurement is
    honest; a partial one wearing the whole one's name is not.

    Returns the ``_Publication`` whose payload it put on the record, so the
    caller can hand the *same read* to whatever else needs it rather than
    re-opening the name (#340).  ``None`` where there is no population to put
    on the record at all -- absent, or a share -- and a share is ``None`` on
    purpose: nothing downstream may bind a status to a slice.
    """

    published = _read_publication(surface_json)
    if published is not None:
        role = published.payload.get("role")
        if role == _SHARD_ROLE:
            record["surface"] = None
            record["shard_at_population_path"] = str(surface_json)
            record["no_surface_means"] = (
                f"the file at {surface_json.name} says it is one xdist "
                f"worker's share ({published.payload.get('worker_id')}), not "
                "this arm's population. A share is a slice of the run; "
                "reporting it as the population would put a fraction of a "
                "suite in the ledger as the whole of it. This is an absent "
                "measurement."
            )
            return None
        record["surface"] = published.payload
        record["surface_path"] = str(surface_json)
        # WHICH bytes the fields above were parsed from.  A resumed receipt is
        # assembled while the pool may still be writing, so naming the
        # publication is what lets a later reader check that the record and
        # whatever is at the path now are the same measurement (#340).
        record["surface_sha256"] = published.digest
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
        return published
    # The distinction that matters: a suite that ran and failed published a
    # surface; one that was never placed, or was refused before collection,
    # did not.  Saying which is the whole value of a receipt.
    record["surface"] = None
    record["no_surface_means"] = (
        "the suite published no population: it was refused, never placed, "
        "or died before the terminal summary. This is not a pass and not a "
        "fail; it is an absent measurement."
    )
    return None


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
    try:
        container = suite_container.command_spec(parts)
        if container is not None:
            return container["inner"], None
    except (ValueError, OSError) as error:
        return None, str(error)
    if len(parts) >= 2 and Path(parts[1]).name == _DEADLINE_WRAPPER:
        if "--" not in parts[2:]:
            return None, (f"`{_DEADLINE_WRAPPER}` was sealed without the `--` "
                          "that separates the command it supervises")
        return _pytest_argv(parts[parts.index("--", 2) + 1:])
    ran = " ".join([program, *parts[1:3]])
    return None, f"`{ran}` is not a pytest invocation this tool parses"


def _targets_of(argv: list) -> list[str]:
    """The test paths ``_command`` put before its first option.

    ``_command`` seals ``<python> -m pytest <targets> -q ...``, so the targets
    are the tokens between ``pytest`` and the first option.  Composing and
    reading them live side by side so the two cannot drift.
    """

    parts = [str(part) for part in argv]
    start = parts.index("pytest") + 1 if "pytest" in parts else len(parts)
    targets = []
    for part in parts[start:]:
        if part.startswith("-"):
            break
        targets.append(part)
    return targets


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


def _container_spec(command):
    """Unwrap only the existing deadline owner and exact container runner."""
    parts = [str(part) for part in command]
    if (len(parts) >= 2 and _PYTHON_NAME.match(Path(parts[0]).name)
            and Path(parts[1]).name == _DEADLINE_WRAPPER and "--" in parts[2:]):
        return _container_spec(parts[parts.index("--", 2) + 1:])
    return suite_container.command_spec(parts)


#: The population's ``counts`` buckets, and the words pytest's summary line
#: spells the same ``terminalreporter.stats`` buckets with.  ``warnings`` and
#: ``deselected`` are on the line and not in the population, so they are not
#: compared.  ``N subtests passed`` is a two-word term on the same line and is
#: read the same way: it names no population bucket, so it is not compared.
_COMPARED_COUNTS = ("passed", "failed", "error", "skipped", "xfailed", "xpassed")
_SUMMARY_WORDS = {"passed": "passed", "failed": "failed", "error": "error",
                  "errors": "error", "skipped": "skipped",
                  "xfailed": "xfailed", "xpassed": "xpassed"}
_SUMMARY_TERM = re.compile(r"^(\d+) ([a-z]+(?: [a-z]+)?)$")
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


def _native_binding_refusal(native: dict, published: "_Publication | None", stdout: str):
    if published is None:
        return "no population publication was read"
    if native.get("query_errors") or native.get("evidence_errors") or not native.get("exit_status_observed"):
        return "native evidence is incomplete: " + "; ".join(
            native.get("query_errors", []) + native.get("evidence_errors", []))
    command = native.get("command", [])
    destination, refusal = _effective_surface_json(command)
    if refusal or destination != str(published.path):
        return "native command does not write the declared population"
    identity = published.payload.get("source_identity") or {}
    expected = native.get("source") or {}
    if (identity.get("verification") != "verified"
            or identity.get("sha256") != expected.get("sha256")
            or identity.get("snapshot_commit") != expected.get("snapshot_commit")
            or published.payload.get("commit") != expected.get("snapshot_commit")):
        return "population source differs from the immutable native launch"
    announced = published_digests(stdout, published.path, POPULATION)
    if published.digest not in announced:
        return "native job did not announce this exact population digest and path"
    counts, _ = _summary_counts(stdout)
    held = published.payload.get("counts") or {}
    if counts is None or any(counts[key] != int(held.get(key) or 0) for key in _COMPARED_COUNTS):
        return "native job terminal summary disagrees with population counts"
    return None


def _resume(name: str, arm: dict, receipt_dir: Path) -> dict:
    record = {"arm": name, "why": arm["why"], "python": None,
              "requires_cuda": bool(arm["strict_cuda"]), "resumed": True,
              "exit_status_observed": False, "returncode": None,
              "exit_status_source": "Flux", "exit_status_note": "native evidence is incomplete"}
    surface_json = receipt_dir / f"surface.{name}.json"
    published = _attach_surface(record, surface_json)
    directory = receipt_dir / f"flux.{name}"
    native = suite_flux.collect(directory)
    record["native_job"] = native
    stdout = (directory / "stdout").read_text() if (directory / "stdout").exists() else ""
    refusal = _native_binding_refusal(native, published, stdout)
    # A native nonzero ending remains conclusive even if publication failed.
    if native.get("exit_status_observed") and native.get("returncode") != 0:
        record["returncode"] = native["returncode"]
    if refusal:
        record["binding_refusal"] = refusal
        record["exit_status_note"] = refusal
    else:
        record.update(returncode=native["returncode"], exit_status_observed=True,
                      exit_status_note="bound to saved native Flux ID " + native["job_id"])
        command, _ = _pytest_argv(native["command"])
        record["pytest_command"] = command
        record["python"] = command[0] if command else None
        record["cpus_used"] = _cpus_of_command(command)
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
        elif counts.get("failed") or counts.get("error"):
            results[record["arm"]] = (
                f"red: {counts.get('failed', 0)} failed, "
                f"{counts.get('error', 0)} error")
        elif record.get("returncode") not in (0, None):
            results[record["arm"]] = f"red: exit {record['returncode']}"
        elif record.get("surface") is None:
            results[record["arm"]] = "no population published"
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
    if any(record.get("status") for record in arms):
        return "not run"
    if any(record.get("surface") is None for record in arms):
        return "incomplete: an arm published no population"
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
time. `yes` is a merge receipt; `no` is a branch's own run; `unknown` means
submission evidence is missing, a checkout/master probe did not answer, or the
row's measured commit is assumed or differs from the captured submission checkout.
New runs persist the comparison and its checkout commit before dispatch and reuse
them on resume; old submit-labelled receipts are not submission evidence. No
remote freshness is implied.

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

`exit` is the status the submitting process observed. Historical `0 (pool)` is a status
this program did not watch and did not guess: the run was resumed, and the
number is the one PrismaBuild's own worker recorded for the action the
population names as its producer, for the attempt whose own output says it
published that population. `not observed` is the remaining case -- a resumed
row with no single finished pool record bound to it, including a population
that names no producer (pre-stamp, or `unknown` source) -- and there the
failure count is still a fact while a zero in it does not make the row green,
because a suite can exit non-zero after a clean summary.

`mode` is how that arm ran, and it changes what the row means as much as the
device does. The legacy `merge_suite.py` wrapper clamps its GPU arm to
`serial`; direct PrismaBuild runs can install scoped xdist and reserve
parallel workers on the same device. Its x86 arm runs `-n <cpus>`, so
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
    # Older direct callers associated their supplied comparison with `commit`.
    # Reassembled receipts always carry the explicit submission association,
    # including None for missing evidence; never fall back from an unknown.
    comparison_commit = population.get("checkout_commit_at_submit", population["commit"])
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
            if record.get("exit_status_source") == "Flux":
                exit_text += " (Flux)"
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
        # The comparison belongs to the checkout observed before dispatch, not
        # the checkout at assembly. Only an established matching measured commit
        # can inherit it; an assumed commit is not submission evidence either.
        if established and comparison_commit and commit == comparison_commit:
            row_head = head_text
        else:
            row_head = "unknown"
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


def _assemble_receipt(arms: list[dict], checkout: Path, *, assembled_by: str,
                      submission_master: dict | None = None) -> dict:
    """The receipt for these arm records: one owner for what a receipt says.

    ``main`` and every tool that submits arms of its own read the same fields
    the same way, so the verdict, the per-arm results and the reading note are
    written here and nowhere else.
    """

    commits = _commits_measured(arms)
    return {
        "schema": "tessera.merge_suite.flux.v2",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "submitted_from": os.uname().nodename,
        "assembled_by": assembled_by,
        "population": _population_of(checkout, submission_master),
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkout", default=str(Path(__file__).resolve().parents[1]),
                    help="tree to test; the x86 arm needs it under /mnt/shared")
    ap.add_argument("--arm", action="append", choices=sorted(ARMS),
                    default=[], help="repeatable; default is both")
    ap.add_argument("--gpu-tag", default="",
                    help="retained GPU placement intent; native GPU admission remains unqualified")
    ap.add_argument("--cpus", type=int, default=1,
                    help="declared native cores and pytest workers, 1..4; default 1")
    ap.add_argument("--mem-gb", type=int, default=16)
    ap.add_argument("--gpu-cpus", type=int, default=None, help="GPU arm process reservation; defaults to --cpus")
    ap.add_argument("--gpu-mem-gb", type=int, default=None, help="GPU aggregate host memory; defaults to --mem-gb")
    ap.add_argument("--gpu-image", default="", help="immutable local image; enables GPU xdist")
    ap.add_argument("--gpu-deps-site", default="", help="readonly scoped pytest/xdist site")
    ap.add_argument("--gpu-deps-sha256", default="", help="suite_dependencies.v1 content seal")
    ap.add_argument("--gpu-cache-dir", default="",
                    help="new owned writable directory on local disk outside the source")
    ap.add_argument("--gpu-data-root", action="append", default=[],
                    help="canonical same-path readonly data mount; repeatable")
    ap.add_argument("--artifact-root", action="append", default=[], metavar="ENV=PATH",
                    help="box_artifacts root override, sealed and mounted; repeatable")
    ap.add_argument("--timeout-s", type=_positive_seconds, default=1200.0,
                    help="positive finite per-attempt inner deadline; TERM then KILL "
                         "after 5s, not a queue/retry lifetime limit")
    ap.add_argument("--wait-s", type=float, default=1800.0)
    ap.add_argument("--out", default="",
                    help="receipt path; default is a timestamped file under "
                         f"{DEFAULT_RECEIPT_ROOT}")
    ap.add_argument("--pytest-arg", action="append", default=[],
                    help="repeatable, passed through to pytest")
    ap.add_argument("--target", action="append", default=[],
                    help="explicit test path or node; repeatable, default tests")
    ap.add_argument("--python", default="", help="explicit CPU suite interpreter")
    ap.add_argument("--record", default="",
                    help="also append one row per arm to this markdown ledger; "
                         "docs/status/suite-populations.md is the one a reader "
                         "of the repo checks")
    ap.add_argument("--resume", default="",
                    help="submit nothing; assemble the receipt from the "
                         "surface.<arm>.json files already in this directory. "
                         "Reads saved native IDs and their terminal records; "
                         "missing evidence stays unobserved")
    ap.add_argument("--dry-run", action="store_true",
                    help="print native payload intentions and submit nothing")
    args = ap.parse_args()
    if args.cpus < 1 or args.mem_gb < 1 or any(value is not None and value < 1
                                              for value in (args.gpu_cpus, args.gpu_mem_gb)):
        ap.error("CPU and memory reservations must be positive")
    if args.gpu_image:
        if not all((args.gpu_deps_site, args.gpu_deps_sha256, args.gpu_cache_dir)):
            ap.error("--gpu-image requires --gpu-deps-site, --gpu-deps-sha256 and --gpu-cache-dir")
    elif any((args.gpu_deps_site, args.gpu_deps_sha256, args.gpu_cache_dir)):
        ap.error("container dependencies/cache require --gpu-image")
    if args.gpu_cache_dir and not args.resume:
        try:
            suite_container.require_local_cache(args.gpu_cache_dir)
        except (OSError, ValueError) as exc:
            ap.error(str(exc))


    args.checkout = Path(args.checkout).resolve()
    wanted = args.arm or sorted(ARMS)
    configured = {name: {**ARMS[name], **({"targets": args.target} if args.target else {}),
                          **({"python": args.python} if args.python and name == "x86" else {})}
                  for name in wanted}
    if not args.resume:
        try:
            for name in wanted:
                _preflight(configured[name], args)
            if not args.dry_run:
                suite_flux.source(args.checkout)
        except (ValueError, OSError) as exc:
            ap.error(str(exc))

    stamp = time.strftime("%Y%m%dT%H%M%S")
    receipt_dir = (Path(args.resume).resolve() if args.resume else
                   Path(args.out).resolve().parent if args.out else DEFAULT_RECEIPT_ROOT / stamp)
    if args.resume and not receipt_dir.is_dir():
        print(f"merge_suite: --resume {receipt_dir} is not a directory",
              file=sys.stderr)
        return 2
    # A dry run composes paths and creates nothing: littering shared storage
    # with an empty directory per invocation is how a receipt root stops being
    # readable.
    if not args.dry_run and not args.resume:
        receipt_dir.mkdir(parents=True, exist_ok=False)
    out = Path(args.out) if args.out else receipt_dir / "receipt.json"

    if args.dry_run and not args.resume:
        submission_master = _submission_master_of(args.checkout)
    else:
        if not args.resume:
            _write_submission_master(args.checkout, receipt_dir)
        submission_master = _read_submission_master(receipt_dir)

    if args.resume:
        arms = [_resume(name, configured[name], receipt_dir) for name in wanted]
    else:
        with ThreadPoolExecutor(max_workers=len(wanted)) as pool:
            futures = {name: pool.submit(_submit, name, configured[name], args,
                                         receipt_dir)
                       for name in wanted}
            arms = []
            for name in wanted:
                try:
                    record = futures[name].result()
                except (ValueError, OSError, subprocess.SubprocessError) as exc:
                    record = _resume(name, configured[name], receipt_dir)
                    record["client_error"] = str(exc)
                arms.append(record)

    receipt = _assemble_receipt(arms, args.checkout,
                                assembled_by="resume" if args.resume else "submit",
                                submission_master=submission_master)
    if args.dry_run:
        print(json.dumps(receipt, indent=2))
        return 0
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
