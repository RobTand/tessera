"""Read-only source identity for test populations, separate from Git history.

A checkout an executor materialized can carry files the executor generated,
for example a closure stamp beside the source.  Such a file is not source, and
two runs of one source can carry different ones, so the identity leaves it out.
It leaves a file out only when a declared **source verifier** vouches for it,
never because of its name.

The verifier is a command, declared in :data:`VERIFIER_ENV` as a shell-quoted
argv.  Tessera runs it with the checkout root and its HEAD commit appended.  It
must exit 0 and print one JSON object on stdout whose ``generated`` member lists
the generated files, each with at least ``path`` (relative to the checkout
root), ``bytes`` and ``sha256``; other fields are recorded as given.  An empty
list means the executor generated nothing.  Tessera still checks every listed
file itself: a regular file inside the checkout, with those bytes and that
digest, equal to its blob at the commit.

Fail closed: a declared verifier that cannot run, exits non-zero, prints
anything else, or lists a file that fails those checks makes the identity
``unknown``.  With no verifier declared, nothing is left out and every tracked
file is source.  Unknown provenance or a modified materialized tree never
establishes equality.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess

#: A receipt identifier, not an import path.  Receipts carrying this
#: string are already on /mnt/shared and are read back by
#: ``tools/merge_suite.py``, so the module moving under ``_dev`` does
#: not move the wire; the version suffix is what a change would use.
SCHEMA = "tessera.suite_source.v1"
#: The environment variable that declares the source verifier command.
VERIFIER_ENV = "TESSERA_SOURCE_VERIFIER"
#: How long the verifier may take before the identity is ``unknown``.
VERIFIER_TIMEOUT_S = 120

#: ``measured_source``'s default: read the verifier from :data:`VERIFIER_ENV`.
FROM_ENVIRONMENT = object()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args],
                                   stderr=subprocess.DEVNULL, timeout=10)


def declared_verifier(env=None):
    """The verifier argv :data:`VERIFIER_ENV` declares, or ``None`` if unset.

    A variable that is set but names no command raises: a declaration that
    cannot be read is not the absence of one.
    """

    value = (os.environ if env is None else env).get(VERIFIER_ENV)
    if value is None:
        return None
    argv = shlex.split(value)
    _require(argv, f"{VERIFIER_ENV} is set but names no command")
    return argv


def _generated_files(root, commit, verifier):
    """The files ``verifier`` says the executor generated, each checked here."""

    try:
        done = subprocess.run([*verifier, str(root), commit], capture_output=True,
                              timeout=VERIFIER_TIMEOUT_S, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"source verifier did not run: {error}") from error
    if done.returncode != 0:
        tail = done.stderr.decode("utf-8", "replace").strip()[-400:]
        raise ValueError(f"source verifier refused (exit {done.returncode}): {tail}")
    try:
        report = json.loads(done.stdout)
    except ValueError as error:
        raise ValueError(f"source verifier printed no JSON object: {error}") from error
    _require(isinstance(report, dict) and isinstance(report.get("generated"), list),
             "source verifier report has no generated list")
    generated, seen = [], set()
    for entry in report["generated"]:
        _require(isinstance(entry, dict), "source verifier entry is not an object")
        path = entry.get("path")
        _require(isinstance(path, str) and path, "source verifier entry names no path")
        relative = PurePosixPath(path)
        _require(not relative.is_absolute() and ".." not in relative.parts
                 and relative.as_posix() == path and path not in seen,
                 "source verifier path escapes the checkout or repeats")
        seen.add(path)
        size, sha = entry.get("bytes"), entry.get("sha256")
        _require(type(size) is int and isinstance(sha, str)
                 and re.fullmatch(r"[0-9a-f]{64}", sha),
                 "source verifier entry has no bytes and sha256")
        on_disk = Path(root, path)
        _require(stat.S_ISREG(on_disk.lstat().st_mode),
                 "generated file is not a regular file")
        raw = on_disk.read_bytes()
        _require(len(raw) == size and hashlib.sha256(raw).hexdigest() == sha,
                 "generated file differs from what the verifier vouched for")
        _require(_git(root, "show", f"{commit}:{path}") == raw,
                 "generated file differs from its blob at the commit")
        generated.append(entry)
    return generated


def _source_files(root, commit, excluded):
    """Hash actual bytes and modes; verify they still equal every HEAD blob."""
    files = []
    roster = _git(root, "ls-tree", "-rz", "--full-tree", commit)
    for entry in roster.split(b"\0"):
        if not entry:
            continue
        header, raw_path = entry.split(b"\t", 1)
        mode, kind, oid = header.split(b" ")
        _require(kind == b"blob", "source contains an unsupported non-blob entry")
        path = root / os.fsdecode(raw_path)
        metadata = path.lstat()
        digest = hashlib.sha256()
        git_digest = hashlib.new("sha1" if len(oid) == 40 else "sha256")
        if mode == b"120000":
            _require(stat.S_ISLNK(metadata.st_mode), "source symlink mode changed")
            data = os.fsencode(os.readlink(path))
            git_digest.update(b"blob " + str(len(data)).encode() + b"\0" + data)
            digest.update(data)
        else:
            actual_mode = b"100755" if metadata.st_mode & stat.S_IXUSR else b"100644"
            _require(stat.S_ISREG(metadata.st_mode) and mode == actual_mode,
                     "source file mode changed")
            git_digest.update(b"blob " + str(metadata.st_size).encode() + b"\0")
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    git_digest.update(chunk)
                    digest.update(chunk)
        _require(git_digest.hexdigest().encode() == oid, "source bytes differ from snapshot")
        if raw_path not in excluded:
            files.append([raw_path.hex(), mode.decode(), digest.hexdigest()])
    return files


def _bound_to_entry(record, entry):
    """Bind a publication-time measurement to the one taken at suite entry.

    Hashing a clean tree proves the bytes are HEAD's bytes *now*.  It does not
    prove they are the bytes the run imported: a shared checkout can be
    fast-forwarded cleanly while a suite is running, and the modules Python
    already holds do not move with it.  The stability check inside
    ``measured_source`` covers its own hashing interval and nothing longer, so
    the span the receipt is about is bracketed here instead -- entry identity
    in, publication identity out, and equality between them is the attestation
    (#219).  Anything else is an explicit unknown that keeps both hashes.
    """

    span = {"entry_sha256": entry.get("sha256"),
            "entry_snapshot_commit": entry.get("snapshot_commit"),
            "entry_verification": entry.get("verification"),
            "agrees": False}
    if (entry.get("verification") == "verified"
            and record.get("verification") == "verified"
            and entry.get("sha256") == record.get("sha256")):
        span["agrees"] = True
        return {**record, "measurement_span": span}
    if record.get("verification") != "verified":
        reason = ("source at suite entry could not be bound: "
                  + str(record.get("reason", "publication is unverified")))
    elif entry.get("verification") != "verified":
        reason = ("the source identity captured at suite entry is not "
                  "verified (" + str(entry.get("reason", "unknown")) + "), so "
                  "this publication attests bytes nothing bound to the run")
    else:
        reason = ("source changed between suite entry and publication: "
                  + str(entry.get("sha256"))[:12] + " -> "
                  + str(record.get("sha256"))[:12] + "; the tests ran against "
                  "the first and this tree is the second")
    return {**record, "verification": "unknown", "sha256": None,
            "excluded_metadata": [], "measurement_span": span,
            "reason": reason}


def is_entry_bound(record):
    """Is this identity a *span*, or a bare sample of one instant?

    ``measured_source(root, entry=...)`` is the only thing that produces a
    ``measurement_span``; ``measured_source(root)`` alone answers "these are
    HEAD's bytes now", which says nothing about the tree a run imported.  The
    distinction is therefore structural rather than a flag someone has to
    remember to set, and this is the one place that reads it.
    """

    return (isinstance(record, dict)
            and isinstance(record.get("measurement_span"), dict))


def agreed_source(record, workers):
    """One identity for a population several processes measured, or unknown.

    Under xdist the canonical population is written by the controller, which
    executed none of the tests it reports: its hash is a fact about the
    controller's filesystem and becomes a fact about the measured source only
    when every process that did the executing agrees.  A worker that reported
    nothing establishes no agreement either, so it is named rather than
    ignored.

    Nor does an *entry* identity establish it (#291).  A worker publishes one
    before it has run anything, so that a process which never finishes is still
    distinguishable from one that never spoke; it is a seed, and accepting it as
    the worker's answer is exactly the failure that let a worker whose own share
    said ``unknown`` be published as agreeing.  Only an entry-BOUND record --
    one that measured the span across the tests the worker ran -- counts.
    """

    if not workers:
        return record
    verdicts, disputed = {}, []
    for name in sorted(workers):
        other = workers[name] if isinstance(workers[name], dict) else {}
        if not other:
            verdicts[name] = "reported no source identity"
            disputed.append(name)
        elif not is_entry_bound(other):
            verdicts[name] = ("reported only an unbound entry identity, never "
                              "measured across the tests it ran")
            disputed.append(name)
        elif other.get("verification") != "verified" or not other.get("sha256"):
            verdicts[name] = "did not establish a verified source identity"
            disputed.append(name)
        elif other.get("sha256") != record.get("sha256"):
            verdicts[name] = "measured " + str(other["sha256"])[:12]
            disputed.append(name)
        else:
            verdicts[name] = "agrees"
    if not disputed and record.get("verification") == "verified":
        return {**record, "workers": verdicts}
    return {**record, "verification": "unknown", "sha256": None,
            "excluded_metadata": [], "workers": verdicts,
            "reason": ("the processes that executed this population did not "
                       "agree on its source: "
                       + "; ".join(f"{name} {verdicts[name]}"
                                   for name in disputed))}


def measured_source(checkout, *, verifier=FROM_ENVIRONMENT, entry=None):
    """Verified effective source hash, or an explicit unknown with the raw ID.

    This performs no repository writes. Dirty input changes already included in
    a materialized snapshot affect its actual file hashes; changes made after
    materialization refuse equality instead of hiding them.

    ``verifier`` is the source verifier argv (see the module docstring), or
    ``None`` for none; by default it is read from :data:`VERIFIER_ENV`.  The
    files it vouches for are left out of the hash and recorded, as it reported
    them, in ``excluded_metadata``.

    ``entry`` is the identity captured before the code under test was
    imported.  Given one, the answer is about the *span* between the two
    measurements rather than about this instant, and it is verified only if
    the source did not move across it.
    """
    record = {"schema": SCHEMA, "snapshot_commit": None, "sha256": None,
              "verification": "unknown", "excluded_metadata": []}
    try:
        if verifier is FROM_ENVIRONMENT:
            verifier = declared_verifier()
        root = Path(os.fsdecode(_git(checkout, "rev-parse", "--show-toplevel").rstrip(b"\n")))
        commit = _git(root, "rev-parse", "HEAD").decode().strip()
        record["snapshot_commit"] = commit
        status_args = ("status", "--porcelain=v1", "--untracked-files=all", "-z")
        _require(not _git(root, *status_args), "source checkout is dirty")
        generated = [] if verifier is None else _generated_files(root, commit, verifier)
        files = _source_files(root, commit,
                              {os.fsencode(item["path"]) for item in generated})
        _require(_git(root, "rev-parse", "HEAD").decode().strip() == commit
                 and not _git(root, *status_args), "source changed while it was measured")
        record.update(verification="verified", sha256=_digest({"schema": SCHEMA, "files": files}),
                      files_verified=len(files), excluded_metadata=generated)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        record["reason"] = str(error)
    return record if entry is None else _bound_to_entry(record, entry)
