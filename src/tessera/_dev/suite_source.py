"""Read-only source identity for test populations, separate from Git history.

PB snapshots contain one action-specific generated closure stamp. Its name is
not ownership proof: exclude it only after verifying the exact sealed action.
Unknown provenance or a modified materialized tree never establishes equality.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess

#: A receipt identifier, not an import path.  Receipts carrying this
#: string are already on /mnt/shared and are read back by
#: ``tools/merge_suite.py``, so the module moving under ``_dev`` does
#: not move the wire; the version suffix is what a change would use.
SCHEMA = "tessera.suite_source.v1"
REQUEST_ROOT = Path("/mnt/shared/prismabuild-fleet/cas/requests")


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


def _verified_stamp(root, commit, request_root, owner):
    """The directory prefix locates requests; only the full checks verify one."""
    match = re.fullmatch(r"([0-9a-f]{12})\.[^/]+", root.parent.name)
    _require(root.name == "checkout" and match, "PB action locator is unavailable")
    prefix = match.group(1)
    candidates = list(Path(request_root, prefix[:2]).glob(prefix + "*.json"))
    _require(len(candidates) == 1, "PB action lookup is missing or ambiguous")
    raw = candidates[0].read_bytes()
    action = json.loads(raw)
    _require(isinstance(action, dict), "PB action is not an object")
    key = action["action_key"]
    body = {name: value for name, value in action.items() if name != "action_key"}
    _require(re.fullmatch(r"[0-9a-f]{64}", key) and key == candidates[0].stem
             and key == _digest(body), "PB action key does not verify")
    _require(action["schema"] == "prismaquant.prismabuild.action.v2"
             and action["task"]["definition_id"] == "fleet/pbrun"
             and action["task"]["definition_version"] == "v1", "unsupported PB action")
    params = action["params"]
    snapshot = params["checkout_snapshot"]
    _require(snapshot["schema"] == "prismaquant.prismabuild.pbrun_checkout_snapshot.v1"
             and snapshot["commit"] == commit, "PB action names another snapshot")
    _require(snapshot["input"] in action["inputs"], "PB snapshot input is not sealed")
    _require(params["cwd"] == snapshot["subdirectory"], "PB logical cwd differs")
    variables = action["environment"]["variables"]
    _require(isinstance(variables, dict), "PB environment variables are not an object")
    _require(owner and owner == variables.get("PRISMABUILD_CONTAINER_OWNER"),
             "PB action owner differs or is unavailable")
    closure = action["code_closure"]
    _require(isinstance(closure, dict), "PB closure is not an object")
    closure_body = {name: value for name, value in closure.items() if name != "closure_sha256"}
    _require(closure["schema"] == "prismaquant.prismabuild.code_closure.v1"
             and closure["closure_sha256"] == _digest(closure_body)
             and len(closure["files"]) == 1, "PB closure does not verify")
    entry = closure["files"][0]
    filename = entry["path"]
    _require(re.fullmatch(r"\.pbrun-closure\.[0-9a-f]{16}\.json", filename),
             "PB closure is not the generated stamp")
    subdirectory = Path(snapshot["subdirectory"])
    _require(not subdirectory.is_absolute() and ".." not in subdirectory.parts,
             "PB snapshot subdirectory escapes the source")
    relative = subdirectory / filename
    stamp_path = root / relative
    _require(stat.S_ISREG(stamp_path.lstat().st_mode), "PB stamp is not a regular file")
    stamp_raw = stamp_path.read_bytes()
    _require(len(stamp_raw) == entry["bytes"]
             and hashlib.sha256(stamp_raw).hexdigest() == entry["sha256"],
             "PB stamp differs from its sealed closure")
    _require(_git(root, "show", f"{commit}:{relative.as_posix()}") == stamp_raw,
             "PB stamp differs from its snapshot blob")
    stamp = json.loads(stamp_raw)
    _require(isinstance(stamp, dict) and set(stamp) == {"cwd", "head", "dirty_sha256"}
             and stamp["cwd"] == params["cwd"]
             and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", stamp["head"])
             and re.fullmatch(r"[0-9a-f]{64}", stamp["dirty_sha256"]),
             "PB stamp payload is not generated closure metadata")
    identity = {name: stamp[name] for name in ("head", "dirty_sha256")}
    # The published pbrun v1 name binds command, demand, environment, source
    # submission identity and placement. No matching-name glob is excluded.
    fingerprint = hashlib.sha256(json.dumps(
        [params["command"], params["cwd"], params["demand"], variables,
         identity, params["placement"]], sort_keys=True).encode()).hexdigest()[:16]
    _require(filename == f".pbrun-closure.{fingerprint}.json"
             and action["task"]["result_path"] == f"pbrun_result.{fingerprint}.txt",
             "PB stamp name does not match its action fingerprint")
    return {**entry, "path": relative.as_posix(), "action_key": key,
            "request_sha256": hashlib.sha256(raw).hexdigest()}


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
        if raw_path != excluded:
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


def agreed_source(record, workers):
    """One identity for a population several processes measured, or unknown.

    Under xdist the canonical population is written by the controller, which
    executed none of the tests it reports: its hash is a fact about the
    controller's filesystem and becomes a fact about the measured source only
    when every process that did the executing agrees.  A worker that reported
    nothing establishes no agreement either, so it is named rather than
    ignored.
    """

    if not workers:
        return record
    verdicts, disputed = {}, []
    for name in sorted(workers):
        other = workers[name] if isinstance(workers[name], dict) else {}
        if other.get("verification") != "verified" or not other.get("sha256"):
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


def measured_source(checkout, *, request_root=REQUEST_ROOT, owner=None,
                    entry=None):
    """Verified effective source hash, or an explicit unknown with the raw ID.

    This performs no repository writes and no full CAS/queue scan. Dirty input
    changes already included in a PB snapshot affect its actual file hashes;
    changes made after materialization refuse equality instead of hiding them.

    ``entry`` is the identity captured before the code under test was
    imported.  Given one, the answer is about the *span* between the two
    measurements rather than about this instant, and it is verified only if
    the source did not move across it.
    """
    record = {"schema": SCHEMA, "snapshot_commit": None, "sha256": None,
              "verification": "unknown", "excluded_metadata": []}
    try:
        root = Path(os.fsdecode(_git(checkout, "rev-parse", "--show-toplevel").rstrip(b"\n")))
        commit = _git(root, "rev-parse", "HEAD").decode().strip()
        record["snapshot_commit"] = commit
        status_args = ("status", "--porcelain=v1", "--untracked-files=all", "-z")
        _require(not _git(root, *status_args), "source checkout is dirty")
        is_snapshot = _git(root, "log", "-1", "--format=%s").strip() == b"PrismaBuild pbrun checkout snapshot v1"
        excluded = None
        if is_snapshot:
            stamp = _verified_stamp(root, commit, request_root,
                                    os.environ.get("PRISMABUILD_CONTAINER_OWNER") if owner is None else owner)
            excluded = os.fsencode(stamp["path"])
        files = _source_files(root, commit, excluded)
        _require(_git(root, "rev-parse", "HEAD").decode().strip() == commit
                 and not _git(root, *status_args), "source changed while it was measured")
        record.update(verification="verified", sha256=_digest({"schema": SCHEMA, "files": files}),
                      files_verified=len(files), excluded_metadata=[stamp] if is_snapshot else [])
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        record["reason"] = str(error)
    return record if entry is None else _bound_to_entry(record, entry)
