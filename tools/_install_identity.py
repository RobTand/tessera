"""Installed-distribution integrity for one module, using only the standard library.

``verify_install(module, expected_commit)`` refuses unless:

- exactly one installed distribution owns ``module``;
- that distribution is a NON-EDITABLE Git install whose ``direct_url.json`` records
  ``commit_id == expected_commit``;
- every hashed RECORD entry exists and its bytes match the recorded hash;
- the module Python actually imports is owned by that RECORD (a checkout or
  PYTHONPATH shadow, and an editable import, are refused);
- the package directory holds no unrecorded file (``.pyc`` aside).

It runs as ordinary repository code from the commit-bound checkout: nothing is
published, re-read for a self-hash, or located through an environment variable.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata as metadata
import importlib.util
import io
import json
from pathlib import Path

_ALGORITHMS = {"sha256", "sha384", "sha512"}


def _digest(path: Path, algorithm: str) -> bytes:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def _distribution(module: str):
    owners = metadata.packages_distributions().get(module, [])
    if len(owners) != 1:
        raise ValueError(f"installed commit=<unknown>; expected one distribution "
                         f"owning {module}, found {owners}")
    dist = metadata.distribution(owners[0])
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    observed = direct.get("vcs_info", {}).get("commit_id", "<unknown>")
    return owners[0], dist, direct, observed, f"distribution={owners[0]} installed commit={observed}"


def verify_record_bytes(module: str) -> dict:
    """Require the imported module's installed bytes to match their RECORD."""
    name, dist, _direct, observed, identity = _distribution(module)
    # RECORD is read as bytes and parsed with the csv module without newline
    # conversion: Distribution.read_text normalizes carriage returns, and
    # Distribution.files hides missing entries and splits quoted lines.
    try:
        record = dist._path.joinpath("RECORD").read_bytes().decode("utf-8")
    except (AttributeError, OSError, KeyError, UnicodeError) as exc:
        raise ValueError(f"{identity}; installed RECORD is missing or unreadable") from exc
    if not record:
        raise ValueError(f"{identity}; installed RECORD is missing")
    recorded = {}
    for row in csv.reader(io.StringIO(record, newline="")):
        if not row or len(row) > 3:
            raise ValueError(f"{identity}; malformed RECORD row: {row}")
        entry = row[0]
        hash_field = row[1] if len(row) > 1 else ""
        size_field = row[2] if len(row) > 2 else ""
        if size_field:
            try:
                int(size_field)  # grammar only; the size is never compared to bytes
            except ValueError:
                raise ValueError(f"{identity}; malformed RECORD row: {row}") from None
        if not entry:
            if hash_field:
                raise ValueError(f"{identity}; malformed RECORD row: {row}")
            continue
        path = Path(dist.locate_file(entry)).resolve()
        if hash_field:
            algorithm, _, value = hash_field.partition("=")
            if algorithm not in _ALGORITHMS:
                raise ValueError(f"{identity}; unsupported RECORD hash: {entry}")
            try:
                digest = _digest(path, algorithm)
            except FileNotFoundError as exc:
                raise ValueError(f"{identity}; recorded file is missing: {entry}") from exc
            if base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != value:
                raise ValueError(f"{identity}; installed bytes differ from RECORD: {entry}")
            recorded[path] = value
    # Distribution metadata alone does not say which module Python will load.
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None or Path(spec.origin).resolve() not in recorded:
        raise ValueError(f"{identity}; imported {module} is not owned by its RECORD")
    for root in spec.submodule_search_locations or []:
        for path in Path(root).rglob("*"):
            if path.is_file() and path.suffix != ".pyc" and path.resolve() not in recorded:
                raise ValueError(f"{identity}; unrecorded package file: {path}")
    return {"module": module, "distribution": name, "installed_commit": observed,
            "origin": spec.origin, "verified_files": len(recorded)}


def verify_install(module: str, expected_commit: str) -> dict:
    """Require an unambiguous, non-editable Git install at ``expected_commit`` with intact bytes."""
    _name, _dist, direct, observed, identity = _distribution(module)
    vcs = direct.get("vcs_info", {})
    if direct.get("dir_info", {}).get("editable") or vcs.get("vcs") != "git" or observed != expected_commit:
        raise ValueError(f"{identity}; require a non-editable Git install at the reviewed commit "
                         "(local-directory installs do not record a Git commit)")
    evidence = verify_record_bytes(module)
    evidence["expected_commit"] = expected_commit
    return evidence
