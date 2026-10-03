"""Bind a qualification module to its hashed, executable-mapped file inode."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


def loaded_native_identity(lib, source, *, expected_sha256: str | None = None) -> dict:
    path = Path(lib.__file__).resolve()
    with path.open("rb") as handle:
        stat = os.fstat(handle.fileno())
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"loaded native bytes changed: {path}")
    matches = []
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and fields[5] == str(path) and "x" in fields[1]:
            major, minor = (int(value, 16) for value in fields[3].split(":"))
            if (major, minor, int(fields[4])) == (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino):
                matches.append(line)
    if not matches:
        raise RuntimeError("returned native module is not mapped from the hashed file inode")
    source = Path(source)
    return dict(path=str(path), sha256=digest, pid=os.getpid(),
                source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                executable_mappings=matches)
