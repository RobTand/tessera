"""Bind a qualification module to its hashed, executable-mapped file inode."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


def mapped_file_device(fd: int) -> tuple[int, int]:
    """Kernel mapping device for this held FD, via its exact mount identity.

    Some filesystems report a per-subvolume st_dev while /proc/maps reports
    the backing superblock device. FD mount provenance bridges those views
    without dropping the mapped-device or inode check.
    """
    ids = [line.split(":", 1)[1].strip()
           for line in Path(f"/proc/self/fdinfo/{fd}").read_text().splitlines()
           if line.startswith("mnt_id:")]
    if len(ids) != 1:
        raise RuntimeError("native file mount identity is incomplete")
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == ids[0]:
            mounts.append(fields)
    if len(mounts) != 1 or len(mounts[0]) < 3:
        raise RuntimeError("native file mount identity is not recorded")
    try:
        device = tuple(int(value) for value in mounts[0][2].split(":"))
    except ValueError as exc:
        raise RuntimeError("native file mount device is malformed") from exc
    if len(device) != 2:
        raise RuntimeError("native file mount device is malformed")
    return device


def loaded_native_identity(lib, source, *, expected_sha256: str | None = None) -> dict:
    path = Path(lib.__file__).resolve()
    with path.open("rb") as handle:
        stat = os.fstat(handle.fileno())
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise RuntimeError(f"loaded native bytes changed: {path}")
        device = mapped_file_device(handle.fileno())
        matches = []
        for line in Path("/proc/self/maps").read_text().splitlines():
            fields = line.split(None, 5)
            if len(fields) == 6 and fields[5] == str(path) and "x" in fields[1]:
                major, minor = (int(value, 16) for value in fields[3].split(":"))
                if (major, minor, int(fields[4])) == (*device, stat.st_ino):
                    matches.append(line)
        if not matches:
            raise RuntimeError("returned native module is not mapped from the hashed file inode")
    source = Path(source)
    return dict(path=str(path), sha256=digest, pid=os.getpid(),
                source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                executable_mappings=matches)
