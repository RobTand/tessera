"""Bind a qualification module to its hashed, executable-mapped file inode."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


def _mountinfo_fields() -> list[list[str]]:
    """The single reader of Linux mount provenance for native files."""
    return [line.split() for line in Path("/proc/self/mountinfo").read_text().splitlines()
            if line.strip()]


def native_cache_mount(path) -> tuple[Path, Path, str]:
    """Resolve a cache's nearest existing parent to its deepest Linux mount.

    Missing, malformed or ambiguous provenance cannot identify a mount.
    """
    import re

    cache = Path(path).resolve()
    parent = cache
    while True:
        try:
            parent.stat()
            break
        except FileNotFoundError:
            if parent == parent.parent:
                raise RuntimeError(f"cache path {cache}: no existing parent for mount lookup")
            parent = parent.parent

    matches = []
    for fields in _mountinfo_fields():
        try:
            separator = fields.index("-")
            if separator < 6 or len(fields) < separator + 4:
                raise ValueError("incomplete mount fields")
            mount_id, parent_id = int(fields[0]), int(fields[1])
            if mount_id <= 0 or parent_id <= 0:
                raise ValueError("invalid mount identity")
            mountpoint = Path(re.sub(r"\\([0-7]{3})",
                                    lambda match: chr(int(match[1], 8)), fields[4]))
            if not mountpoint.is_absolute():
                raise ValueError("relative mount point")
            filesystem = fields[separator + 1]
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"cache path {cache}: mount provenance is malformed") from exc
        if parent.is_relative_to(mountpoint):
            matches.append((mountpoint, filesystem, mount_id, parent_id))
    if not matches:
        raise RuntimeError(f"cache path {cache}: mount provenance is not recorded")
    depth = max(len(match[0].parts) for match in matches)
    deepest = [match for match in matches if len(match[0].parts) == depth]
    parents = {mount_id: parent_id for _, _, mount_id, parent_id in deepest}
    covered = {parent_id for mount_id, parent_id in parents.items() if parent_id != mount_id}
    visible = [match for match in deepest if match[2] not in covered]
    if len(parents) != len(deepest) or len(visible) != 1:
        raise RuntimeError(f"cache path {cache}: mount provenance is ambiguous")
    # A stack is one parent chain ending at its visible top, not sibling
    # mounts or a disconnected cycle. Mount IDs are identities, not order.
    seen = set()
    current = visible[0][2]
    while current in parents and current not in seen:
        seen.add(current)
        current = parents[current]
    if len(seen) != len(parents):
        raise RuntimeError(f"cache path {cache}: mount provenance is ambiguous")
    mountpoint, filesystem, _, _ = visible[0]
    return cache, mountpoint, filesystem


def _fd_mount_id(fd: int) -> str:
    """The mount id the kernel serves this held descriptor through."""
    ids = [line.split(":", 1)[1].strip()
           for line in Path(f"/proc/self/fdinfo/{fd}").read_text().splitlines()
           if line.startswith("mnt_id:")]
    if len(ids) != 1:
        raise RuntimeError("native file mount identity is incomplete")
    return ids[0]


def mapped_file_device(fd: int) -> tuple[int, int]:
    """Kernel mapping device for this held FD, via its exact mount identity.

    Some filesystems report a per-subvolume st_dev while /proc/maps reports
    the backing superblock device. FD mount provenance bridges those views
    without dropping the mapped-device or inode check.
    """
    mounts = [fields for fields in _mountinfo_fields() if fields[0] == _fd_mount_id(fd)]
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
