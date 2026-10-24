"""Bounded data-manifest parsing for submitters and storage roles."""
from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import posixpath
import stat
import zlib

from .core import (
    ActionContractError, _exact_mapping, _fail,
    _nonnegative_integer, _sha256, _text,
)


#: A second content-addressed input: the exact bytes an action will read off
#: the shared mount, so a storage-role loop can make them resident before the
#: action is claimed.  It is an ordinary ``action.inputs`` row, which means the
#: action key covers it -- attaching a manifest to an otherwise identical
#: submission produces a different action.
PBCAMPAIGN_DATA_MANIFEST_INPUT_ID = "pbcampaign.data-manifest"


DATA_MANIFEST_SCHEMA_V1 = "prismaquant.prismabuild.data_manifest.v1"


DATA_MANIFEST_SCHEMA_V2 = "prismaquant.prismabuild.data_manifest.v2"


#: Stored bytes retain the original ceiling. Gzip permits larger read lists
#: without an unbounded decompression; parsed JSON objects cost extra memory.
DATA_MANIFEST_MAX_BYTES = 64 * 1024 * 1024


DATA_MANIFEST_MAX_DECODED_BYTES = 512 * 1024 * 1024


DATA_MANIFEST_MAX_ENTRIES = 1_000_000


DATA_MANIFEST_MAX_READS = 4_000_000


_DATA_MANIFEST_KEYS = frozenset(
    {"schema", "produced_by", "mount_prefix", "entries", "entry_count",
     "total_bytes", "annotations"}
)


_DATA_MANIFEST_V2_KEYS = _DATA_MANIFEST_KEYS | {"read_plan"}


_DATA_MANIFEST_READ_PLAN_KEYS = frozenset({"phases", "read_bytes"})


_DATA_MANIFEST_READ_PHASE_KEYS = frozenset(
    {"name", "entry_indices", "bytes", "cumulative_bytes"}
)


#: The one optional v2 phase key beyond the exact set (#1594): a literal
#: ``resident_before_launch: true``.  Validation strips and checks it before
#: the exact-keys check.  Other unknown keys still refuse.
_DATA_MANIFEST_READ_PHASE_PRELAUNCH_KEY = "resident_before_launch"


_DATA_MANIFEST_ENTRY_KEYS = frozenset({"path", "offset", "bytes", "sha256"})


def _manifest_path(value: object, *, prefix: str, where: str) -> str:
    """One absolute path inside the declared mount, named exactly.

    A manifest is submitter-supplied and the loop that reads it runs on the
    storage host, so this is the boundary that decides what that loop may
    open.  Exact identity means what it says: one literal path per entry, no
    glob, no ``..``, no symlink-resolved detour, nothing outside the prefix the
    manifest itself declares.  ``realpath`` is deliberately not consulted --
    the check must be a property of the manifest bytes, identical on every
    host and stable across a remount; the reader opens ``O_NOFOLLOW`` instead,
    which refuses the link rather than following it somewhere this check
    already approved.
    """

    text = _text(value, where=where)
    if not text.startswith(prefix + "/"):
        _fail(f"{where} must be inside {prefix}")
    if posixpath.normpath(text) != text:
        _fail(f"{where} must already be normalized")
    return text


def validate_data_manifest(value: object) -> dict[str, object]:
    """Normalize the file list an action declares it will read.

    The manifest answers one question -- which bytes, in which order -- and
    deliberately answers no others.  ``sha256`` may be null on every entry:
    hashing a terabyte of calibration captures costs more than the residency
    it buys, and integrity is not what this contract carries.  What binds the
    list to the action is that the *manifest file* is content-addressed in the
    CAS like any other input, so the action key covers these bytes exactly.
    """

    if not isinstance(value, Mapping):
        _fail("data manifest must be an object")
    raw_schema = value.get("schema")
    if raw_schema not in (DATA_MANIFEST_SCHEMA_V1, DATA_MANIFEST_SCHEMA_V2):
        _fail("data manifest schema must be a supported version")
    manifest = _exact_mapping(
        value, keys=(_DATA_MANIFEST_V2_KEYS if raw_schema == DATA_MANIFEST_SCHEMA_V2
                     else _DATA_MANIFEST_KEYS), where="data manifest"
    )
    schema = _text(manifest["schema"], where="data manifest schema")
    if not isinstance(manifest["produced_by"], Mapping):
        _fail("data manifest produced_by must be an object")
    if not isinstance(manifest["annotations"], Mapping):
        _fail("data manifest annotations must be an object")
    prefix = _text(manifest["mount_prefix"], where="data manifest mount_prefix")
    if not prefix.startswith("/") or prefix != posixpath.normpath(prefix):
        _fail("data manifest mount_prefix must be a normalized absolute path")
    if prefix == "/":
        _fail("data manifest mount_prefix must name a mount, not the root")
    raw_entries = manifest["entries"]
    if type(raw_entries) is not list or not raw_entries:
        _fail("data manifest entries must be a non-empty array")
    if len(raw_entries) > DATA_MANIFEST_MAX_ENTRIES:
        _fail(
            "data manifest entries exceed "
            f"{DATA_MANIFEST_MAX_ENTRIES}"
        )
    entries: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    total = 0
    for index, raw in enumerate(raw_entries):
        where = f"data manifest entries[{index}]"
        entry = _exact_mapping(
            raw, keys=_DATA_MANIFEST_ENTRY_KEYS, where=where
        )
        path = _manifest_path(
            entry["path"], prefix=prefix, where=f"{where}.path"
        )
        offset = _nonnegative_integer(entry["offset"], where=f"{where}.offset")
        size = _nonnegative_integer(entry["bytes"], where=f"{where}.bytes")
        if size == 0:
            _fail(f"{where}.bytes must be positive")
        digest = (
            None
            if entry["sha256"] is None
            else _sha256(entry["sha256"], where=f"{where}.sha256")
        )
        if (path, offset) in seen:
            _fail("data manifest entries must not repeat a (path, offset)")
        seen.add((path, offset))
        total += size
        entries.append(
            {"path": path, "offset": offset, "bytes": size, "sha256": digest}
        )
    # Both totals are stated by the producer and recomputed here, because the
    # loop budgets ARC against ``total_bytes`` before it opens anything: a
    # total that disagreed with the list would let a manifest reserve one
    # amount of memory and read another.
    if _nonnegative_integer(
        manifest["entry_count"], where="data manifest entry_count"
    ) != len(entries):
        _fail("data manifest entry_count disagrees with entries")
    if _nonnegative_integer(
        manifest["total_bytes"], where="data manifest total_bytes"
    ) != total:
        _fail("data manifest total_bytes disagrees with entries")
    normalized = {
        "schema": schema,
        "produced_by": dict(manifest["produced_by"]),
        "annotations": dict(manifest["annotations"]),
        "mount_prefix": prefix,
        "entries": entries,
        "entry_count": len(entries),
        "total_bytes": total,
    }
    if schema == DATA_MANIFEST_SCHEMA_V2:
        if "phases" in manifest["annotations"]:
            _fail("data manifest v2 uses read_plan, not annotations.phases")
        plan = _exact_mapping(manifest["read_plan"],
                              keys=_DATA_MANIFEST_READ_PLAN_KEYS,
                              where="data manifest read_plan")
        raw_phases = plan["phases"]
        if type(raw_phases) is not list or not raw_phases:
            _fail("data manifest read_plan.phases must be a non-empty array")
        phases: list[dict[str, object]] = []
        names: set[str] = set()
        used: set[int] = set()
        cumulative = 0
        reads = 0
        for index, raw in enumerate(raw_phases):
            where = f"data manifest read_plan.phases[{index}]"
            # A per-phase prelaunch declaration (#1594): the only key beyond
            # the exact set, allowed only as a literal true.  Other values
            # refuse here.  An older core refuses the unknown key outright.
            prelaunch = None
            if (isinstance(raw, Mapping)
                    and _DATA_MANIFEST_READ_PHASE_PRELAUNCH_KEY in raw):
                prelaunch = raw[_DATA_MANIFEST_READ_PHASE_PRELAUNCH_KEY]
                if prelaunch is not True:
                    _fail(f"{where}.resident_before_launch must be true when present")
                raw = {key: value for key, value in raw.items()
                       if key != _DATA_MANIFEST_READ_PHASE_PRELAUNCH_KEY}
            phase = _exact_mapping(raw, keys=_DATA_MANIFEST_READ_PHASE_KEYS,
                                   where=where)
            name = _text(phase["name"], where=f"{where}.name")
            if name.strip() != name or "\x00" in name:
                _fail(f"{where}.name must match a progress phase name")
            if name in names:
                _fail("data manifest read phase names must be unique")
            names.add(name)
            indices = phase["entry_indices"]
            if type(indices) is not list:
                _fail(f"{where}.entry_indices must be an array")
            reads += len(indices)
            if reads > DATA_MANIFEST_MAX_READS:
                _fail(f"data manifest read references exceed {DATA_MANIFEST_MAX_READS}")
            local: set[int] = set()
            size = 0
            for ref in indices:
                ref = _nonnegative_integer(ref, where=f"{where}.entry_indices")
                if ref >= len(entries):
                    _fail(f"{where}.entry_indices references an absent entry")
                if ref in local:
                    _fail(f"{where}.entry_indices repeats an entry within a phase")
                local.add(ref)
                used.add(ref)
                size += int(entries[ref]["bytes"])
            if _nonnegative_integer(phase["bytes"], where=f"{where}.bytes") != size:
                _fail(f"{where}.bytes disagrees with entry references")
            cumulative += size
            if (_nonnegative_integer(phase["cumulative_bytes"],
                                     where=f"{where}.cumulative_bytes") != cumulative):
                _fail(f"{where}.cumulative_bytes disagrees with entry references")
            normalized_phase: dict[str, object] = {
                "name": name, "entry_indices": list(indices),
                "bytes": size, "cumulative_bytes": cumulative}
            if prelaunch is True:
                normalized_phase[_DATA_MANIFEST_READ_PHASE_PRELAUNCH_KEY] = True
            phases.append(normalized_phase)
        if len(used) != len(entries):
            _fail("data manifest read_plan must reference every unique entry")
        if (_nonnegative_integer(plan["read_bytes"], where="data manifest read_plan.read_bytes")
                != cumulative):
            _fail("data manifest read_plan.read_bytes disagrees with phases")
        normalized["read_plan"] = {"phases": phases, "read_bytes": cumulative}
    return normalized


def read_data_manifest(path: str | Path) -> tuple[dict[str, object], str]:
    """Validate plain JSON or one gzip member and return its wire encoding.

    Detection uses the gzip header, never a suffix: a CAS blob has no suffix.
    Stored bytes and decoded bytes are bounded independently, before JSON
    parsing. Trailing bytes, concatenated members and incomplete streams refuse.
    """

    source = Path(path)
    try:
        # Bound the read itself as well as the size check: a file may grow
        # after stat. O_NONBLOCK prevents a FIFO from blocking the submitter.
        fd = os.open(source, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                _fail("data manifest must be a regular file")
            if info.st_size > DATA_MANIFEST_MAX_BYTES:
                _fail(f"data manifest exceeds {DATA_MANIFEST_MAX_BYTES} bytes: {info.st_size}")
            raw = handle.read(DATA_MANIFEST_MAX_BYTES + 1)
    except OSError as exc:
        raise ActionContractError(f"unreadable data manifest: {exc}") from exc
    if len(raw) > DATA_MANIFEST_MAX_BYTES:
        _fail(f"data manifest exceeds {DATA_MANIFEST_MAX_BYTES} bytes")
    encoding = "gzip" if raw.startswith(b"\x1f\x8b") else "identity"
    if encoding == "gzip":
        try:
            decoder = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
            decoded = decoder.decompress(raw, DATA_MANIFEST_MAX_DECODED_BYTES + 1)
        except zlib.error as exc:
            raise ActionContractError(f"invalid gzip data manifest: {exc}") from exc
        if len(decoded) > DATA_MANIFEST_MAX_DECODED_BYTES:
            _fail(f"decoded data manifest exceeds {DATA_MANIFEST_MAX_DECODED_BYTES} bytes")
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            _fail("gzip data manifest must contain exactly one complete member")
        raw = decoded
    try:
        loaded = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise ActionContractError(f"unreadable data manifest: {exc}") from exc
    return validate_data_manifest(loaded), encoding


def load_data_manifest(path: str | Path) -> dict[str, object]:
    """Read a bounded plain or gzip manifest, preserving the v1 return shape."""

    return read_data_manifest(path)[0]



__all__ = [
    'DATA_MANIFEST_MAX_BYTES',
    'DATA_MANIFEST_MAX_DECODED_BYTES',
    'DATA_MANIFEST_MAX_ENTRIES',
    'DATA_MANIFEST_MAX_READS',
    'DATA_MANIFEST_SCHEMA_V1',
    'DATA_MANIFEST_SCHEMA_V2',
    'PBCAMPAIGN_DATA_MANIFEST_INPUT_ID',
    'load_data_manifest',
    'read_data_manifest',
    'validate_data_manifest',
]
