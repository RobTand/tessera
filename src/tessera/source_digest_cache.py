"""A per-shard sha256 cache for source seals, bound to each shard's stat identity.

A serving part stamps the sha256 of every source shard it reads, and the
merge proves every stamp against one pass over the whole source
(tessera#495). On the GLM-5.3 campaign that re-reads the same unchanged
shards once per part and again per merge: about 21.5 GB of hashing for 3.66
GB of wire per part, plus 642.7 GB per merge (tessera#499). This cache lets a
full read taken once stand for the shard while the shard's stat identity is
unchanged.

What a reused digest claims. It is "the sha256 a recorded full read took while
the shard's inode, size, mtime_ns and ctime_ns were these", not "the sha256
this process read". Every accidental rewrite path changes one of those
fields: an in-place same-size rewrite changes ctime, which ``utime`` cannot
restore, and ``mv`` over changes the inode. A ctime-preserving rewrite (clock
rollback, ``zfs recv``, block-level writes) and bit rot on a reused read are
not detected, so write access to the cache directory is trust base, and
every caller records which digests were reused (:meth:`receipt`).

Conditions, from the tessera#499 design review:

- Only source shard bodies go through this cache. Output shards, config,
  auxiliary files and identity inputs are always hashed.
- The key is ``(leaf name, st_ino, st_size, st_mtime_ns, st_ctime_ns)``.
  ``st_dev`` and the absolute path are recorded, not keyed: the same shard
  has ``st_dev`` 54 on the ZFS server and 64 on an NFS client, with equal
  inode and timestamps.
- Every fingerprint comes from ``open`` plus ``fstat``, never a path
  ``stat``, so an NFS client revalidates attributes (close-to-open) instead
  of answering from its attribute cache.
- A fresh hash re-takes the fingerprint afterwards and raises if it changed.
- A digest is recorded only for a quiescent shard: its newest timestamp must
  be at least ``quiescent_seconds`` older than the moment the hash started,
  which covers the NFS attribute-cache window and clock skew between hosts.
- A corrupt or unknown entry is refused by name, not rehashed around.
- Entries are one file per key, published with ``os.link`` from a private
  temporary file, so concurrent writers never tear an entry and need no lock.
  Two full reads that disagree about one key are refused.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import stat
import tempfile
import threading
import time
from pathlib import Path

SCHEMA = "tessera.source-digest-cache.v1"
RECEIPT_SCHEMA = "tessera.source-digest-receipt.v1"
DEFAULT_QUIESCENT_SECONDS = 300
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


class SourceDigestCache:
    """A directory of stat-bound source shard digests. See the module docstring."""

    def __init__(self, directory, *, source=None,
                 quiescent_seconds: float = DEFAULT_QUIESCENT_SECONDS):
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"source digest cache is not a directory: {directory}")
        resolved = directory.resolve()
        mode = resolved.stat().st_mode
        if mode & stat.S_IWOTH:
            raise ValueError(f"source digest cache is world-writable, and its entries are "
                             f"trusted as source reads: {resolved}")
        if not os.access(resolved, os.W_OK | os.X_OK):
            raise ValueError(f"source digest cache is not writable by this process: {resolved}")
        if source is not None:
            root = Path(source).resolve()
            if resolved == root or root in resolved.parents:
                raise ValueError(f"source digest cache lies inside the source it seals: {resolved}")
        self.directory = resolved
        self.quiescent_ns = int(quiescent_seconds * 1e9)
        self._rows: list = []
        self._lock = threading.Lock()

    @staticmethod
    def fingerprint(path: Path) -> dict:
        with Path(path).open("rb") as handle:
            st = os.fstat(handle.fileno())
        return {"leaf": Path(path).name, "ino": int(st.st_ino), "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns), "ctime_ns": int(st.st_ctime_ns),
                "_dev": int(st.st_dev)}

    @staticmethod
    def _key(fingerprint: dict) -> dict:
        return {k: v for k, v in fingerprint.items() if not k.startswith("_")}

    def _entry_path(self, key: dict) -> Path:
        name = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        return self.directory / f"{name}.json"

    def _read(self, entry: Path, key: dict) -> dict | None:
        try:
            raw = entry.read_bytes()
        except FileNotFoundError:
            return None
        try:
            record = json.loads(raw)
            ok = (isinstance(record, dict) and record.get("schema") == SCHEMA
                  and record.get("algorithm") == "sha256" and record.get("key") == key
                  and isinstance(record.get("digest"), str)
                  and _DIGEST.fullmatch(record["digest"]) is not None)
        except ValueError:
            ok = False
        if not ok:
            raise ValueError(f"source digest cache entry is corrupt or not a {SCHEMA} entry "
                             f"for {key['leaf']}: {entry}; remove it to rehash")
        return record

    def sha256(self, path: Path) -> str:
        """``sha256_file(path)``, or the recorded digest of this exact stat identity."""
        from .serving_parts import sha256_file

        path = Path(path)
        before = self.fingerprint(path)
        key = self._key(before)
        entry = self._entry_path(key)
        record = self._read(entry, key)
        if record is not None:
            self._row({"shard": path.name, "how": "cached", "writer": record.get("writer")})
            return record["digest"]
        started = time.time_ns()
        digest = sha256_file(path)
        finished = time.time_ns()
        if self.fingerprint(path) != before:
            raise ValueError(f"source shard changed while hashing: {path}")
        recorded = max(before["mtime_ns"], before["ctime_ns"]) + self.quiescent_ns <= started
        if recorded:
            self._record(entry, key, digest, {
                "host": socket.gethostname(), "boot_id": _boot_id(), "pid": os.getpid(),
                "resolved_path": str(path.resolve()), "st_dev": before["_dev"],
                "hash_started_at_ns": started, "hash_finished_at_ns": finished,
                "quiescent_seconds": self.quiescent_ns / 1e9})
        self._row({"shard": path.name, "how": "hashed", "recorded": recorded})
        return digest

    def _record(self, entry: Path, key: dict, digest: str, writer: dict) -> None:
        body = json.dumps({"schema": SCHEMA, "algorithm": "sha256", "key": key,
                           "digest": digest, "writer": writer}, indent=2, sort_keys=True)
        fd, temporary = tempfile.mkstemp(prefix=".entry-", dir=self.directory)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o644)
            try:
                os.link(temporary, entry)
            except FileExistsError:
                existing = self._read(entry, key)
                if existing is not None and existing["digest"] != digest:
                    raise ValueError(
                        f"two full reads of {key['leaf']} with one stat identity disagree: "
                        f"{existing['digest']} recorded, {digest} read now ({entry})") from None
        finally:
            os.unlink(temporary)

    def _row(self, row: dict) -> None:
        with self._lock:
            self._rows.append(row)

    def receipt(self) -> dict:
        """How each shard digest this cache served was established, for a receipt."""
        with self._lock:
            rows = sorted(self._rows, key=lambda row: (row["shard"], row["how"]))
        cached = sum(row["how"] == "cached" for row in rows)
        return {"schema": RECEIPT_SCHEMA, "cache": str(self.directory),
                "mode": "stat-bound" if cached else "hashed",
                "cached_shards": cached, "hashed_shards": len(rows) - cached, "shards": rows}
