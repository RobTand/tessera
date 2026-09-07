"""Lossless frame interning and exact canonical JSON hashing for raw snapshots.

This is an observer representation, not an allocation owner or resource price.
The expanded JSON bytes remain the evidence identity. Native Torch snapshots
are fresh, recorder-owned Python values; callers must not mutate retained rows.
"""
import hashlib
import json
import sys


LEGACY_SCHEMA = "tessera.full_engine_resource_capture.v1"
COMPACT_SCHEMA = "tessera.full_engine_resource_capture.v2"


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest_parts(parts):
    digest = hashlib.sha256()
    pending = bytearray()
    for part in parts:
        if len(part) >= 65536:
            if pending:
                digest.update(pending)
                pending.clear()
            digest.update(part)
        else:
            pending.extend(part)
            if len(pending) >= 65536:
                digest.update(pending)
                pending.clear()
    if pending:
        digest.update(pending)
    return digest.hexdigest()


def canonical_equal(before, after):
    """Python equality alone conflates 1/1.0/True and signed floating zero."""
    if type(before) is not type(after):
        return False
    if before is after:
        return True
    if type(before) is dict:
        return (before.keys() == after.keys()
                and all(type(key) is str for key in before)
                and all(canonical_equal(value, after[key]) for key, value in before.items()))
    if type(before) in (list, tuple):
        return len(before) == len(after) and all(canonical_equal(a, b) for a, b in zip(before, after))
    if type(before) is float:
        return before.hex() == after.hex()
    return before == after


class SnapshotFramePool:
    """Own one exact frame value and one canonical byte string per unique stack."""
    def __init__(self):
        self.entries = []
        self._by_key, self._by_identity, self._keys = {}, {}, {}
        self._encoded_ids = set()
        self.encoded_bytes = 0

    def intern(self, value):
        entry = self._by_identity.get(id(value))
        if entry is not None and entry["value"] is value:
            return entry
        # The C JSON encoder is also the exact equality key. A recursive
        # Python type-key walk was slower on retained real stack arrays.
        encoded = _json_bytes(value)
        entry = self._by_key.get(encoded)
        if entry is None:
            entry = {"index": len(self.entries), "value": value, "encoded": encoded}
            self.entries.append(entry)
            self._by_key[encoded] = entry
            self._by_identity[id(value)] = entry
            self._encoded_ids.add(id(encoded))
            self.encoded_bytes += len(encoded)
        return entry

    def intern_value(self, value):
        # Torch shares traceback list objects within a returned snapshot.
        # This local map avoids repeatedly walking the same incoming stack,
        # without retaining duplicate frame objects from successive snapshots.
        seen = {}
        def visit(node):
            if type(node) is dict:
                for key, item in node.items():
                    if key == "frames":
                        entry = seen.get(id(item))
                        if entry is None:
                            entry = self.intern(item)
                            seen[id(item)] = entry
                        node[key] = entry["value"]
                    else:
                        visit(item)
            elif type(node) in (list, tuple):
                for item in node:
                    visit(item)
        visit(value)
        return value

    def parts(self, value):
        """Emit exactly json.dumps(sort_keys=True,separators=(',',':')) bytes."""
        if type(value) is dict and all(type(key) is str for key in value):
            # Allocator/API rows have scalar fields and short coordinate
            # arrays. Encode those together in C, preserving a separately
            # shared frame chunk, instead of calling JSON once per scalar.
            if all(key == "frames" or _flat_json_field(item) for key, item in value.items()):
                if "frames" not in value:
                    yield _json_bytes(value)
                else:
                    before = {key: item for key, item in value.items() if key < "frames"}
                    after = {key: item for key, item in value.items() if key > "frames"}
                    yield (_json_bytes(before)[:-1] + b"," if before else b"{") + b'"frames":'
                    yield self.intern(value["frames"])["encoded"]
                    yield b"," + _json_bytes(after)[1:] if after else b"}"
                return
            yield b"{"
            for index, key in enumerate(sorted(value)):
                if index:
                    yield b","
                if key not in self._keys:
                    self._keys[key] = _json_bytes(key)
                yield self._keys[key]
                yield b":"
                if key == "frames":
                    yield self.intern(value[key])["encoded"]
                else:
                    yield from self.parts(value[key])
            yield b"}"
        elif type(value) in (list, tuple):
            if all(type(item) not in (dict, list, tuple) for item in value):
                yield _json_bytes(value)
                return
            yield b"["
            for index, item in enumerate(value):
                if index:
                    yield b","
                yield from self.parts(item)
            yield b"]"
        else:
            yield _json_bytes(value)

    def digest(self, value):
        return _digest_parts(self.parts(value))

    def row_parts(self, row):
        pending = bytearray()
        parts = []
        for part in self.parts(row):
            if id(part) in self._encoded_ids and len(part) >= 256:
                if pending:
                    parts.append(bytes(pending))
                    pending.clear()
                parts.append(part)
            else:
                pending.extend(part)
        if pending:
            parts.append(bytes(pending))
        return tuple(parts)


def _flat_json_field(value):
    if type(value) in (list, tuple):
        return all(type(item) not in (dict, list, tuple) for item in value)
    return type(value) is not dict


class CanonicalHistoryPrefix:
    """Reuse exact row chunks, including shared frame bytes, across snapshots."""
    def __init__(self, frames=None):
        self.frames = frames if frames is not None else SnapshotFramePool()
        self._previous, self._rows, self._lengths = [], [], []
        self._peak_logical_bytes = 0

    def observe(self, history, *, already_interned=False):
        if not already_interned:
            self.frames.intern_value(history)
        rows, lengths, encoded, reused = [], [], 0, 0
        for index, row in enumerate(history):
            if index < len(self._previous) and canonical_equal(self._previous[index], row):
                parts, length = self._rows[index], self._lengths[index]
                reused += 1
            else:
                parts = self.frames.row_parts(row)
                length = sum(map(len, parts))
                encoded += 1
            rows.append(parts)
            lengths.append(length)
        def prefix_parts():
            yield b"["
            for index, parts in enumerate(rows):
                if index:
                    yield b","
                yield from parts
            yield b"]"
        digest = _digest_parts(prefix_parts())
        logical_bytes = sum(lengths)
        self._peak_logical_bytes = max(self._peak_logical_bytes, logical_bytes)
        self._previous, self._rows, self._lengths = list(history), rows, lengths
        return {"history_prefix_sha256": digest,
                "serialized_history_prefix_bytes": logical_bytes + max(0, len(rows) - 1) + 2,
                "encoded_rows": encoded, "reused_rows": reused,
                "retained_encoded_row_bytes": logical_bytes,
                "peak_retained_encoded_row_bytes": self._peak_logical_bytes,
                "unique_frame_encoded_bytes": self.frames.encoded_bytes,
                "unique_frame_values": len(self.frames.entries),
                "row_cache_shallow_container_bytes": sys.getsizeof(rows) + sys.getsizeof(lengths) + sys.getsizeof(self._previous)
                    + sum(sys.getsizeof(parts) for parts in rows),
                "memory_scope": "logical row bytes share frame chunks; shallow containers exclude scalar byte objects, frame objects and allocator overhead; process/PB peak is separate"}

    def retained_memory(self):
        chunks = {id(part): part for row in self._rows for part in row}
        return {"unique_row_chunk_payload_bytes": sum(map(len, chunks.values())),
                "unique_row_chunk_python_bytes": sum(sys.getsizeof(part) for part in chunks.values()),
                "row_cache_container_python_bytes": sum(map(sys.getsizeof, (self._rows, self._lengths, self._previous)))
                    + sum(sys.getsizeof(row) for row in self._rows),
                "logical_row_payload_bytes": sum(self._lengths),
                "scope": "retained chunk objects and cache containers only; excludes frame values, interning keys and allocator overhead; process/PB peak is required"}


def compact_capture(raw, frames=None):
    if raw.get("schema") != LEGACY_SCHEMA:
        raise ValueError("only a legacy expanded capture can be compacted")
    if "frame_dictionary" in raw:
        raise ValueError("expanded capture already has a frame dictionary")
    pool = frames if frames is not None else SnapshotFramePool()
    def visit(node):
        if type(node) is dict:
            if "frames_ref" in node:
                raise ValueError("expanded capture collides with reserved frames_ref")
            return {"frames_ref" if key == "frames" else key:
                    pool.intern(value)["index"] if key == "frames" else visit(value)
                    for key, value in node.items()}
        if type(node) in (list, tuple):
            return [visit(item) for item in node]
        return node
    result = visit(raw)
    result["schema"] = COMPACT_SCHEMA
    result["frame_dictionary"] = [{"sha256": hashlib.sha256(entry["encoded"]).hexdigest(),
                                    "value": entry["value"]} for entry in pool.entries]
    return result


def expand_capture(raw):
    if raw.get("schema") == LEGACY_SCHEMA:
        return raw
    if raw.get("schema") != COMPACT_SCHEMA:
        raise ValueError("unsupported full-engine capture schema")
    table = raw.get("frame_dictionary")
    if type(table) is not list:
        raise ValueError("missing compact frame dictionary")
    for entry in table:
        if type(entry) is not dict or set(entry) != {"sha256", "value"}:
            raise ValueError("malformed compact frame entry")
        if hashlib.sha256(_json_bytes(entry["value"])).hexdigest() != entry["sha256"]:
            raise ValueError("compact frame dictionary digest mismatch")
    def visit(node):
        if type(node) is dict:
            result = {}
            for key, value in node.items():
                if key == "frames_ref":
                    if "frames" in node or type(value) is not int or not 0 <= value < len(table):
                        raise ValueError("ambiguous or out-of-range compact frame reference")
                    result["frames"] = table[value]["value"]
                else:
                    result[key] = visit(value)
            return result
        if type(node) is list:
            return [visit(item) for item in node]
        return node
    result = visit({key: value for key, value in raw.items() if key != "frame_dictionary"})
    result["schema"] = LEGACY_SCHEMA
    return result


def canonical_snapshot_digest(value):
    pool = SnapshotFramePool()
    pool.intern_value(value)
    return pool.digest(value)
