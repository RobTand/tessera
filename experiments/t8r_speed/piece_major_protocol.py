"""Closed, versioned inputs for the existing benchmark owner's PM comparison."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

SCHEMA = "tessera.routed_piece_major_comparison.v1"
ARTIFACT = "/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported"
GROUP = "experts.R1024.L10"
ORDER = ["legacy", "piece_major", "piece_major", "legacy"]


def hashed_json(path, digest=None):
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if digest is not None and actual != digest:
        raise ValueError(f"comparison input digest differs: {path}")
    return json.loads(raw), actual


def require_digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("comparison requires a full lowercase SHA-256")


def validate(doc):
    fields = {"schema", "artifact", "group", "ms", "order", "warmup", "iters", "power_s",
              "input_manifest", "source_manifest", "kernel_sha256", "native", "routing", "harness"}
    if set(doc) not in (fields, fields | {"numeric_protocol"}) or doc["schema"] != SCHEMA:
        raise ValueError("unknown comparison schema or fields")
    if (doc["artifact"] != ARTIFACT or doc["group"] != GROUP or doc["ms"] != [1, 2048]
            or doc["order"] != ORDER or doc["warmup"] != 10 or doc["iters"] != 30
            or doc["power_s"] != 30):
        raise ValueError("comparison requires exact A8SE L10/M1,2048 and finite ABBA limits")
    require_digest(doc["kernel_sha256"])
    if set(doc["harness"]) != {"experiments/t8r_speed/bench_t8r.py", "experiments/t8r_speed/bench_t8r.sh",
                                "experiments/t8r_speed/piece_major_protocol.py", "experiments/t8r_speed/pb_staged_store.py"}:
        raise ValueError("comparison requires the complete benchmark owner closure")
    for relative, digest in doc["harness"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("comparison harness member must be checkout-relative")
        require_digest(digest)
    for name in ("input_manifest", "source_manifest", "native", "routing"):
        item = doc[name]
        expected = {"path", "sha256", "origin_path", "files"} if name == "native" else {"path", "sha256"}
        if set(item) != expected or not Path(item["path"]).is_absolute():
            raise ValueError(f"comparison requires one absolute, digest-bound {name}")
        require_digest(item["sha256"])
    if not Path(doc["native"]["origin_path"]).is_absolute() or not doc["native"]["files"]:
        raise ValueError("comparison requires the native bank and its recipe bindings")
    if "numeric_protocol" in doc:
        proof = doc["numeric_protocol"]
        if set(proof) != {"path", "sha256"} or not Path(proof["path"]).is_absolute():
            raise ValueError("timing requires the bound original numeric protocol")
        require_digest(proof["sha256"])
    return doc


def numeric_protocol_sha256(doc, current_digest):
    if "numeric_protocol" not in doc:
        return current_digest
    proof = doc["numeric_protocol"]
    original, digest = hashed_json(proof["path"], proof["sha256"])
    validate(original)
    # Launch/proof wiring may change; every measured semantic input remains exact.
    for key in set(original) - {"harness", "numeric_protocol"}:
        if doc[key] != original[key]:
            raise ValueError(f"timing semantic input differs from accepted numeric protocol: {key}")
    return digest


def load(path):
    doc, digest = hashed_json(path)
    validate(doc)
    harness_identity(doc)
    numeric_protocol_sha256(doc, digest)
    return doc, digest


def harness_identity(doc):
    root = Path(__file__).resolve().parents[2]
    for relative, digest in doc["harness"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"comparison harness changed: {relative}")


def require_options(args, doc, *, stubbed=False):
    if (args.artifact != doc["artifact"] or args.groups != doc["group"] or args.ms != "1,2048"
            or not args.no_graph or args.outputs_only or args.routing or args.single_routing_file
            or args.profile_native_file or args.input_manifest != doc["input_manifest"]["path"]
            or args.warmup != 10 or args.iters != 30 or args.power_s != 30
            or args.comparison_phase not in ("numeric", "timing", "ncu")
            or bool(args.ncu) != (args.comparison_phase == "ncu")):
        raise ValueError("comparison arguments differ from the finite protocol")
    if stubbed:
        raise ValueError("comparison refuses stubbed vLLM")
    choices = {"TESSERA_ROUTED_FUSED": "1", "TESSERA_FUSED_E4M3_MMA": "e4m3",
               "TESSERA_ROUTED_FUSED_WIDE": "1"}
    for key, expected in choices.items():
        if os.environ.get(key) != expected:
            raise ValueError(f"comparison requires {key}={expected}")
    if os.environ.get("BENCH_EXPECT_LIBRARY_SHA256") != doc["native"]["sha256"]:
        raise ValueError("comparison native expectation differs from the protocol")
    hashed_json(doc["input_manifest"]["path"], doc["input_manifest"]["sha256"])


def require_numeric_receipt(path, digest, protocol_sha256):
    if not path or not digest:
        raise ValueError("timing/profiling requires a bound passing numeric receipt")
    require_digest(digest)
    doc, _ = hashed_json(path, digest)
    if (doc.get("schema") != SCHEMA + ".receipt" or doc.get("phase") != "numeric"
            or doc.get("status") != "passed" or doc.get("protocol_sha256") != protocol_sha256
            or doc.get("ms") != [1, 2048] or doc.get("intermediate_bits_equal") is not True
            or doc.get("source_files_unchanged") is not True):
        raise ValueError("numeric receipt does not admit this comparison")
    rows = doc.get("results", [])
    if [r.get("M") for r in rows] != [1, 2048]:
        raise ValueError("numeric receipt does not contain both finite cases")
    for row in rows:
        if (row.get("intermediate_bits_equal") is not True
                or set(row.get("bit_hashes", {})) != {"forward", "mode0", "mode1", "mode2"}
                or set(row.get("input_hashes", {})) != {"x", "ids", "weights"}):
            raise ValueError("numeric receipt lacks its actual intermediate/input bits")
        for digest in list(row["bit_hashes"].values()) + list(row["input_hashes"].values()):
            require_digest(digest)
    return doc


def source_identity(doc, source_root):
    manifest, _ = hashed_json(doc["source_manifest"]["path"], doc["source_manifest"]["sha256"])
    root = Path(source_root).resolve().parent
    observed = {}
    for relative, expected in manifest["files"].items():
        path = root / relative
        stat = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (digest != expected["sha256"] or stat.st_size != expected["bytes"]
                or stat.st_mtime_ns != expected["mtime_ns"]):
            raise ValueError(f"comparison source bytes/timestamps differ: {relative}")
        observed[relative] = dict(expected)
    kernel = observed["src/tessera/serving/csrc/routed_fused_window.cu"]["sha256"]
    if kernel != doc["kernel_sha256"]:
        raise ValueError("comparison kernel differs from the source closure")
    return observed


def native_bank_identity(expected):
    path = Path(expected["path"])
    files = {}
    for name, binding in expected["files"].items():
        if Path(name).name != name:
            raise ValueError("native recipe member must be a basename")
        p = path.parent / name
        st = p.stat()
        observed = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
        if observed != binding:
            raise ValueError(f"comparison native bank changed: {name}")
        files[name] = observed
    return files


def mapped_native_identity(lib, expected):
    path = Path(lib.__file__).resolve()
    if (not str(lib.__file__).startswith("/proc/self/fd/")
            or lib.__spec__.origin != lib.__file__ or str(path) != expected["origin_path"]):
        raise ValueError("comparison native module is not the declared held-FD artifact")
    with path.open("rb") as stream:
        stat = os.fstat(stream.fileno())
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected["sha256"]:
        raise ValueError("comparison loaded native digest differs")
    mappings = []
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and fields[5] == str(path) and "x" in fields[1]:
            major, minor = (int(v, 16) for v in fields[3].split(":"))
            if (major, minor, int(fields[4])) == (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino):
                mappings.append(line)
    if not mappings:
        raise ValueError("comparison native file inode is not executable-mapped")
    files = native_bank_identity(expected)
    return {"path": str(path), "sha256": digest, "pid": os.getpid(),
            "executable_mappings": mappings, "files": files}
