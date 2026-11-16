"""Pinned inputs and receipt I/O for the numerics jobs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

COMMIT = "8bf496ab"
KERNEL = "src/tessera/serving/csrc/routed_fused_window.cu"
KERNEL_SHA256 = "8a1536dfbc3df70edbdc37001a0db6947e97d9263926bc85a4d4d3ca231eae4f"
IMAGES = {
    "G": {"local_id": "9e6882df27dd", "image_digest": "sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5",
          "vllm": "0.28.1rc1.dev397+gfd4a15126.d20260904", "torch": "2.13.0+cu130",
          "libraries": ["e2m1", "e4m3mma", "e4m3"]},
    "N": {"local_id": "acc684c06ec7", "image_digest": "sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a",
          "vllm": "0.30.1rc1.dev336+gaf5b4857e.d20260929", "torch": "2.13.0+cu130",
          "libraries": ["value"]},
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_source(root):
    root = Path(root)
    if sha256(root / KERNEL) != KERNEL_SHA256:
        raise ValueError("kernel source sha256 differs from the cell")
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    want = subprocess.check_output(["git", "rev-parse", COMMIT], cwd=root, text=True).strip()
    if rev != want or subprocess.check_output(["git", "status", "--porcelain", "--", "src"], cwd=root):
        raise ValueError("kernel source is not clean commit 8bf496ab")
    return rev


def check_runtime(image):
    expected = IMAGES[image]
    actual = {name: importlib.metadata.version(name) for name in ("torch", "vllm")}
    if any(actual[name] != expected[name] for name in actual):
        raise ValueError(f"runtime differs from the cell: {actual}")
    return {**expected, **actual}


def register_native_ops(image):
    import vllm._custom_ops  # Registers vLLM's compiled torch.ops._C operators.

    checks = []
    if image == "G":
        from tessera.serving.native_ops import require_native_fp8_quant, require_native_fp4_quant

        for require in (require_native_fp8_quant, require_native_fp4_quant):
            require("numerics rows")
            checks.append(require.__name__)
    return {"result": "PASS", "registration": "import vllm._custom_ops", "requirements": checks}


class Invalid(ValueError):
    pass


class BoundaryError(RuntimeError):
    def __init__(self, record, error):
        self.record = record
        self.attributed_to = record["call"] if record["kind"] == "tessera" and record["sync_before"] and record["sync_after"] is False else None
        super().__init__(str(error))


class BoundaryRecorder:
    """Synchronize each call and retain the boundary that first reports an error."""

    def __init__(self, synchronize, records):
        self.synchronize, self.records = synchronize, records
        self.enabled = True
        self.attribute = True

    def call(self, name, kind, function):
        if not self.enabled:
            return function()
        record = {"call": name, "kind": kind, "sync_before": None, "sync_after": None}
        self.records.append(record)
        try:
            self.synchronize()
            record["sync_before"] = True
        except Exception as error:
            record["sync_before"] = False
            record["message"] = str(error)
            raise BoundaryError(record, error) from error
        failure = None
        try:
            result = function()
        except Exception as error:
            failure = error
        nested = failure if isinstance(failure, BoundaryError) else None
        try:
            self.synchronize()
            record["sync_after"] = True
        except Exception as error:
            record["sync_after"] = False
            failure = error
        if failure is not None:
            record["message"] = str(failure)
            error = nested or BoundaryError(record, failure)
            if not self.attribute:
                error.attributed_to = None
            if nested is not None:
                raise nested
            raise error from failure
        return result


def source_digest(root):
    digest = hashlib.sha256()
    root = Path(root)
    paths = subprocess.check_output(["git", "ls-files", "-z", "src"], cwd=root).split(b"\0")
    for name in sorted(p for p in paths if p):
        data = (root / name.decode()).read_bytes()
        digest.update(name + b"\0" + len(data).to_bytes(8, "little") + data)
    return digest.hexdigest()
