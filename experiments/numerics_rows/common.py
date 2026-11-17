"""Pinned inputs and receipt I/O for the numerics jobs."""
from __future__ import annotations

import hashlib
from contextlib import nullcontext
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess

OUTCOMES = ("FAIL", "STOPPED", "INVALID", "UNATTESTED", "PASS")
KERNEL = "src/tessera/serving/csrc/routed_fused_window.cu"
KERNEL_SHA256 = "8a1536dfbc3df70edbdc37001a0db6947e97d9263926bc85a4d4d3ca231eae4f"
IMAGES = {
    "G": {"local_id": "9e6882df27dd", "image_digest": "sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5",
          "vllm": "0.28.1rc1.dev397+gfd4a15126.d20260904", "torch": "2.13.0+cu130",
          "libraries": ["e2m1", "e4m3mma", "e4m3"]},
    "N": {"local_id": "acc684c06ec7", "image_digest": "sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a",
          "vllm": "0.30.1rc1.dev336+gaf5b4857e.d20260929", "torch": "2.13.0+cu130",
          "libraries": ["value", "e4m3mma", "e4m3"]},
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def numpy_value(item):
        import numpy as np

        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=numpy_value) + "\n")
    temporary.replace(path)


def check_source(root, revision):
    root = Path(root)
    if sha256(root / KERNEL) != KERNEL_SHA256:
        raise Invalid("kernel source sha256 differs from the cell")
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    want = subprocess.check_output(["git", "rev-parse", revision], cwd=root, text=True).strip()
    if rev != want or subprocess.check_output(["git", "status", "--porcelain", "--", "src"], cwd=root):
        raise Invalid(f"kernel source is not clean commit {revision}")
    return rev


def check_runtime(image):
    expected = IMAGES[image]
    path = os.environ.get("NUMERICS_IMAGE_INSPECT", "/run/numerics/image-inspect.json")
    if not path or not Path(path).is_file():
        raise Invalid("missing Podman image inspection for this job")
    observed, = json.loads(Path(path).read_text())
    local = observed["Id"].removeprefix("sha256:")
    digests = {observed.get("Digest"), *(value.rsplit("@", 1)[-1] for value in observed.get("RepoDigests", []))}
    if not local.startswith(expected["local_id"]) or expected["image_digest"] not in digests:
        raise Invalid("observed Podman image differs from the cell")
    actual = {name: importlib.metadata.version(name) for name in ("torch", "vllm")}
    if any(actual[name] != expected[name] for name in actual):
        raise Invalid(f"runtime differs from the cell: {actual}")
    return {**expected, **actual, "local_id": local}


def artifact_limit(root):
    """Read the model's configured clamp and convert it to the kernel's fp32 value."""
    import struct

    config = json.loads((Path(root) / "config.json").read_text())
    value = config.get("text_config", config).get("swiglu_limit")
    if value is None:
        return float("inf")
    limit, = struct.unpack("<f", struct.pack("<f", float(value)))
    if math.isnan(limit) or limit <= 0:
        raise Invalid("artifact swiglu_limit must be positive")
    return limit


def load_build(path, runtime):
    built = json.loads(Path(path).read_text())
    if any(built["runtime"][key] != runtime[key] for key in ("image_digest", "local_id", "vllm", "torch")):
        raise Invalid("build runtime differs from this cell image")
    keys = [record["key"] for record in built["libraries"]]
    if not keys or len(set(keys)) != len(keys) or not set(keys) <= set(runtime["libraries"]):
        raise Invalid("build libraries are empty, repeated, or unavailable in this image")
    runtime["libraries"] = keys
    return built


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


class Unattested(ValueError):
    pass


def flag_error(error):
    """Flag parsers can include both environment values in exception text."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        frame = error.__traceback__
        while frame is not None:
            if frame.tb_frame.f_code.co_filename.replace("\\", "/").endswith(("tessera/serving/flags.py", "tessera/flags.py")):
                return True
            frame = frame.tb_next
        error = error.__cause__ or error.__context__
    return False


def error_text(error):
    return type(error).__name__ if flag_error(error) else str(error)


def error_trace(error):
    import traceback
    return type(error).__name__ if flag_error(error) else traceback.format_exc()


class BoundaryError(RuntimeError):
    def __init__(self, record, error):
        self.record = record
        self.attributed_to = record["call"] if record["kind"] == "tessera" and record["sync_before"] and record["sync_after"] is False else None
        super().__init__(error_text(error))


class BoundaryRecorder:
    """Synchronize each call and retain the boundary that first reports an error."""

    def __init__(self, synchronize, records):
        self.synchronize, self.records = synchronize, records
        self.enabled = True
        self.attribute = True
        self.before_call = lambda: None
        self.context = lambda name: nullcontext()

    def call(self, name, kind, function):
        with self.context(name):
            return self._call(name, kind, function)

    def _call(self, name, kind, function):
        self.before_call()
        if not self.enabled:
            return function()
        record = {"call": name, "kind": kind, "sync_before": None, "sync_after": None}
        self.records.append(record)
        try:
            self.synchronize()
            record["sync_before"] = True
        except Exception as error:
            record["sync_before"] = False
            record["message"] = error_text(error)
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
        if isinstance(failure, (Invalid, Unattested)):
            raise failure
        if failure is not None:
            record["message"] = error_text(failure)
            error = nested or BoundaryError(record, failure)
            if not self.attribute:
                error.attributed_to = None
            if nested is not None:
                raise nested
            raise error from failure
        validator = getattr(self, "quantizer_result", None)
        if kind == "quantizer" and validator is not None:
            validator(result)
        return result


def source_digest(root, pathspec="src"):
    digest = hashlib.sha256()
    root = Path(root)
    paths = subprocess.check_output(["git", "ls-files", "-z", pathspec], cwd=root).split(b"\0")
    for name in sorted(p for p in paths if p):
        data = (root / name.decode()).read_bytes()
        digest.update(name + b"\0" + len(data).to_bytes(8, "little") + data)
    return digest.hexdigest()
