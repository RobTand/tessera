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
