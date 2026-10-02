"""Durable KDA evidence publication before semantic progress."""
from __future__ import annotations
import json
import os
import runpy
from pathlib import Path

def kda_commit(out: dict, path: Path, units: int, phase: str = "measure"):
    """Publish each complete measured cell before advancing its progress counter."""
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as f:
        f.write(json.dumps(out, indent=1) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    helper = os.environ.get("PRISMABUILD_ACTION_PROGRESS_HELPER")
    if helper:
        runpy.run_path(helper)["commit"](units, phase)
