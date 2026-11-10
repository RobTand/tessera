"""Durable KDA evidence publication before semantic progress."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools._durable_progress import emit

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
    emit(units, phase)
