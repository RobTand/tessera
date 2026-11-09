"""One home for the durable-progress line: print it only after the unit is on disk."""
from __future__ import annotations

import json
import sys

SCHEMA = "tessera.durable_progress.v1"


def emit(units: int, phase: str, unit: str | None = None, *, stream=None) -> None:
    """Say how much work is COMMITTED. A report is a claim that the unit is already
    durable (fsynced file and directory); the caller orders that, this only prints."""
    record = {"schema": SCHEMA, "units_completed": units, "phase": phase}
    if unit is not None:
        record["unit"] = unit
    print(json.dumps(record, sort_keys=True), file=stream if stream is not None else sys.stdout, flush=True)
