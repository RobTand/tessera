"""Incremental phase records and cooperative termination for row workers."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import signal
import json
import time

from experiments.numerics_rows.common import write_json


@contextmanager
def handled_signals(receipt, out, checkpoint=lambda: None):
    def stop(number, frame):
        checkpoint()
        from experiments.numerics_rows.row import finish_unit
        for unit in receipt["units"]:
            if not unit["findings"]:
                unit["completed"] = False
            finish_unit(unit)
        receipt["row"].update(utc_end=datetime.now(timezone.utc).isoformat(), exit_status=128 + number,
            stop_cause=signal.Signals(number).name, outcome="FAIL" if any(u["findings"] for u in receipt["units"]) else "STOPPED",
            finalization_finished=False)
        write_json(Path(out) / "numerics_row.json", receipt)
        raise SystemExit(128 + number)
    previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


class Progress:
    def __init__(self, out, receipt, phases, meter=None):
        self.out, self.receipt, self.phases, self.meter = Path(out), receipt, phases, meter
        self.start = self.since = time.monotonic()
        self.active = None

    def record(self, phase, unit=None, case=None, *, checkpoint=False):
        now = time.monotonic()
        if self.active is not None:
            self.active["elapsed_seconds"] = now - self.since
            with (self.out / "phase-events.jsonl").open("a") as stream:
                stream.write(json.dumps(self.active, sort_keys=True) + "\n")
        self.phases.change(phase)
        self.active = {"phase": phase, "module": None if unit is None else unit["module"],
                       "cut": None if unit is None else unit["tp"], "case": case,
                       "M": None if unit is None else unit["m_config"]["M"],
                       "utc_start": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": 0,
                       "worker_elapsed_seconds": now - self.start}
        self.since = now
        self.receipt["row"]["active"] = self.active
        self.checkpoint()

    def checkpoint(self):
        if self.active is not None:
            self.active["elapsed_seconds"] = time.monotonic() - self.since
            write_json(self.out / "progress.json", self.active)
