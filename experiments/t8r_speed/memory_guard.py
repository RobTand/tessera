"""Run one admitted payload with the D30 physical-memory abort floor.

The payload has its own process group. Launcher exit does not end that group.
PrismaBuild owns container cleanup and resource admission.
Process-group proof does not qualify Docker termination; PrismaBuild issue 1599 owns its signal relay.
This guard never submits, polls or resubmits PB actions.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

FLOOR_BYTES = 2 << 30
TERM_GRACE_SECONDS = 5


def available_bytes():
    with open('/proc/meminfo', encoding='ascii') as stream:
        for line in stream:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    raise RuntimeError('D30 guard cannot read MemAvailable')


def stop_payload(child):
    """Stop the owned process group, even after its launcher exits."""
    events = []

    def alive():
        child.poll()
        try:
            os.killpg(child.pid, 0)
            return True
        except ProcessLookupError:
            return False

    def send(sig):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            return
        events.append({"signal": sig.name, "monotonic": time.monotonic()})

    if alive():
        send(signal.SIGTERM)
        deadline = time.monotonic() + TERM_GRACE_SECONDS
        while alive() and time.monotonic() < deadline:
            time.sleep(min(.05, max(0, deadline - time.monotonic())))
        if alive():
            send(signal.SIGKILL)
    child.wait(timeout=1)
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a payload command is required')
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    minimum = available_bytes()
    if minimum < FLOOR_BYTES:
        raise RuntimeError('D30 MemAvailable is below 2 GiB before the payload')
    started = time.time()
    child = subprocess.Popen(command, start_new_session=True)
    aborted = False
    try:
        while child.poll() is None:
            minimum = min(minimum, available_bytes())
            if minimum < FLOOR_BYTES:
                aborted = True
                print('D30 abort: MemAvailable is below 2 GiB', flush=True)
                break
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
    finally:
        stop = stop_payload(child)
        result = {'start_unix': started, 'end_unix': time.time(), 'floor_bytes': FLOOR_BYTES,
                  'minimum_available_bytes': minimum, 'memory_aborted': aborted,
                  'payload_returncode': child.returncode, 'command': command,
                  "process_group_signals": stop,
                  "cleanup_owner": "PrismaBuild owns admitted containers and the resource scope"}
        path.write_text(json.dumps(result, indent=2) + '\n')
        print('D30_MEMORY_RESULT ' + json.dumps(result), flush=True)
    return 1 if aborted else (128 - child.returncode if child.returncode < 0 else child.returncode)


if __name__ == '__main__':
    raise SystemExit(main())
