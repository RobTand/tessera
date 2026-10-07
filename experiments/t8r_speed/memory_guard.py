"""Run one admitted payload with the D30 physical-memory abort floor.

The payload has its own process group. PrismaBuild still owns container cleanup
and resource admission. This guard never submits, polls or resubmits PB actions.
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
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=TERM_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                break
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=TERM_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        result = {'start_unix': started, 'end_unix': time.time(), 'floor_bytes': FLOOR_BYTES,
                  'minimum_available_bytes': minimum, 'memory_aborted': aborted,
                  'payload_returncode': child.returncode, 'command': command,
                  'cleanup_owner': 'PrismaBuild owns admitted containers and the resource scope'}
        path.write_text(json.dumps(result, indent=2) + '\n')
        print('D30_MEMORY_RESULT ' + json.dumps(result), flush=True)
    return 1 if aborted else (128 - child.returncode if child.returncode < 0 else child.returncode)


if __name__ == '__main__':
    raise SystemExit(main())
