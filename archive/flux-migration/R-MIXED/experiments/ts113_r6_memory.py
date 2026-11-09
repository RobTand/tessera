"""Summarize completed r6 observer logs and post-teardown parser receipts."""
from pathlib import Path
import hashlib
import json
import re
import time


ROOT = Path('/mnt/shared/tessera-runs')
POP = ROOT / 'ts113-sparklina-population-aa6-r6'
QUEUE = Path('/mnt/shared/prismabuild-fleet/pb-queue')
OBSERVER = 'da61a1522e3aea55c6e15a70a4877e8071d6bf8aafe2bc52dff9ed7e00f1c101'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    while not (POP / 'CAMPAIGN_COMPLETE').is_file() or not (QUEUE / 'done' / f'{OBSERVER}.json').is_file():
        time.sleep(5)
    logs = {}
    for name in ('783ab956', '2f97d7ad'):
        path = ROOT / f'ts113-r6-live-telemetry-{name}.log'
        rows = []
        for line in path.read_text().splitlines():
            match = re.fullmatch(r'utc=(\S+) mem_available_kib=(\d+) power_w=([\d.]+)', line)
            if match:
                utc, mem, power = match.groups()
                rows.append({'utc': utc, 'mem_available_kib': int(mem), 'power_w': float(power)})
        assert rows
        logs[name] = {
            'path': str(path), 'sha256': sha(path), 'samples': len(rows),
            'start_utc': rows[0]['utc'], 'end_utc': rows[-1]['utc'],
            'minimum_memory_sample': min(rows, key=lambda row: row['mem_available_kib']),
            'maximum_power_sample': max(rows, key=lambda row: row['power_w']),
        }
    parsers = {}
    for stage in ('A1', 'A2', 'B1', 'B2'):
        path = POP / 'stages' / stage / f'profile-{stage}-parser-resources.txt'
        text = path.read_text()
        rss = int(re.search(r'Maximum resident set size \(kbytes\): (\d+)', text)[1])
        elapsed = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)', text)[1]
        status = int(re.search(r'Exit status: (\d+)', text)[1])
        swaps = int(re.search(r'Swaps: (\d+)', text)[1])
        assert status == 0
        parsers[stage] = {'path': str(path), 'sha256': sha(path), 'maximum_rss_kib': rss,
                          'elapsed': elapsed, 'exit_status': status, 'swaps': swaps}
    record = {
        'schema': 'tessera.ts113.r6-memory-observer-summary.v1',
        'observer_logs': logs, 'rank0_parsers': parsers,
        'scope': 'Sampled memory/power intervals and standalone post-teardown parser RSS; '
                 'not a hard memory cap or performance comparison.',
    }
    out = ROOT / 'ts113-r6-telemetry-summary.json'
    with out.open('x') as handle:
        json.dump(record, handle, indent=1, sort_keys=True)
        handle.write('\n')
    print(json.dumps(record, indent=1, sort_keys=True))
    print('SUMMARY_SHA256', sha(out))


if __name__ == '__main__':
    main()
