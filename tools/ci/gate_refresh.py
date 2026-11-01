#!/usr/bin/env python3
"""Update bootstrap scheduling and hang limits from retained hosted JUnit."""
from collections import defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from gate_evidence import cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    config = args.root / 'tools/ci'
    durations = json.loads((config / 'gate_durations.json').read_text())
    budget = json.loads((config / 'gate_budget.json').read_text())
    quarantine = json.loads((config / 'gate_quarantine.json').read_text())['lanes']
    grouped = defaultdict(list)
    maximum = defaultdict(float)
    sources = []
    for run in json.loads(args.manifest.read_text()):
        totals = defaultdict(float)
        for case in cases(run['junit']):
            if case['nodeid'] in quarantine[run['lane']]:
                continue
            totals[case['path']] += case['seconds']
            maximum[case['nodeid']] = max(maximum[case['nodeid']], case['seconds'])
        for path, seconds in totals.items():
            grouped[run['lane'], ','.join(run['runner']), path].append(seconds)
        sources.append({'run': run['run_id'], 'artifact': run['artifact'],
                        'archive_sha256': run['archive_sha256'],
                        'junit_sha256': hashlib.sha256(Path(run['junit']).read_bytes()).hexdigest()})
    architecture_medians = defaultdict(list)
    for (lane, arch, path), values in grouped.items():
        architecture_medians[lane, path].append(statistics.median(values))
    for (lane, path), values in architecture_medians.items():
        durations['lanes'][lane][path] = max(values)
    durations['hosted_source'] = durations.get('hosted_source', []) + sources
    durations['hosted_estimator'] = 'latest master-merged hosted file sums: maximum of available per-architecture medians; unobserved files retain prior measured estimates'
    for node, seconds in maximum.items():
        if seconds > budget['seconds']:
            budget['slow'][node] = max(budget['slow'].get(node, 0), math.ceil(2 * seconds))
    budget['source'].extend(sources)
    durations['quarantine_timing_exclusions'] = quarantine
    for name, value in [('gate_budget.json', budget), ('gate_durations.json', durations)]:
        (config / name).write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'measured_lane_files': len(architecture_medians), 'enforced_exceptions': len(budget['slow'])}))


if __name__ == '__main__':
    main()
