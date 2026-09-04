"""Read sealed r6 inputs and write a separate, bounded-CPU comparison receipt."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from tessera.serving import build_identity


POP = Path('/mnt/shared/tessera-runs/ts113-sparklina-population-aa6-r6')
KL = Path('/home/rob/dq-runs/kl_tool.py')
ARMS = ('A1', 'A2', 'B1', 'B2')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, record):
    Path(path).write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')


def base(stage, regime='decode'):
    name = 'teacher_decode' if stage == 'teacher' else f'qwen_ts113_sparklina_aa6_r6_{stage}_{regime}'
    return POP / 'stages' / stage / name


def main():
    output = Path(sys.argv[1])
    assert output.parent == POP.parent, 'analysis must not mutate the sealed population'
    assert output != POP
    assert (POP / 'CAMPAIGN_COMPLETE').is_file(), 'campaign is not yet complete'
    output.mkdir(exist_ok=False)
    inputs = {}
    builds = {}
    fingerprints = {}
    for stage in ('teacher', *ARMS):
        directory = POP / 'stages' / stage
        assert (directory / 'STAGE_COMPLETE').is_file()
        subprocess.run(['sha256sum', '--check', '--strict', '--quiet', 'STAGE_SHA256'],
                       cwd=directory, check=True)
        inputs[f'{stage}/STAGE_SHA256'] = sha(directory / 'STAGE_SHA256')
        build = read(str(base(stage)) + '.build.json')
        assert build['complete'] is True
        assert build['identity']['compiled_forward'] is True
        assert build['identity']['eager'] is False
        assert build['provenance']['fresh_compiles'] > 0
        builds[stage] = build
        for regime in ('decode',) if stage == 'teacher' else ('decode', 'prefill'):
            payload = Path(str(base(stage, regime)) + '.json.npz')
            inputs[str(payload.relative_to(POP))] = sha(payload)
            # Reuse the metric tool's definition rather than inventing equality.
            result = subprocess.run([sys.executable, str(KL), 'fingerprint', str(payload), '--json'],
                                    check=True, capture_output=True, text=True)
            fingerprints[f'{stage}_{regime}'] = json.loads(result.stdout)

    build_pairs = {}
    for left, right in itertools.combinations(builds, 2):
        why = f'r6 {left} vs {right}'
        build_identity.require_same_dispatch(builds[left], builds[right], why=why)
        build_identity.require_distinct_build(builds[left], builds[right], why=why)
        build_pairs[f'{left}--{right}'] = build_identity.compare(builds[left], builds[right])

    rows = []
    pairs = [('teacher', arm, 'decode') for arm in ARMS]
    pairs.extend((left, right, regime) for left, right in itertools.permutations(ARMS, 2)
                 for regime in ('decode', 'prefill'))
    for left, right, regime in pairs:
        label = f'{regime}_{left}_to_{right}'
        path = output / f'{label}.json'
        cmd = [sys.executable, str(KL), 'compare', str(base(left, regime)) + '.json.npz',
               str(base(right, regime)) + '.json.npz', '--teacher-label-override',
               f'compiled-{left}-r6', '--out', str(path)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        (output / f'{label}.log').write_text(result.stdout + result.stderr)
        record = read(path)
        assert record['alignment_checked'] is True
        assert record['unchecked_alignment'] is False
        assert record['positions'] == (256 if regime == 'decode' else 4088)
        row = {'reference': left, 'student': right, 'regime': regime,
               'positions': record['positions'], **record['all'],
               'comparison_sha256': sha(path)}
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    source_root = Path(__file__).resolve().parents[1]
    stamps = list(source_root.glob('.pbrun-closure.*.json'))
    assert len(stamps) == 1
    record = {
        'schema': 'tessera.ts113.r6-pairwise-receipt.v1',
        'produced_at_utc': datetime.now(timezone.utc).isoformat(),
        'device': 'CPU-only postprocessing of sealed served GPU payloads',
        'hostname': os.uname().nodename,
        'population': str(POP),
        'campaign_identity_sha256': sha(POP / 'CAMPAIGN_IDENTITY.json'),
        'continuation_identity_sha256': sha(POP / 'CONTINUATION_IDENTITY.json'),
        'campaign_complete_sha256': sha(POP / 'CAMPAIGN_COMPLETE'),
        'stage_measurement_source': read(POP / 'CONTINUATION_IDENTITY.json')['stage_measurement_source'],
        'postprocessing_source': read(stamps[0]),
        'postprocessing_script_sha256': sha(__file__),
        'kl_tool_sha256': sha(KL),
        'sealed_inputs': inputs,
        'build_fingerprints': {stage: value['build_fingerprint'] for stage, value in builds.items()},
        'build_pairs': build_pairs,
        'payload_fingerprints': fingerprints,
        'comparisons': rows,
        'limits': ['KL values are top-K lower bounds, not full-vocabulary KL.',
                   'All pairs have distinct compiled builds; within-arm pairs measure rebuild variation.',
                   'Two fresh builds per lane state are not a distributional confidence interval.',
                   'No served performance or work-per-joule claim is made.'],
    }
    dump(output / 'receipt.json', record)
    lines = [f'{sha(path)}  {path.name}' for path in sorted(output.iterdir()) if path.is_file()]
    (output / 'ANALYSIS_SHA256').write_text('\n'.join(lines) + '\n')
    (output / 'ANALYSIS_COMPLETE').write_text(f'receipt_sha256={sha(output / "receipt.json")}\n')
    print('TS113_R6_COMPARE_OK', output, sha(output / 'receipt.json'), flush=True)


if __name__ == '__main__':
    main()
