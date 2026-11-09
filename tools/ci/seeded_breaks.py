#!/usr/bin/env python3
"""Plant known breaks and prove the gate's selection catches each one (#1589).

For every seed: commit the break on top of BASE, select tests with the same
code the gate uses (bazel-diff and bazel_gate.gate_decision), run the selection
on remote execution, and require a failing target inside it that the seed is
expected to break. Runs on a Remote Bazel runner (docs/bazel.md); it never
pushes the seed commits. Seeds sit where selection is narrow, so the run costs
tens of targets, not the suite.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bazel_gate  # noqa: E402

RAISE = '\nraise RuntimeError("seeded break")\n'
FAILING_TEST = '\n\ndef test_seeded_break():\n    assert False, "seeded break"\n'

# (place, path, mutation, labels one of which must fail inside the selection)
SEEDS = [
    ('library module', 'src/tessera/rung_allowability.py', ('append', RAISE), ['//tests:test_rung_performant_policy']),
    ('byte-layer provenance', 'src/tessera/provenance.py', ('append', RAISE), ['//tests:test_provenance']),
    ('repository tool', 'tools/tessera_attest.py', ('append', RAISE), ['//tests:test_attest_receipt']),
    ('packaging tool', 'tools/check_wheel.py', ('append', RAISE), ['//tests:test_packaging']),
    ('experiment module', 'experiments/t8r_speed/stageprev_probe.py', ('append', RAISE),
     ['//tests:test_stageprev_probe']),
    ('shell script a test runs', 'experiments/t8_census/profile_unit_encode.sh', ('replace', 'exit 3\n'),
     ['//tests:test_b770_producer_review']),
    ('test file', 'tests/test_alphabet.py', ('append', FAILING_TEST), ['//tests:test_alphabet']),
    ('JSON fixture', 'tests/fixtures/native_operator_resource_qualification.json', ('replace', '{"seeded": break'),
     ['//tests:test_native_operator_resources']),
    ('schema document', 'docs/schema/allowable-rung-table.v2.schema.json', ('replace', '{"seeded": break'),
     ['//tests:test_rung_allowability_structure_spec']),
    ('test helper', 'tests/_whole_phase_sdk.py', ('append', RAISE), ['//tests:test_value_prefetch_numeric']),
    ('gate tooling', 'tools/ci/gate_select.py', ('append', RAISE), ['//tests:test_gate_contract']),
]


def mutate(root, path, mutation):
    kind, text = mutation
    target = Path(root, path)
    if not target.is_file():
        raise bazel_gate.GateError(f'seed target {path} is missing; update SEEDS')
    target.write_text(target.read_text() + text if kind == 'append' else text)


def seed_verdict(selected, statuses, expected):
    failing = sorted(label for label in selected if statuses.get(label) == 'FAILED')
    caught = [label for label in failing if label in expected]
    return {'impacted_tests': len(selected), 'failing_selected': failing, 'caught': bool(caught),
            'example': caught[0] if caught else None}


def table(rows):
    lines = ['| Place | Changed file | Selected tests | Failing selected | Expected failure caught | Example |',
             '|---|---|---:|---:|---|---|']
    for row in rows:
        lines.append(f"| {row['place']} | `{row['path']}` | {row['impacted_tests']} | "
                     f"{len(row['failing_selected'])} | {'yes' if row['caught'] else 'NO'} | "
                     f"`{row['example'] or '-'}` |")
    return '\n'.join(lines)


def select_seeds(value):
    """Resolve comma-separated zero-based indices or place names in SEEDS order."""
    choices = {str(index): index for index in range(len(SEEDS))}
    choices.update({seed[0]: index for index, seed in enumerate(SEEDS)})
    selected = set()
    for item in value.split(','):
        name = item.strip()
        if name not in choices:
            raise argparse.ArgumentTypeError('seeds must be zero-based indices or place names from SEEDS')
        selected.add(choices[name])
    return [seed for index, seed in enumerate(SEEDS) if index in selected]


def run_seed(workspace, base, place, path, mutation, expected):
    workspace.checkout(base)
    mutate(workspace.root, path, mutation)
    subprocess.run(['git', '-c', 'user.name=seed', '-c', 'user.email=seed@invalid', 'commit',
                    '--quiet', '--no-verify', '-am', f'seeded break: {place}'],
                   cwd=workspace.root, check=True, env=bazel_gate.child_environment())
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=workspace.root, text=True,
                                   env=bazel_gate.child_environment()).strip()
    changed, deleted = workspace.changes(base, head)
    forced = bazel_gate.fallback_files(changed | deleted)
    impacted = set() if forced else workspace.bazel_diff(base, head)
    workspace.checkout(head)
    decision = bazel_gate.gate_decision(changed, deleted, impacted, workspace.used_sources(), workspace.tests())
    statuses, _ = workspace.test(decision['selected'], 'remote', {'ROLE': 'SEEDED_BREAK'}, 'seed',
                                 extra=['--jobs=16'])
    return {'place': place, 'path': path, 'selection': decision['selection'],
            **seed_verdict(decision['selected'], statuses, expected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='HEAD')
    parser.add_argument('--seeds', type=select_seeds, default=SEEDS,
                        help='comma-separated zero-based indices or place names; default: all seeds')
    args = parser.parse_args()
    workspace = bazel_gate.Workspace()
    base = subprocess.check_output(['git', 'rev-parse', args.base], cwd=workspace.root, text=True,
                                   env=bazel_gate.child_environment()).strip()
    rows = [run_seed(workspace, base, *seed) for seed in args.seeds]
    workspace.checkout(base)
    report = table(rows)
    print(report)
    output = bazel_gate.ROOT / 'gate-artifacts'
    output.mkdir(parents=True, exist_ok=True)
    try:
        (output / 'seeded-breaks.json').write_text(json.dumps({'base': base, 'rows': rows}, indent=2))
        (output / 'seeded-breaks.md').write_text(report + '\n')
    finally:
        safe_artifacts = bazel_gate.publish_artifacts(output)
    return 0 if safe_artifacts and all(row['caught'] for row in rows) else 1


if __name__ == '__main__':
    bazel_gate.require_scan_python()
    sys.exit(main())
