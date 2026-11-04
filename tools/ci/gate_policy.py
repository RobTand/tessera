#!/usr/bin/env python3
"""The merge gate's base-relative policies, kept from the hosted gate (#1589).

Per-test budgets and the shrink-only slow list may not grow against the base
(tools/ci/gate_run.py), and quarantine grows only in a change of its own, with
an open owner issue for every entry (tools/ci/gate_select.py). Exit 0 pass,
1 policy failure, 75 no verdict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/ci'))
import gate_run  # noqa: E402
import gate_select  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    args = parser.parse_args(argv)
    try:
        gate_run._budget_policy(ROOT, args.base)
        entries = gate_select.exclusion_entries(json.loads((ROOT / gate_select.QUARANTINE_PATH).read_text()))
        unknown = {row['target'].split('::', 1)[0] for row in entries} - set(gate_select.population(ROOT))
        if unknown:
            raise ValueError('quarantine files outside the test population: ' + ', '.join(sorted(unknown)))
        baseline = gate_select.base_quarantine(ROOT, args.base)
        changed = subprocess.run(['git', 'diff', '--name-only', '--no-renames', args.base, 'HEAD'], cwd=ROOT,
                                 text=True, capture_output=True, check=True).stdout.splitlines()
        gate_select.quarantine_policy(entries, gate_select.exclusion_entries(baseline)
                                      if baseline is not None else None, changed)
        gate_select.check_quarantine_owners(entries)
    except (ValueError, gate_run.BudgetFailure) as error:
        print('GATE_POLICY_FAILURE: ' + str(error), file=sys.stderr)
        return 1
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print('NO_VERDICT: ' + str(error), file=sys.stderr)
        return 75
    print('GATE_POLICY_PASS: budgets, quarantine and its owners')
    return 0


if __name__ == '__main__':
    sys.exit(main())
