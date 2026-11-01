#!/usr/bin/env python3
"""The hosted no-torch lane's import, denylist, allowlist and package proofs."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    args = parser.parse_args()
    allowlist = 'tests/standalone_boundary_allowlist.txt'
    previous = subprocess.run(['git', 'show', f'{args.base}:{allowlist}'],
                              capture_output=True, text=True)
    if previous.returncode == 0:
        current = (ROOT / allowlist).read_text()
        if set(current.splitlines()) - set(previous.stdout.splitlines()):
            raise SystemExit('shrink-only standalone boundary allowlist grew')
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
    for name in ('alphabet', 'canonical', 'container', 'errors', 'exact', 'grammar',
                 'layout', 'manifest', 'planes', 'provenance', 'trellis'):
        importlib.import_module('tessera.' + name)
    importlib.import_module('conftest')
    assert 'torch' not in sys.modules, 'byte layer pulled torch into the no-torch process'
    assert 'tessera.slicing' not in sys.modules, 'layout eagerly imported its torch half'
    from tessera.provenance import assert_denylist_populated, Denylist
    from tessera.errors import ProvenanceError
    try:
        assert_denylist_populated(Denylist())
    except ProvenanceError:
        pass
    else:
        raise SystemExit('empty denylist silently certified everything')
    subprocess.run([sys.executable, '-m', 'build'], cwd=ROOT, check=True)
    wheels = sorted((ROOT / 'dist').glob('*.whl'))
    sdists = sorted((ROOT / 'dist').glob('*.tar.gz'))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit('expected exactly one wheel and one sdist')
    subprocess.run([sys.executable, 'tools/check_wheel.py', str(wheels[0]), str(sdists[0])],
                   cwd=ROOT, check=True)
    print('GATE_BOUNDARY_PASS: allowlist, imports, empty denylist, wheel and sdist')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
