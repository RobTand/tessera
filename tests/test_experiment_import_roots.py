"""Issue #58: experiment harnesses must resolve their import root from their
own location, never from a hardcoded checkout path.

Twenty-plus files under ``experiments/`` opened with some spelling of
``sys.path.insert(0, "/home/rob/tessera/src")`` -- the shared working tree,
on whatever branch and dirty state the last worker left it in.  A harness
that hardcodes it does not measure the code it ships with; two runs minutes
apart can be two different codebases, and neither run records which.

The correct form (``bf16_twin_check.py`` and five others) is::

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

This test pins the property, not the roster: it scans every ``*.py`` under
``experiments/`` and fails on any ``sys.path`` insert that names an absolute
source tree without deriving it from ``__file__``.  Reintroducing one
hardcoded line anywhere -- not just in today's 24 files -- fails it.
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


def test_grouped_benchmark_resolves_source_without_pythonpath(tmp_path):
    import pytest

    pytest.importorskip("torch")
    env = dict(os.environ, PYTHONPATH="")
    result = subprocess.run(
        [sys.executable, str(EXPERIMENTS / "t8r_speed/bench_compact_vs_fused.py"), "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr

#: An absolute path literal that names a source tree: ``/.../src`` or
#: ``/.../experiments``.  The shared-checkout form (``/home/rob/tessera/src``),
#: a worktree form (``.../worktrees/<id>/src``, ``.../worktrees/<id>/experiments``)
#: and any future checkout all match; external package roots (``pq-wt/...``),
#: system site dirs (``dist-packages``) and variable-only inserts do not, and
#: are out of this issue's scope.
ABSOLUTE_TREE = re.compile(r"""["']/[^"']*/(src|experiments)/?["']""")


def _offenders():
    bad = []
    for path in sorted(EXPERIMENTS.rglob("*.py")):
        try:
            text = path.read_text()
        except OSError:
            continue
        # The conversions must at least parse; a broken edit is not a fix.
        ast.parse(text, filename=str(path))
        for lineno, line in enumerate(text.splitlines(), 1):
            if "sys.path" not in line:
                continue
            if "__file__" in line:
                continue
            if ABSOLUTE_TREE.search(line):
                bad.append(f"{path.relative_to(EXPERIMENTS)}:{lineno}: {line.strip()}")
    # The same defect in shell form: a driver that PYTHONPATHs the shared
    # checkout measures that tree, not the one it ships with.  The correct
    # form derives the root from the script's own location (``$REPO``/``$WT``
    # siblings do this; ``rd_curve_served.sh`` did not).
    for path in sorted(EXPERIMENTS.rglob("*.sh")):
        try:
            text = path.read_text()
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if "PYTHONPATH" not in line:
                continue
            if "$" in line.split("PYTHONPATH")[1]:
                continue
            if re.search(r"/home/[^ \t]*tessera[^ \t]*/(src|experiments)", line):
                bad.append(f"{path.relative_to(EXPERIMENTS)}:{lineno}: {line.strip()}")
    return bad


def test_no_hardcoded_checkout_import_root():
    """Every ``sys.path`` insert under ``experiments/`` derives from ``__file__``."""
    bad = _offenders()
    assert not bad, (
        "hardcoded checkout import root(s); resolve from __file__ instead:\n"
        + "\n".join(bad)
    )
