"""A spawned child must import the tessera under test, not the installed wheel.

Fleet interpreters -- the shared CPU training venv among them -- carry an
installed ``tessera`` wheel in site-packages.  A test that spawns a child on
``sys.executable`` without a pin silently exercises that wheel, so the branch
under test never runs in the child; the defect only surfaces as an
``AttributeError`` naming a symbol the wheel predates.  Any spawn whose child
can import ``tessera`` takes its environment from :func:`child_env`, which
puts this checkout's ``src`` first on the child's ``PYTHONPATH`` and keeps an
ambient value after it.  The checkout root is derived from this file's own
location, so every worktree and snapshot pins itself.

Scripts under ``experiments/`` and ``tools/`` are exempt from this helper by
the repo's own convention: they self-pin ``parents[1] / "src"`` from their
``__file__`` (the rule ``tests/test_experiment_import_roots.py`` enforces),
and children run under ``-I`` pin themselves through ``sys.path`` for the
same reason -- ``-I`` ignores ``PYTHONPATH`` entirely.
"""
from __future__ import annotations

import os
from pathlib import Path

#: The ``src`` tree of the checkout this file lives in.
SRC = Path(__file__).resolve().parents[1] / "src"


def child_env(**extra: str) -> dict[str, str]:
    """``os.environ`` with this checkout's ``src`` first on ``PYTHONPATH``.

    An existing ``PYTHONPATH`` is preserved, after ``SRC``: a probe or tool
    directory the caller exported keeps working, but it can no longer decide
    which ``tessera`` the child sees.
    """
    env = dict(os.environ)
    ambient = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC) if not ambient else os.pathsep.join([str(SRC), ambient])
    env.update(extra)
    return env
