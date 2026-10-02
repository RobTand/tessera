"""The existing containment owner keeps numeric240 and permits closed timing600."""
import ast
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import os
import signal

import pytest

OWNER = Path('/mnt/shared/astra-resume-20261002/t8_performance/piece-major-common-c236b7aa/execution-owner-timing600/experiments/t8r_speed/paired_k32_action.py')


@pytest.mark.parametrize('deadline,refuses', [(240, True), (600, False)])
def test_phase_deadline_preserves_numeric_and_allows_timing(tmp_path, deadline, refuses):
    node = next(n for n in ast.parse(OWNER.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'run_direct_arm')
    times = iter([0, 241])
    scope = dict(Path=Path, json=json, subprocess=subprocess, signal=signal,
                 os=SimpleNamespace(killpg=lambda *a: None),
                 time=SimpleNamespace(monotonic=lambda: next(times), sleep=lambda *a: None),
                 owned_cleanup=lambda *a: {'state': 'inert'})
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(OWNER), 'exec'), scope)
    class Child:
        pid = 123456
        returncode = 0
        polls = iter([None, 0, 0])
        def poll(self): return next(self.polls)
        def wait(self, **kw): return 0
    kw = dict(pressure=lambda: (80 * (1 << 30), 0), popen=lambda *a, **k: Child())
    if deadline == 600: kw['deadline_s'] = 600
    if refuses:
        with pytest.raises(RuntimeError, match='bound reached'):
            scope['run_direct_arm']([], {}, None, tmp_path, tmp_path / 'guard.jsonl', **kw)
    else:
        assert scope['run_direct_arm']([], {}, None, tmp_path, tmp_path / 'guard.jsonl', **kw) == 0
