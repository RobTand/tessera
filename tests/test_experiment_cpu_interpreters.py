"""CPU adapters take a caller-owned interpreter; teardown cannot remove a hidden default."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('adapter,variable', [
    ('extents_cpu_python.sh', 'TESSERA_CPU_PYTHON'),
    ('glm_control_pytest_python.py', 'TESSERA_CPU_PYTHON'),
    ('regdirect_stage1/run.sh', 'CPU_PY'),
    ('t4_code/fp4_arithmetic_action.sh', 'TESSERA_CPU_PYTHON'),
    ('t8r_speed/bench_t8r.sh', 'BENCH_CPU_PYTHON'),
])
@pytest.mark.parametrize('provided', [False, True])
def test_cpu_adapter_requires_and_executes_its_explicit_interpreter(tmp_path, adapter, variable, provided):
    capture = tmp_path / 'captured'
    interpreter = tmp_path / 'python'
    # Execute an existing shell binary, even when the fixture mount is noexec.
    # Its startup file captures Python-shaped arguments without running Python.
    interpreter.symlink_to('/bin/bash')
    startup = tmp_path / 'capture.sh'
    startup.write_text('if [[ "$0" == *.py ]]; then\n'
                       '  printf "%s\\n" "$0" "$@" > "$INPUT_CAPTURE"\n'
                       '  exit 0\nfi\n')
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path), 'BASH_ENV': str(startup),
           'INPUT_CAPTURE': str(capture), 'BENCH_PY': 'bench_geometry.py',
           'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    if provided:
        env[variable] = str(interpreter)
    command = [sys.executable if adapter.endswith('.py') else 'bash', str(ROOT / 'experiments' / adapter)]
    if adapter in ('regdirect_stage1/run.sh', 't8r_speed/bench_t8r.sh'):
        command += [str(ROOT), str(tmp_path / 'out'), '--cpu-preflight']
    elif adapter == 't4_code/fp4_arithmetic_action.sh':
        command += [str(tmp_path / 'out'), '--cpu-preflight']
    else:
        command += ['-c', ':', 'tests/example.py']
    done = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    if provided:
        assert done.returncode == 0, done.stderr
        assert capture.is_file()
        assert '--cpu-preflight' in capture.read_text() or 'tests/example.py' in capture.read_text()
    else:
        assert done.returncode != 0
        assert variable in done.stderr
        assert not capture.exists()
