"""Nonregular reference sources refuse before a blocking FIFO open."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_hessian_reference_capture import reference

#: The tree under test.  The child below is a fresh interpreter: it does not
#: see the ``sys.path`` entry ``conftest.py`` gives this process, so without
#: this it imports whatever ``tessera`` the ambient environment holds -- none
#: at all under a plain ``pytest`` (ModuleNotFoundError), or an installed pin
#: that is not this checkout, which then passes or fails for that pin's code.
SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.mark.parametrize('intake', ['metadata', 'hessian'])
def test_fifo_reference_refuses_within_bounded_subprocess(reference, intake):
    handoff, _payload, _hessians, canonical, _manifest = reference
    path = handoff if intake == 'metadata' else canonical.parent/'inputs/a.pt'
    path.unlink()
    os.mkfifo(path)
    program = '''
import sys
from tessera.errors import GrammarError
from tessera.hessian_capture import ReferenceHessians
import tessera
if not tessera.__file__.startswith(sys.argv[3]):
    raise AssertionError(f"imported {tessera.__file__}, not the tree under test {sys.argv[3]}")
print("ENTERING_REFERENCE_INTAKE", flush=True)
try:
    with ReferenceHessians(sys.argv[1]) as owner:
        if sys.argv[2] == "hessian":
            owner["a"]
except GrammarError as error:
    if "regular-file byte bound" not in str(error):
        raise
    print("FIFO_REFUSED", flush=True)
else:
    raise AssertionError("nonregular reference source was accepted")
'''
    try:
        env = {**os.environ, 'PYTHONPATH': os.pathsep.join(
            [str(SRC), *filter(None, [os.environ.get('PYTHONPATH')])])}
        result = subprocess.run([sys.executable, '-c', program, str(handoff), intake, str(SRC)],
            capture_output=True, text=True, timeout=5, env=env)
    except subprocess.TimeoutExpired as error:
        # run() kills and reaps this owned child before raising; no FIFO writer
        # or helper process is needed to release an accidentally blocked open.
        assert b'ENTERING_REFERENCE_INTAKE' in (error.stdout or b'')
        pytest.fail(f'{intake} FIFO blocked instead of refusing within 5 seconds')
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['ENTERING_REFERENCE_INTAKE', 'FIFO_REFUSED']
