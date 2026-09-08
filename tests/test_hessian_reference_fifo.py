"""Nonregular reference sources refuse before a blocking FIFO open."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_hessian_reference_capture import reference


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
        result = subprocess.run([sys.executable, '-c', program, str(handoff), intake],
            capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired as error:
        # run() kills and reaps this owned child before raising; no FIFO writer
        # or helper process is needed to release an accidentally blocked open.
        assert b'ENTERING_REFERENCE_INTAKE' in (error.stdout or b'')
        pytest.fail(f'{intake} FIFO blocked instead of refusing within 5 seconds')
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['ENTERING_REFERENCE_INTAKE', 'FIFO_REFUSED']
