"""Nonregular reference sources refuse before a blocking FIFO open."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_hessian_reference_capture import reference
from reuse_authority_fixture import CANONICAL_CAPTURE

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
    with ReferenceHessians(sys.argv[1], canonical_capture=(sys.argv[4], sys.argv[5])) as owner:
        if sys.argv[2] == "hessian":
            owner["a"]
except GrammarError as error:
    if "regular-file byte bound" not in str(error):
        raise
    print("FIFO_REFUSED", flush=True)
else:
    raise AssertionError("nonregular reference source was accepted")
'''
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(
        [str(SRC), *filter(None, [os.environ.get('PYTHONPATH')])])}
    child = subprocess.Popen([sys.executable, '-c', program, str(handoff), intake, str(SRC), *CANONICAL_CAPTURE],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, env=env)
    try:
        # Interpreter start and the import are not what a FIFO can block, so
        # they are not charged to the bound; the child prints this line after
        # the import and just before the open this test is about (#1049).  The
        # pipe is unbuffered so that this read takes nothing past the line and
        # communicate() below, which reads the descriptor, sees the rest.
        entered = child.stdout.readline()
        assert entered == b'ENTERING_REFERENCE_INTAKE\n', (entered, child.communicate()[1])
        stdout, stderr = child.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        # Kill and reap this owned child; no FIFO writer or helper process is
        # needed to release an accidentally blocked open.
        child.kill()
        child.communicate()
        pytest.fail(f'{intake} FIFO blocked instead of refusing within 5 seconds')
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()
    assert child.returncode == 0, stderr.decode()
    assert (entered + stdout).decode().splitlines() == ['ENTERING_REFERENCE_INTAKE', 'FIFO_REFUSED']
