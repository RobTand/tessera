"""Bind the canonical Tessera package before test helpers alter sys.path.

Explicit pytest plugin loading applies in the controller and every xdist worker.
"""
import os
from pathlib import Path

import tessera

expected = Path(os.environ['NATIVE_CONTAINER_SRC']).resolve()
actual = Path(tessera.__file__).resolve().parent.parent
if actual != expected:
    raise RuntimeError(f'foreign Tessera source: {actual}; expected {expected}')
