"""DEPRECATED PATH SHIM for the supported exporter (tessera#687).

The exporter lives at ``src/tessera/export_serving.py`` and its supported
entry is ``python -m tessera.export_serving``.  This shim keeps three legacy
caller shapes working -- the research drivers that import the bare
module by sibling path (``full_model_research_selected_checkpoint.py``,
``moce_source_encode_preflight.py``,
``original_wire_checkpoint.py``, ``full_model_original_wire_checkpoint.py``,
``qualify_historical_selected_wires.py``), the research shell scripts that
invoke ``python experiments/export_tessera_serving.py``, and the tests that
still load this file by location -- for REAL calls: ``main()`` here IS the
exporter's ``main``.  It is a SEPARATE namespace (#691 item 2): the star
import binds copies of the exporter's names into this module, so
monkeypatching an attribute on ``export_tessera_serving`` does NOT affect
the running export.  In-process patchers must target
``importlib.import_module("tessera.export_serving")``.  New callers should
use the module.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.export_serving import *  # noqa: F401,F403
from tessera.export_serving import main  # noqa: F401

if __name__ == "__main__":
    main()
