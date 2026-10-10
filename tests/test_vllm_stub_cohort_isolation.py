"""tessera#1015: a leaked bare vllm stub skips vllm consumers, never errors them.

The contaminator is the pre-ddd2137 import-time ``_install_vllm_stubs`` in
``experiments/t8r_speed/bench_t8r.py``: a bare ``ModuleType("vllm")`` with no
``__path__``, published into ``sys.modules`` at import and fanned into the
co-run by eleven importing test files. A consumer that guards on the bare
name passes its capability check on that stub, then fails the real import
with ``'vllm' is not a package``. The repaired consumers guard on the
submodule they import, so the same stub skips them.

This test replays that co-run shape in one child interpreter: the stub is
present before collection (as the old import left it), the cohort holds the
real intake consumer, and the run must stay green with the consumer skipped.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

CHECKOUT = Path(__file__).resolve().parents[1]
INTAKE = CHECKOUT / "tests" / "test_serving_moe_bf16_tp1_intake.py"

#: The old import-time contaminator, verbatim in shape: bare non-package
#: ``vllm`` in ``sys.modules`` at module import, with no restore.
_CONTAMINATOR = """
import sys
from types import ModuleType

sys.modules["vllm"] = ModuleType("vllm")


def test_contaminator_marks_the_cohort():
    assert sys.modules["vllm"].__name__ == "vllm"
"""

#: The stub is present before collection starts, as the old import left it.
_PLUGIN = """
import sys
from types import ModuleType

sys.modules["vllm"] = ModuleType("vllm")
"""

_DUMMY = """
def test_dummy_marks_the_cohort():
    pass
"""


def test_a_bare_vllm_stub_skips_the_intake_consumer(tmp_path):
    (tmp_path / "stub_plugin.py").write_text(_PLUGIN)
    (tmp_path / "test_contaminator.py").write_text(_CONTAMINATOR)
    (tmp_path / "test_dummy.py").write_text(_DUMMY)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:xdist",
         "-p", "stub_plugin", "-q",
         str(tmp_path / "test_contaminator.py"), str(INTAKE),
         str(tmp_path / "test_dummy.py")],
        cwd=CHECKOUT, env=env, text=True, capture_output=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    assert re.search(r"\d+ skipped", result.stdout), result.stdout + result.stderr
