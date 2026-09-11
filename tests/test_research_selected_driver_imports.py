"""The bullet-1 export driver's module graph must resolve before a GPU window.

Issue #442.  The first launcher for this driver ran the frozen per-job
installer, which pip-installs Tessera from an archive pinned at ``382a1a97`` --
a tree with no ``tessera/moe_execution.py``.  The driver imports
``ResearchSelectedMoeInput`` from that module at module level, so the job would
have died on its first import and spent the window doing it.  Nothing caught
that, because no test asks whether this driver can import at all.

This asks.  ``--help`` runs every module-level import and exits before it
touches a device, a model or an output path, so the check is a few seconds on a
CPU box and covers the two resolutions the container has to get right:

* ``tessera.*`` comes from the checkout on ``PYTHONPATH``, which is what
  ``experiments/checkout_runtime_identity.py`` prepends before it execs;
* ``export_tessera_serving`` is a bare module name and resolves from the
  driver's own directory, which is ``sys.path[0]`` when Python runs a script by
  path.

It is not a substitute for the run.  An import that resolves says nothing about
what the encode computes.
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest

# The driver imports torch at module level (``:18``), so on an interpreter
# without it ``--help`` exits 1 and this file reports a failure where it should
# report an absence -- which is what it did on the GitHub runner, whose
# environment installs no torch.  Both tests below run the driver, so the guard
# belongs at module scope here: there is nothing in this file that a
# torch-free box could still check.
#
# The guard is also the honest scope statement.  This file asks whether the
# driver's module graph resolves against a checkout, and the graph includes
# torch; a box that cannot import torch cannot answer the question, and
# claiming a pass there would be claiming coverage the run does not have.
pytest.importorskip("torch", reason="the export driver imports torch at module level")

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = [
    "full_model_research_selected_checkpoint.py",
]


@pytest.mark.parametrize("name", DRIVERS)
def test_driver_module_graph_resolves_from_the_checkout(name):
    driver = ROOT / "experiments" / name
    assert driver.is_file(), driver
    # The shape checkout_runtime_identity builds (``:150-151``): the checkout's
    # src on PYTHONPATH and nothing installed.  The container sets no PYTHONPATH
    # of its own, so the prepend there resolves to src alone -- and this drops
    # any inherited one for the same reason a runner's environment must not be
    # what satisfies the import.  The test harness exports PYTHONPATH=src:
    # experiments (``pbtest.py:450``); inheriting it would let the harness, not
    # the checkout, answer the question this file asks.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run([sys.executable, str(driver), "--help"], env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, (
        f"{name} --help exited {proc.returncode}. Its module-level imports do not "
        f"resolve against this checkout, so the job it launches would die on the "
        f"first import.\nstderr:\n{proc.stderr[-4000:]}")
    assert "--execution-json" in proc.stdout and "--layers" in proc.stdout


@pytest.mark.parametrize("name", DRIVERS)
def test_driver_reaches_the_checkout_tessera_and_not_an_installed_one(name):
    """The import that the frozen installer's tree could not satisfy."""
    src = ROOT / "src"
    code = (
        "import sys, pathlib\n"
        f"sys.path.insert(0, {str(src)!r})\n"
        "from tessera.moe_execution import ResearchSelectedMoeInput\n"
        "import tessera\n"
        f"assert pathlib.Path(tessera.__file__).resolve().parent == pathlib.Path({str(src)!r})/'tessera'\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0 and proc.stdout.strip().endswith("ok"), (
        "tessera.moe_execution does not import from this checkout.\n"
        f"stderr:\n{proc.stderr[-4000:]}")
