#!/usr/bin/env python3
"""Prove a built wheel ships what the plugin and the producer open at run time.

``tests/test_packaging.py`` holds the source tree to the package-data table.
This holds the *artifact*: the wheel is installed into an empty directory,
with no dependencies, and imported from there -- the source tree is not on
the path -- so a data file the table names but the build dropped, or an
entry point that resolves in the tree and not in the wheel, is refused before
``pypa/gh-action-pypi-publish`` ever sees it.  Torch-free, like the wheel's
import surface has to be.

Usage: ``python tools/check_wheel.py dist/tessera_quant-*.whl``
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

#: Files the wheel must contain, by path inside the wheel.  A route that
#: JIT-builds a kernel opens the .cu; the producer preflight opens the contract.
REQUIRED = (
    "tessera/csrc/window_gemv.cu",
    "tessera/serving/csrc/window_gemv.cu",
    "tessera/serving/csrc/tessera_nvfp4.cu",
    "tessera/serving/runtime_contract.json",
)

CHECK = r"""
import importlib.metadata as md
import sys
assert "torch" not in sys.modules
from importlib import resources
eps = [ep for ep in md.entry_points(group="vllm.general_plugins") if ep.name == "tessera"]
assert len(eps) == 1, f"expected one vllm.general_plugins entry point named tessera, found {eps}"
assert eps[0].value == "tessera.serving:register", eps[0].value
assert callable(eps[0].load())
from tessera.serving import contract
assert contract.load_serving_contract()["schema"] == contract.CONTRACT_SCHEMA
for package, name in (
    ("tessera", "csrc/window_gemv.cu"),
    ("tessera.serving", "csrc/window_gemv.cu"),
    ("tessera.serving", "csrc/tessera_nvfp4.cu"),
):
    assert resources.files(package).joinpath(name).is_file(), f"{package}: {name}"
assert "torch" not in sys.modules, "importing the plugin's registration surface pulled torch in"
import tessera
print("wheel check passed:", tessera.__file__)
"""


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    wheel = Path(argv[1])
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    missing = [name for name in REQUIRED if name not in names]
    if missing:
        print(f"{wheel.name} is missing: {missing}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "site"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
             "--target", str(target), str(wheel)],
            check=True,
        )
        env = dict(os.environ, PYTHONPATH=str(target))
        env.pop("PYTHONSAFEPATH", None)
        # cwd is the temp dir so a source tree on the caller's path cannot
        # stand in for the wheel.
        subprocess.run([sys.executable, "-c", CHECK], check=True, cwd=tmp, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
