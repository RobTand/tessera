#!/usr/bin/env python3
"""Prove a built wheel ships what the plugin and the producer open at run time.

``tests/test_packaging.py`` holds the source tree to the package-data table.
This holds the *artifact*: the wheel is installed into an empty directory,
with no dependencies, and imported from there -- the source tree is not on
the path -- so a data file the table names but the build dropped, or an
entry point that resolves in the tree and not in the wheel, is refused before
``pypa/gh-action-pypi-publish`` ever sees it.  Torch-free, like the wheel's
import surface has to be.

Everything this asserts about the wheel's identity -- its version, its entry
point -- is read from ``pyproject.toml``'s one declaration rather than
restated here, so this file cannot be the copy that drifts.

Usage: ``python tools/check_wheel.py dist/tessera_quant-*.whl``
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Files the wheel must contain, by path inside the wheel.  A route that
#: JIT-builds a kernel opens the .cu; the producer preflight opens the contract.
REQUIRED = (
    "tessera/csrc/window_gemv.cu",
    "tessera/serving/csrc/window_gemv.cu",
    "tessera/serving/csrc/tessera_nvfp4.cu",
    "tessera/serving/runtime_contract.json",
)

#: Substituted into CHECK below: what pyproject declares, carried into the
#: subprocess that has only the installed wheel to read.
CHECK = r"""
import importlib.metadata as md
import json
import sys
declared = json.loads(DECLARED)
assert "torch" not in sys.modules
from importlib import resources
eps = [ep for ep in md.entry_points(group="vllm.general_plugins") if ep.name == "tessera"]
assert len(eps) == 1, f"expected one vllm.general_plugins entry point named tessera, found {eps}"
assert eps[0].value == declared["entry_point"], (eps[0].value, declared["entry_point"])
assert callable(eps[0].load())
assert md.version(declared["name"]) == declared["version"], (
    f'installed {declared["name"]} is {md.version(declared["name"])}, '
    f'pyproject declares {declared["version"]}')
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
# The installed wheel has no pyproject, so this is the metadata reader in
# tessera/__init__.py -- the path a consumer takes and a checkout never does.
assert tessera.__version__ == declared["version"], (
    f'the installed package reports {tessera.__version__}, '
    f'pyproject declares {declared["version"]}')
print("wheel check passed:", tessera.__file__, tessera.__version__)
"""


def declared() -> dict[str, str]:
    """The distribution's identity, from its one declaration."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    entry_points = project["entry-points"]["vllm.general_plugins"]
    assert list(entry_points) == ["tessera"], entry_points
    return {"name": project["name"], "version": project["version"],
            "entry_point": entry_points["tessera"]}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    wheel = Path(argv[1])
    want = declared()
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        # Located, not constructed: a wheel whose dist-info names a version
        # pyproject does not declare is the drift this refuses, so the path
        # cannot be built out of the version it is checking.
        metadata = [name for name in names
                    if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            print(f"{wheel.name} holds {len(metadata)} dist-info METADATA files",
                  file=sys.stderr)
            return 1
        declared_version = [
            line.split(": ", 1)[1]
            for line in archive.read(metadata[0]).decode().splitlines()
            if line.startswith("Version: ")]
    missing = [name for name in REQUIRED if name not in names]
    if missing:
        print(f"{wheel.name} is missing: {missing}", file=sys.stderr)
        return 1
    if declared_version != [want["version"]]:
        print(f"{wheel.name} declares Version {declared_version}, pyproject "
              f"declares {want['version']}", file=sys.stderr)
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
        check = f"DECLARED = {json.dumps(json.dumps(want))}\n" + CHECK
        subprocess.run([sys.executable, "-c", check], check=True, cwd=tmp, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
