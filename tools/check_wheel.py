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

Given the sdist as well, it holds that too, and *derived from the wheel*
rather than from a list kept here: the sdist is the source that rebuilds
this wheel, so it must carry every source the wheel ships and the build
inputs beside them, and nothing else.  setuptools' defaults decide sdist
contents by sweeping directories, which is how 149 test modules shipped
without the ``conftest.py`` that collects them (issue #151); a sweep is not
a decision, and this is where the decision is enforced.

In the other direction it holds what must *not* ship.  ``pyproject.toml``'s
``[tool.setuptools.packages.find] exclude`` keeps the repository's own
tooling (``tessera._dev``) out of the distribution; that line is intent, and
a moved module, a namespace sweep or a stale ``build/`` tree reinstates the
package without saying so.  The patterns are read from that config -- there
is no roster of module names here -- and matched against the wheel's and the
sdist's own namelists.

Usage: ``python tools/check_wheel.py dist/tessera_quant-*.whl [dist/*.tar.gz]``
"""
from __future__ import annotations

import fnmatch
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Files the wheel must contain, by path inside the wheel.  A route that
#: JIT-builds a kernel opens the .cu; the producer preflight opens the contract.
REQUIRED = (
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
from tessera.serving import ext
for package, name in (
    ("tessera.serving", "csrc/window_gemv.cu"),
    ("tessera.serving", "csrc/tessera_nvfp4.cu"),
):
    assert resources.files(package).joinpath(name).is_file(), f"{package}: {name}"
for entry in ext.NATIVE_EXTENSIONS:
    assert resources.files("tessera.serving").joinpath(entry["source"]).is_file(), entry["source"]
    assert ext.native_source_path(entry["module_name_prefix"]).endswith(entry["source"].split("/")[-1])
assert "torch" not in sys.modules, "importing the plugin's registration surface pulled torch in"
import tessera
# The installed wheel has no pyproject, so this is the metadata reader in
# tessera/__init__.py -- the path a consumer takes and a checkout never does.
assert tessera.__version__ == declared["version"], (
    f'the installed package reports {tessera.__version__}, '
    f'pyproject declares {declared["version"]}')
print("wheel check passed:", tessera.__file__, tessera.__version__)
"""


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def declared() -> dict[str, str]:
    """The distribution's identity, from its one declaration."""
    project = _pyproject()["project"]
    entry_points = project["entry-points"]["vllm.general_plugins"]
    assert list(entry_points) == ["tessera"], entry_points
    return {"name": project["name"], "version": project["version"],
            "entry_point": entry_points["tessera"]}


def excluded_packages() -> list[str]:
    """The packages the build is configured not to ship.

    Read from ``[tool.setuptools.packages.find] exclude`` rather than listed
    here, so this file holds no second roster of module names to fall out of
    date with the config that actually drives the build.
    """
    find = _pyproject()["tool"]["setuptools"]["packages"]["find"]
    return list(find.get("exclude", []))


def _package_of(member: str) -> str | None:
    """The dotted package a distribution member belongs to, or ``None``.

    ``tessera/_dev/suite_source.py`` -> ``tessera._dev``.  Only Python
    modules carry a package; data files are covered by the package-data
    table, which cannot name a package ``find`` excluded.
    """
    if not member.endswith(".py"):
        return None
    return os.path.dirname(member).replace("/", ".") or None


def refuse_excluded(label: str, members: set[str], patterns: list[str]) -> int:
    """Refuse a built artifact that ships a package the config excludes.

    ``exclude`` is a statement of intent that setuptools may or may not have
    carried out -- a moved module, a namespace sweep, a stale ``build/``
    directory all reinstate it silently.  This reads the artifact's own
    namelist and matches it against the patterns with setuptools' own
    ``fnmatch`` semantics.
    """
    shipped = sorted(
        member for member in members
        if (package := _package_of(member)) is not None
        and any(fnmatch.fnmatch(package, pattern) for pattern in patterns)
    )
    if shipped:
        print(f"{label} ships {len(shipped)} module(s) from a package "
              f"{patterns} excludes: {shipped}", file=sys.stderr)
        return 1
    return 0


#: What an sdist carries beyond the wheel's own sources: the inputs that
#: rebuild it, and the metadata setuptools regenerates on the way.  Anything
#: else in an sdist got there by a sweep nobody decided.
SDIST_BUILD_INPUTS = ("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE",
                      "PKG-INFO", "setup.cfg")


def check_sdist(sdist: Path, wheel_payload: set[str], want: dict[str, str]) -> int:
    """Hold the sdist to the wheel it has to rebuild.

    The expected set is computed from the wheel's own namelist, so it stays
    exact as the package grows and there is no roster here to fall out of
    date (AGENTS.md principle 3).
    """
    with tarfile.open(sdist) as archive:
        members = [name for name in archive.getnames()]
    root = f"{want['name'].replace('-', '_')}-{want['version']}"
    outside = [name for name in members
               if name != root and not name.startswith(f"{root}/")]
    if outside:
        print(f"{sdist.name} holds paths outside {root}/: {sorted(outside)[:5]}",
              file=sys.stderr)
        return 1
    held = {name[len(root) + 1:] for name in members} - {""}
    expected = {f"src/{name}" for name in wheel_payload} | set(SDIST_BUILD_INPUTS)
    # Directories are entries in a tar; a directory is expected exactly when
    # something expected lives under it.
    for name in sorted(expected):
        parent = os.path.dirname(name)
        while parent:
            expected.add(parent)
            parent = os.path.dirname(parent)
    # setuptools regenerates this on every build and puts it in the sdist;
    # it is its own metadata, not a source, and pruning it fights the backend.
    held = {name for name in held if ".egg-info" not in name}
    unwanted = sorted(held - expected)
    if unwanted:
        print(f"{sdist.name} ships {len(unwanted)} path(s) the wheel it rebuilds "
              f"does not need, e.g. {unwanted[:5]}", file=sys.stderr)
        return 1
    absent = sorted(expected - held)
    if absent:
        print(f"{sdist.name} cannot rebuild the wheel: {absent[:5]} missing",
              file=sys.stderr)
        return 1
    # Belt to the braces above rather than new coverage: `held == expected`
    # already excludes anything the wheel does not ship, so an sdist carrying
    # an excluded package is refused as `unwanted`.  This says *why* in the
    # message when that is the reason, and keeps holding if the derivation
    # above is ever loosened.
    stripped = {name[len("src/"):] for name in held if name.startswith("src/")}
    status = refuse_excluded(sdist.name, stripped, excluded_packages())
    if status:
        return status
    print(f"sdist check passed: {sdist.name}, {len(held)} path(s)")
    return 0


def main(argv: list[str]) -> int:
    paths = [Path(argument) for argument in argv[1:]]
    wheels = [path for path in paths if path.suffix == ".whl"]
    sdists = [path for path in paths if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) > 1 or len(wheels) + len(sdists) != len(paths):
        print(__doc__, file=sys.stderr)
        return 2
    wheel = wheels[0]
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
    payload = {name for name in names if ".dist-info/" not in name}
    status = refuse_excluded(wheel.name, payload, excluded_packages())
    if status:
        return status
    if sdists:
        return check_sdist(sdists[0], payload, want)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
