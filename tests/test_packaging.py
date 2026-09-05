"""What a wheel ships, checked from the tree the wheel is cut from.

An editable install reads the source tree and cannot notice a data file the
package-data table forgot; the first wheel would, at a user's first call,
as a FileNotFoundError from a JIT build or a missing runtime contract.  These
tests hold the table to the tree and the tree to the table, and hold the
plugin entry point to a name that resolves.  ``tools/check_wheel.py`` is the
other half: the same properties asserted on a built wheel, in CI, before it
is published.  Torch-free by construction so the bytes-only job runs them.
"""
from __future__ import annotations

import importlib
import re
from fnmatch import fnmatch
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

#: Every non-Python file a route or a producer opens at run time.  A new one
#: belongs in this list AND in pyproject's package-data table; the tests below
#: refuse either half on its own.
RUNTIME_DATA_SUFFIXES = (".cu", ".json")


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _package_data() -> dict[str, list[str]]:
    return _pyproject()["tool"]["setuptools"]["package-data"]


def _package_dir(package: str) -> Path:
    return SRC.joinpath(*package.split("."))


def test_every_package_data_glob_matches_a_file():
    """A glob that matches nothing is a package-data line that ships nothing."""
    for package, globs in _package_data().items():
        base = _package_dir(package)
        assert base.is_dir(), f"package-data names {package!r}, which is not a package"
        for pattern in globs:
            assert list(base.glob(pattern)), (
                f"package-data {package!r}: {pattern!r} matches no file under {base}"
            )


def test_every_runtime_data_file_in_the_tree_is_declared():
    """The other direction: a data file the table does not name is not in
    the wheel, and an editable install would never say so."""
    table = _package_data()
    undeclared = []
    for path in sorted(SRC.joinpath("tessera").rglob("*")):
        if not path.is_file() or path.suffix not in RUNTIME_DATA_SUFFIXES:
            continue
        if "__pycache__" in path.parts:
            continue
        declared = any(
            path in _package_dir(package).glob(pattern)
            for package, globs in table.items()
            for pattern in globs
        )
        if not declared:
            undeclared.append(str(path.relative_to(SRC)))
    assert not undeclared, (
        "runtime data files no package-data glob covers (a wheel would omit "
        f"them): {undeclared}"
    )


def test_the_vllm_plugin_entry_point_resolves_to_a_callable():
    """vLLM discovers the plugin by this entry point and calls what it names.
    The name has to resolve without vLLM present: registration imports vLLM
    lazily, inside the call, and this test runs where there is none."""
    entry_points = _pyproject()["project"]["entry-points"]["vllm.general_plugins"]
    assert list(entry_points) == ["tessera"]
    module_name, _, attribute = entry_points["tessera"].partition(":")
    module = importlib.import_module(module_name)
    assert callable(getattr(module, attribute)), entry_points["tessera"]


def test_the_runtime_contract_is_read_from_inside_the_serving_package():
    """The contract must resolve through importlib.resources to the file the
    package-data table ships, never through repo-root arithmetic, so a wheel,
    an editable install and a checkout read the same bytes."""
    from tessera.serving import contract

    path = contract.contract_path()
    assert path.is_file()
    shipped = _package_dir("tessera.serving") / contract.CONTRACT_FILENAME
    assert Path(str(path)).resolve() == shipped.resolve()
    assert contract.load_serving_contract()["schema"] == contract.CONTRACT_SCHEMA


def test_the_kernel_sources_resolve_as_package_resources():
    """The routes JIT-build these through importlib.resources; the serving
    package's csrc has to be reachable that way, not only as a tree path -- and
    it is the only csrc: every published ``native_extensions[].source``
    resolves there (#134)."""
    from importlib import resources

    from tessera.serving import ext
    for package, name in (
        ("tessera.serving", "csrc/window_gemv.cu"),
        ("tessera.serving", "csrc/tessera_nvfp4.cu"),
    ):
        assert resources.files(package).joinpath(name).is_file(), f"{package}: {name}"
    for entry in ext.NATIVE_EXTENSIONS:
        assert resources.files("tessera.serving").joinpath(entry["source"]).is_file(), entry


def _declared_version() -> str:
    """``pyproject.toml``'s ``[project] version`` -- the one declaration."""
    return _pyproject()["project"]["version"]


def _declared_entry_points() -> dict[str, str]:
    return _pyproject()["project"]["entry-points"]["vllm.general_plugins"]


def test_every_version_copy_derives_from_the_one_declaration():
    """The version is declared in pyproject and read from there or from the
    installed distribution's metadata -- never restated.  A literal in a
    package is a copy a release bumps or forgets, and this string is not only
    metadata: it is an input to the vLLM compile-cache key
    (``serving/compile_identity.py``), so a stale one is a stale cache."""
    import tessera
    from tessera import serving

    declared = _declared_version()
    assert tessera.__version__ == declared, (
        f"tessera.__version__ {tessera.__version__!r} != pyproject "
        f"{declared!r}: the package restates the version instead of reading it")
    assert serving.__version__ == declared, (
        f"tessera.serving.__version__ {serving.__version__!r} != pyproject "
        f"{declared!r}")


def test_the_documentation_url_names_the_version_it_documents():
    """``Documentation`` points at a tag, so it is a fifth copy of the version.
    Nothing can derive it from a static table; this is what refuses the drift."""
    url = _pyproject()["project"]["urls"]["Documentation"]
    assert f"/blob/v{_declared_version()}/" in url, (
        f"Documentation URL {url!r} does not name v{_declared_version()}")


def test_the_contract_states_no_version_the_distribution_does_not_have():
    """``runtime_contract.json`` is read by producers as an attestation about
    this package.  Whatever it stores about the package's own identity must be
    what the package has; a contract that may derive these instead of storing
    them is covered too, because what is not stored cannot drift."""
    from tessera.serving import contract

    versions = contract.load_serving_contract()["versions"]
    if "tessera" in versions:
        assert versions["tessera"] == _declared_version(), (
            f"runtime_contract versions.tessera {versions['tessera']!r} != "
            f"pyproject {_declared_version()!r}")
    if "plugin_entry_point" in versions:
        expected = "\n".join(
            f"{name} = {value}" for name, value in _declared_entry_points().items())
        assert versions["plugin_entry_point"] == expected, (
            f"runtime_contract versions.plugin_entry_point "
            f"{versions['plugin_entry_point']!r} != the declared entry point "
            f"{expected!r}")


def _excluded_packages() -> list[str]:
    """``[tool.setuptools.packages.find] exclude`` -- the one place the
    distribution says which packages in the tree it does not ship."""
    find = _pyproject()["tool"]["setuptools"]["packages"]["find"]
    return list(find.get("exclude", []))


def test_every_excluded_package_pattern_matches_a_package():
    """A pattern that matches nothing excludes nothing.

    ``tessera._dev`` is repository tooling -- the merge-suite deadline helper,
    the source-identity reader, the import-graph analyser -- that lives under
    ``src/`` because ``tools/`` imports it by module name.  One config pattern
    keeps it out of the wheel, which is why there is no roster of module names
    anywhere; but a rename would leave the pattern behind still looking like a
    policy while shipping the modules again.  ``tools/check_wheel.py`` refuses
    the built artifact; this refuses the vacuous pattern that would let the
    artifact pass by excluding nothing at all."""
    patterns = _excluded_packages()
    assert patterns, (
        "packages.find declares no exclude, so every package under src/ ships, "
        "including the repository's own tooling")
    packages = {
        ".".join(path.relative_to(SRC).parts)
        for path in SRC.rglob("*")
        if path.is_dir() and "__pycache__" not in path.parts
    }
    for pattern in patterns:
        assert any(fnmatch(package, pattern) for package in packages), (
            f"packages.find exclude {pattern!r} matches no package under "
            f"{SRC}; it excludes nothing and the modules it named ship")


def test_no_runtime_module_imports_the_excluded_tooling():
    """The exclusion is only safe while nothing shipped needs what it drops.

    An import of ``tessera._dev`` from a shipped module would install a
    package whose first call is an ImportError on a consumer's box, and no
    test that runs from this checkout would ever see it -- the tree has the
    modules the wheel does not."""
    offenders = []
    for path in sorted(SRC.joinpath("tessera").rglob("*.py")):
        if "_dev" in path.relative_to(SRC).parts or "__pycache__" in path.parts:
            continue
        if "tessera._dev" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(SRC)))
    assert not offenders, (
        "shipped modules name tessera._dev, which the wheel does not carry: "
        f"{offenders}")


def test_the_sdist_policy_names_paths_that_exist():
    """``MANIFEST.in`` is the whole sdist policy, and setuptools sweeps
    directories when the policy is silent: a directive naming a path that has
    since moved stops excluding anything, the sweep comes back, and nothing
    says so.  ``tools/check_wheel.py`` holds the built artifact; this holds
    the policy to the tree it is written against."""
    manifest = ROOT / "MANIFEST.in"
    directives = [
        line.split()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert directives, (
        "MANIFEST.in states no policy, so setuptools' directory sweep decides "
        "what the sdist ships")
    for command, *arguments in directives:
        for argument in arguments:
            if any(character in argument for character in "*?["):
                continue  # a pattern, not a path; the built-artifact check covers it
            assert (ROOT / argument).exists(), (
                f"MANIFEST.in '{command} {argument}': no such path, so the "
                "directive includes or excludes nothing")


#: Every ref a README link resolves through, whatever it is spelled as.  The
#: ``v`` is NOT in the pattern: a gate that only sees ``blob/v<something>/``
#: cannot see the one link this rule exists to catch -- a ``master`` or a bare
#: sha, which resolves from PyPI today and to a different file tomorrow.  The
#: ref is captured and compared, so an unpinned link is a finding rather than
#: a line the scan skips.
_README_LINK = re.compile(r"github\.com/RobTand/tessera/(?:blob|tree)/([^/]+)/")


def readme_link_refs(text: str) -> list:
    """The git refs the README's own links are pinned to, in order."""
    return _README_LINK.findall(text)


def test_the_readme_pins_its_links_to_the_version_it_ships_with():
    """README.md says its links are pinned to the release tag so they resolve
    from PyPI, where relative paths do not.  That makes every one of them a
    copy of the version: a bump that leaves them behind ships a page
    describing one release and linking to another."""
    declared = _declared_version()
    stale = sorted({
        ref for ref in readme_link_refs((ROOT / "README.md").read_text(encoding="utf-8"))
        if ref != f"v{declared}"
    })
    assert not stale, (
        f"README.md links resolve through {stale} but the distribution is "
        f"v{declared}; every link is pinned to the release tag, or the page "
        "documents one release and links to another")


def test_a_readme_link_on_a_moving_ref_is_a_finding_not_a_line_the_scan_skips():
    """The pre-fix failure this test was written for::

        AssertionError: ['9.9.9']
        assert 'master' in ['9.9.9']

    A link to ``blob/master/`` or to a bare sha is exactly the link that stops
    documenting this release the day master moves, and the version-shaped
    pattern was blind to both."""
    unpinned = (
        "see https://github.com/RobTand/tessera/blob/master/AGENTS.md and "
        "https://github.com/RobTand/tessera/tree/1fa2dbf0/experiments/ and "
        "https://github.com/RobTand/tessera/blob/v9.9.9/README.md\n")
    refs = readme_link_refs(unpinned)
    assert "master" in refs, refs
    assert "1fa2dbf0" in refs, refs
    assert "v9.9.9" in refs, refs


def test_the_jit_toolchain_is_named_by_one_extra_and_referenced_by_the_rest():
    """Both native routes build a packaged ``.cu`` with torch's JIT at first
    use, which needs a ``ninja``.  An extra that installs a runtime able to
    reach that build and does not carry the builder installs a consumer
    straight onto the fallback decode -- silently, because the fallback is a
    named substitute rather than an error.  The builder is named in one extra
    and referenced by the others, so there is no second copy to bump."""
    project = _pyproject()["project"]
    extras = project["optional-dependencies"]
    assert "native" in extras, (
        "no extra declares the JIT build toolchain; ninja is a build "
        "requirement of both native routes and was declared nowhere")
    reference = f"{project['name']}[native]"
    for extra, requirements in extras.items():
        if extra == "native":
            continue
        assert reference in requirements, (
            f"extra {extra!r} installs a runtime that reaches the JIT build "
            f"and does not carry {reference}; if it genuinely cannot reach "
            "one, say so here")
    restated = sorted(
        extra for extra, requirements in extras.items()
        if extra != "native" and any("ninja" in item for item in requirements))
    assert not restated, (
        f"extras {restated} name ninja directly instead of referencing "
        f"{reference}; that is the copy this test exists to prevent")
