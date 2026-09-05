"""The impacted-test receipt describes the tree it actually classified."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "impacted_tests.py"
_SPEC = importlib.util.spec_from_file_location("impacted_tests", SCRIPT)
impacted = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(impacted)  # type: ignore[union-attr]


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Tessera test")
    (repo / "seed.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "seed.txt")
    _git(repo, "commit", "-qm", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def _selector(repo: Path, ref: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(repo), "--ref", ref, "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_parentless_snapshot_uses_a_direct_base_tree_comparison(
    tmp_path: Path,
) -> None:
    """A fetched base object is enough; synthetic snapshots have no ancestry."""

    repo, base = _repo(tmp_path)
    _git(repo, "checkout", "--orphan", "pbrun-snapshot")
    (repo / "seed.txt").write_text("branch\n", encoding="utf-8")
    _git(repo, "add", "seed.txt")
    _git(repo, "commit", "-qm", "parentless snapshot")
    head = _git(repo, "rev-parse", "HEAD")

    result = _selector(repo, f"{base}...HEAD")

    assert result["comparison"] == (
        f"direct {base}..{head} "
        f"(no merge base; requested {base}...HEAD)"
    )
    assert result["changed"] == 1
    assert result["verdict"] == "none"
    assert result["reason"] == "inert changed paths require no tests"


def test_a_closure_shaped_tracked_file_is_not_ownership_proof(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)
    names = [
        ".pbrun-closure.0123456789abcdef.json",
        ".pbrun-closure.0123456789abcde.json",
        ".pbrun-closure.0123456789abcdef0.json",
        ".pbrun-closure.0123456789abcdeF.json",
        "prefix.pbrun-closure.0123456789abcdef.json",
        ".pbrun-closure.0123456789abcdef.json.bak",
    ]
    for name in names:
        (repo / name).write_text("{}\n", encoding="utf-8")
    _git(repo, "add", *names)
    _git(repo, "commit", "-qm", "generated and lookalike names")

    changed, comparison = impacted.changed_files(f"{base}...HEAD", repo)

    assert comparison == f"{base}...HEAD"
    assert changed == sorted(names)
    result = _selector(repo, f"{base}...HEAD")
    assert result["verdict"] == "full"
    assert names[0] in result["forces_full"]


def test_verified_action_metadata_preserves_narrowed_selection(tmp_path, monkeypatch):
    from test_suite_source import _snapshot
    from tessera.suite_source import measured_source

    repo, requests, _, stamp = _snapshot(tmp_path, "gpu")
    empty = subprocess.check_output(
        ["git", "-C", str(repo), "hash-object", "-t", "tree", "-w", "--stdin"],
        input=b"",
    ).decode().strip()
    verified = []

    def inspect_source(root):
        record = measured_source(root, request_root=requests, owner="e" * 64)
        assert record["verification"] == "verified", record
        verified.append(record)
        return record

    monkeypatch.setattr(impacted, "measured_source", inspect_source, raising=False)
    changed, _ = impacted.changed_files(f"{empty}..HEAD", repo)
    assert len(verified) == 1, "selector skipped metadata without checking its action"
    assert changed == ["source.py"]
    assert verified[0]["excluded_metadata"][0]["path"] == stamp


@pytest.mark.parametrize("directory", ["tab\tname", "line\nname"])
@pytest.mark.parametrize("parentless", [False, True])
def test_quoted_path_cannot_evade_unverified_metadata_fallback(tmp_path, directory, parentless):
    repo, base = _repo(tmp_path)
    if parentless:
        _git(repo, "checkout", "--orphan", "snapshot")
    relative = f"{directory}/.pbrun-closure.0123456789abcdef.json"
    path = repo / relative
    path.parent.mkdir()
    path.write_text("{}\n")
    _git(repo, "add", relative)
    _git(repo, "commit", "-qm", "metadata-shaped tracked input")
    changed, _ = impacted.changed_files(f"{base}...HEAD", repo)
    assert changed == [relative], "display-quoted Git paths are not filesystem paths"
    result = _selector(repo, f"{base}...HEAD")
    assert result["verdict"] == "full"
    assert result["forces_full"] == [relative]


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("notes.md", "inert changed paths require no tests"),
        (
            "settings.yaml",
            "non-Python changed paths have no text-matched tests",
        ),
        (
            "module.py",
            "Python import graph found no reverse-reachable tests",
        ),
    ],
)
def test_no_test_reason_names_why_the_path_selected_nothing(
    tmp_path: Path,
    name: str,
    reason: str,
) -> None:
    repo, base = _repo(tmp_path)
    (repo / name).write_text("changed\n", encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-qm", "change")

    result = _selector(repo, f"{base}...HEAD")

    assert result["verdict"] == "none"
    assert result["reason"] == reason


def _dynamic_repo(tmp_path: Path, source: str, extra=None) -> tuple[Path, str]:
    repo, _ = _repo(tmp_path)
    files = {
        "tools/driver.py": "VALUE = 1\n",
        "tools/other.py": "VALUE = 2\n",
        "tests/test_dynamic.py": textwrap.dedent(source),
        **(extra or {}),
    }
    for relative, body in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body), encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "dynamic import fixture")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_real_route_census_path_import_is_an_edge():
    _, importers = impacted.build_graph(ROOT)
    target = impacted._module_name(ROOT / "tools/tessera_route_census.py", ROOT)
    consumer = impacted._module_name(ROOT / "tests/test_route_census_module_space.py", ROOT)
    assert consumer in importers.get(target, set()), (
        "loading the census by its file path must select its module-space regression"
    )


@pytest.mark.parametrize("source", [
    pytest.param('''
        import importlib.util as iu
        from pathlib import Path as P
        ROOT = P(__file__).resolve().parents[1]
        TOOL = ROOT / "tools" / "driver.py"
        def _load():
            return iu.spec_from_file_location("tools.other", TOOL)
    ''', id="global-path-and-module-aliases"),
    pytest.param('''
        from importlib.util import spec_from_file_location as load
        import pathlib as pl
        def _load():
            path = pl.Path(__file__).resolve().parent.parent / "tools" / "driver.py"
            return load(name="tools.other", location=path)
    ''', id="local-path-keyword-location-and-direct-alias"),
    pytest.param('''
        import importlib.util
        from pathlib import Path
        load = importlib.util.spec_from_file_location
        ROOT = Path(__file__).parents[1]
        def _load():
            return load("tools.other", ROOT / "tools" / "driver.py")
    ''', id="assigned-loader-alias-and-file-parent"),
])
def test_dynamic_import_edges_follow_the_path_not_the_module_label(tmp_path, source):
    repo, _ = _dynamic_repo(tmp_path, source)
    _, importers = impacted.build_graph(repo)
    assert "tests.test_dynamic" in importers.get("tools.driver", set())
    assert "tests.test_dynamic" not in importers.get("tools.other", set()), (
        "spec's arbitrary module label is not the file that executes"
    )
    assert "tests.test_dynamic" not in importers.get("*", set())


def test_a_loader_parameter_shadows_a_resolvable_global_path(tmp_path):
    repo, _ = _dynamic_repo(tmp_path, '''
        import importlib.util
        from pathlib import Path
        TOOL = Path(__file__).resolve().parents[1] / "tools" / "other.py"
        def _load(TOOL):
            return importlib.util.spec_from_file_location("loaded", TOOL)
    ''')
    _, importers = impacted.build_graph(repo)
    assert "tests.test_dynamic" in importers.get("*", set()), (
        "an unknown argument cannot borrow the global's unrelated file path"
    )


def test_conditional_path_reassignment_keeps_every_possible_import(tmp_path):
    repo, _ = _dynamic_repo(tmp_path, '''
        import importlib.util
        from pathlib import Path
        ROOT = Path(__file__).resolve().parents[1]
        def _load(toggle):
            path = ROOT / "tools" / "driver.py"
            if toggle:
                path = ROOT / "tools" / "other.py"
            return importlib.util.spec_from_file_location("loaded", path)
    ''')
    _, importers = impacted.build_graph(repo)
    for target in ("tools.driver", "tools.other"):
        assert "tests.test_dynamic" in (
            importers.get(target, set()) | importers.get("*", set())
        ), "a conditional assignment must not erase a possible import edge"


def test_local_path_bindings_do_not_leak_between_loader_functions(tmp_path):
    repo, _ = _dynamic_repo(tmp_path, '''
        import importlib.util
        from pathlib import Path
        ROOT = Path(__file__).resolve().parents[1]
        path = ROOT / "tools" / "driver.py"
        def _unrelated():
            path = ROOT / "tools" / "other.py"
            return path
        def _load():
            return importlib.util.spec_from_file_location("loaded", path)
    ''')
    _, importers = impacted.build_graph(repo)
    assert "tests.test_dynamic" in (
        importers.get("tools.driver", set()) | importers.get("*", set())
    ), "a different function's local must not replace the loader's global"


@pytest.mark.parametrize("changed_path", ["tools/driver.py", "settings.yaml"])
def test_unknown_loader_selects_its_static_downstream_test(tmp_path, changed_path):
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "support/dynamic.py": '''
            from importlib.util import spec_from_file_location as load_spec
            def load(location):
                return load_spec("runtime_selected", location)
        ''',
        "tests/test_consumer.py": '''
            from support.dynamic import load
            def test_consumer():
                assert callable(load)
        ''',
        "settings.yaml": "value: before\n",
    })
    (repo / changed_path).write_text("# changed\n", encoding="utf-8")
    _git(repo, "add", changed_path)
    _git(repo, "commit", "-qm", "non-inert source change")
    result = _selector(repo, f"{base}...HEAD")
    assert result["verdict"] == "narrowed"
    assert "tests/test_consumer.py" in result["tests"]
    assert "tests/test_dynamic.py" not in result["tests"]

    inert_base = _git(repo, "rev-parse", "HEAD")
    (repo / "seed.txt").write_text("notes only\n", encoding="utf-8")
    _git(repo, "add", "seed.txt")
    _git(repo, "commit", "-qm", "inert change")
    assert _selector(repo, f"{inert_base}...HEAD")["verdict"] == "none"


@pytest.mark.parametrize("indirect", [False, True], ids=["direct", "through-helper"])
def test_unknown_conftest_loader_forces_the_full_population(tmp_path, indirect):
    loader = '''
        import importlib.util
        def load(location):
            return importlib.util.spec_from_file_location("unknown", location)
    '''
    extra = {
        "tests/conftest.py": "from support.dynamic import load\n" if indirect else loader,
    }
    if indirect:
        extra["support/dynamic.py"] = loader
    repo, base = _dynamic_repo(tmp_path, "def test_example(): pass\n", extra)
    (repo / "tools/driver.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(repo, "add", "tools/driver.py")
    _git(repo, "commit", "-qm", "source change")
    assert _selector(repo, f"{base}...HEAD")["verdict"] == "full"


def test_finite_test_glob_imports_preserve_narrowed_selection(tmp_path):
    repo, base = _dynamic_repo(tmp_path, '''
        from tools.driver import VALUE
        def test_example():
            assert VALUE
    ''', {
        "tests/test_unrelated.py": "def test_unrelated(): pass\n",
        "tests/conftest.py": '''
            import importlib.util
            from pathlib import Path
            def collect_modules():
                here = Path(__file__).resolve().parent
                for path in sorted(here.glob("test_*.py")):
                    spec = importlib.util.spec_from_file_location(path.stem, path)
        ''',
    })
    _, importers = impacted.build_graph(repo)
    assert "tests.conftest" not in importers.get("*", set())
    for test_path in (repo / "tests").glob("test_*.py"):
        target = impacted._module_name(test_path, repo)
        assert "tests.conftest" in importers.get(target, set())
    (repo / "tools/driver.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(repo, "add", "tools/driver.py")
    _git(repo, "commit", "-qm", "one test's source changed")
    result = _selector(repo, f"{base}...HEAD")
    assert result["verdict"] == "narrowed"
    assert result["tests"] == ["tests/test_dynamic.py"]


@pytest.mark.parametrize("definition", [
    pytest.param('''
        def _load(TOOL=load_spec("loaded", TOOL)):
            return TOOL
    ''', id="default-evaluated-in-enclosing-scope"),
    pytest.param('''
        def decorate(value):
            return lambda function: function
        @decorate(load_spec("loaded", TOOL))
        def _load(TOOL):
            return TOOL
    ''', id="decorator-evaluated-in-enclosing-scope"),
])
def test_function_headers_keep_their_dynamic_import_edges(tmp_path, definition):
    source = '''\
from importlib.util import spec_from_file_location as load_spec
from pathlib import Path
TOOL = Path(__file__).resolve().parents[1] / "tools" / "driver.py"
''' + textwrap.dedent(definition)
    repo, _ = _dynamic_repo(tmp_path, source)
    _, importers = impacted.build_graph(repo)
    assert "tests.test_dynamic" in importers.get("tools.driver", set()), (
        "executable defaults and decorators are not part of the function's local scope"
    )


@pytest.mark.parametrize("definition", [
    pytest.param('''
        def _load(runtime_paths):
            return [load_spec("loaded", location) for location in runtime_paths]
    ''', id="comprehension-target"),
    pytest.param('''
        def _load(data):
            match data:
                case {"path": location}:
                    return load_spec("loaded", location)
    ''', id="match-pattern-target"),
])
def test_runtime_bindings_cannot_borrow_a_global_loader_path(tmp_path, definition):
    source = '''\
from importlib.util import spec_from_file_location as load_spec
from pathlib import Path
location = Path(__file__).resolve().parents[1] / "tools" / "other.py"
''' + textwrap.dedent(definition)
    repo, _ = _dynamic_repo(tmp_path, source)
    _, importers = impacted.build_graph(repo)
    assert "tests.test_dynamic" in importers.get("*", set()), (
        "a runtime binding must remain unknown, not resolve to the shadowed global"
    )


def test_shadowed_loader_callable_is_not_a_proven_import_api(tmp_path):
    repo, _ = _dynamic_repo(tmp_path, '''
        from importlib.util import spec_from_file_location as load
        from pathlib import Path
        TOOL = Path(__file__).resolve().parents[1] / "tools" / "driver.py"
        def _load(load):
            return load("loaded", TOOL)
    ''')
    _, importers = impacted.build_graph(repo)
    assert "tests.test_dynamic" in importers.get("*", set()), (
        "a known path argument does not establish an unknown callable's dependencies"
    )


@pytest.mark.parametrize(("base", "pattern"), [
    pytest.param('ROOT / ".."', "test_external_*.py", id="escaping-base"),
    pytest.param("ROOT", "../test_external_*.py", id="escaping-pattern"),
])
def test_escaping_glob_is_unknown_without_enumerating_outside_source(
    tmp_path, monkeypatch, base, pattern,
):
    source = '''\
import importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
''' + f'''
def _load():
    outside = {base}
    for path in outside.glob({pattern!r}):
        importlib.util.spec_from_file_location("loaded", path)
'''
    repo, _ = _dynamic_repo(tmp_path, source)
    original_glob = Path.glob

    def refuse_external_glob(path, pattern, *args, **kwargs):
        if "test_external_" in str(pattern):
            pytest.fail("dependency discovery tried to enumerate outside its source root")
        return original_glob(path, pattern, *args, **kwargs)

    monkeypatch.setattr(Path, "glob", refuse_external_glob)
    _, importers = impacted.build_graph(repo)
    assert "tests.test_dynamic" in importers.get("*", set())


@pytest.mark.parametrize(("definition", "target"), [
    pytest.param('def f(TOOL: load_spec("loaded", TOOL)): pass', "tools.driver", id="parameter"),
    pytest.param('def f(TOOL) -> load_spec("loaded", TOOL): pass', "tools.driver", id="return"),
    pytest.param('def f(*, TOOL: load_spec("loaded", TOOL)): pass', "tools.driver", id="keyword-only"),
    pytest.param('def f(*TOOL: load_spec("loaded", TOOL)): pass', "tools.driver", id="varargs"),
    pytest.param('def f(**TOOL: load_spec("loaded", TOOL)): pass', "tools.driver", id="kwargs"),
    pytest.param('value: load_spec("loaded", TOOL) = 1', "tools.driver", id="assignment-with-value"),
    pytest.param('value: load_spec("loaded", TOOL)', "tools.driver", id="assignment-no-value"),
    pytest.param('''
        class Example:
            def method(self, TOOL: load_spec("loaded", TOOL)): pass
    ''', "tools.driver", id="method-annotation"),
    pytest.param('''
        def f(TOOL):
            value: load_spec("loaded", TOOL)
    ''', "*", id="local-annotation-unknown-parameter"),
])
def test_annotation_expressions_keep_file_loader_edges(tmp_path, definition, target):
    source = '''\
from importlib.util import spec_from_file_location as load_spec
from pathlib import Path
TOOL = Path(__file__).resolve().parents[1] / "tools" / "driver.py"
''' + textwrap.dedent(definition) + "\n"
    repo, base = _dynamic_repo(tmp_path, source)
    _, importers = impacted.build_graph(repo)
    assert "tests.test_dynamic" in importers.get(target, set()), (
        "annotation expressions must retain their explicit or uncertain file dependency"
    )
    (repo / "tools/driver.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(repo, "add", "tools/driver.py")
    _git(repo, "commit", "-qm", "annotation's dependency changed")
    result = _selector(repo, f"{base}...HEAD")
    assert result["verdict"] == "narrowed"
    assert result["tests"] == ["tests/test_dynamic.py"]


def test_generic_annotation_shadow_does_not_borrow_or_pollute_outer_paths(tmp_path):
    import ast
    from types import SimpleNamespace
    from tessera.source_dependencies import file_imports

    source = '''\
from importlib.util import spec_from_file_location as load_spec
from pathlib import Path
TOOL = Path(__file__).resolve().parents[1] / "tools" / "driver.py"
def generic(value: load_spec("loaded", TOOL)): pass
def ordinary(value: load_spec("loaded", TOOL)): pass
'''
    repo, _ = _dynamic_repo(tmp_path, source)
    tree = ast.parse(source)
    # Exercise the type-parameter name field on every supported Python,
    # including versions whose parser predates generic-function syntax.
    tree.body[-2].type_params = [SimpleNamespace(name="TOOL")]
    found, unknown = file_imports(tree, repo / "tests/test_dynamic.py", repo)
    assert unknown, "a type parameter is not the shadowed global file path"
    assert repo / "tools/driver.py" in found, "annotation scope must not leak to a sibling"


@pytest.mark.parametrize(("definition", "target"), [
    pytest.param('''
        def consume():
            tree = ast.parse(TOOL.read_text())
            exec(compile(tree, str(TOOL), "exec"), {})
    ''', "tools.driver", id="stage-attempt-literal-path-read-and-ast-exec"),
    pytest.param('''
        def consume(driver):
            tree = ast.parse((ROOT / "tools" / driver).read_text())
            exec(compile(tree, driver, "exec"), {})
    ''', "*", id="bound-collision-parameterized-source-read"),
    pytest.param('''
        def consume():
            return ast.parse(TOOL.read_bytes())
    ''', "tools.driver", id="path-read-bytes"),
    pytest.param('''
        read_source = TOOL.read_text
        def consume():
            return ast.parse(read_source())
    ''', "tools.driver", id="aliased-path-reader"),
    pytest.param('''
        import builtins as bi
        source_open = bi.open
        def consume():
            with source_open(TOOL) as source:
                return ast.parse(source.read())
    ''', "tools.driver", id="aliased-builtin-open"),
    pytest.param('''
        def consume():
            with TOOL.open() as source:
                return ast.parse(source.read())
    ''', "tools.driver", id="path-open"),
    pytest.param('''
        def consume(TOOL):
            return ast.parse(TOOL.read_text())
    ''', "*", id="runtime-path-must-not-borrow-global"),
])
def test_explicit_source_reads_are_edges_without_importlib(tmp_path, definition, target):
    source = '''\
import ast
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "driver.py"
''' + textwrap.dedent(definition)
    repo, base = _dynamic_repo(tmp_path, source)
    _, importers = impacted.build_graph(repo)
    assert "tests.test_dynamic" in importers.get(target, set()), (
        "reading Python source couples its consumer even without an importlib loader"
    )
    (repo / "tools/driver.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(repo, "add", "tools/driver.py")
    _git(repo, "commit", "-qm", "AST-executed source changed")
    result = _selector(repo, f"{base}...HEAD")
    assert result["verdict"] == "narrowed"
    assert result["tests"] == ["tests/test_dynamic.py"]


def test_unknown_source_reader_reaches_static_test_consumers(tmp_path):
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "support/dynamic.py": '''
            import ast
            def consume(path):
                return ast.parse(path.read_text())
        ''',
        "tests/test_consumer.py": "from support.dynamic import consume\n",
    })
    (repo / "tools/driver.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(repo, "add", "tools/driver.py")
    _git(repo, "commit", "-qm", "dynamically read source changed")
    result = _selector(repo, f"{base}...HEAD")
    assert result["verdict"] == "narrowed"
    assert result["tests"] == ["tests/test_consumer.py"]


def test_a_submodule_import_is_an_edge_to_every_package_it_executes(tmp_path):
    """Importing ``pkg.inner.leaf`` runs two ``__init__`` files on the way in."""

    repo, _ = _repo(tmp_path)
    files = {
        "src/pkg/__init__.py": "VERSION = 1\n",
        "src/pkg/inner/__init__.py": "",
        "src/pkg/inner/leaf.py": "VALUE = 1\n",
        "tests/test_leaf.py": "from pkg.inner.leaf import VALUE\n",
    }
    for relative, body in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "package fixture")

    _, importers = impacted.build_graph(repo)

    for package in ("pkg", "pkg.inner", "pkg.inner.leaf"):
        assert "tests.test_leaf" in importers.get(package, set()), (
            f"a change to {package} reaches the importer of pkg.inner.leaf"
        )


def _modules_naming_the_package(root: Path, package: str) -> set[str]:
    """Test files whose own AST imports ``package`` or a submodule of it.

    Ground truth read from the tests, not from the selector's graph, so the
    two cannot agree by sharing a bug.
    """
    import ast

    found = set()
    for path in sorted((root / "tests").rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:  # pragma: no cover - a broken test file is its own bug
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            else:
                continue
            if any(name == package or name.startswith(package + ".")
                   for name in names):
                found.add(str(path.relative_to(root)))
                break
    return found


def test_a_package_init_change_selects_every_test_that_imports_the_package():
    """#148: longest-prefix attribution dropped the package edge entirely."""

    selected = set(impacted.select(ROOT, ["src/tessera/__init__.py"])["tests"])
    expected = _modules_naming_the_package(ROOT, "tessera")

    assert expected, "the fixture is vacuous if no test imports tessera"
    assert not expected - selected, (
        f"{len(expected - selected)} test modules import a tessera submodule "
        f"and were not selected: {sorted(expected - selected)[:5]}"
    )


def test_this_repository_narrows_for_a_leaf_source_edit():
    """#148: the verdict was ``full`` for every change this tree can make."""

    result = impacted.select(ROOT, ["src/tessera/wire.py"])

    assert result["verdict"] == "narrowed", result["forces_full"]
    assert result["tests"], "a narrowed verdict selecting nothing is not a selection"
    population = len(list((ROOT / "tests").rglob("test_*.py")))
    assert len(result["tests"]) < population, (
        "narrowed must mean fewer than the whole population"
    )


_GROUND_TRUTH_PROBE = '''
import importlib.util, json, sys, types
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location("under_test", root / sys.argv[2])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

candidates = list(sys.modules.values()) + [
    value for value in vars(module).values() if isinstance(value, types.ModuleType)
]
seen = set()
for loaded in candidates:
    path = getattr(loaded, "__file__", None)
    if not path:
        continue
    resolved = Path(path).resolve()
    if resolved.is_relative_to(root):
        seen.add(str(resolved.relative_to(root)))
print(json.dumps(sorted(seen)))
'''


def _files_an_import_actually_reaches(repo: Path, relative: str) -> set[str]:
    """Ground truth: import the test module for real and see what it loaded."""

    completed = subprocess.run(
        [sys.executable, "-c", _GROUND_TRUTH_PROBE, str(repo), relative],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return set(json.loads(completed.stdout))


def test_a_narrowed_selection_covers_what_importing_actually_reaches(tmp_path):
    """The selector's answer against a ground truth built by importing.

    Both halves of #148 are in this fixture: ``pkg/__init__.py`` is reached
    only through a submodule import, and ``tools/script.py`` only through an
    explicit file load.
    """

    repo, _ = _repo(tmp_path)
    files = {
        "src/pkg/__init__.py": "from pkg.core import CORE\n",
        "src/pkg/core.py": "CORE = 1\n",
        "src/pkg/leaf.py": "LEAF = 2\n",
        "tools/script.py": "SCRIPT = 3\n",
        "tests/test_core.py": "from pkg.core import CORE\ndef test_core(): assert CORE\n",
        "tests/test_leaf.py": "import pkg.leaf\ndef test_leaf(): assert pkg.leaf.LEAF\n",
        "tests/test_script.py": '''
            import importlib.util
            from pathlib import Path
            PATH = Path(__file__).resolve().parents[1] / "tools" / "script.py"
            _spec = importlib.util.spec_from_file_location("script", PATH)
            script = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(script)
            def test_script(): assert script.SCRIPT
        ''',
        "tests/test_alone.py": "def test_alone(): pass\n",
    }
    for relative, body in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body), encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "ground truth fixture")

    test_files = sorted(
        str(p.relative_to(repo)) for p in (repo / "tests").glob("test_*.py")
    )
    reached = {name: _files_an_import_actually_reaches(repo, name)
               for name in test_files}

    for edit in ("src/pkg/__init__.py", "src/pkg/core.py", "src/pkg/leaf.py",
                 "tools/script.py"):
        truth = {name for name, files in reached.items() if edit in files}
        assert truth, f"the fixture never imports {edit}"
        result = impacted.select(repo, [edit])
        assert result["verdict"] != "full", (edit, result["forces_full"])
        assert not truth - set(result["tests"]), (
            f"editing {edit} can change {sorted(truth - set(result['tests']))}, "
            f"which the selector did not select"
        )


def test_an_unnameable_data_read_is_not_an_unknown_python_dependency(tmp_path):
    """#148: a module that reads bytes it never parses is reading data.

    Calling that "any module in the tree" is what put ``tessera.suite_source``
    -- which hashes the tree and executes none of it -- in the conftest's
    dependency closure, and every run at ``full``.
    """

    repo, base = _dynamic_repo(tmp_path, '''
        from tools.driver import VALUE
        def test_example(): assert VALUE
    ''', {
        "support/reader.py": '''
            import json
            def load(path):
                return json.loads(path.read_text())
        ''',
        "tests/conftest.py": "from support.reader import load\n",
    })
    (repo / "tools/driver.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(repo, "add", "tools/driver.py")
    _git(repo, "commit", "-qm", "source change")

    result = _selector(repo, f"{base}...HEAD")

    assert result["verdict"] == "narrowed", result["forces_full"]
    assert result["tests"] == ["tests/test_dynamic.py"]
    assert result["unresolved_file_loaders"] == []


def test_a_conftest_probing_its_own_tests_does_not_inherit_their_uncertainty(tmp_path):
    """The probe closes a cycle: conftest execs every test, every test needs it."""

    repo, base = _dynamic_repo(tmp_path, '''
        import ast
        def parse(path):
            return ast.parse(path.read_text())
        def test_dynamic(): pass
    ''', {
        "tests/conftest.py": '''
            import importlib.util
            from pathlib import Path
            def collect():
                here = Path(__file__).resolve().parent
                for path in sorted(here.glob("test_*.py")):
                    importlib.util.spec_from_file_location(path.stem, path)
        ''',
    })
    (repo / "tools/driver.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(repo, "add", "tools/driver.py")
    _git(repo, "commit", "-qm", "source change")

    result = _selector(repo, f"{base}...HEAD")

    assert result["verdict"] == "narrowed", result["forces_full"]
    assert "tests/test_dynamic.py" in result["tests"], (
        "the uncertain test file is still selected; only the escalation went"
    )


def test_a_conftest_that_imports_a_test_module_outright_still_forces_full(tmp_path):
    """A probe is a file-path edge. An ``import`` is a code dependency."""

    repo, base = _dynamic_repo(tmp_path, '''
        import ast
        HELPER = 1
        def parse(path):
            return ast.parse(path.read_text())
    ''', {
        "tests/conftest.py": "from tests.test_dynamic import HELPER\n",
    })
    (repo / "tools/driver.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(repo, "add", "tools/driver.py")
    _git(repo, "commit", "-qm", "source change")

    assert _selector(repo, f"{base}...HEAD")["verdict"] == "full"


def test_a_named_data_file_is_an_edge_to_the_code_that_reads_it(tmp_path):
    """A resolved non-Python read was an unknown; it is the exact edge it is."""

    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "support/config.py": '''
            import json
            from pathlib import Path
            SPEC = Path(__file__).resolve().parents[1] / "data" / "spec.json"
            def load():
                return json.loads(SPEC.read_text())
        ''',
        "tests/test_config.py": '''
            from support.config import load
            def test_config(): assert load
        ''',
        "data/spec.json": '{"value": 1}\n',
    })
    (repo / "data/spec.json").write_text('{"value": 2}\n', encoding="utf-8")
    _git(repo, "add", "data/spec.json")
    _git(repo, "commit", "-qm", "the spec the reader names")

    result = _selector(repo, f"{base}...HEAD")

    assert result["verdict"] == "narrowed"
    assert result["tests"] == ["tests/test_config.py"]
    assert result["unresolved_file_loaders"] == []
    assert result["reason"] == (
        "non-Python changed paths reached their readers through the import graph"
    )
