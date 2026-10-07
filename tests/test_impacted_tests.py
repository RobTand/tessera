"""The impacted-test receipt describes the tree it actually classified."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "impacted_tests.py"

#: A reader under tests/ anchored with ``__file__``: its base does not depend on the process
#: directory (a relative one is kept as an unplaced read instead).
_ROOT_ANCHOR = "from pathlib import Path\n\nROOT = Path(__file__).resolve().parents[1]\n"
_SPEC = importlib.util.spec_from_file_location("impacted_tests", SCRIPT)
impacted = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(impacted)  # type: ignore[union-attr]


@pytest.fixture(scope="module")
def _checkout_import_graph():
    """Analyse the read-only checkout once; mutable fixture repos stay fresh."""
    original = impacted.import_graph
    graph = None
    guards = {}
    unnamed = {}

    def read(root, *, guarded_edges=None, unnamed_reads=None):
        nonlocal graph
        if root != ROOT:
            return original(root, guarded_edges=guarded_edges, unnamed_reads=unnamed_reads)
        if graph is None:
            graph = original(root, guarded_edges=guards, unnamed_reads=unnamed)
        if guarded_edges is not None:
            guarded_edges.update(deepcopy(guards))
        if unnamed_reads is not None:
            unnamed_reads.update(deepcopy(unnamed))
        return deepcopy(graph)

    return read


@pytest.fixture(autouse=True)
def _use_checkout_import_graph(monkeypatch, _checkout_import_graph):
    monkeypatch.setattr(impacted, "import_graph", _checkout_import_graph)


def test_readonly_checkout_queries_do_not_repeat_source_analysis(monkeypatch):
    """Two queries of this unchanged checkout need at most one source read."""
    reads = []
    original = impacted._read_python_source

    def record(path, root, unreadable):
        if path == SCRIPT:
            reads.append(path)
        return original(path, root, unreadable)

    monkeypatch.setattr(impacted, "_read_python_source", record)
    first = impacted.build_graph(ROOT)
    second = impacted.build_graph(ROOT)

    assert SCRIPT in first[0].values(), "the actual checkout was not analysed"
    assert first == second
    assert len(reads) <= 1, "repeated checkout queries rebuilt the same source graph"


def test_checkout_graph_results_and_guard_metadata_are_isolated():
    guards = {}
    first = impacted.import_graph(ROOT, guarded_edges=guards)
    expected_guards = deepcopy(guards)
    first[0].clear()
    first[1].clear()
    first[2].add(("fixture-only", "fixture-only"))
    first[3]["fixture-only"] = "fixture-only"
    for fact in guards.values():
        fact["names"].append("fixture-only")
    guards[("fixture-only", "fixture-only")] = {"names": []}

    fresh_guards = {}
    second = impacted.import_graph(ROOT, guarded_edges=fresh_guards)
    assert SCRIPT in second[0].values()
    assert second[1], "caller mutation erased the reverse graph"
    assert ("fixture-only", "fixture-only") not in second[2]
    assert "fixture-only" not in second[3]
    assert fresh_guards == expected_guards


def test_temporary_repository_queries_still_read_changed_sources(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "source.py"
    source.write_text("import first\n")
    reads = []
    original = impacted._read_python_source

    def record(path, root, unreadable):
        if path == source:
            reads.append(path)
        return original(path, root, unreadable)

    monkeypatch.setattr(impacted, "_read_python_source", record)
    impacted.build_graph(repo)
    source.write_text("this is not valid Python !!!\n")
    _, _, _, unreadable = impacted.import_graph(repo)
    assert len(reads) == 2, "a mutable temporary repository reused stale analysis"
    assert "source.py" in unreadable


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
    from test_suite_source import _git as _snapshot_git, _snapshot, _verifier
    from tessera._dev.suite_source import measured_source

    stamp = ".pbrun-closure.0123456789abcdef.json"
    repo, entry = _snapshot(tmp_path, "gpu")
    # The snapshot repo carries no identity of its own; its helper supplies one.
    _snapshot_git(repo, "mv", entry["path"], stamp)
    _snapshot_git(repo, "commit", "-qm", "generated file under its executor's name")
    entry = dict(entry, path=stamp)
    verifier, _ = _verifier(tmp_path, generated=[entry])
    empty = subprocess.check_output(
        ["git", "-C", str(repo), "hash-object", "-t", "tree", "-w", "--stdin"],
        input=b"",
    ).decode().strip()
    verified = []

    def inspect_source(root):
        record = measured_source(root, verifier=verifier)
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


_READER_MODULE = '''\
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"


def _markdown():
    return sorted(DOCS.rglob("*.md"))


def test_reads_every_doc():
    assert all(path.read_text() for path in _markdown())
'''


def _docs_reader_repo(tmp_path: Path) -> tuple[Path, str]:
    """A tracked doc and a test whose only coupling to it is the rglob.

    The reader names no file: neither the changed path nor its basename
    appears in its source, so neither the import graph nor the text fallback
    can see it.  The directory-wide enumeration is the whole relationship.
    """
    repo, _ = _repo(tmp_path)
    (repo / "docs").mkdir()
    (repo / "docs" / "ARCHITECTURE.md").write_text("# base\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_docs_reader.py").write_text(
        _READER_MODULE, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "doc and its directory-wide reader")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_a_docs_change_selects_its_directory_wide_reader(tmp_path):
    """tessera#923, TS921: a docs change must select the tests that rglob docs/.

    On the unfixed selector the verdict here was ``none`` -- "inert changed
    paths require no tests" -- while the full run caught what the narrowed
    list dropped.
    """
    repo, base = _docs_reader_repo(tmp_path)
    (repo / "docs" / "ARCHITECTURE.md").write_text("# changed\n", encoding="utf-8")
    _git(repo, "add", "docs/ARCHITECTURE.md")
    _git(repo, "commit", "-qm", "docs-only change")

    result = _selector(repo, f"{base}...HEAD")

    assert result["verdict"] == "narrowed", result
    assert "tests/test_docs_reader.py" in result["tests"], result


def test_a_new_doc_selects_the_directory_reader(tmp_path):
    repo, base = _docs_reader_repo(tmp_path)
    (repo / "docs" / "extra.md").write_text("# new\n", encoding="utf-8")
    _git(repo, "add", "docs/extra.md")
    _git(repo, "commit", "-qm", "new doc")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_docs_reader.py" in result["tests"], result


def test_a_deleted_doc_still_selects_the_directory_reader(tmp_path):
    """A deleted member is the reader's concern too: the last triage doc
    going away is exactly what the existence guard exists to catch."""
    repo, base = _docs_reader_repo(tmp_path)
    _git(repo, "rm", "docs/ARCHITECTURE.md")
    _git(repo, "commit", "-qm", "delete the doc")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_docs_reader.py" in result["tests"], result


_GLOB_READERS = {
    # Each reader names no file: the directory-wide glob is the whole relationship.
    "names-only": 'def test_lists():\n    assert sorted(DOCS.glob("*.md"))\n',
    "reads-each": ('def test_reads():\n    for path in sorted(DOCS.glob("*.md")):\n'
                   '        assert path.read_text()\n'),
    "recursive": ('def test_reads():\n    for path in DOCS.glob("**/*.md"):\n'
                  '        assert path.read_text()\n'),
    # The receiver is still named: bound to the directory, or the explicit self
    # of the unbound method.
    "bound-alias": 'scan = DOCS.glob\n\n\ndef test_lists():\n    assert sorted(scan("*.md"))\n',
    "unbound-alias": ('original = Path.glob\n\n\ndef test_lists():\n'
                      '    assert sorted(original(DOCS, "*.md"))\n'),
    "direct-unbound": 'def test_lists():\n    assert sorted(Path.glob(DOCS, "*.md"))\n',
    "direct-unbound-rglob": 'def test_lists():\n    assert sorted(Path.rglob(DOCS, "*.md"))\n',
}


def _glob_reader_repo(tmp_path: Path, spelling: str) -> tuple[Path, str]:
    repo, _ = _repo(tmp_path)
    (repo / "docs" / "nested").mkdir(parents=True)
    (repo / "docs" / "first.md").write_text("# one\n", encoding="utf-8")
    (repo / "docs" / "nested" / "second.md").write_text("# two\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_glob_reader.py").write_text(
        "from pathlib import Path\n\n"
        'DOCS = Path(__file__).resolve().parents[1] / "docs"\n\n\n'
        + _GLOB_READERS[spelling], encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "docs and a glob reader")
    return repo, _git(repo, "rev-parse", "HEAD")


def _change_docs(repo: Path, change: str) -> None:
    if change == "modify":
        (repo / "docs" / "first.md").write_text("# changed\n", encoding="utf-8")
        _git(repo, "add", "docs/first.md")
    elif change == "add":
        (repo / "docs" / "extra.md").write_text("# new\n", encoding="utf-8")
        _git(repo, "add", "docs/extra.md")
    elif change == "add-nested":
        (repo / "docs" / "nested" / "deeper").mkdir()
        (repo / "docs" / "nested" / "deeper" / "third.md").write_text("# new\n", encoding="utf-8")
        _git(repo, "add", "docs/nested/deeper/third.md")
    elif change == "delete":
        _git(repo, "rm", "-q", "docs/first.md")
    elif change == "delete-nested":
        _git(repo, "rm", "-q", "docs/nested/second.md")
    elif change == "rename":
        _git(repo, "mv", "docs/first.md", "docs/renamed.md")
    else:
        raise ValueError(change)
    _git(repo, "commit", "-qm", change)


@pytest.mark.parametrize("spelling, change", [
    ("names-only", "add"), ("names-only", "delete"), ("names-only", "rename"),
    ("reads-each", "delete"),
    ("recursive", "modify"), ("recursive", "add"), ("recursive", "add-nested"),
    ("recursive", "delete"), ("recursive", "delete-nested"), ("recursive", "rename"),
    ("bound-alias", "add"), ("bound-alias", "delete"),
    ("unbound-alias", "add"), ("unbound-alias", "delete"),
    ("direct-unbound", "add"), ("direct-unbound", "delete"),
    ("direct-unbound-rglob", "add"), ("direct-unbound-rglob", "delete"),
])
def test_a_membership_change_selects_a_glob_reader(tmp_path, spelling, change):
    """PB1496: ``Path.glob`` consumes a directory's membership like ``rglob`` does.

    Before this, a deleted member of a glob that reads its matches, an added,
    deleted or renamed member of a names-only glob, and every change under the
    recursive spelling gave verdict ``none``: the full run caught what the
    narrowed list dropped.
    """
    repo, base = _glob_reader_repo(tmp_path, spelling)
    _change_docs(repo, change)

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_glob_reader.py" in result["tests"], result


@pytest.mark.parametrize("spelling", sorted(_GLOB_READERS))
def test_a_glob_reader_is_not_selected_by_an_unrelated_directory(tmp_path, spelling):
    """The directory node is the base the glob names, not every directory."""
    repo, base = _glob_reader_repo(tmp_path, spelling)
    (repo / "elsewhere").mkdir()
    (repo / "elsewhere" / "first.md").write_text("# same basename, other directory\n", encoding="utf-8")
    _git(repo, "add", "elsewhere/first.md")
    _git(repo, "commit", "-qm", "an unrelated directory")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_glob_reader.py" not in result["tests"], result


@pytest.mark.parametrize("pattern, linked", [
    ("../data/*.md", False),
    ("link/*.md", True),
], ids=["parent-component", "directory-symlink"])
def test_a_glob_pattern_that_leaves_its_receiver_selects_the_reader(tmp_path, pattern, linked):
    """A pattern with separators can read another directory's membership.

    ``docs`` is the receiver, but ``../data/*.md`` and ``link/*.md`` (``link``
    pointing at ``data``) list ``data``: a member added there must still select
    the reader, though the receiver directory never changed (PB1496 review).
    """
    repo, _ = _repo(tmp_path)
    (repo / "docs").mkdir()
    (repo / "data").mkdir()
    (repo / "docs" / "keep.md").write_text("# keep\n", encoding="utf-8")
    (repo / "data" / "first.md").write_text("# first\n", encoding="utf-8")
    if linked:
        (repo / "docs" / "link").symlink_to("../data", target_is_directory=True)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_pattern_reader.py").write_text(
        "from pathlib import Path\n\n"
        'DOCS = Path(__file__).resolve().parents[1] / "docs"\n\n\n'
        f'def test_lists():\n    assert sorted(DOCS.glob({pattern!r}))\n',
        encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "a reader whose pattern leaves its receiver")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "data" / "extra.md").write_text("# new\n", encoding="utf-8")
    _git(repo, "add", "data/extra.md")
    _git(repo, "commit", "-qm", "a member of the other directory")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_pattern_reader.py" in result["tests"], result


def test_a_named_base_with_an_unknown_pattern_keeps_its_dependency(tmp_path):
    """A resolvable base with a dynamic pattern names the directory but not
    the membership: the #338 unplaced-read uncertainty, never a silent drop."""
    repo, _ = _repo(tmp_path)
    (repo / "docs").mkdir()
    (repo / "docs" / "ARCHITECTURE.md").write_text("# base\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_docs_reader.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        'DOCS = Path(__file__).resolve().parents[1] / "docs"\n'
        'PATTERN = os.environ.get("PATTERN", "*.md")\n'
        "def test_reads_matches():\n"
        "    assert all(path.read_text() for path in DOCS.rglob(PATTERN))\n",
        encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "reader with a dynamic pattern")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "seed.txt").write_text("changed\n", encoding="utf-8")
    _git(repo, "add", "seed.txt")
    _git(repo, "commit", "-qm", "unrelated inert change")

    result = _selector(repo, f"{base}...HEAD")

    assert result["unplaced_data_reads"] == ["tests/test_docs_reader.py"], result
    assert "tests/test_docs_reader.py" in result["tests"], result


def _listing_conftest_repo(tmp_path: Path) -> tuple[Path, str]:
    """A conftest that lists the repository root to bootstrap import roots."""
    repo, _ = _repo(tmp_path)
    (repo / "src" / "pkg").mkdir(parents=True)
    (repo / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "src" / "pkg" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "conftest.py").write_text(
        "from pathlib import Path\n"
        "REPO = Path(__file__).resolve().parents[1]\n"
        "def _roots():\n"
        "    return sorted(child.name for child in REPO.iterdir())\n"
        "ROOTS = _roots()\n",
        encoding="utf-8")
    (repo / "tests" / "test_consumer.py").write_text(
        "from pkg.mod import VALUE\n\n\ndef test_value():\n    assert VALUE\n",
        encoding="utf-8")
    (repo / "tests" / "test_other.py").write_text(
        "def test_other():\n    assert True\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "listing conftest fixture")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_a_conftest_enumeration_is_a_probe_not_a_whole_tree_edge(tmp_path):
    """The conftest's root listing is collection machinery, not data coupling.

    pytest imports the conftest for every test in its scope whatever changed,
    so a per-change edge would let every changed path's ancestor listing reach
    the conftest and hold every verdict at full (#148).  The edge is kept and
    probe-marked: excluded from the walks that select or scope, like the
    exec-probe edge one paragraph of the selector's contract already removes.
    """
    repo, base = _listing_conftest_repo(tmp_path)
    (repo / "src" / "pkg" / "mod.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "add", "src/pkg/mod.py")
    _git(repo, "commit", "-qm", "module change")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_consumer.py" in result["tests"], result
    assert "tests/test_other.py" not in result["tests"], result
    _, importers, probes, _ = impacted.import_graph(repo)
    assert (".", "tests.conftest") in probes, sorted(probes)
    assert "tests.conftest" in importers.get(".", set()), importers


def test_a_docs_architecture_change_selects_the_issue_ref_guard():
    """TS921, on the tree this selector actually serves.

    The real 317-file selector run omitted tests/test_issue_refs.py for a
    docs/ARCHITECTURE.md change and the pure CI run then caught the dangling
    reference the guard exists for.  docs/ is consumed through
    DOCS.rglob("*.md"), a directory-wide read with no nameable file.
    """
    result = impacted.select(ROOT, ["docs/ARCHITECTURE.md"])

    assert "tests/test_issue_refs.py" in result["tests"], result


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
    # A prose suffix does not prove the unknown loader left the file unread
    # (PB1496): this used to be verdict ``none``.  The control that an inert
    # change with no unknown loader stays ``none`` is
    # test_no_test_reason_names_why_the_path_selected_nothing.
    inert = _selector(repo, f"{inert_base}...HEAD")
    assert inert["verdict"] == "narrowed"
    assert "tests/test_consumer.py" in inert["tests"]


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
    from tessera._dev.source_dependencies import file_imports

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
    found, unknown, _ = file_imports(tree, repo / "tests/test_dynamic.py", repo)
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

    result = impacted.select(ROOT, ["src/tessera/__init__.py"])
    selected = set(result["tests"])
    excluded = {entry["path"] for entry in result["excluded_tests"]}
    expected = _modules_naming_the_package(ROOT, "tessera")

    assert expected, "the fixture is vacuous if no test imports tessera"
    missing = expected - selected - excluded
    assert not missing, (
        f"{len(missing)} test modules import a tessera submodule "
        f"and were neither selected nor explicitly excluded: {sorted(missing)[:5]}"
    )
    assert not selected & excluded


def test_an_opaque_path_outranks_the_generic_inert_suffix_rule():
    """#216: the wire is named in OPAQUE and is written in Markdown.

    ``docs/schema/`` is in ``OPAQUE`` because a wire change is exactly what
    nothing here can reason about -- and the classification excluded every
    ``.md`` path before that rule was ever consulted, so the one file the rule
    exists for was the one file it could not reach.
    """

    opaque_docs = sorted(
        str(path.relative_to(ROOT))
        for prefix in impacted.OPAQUE
        for path in ROOT.glob(prefix + "*")
        if path.is_file() and path.suffix in impacted.INERT
    )
    assert opaque_docs, "no opaque path has an inert suffix; fixture is vacuous"

    for changed in opaque_docs:
        result = impacted.select(ROOT, [changed])
        assert result["verdict"] == "full", (changed, result)
        assert changed in result["forces_full"], (changed, result)

    # Ordinary Markdown cannot force the population by suffix. Refused
    # data readers can still select their consumers conservatively (#355).
    ordinary = impacted.select(ROOT, ["docs/measurements/a-note-2026-09-05.md"])
    assert ordinary["verdict"] == ("narrowed" if ordinary["tests"] else "none"), ordinary
    assert ordinary["forces_full"] == [], ordinary


def _leaves_outside_the_shared_conftest(root: Path) -> list[str]:
    """Source files ``tests/conftest.py`` does not reach, derived from the graph.

    Which files those are is not this test's business to pin -- the shared
    conftest imports the container, the grammar and the layout, and a file
    below any of those is one pytest re-imports for every test in ``tests/``.
    The rule is that a source file OUTSIDE that closure still narrows.
    """

    by_name, importers, probes, _ = impacted.import_graph(root)
    leaves = []
    for name, path in sorted(by_name.items()):
        if not str(path.relative_to(root)).startswith("src/"):
            continue
        if "tests.conftest" in impacted._reverse_reachable({name}, importers,
                                                           skip=probes):
            continue
        leaves.append(str(path.relative_to(root)))
    return leaves


def test_this_repository_narrows_for_a_leaf_source_edit():
    """#148: the verdict was ``full`` for every change this tree can make."""

    population = len(list((ROOT / "tests").rglob("test_*.py")))
    leaves = _leaves_outside_the_shared_conftest(ROOT)
    assert leaves, "no source file is outside the shared conftest's closure"

    # Sample the ends and the middle of the derived list to exercise distinct
    # paths through the selector with the same checkout graph.
    for leaf in {leaves[0], leaves[len(leaves) // 2], leaves[-1]}:
        result = impacted.select(ROOT, [leaf])
        assert result["verdict"] == "narrowed", (leaf, result["forces_full"])
        assert result["tests"], f"{leaf}: a narrowed verdict selecting nothing"
        assert len(result["tests"]) < population, (
            f"{leaf}: narrowed must mean fewer than the whole population")


def test_a_file_the_shared_conftest_imports_selects_every_test_below_it():
    """The other half of the same rule, and it is not a regression (#215).

    ``tests/conftest.py`` imports the container, so pytest re-imports the wire
    for every test in ``tests/`` whether or not that test names it.  Selecting
    fewer than all of them was under-selection, and the receipt says which
    edge widened it rather than leaving a reader to wonder.
    """

    result = impacted.select(ROOT, ["src/tessera/wire.py"])
    population = {str(p.relative_to(ROOT))
                  for p in (ROOT / "tests").rglob("test_*.py")}

    assert result["verdict"] == "narrowed", result["forces_full"]
    excluded = {entry["path"] for entry in result["excluded_tests"]}
    assert not population - (set(result["tests"]) | excluded)
    assert not set(result["tests"]) & excluded
    assert "conftest" in result["reason"], result["reason"]


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

    Calling that "any module in the tree" is what put ``tessera._dev.suite_source``
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


def test_a_conftest_helper_selects_the_tests_that_request_its_fixtures(tmp_path):
    """#215: pytest imports the conftest; the tests never import the helper.

    A fixture consumer names the fixture, not the module the fixture's value
    came from.  So the only path from ``tests/helper.py`` to
    ``tests/test_consumer.py`` runs through the conftest, and it runs the way
    pytest runs it -- by import at collection time, for every test at or below
    the conftest's directory.  Reaching the conftest and stopping there
    selected nothing at all.
    """

    repo, base = _dynamic_repo(tmp_path, "def test_dynamic(): pass\n", {
        "tests/helper.py": "VALUE = 1\n",
        "tests/conftest.py": '''
            import pytest
            from helper import VALUE

            @pytest.fixture
            def value():
                return VALUE
        ''',
        "tests/test_consumer.py": '''
            def test_consumer(value):
                assert value
        ''',
    })
    (repo / "tests/helper.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "add", "tests/helper.py")
    _git(repo, "commit", "-qm", "the fixture's helper changed")

    result = _selector(repo, f"{base}...HEAD")

    assert result["verdict"] == "narrowed", result["forces_full"]
    assert "tests/test_consumer.py" in result["tests"], result


def test_a_changed_conftest_dependency_names_the_scope_it_selected(tmp_path):
    """The receipt says WHY tests nothing imports the changed file appeared."""

    repo, base = _dynamic_repo(tmp_path, "def test_dynamic(): pass\n", {
        "support/state.py": "VALUE = 1\n",
        "tests/conftest.py": "from support.state import VALUE\n",
    })
    (repo / "support/state.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "add", "support/state.py")
    _git(repo, "commit", "-qm", "a conftest dependency changed")

    result = _selector(repo, f"{base}...HEAD")

    assert result["tests"] == ["tests/test_dynamic.py"], result
    assert "conftest" in result["reason"], result["reason"]


def test_a_relative_re_export_is_an_edge_from_the_child_to_the_package(tmp_path):
    """#215: ``from .child import VALUE`` in a package's own initializer.

    ``src/tessera/__init__.py`` is spelled exactly this way.  The package name
    of an ``__init__.py`` is the package itself, not its parent, so the
    relative import resolved to a top-level ``child`` that does not exist and
    the ``pkg.child -> pkg`` edge was dropped.
    """

    repo, base = _repo(tmp_path)
    files = {
        "src/pkg/__init__.py": "from .child import VALUE\n",
        "src/pkg/child.py": "VALUE = 1\n",
        "src/pkg/inner/__init__.py": "from ..child import VALUE as V\n",
        "tests/test_consumer.py": "from pkg import VALUE\ndef test_v(): assert VALUE\n",
    }
    for relative, body in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "relative re-export fixture")

    _, importers = impacted.build_graph(repo)
    assert "pkg" in importers.get("pkg.child", set()), (
        "an initializer's relative import climbs from its own package")
    assert "pkg.inner" in importers.get("pkg.child", set()), (
        "a two-dot import from a subpackage's initializer climbs one level")

    (repo / "src/pkg/child.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "add", "src/pkg/child.py")
    _git(repo, "commit", "-qm", "the re-exported child changed")
    result = _selector(repo, f"{base}...HEAD")
    assert result["verdict"] == "narrowed", result["forces_full"]
    assert "tests/test_consumer.py" in result["tests"], result


def _modules_importing_bare(root: Path, name: str) -> set[str]:
    """Test files whose own AST imports ``name`` with no package qualifier.

    Ground truth read from the tests themselves, so the selector's graph and
    the expectation cannot agree by sharing a bug.
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
            if any(spelled == name or spelled.startswith(name + ".")
                   for spelled in names):
                found.add(str(path.relative_to(root)))
                break
    return found


@pytest.mark.parametrize("helper", ["box_artifacts", "conftest"])
def test_pytest_import_roots_resolve_the_bare_helper_imports(helper):
    """#215: ``tests/`` is on ``sys.path``, so ``import box_artifacts`` works.

    The graph named the file ``tests.box_artifacts`` and nothing else, while
    every test that uses it spells it bare -- the spelling pytest's own
    rootdir/basedir rule makes valid, and the one ``tests/conftest.py``
    guarantees by inserting its own directory.  Those imports resolved to no
    node, so editing the helper selected none of its consumers.
    """

    consumers = _modules_importing_bare(ROOT, helper)
    assert consumers, f"the fixture is vacuous if no test imports {helper}"
    _, importers = impacted.build_graph(ROOT)
    reached = {
        str((ROOT / m.replace(".", "/")).with_suffix(".py").relative_to(ROOT))
        for m in importers.get(f"tests.{helper}", set())
    }
    assert not consumers - reached, sorted(consumers - reached)


def test_a_box_artifacts_edit_selects_the_tests_that_read_its_roots():
    """#215's own example: the shipped-checkpoint gate reads those roots."""

    result = impacted.select(ROOT, ["tests/box_artifacts.py"])

    assert result["verdict"] != "full", result["forces_full"]
    consumers = _modules_importing_bare(ROOT, "box_artifacts")
    excluded = {entry["path"] for entry in result["excluded_tests"]}
    missing = consumers - set(result["tests"]) - excluded
    assert not missing, sorted(missing)
    assert not set(result["tests"]) & excluded
    shipped = "tests/test_shipped_checkpoint_minor7.py"
    if (ROOT / shipped).exists():
        assert shipped in result["tests"], "the audit's named example"


def test_an_unparseable_file_beside_the_change_is_uncertain_not_a_crash(tmp_path):
    """A file the branch is mid-edit on must not take the selector down.

    ``_imports`` answers with three sets and answered a syntax error with one,
    so every caller unpacked it into a ``ValueError`` -- and a selector that
    raises selects nothing at all, which is the failure mode this tool exists
    to refuse.  Nothing imports this one, so its uncertainty reaches no test
    and the narrowed list is unchanged; the receipt still names it (#293).
    """

    repo, base = _dynamic_repo(tmp_path, '''
        from tools.driver import VALUE
        def test_example(): assert VALUE
    ''', {"support/broken.py": "def (\n"})
    (repo / "tools/driver.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(repo, "add", "tools/driver.py")
    _git(repo, "commit", "-qm", "source change beside an unparseable file")

    _, importers = impacted.build_graph(repo)
    assert importers.get("support.broken", set()) == set()
    result = _selector(repo, f"{base}...HEAD")
    assert result["tests"] == ["tests/test_dynamic.py"], result
    assert result["verdict"] == "narrowed", result
    assert "SyntaxError" in result["unreadable_sources"]["support/broken.py"]


def test_an_unparseable_importer_of_the_change_is_selected_not_erased(tmp_path):
    """#293: a file that does not parse states no dependency, not none.

    ``_imports`` answered a ``SyntaxError``/``OSError`` with three empty sets,
    which reads as proof that the module imports nothing.  A test that imports
    the changed leaf and does not parse therefore lost its edge, and the
    selected branch check ran nothing while collecting that test would fail.
    """

    repo, _ = _repo(tmp_path)
    files = {
        "src/pkg/__init__.py": "",
        "src/pkg/leaf.py": "VALUE = 1\n",
        "tests/test_consumer.py": '''
            from pkg.leaf import VALUE
            def test_consumer(): assert VALUE
            def (
        ''',
        "tests/test_unrelated.py": "def test_unrelated(): pass\n",
    }
    for relative, body in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body), encoding="utf-8")

    result = impacted.select(repo, ["src/pkg/leaf.py"])

    assert "tests/test_consumer.py" in result["tests"], result
    assert "tests/test_unrelated.py" not in result["tests"], (
        "one unreadable file must not widen the selection to the population")
    assert "SyntaxError" in result["unreadable_sources"]["tests/test_consumer.py"]
    assert "tests/test_consumer.py" in result["reason"], result["reason"]
    assert result["unresolved_file_loaders"] == [], (
        "an unreadable file is not an unresolved file loader")


def test_an_unreadable_source_names_the_file_and_the_read_failure(tmp_path):
    """The other half of the same `except`: the file could not be read at all.

    A broken link, a permission, a file the branch deleted under the analysis.
    The operator has to be told which file and what went wrong to repair it.
    """

    repo, _ = _repo(tmp_path)
    (repo / "src/pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "src/pkg/__init__.py").write_text("", encoding="utf-8")
    (repo / "src/pkg/leaf.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "tests/test_consumer.py").symlink_to(repo / "tests/gone.py")

    result = impacted.select(repo, ["src/pkg/leaf.py"])

    assert "tests/test_consumer.py" in result["tests"], result
    reason = result["unreadable_sources"]["tests/test_consumer.py"]
    assert reason.startswith("FileNotFoundError"), reason


def test_an_unreadable_conftest_forces_the_population_it_gates(tmp_path):
    """#293: a conftest that cannot be parsed makes its whole scope unknown.

    Every test at or below it is collected through it, so nothing below it can
    be reasoned about -- the same escalation an unresolved loader in a conftest
    already gets.
    """

    repo, _ = _repo(tmp_path)
    files = {
        "src/pkg/__init__.py": "",
        "src/pkg/leaf.py": "VALUE = 1\n",
        "tests/conftest.py": "import pytest\ndef (\n",
        "tests/test_unrelated.py": "def test_unrelated(): pass\n",
    }
    for relative, body in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body), encoding="utf-8")

    result = impacted.select(repo, ["src/pkg/leaf.py"])

    assert result["verdict"] == "full", result
    assert "tests/conftest.py" in result["forces_full"], result
    assert "SyntaxError" in result["unreadable_sources"]["tests/conftest.py"]


def test_supported_syntax_still_narrows_and_reports_nothing_unreadable(tmp_path):
    """The control: the uncertainty is the failure, not the file's presence."""

    repo, _ = _repo(tmp_path)
    files = {
        "src/pkg/__init__.py": "",
        "src/pkg/leaf.py": "VALUE = 1\n",
        "tests/conftest.py": "import pytest\n",
        "tests/test_consumer.py": (
            "from pkg.leaf import VALUE\ndef test_consumer(): assert VALUE\n"),
        "tests/test_unrelated.py": "def test_unrelated(): pass\n",
    }
    for relative, body in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body), encoding="utf-8")

    result = impacted.select(repo, ["src/pkg/leaf.py"])

    assert result["verdict"] == "narrowed", result
    assert result["tests"] == ["tests/test_consumer.py"], result
    assert result["unreadable_sources"] == {}, result


def test_this_repository_has_no_unreadable_source():
    """A tree with an unreadable file cannot narrow, so say when one appears."""

    result = impacted.select(ROOT, ["src/tessera/kernel.py"])
    assert result["unreadable_sources"] == {}, result["unreadable_sources"]


def _shadowed_repo(tmp_path: Path) -> Path:
    """Two files with one canonical name, and the shadowed one is what runs.

    ``_module_name`` strips a leading ``src/``, so ``helper.py`` and
    ``src/helper.py`` are both ``helper``.  ``tests/conftest.py`` inserts
    ``src/`` and then its own directory, exactly as this repository's does at
    lines 11-12, so ``from helper import VALUE`` resolves to ``src/helper.py``
    -- the file the graph dropped.
    """

    repo, _ = _repo(tmp_path)
    files = {
        "src/pkg/__init__.py": "",
        "src/pkg/leaf.py": "VALUE = 1\n",
        "helper.py": "VALUE = 'root'\n",
        "src/helper.py": "from pkg.leaf import VALUE\n",
        "tests/conftest.py": '''
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
            sys.path.insert(0, str(Path(__file__).resolve().parent))
        ''',
        "tests/test_consumer.py": _IMPORT_ROOT_PROBE,
        "tests/test_unrelated.py": "def test_unrelated(): pass\n",
    }
    for relative, body in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body), encoding="utf-8")
    return repo


def test_a_file_shadowed_on_its_canonical_name_still_has_a_node(tmp_path):
    """#317: every file gets a node, so every file's own imports are edges."""

    repo = _shadowed_repo(tmp_path)

    by_name, _, _, _ = impacted.import_graph(repo)
    analysed = {str(p.relative_to(repo)) for p in by_name.values()}
    assert {"helper.py", "src/helper.py"} <= analysed, sorted(analysed)


def test_a_shadowed_files_own_imports_are_not_lost(tmp_path):
    """#317: the under-selection the missing node caused.

    ``src/helper.py`` was never parsed, so the ``pkg.leaf -> helper`` edge did
    not exist and a change to the leaf selected nothing at all -- while pytest
    imports that very file for ``from helper import VALUE``.
    """

    repo = _shadowed_repo(tmp_path)
    executed = _helper_pytest_executes(repo, tmp_path / "probe.out")
    assert executed == "src/helper.py", (
        f"the fixture is not the shadowing case if pytest runs {executed}")

    result = impacted.select(repo, ["src/pkg/leaf.py"])

    assert result["verdict"] == "narrowed", result
    assert "tests/test_consumer.py" in result["tests"], result
    assert "tests/test_unrelated.py" not in result["tests"], result


@pytest.mark.parametrize("changed", ["helper.py", "src/helper.py"])
def test_either_file_spelling_one_name_selects_the_bare_consumer(tmp_path, changed):
    """Which of the two `sys.path` gives the importer is unmodelled, so union."""

    repo = _shadowed_repo(tmp_path)

    result = impacted.select(repo, [changed])

    assert result["verdict"] == "narrowed", result
    assert "tests/test_consumer.py" in result["tests"], result


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


_IMPORT_ROOT_PROBE = '''
import os

import helper
from helper import VALUE


def test_which_helper_pytest_executed():
    with open(os.environ["TESSERA_PROBE_OUT"], "w", encoding="utf-8") as out:
        out.write(helper.__file__)
    assert VALUE
'''


def _helper_pytest_executes(repo: Path, out: Path) -> str:
    """Import-root probe: run pytest for real and record which file it imported.

    The graph cannot model import-root precedence, so the test may not assume
    an answer either.  This asks pytest.
    """

    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    env["TESSERA_PROBE_OUT"] = str(out)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_consumer.py",
         "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=str(repo), capture_output=True, text=True, env=env, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    executed = Path(out.read_text(encoding="utf-8")).resolve()
    return str(executed.relative_to(repo.resolve()))


def test_a_canonical_short_name_does_not_outrank_its_alias_candidates(tmp_path):
    """#292: a root ``helper.py`` must not hide ``tests/helper.py``.

    ``_targets`` returned the canonical module alone whenever the spelling was
    a canonical name, so the alias candidate on the import root that
    ``tests/conftest.py`` inserts -- the one pytest actually executes for
    ``from helper import VALUE`` -- got no edge, and editing it selected
    nothing.  Nothing here models import-root precedence, so the honest answer
    is the union: both candidates get the edge.
    """

    repo, _ = _repo(tmp_path)
    files = {
        "helper.py": "VALUE = 'root'\n",
        "tests/helper.py": "VALUE = 'tests'\n",
        "tests/conftest.py": '''
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parent))
        ''',
        "tests/test_consumer.py": _IMPORT_ROOT_PROBE,
    }
    for relative, body in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body), encoding="utf-8")

    candidates = ["helper.py", "tests/helper.py"]
    executed = _helper_pytest_executes(repo, tmp_path / "probe.out")
    assert executed in candidates, executed

    _, importers = impacted.build_graph(repo)
    executed_name = impacted._module_name(repo / executed, repo)
    assert "tests.test_consumer" in importers.get(executed_name, set()), (
        f"pytest imports {executed} for `from helper import VALUE`, and the "
        "graph holds no edge from it")
    for candidate in candidates:
        name = impacted._module_name(repo / candidate, repo)
        assert "tests.test_consumer" in importers.get(name, set()), candidate
        result = impacted.select(repo, [candidate])
        assert result["verdict"] == "narrowed", (candidate, result)
        assert "tests/test_consumer.py" in result["tests"], (candidate, result)


def _alias_reader_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    """The #338 fixture: an in-tree JSON a reader names through an alias.

    ``alias/`` is a symlink to the checkout, so ``alias/data/...`` and
    ``repo/data/...`` are the same bytes with two spellings.  The alias is
    local environment state -- nothing in the repository records it -- which
    is exactly why the selector must not need to resolve it, and exactly why
    refusing to resolve it cannot be read as "no dependency".  Everything
    here is a disposable file under ``tmp_path``; no mount is involved.
    """
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "data").mkdir()
    data = root / "data" / "runtime-settings.json"
    data.write_text('{"value": 1}\n', encoding="utf-8")
    reader = root / "src" / "pkg" / "reader.py"
    reader.write_text(
        "from pathlib import Path\n"
        f"TARGET = Path({str(alias / 'data/runtime-settings.json')!r})\n"
        "def load():\n    return TARGET.read_text()\n",
        encoding="utf-8",
    )
    (root / "tests").mkdir()
    (root / "tests" / "test_consumer.py").write_text(
        "from pkg.reader import load\n"
        "def test_consumer():\n    assert load()\n",
        encoding="utf-8",
    )
    return root, reader, data


def test_outside_spelled_data_read_still_selects_its_reader(tmp_path: Path) -> None:
    """tessera#338 -- refusing to resolve an alias is not proof of independence.

    The reader's bytes change when the tracked JSON changes; editing the JSON
    must therefore reach the reader's test.  Before this, the lexical guard
    refused the outside spelling and ``wildcard(reading)`` reported nothing
    for a plain reader, so the receipt was ``verdict='none'``, ``tests=[]``,
    with no unresolved or unreadable entry naming the file it had dropped --
    under-selection by a tool whose stated contract is that it never
    under-selects.
    """
    root, reader, data = _alias_reader_tree(tmp_path)

    result = impacted.select(root, ["data/runtime-settings.json"])

    assert result["verdict"] == "narrowed", result
    assert "tests/test_consumer.py" in result["tests"], result
    # The receipt has to SAY the dependency is unplaced; the defect was a
    # silent drop, and a selection with no diagnostic repeats it.
    assert "src/pkg/reader.py" in result["unplaced_data_reads"], result
    assert "refused to place" in result["reason"], result["reason"]


def test_outside_spelled_read_is_not_promoted_to_an_unknown_importer(
    tmp_path: Path,
) -> None:
    """The dependency is data, not "any module in the tree" (#148 stands).

    A reader that executes nothing imports nothing, so it must not appear as
    an unresolved file loader, and it must not force the full run.
    """
    root, _, _ = _alias_reader_tree(tmp_path)

    result = impacted.select(root, ["data/runtime-settings.json"])

    assert result["unresolved_file_loaders"] == [], result
    assert result["forces_full"] == [], result
    _, importers = impacted.build_graph(root)
    assert "pkg.reader" not in importers.get("*", set()), (
        "a plain data reader is not an unknown Python importer")


def test_in_root_spelling_of_the_same_read_keeps_its_exact_edge(
    tmp_path: Path,
) -> None:
    """The negative control from the #338 proof: nothing about narrowing moved.

    Spelling the identical file from inside the tree still yields the exact
    data edge -- not the unplaced fallback -- so the fix buys uncertainty
    only where the resolver actually declined to look.
    """
    root, reader, _ = _alias_reader_tree(tmp_path)
    reader.write_text(
        reader.read_text(encoding="utf-8").replace(
            str(tmp_path / "alias"), str(root)),
        encoding="utf-8",
    )

    result = impacted.select(root, ["data/runtime-settings.json"])

    assert result["verdict"] == "narrowed", result
    assert result["tests"] == ["tests/test_consumer.py"], result
    assert result["unplaced_data_reads"] == [], result
    _, importers = impacted.build_graph(root)
    assert importers.get("data/runtime-settings.json") == {"pkg.reader"}


def test_an_unnameable_read_still_states_no_dependency(tmp_path: Path) -> None:
    """#148's rule is untouched: a target that was never NAMED is silence.

    Only a named-and-refused target survives its refusal.  A runtime filename
    names no file at all, so it must remain what it was -- no edge, no
    unplaced read, no wildcard -- or every reader depends on every file again.
    """
    root, reader, _ = _alias_reader_tree(tmp_path)
    reader.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "def load(name):\n"
        "    return (Path(os.environ['DIR']) / name).read_text()\n",
        encoding="utf-8",
    )

    result = impacted.select(root, ["data/runtime-settings.json"])

    assert result["verdict"] == "none", result
    assert result["tests"] == [], result
    assert result["unplaced_data_reads"] == [], result
    _, importers = impacted.build_graph(root)
    assert "pkg.reader" not in importers.get("*", set())
    assert "pkg.reader" not in importers.get("*data", set())


@pytest.mark.parametrize("suffix", [".md", ".txt", ".rst", ".json"])
@pytest.mark.parametrize("alias", [True, False], ids=["refused-alias", "exact-edge"])
def test_text_data_changes_reach_readers_regardless_of_suffix(tmp_path, monkeypatch, suffix, alias):
    root, reader, original_data = _alias_reader_tree(tmp_path)
    data = original_data.with_suffix(suffix)
    original_data.rename(data)
    source = reader.read_text().replace("runtime-settings.json", data.name)
    if not alias:
        source = source.replace(str(tmp_path / "alias"), str(root))
    reader.write_text(source)
    # Execute only this disposable local fixture to establish observable impact.
    namespace = {}
    exec(source, namespace)
    before = namespace["load"]()
    data.write_text("changed bytes\n")
    assert namespace["load"]() != before

    alias_path = tmp_path / "alias"
    def guarded(original):
        def call(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                assert not Path(os.fsdecode(path)).is_relative_to(alias_path), (
                    "selector approached the refused alias")
            return original(path, *args, **kwargs)
        return call

    for name in ("stat", "lstat", "readlink", "scandir", "listdir"):
        monkeypatch.setattr(os, name, guarded(getattr(os, name)))
    result = impacted.select(root, [str(data.relative_to(root))])
    assert result["verdict"] == "narrowed", result
    assert result["tests"] == ["tests/test_consumer.py"], result
    assert "require no tests" not in result["reason"]
    assert result["unplaced_data_reads"] == (["src/pkg/reader.py"] if alias else [])
    assert result["unresolved_file_loaders"] == []
    assert result["forces_full"] == []


@pytest.mark.parametrize("suffix", [".md", ".rst", ".txt"])
def test_helper_read_selects_a_test_naming_an_inert_input(tmp_path, suffix):
    repo, _ = _repo(tmp_path)
    (repo / "tests").mkdir()
    name = "contract" + suffix
    (repo / name).write_text("contract\n")
    (repo / "tests/test_contract.py").write_text(
        "from pathlib import Path\n"
        "ROOT = Path(__file__).resolve().parents[1]\n"
        "def _read(rel): return (ROOT / rel).read_text()\n"
        f"def test_contract(): assert _read({name!r})\n"
    )
    result = impacted.select(repo, [name])
    assert result["verdict"] == "narrowed"
    assert "tests/test_contract.py" in result["tests"]
    assert "inert changed paths require no tests" not in result["reason"]


def test_readme_selects_its_existing_helper_read_contract():
    result = impacted.select(ROOT, ["README.md"])
    assert "tests/test_doc_scope_69.py" in result["tests"]


@pytest.mark.parametrize("directory", [False, True])
def test_repointing_a_tracked_data_link_selects_transitive_readers(tmp_path, directory):
    repo, _ = _repo(tmp_path)
    for folder in ["data/first", "data/second", "support", "tests"]:
        (repo / folder).mkdir(parents=True)
    for folder in ["first", "second"]:
        (repo / "data" / folder / "payload.json").write_text(folder)
    link = repo / "data/chosen"
    link.symlink_to("first" if directory else "first/payload.json")
    expression = 'Path("data/chosen") / "payload.json"' if directory else 'Path("data/chosen")'
    (repo / "support/reader.py").write_text(
        'from pathlib import Path\n' + f'PAYLOAD = ({expression}).read_text()\n')
    (repo / "tests/test_reader.py").write_text('from support.reader import PAYLOAD\n')
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "reader and its tracked data link")
    base = _git(repo, "rev-parse", "HEAD")
    link.unlink()
    link.symlink_to("second" if directory else "second/payload.json")
    _git(repo, "add", "data/chosen")
    _git(repo, "commit", "-qm", "repoint data input")
    result = _selector(repo, f"{base}...HEAD")
    assert result["verdict"] == "narrowed"
    assert "tests/test_reader.py" in result["tests"]


@pytest.mark.parametrize("read", [
    "path.read_text()", "path.read_bytes()", "open(path).read()",
])
@pytest.mark.parametrize("binding, call", [
    ("from support.compiler import execute", "execute(text)"),
    ("from support.compiler import execute as rebuild", "rebuild(text)"),
    ("import support.compiler as compiler", "compiler.execute(text)"),
])
def test_split_source_reader_keeps_unknown_execution_edges(tmp_path, read, binding, call):
    """#808: moving exec to a helper must not erase the reader's uncertainty."""
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "support/reader.py": f'''
            {binding}
            def consume(path):
                text = {read}
                return {call}
        ''',
        "support/compiler.py": '''
            def execute(text):
                exec(text, {})
        ''',
        "tests/conftest.py": "from support.reader import consume\n",
    })
    (repo / "tools/driver.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(repo, "add", "tools/driver.py")
    _git(repo, "commit", "-qm", "source an unknown reader can execute changed")
    result = _selector(repo, f"{base}...HEAD")
    assert result["verdict"] == "full", result
    assert result["unresolved_file_loaders"] == ["support/reader.py"]


@pytest.mark.parametrize("expression", [
    "Path(stock.__file__).read_text()",
    "Path(stock.__file__).read_bytes()",
    "open(stock.__file__).read()",
])
def test_unproven_import_source_read_keeps_unknown_origin(tmp_path, expression):
    """#808: an absent import origin is unknown, not proven external source."""
    repo, base = _dynamic_repo(tmp_path, '''
        from tools.driver import VALUE
        def test_dynamic(): assert VALUE
    ''', {
        "support/reader.py": f'''
            from pathlib import Path
            import foreign_runtime as stock
            def consume():
                text = {expression}
                exec(text, {{}})
        ''',
        "tests/conftest.py": "from support.reader import consume\n",
    })
    (repo / "tools/driver.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(repo, "add", "tools/driver.py")
    _git(repo, "commit", "-qm", "leaf change does not prove imported source origin")
    result = _selector(repo, f"{base}...HEAD")
    assert result["verdict"] == "full", result
    assert result["tests"] == ["tests/test_dynamic.py"]
    assert result["unresolved_file_loaders"] == ["support/reader.py"]


_UNKNOWN_READER = '''
    def consume(handle):
        exec("pass", {})
        return open(handle).read()
'''


@pytest.mark.parametrize("suffix", [".md", ".txt", ".rst"])
def test_a_doc_change_cannot_hide_a_conftest_unknown_reader(tmp_path, suffix):
    """PB1496: an inert suffix is not proof that an unknown loader cannot read it.

    ``consume`` executes source and opens a path nothing here can name, so it
    may read any file the diff holds, a Markdown note included.  A conftest
    imports it, so every test below that conftest is unpredictable and a diff
    holding only prose must still force the population, as a non-inert
    suffix already does.
    """
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "support/reader.py": _UNKNOWN_READER,
        "tests/conftest.py": "from support.reader import consume\n",
        f"docs/note{suffix}": "before\n",
    })
    (repo / f"docs/note{suffix}").write_text("after\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "prose changed")

    result = _selector(repo, f"{base}...HEAD")

    assert result["verdict"] == "full", result
    assert result["unresolved_file_loaders"] == ["support/reader.py"], result


@pytest.mark.parametrize("suffix", [".md", ".txt", ".rst"])
def test_a_doc_change_selects_the_consumers_of_an_unknown_reader(tmp_path, suffix):
    """The same unknown reader behind a test, not a conftest, selects that test."""
    repo, base = _dynamic_repo(
        tmp_path,
        "from support.reader import consume\n\n\ndef test_reads(): pass\n",
        {"support/reader.py": _UNKNOWN_READER, f"docs/note{suffix}": "before\n"})
    (repo / f"docs/note{suffix}").write_text("after\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "prose changed")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_dynamic.py" in result["tests"], result


_UNNAMED_BASE_READER = '''
    import os
    from pathlib import Path

    DIR = Path(os.environ["READ_DIR"])


    def test_reads():
        assert sorted(DIR.rglob("*.md"))
'''


def test_a_directory_read_of_an_unnameable_base_is_listed_not_selected(tmp_path):
    """PB1496: the documented limit stays, and is no longer silent.

    A reader that executes nothing and enumerates a directory nothing names
    states no dependency, so a docs change selects nothing (#148).  The result
    lists the reader and the line, and the text receipt prints it, so a reader
    of this shape is seen and not just unselected.
    """
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "tests/test_r.py": _UNNAMED_BASE_READER, "docs/a.md": "before\n"})
    (repo / "docs/a.md").write_text("after\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "docs changed")

    result = _selector(repo, f"{base}...HEAD")
    line = textwrap.dedent(_UNNAMED_BASE_READER).splitlines().index(
        '    assert sorted(DIR.rglob("*.md"))') + 1
    assert result["unnamed_directory_reads"] == {"tests/test_r.py": [line]}, result
    assert "tests/test_r.py" not in result["tests"], result

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(repo), "--ref", f"{base}...HEAD"],
        capture_output=True, text=True, check=False)
    assert "base cannot be named" in completed.stdout, completed.stdout
    assert "tests/test_r.py" in completed.stdout, completed.stdout


@pytest.mark.parametrize("reader, why", [
    ('from pathlib import Path\n\nDIR = Path(__file__).resolve().parent / "data"\n\n\n'
     'def test_reads():\n    assert sorted(DIR.rglob("*.md"))\n', "the base is named"),
    ('import os\nfrom pathlib import Path\n\nDIR = Path(os.environ["READ_DIR"])\n\n\n'
     'def test_reads():\n    exec("pass")\n    assert sorted(DIR.rglob("*.md"))\n',
     "a module that executes source is an unknown loader, escalated not listed"),
])
def test_only_the_silent_directory_read_is_listed(tmp_path, reader, why):
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "tests/test_r.py": reader, "docs/a.md": "before\n"})
    (repo / "docs/a.md").write_text("after\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "docs changed")

    result = _selector(repo, f"{base}...HEAD")

    assert result["unnamed_directory_reads"] == {}, (why, result)


# What the real tree held when this guard was added: modules with a directory
# read whose base nothing names, in a module that executes nothing, and the
# read sites in them.  Most take the directory as a parameter or are test
# scaffolding, and some are the standard library's glob.glob(pattern), whose
# base sits in the pattern string; the count is a ceiling, not a verdict on any
# of them (it was 46 and 58 before Path.glob became an enumeration, and 117 and 178
# before a function the file defines stopped being read as os.walk).  A new one
# should name its base; if it cannot, raise these numbers in the same commit
# and say why (PB1496).  Measured again on master 30418c445 (2026-10-07): 114 and
# 167, one below the 115 and 168 pinned before, so the ceiling follows it down
# (#1014).
_UNNAMED_DIRECTORY_READ_MODULES = 114
_UNNAMED_DIRECTORY_READ_SITES = 167


def _exceeds_unnamed_directory_read_ceiling(modules: int, sites: int) -> bool:
    return modules > _UNNAMED_DIRECTORY_READ_MODULES or sites > _UNNAMED_DIRECTORY_READ_SITES


@pytest.mark.parametrize("modules, sites, exceeds", [
    (_UNNAMED_DIRECTORY_READ_MODULES, _UNNAMED_DIRECTORY_READ_SITES, False),
    (_UNNAMED_DIRECTORY_READ_MODULES - 1, _UNNAMED_DIRECTORY_READ_SITES, False),
    (_UNNAMED_DIRECTORY_READ_MODULES, _UNNAMED_DIRECTORY_READ_SITES - 1, False),
    (_UNNAMED_DIRECTORY_READ_MODULES + 1, _UNNAMED_DIRECTORY_READ_SITES, True),
    (_UNNAMED_DIRECTORY_READ_MODULES, _UNNAMED_DIRECTORY_READ_SITES + 1, True),
    # Fewer modules with more sites is still a gained read; a tuple comparison
    # orders on the module count first and lets it through.
    (_UNNAMED_DIRECTORY_READ_MODULES - 1, _UNNAMED_DIRECTORY_READ_SITES + 1, True),
    (1, _UNNAMED_DIRECTORY_READ_SITES + 1, True),
    (_UNNAMED_DIRECTORY_READ_MODULES + 1, 1, True),
])
def test_the_unnamed_directory_read_ceiling_bounds_modules_and_sites_independently(
        modules, sites, exceeds):
    assert _exceeds_unnamed_directory_read_ceiling(modules, sites) is exceeds


def test_a_local_wrapper_around_walk_still_selects_its_reader(tmp_path):
    """A wrapper named like an enumeration is the reader of the directory it is called with."""
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "tests/test_wrapper.py": (
            "import os\nfrom pathlib import Path\n\n\ndef walk(root):\n    return os.walk(root)\n\n\n"
            "def test_reads():\n    assert list(walk(Path('docs')))\n"),
        "docs/a.md": "before\n"})
    (repo / "docs/b.md").write_text("new\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "a member added")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_wrapper.py" in result["tests"], result


_FIXTURE_CONFTEST = (
    "import os\nfrom pathlib import Path\nimport pytest\n{imports}\n\n{probe}\n\n\n"
    "@pytest.fixture\ndef value():\n    return {value}\n")


def _fixture_repo(tmp_path, *, imports, probe, value):
    # No tests/__init__.py: pytest's import-root alias resolver reads this layout.
    return _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "tests/conftest.py": _FIXTURE_CONFTEST.format(imports=imports, probe=probe, value=value),
        "tests/test_values.py": "VALUE = 1\n",
        "tests/test_consumer.py": "def test_value(value):\n    assert value == 1\n"})


@pytest.mark.parametrize("probe", [
    "pass",
    "Path('tests/test_values.py').read_text()",
    "list(os.walk(Path('tests/test_values.py')))",
], ids=["no-probe", "path-read", "walk-of-path"])
def test_an_ordinary_import_is_not_masked_by_a_probe_of_the_same_file(tmp_path, probe):
    """PB1496 review: a conftest that imports ``test_values`` AND reaches the file by
    path has one dependency that matters, the import.  The probe exclusion is keyed
    by the (target, importer) pair, so it also removed the ordinary import's edge and
    the fixture's consumer was never selected when the helper changed."""
    repo, base = _fixture_repo(tmp_path, imports="from test_values import VALUE",
                               probe=probe, value="VALUE")
    (repo / "tests/test_values.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "the helper the fixture imports changed")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_consumer.py" in result["tests"], result


def test_a_probe_alone_still_does_not_select_the_fixture_consumers(tmp_path):
    """The exclusion stays for what it is for: a conftest that only reaches a test file
    by path is collection machinery, not a dependency of the tests it serves."""
    repo, base = _fixture_repo(tmp_path, imports="",
                               probe="Path('tests/test_values.py').read_text()", value="1")
    (repo / "tests/test_values.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "a test file the conftest only probes changed")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_consumer.py" not in result["tests"], result


@pytest.mark.parametrize("spelling", [
    "os.listdir(str(ROOT / 'docs'))", "list(os.walk(str(ROOT / 'docs')))",
    "list(os.scandir(str(ROOT / 'docs')))",
], ids=["listdir", "walk", "scandir"])
def test_a_string_path_directory_reader_is_selected_by_a_new_member(tmp_path, spelling):
    """A reader that lists a directory by string path depends on its membership."""
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "tests/test_lister.py": (
            "import os\n" + _ROOT_ANCHOR + f"\n\ndef test_lists():\n    assert {spelling}\n"),
        "docs/a.md": "before\n"})
    (repo / "docs/b.md").write_text("new\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "a member added")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_lister.py" in result["tests"], result


def test_a_string_argument_does_not_hide_an_unknown_loader_from_its_consumers(tmp_path):
    """A module that executes source and calls a custom ``walk(str(ROOT / "mode"))`` stays an
    unknown loader: resolving the string as a directory adds an edge, it must not
    replace the wildcard, or a change it may load selects its consumer no more."""
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "support/helper.py": (
            _ROOT_ANCHOR + "\n\ndef walk(mode):\n    return mode\n\n\ndef run():\n    exec('pass')\n"
            "    return walk(str(ROOT / 'mode'))\n"),
        "tests/test_consumer.py": "from support.helper import run\n\n\ndef test_run():\n    assert run\n",
        "mode/seed.txt": "x\n"})
    (repo / "tools/driver.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "a source file the unknown loader may execute changed")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_consumer.py" in result["tests"], result


@pytest.mark.parametrize("probe", [
    "list(os.walk(str(Path(__file__).resolve().parent / 'test_values.py')))",
    "os.listdir(str(Path(__file__).resolve().parent / 'test_values.py'))",
], ids=["walk-of-string", "listdir-of-string"])
def test_a_string_path_probe_does_not_mask_an_ordinary_import_of_the_same_file(tmp_path, probe):
    """PB1496 review: resolving a string adds an edge, and in a conftest that edge can be
    a collection probe of a file the conftest ALSO imports.  The ordinary import is the
    dependency that matters; masking it dropped the fixture's consumers.  The string is
    anchored with ``__file__`` (a relative one is unplaced, which would select the reader
    whatever the probe did).  No tests/__init__.py: the pytest import-root alias
    resolver reads this layout."""
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "tests/conftest.py": (
            "import os\nfrom pathlib import Path\nimport pytest\nfrom test_values import VALUE\n\n"
            + probe + "\n\n\n@pytest.fixture\ndef value():\n    return VALUE\n"),
        "tests/test_values.py": "VALUE = 1\n",
        "tests/test_consumer.py": "def test_value(value):\n    assert value == 1\n"})
    (repo / "tests/test_values.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "the helper the fixture imports changed")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_consumer.py" in result["tests"], result


def test_a_string_path_directory_reader_is_not_selected_by_another_directory(tmp_path):
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "tests/test_lister.py": ("import os\n" + _ROOT_ANCHOR + "\n\ndef test_lists():\n"
                                 "    assert os.listdir(str(ROOT / 'docs'))\n"),
        "docs/a.md": "before\n", "elsewhere/x.md": "other\n"})
    (repo / "elsewhere/y.md").write_text("new\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "a member added elsewhere")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_lister.py" not in result["tests"], result


@pytest.mark.parametrize("reader", [
    "import glob\n" + _ROOT_ANCHOR + "\n\ndef test_lists():\n    assert sorted(glob.glob(str(ROOT / 'docs' / '*.md')))\n",
    "import glob\n" + _ROOT_ANCHOR + "\n\ndef test_lists():\n    assert sorted(glob.iglob(str(ROOT / 'docs' / '*.md')))\n",
    "from glob import glob\n" + _ROOT_ANCHOR + "\n\ndef test_lists():\n    assert sorted(glob(str(ROOT / 'docs' / '*.md')))\n",
], ids=["module-glob", "module-iglob", "from-import"])
def test_a_module_glob_reader_is_selected_by_a_new_member(tmp_path, reader):
    """#1010: a reader listing a directory with the module function depends on its
    membership, exactly as one using ``Path.glob`` does."""
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "tests/test_lister.py": reader, "docs/a.md": "before\n"})
    (repo / "docs/b.md").write_text("new\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "a member added")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_lister.py" in result["tests"], result


def test_a_module_glob_reader_is_not_selected_by_another_directory(tmp_path):
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "tests/test_lister.py": ("import glob\n" + _ROOT_ANCHOR + "\n\ndef test_lists():\n"
                                 "    assert glob.glob(str(ROOT / 'docs' / '*.md'))\n"),
        "docs/a.md": "before\n", "elsewhere/x.md": "other\n"})
    (repo / "elsewhere/y.md").write_text("new\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "a member added elsewhere")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_lister.py" not in result["tests"], result


def test_a_glob_reader_is_selected_by_a_new_member_behind_a_wildcard_link(tmp_path):
    """#1011: ``docs/link`` is a link to ``data``; ``docs/*/x.md`` lists it too."""
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "tests/test_lister.py": (
            "from pathlib import Path\n\nDOCS = Path(__file__).resolve().parents[1] / 'docs'\n\n\n"
            "def test_lists():\n    assert list(DOCS.glob('*/*.md')) is not None\n"),
        "docs/plain/a.md": "x\n", "data/seed.md": "x\n"})
    (repo / "docs" / "link").symlink_to("../data", target_is_directory=True)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "a link under the globbed directory")
    base = _git(repo, "rev-parse", "HEAD")
    # A basename the reader never names, so the text fallback cannot select it.
    (repo / "data" / "later.md").write_text("new\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "a member added behind the link")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_lister.py" in result["tests"], result


_LINK_READER = (
    "from pathlib import Path\n\nDOCS = Path(__file__).resolve().parents[1] / 'docs'\n\n\n"
    "def test_lists():\n    assert list(DOCS.glob({pattern!r})) is not None\n")


def test_a_literal_link_after_a_wildcard_selects_the_reader_for_its_target(tmp_path):
    """The wildcard reaches an ordinary directory, the literal ``link`` leads to ``data``."""
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "tests/test_lister.py": _LINK_READER.format(pattern="*/link/*.md"),
        "docs/plain/a.md": "x\n", "data/seed.md": "x\n"})
    (repo / "docs" / "plain" / "link").symlink_to("../../data", target_is_directory=True)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "a link behind an ordinary directory")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "data" / "later.md").write_text("new\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "a member added behind the link")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_lister.py" in result["tests"], result


def test_deleting_the_last_member_behind_an_unchanged_link_selects_the_reader(tmp_path):
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "tests/test_lister.py": _LINK_READER.format(pattern="*/*.md"),
        "docs/plain/a.md": "x\n", "data/seed.md": "x\n"})
    (repo / "docs" / "link").symlink_to("../data", target_is_directory=True)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "a link to data")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "rm", "-q", "data/seed.md")
    _git(repo, "commit", "-qm", "the last member behind the link goes")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_lister.py" in result["tests"], result


def test_an_outside_bridge_behind_a_wildcard_selects_the_reader_for_any_change(tmp_path):
    """``docs/link`` points outside the tree; the bridge may lead back to ``data``.
    The selector will not look, so the reader cannot be tied to a file and is kept
    for every change (#338): ``data/later.md`` is changed and unmentioned."""
    outside = tmp_path / "outside"
    outside.mkdir()
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "tests/test_lister.py": _LINK_READER.format(pattern="*/*.md"),
        "docs/plain/a.md": "x\n", "data/seed.md": "x\n"})
    (repo / "docs" / "link").symlink_to(outside, target_is_directory=True)
    (outside / "bridge").symlink_to(repo / "data", target_is_directory=True)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "a link that leaves the tree")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "data" / "later.md").write_text("new\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "an unmentioned member of data")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_lister.py" in result["tests"], result


def test_a_module_glob_after_a_chdir_selects_the_reader_for_the_runtime_directory(tmp_path):
    """``os.chdir(.../nested)`` then ``glob.glob('docs/*.md')`` lists ``nested/docs``, not
    the root's ``docs``.  A member added there, under a basename the reader never
    names, must select it."""
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "tests/test_cwd.py": (
            "import glob\nimport os\nfrom pathlib import Path\n\n\ndef test_lists():\n"
            "    os.chdir(Path(__file__).resolve().parents[1] / 'nested')\n"
            "    assert sorted(glob.glob('docs/*.md'))\n"),
        "nested/docs/a.md": "x\n", "docs/a.md": "x\n"})
    (repo / "nested" / "docs" / "later.md").write_text("new\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "a member added under the runtime directory")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_cwd.py" in result["tests"], result


def test_deleting_the_last_member_behind_a_link_selects_a_directory_only_glob(tmp_path):
    """``docs/*/`` lists directories, so deleting ``data``'s only file changes its result."""
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "tests/test_lister.py": ("import glob\n" + _ROOT_ANCHOR + "\n\ndef test_lists():\n"
                                 "    assert glob.glob(str(ROOT / 'docs' / '*') + '/') is not None\n"),
        "docs/plain/a.md": "x\n", "data/seed.md": "x\n"})
    (repo / "docs" / "link").symlink_to("../data", target_is_directory=True)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "a link to data")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "rm", "-q", "data/seed.md")
    _git(repo, "commit", "-qm", "data loses its only file")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_lister.py" in result["tests"], result


def test_editing_the_file_a_terminal_link_points_to_selects_the_reader(tmp_path):
    """The reader names ``manifest.md``; the changed file is ``payload.md``, a basename it
    never mentions, so only the link makes it a dependency."""
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "tests/test_lister.py": (
            "from pathlib import Path\n\nDOCS = Path(__file__).resolve().parents[1] / 'docs'\n\n\n"
            "def test_reads():\n    for p in DOCS.glob('*/manifest.md'):\n"
            "        assert p.read_text() == 'expected'\n"),
        "docs/plain/a.md": "x\n", "data/payload.md": "expected\n"})
    (repo / "docs" / "plain" / "manifest.md").symlink_to("../../data/payload.md")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "a file link behind a wildcard directory")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "data" / "payload.md").write_text("changed\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "the target file changes")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_lister.py" in result["tests"], result


def test_a_module_glob_after_an_imported_cwd_helper_selects_the_reader(tmp_path):
    """``support.cwd.enter()`` changes to ``nested``; the test then lists ``docs/*.md``.  The
    reader's tree contains no chdir and the helper no enumeration, so only treating a
    relative base as unplaced selects it for ``nested/docs/later.md`` (a basename the
    reader never names)."""
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "support/cwd.py": "import os\nfrom pathlib import Path\n\n\ndef enter():\n    os.chdir(Path(__file__).resolve().parents[1] / 'nested')\n",
        "tests/test_cwd.py": (
            "import glob\n\nfrom support.cwd import enter\n\n\ndef test_lists():\n"
            "    enter()\n    assert sorted(glob.glob('docs/*.md'))\n"),
        "nested/docs/a.md": "x\n", "docs/a.md": "x\n"})
    (repo / "nested" / "docs" / "later.md").write_text("new\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "a member added under the runtime directory")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_cwd.py" in result["tests"], result


def test_an_anchored_module_glob_is_selected_only_by_its_own_directory(tmp_path):
    """The precise case: anchored with ``__file__``, so another directory does not select it."""
    reader = ("import glob\nfrom pathlib import Path\n\nDOCS = Path(__file__).resolve().parents[1] / 'docs'\n\n\n"
              "def test_lists():\n    assert glob.glob(str(DOCS / '*.md'))\n")
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "tests/test_lister.py": reader, "docs/a.md": "x\n", "elsewhere/x.md": "x\n"})
    (repo / "elsewhere" / "y.md").write_text("new\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "a member added elsewhere")
    elsewhere = _selector(repo, f"{base}...HEAD")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "docs" / "b.md").write_text("new\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "a member added under docs")
    own = _selector(repo, f"{base}...HEAD")

    assert "tests/test_lister.py" not in elsewhere["tests"], elsewhere
    assert "tests/test_lister.py" in own["tests"], own


def test_an_anchored_module_glob_behind_an_outside_bridge_selects_the_reader_for_any_change(tmp_path):
    """``docs`` leaves the tree and the bridge may lead back to ``data``: a named in-tree base the
    guard declines keeps the reader for every change (``data/later.md``, a basename it never
    names)."""
    outside = tmp_path / "outside"
    outside.mkdir()
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "tests/test_lister.py": ("import glob\n" + _ROOT_ANCHOR + "\n\ndef test_lists():\n"
                                 "    assert glob.glob(str(ROOT / 'docs' / '*.md')) is not None\n"),
        "data/seed.md": "x\n"})
    (outside / "bridge").symlink_to(repo / "data", target_is_directory=True)
    (repo / "docs").symlink_to(outside, target_is_directory=True)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "docs leaves the tree")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "data" / "later.md").write_text("new\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "an unmentioned member of data")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_lister.py" in result["tests"], result


@pytest.mark.parametrize("order", ["outside-first", "inside-first"])
def test_an_outside_alternative_does_not_hide_an_in_tree_module_glob(tmp_path, order):
    """One pattern name bound to a box path that climbs after a wildcard and to an anchored
    ``docs`` pattern: the box path is never stat'ed, but the ``docs`` alternative still reads
    ``docs``, so a new member there selects the reader whichever binding comes first."""
    bindings = ["'/usr/*/../x/*.md'", "str(ROOT / 'docs' / '*.md')"]
    if order == "inside-first":
        bindings.reverse()
    repo, base = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "tests/test_lister.py": ("import glob\n" + _ROOT_ANCHOR + f"\nPATTERN = {bindings[0]}\n"
                                 f"PATTERN = {bindings[1]}\n\n"
                                 "def test_lists():\n    assert glob.glob(PATTERN) is not None\n"),
        "docs/seed.md": "x\n"})
    (repo / "docs" / "added.md").write_text("new\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "a new member of docs")

    result = _selector(repo, f"{base}...HEAD")

    assert "tests/test_lister.py" in result["tests"], result


def test_this_repository_does_not_gain_an_unnamed_directory_read():
    listed = impacted.select(ROOT, ["README.md"])["unnamed_directory_reads"]
    sites = sum(len(lines) for lines in listed.values())
    assert not _exceeds_unnamed_directory_read_ceiling(len(listed), sites), (
        "a directory read whose base cannot be named was added; name its base, "
        "or raise the ceiling here with the reason", len(listed), sites, sorted(listed))
