"""tessera#325 -- an absolute or ``..``-escaping ``Path`` literal is an
unknown dependency, and establishing that never touches the filesystem
outside the scanned root.

``tessera._dev.source_dependencies._values`` evaluates literal ``Path(...)``
expressions and, at three call sites (a literal ``.resolve()``, a
``glob``/``rglob`` base, and the final per-value resolve in
``file_imports``), called ``.resolve()`` on them whatever their value --
absolute ones included.  ``.resolve()`` on a path under a hard-mounted NFS
export is an uninterruptible RPC when the server stalls, so a selector run
that scans a source file containing an absolute literal like
``Path("/mnt/shared/...")`` had its wall time bounded by that mount, not by
the tree it was measuring.  The module already refused this for globs
(":314-316": "they must not trigger a filesystem crawl outside this root");
this is the same escape by another spelling, and one rule -- a literal that
cannot be walked entirely inside root is an unknown dependency, decided
before any ``resolve()``/``stat()`` -- now owns all three call sites.

tessera#338 is the other half of that rule: refusing to *resolve* a path is
not the same claim as the file it names being independent of this
repository.  An outside spelling can be a local alias for a tracked file --
environment state the repository never records -- so the refusal keeps a
data dependency of its own, reported apart from the module wildcard that a
plain reader must never become (#148).
"""

from __future__ import annotations

import ast
import os.path
from pathlib import Path

import pytest

from tessera._dev.source_dependencies import _MAX_LINK_DEPTH, file_imports


def _scan(source, root, *, consumer="consumer.py"):
    tree = ast.parse(source)
    found, unknown, _ = file_imports(tree, root / consumer, root)
    return found, unknown


def _scan_full(source, root, *, consumer="consumer.py"):
    """``(found, unknown, unplaced)`` -- the unplaced-read channel included."""
    return file_imports(ast.parse(source), root / consumer, root)


def _guard_resolve_to_root(monkeypatch, root, scratch=None):
    """Fail if anything reaches a location under *scratch* but outside *root*.

    The #325 version of this guarded ``Path.resolve``'s *argument*, and
    normalized it first.  Both concessions were holes (#339).  Normalizing
    accepts ``<scratch>/outside/../repo/driver.py``, whose destination is
    inside the tree and whose *walk* is not: ``resolve()`` stats ``outside``
    before ``..`` collapses.  And guarding the argument sees only the
    spelling, never the steps -- an in-root symlink is followed to its
    outside target with no outside argument passed to anything.

    The syscall is the observable, because the syscall is what blocks in D
    state on a stalled mount, so this guards the syscalls the walk can make
    and normalizes nothing: each call is judged on the string it is handed.
    It is scoped to *scratch* (the fixture's own temporary directory,
    defaulting to root's parent) so that unrelated machinery running in the
    same process -- the interpreter's own imports above all -- is not
    caught by a guard it was never about.
    """
    root_str = os.path.normpath(str(root))
    scratch_str = os.path.normpath(str(root.parent if scratch is None else scratch))

    def offending(argument):
        try:
            raw = os.fspath(argument)
        except TypeError:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "surrogateescape")
        if raw != scratch_str and not raw.startswith(scratch_str + os.sep):
            return None                     # not part of this fixture at all
        if raw == root_str or raw.startswith(root_str + os.sep):
            return None                     # inside the approved tree
        if root_str.startswith(raw + os.sep):
            # An ancestor of root.  ``Path.resolve`` walks down from ``/`` and
            # touches every one of them; that is not the escape, and flagging
            # it would make the in-root controls fail for the wrong reason.
            return None
        return raw

    def guarded(name, original, take=lambda args, kwargs: args[0] if args else None):
        def call(*args, **kwargs):
            reached = offending(take(args, kwargs))
            if reached is not None:
                raise AssertionError(
                    f"{name} reached {reached}, outside root {root_str} "
                    "(tessera#325/#339: no filesystem step may leave the tree)"
                )
            return original(*args, **kwargs)
        return call

    for name in ("lstat", "stat", "readlink", "scandir", "listdir"):
        monkeypatch.setattr(os, name, guarded(f"os.{name}", getattr(os, name)))
    monkeypatch.setattr(
        Path, "resolve",
        guarded("Path.resolve", Path.resolve, lambda args, kwargs: args[0]))


#: The three places a path reaches the filesystem: a bare loader argument, an
#: explicit ``.resolve()``, and a glob base.  One home for the three, because
#: every rule about the boundary has to hold at all of them (#325, #339) and a
#: shape that exists in only one test is the one that stops being covered.
_ENTRY_POINTS = [
    pytest.param(
        "bare-literal (:426)",
        '''\
import importlib.util
from pathlib import Path
TARGET = Path({outside!r})
importlib.util.spec_from_file_location("mod", TARGET)
''',
        id="bare-literal",
    ),
    pytest.param(
        "explicit-resolve (:308)",
        '''\
import importlib.util
from pathlib import Path
TARGET = Path({outside!r}).resolve()
importlib.util.spec_from_file_location("mod", TARGET)
''',
        id="explicit-resolve",
    ),
    pytest.param(
        "glob-base (:318)",
        '''\
import importlib.util
from pathlib import Path
BASE = Path({outside!r})
for candidate in BASE.glob("*.py"):
    importlib.util.spec_from_file_location("mod", candidate)
''',
        id="glob-base",
    ),
]


@pytest.mark.parametrize(("shape", "template"), _ENTRY_POINTS)
def test_absolute_literal_outside_root_is_unknown_and_never_touches_fs(
    tmp_path, monkeypatch, shape, template,
):
    """An absolute ``Path`` literal outside the scanned root must resolve
    to the selector's existing unknown/wildcard outcome, and must never
    reach ``Path.resolve()`` with a path outside root while deciding it --
    that call is what stats an NFS export component by component and
    blocks in D state when the server stalls."""
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    outside = tmp_path / "outside" / "target.py"

    source = template.format(outside=str(outside))
    _guard_resolve_to_root(monkeypatch, root)

    found, unknown = _scan(source, root)

    assert unknown, f"{shape}: an absolute literal outside root must be an unknown dependency"
    assert found == set(), f"{shape}: an absolute literal outside root must not become an edge"


def test_relative_literal_under_root_is_unchanged(tmp_path):
    """A plain relative literal under root keeps resolving to an exact
    edge -- the #325 fix touches only literals that escape root."""
    root = (tmp_path / "repo").resolve()
    root.mkdir()

    source = '''\
import importlib.util
from pathlib import Path
TARGET = Path("target.py")
importlib.util.spec_from_file_location("mod", TARGET)
'''
    found, unknown = _scan(source, root)

    assert found == {root / "target.py"}
    assert not unknown


def test_dotdot_escaping_relative_literal_is_unknown_and_never_touches_fs(
    tmp_path, monkeypatch,
):
    """A relative literal that lexically escapes root via ``..`` is the
    same escape as an absolute literal outside root: unknown, and never
    resolved outside root to find out."""
    root = (tmp_path / "repo").resolve()
    root.mkdir()

    source = '''\
import importlib.util
from pathlib import Path
TARGET = Path("../outside/target.py")
importlib.util.spec_from_file_location("mod", TARGET)
'''
    _guard_resolve_to_root(monkeypatch, root)

    found, unknown = _scan(source, root)

    assert unknown
    assert found == set()


_READ_SHAPES = [
    pytest.param(
        '''\
from pathlib import Path
TARGET = Path({outside!r})
def load():
    return TARGET.read_text()
''',
        id="bare-literal-read",
    ),
    pytest.param(
        '''\
from pathlib import Path
TARGET = Path({outside!r}).resolve()
def load():
    return TARGET.read_bytes()
''',
        id="explicit-resolve-read",
    ),
    pytest.param(
        '''\
from pathlib import Path
BASE = Path({outside!r})
def load():
    text = ""
    for candidate in BASE.glob("*.json"):
        text += candidate.read_text()
    return text
''',
        id="glob-base-read",
    ),
]


@pytest.mark.parametrize("template", _READ_SHAPES)
def test_outside_spelled_read_keeps_its_own_uncertainty(
    tmp_path, monkeypatch, template,
):
    """tessera#338 -- the guard refuses to resolve, not to depend.

    ``found=set(), unknown=False`` was the whole receipt for a plain reader
    whose target the guard refused, so the caller created neither a data edge
    nor an uncertainty edge and the selector answered ``none`` for a change
    that moved the reader's bytes.  The refusal is now its own third value:
    a data read with no placeable target.
    """
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    outside = tmp_path / "alias" / "data" / "runtime-settings.json"
    _guard_resolve_to_root(monkeypatch, root)

    found, unknown, unplaced = _scan_full(
        template.format(outside=str(outside)), root)

    assert found == set(), "an outside spelling is still never an edge"
    assert not unknown, "a plain reader executes nothing and imports nothing (#148)"
    assert unplaced, (
        "an outside-spelled read is a dependency this resolver declined to "
        "place, not an absence of one"
    )


@pytest.mark.parametrize("template", _READ_SHAPES)
def test_outside_spelled_read_that_can_execute_stays_a_module_wildcard(
    tmp_path, monkeypatch, template,
):
    """A reader that runs what it read keeps the stronger claim.

    The split is between kinds of consumer, not kinds of path: a module that
    can ``exec`` the bytes may import anything, so it stays a ``WILDCARD``
    and never degrades to the weaker data-only uncertainty.
    """
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    outside = tmp_path / "alias" / "data" / "runtime-settings.json"
    source = template.format(outside=str(outside)) + "exec(load())\n"
    _guard_resolve_to_root(monkeypatch, root)

    found, unknown, unplaced = _scan_full(source, root)

    assert found == set()
    assert unknown, "a source-executing reader may import anything it read"
    assert not unplaced, "the module wildcard already subsumes the data claim"


def test_outside_spelled_loader_stays_a_module_wildcard(tmp_path, monkeypatch):
    """A recognized loader is unchanged by #338: it always executes."""
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    outside = tmp_path / "alias" / "driver.py"
    source = '''\
import importlib.util
from pathlib import Path
TARGET = Path({outside!r})
importlib.util.spec_from_file_location("mod", TARGET)
'''.format(outside=str(outside))
    _guard_resolve_to_root(monkeypatch, root)

    found, unknown, unplaced = _scan_full(source, root)

    assert found == set()
    assert unknown
    assert not unplaced


def test_a_read_that_names_no_file_states_no_dependency(tmp_path):
    """#148 is untouched: never-named is not named-and-refused.

    A filename assembled from runtime state names nothing the diff can hold,
    so it must produce no edge, no wildcard and no unplaced read.  Reading it
    as uncertainty is what made every verdict ``full``.
    """
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    source = '''\
import os
from pathlib import Path
def load(name):
    return (Path(os.environ["DIR"]) / name).read_text()
'''
    found, unknown, unplaced = _scan_full(source, root)

    assert found == set()
    assert not unknown
    assert not unplaced


def test_in_root_read_is_still_an_exact_edge(tmp_path):
    """The control: nothing about ordinary narrowing moved."""
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    source = '''\
from pathlib import Path
TARGET = Path("data/runtime-settings.json")
def load():
    return TARGET.read_text()
'''
    found, unknown, unplaced = _scan_full(source, root)

    assert found == {root / "data/runtime-settings.json"}
    assert not unknown
    assert not unplaced


@pytest.mark.parametrize(("shape", "template"), _ENTRY_POINTS)
def test_reentering_spelling_never_walks_outside_root(
    tmp_path, monkeypatch, shape, template,
):
    """tessera#339 -- the destination is inside; the walk to it was not.

    ``<scratch>/outside/../repo/target.py`` normalizes to an in-root path, so
    the lexical guard admitted it -- and then ``resolve()`` was handed the
    original spelling, which it walks as written: ``lstat`` on ``outside``
    before ``..`` collapses.  That is precisely the syscall #325 was about,
    reachable again with no symlink involved.  Bounding the destination is not
    bounding the resolution, so the spelling is refused whole, before any
    filesystem call at all.
    """
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    reentering = tmp_path / "outside" / ".." / "repo" / "target.py"
    assert ".." in reentering.parts, "the fixture must keep the escaping segment"

    source = template.format(outside=str(reentering))
    _guard_resolve_to_root(monkeypatch, root)

    found, unknown = _scan(source, root)

    assert unknown, f"{shape}: a spelling that leaves the tree is an unknown dependency"
    assert found == set(), f"{shape}: it is not an edge either"


@pytest.mark.parametrize(("shape", "template"), _ENTRY_POINTS)
def test_in_root_symlink_to_outside_is_never_followed(
    tmp_path, monkeypatch, shape, template,
):
    """tessera#339 -- the other escape: every component of the spelling is
    in-root, and the *link* is what leaves.

    ``<root>/bridge`` passes any string check there is, and ``resolve()``
    follows it to its outside target before the ``is_relative_to(root)`` check
    ever runs.  The link is read -- it is in the tree, and reading it is how
    we learn where it points -- but its target is never approached.  The
    target here is deliberately never created: nothing about this case
    requires the outside location to exist, only to be named.
    """
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    denied = tmp_path / "never-accessed-target"
    (root / "bridge").symlink_to(denied, target_is_directory=True)
    assert not denied.exists(), "the fixture never creates the outside target"

    source = template.format(outside=str(root / "bridge" / "target.py"))
    _guard_resolve_to_root(monkeypatch, root)

    found, unknown = _scan(source, root)

    assert unknown, f"{shape}: a link out of the tree is an unknown dependency"
    assert found == set(), f"{shape}: and never an edge to a file outside it"


def test_reentering_spelling_in_a_data_read_keeps_its_unplaced_dependency(
    tmp_path, monkeypatch,
):
    """The #339 guard must not undo #338: refusing is still not independence.

    A plain reader whose spelling leaves the tree keeps the same unplaced-read
    uncertainty an outside literal gets, rather than falling back to the
    silence that #338 removed.
    """
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    reentering = tmp_path / "outside" / ".." / "repo" / "data.json"
    source = '''\
from pathlib import Path
TARGET = Path({outside!r})
def load():
    return TARGET.read_text()
'''.format(outside=str(reentering))
    _guard_resolve_to_root(monkeypatch, root)

    found, unknown, unplaced = _scan_full(source, root)

    assert found == set()
    assert not unknown, "a plain reader still executes nothing (#148)"
    assert unplaced, "a refused spelling is still a dependency this cannot place"


def test_in_root_dotdot_still_resolves_to_an_exact_edge(tmp_path, monkeypatch):
    """The control for over-refusal: ``..`` inside the tree is ordinary.

    ``<root>/pkg/../target.py`` never leaves root at any step, so it must keep
    the exact edge it always had.  The walk applies ``..`` to a prefix it has
    already established is not a symlink, which is why it can mean what the
    filesystem means by it.
    """
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / "pkg").mkdir()
    (root / "target.py").write_text("VALUE = 1\n", encoding="utf-8")

    source = '''\
import importlib.util
from pathlib import Path
TARGET = Path("pkg") / ".." / "target.py"
importlib.util.spec_from_file_location("mod", TARGET)
'''
    _guard_resolve_to_root(monkeypatch, root)

    found, unknown = _scan(source, root)

    assert found == {root / "target.py"}
    assert not unknown


def test_in_root_symlink_to_an_in_root_file_still_resolves(tmp_path, monkeypatch):
    """The other control: a link the tree owns is followed, as it always was.

    Refusing to leave root is not refusing to resolve.  ``<root>/link.py``
    pointing at ``<root>/target.py`` produces edges to the link and target,
    which is what makes the exact-edge narrowing worth having.
    """
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "link.py").symlink_to("target.py")

    source = '''\
import importlib.util
from pathlib import Path
TARGET = Path("link.py")
importlib.util.spec_from_file_location("mod", TARGET)
'''
    _guard_resolve_to_root(monkeypatch, root)

    found, unknown = _scan(source, root)

    assert found == {root / "target.py", root / "link.py"}
    assert not unknown


def test_a_symlink_loop_inside_root_terminates_as_unknown(tmp_path, monkeypatch):
    """A cycle the tree owns is bounded, not walked forever.

    ``resolve()`` answers ``ELOOP``; the walk answers with the same refusal it
    gives any other step it cannot complete, so a pathological tree cannot
    hang the selector either.
    """
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / "a").symlink_to("b")
    (root / "b").symlink_to("a")

    source = '''\
import importlib.util
from pathlib import Path
TARGET = Path("a") / "target.py"
importlib.util.spec_from_file_location("mod", TARGET)
'''
    _guard_resolve_to_root(monkeypatch, root)

    found, unknown = _scan(source, root)

    assert found == set()
    assert unknown


@pytest.mark.parametrize(("shape", "template"), _ENTRY_POINTS)
@pytest.mark.parametrize("links", ["absolute", "relative", "mixed"])
def test_link_cycles_share_a_bounded_budget(tmp_path, monkeypatch, shape, template, links):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a").symlink_to(root / "b" if links != "relative" else "b")
    (root / "b").symlink_to(root / "a" if links == "absolute" else "a")
    _guard_resolve_to_root(monkeypatch, root)
    original = os.readlink
    followed = []
    # Each resolution of the cycle gets one fresh budget.  A glob base is
    # resolved twice: once as the directory node and once for its exact edges.
    limit = (_MAX_LINK_DEPTH + 1) * (2 if shape.startswith("glob-base") else 1)

    def bounded_readlink(path, *args, **kwargs):
        if Path(path).name in {"a", "b"}:
            followed.append(path)
            assert len(followed) <= limit, "symlink cycle exceeded its traversal budget"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", bounded_readlink)
    found, unknown = _scan(template.format(outside=str(root / "a")), root)
    assert found == set()
    assert unknown


@pytest.mark.parametrize("absolute", [False, True])
def test_link_parent_semantics_retain_exact_edge(tmp_path, monkeypatch, absolute):
    root = tmp_path / "repo"
    (root / "nested" / "child").mkdir(parents=True)
    (root / "nested" / "target.py").write_text("VALUE = 1\n")
    (root / "link").symlink_to(root / "nested" / "child" if absolute else "nested/child")
    _guard_resolve_to_root(monkeypatch, root)
    found, unknown = _scan('from pathlib import Path\nimport runpy\n'
                          'runpy.run_path(Path("link") / ".." / "target.py")', root)
    assert found == {root / "nested" / "target.py", root / "link"}
    assert not unknown


@pytest.mark.parametrize("reader", [False, True], ids=["loader", "data-read"])
@pytest.mark.parametrize("pattern", ["*/", "*"])
def test_glob_checks_links_before_directory_filtering(tmp_path, monkeypatch, reader, pattern):
    root = tmp_path / "repo"
    root.mkdir()
    # Never create or approach the target: guard the C-level following call.
    (root / "bridge").symlink_to(tmp_path / "unvisited", target_is_directory=True)
    _guard_resolve_to_root(monkeypatch, root)
    original = os.scandir

    class Entry:
        def __init__(self, entry):
            self.entry = entry

        def __getattr__(self, name):
            return getattr(self.entry, name)

        def is_dir(self, *, follow_symlinks=True):
            assert not (self.name == "bridge" and follow_symlinks), (
                "directory glob followed a link before boundary placement")
            return self.entry.is_dir(follow_symlinks=follow_symlinks)

    class Entries:
        def __init__(self, path):
            self.entries = original(path)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.entries.close()

        def __iter__(self):
            return (Entry(entry) for entry in self.entries)

    monkeypatch.setattr(os, "scandir", Entries)
    action = '(p / "data.txt").read_text()' if reader else 'runpy.run_path(p / "driver.py")'
    found, unknown, unplaced = _scan_full(
        f'from pathlib import Path\nimport runpy\nfor p in Path(".").glob({pattern!r}):\n    {action}\n', root)
    # The link leaves the tree, so the call is kept as an unplaced read (#338; flags
    # asserted below), which supersedes the base node it used to return alone: the
    # last pattern component is now scanned for links and the guard declines this one
    # without approaching it (#1011 review).
    assert found == set()
    assert unknown is (not reader)
    assert unplaced is reader


def test_plain_glob_retains_exact_edges(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "driver.py"
    target.write_text("VALUE = 1\n")
    _guard_resolve_to_root(monkeypatch, root)
    found, unknown, unplaced = _scan_full(
        'from pathlib import Path\nimport runpy\nfor p in Path(".").glob("*.py"):\n    runpy.run_path(p)\n', root)
    assert found == {target, root}
    assert not unknown
    assert not unplaced


@pytest.mark.parametrize("expression", [
    'Path("link") / "chosen.json"',
    '(Path("link") / "chosen.json").resolve()',
    'next_path',
    'Path("link") / ".." / ".." / "target.json"',
])
def test_read_dependencies_keep_each_traversed_link(tmp_path, monkeypatch, expression):
    root = tmp_path / "repo"
    (root / "nested" / "child").mkdir(parents=True)
    (root / "link").symlink_to("nested/child")
    (root / "nested" / "child" / "chosen.json").symlink_to("../../target.json")
    (root / "target.json").write_text("{}")
    _guard_resolve_to_root(monkeypatch, root)
    source = ('from pathlib import Path\n'
              'for next_path in Path("link").glob("*.json"):\n'
              f'    value = ({expression}).read_text()\n')
    found, unknown, unplaced = _scan_full(source, root)
    # ``chosen.json`` is a link entry the glob's own terminal component matches, so it is
    # a dependency whatever the read spells (#1011 review); before, only a read through
    # it was.
    expected = {root / "target.json", root / "link", root / "nested" / "child",
                root / "nested" / "child" / "chosen.json"}
    assert found == expected
    assert not unknown and not unplaced


@pytest.mark.parametrize("method", ["glob", "rglob"])
@pytest.mark.parametrize("form", ["unbound", "bound", "direct"])
def test_an_aliased_glob_method_keeps_its_named_base(tmp_path, method, form):
    # ``original = Path.glob; original(path, pattern)``,
    # ``scan = DOCS.glob; scan(pattern)`` and the direct ``Path.glob(path,
    # pattern)`` still name their directory (PB1496).
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    if form == "unbound":
        source = ('from pathlib import Path\n'
                  f'original = Path.{method}\n'
                  'original(Path("docs"), "*.json")\n')
    elif form == "direct":
        source = ('import pathlib\n'
                  f'pathlib.Path.{method}(pathlib.Path("docs"), "*.json")\n')
    else:
        source = ('from pathlib import Path\n'
                  f'scan = Path("docs").{method}\n'
                  'scan("*.json")\n')
    found, unknown, unplaced = _scan_full(source, root)
    assert found == {root / "docs"}
    assert not unknown and not unplaced


@pytest.mark.parametrize("method", ["glob", "rglob"])
def test_an_aliased_glob_method_with_no_nameable_receiver_names_no_base(tmp_path, method):
    root = tmp_path / "repo"
    root.mkdir()
    found, unknown, unplaced = _scan_full(
        'from pathlib import Path\n'
        f'original = Path.{method}\n'
        'original(somewhere, "*.json")\n', root)
    assert found == set()
    assert not unknown and not unplaced


@pytest.mark.parametrize("pattern", ["../data/*.json", "/abs/*.json"])
def test_a_glob_pattern_that_can_leave_its_receiver_is_refused(tmp_path, pattern):
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    found, unknown, unplaced = _scan_full(
        'from pathlib import Path\n'
        f'Path("docs").glob({pattern!r})\n', root)
    assert found == set()
    assert unplaced or unknown


def test_a_glob_prefix_that_is_a_link_keeps_the_link_and_its_target(tmp_path):
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "docs" / "link").symlink_to("../data", target_is_directory=True)
    found, unknown, unplaced = _scan_full(
        'from pathlib import Path\n'
        'Path("docs").glob("link/*.json")\n', root)
    assert {root / "docs", root / "data", root / "docs" / "link"} <= found
    assert not unknown and not unplaced


def _unnamed_directory_reads(source, root):
    unnamed = []
    file_imports(ast.parse(source), root / "consumer.py", root, unnamed=unnamed)
    return unnamed


@pytest.mark.parametrize("source", [
    "def f(x):\n    def walk(a):\n        return a\n    return walk(x)\n",
    "def walk(a):\n    return a\n\n\ndef f(x):\n    return walk(x)\n",
    "def glob(a):\n    return a\n\n\ndef f(x):\n    return glob(x)\n",
], ids=["nested-def", "module-def", "glob"])
def test_a_function_the_file_defines_is_not_a_directory_read(tmp_path, source):
    # A bare ``walk(...)`` is os.walk only if something names it so.  A name the
    # file defines itself and never imports is that function, not an enumeration
    # (PB1496: the codebook's recursive ``walk`` was listed as an unnamed read).
    assert _unnamed_directory_reads(source, tmp_path) == []


@pytest.mark.parametrize("source", [
    "from os import walk\n\n\ndef f(x):\n    return list(walk(x))\n",
    "import os\n\n\ndef f(x):\n    return list(os.walk(x))\n",
    "from os import walk as step\n\n\ndef f(x):\n    return list(step(x))\n",
    "import os\nwalk = os.walk\n\n\ndef f(x):\n    return list(walk(x))\n",
    # A star import may bring os.walk in; a same-named def elsewhere cannot rule it out.
    "from os import *\n\n\ndef g():\n    def walk(a):\n        return a\n\n\ndef f(x):\n    return list(walk(x))\n",
    # An import of the real name keeps the alias even if the file also defines one.
    "from os import walk\n\n\ndef g():\n    def walk(a):\n        return a\n\n\ndef f(x):\n    return list(walk(x))\n",
], ids=["from-os", "os-attribute", "renamed-import", "assigned-alias", "star-import", "import-and-def"])
def test_a_real_directory_walk_is_still_listed(tmp_path, source):
    assert _unnamed_directory_reads(source, tmp_path), source


def _unnamed_and_unknown(source, root):
    unnamed = []
    _, unknown, _ = file_imports(ast.parse(source), root / "consumer.py", root, unnamed=unnamed)
    return unnamed, unknown


# The name that is called must resolve, lexically, to a plain def or class: an
# unrelated definition elsewhere in the file proves nothing about this call.
_SHADOWED_WALK = {
    "method-elsewhere-and-parameter-default": (
        "import os\n\n\nclass T:\n    def walk(self):\n        return 1\n\n\n"
        "def f(x, walk=os.walk):\n    return list(walk(x))\n"),
    "decorator-returns-os-walk": (
        "import os\n\n\ndef replace(fn):\n    return os.walk\n\n\n@replace\n"
        "def walk(a):\n    return a\n\n\ndef f(x):\n    return list(walk(x))\n"),
    "parameter-shadows-module-def": (
        "def walk(a):\n    return a\n\n\ndef f(x, walk):\n    return walk(x)\n"),
    "def-in-one-branch-assignment-in-the-other": (
        "import os\n\nif os.environ:\n    def walk(a):\n        return a\nelse:\n"
        "    walk = os.walk\n\n\ndef f(x):\n    return list(walk(x))\n"),
    "import-with-def-fallback": (
        "try:\n    from os import walk\nexcept ImportError:\n    def walk(a):\n        return a\n\n\n"
        "def f(x):\n    return list(walk(x))\n"),
    "global-rebinding": (
        "import os\n\n\ndef walk(a):\n    return a\n\n\ndef g():\n    global walk\n"
        "    walk = os.walk\n\n\ndef f(x):\n    return list(walk(x))\n"),
    "loop-variable": (
        "import os\n\n\ndef walk(a):\n    return a\n\n\ndef f(x):\n"
        "    for walk in (os.walk,):\n        return list(walk(x))\n"),
    "star-import-after-def": (
        "def walk(a):\n    return a\n\n\nfrom os import *\n\n\ndef f(x):\n    return list(walk(x))\n"),
    "conditional-class-binding": (
        "import os\n\nif os.environ:\n    class walk:\n        pass\nelse:\n    walk = os.walk\n\n\n"
        "def f(x):\n    return list(walk(x))\n"),
    "metaclass-binds-os-walk": (
        "import os\n\n\nclass Meta(type):\n    def __new__(mcs, name, bases, namespace):\n"
        "        return os.walk\n\n\nclass walk(metaclass=Meta):\n    pass\n\n\n"
        "def f(x):\n    return list(walk(x))\n"),
    "class-body-comprehension": (
        "import os\n\n\ndef walk(a):\n    return a\n\n\nclass A:\n    walk = os.walk\n"
        "    results = [walk(x) for x in range(3)]\n"),
    "def-in-another-function": (
        "import os\n\n\ndef g():\n    def walk(a):\n        return a\n    return walk\n\n\n"
        "def f(x, walk=os.walk):\n    return list(walk(x))\n"),
}


@pytest.mark.parametrize("source", list(_SHADOWED_WALK.values()), ids=list(_SHADOWED_WALK))
def test_a_defined_name_does_not_hide_a_call_that_resolves_elsewhere(tmp_path, source):
    unnamed, _ = _unnamed_and_unknown(source, tmp_path)
    assert unnamed, source


@pytest.mark.parametrize("source", list(_SHADOWED_WALK.values()), ids=list(_SHADOWED_WALK))
def test_a_module_that_executes_source_keeps_its_unknown_loader_flag(tmp_path, source):
    # The misread would also have dropped the unknown-loader flag of a module
    # that can run what it reads, which is the escalation a real walk gets.
    unnamed, unknown = _unnamed_and_unknown(source + '\n\nexec("pass")\n', tmp_path)
    assert unknown, source


def test_a_local_wrapper_around_walk_keeps_the_directory_it_names(tmp_path):
    # ``walk(Path("docs"))`` reads docs through the wrapper: the call site names the
    # directory, so the resolved dependency must survive the exemption.  Only the
    # warning that a base is unnamed may be suppressed, never the call (PB1496).
    (tmp_path / "docs").mkdir()
    source = ("import os\nfrom pathlib import Path\n\n\ndef walk(root):\n    return os.walk(root)\n\n\n"
              "def f():\n    return list(walk(Path('docs')))\n")
    found, unknown, unplaced = file_imports(ast.parse(source), tmp_path / "consumer.py", tmp_path)
    assert tmp_path / "docs" in found, (found, unknown, unplaced)


def test_a_recursive_local_def_is_still_not_a_directory_read(tmp_path):
    source = ("def f(items):\n    def walk(level):\n        if level == 0:\n"
              "            return [level]\n        return walk(level - 1) + walk(level - 1)\n"
              "    return walk(items)\n")
    assert _unnamed_directory_reads(source, tmp_path) == []


@pytest.mark.parametrize("source", [
    "import os\nx = os.listdir('docs')\n",
    "import os\nx = list(os.walk('docs'))\n",
    "import os\nx = list(os.scandir('docs'))\n",
    "import os\nDOCS = 'docs'\nx = os.listdir(DOCS)\n",
    "from os import listdir\nx = listdir('docs')\n",
    "import os\nx = os.listdir('./docs')\n",
], ids=["listdir", "walk", "scandir", "constant", "from-import", "dot-slash"])
def test_a_string_path_names_the_directory_an_enumeration_reads(tmp_path, source):
    # ``os.listdir("docs")`` reads the same directory as ``os.listdir(Path("docs"))``;
    # only the Path spelling used to be resolved, so a file added under the
    # directory selected no reader (PB1496).
    (tmp_path / "docs").mkdir()
    found, unknown, unplaced = _scan_full(source, tmp_path)
    assert found == {tmp_path / "docs"}, (found, unknown, unplaced)
    assert not unknown and not unplaced


@pytest.mark.parametrize("source", [
    # A custom ``walk`` is recognized by its name alone; its string argument may
    # not be a path at all, so resolving it must ADD an edge and never replace the
    # unknown-loader flag a module that can execute source already had.
    "def walk(mode):\n    return mode\n\n\ndef f():\n    exec('pass')\n    return walk('mode')\n",
    "import os\n\n\ndef f():\n    exec('pass')\n    return os.listdir('mode')\n",
    "from os import walk\n\n\ndef f():\n    exec('pass')\n    return list(walk('mode'))\n",
], ids=["custom-walk", "os-listdir", "from-import-walk"])
def test_a_string_base_adds_an_edge_without_dropping_the_unknown_loader_flag(tmp_path, source):
    (tmp_path / "mode").mkdir()
    found, unknown, unplaced = _scan_full(source, tmp_path)
    assert tmp_path / "mode" in found, (found, unknown, unplaced)
    assert unknown, (found, unknown, unplaced)


@pytest.mark.parametrize("source", [
    "import os\nx = os.listdir('/etc')\n",
    "import os\nx = os.listdir('../outside')\n",
], ids=["absolute", "escaping"])
def test_a_string_path_outside_the_tree_is_refused_not_resolved(tmp_path, source):
    # The boundary guard is the same one the Path spelling meets: refused, kept
    # as an unplaced read, and never stat'ed outside the tree.
    (tmp_path / "docs").mkdir()
    found, unknown, unplaced = _scan_full(source, tmp_path)
    assert found == set()
    assert unplaced and not unknown


@pytest.mark.parametrize("source", [
    "import glob\nx = glob.glob('docs/*.md')\n",
    "import glob as g\nx = g.glob('docs/*.md')\n",
    "from glob import glob\nx = glob('docs/*.md')\n",
    "import glob\nx = list(glob.iglob('docs/*.md'))\n",
    "from glob import iglob\nx = list(iglob('docs/*.md'))\n",
    "import glob\nx = glob.glob('docs/**/*.md', recursive=True)\n",
    "import glob\nPATTERN = 'docs/*.md'\nx = glob.glob(PATTERN)\n",
    "import glob\nx = glob.glob('./docs/*.md')\n",
], ids=["module", "aliased-module", "from-import", "iglob", "from-import-iglob", "recursive",
        "constant", "dot-slash"])
def test_a_module_glob_names_the_directory_in_front_of_its_wildcard(tmp_path, source):
    # ``glob.glob("docs/*.md")`` carries its base in the pattern string, not in a
    # Path receiver; only the Path spelling used to resolve, and ``iglob`` was not
    # recognised at all, so a file added under ``docs`` selected no reader (#1010).
    (tmp_path / "docs").mkdir()
    found, unknown, unplaced = _scan_full(source, tmp_path)
    assert found == {tmp_path / "docs"}, (found, unknown, unplaced)
    assert not unknown and not unplaced


def test_a_module_glob_places_the_whole_literal_prefix(tmp_path):
    (tmp_path / "docs" / "sub").mkdir(parents=True)
    found, _, _ = _scan_full("import glob\nx = glob.glob('docs/sub/*.md')\n", tmp_path)
    assert found == {tmp_path / "docs" / "sub"}


@pytest.mark.parametrize("source", [
    "import glob\nx = glob.glob('/etc/*.conf')\n",
    "import glob\nx = glob.glob('../outside/*.md')\n",
    "import glob\nx = glob.glob('docs/*/../../outside/*.md')\n",
], ids=["absolute", "escaping", "parent-after-wildcard"])
def test_a_module_glob_that_leaves_the_tree_stays_unnamed(tmp_path, source):
    # The same boundary guard as the Path spelling, never stat'ed outside the tree.
    # Unlike a Path read it is NOT kept as an unplaced read: this tree's module globs
    # of that kind name box locations (/usr/local/cuda-*, /mnt/shared/...), and an
    # unplaced read seeds its reader's consumers on every change (#148).  It stays
    # listed as an unnamed read, exactly as before.
    (tmp_path / "docs").mkdir()
    unnamed = []
    found, unknown, unplaced = file_imports(
        ast.parse(source), tmp_path / "consumer.py", tmp_path, unnamed=unnamed)
    assert found == set() and unnamed, (found, unnamed)
    assert not unplaced and not unknown, (unknown, unplaced)


def test_an_unnamed_module_glob_still_escalates_in_a_module_that_executes_source(tmp_path):
    found, unknown, unplaced = _scan_full(
        "import glob\nexec('pass')\nx = glob.glob('/usr/local/cuda-*')\n", tmp_path)
    assert found == set() and unknown and not unplaced


@pytest.mark.parametrize("source", [
    "import glob\n\n\ndef f(pattern):\n    return glob.glob(pattern)\n",
    "import glob\nx = glob.glob('docs/*.md', root_dir='elsewhere')\n",
], ids=["unnameable-pattern", "root-dir"])
def test_a_module_glob_with_no_nameable_base_stays_unnamed(tmp_path, source):
    (tmp_path / "docs").mkdir()
    unnamed = []
    found, unknown, unplaced = file_imports(
        ast.parse(source), tmp_path / "consumer.py", tmp_path, unnamed=unnamed)
    assert found == set() and unnamed, (found, unnamed)


@pytest.mark.parametrize("source", [
    "import glob, os\nos.chdir('nested')\nx = glob.glob('docs/*.md')\n",
    "import glob\n\n\ndef test_x(monkeypatch):\n    monkeypatch.chdir('nested')\n    return glob.glob('docs/*.md')\n",
    "import glob, os\nfrom pathlib import Path\nos.chdir(Path(__file__).resolve().parents[1] / 'nested')\nx = list(glob.iglob('docs/*.md'))\n",
    "import os\nos.chdir('nested')\nx = os.listdir('docs')\n",
    "import os\nfrom contextlib import chdir\nwith chdir('nested'):\n    x = os.walk('docs')\n",
], ids=["os-chdir", "monkeypatch", "path-argument-iglob", "listdir-string", "contextlib-chdir"])
def test_a_relative_base_is_not_assumed_root_relative_in_a_module_that_changes_directory(
        tmp_path, source):
    # The runtime directory is not the tree's root once the module has called chdir, so a
    # relative pattern or string names a directory nothing here can place.  Naming the
    # root's ``docs`` would miss ``nested/docs``; keep the read as an unplaced one.
    (tmp_path / "docs").mkdir()
    (tmp_path / "nested" / "docs").mkdir(parents=True)
    found, unknown, unplaced = _scan_full(source, tmp_path)
    assert tmp_path / "docs" not in found
    assert unplaced and not unknown, (found, unknown, unplaced)


def test_an_absolute_pattern_after_a_chdir_is_unaffected(tmp_path):
    found, unknown, unplaced = _scan_full(
        "import glob, os\nos.chdir('nested')\nx = glob.glob('/usr/local/cuda-*')\n", tmp_path)
    assert found == set() and not unplaced and not unknown


@pytest.mark.parametrize("source", [
    "import glob\nx = glob.glob('docs/*.md')\n",
    "import glob\nx = list(glob.iglob('docs/*.md'))\n",
    "import os\nx = os.listdir('docs')\n",
    "from os import chdir as cd\nimport glob\ncd('nested')\nx = glob.glob('docs/*.md')\n",
    "import os, glob\nmove = os.chdir\nmove('nested')\nx = glob.glob('docs/*.md')\n",
    "from contextlib import chdir as enter\nimport glob\nwith enter('nested'):\n    x = glob.glob('docs/*.md')\n",
    "import glob\nfrom support.cwd import enter\nenter()\nx = glob.glob('docs/*.md')\n",
], ids=["relative-module", "relative-iglob", "relative-string", "from-import-alias",
        "assigned-alias", "aliased-contextlib", "imported-helper"])
def test_a_relative_base_is_unplaced_whatever_changes_the_working_directory(tmp_path, source):
    # Nothing here proves the process directory is the tree's root: a helper, a fixture, an
    # alias or pytest itself can change it.  A relative pattern or string base is therefore
    # kept as an unplaced read, never resolved against the root (#1010 review).
    (tmp_path / "docs").mkdir()
    found, unknown, unplaced = _scan_full(source, tmp_path)
    assert tmp_path / "docs" not in found
    assert unplaced and not unknown, (found, unknown, unplaced)


@pytest.mark.parametrize("source", [
    "import glob\nfrom pathlib import Path\nHERE = Path(__file__).resolve().parent\nx = glob.glob(str(HERE / 'docs' / '*.md'))\n",
    "import glob\nfrom pathlib import Path\nHERE = Path(__file__).resolve().parent\nx = list(glob.iglob(str(HERE / 'docs/*.md')))\n",
    "import os\nfrom pathlib import Path\nHERE = Path(__file__).resolve().parent\nx = os.listdir(str(HERE / 'docs'))\n",
], ids=["module", "iglob", "string"])
def test_an_anchored_base_resolves_whatever_the_working_directory(tmp_path, source):
    # Built from ``__file__``, the base does not depend on the process directory.
    (tmp_path / "docs").mkdir()
    found, unknown, unplaced = _scan_full(source, tmp_path)
    assert found == {tmp_path / "docs"} and not unknown and not unplaced, (found, unknown, unplaced)


def test_a_module_glob_keeps_the_unknown_loader_flag_of_an_executing_module(tmp_path):
    # A pattern need not name a directory a custom ``glob`` reads; naming it adds
    # the edge and must never replace the flag (the #1000 lesson).
    (tmp_path / "docs").mkdir()
    found, unknown, _ = _scan_full(
        "import glob\nexec('pass')\nx = glob.glob('docs/*.md')\n", tmp_path)
    assert tmp_path / "docs" in found and unknown


def _linked_tree(root):
    """docs/link -> ../data, with a file in data, and docs/plain as an ordinary directory."""
    (root / "docs" / "plain").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "data" / "member.md").write_text("x\n")
    (root / "docs" / "link").symlink_to("../data", target_is_directory=True)


@pytest.mark.parametrize("source", [
    "from pathlib import Path\nx = list(Path('docs').glob('*/x.md'))\n",
    "from pathlib import Path\nx = list(Path('docs').rglob('*.md'))\n",
    "from pathlib import Path\nx = list(Path('docs').glob('**/x.md'))\n",
    "import glob\nx = glob.glob('docs/*/x.md')\n",
    "import glob\nx = glob.glob('docs/**/x.md', recursive=True)\n",
    "from pathlib import Path\nx = list(Path('docs').glob('*/plain/../x.md'))\n",
], ids=["glob-star", "rglob", "glob-doublestar", "module-star", "module-doublestar", "parent-after-wildcard"])
def test_a_link_reached_through_a_wildcard_component_is_followed(tmp_path, monkeypatch, source):
    # ``docs/link`` points at ``data``; a pattern that wildcards over ``docs`` reads
    # ``data`` through it, so ``data`` and the link are dependencies (#1011).  The
    # parent-after-wildcard spelling is refused or followed, never silently dropped.
    _linked_tree(tmp_path)
    _guard_resolve_to_root(monkeypatch, tmp_path)
    found, unknown, unplaced = _scan_full(source, tmp_path)
    followed = {tmp_path / "data", tmp_path / "docs" / "link"} <= found
    assert followed or unplaced, (found, unknown, unplaced)
    assert tmp_path / "docs" in found or unplaced


@pytest.mark.parametrize("source", [
    "from pathlib import Path\nx = list(Path('docs').glob('*/link/*.md'))\n",
    "import glob\nx = glob.glob('docs/*/link/*.md')\n",
], ids=["path", "module"])
def test_a_link_after_a_wildcard_and_a_literal_component_retains_its_target(tmp_path, monkeypatch, source):
    # ``docs/plain/link`` -> ``../../data``: the wildcard reaches ``plain`` (an ordinary
    # directory) and the literal ``link`` is traversed, so ``data`` is read and must be a
    # dependency, not just traversal state (#1011 review).
    (tmp_path / "docs" / "plain").mkdir(parents=True)
    (tmp_path / "data").mkdir()
    (tmp_path / "docs" / "plain" / "link").symlink_to("../../data", target_is_directory=True)
    _guard_resolve_to_root(monkeypatch, tmp_path)
    found, unknown, unplaced = _scan_full(source, tmp_path)
    assert tmp_path / "data" in found, (found, unknown, unplaced)


def test_a_dangling_link_behind_a_wildcard_still_retains_its_in_tree_target(tmp_path, monkeypatch):
    # The target directory was deleted: the link is unchanged and now dangling, but the
    # reader still depends on that path, so deleting the last member must select it.
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "link").symlink_to("../data", target_is_directory=True)
    _guard_resolve_to_root(monkeypatch, tmp_path)
    found, unknown, unplaced = _scan_full(
        "from pathlib import Path\nx = list(Path('docs').glob('*/*.md'))\n", tmp_path)
    assert tmp_path / "data" in found, (found, unknown, unplaced)


@pytest.mark.parametrize("source", [
    "from pathlib import Path\nx = list(Path('docs').glob('*/'))\n",
    "import glob\nx = glob.glob('docs/*/')\n",
], ids=["path", "module"])
def test_a_trailing_separator_pattern_still_scans_its_last_component(tmp_path, monkeypatch, source):
    # PurePath drops the trailing separator and the last component was never scanned, so
    # ``docs/*/`` found nothing behind ``docs/link`` (#1011 review).  The link is dangling.
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "link").symlink_to("../data", target_is_directory=True)
    _guard_resolve_to_root(monkeypatch, tmp_path)
    found, unknown, unplaced = _scan_full(source, tmp_path)
    assert tmp_path / "data" in found, (found, unknown, unplaced)


@pytest.mark.parametrize("source", [
    "from pathlib import Path\nx = [p.read_text() for p in Path('docs').glob('*/manifest.md')]\n",
    "from pathlib import Path\nx = [p.read_text() for p in Path('docs').glob('*/*.md')]\n",
    "import glob\nx = glob.glob('docs/*/manifest.md')\n",
], ids=["terminal-literal", "terminal-wildcard", "module-literal"])
def test_a_file_link_matched_by_the_terminal_component_retains_its_target(tmp_path, monkeypatch, source):
    # ``docs/plain/manifest.md`` -> ``../../data/payload.md``.  Only ``docs/plain`` holds a
    # link; ``docs`` itself holds none, so the ``*`` scan cannot find it.  The terminal
    # component matches a link to a FILE; reading it reads ``data/payload.md``.
    (tmp_path / "docs" / "plain").mkdir(parents=True)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "payload.md").write_text("x\n")
    (tmp_path / "docs" / "plain" / "manifest.md").symlink_to("../../data/payload.md")
    _guard_resolve_to_root(monkeypatch, tmp_path)
    found, unknown, unplaced = _scan_full(source, tmp_path)
    assert tmp_path / "data" / "payload.md" in found, (found, unknown, unplaced)


@pytest.mark.parametrize("source", [
    "from pathlib import Path\nx = [p.read_text() for p in Path('docs').glob('manifest.md')]\n",
    "from pathlib import Path\nx = [p.read_text() for p in Path('docs').glob('*.md')]\n",
], ids=["literal", "wildcard"])
def test_a_file_link_in_the_base_itself_retains_its_target(tmp_path, monkeypatch, source):
    (tmp_path / "docs").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "payload.md").write_text("x\n")
    (tmp_path / "docs" / "manifest.md").symlink_to("../data/payload.md")
    _guard_resolve_to_root(monkeypatch, tmp_path)
    found, unknown, unplaced = _scan_full(source, tmp_path)
    assert tmp_path / "data" / "payload.md" in found, (found, unknown, unplaced)


def test_a_wildcard_over_an_ordinary_directory_adds_nothing(tmp_path, monkeypatch):
    (tmp_path / "docs" / "plain").mkdir(parents=True)
    _guard_resolve_to_root(monkeypatch, tmp_path)
    found, unknown, unplaced = _scan_full(
        "from pathlib import Path\nx = list(Path('docs').glob('*/x.md'))\n", tmp_path)
    assert found == {tmp_path / "docs"} and not unknown and not unplaced


@pytest.mark.parametrize("source, link", [
    ("from pathlib import Path\nx = list(Path('docs').glob('*/*.md'))\n", "docs/out"),
    ("from pathlib import Path\nx = list(Path('docs').glob('*/link/*.md'))\n", "docs/plain/link"),
    ("import glob\nx = glob.glob('docs/*/*.md')\n", "docs/out"),
], ids=["wildcard", "literal-after-wildcard", "module"])
def test_a_link_leaving_the_tree_keeps_the_read_unplaced_and_is_never_approached(
        tmp_path, monkeypatch, source, link):
    # An outside bridge may lead straight back into the tree, so the read cannot be
    # attributed to any file.  The guard declines to look; the answer is the #338
    # uncertainty (select the reader's consumers), not a silent success (#1011 review).
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    root = tmp_path / "repo"
    (root / "docs" / "plain").mkdir(parents=True)
    (root / link).symlink_to(outside, target_is_directory=True)
    _guard_resolve_to_root(monkeypatch, root, scratch=tmp_path.parent)
    found, unknown, unplaced = _scan_full(source, root)
    assert outside not in found
    assert unplaced and not unknown, (found, unknown, unplaced)


def test_a_scan_over_its_budget_falls_back_to_an_unplaced_read(tmp_path, monkeypatch):
    # Not provable within the budget: select more, never less.
    import tessera._dev.source_dependencies as dependencies
    (tmp_path / "docs").mkdir()
    for index in range(6):
        (tmp_path / "docs" / f"d{index}").mkdir()
    monkeypatch.setattr(dependencies, "_LINK_SCAN_BUDGET", 2)
    found, unknown, unplaced = _scan_full(
        "from pathlib import Path\nx = list(Path('docs').glob('*/x.md'))\n", tmp_path)
    assert found == set() and unplaced and not unknown


def test_empty_glob_keeps_the_link_that_controls_its_members(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "empty").mkdir(parents=True)
    (root / "link").symlink_to("empty")
    _guard_resolve_to_root(monkeypatch, root)
    found, unknown, unplaced = _scan_full(
        'from pathlib import Path\n'
        'for item in Path("link").glob("*.json"):\n'
        '    item.read_text()\n', root)
    assert found == {root / "link", root / "empty"}
    assert not unknown and not unplaced


def test_an_enumerated_directory_is_an_edge_to_its_base(tmp_path):
    """A directory-wide read consumes the base's membership, so the base is
    the dependency: one node every changed, added or deleted path under it
    reaches (tessera#923)."""
    (tmp_path / "docs").mkdir()
    found, unknown, unplaced = _scan_full(
        'from pathlib import Path\n'
        'DOCS = Path(__file__).resolve().parent / "docs"\n'
        'for path in DOCS.rglob("*.md"):\n'
        '    path.read_text()\n', tmp_path)
    assert found == {tmp_path / "docs"}, (found, unknown, unplaced)
    assert not unknown and not unplaced


def test_iterdir_names_its_base(tmp_path):
    (tmp_path / "docs").mkdir()
    found, unknown, unplaced = _scan_full(
        'from pathlib import Path\n'
        'DOCS = Path(__file__).resolve().parent / "docs"\n'
        'NAMES = sorted(child.name for child in DOCS.iterdir())\n', tmp_path)
    assert found == {tmp_path / "docs"}, (found, unknown, unplaced)
    assert not unknown and not unplaced


def test_an_out_of_tree_enumeration_base_keeps_the_unplaced_read(
        tmp_path, monkeypatch):
    _guard_resolve_to_root(monkeypatch, tmp_path)
    outside = tmp_path.parent / "outside"
    found, unknown, unplaced = _scan_full(
        'from pathlib import Path\n'
        f'OUT = Path({str(outside)!r})\n'
        'for path in OUT.rglob("*.md"):\n'
        '    path.read_text()\n', tmp_path)
    assert found == set()
    assert not unknown and unplaced


def test_a_dynamic_pattern_on_a_named_base_keeps_the_unplaced_read(tmp_path):
    """The base is named, the membership is not: #338 uncertainty, never a
    silent drop of a directory the diff can still reach."""
    (tmp_path / "docs").mkdir()
    found, unknown, unplaced = _scan_full(
        'import os\nfrom pathlib import Path\n'
        'DOCS = Path(__file__).resolve().parent / "docs"\n'
        'for path in DOCS.rglob(os.environ.get("PATTERN", "*.md")):\n'
        '    path.read_text()\n', tmp_path)
    assert found == set()
    assert not unknown and unplaced


def test_a_source_executing_module_with_an_unnameable_base_is_a_wildcard(
        tmp_path):
    found, unknown, unplaced = _scan_full(
        'import os\nfrom pathlib import Path\nimport runpy\n'
        'for path in Path(os.environ["DOCS"]).rglob("*.py"):\n'
        '    runpy.run_path(path)\n', tmp_path)
    assert found == set() and not unplaced
    assert unknown
