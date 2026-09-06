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

from tessera._dev.source_dependencies import file_imports


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

    def bounded_readlink(path, *args, **kwargs):
        if Path(path).name in {"a", "b"}:
            followed.append(path)
            assert len(followed) <= 41, "symlink cycle exceeded its traversal budget"
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
    assert found == {target}
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
    expected = {root / "target.json", root / "link"}
    if '".."' not in expression:
        expected.add(root / "nested" / "child" / "chosen.json")
    assert found == expected
    assert not unknown and not unplaced
