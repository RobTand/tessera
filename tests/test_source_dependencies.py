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
does not resolve *lexically* under root is an unknown dependency, decided
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


def _guard_resolve_to_root(monkeypatch, root):
    """Fail immediately if ``Path.resolve()`` is asked about anything
    outside *root* -- the exact call that blocks in D state under a
    stalled NFS mount (tessera#325). Patches only ``Path.resolve``, the
    module's own accessor for this; the module never calls
    ``Path.stat``/``os.stat`` directly, so patching those would just add
    unrelated flakiness from other machinery running in the same process.
    """
    # Normalized once here the same way the fix normalizes in production
    # (os.path.normpath, no filesystem access) so this guard catches an
    # un-resolved ``..``-bearing argument too, not just an already-bare
    # absolute one -- a literal target need not be pre-normalized before
    # it reaches ``resolve()``.
    root_str = os.path.normpath(str(root))
    original_resolve = Path.resolve

    def guarded_resolve(self, *args, **kwargs):
        normalized = os.path.normpath(str(self))
        if normalized != root_str and not normalized.startswith(root_str + os.sep):
            raise AssertionError(
                f"Path.resolve() touched {self!s}, outside root {root_str} "
                "(tessera#325: literals outside root must never be resolved)"
            )
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)


@pytest.mark.parametrize(("shape", "template"), [
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
])
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
