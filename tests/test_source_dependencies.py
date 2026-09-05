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
"""

from __future__ import annotations

import ast
import os.path
from pathlib import Path

import pytest

from tessera._dev.source_dependencies import file_imports


def _scan(source, root, *, consumer="consumer.py"):
    tree = ast.parse(source)
    return file_imports(tree, root / consumer, root)


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
