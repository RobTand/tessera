"""Freeze every reference ``src/tessera`` makes to PrismaQuant or PrismaBuild.

Tessera stands alone. PrismaQuant (PQ) may use Tessera, but Tessera may not
know PQ or PrismaBuild (PB) except through a plugin interface Tessera
defines (Rob, 2026-09-24, restated 2026-09-27). This test is the
mechanical half of that rule, step 0 of the decoupling plan in the
2026-09-27 coupling inventory.

It parses every module under ``src/tessera`` with ``ast`` and finds:

- ``import prismaquant...``, ``import prismabuild...`` and their
  ``from ... import`` forms, at any depth, including inside functions;
- ``str`` and ``bytes`` constants, outside docstrings, that match
  ``\\b(prismaquant|prismabuild)\\.[a-z0-9_.]+`` or contain
  ``prismabuild-fleet``. Constants inside f-strings count.

Comments, docstrings, docs, test fixtures and measured provenance such as
the ``image`` values in ``runtime_contract.json`` are out of scope: the
gate reads code only.

Each finding is one line of ``standalone_boundary_allowlist.txt``:
``path<TAB>kind<TAB>value``, one line per occurrence. The value is the
module name for an import and the matched token for a constant, never the
whole constant or a line number, so an unrelated edit to a message does
not churn the list. The test requires the findings to equal the file as a
multiset:

- a new reference, or one more occurrence of an allowlisted one, fails;
- a removed reference fails until its line is deleted in the same change,
  so the list tracks the code.

The allowlist only shrinks. CI refuses any pull request that adds a line
to it (``.github/workflows/ci.yml``). It now holds the end state: the
``prismaquant.tessera.v1`` wire-ID literals alone, which stay until the
wire-version decision (step 8). The ``cached_unit.py`` and
``hessian_capture.py`` entries went with step 1, and the
``_dev/suite_source.py`` entries went when its PrismaBuild record checks moved
behind the source-verifier seam (tessera#599 step 3).
"""
from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ROOT / "src" / "tessera"
ALLOWLIST = Path(__file__).with_name("standalone_boundary_allowlist.txt")
CLIENTS = ("prismaquant", "prismabuild")
_TOKEN = re.compile(r"\b(?:prismaquant|prismabuild)\.[a-z0-9_.]+")
_FLEET = "prismabuild-fleet"

_DOC_OWNERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, _DOC_OWNERS) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, (str, bytes))):
                ids.add(id(first.value))
    return ids


def _text(value: str | bytes) -> str:
    return value.decode("latin-1") if isinstance(value, bytes) else value


def _escape(value: str) -> str:
    return value.encode("unicode_escape").decode("ascii")


def findings(source: str) -> list[tuple[str, str]]:
    """``(kind, value)`` for every client reference in one module."""
    tree = ast.parse(source)
    docstrings = _docstring_ids(tree)
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [("import", a.name) for a in node.names
                      if a.name.split(".")[0] in CLIENTS]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and node.module.split(".")[0] in CLIENTS:
                found.append(("import", node.module))
        elif (isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes))
                and id(node) not in docstrings):
            text = _text(node.value)
            found += [("literal", _escape(m)) for m in _TOKEN.findall(text)]
            found += [("literal", _FLEET)] * text.count(_FLEET)
    return found


def scan(root: Path = ROOT) -> Counter:
    counts: Counter = Counter()
    for path in sorted((root / "src" / "tessera").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        for kind, value in findings(path.read_text(encoding="utf-8")):
            counts[f"{rel}\t{kind}\t{value}"] += 1
    return counts


def allowed(path: Path = ALLOWLIST) -> Counter:
    lines = path.read_text(encoding="utf-8").splitlines()
    return Counter(line for line in lines if line and not line.startswith("#"))


def test_scanner_sees_every_reference_form():
    source = (
        '"""prismaquant.docstring.v1 is prose."""\n'
        "import prismaquant.allocator as a, json\n"
        "from prismabuild.core import seal\n"
        "def f():\n"
        "    '''prismabuild.core in a docstring is prose.'''\n"
        "    import prismabuild\n"
        "    return b'prismaquant.tessera.v1/x', f'{a}prismabuild.pool.v2'\n"
        "X = 'fleet:prismabuild-fleet/cas'\n"
        "Y = 'prismaquantX.no'  # prismaquant.comment is prose\n"
    )
    assert sorted(findings(source)) == [
        ("import", "prismabuild"), ("import", "prismabuild.core"),
        ("import", "prismaquant.allocator"),
        ("literal", "prismabuild-fleet"), ("literal", "prismabuild.pool.v2"),
        ("literal", "prismaquant.tessera.v1"),
    ]


def test_client_references_equal_the_frozen_allowlist():
    live, frozen = scan(), allowed()
    new = {k: v for k, v in live.items() if k not in frozen}
    grown = {k: (frozen[k], v) for k, v in live.items() if k in frozen and v > frozen[k]}
    shrunk = {k: (n, live.get(k, 0)) for k, n in frozen.items() if live.get(k, 0) < n}
    assert not new, (
        f"new PrismaQuant/PrismaBuild references in src/tessera {sorted(new)}: "
        "Tessera stands alone; take the value from the caller or a plugin "
        "Tessera defines")
    assert not grown, f"allowlisted references gained occurrences (allowed, live): {grown}"
    assert not shrunk, (
        f"references removed (allowed, live): {shrunk}; delete their lines "
        f"from {ALLOWLIST.name} in the same change")
