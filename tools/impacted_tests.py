#!/usr/bin/env python3
"""Select the tests a change can reach, by traversing the import graph.

The rule this serves says a branch owes evidence for the files it changed and
the files that import them.  "Generously" is how that was first stated, which
is a judgement call made by whoever is in a hurry.  This computes it instead:
parse every module's imports, invert the edges, and take everything
reverse-reachable from the changed set.

**It fails open, and that is the point.**  An import graph is sound only for
coupling that an ``import`` statement expresses.  Three kinds here do not:

* *conftest.py* is imported by pytest, not by the tests.  A changed conftest
  impacts every test at or below its directory, and that edge is added
  explicitly.
* *Data coupling* -- a JSON spec, a wire schema, or a shell harness is read at
  runtime by code that never imports it as Python.  For a non-Python file the
  graph has nothing to traverse, so the fallback is textual: any test that
  mentions the path or its basename is impacted, and a file no test mentions
  is inert *for test selection*.
* *The wire and the packaging* are un-analysable by either route and are named
  in ``OPAQUE``.

One thing this deliberately does **not** do is certify serving behaviour.
``tessera.serving`` is loaded by vLLM through an entry point, so nothing here
imports it the way the runtime does -- but tests that import it are still
found by the graph, because that is a different question. Selecting the right
tests is not the same as proving the plugin still serves, and a served check
remains its own gate.

So the answer is a *selection plus a verdict*.  When the verdict is
``full``, the caller runs everything; a narrowed list is only ever returned
when every changed path was something the graph can actually reason about.
Silent narrowing is how a selector like this turns into a hole.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path

# Coupling no import statement expresses.  A change at or below any of these
# forces the full suite rather than a narrowed list.
OPAQUE = (
    "docs/schema/",                # the wire; read, never imported
    "pyproject.toml",
)
# Scratch copies of the tree: worker output dirs hold whole checkouts, and
# walking them selects tests that are not the ones about to run.
SKIP_DIRS = {".git", ".claude", "archive", "build", ".venv", "node_modules",
             "muse-out", "worktrees", "__pycache__"}
# Extensions that cannot change behaviour and never force a full run.
INERT = {".md", ".txt", ".rst"}


def _module_name(path: Path, root: Path) -> str | None:
    """Dotted name for a file, using the layout this repo actually has."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    if rel.suffix != ".py":
        return None
    parts = list(rel.parts)
    if parts[0] == "src":
        parts = parts[1:]
    parts[-1] = rel.stem
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def _imports(path: Path, own: str) -> set[str]:
    """Every module this file imports, relative imports resolved against own."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return set()
    package = own.rsplit(".", 1)[0] if "." in own else ""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # A relative import climbs from this module's package.
                base = package.split(".") if package else []
                climb = node.level - 1
                base = base[:len(base) - climb] if climb else base
                prefix = ".".join(base + ([node.module] if node.module else []))
            else:
                prefix = node.module or ""
            if not prefix:
                continue
            found.add(prefix)
            # "from x import y" may name a submodule rather than an attribute;
            # both readings are recorded because only the graph can tell.
            found.update(f"{prefix}.{alias.name}" for alias in node.names)
    return found


def build_graph(root: Path) -> tuple[dict[str, Path], dict[str, set[str]]]:
    """Module name -> file, and module name -> the modules that import it."""
    files = [
        p for p in root.rglob("*.py")
        if not any(part in SKIP_DIRS for part in p.parts)
    ]
    by_name: dict[str, Path] = {}
    for path in files:
        name = _module_name(path, root)
        if name:
            by_name.setdefault(name, path)
    importers: dict[str, set[str]] = defaultdict(set)
    for name, path in by_name.items():
        for target in _imports(path, name):
            # Attribute the edge to the longest known module prefix: an import
            # of tessera.encode.foo is an edge to tessera.encode.
            parts = target.split(".")
            for cut in range(len(parts), 0, -1):
                candidate = ".".join(parts[:cut])
                if candidate in by_name:
                    importers[candidate].add(name)
                    break
    return by_name, importers


def changed_files(ref: str, root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", ref],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"git diff {ref} failed: {out.stderr.strip()}")
    return [line for line in out.stdout.splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Tests reverse-reachable from a change, or a full-run verdict."
    )
    ap.add_argument("--ref", default="master...HEAD",
                    help="git diff spec naming the change (default master...HEAD)")
    ap.add_argument("--root", default=".")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    changed = changed_files(args.ref, root)
    if not changed:
        print("no changes", file=sys.stderr)
        return 0

    forced = [
        f for f in changed
        if Path(f).suffix not in INERT
        and any(f.startswith(o) for o in OPAQUE)
    ]
    # The root conftest is imported by pytest for the whole tree.
    forced += [f for f in changed if f == "conftest.py"]

    by_name, importers = build_graph(root)
    name_of = {str(p.relative_to(root)): n for n, p in by_name.items()}

    # Seed from the path, not from a lookup in the checked-out tree.  The
    # graph is built from whatever is checked out; the changed list comes from
    # the branch.  A file the branch ADDS has no node here, so a lookup drops
    # it silently -- which is how the first version of this tool missed the two
    # tests a human had already named for #92.  Deriving the name from the path
    # seeds it whether or not the file is present.
    seeds = set()
    for f in changed:
        name = name_of.get(f) or _module_name(root / f, root)
        if name:
            seeds.add(name)
    missing = [f for f in changed
               if f.endswith(".py") and not (root / f).exists()]
    # Reverse-reachable closure: everything that imports a changed module,
    # transitively.
    seen, queue = set(seeds), deque(seeds)
    while queue:
        node = queue.popleft()
        for importer in importers.get(node, ()):
            if importer not in seen:
                seen.add(importer)
                queue.append(importer)

    # A module in `seen` may have no file in this checkout -- a test the branch
    # ADDS is exactly that case, and it is the one selection can least afford
    # to miss.  Fall back to the changed path itself.
    added = {(_module_name(root / f, root) or ""): f for f in changed}
    found: set[str] = set()
    for name in seen:
        path = by_name.get(name)
        rel = str(path.relative_to(root)) if path else added.get(name)
        if rel and Path(rel).name.startswith("test_"):
            found.add(rel)
    tests = sorted(found)
    # Non-Python changes have no import edges.  Fall back to text: a test that
    # names the file is coupled to it, and one that never mentions it is not.
    opaque_py = {f for f in changed
                 if Path(f).suffix not in INERT and Path(f).suffix != ".py"}
    if opaque_py:
        all_tests = [p for p in (root / "tests").rglob("test_*.py")]
        for path in all_tests:
            body = path.read_text(encoding="utf-8", errors="replace")
            if any(f in body or Path(f).name in body for f in opaque_py):
                tests.append(str(path.relative_to(root)))
        tests = sorted(set(tests))

    # A changed conftest is imported by pytest, not by the tests it serves.
    for f in changed:
        if Path(f).name == "conftest.py" and f != "conftest.py":
            scope = str(Path(f).parent)
            tests = sorted(set(tests) | {
                str(q.relative_to(root))
                for q in (root / scope).rglob("test_*.py")
            })

    # Edges out of a file that is not in this tree cannot be read, so a branch
    # analysed from another checkout is under-approximated.  Say so loudly
    # rather than returning a confidently short list.
    if missing:
        forced = forced + missing
    verdict = "full" if forced else ("narrowed" if tests else "none")
    result = {
        "verdict": verdict,
        "changed": len(changed),
        "tests": tests,
        "forces_full": forced,
        "reason": (
            "run this in the branch's own worktree: changed files absent from "
            "this checkout have unreadable edges" if missing else
            "changed paths the import graph cannot reason about; run everything"
            if forced else
            "every changed path is a Python module the graph covers"
        ),
    }
    if args.json:
        print(json.dumps(result, indent=1))
    else:
        print(f"verdict: {verdict}  ({len(changed)} changed, {len(tests)} tests)")
        if forced:
            print("forces full run:")
            for f in forced:
                print(f"  {f}")
        for t in tests:
            print(f"  {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
