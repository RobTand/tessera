#!/usr/bin/env python3
"""Select the tests a change can reach, by traversing the import graph.

The rule this serves says a branch owes evidence for the files it changed and
the files that import them.  "Generously" is how that was first stated, which
is a judgement call made by whoever is in a hurry.  This computes it instead:
parse every module's imports, invert the edges, and take everything
reverse-reachable from the changed set.

**It fails open, and that is the point.** Besides ordinary imports, explicit
file loaders contribute edges from their resolved paths, not their module
labels. Unresolved loader paths conservatively select the importing module's
reverse-reachable tests for any non-inert change (a conftest forces full).
Other kinds of coupling do not have ordinary import edges:

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

PrismaBuild snapshots add one generated source-closure member. Only the exact
member independently verified by ``tessera.suite_source`` against the sealed
action is removed before classification. A matching basename is not ownership
proof; unverified closure-shaped changes force a full selection. A snapshot
commit is intentionally parentless. When
both endpoints of ``BASE...HEAD`` exist but have no merge base, the selector
records and uses the equivalent direct ``BASE..HEAD`` tree comparison.

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
import re
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tessera.suite_source import measured_source  # noqa: E402
from tessera.source_dependencies import WILDCARD, file_imports  # noqa: E402

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

# This grammar only identifies metadata that needs verification. It never
# proves ownership or grants an exclusion: the sealed action does that.
PBRUN_CLOSURE_CANDIDATE = re.compile(
    r"\.pbrun-closure\.[0-9a-f]{16}\.json\Z"
)


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


def _imports(path: Path, own: str, root: Path) -> set[str]:
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
    paths, unknown = file_imports(tree, path, root)
    found.update(name for target in paths if (name := _module_name(target, root)))
    if unknown:
        found.add(WILDCARD)
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
        for target in _imports(path, name, root):
            if target == WILDCARD:
                importers[WILDCARD].add(name)
                continue
            # Attribute the edge to the longest known module prefix: an import
            # of tessera.encode.foo is an edge to tessera.encode.
            parts = target.split(".")
            for cut in range(len(parts), 0, -1):
                candidate = ".".join(parts[:cut])
                if candidate in by_name:
                    importers[candidate].add(name)
                    break
    return by_name, importers


def _reverse_reachable(seeds, importers):
    seen, queue = set(seeds), deque(seeds)
    while queue:
        node = queue.popleft()
        for importer in importers.get(node, ()):
            if importer not in seen:
                seen.add(importer)
                queue.append(importer)
    return seen


def _resolved_commit(ref: str, root: Path) -> str | None:
    out = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip() if out.returncode == 0 else None


def _parentless_direct_diff(ref: str, root: Path) -> tuple[list[str], str] | None:
    """Compare both trees directly when a valid three-dot pair has no base."""

    if ref.count("...") != 1:
        return None
    left, right = ref.split("...", 1)
    if not left or not right:
        return None
    merge_base = subprocess.run(
        ["git", "-C", str(root), "merge-base", left, right],
        capture_output=True,
        text=True,
    )
    # ``merge-base`` returns one for two valid, unrelated histories. Bad refs
    # return a different failure and must retain the original refusal.
    if merge_base.returncode != 1 or merge_base.stdout.strip():
        return None
    left_commit = _resolved_commit(left, root)
    right_commit = _resolved_commit(right, root)
    if left_commit is None or right_commit is None:
        return None
    out = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", "-z", left_commit, right_commit],
        capture_output=True,
        text=True, errors="surrogateescape",
    )
    if out.returncode != 0:
        return None
    comparison = (
        f"direct {left_commit}..{right_commit} "
        f"(no merge base; requested {left}...{right})"
    )
    return out.stdout.split("\0"), comparison


def changed_files(ref: str, root: Path) -> tuple[list[str], str]:
    out = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", "-z", ref],
        capture_output=True, text=True, errors="surrogateescape",
    )
    if out.returncode != 0:
        fallback = _parentless_direct_diff(ref, root)
        if fallback is None:
            raise SystemExit(f"git diff {ref} failed: {out.stderr.strip()}")
        lines, comparison = fallback
    else:
        lines, comparison = out.stdout.split("\0"), ref
    # Display-quoted names are not paths: tabs/newlines must not hide a
    # closure candidate (or split any other path) from classification.
    changed = [line for line in lines if line]
    if any(PBRUN_CLOSURE_CANDIDATE.fullmatch(Path(line).name) for line in changed):
        source = measured_source(root)
        verified = ({member["path"] for member in source["excluded_metadata"]}
                    if source["verification"] == "verified" else set())
        changed = [line for line in changed if line not in verified]
    return sorted(changed), comparison


def _selection_reason(
    changed: list[str],
    *,
    missing: list[str],
    forced: list[str],
    tests: list[str],
    text_matched: set[str],
) -> str:
    if missing:
        return (
            "run this in the branch's own worktree: changed files absent from "
            "this checkout have unreadable edges"
        )
    if forced:
        return "changed paths the import graph cannot reason about; run everything"

    python_paths = {path for path in changed if Path(path).suffix == ".py"}
    inert_paths = {path for path in changed if Path(path).suffix in INERT}
    non_python = set(changed) - python_paths - inert_paths
    parts: list[str] = []
    if python_paths:
        parts.append(
            "Python import graph selected reverse-reachable tests"
            if tests
            else "Python import graph found no reverse-reachable tests"
        )
    if text_matched:
        parts.append("text matches selected tests for non-Python changed paths")
    if non_python - text_matched:
        parts.append("non-Python changed paths have no text-matched tests")
    if inert_paths:
        parts.append("inert changed paths require no tests")
    return "; ".join(parts) or "no changed path requires a test"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Tests reverse-reachable from a change, or a full-run verdict."
    )
    ap.add_argument(
        "--ref",
        default="master...HEAD",
        help=("git diff spec naming the change (default master...HEAD); a "
              "parentless snapshot with valid endpoints falls back from "
              "BASE...HEAD to a direct BASE..HEAD tree comparison"),
    )
    ap.add_argument("--root", default=".")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    changed, comparison = changed_files(args.ref, root)
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
    # An unowned metadata-shaped file remains a real change. We cannot infer
    # that it is harmless scaffolding from its spelling, so fail open.
    forced += [f for f in changed if PBRUN_CLOSURE_CANDIDATE.fullmatch(Path(f).name)]

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
    unresolved = (importers.get(WILDCARD, set())
                  if any(Path(f).suffix not in INERT for f in changed) else set())
    seeds.update(unresolved)
    uncertain_consumers = _reverse_reachable(unresolved, importers)
    forced += [str(by_name[name].relative_to(root)) for name in sorted(uncertain_consumers)
               if by_name[name].name == "conftest.py"]
    missing = [f for f in changed
               if f.endswith(".py") and not (root / f).exists()]
    # Reverse-reachable closure: everything that imports a changed module,
    # transitively.
    seen = _reverse_reachable(seeds, importers)

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
    text_matched: set[str] = set()
    if opaque_py:
        all_tests = [p for p in (root / "tests").rglob("test_*.py")]
        for path in all_tests:
            body = path.read_text(encoding="utf-8", errors="replace")
            matches = {
                f for f in opaque_py if f in body or Path(f).name in body
            }
            if matches:
                tests.append(str(path.relative_to(root)))
                text_matched.update(matches)
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
        "comparison": comparison,
        "tests": tests,
        "forces_full": forced,
        "unresolved_file_loaders": sorted(
            str(by_name[name].relative_to(root)) for name in unresolved),
        "reason": _selection_reason(
            changed,
            missing=missing,
            forced=forced,
            tests=tests,
            text_matched=text_matched,
        ),
    }
    if unresolved:
        result["reason"] += "; unresolved file loaders conservatively select their consumers"
    if args.json:
        print(json.dumps(result, indent=1))
    else:
        print(f"verdict: {verdict}  ({len(changed)} changed, {len(tests)} tests)")
        print(f"comparison: {comparison}")
        if forced:
            print("forces full run:")
            for f in forced:
                print(f"  {f}")
        for t in tests:
            print(f"  {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
