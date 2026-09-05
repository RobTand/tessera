#!/usr/bin/env python3
"""Select the tests a change can reach, by traversing the import graph.

The rule this serves says a branch owes evidence for the files it changed and
the files that import them.  "Generously" is how that was first stated, which
is a judgement call made by whoever is in a hurry.  This computes it instead:
parse every module's imports, invert the edges, and take everything
reverse-reachable from the changed set.

**It fails open to the full run, and never under-selects: that is the whole
contract.** Every gap in what this can prove resolves towards running more
tests -- an ambiguous spelling edges to every file it can name, a module whose
dependencies cannot be established selects its consumers, and an unresolvable
scope forces the whole suite. A narrowed list is a claim that the graph read
everything relevant; where it did not, the honest answer is ``full``.

Besides ordinary imports, explicit
file loaders and source reads contribute edges from resolved paths, not module
labels. A path it cannot resolve conservatively selects the reading module's
reverse-reachable tests for any non-inert change, and an unresolved *loader*
reaching a conftest forces full -- a conftest that can run code it cannot name
makes every test below it unpredictable. A file that will not parse or read is
the same uncertainty from the other end: it states no dependency, which is not
the same as having none, so it too may import anything, its consumers are
selected, and an unreadable conftest forces the population it gates. The
receipt names it and the failure, because that one is repairable (#293). An
unresolved *read* does not: bytes
become a Python dependency only when something parses or executes them, and
``tessera._dev.source_dependencies`` says which readers can. Reading that
distinction the other way is what made this tool return ``full`` for every
change this repository can make (#148): one module hashes every tracked file,
the root conftest imports it, and the whole tree was uncertain forever.
But "not a module edge" is not "no edge". A read whose target the resolver
*named* and then refused to place -- an absolute spelling outside the tree,
which a local alias directory can carry straight back into it -- keeps a data
dependency it cannot attribute to a file, so its reader is seeded for every
non-inert change instead. Dropping that was under-selection with no
diagnostic at all: the changed JSON moved the reader's bytes and the verdict
was a confident ``none`` (#338). It is deliberately weaker than the module
wildcard -- it selects the reader's consumers and never forces the full run --
because a reader that executes nothing imports nothing.

Other kinds of coupling do not have ordinary import edges:

* *conftest.py* is imported by pytest, not by the tests.  A conftest the
  change *reaches* -- because it was edited, or because it imports something
  that was -- impacts every test at or below its directory, and that edge is
  added explicitly.  A test names a fixture, never the module the conftest
  built the fixture from, so this is the only path there is from a conftest's
  helper to its consumers.  The reverse -- a conftest that execs each ``test_*.py`` below it
  to decide what it can collect -- is a collection probe, and it is excluded
  from the walk that forces full: read as a dependency it closes a cycle
  through which any one uncertain test file makes the whole population
  uncertain.  An ordinary ``import`` in a conftest is not a probe.
* *Data coupling* -- a JSON spec, a wire schema, or a shell harness is read at
  runtime by code that never imports it as Python.  A file a module names by
  an explicit path is a node in the graph under that path, so changing it
  selects that module's tests.  For the rest the fallback is textual: any test
  that mentions the path or its basename is impacted, and a file no test
  mentions and no module reads is inert *for test selection*.
* *The wire and the packaging* are un-analysable by either route and are named
  in ``OPAQUE``.

PrismaBuild snapshots add one generated source-closure member. Only the exact
member independently verified by ``tessera._dev.suite_source`` against the sealed
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
from tessera._dev.suite_source import measured_source  # noqa: E402
from tessera._dev.source_dependencies import (  # noqa: E402
    DATA_WILDCARD, WILDCARD, file_imports)

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


def _package_name(path: Path, root: Path) -> str | None:
    """The name this file has under its own package root.

    pytest's rootdir rule, and the one ``tests/conftest.py`` makes explicit by
    inserting its own directory on ``sys.path``: a module's package is the
    chain of directories above it that hold an ``__init__.py``, and the first
    directory that does not is an import root.  ``tests/`` holds no
    ``__init__.py``, so ``tests/box_artifacts.py`` is importable as
    ``box_artifacts`` -- which is how every test in this tree spells it, and
    how none of them spells ``tests.box_artifacts``.  An import that resolves
    to no node selects nothing, so the alias is the difference between an edge
    and a silent drop (#215).
    """

    name = _module_name(path, root)
    if name is None:
        return None
    parts = name.split(".")
    directory = path.parent
    depth = 0
    while (directory / "__init__.py").exists() and directory != root:
        depth += 1
        directory = directory.parent
    alias = ".".join(parts[len(parts) - depth - 1:]) if path.stem != "__init__" \
        else ".".join(parts[len(parts) - depth:])
    return alias or None


def _imports(
    path: Path,
    own: str,
    root: Path,
    unreadable: dict[str, str] | None = None,
    nodes: dict[Path, str] | None = None,
) -> tuple[set[str], set[str], set[str]]:
    """What this file depends on, split by how the dependency was established.

    Statement imports, modules named by an explicit file path, and repository
    files read by an explicit path that are not Python.  The split is not
    cosmetic: importing ``pkg.mod`` executes ``pkg/__init__.py`` and loading
    ``pkg/mod.py`` by path does not, and a conftest that reaches a test file
    by path is probing its own collection targets rather than depending on
    them.

    ``nodes`` maps a file to the node the graph holds it under, which is the
    only thing that can answer a *path* load: the dotted name a path spells
    can belong to another file (#317), so deriving the node from the name
    would attribute the edge to the wrong one.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError) as exc:
        # Three sets, like every other return here: answering with one made
        # the caller's unpack raise, and a selector that raises selects
        # nothing.  But three EMPTY sets state that this module imports
        # nothing, which is the opposite of what a failed parse establishes.
        # A file nobody could read states no dependency, and "no dependency
        # stated" is uncertainty, not independence (#293): a test importing
        # the changed leaf that does not parse lost its edge and its
        # selection, while collecting it would have failed.  So it is the
        # WILDCARD case the unresolved-loader path already models -- this
        # module may import anything -- and the file and its failure are
        # recorded so the operator can repair it.
        if unreadable is not None:
            unreadable[str(path.relative_to(root))] = f"{type(exc).__name__}: {exc}"
        return {WILDCARD}, set(), set()
    # An ``__init__.py`` IS its package: ``from .child import VALUE`` there
    # names ``pkg.child``, not a top-level ``child``.  Climbing from the
    # parent instead lost every relative re-export a package initializer
    # makes -- the form ``src/tessera/__init__.py`` is written in (#215).
    if path.name == "__init__.py":
        package = own
    else:
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
    paths, unknown, unplaced = file_imports(tree, path, root)
    loaded, data = set(), set()
    for target in paths:
        held = nodes.get(target) if nodes is not None else None
        if held is None:
            held = _module_name(target, root)
        if held:
            loaded.add(held)
        else:
            data.add(str(target.relative_to(root)))
    if unknown:
        found.add(WILDCARD)
    if unplaced:
        # A data node, not a module one: this reader executes nothing, so it
        # imports nothing, but it does read a file the resolver named and
        # refused to place.  ``DATA_WILDCARD`` is the node that file is held
        # under when no single path can be (#338).
        data.add(DATA_WILDCARD)
    return found, loaded, data


def _is_collection_probe(importer: Path, target: Path | None) -> bool:
    """A conftest reaching a test file *by path* is probing, not depending.

    ``tests/conftest.py`` execs each ``test_*.py`` to decide what it can
    collect.  That is a real call, but the dependency it expresses runs the
    other way: pytest imports the conftest for those tests, and each test's
    own uncertainty is already carried by that test's own selection.  Read as
    an ordinary edge it closes a cycle -- conftest imports every test, every
    test depends on conftest -- through which one uncertain test file makes
    the whole population uncertain.  An ordinary ``from tests.helper import
    x`` in a conftest is a code dependency and is NOT this.
    """
    return (target is not None
            and importer.name == "conftest.py"
            and target.name.startswith("test_")
            and target.is_relative_to(importer.parent))


def import_graph(
    root: Path,
) -> tuple[dict[str, Path], dict[str, set[str]],
           set[tuple[str, str]], dict[str, str]]:
    """The graph, the collection-probe reverse edges, and what would not read.

    ``importers`` is keyed by node.  A node is a file's dotted module name;
    or its repository-relative path, both for a file read by an explicit path
    that is not Python -- so a JSON spec or a shell harness a module reads is
    a node like any other -- and for a Python file whose dotted name another
    file already answers to (#317); or ``WILDCARD``/``DATA_WILDCARD``, the two
    nodes a dependency is held under when no file can be.  ``unreadable``
    maps the repository-relative path of every file that would not parse or
    read to the failure, and each of those files is a ``WILDCARD`` importer:
    the graph does not know what it depends on (#293).  A ``DATA_WILDCARD``
    importer is the weaker claim -- it reads a file the resolver named and
    refused to place, and imports nothing (#338).
    """
    files = [
        p for p in root.rglob("*.py")
        if not any(part in SKIP_DIRS for part in p.parts)
    ]
    # Every file gets a node, and a node's identity is its file.  A dotted
    # name is only a *spelling*, and a spelling can name more than one file:
    # ``_module_name`` strips a leading ``src/``, so ``helper.py`` and
    # ``src/helper.py`` are both ``helper``.  Keeping the first and dropping
    # the second left that file out of the graph entirely -- never parsed, so
    # its own imports were not edges, so a test reaching a changed module only
    # through it was not selected and the verdict was a confident ``none``
    # (#317).  The loser therefore gets a node under its own repository path,
    # which is what the graph already does for the data files a module reads,
    # and both files answer to the shared spelling the way an ambiguous alias
    # does (#292).  ``module_of`` keeps each node's dotted name, which is what
    # relative imports climb from.
    by_name: dict[str, Path] = {}
    module_of: dict[str, str] = {}
    for path in files:
        name = _module_name(path, root)
        if not name:
            continue
        node = name if name not in by_name else str(path.relative_to(root))
        by_name[node] = path
        module_of[node] = name
    # A path never spells a module name, so a path-keyed node cannot collide
    # with a dotted one, and one file is one node either way.
    nodes = {path: node for node, path in by_name.items()}
    # The other spellings the same files have on the import roots that are
    # actually on ``sys.path``.  An alias can be ambiguous -- two import roots
    # can each hold a ``helper.py``, and one of those roots can be the
    # repository root, which owns the canonical name -- and every candidate
    # gets the edge, because the graph cannot tell which one ran and a dropped
    # edge is the failure mode this tool refuses.
    aliases: dict[str, set[str]] = defaultdict(set)
    for node, path in by_name.items():
        canonical = module_of[node]
        if canonical != node:
            aliases[canonical].add(node)
        alias = _package_name(path, root)
        if alias and alias not in (node, canonical):
            aliases[alias].add(node)

    def _targets(candidate: str) -> tuple[str, ...]:
        """Every file this spelling can name, deduplicated by path.

        Owning the canonical name is not precedence.  A root ``helper.py``
        answers to ``helper`` and so does ``tests/helper.py`` on the import
        root ``tests/conftest.py`` inserts -- and pytest executes the second
        one for ``from helper import VALUE``.  Returning the canonical module
        alone dropped exactly the candidate that runs (#292).  Nothing here
        models import-root precedence, so nothing here may pick a winner:
        the union is the only answer this graph can prove, and over-selecting
        is the direction the tool fails in.
        """
        names = set(aliases.get(candidate, ()))
        if candidate in by_name:
            names.add(candidate)
        seen_paths: set[Path] = set()
        resolved: list[str] = []
        for name in sorted(names):
            path = by_name[name]
            if path in seen_paths:
                continue
            seen_paths.add(path)
            resolved.append(name)
        return tuple(resolved)

    importers: dict[str, set[str]] = defaultdict(set)
    probes: set[tuple[str, str]] = set()
    unreadable: dict[str, str] = {}
    for node, path in by_name.items():
        statements, loaded, data = _imports(
            path, module_of[node], root, unreadable, nodes)
        for target in statements:
            if target == WILDCARD:
                importers[WILDCARD].add(node)
                continue
            # Attribute the edge to the longest known module prefix: an import
            # of tessera.encode.foo is an edge to tessera.encode.  Importing a
            # submodule also EXECUTES every package __init__ above it, so those
            # are edges too.  Stopping at the longest prefix dropped them, and
            # a package is exactly where a re-export lives: at #148 a change to
            # src/tessera/__init__.py selected 98 of the 123 test modules that
            # reach the package, and src/tessera/serving/__init__.py 31 of 70.
            parts = target.split(".")
            matched = False
            for cut in range(len(parts), 0, -1):
                candidate = ".".join(parts[:cut])
                known = _targets(candidate)
                if not known:
                    continue
                for resolved in known:
                    if not matched or by_name[resolved].name == "__init__.py":
                        importers[resolved].add(node)
                matched = True
        for target in loaded:
            # An exact path names an exact file.  It does not execute the
            # packages above it, so it gets no prefix edges.
            importers[target].add(node)
            if _is_collection_probe(path, by_name.get(target)):
                probes.add((target, node))
        for target in data:
            importers[target].add(node)
    return by_name, importers, probes, unreadable


def build_graph(root: Path) -> tuple[dict[str, Path], dict[str, set[str]]]:
    """Node -> file, and node -> the nodes that import it.

    A node is a file's dotted module name, or its repository-relative path
    when another file already answers to that name (#317) and for the
    non-Python files a module reads by an explicit path.
    """
    by_name, importers, _, _ = import_graph(root)
    return by_name, importers


def _reverse_reachable(seeds, importers, skip=frozenset()):
    seen, queue = set(seeds), deque(seeds)
    while queue:
        node = queue.popleft()
        for importer in importers.get(node, ()):
            if importer in seen or (node, importer) in skip:
                continue
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
    data_matched: set[str] = frozenset(),
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
    if data_matched:
        parts.append("non-Python changed paths reached their readers "
                     "through the import graph")
    if non_python - text_matched - data_matched:
        parts.append("non-Python changed paths have no text-matched tests")
    if inert_paths - data_matched:
        parts.append("inert changed paths require no tests")
    return "; ".join(parts) or "no changed path requires a test"


def select(root: Path, changed: list[str], *, comparison: str = "") -> dict:
    """The receipt for a changed-file list: a verdict, a selection and a reason.

    Separated from ``main`` so a caller -- a test, above all -- can drive the
    classification without a git checkout to diff.  The tool's own regression
    could not reach this code before: it went through ``changed_files``, so
    every case had to be a synthetic repository.
    """
    # An explicitly OPAQUE path outranks the generic inert-suffix rule: the
    # wire spec IS a Markdown file, so filtering ``.md`` first meant the one
    # path ``docs/schema/`` was listed for could never reach the rule that
    # lists it, and a wire change certified a narrowed selection (#216).
    forced = [f for f in changed if any(f.startswith(o) for o in OPAQUE)]
    # The root conftest is imported by pytest for the whole tree.
    forced += [f for f in changed if f == "conftest.py"]
    # An unowned metadata-shaped file remains a real change. We cannot infer
    # that it is harmless scaffolding from its spelling, so fail open.
    forced += [f for f in changed if PBRUN_CLOSURE_CANDIDATE.fullmatch(Path(f).name)]

    by_name, importers, probes, unreadable = import_graph(root)
    name_of = {str(p.relative_to(root)): n for n, p in by_name.items()}

    # Seed from the path, not from a lookup in the checked-out tree.  The
    # graph is built from whatever is checked out; the changed list comes from
    # the branch.  A file the branch ADDS has no node here, so a lookup drops
    # it silently -- which is how the first version of this tool missed the two
    # tests a human had already named for #92.  Deriving the name from the path
    # seeds it whether or not the file is present.
    seeds = set()
    data_changed = set()
    for f in changed:
        name = name_of.get(f) or _module_name(root / f, root)
        if name:
            seeds.add(name)
        else:
            # Not a module, so it is a node under its own path: the graph
            # holds an edge for every file a module reads by an explicit
            # path, whatever its suffix.
            seeds.add(f)
            data_changed.add(f)
    # Two kinds of module state an unknown dependency: one that loads a file
    # by a path this cannot resolve, and one that would not parse or read at
    # all.  They seed identically -- their consumers are selected, and a
    # conftest among them forces the population -- and are reported apart,
    # because "repair this file" is the only action one of them admits.
    non_inert = any(Path(f).suffix not in INERT for f in changed)
    uncertain = importers.get(WILDCARD, set()) if non_inert else set()
    seeds.update(uncertain)
    unresolved = {name for name in uncertain
                  if str(by_name[name].relative_to(root)) not in unreadable}
    # A third kind: a module that reads a file it named and the resolver
    # refused to place.  It is NOT in ``uncertain`` -- it imports nothing, so
    # it must not force the population the way an unknown loader does -- but
    # its own bytes-in depend on a file the diff may hold, so it is a seed
    # like any changed module and its consumers are selected (#338).  The
    # alternative was the receipt this tool cannot afford: verdict ``none``,
    # no tests, no diagnostic, for a change that moved the reader's output.
    unplaced = importers.get(DATA_WILDCARD, set()) if non_inert else set()
    seeds.update(unplaced)
    # The probe edges are excluded HERE and nowhere else: a conftest's own
    # uncertainty still forces the population, but a test file's does not
    # become the conftest's by way of the conftest having exec'd it.
    uncertain_consumers = _reverse_reachable(uncertain, importers, skip=probes)
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
    data_matched = {f for f in data_changed if importers.get(f)}
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

    # A conftest is imported by pytest, not by the tests it serves, so its
    # scope -- every test at or below its directory -- is an edge no import
    # statement expresses.  REACHING one is the same fact as changing one: a
    # fixture consumer names the fixture, never the module the conftest built
    # it from, so a helper the conftest imports has no other path to the tests
    # that request it (#215).  The probe edges are excluded here for the same
    # reason they are excluded from the escalation walk: a conftest that execs
    # the test files below it would otherwise make any one changed test file
    # select the whole population.
    reached = _reverse_reachable(seeds, importers, skip=probes)
    scopes = {by_name[name].parent for name in reached
              if name in by_name and by_name[name].name == "conftest.py"}
    scopes |= {(root / f).parent for f in changed if Path(f).name == "conftest.py"}
    for scope in sorted(scopes):
        tests = sorted(set(tests) | {
            str(q.relative_to(root)) for q in scope.rglob("test_*.py")
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
        "unplaced_data_reads": sorted(
            str(by_name[name].relative_to(root)) for name in unplaced),
        # A property of the tree, not of this change: report it whether or not
        # this change reaches it, because it is a defect to repair either way.
        "unreadable_sources": {path: unreadable[path] for path in sorted(unreadable)},
        "reason": _selection_reason(
            changed,
            missing=missing,
            forced=forced,
            tests=tests,
            text_matched=text_matched,
            data_matched=data_matched,
        ),
    }
    if unresolved:
        result["reason"] += "; unresolved file loaders conservatively select their consumers"
    if unplaced:
        result["reason"] += (
            "; reads of paths this resolver named but refused to place "
            "conservatively select their readers' consumers")
    unreadable_reached = sorted(str(by_name[name].relative_to(root))
                                for name in uncertain - unresolved)
    if unreadable_reached:
        result["reason"] += (
            "; sources that would not parse or read state no dependency and "
            "conservatively select their consumers -- repair "
            + ", ".join(f"{path} ({unreadable[path].split(':', 1)[0]})"
                        for path in unreadable_reached))
    if scopes:
        result["reason"] += (
            "; a changed path reaches a conftest, which pytest imports for "
            "every test in its scope")
    return result


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

    result = select(root, changed, comparison=comparison)
    verdict = result["verdict"]
    tests = result["tests"]
    forced = result["forces_full"]
    if args.json:
        print(json.dumps(result, indent=1))
    else:
        print(f"verdict: {verdict}  ({len(changed)} changed, {len(tests)} tests)")
        print(f"comparison: {comparison}")
        if result["unreadable_sources"]:
            print("would not parse or read (dependencies unknown; repair these):")
            for path, failure in result["unreadable_sources"].items():
                print(f"  {path}: {failure}")
        # Printed, because the failure #338 records was a selector that said
        # ``none`` and gave the operator nothing to read.
        if result["unplaced_data_reads"]:
            print("reads a path outside this tree's spelling "
                  "(dependency kept, target unnamed):")
            for path in result["unplaced_data_reads"]:
                print(f"  {path}")
        if forced:
            print("forces full run:")
            for f in forced:
                print(f"  {f}")
        # Labelled, because the two lists print with the same indent: an
        # unlabelled tail read as more of the forcing list, which says the
        # opposite of what a selection says.
        if tests:
            print(f"selected tests ({len(tests)}):")
        for t in tests:
            print(f"  {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
