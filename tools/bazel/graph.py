"""Each test file's Bazel input closure, from tools/bazel/evidence.py.

A test target holds its own file, every module it imports transitively, the
modules tests/conftest.py imports, and the data of each. Bazel reruns the test
when any of those files changes. tools/bazel/build_files.py writes these
targets.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath

from evidence import Evidence, is_module, is_test, is_test_side

CONFTEST = 'tests/conftest.py'
# Every test target's shared inputs besides conftest: what the runner reads.
SHARED_FILES = ('pyproject.toml',)


def strongly_connected(nodes, edges):
    """Tarjan's algorithm, iterative. Returns components in a stable order."""
    index, low, on_stack, stack, components = {}, {}, set(), [], []
    counter = 0
    for start in sorted(nodes):
        if start in index:
            continue
        work = [(start, iter(sorted(edges.get(start, ()))))]
        index[start] = low[start] = counter
        counter += 1
        stack.append(start)
        on_stack.add(start)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(edges.get(child, ())))))
                    advanced = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(sorted(component))
    return sorted(components)


class Graph:
    def __init__(self, evidence: Evidence | None = None):
        self.evidence = evidence or Evidence()
        self.edges = {path: self.evidence.edges(path) for path in self.evidence.modules}
        self.data = {path: self.evidence.data(path) for path in self.evidence.modules}
        self.tests = [path for path in self.evidence.modules if is_test(path)]
        self._reach = {}
        self._launch = {}

    def reach(self, start):
        """Every module `start` imports, transitively, itself included."""
        if start not in self._reach:
            seen, pending = {start}, [start]
            while pending:
                for module in self.edges.get(pending.pop(), ()):
                    if module not in seen:
                        seen.add(module)
                        pending.append(module)
            self._reach[start] = frozenset(seen)
        return self._reach[start]

    def test_data(self, test):
        """A test's own extra inputs: tree reads of the production modules it imports directly."""
        return set().union(set(), *(self.evidence.tree_reads(module) for module in self.edges[test]
                                    if not is_test_side(module)))

    def launched(self, test):
        """Modules the test's code launches or loads by path, or that the
        shell scripts it reaches run, and what those launch in turn. Each
        one's library brings its own imports."""
        return self._launches(test)[0]

    def script_files(self, test):
        """Repository files the shell scripts a test reaches name."""
        return self._launches(test)[1]

    def _launches(self, test):
        if test in self._launch:
            return self._launch[test]
        reached = set(self.reach(test) | self.reach(CONFTEST))
        found, files = set(), set()
        while True:
            new = set().union(set(), *(self.evidence.programs(module) for module in reached))
            scripts = {file for file in set().union(self.test_data(test), *(self.data[module] for module in reached))
                       if file in self.evidence.scripts}
            seen = set()
            while scripts - seen:
                script = sorted(scripts - seen)[0]
                seen.add(script)
                modules, named = self.evidence.scripts[script]
                new |= modules
                files |= named | {script}
                scripts |= {file for file in named if file in self.evidence.scripts}
            new -= reached | found
            if not new:
                self._launch[test] = (frozenset(found), frozenset(files))
                return self._launch[test]
            found |= new
            for module in new:
                reached |= self.reach(module)

    def closure(self, test):
        """Every repository file the test target holds."""
        modules = self.reach(test) | self.reach(CONFTEST)
        for program in self.launched(test):
            modules |= self.reach(program)
        files = set(modules) | set(SHARED_FILES) | self.script_files(test)
        for module in modules:
            files |= self.data[module]
        return files | self.test_data(test)

    def closures(self):
        return {test: self.closure(test) for test in self.tests}


def selectors(closures):
    """{file: tests whose closure holds it}."""
    found = defaultdict(set)
    for test, files in closures.items():
        for file in files:
            found[file].add(test)
    return found


# A change to one of these rebuilds the graph or the toolchain: every test.
GLOBAL_INPUTS = ('MODULE.bazel', 'MODULE.bazel.lock', '.bazelrc', '.bazelversion', '.bazelignore',
                 'tools/bazel/', '.github/workflows/gate.yml', 'tools/ci/gate_requirements.txt',
                 'tools/ci/gate_torch_requirements.txt')


def is_global(path):
    return path.startswith(GLOBAL_INPUTS) or PurePosixPath(path).name == 'BUILD.bazel'
