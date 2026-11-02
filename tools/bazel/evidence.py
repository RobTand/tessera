"""Each Python file's dependencies, as Bazel needs them: imports, launches, reads.

The merge gate's selector (``tools/impacted_tests.py``) answers a different
question. It must never miss a test, so a string that might name a module is
an import, a read it can't place couples the reader to the whole tree, and
any module that can run source may import anything. On a hosted runner that
caution is the only safety net. Under Bazel it isn't: a remote test action
holds only its declared inputs, so an import or read this evidence misses
fails the test instead of passing it. That lets the evidence keep only what
the code does:

- **Imports** are import statements anywhere in a file (inside functions too),
  ``pytest_plugins`` entries, literal ``import_module``/``__import__``/
  ``run_module`` names, and imports inside embedded Python snippets
  (``python -c``, a script a test writes).
- **Paths** are values ``tessera._dev.source_dependencies`` resolves exactly
  (``Path(__file__).parents[1] / "tools" / "x.py"``, string constants, names
  bound to either) in call arguments, plus string constants that spell a
  repository path. A path names a file; it is an import only when the file
  is Python that something runs: a test naming a non-test module, or a
  production module that launches or loads a file by path.
- **Directories** a file enumerates (``glob``, ``iterdir``, ``os.walk``) are
  reads of the whole directory.

Nothing here executes the code it reads.
"""
from __future__ import annotations

import ast
from collections import defaultdict
import fnmatch
import posixpath
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from tessera._dev import source_dependencies as sd  # noqa: E402

# Bazel packages that hold modules. Every other file belongs to the root
# package. tools/bazel is written by hand.
PACKAGES = ('src/tessera', 'tools', 'tests', 'experiments')
HANDWRITTEN = ('tools/bazel/',)
SKIPPED = ('.git/', 'bazel-')
IMPORT_LINE = re.compile(r'^\s*(import\s+[A-Za-z_][\w.]*|from\s+[A-Za-z_.][\w.]*\s+import\s+[\w*(])')
NAME_IMPORTERS = frozenset({'import_module', '__import__', 'run_module'})
MODULE_NAME = re.compile(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*')
SCRIPT_IMPORT = re.compile(r'(?:^|[\s;"\'(])(?:from\s+([A-Za-z_][\w.]*)\s+import\s+([\w, ]+)|import\s+([A-Za-z_][\w., ]*))')
SCRIPT_PATH = re.compile(r'[\w./-]*[\w-]\.(?:sh|py|json|txt|toml)\b')
MODULE_FILE = re.compile(r'[\w][\w.+-]*')
# Calls that start a program or load a file as code, by their last name part.
LAUNCHERS = frozenset({
    'run', 'Popen', 'call', 'check_call', 'check_output', 'getoutput', 'getstatusoutput',
    'execv', 'execve', 'execvp', 'execvpe', 'execl', 'execle', 'execlp', 'execlpe',
    'spawnv', 'spawnve', 'spawnvp', 'spawnvpe', 'posix_spawn', 'posix_spawnp', 'system',
    'run_path', 'spec_from_file_location', 'SourceFileLoader', 'create_subprocess_exec',
    'create_subprocess_shell'})
LAUNCH_MODULES = frozenset({'subprocess', 'os', 'runpy', 'importlib', 'asyncio'})


_WALKED = {}


def walk(tree):
    """ast.walk, once per tree: every reader below visits the same nodes."""
    key = id(tree)
    if key not in _WALKED or _WALKED[key][0] is not tree:
        _WALKED.clear()
        _WALKED[key] = (tree, list(ast.walk(tree)))
    return _WALKED[key][1]


def repository_files(root=ROOT):
    """Tracked and untracked, unignored files: a new test counts before `git add`."""
    output = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                            cwd=root, capture_output=True, check=True).stdout.decode()
    return sorted(path for path in set(output.split('\0')) if path and not path.startswith(SKIPPED)
                  and PurePosixPath(path).name != 'BUILD.bazel'
                  and (root / path).is_file() and not (root / path).is_symlink())


def package_of(path):
    if path.startswith(HANDWRITTEN):
        return HANDWRITTEN[0].rstrip('/')
    for package in PACKAGES:
        if path.startswith(package + '/'):
            return package
    return ''


def is_module(path):
    return (path.endswith('.py') and not path.startswith(HANDWRITTEN)
            and path.startswith(tuple(package + '/' for package in PACKAGES)))


def is_test(path):
    return path.startswith('tests/') and PurePosixPath(path).name.startswith('test_') and path.endswith('.py')


def is_test_side(path):
    return path.startswith('tests/')


def module_aliases(path):
    """The names a module is imported by.

    src/tessera/a/b.py is tessera.a.b. A tests/, tools/ or experiments/ module
    is also every dotted suffix of its path: tests/conftest.py puts tests/ on
    sys.path, the repository root is on it under `python -m pytest`, and tests
    put experiment directories on it, so experiments/graph_attest_702/
    rank_window.py is also rank_window.
    """
    parts = path.removesuffix('.py').split('/')
    if parts[-1] == '__init__':
        parts = parts[:-1]
    if path.startswith('src/'):
        return {'.'.join(parts[1:])}
    return {'.'.join(parts[start:]) for start in range(len(parts))}


def parse_snippets(text):
    """Python embedded in a string: a `-c` argument or a script a test writes."""
    try:
        return [ast.parse(textwrap.dedent(text))]
    except (SyntaxError, ValueError):
        pass
    found = []
    for line in text.splitlines():
        if IMPORT_LINE.match(line):
            try:
                found.append(ast.parse(line.split(';', 1)[0].strip()))
            except (SyntaxError, ValueError):
                continue
    return found


def statement_imports(tree, path):
    """Every module an import statement or pytest_plugins entry names."""
    names = set()
    for node in walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = node.module or ''
            if node.level:
                package = list(PurePosixPath(path.removeprefix('src/')).parent.parts)
                package = package[:len(package) - node.level + 1]
                prefix = '.'.join([*package, *([prefix] if prefix else [])])
            if prefix:
                names.add(prefix)
            names.update(f'{prefix}.{alias.name}' if prefix else alias.name for alias in node.names)
        elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'pytest_plugins'
                                                  for target in node.targets):
            names.update(value.value for value in ast.walk(node.value)
                         if isinstance(value, ast.Constant) and isinstance(value.value, str))
    return names


def callee(call):
    function = call.func
    if isinstance(function, ast.Attribute):
        return function.attr
    if isinstance(function, ast.Name):
        return function.id
    return ''


def named_imports(tree):
    """Literal names passed to import_module, __import__ or runpy.run_module."""
    return {call.args[0].value for call in walk(tree)
            if isinstance(call, ast.Call) and callee(call) in NAME_IMPORTERS and call.args
            and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)
            and MODULE_NAME.fullmatch(call.args[0].value)}


def launches(tree):
    """Whether this file starts a program or loads a file as code."""
    imported = {alias.name.split('.')[0] for node in walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or '')])}
    return bool(imported & LAUNCH_MODULES) and any(
        isinstance(call, ast.Call) and callee(call) in LAUNCHERS for call in walk(tree))


def _leaves(expression):
    """Sub-expressions that may evaluate to a path, below containers."""
    if isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
        for element in expression.elts:
            yield from _leaves(element)
    elif isinstance(expression, ast.Starred):
        yield from _leaves(expression.value)
    elif isinstance(expression, ast.Dict):
        for value in [*expression.keys, *expression.values]:
            if value is not None:
                yield from _leaves(value)
    elif isinstance(expression, ast.Call) and callee(expression) in {'str', 'fspath', 'Path', 'resolve'}:
        yield expression
        for argument in expression.args:
            yield from _leaves(argument)
    else:
        yield expression


def package_data(root, files):
    """{package __init__.py: files} from pyproject.toml's package-data table.

    Importing a package makes its package data readable, as in an installed
    wheel: the runtime contract and the CUDA sources a loader resolves through
    importlib.resources.
    """
    import tomllib
    table = tomllib.loads((root / 'pyproject.toml').read_text()).get('tool', {}).get('setuptools', {})
    found = defaultdict(set)
    for package, patterns in table.get('package-data', {}).items():
        directory = 'src/' + package.replace('.', '/')
        for pattern in patterns:
            found[directory + '/__init__.py'] |= {file for file in files if file.startswith(directory + '/')
                                                  and fnmatch.fnmatchcase(file[len(directory) + 1:], pattern)}
    return found


@dataclass
class Facts:
    names: set = field(default_factory=set)        # module names imported
    paths: set = field(default_factory=set)        # repository files named exactly
    directories: set = field(default_factory=set)  # repository directories read whole
    patterns: set = field(default_factory=set)     # repository files a built path may name
    bases: set = field(default_factory=set)        # repository directories paths are built on
    launches: bool = False


class Evidence:
    """Every module's facts, resolved against the repository's files."""

    def __init__(self, root=ROOT, files=None):
        self.root = Path(root)
        self.files = repository_files(self.root) if files is None else files
        self.file_set = set(self.files)
        self.directories = {'/'.join(parts[:end]) + '/' for parts in (file.split('/') for file in self.files)
                            for end in range(1, len(parts))}
        self.modules = sorted(path for path in self.files if is_module(path))
        self.aliases = defaultdict(set)
        for path in self.modules:
            for alias in module_aliases(path):
                self.aliases[alias].add(path)
        self.facts = {path: self.read(path) for path in self.modules}
        self.scripts = {path: self.read_script(path) for path in self.files if path.endswith('.sh')}
        self.package_data = package_data(self.root, self.files)
        self.package_imports = {init: set().union(set(), *(self.named_in_json(file) for file in files))
                                for init, files in self.package_data.items()}

    def place(self, value, path):
        """A repository-relative file or directory for one resolved value, or None."""
        if isinstance(value, Path):
            candidate = value
        elif isinstance(value, str) and value and '\n' not in value and len(value) < 300:
            candidate = Path(value)
        else:
            return None
        try:
            if candidate.is_absolute():
                relative = candidate.relative_to(self.root)
            else:
                relative = candidate
            text = PurePosixPath(relative.as_posix()).as_posix()
        except ValueError:
            return None
        if '..' in PurePosixPath(text).parts:
            try:
                text = (self.root / text).resolve().relative_to(self.root.resolve()).as_posix()
            except ValueError:
                return None
        if text in ('', '.'):
            return None
        if text in self.file_set:
            return text
        try:
            # Prose can reach here; a name too long for the filesystem names nothing.
            directory = (self.root / text).is_dir()
        except OSError:
            return None
        return text.rstrip('/') + '/' if directory and not text.startswith(SKIPPED) else None

    def read(self, path):
        source = (self.root / path).read_text()
        tree = ast.parse(source)
        nodes = walk(tree)
        trees = [tree] + [snippet for node in nodes
                          if isinstance(node, ast.Constant) and isinstance(node.value, str) and 'import ' in node.value
                          for snippet in parse_snippets(node.value)]
        facts = Facts(launches=launches(tree))
        for each in trees:
            facts.names |= statement_imports(each, path) | named_imports(each)
        facts.names |= self.tabled_imports(tree, path)
        absolute = self.root / path
        scanner = sd._Scanner(absolute)
        scanner.visit(tree)
        for call, scope in scanner.calls:
            arguments = [*call.args, *(keyword.value for keyword in call.keywords)]
            if isinstance(call.func, ast.Attribute):
                arguments.append(call.func.value)
            for argument in arguments:
                for leaf in _leaves(argument):
                    if isinstance(leaf, ast.Constant) and not isinstance(leaf.value, str):
                        continue
                    try:
                        values = sd._values(leaf, scope, self.root)
                    except (OSError, ValueError, TypeError, RecursionError):
                        values = None
                    for value in values or ():
                        self.note(facts, value, path, directories=False)
        # Enumerated directories: the base of a glob, iterdir, listdir, scandir
        # or walk, resolved exactly.
        for call, scope in scanner.calls:
            if callee(call) not in sd._ENUMERATIONS:
                continue
            if isinstance(call.func, ast.Attribute) and callee(call) in ('glob', 'rglob', 'iterdir') \
                    and not ast.unparse(call.func.value) == 'glob':
                base = call.func.value
            elif call.args:
                base = call.args[0]
            else:
                continue
            try:
                values = sd._values(base, scope, self.root)
            except (OSError, ValueError, TypeError, RecursionError):
                values = None
            for value in values or ():
                if isinstance(value, str) and any(char in value for char in '*?['):
                    value = value.split('*', 1)[0].rsplit('/', 1)[0] or '.'
                self.note(facts, value, path)
        here = posixpath.dirname(path)
        for node in nodes:
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)) or len(node.value) > 200:
                continue
            if '/' in node.value:
                self.note(facts, node.value, path, directories=False)
            # A manifest of the module's own files, relative to its directory:
            # PRODUCER_FILES = ("tp2_recipe.py", "observer/usercustomize.py").
            if MODULE_FILE.fullmatch(node.value.replace('/', '')) and '.' in node.value:
                sibling = posixpath.normpath(posixpath.join(here, node.value))
                if sibling in self.file_set:
                    facts.paths.add(sibling)
        self._bases = facts.bases
        facts.patterns = (self.built_paths(tree, scanner) | self.siblings(tree)
                          | self.tabled(tree, facts.bases)) - facts.paths
        facts.paths.discard(path)
        facts.patterns.discard(path)
        return facts

    def built_paths(self, tree, scanner):
        """Repository files a path built from partly unknown parts may name.

        `RESULTS / f"stub_{name}_census.json"` or `LEGACY / name` names a file
        this resolver can't pin down; read as a pattern, with `*` for each
        unknown part, it names the files it could be. A pattern needs a
        resolved base inside the repository, and a literal in its file name or
        a directory below the top level. `os.path.join` and `joinpath` chains
        count as `/` chains, and a chain that resolves fully is an exact file.
        """
        scopes = {id(node): scope for node, scope in scanner.references}
        parents = {child: node for node in walk(tree) for child in ast.iter_child_nodes(node)}
        found, bases = set(), self._bases

        def text(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            if isinstance(node, ast.JoinedStr):
                return ''.join(value.value if isinstance(value, ast.Constant) else '*' for value in node.values)
            return None

        chains = []
        for node in walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                if isinstance(parents.get(node), ast.BinOp) and isinstance(parents[node].op, ast.Div):
                    continue
                operands = []
                while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                    operands.append(node.right)
                    node = node.left
                chains.append((node, operands[::-1]))
            elif isinstance(node, ast.Call) and node.args and not node.keywords:
                if ast.unparse(node.func).endswith('path.join'):
                    chains.append((node.args[0], node.args[1:]))
                elif callee(node) == 'joinpath' and isinstance(node.func, ast.Attribute):
                    chains.append((node.func.value, node.args))
        for base, rest in chains:
            scope = next((scopes[id(name)] for name in ast.walk(base) if id(name) in scopes), scanner.scope)
            try:
                values = sd._values(base, scope, self.root)
            except (OSError, ValueError, TypeError, RecursionError):
                values = None
            if not values or not all(isinstance(value, (Path, str)) for value in values):
                continue
            parts = [text(operand) for operand in rest]
            tail = '/'.join(part if part is not None else '*' for part in parts)
            for value in values:
                value = Path(value)
                if not value.is_absolute():
                    continue
                try:
                    base_text = value.relative_to(self.root).as_posix()
                except ValueError:
                    continue
                if base_text not in ('', '.') and base_text + '/' in self.directories:
                    bases.add(base_text + '/')
                pattern = posixpath.normpath(posixpath.join(base_text, tail)) if tail else base_text
                if '*' not in pattern:
                    if pattern in self.file_set:
                        found.add(pattern)  # Exact: a path no call argument carried.
                    continue
                # A pattern must say something about the file: a literal in its
                # name (`f"stub_{x}_census.json"`), or a directory specific
                # enough to be a table's home (`serving / name`).
                # `ROOT / "tests" / name` names nothing in particular.
                directory, _, filename = pattern.rpartition('/')
                if not filename.strip('*?') and directory.count('/') < 1:
                    continue
                if pattern.startswith('..') or pattern in ('.', '*'):
                    continue
                found |= self.matching(pattern)
        return found

    def tabled_imports(self, tree, path):
        """Module names a file keeps in strings when it imports by a name it
        computes (`import_module(builder[0])`): scheme.py's route builders, a
        test's `_INTAKE = "test_x"`. In production code only full `tessera.`
        names count; a bare word is too often not a module."""
        dynamic = any(isinstance(call, ast.Call) and callee(call) in NAME_IMPORTERS and call.args
                      and not isinstance(call.args[0], ast.Constant) for call in walk(tree))
        if is_test_side(path) and not dynamic:
            return set()
        # Production tables and their importers are often different modules
        # (scheme.py's builders, config.py's import_module), so there a string
        # that is exactly a tessera module's full name counts wherever it is.
        return {node.value for node in walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
                and MODULE_NAME.fullmatch(node.value) and node.value in self.aliases
                and (dynamic if is_test_side(path) else node.value.startswith('tessera.'))}

    def tabled(self, tree, bases):
        """Files a table names by bare file name, under a directory this file
        builds paths on: `RESULTS / name` over `{"x": "a_census.json"}`. A bare
        name also names that module, `load("sass_dump")` its .py file."""
        names = {node.value for node in walk(tree) if isinstance(node, ast.Constant)
                 and isinstance(node.value, str) and node.value and '/' not in node.value
                 and len(node.value) < 120 and MODULE_FILE.fullmatch(node.value)}
        found = set()
        for base in bases:
            if base.rstrip('/').count('/') < 1 and base.rstrip('/') in ('', 'src', 'tests', 'tools', 'experiments', 'docs'):
                continue  # A top-level directory is an import root or a tree, not a table's base.
            for name in names:
                for candidate in (base + name, base + name + '.py'):
                    if candidate in self.file_set:
                        found.add(candidate)
        return found

    def siblings(self, tree):
        """Files named by `.with_name("x.py")` on a path this resolver can't
        place (`Path(module.__file__)`): every repository file with that name."""
        names = {call.args[0].value for call in walk(tree)
                 if isinstance(call, ast.Call) and callee(call) == 'with_name' and call.args
                 and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)}
        return {file for file in self.files if PurePosixPath(file).name in names} if names else set()

    def matching(self, pattern):
        """Repository files a segment-wise glob names."""
        wanted = pattern.split('/')
        literal = '/'.join(part for part in wanted if '*' not in part and '?' not in part)
        return {file for file in self.files if len(file.split('/')) == len(wanted)
                and all(fnmatch.fnmatchcase(have, want) for have, want in zip(file.split('/'), wanted))} \
            if literal else set()

    def note(self, facts, value, path, directories=True):
        """Record one resolved value. A directory counts only where it is
        enumerated: elsewhere it is an import root or a base a path is built on."""
        placed = self.place(value, path)
        if placed is None:
            return
        if not placed.endswith('/'):
            facts.paths.add(placed)
        else:
            facts.bases.add(placed)
            if directories:
                facts.directories.add(placed)

    def read_script(self, path):
        """(modules, files) a shell script needs: the modules its embedded
        Python imports, and the repository files it names, relative to its own
        directory or the repository root, by full path or bare name."""
        text = (self.root / path).read_text(errors='replace')
        names = set()
        for match in SCRIPT_IMPORT.finditer(text):
            module, members, imported = match[1], match[2], match[3]
            if module:
                names.add(module)
                names |= {f'{module}.{name.strip()}' for name in members.split(',') if name.strip()}
            else:
                names |= {name.strip().split(' ')[0] for name in imported.split(',') if name.strip()}
        modules = set().union(set(), *(self.imported(name) for name in names))
        here = posixpath.dirname(path)
        files = set()
        for token in SCRIPT_PATH.findall(text):
            # `${VAR:-path}` leaves its `-` on the token; `./x` is x.
            token = token.lstrip('-').removeprefix('./').strip('/')
            for candidate in (token, posixpath.normpath(posixpath.join(here, token))):
                if candidate in self.file_set:
                    files.add(candidate)
            if '/' not in token:
                files |= {file for file in self.files if PurePosixPath(file).name == token
                          and file.startswith(here + '/')}
        files.discard(path)
        modules |= {file for file in files if is_module(file)}
        return modules, files - modules

    def named_in_json(self, file):
        """Modules a packaged JSON table names in full: runtime_contract.json's
        loaders, which tessera.serving.contract finds with find_spec."""
        if not file.endswith('.json'):
            return set()
        found = set()
        for value in re.findall(r'"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)"', (self.root / file).read_text()):
            if value.split('.')[0] == 'tessera':
                found |= {module for module in self.aliases.get(value, ()) if module.startswith('src/')}
        return found

    def imported(self, name):
        parts = name.split('.')
        found = set()
        for end in range(1, len(parts) + 1):
            found.update(self.aliases.get('.'.join(parts[:end]), ()))
        return found

    def edges(self, path):
        """Modules `path` needs as libraries: what it imports, and for a test
        module the modules it names by path (it runs or loads them)."""
        facts = self.facts[path]
        modules = set().union(*(self.imported(name) for name in facts.names)) if facts.names else set()
        modules |= self.package_imports.get(path, set())
        if is_test_side(path):
            modules |= {file for file in facts.paths if is_module(file)}
            if facts.launches:
                modules |= {file for file in facts.patterns if is_module(file)}
        modules.discard(path)
        return modules

    def programs(self, path):
        """Modules a production module launches or loads by path. They are
        inputs of the tests that reach it, never library edges: on a library
        they would close cycles across packages."""
        facts = self.facts[path]
        if is_test_side(path) or not facts.launches:
            return set()
        return {file for file in facts.paths | facts.patterns if is_module(file)} - {path}

    def data(self, path):
        """Repository files `path` reads: every exact file it names that is
        not one of its imports, and for tests every directory it enumerates
        and every file a built path may name."""
        facts = self.facts[path]
        imports = self.edges(path) | self.programs(path)
        files = {file for file in facts.paths if file not in imports}
        files |= self.package_data.get(path, set())
        # conftest enumerates import roots for their names (which top-level
        # names are this tree's own); the files under them are not inputs.
        if is_test_side(path) and PurePosixPath(path).name != 'conftest.py':
            files |= {file for file in facts.patterns if file not in imports}
            for directory in facts.directories:
                files |= {file for file in self.files if file.startswith(directory)}
        return files

    def tree_reads(self, path):
        """Files under the directories a production module enumerates, and the
        files its built paths may name."""
        facts = self.facts[path]
        found = {file for file in facts.patterns if file not in self.edges(path) | self.programs(path)}
        if facts.directories:
            found |= {file for file in self.files if file.startswith(tuple(facts.directories))}
        return found
