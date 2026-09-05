"""Conservative, non-executing discovery of Python file-consumer dependencies.

The supported expressions are finite Path constructions, not arbitrary Python.
A resolved target inside the tree is an exact edge whatever its suffix: a
``.py`` file is a module dependency, anything else is the data dependency it
is.  A resolved target outside the tree is neither -- this repository's diff
cannot change it.  An unresolved target is a wildcard edge instead of no edge.

**An unresolved target is a wildcard over Python only when the reader can run
Python.**  A loader always can, by definition.  A plain read cannot: bytes are
a Python dependency only once something parses or executes them, so a module
that never does is reading data, and calling its unnameable path "any module
in the tree" made the whole tree depend on the whole tree.  ``_SOURCE_*`` is
the recognition set for that -- the standard library's source-execution API,
which this repository does not own and cannot derive from its own code, and
which is deliberately matched by resolved symbol rather than by bare
attribute name (``re.compile`` and ``model.eval()`` are not source
execution).  What it misses is a source read this module never sees at all,
``subprocess.run([sys.executable, path])`` above all; that was never an edge
here and this does not change it.
"""
from __future__ import annotations

import ast
import os.path
from collections import defaultdict
from pathlib import Path

WILDCARD = "*"
_LOADERS = {"spec_from_file_location": (1, "location"),
            "SourceFileLoader": (1, "path"), "run_path": (0, "path_name")}
_SYMBOLS = {"spec_from_file_location": "importlib.util.spec_from_file_location",
            "SourceFileLoader": "importlib.machinery.SourceFileLoader",
            "run_path": "runpy.run_path"}
_READ_METHODS = {"read_text", "read_bytes", "open"}
_KINDS = set(_LOADERS) | _READ_METHODS


class _Scope:
    def __init__(self, parent=None, *, class_body=False):
        self.parent = parent
        self.class_body = class_body
        self.bindings = defaultdict(list)

    def bind(self, target, value):
        for node in ast.walk(target):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                self.bindings[node.id].append(value)


class _Scanner(ast.NodeVisitor):
    def __init__(self, path):
        self.scope = _Scope()
        self.scope.bindings["__file__"].append(ast.Constant(str(path)))
        self.calls = []

    def visit_Import(self, node):
        for alias in node.names:
            self.scope.bindings[alias.asname or alias.name.split(".")[0]].append(
                ("symbol", alias.name if alias.asname else alias.name.split(".")[0]))

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.scope.bindings[alias.asname or alias.name].append(
                ("symbol", f"{node.module}.{alias.name}") if not node.level else None)

    def visit_Assign(self, node):
        for target in node.targets:
            self.scope.bind(target, node.value)
        self.visit(node.value)

    def visit_AnnAssign(self, node):
        self.scope.bind(node.target, node.value)
        self.visit(node.annotation)
        if node.value:
            self.visit(node.value)

    def visit_AugAssign(self, node):
        self.scope.bind(node.target, None)
        self.visit(node.value)

    def visit_NamedExpr(self, node):
        self.scope.bind(node.target, node.value)
        self.visit(node.value)

    def visit_For(self, node):
        self.scope.bind(node.target, node.iter if isinstance(node.target, ast.Name) else None)
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars:
                self.scope.bind(item.optional_vars, None)
        self.generic_visit(node)

    visit_AsyncWith = visit_With

    def visit_ExceptHandler(self, node):
        if node.name:
            self.scope.bindings[node.name].append(None)
        self.generic_visit(node)

    def visit_Global(self, node):
        for name in node.names:
            self.scope.bindings[name].append(None)
            parent = self.scope.parent
            while parent:
                if name in parent.bindings or parent.parent is None:
                    parent.bindings[name].append(None)
                parent = parent.parent

    visit_Nonlocal = visit_Global

    def visit_Delete(self, node):
        for target in node.targets:
            for name in ast.walk(target):
                if isinstance(name, ast.Name):
                    self.scope.bindings[name.id].append(None)

    def visit_match_case(self, node):
        for pattern in ast.walk(node.pattern):
            if isinstance(pattern, (ast.MatchAs, ast.MatchStar)) and pattern.name:
                self.scope.bindings[pattern.name].append(None)
            elif isinstance(pattern, ast.MatchMapping) and pattern.rest:
                self.scope.bindings[pattern.rest].append(None)
        self.generic_visit(node)

    def _nested(self, node, *, class_body=False):
        prior = self.scope
        # Decorators, defaults and class bases execute in the enclosing scope.
        for expression in getattr(node, "decorator_list", []):
            self.visit(expression)
        if hasattr(node, "args"):
            for expression in node.args.defaults + node.args.kw_defaults:
                if expression:
                    self.visit(expression)
            # Annotations are potential dependencies even when evaluation is
            # deferred. Value parameters do not bind in their annotation's
            # defining scope; generic type parameters may shadow outer names.
            annotation_scope = _Scope(prior)
            for parameter in getattr(node, "type_params", []):
                annotation_scope.bindings[parameter.name].append(None)
            self.scope = annotation_scope
            parameters = (node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                          + [node.args.vararg, node.args.kwarg])
            for parameter in parameters:
                if parameter is not None and parameter.annotation is not None:
                    self.visit(parameter.annotation)
            if getattr(node, "returns", None) is not None:
                self.visit(node.returns)
            self.scope = prior
        for expression in getattr(node, "bases", []):
            self.visit(expression)
        for keyword in getattr(node, "keywords", []):
            self.visit(keyword.value)
        if hasattr(node, "name"):
            prior.bindings[node.name].append(None)
        parent = prior.parent if prior.class_body and not class_body else prior
        self.scope = _Scope(parent, class_body=class_body)
        if hasattr(node, "args"):
            for arg in ast.walk(node.args):
                if isinstance(arg, ast.arg):
                    self.scope.bindings[arg.arg].append(None)
        body = node.body if isinstance(node.body, list) else [node.body]
        for statement in body:
            self.visit(statement)
        self.scope = prior

    def visit_FunctionDef(self, node):
        self._nested(node)

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef

    def visit_ClassDef(self, node):
        self._nested(node, class_body=True)

    def visit_Call(self, node):
        self.calls.append((node, self.scope))
        self.generic_visit(node)

    def visit_ListComp(self, node):
        prior = self.scope
        self.scope = _Scope(prior)
        for generator in node.generators:
            self.scope.bind(generator.target, None)
        self.generic_visit(node)
        self.scope = prior

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp


#: Calls that turn bytes into running Python.  Bare names only for the
#: builtins -- ``model.eval()`` and ``re.compile()`` are attributes and are not
#: this.  Attribute names only where the name itself is the API.
_SOURCE_BUILTINS = {"exec", "eval", "compile", "execfile", "__import__"}
_SOURCE_ATTRIBUTES = {"run_path", "run_module", "spec_from_file_location",
                      "SourceFileLoader", "SourcelessFileLoader", "exec_module",
                      "source_to_code", "get_code", "compile_command"}
#: ``module: attribute`` pairs whose attribute is too common to match alone.
_SOURCE_QUALIFIED = {"ast": {"parse"}, "py_compile": {"compile"}}


def _executes_python_source(tree):
    """Whether this module can turn file bytes into Python it runs or parses."""
    direct, modules = set(), {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _SOURCE_QUALIFIED:
                    modules[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and not node.level:
            owned = _SOURCE_QUALIFIED.get(node.module or "", set())
            for alias in node.names:
                if alias.name in owned or alias.name in _SOURCE_ATTRIBUTES:
                    direct.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name):
            if function.id in _SOURCE_BUILTINS or function.id in direct:
                return True
        elif isinstance(function, ast.Attribute):
            if function.attr in _SOURCE_ATTRIBUTES:
                return True
            owner = function.value
            if (isinstance(owner, ast.Name)
                    and function.attr in _SOURCE_QUALIFIED.get(
                        modules.get(owner.id, ""), set())):
                return True
    return False


def _lexically_under_root(path, root):
    """Whether *path* denotes a location under *root* -- string work only.

    ``os.path.normpath`` collapses ``.``/``..`` segments the same way
    ``resolve()`` would, without a filesystem call, so this decides the
    question ``resolve()`` decides -- modulo a symlink -- for free.  An
    absolute literal outside the repository, or a relative literal that
    escapes via ``..``, reads false here and is never resolved or stat'ed
    to confirm it: the same rule that keeps a glob from crawling outside
    this root at :314-316 keeps a bare literal from being resolved outside
    it. A path that reads true here may still be resolved, but that
    resolve() only ever walks locations already established as in-root.
    """
    absolute = path if path.is_absolute() else root / path
    normalized = Path(os.path.normpath(str(absolute)))
    root_normalized = Path(os.path.normpath(str(root)))
    return normalized == root_normalized or root_normalized in normalized.parents


def _values(node, scope, root, visiting=frozenset()):
    """All statically established values; None means some alternative is unknown."""
    if isinstance(node, tuple):
        return {node}
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
        return {node.value}
    if isinstance(node, ast.Name):
        here = scope
        while here and node.id not in here.bindings:
            here = here.parent
        if here is None:
            return {("symbol", "builtins." + node.id)} if node.id in {
                "str", "sorted", "list", "tuple", "set", "open"} else None
        key = (id(here), node.id)
        if key in visiting:
            return None
        result = set()
        for expression in here.bindings[node.id]:
            value = _values(expression, here, root, visiting | {key})
            if value is None:
                return None
            result.update(value)
        return result
    if isinstance(node, ast.Attribute):
        values = _values(node.value, scope, root, visiting)
        if values is None:
            return None
        result = set()
        for value in values:
            if isinstance(value, tuple) and value[0] == "symbol":
                result.add(("symbol", value[1] + "." + node.attr))
            elif isinstance(value, Path) and node.attr == "parent":
                result.add(value.parent)
            elif isinstance(value, Path) and node.attr in _READ_METHODS:
                result.add(("file_reader", value))
            else:
                return None
        return result
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "parents":
        paths = _values(node.value.value, scope, root, visiting)
        indices = _values(node.slice, scope, root, visiting)
        if paths is not None and indices is not None:
            if not all(isinstance(path, Path) for path in paths) or not all(
                    isinstance(index, int) for index in indices):
                return None
            try:
                return {path.parents[index] for path in paths for index in indices}
            except IndexError:
                return None
    if isinstance(node, ast.BinOp):
        left = _values(node.left, scope, root, visiting)
        right = _values(node.right, scope, root, visiting)
        if left is None or right is None:
            return None
        result = set()
        for a in left:
            for b in right:
                if isinstance(node.op, ast.Div) and isinstance(a, Path) and isinstance(b, (str, Path)):
                    result.add(a / b)
                elif isinstance(node.op, ast.Add) and isinstance(a, str) and isinstance(b, str):
                    result.add(a + b)
                else:
                    return None
        return result
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"resolve", "glob", "rglob", "joinpath"}:
            paths = _values(node.func.value, scope, root, visiting)
            if paths is None or not all(isinstance(path, Path) for path in paths):
                return None
            if node.func.attr == "resolve" and not node.args and not node.keywords:
                # A literal ``.resolve()`` on a path outside root is the
                # same escape a crawling glob would be: unknown, and never
                # stat'ed to find out (rule 4; was :308).
                if not all(_lexically_under_root(path, root) for path in paths):
                    return None
                return {(path if path.is_absolute() else root / path).resolve() for path in paths}
            if len(node.args) == 1 and not node.keywords:
                args = _values(node.args[0], scope, root, visiting)
                if args is None or not all(isinstance(arg, str) for arg in args):
                    return None
                if node.func.attr == "joinpath":
                    return {path / arg for path in paths for arg in args}
                # Only one-directory globs are finite within the checked tree.
                # Recursive/escaping patterns remain an unknown dependency;
                # they must not trigger a filesystem crawl outside this root.
                # Nor may the base itself be resolved from outside root --
                # an absolute literal base is the identical escape (rule 4;
                # was :318), refused lexically before any stat.
                if not all(_lexically_under_root(path, root) for path in paths):
                    return None
                bases = {path.resolve() for path in paths}
                if (node.func.attr == "glob"
                        and all(base.is_relative_to(root) for base in bases)
                        and all(len(Path(arg).parts) == 1 and arg not in {".", ".."}
                                and "**" not in arg for arg in args)):
                    return {item for base in bases for arg in args
                            for item in base.glob(arg)}
            return None
        functions = _values(node.func, scope, root, visiting)
        if functions is None or len(node.args) != 1 or node.keywords:
            return None
        args = _values(node.args[0], scope, root, visiting)
        if args is None:
            return None
        result = set()
        for function in functions:
            if function == ("symbol", "pathlib.Path") and all(isinstance(arg, (str, Path)) for arg in args):
                result.update(Path(arg) for arg in args)
            elif function == ("symbol", "builtins.str"):
                result.update(str(arg) for arg in args)
            elif function in {("symbol", "builtins." + name) for name in ("sorted", "list", "tuple", "set")}:
                result.update(args)
            else:
                return None
        return result
    return None


def file_imports(tree, path, root):
    """Return repository-relative .py dependencies and an unknown-loader flag."""
    scanner = _Scanner(path)
    scanner.visit(tree)
    aliases = {name: {name} for name in _KINDS}
    assignments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _KINDS:
                    aliases.setdefault(alias.asname or alias.name, set()).add(alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            assignments.append(node)
    def kind(expression):
        if isinstance(expression, ast.Attribute) and expression.attr in _KINDS:
            return {expression.attr}
        if isinstance(expression, ast.Name):
            return aliases.get(expression.id, set())
        return set()
    # Aliases are an over-approximation for recognition only. Lexical value
    # resolution below still refuses parameter/reassignment uncertainty.
    while True:
        before = {name: set(kinds) for name, kinds in aliases.items()}
        for assignment in assignments:
            loader = kind(assignment.value)
            if loader:
                targets = assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        aliases.setdefault(target.id, set()).update(loader)
        if aliases == before:
            break
    executes = _executes_python_source(tree)

    def wildcard(reading):
        """An unnameable target is an unknown *module* only if it can run."""
        return executes or not reading

    found, unknown = set(), False
    for call, scope in scanner.calls:
        loaders = kind(call.func)
        if not loaders:
            continue
        reading = loaders <= _READ_METHODS
        if len(loaders) != 1:
            unknown = unknown or wildcard(reading)
            continue
        loader = next(iter(loaders))
        try:
            functions = _values(call.func, scope, root)
            if loader in _READ_METHODS:
                # Reading source bytes is already a dependency, whether the
                # consumer later ast.parse/execs them or asserts on the text.
                # No execution/data-flow guess or hardcoded consumer roster.
                if functions is not None and all(
                        isinstance(function, tuple) and function[0] == "file_reader"
                        for function in functions):
                    values = {function[1] for function in functions}
                elif loader == "open" and functions is not None and functions <= {
                        ("symbol", "builtins.open"), ("symbol", "io.open")}:
                    expression = call.args[0] if call.args else next(
                        (arg.value for arg in call.keywords if arg.arg == "file"), None)
                    values = _values(expression, scope, root)
                else:
                    values = None
            else:
                position, keyword = _LOADERS[loader]
                expression = call.args[position] if len(call.args) > position else next(
                    (arg.value for arg in call.keywords if arg.arg == keyword), None)
                if functions != {("symbol", _SYMBOLS[loader])}:
                    unknown = True
                values = _values(expression, scope, root)
        except (OSError, ValueError, TypeError, RecursionError):
            values = None
        if values is None or not all(isinstance(value, (str, Path)) for value in values):
            unknown = unknown or wildcard(reading)
            continue
        for value in values:
            try:
                target = Path(value)
            except (OSError, ValueError):
                unknown = unknown or wildcard(reading)
                continue
            if not _lexically_under_root(target, root):
                # An absolute (or ``..``-escaping) literal outside root is
                # the same escape the glob and resolve() guards above
                # refuse: unknown rather than resolved, so a stalled mount
                # under the literal's real location never blocks the
                # selector (rule 4; was :426). This is the one case where
                # "outside the tree" now reads unknown instead of the
                # silent drop below, because we no longer pay a resolve()
                # to tell "genuinely outside" from "a symlink away".
                unknown = unknown or wildcard(reading)
                continue
            try:
                target = (target if target.is_absolute() else root / target).resolve()
            except (OSError, ValueError):
                unknown = unknown or wildcard(reading)
                continue
            # Named and inside the tree: an exact edge, ``.py`` or not. Named
            # and outside it: nothing this repository's diff can move, so it
            # is neither an edge nor an unknown.
            if target.is_relative_to(root):
                found.add(target)
    return found, unknown
