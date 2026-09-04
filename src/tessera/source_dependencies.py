"""Conservative, non-executing discovery of Python file-loader dependencies.

The supported expressions are finite Path constructions, not arbitrary Python.
An unresolved recognized loader returns a wildcard edge instead of no edge.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

WILDCARD = "*"
_LOADERS = {"spec_from_file_location": (1, "location"),
            "SourceFileLoader": (1, "path"), "run_path": (0, "path_name")}
_SYMBOLS = {"spec_from_file_location": "importlib.util.spec_from_file_location",
            "SourceFileLoader": "importlib.machinery.SourceFileLoader",
            "run_path": "runpy.run_path"}


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
                "str", "sorted", "list", "tuple", "set"} else None
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
    aliases = {name: {name} for name in _LOADERS}
    assignments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _LOADERS:
                    aliases.setdefault(alias.asname or alias.name, set()).add(alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            assignments.append(node)
    def kind(expression):
        if isinstance(expression, ast.Attribute) and expression.attr in _LOADERS:
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
    found, unknown = set(), False
    for call, scope in scanner.calls:
        loaders = kind(call.func)
        if not loaders:
            continue
        if len(loaders) != 1:
            unknown = True
            continue
        loader = next(iter(loaders))
        position, keyword = _LOADERS[loader]
        expression = call.args[position] if len(call.args) > position else next(
            (arg.value for arg in call.keywords if arg.arg == keyword), None)
        try:
            functions = _values(call.func, scope, root)
            if functions != {("symbol", _SYMBOLS[loader])}:
                unknown = True
            values = _values(expression, scope, root)
        except (OSError, ValueError, TypeError, RecursionError):
            values = None
        if values is None or not all(isinstance(value, (str, Path)) for value in values):
            unknown = True
            continue
        for value in values:
            try:
                target = Path(value)
                target = (target if target.is_absolute() else root / target).resolve()
            except (OSError, ValueError):
                unknown = True
                continue
            if target.is_relative_to(root) and target.suffix == ".py":
                found.add(target)
            else:
                unknown = True
    return found, unknown
