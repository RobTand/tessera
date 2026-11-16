"""Conservative, non-executing discovery of Python file-consumer dependencies.

The supported expressions are finite Path constructions, not arbitrary Python.
A resolved target inside the tree is an exact edge whatever its suffix: a
``.py`` file is a module dependency, anything else is the data dependency it
is. A boundary refusal retains module or data uncertainty; an outside spelling
does not prove independence from this tree. An unresolved target retains the
consumer's corresponding uncertainty instead of silently dropping its edge.

**An unresolved target is a wildcard over Python only when the reader can run
Python.**  A loader always can, by definition.  A plain read cannot: bytes are
a Python dependency only once something parses or executes them, so a module
that never does is reading data, and calling its unnameable path "any module
in the tree" made the whole tree depend on the whole tree.  ``_SOURCE_*`` is
the recognition set for that -- the standard library's source-execution API,
which this repository does not own and cannot derive from its own code, and
which qualifies common attribute names (``re.compile`` and ``model.eval()``
are not source execution). ``source_execution_modules`` propagates this capability over
recognized calls to imported source-executing helpers; import presence alone
is not execution, and helper capability never proves a read's external origin.
What it misses is a source read this module never sees at all,
``subprocess.run([sys.executable, path])`` above all; that was never an edge
here and this does not change it.

**A target this resolver could not NAME and one it named and then declined to
PLACE are different facts, and only the first is silence.**  A read whose
filename is runtime state states no dependency, and reading that as "any
module in the tree" is what held every verdict at ``full`` (#148).  But the
boundary guard refuses a path this module named exactly -- an absolute
spelling outside the tree, which a local alias directory can carry straight
back into it -- and refusing to resolve it does not make the file it names
stop changing.  That refusal is reported as its own kind of uncertainty, a
data read with no placeable target, so a caller can keep the reader coupled
to the diff without promoting a plain reader into an unknown Python importer
(#338).
"""
from __future__ import annotations

import ast
import os.path
from collections import defaultdict
from pathlib import Path, PurePath

_LOADERS = {"spec_from_file_location": (1, "location"),
            "SourceFileLoader": (1, "path"), "run_path": (0, "path_name")}
_SYMBOLS = {"spec_from_file_location": "importlib.util.spec_from_file_location",
            "SourceFileLoader": "importlib.machinery.SourceFileLoader",
            "run_path": "runpy.run_path"}
_READ_METHODS = {"read_text", "read_bytes", "open"}
#: Directory-enumeration reads.  A call one of these names consumes the base
#: directory's *membership*, not one named file: what can change under it is
#: any path at or below the base -- added, edited or deleted -- so the edge it
#: resolves to is the base directory itself, and no pattern is matched
#: (tessera#923).  ``Path.glob`` is one of them (PB1496): its matches are
#: still exact edges through the expression resolver where a loader or reader
#: consumes them, but that edge names only the files that exist now, so a
#: deleted, added or renamed member -- and every change under the recursive
#: ``**`` spelling, which that resolver leaves unbounded -- selected nothing.
_ENUMERATIONS = {"glob", "iglob", "rglob", "iterdir", "listdir", "scandir", "walk"}
_KINDS = set(_LOADERS) | _READ_METHODS | _ENUMERATIONS
#: Modules whose enumeration calls read the filesystem.  A call named like an
#: enumeration whose resolved owners are all elsewhere (``ast.walk``) reads no
#: directory; an owner nothing resolves still counts.
_FILESYSTEM_OWNERS = ("os.", "glob.", "pathlib.")


class _Scope:
    def __init__(self, parent=None, *, class_body=False):
        self.parent = parent
        self.class_body = class_body
        self.bindings = defaultdict(list)

    def bind(self, target, value):
        for node in ast.walk(target):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                self.bindings[node.id].append(value)

    def bind_iteration(self, target, iterable):
        # Sequence elements are alternatives only when the loop iterates them.
        # A scalar Path((a, b)) or str([a, b]) must not resolve as either element.
        if isinstance(target, ast.Name) and isinstance(iterable, (ast.Tuple, ast.List, ast.Set)):
            for element in iterable.elts:
                self.bind(target, element)
        else:
            self.bind(target, iterable if isinstance(target, ast.Name) else None)


class _Scanner(ast.NodeVisitor):
    def __init__(self, path, module=None):
        self.module = module
        self.path_is_package = path.name == "__init__.py"
        self.scope = _Scope()
        self.scope.bindings["__file__"].append(ast.Constant(str(path)))
        self.calls = []
        self.references = []
        self.functions = defaultdict(list)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.references.append((node, self.scope))

    def _target_references(self, target):
        # Assignment/deletion targets can inspect an imported object's
        # namespace too. Record references without changing executor facts.
        self.references.extend((node, self.scope) for node in ast.walk(target)
                               if isinstance(node, ast.Name))

    def visit_Import(self, node):
        for alias in node.names:
            self.scope.bindings[alias.asname or alias.name.split(".")[0]].append(
                ("symbol", alias.name if alias.asname else alias.name.split(".")[0]))

    def visit_ImportFrom(self, node):
        prefix = node.module
        if node.level and self.module is not None:
            package = self.module if self.path_is_package else self.module.rpartition(".")[0]
            parts = package.split(".") if package else []
            climb = node.level - 1
            parts = parts[:len(parts) - climb] if climb else parts
            prefix = ".".join(parts + ([node.module] if node.module else []))
        for alias in node.names:
            self.scope.bindings[alias.asname or alias.name].append(
                ("symbol", f"{prefix}.{alias.name}")
                if prefix and (not node.level or self.module is not None) else None)

    def visit_Assign(self, node):
        for target in node.targets:
            self.scope.bind(target, node.value)
            self._target_references(target)
        self.visit(node.value)

    def visit_AnnAssign(self, node):
        self.scope.bind(node.target, node.value)
        self._target_references(node.target)
        self.visit(node.annotation)
        if node.value:
            self.visit(node.value)

    def visit_AugAssign(self, node):
        self.scope.bind(node.target, None)
        self._target_references(node.target)
        self.visit(node.value)

    def visit_NamedExpr(self, node):
        self.scope.bind(node.target, node.value)
        self.visit(node.value)

    def visit_For(self, node: ast.For | ast.AsyncFor):
        self.scope.bind_iteration(node.target, node.iter)
        self.generic_visit(node)

    def visit_AsyncFor(self, node):
        self.visit_For(node)

    def visit_With(self, node: ast.With | ast.AsyncWith):
        for item in node.items:
            if item.optional_vars:
                self.scope.bind(item.optional_vars, None)
        self.generic_visit(node)

    def visit_AsyncWith(self, node):
        self.visit_With(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.scope.bindings[node.name].append(None)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global | ast.Nonlocal):
        for name in node.names:
            self.scope.bindings[name].append(None)
            parent = self.scope.parent
            while parent:
                if name in parent.bindings or parent.parent is None:
                    parent.bindings[name].append(None)
                parent = parent.parent

    def visit_Nonlocal(self, node):
        self.visit_Global(node)

    def visit_Delete(self, node):
        for target in node.targets:
            self._target_references(target)
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
        if (self.module is not None and prior.parent is None
                and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
            self.functions[node.name].append(node)
        if hasattr(node, "name"):
            prior.bindings[node.name].append(
                ("symbol", f"{self.module}.{node.name}")
                if self.module is not None and prior.parent is None else None)
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

    def visit_AsyncFunctionDef(self, node):
        self._nested(node)

    def visit_Lambda(self, node):
        self._nested(node)

    def visit_ClassDef(self, node):
        self._nested(node, class_body=True)

    def visit_Call(self, node):
        self.calls.append((node, self.scope))
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp):
        prior = self.scope
        self.scope = _Scope(prior)
        for generator in node.generators:
            self.scope.bind(generator.target, None)
        self.generic_visit(node)
        self.scope = prior

    def visit_SetComp(self, node):
        self.visit_ListComp(node)

    def visit_DictComp(self, node):
        self.visit_ListComp(node)

    def visit_GeneratorExp(self, node):
        self.visit_ListComp(node)


#: Calls that turn bytes into running Python.  Bare names only for the
#: builtins -- ``model.eval()`` and ``re.compile()`` are attributes and are not
#: this.  Attribute names only where the name itself is the API.
_SOURCE_BUILTINS = {"exec", "eval", "compile", "execfile", "__import__"}
_SOURCE_ATTRIBUTES = {"run_path", "run_module", "spec_from_file_location",
                      "SourceFileLoader", "SourcelessFileLoader", "exec_module",
                      "source_to_code", "get_code", "compile_command"}
#: ``module: attribute`` pairs whose attribute is too common to match alone.
_SOURCE_QUALIFIED = {"ast": {"parse"}, "py_compile": {"compile"}}


def _source_call(call, symbols):
    """One conservative recognition predicate for module and helper facts."""
    function = call.func
    if isinstance(function, ast.Name) and function.id in _SOURCE_BUILTINS:
        return True
    if isinstance(function, ast.Attribute) and function.attr in _SOURCE_ATTRIBUTES:
        return True
    for symbol in symbols:
        module, _, name = symbol.rpartition(".")
        if (name in _SOURCE_ATTRIBUTES
                or name in _SOURCE_QUALIFIED.get(module, ())
                or module == "builtins" and name in _SOURCE_BUILTINS):
            return True
    return False


def _executes_python_source(tree):
    """Whether this module can turn file bytes into Python it runs or parses."""
    scanner = _Scanner(Path("source.py"))
    scanner.visit(tree)
    return any(_source_call(call, _possible_symbols(call.func, scope))
               for call, scope in scanner.calls)


#: A symlink chain longer than this is a loop for our purposes, and the walk
#: below ends it the way the kernel's own ``ELOOP`` does rather than
#: recursing forever.
_MAX_LINK_DEPTH = 40


def _resolve_within_root(path, root, budget=None, links=None):
    """Resolve *path* without ever naming a location outside *root*.

    Returns the resolved absolute path, or ``None`` when the spelling or the
    walk would leave the tree.  ``None`` is the same refusal a literal
    outside root already gets: unknown, decided without a syscall out there.

    **Membership of the destination is not a bound on the resolution.**
    ``Path.resolve`` walks the spelling as written, so a sibling-relative
    spelling like ``.../outside/../repo/driver.py`` -- which normalizes to
    a path inside the tree -- still ``lstat``s ``outside`` on the way, and that is
    the uninterruptible RPC #325 was about.  Collapsing ``..`` lexically
    first is no answer either: ``link/..`` is the parent of the *link's
    target*, so the string answer and the filesystem answer differ exactly
    where a symlink is involved, and an in-root link to an outside directory
    is followed before anyone can ask whether its target is in the tree.
    Both are one defect -- a normalized final membership says nothing about
    the steps taken to reach it (#339).

    So this walks instead.  *root* is the approved tree and is taken as
    canonical.  A spelling whose leading components are not *root*'s is
    refused before any filesystem call at all.  After that every component is
    examined only once the prefix it extends is known to be inside root;
    ``..`` is applied to a prefix already free of symlinks, so it means what
    the filesystem means by it; and a symlink whose target leaves the tree
    ends the walk -- the link itself is in-root and readable, its target is
    never approached.

    ``links`` collects the in-root link entries actually traversed. Repointing
    one changes the read even when its old and new targets are both in-root,
    so these entries are dependencies alongside the resolved destination.
    """
    base = Path(os.path.normpath(str(root)))
    absolute = path if path.is_absolute() else base / path
    parts = absolute.parts
    prefix = base.parts
    if parts[:len(prefix)] != prefix:
        # Not even spelled from inside the tree.  Whatever ``..`` would do to
        # it later, walking it means stat'ing outside root first.
        return None
    return _walk_within_root(
        base, parts[len(prefix):], base, [_MAX_LINK_DEPTH] if budget is None else budget, links)


def _walk_within_root(current, parts, base, budget, links=None):
    """One component at a time from *current*, which is already inside *base*."""
    for part in parts:
        if not part or part == ".":
            continue
        if part == "..":
            if current == base:
                return None                 # one step above the approved tree
            current = current.parent
            continue
        candidate = current / part
        try:
            # ``readlink`` on an in-root path: the only filesystem question
            # this walk ever asks, and never about a location outside root.
            link = os.readlink(candidate)
        except OSError:
            # Not a link, or nothing there to be one.  The name stands, which
            # is what ``resolve(strict=False)`` does with it too.
            current = candidate
            continue
        if links is not None:
            links.add(candidate)
        # One budget covers every link, including absolute targets and later
        # components after a recursive target walk returns (#353).
        if budget[0] == 0:
            return None
        budget[0] -= 1
        target = Path(link)
        if target.is_absolute():
            current = _resolve_within_root(target, base, budget, links)
        else:
            current = _walk_within_root(current, target.parts, base, budget, links)
        if current is None:
            return None
    return current


def _place(paths, root, refused, links=None):
    """The paths the boundary guard will not let us place, resolved if it will.

    Returns ``None`` when any of *paths* is refused -- recorded in *refused*
    rather than merely returned, because the caller's two unknowns are not the
    same fact: an expression this resolver cannot evaluate names no file,
    while a path it evaluated exactly and then refused to touch names one file
    it declines to identify (#338).  Only the second keeps a data dependency
    alive.
    """
    resolved = set()
    for path in paths:
        within = _resolve_within_root(path, root, links=links)
        if within is None:
            if refused is not None:
                refused.append(path)
            return None
        resolved.add(within)
    return resolved


def _values(node, scope, root, visiting=frozenset(), refused=None, links=None):
    """All statically established values; None means some alternative is unknown.

    ``refused``, when given, collects the paths a boundary guard declined to
    place, so a ``None`` return caused by the guard can be told from a
    ``None`` return caused by an unnameable expression.
    """
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
            value = _values(expression, here, root, visiting | {key}, refused, links)
            if value is None:
                return None
            result.update(value)
        return result
    if isinstance(node, ast.Attribute):
        values = _values(node.value, scope, root, visiting, refused, links)
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
        paths = _values(node.value.value, scope, root, visiting, refused, links)
        indices = _values(node.slice, scope, root, visiting, refused, links)
        if paths is not None and indices is not None:
            if not all(isinstance(path, Path) for path in paths) or not all(
                    isinstance(index, int) for index in indices):
                return None
            try:
                return {path.parents[index] for path in paths for index in indices}
            except IndexError:
                return None
    if isinstance(node, ast.BinOp):
        left = _values(node.left, scope, root, visiting, refused, links)
        right = _values(node.right, scope, root, visiting, refused, links)
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
            paths = _values(node.func.value, scope, root, visiting, refused, links)
            if paths is None or not all(isinstance(path, Path) for path in paths):
                return None
            if node.func.attr == "resolve" and not node.args and not node.keywords:
                # A literal ``.resolve()`` on a path outside root is the
                # same escape a crawling glob would be: unknown, and never
                # stat'ed to find out (rule 4; was :308).  ``_place`` is the
                # resolve, so this expression never reaches ``Path.resolve``
                # and never walks a step outside the tree (#339).
                return _place(paths, root, refused, links)
            if len(node.args) == 1 and not node.keywords:
                args = _values(node.args[0], scope, root, visiting, refused, links)
                if args is None or not all(isinstance(arg, str) for arg in args):
                    return None
                if node.func.attr == "joinpath":
                    return {path / arg for path in paths for arg in args}
                # Only one-directory globs are finite within the checked tree.
                # Recursive/escaping patterns remain an unknown dependency;
                # they must not trigger a filesystem crawl outside this root.
                # Nor may the base itself be resolved from outside root --
                # an absolute literal base is the identical escape (rule 4;
                # was :318), refused before any stat out there.  A base that
                # ``_place`` returns is inside root by construction, so the
                # membership re-check this used to make is gone with the
                # ``Path.resolve`` that needed it (#339); a relative base is
                # now joined to root rather than to the process's cwd, which
                # is what the guard above already assumed of it.
                bases = _place(paths, root, refused, links)
                if bases is None:
                    return None
                # pathlib's trailing-separator filter follows DirEntry links
                # before we can place its matches. Refuse the named pattern
                # without enumerating it, retaining data-read uncertainty.
                if any(arg.endswith(tuple(sep for sep in (os.sep, os.altsep) if sep))
                       for arg in args):
                    if refused is not None:
                        refused.extend(bases)
                    return None
                if (node.func.attr == "glob"
                        and all(len(Path(arg).parts) == 1 and arg not in {".", ".."}
                                and "**" not in arg for arg in args)):
                    return {item for base in bases for arg in args
                            for item in base.glob(arg)}
            return None
        functions = _values(node.func, scope, root, visiting, refused, links)
        if functions is None or len(node.args) != 1 or node.keywords:
            return None
        args = _values(node.args[0], scope, root, visiting, refused, links)
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


def _file_consumer_scan(tree, path):
    """Lexical facts and recognition aliases; neither classifies dependencies."""
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

    # Aliases recognize possible consumers; lexical resolution proves targets.
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
    return scanner, kind


def _possible_symbols(expression, scope):
    """Recognition only: union aliases, including shadowed alternatives.

    The worklist walks finite binding paths without consuming Python's call
    stack. Binding-cycle guards apply per path, before extending attributes,
    so ``alias = alias.member`` cannot produce an infinite suffix chain.
    This never proves a callable or an external file origin.
    """
    result = set()
    pending = [(expression, scope, "", frozenset())]
    while pending:
        value, here, suffix, visiting = pending.pop()
        if isinstance(value, tuple) and value[0] == "symbol":
            result.add(value[1] + suffix)
        elif isinstance(value, ast.Attribute):
            pending.append((value.value, here, "." + value.attr + suffix, visiting))
        elif isinstance(value, ast.Name):
            while here is not None:
                key = (id(here), value.id)
                if key not in visiting:
                    pending.extend((bound, here, suffix, visiting | {key})
                                   for bound in here.bindings.get(value.id, ()))
                here = here.parent
    return result


def _helper_capability(capable, edges):
    """One monotone fixed point for effects of recognized helper calls."""
    while True:
        expanded = capable | {key for key, dependencies in edges.items()
                              if dependencies & capable}
        if expanded == capable:
            return capable
        capable = expanded


def source_execution_modules(trees, modules, targets):
    """Files that call a source executor, including known imported helpers.

    ``targets`` is the caller's ambiguity-preserving module spelling
    resolver. Only top-level functions are exported summaries. Their
    source-execution capability grows to a fixed point over helper calls; it
    never depends on import presence alone and never proves read provenance.
    Unknown external origins and generic read parameters remain unknown.
    """
    functions = {}
    scopes = {}
    calls = {}
    for path, tree in trees.items():
        scanner = _Scanner(path, modules[path])
        scanner.visit(tree)
        functions[path] = scanner.functions
        scopes[path] = scanner.scope
        calls[path] = {
            call: _possible_symbols(call.func, scope)
            for call, scope in scanner.calls
        }

    def helpers(symbols):
        pending, seen = set(symbols), set()
        while pending:
            symbol = pending.pop()
            if symbol in seen:
                continue
            seen.add(symbol)
            module, _, name = symbol.rpartition(".")
            for path in targets(module):
                if name in functions.get(path, {}):
                    yield path, name
                if path in scopes:
                    pending.update(_possible_symbols(ast.Name(id=name), scopes[path]) - seen)

    capable = {
        (path, name) for path, defined in functions.items()
        for name, alternatives in defined.items()
        if any(_source_call(node, calls[path].get(node, ()))
               for function in alternatives for node in ast.walk(function)
               if isinstance(node, ast.Call))
    }
    edges = {
        (path, name): {
            helper for function in alternatives
            for node in ast.walk(function) if isinstance(node, ast.Call)
            for helper in helpers(calls[path].get(node, ()))
        }
        for path, defined in functions.items() for name, alternatives in defined.items()
    }
    capable = _helper_capability(capable, edges)
    return {
        path for path, tree in trees.items()
        if any(_source_call(call, symbols) for call, symbols in calls[path].items())
        or any(helper in capable for symbols in calls[path].values()
               for helper in helpers(symbols))
    }


def _literal_prefix(pattern):
    """The directory names in front of a pattern's first wildcard component."""
    prefix = []
    for part in PurePath(pattern).parts[:-1]:
        if any(wildcard in part for wildcard in "*?["):
            break
        prefix.append(part)
    return prefix


def _glob_receiver(loader, call, scope, root, refused, links):
    """``(bases, pattern arguments)`` of a ``glob``/``rglob`` call, or None.

    The method is named directly (``DOCS.glob(...)``, ``Path.glob(DOCS, ...)``)
    or through a name, which resolves only when it has one binding that is
    lexically the method.  Whichever way it is spelled, the owner is either a
    directory, which is the receiver, or the class ``Path``, whose receiver is
    the first argument.  Anything else names nothing, which the caller treats as
    an unnameable base.
    """
    if isinstance(call.func, ast.Attribute):
        owner = _values(call.func.value, scope, root, refused=refused, links=links)
    else:
        here = scope
        while here and call.func.id not in here.bindings:
            here = here.parent
        if here is None or len(here.bindings[call.func.id]) != 1:
            return None
        expression = here.bindings[call.func.id][0]
        if not (isinstance(expression, ast.Attribute) and expression.attr == loader):
            return None
        owner = _values(expression.value, here, root, refused=refused, links=links)
    if owner == {("symbol", "pathlib.Path")}:
        if not call.args:
            return None
        return _values(call.args[0], scope, root, refused=refused, links=links), call.args[1:]
    return owner, call.args


_MODULE_GLOBS = {("symbol", "glob.glob"), ("symbol", "glob.iglob")}

#: Directory entries the link scan may read for one call.  Over it, the read is kept
#: as an unplaced one: not provable within the budget, so select more, never less.
_LINK_SCAN_BUDGET = 20000


def _wildcard_directory_components(pattern, prefix=(), recursive=False):
    """The components of *pattern* after its literal prefix, the last one included.

    The last component is scanned like the others: it can match a link to a file, and a
    trailing separator (``docs/*/``, which ``PurePath`` drops) makes it a directory
    component.  ``rglob(pattern)`` is ``glob('**/' + pattern)``: the recursion runs over the
    whole base, so its literal components are not a prefix of the base.
    """
    return (["**"] if recursive else []) + list(PurePath(pattern).parts[len(prefix):])


def _scan_children(directories, recursive, root, links, found, budget, refused):
    """Subdirectories of *directories* (all depths when *recursive*), or None over budget.

    Entries are read without following links.  A link entry is resolved by ``_place``,
    which records it and stops at a target outside the tree, so that target is never
    approached.  A link the guard declines is appended to *refused*: where it leads is
    unknown and may be straight back into the tree.  An in-tree target is a
    dependency (*found*) and a directory joins the result.
    """
    seen = set(directories)
    reached = set(directories) if recursive else set()
    pending = list(directories)
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        budget[0] -= len(entries)
        if budget[0] < 0:
            return None
        for entry in entries:
            path = directory / entry.name
            if entry.is_symlink():
                placed = _place({path}, root, refused, links)
                target = next(iter(placed)) if placed else None
                if target is None:
                    continue
                # A dependency even when absent: deleting the last member behind an
                # unchanged link leaves it dangling and the read still names it.  Only
                # traversal needs a directory.
                found.add(target)
                if not target.is_dir():
                    continue
            elif entry.is_dir(follow_symlinks=False):
                target = path
            else:
                continue
            if target in seen:
                continue
            seen.add(target)
            reached.add(target)
            if recursive:
                pending.append(target)
    return reached


def _wildcard_link_dirs(start, components, root, links, refused):
    """Directories a link reached through a wildcard component points to (#1011).

    *start* is a placed directory.  Each wildcard component is expanded over the
    directories reached so far and each literal one is placed.  ``None`` when the
    scan would exceed its budget.
    """
    found, frontier, budget = set(), {start}, [_LINK_SCAN_BUDGET]
    for component in components:
        if component in ("", "."):
            continue
        if any(wildcard in component for wildcard in "*?["):
            frontier = _scan_children(frontier, component == "**", root, links, found, budget, refused)
            if frontier is None:
                return None
        else:
            reached = set()
            for directory in frontier:
                for target in _place({directory / component}, root, refused, links) or ():
                    reached.add(target)
                    # A literal component can itself be a link: where it leads is read, so
                    # it is a dependency and not only traversal state.  A plain path under
                    # the base is already covered by the base node.
                    if target != directory / component:
                        found.add(target)
            frontier = reached
    return found


def _with_link_targets(placed, specs, root, refused, links):
    """*placed* plus the directories links behind wildcard components lead to."""
    found = set()
    for start, components in specs:
        resolved = _place({start}, root, [], links)
        if not resolved:
            continue
        declined = []
        targets = _wildcard_link_dirs(next(iter(resolved)), components, root, links, declined)
        if targets is None or declined:
            # Over budget, or a link the guard declined to follow: not provable, so keep
            # the read as an unplaced one (#338) without looking where it leads.
            refused.extend(declined or [start])
            return None
        found |= targets
    return placed | found


def _module_glob_bases(call, scope, root, refused, links, from_string):
    """The directory a module-level ``glob.glob``/``glob.iglob`` call reads, or None.

    Unlike ``Path.glob`` the base is not a receiver: it is the literal directory
    in front of the pattern's first wildcard, relative to the process (joined to
    the tree like every other relative spelling), placed by the same boundary
    guard.  Only a base that lands inside the tree adds an edge.  A base the guard
    declines -- an absolute path such as ``/usr/local/cuda-*`` or a pattern that
    climbs out -- stays *unnamed*, as before, and is never kept as an unplaced
    read: this tree's module globs of that kind name box locations, not
    repository files, and an unplaced read seeds its reader's consumers on every
    change, which made ``serving/ext.py`` select the whole population (#148).
    ``root_dir``, ``dir_fd``, extra positional arguments, a pattern nothing names
    and a ``..`` after the first wildcard also stay unnamed (#1010), but only for that
    alternative: another alternative of the same call that names an in-tree base still
    places it or keeps it as an unplaced read, whatever order they come in.  A
    pattern spelled from the tree's own root names a base inside it: a refusal or a
    ``..`` there is kept as an unplaced read (#1010 review).

    A RELATIVE pattern names a directory relative to the process, and nothing here
    proves where that is: a ``chdir`` in this module, an alias of it, an imported
    helper, a fixture or pytest itself can move it.  It is kept as an unplaced read
    and never resolved against the tree's root; a pattern anchored with ``__file__``
    does not depend on the process directory and resolves (#1010 review).
    """
    if len(call.args) != 1 or isinstance(call.args[0], ast.Starred) or any(
            keyword.arg not in ("recursive", "include_hidden") for keyword in call.keywords):
        return None
    patterns = _values(call.args[0], scope, root, refused=refused, links=links)
    if patterns is None or not all(isinstance(pattern, str) for pattern in patterns):
        return None
    bases, specs, names_tree, climbs = set(), [], False, False
    root_parts = Path(os.path.normpath(str(root))).parts
    for pattern in patterns:
        prefix = _literal_prefix(pattern)
        base = Path(*prefix) if prefix else Path(".")
        if not PurePath(pattern).is_absolute():
            refused.append(base)
            return None
        # A pattern spelled from the tree's own root names a base INSIDE it, even when
        # its literal prefix stops early (a checkout name with glob metacharacters).
        in_tree = PurePath(pattern).parts[:len(root_parts)] == root_parts
        names_tree = names_tree or in_tree
        if ".." in PurePath(pattern).parts[len(prefix):]:
            # This alternative names nothing it can place.  An outside one is unnamed; an
            # in-tree one is an unplaced read.  Either way the other alternatives are still
            # read, so keep going and let them place or refuse (#1010 review).
            if in_tree:
                refused.append(base)
                climbs = True
            continue
        bases.add(base)
        components = _wildcard_directory_components(pattern, prefix)
        if components:
            specs.append((base, components))
    if climbs or not bases:
        return None
    declined = []
    placed = _place(bases, root, declined, links)
    if placed is None and names_tree:
        # A base named inside the tree that the guard declines (a directory that is a link out
        # of it) cannot be attributed to a file: the #338 uncertainty.  A base outside the tree,
        # a box path, stays unnamed.
        refused.extend(declined)
        return None
    if placed is not None and specs:
        placed = _with_link_targets(placed, specs, root, refused, links)
    if placed is not None and from_string is not None:
        from_string.append(True)
    return placed


def _enumeration_bases(loader, call, scope, root, refused, links, from_string=None):
    """The base directories an enumeration call consumes, or ``None``.

    A directory-wide read consumes the directory's *membership*: what can
    change under it is any path at or below the base, not one named file.  So
    the dependency is the base itself -- the selector holds it as a node under
    its repository path and seeds every changed path's ancestor directories
    against it, which is what carries added and deleted members a per-file
    edge would miss (#923).  No pattern is matched and nothing is enumerated
    here: a pattern, a flat listing and a recursive walk of one base all hold
    the same node, which is the sound direction -- matching the pattern would
    trade that over-selection for an under-selection any new file can trigger.

    ``None`` names nothing: the caller applies the named/unnamed rule (#148).
    A base that was named and then refused by the boundary guard -- or named
    with a pattern this resolver cannot resolve, which leaves the membership
    unknown -- is appended to ``refused`` so the caller keeps the #338
    unplaced-read uncertainty instead of dropping the directory.
    """
    if loader in ("glob", "iglob"):
        functions = _values(call.func, scope, root)
        if functions and functions <= _MODULE_GLOBS:
            return _module_glob_bases(call, scope, root, refused, links, from_string)
    specs = []
    if loader == "iterdir":
        bases = _values(call.func.value, scope, root, refused=refused, links=links)
        if bases is not None and (call.args or call.keywords):
            refused.extend(base for base in bases if isinstance(base, Path))
            return None
    elif loader in {"listdir", "scandir", "walk"}:
        if not call.args:
            return None
        bases = _values(call.args[0], scope, root, refused=refused, links=links)
        if bases is not None:
            # ``os.listdir("docs")`` reads the directory ``Path("docs")`` names:
            # a string path is the same base, placed (or refused) by the same
            # boundary guard.  Only the Path spelling was resolved, so a file
            # added under a directory listed by string selected no reader.
            if any(isinstance(base, str) and not PurePath(base).is_absolute() for base in bases):
                # A relative string names a directory relative to the process; nothing here
                # proves where that is, so it is unplaced and never root-relative.
                refused.extend(Path(base) for base in bases if isinstance(base, str))
                return None
            if from_string is not None and any(isinstance(base, str) for base in bases):
                from_string.append(True)
            bases = {Path(base) if isinstance(base, str) else base for base in bases}
    else:  # glob and rglob: the receiver names the tree, the argument the pattern.
        receiver = _glob_receiver(loader, call, scope, root, refused, links)
        if receiver is None:
            return None
        bases, arguments = receiver
        if bases is None:
            return None
        if len(arguments) != 1 or call.keywords:
            refused.extend(base for base in bases if isinstance(base, Path))
            return None
        patterns = _values(arguments[0], scope, root, refused=refused, links=links)
        if patterns is None or not all(
                isinstance(pattern, str) for pattern in patterns):
            refused.extend(base for base in bases if isinstance(base, Path))
            return None
        if any(PurePath(pattern).is_absolute() or ".." in PurePath(pattern).parts
               for pattern in patterns):
            # A pattern can leave the receiver; nothing here proves where.
            refused.extend(base for base in bases if isinstance(base, Path))
            return None
        if all(isinstance(base, Path) for base in bases):
            # A literal directory in front of the first wildcard may be a link
            # to another directory.  Placing it keeps the link and the target.
            recursive = loader == "rglob"
            for base in bases:
                for pattern in patterns:
                    prefix = _literal_prefix(pattern)
                    start = base if recursive or not prefix else base.joinpath(*prefix)
                    components = _wildcard_directory_components(pattern, () if recursive else prefix, recursive)
                    if components:
                        specs.append((start, components))
            bases = bases | {
                base.joinpath(*prefix)
                for base in bases for prefix in (_literal_prefix(pattern) for pattern in patterns)
                if prefix}
    if bases is None or not all(isinstance(base, Path) for base in bases):
        return None
    placed = _place(bases, root, refused, links)
    if placed is None or not specs:
        return placed
    return _with_link_targets(placed, specs, root, refused, links)


def _sole_plain_definitions(tree):
    """Names the file binds exactly once, in any scope, with an undecorated ``def``.

    This is a fallback, not a resolver (PB1496), and it only decides whether the
    unnamed-read warning is printed: no call is ever dropped from analysis.  A
    bare ``walk(...)`` is the file's own function only when nothing else in the
    file can bind the name:
    one ``def`` and no parameter, assignment, import, ``global``, loop or
    ``with`` target, ``except`` name, pattern capture, type parameter, decorator
    or class.  A star import anywhere voids the proof for every name, since it
    can rebind any of them.  Anything this cannot prove keeps its warning.
    """
    if any(isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names)
           for node in ast.walk(tree)):
        return set()
    count, plain = defaultdict(int), set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = [node.name]
            if not node.decorator_list:
                plain.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names = [node.name]
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            names = [node.id]
        elif isinstance(node, ast.arg):
            names = [node.arg]
        elif isinstance(node, ast.alias):
            names = [node.asname or node.name.split(".")[0]]
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names = list(node.names)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            names = [node.name] if node.name else []
        elif isinstance(node, ast.MatchMapping):
            names = [node.rest] if node.rest else []
        elif type(node).__name__ in ("TypeVar", "ParamSpec", "TypeVarTuple"):
            names = [node.name]
        for name in names:
            count[name] += 1
    return {name for name in plain if count[name] == 1}


def file_imports(tree, path, root, *, executes_source=None, unnamed=None):
    """Return in-tree dependencies, an unknown-loader flag, and an unplaced-read flag.

    ``unnamed``, when a list is given, receives the line of each directory read
    whose base nothing names in a module that executes nothing: the limit that
    selects no test and forces no run, listed so it is seen (PB1496).

    The third value is the one #338 exists for.  ``unknown`` says this module
    may import Python it cannot name; ``unplaced`` says it reads a file it
    named exactly and this resolver refused to place -- an outside spelling
    that an alias directory can carry back into the tree.  A caller that
    collapsed the two either lost the dependency (a plain reader is not an
    unknown importer, so it recorded nothing at all) or lost #148 (an
    unnameable read is not "every module in the tree").

    Directory-enumeration calls (``glob``, ``rglob``, ``iterdir``, ``os.listdir``/
    ``scandir``/``walk``) resolve to the base directory itself, under the same
    boundary guard and the same named/unnamed split: a resolvable base comes
    back in ``found`` as one directory node, a named-but-refused one as
    ``unplaced``, a base nothing names as neither (#923).
    """
    scanner, kind = _file_consumer_scan(tree, path)
    executes = (_executes_python_source(tree) if executes_source is None
                else executes_source)

    def wildcard(reading):
        """An unnameable target is an unknown *module* only if it can run."""
        return executes or not reading

    found, unknown, unplaced = set(), False, False
    own_definitions = _sole_plain_definitions(tree)

    def refuse(reading):
        """Record a target this resolver named and then declined to place.

        A loader, or any module that can execute source, still edges to the
        whole tree: it may run what it read.  A plain reader does not -- that
        reading is #148 -- but the file it named can still be in the diff, so
        the dependency is kept as its own uncertainty rather than dropped
        (#338).  ``wildcard`` decides which of the two this is, exactly as it
        does for a target that was never nameable.
        """
        nonlocal unknown, unplaced
        if wildcard(reading):
            unknown = True
        else:
            unplaced = True

    for call, scope in scanner.calls:
        loaders = kind(call.func)
        if not loaders:
            continue
        reading = loaders <= _READ_METHODS | _ENUMERATIONS
        if len(loaders) != 1:
            unknown = unknown or wildcard(reading)
            continue
        loader = next(iter(loaders))
        if loader in _ENUMERATIONS:
            owners = _possible_symbols(call.func, scope)
            if owners and not any(owner.startswith(_FILESYSTEM_OWNERS) for owner in owners):
                continue
            refused, links, from_string = [], set(), []
            try:
                targets = _enumeration_bases(
                    loader, call, scope, root, refused, links, from_string)
            except (OSError, ValueError, TypeError, RecursionError):
                targets = None
            if targets is None:
                # A base this resolver named and the boundary guard refused is
                # the #338 refusal; a base nothing named follows the #148 rule,
                # widened by ``reading``: a module that can execute source may
                # run what any directory holds, so it stays a wildcard.
                if refused:
                    refuse(True)
                else:
                    if (unnamed is not None and not wildcard(True)
                            and not (isinstance(call.func, ast.Name)
                                     and call.func.id in own_definitions)):
                        # A directory read of a base nothing names, in a module
                        # that executes nothing, states no dependency (#148).
                        # It is the one case the selector can neither select
                        # nor escalate, so it is listed (PB1496).  Only this
                        # warning is ever withheld, for a call to the file's own
                        # sole plain def: the call itself is processed exactly
                        # as for any other, so a base it names keeps its edge.
                        unnamed.append(call.lineno)
                    unknown = unknown or wildcard(True)
            else:
                found.update(targets)
                found.update(links)
                if from_string:
                    # A bare ``walk("mode")`` is recognized by its name alone, and
                    # a string need not be a path.  Naming it adds the edge; it
                    # must not replace the unknown-loader flag a module that can
                    # execute source had while the string was "unnamed".
                    unknown = unknown or wildcard(True)
            continue
        # Refusals by the boundary guard anywhere inside this call's
        # expressions, so the ``values is None`` below can tell "no target
        # was nameable" from "a named target was not placeable".
        refused = []
        links = set()
        try:
            functions = _values(call.func, scope, root, refused=refused, links=links)
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
                    values = _values(expression, scope, root, refused=refused, links=links)
                else:
                    values = None
            else:
                position, keyword = _LOADERS[loader]
                expression = call.args[position] if len(call.args) > position else next(
                    (arg.value for arg in call.keywords if arg.arg == keyword), None)
                if functions != {("symbol", _SYMBOLS[loader])}:
                    unknown = True
                values = _values(expression, scope, root, refused=refused, links=links)
        except (OSError, ValueError, TypeError, RecursionError):
            values = None
        if values is None or not all(isinstance(value, (str, Path)) for value in values):
            # A guard refusal is a named file; anything else named none.
            if refused:
                refuse(reading)
            else:
                unknown = unknown or wildcard(reading)
            continue
        # A resolved glob base is a dependency even when it yields no files.
        found.update(links)
        for value in values:
            try:
                target = Path(value)
            except (OSError, ValueError):
                unknown = unknown or wildcard(reading)
                continue
            # An absolute (or ``..``-escaping) literal outside root is the
            # same escape the glob and resolve() guards above refuse:
            # unknown rather than resolved, so a stalled mount under the
            # literal's real location never blocks the selector (rule 4; was
            # :426).  ``_place`` decides the whole question -- spelling,
            # every resolution step and the destination -- and it is the only
            # thing here that touches the filesystem (#339).  The refusal is
            # not the same as independence: an outside spelling can be a
            # local alias for a tracked file, and the alias is environment
            # state this repository never sees, so the dependency survives it
            # as an unplaced read (#338).
            try:
                placed = _place({target}, root, None, links)
            except (OSError, ValueError):
                placed = None
            if placed is None:
                refuse(reading)
                continue
            # Named and inside the tree: an exact edge, ``.py`` or not.  There
            # is no third outcome left here -- a target that resolved outside
            # the tree used to be dropped in silence, and it is now the same
            # refusal as any other step that leaves root.
            found.update(placed)
            found.update(links)
    return found, unknown, unplaced
