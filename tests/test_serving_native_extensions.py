"""What the plugin LOADS, checked against what the contract publishes.

``runtime_contract.json``'s ``formats``/``lane_eligibility`` say what a Tessera
serve EXECUTES.  ``native_extensions`` says what it can map into the serving
process, and it exists because a consumer keys reproducibility on exactly that:
PrismaQuant's serve fingerprint records a KL as comparable only across serves
whose native-extension residency matches, so a lane whose ``.so`` it cannot
name fingerprints identically to a stock serve.  With no table to read, that
consumer mirrored the basename in its own repository -- principle 14's failure,
one repository over.

A hand-kept list of names would be the same defect on this side of the wall, so
two mechanisms hold it up and they answer different questions:

* ``contract.validate_serving_contract`` refuses any block that is not
  ``ext.NATIVE_EXTENSIONS``, whose ``module_name_prefix`` IS the constant
  ``ext._load_locked`` passes to ``cpp_extension.load``.  That makes the
  published name the loaded name by construction: it cannot be wrong about the
  extension it declares.
* :func:`scan_jit_extension_loads` below walks the import graph from
  ``tessera.serving`` and finds every JIT load site reachable from it.  That
  answers the question the first mechanism cannot: is the list SHORT?  The
  scope note on RobTand/tessera#28 -- ``tessera.kernel_window_gemv`` builds
  ``tessera_window_gemv``, but nothing under ``tessera/serving/`` reaches it,
  so it is producer-side -- is a claim about the import graph, and this is that
  claim mechanised.  The day a route wires the window GEMV in, the table goes
  red instead of staying quietly short.

The scanner is static (``ast``) and deliberately unforgiving: a load site whose
module name it cannot read is a FAILURE, not a skip.  A JIT loader that hides
its name from a reader hides it from every consumer of this contract too.
"""
from __future__ import annotations

import ast
import fnmatch
import textwrap
from pathlib import Path

import pytest

from tessera.serving import ext
from tessera.serving.contract import load_serving_contract, validate_serving_contract

SRC = Path(__file__).resolve().parents[1] / "src"

#: Call shapes that put native code into a process.  ``load``/``load_inline``
#: are matched only when the file imports them from ``cpp_extension`` (bare
#: ``load`` is also ``json.load``); the loader attributes are unambiguous.
_CPP_EXTENSION_NAMES = ("load", "load_inline")
_LOADER_ATTRS = ("CDLL", "LoadLibrary", "load_library")


def _module_path(module: str, src: Path) -> Path | None:
    parts = module.split(".")
    candidate = src.joinpath(*parts).with_suffix(".py")
    if candidate.is_file():
        return candidate
    package = src.joinpath(*parts, "__init__.py")
    return package if package.is_file() else None


def _imports(tree: ast.AST, module: str) -> set[str]:
    """First-party modules this one imports, function-local imports included.

    Function-local is not an edge case here: ``ops`` reaches ``tessera.fused``
    and ``fp8_route`` reaches ``tessera.decode`` exactly that way, to keep the
    contract reader torch-free.  A walk that only read module-level imports
    would declare almost the whole package unreachable and go vacuous.
    """
    package = module.rsplit(".", 1)[0] if "." in module else module
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[:len(base) - node.level + 1]
                root = ".".join(base + ([node.module] if node.module else []))
            else:
                root = node.module or ""
            if not root:
                continue
            found.add(root)
            # ``from tessera import decode`` names a MODULE, not an attribute.
            found.update(f"{root}.{alias.name}" for alias in node.names)
    return {m for m in found if m == "tessera" or m.startswith("tessera.")}


#: What a name computed at run time becomes, so the glob's ``*`` is tested
#: against a name the site can really produce rather than against a wildcard.
_VARIES = "0123456789abcdef"


def _name_argument(call: ast.Call) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == "name":
            return keyword.value
    return call.args[0] if call.args else None


def _assigned_in_scope(tree: ast.AST, target: str, before: int) -> ast.expr | None:
    """The last value bound to ``target`` above line ``before``.

    Deliberately not scope-aware: one flat pass over the module, nearest
    preceding assignment wins.  That is exact on a tree where a load site's
    name is bound once (today's, and the shape a legible loader has); a name
    shadowed in an inner scope would resolve to the wrong binding, and the
    honest failure mode is that the site then reads as undeclared -- red, not
    quietly passing.
    """
    best: tuple[int, ast.expr] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or node.lineno > before:
            continue
        for slot in node.targets:
            if isinstance(slot, ast.Name) and slot.id == target:
                if best is None or node.lineno > best[0]:
                    best = (node.lineno, node.value)
    return best[1] if best else None


def _constructor_attribute(expr: ast.Attribute, tree: ast.AST,
                           lineno: int) -> ast.expr | None:
    """Read one unambiguous field of a same-module constructor.

    This follows ``self.build = Build(...)`` into ``Build.__init__``;
    unrelated classes' ``self.name`` fields are never candidate bindings.
    Multiple writes, properties, dynamic factories and inherited fields stay
    unreadable. They require a richer proof, not a guessed published prefix.
    """
    if isinstance(expr.value, ast.Name) and expr.value.id == "self":
        owners = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
                  and n.lineno <= lineno <= n.end_lineno]
        owner = min(owners, key=lambda n: n.end_lineno-n.lineno) if owners else None
    elif isinstance(expr.value, ast.Attribute):
        bound = _constructor_attribute(expr.value, tree, lineno)
        if not (isinstance(bound, ast.Call) and isinstance(bound.func, ast.Name)):
            return None
        owners = [n for n in tree.body if isinstance(n, ast.ClassDef)
                  and n.name == bound.func.id]
        owner = owners[0] if len(owners) == 1 else None
        # A constructor with a rebound identifier is not an explicit owner.
        if any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
               and n.id == bound.func.id for n in ast.walk(tree)):
            return None
        if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == bound.func.id for n in tree.body):
            return None
        lineno = owner.end_lineno if owner else lineno
    else:
        return None
    if owner is None or owner.bases:
        return None
    if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
           and n.name in (expr.attr, "__getattribute__", "__getattr__", "__setattr__")
           for n in owner.body):
        return None
    initializers = [n for n in owner.body if isinstance(n, ast.FunctionDef)
                    and n.name == "__init__"]
    if len(initializers) != 1:
        return None
    initializer = initializers[0]
    writes = [n for n in ast.walk(owner) if isinstance(n, ast.Attribute)
              and isinstance(n.ctx, ast.Store) and n.attr == expr.attr
              and isinstance(n.value, ast.Name) and n.value.id == "self"]
    if len(writes) != 1 or writes[0].lineno >= lineno:
        return None
    for assignment in ast.walk(initializer):
        if isinstance(assignment, ast.Assign) and writes[0] in assignment.targets:
            return assignment.value
    return None


def _producible_name(expr: ast.expr | None, tree: ast.AST, lineno: int,
                     active: frozenset[int] = frozenset()) -> str | None:
    """A module name this call site can produce, or ``None`` if unreadable.

    A literal gives the name exactly (the window loader spells
    ``name="tessera_window_gemv"`` as a literal for exactly this reader).  An
    f-string gives its leading constant segment plus a placeholder for what
    varies. Bare names and explicit constructor fields resolve through their
    assignments; cycles and opaque owner/path expressions remain unreadable.
    """
    if expr is None or id(expr) in active:
        return None
    active = active | {id(expr)}

    def read(value, at=lineno):
        return _producible_name(value, tree, at, active)

    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    if isinstance(expr, ast.JoinedStr):
        out = []
        for piece in expr.values:
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                out.append(piece.value)
                continue
            inner = piece.value if isinstance(piece, ast.FormattedValue) else None
            resolved = read(inner) if inner is not None else None
            # A segment that resolves to a constant is part of the name; one
            # computed at run time is what the glob's ``*`` stands for.
            out.append(resolved if resolved is not None else _VARIES)
        name = "".join(out)
        # A variable directory/hash suffix is fine; an opaque basename prefix
        # cannot stand in for an explicitly published library identity.
        return None if name.rsplit("/", 1)[-1].startswith(_VARIES) else name
    if isinstance(expr, ast.Name):
        bound = _assigned_in_scope(tree, expr.id, lineno)
        if bound is None or isinstance(bound, ast.Name):
            return None
        return read(bound)
    if isinstance(expr, ast.Attribute):
        bound = _constructor_attribute(expr, tree, lineno)
        return read(bound, bound.lineno) if bound is not None else None
    if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)
            and expr.func.id == "str" and len(expr.args) == 1 and not expr.keywords):
        return read(expr.args[0])
    if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)
            and expr.func.id == "Path" and len(expr.args) == 1 and not expr.keywords):
        return read(expr.args[0])
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
        # cpp_extension.load(..., is_python_module=False) returns the path of
        # the library it compiled under its explicit name. Follow that real
        # compiler result, not a predicted directory or an invented prefix.
        imports = {alias.asname or alias.name for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom) and node.module == "torch.utils.cpp_extension"
                   for alias in node.names if alias.name == "load"}
        rebound = any((isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                       and node.name == expr.func.id) or
                      (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                       and node.id == expr.func.id) for node in ast.walk(tree))
        non_python = any(keyword.arg == "is_python_module"
                         and isinstance(keyword.value, ast.Constant)
                         and keyword.value.value is False for keyword in expr.keywords)
        if expr.func.id in imports and non_python and not rebound:
            return read(_name_argument(expr))
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        left = read(expr.left)
        right = read(expr.right)
        return None if left is None or right is None else left + right
    if (isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div)
            and isinstance(expr.left, ast.Call) and isinstance(expr.left.func, ast.Name)
            and expr.left.func.id == "Path" and len(expr.left.args) == 1
            and not expr.left.keywords):
        # Only the directory varies; an unreadable basename must still refuse.
        basename = read(expr.right)
        return None if basename is None else _VARIES + "/" + basename
    return None


def scan_jit_extension_loads(src: Path, roots: list[str]) -> list[dict[str, object]]:
    """Every native-load site reachable from ``roots``, as ``{module, line, name}``.

    ``name`` is a module name the site can produce, with anything it computes
    replaced by a placeholder, or ``None`` when the site is not statically
    legible -- which the caller must treat as a failure.
    """
    seen: set[str] = set()
    queue = list(roots)
    sites: list[dict[str, object]] = []
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        path = _module_path(module, src)
        if path is None:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        queue.extend(_imports(tree, module) - seen)
        aliased = {
            alias.asname or alias.name
            for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
            and (node.module or "").endswith("cpp_extension")
            for alias in node.names if alias.name in _CPP_EXTENSION_NAMES
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            hit = (isinstance(func, ast.Name) and func.id in aliased) or (
                isinstance(func, ast.Attribute)
                and (func.attr in _LOADER_ATTRS
                     or (func.attr in _CPP_EXTENSION_NAMES
                         and isinstance(func.value, ast.Name)
                         and func.value.id.endswith("cpp_extension"))))
            if not hit:
                continue
            name = _producible_name(_name_argument(node), tree, node.lineno)
            if (name is not None and isinstance(func, ast.Attribute)
                    and func.attr in _LOADER_ATTRS):
                # A loader attribute takes a PATH, and a glob is matched
                # against the BASENAME of a mapped library (``match``), so the
                # name a site declares is the basename without its suffix.
                name = name.rsplit("/", 1)[-1].removesuffix(".so")
            sites.append({"module": module, "line": node.lineno, "name": name})
    return sites


def _declared_libraries() -> list[dict[str, object]]:
    """Every library the contract publishes, from BOTH blocks that publish one."""
    return [*ext.NATIVE_EXTENSIONS,
            *(entry["library"] for entry in ext.STOCK_KERNEL_OVERRIDES if entry["library"])]


def _undeclared(sites: list[dict[str, object]],
                libraries: list[dict[str, object]]) -> list[dict[str, object]]:
    """The readable load sites whose library no published glob matches."""
    globs = [library["filename_glob"] for library in libraries]
    return [s for s in sites if s["name"] is not None
            and not any(fnmatch.fnmatch(f"{s['name']}.so", g) for g in globs)]


def _serving_modules() -> list[str]:
    package = SRC / "tessera" / "serving"
    return sorted(f"tessera.serving.{p.stem}" if p.stem != "__init__" else "tessera.serving"
                  for p in package.glob("*.py"))


# --- the table is the load path's, not a copy of it ---------------------------

def test_the_contract_publishes_exactly_what_this_build_loads():
    contract = load_serving_contract()
    assert contract["native_extensions"] == list(ext.NATIVE_EXTENSIONS)


def test_the_published_name_is_the_constant_the_load_path_asks_for():
    """Not "agrees with": the same name.

    ``kernel_window_gemv._ext`` spells the module ``name="tessera_window_gemv"``
    as a LITERAL (so this contract reader can read it statically), and the
    table publishes ``WINDOW_GEMV_MODULE_NAME``.  Both the literal and the
    constant are asserted here, so a rename on either side is red rather than
    a table and a load path that quietly differ.
    """
    source = (SRC / "tessera" / "kernel_window_gemv.py").read_text(encoding="utf-8")
    assert 'name="tessera_window_gemv"' in source
    assert ext.NATIVE_EXTENSIONS[0]["module_name_prefix"] == ext.WINDOW_GEMV_MODULE_NAME
    assert ext.WINDOW_GEMV_MODULE_NAME == "tessera_window_gemv"


def test_the_glob_matches_the_library_torch_actually_writes():
    """The suffix is torch's ``LIB_EXT``, not our guess at a platform."""
    pytest.importorskip("torch")
    from torch.utils import cpp_extension

    entry = ext.NATIVE_EXTENSIONS[0]
    built = f"{entry['module_name_prefix']}{cpp_extension.LIB_EXT}"
    assert fnmatch.fnmatch(built, entry["filename_glob"])
    # The window module name carries no build-identity hash, so the exact file
    # IS the published name plus the suffix.
    assert entry["filename_glob"] == ext.WINDOW_GEMV_MODULE_NAME + "*.so"


def test_the_fallback_the_table_publishes_is_the_window_lane_s():
    """``when_unavailable`` is what the window lanes' preparation gates on."""
    pytest.importorskip("torch")
    from tessera.serving.lane import MODE_RESIDENT, MODE_STREAMED
    from tessera.serving.telemetry import DECODERS, DECODER_TORCH_WINDOW

    assert ext.substitutes_when_unavailable(
        MODE_RESIDENT, ext.WINDOW_GEMV_MODULE_NAME) is True
    assert ext.substitutes_when_unavailable(
        MODE_STREAMED, ext.WINDOW_GEMV_MODULE_NAME) is True
    behaviours = ext.NATIVE_EXTENSIONS[0]["when_unavailable"]
    for mode in (MODE_RESIDENT, MODE_STREAMED):
        assert behaviours[mode]["decoder"] in DECODERS
        assert behaviours[mode]["decoder"] == DECODER_TORCH_WINDOW
    # The lane the entry is for: the FP8 and BF16 routes' streamed preparation
    # loads it through ``fp8_gemv``'s module name constant.
    assert ext.NATIVE_EXTENSIONS[0]["routes"] == ["TESSERA_FP8", "TESSERA_BF16"]


def test_an_unknown_residency_is_a_refusal_not_a_default():
    with pytest.raises(ValueError, match="publishes no behaviour for residency"):
        ext.substitutes_when_unavailable("hybrid", ext.WINDOW_GEMV_MODULE_NAME)


# --- the list cannot go quietly short -----------------------------------------

def test_every_native_load_reachable_from_serving_is_declared():
    """The scope note, mechanised.

    Reachability is the criterion because residency is: an extension loaded by
    anything the serving package can import is an extension that can be mapped
    into the serving process, whichever module holds the ``load`` call.
    """
    sites = scan_jit_extension_loads(SRC, _serving_modules())
    assert sites, ("the scan found no native load site at all; the walk is broken, and a "
                   "scan that finds nothing agrees with every table")
    unreadable = [s for s in sites if s["name"] is None]
    assert not unreadable, (
        f"cannot statically read the module name at {unreadable}; a JIT loader that hides its "
        "name from a reader hides it from every consumer of this contract")
    undeclared = _undeclared(sites, _declared_libraries())
    assert not undeclared, (
        f"{undeclared} is loadable from tessera.serving and is not in "
        "runtime_contract.json's native_extensions or stock_kernel_overrides. Either it belongs "
        "there -- a library that can be resident in a serving process is a library a "
        "fingerprint must be able to see -- or the route that reaches it should not.")


def test_every_declared_extension_is_actually_loadable_from_serving():
    """The other direction: no entry outlives the code that loaded it."""
    sites = scan_jit_extension_loads(SRC, _serving_modules())
    for library in _declared_libraries():
        assert any(fnmatch.fnmatch(f"{s['name']}.so", library["filename_glob"])
                   for s in sites if s["name"]), (
            f"{library['filename_glob']} is published and no reachable code loads it")


# --- the scanner itself has teeth ---------------------------------------------

def test_the_scanner_finds_an_undeclared_loader_through_an_import(tmp_path):
    """Vacuity check, on a synthetic tree: it must catch a SECOND loader.

    Reachable only through a function-local import from the serving package,
    which is how the real one is reached, so this exercises the walk and not
    just the call matcher.
    """
    root = tmp_path / "src" / "tessera"
    (root / "serving").mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "serving" / "__init__.py").write_text("")
    (root / "serving" / "route.py").write_text(textwrap.dedent("""
        def go():
            from tessera.smuggled import build
            return build()
    """))
    (root / "smuggled.py").write_text(textwrap.dedent("""
        from torch.utils.cpp_extension import load

        def build():
            return load(name="tessera_smuggled", sources=["x.cu"])
    """))
    sites = scan_jit_extension_loads(tmp_path / "src",
                                     ["tessera.serving", "tessera.serving.route"])
    assert [(s["module"], s["name"]) for s in sites] == [
        ("tessera.smuggled", "tessera_smuggled")]


def test_an_undeclared_ctypes_loader_reachable_from_serving_is_still_undeclared(tmp_path):
    """The mutant, for the second block: reading stock_kernel_overrides widens
    what counts as declared, and must not widen it to everything.

    A ``ctypes.CDLL`` of a library neither block publishes, reached from the
    serving package the way an override's install path is, is undeclared
    against the real published globs.  The same site counts as declared only
    once a library with its glob is published.
    """
    root = tmp_path / "src" / "tessera"
    (root / "serving").mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "serving" / "__init__.py").write_text(textwrap.dedent("""
        def register():
            from tessera.serving import smuggled_override
            smuggled_override.install()
    """))
    (root / "serving" / "smuggled_override.py").write_text(textwrap.dedent("""
        import ctypes

        PREFIX = "tessera_smuggled_"

        def install():
            directory = cache_dir()
            key = build_key()
            return ctypes.CDLL(f"{directory}/{PREFIX}{key}.so")
    """))
    sites = scan_jit_extension_loads(tmp_path / "src", ["tessera.serving"])
    assert [(s["module"], s["name"]) for s in sites] == [
        ("tessera.serving.smuggled_override", "tessera_smuggled_0123456789abcdef")]
    assert _undeclared(sites, _declared_libraries()) == sites
    published = {"module_name_prefix": "tessera_smuggled_",
                 "filename_glob": "tessera_smuggled_*.so",
                 "match": ext.MATCH_BASENAME_FNMATCH, "source": "csrc/window_gemv.cu"}
    assert _undeclared(sites, [*_declared_libraries(), published]) == []


def test_the_scanner_refuses_a_load_site_it_cannot_read(tmp_path):
    root = tmp_path / "src" / "tessera" / "serving"
    root.mkdir(parents=True)
    (root.parent / "__init__.py").write_text("")
    (root / "__init__.py").write_text(textwrap.dedent("""
        from torch.utils.cpp_extension import load

        def build(chosen):
            return load(name=chosen, sources=["x.cu"])
    """))
    sites = scan_jit_extension_loads(tmp_path / "src", ["tessera.serving"])
    assert [s["name"] for s in sites] == [None]



def _shared_builder_sites(tmp_path, *, name='f"{PREFIX}{key}"',
                          owner='Build()', extra='', other=''):
    root = tmp_path / "src" / "tessera"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "builder.py").write_text(textwrap.dedent(f"""
        import ctypes
        from pathlib import Path
        from torch.utils.cpp_extension import load
        PREFIX = "tessera_shared_"
        class Build:
            def __init__(self):
                self.name = {name}
                {extra}
                self.library_path = Path(directory) / f"{{self.name}}.so"
                load(name=self.name, sources=["x.cu"])
        class Runtime:
            def __init__(self):
                self.build = {owner}
                ctypes.CDLL(str(self.build.library_path))
        {other}
    """))
    return scan_jit_extension_loads(tmp_path / "src", ["tessera.builder"])


def test_shared_builder_name_and_retained_path_have_one_inventory_identity(tmp_path):
    sites = _shared_builder_sites(tmp_path)
    assert [s['name'] for s in sites] == ['tessera_shared_'+_VARIES]*2
    assert _undeclared(sites, []) == sites
    assert _undeclared(sites, [{'filename_glob':'tessera_shared_*.so'}]) == []


@pytest.mark.parametrize('changes,expected', [
    ({'name':'chosen'}, [None,None]),
    ({'owner':'dynamic_builder()'}, ['tessera_shared_'+_VARIES,None]),
    ({'extra':'self.name = chosen'}, [None,None]),
    ({'name':'self.name'}, [None,None]),
    ({'other':'class Other: name = "tessera_foreign"'}, ['tessera_shared_'+_VARIES]*2),
])
def test_shared_builder_inventory_refuses_opaque_or_ambiguous_names(tmp_path, changes, expected):
    assert [s['name'] for s in _shared_builder_sites(tmp_path, **changes)] == expected

def _compiler_result_sites(tmp_path, *, builder="load", module_name='"tessera_compiler_result"',
                           is_python="False", extra=""):
    root = tmp_path / "src" / "tessera"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text(textwrap.dedent(f"""
        import ctypes
        from pathlib import Path
        from torch.utils.cpp_extension import load
        {extra}
        def build():
            name = {module_name}
            built = {builder}(name=name, sources=["x.cu"], is_python_module={is_python})
            ctypes.CDLL(str(Path(built)))
    """))
    return scan_jit_extension_loads(tmp_path / "src", ["tessera"])


def test_compiler_returned_library_path_keeps_its_explicit_inventory_name(tmp_path):
    sites = _compiler_result_sites(tmp_path)
    assert [site["name"] for site in sites] == ["tessera_compiler_result"] * 2
    assert _undeclared(sites, []) == sites
    assert not _undeclared(sites, [{"filename_glob": "tessera_compiler_result.so"}])


@pytest.mark.parametrize("changes,expected", [
    ({"builder": "choose_runtime_library"}, [None]),
    ({"module_name": "unknown_name"}, [None, None]),
    ({"is_python": "True"}, ["tessera_compiler_result", None]),
    ({"extra": "load = choose_runtime_library"}, ["tessera_compiler_result", None]),
])
def test_compiler_result_proof_keeps_opaque_and_rebound_factories_unreadable(tmp_path, changes, expected):
    assert [site["name"] for site in _compiler_result_sites(tmp_path, **changes)] == expected


def test_the_scanner_does_not_trip_on_an_ordinary_load(tmp_path):
    root = tmp_path / "src" / "tessera" / "serving"
    root.mkdir(parents=True)
    (root.parent / "__init__.py").write_text("")
    (root / "__init__.py").write_text(textwrap.dedent("""
        import json

        def read(handle):
            return json.load(handle)
    """))
    assert scan_jit_extension_loads(tmp_path / "src", ["tessera.serving"]) == []


# --- the validator refuses a table that drifted -------------------------------

def _mutated(**changes):
    import copy

    contract = copy.deepcopy(load_serving_contract())
    contract["native_extensions"][0].update(changes)
    return contract


def test_a_missing_block_is_refused():
    import copy

    contract = copy.deepcopy(load_serving_contract())
    del contract["native_extensions"]
    with pytest.raises(ValueError, match="missing.*native_extensions"):
        validate_serving_contract(contract)


def test_a_dropped_entry_is_refused():
    import copy

    contract = copy.deepcopy(load_serving_contract())
    contract["native_extensions"] = []
    with pytest.raises(ValueError, match="not what this build loads"):
        validate_serving_contract(contract)


def test_a_glob_that_matches_nothing_the_load_path_builds_is_refused():
    contract = _mutated(filename_glob="tessera_nvfp4.so")
    with pytest.raises(ValueError, match="not what this build loads"):
        validate_serving_contract(contract)


def test_an_optional_boolean_is_not_an_answer():
    """The shape the issue proposed, refused by name.

    ``optional: true`` says absence is survivable somewhere; it does not say
    which serve ran instead, and the answer differs by residency.
    """
    import copy

    contract = copy.deepcopy(load_serving_contract())
    entry = contract["native_extensions"][0]
    del entry["when_unavailable"]
    entry["optional"] = True
    with pytest.raises(ValueError, match="not what this build loads"):
        validate_serving_contract(contract)


@pytest.mark.parametrize("field, value, message", [
    ("match", "prefix", "match is"),
    ("source", "csrc/absent.cu", "not packaged with this build"),
    ("loaded_by", "tessera.kernel_window_gemv", "loaded_by is"),
    ("routes", ["TESSERA_INVENTED"], "routes names"),
])
def test_a_structurally_wrong_entry_is_refused(field, value, message, monkeypatch):
    """With the authority check stood down, so the STRUCTURE checks are reached.

    The equality against ``ext.NATIVE_EXTENSIONS`` would refuse every one of
    these first, and then nothing below it would ever run -- which is how a
    validator grows unreachable branches.
    """
    import copy

    contract = copy.deepcopy(load_serving_contract())
    contract["native_extensions"][0][field] = value
    monkeypatch.setattr(ext, "NATIVE_EXTENSIONS", contract["native_extensions"])
    with pytest.raises(ValueError, match=message):
        validate_serving_contract(contract)


@pytest.mark.parametrize("behaviours, message", [
    ({"resident": {"status": "substituted", "decoder": "torch_materialize_stock"}},
     "exactly the residencies"),
    ({"resident": {"status": "substituted", "decoder": None},
      "streamed": {"status": "refused", "decoder": None}},
     "substitutes and names no decoder"),
    ({"resident": {"status": "substituted", "decoder": "torch_materialize_stock"},
      "streamed": {"status": "refused", "decoder": "torch_materialize_stock"}},
     "refuses and still names a decoder"),
])
def test_a_fallback_block_that_answers_the_wrong_question_is_refused(
        behaviours, message, monkeypatch):
    import copy

    contract = copy.deepcopy(load_serving_contract())
    contract["native_extensions"][0]["when_unavailable"] = behaviours
    monkeypatch.setattr(ext, "NATIVE_EXTENSIONS", contract["native_extensions"])
    with pytest.raises(ValueError, match=message):
        validate_serving_contract(contract)


# --- the stock-kernel overrides block (contract v54) --------------------------

def test_the_contract_publishes_exactly_the_overrides_this_build_installs():
    contract = load_serving_contract()
    assert contract["stock_kernel_overrides"] == list(ext.STOCK_KERNEL_OVERRIDES)


def _override(**changes):
    """A legible synthetic entry, built from files and modules this build has."""
    entry = {
        "kind": "attention_backend",
        "overrides": {"backend": "FLASHINFER_MLA_SPARSE_SM120",
                      "kernel": "sparse_mla_prefill_mg_kernel"},
        "enabled_by": "TESSERA_SYNTHETIC_OVERRIDE",
        "default": ext.OVERRIDE_DEFAULT_OFF,
        "loaded_by": "tessera.serving.ext",
        "library": {"module_name_prefix": "tessera_synthetic_override_",
                    "filename_glob": "tessera_synthetic_override_*.so",
                    "match": ext.MATCH_BASENAME_FNMATCH,
                    "source": ext.WINDOW_GEMV_SOURCE},
        "required_identity": ext.IDENTITY_BITWISE_VS_STOCK,
        "evidence": [],
    }
    entry.update(changes)
    return entry


def _with_overrides(monkeypatch, *entries):
    """The packaged contract carrying ``entries``, with the authority check stood
    down so the STRUCTURE checks are reached (as for native_extensions above)."""
    import copy

    contract = copy.deepcopy(load_serving_contract())
    contract["stock_kernel_overrides"] = list(entries)
    monkeypatch.setattr(ext, "STOCK_KERNEL_OVERRIDES", contract["stock_kernel_overrides"])
    return contract


@pytest.mark.parametrize("changes", [
    {},
    {"library": None},
    {"evidence": [{"gate": "served TR3", "receipt": "docs/measurements/x.md"}]},
])
def test_a_legible_override_is_admitted(monkeypatch, changes):
    # Each synthetic installed table needs a fresh monkeypatch scope. Reading
    # the real empty packaged contract while a prior synthetic table is still
    # installed correctly fails its authority check.
    validate_serving_contract(_with_overrides(monkeypatch, _override(**changes)))


def test_a_missing_overrides_block_is_refused():
    import copy

    contract = copy.deepcopy(load_serving_contract())
    del contract["stock_kernel_overrides"]
    with pytest.raises(ValueError, match="missing.*stock_kernel_overrides"):
        validate_serving_contract(contract)


def test_an_override_this_build_does_not_install_is_refused():
    import copy

    contract = copy.deepcopy(load_serving_contract())
    contract["stock_kernel_overrides"] = [_override()]
    with pytest.raises(ValueError, match="not what this build installs"):
        validate_serving_contract(contract)


@pytest.mark.parametrize("changes, message", [
    ({"kind": "unquantized_gemm"}, "kind is"),
    ({"overrides": {"backend": "FLASHINFER_MLA_SPARSE_SM120"}}, r"overrides is missing \['kernel'\]"),
    ({"overrides": {"backend": "X", "kernel": "k", "layer": "l"}}, "overrides carries unknown"),
    ({"overrides": {"backend": "", "kernel": "k"}}, "must name the stock object"),
    ({"enabled_by": "MLA_PREFILL"}, "enabled_by must be the TESSERA_"),
    ({"default": "on"}, "default is 'on'"),
    ({"loaded_by": "tessera.kernel_window_gemv"}, "loaded_by is"),
    ({"required_identity": "allclose"}, "required_identity is 'allclose'"),
    ({"evidence": {}}, "evidence must be a JSON array"),
    ({"evidence": [{"gate": "served TR3"}]}, r"evidence\[0\] is missing \['receipt'\]"),
    ({"library": {"module_name_prefix": "tessera_synthetic_override_",
                  "filename_glob": "tessera_synthetic_override_*.so",
                  "match": "prefix", "source": ext.WINDOW_GEMV_SOURCE}}, "match is"),
    ({"library": {"module_name_prefix": "tessera_synthetic_override_",
                  "filename_glob": "tessera_synthetic_override_*.so",
                  "match": ext.MATCH_BASENAME_FNMATCH, "source": "csrc/absent.cu"}},
     "not packaged with this build"),
    # A prefix a native_extensions glob already matches: the window GEMV's
    # glob is ``tessera_window_gemv*.so``.
    ({"library": {"module_name_prefix": "tessera_window_gemv_prefill_",
                  "filename_glob": "tessera_window_gemv_prefill_*.so",
                  "match": ext.MATCH_BASENAME_FNMATCH, "source": ext.WINDOW_GEMV_SOURCE}},
     "also matches"),
])
def test_an_illegible_override_is_refused(changes, message, monkeypatch):
    with pytest.raises(ValueError, match=message):
        validate_serving_contract(_with_overrides(monkeypatch, _override(**changes)))


def test_one_flag_installs_one_override(monkeypatch):
    second = _override(library={**_override()["library"],
                                "module_name_prefix": "tessera_synthetic_second_",
                                "filename_glob": "tessera_synthetic_second_*.so"})
    with pytest.raises(ValueError, match="installs two overrides"):
        validate_serving_contract(_with_overrides(monkeypatch, _override(), second))


def test_the_install_path_is_held_to_its_published_entry(monkeypatch):
    """``stock_kernel_override_refusal`` is the install path's whole permission."""
    from tessera.serving.contract import stock_kernel_override_refusal

    entry = _override()
    contract = _with_overrides(monkeypatch, entry)
    validate_serving_contract(contract)
    asking = dict(kind=entry["kind"], overrides=entry["overrides"], loaded_by=entry["loaded_by"],
                  library_prefix=entry["library"]["module_name_prefix"], contract=contract)
    assert stock_kernel_override_refusal("TESSERA_SYNTHETIC_OVERRIDE", **asking) is None
    assert "publishes no stock_kernel_overrides entry enabled by TESSERA_OTHER" in (
        stock_kernel_override_refusal("TESSERA_OTHER", **asking))
    for field, value in (("kind", "unquantized_linear"),
                         ("overrides", {**entry["overrides"], "kernel": "another_kernel"}),
                         ("loaded_by", "tessera.serving.glm53_nope"),
                         ("library_prefix", "tessera_other_")):
        reason = stock_kernel_override_refusal("TESSERA_SYNTHETIC_OVERRIDE",
                                               **{**asking, field: value})
        assert reason is not None and "this install would use" in reason, (field, reason)
    # With no document passed, the PACKAGED contract is the one read, and it
    # publishes nothing under the synthetic flag.
    monkeypatch.undo()
    assert stock_kernel_override_refusal(
        "TESSERA_SYNTHETIC_OVERRIDE", kind="attention_backend", overrides={},
        loaded_by="tessera.serving.ext", library_prefix=None) is not None


# --- one source, one path (#134) ------------------------------------------------

def test_there_is_one_window_gemv_source_and_it_is_the_published_one():
    """One ``window_gemv.cu`` under ``src/``, at the path the contract
    publishes and the loader resolves through ``ext.native_source_path``.  Two
    byte-identical copies were a test away from drifting; one file cannot."""
    import os
    from pathlib import Path

    from tessera.serving import ext
    src = Path(__file__).resolve().parents[1] / "src"
    copies = sorted(p for p in src.rglob("window_gemv.cu"))
    published = Path(ext.native_source_path(ext.WINDOW_GEMV_MODULE_NAME))
    assert copies == [published], copies
    assert published == Path(ext.csrc_dir()) / "window_gemv.cu"
    entry = next(e for e in ext.NATIVE_EXTENSIONS
                 if e["module_name_prefix"] == ext.WINDOW_GEMV_MODULE_NAME)
    assert entry["source"] == ext.WINDOW_GEMV_SOURCE == "csrc/window_gemv.cu"
    assert os.path.isfile(published)


def test_native_source_path_resolves_every_published_source_and_nothing_else():
    import os

    from tessera.serving import ext
    for entry in ext.NATIVE_EXTENSIONS:
        path = ext.native_source_path(entry["module_name_prefix"])
        assert path == os.path.join(ext.csrc_dir(), *entry["source"].split("/")[1:])
        assert os.path.isfile(path)
    with pytest.raises(KeyError, match="no native extension"):
        ext.native_source_path("tessera_absent")


# ---------------------- an explicit toolkit chosen after torch's import ------


def _fake_toolkit(tmp_path, name: str, version: str):
    """A directory shaped like a CUDA toolkit, with an executable ``bin/nvcc``."""
    root = tmp_path / name
    (root / "bin").mkdir(parents=True)
    nvcc = root / "bin" / "nvcc"
    nvcc.write_text(f"#!/bin/sh\necho 'fake nvcc {version}'\n")
    nvcc.chmod(0o755)
    return root


# -------- an explicit toolkit chosen after torch's import (issue #298) -------
#
# ``CUDA_HOME``/``CUDA_PATH`` in the environment is an operator NAMING a
# toolkit, and ``ext.py``'s TOOLCHAIN note says that choice always wins.  It
# can only win if it reaches the mechanism the build reads: ``load()`` takes
# its nvcc from ``cpp_extension.CUDA_HOME``, a module global torch freezes at
# IMPORT, so a choice made after that import is adopted into that global as
# well as the environment or it is a report about a compiler nothing runs.
# The prior cached root is what decides the two shapes of that mismatch -- a
# complete one silently builds with the previous toolkit, an incomplete one
# fails the build while the resolver reports a complete selected toolkit -- so
# both are cases here.  CPU-only and fully mocked: fake toolkits with
# executable ``bin/nvcc`` scripts, nothing is compiled.


def _incomplete_toolkit(tmp_path, name: str):
    """A toolkit root that EXISTS and holds no compiler (the partial install)."""
    root = tmp_path / name
    (root / "include").mkdir(parents=True)
    return root


def _clear_toolkit_environment(monkeypatch):
    """Unset the toolkit variables, registered so the test restores them.

    ``monkeypatch.delenv`` of an already-absent name records nothing to undo,
    and the resolver SETS ``CUDA_HOME`` -- so a bare ``delenv`` would leak this
    test's adoption into the rest of the session.
    """
    import os

    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))   # _resolve_ninja may prepend
    monkeypatch.delenv("PYTORCH_NVCC", raising=False)
    for var in ("CUDA_HOME", "CUDA_PATH"):
        monkeypatch.setenv(var, "registered-for-restore")
        monkeypatch.delenv(var)


@pytest.mark.parametrize("prior", ["complete", "incomplete", "unset"])
@pytest.mark.parametrize("variable", ["CUDA_HOME", "CUDA_PATH"])
def test_an_explicit_toolkit_chosen_after_torch_import_is_the_one_the_build_runs(
        tmp_path, monkeypatch, prior, variable):
    import os

    torch = pytest.importorskip("torch")   # collectable without it (tessera#309)
    from torch.utils import cpp_extension

    tag = f"{prior}-{variable}"
    selected = _fake_toolkit(tmp_path, f"cuda-selected-{tag}", "SELECTED")
    cached = {
        "complete": lambda: str(_fake_toolkit(tmp_path, f"cuda-old-{tag}", "OLD")),
        "incomplete": lambda: str(_incomplete_toolkit(tmp_path, f"cuda-partial-{tag}")),
        "unset": lambda: None,
    }[prior]()

    _clear_toolkit_environment(monkeypatch)
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", cached)   # frozen at torch's import
    monkeypatch.setenv(variable, str(selected))               # the operator's choice, after

    assert ext._resolve_cuda_home(torch) == str(selected)
    assert ext._nvcc_for_build() == os.path.join(str(selected), "bin", "nvcc"), (
        "the resolver's answer must BE the build's compiler: cpp_extension.load "
        "builds <cpp_extension.CUDA_HOME>/bin/nvcc and never reads the environment")
    assert cpp_extension.CUDA_HOME == str(selected)
    assert os.environ["CUDA_HOME"] == str(selected)
    report = ext.toolchain_report(torch)
    assert report["cuda_home"] == str(selected)
    assert report["nvcc"] == ext._nvcc_for_build()


def test_a_refused_explicit_toolkit_does_not_leave_another_one_compiling(tmp_path, monkeypatch):
    """An explicit root with no ``nvcc`` stays fail-closed -- and the toolkit
    the operator did NOT name does not quietly take its place."""
    import os

    torch = pytest.importorskip("torch")   # collectable without it (tessera#309)
    from torch.utils import cpp_extension

    chosen = _incomplete_toolkit(tmp_path, "cuda-chosen-empty")
    other = _fake_toolkit(tmp_path, "cuda-other", "OTHER")

    _clear_toolkit_environment(monkeypatch)
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", str(other))
    monkeypatch.setenv("CUDA_HOME", str(chosen))

    assert ext._resolve_cuda_home(torch) is None
    assert ext.toolchain_report(torch)["complete"] is False
    assert ext._nvcc_for_build() == os.path.join(str(chosen), "bin", "nvcc"), (
        "a refused explicit selection must not leave the displaced toolkit as the "
        "build's compiler: the operator named a root and the build looks there")
    assert os.environ["CUDA_HOME"] == str(chosen)


def test_mapped_file_device_uses_exact_held_fd_mount(monkeypatch):
    """Kernel superblock device, not a filesystem's virtual st_dev (#915)."""
    from tessera._dev.native_identity import mapped_file_device
    metadata = {"/proc/self/fdinfo/17": "pos:\t0\nmnt_id:\t42\n",
                "/proc/self/mountinfo": "9 1 8:2 / /other rw - ext4 /dev/other rw\n42 1 0:30 / / rw - btrfs /dev/root rw\n"}
    monkeypatch.setattr(Path, "read_text", lambda path: metadata[str(path)])
    assert mapped_file_device(17) == (0, 30)


@pytest.mark.parametrize("fdinfo,mountinfo", [
    ("pos:\t0\n", "42 1 0:30 / / rw - btrfs /dev/root rw\n"),
    ("mnt_id:\t42\nmnt_id:\t43\n", "42 1 0:30 / / rw - btrfs /dev/root rw\n"),
    ("mnt_id:\t42\n", "9 1 8:2 / /other rw - ext4 /dev/other rw\n"),
    ("mnt_id:\t42\n", "42 1 0:30 / / rw\n42 1 0:30 / / rw\n"),
    ("mnt_id:\t42\n", "42 1 bad:device / / rw\n"),
    ("mnt_id:\t42\n", "42 1 30 / / rw\n"),
])
def test_mapped_file_device_refuses_missing_or_ambiguous_mount_provenance(monkeypatch, fdinfo, mountinfo):
    from tessera._dev.native_identity import mapped_file_device
    metadata = {"/proc/self/fdinfo/17": fdinfo, "/proc/self/mountinfo": mountinfo}
    monkeypatch.setattr(Path, "read_text", lambda path: metadata[str(path)])
    with pytest.raises(RuntimeError, match="mount"):
        mapped_file_device(17)
