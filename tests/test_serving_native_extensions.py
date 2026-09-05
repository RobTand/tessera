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


def _producible_name(expr: ast.expr | None, tree: ast.AST, lineno: int) -> str | None:
    """A module name this call site can produce, or ``None`` if unreadable.

    A literal gives the name exactly.  An f-string gives its leading constant
    segment plus a placeholder for what varies -- which is the whole point:
    ``ext`` builds ``f"{NVFP4_MODULE_PREFIX}{identity}"``, so no exact basename
    exists and the contract publishes a glob.  A bare name is resolved one hop
    to its assignment.
    """
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    if isinstance(expr, ast.JoinedStr):
        out = []
        for piece in expr.values:
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                out.append(piece.value)
                continue
            inner = piece.value if isinstance(piece, ast.FormattedValue) else None
            resolved = _producible_name(inner, tree, lineno) if inner is not None else None
            # A segment that resolves to a constant is part of the name; one
            # computed at run time is what the glob's ``*`` stands for.
            out.append(resolved if resolved is not None else _VARIES)
        return "".join(out)
    if isinstance(expr, ast.Name):
        bound = _assigned_in_scope(tree, expr.id, lineno)
        if bound is None or isinstance(bound, ast.Name):
            return None
        return _producible_name(bound, tree, lineno)
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        left = _producible_name(expr.left, tree, lineno)
        right = _producible_name(expr.right, tree, lineno)
        return None if left is None or right is None else left + right
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
            sites.append({
                "module": module,
                "line": node.lineno,
                "name": _producible_name(_name_argument(node), tree, node.lineno),
            })
    return sites


def _serving_modules() -> list[str]:
    package = SRC / "tessera" / "serving"
    return sorted(f"tessera.serving.{p.stem}" if p.stem != "__init__" else "tessera.serving"
                  for p in package.glob("*.py"))


# --- the table is the load path's, not a copy of it ---------------------------

def test_the_contract_publishes_exactly_what_this_build_loads():
    contract = load_serving_contract()
    assert contract["native_extensions"] == list(ext.NATIVE_EXTENSIONS)


def test_the_published_prefix_is_the_constant_the_load_path_asks_for():
    """Not "agrees with": the same object.

    ``_load_locked`` interpolates ``NVFP4_MODULE_PREFIX`` into the module name,
    so a rename moves the contract with it.  This test reads the source rather
    than the value, because a second literal spelled the same way would pass a
    value comparison and be exactly the drift this closes.
    """
    source = (SRC / "tessera" / "serving" / "ext.py").read_text(encoding="utf-8")
    assert 'module_name = f"{NVFP4_MODULE_PREFIX}{identity}"' in source
    assert ext.NATIVE_EXTENSIONS[0]["module_name_prefix"] == ext.NVFP4_MODULE_PREFIX


def test_the_glob_matches_the_library_torch_actually_writes():
    """The suffix is torch's ``LIB_EXT``, not our guess at a platform."""
    pytest.importorskip("torch")
    from torch.utils import cpp_extension

    entry = ext.NATIVE_EXTENSIONS[0]
    built = f"{entry['module_name_prefix']}{'0' * 64}{cpp_extension.LIB_EXT}"
    assert fnmatch.fnmatch(built, entry["filename_glob"])
    # And an exact-stem reading would match nothing, which is why ``match`` is
    # published as a value instead of left to a consumer's guess.
    assert not fnmatch.fnmatch(built, entry["module_name_prefix"].rstrip("_") + ".so")


def test_the_fallback_the_table_publishes_is_the_one_the_route_takes():
    """``when_unavailable`` is what the route gates on, not a parallel note."""
    pytest.importorskip("torch")
    from tessera.serving.lane import MODE_RESIDENT, MODE_STREAMED
    from tessera.serving.telemetry import DECODERS

    assert ext.substitutes_when_unavailable(MODE_RESIDENT) is True
    assert ext.substitutes_when_unavailable(MODE_STREAMED) is False
    behaviours = ext.NATIVE_EXTENSIONS[0]["when_unavailable"]
    assert behaviours[MODE_RESIDENT]["decoder"] in DECODERS
    # The value the resident fallback actually stamps on its route record.
    from tessera.serving.telemetry import DECODER_TORCH_STOCK
    assert behaviours[MODE_RESIDENT]["decoder"] == DECODER_TORCH_STOCK
    source = (SRC / "tessera" / "serving" / "nvfp4_route.py").read_text(encoding="utf-8")
    assert "allow_torch_fallback=substitutes_when_unavailable(self._mode)" in source


def test_an_unknown_residency_is_a_refusal_not_a_default():
    with pytest.raises(ValueError, match="publishes no behaviour for residency"):
        ext.substitutes_when_unavailable("hybrid")


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
    globs = [entry["filename_glob"] for entry in ext.NATIVE_EXTENSIONS]
    unreadable = [s for s in sites if s["name"] is None]
    assert not unreadable, (
        f"cannot statically read the module name at {unreadable}; a JIT loader that hides its "
        "name from a reader hides it from every consumer of this contract")
    undeclared = [s for s in sites
                  if not any(fnmatch.fnmatch(f"{s['name']}.so", g) for g in globs)]
    assert not undeclared, (
        f"{undeclared} is loadable from tessera.serving and is not in "
        "runtime_contract.json's native_extensions. Either it belongs there -- a library that "
        "can be resident in a serving process is a library a fingerprint must be able to see -- "
        "or the route that reaches it should not.")


def test_every_declared_extension_is_actually_loadable_from_serving():
    """The other direction: no entry outlives the code that loaded it."""
    sites = scan_jit_extension_loads(SRC, _serving_modules())
    for entry in ext.NATIVE_EXTENSIONS:
        assert any(fnmatch.fnmatch(f"{s['name']}.so", entry["filename_glob"])
                   for s in sites if s["name"]), (
            f"{entry['filename_glob']} is published and no reachable code loads it")


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


# ---------------------- the identity names the build's compiler (issue #242) --
#
# torch's JIT loader picks its CUDA compiler by ONE rule
# (torch/utils/cpp_extension._write_ninja_file): ``PYTORCH_NVCC`` when set,
# else ``cpp_extension.CUDA_HOME/bin/nvcc``.  ``$NVCC``/PATH is not consulted.
# The build identity must hash the compiler that rule selects, or two builds
# with different toolkits share a module name -- and an ``NVCC``-only change
# renames a build whose compiler did not move.  No compilation is needed to
# pin the selection: fake toolkits with executable ``bin/nvcc`` scripts are
# enough for ``_compiler_identity`` to resolve and version.


def _fake_toolkit(tmp_path, name: str, version: str):
    import os

    root = tmp_path / name
    (root / "bin").mkdir(parents=True)
    nvcc = root / "bin" / "nvcc"
    nvcc.write_text(f"#!/bin/sh\necho 'fake nvcc {version}'\n")
    nvcc.chmod(0o755)
    return root


def test_the_identity_hashes_the_compiler_the_build_will_invoke(tmp_path, monkeypatch):
    import os

    import torch
    from torch.utils import cpp_extension

    a = _fake_toolkit(tmp_path, "cuda-a", "A")
    b = _fake_toolkit(tmp_path, "cuda-b", "B")
    source = ext.native_source_path(ext.NVFP4_MODULE_PREFIX)
    cc = (12, 1)

    monkeypatch.delenv("PYTORCH_NVCC", raising=False)
    monkeypatch.setenv("NVCC", str(b / "bin" / "nvcc"))      # NOT torch's selector
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", str(a))  # torch's selector

    ident_a, payload_a = ext._build_identity(torch, source=source, capability=cc)
    assert payload_a["nvcc"]["path"] == os.path.realpath(str(a / "bin" / "nvcc")), (
        "the identity must hash the compiler torch's loader selects "
        "(cpp_extension.CUDA_HOME/bin/nvcc), not $NVCC or PATH")

    # a CUDA_HOME change IS a compiler change and must move the identity ...
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", str(b))
    ident_b, payload_b = ext._build_identity(torch, source=source, capability=cc)
    assert payload_b["nvcc"]["path"] == os.path.realpath(str(b / "bin" / "nvcc"))
    assert ident_a != ident_b

    # ... and PYTORCH_NVCC is the loader's first choice, over CUDA_HOME
    monkeypatch.setenv("PYTORCH_NVCC", str(a / "bin" / "nvcc"))
    _, payload_p = ext._build_identity(torch, source=source, capability=cc)
    assert payload_p["nvcc"]["path"] == os.path.realpath(str(a / "bin" / "nvcc"))


def test_an_nvcc_only_environment_change_does_not_rename_the_build(tmp_path, monkeypatch):
    """``$NVCC`` moves nothing torch's loader reads, so it must move nothing
    in the identity either -- pre-#242 it renamed the build namespace while
    the compiler stayed put."""
    import torch
    from torch.utils import cpp_extension

    a = _fake_toolkit(tmp_path, "cuda-a", "A")
    b = _fake_toolkit(tmp_path, "cuda-b", "B")
    source = ext.native_source_path(ext.NVFP4_MODULE_PREFIX)
    monkeypatch.delenv("PYTORCH_NVCC", raising=False)
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", str(a))

    monkeypatch.setenv("NVCC", str(a / "bin" / "nvcc"))
    one, _ = ext._build_identity(torch, source=source, capability=(12, 1))
    monkeypatch.setenv("NVCC", str(b / "bin" / "nvcc"))
    two, _ = ext._build_identity(torch, source=source, capability=(12, 1))
    assert one == two


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
