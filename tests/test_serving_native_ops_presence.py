"""Per-op presence, and the two refusals a quantized route owes a reader.

``native_ops`` used to ask ONE question -- is ``cutlass_scaled_mm``
registered? -- and answer three with it.  That reading is wrong on any build
that ships a subset of vLLM's CUDA quantization operators, which is every ROCm
build: the FP8 quantizer can be present while CUTLASS is absent, and the
sentinel reported the FP8 quantizer missing.  These tests pin the three
predicates separately, and pin the refusal that has to come BEFORE them --
the pinned runtime contract publishing a family as unbacked on this platform,
which is a fact about the route and not about the build's ABI.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tessera.serving import contract as contract_module
from tessera.serving import native_ops
from tessera.serving.ext import NativeKernelUnavailableError

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
NATIVE_OPS = "tessera.serving.native_ops"


def _stub_ops(monkeypatch, **ops):
    """Replace ``torch.ops._C`` with a namespace carrying exactly ``ops``.

    ``SimpleNamespace`` answers ``getattr(..., name, None)`` with the default
    for every name it does not carry, which is how ``torch.ops`` itself
    behaves (``_OpNamespace.__getattr__`` raises ``AttributeError`` for an
    operator this build did not register).  So a stub is a faithful stand-in
    for a build that compiled a subset, and no CUDA device is needed to ask
    the question this module exists to ask.
    """
    monkeypatch.setattr(torch.ops, "_C", SimpleNamespace(**ops), raising=False)


def _no_platform(monkeypatch):
    """No device: the contract has attested nothing about where we are."""
    monkeypatch.setattr(native_ops, "_platform_token", lambda: None)


# --------------------------------------------------------------------------
# The three predicates


def test_a_build_with_only_the_fp8_quantizer_reports_exactly_that(monkeypatch):
    """The case the sentinel got wrong, and the one ROCm actually ships."""
    _stub_ops(monkeypatch, dynamic_per_token_scaled_fp8_quant=lambda *a: None)
    assert native_ops.has_fp8_quant() is True
    assert native_ops.has_cutlass_mm() is False
    assert native_ops.has_fp4_quant() is False


def test_each_predicate_reads_its_own_operator_and_no_other(monkeypatch):
    for present, expected in (
        ("dynamic_per_token_scaled_fp8_quant", (True, False, False)),
        ("scaled_fp4_quant", (False, True, False)),
        ("cutlass_scaled_mm", (False, False, True)),
    ):
        _stub_ops(monkeypatch, **{present: lambda *a: None})
        got = (native_ops.has_fp8_quant(), native_ops.has_fp4_quant(),
               native_ops.has_cutlass_mm())
        assert got == expected, present


def test_a_non_callable_attribute_is_not_an_operator(monkeypatch):
    """Presence is callability: a name bound to data registers nothing."""
    _stub_ops(monkeypatch, dynamic_per_token_scaled_fp8_quant=object())
    assert native_ops.has_fp8_quant() is False


@pytest.mark.skipif(importlib.util.find_spec("vllm") is not None,
                    reason="the guard's fallback import succeeds where vLLM is installed, "
                           "so this box cannot tell the two readings apart")
def test_the_library_counts_as_registered_when_any_known_op_is_present(monkeypatch):
    """The sentinel's other half: the guard must not re-import vLLM.

    ``_load_native_ops`` exists to register the namespace.  Keyed on CUTLASS
    alone it tried to import vLLM on a build that had already registered the
    FP8 quantizer -- and where vLLM cannot be imported that raises, so a
    perfectly serviceable FP8 build was refused for the absence of a kernel
    Tessera never calls.  Here the FP8 quantizer is registered and vLLM is
    absent: the old reading raises, the per-op reading returns.
    """
    _stub_ops(monkeypatch, dynamic_per_token_scaled_fp8_quant=lambda *a: None)
    native_ops._load_native_ops("ctx")


def test_every_predicate_the_guard_accepts_is_one_it_can_be_asked_about():
    """``_KNOWN_OPS`` and the three predicates name the same set."""
    assert set(native_ops._KNOWN_OPS) == {
        native_ops.FP8_QUANT_OP, native_ops.FP4_QUANT_OP, native_ops.CUTLASS_MM_OP}


def test_require_fp8_passes_on_a_build_that_registers_only_it(monkeypatch):
    _no_platform(monkeypatch)
    _stub_ops(monkeypatch, dynamic_per_token_scaled_fp8_quant=lambda *a: None)
    native_ops.require_native_fp8_quant("ctx")


def test_require_fp4_still_refuses_a_build_missing_its_operator(monkeypatch):
    """Existing CUDA behaviour: the ABI probe is per operator, as before."""
    _no_platform(monkeypatch)
    _stub_ops(monkeypatch, dynamic_per_token_scaled_fp8_quant=lambda *a: None)
    with pytest.raises(NativeKernelUnavailableError) as excinfo:
        native_ops.require_native_fp4_quant("ctx")
    assert "scaled_fp4_quant" in str(excinfo.value)


# --------------------------------------------------------------------------
# The contract refusal


def _contract_with(executes, platform="gfx1151"):
    """A document whose platform axis says exactly one thing."""
    return {"lane_eligibility": {"platforms": {platform: {
        "backend": "hip", "gcn_arch": platform, "executes": executes}}}}


def _platform(monkeypatch, token, *, backend="hip", contract=None):
    monkeypatch.setattr(native_ops, "_platform_token", lambda: token)
    monkeypatch.setattr(native_ops, "_backend", lambda: backend)
    if contract is not None:
        monkeypatch.setattr(contract_module, "cached_serving_contract", lambda: contract)


@pytest.mark.parametrize("require, family, op", [
    (native_ops.require_native_fp8_quant, "TESSERA_E4M3_K1",
     "dynamic_per_token_scaled_fp8_quant"),
    (native_ops.require_native_fp4_quant, "TESSERA_E2M1_K2", "scaled_fp4_quant"),
])
def test_a_null_executes_entry_refuses_even_when_the_operator_exists(
        monkeypatch, require, family, op):
    """The refusal is the contract's, so a registered kernel cannot mask it.

    Both operators are stubbed present.  If the ABI probe ran first this would
    pass, and the serve would run a route the pinned runtime publishes as
    having no native path on this device.
    """
    _platform(monkeypatch, "gfx1151", contract=_contract_with(
        {"TESSERA_E4M3_K1": None, "TESSERA_E2M1_K2": None,
         "TESSERA_BF16_K1": "bf16_unquantized"}))
    _stub_ops(monkeypatch, dynamic_per_token_scaled_fp8_quant=lambda *a: None,
              scaled_fp4_quant=lambda *a: None, cutlass_scaled_mm=lambda *a: None)
    with pytest.raises(NativeKernelUnavailableError) as excinfo:
        require("loading a Linear")
    message = str(excinfo.value)
    assert "unbacked" in message, message          # the contract word
    assert "hip" in message, message               # the backend
    assert "gfx1151" in message, message           # the platform this is about
    assert family in message, message              # which bytes were refused
    assert op not in message, message              # NOT an ABI diagnosis


def test_a_backed_entry_refuses_nothing(monkeypatch):
    _platform(monkeypatch, "sm_121", backend="cuda", contract=_contract_with(
        {"TESSERA_E4M3_K1": "fp8_per_token_dynamic"}, platform="sm_121"))
    _stub_ops(monkeypatch, dynamic_per_token_scaled_fp8_quant=lambda *a: None)
    native_ops.require_native_fp8_quant("ctx")


def test_an_unlisted_platform_refuses_nothing(monkeypatch):
    """A silence is not a refusal: nothing was attested about this device."""
    _platform(monkeypatch, "gfx1201", contract=_contract_with(
        {"TESSERA_E4M3_K1": None}, platform="gfx1151"))
    _stub_ops(monkeypatch, dynamic_per_token_scaled_fp8_quant=lambda *a: None)
    native_ops.require_native_fp8_quant("ctx")


def test_the_packaged_contract_refuses_only_where_it_attests_an_absence():
    """What the shipped document makes this module do, said out loud.

    At ``contract_version`` 22 this test read the other way: no platform
    entry carried an ``executes`` key, every family on every platform was
    ``unstated``, and the answer was "refuses nothing". v23 is exactly the
    release it was written to catch -- the platform axis lands and two
    platforms publish a null. So the assertion is now per platform, and the
    half that matters for the CUDA path is unchanged: ``sm_121`` is backed
    for all three families, so nothing on an NVIDIA box refuses here.
    """
    shipped = contract_module.cached_serving_contract()
    families = ("TESSERA_E2M1_K2", "TESSERA_E4M3_K1", "TESSERA_BF16_K1")
    states = {
        (platform, family): contract_module.platform_execution_contract(
            family, platform, shipped)[0]
        for platform in shipped["lane_eligibility"]["platforms"]
        for family in families
    }
    for family in families:
        assert states[("sm_121", family)] == contract_module.PLATFORM_BACKED, family
        assert contract_module.platform_backs(family, "sm_121", shipped)
    for platform in ("gfx1151", "gfx1201"):
        assert states[(platform, "TESSERA_BF16_K1")] == contract_module.PLATFORM_BACKED
        for family in ("TESSERA_E2M1_K2", "TESSERA_E4M3_K1"):
            assert states[(platform, family)] == contract_module.PLATFORM_UNBACKED
            assert not contract_module.platform_backs(family, platform, shipped)


def test_an_undeclared_platform_is_unstated_against_the_shipped_document():
    """The third state survives the axis: a box the table does not name is
    not refused, because nothing was attested about it."""
    shipped = contract_module.cached_serving_contract()
    state, contract_name = contract_module.platform_execution_contract(
        "TESSERA_E4M3_K1", "sm_90", shipped)
    assert state == contract_module.PLATFORM_UNSTATED
    assert contract_name is None
    assert contract_module.platform_backs("TESSERA_E4M3_K1", "sm_90", shipped)


def test_the_reader_refuses_a_family_the_contract_does_not_publish():
    with pytest.raises(KeyError):
        contract_module.platform_execution_contract("TESSERA_E4M3_K2", "sm_121")


# --------------------------------------------------------------------------
# The BF16 route reaches none of this


def _module_file(name: str) -> Path | None:
    """The file a dotted in-tree module name resolves to, or ``None``."""
    parts = name.split(".")
    candidate = SRC.joinpath(*parts).with_suffix(".py")
    if candidate.exists():
        return candidate
    package = SRC.joinpath(*parts, "__init__.py")
    return package if package.exists() else None


def _direct_imports(path: Path) -> set[str]:
    """The modules THIS file's import statements name, and only those.

    Deliberately narrower than ``tools/impacted_tests.py``'s graph, and the
    difference is the point.  That tool must never UNDER-select, so it adds
    every reading an import could have -- the bare package prefix of a
    relative import included.  Through that prefix every module in
    ``tessera.serving`` reaches every other one, because importing any of them
    runs the package initializer.  A generous graph is the right instrument
    for choosing which tests to run and the wrong one for a negative claim:
    here the edges have to be the ones the source actually writes.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for name in _named_by(node, path)}


def _named_by(node: ast.Import | ast.ImportFrom, path: Path) -> set[str]:
    """The dotted names one import statement in ``path`` can mean."""
    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names}
    own = path.relative_to(SRC).with_suffix("")
    package = ".".join(own.parts[:-1] if own.name != "__init__" else own.parts)
    if node.level:
        base = package.split(".")
        base = base[:len(base) - (node.level - 1)] if node.level > 1 else base
        prefix = ".".join(base + ([node.module] if node.module else []))
    else:
        prefix = node.module or ""
    if not prefix:
        return set()
    # ``from x import y`` may name a submodule; the caller keeps only the
    # readings that resolve to a file, which is the whole ambiguity resolved.
    return {prefix} | {f"{prefix}.{alias.name}" for alias in node.names}


def _import_closure(start: str) -> set[str]:
    """Every in-tree module reachable from ``start`` through written imports."""
    seen: set[str] = set()
    pending = [start]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        path = _module_file(current)
        if path is None:
            continue
        seen.add(current)
        pending.extend(_direct_imports(path))
    return seen


#: The FP8-quantizer entry points: the per-token quantizer the e4m3 family
#: runs, and the availability check preparation calls.  A BF16 serve on a
#: build that registers no quantization operator (ROCm) survives only while
#: the value family reaches neither.
_FP8_QUANTIZER_SYMBOLS = ("native_fp8_quant", "require_native_fp8_quant")


def test_the_bf16_route_never_calls_the_fp8_quantizer():
    """W16A16 quantizes nothing, so the BF16 path must never CALL vLLM's FP8
    quantizer or its availability check (tessera#543).

    This is what lets the plugin serve ``TESSERA_BF16_K1`` on a vLLM build
    that registers no quantization operators at all -- a ROCm build -- and it
    is a STATIC property, not a claim about one execution.  The previous
    revision drew the property as an import-closure prohibition, which the
    packed native window GEMM broke honestly: ``bf16_route`` prepares through
    ``native_window``, which shares ``window_gemm`` with the e4m3 family, so
    the closure reaches ``native_ops`` through imports the value family never
    executes.  What matters is the call, so the test reads calls:

    * the two files the BF16 route owns (``bf16_route.py``,
      ``native_window.py``) never name either symbol;
    * ``fp8_route.py`` names both -- the positive control that the search
      sees the edge it claims is absent above;
    * in the shared ``window_gemm.py``, every live reference to either
      symbol sits in the ``else`` of a ``X.family == "value"`` branch --
      the e4m3 half -- so the value family cannot reach it whatever the
      imports say.
    """
    owned = {"bf16_route.py": SRC / "tessera" / "serving" / "bf16_route.py",
             "native_window.py": SRC / "tessera" / "serving" / "native_window.py"}
    for name, path in owned.items():
        source = path.read_text(encoding="utf-8")
        for symbol in _FP8_QUANTIZER_SYMBOLS:
            assert symbol not in source, (name, symbol)
    fp8_source = (SRC / "tessera" / "serving" / "fp8_route.py").read_text(encoding="utf-8")
    for symbol in _FP8_QUANTIZER_SYMBOLS:
        assert symbol in fp8_source, symbol
    shared = (SRC / "tessera" / "window_gemm.py").read_text(encoding="utf-8")
    tree = ast.parse(shared, filename="window_gemm.py")
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    references = [node for node in ast.walk(tree)
                  if (isinstance(node, ast.Name) and node.id in _FP8_QUANTIZER_SYMBOLS)
                  or (isinstance(node, ast.Attribute) and node.attr in _FP8_QUANTIZER_SYMBOLS)]
    assert references, "the walk found no quantizer reference to guard"
    for node in references:
        assert _only_behind_a_value_guard(node, parents), ast.dump(node)


def _is_value_guard(test: ast.AST) -> bool:
    """Whether ``test`` is an ``X.family == "value"`` comparison."""
    return (isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.left, ast.Attribute) and test.left.attr == "family"
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "value")


def _only_behind_a_value_guard(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    """Whether ``node`` executes only on the non-value side of a family split.

    Climbs to every enclosing ``if``: the first ``X.family == "value"``
    guard met must hold the node in its ``else`` half -- a node in the
    guard's own body is a value-family call no outer context excuses.
    Guards about anything else are transparent to this question.
    """
    current = node
    while id(current) in parents:
        parent = parents[id(current)]
        if isinstance(parent, ast.If) and _is_value_guard(parent.test):
            return any(current is stmt for stmt in parent.orelse)
        current = parent
    return False


def test_the_quantized_routes_do_reach_it():
    """The negative above is only evidence if the walk can see the edge."""
    for route in ("tessera.serving.fp8_route", "tessera.serving.nvfp4_route"):
        assert NATIVE_OPS in _import_closure(route), route


def test_importing_native_ops_touches_no_native_operator():
    """Being in the same process is not a call.

    The package initializer imports every route, so ``native_ops`` is imported
    whenever anything in ``tessera.serving`` is -- including on a BF16-only
    serve.  That is harmless exactly as far as this test says it is: the
    module's own closure is torch and the plugin's error type, and every read
    of ``torch.ops._C`` in it happens inside a function body.
    """
    path = _module_file(NATIVE_OPS)
    assert path is not None
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    at_module_scope = {node for node in tree.body
                       if isinstance(node, (ast.Import, ast.ImportFrom))}
    in_tree = {name
               for node in at_module_scope
               for name in _named_by(node, path)
               if _module_file(name)}
    assert in_tree == {"tessera.serving.ext"}, (
        "native_ops grew a module-level dependency; a BF16-only serve imports this "
        f"module through the package initializer, so its own imports run too: {in_tree}")
    # The contract read is deliberately NOT one of them: it is written inside
    # ``_require_platform_backs``, so a serve that calls neither ``require_``
    # entry parses no contract on this module's account.
    inside_function = {
        node for function in ast.walk(tree)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(function)
    }
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
                and node.value.attr == "ops"):
            assert node in inside_function, (
                f"torch.ops is read at module scope (line {node.lineno}); importing "
                "native_ops would then touch the operator namespace")
