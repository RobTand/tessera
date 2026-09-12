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
BF16_ROUTE = "tessera.serving.bf16_route"
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


def test_the_packaged_contract_refuses_nothing_today(monkeypatch):
    """CUDA behaviour is unchanged BY CONSTRUCTION on the shipped document.

    ``contract_version`` 22 publishes platform entries with no ``executes``
    key at all, so every family on every platform reads ``unstated`` and this
    module behaves exactly as it did before the platform axis existed.  The
    day a released contract publishes a null entry, this test is the one that
    says so out loud.
    """
    shipped = contract_module.cached_serving_contract()
    for platform in shipped["lane_eligibility"]["platforms"]:
        for family in ("TESSERA_E4M3_K1", "TESSERA_E2M1_K2", "TESSERA_BF16_K1"):
            state, _ = contract_module.platform_execution_contract(
                family, platform, shipped)
            assert state != contract_module.PLATFORM_UNBACKED, (platform, family)
            assert contract_module.platform_backs(family, platform, shipped)


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


def test_the_bf16_route_never_reaches_native_ops():
    """W16A16 quantizes nothing, so it must not depend on vLLM's quantizers.

    This is what lets the plugin serve ``TESSERA_BF16_K1`` on a vLLM build
    that registers no quantization operators at all -- a ROCm build -- and it
    is a STATIC property, not a claim about one execution: the walk reads
    every import in the closure, including the ones inside functions.

    ``bf16_route`` is reached by name through ``scheme.ROUTES`` rather than by
    a written import, so nothing here draws the edge INTO it.  That direction
    is not what is claimed: the claim is about what the BF16 route can reach
    once entered, and a module cannot execute an import statement that no file
    in its closure contains.
    """
    closure = _import_closure(BF16_ROUTE)
    assert BF16_ROUTE in closure, "the walk found nothing"
    assert NATIVE_OPS not in closure, sorted(closure)


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
