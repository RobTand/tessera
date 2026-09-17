"""The dense streamed routes' prefill regime, pinned where it is decided.

WHAT THIS FILE IS FOR.  ``fp8_gemv`` and ``bf16_route`` each carry a window-GEMV
lane whose dispatch reads the wire directly at ``M <= GEMV_MAX_M`` and, past it,
decodes a whole ``[rows, columns]`` tile and hands it to ``torch._scaled_mm`` /
``torch.mm``.  The dense routes no longer reach that lane: the compact loader
prepares the packed window bundles (``serving.native_window`` /
``tessera.window_gemm``) and ``apply`` runs the packed GEMM at every M in both
residencies.  The materialising prefill branch therefore survives as an orphan,
and "production does not materialise a dense weight tile" is a claim about
REACHABILITY -- the shape a grep cannot hold and a green suite cannot show.

Each test below fails on the pre-substitution source (``origin/master`` at the
time of writing): the route imports ``fp8_gemv`` and branches on
``layer.tessera_gemv`` / ``layer.tessera_prepared``, ``apply`` reaches
``torch._scaled_mm``, and the load path expands the wire through
``parse_tessera_blob_for_scheme``.  They pass on the native-lane source.  That
pair is the point: a test that cannot fail on the code the replacement retires
cannot hold the replacement.

The reachability is computed from the AST, not from the text: docstrings and
comments name these symbols freely (they are how the lane is explained), so a
textual ban would fail on the explanation rather than on the dispatch.
"""
from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
SERVING = REPO / "src" / "tessera" / "serving"

#: The two dense window routes, by file.  A third family adds its file here.
DENSE_ROUTES = ("fp8_route.py", "bf16_route.py")

#: Every symbol a route reaching a materialised (or per-forward kernel-decoded)
#: weight tile must name -- the stock GEMMs, the reference decoders, the GEMV
#: lane's entry points and the per-layer attributes the old dispatch branched
#: on.  ``parse_tessera_blob_for_scheme`` is the MATERIALISING reader; the
#: compact one is what the native lane loads through.
MATERIALISING_SYMBOLS = frozenset({
    "_scaled_mm",
    "materialize_fp8",
    "materialize_bf16",
    "parse_tessera_blob_for_scheme",
    "shard_parsed_roles",
    "streamed_apply",
    "_gemv_path",
    "_materialised_path",
    "prepare_fp8_gemv",
    "prepare_bf16_gemv",
    "decode_is_gemv",
    "tessera_gemv",
    "tessera_prepared",
    "weight_fp8",
    "weight_bf16",
    "fp8_gemv",
})

#: The method-class members a serve executes: the load path's two halves and
#: the forward.  Reachability starts here, because here is where the serve
#: starts.
ENTRY_POINTS = ("create_weights", "process_weights_after_loading", "apply")

#: The modules that DEFINE the orphaned lane.  Everything else in the serving
#: tree referring to it would be a caller.
LANE_OWNERS = frozenset({"fp8_gemv.py", "bf16_route.py"})

#: The lane's entry points and its per-layer attributes, by name.
LANE_SYMBOLS = frozenset({
    "fp8_gemv",
    "prepare_fp8_gemv",
    "prepare_bf16_gemv",
    "streamed_apply",
    "_gemv_path",
    "_materialised_path",
    "decode_is_gemv",
    "tessera_gemv",
    "GEMV_MODULE_NAME",
})


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _names_in(node: ast.AST) -> set:
    """Every ``Name``/attribute word and imported name under ``node``.

    Docstrings and comments are excluded by construction: they are
    ``Constant``/tokens, not names, and the lane is explained in prose that
    names all of these symbols.
    """
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            out.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            out.add(sub.attr)
        elif isinstance(sub, ast.Import):
            for alias in sub.names:
                out.add(alias.name)
                out.add(alias.name.split(".")[0])
        elif isinstance(sub, ast.ImportFrom):
            if sub.module:
                out.add(sub.module)
                out.add(sub.module.split(".")[-1])
            for alias in sub.names:
                out.add(alias.asname or alias.name)
    return out


def _module_imports_in(tree: ast.Module) -> set:
    """The names this MODULE binds by import, at module level only.

    An import inside a function is reached only by calling that function, so it
    belongs to the function's own symbols and not to the module's: a route that
    keeps ``from tessera.decode import materialize_fp8`` inside its retained
    reference preparation has not reached it from ``apply``.
    """
    out = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module)
                out.add(node.module.split(".")[-1])
            for alias in node.names:
                out.add(alias.asname or alias.name)
    return out


def _entry_nodes(tree: ast.Module):
    """The route builder's method-class entry points, by walking the source."""
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for member in node.body:
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if member.name in ENTRY_POINTS and member.name not in found:
                    found[member.name] = member
    return found


def _reachable(path: Path) -> set:
    """Names a serve can reach from the route's entry points.

    Within-module calls are followed (``apply`` -> ``_materialised_path`` on the
    pre-substitution source); imported names are taken from the file's imports,
    since a module this file imports is reached by importing it.  The walk is
    deliberately conservative in one direction only: it can miss a symbol
    reached through an import it does not follow, never invent one.
    """
    tree = _parse(path)
    module_functions = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    entry = _entry_nodes(tree)
    missing = [name for name in ENTRY_POINTS if name not in entry]
    assert not missing, f"{path.name}: no {missing} in the route's method class"
    seen, todo, names = set(), list(entry.values()), set()
    while todo:
        node = todo.pop()
        if node.name in seen:
            continue
        seen.add(node.name)
        here = _names_in(node)
        names |= here
        for symbol in here:
            callee = module_functions.get(symbol)
            if callee is not None and callee.name not in seen:
                todo.append(callee)
    return names | _module_imports_in(tree)


@pytest.mark.parametrize("name", DENSE_ROUTES)
def test_the_dense_route_cannot_reach_a_materialised_weight_path(name):
    """One assertion per route: the served names include no materialiser.

    Red on the pre-substitution source, where ``apply`` branches on
    ``layer.tessera_gemv``/``layer.tessera_prepared`` and reaches
    ``torch._scaled_mm`` (FP8) or the reference preparation's decode plus
    ``torch.mm`` (BF16).
    """
    reached = _reachable(SERVING / name)
    offenders = sorted(reached & MATERIALISING_SYMBOLS)
    assert not offenders, (
        f"{name}: the route can still reach {offenders} from "
        f"{list(ENTRY_POINTS)}; the packed native window lane is the only "
        "dense path this build wires"
    )
    # And the positive half, so an empty or moved file cannot pass by omission:
    # the native lane's preparer and the symbol it stamps are what is reached.
    assert "prepare_dense_native_module" in reached, name
    assert "WINDOW_GEMM_SYMBOL" in reached, name


def test_the_orphaned_lane_has_no_caller_in_the_serving_tree():
    """The GEMV lane's materialising prefill branch has no production caller.

    The lane's own tests drive it directly, and that is the point: the branch
    is reachable from a test and from nothing else.  A serving module that
    prepared or dispatched it would re-open the materialised prefill path
    without changing a line of the routes, so the import graph is where the
    claim has to be held.
    """
    offenders = {}
    for path in sorted(SERVING.glob("*.py")):
        if path.name in LANE_OWNERS:
            continue
        tree = _parse(path)
        hit = _names_in(tree) & LANE_SYMBOLS
        if hit:
            offenders[path.name] = sorted(hit)
    assert not offenders, (
        "these serving modules reach the orphaned window-GEMV lane: "
        f"{offenders}.  The dense routes serve the packed native window lane; "
        "the GEMV lane's tests are its only callers"
    )


def _install_vllm_stubs(monkeypatch):
    """The two vLLM bases ``create_weights`` reads, and nothing else.

    The same seam ``test_serving_fp8_route.py`` uses: the refusal under test
    happens long before a kernel, so the operator library is not needed to
    reach it.
    """
    class _LinearMethodBase:
        pass

    def _param(data, **_kw):
        return torch.nn.Parameter(data, requires_grad=False)

    linear = types.ModuleType("vllm.model_executor.layers.linear")
    linear.LinearMethodBase = _LinearMethodBase
    parameter = types.ModuleType("vllm.model_executor.parameter")
    parameter.ModelWeightParameter = _param
    parameter.BasevLLMParameter = _param
    for name, mod in (("vllm", types.ModuleType("vllm")),
                      ("vllm.model_executor", types.ModuleType("vllm.model_executor")),
                      ("vllm.model_executor.layers", types.ModuleType("vllm.model_executor.layers")),
                      ("vllm.model_executor.layers.linear", linear),
                      ("vllm.model_executor.parameter", parameter)):
        monkeypatch.setitem(sys.modules, name, mod)


class _Layer(torch.nn.Module):
    """A vLLM ``LinearBase`` stand-in on one rank (its TP coordinates)."""

    tp_rank, tp_size = 0, 1


def _scheme(family):
    if family == "TESSERA_FP8":
        return {"family": family, "grid": "E4M3", "body": "WINDOW", "plane": "CHANNEL",
                "q256": 1024, "rows": 256, "columns": 1024, "wire_bytes": 4096,
                "roles": [["weight", 256]]}
    return {"family": family, "grid": "BF16", "body": "WINDOW", "plane": "CHANNEL",
            "q256": 1792, "rows": 64, "columns": 512, "wire_bytes": 4096,
            "roles": [["weight", 64]]}


@pytest.mark.parametrize("family,builder", [
    ("TESSERA_FP8", "build_tessera_fp8_method"),
    ("TESSERA_BF16", "build_tessera_bf16_method"),
])
def test_the_dense_apply_refuses_without_a_prepared_native_module(monkeypatch, family, builder):
    """A module whose load never ran FAILS CLOSED -- it does not materialise.

    The pre-substitution source answers the same call by decoding the prepared
    planes and running a stock GEMM (a materialised tile per forward), so this
    assertion is the refusal the native lane owns: no ``weight_fp8`` /
    ``weight_bf16`` / ``tessera_prepared`` / ``tessera_gemv`` fallback is read,
    and the message says the materialising path is not wired.
    """
    from tessera.serving import bf16_route, fp8_route, native_ops
    from tessera.serving.lane import MODE_STREAMED

    # The A-side quantiser is the first statement of the FP8 forward and needs
    # the operator library; the refusal under test is the statement after it.
    monkeypatch.setattr(
        native_ops, "native_fp8_quant",
        lambda x: (torch.empty(x.shape, dtype=torch.float8_e4m3fn),
                   torch.empty(x.shape[0], 1, dtype=torch.float32)))

    _install_vllm_stubs(monkeypatch)
    module = {"build_tessera_fp8_method": fp8_route,
              "build_tessera_bf16_method": bf16_route}[builder]
    scheme = _scheme(family)
    method = getattr(module, builder)(scheme, "test.layer", MODE_STREAMED)
    layer = _Layer()
    method.create_weights(layer, input_size_per_partition=scheme["columns"],
                          output_partition_sizes=[r for _, r in scheme["roles"]],
                          input_size=scheme["columns"], output_size=scheme["rows"],
                          params_dtype=torch.bfloat16)
    assert hasattr(layer, "wire_bytes")
    x = torch.randn(16, scheme["columns"], dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="no longer wires"):
        method.apply(layer, x)
