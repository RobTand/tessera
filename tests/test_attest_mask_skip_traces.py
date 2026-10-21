"""Fail-closed, read-only audit of per-rank eager MLA dispatch histograms.

These fixtures attest native dispatch and stock fallback presence, NOT decode
execution, numerical correctness, graph coverage, or live serving qualification.
The source pin deliberately parses the emitter; importing it would need vLLM.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "tessera_attest.py"
OWNER = ROOT / "src" / "tessera" / "serving" / "mla_sparse_sm120.py"
TELEMETRY = OWNER.with_name("telemetry.py")

# Explicit consumer vocabulary: the role-specific source pin below binds these
# names to what the real emitter passes, not to arbitrary strings in its file.
WORDS = {
    "NATIVE_DECODER": "native_mg_mask_skip",
    "STOCK_DECODER": "stock",
    "NATIVE_SYMBOL": "tessera_mla_prefill_mg_l0",
    "STOCK_SYMBOL": "stock",
    "NATIVE_SCHEDULE": "mg_mask_skip_pass_buffers",
    "STOCK_SCHEDULE": "stock_mg",
    "MASK_SKIP_KIND": "attention_backend",
    "MASK_SKIP_POLICY": "mla_mask_skip_eager_dispatch",
    "MASK_SKIP_CONTRACT": "stock_kernel_overrides",
    "MASK_SKIP_TRACE_SCHEMA": "tessera.route_trace/1",
}


def _module():
    spec = importlib.util.spec_from_file_location("_mask_skip_attest", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def audit():
    module = _module()

    def call(paths, world_size):
        function = getattr(module, "audit_mla_mask_skip_traces", None)
        assert callable(function), "tessera_attest must expose audit_mla_mask_skip_traces(paths, world_size)"
        return function(paths, world_size=world_size)

    return call


def _entry(native=True, **overrides):
    entry = {
        "kind": "attention_backend", "policy": "mla_mask_skip_eager_dispatch",
        "contract": "stock_kernel_overrides",
        "decoder": "native_mg_mask_skip" if native else "stock",
        "symbol": "tessera_mla_prefill_mg_l0" if native else "stock",
        "shape": "T128:H32:D512" if native else "T1:H32:D512",
        "launches": 3 if native else 5,
        "modules": 1, "module_names": ["model.layers.0.self_attn"],
        "unnamed_modules": 0, "dispatches_without_prefix": 0,
    }
    # Histograms intentionally have no kernel_schedule or reason field.
    entry.update(overrides)
    return entry


def _trace(rank=0, world_size=2):
    return {
        "schema": "tessera.route_trace/1", "rank": rank,
        "world_size": world_size, "rank_source": "torch.distributed",
        "rank_conflict": None,
        "dispatch_coverage": {
            "python_dispatches_counted": True,
            "torch_compile_tracing_counted": False,
            "cuda_graph_replays_counted": False,
        },
        "entries": [_entry(), _entry(False)],
    }


def _files(tmp_path, documents):
    paths = []
    for index, document in enumerate(documents):
        # The header, not the filename or the input order, identifies the rank.
        path = tmp_path / f"worker-{index + 10}.json"
        path.write_text(json.dumps(document))
        paths.append(path)
    return paths


def _verdict(result, expected):
    assert isinstance(result, dict), "the audit returns a JSON-compatible verdict"
    assert result.get("ok") is expected, result
    problems = result.get("problems")
    assert isinstance(problems, list), result
    if expected:
        assert not problems, result
    else:
        assert problems and all(isinstance(p, str) and p.strip() for p in problems), result


def test_complete_eager_rank_set_is_accepted(audit, tmp_path):
    documents = [_trace(1), _trace(0)]
    # An unrelated route is not evidence for or against this override.
    documents[0]["entries"].append(_entry(policy="another_policy", decoder="another_decoder"))
    paths = _files(tmp_path, documents)
    before = [path.read_bytes() for path in paths]
    _verdict(audit(paths, world_size=2), True)
    assert [path.read_bytes() for path in paths] == before, "audit must not rewrite trace evidence"


def test_unreadable_file_is_refused(audit, tmp_path):
    # A missing file is reliably unreadable even in a root-owned runtime image.
    paths = _files(tmp_path, [_trace(0)]) + [tmp_path / "absent.json"]
    _verdict(audit(paths, world_size=2), False)


@pytest.mark.parametrize("contents", ["{broken", "[]", "null"],
                         ids=["invalid_json", "array_document", "null_document"])
def test_unreadable_document_is_refused(audit, tmp_path, contents):
    paths = _files(tmp_path, [_trace(0), _trace(1)])
    paths[1].write_text(contents)
    _verdict(audit(paths, world_size=2), False)


@pytest.mark.parametrize("case", [
    "foreign_schema", "missing_rank_header", "world_mismatch", "duplicate_rank",
    "unknown_decoder", "missing_native_launch", "missing_stock_launch",
    "missing_native_entry", "missing_stock_entry",
    "counted_graph_replays", "counted_compile_tracing",
    "missing_graph_coverage", "missing_compile_coverage", "rank_conflict",
    "native_symbol_mismatch", "stock_symbol_mismatch", "foreign_contract",
], ids=str)
def test_refusal_shapes(audit, tmp_path, case):
    documents = [_trace(0), _trace(1)]
    bad = documents[1]
    if case == "foreign_schema":
        bad["schema"] = "foreign.route_trace/1"
    elif case == "missing_rank_header":
        del bad["rank"]
    elif case == "world_mismatch":
        bad["world_size"] = 3
    elif case == "duplicate_rank":
        bad["rank"] = 0
    elif case == "unknown_decoder":
        bad["entries"].append(_entry(decoder="future_native_decoder"))
    elif case == "missing_native_launch":
        bad["entries"][0]["launches"] = 0
    elif case == "missing_stock_launch":
        bad["entries"][1]["launches"] = 0
    elif case == "missing_native_entry":
        bad["entries"].pop(0)
    elif case == "missing_stock_entry":
        bad["entries"].pop(1)
    elif case == "counted_graph_replays":
        bad["dispatch_coverage"]["cuda_graph_replays_counted"] = True
    elif case == "counted_compile_tracing":
        bad["dispatch_coverage"]["torch_compile_tracing_counted"] = True
    elif case == "missing_graph_coverage":
        del bad["dispatch_coverage"]["cuda_graph_replays_counted"]
    elif case == "missing_compile_coverage":
        del bad["dispatch_coverage"]["torch_compile_tracing_counted"]
    elif case == "rank_conflict":
        bad["rank_conflict"] = {"rank": 0, "world_size": 2, "source": "env"}
    elif case == "native_symbol_mismatch":
        bad["entries"][0]["symbol"] = "some_other_kernel"
    elif case == "stock_symbol_mismatch":
        bad["entries"][1]["symbol"] = "some_other_fallback"
    elif case == "foreign_contract":
        bad["entries"][0]["contract"] = "unrelated_contract"
    _verdict(audit(_files(tmp_path, documents), world_size=2), False)


def test_missing_rank_file_is_refused(audit, tmp_path):
    _verdict(audit(_files(tmp_path, [_trace(0)]), world_size=2), False)


def test_empty_rank_set_is_refused(audit):
    _verdict(audit([], world_size=2), False)


@pytest.mark.parametrize("world_size", [0, -1, True, "2", 2.5],
                         ids=["zero", "negative", "boolean", "string", "fractional"])
def test_invalid_world_is_refused(audit, tmp_path, world_size):
    documents = [] if world_size == 0 else [_trace(0, world_size), _trace(1, world_size)]
    _verdict(audit(_files(tmp_path, documents), world_size=world_size), False)


@pytest.mark.parametrize("rank", [-1, 2, True, "1", 1.5, [], {}],
                         ids=["negative", "out_of_range", "boolean", "string", "fractional", "list", "dict"])
def test_invalid_rank_is_refused_without_raising(audit, tmp_path, rank):
    documents = [_trace(0), _trace(rank)]
    _verdict(audit(_files(tmp_path, documents), world_size=2), False)


def _source_words(tree):
    """Read actual roles in _emit and its _run_mqa_kernel call sites, never exec."""
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == "TesseraMLASparseSM120Impl")
    emit = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_emit")
    calls = [n for n in ast.walk(emit) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
             and n.func.value.id == "telemetry" and n.func.attr == "emit_route"]
    assert len(calls) == 1, "pin the real telemetry delegation"
    kwargs = {kw.arg: kw.value for kw in calls[0].keywords}
    assert isinstance(kwargs["symbol"], ast.Name) and kwargs["symbol"].id == "symbol"
    words = {f"MASK_SKIP_{name.upper()}": ast.literal_eval(kwargs[name])
             for name in ("kind", "policy", "contract")}
    for field, suffix in (("decoder", "DECODER"), ("kernel_schedule", "SCHEDULE")):
        choice = kwargs[field]
        assert isinstance(choice, ast.IfExp)
        assert isinstance(choice.test, ast.Name) and choice.test.id == "reason"
        words[f"STOCK_{suffix}"] = ast.literal_eval(choice.body)
        words[f"NATIVE_{suffix}"] = ast.literal_eval(choice.orelse)
    run = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_run_mqa_kernel")
    role_symbols = {"STOCK": [], "NATIVE": []}
    for call in ast.walk(run):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name) and call.func.value.id == "self"
                and call.func.attr == "_emit"):
            # _emit(q, symbol[, reason]); both refusal branches must keep stock.
            role = "STOCK" if len(call.args) >= 3 or any(k.arg == "reason" for k in call.keywords) else "NATIVE"
            role_symbols[role].append(ast.literal_eval(call.args[1]))
    for role, symbols in role_symbols.items():
        assert symbols and len(set(symbols)) == 1, (role, symbols)
        words[f"{role}_SYMBOL"] = symbols[0]
    return words


def test_audit_vocabulary_matches_emit_owner_without_vllm():
    # A fresh process avoids assertions about process-wide sys.modules that
    # depend on unrelated earlier tests or whether the host has vLLM installed.
    script = """
import importlib.abc, json, runpy, sys
class NoVllm(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'vllm' or fullname.startswith('vllm.'):
            raise AssertionError('source pin must not import vLLM: ' + fullname)
sys.meta_path.insert(0, NoVllm())
module = runpy.run_path(sys.argv[1], run_name='_audit_pin')
names = json.loads(sys.argv[2])
missing = [name for name in names if name not in module]
assert not missing, 'audit vocabulary constants missing: ' + ', '.join(missing)
print(json.dumps({name: module[name] for name in names}))
"""
    process = subprocess.run([sys.executable, "-c", script, str(TOOL), json.dumps(list(WORDS))],
                             cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    actual = json.loads(process.stdout)
    assert actual == WORDS
    tree = ast.parse(OWNER.read_text())
    expected = _source_words(tree)
    telemetry = ast.parse(TELEMETRY.read_text())
    expected["MASK_SKIP_TRACE_SCHEMA"] = next(
        ast.literal_eval(node.value) for node in telemetry.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "ROUTE_TRACE_SCHEMA" for t in node.targets))
    assert actual == expected

    # Prove that changing an actual decoder, schedule, or executed symbol (not
    # an unrelated occurrence of the same word) invalidates this source pin.
    for field in ("NATIVE_DECODER", "STOCK_DECODER", "NATIVE_SCHEDULE", "STOCK_SCHEDULE",
                  "NATIVE_SYMBOL", "STOCK_SYMBOL"):
        changed = copy.deepcopy(tree)
        for node in ast.walk(changed):
            if isinstance(node, ast.Constant) and node.value == WORDS[field]:
                node.value = "changed_" + node.value
        changed_words = _source_words(changed)
        assert changed_words[field] != actual[field], f"source pin missed {field} drift"


@pytest.mark.parametrize("refuse", [False, True], ids=["accept_complete_pair", "refuse_unknown_decoder"])
def test_cli_mask_skip_traces_has_an_independent_json_verdict(tmp_path, refuse):
    documents = [_trace(0), _trace(1)]
    if refuse:
        documents[1]["entries"].append(_entry(decoder="unknown"))
    paths = _files(tmp_path, documents)
    process = subprocess.run([
        sys.executable, str(TOOL), "--mask-skip-traces", *map(str, paths), "--world-size", "2",
    ], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert process.returncode == (1 if refuse else 0), (process.stdout, process.stderr)
    _verdict(json.loads(process.stdout), not refuse)
    assert "Traceback" not in process.stderr
