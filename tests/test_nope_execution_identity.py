"""CPU tests for the NoPE verdict carrier, not serving-equivalence measurements.

The backend module imports vLLM's CUDA runtime. Execute its existing two pure
reporting functions with enum/config doubles instead; exercise the real route
trace writer, and leave the runtime's arithmetic verdict unchanged.
"""
import ast
from enum import Enum, auto
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tessera.serving import graph_equivalence, telemetry


class Graph(Enum):
    NONE = auto()
    FULL = auto()
    FULL_DECODE_ONLY = auto()
    FULL_AND_PIECEWISE = auto()


def reporter(gap):
    path = Path(telemetry.__file__).with_name("glm53_nope.py")
    tree = ast.parse(path.read_text())
    names = {"_graph_mode", "_report_equivalence"}
    body = [node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in body} == names
    namespace = {"__package__": "tessera.serving", "sys": sys,
                 "CUDAGraphMode": Graph, "_REPORTED": set(),
                 "graph_mode": graph_equivalence.graph_mode,
                 "eager_equivalence_gap": lambda config: gap,
                 "_speculative_key": lambda config: None}
    exec(compile(ast.Module(body=[*body], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_report_equivalence"]


def config(graph=Graph.FULL, splits=False):
    return SimpleNamespace(compilation_config=SimpleNamespace(
        mode=SimpleNamespace(name="NONE"), cudagraph_mode=graph,
        splitting_ops_contain_attention=lambda: splits))


@pytest.fixture
def tracing(tmp_path):
    path = tmp_path / "route-trace.json"
    trace = telemetry.start_route_trace(path)
    try:
        yield trace, path
    finally:
        telemetry.stop_route_trace()


@pytest.mark.parametrize("gap", [None, "padded replay differs from eager"])
@pytest.mark.parametrize("splits", [False, True])
def test_report_writes_resolved_mode_and_existing_verdict(tracing, gap, splits):
    trace, path = tracing
    report = reporter(gap)
    report(config(splits=splits))
    trace.flush()
    written = json.loads(path.read_text())
    identity = written["backend_execution_identity"]
    assert identity == {
        "backend": "glm53_nope", "compilation_mode": "NONE",
        "cuda_graph_mode": ("FULL_AND_PIECEWISE" if splits else "FULL_DECODE_ONLY"),
        "eager_equivalence": gap is None, "eager_equivalence_gap": gap,
    }
    assert written["backend_execution_identity_conflict"] is None
    # Even with no launches, the model process's verdict must be persisted.
    assert written["entries"] == []
    report(config(splits=splits))
    trace.flush()
    assert json.loads(path.read_text())["backend_execution_identity"] == identity


def test_unknown_verdict_and_graph_replay_coverage_are_explicit(tracing):
    trace, path = tracing
    written = json.loads(path.read_text())
    assert written["backend_execution_identity"] is None
    assert written["dispatch_coverage"] == {
        "python_dispatches_counted": True,
        "torch_compile_tracing_counted": False,
        "cuda_graph_replays_counted": False,
    }
    assert trace.snapshot()["dispatch_coverage"] == written["dispatch_coverage"]


def test_different_modes_do_not_silently_replace_the_first_identity(tracing):
    trace, path = tracing
    report = reporter(None)
    report(config(graph=Graph.NONE))
    report(config(graph=Graph.FULL))
    trace.flush()
    written = json.loads(path.read_text())
    assert written["backend_execution_identity"]["cuda_graph_mode"] == "NONE"
    assert written["backend_execution_identity_conflict"]["cuda_graph_mode"] == "FULL_DECODE_ONLY"


def test_empty_backend_report_does_not_erase_another_process_histogram(tmp_path):
    path = tmp_path / "route-trace.json"
    existing = {"pid": -1, "entries": [{"launches": 3}]}
    path.write_text(json.dumps(existing))
    trace = telemetry.start_route_trace(path)
    try:
        reporter(None)(config())
        trace.flush()
        assert json.loads(path.read_text()) == existing
        assert trace.snapshot()["backend_execution_identity"] is not None
    finally:
        telemetry.stop_route_trace()


def test_reporting_with_trace_disabled_creates_no_trace(tmp_path):
    telemetry.stop_route_trace()
    reporter(None)(config())
    assert telemetry.route_trace() is None
    assert list(tmp_path.iterdir()) == []
