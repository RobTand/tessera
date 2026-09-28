"""The standalone measurement must not turn a caught workload failure green."""
import ast
import json
import sys
import traceback
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


@pytest.mark.parametrize("family", ["e4m3", "bf16", "e2m1"])
def test_profile_refuses_failed_family(tmp_path, monkeypatch, family):
    # Execute the actual driver without importing CUDA/vLLM into the CPU arm.
    source = Path(__file__).resolve().parents[1] / "experiments/routed_pair_oracle.py"
    tree = ast.parse(source.read_text())
    driver = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "run_profile")
    workspace: Any = ModuleType("vllm.v1.worker.workspace")
    workspace.init_workspace_manager = lambda *args: None
    monkeypatch.setitem(sys.modules, workspace.__name__, workspace)

    def fail_load(*args):
        raise RuntimeError("injected wire-loading failure")

    sampler = SimpleNamespace(start=lambda: None, source="test", window=lambda *args: {},
                              stop_flag=False)
    namespace: dict[str, Any] = {
        "Path": Path, "json": json, "traceback": traceback,
        "PowerSampler": lambda **kwargs: sampler, "provenance": lambda args: {},
        "time": SimpleNamespace(time=lambda: 0, sleep=lambda seconds: None),
        "torch": SimpleNamespace(device=lambda *args: None,
                                 cuda=SimpleNamespace(current_device=lambda: 0)),
        "FAMILIES": {family: {"payload": family, "rung": "fixture"}},
        "load_wires": fail_load, "log": lambda *args: None,
    }
    # Only repository-owned source is executed; no external input enters this AST.
    exec(compile(ast.Module(body=[driver], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    args = SimpleNamespace(out=str(tmp_path), profile_experts=288, layer=3,
                           idle_s=0, families=family, m="1,64,512")
    status = namespace["run_profile"](args)
    report = json.loads((tmp_path / "profiles.json").read_text())
    assert "injected wire-loading failure" in report["families"][family]["error"]
    assert status != 0, "caught profile failure must produce a failing process exit"
