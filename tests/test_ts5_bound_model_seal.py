"""An explicit served-artifact override carries its matching assembly seal."""
import argparse
import ast
import json
from pathlib import Path

import pytest


DRIVER = Path(__file__).resolve().parents[1] / "experiments/ts5_lfm_served_bound.py"


def _configuration(monkeypatch, arguments):
    """Execute the driver's real CLI setup without telemetry or subprocesses."""
    tree = ast.parse(DRIVER.read_text())
    start = next(i for i, node in enumerate(tree.body)
                 if isinstance(node, ast.FunctionDef)
                 and node.name == "campaign_stage_paths")
    end = next(i for i, node in enumerate(tree.body)
               if isinstance(node, ast.FunctionDef) and node.name == "interrupted")
    monkeypatch.setattr("sys.argv", [str(DRIVER), "census", *arguments])
    namespace = {"Path": Path, "argparse": argparse, "json": json, "__doc__": "test"}
    exec(compile(ast.Module(body=tree.body[start:end], type_ignores=[]),
                 str(DRIVER), "exec"), namespace)
    return namespace


def test_explicit_artifact_pair_replaces_both_defaults(monkeypatch, tmp_path):
    default = _configuration(monkeypatch, [])
    assert default["MODEL"] == default["CAMPAIGN"] / "full-model"
    model, seal = tmp_path / "full-model-r3", tmp_path / "new-seal.json"
    selected = _configuration(monkeypatch, ["--model", str(model), "--seal", str(seal)])
    assert selected["MODEL"] == model
    assert selected["SEAL"] == seal
    assert default["SEAL"] == default["CAMPAIGN"] / "merge-action-r1/artifact-seal.json"


@pytest.mark.parametrize("flag", ["--model", "--seal"])
def test_partial_artifact_override_is_refused_by_pair_gate(monkeypatch, flag):
    with pytest.raises(ValueError, match="--model and --seal must be supplied together"):
        _configuration(monkeypatch, [flag, "/new-artifact"])


def _check_binding(tmp_path, checkpoint, identity):
    """Run the real prelaunch statements through the existing identity gate."""
    tree = ast.parse(DRIVER.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "read_artifact_seal"]
    preflight = next(node.body for node in tree.body
                     if isinstance(node, ast.Try) and node.finalbody)
    end = next(i for i, node in enumerate(preflight)
               if isinstance(node, ast.Assert)
               and "merged checkpoint differs" in ast.unparse(node))
    model = tmp_path / "full-model"
    seal_path = tmp_path / "merge-action-r1/artifact-seal.json"
    seal_path.parent.mkdir()
    seal_path.write_text(json.dumps({"checkpoint": checkpoint,
                                     "checkpoint_identity": identity}))
    reads = []

    def source_identity(path):
        reads.append(path)
        return {"files": {"shard": "actual-bytes"}}

    namespace = {"Path": Path, "json": json, "CAMPAIGN": tmp_path,
                 "MODEL": model, "SEAL": seal_path, "NAME": "unused",
                 "capture": lambda command: "", "gpu_processes": lambda: "",
                 "source_identity": source_identity}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(DRIVER), "exec"), namespace)
    return namespace, reads, ast.Module(body=preflight[:end + 1], type_ignores=[])


@pytest.mark.parametrize("checkpoint", [None, "/different/model"])
def test_seal_path_mismatch_is_refused_before_reading_model(tmp_path, checkpoint):
    namespace, reads, statements = _check_binding(
        tmp_path, checkpoint, {"files": {"shard": "actual-bytes"}})
    with pytest.raises(ValueError, match="seal checkpoint does not match selected model"):
        exec(compile(statements, str(DRIVER), "exec"), namespace)
    assert reads == []


def test_selected_seal_still_requires_exact_checkpoint_identity(tmp_path):
    namespace, reads, statements = _check_binding(
        tmp_path, str(tmp_path / "full-model"), {"files": {"shard": "different-bytes"}})
    with pytest.raises(AssertionError, match="merged checkpoint differs from checked assembly"):
        exec(compile(statements, str(DRIVER), "exec"), namespace)
    assert reads == [tmp_path / "full-model"]
