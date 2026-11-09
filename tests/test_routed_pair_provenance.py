"""Run provenance names the Flux job, not a PrismaBuild action (receipt v2)."""
import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "experiments/routed_pair_oracle.py"


def _provenance(monkeypatch):
    tree = ast.parse(SOURCE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "provenance")
    device = SimpleNamespace(get_device_name=lambda index: "fixture-device", get_device_capability=lambda index: (12, 1))
    namespace = {
        "__file__": str(SOURCE), "Path": Path, "os": os, "sys": sys, "socket": SimpleNamespace(gethostname=lambda: "fixture-host"),
        "time": SimpleNamespace(strftime=lambda *a: "T", gmtime=lambda: None),
        "torch": SimpleNamespace(cuda=device, __version__="0"), "src_tree_digest": lambda here: "d",
        "__builtins__": {"__import__": lambda name, *a, **k: SimpleNamespace(__version__="v"), "list": list},
    }
    # Only repository-owned source is executed; no external input enters this AST.
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)  # noqa: S102
    return namespace["provenance"](SimpleNamespace(image="i", tessera_head="h", tessera_state="s"))


@pytest.mark.parametrize("pb_env", [{}, {"PB_ACTION_KEY": "old", "PRISMABUILD_ACTION_KEY": "old"}])
def test_provenance_records_the_flux_job_id_and_never_a_pb_action(monkeypatch, pb_env):
    for name in ("PB_ACTION_KEY", "PRISMABUILD_ACTION_KEY", "FLUX_JOB_ID"):
        monkeypatch.delenv(name, raising=False)
    for name, value in pb_env.items():
        monkeypatch.setenv(name, value)
    assert _provenance(monkeypatch)["flux_job_id"] is None
    monkeypatch.setenv("FLUX_JOB_ID", "f123")
    record = _provenance(monkeypatch)
    assert record["flux_job_id"] == "f123" and "pb_action" not in record


def test_both_receipt_schemas_move_to_v2_together():
    text = SOURCE.read_text()
    assert '"schema": "tessera.routed_pair_oracle.v2"' in text and '"schema": "tessera.routed_pair_profiles.v2"' in text
