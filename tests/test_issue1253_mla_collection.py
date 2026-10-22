"""Isolated import-gate regressions; synthetic dependencies are not CUDA evidence."""
from pathlib import Path
import json
import subprocess
import sys

import pytest

from child_env import child_env


ROOT = Path(__file__).resolve().parents[1]
PROBE = r'''
import ast
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

root, scratch = Path(sys.argv[1]), Path(sys.argv[2])
symbol, mode = sys.argv[3:5]
source = root / "src/tessera/serving/mla_sparse_sm120.py"
imports = {alias.name: node.module
           for node in ast.walk(ast.parse(source.read_text()))
           if isinstance(node, ast.ImportFrom) and node.module
           and node.module.startswith("vllm.")
           for alias in node.names}
missing = imports[symbol]
unrelated = "issue1253_unrelated_dependency"
classes = {name: type(name, (), {}) for name in (
    "FlashInferMLASparseSM120Backend", "FlashInferMLASparseSM120Impl",
    "HiSparseMLAIndexGroup")}

class VllmProbe(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm" or fullname.startswith("vllm."):
            if fullname == missing and mode == "missing":
                raise ModuleNotFoundError(f"No module named '{fullname}'", name=fullname)
            return importlib.util.spec_from_loader(fullname, self, is_package=True)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if module.__name__ == missing and mode == "broken":
            raise ModuleNotFoundError(f"No module named '{unrelated}'", name=unrelated)
        module.__file__ = str(source)
        module.__dict__.update(classes)

sys.meta_path.insert(0, VllmProbe())
torch = ModuleType("torch")
torch.__spec__ = importlib.util.spec_from_loader("torch", loader=None)
torch.cuda = SimpleNamespace(is_available=lambda: False)
sys.modules["torch"] = torch
native = ModuleType("tessera.serving.mla_prefill")
native.MlaPrefillLibrary = type("MlaPrefillLibrary", (), {})
sys.modules[native.__name__] = native

peer = scratch / "test_collection_peer.py"
peer.write_text("def test_other_gpu_module_can_collect():\n    pass\n")

class Reports:
    def __init__(self):
        self.skips, self.errors, self.items = [], [], []

    def pytest_collectreport(self, report):
        if report.skipped:
            self.skips.append(str(report.longrepr))
        if report.failed:
            self.errors.append(str(report.longrepr))

    def pytest_collection_finish(self, session):
        # Out-of-root files have version-dependent node IDs. Bind the peer to
        # its resolved file AND function instead, retaining node IDs for diagnosis.
        self.items = [dict(path=str(item.path.resolve()), name=item.name,
                           nodeid=item.nodeid) for item in session.items]

reports = Reports()
status = pytest.main([
    str(root / "tests/test_serving_mla_mask_runtime_gpu.py"), str(peer),
    "--collect-only", "--noconftest", "-q", "-rs", "-p", "no:cacheprovider",
    "--basetemp=" + str(scratch / "pytest"),
], plugins=[reports])
print("ISSUE1253=" + json.dumps(dict(
    status=int(status), missing=missing, unrelated=unrelated,
    skips=reports.skips, errors=reports.errors, items=reports.items,
    peer=str(peer.resolve()))))
'''


def _collect(tmp_path, symbol, mode):
    scratch = tmp_path / mode
    scratch.mkdir()
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(ROOT), str(scratch), symbol, mode],
        cwd=ROOT,
        env=child_env(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="",
                      TESSERA_STRICT_CUDA="0"),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [line.removeprefix("ISSUE1253=") for line in result.stdout.splitlines()
            if line.startswith("ISSUE1253=")]
    assert len(rows) == 1, result.stdout + result.stderr
    return json.loads(rows[0])


@pytest.mark.parametrize("symbol", ["HiSparseMLAIndexGroup", "FlashInferMLASparseSM120Impl"])
def test_runtime_collection_handles_partial_vllm(tmp_path, symbol):
    runtime = str((ROOT / "tests/test_serving_mla_mask_runtime_gpu.py").resolve())
    supported = _collect(tmp_path, symbol, "present")
    assert supported["status"] == 0, supported
    assert not supported["skips"] and not supported["errors"], supported
    assert sum(item["path"] == runtime for item in supported["items"]) == 2, supported

    broken = _collect(tmp_path, symbol, "broken")
    assert broken["status"] != 0, "unrelated transitive import failure was hidden"
    assert not broken["skips"], broken
    assert broken["unrelated"] in "\n".join(broken["errors"]), broken

    missing = _collect(tmp_path, symbol, "missing")
    assert missing["status"] == 0, (
        "missing declared MLA prerequisite aborted suite collection: "
        + "\n".join(missing["errors"]))
    assert not missing["errors"], missing
    assert len(missing["skips"]) == 1, missing
    reason = missing["skips"][0]
    assert missing["missing"] in reason or symbol in reason, reason
    assert any(item["path"] == missing["peer"]
               and item["name"] == "test_other_gpu_module_can_collect"
               for item in missing["items"]), missing
    assert not any(item["path"] == runtime for item in missing["items"]), missing
