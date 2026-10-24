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
symbol, mode, consumer = sys.argv[3:6]
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
        # Registration collection only imports the enum; no registration runs.
        module.AttentionBackendEnum = type("AttentionBackendEnum", (), {})

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
    str(root / "tests" / consumer), str(peer),
    "--collect-only", "--noconftest", "-q", "-rs", "-p", "no:cacheprovider",
    "--basetemp=" + str(scratch / "pytest"),
], plugins=[reports])
staged_scopes = []
if mode == "present" and status == 0:
    from tessera.serving import mla_sparse_sm120 as adapter
    torch.compiler = SimpleNamespace(is_compiling=lambda: False)
    classes["FlashInferMLASparseSM120Impl"].forward_mqa = (
        lambda self, *args: self._tessera_mla_context)
    metadata = SimpleNamespace(num_decode_tokens=0, num_decodes=0,
                               num_prefills=1, prefill=object())
    for group_type in (classes["HiSparseMLAIndexGroup"],
                       type("StagedSubclass", (classes["HiSparseMLAIndexGroup"],), {})):
        impl = object.__new__(adapter.TesseraMLASparseSM120Impl)
        impl.index_group = group_type()
        previous, layer = object(), object()
        impl._tessera_mla_context = previous
        scope, seen_layer = impl.forward_mqa(object(), object(), metadata, layer)
        staged_scopes.append(dict(scope=scope, layer_preserved=seen_layer is layer,
                                  restored=impl._tessera_mla_context is previous))
print("ISSUE1253=" + json.dumps(dict(
    status=int(status), missing=missing, unrelated=unrelated,
    skips=reports.skips, errors=reports.errors, items=reports.items,
    peer=str(peer.resolve()), staged_scopes=staged_scopes)))
'''


def _collect(tmp_path, symbol, mode, consumer):
    scratch = tmp_path / mode
    scratch.mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(ROOT), str(scratch), symbol, mode, consumer],
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


@pytest.mark.parametrize("consumer,expected_count", [
    ("test_serving_mla_mask_runtime_gpu.py", 2),
    ("test_serving_mla_mask_registration.py", 15),
])
@pytest.mark.parametrize("symbol", ["HiSparseMLAIndexGroup", "FlashInferMLASparseSM120Impl"])
def test_runtime_collection_handles_partial_vllm(tmp_path, symbol, consumer, expected_count):
    runtime = str((ROOT / "tests" / consumer).resolve())
    supported = _collect(tmp_path, symbol, "present", consumer)
    assert supported["status"] == 0, supported
    assert not supported["skips"] and not supported["errors"], supported
    assert sum(item["path"] == runtime for item in supported["items"]) == expected_count, supported
    assert supported["staged_scopes"] == [
        dict(scope=False, layer_preserved=True, restored=True),
        dict(scope=False, layer_preserved=True, restored=True),
    ], supported

    # Reproduce the original import failure without requiring Git history in a
    # fleet snapshot. This frozen boundary has only the old backend guard.
    original = _collect(tmp_path / "unfixed", symbol, "missing",
                        "fixtures/mla_runtime_original_imports.py")
    assert original["status"] != 0 and not original["skips"], original
    assert original["missing"] in "\n".join(original["errors"]), original
    print(f"Unfixed guard {symbol}: status={original['status']}\n"
          + "\n".join(original["errors"]))

    missing = _collect(tmp_path, symbol, "missing", consumer)
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

    broken = _collect(tmp_path, symbol, "broken", consumer)
    assert broken["status"] != 0, "unrelated transitive import failure was hidden"
    assert not broken["skips"], broken
    assert broken["unrelated"] in "\n".join(broken["errors"]), broken
