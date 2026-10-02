"""Task842: actual pinned source roots and closed two-arm descriptors."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("lut_owner", ROOT / "experiments/t8r_speed/routed_lut/owner.py")
O = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(O)


def descriptor():
    return {"schema": "tessera.routed_lut_pair.v1", "arms": {
        arm: {"binary": "/mnt/shared/test/" + arm + ".so",
              "binary_sha256": O.BASELINE_BINARY if arm == "A" else "b" * 64,
              "kernel_sha256": O.BASELINE_SOURCE if arm == "A" else O.CANDIDATE_SOURCE,
              "owners": O.OWNERS, "compile_flags": ["-O3"]}
        for arm in ["A", "B"]}}


@pytest.mark.parametrize("arm", ["A", "B"])
def test_closed_arm_descriptor_accepts_declared_identity(arm):
    assert O.arm_descriptor(json.dumps(descriptor()), arm, ["-O3"])["owners"] == O.OWNERS


@pytest.mark.parametrize("mutation", ["source", "binary", "flags", "owners", "extra", "path"])
def test_altered_arm_descriptor_refuses_before_native_load(mutation):
    doc = copy.deepcopy(descriptor()); arm = doc["arms"]["A"]
    if mutation == "source": arm["kernel_sha256"] = O.CANDIDATE_SOURCE
    if mutation == "binary": arm["binary_sha256"] = "b" * 64
    if mutation == "flags": arm["compile_flags"] = []
    if mutation == "owners": arm["owners"] = {}
    if mutation == "extra": arm["unbound"] = True
    if mutation == "path": arm["binary"] = "/mnt/shared/../foreign.so"
    with pytest.raises(ValueError): O.arm_descriptor(json.dumps(doc), "A", ["-O3"])


def test_actual_public_pinned_archive_and_patch_admit_exact_arm_roots(tmp_path):
    sys.path.insert(0, str(ROOT / "experiments/t8r_speed"))
    from pb_staged_store import StagedInputs
    reader = StagedInputs(ROOT / "experiments/configs/routed_lut_842_sources.json")
    try:
        args = [reader, "/mnt/shared/astra-routed-lut-20261002/inputs/baseline-source.tar.gz",
                "/mnt/shared/astra-routed-lut-20261002/inputs/readonly.patch"]
        a = O.source_root(*args, tmp_path / "A", "A")
        b = O.source_root(*args, tmp_path / "B", "B")
        assert O.digest((a / O.KERNEL).read_bytes()) == O.BASELINE_SOURCE
        assert O.digest((b / O.KERNEL).read_bytes()) == O.CANDIDATE_SOURCE
        modules = [SimpleNamespace(__file__=str(a / p)) for p in O.OWNERS]
        modules[-1].native_source_path = lambda name: str(a / O.KERNEL)
        O.verify_imports(a, "A", *modules)
        modules[1].__file__ = str(b / "tessera/routed_fused.py")
        with pytest.raises(ValueError, match="origin"): O.verify_imports(a, "A", *modules)
        (b / O.KERNEL).write_bytes(b"foreign source")
        with pytest.raises(ValueError, match="source owner"): O.verify_root(b, "B")
        assert len(reader.reads) == 4
    finally:
        reader.close()


def test_altered_pinned_source_bytes_refuse_before_extraction(tmp_path):
    reader = SimpleNamespace(read=lambda path: b"altered")
    with pytest.raises(ValueError, match="archive/patch"):
        O.source_root(reader, "archive", "patch", tmp_path / "absent", "A")
    assert not (tmp_path / "absent").exists()
