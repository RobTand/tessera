"""Actual public source admission; PrismaBuild is an explicit dependency."""
import sys
from types import SimpleNamespace
from pathlib import Path
from prismabuild import client
import pytest
from test_routed_lut_experiment_owner import O, ROOT

def test_actual_public_pinned_archive_and_patch_admit_exact_arm_roots(tmp_path):
    sys.path.insert(0, str(ROOT / "experiments/t8r_speed"))
    from pb_staged_store import StagedInputs
    reader = StagedInputs(ROOT / "experiments/configs/routed_lut_842_sources.json")
    try:
        def declared_path(name):
            matches = [e["path"] for e in reader.manifest["entries"]
                       if Path(e["path"]).name == name and e["offset"] == 0]
            assert len(matches) == 1, "source manifest must identify one input"
            return matches[0]
        args = [reader, declared_path("baseline-source.tar.gz"),
                declared_path("readonly.patch")]
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

