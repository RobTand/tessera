"""CPU probe failure custody and real-disassembly syntax regression controls."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "experiments/t4_code"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def probe(monkeypatch):
    sass = load("sass_dump")
    monkeypatch.setitem(sys.modules, "sass_dump", sass)
    return load("f16_prefetch_probe")


def test_missing_compiler_preserves_input_evidence(probe, monkeypatch, tmp_path):
    off, on, out = tmp_path / "off.cu", tmp_path / "on.cu", tmp_path / "evidence"
    off.write_text("// hint off\n")
    on.write_text("// hint on\n")

    def unavailable(args, runner, manifest):
        manifest["tools"] = {"nvcc": {"available": False}}

    monkeypatch.setattr(probe, "configuration", unavailable)
    status = probe.main(["--off", str(off), "--on", str(on),
                         "--off-identity", "recovered", "--on-identity", "reconstruction",
                         "--out", str(out)])
    assert status == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "toolchain/includes unavailable" in manifest["error"]
    assert not manifest["commands"]
    assert (out / "off/off.cu").read_bytes() == b"// hint off\n"
    assert (out / "on/on.cu").read_bytes() == b"// hint on\n"
    assert "-// hint off\n+// hint on\n" in (out / "sources.diff").read_text()
    assert set(manifest["artifacts"]) == {"off/off.cu", "on/on.cu", "sources.diff"}


def test_emitted_underscore_mnemonics_are_valid(probe):
    result = probe.sass_dump.analyze_prefetch_sass('''arch = sm_121
Function : target
/*0000*/ BAR.SYNC.DEFER_BLOCKING 0x5, 0x100 ;
/*0010*/ F2FP.F16.E4M3.UNPACK_B R2, R3 ;
/*0020*/ CCTL.E.PF1 [R16] ;
''', architecture="sm_121")
    assert result["status"] == "complete", result
    assert result["hint_count"] == 1
    assert result["instructions"][0]["pc"] == 0x20


def test_malformed_pc_cannot_hide_hint(probe):
    result = probe.sass_dump.analyze_prefetch_sass('''arch = sm_121
Function : target
/*0000*/ EXIT ;
/*zzzz*/ CCTL.E.PF1 [R16] ;
''', architecture="sm_121")
    assert result["status"] == "inconclusive"
    assert result["hint_count"] is None
    assert any("PC" in error for error in result["errors"])
