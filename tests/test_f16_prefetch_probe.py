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


@pytest.mark.parametrize("line", [
    "/*zzzz*/ CCTL.E.PF1 [R16] ;",
    "/*0010 */ CCTL.E.PF1 [R16] ;",
    "/* 0010*/ @!P1 CCTL.E.PF1 [R16] ;",
    "CCTL.E.PF1 [R16] ;",
    "@P1 CCTL.E.UNKNOWN [R16] ;",
])
def test_malformed_pc_cannot_hide_hint(probe, line):
    result = probe.sass_dump.analyze_prefetch_sass('''arch = sm_121
Function : target
/*0000*/ EXIT ;
''' + line, architecture="sm_121")
    assert result["status"] == "inconclusive"
    assert result["hint_count"] is None
    assert any("PC" in error for error in result["errors"])


def test_comments_and_encoding_continuations_are_not_executable(probe):
    result = probe.sass_dump.analyze_prefetch_sass('''arch = sm_121
Function : target
// CCTL.E.PF1 [R16] ;
/* CCTL.E.UNKNOWN [R16] ; */
/* A multiline annotation:
   CCTL.E.PF1 [R16] ;
*/
// /*0010 */ CCTL.E.PF1 [R16] ;
/*0000*/ EXIT ; /* 0x000000000000794d */
                  /* 0x000fea0003800000 */
''', architecture="sm_121")
    assert result["status"] == "complete", result
    assert result["hint_count"] == 0
    assert result["errors"] == []


def test_kernel_correspondence_across_unequal_source_basenames(probe, tmp_path):
    # Valid Itanium source-name lengths; these are synthetic nvcc-shaped names,
    # not evidence of a fresh compiler run. The following component starts with
    # hex letters, so a greedy hash regex must not consume it or its length.
    names = (
        "_ZN25_GLOBAL__N__a_1_base_cu_b6abcdefILi1EEEvv",
        "_ZN24_GLOBAL__N__c_1_new_cu_d6abcdefILi1EEEvv",
    )
    expected = "_ZN11_GLOBAL__N_6abcdefILi1EEEvv"
    for side, name in zip(("off", "on"), names):
        path = tmp_path / f"{side}.json"
        path.write_text(json.dumps({"kernels": {name: ["EXIT"]}}))
        assert probe.sass_dump._kernels(str(path)) == {expected: ["EXIT"]}


def test_kernel_normalization_preserves_specialization_and_public_names(probe):
    assert probe.sass_dump._kernel_name(
        "_ZN24_GLOBAL__N__c_1_new_cu_d6abcdefILi2EEEvv"
    ) == "_ZN11_GLOBAL__N_6abcdefILi2EEEvv"
    public = "_ZN4user6abcdefILi1EEEvv"
    assert probe.sass_dump._kernel_name(public) == public


@pytest.mark.parametrize("unpaired", [False, True])
def test_paired_packet_requires_corresponding_specializations(probe, monkeypatch, tmp_path, unpaired):
    """Synthetic compiler output tests packet assembly, not CUDA availability."""
    off, on, out = tmp_path / "base.cu", tmp_path / "new.cu", tmp_path / "packet"
    off.write_text("// off\n")
    on.write_text("// on\n")
    names = {
        "off": "_ZN25_GLOBAL__N__a_1_base_cu_b6abcdefILi1EEEvv",
        "on": ("_ZN24_GLOBAL__N__c_1_new_cu_d6abcdefILi2EEEvv" if unpaired else
               "_ZN24_GLOBAL__N__c_1_new_cu_d6abcdefILi1EEEvv"),
    }

    def configuration(args, runner, manifest):
        manifest["config"] = {"ptx_flags": [], "includes": [], "abi": 1}
        manifest["tools"] = {name: {"available": True, "resolved": name}
                             for name in ("nvcc", "ptxas", "cuobjdump")}
        return manifest["config"]

    def command_output(runner, name, argv):
        side, stage = name.split("-", 1)
        stdout, stderr = runner.out / f"{name}.stdout", runner.out / f"{name}.stderr"
        stdout.write_text("")
        stderr.write_text("")
        runner.artifacts.update((stdout, stderr))
        if stage == "dependencies":
            stdout.write_text(f"object: {off if side == 'off' else on}\n")
        elif stage == "nvcc":
            Path(argv[-1]).write_text("prefetch.global.L1 [%rd1];\n" if side == "on" else "")
        elif stage == "ptxas":
            Path(argv[-1]).write_bytes(b"synthetic binary")
        elif stage == "sass":
            extra = "/*0010*/ @!P1 CCTL.E.PF1 [R16] ;\n" if side == "on" else ""
            stdout.write_text(f"arch = sm_121\nFunction : {names[side]}\n/*0000*/ EXIT ;\n{extra}")
        else:
            pytest.fail(f"unexpected command stage: {stage}")
        return True

    monkeypatch.setattr(probe, "configuration", configuration)
    monkeypatch.setattr(probe.Runner, "run", command_output)
    code = probe.main(["--off", str(off), "--on", str(on), "--out", str(out),
                       "--off-identity", "reconstruction", "--on-identity", "reconstruction",
                       "--architecture", "sm_121"])
    packet = json.loads((out / "manifest.json").read_text())
    if unpaired:
        assert code == 1
        assert packet["status"] == "failed"
        assert "unpaired kernels" in packet["error"]
        assert {r["status"] for r in packet["comparison"].values()} == {"only_off", "only_on"}
    else:
        assert code == 0
        assert packet["status"] == "complete"
        assert set(packet["comparison"]) == {"_ZN11_GLOBAL__N_6abcdefILi1EEEvv"}
        assert packet["comparison"]["_ZN11_GLOBAL__N_6abcdefILi1EEEvv"]["status"] == "differs"
    assert packet["sides"]["off"]["analyzer"]["hint_count"] == 0
    assert packet["sides"]["on"]["analyzer"]["hint_count"] == 1
