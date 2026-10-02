"""A failed or vacuous numerical screen must fail the admitted action."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest








@pytest.fixture
def probe():
    path = Path(__file__).resolve().parents[1] / "experiments/kda/conv_gate.py"
    spec = importlib.util.spec_from_file_location("kda_conv_gate_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ex2_control(probe, tmp_path):
    # Three actual rows from the banked GPU control: +tiny, -tiny, and a
    # positive accumulator whose exponential is subnormal. Files stand in
    # for binding validation only; these CPU tests claim no CUDA execution.
    module = tmp_path / "control.so"
    module.write_bytes(b"CPU identity fixture")
    sass = tmp_path / "control.sass"
    sass.write_bytes(b"CPU SASS identity fixture")
    return {"schema": "tessera.kda_ex2_control.v1", "finite_accumulator_count": 3,
            "all_accumulators_finite": True,
            "raw_fp32_bits": [
                [1, -2147483647, 1065353216, 1065353216, 1073741824, 1073741824, 0, 0],
                [-2147483647, 1, 1065353216, 1065353216, 1073741824, 1073741824, -2147483648, -2147483648],
                [1118766345, -1023639552, 7053949, 0, 1065353216, 1065353216, 1118766345, 1118766345]],
            "raw_bf16_bits": [[0, 0], [-32768, -32768], [17071, 17071]],
            "ex2": {"bit_equal": False, "bits_differing": 1},
            "denominator": {"bit_equal": True, "bits_differing": 0},
            "output_fp32": {"bit_equal": True, "bits_differing": 0},
            "output_bf16": {"bit_equal": True, "bits_differing": 0},
            "input_subnormal_count": 2, "input_subnormal_ex2_is_one_both": True,
            "changed_ex2_is_subnormal_flushed_to_positive_zero": True,
            "ptx_source_sha256": hashlib.sha256(probe.KDA_CONV_PTX_SRC.encode()).hexdigest(),
            "cuda_flags": ["-O3"], "compiled_module": str(module),
            "compiled_module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
            "sass": {"path": str(sass), "sha256": hashlib.sha256(sass.read_bytes()).hexdigest()}}




@pytest.mark.parametrize("fault", ["absent", "raw_shape", "raw_type", "false_count", "vacuous",
                                  "denominator", "fp32_output", "bf16_output", "nonfinite_acc",
                                  "source", "compiled", "sass", "flags"])
def test_ex2_control_fails_closed_on_bad_evidence(probe, tmp_path, fault):
    c = _ex2_control(probe, tmp_path)
    if fault == "absent":
        c = None
    elif fault == "raw_shape":
        c["raw_fp32_bits"][0].pop()
    elif fault == "raw_type":
        c["raw_fp32_bits"][0][0] = True
    elif fault == "false_count":
        c["ex2"]["bits_differing"] = True
    elif fault == "vacuous":
        c["raw_fp32_bits"][2][3] = 7053949
    elif fault == "denominator":
        c["raw_fp32_bits"][2][5] ^= 1
    elif fault == "fp32_output":
        c["raw_fp32_bits"][2][7] ^= 1
    elif fault == "bf16_output":
        c["raw_bf16_bits"][2][1] ^= 1
    elif fault == "nonfinite_acc":
        c["raw_fp32_bits"][0][0] = 0x7f800000
    elif fault == "source":
        c["ptx_source_sha256"] = "0" * 64
    elif fault == "compiled":
        Path(c["compiled_module"]).write_bytes(b"changed fixture")
    elif fault == "sass":
        Path(c["sass"]["path"]).write_bytes(b"changed fixture")
    elif fault == "flags":
        c["cuda_flags"] = ["-O3", "--use_fast_math"]
    assert probe.kdaex2_gate_errors(c)


def _cmp(n):
    return {"bit_equal": n == 0, "bits_differing": n}


def _screen(probe, tmp_path):
    """CPU policy stand-in: complete roster and internally consistent comparisons."""
    cases = []
    for layout in ("SD", "DS"):
        for state_len in (probe.KDA_WIDTH - 1, probe.KDA_WIDTH + 2):
            for name, lens, has, *_ in probe.KDA_PTX_CASES:
                row = {"layout": layout, "state_len": state_len, "case": name,
                       "lens": list(lens), "has": list(has)}
                for mode, label in probe.KDA_PTX_MODES.items():
                    output = int(mode in (1, 3) or (mode == 4 and state_len == 6))
                    state = int(mode in (4, 5) and state_len == 6)
                    row[label] = dict(_cmp(output), qkv={"q": _cmp(output), "k": _cmp(0), "v": _cmp(0)},
                                      conv_state=_cmp(state))
                    if mode:
                        row[label].update(vs_candidate=_cmp(output), state_vs_candidate=_cmp(state))
                cases.append(row)
    return {"gate_contract": "tessera.kda_conv_screen.v2", "p": probe.KDA_P, "width": probe.KDA_WIDTH,
            "served_bit_equal_all": True, "served_output_bit_equal_all": True,
            "served_conv_state_bit_equal_all": True, "cases": cases,
            "mutants_seen": {name: mode != 2 for mode, name in probe.KDA_PTX_MODES.items() if mode},
            "ex2_equivalence": _ex2_control(probe, tmp_path)}


@pytest.mark.parametrize("fault", ["absent", "subset", "duplicate", "wrong_label", "wrong_lens",
                                  "wrong_has", "bad_width", "counterflag", "qkvtotals",
                                  "mutant_summary", "served_summary"])
def test_numerical_gate_derives_summaries_from_complete_case_evidence(probe, tmp_path, fault):
    s = _screen(probe, tmp_path)
    if fault == "absent":
        s.pop("cases")
    elif fault == "subset":
        s["cases"].pop()
    elif fault == "duplicate":
        s["cases"][-1] = s["cases"][0]
    elif fault == "wrong_label":
        s["cases"][0]["case"] = "unknown"
    elif fault == "wrong_lens":
        s["cases"][0]["lens"][0] += 1
    elif fault == "wrong_has":
        s["cases"][0]["has"][0] = False
    elif fault == "bad_width":
        s["width"] = 5
    elif fault == "counterflag":
        s["cases"][0]["served_sass"]["bits_differing"] = 1
    elif fault == "qkvtotals":
        s["cases"][0]["mutant_two_roundings_per_tap"]["qkv"]["q"]["bits_differing"] = 2
    elif fault == "mutant_summary":
        for row in s["cases"]:
            row["mutant_two_roundings_per_tap"] = dict(_cmp(0), qkv={k: _cmp(0) for k in ("q", "k", "v")},
                                                     conv_state=_cmp(0), vs_candidate=_cmp(0), state_vs_candidate=_cmp(0))
    elif fault == "served_summary":
        s["cases"][0]["served_sass"].update(_cmp(1))
        s["cases"][0]["served_sass"]["qkv"]["q"] = _cmp(1)
    assert probe.kdaptx_gate_errors(s)


@pytest.mark.parametrize("owner", ["kda/conv_gate.py", "kda/conv_progress.py"])
def test_each_new_source_owner_changes_and_refuses_the_identity(probe, tmp_path, owner):
    for name in probe.SOURCE_OWNERS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"initial owned source")
    record = probe.harness_source_identity(tmp_path)
    assert probe.harness_source_identity_matches(record, tmp_path)
    (tmp_path / owner).write_bytes(b"changed owner")
    assert not probe.harness_source_identity_matches(record, tmp_path)


def test_gate_and_progress_import_without_scientific_dependencies():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    script = '''import importlib.abc,sys
sys.path.insert(0,sys.argv[1]+"/experiments/kda")
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split(".")[0] in {"torch","vllm","numpy","prismaquant","prismabuild"}: raise ImportError(fullname)
sys.meta_path.insert(0,Block())
import conv_gate,conv_progress
assert conv_gate.kdaptx_gate_errors({})
assert conv_gate.harness_source_identity_matches(conv_gate.harness_source_identity())
assert callable(conv_progress.kda_commit)
'''
    result = subprocess.run([sys.executable, "-I", "-c", script, str(root)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
