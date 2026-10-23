"""CPU-only regression contract for the bounded f16 prefetch decision probe.

These are synthetic sm_121 instruction fixtures, not retained compiler output
or timing evidence.  The probe must separately attest its compiler and inputs.
No test needs CUDA, an external artifact store, or historical git objects.
"""
from pathlib import Path
import importlib.util

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/t4_code/sass_dump.py"
SPEC = importlib.util.spec_from_file_location("f16_probe_sass_dump", SCRIPT)
sass_dump = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sass_dump)
KERNEL = "_Z16f16_one_run_testv"


def _ptx(hint=False, *, kernel=KERNEL):
    # The name/comment deliberately contains the old detector's search word.
    body = "    prefetch.global.L1 [%rd2];\n" if hint else ""
    return (
        ".version 9.0\n.target sm_121\n.address_size 64\n"
        "// prefetch in a comment is not an instruction\n"
        f".visible .entry {kernel}()\n{{\n"
        "    .reg .b64 %rd<3>;\n" + body + "    ret;\n}\n"
    )


def _sass(control=None, *, kernel=KERNEL):
    middle = f"        /*0010*/ {control}; /* 0x0000000000000000 */\n" if control else ""
    return (
        "Fatbin elf code:\n================\narch = sm_121\n"
        f"        Function : {kernel}\n"
        "// prefetch in a comment is not an instruction\n"
        "        /*0000*/ LDG.E R4, [R2.64]; /* 0x0000000000000000 */\n"
        + middle
        + "        /*0020*/ EXIT; /* 0x0000000000000000 */\n"
    )


def _analyze(off_sass=None, on_sass=None, *, off_ptx=None, on_ptx=None,
             token="sm_121", kernel=KERNEL):
    analyze = getattr(sass_dump, "analyze_prefetch_pair", None)
    assert callable(analyze), (
        "f16 probe lacks analyze_prefetch_pair: text matches cannot decide "
        "whether a paired sm_121 binary retains prefetch.global.L1"
    )
    result = analyze(
        _ptx() if off_ptx is None else off_ptx,
        _ptx(True) if on_ptx is None else on_ptx,
        off_sass, on_sass, token=token, kernel=kernel,
    )
    assert isinstance(result, dict)
    assert result.get("reason"), "every disposition must explain its evidence or uncertainty"
    return result


@pytest.mark.parametrize("predicate", ["", "@P0 ", "@!P0 "])
def test_paired_l1_cache_control_retains_hint_and_instruction_excerpt(predicate):
    result = _analyze(_sass(), _sass(predicate + "CCTL.E.PF1 [R6.64]"))
    assert result["status"] == "retained", result
    assert result["ptx_counts"] == {"off": 0, "on": 1}
    assert result["sass_counts"] == {"off": 0, "on": 1}
    # Keep the actual address/opcode/modifiers/operands, not a renamed grep count.
    assert any("0010" in line and predicate + "CCTL.E.PF1 [R6.64]" in line
               for line in result["evidence"]["on"]), result


@pytest.mark.parametrize("control", ["CCTL.IVALL", "CCTL.E.IV [R6.64]",
                                     "CCTL.E.PF2 [R6.64]", "MEMBAR.SC.GPU"])
def test_unrelated_control_is_not_l1_hint(control):
    result = _analyze(_sass(), _sass(control))
    assert result["status"] == "not_observed", result
    assert result["sass_counts"] == {"off": 0, "on": 0}
    if control.startswith("CCTL"):
        assert any(control in line for line in result["evidence"]["on"]), result


def test_complete_absent_hint_is_not_a_text_search_or_a_timing_claim():
    result = _analyze(_sass(), _sass())
    assert result["status"] == "not_observed", result
    assert result["ptx_counts"] == {"off": 0, "on": 1}
    assert result["sass_counts"] == {"off": 0, "on": 0}


def test_background_prefetch_present_in_both_arms_does_not_prove_retention():
    background = _sass("CCTL.E.PF1 [R6.64]")
    result = _analyze(background, background)
    assert result["status"] == "inconclusive", result


@pytest.mark.parametrize("bad", [None, "", "cuobjdump failed: no ELF image",
                                  f"Function : {KERNEL}\n",
                                  f"Function : {KERNEL}\n/*0010*/ CCTL.E.??? [R6.64];\n"])
@pytest.mark.parametrize("arm", ["off", "on"])
def test_missing_empty_or_malformed_disassembly_never_demonstrates_removal(bad, arm):
    before, after = (_sass(), bad) if arm == "on" else (bad, _sass())
    result = _analyze(before, after)
    assert result["status"] == "inconclusive", result


def test_wrong_kernel_cannot_supply_positive_or_negative_evidence():
    result = _analyze(_sass(), _sass("CCTL.E.PF1 [R6.64]", kernel="other"))
    assert result["status"] == "inconclusive", result


def test_unsupported_architecture_is_not_silently_classified_as_sm121():
    result = _analyze(_sass(), _sass("CCTL.E.PF1 [R6.64]"), token="sm_unknown")
    assert result["status"] == "inconclusive", result


@pytest.mark.parametrize("on_ptx", ["", "ptx generation failed", _ptx()])
def test_missing_or_hint_free_ptx_cannot_prove_hint_removal(on_ptx):
    result = _analyze(_sass(), _sass(), on_ptx=on_ptx)
    assert result["status"] == "inconclusive", result


def test_instruction_from_another_kernel_cannot_explain_the_selected_hint():
    result = _analyze(_sass(), _sass() + _sass("CCTL.E.PF1 [R6.64]", kernel="other"))
    assert result["status"] == "not_observed", result
    assert result["sass_counts"]["on"] == 0
