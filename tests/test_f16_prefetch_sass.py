"""Static instruction controls for the f16 prefetch decision gate (tessera#1369).

Synthetic cuobjdump fixtures, not GPU/timing evidence. PF1 and IVALL spellings
and the negated predicate/offset form were observed in the retained paired
sm_121 disassembly. Addresses and kernel names below are deliberately synthetic.
No historical checkout, compiler, artifact mount or CUDA device is needed.

Additive API in experiments/t4_code/sass_dump.py:
  analyze_prefetch_sass(text, *, architecture) -> dict
  status: complete | inconclusive
  hint_count: int for complete input, None for inconclusive input
  instructions: cache-control records, each with kernel, pc (integer),
                predicate (string, empty when absent), opcode, operands
  errors: nonempty diagnostics when inconclusive
An absent hint is meaningful only for a valid, recognized disassembly.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/t4_code/sass_dump.py"
SPEC = importlib.util.spec_from_file_location("f16_sass_dump", SCRIPT)
sass_dump = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sass_dump)


# Complete small cuobjdump sections, including an encoding-only continuation.
ABSENT = """
Fatbin elf code:
================
arch = sm_121
        Function : _Z6targetv
.headerflags @\"EF_CUDA_SM121 EF_CUDA_PTX_SM(EF_CUDA_SM121)\"
        /*0000*/ MOV R1, c[0x0][0x28] ; /* 0x00000a0000017a02 */
                                           /* 0x000fc40000000f00 */
        /*0010*/ LDG.E R2, [R4.64] ; /* 0x0000000400027981 */
        /*0020*/ EXIT ; /* 0x000000000000794d */
"""
UNRELATED = ABSENT.replace(
    "/*0020*/ EXIT ;",
    "/*0020*/ CCTL.IVALL ;\n"
    "        /*0030*/ MEMBAR.SC.GPU ;\n"
    "        /*0040*/ DEPBAR.LE SB0, 0x1 ;\n"
    "        /*0050*/ EXIT ;",
)
POSITIVE = UNRELATED.replace(
    "/*0050*/ EXIT ;",
    "/*0050*/ @!P1 CCTL.E.PF1 [R18+0x40] ;\n"
    "        /*0060*/ CCTL.E.PF1 [R16] ;\n"
    "        /*0070*/ EXIT ;",
)


def analyze(text, architecture="sm_121"):
    function = getattr(sass_dump, "analyze_prefetch_sass", None)
    assert callable(function), (
        "Missing opcode-aware analyzer: add analyze_prefetch_sass(text, *, "
        "architecture) to sass_dump; a prefetch text search cannot inspect CCTL modifiers"
    )
    return function(text, architecture=architecture)


def test_positive_cache_control_preserves_instruction_identity():
    result = analyze(POSITIVE)
    assert result["status"] == "complete", result
    assert result["hint_count"] == 2, result
    assert not result["errors"], result
    records = result["instructions"]
    hints = [row for row in records if row["opcode"] == "CCTL.E.PF1"]
    assert len(hints) == 2, result
    assert [(row["kernel"], row["pc"], row["predicate"], row["operands"])
            for row in hints] == [
        ("_Z6targetv", 0x50, "@!P1", "[R18+0x40]"),
        ("_Z6targetv", 0x60, "", "[R16]"),
    ]
    assert any(row["opcode"] == "CCTL.IVALL" for row in records), result


def test_unrelated_control_and_absent_hint_are_not_prefetch():
    # Each fixture is a valid disassembly; neither is missing evidence.
    for text in (ABSENT, UNRELATED):
        result = analyze(text)
        assert result["status"] == "complete", result
        assert result["hint_count"] == 0, result
        assert not result["errors"], result
    result = analyze(UNRELATED)
    assert [row["opcode"] for row in result["instructions"]] == ["CCTL.IVALL"]


def test_comments_symbols_and_other_kernels_do_not_create_or_move_hints():
    decoy = UNRELATED.replace("_Z6targetv", "_Z23prefetch_CCTL_E_PF1_fakev")
    decoy += "\n// prefetch.global.L1 CCTL.E.PF1 [R16];\n"
    decoy += "/* CCTL.E.PF1 [R16]; */\n"
    result = analyze(decoy + POSITIVE)
    assert result["status"] == "complete", result
    assert result["hint_count"] == 2, result
    hints = [row for row in result["instructions"] if row["opcode"] == "CCTL.E.PF1"]
    assert all(row["kernel"] == "_Z6targetv" for row in hints), result


@pytest.mark.parametrize("text", [
    None,
    "",
    "cuobjdump fatal: could not open input file",
    "arch = sm_121\nFunction : _Z6targetv\n",
    # Valid instructions must not hide a truncated, unparsed instruction.
    POSITIVE + "\n        /*0080*/ @!P2 ;\n",
    POSITIVE.replace("CCTL.E.PF1 [R16] ;", "CCTL.E.PF1 [R16]"),
])
def test_missing_or_malformed_disassembly_is_inconclusive(text):
    result = analyze(text)
    assert result["status"] == "inconclusive", result
    assert result["hint_count"] is None, result
    assert result["errors"], result


@pytest.mark.parametrize("architecture,text", [
    ("sm_unknown", POSITIVE),
    ("sm_121", POSITIVE.replace("arch = sm_121", "arch = sm_90")),
    ("sm_121", POSITIVE.replace("CCTL.E.PF1", "CCTL.E.UNKNOWN")),
])
def test_unsupported_or_mismatched_architecture_and_unknown_control_fail_closed(architecture, text):
    result = analyze(text, architecture=architecture)
    assert result["status"] == "inconclusive", result
    assert result["hint_count"] is None, result
    assert result["errors"], result
