"""P3 must bind the dependency graph and constants to the same binary dump."""
import json
from pathlib import Path

import pytest

from experiments.numerics_rows.common import sha256
from experiments.numerics_rows import sass_slice
from experiments.numerics_rows.sass_slice import Instruction, Slice, constants, instructions, prove

KERNEL = "routed_fused_kernelILi0E"


def sass(name, lines):
    return "Function : " + name + "\n" + "\n".join(f"/*{i * 16:04x}*/ {line};" for i, line in enumerate(lines))


def expression(gf, first, constant="c[0x3][0x0]"):
    return [f"FMUL R{first}, -R{gf}, {constant}",
            f"MUFU.EX2 R{first + 1}, R{first}",
            f"FADD R{first + 2}, R{first + 1}, 1",
            f"MUFU.RCP R{first + 3}, R{first + 2}",
            f"FMUL R{first + 4}, R{gf}, R{first + 3}",
            f"STG.E [R50], R{first + 4}"]


def elf(name, value="3fb8aa3b"):
    return f".nv.constant3.{name}\n0x{value}\n"


def evidence():
    probe = sass("swiglu_probe", ["LDG.E R1, [R40]"] + expression(1, 2))
    library = sass(KERNEL, ["FMNMX R11, R20, c[0x0][0x160], PT", "IADD3 R60, R61, 8, RZ"] + expression(11, 12))
    return library, elf(KERNEL), probe, elf("swiglu_probe")


def test_register_names_and_unrelated_schedule_do_not_count():
    result = prove(*evidence())
    assert result["matched"]
    assert result["mode0_instantiations"] == result["quotients"] == 1
    assert result["constant_operands"]
    assert result["unresolved"] == []


@pytest.mark.parametrize("before,after", (("FMUL R16, R11, R15", "FMUL.FTZ R16, R11, R15"),
    ("FMUL R16, R11, R15", "FMUL R16, R15, R11"),
    ("FADD R14, R13, 1", "FADD R14, R13, 2"),
    ("-R11", "R11"), ("MUFU.EX2", "MUFU.SIN")))
def test_changed_opcode_modifier_immediate_or_operand_position_is_unmatched(before, after):
    library, lelf, probe, pelf = evidence()
    assert before in library
    assert not prove(library.replace(before, after), lelf, probe, pelf)["matched"]


@pytest.mark.parametrize("constant", ("c[0x0][0x160]", "c[0x3][0x4]", "c[0x3][R70]", "R70"))
def test_unresolved_input_is_unmatched(constant):
    library, lelf, probe, pelf = evidence()
    result = prove(library.replace("c[0x3][0x0]", constant), lelf, probe, pelf)
    assert not result["matched"] and result["unresolved"]


def test_same_constant_address_with_different_binary_bytes_is_unmatched():
    library, _, probe, pelf = evidence()
    result = prove(library, elf(KERNEL, "3fb8aa3c"), probe, pelf)
    assert not result["matched"]


def test_missing_constant_section_is_unmatched():
    library, _, probe, pelf = evidence()
    assert not prove(library, "", probe, pelf)["matched"]


def test_every_mode0_instantiation_must_match():
    library, lelf, probe, pelf = evidence()
    second = KERNEL + "Lb1E"
    library += "\n" + sass(second, ["FMNMX R11, R20, c[0x0][0x160], PT"] + expression(11, 12))
    result = prove(library, lelf + elf(second), probe, pelf)
    assert result["matched"] and result["mode0_instantiations"] == 2
    library = library.replace("MUFU.RCP", "MUFU.SQRT", 1)
    assert not prove(library, lelf + elf(second), probe, pelf)["matched"]


def test_unknown_branch_reaching_definition_is_unmatched():
    library, lelf, probe, pelf = evidence()
    library = library.replace("MUFU.RCP R15, R14;", "@P0 BRA 0x80;\n/*0078*/ MUFU.RCP R15, R14;")
    assert not prove(library, lelf, probe, pelf)["matched"]


def test_elf_words_are_data_and_sections_keep_their_function():
    text = "Index Offset Size Name\n1 40 8 .nv.constant3.a\n"
    text += ".nv.constant3.a\n0x3fb8aa3b 0x3f800000\n.nv.constant3.b\n0x40000000\n"
    text += ".section .rela.nv.constant3.b RELA\n0x0 symbol R_CUDA_64 0x0\n"
    parsed = constants(text)
    assert bytes(parsed[3, "a"].values()).hex() == "3baab83f0000803f"
    assert parsed[3, "b"] == {}


def fatbin(arch, identifier, content):
    return f"Fatbin elf code:\n================\narch = {arch}\nidentifier = {identifier}\n\n{content}"


def test_fatbinary_constants_do_not_cross_cubin_boundaries():
    library, lelf, probe, pelf = evidence()
    args = tuple("\n".join(fatbin(arch, "library" if i < 2 else "probe", value)
                            for arch in ("sm_120", "sm_121"))
                 for i, value in enumerate((library, lelf, probe, pelf)))
    result = prove(*args)
    assert result["matched"] and result["mode0_instantiations"] == 2
    changed = args[1].rsplit("0x3fb8aa3b", 1)
    result = prove(args[0], "0x3fb8aa3c".join(changed), *args[2:])
    assert result["status"] == "DIFFERENT" and len(result["differences"]) == 1


def test_barriers_and_uniform_moves_have_known_destination_shapes():
    assert Instruction(0, "", "BAR.SYNC.DEFER_BLOCKING", ("R90", "0x200")).destinations == ()
    assert Instruction(0, "", "UMOV", ("UR4", "0x1")).destinations == ("UR4",)
    assert Instruction(0, "", "IADD3", ("R2", "PT", "PT", "R3", "0x1", "RZ")).destination_count == 3
    assert Instruction(0, "", "LOP3.LUT", ("P0", "RZ", "R3", "0x7fffffff", "R0", "0xc8", "!PT")).destination_count == 2


@pytest.mark.parametrize("opcode,operands,bits", [
    ("HFMA2", ("R2", "-RZ", "RZ", "0.96630859375", "-0.0022525787353515625"), 0x3BBB989D),
    ("HFMA2", ("R2", "RZ", "RZ", "-0.0", "-0.0"), 0),
    ("HFMA2", ("R2", "-RZ", "RZ", "-0.0", "-0.0"), 0x80008000),
    ("IMAD.MOV.U32", ("R2", "RZ", "RZ", "0x3bbb989d"), 0x3BBB989D),
    ("IADD3", ("R2", "PT", "PT", "RZ", "0x3bbb989d", "RZ"), 0x3BBB989D),
    ("MOV", ("R2", "-0x1"), 0xFFFFFFFF),
])
def test_constant_instructions_decode_written_bits(opcode, operands, bits):
    assert sass_slice.immediate32(Instruction(0, "", opcode, operands)) == bits


def constant_evidence():
    probe = sass("swiglu_probe", ["LDG.E R1, [R40]",
        "HFMA2 R2, -RZ, RZ, 0.96630859375, -0.0022525787353515625"] + expression(1, 2, "R2"))
    library = sass(KERNEL, ["MOV R12, 0x3bbb989d", "FMNMX R11, R20, c[0x0][0x160], PT"]
                   + expression(11, 12, "R12"))
    return library, "", probe, ""


def test_equal_written_constants_match_and_one_ulp_difference_is_different():
    library, lelf, probe, pelf = constant_evidence()
    assert prove(library, lelf, probe, pelf)["matched"]
    result = prove(library.replace("0x3bbb989d", "0x3bbb989e"), lelf, probe, pelf)
    assert result["status"] == "DIFFERENT" and result["differences"] and not result["unresolved"]


def test_constant_instruction_with_a_runtime_register_is_unresolved():
    library, lelf, probe, pelf = constant_evidence()
    result = prove(library, lelf, probe.replace("-RZ, RZ", "-R70, RZ"), pelf)
    assert result["status"] == "UNSUPPORTED" and result["unresolved"]


def real_slice(key, kind, text=None):
    root = Path(__file__).parent / "data" / "numerics_rows"
    record = json.loads((root / "provenance.json").read_text())[key]
    path = root / f"{key}.{kind}.sass"
    assert sha256(path) == record[kind]["excerpt_sha256"]
    functions = instructions(path.read_text() if text is None else text)
    name, = functions
    seq = functions[name]
    check = next(i for i, ins in enumerate(seq) if ins.opcode == "FCHK")
    gf = seq[check].operands[1]
    boundary = next(i for i in range(check - 1, -1, -1) if gf in seq[i].destinations)
    branch = next(ins for ins in seq[check + 1:] if ins.opcode.startswith("BRA"))
    join = next(i for i, ins in enumerate(seq) if ins.address == int(branch.operands[-1], 16))
    final = next(ins for ins in reversed(seq[check + 1:seq.index(branch)]) if ins.opcode.startswith(("FFMA", "FMUL")))
    sliced = Slice(functions, {}, name, gf, boundary)
    graph = sliced.root(join, final.destinations[0])
    if kind == "probe":
        assert sass_slice.quotient_regions(functions, name, probe=True) == [(boundary, gf, join, final.destinations[0])]
    return graph, sliced, path.read_text()


@pytest.mark.parametrize("key", ("e2m1", "e4m3mma", "e4m3", "value"))
@pytest.mark.parametrize("kind", ("library", "probe"))
def test_real_ieee_division_includes_fchk_call_ret_and_join(key, kind):
    graph, sliced, text = real_slice(key, kind)
    used = [sliced.nodes[n] for n in sliced.used]
    assert {"branch", "call", "gf"} <= {node[0] for node in used}
    assert {"FCHK", "MUFU.RCP", "MUFU.EX2"} <= {node[1] for node in used if node[0] == "op"}
    changed, _, _ = real_slice(key, kind, text.replace("12582913", "12582912", 1))
    assert graph != changed


def test_real_slice_stops_at_the_clamp_result_and_discards_the_old_up_operand():
    graph, _, text = real_slice("e4m3mma", "library")
    text = text.replace("/*6a30*/", "/*6a20*/ LDS.128 R105, [R70];\n/*6a30*/")
    text = text.replace("FMNMX R105, R105, UR6, PT", "FMNMX R105, R70, c[0x0][0x160], PT")
    changed, _, _ = real_slice("e4m3mma", "library", text)
    assert graph == changed
