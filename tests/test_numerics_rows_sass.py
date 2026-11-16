"""P3 must bind the dependency graph and constants to the same binary dump."""
import pytest

from experiments.numerics_rows.sass_slice import prove

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
    return f".nv.constant3.{name}:\n0x00000000 0x{value}\n"


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
