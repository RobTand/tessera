"""Only an exact nvcc PID token and symbol-record order may be canonicalized."""
import struct

import pytest

from experiments.numerics_rows.canonical import canonical_identity, explain, mask, negative_controls
from tests.test_numerics_rows_receipts import elf_bytes


def token(pid):
    return b"tmpxft_" + pid + b"_00000000-6_routed_fused_window.cudafe1.cpp"


def symbolic(*, pid=b"0000001b", reverse=False, rename=False, value=16, wide=True, order="<", text=b"code"):
    strings = b"\0alpha\0beta\0" + token(pid) + b"\0"
    rows = [(0, 0, 0, 0, 0, 0), (1, value, 4, 18, 0, 2), (7, 32, 4, 18, 0, 2), (12, 0, 0, 4, 0, 65521)]
    if reverse:
        rows[1:] = reversed(rows[1:])
    data = b"".join(struct.pack(order + ("IBBHQQ" if wide else "IIIBBH"),
                  *((name, info, other, shndx, value, size) if wide else (name, value, size, info, other, shndx)))
                  for name, value, size, info, other, shndx in rows)
    if rename:
        strings = strings.replace(b"alpha", b"alphb")
    return elf_bytes(wide=wide, order=order, text=text, strtab=strings,
                     extra_sections=[(".symtab", 2, 0, data, len(data))])


def test_exact_token_mask_records_only_the_eight_pid_digits():
    source = b"\0" + token(b"0000001b") + b"\0"
    changed, offsets = mask(source)
    assert changed == b"\0" + token(b"00000000") + b"\0"
    assert offsets == [{"offset": 8, "size": 8, "token": token(b"0000001b").decode()}]
    sass = b'file "' + token(b"0000001b") + b'"\n'
    assert mask(sass)[0] == b'file "' + token(b"00000000") + b'"\n'
    for unknown in (source.replace(b"-6_", b"-7_"), source.replace(b"window", b"windox"), source.replace(b"0000001b", b"0000001B")):
        assert mask(unknown) == (unknown, [])


@pytest.mark.parametrize("wide,order", [(False, "<"), (False, ">"), (True, "<"), (True, ">")])
def test_pid_and_symbol_record_order_are_the_only_symbol_equivalences(wide, order):
    a = symbolic(wide=wide, order=order)
    equal = symbolic(wide=wide, order=order, pid=b"0000000f", reverse=True)
    assert a != equal and canonical_identity(a)["sha256"] == canonical_identity(equal)["sha256"]
    for changed in (symbolic(wide=wide, order=order, rename=True), symbolic(wide=wide, order=order, value=17)):
        assert canonical_identity(a)["sha256"] != canonical_identity(changed)["sha256"]


@pytest.mark.parametrize("change", [{"text": b"codf"}, {"bss_size": 9},
                                    {"extra_sections": [(".note.gnu.build-id", 7, 0, b"note", 4)]}])
def test_non_token_code_nobits_and_notes_remain_strict(change):
    assert canonical_identity(elf_bytes())["sha256"] != canonical_identity(elf_bytes(**change))["sha256"]


def test_compressed_fatbin_is_bound_to_every_extracted_cubin():
    a = elf_bytes(extra_sections=[(".nv_fatbin", 1, 2, b"compressed-a", 12)])
    b = elf_bytes(extra_sections=[(".nv_fatbin", 1, 2, b"compressed-b", 12)])
    cubes = [("sm_121.cubin", canonical_identity(elf_bytes())["sha256"])]
    assert canonical_identity(a, cubes)["sha256"] == canonical_identity(b, cubes)["sha256"]
    changed = [("sm_121.cubin", canonical_identity(elf_bytes(text=b"bad!"))["sha256"])]
    assert canonical_identity(a, cubes)["sha256"] != canonical_identity(a, changed)["sha256"]
    with pytest.raises(ValueError, match="extracted cubins"):
        canonical_identity(a)


def test_each_requested_negative_control_changes_canonical_identity():
    controls = negative_controls(symbolic(), [elf_bytes()], ["sm_121.cubin"])
    assert set(controls) == {"host_text_byte", "non_token_symbol_byte", "cubin_code_byte"}
    assert all(r["verdict"] == "DIFFERENT" and r["before"] != r["after"] for r in controls.values())


def test_raw_file_audit_does_not_explain_an_unmasked_byte(tmp_path):
    a, b = tmp_path / "a.so", tmp_path / "b.so"
    a.write_bytes(symbolic())
    b.write_bytes(symbolic(pid=b"0000000f", reverse=True))
    left, right = canonical_identity(a.read_bytes()), canonical_identity(b.read_bytes())
    assert explain(a, b, left["sections"], right["sections"])["all_differences_explained"]
    b.write_bytes(symbolic(pid=b"0000000f", reverse=True, text=b"codf"))
    right = canonical_identity(b.read_bytes())
    report = explain(a, b, left["sections"], right["sections"])
    assert not report["all_differences_explained"] and report["unexplained_bytes"] == 1


def test_elf_layout_bytes_are_not_discarded():
    original = elf_bytes()
    changed = bytearray(original)
    changed[18] ^= 1  # e_machine, outside every section payload.
    assert canonical_identity(original)["sha256"] != canonical_identity(bytes(changed))["sha256"]
