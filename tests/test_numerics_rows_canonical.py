"""Only an exact nvcc PID token may change a symbol record."""
import struct

import pytest

from experiments.numerics_rows.canonical import canonical_identity, explain, mask, negative_controls
from experiments.numerics_rows.build import elf_sections
from tests.test_numerics_rows_receipts import elf_bytes


def token(pid):
    return b"tmpxft_" + pid + b"_00000000-6_routed_fused_window.cudafe1.cpp"


def symbolic(*, pid=b"0000001b", reverse=False, rename=False, value=16, wide=True, order="<", text=b"code", extra_sections=()):
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
                     extra_sections=[(".symtab", 2, 0, data, len(data)), *extra_sections])


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
def test_only_pid_digits_may_change_symbol_records(wide, order):
    a = symbolic(wide=wide, order=order)
    equal = symbolic(wide=wide, order=order, pid=b"0000000f")
    assert a != equal and canonical_identity(a)["sha256"] == canonical_identity(equal)["sha256"]
    for changed in (symbolic(wide=wide, order=order, rename=True), symbolic(wide=wide, order=order, value=17),
                    symbolic(wide=wide, order=order, reverse=True)):
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
    host = symbolic(extra_sections=[(".nv_fatbin", 1, 2, b"compressed-a", 12)])
    controls = negative_controls(host, [elf_bytes()], ["sm_121.cubin"])
    assert set(controls) == {"host_text_byte", "non_token_symbol_byte", "cubin_code_byte"}
    for name, record in controls.items():
        assert record["verdict"] == "DIFFERENT" and record["before"] != record["after"], (name, record)
    device = controls["cubin_code_byte"]
    assert device["original_cubin_canonical_sha256"] != device["changed_cubin_canonical_sha256"]


def test_raw_file_audit_does_not_explain_an_unmasked_byte(tmp_path):
    a, b = tmp_path / "a.so", tmp_path / "b.so"
    a.write_bytes(symbolic())
    b.write_bytes(symbolic(pid=b"0000000f"))
    left, right = canonical_identity(a.read_bytes()), canonical_identity(b.read_bytes())
    assert explain(a, b, left["sections"], right["sections"])["all_differences_explained"]
    b.write_bytes(symbolic(pid=b"0000000f", text=b"codf"))
    right = canonical_identity(b.read_bytes())
    report = explain(a, b, left["sections"], right["sections"])
    assert not report["all_differences_explained"] and report["unexplained_bytes"] == 1


def test_elf_layout_bytes_are_not_discarded():
    original = elf_bytes()
    changed = bytearray(original)
    changed[18] ^= 1  # e_machine, outside every section payload.
    assert canonical_identity(original)["sha256"] != canonical_identity(bytes(changed))["sha256"]


def aliased_elf(*, kind=1, start=0, size=4):
    data = bytearray(elf_bytes(extra_sections=[(".alias", kind, 6, b"", 0)]))
    table = struct.unpack_from("<Q", data, 40)[0]
    offset = struct.unpack_from("<Q", data, table + 2 * 64 + 24)[0]
    struct.pack_into("<QQ", data, table + 5 * 64 + 24, offset + start, size)
    return bytes(data)


def test_exact_payload_alias_keeps_both_full_headers():
    result = canonical_identity(aliased_elf())
    text, alias = result["sections"][2], result["sections"][5]
    assert text["payload_source_section"] == alias["payload_source_section"] == 2
    assert result["alias_groups"] == [{"offset": text["offset"], "size": 4, "sections": [
        {"index": 2, "name": ".text", "sh_type": 1}, {"index": 5, "name": ".alias", "sh_type": 1}]}]
    for field in ("addr", "link", "info", "align", "entsize"):
        assert field in text and field in alias


def test_partial_payload_overlap_names_both_sections_types_and_ranges():
    original = aliased_elf(start=1, size=3)
    text, alias = elf_sections(original)[2][2], elf_sections(original)[2][5]
    with pytest.raises(ValueError) as error:
        canonical_identity(original)
    for record in (text, alias):
        assert f"{record['name']!r} type=0x1 [{record['offset']:#x}, {record['offset'] + record['sh_size']:#x})" in str(error.value)


def test_one_byte_in_an_aliased_payload_changes_identity():
    original = aliased_elf()
    changed = bytearray(original)
    changed[elf_sections(original)[2][2]["offset"]] ^= 1
    assert canonical_identity(original)["sha256"] != canonical_identity(bytes(changed))["sha256"]


@pytest.mark.parametrize("kind,size", [(8, 32), (1, 0)])
def test_nobits_and_zero_size_have_no_file_payload(kind, size):
    result = canonical_identity(aliased_elf(kind=kind, start=1, size=size))
    assert result["alias_groups"] == []
    section = result["sections"][5]
    assert section["sh_size"] == size and section["canonical_size"] == 0
    assert section["payload_source_section"] is None


def test_symbol_swap_with_unchanged_relocation_is_different():
    relocation = struct.pack("<QQ", 0, (1 << 32) | 1)
    sections = [(".rel.text", 9, 0, relocation, len(relocation))]
    original, changed = symbolic(extra_sections=sections), symbolic(reverse=True, extra_sections=sections)
    assert elf_sections(original)[2][-1]["data"] == elf_sections(changed)[2][-1]["data"]
    assert canonical_identity(original)["sha256"] != canonical_identity(changed)["sha256"]


@pytest.mark.parametrize("program", ["", ".version 9.0\n.visible .entry kernel() { ret; }\n"], ids=["header-only", "ptx-entry"])
def test_ptx_fatbin_header_is_not_a_program(tmp_path, monkeypatch, program):
    from experiments.numerics_rows import serving_build

    # Retained cuobjdump -ptx output from f2tBGwuMrUX, exactly 250 bytes.
    header = "\nFatbin elf code:\n================\narch = sm_121\ncode version = [1,8]\nhost = linux\ncompile_size = 64bit\ncompressed\nidentifier = /mnt/shared/flux-jobs/tessera-1717-P-N-A8S-2ab48a03-20261116T150541Z/cell/src/tessera/serving/csrc/routed_fused_window.cu\n"
    assert len(header.encode()) == 250
    host, device, sass = [tmp_path / name for name in ("host.so", "device.cubin", "host.sass")]
    host.write_bytes(symbolic(extra_sections=[(".nv_fatbin", 1, 2, b"compressed-a", 12)]))
    device.write_bytes(elf_bytes())
    sass.write_text("\ncompressed\n")
    cubin = canonical_identity(device.read_bytes())
    identity = canonical_identity(host.read_bytes(), [(device.name, cubin["sha256"])])
    measured = {"snapshot_path": str(host), "library_canonical_code_sha256": identity["sha256"],
                "library_canonical_sections": identity["sections"], "cubin_canonical_sha256s": [cubin["sha256"]],
                "canonical_cubins": [{"path": str(device), "sections": cubin["sections"]}]}
    served = {**measured, "sass": {"path": str(sass)}}
    def cuobjdump(command, *, stdout, check):
        assert command[:2] == ["cuobjdump", "-ptx"] and check
        stdout.write(header + (program if command[2] == str(host) else ""))
    monkeypatch.setattr(serving_build.subprocess, "run", cuobjdump)
    result = serving_build.compare_binaries(measured, served, sass, tmp_path / "comparison")
    assert result["identity_equal"] is (not program)
    assert all(not r["empty"] for r in result["ptx"].values())
    assert all(r["has_program"] is bool(program) for r in result["ptx"].values())
