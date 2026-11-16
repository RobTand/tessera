"""Canonical ELF identities. Only the observed nvcc PID token is masked."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import struct

# Keep the stage, source basename, extension, sequence and token length exact.
TOKEN = re.compile(rb"tmpxft_([0-9a-f]{8})_00000000-6_routed_fused_window\.cudafe1\.cpp(?=\x00|$)")
CANONICAL_FIELDS = ("library_canonical_code_sha256", "cubin_canonical_sha256s")


def packed(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("ascii") + b"\n"


def mask(data):
    """Replace only eight PID digits by ASCII zeroes; return section-relative spans."""
    result, offsets = bytearray(data), []
    for match in TOKEN.finditer(data):
        start, end = match.span(1)
        result[start:end] = b"0" * (end - start)
        offsets.append({"offset": start, "size": end - start, "token": match.group().decode("ascii")})
    return bytes(result), offsets


def symbols(order, wide, records, record):
    strings = records[record["link"]]
    pattern = order + ("IBBHQQ" if wide else "IIIBBH")
    stride = struct.calcsize(pattern)
    raw = record["data"]
    if record["entsize"] != stride or len(raw) % stride:
        raise ValueError("unsupported ELF symbol-table stride")
    for start in range(0, len(raw), stride):
        fields = struct.unpack_from(pattern, raw, start)
        name, info, other, shndx, value, size = fields if wide else (fields[0], *fields[3:], *fields[1:3])
        end = strings["data"].find(b"\0", name)
        if name >= len(strings["data"]) or end < 0:
            raise ValueError("invalid ELF symbol name")
        yield strings["offset"] + name, (strings["data"][name:end], value, size, info, other, shndx)


def canonical_identity(data, cubins=()):
    r"""Hash canonical sections in section-header order, including .strtab.

    Each header is packed([name, sh_type, sh_flags, canonical_payload_size])
    followed by its canonical payload. NOBITS uses sh_size and no payload.
    packed uses ASCII JSON without spaces, followed by one LF. Ordinary
    payloads are byte-exact except mask()'s eight PID digits in the one exact
    observed token form. No other string, hash, instruction or note is masked.

    SHT_SYMTAB and SHT_DYNSYM payloads are packed(sorted(records)), where each
    record is [masked_name, value, size, info, other, shndx]. Names are decoded
    as Latin-1, so every byte is retained. Symbol order alone is immaterial.
    Their associated string-table payloads remain strict, with only PID masking.

    A host .nv_fatbin payload is represented by packed([(extracted_cubin_name,
    canonical_cubin_sha256), ...]) in extraction-name order. The proof checks
    that the retained fatbin is compressed and has no PTX. Raw container bytes,
    size and hash remain recorded; raw compression bytes are never masked.
    Every extracted cubin section is recursively checked under this rule.
    All bytes outside section payloads, including ELF/program/section headers
    and padding, enter a separate layout hash. Layout changes remain strict.
    """
    from experiments.numerics_rows.build import elf_sections

    order, wide, records = elf_sections(data)
    layout, cursor = hashlib.sha256(), 0
    for r in sorted((r for r in records if r["data"]), key=lambda r: r["offset"]):
        if r["offset"] < cursor:
            raise ValueError("overlapping ELF section payloads")
        layout.update(data[cursor:r["offset"]])
        cursor = r["offset"] + len(r["data"])
    layout.update(data[cursor:])
    digest, sections = hashlib.sha256(packed(["layout", layout.hexdigest()])), []
    for record in records:
        raw = record["data"]
        content, offsets = mask(raw)
        method = "token-mask" if offsets else "bytes"
        kind = record["sh_type"]
        if kind in (2, 11):
            rows = []
            for _, (name, *fields) in symbols(order, wide, records, record):
                masked_name, _ = mask(name + b"\0")
                rows.append([masked_name[:-1].decode("latin1"), *fields])
            content, method = packed(sorted(rows)), "sorted-symbol-records"
        elif record["name"] == ".nv_fatbin":
            if not cubins:
                raise ValueError("fatbin identity requires extracted cubins")
            content, method = packed(list(cubins)), "extracted-cubins"
        size = record["sh_size"] if kind == 8 else len(content)
        header = [record["name"], kind, record["sh_flags"], size]
        digest.update(packed(header))
        digest.update(content)
        sections.append({k: record[k] for k in ("index", "name", "sh_type", "sh_flags", "sh_size", "offset")} | {
            "raw_sha256": hashlib.sha256(raw).hexdigest(), "canonical_size": size,
            "canonical_sha256": hashlib.sha256(content).hexdigest(), "method": method, "masked_offsets": offsets})
    return {"sha256": digest.hexdigest(), "layout_sha256": layout.hexdigest(), "sections": sections}


def ranges(indices):
    result = []
    for index in indices:
        index = int(index)
        if result and result[-1][1] == index:
            result[-1][1] += 1
        else:
            result.append([index, index + 1])
    return result


def explain(left, right, left_sections, right_sections):
    """Enumerate every differing file byte. Do not infer a cause from a hash."""
    import numpy as np

    a, b = Path(left).read_bytes(), Path(right).read_bytes()
    aa, bb = np.frombuffer(a, dtype=np.uint8), np.frombuffer(b, dtype=np.uint8)
    length = min(len(a), len(b))
    remaining = np.zeros(max(len(a), len(b)), dtype=bool)
    remaining[:length] = aa[:length] != bb[:length]
    remaining[length:] = True
    differences = []
    for x, y in zip(left_sections, right_sections):
        if x["name"] != y["name"] or x["offset"] != y["offset"] or x["sh_size"] != y["sh_size"]:
            continue  # Layout differences remain explicitly unexplained below.
        start, size = x["offset"], 0 if x["sh_type"] == 8 else x["sh_size"]
        offsets = np.flatnonzero(remaining[start:start + size])
        if not len(offsets):
            continue
        equal = (x["sh_type"], x["sh_flags"], x["canonical_size"], x["canonical_sha256"]) == (
                 y["sh_type"], y["sh_flags"], y["canonical_size"], y["canonical_sha256"])
        differences.append({"section": x["name"], "offsets": ranges(offsets), "bytes": len(offsets),
                            "explained": equal, "method": x["method"], "left_masks": x["masked_offsets"], "right_masks": y["masked_offsets"]})
        if equal:
            remaining[start:start + size] = False
    unexplained = np.flatnonzero(remaining)
    return {"raw_equal": a == b, "sections": differences, "all_differences_explained": not len(unexplained),
            "unexplained_file_offsets": ranges(unexplained), "unexplained_bytes": len(unexplained)}


def negative_controls(host, cubin_data, cubin_names):
    from experiments.numerics_rows.build import elf_sections

    embedded = [(name, canonical_identity(data)["sha256"]) for name, data in zip(cubin_names, cubin_data)]
    baseline = canonical_identity(host, embedded)["sha256"]
    def changed(data, offset):
        result = bytearray(data)
        result[offset] ^= 1
        return bytes(result)
    order, wide, sections = elf_sections(host)
    text = next(r for r in sections if r["name"] == ".text" and r["data"])
    renamed = next(offset for r in sections if r["sh_type"] == 2 for offset, (name, _, size, info, _, _) in
                   symbols(order, wide, sections, r) if name and size and info & 15 == 2 and not TOKEN.search(name + b"\0"))
    _, _, device_sections = elf_sections(cubin_data[0])
    code = next(r for r in device_sections if r["sh_flags"] & 4 and r["data"])
    modified_cubin = changed(cubin_data[0], code["offset"])
    modified = [(cubin_names[0], canonical_identity(modified_cubin)["sha256"]), *embedded[1:]]
    def control(data, offset, canonical_sha):
        return {"verdict": "DIFFERENT" if canonical_sha != baseline else "SAME", "offset": offset,
                "before": data[offset], "after": data[offset] ^ 1, "original_canonical_sha256": baseline,
                "changed_canonical_sha256": canonical_sha}
    return {"host_text_byte": control(host, text["offset"], canonical_identity(changed(host, text["offset"]), embedded)["sha256"]),
            "non_token_symbol_byte": control(host, renamed, canonical_identity(changed(host, renamed), embedded)["sha256"]),
            "cubin_code_byte": {**control(cubin_data[0], code["offset"], canonical_identity(host, modified)["sha256"]),
                                "cubin": cubin_names[0], "original_cubin_canonical_sha256": embedded[0][1],
                                "changed_cubin_canonical_sha256": modified[0][1]}}
