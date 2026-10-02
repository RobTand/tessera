#!/usr/bin/env python3
"""Compare two CUDA cubins section by section, byte for byte.

Usage: cubin_cmp.py IMAGE.cubin REBUILT.cubin --out OUT.json

Reads the ELF64 section table of each cubin and hashes every section's
bytes (NOBITS sections such as .nv.shared.* compare by size). The verdict
rests on the per-kernel .text.<mangled> sections: if every kernel's .text
bytes match, the rebuilt toolchain reproduces the image's SASS exactly,
whatever any disassembler prints. Notes and other metadata are reported
alongside, not gated.
"""
import argparse
import hashlib
import json
import struct
import sys

SHT_NOBITS = 8


def sections(path):
    data = open(path, "rb").read()
    if data[:4] != b"\x7fELF" or data[4] != 2:
        raise SystemExit(f"{path}: not an ELF64 file")
    (e_shoff,) = struct.unpack_from("<Q", data, 0x28)
    e_flags, = struct.unpack_from("<I", data, 0x30)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    hdrs = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        name, typ, flags, addr, offset, size, link, info, align, entsize = \
            struct.unpack_from("<IIQQQQIIQQ", data, off)
        hdrs.append((name, typ, offset, size))
    strtab_off, strtab_size = hdrs[e_shstrndx][2], hdrs[e_shstrndx][3]
    strtab = data[strtab_off:strtab_off + strtab_size]
    out = {}
    for name, typ, offset, size in hdrs[1:]:
        end = strtab.index(b"\0", name)
        sname = strtab[name:end].decode()
        if typ == SHT_NOBITS:
            rec = {"type": "nobits", "size": size}
        else:
            blob = data[offset:offset + size]
            rec = {"type": typ, "size": size,
                   "sha256": hashlib.sha256(blob).hexdigest()}
            if sname.startswith(".note.nv.tkinfo"):
                rec["text"] = blob.replace(b"\0", b"|").decode(
                    "latin-1", "replace")
        if sname in out:
            sname = f"{sname}#{len([k for k in out if k.startswith(sname)])}"
        out[sname] = rec
    return {"e_flags": hex(e_flags), "sections": out,
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("rebuilt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A, B = sections(a.image), sections(a.rebuilt)
    sa, sb = A["sections"], B["sections"]
    rows = []
    for name in sorted(set(sa) | set(sb)):
        ra, rb = sa.get(name), sb.get(name)
        same = (ra is not None and rb is not None
                and ra.get("sha256") == rb.get("sha256")
                and ra.get("size") == rb.get("size"))
        rows.append({"section": name, "equal": same,
                     "image": ra, "rebuilt": rb})
    text = [r for r in rows if r["section"].startswith(".text.")]
    verdict = {
        "image": {"path": a.image, "sha256": A["sha256"], "bytes": A["bytes"],
                  "e_flags": A["e_flags"]},
        "rebuilt": {"path": a.rebuilt, "sha256": B["sha256"],
                    "bytes": B["bytes"], "e_flags": B["e_flags"]},
        "whole_file_equal": A["sha256"] == B["sha256"],
        "kernel_text_sections": len(text),
        "kernel_text_equal": sum(r["equal"] for r in text),
        "kernel_text_all_equal": bool(text) and all(r["equal"] for r in text),
        "kernel_text_differs": [r["section"] for r in text if not r["equal"]],
        "other_sections_differ": [r["section"] for r in rows
                                  if not r["equal"]
                                  and not r["section"].startswith(".text.")],
        "rows": rows,
    }
    json.dump(verdict, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in verdict.items() if k != "rows"},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
