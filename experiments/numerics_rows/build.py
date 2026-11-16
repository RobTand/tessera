"""Build cell libraries and probes, then save their binary evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

from experiments.numerics_rows.common import IMAGES, KERNEL, check_runtime, check_source, sha256, write_json


def code_identity(path):
    r"""Return the code digest and its ordered ELF section records.

    Visit sections in section-header order, including section zero. Exclude
    only sections named exactly '.strtab'. For each included section, hash
    this header followed immediately by its file payload:

        json.dumps([name, sh_type, sh_flags, sh_size], ensure_ascii=True,
                   separators=(',', ':')).encode('ascii') + b'\n'

    The name is decoded as UTF-8. Integers use unsigned decimal JSON numbers.
    Headers have no spaces; each ends with one LF byte. No other separators
    or bytes enter the digest. SHT_NOBITS has an empty payload, regardless
    of sh_size. Its header still includes sh_size. Each section's sha256
    hashes only its payload, with no header. Offsets and addresses do not
    enter the digest. ELF32 and ELF64 use the same encoding.
    """
    data = Path(path).read_bytes()
    if data[:4] != b"\x7fELF" or len(data) < 16 or data[4] not in (1, 2) or data[5] not in (1, 2):
        raise ValueError("unsupported ELF encoding")
    order, wide = "<" if data[5] == 1 else ">", data[4] == 2

    def unpack(pattern, offset):
        try:
            return struct.unpack_from(order + pattern, data, offset)
        except struct.error as error:
            raise ValueError("truncated ELF header") from error

    def payload(offset, size):
        if offset + size > len(data):
            raise ValueError("truncated ELF section")
        return data[offset:offset + size]

    header = unpack("HHIQQQIHHHHHH" if wide else "HHIIIIIHHHHHH", 16)
    offset, stride, count, names_index = header[5], header[10], header[11], header[12]
    pattern = "IIQQQQIIQQ" if wide else "IIIIIIIIII"
    if not offset or stride < struct.calcsize(order + pattern):
        raise ValueError("missing ELF section table")
    first = unpack(pattern, offset)
    count = count or first[5]
    names_index = first[6] if names_index == 65535 else names_index
    if not 0 < names_index < count or offset + count * stride > len(data):
        raise ValueError("invalid ELF section table")
    table = [unpack(pattern, offset + i * stride) for i in range(count)]
    strings = table[names_index]
    names = payload(strings[4], strings[5])
    digest, sections = hashlib.sha256(), []
    for index, section in enumerate(table):
        name_index, kind, flags, _, start, size = section[:6]
        end = names.find(b"\0", name_index)
        if end < 0:
            raise ValueError("unterminated ELF section name")
        name = names[name_index:end].decode("utf-8")
        if name == ".strtab":
            continue
        content = b"" if kind == 8 else payload(start, size)
        digest.update(json.dumps([name, kind, flags, size], ensure_ascii=True, separators=(",", ":")).encode("ascii") + b"\n")
        digest.update(content)
        sections.append({"index": index, "name": name, "sh_type": kind, "sh_flags": flags,
                         "sh_size": size, "sha256": hashlib.sha256(content).hexdigest()})
    return {"library_code_sha256": digest.hexdigest(), "library_code_sections": sections}


def library_identity(path, out):
    """Extract cubins in a new job-local directory and record their exact hashes."""
    path, out = Path(path).resolve(), Path(out)
    identity = {"library_so_sha256": sha256(path), **code_identity(path)}
    out.mkdir(parents=True)
    with (out / "cuobjdump.stdout").open("w") as stdout, (out / "cuobjdump.stderr").open("w") as stderr:
        subprocess.run(["cuobjdump", "-xelf", "all", str(path)], cwd=out, stdout=stdout, stderr=stderr, check=True)
    cubins = [p for p in sorted(out.iterdir()) if p.read_bytes()[:4] == b"\x7fELF"]
    if not cubins:
        raise ValueError("cuobjdump extracted no cubins")
    identity["cubin_sha256s"] = [sha256(p) for p in cubins]
    return identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--image", choices=IMAGES, required=True)
    p.add_argument("--library", action="append", choices=("e2m1", "e4m3mma", "e4m3", "value"))
    p.add_argument("--revision", choices=("8bf496ab", "e89757b9"), default="8bf496ab")
    args = p.parse_args()
    commit = check_source(args.source, args.revision)
    runtime = check_runtime(args.image)
    selected = args.library or runtime["libraries"]
    if len(set(selected)) != len(selected) or not set(selected) <= set(runtime["libraries"]):
        p.error("libraries must be distinct and available in the cell image")
    runtime["libraries"] = selected
    sys.path.insert(0, str(args.source / "src"))
    from tessera.routed_fused import LIBRARIES, _cflags
    from torch.utils.cpp_extension import load

    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for key in selected:
        name = "tessera_routed_fused_e2m1" if key == "e2m1" else LIBRARIES[key][0]
        flags = _cflags("sm_121", key in ("e4m3", "e4m3mma"), key == "e4m3mma", key == "e2m1")
        flags.append("--objdir-as-tempdir")
        build = args.out / name
        build.mkdir()
        lib = load(name=name, sources=[str(args.source / KERNEL)], build_directory=str(build),
                   extra_cuda_cflags=flags, verbose=True)
        probe = args.out / f"swiglu_{key}"
        probe_source = Path(__file__).with_name("swiglu_probe.cu")
        subprocess.run(["nvcc", *flags, str(probe_source), "-o", str(probe)], check=True)
        binaries = {}
        for label, path in (("library", Path(lib.__file__)), ("probe", probe)):
            dumps = {}
            for kind in ("sass", "elf"):
                dump = args.out / f"{key}.{label}.{kind}"
                with dump.open("w") as stream:
                    subprocess.run(["cuobjdump", f"-{kind}", str(path)], stdout=stream, check=True)
                dumps[kind] = {"path": str(dump), "sha256": sha256(dump)}
            binaries[label] = {"path": str(path), "sha256": sha256(path), **dumps}
        records.append({"key": key, "name": name, "cflags": flags, **binaries,
                        "probe_source_sha256": sha256(probe_source)})
        write_json(args.out / "build.json", {"commit": commit, "runtime": runtime, "libraries": records})


if __name__ == "__main__":
    main()
