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

from experiments.numerics_rows.common import IMAGES, KERNEL, check_runtime, check_source, load_build, sha256, write_json

IDENTITY_FIELDS = ("library_so_sha256", "library_code_sha256", "library_code_sections", "cubin_sha256s")


def elf_sections(path):
    """Read bounded section payloads in section-header order."""
    data = path if isinstance(path, bytes) else Path(path).read_bytes()
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
    sections = []
    for index, section in enumerate(table):
        name_index, kind, flags, _, start, size = section[:6]
        end = names.find(b"\0", name_index)
        if end < 0:
            raise ValueError("unterminated ELF section name")
        name = names[name_index:end].decode("utf-8")
        content = b"" if kind == 8 else payload(start, size)
        sections.append({"index": index, "name": name, "sh_type": kind, "sh_flags": flags, "sh_size": size,
                         "offset": start, "link": section[6], "entsize": section[9], "data": content})
    return order, wide, sections


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
    _, _, records = elf_sections(path)
    digest, sections = hashlib.sha256(), []
    for r in records:
        if r["name"] == ".strtab":
            continue
        header = [r[k] for k in ("name", "sh_type", "sh_flags", "sh_size")]
        digest.update(json.dumps(header, ensure_ascii=True, separators=(",", ":")).encode("ascii") + b"\n")
        digest.update(r["data"])
        sections.append({k: r[k] for k in ("index", "name", "sh_type", "sh_flags", "sh_size")} | {
            "sha256": hashlib.sha256(r["data"]).hexdigest()})
    return {"library_code_sha256": digest.hexdigest(), "library_code_sections": sections}


def library_identity(path, out, *, data=None):
    """Hash one byte buffer. Extract cubins from its job-local snapshot."""
    path, out = Path(path).resolve(), Path(out)
    data = path.read_bytes() if data is None else data
    identity = {"library_so_sha256": hashlib.sha256(data).hexdigest(), **code_identity(data)}
    out.mkdir(parents=True)
    snapshot = out.parent / (out.name + ".so")
    snapshot.write_bytes(data)
    identity["snapshot_path"] = str(snapshot)
    with (out / "cuobjdump.stdout").open("w") as stdout, (out / "cuobjdump.stderr").open("w") as stderr:
        subprocess.run(["cuobjdump", "-xelf", "all", str(snapshot)], cwd=out, stdout=stdout, stderr=stderr, check=True)
    cubins = [p for p in sorted(out.iterdir()) if p.read_bytes()[:4] == b"\x7fELF"]
    if not cubins:
        raise ValueError("cuobjdump extracted no cubins")
    identity["cubin_sha256s"] = [sha256(p) for p in cubins]
    from experiments.numerics_rows.canonical import TOKEN, canonical_identity

    embedded = [canonical_identity(p.read_bytes()) for p in cubins]
    canonical = canonical_identity(data, [(p.name, r["sha256"]) for p, r in zip(cubins, embedded)])
    identity["canonicalization"] = {"schema": "tessera.nvcc_pid.v1", "pattern": TOKEN.pattern.decode("ascii"), "replacement": "00000000"}
    identity.update(library_canonical_code_sha256=canonical["sha256"], library_canonical_sections=canonical["sections"],
                    cubin_canonical_sha256s=[r["sha256"] for r in embedded],
                    canonical_cubins=[{"path": str(p), **r} for p, r in zip(cubins, embedded)])
    return identity


def add_identities(build_path, out):
    """Write a P receipt with code identities. Keep the original build unchanged."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    built = json.loads(Path(build_path).read_text())
    for record in built["libraries"]:
        for label in ("library", "probe"):
            if sha256(record[label]["path"]) != record[label]["sha256"]:
                raise ValueError(f"changed {label} binary")
        record.update(library_identity(record["library"]["path"], out / record["key"] / "cubins"))
    built.update(build_path=str(build_path), build_sha256=sha256(build_path))
    write_json(out / "p.json", built)
    return built


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--image", choices=IMAGES, required=True)
    p.add_argument("--library", action="append", choices=("e2m1", "e4m3mma", "e4m3", "value"))
    p.add_argument("--revision", choices=("8bf496ab", "e89757b9"), default="8bf496ab")
    p.add_argument("--existing-build", type=Path)
    args = p.parse_args()
    commit = check_source(args.source, args.revision)
    runtime = check_runtime(args.image)
    if args.existing_build:
        load_build(args.existing_build, runtime)
        add_identities(args.existing_build, args.out)
        return
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
                        **library_identity(binaries["library"]["path"], args.out / key / "cubins"),
                        "probe_source_sha256": sha256(probe_source)})
        write_json(args.out / "build.json", {"commit": commit, "runtime": runtime, "libraries": records})


if __name__ == "__main__":
    main()
