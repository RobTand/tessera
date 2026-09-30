"""Per-kernel SASS of the fused window libraries, and a comparison of two dumps.

The lead's bar for a change to ``routed_fused_window.cu`` that must leave the
existing libraries' launches alone (tessera#750): every kernel's SASS is the
same instruction multiset up to commuted operands and register renaming.
This tool gives that comparison a receipt.

``dump``: compile ``--source`` once per library with the defines and flags
``tessera.routed_fused._cflags`` gives it (device code only, ``nvcc -cubin``),
disassemble with ``cuobjdump -sass`` and write ``<out>/<library>.json``: the
toolchain, the flags and, per kernel (mangled name), its instruction list
with addresses and encodings stripped.

``compare``: read two dump directories and, per library present in both,
classify every kernel as

* ``identical``: the same instruction sequence;
* ``multiset``: the same multiset of instructions once register and
  predicate numbers are renamed away and each instruction's source operands
  are sorted (commuted operands, renaming, and a changed schedule);
* ``differs``: anything else, with the opcode count delta;
* ``only_before`` / ``only_after``: a kernel one side does not have.

No device is touched; both subcommands are CPU work.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys

LIBRARIES = {
    # library: (fp8, mma8, fp4)
    "value": (False, False, False),
    "e4m3": (True, False, False),
    "e4m3mma": (True, True, False),
    "e2m1": (False, False, True),
}


def _flags(library: str, token: str) -> list:
    fp8, mma8, fp4 = LIBRARIES[library]
    digits = token[len("sm_"):] + ("a" if fp4 else "")
    flags = ["-O3", "-lineinfo", "-std=c++17",
             f"-DTESSERA_ROUTED_FUSED_FP8={1 if fp8 else 0}",
             f"-DTESSERA_ROUTED_FUSED_MMA8={1 if mma8 else 0}"]
    if fp4:
        flags.append("-DTESSERA_ROUTED_FUSED_FP4=1")
    return flags + ["-gencode", f"arch=compute_{digits},code=sm_{digits}"]


def _includes() -> list:
    import sysconfig

    from torch.utils.cpp_extension import include_paths

    out = []
    for p in include_paths() + [sysconfig.get_paths()["include"]]:
        out += ["-I", p]
    return out


_ADDR = re.compile(r"/\*[0-9a-f]{4,}\*/")
_ENC = re.compile(r"/\*\s*0x[0-9a-f]+\s*\*/")
_FUNC = re.compile(r"^\s*Function\s*:\s*(\S+)")


def parse_sass(text: str) -> dict:
    kernels, name = {}, None
    for line in text.splitlines():
        m = _FUNC.match(line)
        if m:
            name = m.group(1)
            kernels[name] = []
            continue
        if name is None:
            continue
        if _ENC.search(line) and not _ADDR.search(line):
            continue                      # the second encoding line of an instruction
        if not _ADDR.search(line):
            continue
        ins = _ENC.sub("", _ADDR.sub("", line)).strip().rstrip(";").strip()
        if ins:
            kernels[name].append(ins)
    return kernels


def _dump_one(args, library: str, version: str, torch_version: str) -> str:
    nvcc = os.environ.get("NVCC", "nvcc")
    cubin = os.path.join(args.out, f"{library}.cubin")
    cmd = [nvcc, "-cubin", "-o", cubin, *_flags(library, args.token), *_includes(),
           "-D_GLIBCXX_USE_CXX11_ABI=1", "-DTORCH_EXTENSION_NAME=sass_dump", args.source]
    if args.ptxas_verbose:
        cmd[1:1] = ["-Xptxas", "-v"]
    print("+", " ".join(cmd), flush=True)
    built = subprocess.run(cmd, capture_output=True, text=True)
    if built.returncode != 0:
        print(built.stdout + built.stderr, flush=True)
        raise subprocess.CalledProcessError(built.returncode, cmd)
    resources = parse_ptxas(built.stderr) if args.ptxas_verbose else {}
    sass = subprocess.run(["cuobjdump", "-sass", cubin], capture_output=True, text=True, check=True).stdout
    kernels = parse_sass(sass)
    rec = {"library": library, "source": os.path.abspath(args.source), "token": args.token,
           "flags": _flags(library, args.token), "nvcc": version, "torch": torch_version,
           "kernels": kernels, "resources": resources}
    for name, r in sorted(resources.items()):
        print(f"  {library} {name[:110]}: {r}", flush=True)
    with open(os.path.join(args.out, f"{library}.json"), "w") as f:
        json.dump(rec, f)
    os.remove(cubin)
    return f"{library}: {len(kernels)} kernels, {sum(len(v) for v in kernels.values())} instructions"


def dump(args) -> int:
    from concurrent.futures import ThreadPoolExecutor

    import torch

    os.makedirs(args.out, exist_ok=True)
    nvcc = os.environ.get("NVCC", "nvcc")
    version = subprocess.run([nvcc, "--version"], capture_output=True, text=True,
                             check=True).stdout.strip().splitlines()[-1]
    libs = args.libraries.split(",")
    with ThreadPoolExecutor(len(libs)) as pool:
        for line in pool.map(lambda lib: _dump_one(args, lib, version, torch.__version__), libs):
            print(line, flush=True)
    return 0


_PTXAS_ENTRY = re.compile(r"Compiling entry function '([^']+)'")
_PTXAS_PROPS = re.compile(r"Function properties for (\S+)")
_PTXAS_SPILL = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
_PTXAS_REGS = re.compile(r"Used (\d+) registers")


def parse_ptxas(text: str) -> dict:
    """Per entry function: registers, stack frame and spill bytes (``-Xptxas -v``)."""
    out, name = {}, None
    for line in text.splitlines():
        m = _PTXAS_ENTRY.search(line) or _PTXAS_PROPS.search(line)
        if m:
            name = m.group(1)
            out.setdefault(name, {})
            continue
        if name is None:
            continue
        m = _PTXAS_SPILL.search(line)
        if m:
            out[name].update(stack=int(m.group(1)), spill_stores=int(m.group(2)), spill_loads=int(m.group(3)))
        m = _PTXAS_REGS.search(line)
        if m:
            out[name]["registers"] = int(m.group(1))
    return out


_REG = re.compile(r"\b(U?R|U?P)(\d+|Z|T)\b")


def _canon(ins: str) -> str:
    """Registers renamed away, source operands sorted (commutation)."""
    ins = _REG.sub(lambda m: m.group(1), ins)
    head, _, ops = ins.partition(" ")
    parts = [p.strip() for p in ops.split(",")] if ops else []
    if len(parts) > 2:
        parts = parts[:1] + sorted(parts[1:])
    return head + " " + ",".join(parts)


def _opcodes(seq) -> collections.Counter:
    return collections.Counter(s.split(" ")[0] for s in seq)


def compare(args) -> int:
    report = {}
    for library in LIBRARIES:
        a = os.path.join(args.before, f"{library}.json")
        b = os.path.join(args.after, f"{library}.json")
        if not (os.path.exists(a) and os.path.exists(b)):
            continue
        ka = json.load(open(a))["kernels"]
        kb = json.load(open(b))["kernels"]
        rows = {"identical": [], "multiset": [], "differs": [], "only_before": [], "only_after": []}
        for name in sorted(set(ka) | set(kb)):
            if name not in kb:
                rows["only_before"].append(name)
            elif name not in ka:
                rows["only_after"].append(name)
            elif ka[name] == kb[name]:
                rows["identical"].append(name)
            elif collections.Counter(map(_canon, ka[name])) == collections.Counter(map(_canon, kb[name])):
                rows["multiset"].append(name)
            else:
                da, db = _opcodes(ka[name]), _opcodes(kb[name])
                delta = {op: db[op] - da[op] for op in set(da) | set(db) if db[op] != da[op]}
                rows["differs"].append({"kernel": name, "len": [len(ka[name]), len(kb[name])],
                                        "opcode_delta": delta})
        report[library] = rows
        print(f"{library}: {len(rows['identical'])} identical, {len(rows['multiset'])} same multiset, "
              f"{len(rows['differs'])} differ, {len(rows['only_before'])} only before, "
              f"{len(rows['only_after'])} only after (of {len(set(ka) | set(kb))})")
        for d in rows["differs"]:
            print("   differs:", d["kernel"][:120], d["len"], d["opcode_delta"])
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    return 1 if any(r["differs"] or r["only_before"] for r in report.values()) else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--source", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--libraries", default="value,e4m3,e4m3mma")
    d.add_argument("--token", default="sm_121")
    d.add_argument("--ptxas-verbose", action="store_true",
                   help="also record each kernel's registers and spill bytes (-Xptxas -v)")
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    c.add_argument("--json")
    args = ap.parse_args(argv)
    return dump(args) if args.cmd == "dump" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
