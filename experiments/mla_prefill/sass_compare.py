#!/usr/bin/env python3
"""Compare SASS of the served FlashInfer kernel with our copy and L0 kernels.

    sass_compare.py SERVED_SASS OURS_SASS

Each SASS file is a cuobjdump -sass dump. An instruction is normalized to its
text plus both 64-bit encoding words (the second word carries the scheduling
control bits), so "identical" means the same instruction stream, not only the
same mnemonics. Prints COPY_SASS_IDENTICAL, STOCK_TEMPLATE_SASS_IDENTICAL, and
for the L0 kernel the local-memory traffic and the positive-zero FADD sites.
"""
import difflib
import re
import sys

STOCK = ("_Z28sparse_mla_prefill_mg_kernelIL9ModelType3EL13QkComputeMode0ELi32ELi64ELi2EE"
         "vPK13__nv_bfloat16PKhPKiPS2_PfPKf17PrefillColdParams")


def functions(text):
    out, name, body = {}, None, []
    for line in text.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            if name:
                out[name] = body
            name, body = m.group(1), []
        elif name is not None:
            body.append(line)
    if name:
        out[name] = body
    return out


def norm(lines):
    res, pending = [], None
    for line in lines:
        m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s*(.*?)\s*;?\s*/\*\s*(0x[0-9a-f]+)\s*\*/\s*$", line)
        if m:
            pending = [m.group(1), re.sub(r"\s+", " ", m.group(2)), m.group(3)]
            continue
        m = re.match(r"\s*/\*\s*(0x[0-9a-f]+)\s*\*/\s*$", line)
        if m and pending is not None:
            pending.append(m.group(1))
            res.append(tuple(pending))
            pending = None
    return res


def stream(insns):
    return [" | ".join(i[1:]) for i in insns]


def compare(label, served, ours):
    a, b = stream(served), stream(ours)
    same = a == b
    print(f"{label} {same} served={len(a)} ours={len(b)}")
    if not same:
        d = list(difflib.unified_diff(a, b, "served", "ours", n=1, lineterm=""))
        print("\n".join(d[:120]))
    return same


def main():
    served = functions(open(sys.argv[1]).read())
    ours = functions(open(sys.argv[2]).read())
    print("served functions:", [k for k in served if "prefill_mg_kernel" in k])
    print("ours functions:", len(ours), [k for k in ours if "tessera" in k or k == STOCK])
    if STOCK not in served:
        print("SERVED_KERNEL_MISSING")
        return 2
    s = norm(served[STOCK])
    rc = 0
    if "tessera_mla_prefill_mg_copy" in ours:
        rc |= 0 if compare("COPY_SASS_IDENTICAL", s, norm(ours["tessera_mla_prefill_mg_copy"])) else 1
    else:
        print("COPY_KERNEL_MISSING")
        rc |= 1
    if STOCK in ours:
        compare("STOCK_TEMPLATE_SASS_IDENTICAL", s, norm(ours[STOCK]))
    if "tessera_mla_prefill_mg_l0" in ours:
        l0 = norm(ours["tessera_mla_prefill_mg_l0"])
        print(f"L0 instructions={len(l0)} stock={len(s)}")
        for addr, text, *_ in l0:
            if re.search(r"\b(LDL|STL)\b", text):
                print(f"L0 local {addr} {text}")
        for addr, text, *_ in l0:
            if text.startswith("FADD") or " FADD" in text:
                if re.search(r"\bRZ\b", text):
                    print(f"L0 fadd-zero {addr} {text}")
        bars = sorted({t for _, t, *_ in l0 if re.match(r"(@!?U?P\d+ )?BAR", t)})
        print("L0 barrier forms:", bars[:20])
    return rc


if __name__ == "__main__":
    sys.exit(main())
