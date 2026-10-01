#!/usr/bin/env python3
"""Per-kernel SASS and resource-usage comparison of CUDA binaries (cubin or host object).

Usage: sass_cmp.py <cuobjdump> <spec.json> <out.json>
  spec.json: {"bins": {name: path}, "pairs": [[label, ref, cand, expect], ...]}
  expect:
    identical              every kernel's SASS and resources match, and the kernel sets match
    single-rate-identical  the routed kernels with TWO=0, and every non-routed kernel, match
    two-run-identical      the routed kernels with TWO=1, and every non-routed kernel, match
    report                 no verdict
  Every pair also fails on a spill regression: a kernel whose LOCAL or STACK grows from ref
  to cand.
Exit status: 0 when every pair with a verdict passes, 1 otherwise.

Kernel names are normalized before matching: nvcc's anonymous-namespace and internal-linkage
hashes (``_GLOBAL__N__<8 hex>_<n>_<file>_cu_<8 hex>``, ``_INTERNAL_<8 hex>_...``) depend on
the path a source is compiled from; the template arguments do not.  The replacement keeps each
hash's length, so the mangled length prefixes still parse.
"""
import hashlib
import json
import multiprocessing
import os
import re
import subprocess
import sys
from collections import Counter

HASH8 = [(re.compile(r"_GLOBAL__N__[0-9a-f]{8}_"), "_GLOBAL__N__xxxxxxxx_"),
         (re.compile(r"_INTERNAL_[0-9a-f]{8}_"), "_INTERNAL_xxxxxxxx_"),
         (re.compile(r"_cu_[0-9a-f]{8}"), "_cu_xxxxxxxx")]
ROUTED = re.compile(r"(routed_fused(?:_fp4)?_kernel)I((?:L[bi]n?\d+E)+)E")
# The TWO template argument's index: routed_fused_kernel<FP8, MODE, DENSE, SPLIT, RL, TWO, BMT>,
# routed_fused_fp4_kernel<MODE, DENSE, SPLIT, RL, TWO>.
TWO_AT = {"routed_fused_kernel": 5, "routed_fused_fp4_kernel": 4}
ADDR = re.compile(r"/\*[0-9a-f]{4,}\*/")
ENC = re.compile(r"/\*\s*0x[0-9a-f]{16}\s*\*/")
# cuobjdump's per-ELF banner (a host object's fatbin carries one per embedded ELF).
BANNER = re.compile(r"^(Fatbin|code for|=+$|arch = |code version = |host = |compile_size = )")


def norm(s: str) -> str:
    for rx, rep in HASH8:
        s = rx.sub(rep, s)
    return s


def kernel_info(name: str):
    """(short name, template args, TWO) for a routed kernel; (name, None, None) otherwise."""
    m = ROUTED.search(name)
    if not m:
        return name, None, None
    args = [int(v.replace("n", "-")) for _, v in re.findall(r"L([bi])(n?\d+)E", m.group(2))]
    two = bool(args[TWO_AT[m.group(1)]])
    return f"{m.group(1)}<{','.join(map(str, args))}>", args, two


def run(cuobjdump: str, flag: str, path: str) -> str:
    return subprocess.run([cuobjdump, flag, path], check=True, capture_output=True, text=True).stdout


def parse_sass(text: str) -> dict:
    funcs, cur, lines = {}, None, []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Function :"):
            if cur is not None:
                funcs[cur] = lines
            cur, lines = norm(s.split(":", 1)[1].strip()), []
        elif cur is not None:
            if not s or BANNER.match(s) or set(s) <= {"."}:
                continue
            lines.append(norm(s))
    if cur is not None:
        funcs[cur] = lines
    return funcs


def parse_res(text: str) -> dict:
    res, cur = {}, None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Function ") and s.endswith(":"):
            cur = norm(s[len("Function "):-1].strip())
        elif cur is not None and "REG:" in s:
            res[cur] = {k: v for k, v in (t.split(":", 1) for t in s.split() if ":" in t)}
            cur = None
    return res


def instr(lines):
    """Instruction text only: no addresses, no encodings, no encoding-only lines."""
    out = []
    for s in lines:
        s = ENC.sub("", ADDR.sub("", s)).strip()
        if s and not s.startswith(".headerflags"):
            out.append(" ".join(s.split()))
    return out


def load(cuobjdump: str, path: str) -> dict:
    sass = parse_sass(run(cuobjdump, "-sass", path))
    res = parse_res(run(cuobjdump, "-res-usage", path))
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    spills = sorted(kernel_info(k)[0] for k, r in res.items()
                    if r.get("LOCAL", "0") != "0" or r.get("STACK", "0") != "0")
    regs = [int(r["REG"]) for r in res.values() if "REG" in r]
    return {"path": path, "sha256": digest, "sass": sass, "res": res, "spills": spills,
            "max_reg": max(regs) if regs else None,
            "instructions": sum(len(instr(v)) for v in sass.values())}


def must_match(expect: str, two):
    if expect == "identical":
        return True
    if expect == "single-rate-identical":
        return two is None or not two
    if expect == "two-run-identical":
        return two is None or two
    return False


def compare(label, ref, cand, expect, a, b):
    ka, kb = set(a["sass"]), set(b["sass"])
    rows, identical, failures = [], 0, []
    for k in sorted(ka & kb):
        short, args, two = kernel_info(k)
        sass_eq = a["sass"][k] == b["sass"][k]
        res_eq = a["res"].get(k) == b["res"].get(k)
        ra, rb = a["res"].get(k, {}), b["res"].get(k, {})
        spill_up = any(int(rb.get(x, "0")) > int(ra.get(x, "0")) for x in ("LOCAL", "STACK"))
        if spill_up:
            failures.append(f"{short}: LOCAL/STACK grew {ra.get('LOCAL')}/{ra.get('STACK')} -> "
                            f"{rb.get('LOCAL')}/{rb.get('STACK')}")
        if sass_eq and res_eq:
            identical += 1
            continue
        ia, ib = instr(a["sass"][k]), instr(b["sass"][k])
        # Instructions (text, branch targets included) in one and not the other, with
        # multiplicity: a size for the change, linear in the kernel's length.
        ca, cb = Counter(ia), Counter(ib)
        changed = sum(((ca - cb) + (cb - ca)).values())
        row = {"kernel": short, "args": args, "two": two, "sass_equal": sass_eq, "res_equal": res_eq,
               "res_ref": ra, "res_cand": rb, "instr_ref": len(ia), "instr_cand": len(ib),
               "instr_text_equal": ia == ib, "instr_multiset_delta": changed}
        rows.append(row)
        if must_match(expect, two):
            failures.append(f"{short}: {'SASS' if not sass_eq else 'resources'} differ")
    only_a = sorted(kernel_info(k)[0] for k in ka - kb)
    only_b = sorted(kernel_info(k)[0] for k in kb - ka)
    if expect == "identical" and (only_a or only_b):
        failures.append(f"kernel sets differ: {len(only_a)} only in {ref}, {len(only_b)} only in {cand}")
    verdict = "report" if expect == "report" and not failures else ("pass" if not failures else "fail")
    return {"label": label, "ref": ref, "cand": cand, "expect": expect, "verdict": verdict,
            "kernels_ref": len(ka), "kernels_cand": len(kb), "common": len(ka & kb),
            "identical": identical, "differ": rows, "only_ref": only_a, "only_cand": only_b,
            "failures": failures}


def main():
    cuobjdump, spec_path, out_path = sys.argv[1:4]
    spec = json.load(open(spec_path))
    names = list(spec["bins"])
    # One process per binary: cuobjdump and the parse are single-threaded (~20 s each here).
    with multiprocessing.Pool(min(len(names), len(os.sched_getaffinity(0)))) as pool:
        loaded = pool.starmap(load, [(cuobjdump, spec["bins"][n]) for n in names])
    bins = dict(zip(names, loaded))
    pairs = [compare(lbl, r, c, e, bins[r], bins[c]) for lbl, r, c, e in spec["pairs"]]
    out = {"bins": {n: {k: v for k, v in b.items() if k not in ("sass", "res")} | {"kernels": len(b["sass"])}
                    for n, b in bins.items()},
           "pairs": pairs}
    json.dump(out, open(out_path, "w"), indent=1)
    for n, b in out["bins"].items():
        print(f"bin {n}: {b['kernels']} kernels, {b['instructions']} instructions, max REG {b['max_reg']}, "
              f"spills {b['spills'] or 'none'}, sha256 {b['sha256'][:16]}")
    ok = True
    for p in pairs:
        diff_two = sorted({str(r["two"]) for r in p["differ"]})
        print(f"pair {p['label']} ({p['ref']} -> {p['cand']}, expect {p['expect']}): {p['verdict']}; "
              f"{p['identical']}/{p['common']} kernels identical, {len(p['differ'])} differ "
              f"(TWO of the differing: {', '.join(diff_two) or '-'}), only-ref {len(p['only_ref'])}, "
              f"only-cand {len(p['only_cand'])}")
        for f in p["failures"][:12]:
            print(f"   FAIL {f}")
        ok &= p["verdict"] != "fail"
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
