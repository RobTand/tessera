"""Targeted native arithmetic probes with exact rational alternatives.

The method follows Fasi et al. (2021), not their older-device conclusions.
The raw input file retains every nibble, scale byte, and accumulator bit.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tessera.fp4_arithmetic import (ATOM_ERROR, INSTRUCTION, PROPERTIES,
    FP4QualificationError, derive_attested_fp4_bound, require_probe_contract,
    require_t4_device_qualification)


def power(e):
    return F(1 << e) if e >= 0 else F(1, 1 << -e)


def e2m1(c):
    magnitude = (F(0), F(1, 2), F(1), F(3, 2), F(2), F(3), F(4), F(6))[c & 7]
    return -magnitude if c & 8 else magnitude


def ue4m3(c):
    if type(c) is not int or not 0 <= c <= 126:
        raise ValueError("UE4M3 requires a finite unsigned scale byte 0..126")
    e, m = c >> 3, c & 7
    return F(m, 8) * power(-6) if not e else F(8 + m, 8) * power(e - 7)


def exponent(x):
    x = abs(x)
    e = x.numerator.bit_length() - x.denominator.bit_length()
    return e if x >= power(e) else e - 1


def quantize(x, q, mode):
    y = x / q
    lo = y.numerator // y.denominator
    if mode == "rd":
        n = lo
    elif mode == "ru":
        n = lo if y == lo else lo + 1
    elif mode == "rz":
        n = lo if y >= 0 or y == lo else lo + 1
    elif mode == "rn":
        remainder = y - lo
        n = lo + int(remainder > F(1, 2) or (remainder == F(1, 2) and lo % 2))
    else:
        raise ValueError("unknown rounding mode")
    return n * q


def round32(x, mode="rn"):
    if not x:
        return F(0)
    return quantize(x, power(max(-149, exponent(x) - 23)), mode)


def bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def from_bits(v):
    value = struct.unpack("<f", struct.pack("<I", v))[0]
    if not math.isfinite(value):
        raise ValueError("nonfinite native output")
    return F(value)


def base(name, prop, *, c=F(0)):
    return {"name": name, "property": prop, "a": bytearray(1024), "b": bytearray(512),
            "sa": bytearray([56] * 64), "sb": bytearray([56] * 32), "c": [bits(c)] * 128}


def term(case, k, a, b, sa=56, sb=56):
    for row in range(16):
        case["a"][row * 64 + k] = a
        case["sa"][row * 4 + k // 16] = sa
    for col in range(8):
        case["b"][col * 64 + k] = b
        case["sb"][col * 4 + k // 16] = sb


def scale_power(e):
    if e == -9:
        return 1
    if e == -8:
        return 2
    if e == -7:
        return 4
    if -6 <= e <= 8:
        return (e + 7) << 3
    raise ValueError("scale power lies outside UE4M3")


def large(case, k, e, negative=False):
    # Power-of-two products through exponent 20, with exact scale factors.
    code = 6 if e > 16 else 2
    scale_exponent = e - (4 if e > 16 else 0)
    ea = max(-9, min(8, scale_exponent // 2))
    eb = scale_exponent - ea
    term(case, k, code | (8 if negative else 0), code, scale_power(ea), scale_power(eb))


def cases():
    result = []
    # Deterministic layout controls precede arithmetic interpretation.
    for k in range(64):
        row, col = k % 16, (k // 8) % 8
        case = base(f"layout-r{row}-n{col}-k{k}", "layout")
        case["a"][row * 64 + k] = 3
        case["b"][col * 64 + k] = 5
        case["sa"][row * 4 + k // 16] = 60
        case["sb"][col * 4 + k // 16] = 64
        result.append(case)
    # All signed E2M1 pairs. Every finite UE4M3 byte meets scale corners.
    for sa in range(127):
        for sb in (0, 1, 7, 8, 15, 56, 63, 126):
            for half in (0, 8):
                case = base(f"product-sa{sa}-sb{sb}-half{half}", "product_exactness")
                for row in range(16):
                    case["a"][row * 64] = row
                    case["sa"][row * 4] = sa
                for col in range(8):
                    case["b"][col * 64] = col + half
                    case["sb"][col * 4] = sb
                result.append(case)
    # Distinct group scales detect post-reduction or crossed scale application.
    for shift in range(4):
        case = base(f"scale-groups-{shift}", "scale_application_order")
        for group in range(4):
            term(case, group * 16, 3, 5, (56, 60, 64, 68)[(group + shift) % 4], (72, 64, 60, 56)[group])
        result.append(case)
    # Cancellation reveals bits lost during alignment, not final output rounding.
    for j in range(10, 41):
        for sign in (1, -1):
            case = base(f"alignment-gap{j}-sign{sign}", "accumulation_alignment_width", c=sign * power(j - 20))
            large(case, 0, j - 20, negative=sign > 0)
            term(case, 16, 1 | (8 if sign < 0 else 0), 1, 1, 1)
            case["gap"] = j
            result.append(case)
    # More than half an ULP, exact ties, odd/even significands, and both signs.
    for e in (3, 4, 5, 6):
        for sign in (1, -1):
            for odd in (0, 1):
                c = sign * (power(e) + odd * power(e - 23))
                case = base(f"round-e{e}-sign{sign}-odd{odd}", "rounding_mode", c=c)
                term(case, 0, 3 | (8 if sign < 0 else 0), 1, 1, 1)
                result.append(case)
    for byte in (1, 7, 8):
        case = base(f"subnormal-scale-{byte}", "subnormal_handling")
        term(case, 0, 1, 1, byte, byte)
        result.append(case)
    for c_bits in (1, 0x007fffff, 0x00800000, 0x80000001, 0x807fffff):
        case = base(f"subnormal-accumulator-{c_bits:x}", "subnormal_handling")
        case["c"] = [c_bits] * 128
        result.append(case)
    # Maximum product and full atom, carry growth, and normalization cancellation.
    case = base("maximum-full-atom", "intermediate_domain")
    for k in range(64):
        term(case, k, 7, 7, 126, 126)
    result.append(case)
    for gap in (20, 23, 24, 25, 26, 30, 40):
        for permutation in ((0, 16, 32), (32, 0, 16), (16, 32, 0)):
            case = base(f"normalization-gap{gap}-order{permutation}", "intermediate_domain")
            large(case, permutation[0], gap - 20)
            term(case, permutation[1], 1, 1, 1, 1)
            large(case, permutation[2], gap - 20, negative=True)
            result.append(case)
    for e in (15, 16, 17, 18, 19, 20):
        for sign in (1, -1):
            case = base(f"group-before-alignment-e{e}-sign{sign}", "intermediate_domain", c=sign * power(e))
            for k in range(16):
                term(case, k, 1 | (8 if sign < 0 else 0), 1, 1, 1)
            large(case, 16, e, negative=sign > 0)
            result.append(case)
    case = base("seven-carry-bits", "intermediate_domain", c=F(2025, 256))
    for k in range(64):
        term(case, k, 3, 3, 63, 63)
    result.append(case)
    return result


def exact_terms(case, row, col):
    terms = [from_bits(case["c"][row * 8 + col])]
    for k in range(64):
        a, b = case["a"][row * 64 + k], case["b"][col * 64 + k]
        if (a & 7) and (b & 7):
            terms.append(e2m1(a) * e2m1(b) * ue4m3(case["sa"][row * 4 + k // 16]) * ue4m3(case["sb"][col * 4 + k // 16]))
    return terms


def exact_group_terms(case, row, col):
    result = [from_bits(case["c"][row * 8 + col])]
    for group in range(4):
        dot = sum((e2m1(case["a"][row * 64 + k]) * e2m1(case["b"][col * 64 + k]) for k in range(group * 16, (group + 1) * 16)), F(0))
        result.append(dot * ue4m3(case["sa"][row * 4 + group]) * ue4m3(case["sb"][col * 4 + group]))
    return result


def alternatives(terms, grouped=None):
    exact = sum(terms, F(0))
    result = {"exact": str(exact)}
    for mode in ("rn", "rz", "rd", "ru"):
        result[mode] = bits(round32(exact, mode))
    maximum = max(map(abs, terms))
    if maximum:
        for p in (23, 24, 25, 26, 27, 32, 35, 36, 37, 40):
            q = power(exponent(maximum) - p + 1)
            aligned = sum((quantize(x, q, "rz") for x in terms), F(0))
            for mode in ("rn", "rz"):
                result[f"align{p}-{mode}"] = bits(round32(aligned, mode))
    if grouped is not None:
        maximum = max(map(abs, grouped))
        if maximum:
            for p in (23, 24, 25, 26, 27, 32, 35, 36, 37, 40):
                q = power(exponent(maximum) - p + 1)
                aligned = sum((quantize(x, q, "rz") for x in grouped), F(0))
                for mode in ("rn", "rz"):
                    result[f"group-align{p}-{mode}"] = bits(round32(aligned, mode))
    return result


def pack(case):
    return bytes(case["a"] + case["b"] + case["sa"] + case["sb"]) + struct.pack("<128I", *case["c"])


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def analyze(population, native, device):
    properties = {name: {"status": "pass", "cases": 0, "failures": []} for name in PROPERTIES}
    rows = []
    layout_failures = []
    rounding_candidates = set(("rn", "rz", "rd", "ru"))
    alignment_candidates = {f"{kind}{p}-{mode}" for kind in ("align", "group-align") for p in (24, 25, 26, 27, 32, 35, 36, 37, 40) for mode in ("rn", "rz")}
    for index, case in enumerate(population):
        prop = case["property"]
        target = properties.get(prop)
        if target is not None:
            target["cases"] += 1
        expected, actual, failures = [], native[index * 128:(index + 1) * 128], []
        for row in range(16):
            for col in range(8):
                position = row * 8 + col
                terms = exact_terms(case, row, col)
                grouped = exact_group_terms(case, row, col) if prop in ("accumulation_alignment_width", "intermediate_domain") else None
                options = alternatives(terms, grouped)
                got_bits = actual[position]
                expected.append(options)
                try:
                    got = from_bits(got_bits)
                    exact = sum(terms, F(0))
                    magnitude = sum(map(abs, terms), F(0))
                    error = abs(got - exact)
                    if prop in ("layout", "product_exactness", "scale_application_order", "subnormal_handling"):
                        passed = got == exact
                    else:
                        passed = error <= ATOM_ERROR * magnitude
                    if prop == "rounding_mode":
                        rounding_candidates.intersection_update(key for key in ("rn", "rz", "rd", "ru") if options[key] == got_bits)
                    if prop in ("accumulation_alignment_width", "intermediate_domain"):
                        alignment_candidates.intersection_update(key for key in alignment_candidates.copy() if options.get(key) == got_bits)
                    if not passed:
                        failures.append({"output": position, "bits": got_bits, "exact": str(exact), "error": str(error), "atom_bound": str(ATOM_ERROR * magnitude)})
                except ValueError as exc:
                    failures.append({"output": position, "error": str(exc), "bits": got_bits})
        rows.append({"name": case["name"], "property": prop, "input_offset": index * 2144,
                     "output_bits": actual, "independent_alternatives": expected, "failures": failures})
        if failures:
            if target is None:
                layout_failures.extend(failures)
            else:
                target["status"] = "fail"
                target["failures"].append({"case": case["name"], "outputs": failures})
    if not any(candidate.startswith(("align36-", "group-align36-")) for candidate in alignment_candidates):
        properties["accumulation_alignment_width"]["status"] = "unsupported_model"
    if len(rounding_candidates) != 1:
        properties["rounding_mode"]["status"] = "unsupported_model"
    report = {"schema": "tessera.fp4_mma_attestation.v1", "instruction": INSTRUCTION,
              "device": device, "harness_status": "fail" if layout_failures else "pass",
              "layout_failures": layout_failures, "properties": properties,
              "alignment_candidates": sorted(alignment_candidates), "rounding_candidates": sorted(rounding_candidates),
              "contract": {"alignment_bits_min": 36, "atom_summands": 65, "scaled_product_error": "0", "atom_error": "(65*2^-35+2^-23)*S"},
              "physical_devices": [os.environ["HOST_NAME"] + ":cuda:0"] if os.environ.get("HOST_NAME") else [],
              "arithmetic_qualified": False, "reviews": {"kernels_parent": False, "independent": False},
              "scope": "targeted finite implementation attestation, not universal hardware proof",
              "unsupported": ["other instructions or devices", "scale NaN code 127", "nonzero initial accumulator in the derived bound", "overflow", "uncharacterized library reference GEMM", "activation quantization or division", "SwiGLU, router weights, split sums, or token sums"]}
    return report, rows


def cpu_controls():
    # Two independent references: rational rounding and host IEEE conversion.
    for e in range(-140, 121, 13):
        for n in (-7, -3, -1, 0, 1, 3, 7):
            x = F(n) * power(e - 25)
            assert from_bits(bits(round32(x))) == from_bits(bits(x))
    assert round32(F(1) + power(-24)) == 1
    assert round32(F(1) + 3 * power(-25)) == 1 + power(-23)
    assert ue4m3(1) == power(-9) and ue4m3(126) == 448
    assert e2m1(7) == 6 and e2m1(15) == -6


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cpu_controls()
    if args.corrective_stock_audit:
        if not args.native_attestation or not args.retained_outputs:
            raise ValueError("a corrective audit requires a native attestation and retained outputs")
        from fp4_corrective_audit import run
        return run(args, sys.modules[__name__])
    if args.original_stock_regressions:
        if not args.native_attestation:
            raise ValueError("original comparison regressions require --native-attestation")
        from fp4_stock_reference_attest import run_original
        return run_original(args, sys.modules[__name__])
    if args.stock_reference:
        if not args.native_attestation:
            raise ValueError("stock reference probes require --native-attestation")
        from fp4_stock_reference_attest import run
        return run(args, sys.modules[__name__])
    if args.output_boundaries:
        if not args.native_attestation:
            raise ValueError("output boundary probes require --native-attestation")
        from fp4_output_boundaries import run
        return run(args, sys.modules[__name__])
    population = cases()
    inputs = b"".join(map(pack, population))
    input_path = out / "inputs.bin"
    input_path.write_bytes(inputs)
    save(out / "inputs.json", {"layout": "little endian: A1024,B512,SA64,SB32,C128 uint32; 2144 bytes per atom", "cases": [{"name": x["name"], "property": x["property"]} for x in population], "sha256": hashlib.sha256(inputs).hexdigest()})
    if args.cpu_preflight:
        # Exercise each property and the real gate without claiming device proof.
        selected = [next(x for x in population if x["property"] == p) for p in ("layout", "product_exactness", "scale_application_order", "subnormal_handling")]
        selected += [x for x in population if x["property"] in ("rounding_mode", "accumulation_alignment_width", "intermediate_domain")]
        outputs = [alternatives(exact_terms(x, r, c)).get("align36-rz", 0) for x in selected for r in range(16) for c in range(8)]
        report, _ = analyze(selected, outputs, {"device": "CPU reference control"})
        altered = outputs.copy()
        altered[128] = bits(F(1))
        bad, bad_rows = analyze(selected, altered, {"device": "CPU reference control"})
        save(out / "negative-references.json", [bad_rows[1]])
        try:
            require_probe_contract(bad, device="CPU reference control")
        except FP4QualificationError as exc:
            refusal = str(exc)
        else:
            raise AssertionError("negative control received a native contract")
        bound = derive_attested_fp4_bound(F(64), k=64, report=report, device="CPU reference control")
        try:
            require_t4_device_qualification(report, device="CPU reference control", physical_device="CPU reference control", comparison="exact_represented_operands", k=64, shape=(16, 8, 64))
        except FP4QualificationError as exc:
            review_refusal = str(exc)
        else:
            raise AssertionError("unreviewed arithmetic passed the qualification gate")
        result = {"mode": "CPU preflight", "cases": len(population), "GPU_exercised": False, "interpreter_controls": "pass", "negative_refusal": refusal, "review_refusal": review_refusal, "bound_api": bound}
        save(out / "cpu-preflight.json", result)
        print(json.dumps(result), flush=True)
        if args.negative_control:
            print(refusal, file=sys.stderr)
            return 5
        return 0
    source = Path(__file__).with_name("fp4_arithmetic_probe.cu")
    shutil.copy2(source, out / source.name)
    compiler = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    binary = out / "probe"
    build = [compiler, "-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a", "--ftz=false", "--fmad=false", "-o", str(binary), str(source)]
    context = {"build": build, "compiler": subprocess.check_output([compiler, "--version"], text=True), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "action_key": os.environ.get("PRISMABUILD_ACTION_KEY", os.environ.get("PB_ACTION_KEY")), "image": os.environ.get("ORACLE_IMAGE"), "python": sys.version, "platform": sys.platform}
    save(out / "context.json", context)
    build_result = subprocess.run(build, capture_output=True, text=True)
    (out / "compile.stdout").write_text(build_result.stdout)
    (out / "compile.stderr").write_text(build_result.stderr)
    if build_result.returncode:
        save(out / "failure.json", {"classification": "harness_compile_failure", "returncode": build_result.returncode})
        return build_result.returncode
    # Retain native instruction code for independent inspection.
    disassembler = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    disassembly = subprocess.run([disassembler, "--dump-sass", str(binary)], capture_output=True, text=True)
    (out / "probe.sass").write_text(disassembly.stdout)
    context["disassembly_returncode"] = disassembly.returncode
    res = subprocess.run([str(binary), str(input_path), str(out / "outputs.bin")], capture_output=True, text=True)
    (out / "native.stdout").write_text(res.stdout)
    (out / "native.stderr").write_text(res.stderr)
    if res.returncode:
        save(out / "failure.json", {"classification": "native_execution_failure", "returncode": res.returncode, "native_arithmetic_observed": False})
        return res.returncode
    device = json.loads(res.stdout)
    output_bytes = (out / "outputs.bin").read_bytes()
    if len(output_bytes) != len(population) * 128 * 4:
        raise ValueError("native output shape failed")
    native = list(struct.unpack(f"<{len(population) * 128}I", output_bytes))
    report, rows = analyze(population, native, device)
    report["context"] = context
    report["inputs_sha256"] = hashlib.sha256(inputs).hexdigest()
    report["outputs_sha256"] = hashlib.sha256(output_bytes).hexdigest()
    save(out / "references.json", rows)
    altered = native.copy()
    altered[64 * 128] = bits(F(1))
    bad, bad_rows = analyze(population, altered, device)
    save(out / "negative-references.json", [bad_rows[64]])
    try:
        require_probe_contract(bad, device=device["device"])
    except FP4QualificationError as exc:
        report["negative_control_refusal"] = str(exc)
    else:
        raise AssertionError("an altered native product received a contract")
    if args.negative_control:
        report = bad
        report["negative_control"] = "one captured output changed to one; not a device failure"
    failures = []
    try:
        require_probe_contract(report, device=device["device"])
        report["bound_smoke"] = derive_attested_fp4_bound(F(64) * 36 * 448 * 448, k=64, report=report, device=device["device"])
    except FP4QualificationError as exc:
        failures.append(str(exc))
    try:
        require_t4_device_qualification(report, device=device["device"], physical_device=os.environ.get("HOST_NAME", "unidentified") + ":cuda:0", comparison="exact_represented_operands", k=64, shape=(16, 8, 64))
    except FP4QualificationError as exc:
        report["qualification_refusal"] = str(exc)
    else:
        raise AssertionError("the unreviewed report qualified a device")
    report["failures"] = failures
    report["classification"] = ("injected_negative_control" if args.negative_control else "harness_layout_failure" if report["harness_status"] != "pass" else "unsupported_arithmetic_model" if any(x["status"] == "unsupported_model" for x in report["properties"].values()) else "device_inequality_failure" if failures else "measured_contract_pending_reviews")
    save(out / "attestation.json", report)
    print(json.dumps({"device": device, "classification": report["classification"], "properties": {k: {"status": v["status"], "cases": v["cases"]} for k, v in report["properties"].items()}, "alignment_candidates": report["alignment_candidates"], "rounding_candidates": report["rounding_candidates"], "failures": failures, "arithmetic_qualified": False}), flush=True)
    return 5 if failures else 0


def guarded(args):
    if args.guarded_child:
        return main(args)
    sys.path.insert(0, str(ROOT / "experiments" / "graph_attest_702"))
    from managed_window import Envelope
    samples = []
    def sample():
        value = next(line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
        available = int(value.split()[1]) * 1024
        samples.append({"unix": time.time(), "available_bytes": available})
        if available < 2 * (1 << 30):
            raise RuntimeError("D30: MemAvailable fell below two GiB")
    sample()
    envelope = Envelope(time.time() + 1200, cleanup_seconds=20)
    result = None
    try:
        result = envelope.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--guarded-child"], check=False, tick=sample, stdout=sys.stdout, limit=1180)
        return result.returncode
    finally:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        save(out / "memory-guard.json", {"samples": samples, "abort_below_bytes": 2 * (1 << 30), "terminations": envelope.terminations, "returncode": None if result is None else result.returncode})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cpu-preflight", action="store_true")
    parser.add_argument("--negative-control", action="store_true")
    parser.add_argument("--output-boundaries", action="store_true")
    parser.add_argument("--native-attestation")
    parser.add_argument("--stock-reference", action="store_true")
    parser.add_argument("--boundary-attestation")
    parser.add_argument("--original-stock-regressions", action="store_true")
    parser.add_argument("--corrective-stock-audit", action="store_true")
    parser.add_argument("--retained-outputs")
    parser.add_argument("--guarded-child", action="store_true", help=argparse.SUPPRESS)
    raise SystemExit(guarded(parser.parse_args()))
