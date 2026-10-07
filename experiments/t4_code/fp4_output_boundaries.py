"""Target the actual optional scalar boundaries, without another matrix run."""
from __future__ import annotations

import copy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

from tessera.fp4_arithmetic import (
    FP4QualificationError, derive_attested_fp4_bound,
    require_probe_contract, require_t4_device_qualification,
)


def run(args, reference):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads(Path(args.native_attestation).read_text())
    require_probe_contract(report, device=report["device"]["device"])
    # Values separate nearest-even conversion, truncation, and gradual underflow.
    values = [F(0), F(1), -F(1), F(1) + reference.power(-8),
              F(1) + 3 * reference.power(-8), -F(1) - reference.power(-8),
              -F(1) - 3 * reference.power(-8), reference.power(-20),
              reference.power(-126), reference.power(-133), reference.power(-134),
              3 * reference.power(-134), reference.power(-149),
              3 * reference.power(-149), -3 * reference.power(-149),
              reference.power(20), -reference.power(20)]
    scales = [F(0), F(1, 2), F(1), F(896), reference.power(-126),
              reference.power(-130), 3 * reference.power(-149)]
    inputs = [(value, scale) for value in values for scale in scales]
    raw = b"".join(struct.pack("<II", reference.bits(v), reference.bits(s)) for v, s in inputs)
    (out / "boundary-inputs.bin").write_bytes(raw)
    reference.save(out / "boundary-inputs.json", {"layout": "little endian uint32 pairs: value bits, scale bits", "inputs": [{"value": str(v), "scale": str(s)} for v, s in inputs], "sha256": hashlib.sha256(raw).hexdigest()})
    context = {"mode": "CPU reference control" if args.cpu_preflight else "native output boundaries", "native_attestation": args.native_attestation, "native_attestation_sha256": hashlib.sha256(Path(args.native_attestation).read_bytes()).hexdigest()}
    if args.cpu_preflight:
        native = []
        for value, scale in inputs:
            product = reference.round32(value * scale)
            bf16 = reference.quantize(product, reference.power(max(-133, reference.exponent(product) - 7)), "rn") if product else F(0)
            native.extend((reference.bits(product), reference.bits(bf16)))
        device = {"device": "CPU reference control"}
    else:
        source = Path(__file__).with_name("fp4_output_boundary_probe.cu")
        shutil.copy2(source, out / source.name)
        compiler = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        binary = out / "boundary-probe"
        build = [compiler, "-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a", "--ftz=false", "--fmad=false", "-o", str(binary), str(source)]
        context.update(build=build, compiler=subprocess.check_output([compiler, "--version"], text=True), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), python=sys.version)
        compiled = subprocess.run(build, capture_output=True, text=True)
        (out / "compile.stdout").write_text(compiled.stdout)
        (out / "compile.stderr").write_text(compiled.stderr)
        if compiled.returncode:
            reference.save(out / "failure.json", {"classification": "boundary_harness_compile_failure", "returncode": compiled.returncode})
            return compiled.returncode
        disassembler = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
        sass = subprocess.run([disassembler, "--dump-sass", str(binary)], capture_output=True, text=True)
        (out / "boundary-probe.sass").write_text(sass.stdout)
        result = subprocess.run([str(binary), str(out / "boundary-inputs.bin"), str(out / "boundary-outputs.bin")], capture_output=True, text=True)
        (out / "native.stdout").write_text(result.stdout)
        (out / "native.stderr").write_text(result.stderr)
        if result.returncode:
            reference.save(out / "failure.json", {"classification": "boundary_native_execution_failure", "returncode": result.returncode})
            return result.returncode
        device = json.loads(result.stdout)
        outputs = (out / "boundary-outputs.bin").read_bytes()
        if len(outputs) != len(inputs) * 8:
            raise ValueError("boundary output shape failed")
        native = struct.unpack(f"<{len(inputs) * 2}I", outputs)
        context["outputs_sha256"] = hashlib.sha256(outputs).hexdigest()
    rows, multiply_failures, convert_failures = [], [], []
    for i, (value, scale) in enumerate(inputs):
        product = reference.from_bits(native[2 * i])
        converted = reference.from_bits(native[2 * i + 1])
        exact = value * scale
        product_error = abs(product - exact)
        product_bound = reference.power(-24) * abs(exact) + reference.power(-150)
        bf16_quantum = reference.power(max(-133, reference.exponent(product) - 7)) if product else reference.power(-133)
        expected_bf16 = reference.quantize(product, bf16_quantum, "rn")
        conversion_error = abs(converted - product)
        conversion_bound = reference.power(-8) * abs(product) + reference.power(-134)
        if product != reference.round32(exact) or product_error > product_bound:
            multiply_failures.append(i)
        if converted != expected_bf16 or conversion_error > conversion_bound:
            convert_failures.append(i)
        rows.append({"input": i, "value": str(value), "scale": str(scale), "product_bits": native[2 * i], "conversion_bits": native[2 * i + 1], "exact_product": str(exact), "product_error": str(product_error), "product_bound": str(product_bound), "conversion_error": str(conversion_error), "conversion_bound": str(conversion_bound), "product_alternatives": {mode: reference.bits(reference.round32(exact, mode)) for mode in ("rn", "rz", "rd", "ru")}, "conversion_alternatives": {mode: reference.bits(reference.quantize(product, bf16_quantum, mode)) for mode in ("rn", "rz", "rd", "ru")}})
    report = copy.deepcopy(report)
    report["device"] = device
    report["output_boundaries"] = {"float32_multiply": {"status": "fail" if multiply_failures else "pass", "cases": len(inputs), "failed_inputs": multiply_failures}, "bfloat16_conversion": {"status": "fail" if convert_failures else "pass", "cases": len(inputs), "failed_inputs": convert_failures}}
    report["boundary_context"] = context
    report["arithmetic_qualified"] = False
    report["reviews"] = {"kernels_parent": False, "independent": False}
    negative = copy.deepcopy(report)
    negative["output_boundaries"]["float32_multiply"]["status"] = "fail"
    try:
        derive_attested_fp4_bound(1, k=64, report=negative, device=device["device"], output_scale=F(1, 2))
    except FP4QualificationError as exc:
        report["boundary_negative_refusal"] = str(exc)
    else:
        raise AssertionError("a failed scalar boundary received an output bound")
    failures = multiply_failures + convert_failures
    if not failures:
        report["boundary_bound_smoke"] = [derive_attested_fp4_bound(F(7), k=k, report=report, device=device["device"], output_scale=scale, output_dtype=dtype) for k, scale, dtype in ((64, F(1, 2), "float32"), (128, F(896), "bfloat16"), (4096, reference.power(-130), "bfloat16"))]
    try:
        require_t4_device_qualification(report, device=device["device"], physical_device=next(iter(report.get("physical_devices", ())), "unidentified"), comparison="exact_represented_operands")
    except FP4QualificationError as exc:
        report["qualification_refusal"] = str(exc)
    else:
        raise AssertionError("the unreviewed composite contract qualified a device")
    reference.save(out / "boundary-references.json", rows)
    reference.save(out / "attestation.json", report)
    print(json.dumps({"mode": context["mode"], "device": device, "output_boundaries": report["output_boundaries"], "bound_smoke_count": len(report.get("boundary_bound_smoke", [])), "arithmetic_qualified": False}), flush=True)
    return 5 if failures else 0
