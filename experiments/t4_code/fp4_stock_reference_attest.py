"""Characterize the four operations in the original packed stock comparison."""
from __future__ import annotations

import copy
from dataclasses import replace
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

from tessera.fp4_arithmetic import (
    FP4QualificationError, _gamma_exact, check_packed_stock_arithmetic,
    derive_packed_stock_bound, require_probe_contract, stock_magnitude_upper, fp64_magnitude_contract,
)
from tessera.stock import _nvfp4_values, materialize_stock, stock_dequant

MS = (1, 2, 3, 16, 33, 65)
N, K = 64, 256


def _profile(call, out, name):
    from torch.profiler import profile, ProfilerActivity
    call()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(out / (name + ".trace.json")))
    return sorted({event.key for event in prof.events() if str(event.device_type).endswith("CUDA")})


def _tensor_bits(tensor):
    return tensor.detach().cpu().contiguous().view(torch.int32).reshape(-1).tolist()


def _make_unit(device, q=896):
    from tessera.alphabet import grid_for_name
    from tessera.compact_prep import parse_compact_wire, prepare_a4_wire_compact
    from tessera.export import encode_linear, served_recipe
    from tessera.structure import STRUCTURE_DENSE
    from tessera.unit_artifact import parse_unit_artifact
    grid = grid_for_name("E2M1x2")
    recipe = served_recipe(grid, q, STRUCTURE_DENSE)
    index = torch.arange(32 * K, device=device).reshape(32, K)
    weights = ((index % 13) - 6).float().mul(2.0 ** -8).to(torch.bfloat16).repeat(2, 1)
    encoded = encode_linear(weights, grid=grid, q256=q, body=recipe.body, span=recipe.span,
        scale_plane=recipe.scale_plane, window_bits=recipe.window_bits,
        window_seed=recipe.window_seed, window_sigma=recipe.window_sigma,
        channel_sigma=recipe.channel_sigma)
    unit = prepare_a4_wire_compact(parse_compact_wire(encoded.blob, device=device), device=device)
    parsed = parse_unit_artifact(encoded.blob, device="cpu")
    stock = materialize_stock(parsed.unit, parsed.forests, parsed.code)
    return unit, stock, encoded.blob


def _library_patterns(reference, dtype):
    precision = 24 if dtype == torch.float32 else 53
    small = reference.power(-24 if precision == 24 else -53)
    split = 12 if precision == 24 else 27
    return [
        ("product-low-bits", [(F(1) + reference.power(-23), F(1) + reference.power(-23))]),
        ("above-half-positive", [(F(1), F(1)), (reference.power(-split), 3 * reference.power(-(precision + 1 - split)))]),
        ("above-half-negative", [(-F(1), F(1)), (-reference.power(-split), 3 * reference.power(-(precision + 1 - split)))]),
        ("below-half", [(F(1), F(1)), (reference.power(-split), reference.power(-(precision + 1 - split)))]),
        ("cancellation", [(F(1), F(1)), (-F(1), F(1)), (F(1), small)]),
        ("carry", [(F(3, 2), F(3, 2))] * 64),
    ]


def run(args, reference):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cpu" if args.cpu_preflight else "cuda"
    base = json.loads(Path(args.native_attestation).read_text())
    require_probe_contract(base, device=base["device"]["device"])
    if not args.boundary_attestation:
        raise ValueError("the stock comparison requires --boundary-attestation")
    boundaries = json.loads(Path(args.boundary_attestation).read_text())
    if any(boundaries.get("output_boundaries", {}).get(name, {}).get("status") != "pass" for name in ("float32_multiply", "bfloat16_conversion")):
        raise ValueError("the stock comparison has no scalar output contract")
    unit, stock, blob = _make_unit(device)
    (out / "unit.blob").write_bytes(blob)
    stock_weight = stock_dequant(stock).to(device)
    gs = torch.tensor(896.0, dtype=torch.float32, device=device)
    ctx = {"torch": torch.__version__, "torch_git": torch.version.git_version,
           "cuda": torch.version.cuda, "device": device, "native_attestation": args.native_attestation,
           "boundary_attestation": args.boundary_attestation, "image": os.environ.get("ORACLE_IMAGE"),
           "action_key": os.environ.get("PRISMABUILD_ACTION_KEY"), "host": os.environ.get("HOST_NAME"),
           "python": sys.version, "unit_sha256": hashlib.sha256(blob).hexdigest()}
    if args.cpu_preflight:
        reference.save(out / "stock-cpu-preflight.json", {"imports": "pass", "unit_bytes": len(blob),
            "weight_shape": list(stock_weight.shape), "ratio": _tensor_bits(unit.epilogue_for(gs)),
            "GPU_exercised": False, "context": ctx})
        print(json.dumps({"mode": "stock processor preflight", "unit_bytes": len(blob), "weight_shape": list(stock_weight.shape), "GPU_exercised": False}), flush=True)
        return 0
    # This is the original correctness path's explicit setting, not a new default.
    torch.backends.cuda.matmul.allow_tf32 = False
    physical = os.environ["HOST_NAME"] + ":cuda:0"
    ctx.update(allow_tf32=torch.backends.cuda.matmul.allow_tf32,
               float32_matmul_precision=torch.get_float32_matmul_precision(),
               device_name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))
    report = copy.deepcopy(base)
    report["physical_devices"] = [physical]
    report["output_boundaries"] = copy.deepcopy(boundaries["output_boundaries"])
    props = {name: {"status": "pass", "cases": 0, "failures": []} for name in ("activation_division", "ratio_formation", "library_fp32", "magnitude_fp64")}
    records = []
    # All signed codes and finite scale bytes cover the complete rendered-input domain.
    codes = torch.arange(16, device=device).repeat_interleave(127)
    scale_codes = torch.arange(127, dtype=torch.uint8, device=device).repeat(16)
    raw = _nvfp4_values(codes) * scale_codes.view(torch.float8_e4m3fn).float()
    divided = raw / gs
    reciprocal = reference.round32(F(1, 896))
    direct_matches = reciprocal_matches = True
    for i, (code, scale) in enumerate(zip(codes.cpu().tolist(), scale_codes.cpu().tolist())):
        exact_raw = reference.e2m1(code) * reference.ue4m3(scale)
        exact = exact_raw / 896
        got = F(float(divided[i]))
        rn = reference.round32(exact)
        rec = reference.round32(exact_raw * reciprocal)
        direct_matches &= got == rn
        reciprocal_matches &= got == rec
        props["activation_division"]["cases"] += 1
        if got not in (rn, rec):
            props["activation_division"]["status"] = "fail"
            props["activation_division"]["failures"].append(i)
        records.append({"operation": "rendered_activation_division", "code": code, "scale_byte": scale,
            "raw": str(exact_raw), "exact": str(exact), "output_bits": reference.bits(got),
            "rn_divide_bits": reference.bits(rn), "rn_reciprocal_multiply_bits": reference.bits(rec)})
    division_steps = 1 if direct_matches else 2 if reciprocal_matches else 0
    ratio_direct = ratio_reciprocal = True
    for exponent in range(-73, 117):
        g = reference.power(exponent)
        ratio = replace(unit, global_scale=float(g)).epilogue_for(gs)
        actual = F(float(ratio[0]))
        exact = g / 896
        direct = reference.round32(exact)
        rec = reference.round32(g * reciprocal)
        ratio_direct &= actual == direct
        ratio_reciprocal &= actual == rec
        props["ratio_formation"]["cases"] += 1
        if actual not in (direct, rec):
            props["ratio_formation"]["status"] = "fail"
            props["ratio_formation"]["failures"].append(exponent)
        records.append({"operation": "A4WireUnit.epilogue_for", "global_exponent": exponent,
            "exact": str(exact), "output_bits": reference.bits(actual),
            "rn_divide_bits": reference.bits(direct), "rn_reciprocal_multiply_bits": reference.bits(rec)})
    ratio_steps = 1 if ratio_direct else 2 if ratio_reciprocal else 0
    profiles = []
    for dtype, prop, eps in ((torch.float32, "library_fp32", reference.power(-23)), (torch.float64, "magnitude_fp64", reference.power(-52))):
        gamma = _gamma_exact(2 * K, eps)
        for m in MS:
            for name, pairs in _library_patterns(reference, dtype):
                # The actual positive magnitude has nonnegative operands.
                if dtype == torch.float64:
                    pairs = [(abs(a), abs(b)) for a, b in pairs]
                a = torch.zeros((m, K), dtype=torch.float32, device=device)
                w = torch.zeros((N, K), dtype=torch.float32, device=device)
                for column, (x, y) in enumerate(pairs):
                    a[:, column] = float(x)
                    w[:, column] = float(y)
                aa, ww = (a, w) if dtype == torch.float32 else (a.double().abs(), w.double().abs())
                result = aa @ ww.T
                exact = sum((F(float(a[0, column])) * F(float(w[0, column])) for column in range(len(pairs))), F(0))
                magnitude = sum((abs(F(float(a[0, column])) * F(float(w[0, column]))) for column in range(len(pairs))), F(0))
                got = F(float(result[0, 0]))
                error = abs(got - exact)
                props[prop]["cases"] += 1
                if not torch.equal(result, result[0, 0].expand_as(result)) or error > gamma * magnitude:
                    props[prop]["status"] = "fail"
                    props[prop]["failures"].append([m, name])
                # These named products cannot survive an input cast to TF32 or bfloat16.
                if name == "product-low-bits" and dtype == torch.float32 and got != reference.round32(exact):
                    props[prop]["status"] = "fail"
                    props[prop]["failures"].append([m, name, "narrow product"])
                records.append({"operation": prop, "shape": [m, N, K], "pattern": name,
                    "inputs": [[str(x), str(y)] for x, y in pairs], "native_output": str(got),
                    "exact_reference": str(exact), "error": str(error), "derived_bound": str(gamma * magnitude)})
            kernels = _profile(lambda: aa @ ww.T, out, prop + "-m" + str(m))
            profiles.append({"operation": prop, "shape": [m, N, K], "kernels": kernels})
            # Supported conventional dot kernels have K products and at most K sums.
            accepted = ("sgemm", "gemv", "gemmsn_tn_kernel<float", "splitkreduce_kernel") if dtype == torch.float32 else ("dgemm", "gemv", "d884", "scal_kernel<double", "memset (device)")
            if not kernels or not all(any(label in kernel.lower() for label in accepted) for kernel in kernels):
                props[prop]["status"] = "unsupported_kernel"
                props[prop]["failures"].append({"shape": [m, N, K], "kernels": kernels})
    fp64_contracts = [fp64_magnitude_contract({"profiles": profiles}, shape=(m, N, K), device=report["device"]["device"]) for m in MS]
    report["stock_reference"] = {"properties": props, "physical_devices": [physical],
        "input_global_scale": 896, "allow_tf32": False, "activation_division_steps": division_steps,
        "ratio_formation_steps": ratio_steps, "library_fp32_precision": 24,
        "magnitude_fp64_precision": min(item["precision_bits"] for item in fp64_contracts),
        "magnitude_fp64_model": "ptx_9_0_f64_fma", "magnitude_fp64_contracts": fp64_contracts,
        "reference_shapes": [[m, N, K] for m in MS], "current_k": K, "profiles": profiles,
        "weight_global_domain": [-73, 116], "context": ctx}
    report["arithmetic_qualified"] = False
    report["reviews"] = {"kernels_parent": False, "independent": False}
    reference.save(out / "stock-reference-probes.json", records)
    reference.save(out / "attestation.json", report)
    if any(item["status"] != "pass" for item in props.values()):
        print(json.dumps({"mode": "stock-reference probes", "properties": props, "profiles": profiles, "arithmetic_qualified": False}), flush=True)
        return 5
    # Actual packed reader, runtime quantizer, stock renderer, and original GEMM.
    from tessera.kernel_a4 import a4_quantize_activation
    from tessera.kernel_a4_wire import PreparedA4Wire, decode_wire_codes
    actual_rows = []
    for q in (895, 896):
        unit, stock, unit_blob = _make_unit(device, q)
        (out / ("unit-q" + str(q) + ".blob")).write_bytes(unit_blob)
        weight = stock_dequant(stock).to(device)
        native_codes, native_scales = decode_wire_codes(unit)
        if not torch.equal(native_codes.cpu(), stock["weight_packed"]) or not torch.equal(native_scales.cpu(), stock["weight_scale"].view(torch.uint8)):
            raise ValueError("actual packed code or scale bytes differ from stock")
        prepared = PreparedA4Wire([unit], gs)
        for m in (1, 16, 33, 65):
            index = torch.arange(m * K, device=device).reshape(m, K)
            x = ((index % 17) - 8).float().mul(2.0 ** -4).to(torch.bfloat16)
            packed, scales = a4_quantize_activation(x, gs)
            nib = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(m, K)
            rendered = _nvfp4_values(nib) * scales.float().repeat_interleave(16, dim=1) / gs
            measured = float((rendered.double().abs() @ weight.double().abs().T).max())
            upper = stock_magnitude_upper(measured, k=K, report=report, device=report["device"]["device"], physical_device=physical, shape=(m, N, K))
            expected = rendered @ weight.T
            torch.save({"x_bfloat16": x.cpu(), "packed_activation": packed.cpu(), "activation_scale_bytes": scales.cpu().view(torch.uint8),
                "rendered_x": rendered.cpu(), "stock_weight": weight.cpu(), "stock_result": expected.cpu(),
                "weight_global": unit.global_scale, "stored_ratio": prepared.epilogue.cpu()}, out / f"actual-inputs-q{q}-m{m}.pt")
            actual = prepared(x, out_dtype=torch.float32)
            receipt = check_packed_stock_arithmetic(actual, expected, upper, k=K, global_scale=unit.global_scale,
                report=report, physical_device=physical)
            receipt.update(q256=q, M=m, codes_scales="byte exact", global_scale=unit.global_scale,
                actual_bits_sha256=hashlib.sha256(actual.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
            actual_rows.append(receipt)
            allowance, _ = derive_packed_stock_bound(upper, k=K, global_scale=unit.global_scale, report=report,
                device=report["device"]["device"], physical_device=physical, shape=(m, N, K))
            bad = actual.clone()
            bad[0, 0] = expected[0, 0] + max(allowance["atol"] * 4, float(torch.finfo(torch.float32).eps))
            try:
                check_packed_stock_arithmetic(bad, expected, upper, k=K, global_scale=unit.global_scale, report=report, physical_device=physical)
            except FP4QualificationError as exc:
                receipt["negative_control_refusal"] = str(exc)
            else:
                raise AssertionError("a wrong actual-path output passed the stock comparison")
        del prepared
    report["actual_path_regressions"] = actual_rows
    reference.save(out / "actual-stock-comparison.json", actual_rows)
    reference.save(out / "attestation.json", report)
    print(json.dumps({"mode": "complete stock-reference comparison", "physical_device": physical,
        "properties": {name: {"status": item["status"], "cases": item["cases"]} for name, item in props.items()},
        "activation_division_steps": division_steps, "ratio_formation_steps": ratio_steps,
        "actual_path_regressions": len(actual_rows), "arithmetic_qualified": False}), flush=True)
    return 0



def run_original(args, reference):
    from types import SimpleNamespace
    from tessera.alphabet import grid_for_name
    from tessera.fp4_arithmetic import require_stock_reference_contract
    from bench_geometry_e2m1 import run_correctness
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads(Path(args.native_attestation).read_text())
    physical = report["stock_reference"]["physical_devices"][0]
    require_stock_reference_contract(report, device=report["device"]["device"], physical_device=physical, k=K, shape=(1, N, K))
    index = torch.arange(32 * K).reshape(32, K)
    samples = {"gate": ((index % 13) - 6).float().mul(2.0 ** -8).to(torch.bfloat16),
               "up": ((index % 11) - 5).float().mul(2.0 ** -8).to(torch.bfloat16)}
    torch.save(samples, out / "samples.pt")
    if args.cpu_preflight:
        _, stock, blob = _make_unit("cpu", 129)
        reference.save(out / "original-cpu-preflight.json", {"imports": "pass", "samples": {k: list(v.shape) for k, v in samples.items()}, "unit_bytes": len(blob), "stock_shape": list(stock_dequant(stock).shape), "GPU_exercised": False})
        print(json.dumps({"mode": "original comparison processor preflight", "unit_bytes": len(blob), "GPU_exercised": False}), flush=True)
        return 0
    options = SimpleNamespace(packed_reader=True, part="all", qs=(129, 383, 641, 895, 896),
        out=str(out), arithmetic_attestation=args.native_attestation)
    run_correctness(options, grid_for_name("E2M1x2"), samples)
    result = json.loads((out / "correctness.json").read_text())
    rows = result["rows"]
    comparisons = [item for row in rows for item in row["arithmetic_comparisons"]]
    print(json.dumps({"mode": "original packed stock comparison", "rows": len(rows),
        "comparisons": len(comparisons), "max_abs_error": max(row["native_max_abs_error"] for row in rows),
        "arithmetic_qualified": False, "physical_device": physical}), flush=True)
    return 0

