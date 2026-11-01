"""Audit corrected APIs against retained device outputs without a GPU replay."""
from __future__ import annotations

import copy
from fractions import Fraction
import json
from pathlib import Path

import torch

from tessera.fp4_arithmetic import (
    FP4QualificationError, _outward_float, check_packed_stock_arithmetic,
    require_stock_reference_contract, stock_magnitude_upper,
)


def run(args, reference):
    report_path = Path(args.native_attestation)
    report = json.loads(report_path.read_text())
    physical = report["stock_reference"]["physical_devices"][0]
    source = Path(args.retained_outputs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    corrected = copy.deepcopy(report)
    contracts = []
    for shape in report["stock_reference"]["reference_shapes"]:
        stock = require_stock_reference_contract(report, device=report["device"]["device"],
            physical_device=physical, k=shape[2], shape=shape)
        contracts.append(stock["magnitude_fp64_contract"])
    corrected["stock_reference"]["magnitude_fp64_model"] = "ptx_9_0_f64_fma"
    corrected["stock_reference"]["magnitude_fp64_contracts"] = contracts
    corrected["arithmetic_qualified"] = False
    corrected["reviews"] = {"kernels_parent": False, "independent": False}
    checks = []
    negatives = []
    largest_error = 0.0
    for path in sorted(source.glob("dense-*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False)
        previous = data["bound"]
        k = int(data["rendered_x"].shape[1])
        shape = (int(data["actual"].shape[0]), int(data["actual"].shape[1]), k)
        # The retained scalar is a ceiling of Mobs/(1-GM_old). Therefore its
        # product with (1-GM_old) is a rigorous upper estimate of the same
        # retained GPU observation. No new magnitude operation is measured.
        observed_upper = _outward_float(Fraction(previous["operand_magnitude_upper"]) *
            (1 - Fraction(previous["magnitude_gamma"])))
        magnitude = stock_magnitude_upper(observed_upper, k=k, report=corrected,
            device=corrected["device"]["device"], physical_device=physical, shape=shape)
        global_scale = Fraction(2) ** previous["weight_global_exponent"]
        receipt = check_packed_stock_arithmetic(data["actual"], data["expected"], magnitude,
            k=k, global_scale=global_scale, report=corrected, physical_device=physical)
        receipt.update(source=str(path), shape=list(shape), previous_magnitude=previous["operand_magnitude_upper"],
            recovered_observation_upper=observed_upper, native_execution="retained original output; no replay")
        largest_error = max(largest_error, receipt["max_abs_error"])
        checks.append(receipt)
        try:
            check_packed_stock_arithmetic(data["negative_output"], data["expected"], magnitude,
                k=k, global_scale=global_scale, report=corrected, physical_device=physical)
        except FP4QualificationError as exc:
            negatives.append({"source": str(path), "refusal": str(exc)})
        else:
            raise AssertionError("a retained wrong output passed the corrected comparison")
    for path in sorted(source.glob("grouped-*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False)
        k = int(data["x"].shape[1])
        for i, (first, last) in enumerate(((0, 3), (3, 5))):
            previous = data["segment_bounds"][i]
            actual = data["actual"][first:last]
            expected = data["expected"][first:last]
            shape = (last - first, int(actual.shape[1]), k)
            observed_upper = _outward_float(Fraction(previous["operand_magnitude_upper"]) *
                (1 - Fraction(previous["magnitude_gamma"])))
            magnitude = stock_magnitude_upper(observed_upper, k=k, report=corrected,
                device=corrected["device"]["device"], physical_device=physical, shape=shape)
            receipt = check_packed_stock_arithmetic(actual, expected, magnitude, k=k,
                global_scale=Fraction(2) ** previous["weight_global_exponent"], report=corrected,
                physical_device=physical)
            receipt.update(source=str(path), segment=i, shape=list(shape),
                previous_magnitude=previous["operand_magnitude_upper"], recovered_observation_upper=observed_upper,
                native_execution="retained original output; no replay")
            largest_error = max(largest_error, receipt["max_abs_error"])
            checks.append(receipt)
    if not checks or not negatives:
        raise ValueError("the corrective audit requires retained actual outputs and negative controls")
    result = {"schema": "tessera.fp64_corrective_audit.v1", "physical_device": physical,
        "source_report": str(report_path), "retained_outputs": str(source), "checks": checks,
        "negative_controls": negatives, "checked_segments": len(checks), "negative_refusals": len(negatives),
        "max_abs_error": largest_error, "GPU_exercised": False, "native_replays": 0,
        "arithmetic_qualified": False, "magnitude_basis": "normative PTX 9.0 double FMA contract"}
    reference.save(out / "audit.json", result)
    reference.save(out / "attestation.json", corrected)
    print(json.dumps({key: result[key] for key in ("physical_device", "checked_segments", "negative_refusals",
        "max_abs_error", "GPU_exercised", "native_replays", "arithmetic_qualified")}), flush=True)
    return 0
