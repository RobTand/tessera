"""Run eager numerics units and write the v2 receipt. Diagnostics do not decide outcomes."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import traceback

from experiments.numerics_rows.common import BoundaryError, BoundaryRecorder, Invalid, IMAGES, KERNEL_SHA256, check_runtime, check_source, sha256, write_json
from experiments.numerics_rows.reference import compare, selftest


def outcome(unit):
    if unit["findings"]:
        return "FAIL"
    if not unit["completed"] or unit["cuda_unattributed"]:
        return "STOPPED"
    if unit["invalid"]:
        return "INVALID"
    if unit["unattested"]:
        return "UNATTESTED"
    return "PASS"


def new_unit(scope):
    return {**scope, "findings": [], "invalid": [], "unattested": [], "completed": False,
            "cuda_unattributed": False, "boundaries": [], "cuda_errors": [], "diagnostic": [],
            "first_mismatch": None, "decoder_bit_exact": False, "graph_replay_equal": [],
            "exact": {"sweep": {"columns": 0, "experts": 0, "elements": 0, "mismatches": 0},
                      "separation": [], "scale_index": [], "chunk_pairs": {"rows": 0, "x_elements": 0, "mismatches": 0},
                      "down_input": {"elements": 0, "mismatches": 0}, "swiglu": {"elements": 0, "mismatches": 0},
                      "token_sum": {"elements": 0, "mismatches": 0}, "composition": {"cases": 0, "equal": True},
                      "signed_zero_diffs": 0}}


def compare_output(unit, case, got, ref, mask=None, operands=None):
    import torch

    if got.shape != ref.shape or got.dtype != ref.dtype:
        unit["findings"].append(f"{case}: wrong shape or dtype")
        return {"equal": False, "mismatches": max(got.numel(), ref.numel())}
    if not bool(torch.isfinite(ref).all()):
        raise Invalid(f"{case}: reference is not finite")
    positions = mask.nonzero() if mask is not None else None
    if mask is not None:
        got, ref = got[mask], ref[mask]
    result = compare(got, ref)
    unit["exact"]["signed_zero_diffs"] += result["signed_zero_diffs"]
    if not result["equal"]:
        unit["findings"].append(f"{case}: exact output differs")
        if unit["first_mismatch"] is None:
            first = dict(result["first_mismatch"])
            if positions is not None:
                first["index"] = positions[first["index"][0]].tolist()
            unit["first_mismatch"] = {"case": case, **first, "operands": operands(first["index"]) if callable(operands) else operands}
    return result


def dispatch(unit, expected, emitted, loaded, allowed, lane_reason=None):
    unit["arm"].update(emitted_pairs=emitted, loaded_modules=loaded, lane_reason=lane_reason)
    if not emitted:
        if lane_reason:
            unit["unattested"].append(f"NOT_EXERCISED: {lane_reason}")
        else:
            unit["findings"].append("cell pair did not dispatch and has no named predicate refusal")
    elif any(tuple(pair) != tuple(expected) for pair in emitted):
        unit["findings"].append("telemetry pair is not the unit's exact pair")
    if (not loaded and unit["decoder"] != "native_window_gemm") or any(module["sha256"] not in allowed for module in loaded):
        unit["findings"].append("loaded module is not the unit's binary")


def coverage(unit, section, complete, claimed):
    if complete:
        return
    if claimed:
        unit["invalid"].append(f"C1 claimed complete {section} coverage")
    else:
        unit["unattested"].append(f"incomplete {section} coverage before the freeze")


def x_products(products):
    """Vector X check on fp64 products of the actual represented dot operands."""
    import numpy as np
    import torch

    array = products.detach().cpu().numpy().astype(np.float64)
    absolute = np.abs(array)
    bits = absolute.view(np.uint64)
    exponent = ((bits >> 52) & 2047).astype(np.int64) - 1023 - 52
    mantissa = (bits & np.uint64(2**52 - 1)) | np.uint64(2**52)
    low = mantissa & (~mantissa + np.uint64(1))
    shift = np.frexp(low.astype(np.float64))[1] - 1
    valuations = np.where(array == 0, 10000, exponent + shift)
    qexp = valuations.min(axis=-1)
    total = absolute.sum(axis=-1)
    zero = total == 0
    q = np.ldexp(np.ones(qexp.shape), np.minimum(qexp, 1023))
    valid = zero | (np.isfinite(total) & (qexp >= -126) & (total < 2**24 * q) & (total < 2.0**128))
    return torch.as_tensor(valid, device=products.device)


def dot_reference(a, weight, a_scale, scale, *, exact=True):
    import torch

    # Sparse exact cases use only the columns that the quantizer actually kept nonzero.
    outputs, masks = [], []
    for row in range(a.shape[0]):
        columns = (a[row] != 0).nonzero().flatten()
        if columns.numel():
            products = weight[:, columns.to(weight.device)].to(a.device).double() * a[row, columns].double()[None, :]
            mask = x_products(products) if exact else torch.ones(weight.shape[0], dtype=torch.bool, device=a.device)
            acc = products.sum(-1)
        else:
            mask = torch.ones(weight.shape[0], dtype=torch.bool, device=a.device)
            acc = torch.zeros(weight.shape[0], dtype=torch.float64, device=a.device)
        value = acc.float()
        if a_scale is not None:
            value = value * a_scale[row].float()
        value = value * scale.float()
        outputs.append(value.to(torch.bfloat16))
        masks.append(mask)
    return torch.stack(outputs), torch.stack(masks)


def diagnostic(got, ref, products_abs, k, splits, distribution, seed):
    import torch
    from experiments.routed_pair_oracle import gamma

    diff = (got.double() - ref.double()).abs()
    denominator = gamma(k + splits, 2**-23) * products_abs
    finite = torch.isfinite(diff)
    rowmax = ref.double().abs().amax(-1, keepdim=True).clamp_min(2**-126)
    ulps = diff / torch.exp2(torch.floor(torch.log2(rowmax)) - 7)
    def number(value):
        value = float(value)
        return value if math.isfinite(value) else None
    return {"distribution": distribution, "seed": seed,
            "max_abs_err": number(diff[finite].max()) if finite.any() else None,
            "max_rel_err": number((diff / ref.double().abs().clamp_min(torch.finfo(torch.float64).tiny))[finite].max()) if finite.any() else None,
            "ulps_row_max": {"max": number(ulps[finite].max()) if finite.any() else None,
                             "p99": number(torch.quantile(ulps[finite], 0.99)) if finite.any() else None},
            "higham_ratio_max": number((diff / denominator.clamp_min(torch.finfo(torch.float64).tiny))[finite].max()) if finite.any() else None,
            "nonfinite": {"got": int((~torch.isfinite(got)).sum()), "ref": int((~torch.isfinite(ref)).sum())}}



@contextlib.contextmanager
def native_boundaries(libraries, recorder):
    """Wrap loaded calls, including calls made inside apply. Restore all functions on exit."""
    from tessera.serving import native_ops
    from tessera import kernel_a4

    saved = []
    def wrap(owner, name, kind):
        function = getattr(owner, name)
        saved.append((owner, name, function))
        def called(*args, **kwargs):
            return recorder.call(name, kind, lambda: function(*args, **kwargs))
        setattr(owner, name, called)
    for library in libraries.values():
        for name in dir(library):
            if name.startswith(("dense_forward", "dense_module", "routed_fused_forward", "token_sum")):
                wrap(library, name, "tessera")
    quantizers = (native_ops.native_fp8_quant, kernel_a4.a4_quantize_activation)
    for module in tuple(sys.modules.values()):
        if module is None or not getattr(module, "__name__", "").startswith("tessera"):
            continue
        for name in ("native_fp8_quant", "a4_quantize_activation"):
            if getattr(module, name, None) in quantizers:
                wrap(module, name, "quantizer")
    try:
        yield
    finally:
        for owner, name, function in reversed(saved):
            setattr(owner, name, function)


class Dense:
    def __init__(self, entry, cut, manifest):
        from experiments import dense_fused_oracle as dense
        from experiments.numerics_rows.cases import decode_roles

        self.entry, self.cut, self.store = entry, cut, dense.Store(entry["root"])
        self.name, self.library = entry["name"], entry["library"]
        self.layer, self.method, self.info = dense.build_served(
            self.store, self.name, mode="resident", tp_rank=cut["rank"], tp_size=cut["size"],
            fused=self.library != "triton", instruction="f16" if self.library == "e4m3" else "e4m3")
        self.scheme = self.store.schemes[self.name]
        wire = self.store.wire(self.name).contiguous().numpy().tobytes()
        if hashlib.sha256(wire).hexdigest() != manifest["wire_sha256"]:
            raise Invalid("loaded wire differs from the C1 reader artifact")
        self.roles = decode_roles(wire, self.scheme, self.name, device="cuda", plan=self.layer.tessera_shard_plan)
        self.family = {"TESSERA_FP8": "e4m3", "TESSERA_NVFP4": "e2m1", "TESSERA_BF16": "value"}[self.scheme["family"]]
        self.cols, self.rows = int(self.layer.tessera_columns), int(self.layer.tessera_rows)
        self.gs = float(self.layer.trellis_input_global_scale) if self.family == "e2m1" else 1.0

    def apply(self, x):
        return self.method.apply(self.layer, x)

    def quantized(self, x, recorder):
        from experiments.numerics_rows.quantizers import quantize
        import torch
        from experiments.routed_pair_oracle import e2m1_table

        pair = recorder.call("reference:quantizer", "quantizer", lambda: quantize(self.family, x, self.gs))
        again = recorder.call("reference:quantizer-repeat", "quantizer", lambda: quantize(self.family, x, self.gs))
        if any(not torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(pair, again)):
            raise Invalid("two quantizer calls gave different bytes")
        q, sf = pair
        if self.family == "e2m1":
            codes = q.view(torch.uint8).reshape(x.shape[0], -1)[:, :self.cols // 2]
            indices = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(x.shape[0], self.cols)
            scale = sf.view(torch.uint8).reshape(x.shape[0], -1)[:, :self.cols // 16].contiguous().view(torch.float8_e4m3fn)
            a = e2m1_table(x.device)[indices.long()] * scale.double().repeat_interleave(16, dim=1)
            return q, sf, a, None
        return q, sf, q.double(), sf.reshape(-1) if self.family == "e4m3" else None

    def reference(self, a, a_scale, exact=True):
        import torch

        parts, masks = [], []
        for _, weight, scale, _ in self.roles:
            if self.family == "e2m1":
                scale = torch.tensor([float(scale)], device=weight.device, dtype=torch.float32) / torch.tensor(self.gs, device=weight.device, dtype=torch.float32)
            y, mask = dot_reference(a, weight, a_scale, scale, exact=exact)
            parts.append(y)
            masks.append(mask)
        return torch.cat(parts, 1), torch.cat(masks, 1)

    def staged(self, q, sf, recorder):
        import torch
        from tessera import routed_fused as rf
        from tessera.routed_fused_e2m1 import dense_forward_quantized

        output = torch.empty(q.shape[0], self.rows, dtype=torch.bfloat16, device=q.device)
        if self.family == "e2m1":
            offset = 0
            for role in self.layer.tessera_a4_roles:
                dense_forward_quantized(role, q, sf, out=output[:, offset:offset + role.rows])
                offset += role.rows
        else:
            native = self.layer.tessera_native
            offset = 0
            for bundle in native.role_bundles:
                out = output[:, offset:offset + bundle.rows]
                if self.library == "triton":
                    part = recorder.call("triton:staged", "tessera", lambda: bundle(q, sf if self.family == "e4m3" else None))
                    out.copy_(part)
                else:
                    role = rf.prepare_dense_role(bundle)
                    counter = torch.zeros(1, dtype=torch.int32, device=q.device)
                    rf.dense_forward(role, q, sf if self.family == "e4m3" else None, out, counter)
                offset += bundle.rows
        return output

    def pairs(self):
        from experiments.dense_fused_oracle import route_record

        record = route_record(self.layer)
        return [] if record is None else [(record["symbol"], record["decoder"])]

    def p4(self):
        import torch

        if self.family == "e2m1":
            return all(torch.equal(role.ratio.view(torch.int32),
                                   (torch.tensor([float(scale)], device="cuda", dtype=torch.float32) / role.gs).view(torch.int32))
                       for role, (_, _, scale, _) in zip(self.layer.tessera_a4_roles, self.roles))
        bundles = self.layer.tessera_native.role_bundles
        return all(torch.equal(bundle.scale.reshape(-1).view(torch.int32), scale.reshape(-1).view(torch.int32)) for bundle, (_, _, scale, _) in zip(bundles, self.roles))


def run_dense_case(adapter, unit, recipe, recorder, *, exact=True):
    import torch

    m = unit["m_config"]["M"]
    x = torch.zeros(m, adapter.cols, dtype=torch.bfloat16, device="cuda")
    for row, columns, values in recipe:
        x[row % m, columns] = torch.tensor(values, dtype=torch.bfloat16, device="cuda")
    q, sf, a, a_scale = adapter.quantized(x, recorder)
    reference, mask = adapter.reference(a, a_scale, exact=exact)
    if exact:
        require_x(mask, sweep=all(len(columns) <= 1 for _, columns, _ in recipe))
    try:
        got = recorder.call("dense:apply", "apply", lambda: opaque_apply(adapter.apply, recorder, x))
    except BoundaryError as error:
        if error.attributed_to is None:
            adapter.staged(q, sf, recorder)
        raise
    composition = adapter.staged(q, sf, recorder)
    composed = compare(got, composition)
    unit["exact"]["composition"]["cases"] += 1
    unit["exact"]["composition"]["equal"] &= composed["equal"]
    if not composed["equal"]:
        unit["findings"].append("apply differs from the same staged launches")
    def actual_operands(index):
        row, column = index
        offset = 0
        for name, w, scale, _ in adapter.roles:
            if column < offset + w.shape[0]:
                columns = (a[row] != 0).nonzero().flatten()
                return {"role": name, "columns": columns.tolist(), "a": a[row, columns].tolist(),
                        "w": w[column - offset, columns].tolist(), "a_scale": None if a_scale is None else float(a_scale[row]),
                        "scale": scale.reshape(-1).tolist() if scale.numel() == 1 else float(scale[column - offset])}
            offset += w.shape[0]
    return compare_output(unit, str(recipe), got, reference, mask, actual_operands) if exact else (got, reference, a)


def dense_cases(adapter, unit, manifest, recorder):
    from experiments.numerics_rows.cases import chunk_pairs, split_columns

    m = unit["m_config"]["M"]
    for role in manifest["roles"]:
        record = role["separation"]
        coverage(unit, "separation", record["complete"], manifest["complete"])
        unit["exact"]["separation"].append({k: v for k, v in record.items() if k not in ("recipes", "uncovered")})
        pending = [(recipe["column"], activation) for recipe in record["recipes"] for activation in recipe["values"]]
        for start in range(0, len(pending), m):
            batch = [(row, [column], [activation]) for row, (column, activation) in enumerate(pending[start:start + m])]
            result = run_dense_case(adapter, unit, batch, recorder)
            unit["exact"]["sweep"]["columns"] += len(batch)
            unit["exact"]["sweep"]["elements"] += len(batch) * adapter.rows
            unit["exact"]["sweep"]["mismatches"] += result["mismatches"]
    for pair in chunk_pairs(adapter.cols, unit["m_config"]["chunk_size"]):
        result = run_dense_case(adapter, unit, [(row, list(pair), [1, 1]) for row in range(m)], recorder)
        unit["exact"]["chunk_pairs"]["rows"] += m
        unit["exact"]["chunk_pairs"]["x_elements"] += result.get("elements", 0)
        unit["exact"]["chunk_pairs"]["mismatches"] += result["mismatches"]
    if unit["m_config"]["k_split"] > 1:
        columns = split_columns(adapter.cols, unit["m_config"]["chunk_size"], unit["m_config"]["k_split"])
        run_dense_case(adapter, unit, [(row, columns, [1] * len(columns)) for row in range(m)], recorder)


def cell_verdicts(units):
    result = []
    for cell in sorted({unit["cell"] for unit in units}):
        owned = [u for u in units if u["cell"] == cell and u["role"] == "unit"]
        failed = sum(u["outcome"] == "FAIL" for u in owned)
        passed = sum(u["outcome"] == "PASS" for u in owned)
        result.append({"id": cell, "verdict": "FAIL" if failed else "PASS" if owned and passed == len(owned) else "UNATTESTED",
                       "units_pass": passed, "units_failed": failed, "units_unattested": len(owned) - failed - passed})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "out", "manifest", "population", "build", "q1", "h1", "rule", "freeze"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--image", choices=("G", "N"), required=True)
    p.add_argument("--name", required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "tessera.numerics_row.v2",
               "row": {"name": args.name, "flux_job_id": os.environ.get("FLUX_JOB_ID"), "host": socket.gethostname(),
                       "utc_start": datetime.now(timezone.utc).isoformat(), "utc_end": None, "exit_status": None, "stop_cause": None},
               "rule": {}, "runtime": {}, "tessera": {}, "probe": {}, "inputs": {},
               "modules": [], "units": [], "negative_controls": [], "cells": [], "findings": []}
    try:
        run(args, receipt)
    except Exception as error:
        receipt["row"].update(utc_end=datetime.now(timezone.utc).isoformat(), exit_status=3, stop_cause=str(error))
        receipt["findings"].append(traceback.format_exc())
        write_json(args.out / "numerics_row.json", receipt)
        raise SystemExit(3) from error


def run(args, receipt):
    commit, runtime = check_source(args.source), check_runtime(args.image)
    sys.path.insert(0, str(args.source / "src"))
    import tessera
    import torch
    from experiments import dense_fused_oracle as dense
    from experiments.numerics_rows.h1 import install_libraries, negative_controls
    from experiments.numerics_rows.cases import admitted_units, DTable
    from experiments.numerics_rows.routed import Routed
    from experiments.numerics_rows import run_routed
    from experiments.numerics_rows.scale_launch import check as scale_index_checks
    from experiments.numerics_rows.sass_slice import match_library
    from experiments.numerics_rows.common import source_digest

    torch.set_num_threads(1)
    if torch.cuda.get_device_capability() != (12, 1):
        raise Invalid("device capability differs from the sm_121 cell")
    freeze = json.loads(args.freeze.read_text())
    if freeze["sha256"] != sha256(args.rule) or not freeze["frozen_utc"] or not freeze["reviewer"]:
        raise Invalid("rule is not the reviewed, frozen rule")
    manifest, population, built, h1 = (json.loads(path.read_text()) for path in (args.manifest, args.population, args.build, args.h1))
    if manifest["population_sha256"] != sha256(args.population) or manifest["q1_table_sha256"] != sha256(args.q1):
        raise Invalid("case manifest input identity differs")
    if len(manifest["modules"]) != len(population["modules"]):
        raise Invalid("case manifest omitted a population module")
    keys = {e["library"] if e["library"] != "triton" else "e4m3mma" for e in population["modules"]}
    if len(keys) != 1:
        raise Invalid("a row must contain one library arm")
    key, = keys
    if h1["build_sha256"] != sha256(args.build) or h1["P1"]["mismatches"] or any(not v.get("P4") or not v.get("P5", True) for v in h1["libraries"].values()):
        raise Invalid("H1 did not confirm this binary set before the freeze")
    q1 = json.loads(args.q1.read_text())
    loaded = install_libraries(built["libraries"])
    if built["commit"] != commit or built["runtime"]["image_digest"] != runtime["image_digest"]:
        raise Invalid("library build does not match the cell")
    contract = json.loads((args.source / "src/tessera/serving/runtime_contract.json").read_text())
    slices = {r["key"]: match_library(r) for r in built["libraries"]}
    for library in built["libraries"]:
        d, = [t for t in q1["tables"] if t["key"] == library["key"]]
        if d["library_so_sha256"] != library["library"]["sha256"] or d["probe_sha256"] != library["probe"]["sha256"]:
            raise Invalid("Q1 D table belongs to a different library or probe")
    library, = [r for r in built["libraries"] if r["key"] == key]
    d, = [t for t in q1["tables"] if t["key"] == key]
    receipt.update({"rule": {"path": str(args.rule), **freeze},
               "runtime": {**runtime, "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
                           "driver": __import__("subprocess").check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()},
               "tessera": {"commit": commit, "src_tree_sha256": source_digest(args.source), "kernel_source_sha256": KERNEL_SHA256,
                           "cell_kernel_build": f"legacy-toolchain/{runtime['vllm']}/{runtime['torch']}", "matches_cell": True},
               "probe": {"source_sha256": library["probe_source_sha256"], "cflags": library["cflags"],
                         "so_sha256": library["probe"]["sha256"], "L_bits": d["L_bits"], "dom_size": d["dom_size"],
                         "d_table_sha256": d["d_table_sha256"], "d_nondecreasing_0_L": d["d_nondecreasing_0_L"],
                         "sass_slice": slices[key]},
               "inputs": {"case_manifest_sha256": sha256(args.manifest), "q1_table_sha256": sha256(args.q1), "reference_selftest": selftest()},
               "negative_controls": negative_controls()})
    exit_status = 0
    try:
        with dense.init_vllm_world1():
            for entry, whole_module in zip(population["modules"], manifest["modules"]):
                cuts = ({"rank": 0, "size": 1}, {"rank": 0, "size": 2}, {"rank": 1, "size": 2})
                for cut in cuts:
                    local, = [c for c in whole_module["cuts"] if c["tp"] == cut]
                    module = {**whole_module, **local}
                    with dense.e4m3_instruction("f16" if entry["library"] == "e4m3" else "e4m3"), dense.dense_lane(entry["library"] != "triton"):
                        routed = module["structure"] == "routed_moe"
                        key = entry["library"] if entry["library"] != "triton" else "e4m3mma"
                        table = DTable(next(t for t in q1["tables"] if t["key"] == key)) if routed else None
                        adapter = Routed(entry, cut, module, table) if routed else Dense(entry, cut, module)
                        receipt["modules"].append(module_receipt(module, adapter, routed))
                        for config in module["m_configs"]:
                            for scope in admitted_units(contract, runtime["image_digest"], entry["contract_family"], module["structure"], module["q256"], entry["pair"], config["M"]):
                                unit = new_unit({**scope, "symbol": entry["pair"][0], "decoder": entry["pair"][1], "rung": module["q256"], "m_config": config,
                                                 "tp": dict(cut), "arm": {"selector_env": entry.get("selector_env", {}), "profiler_kernels": []}})
                                receipt["units"].append(unit)
                                if cut["size"] > 1 and entry["contract_family"] != "TESSERA_E4M3_K1":
                                    unit["role"], unit["admitted_by_v68"] = "observation", False
                                unit["tp"].update(tp_facts(adapter, module, routed))
                                recorder = BoundaryRecorder(torch.cuda.synchronize, unit["boundaries"])
                                try:
                                    if receipt["inputs"]["reference_selftest"]["mismatches"] or not all(c["flagged_exactly"] for c in receipt["negative_controls"]):
                                        raise Invalid("reference self-test or mutated reference control failed")
                                    if any(not role["matches_attested_wire"] for role in module["roles"]):
                                        raise Invalid("wire stamp differs from attested_wire")
                                    unit["scales"] = scale_receipt(adapter, routed)
                                    ready = check_preconditions(unit, scale_equal=unit["scales"]["equal"], routed=routed,
                                        slice_matched=slices[key]["matched"], monotonic=next(t for t in q1["tables"] if t["key"] == key)["d_nondecreasing_0_L"])
                                    if not ready:
                                        unit["completed"] = True
                                    else:
                                        verify_cases(adapter, unit, module, q1, table)
                                        with native_boundaries(loaded, recorder):
                                            if routed:
                                                run_routed.run(adapter, unit, module, recorder)
                                            else:
                                                dense_cases(adapter, unit, module, recorder)
                                                dense_diagnostics(adapter, unit, recorder)
                                            scale_index_checks(adapter, unit, module, recorder)
                                            profile_unit(adapter, unit, recorder, routed=routed)
                                        dispatch(unit, entry["pair"], adapter.pairs(),
                                                 unit_libraries(adapter, routed),
                                                 {r["library"]["sha256"] for r in built["libraries"] if r["key"] == key}, adapter.info.get("lane_reason"))
                                        unit["completed"] = True
                                except BoundaryError as error:
                                    record_cuda_error(unit, error)
                                except Invalid as error:
                                    unit["invalid"].append(str(error))
                                    unit["completed"] = True
                                unit["outcome"] = outcome(unit)
                                unit["decoder_bit_exact"] = unit["outcome"] == "PASS" and scope["role"] == "unit"
                                unit["unattested_reason"] = unit["unattested"]
                                unit["reasons"] = unit["findings"] + unit["invalid"] + unit["unattested"]
                                write_json(args.out / "numerics_row.json", receipt)
    except Exception as error:
        receipt["row"]["stop_cause"] = str(error)
        receipt["findings"].append(traceback.format_exc())
        exit_status = 3
    if not receipt["units"]:
        receipt["row"]["stop_cause"] = receipt["row"]["stop_cause"] or "no population units ran"
        exit_status = 3
    for unit in receipt["units"]:
        unit["outcome"] = outcome(unit)
        unit["decoder_bit_exact"] = unit["outcome"] == "PASS" and unit["role"] == "unit"
        unit["unattested_reason"] = unit["unattested"]
        unit["reasons"] = unit["findings"] + unit["invalid"] + unit["unattested"]
    if any(unit["outcome"] in ("FAIL", "STOPPED", "INVALID") for unit in receipt["units"]):
        exit_status = 3
    receipt["units"].extend(historical_units(contract, runtime, receipt["units"]))
    receipt["cells"] = cell_verdicts(receipt["units"])
    receipt["row"].update(utc_end=datetime.now(timezone.utc).isoformat(), exit_status=exit_status)
    write_json(args.out / "numerics_row.json", receipt)
    raise SystemExit(exit_status)



def dense_diagnostics(adapter, unit, recorder):
    import torch
    from experiments.numerics_rows.run_routed import graph_record

    m = unit["m_config"]["M"]
    for seed in (0, 1, 2):
        for distribution in ("gauss", "outlier", "zero"):
            x = torch.randn(m, adapter.cols, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed)).bfloat16()
            if distribution == "zero":
                x.zero_()
            elif distribution == "outlier":
                x.flatten()[::256] *= 128
                x = (x.float() * torch.exp2(torch.linspace(-8, 8, m, device="cuda"))[:, None]).bfloat16()
                if adapter.family == "e2m1":
                    x = (x.float() * min(1.0, 6 * 448 / (float(x.abs().max()) * adapter.gs))).bfloat16()
            q, sf, a, a_scale = adapter.quantized(x, recorder)
            got = recorder.call("diagnostic:apply", "apply", lambda: opaque_apply(adapter.apply, recorder, x))
            staged = adapter.staged(q, sf, recorder)
            if not compare(got, staged)["equal"]:
                unit["findings"].append("realistic apply differs from its staged composition")
            reference, totals = [], []
            for _, weight, scale, _ in adapter.roles:
                c = scale.double() / (adapter.gs if adapter.family == "e2m1" else 1)
                aa = a.double() * (a_scale[:, None].double() if a_scale is not None else 1)
                reference.append((aa @ weight.double().T) * c)
                totals.append(aa.abs() @ weight.double().abs().T)
            unit["diagnostic"].append(diagnostic(got, torch.cat(reference, 1), torch.cat(totals, 1), adapter.cols,
                                                 unit["m_config"]["k_split"], distribution, seed))
            if seed == 0 and distribution == "gauss":
                buffer = torch.empty_like(got)
                graph_record(unit, "dense-staged", lambda: buffer.copy_(adapter.staged(q, sf, recorder)), buffer, recorder)


def profile_unit(adapter, unit, recorder, *, routed):
    import torch
    from experiments.routed_pair_oracle import kernel_table
    from experiments.numerics_rows.cases import routing

    m = unit["m_config"]["M"]
    x = torch.zeros(m, adapter.cols, dtype=torch.bfloat16, device="cuda")
    if routed:
        ids, weights = routing(m, adapter.experts, m, 0)
        ids, weights = torch.tensor(ids, device="cuda"), torch.tensor(weights, device="cuda")
        call = lambda: adapter.apply(x, ids, weights)
    else:
        call = lambda: adapter.apply(x)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
        recorder.call("dispatch-profiler:apply", "apply", lambda: opaque_apply(call, recorder))
    unit["arm"]["profiler_kernels"] = kernel_table(profile)


def opaque_apply(function, recorder, *args):
    previous = recorder.attribute
    recorder.attribute = False
    try:
        return function(*args)
    finally:
        recorder.attribute = previous


def unit_libraries(adapter, routed):
    if not routed and adapter.library == "triton":
        return []
    from tessera._dev.native_identity import loaded_native_identity
    from tessera import routed_fused as rf, routed_fused_e2m1 as fe
    key = ("e2m1" if adapter.family == "e2m1" else adapter.native.library) if routed else adapter.library
    actual = (adapter.native.uniform.module if routed and key != "e2m1" else
              fe._ext() if key == "e2m1" else rf._ext(key))
    evidence = loaded_native_identity(actual, rf.__file__)
    return [{"name": actual.__name__, **evidence}]


def module_receipt(module, adapter, routed):
    roles = module["roles"]
    return {k: module[k] for k in ("name", "family", "structure", "q256", "rows", "cols", "experts", "wire_sha256")} | {
        "tp": dict(adapter.cut), "roles": [{"name": r["name"], "expert": r["expert"]} for r in roles],
        "runs_table": [r["runs_table"] for r in roles], "wire_stamp": [r["wire_stamp"] for r in roles],
        "matches_attested_wire": all(r["matches_attested_wire"] for r in roles), "scales": scale_receipt(adapter, routed)}


def scale_receipt(adapter, routed):
    import torch
    if routed:
        tensors = [getattr(adapter.native, "ratio_" + field) if adapter.family == "e2m1" else getattr(adapter.native, field).scale_all for field in ("gate", "up", "down")]
    elif adapter.family == "e2m1":
        tensors = [r.ratio for r in adapter.layer.tessera_a4_roles]
    else:
        tensors = [r.scale for r in adapter.layer.tessera_native.role_bundles]
    data = b"".join(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes() for t in tensors)
    return {"c_loaded_sha256": hashlib.sha256(data).hexdigest(),
            "reader_formula": "float32(global) / float32(gs)" if adapter.family == "e2m1" else "scale_rows.float() * float(scale_global)",
            "equal": adapter.p4()}


def tp_facts(adapter, module, routed):
    import torch
    if routed:
        tensors = [getattr(adapter.native, field).init_all for field in ("gate", "up", "down")]
        has_init = any(bool(getattr(adapter.native, field).has_init.any()) for field in ("gate", "up", "down"))
    elif adapter.family == "e2m1":
        tensors = [r.init for r in adapter.layer.tessera_a4_roles]
        has_init = any(bool(r.has_init.any()) for r in adapter.layer.tessera_a4_roles)
    else:
        tensors = [r.init_perm for r in adapter.layer.tessera_native.role_bundles]
        has_init = any(r.has_init for r in adapter.layer.tessera_native.role_bundles)
    data = b"".join(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes() for t in tensors if t is not None)
    return {"cut_axis": [r["cut_axis"] for r in module["roles"]], "cut_range": [r["cut_range"] for r in module["roles"]],
            "has_init": has_init, "start_state_sha256": hashlib.sha256(data).hexdigest()}


def verify_cases(adapter, unit, module, q1, table):
    from experiments.numerics_rows.cases import QuantizerTable, column_coverage
    for role in module["roles"]:
        if sha256(role["reader_path"]) != role["reader_sha256"]:
            raise Invalid("C1 reader artifact changed")
        routed = module["structure"] == "routed_moe"
        if routed:
            weight, scale = adapter.weight(role["expert"], role["name"])
            peer_name = "up_proj" if role["name"] == "gate_proj" else "gate_proj"
            peer = adapter.weight(role["expert"], peer_name) if role["name"] != "down_proj" else None
        else:
            _, weight, scale, _ = next(r for r in adapter.roles if r[0] == role["name"])
            weight, scale, peer = weight.cpu(), scale.cpu(), None
        scale = scale.float() / (role["gs"] if adapter.family == "e2m1" else 1)
        if scale.numel() == 1:
            scale = scale.expand(weight.shape[0])
        peer_scale = None if peer is None else peer[1].float() / (role["gs"] if adapter.family == "e2m1" else 1)
        if peer_scale is not None and peer_scale.numel() == 1:
            peer_scale = peer_scale.expand(weight.shape[0])
        # C1 stores the complete value alphabet, including each unit's E2M1 LUT.
        checked = column_coverage(weight.numpy(), scale.numpy(), adapter.family, QuantizerTable(q1, adapter.family, role["gs"]),
            peer=None if peer is None else peer[0].numpy(), peer_scales=None if peer is None else peer_scale.numpy(),
            table=table, stage=role["separation"]["stage"], lut=role["lut"])
        coverage(unit, "separation", checked["complete"], role["separation"]["complete"])
        for field in ("elements", "lower_excluded", "upper_excluded", "both", "recipes"):
            if checked[field] != role["separation"][field]:
                raise Invalid("C1 separation does not match the loaded reader operands")


def historical_units(contract, runtime, units):
    result, seen = [], set()
    for source in units:
        if runtime["local_id"] != IMAGES["G"]["local_id"] or source["decoder"] != "native_routed_window_classes_e4m3mma":
            continue
        cell, = [c for c in contract["lane_eligibility"]["cells"] if c["id"] == source["cell"]]
        for pair in cell["executes"]:
            if pair["decoder"] == source["decoder"]:
                continue
            for rung in pair.get("rungs_q256", cell["rungs_q256"]):
                key = source["cell"], pair["symbol"], pair["decoder"], rung, json.dumps(source["m_config"], sort_keys=True), source["tp"]["size"], source["tp"]["rank"]
                if key in seen:
                    continue
                seen.add(key)
                unit = new_unit({k: source[k] for k in ("cell", "m_config", "tp", "arm")})
                unit.update(symbol=pair["symbol"], decoder=pair["decoder"], rung=rung, admitted_by_v68=True, role="unit", completed=True)
                unit["unattested"].append("NOT_DISPATCHABLE (D1)")
                unit.update(outcome="UNATTESTED", unattested_reason=unit["unattested"], reasons=unit["unattested"])
                result.append(unit)
    return result


def require_x(mask, *, sweep=False):
    valid = bool(mask.all()) if sweep else bool(mask.any(-1).all())
    if not valid:
        raise Invalid("X false in a case that C1 claimed was exact")


def check_preconditions(unit, *, scale_equal, routed=False, slice_matched=True, monotonic=True):
    if not scale_equal:
        unit["findings"].append("P4: loaded scale differs from the reader formula")
    if routed and (not slice_matched or not monotonic):
        unit["unattested"].append("P3 or D monotonicity is not established")
        return False
    return True


def record_cuda_error(unit, error):
    unit["cuda_errors"].append({"call": error.record["call"], "attributed_to": error.attributed_to, "message": str(error)})
    if error.attributed_to:
        unit["findings"].append("CUDA error attributed to a Tessera launch")
    else:
        unit["cuda_unattributed"] = True


if __name__ == "__main__":
    main()
