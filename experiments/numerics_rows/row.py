"""Run eager numerics units. Check bounded finiteness on exact and diagnostic inputs."""
from __future__ import annotations

import argparse
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import subprocess
import traceback

from experiments.numerics_rows.common import BoundaryError, BoundaryRecorder, Invalid, IMAGES, KERNEL_SHA256, check_runtime, check_source, sha256, write_json
from experiments.numerics_rows.build import IDENTITY_FIELDS, library_identity
from experiments.numerics_rows.canonical import CANONICAL_FIELDS
from experiments.numerics_rows.reference import FiniteBound, compare, selftest


def outcome(unit):
    aggregate_exact(unit)
    if unit["findings"]:
        return "FAIL"
    if not unit["completed"] or unit["cuda_unattributed"]:
        return "STOPPED"
    if unit["invalid"]:
        return "INVALID"
    if unit["unattested"]:
        return "UNATTESTED"
    if missing_cases(unit):
        return "STOPPED"
    return "PASS"


def missing_cases(unit):
    exact = unit["exact"]
    required = {"sweep": exact["sweep"]["elements"],
                "separation": sum(r["both"] for r in exact["separation"]["records"]),
                "scale_index": sum(r.get("launches", 0) for r in exact["scale_index"]["arrays"]),
                "chunk_pairs": exact["chunk_pairs"]["x_elements"], "composition": exact["composition"]["cases"]}
    if unit.get("structure") == "routed_moe":
        required.update({name: exact[name]["elements"] for name in ("swiglu", "down_input", "token_sum")})
    return [name for name, count in required.items() if count <= 0]


def new_unit(scope):
    scope = copy.deepcopy(scope)
    if "m_config" in scope:
        scope["m_config"].setdefault("compiled_kernel_symbols", [])
        scope["m_config"].setdefault("separation_source", None)
    return {**scope, "findings": [], "invalid": [], "unattested": [], "completed": False,
            "cuda_unattributed": False, "boundaries": [], "cuda_errors": [], "diagnostic": [],
            "first_mismatch": None, "decoder_bit_exact": False, "graph_replay_equal": [],
            "exact": {"sweep": {"columns": 0, "experts": 0, "elements": 0, "mismatches": 0},
                      "separation": {"records": []}, "scale_index": {"arrays": []}, "chunk_pairs": {"rows": 0, "x_elements": 0, "mismatches": 0},
                      "down_input": {"elements": 0, "mismatches": 0}, "swiglu": {"elements": 0, "mismatches": 0},
                      "token_sum": {"elements": 0, "mismatches": 0}, "composition": {"cases": 0, "equal": True},
                      "finite_certificate": {"elements": 0, "held": 0, "nonfinite_where_held": 0},
                      "signed_zero_diffs": 0}}


def aggregate_exact(unit):
    sep, scales = unit["exact"]["separation"], unit["exact"]["scale_index"]
    records = sep["records"]
    sep.update(stage=sorted({r.get("stage", "dense") for r in records}))
    for field in ("elements", "lower_excluded", "upper_excluded", "both"):
        sep[field] = sum(r.get(field, 0) for r in records)
    sep["by_sign"] = {sign: [sum(r.get("by_sign", {}).get(sign, [0, 0])[side] for r in records) for side in (0, 1)]
                      for sign in ("positive", "negative")}
    arrays = scales["arrays"]
    scales.update(scales=sorted({r.get("scale", "row") for r in arrays}), bases=sorted({r["base"] for r in arrays if "base" in r}),
                  families=sorted({f for r in arrays for f in r.get("families", [])}))
    for field in ("passes_total", "reads", "reads_covered", "mismatches", "launches"):
        scales[field] = sum(r.get(field, 0) for r in arrays)
    scales["uncovered_reads"] = [{"scale": r["scale"], "read": read["read"]} for r in arrays for read in r.get("uncovered_reads", [])]


def unit_identity(unit):
    config = {k: v for k, v in unit.get("m_config", {}).items() if k not in ("compiled_kernel_symbols", "separation_source")}
    cut = {k: v for k, v in unit.get("tp", {}).items() if k in ("rank", "size", "cut_axis", "cut_range")}
    return copy.deepcopy([unit.get(k) for k in ("cell", "symbol", "decoder", "rung")] + [config, cut])


def finish_unit(unit):
    unit["unit_id"] = unit_identity(unit)
    unit["outcome"] = outcome(unit)
    unit["decoder_bit_exact"] = unit["outcome"] == "PASS" and unit["role"] == "unit"
    unit["unattested_reason"] = unit["unattested"]
    unit["reasons"] = unit["findings"] + unit["invalid"] + unit["unattested"]
    if unit["outcome"] == "STOPPED":
        unit["reasons"] += [f"required cases did not run: {name}" for name in missing_cases(unit)]


def finite_outputs(unit, case, got, bound):
    """Check every output for which all stage bounds hold, before selection by X."""
    import torch

    if got.shape != bound.value.shape:
        raise Invalid(f"{case}: finite-output bound has the wrong shape")
    held = bound.held.to(got.device)
    bad = (~torch.isfinite(got)) & held
    counts = unit["exact"]["finite_certificate"]
    for key, value in (("elements", got.numel()), ("held", int(held.sum())), ("nonfinite_where_held", int(bad.sum()))):
        counts[key] += value
    if bool(bad.any()):
        unit["findings"].append(f"{case}: {int(bad.sum())} nonfinite outputs where the finite certificate holds")


def finite_reference(reference):
    import torch
    if not bool(torch.isfinite(reference).all()):
        raise Invalid("reference is not finite before the native call")


def bind_library_identity(library, proof, precondition):
    identity = {k: proof[k] for k in IDENTITY_FIELDS}
    if identity["library_so_sha256"] != library["library"]["sha256"] or any(precondition.get(k) != v for k, v in identity.items()):
        raise Invalid("H1 code identity differs from the row's measured library")
    library.update(proof)


def check_served_identity(unit, served, measured):
    """Refuse a mapped library mismatch before any numerics case can run."""
    unit["arm"].update(loaded_modules=served, library_identity_verified=False)
    if not served and unit["decoder"] == "native_window_gemm":
        unit["arm"]["library_identity_verified"] = None
        return
    if unit["decoder"] == "native_window_gemm" or not served or any(
            not measured.get(key) or r.get(key) != measured[key] for r in served for key in CANONICAL_FIELDS):
        raise Invalid("served library code or cubin identity differs from the H1/P build")
    unit["arm"]["library_identity_verified"] = True


def check_served_after(unit, before, after):
    unit["arm"]["loaded_modules_after"] = after
    unit["arm"]["library_identity_unchanged"] = False
    fields = ("name", "path", "library_so_sha256", "executable_mappings")
    if [[r[k] for k in fields] for r in before] != [[r[k] for k in fields] for r in after]:
        raise Invalid("served library mapping or bytes changed during the cases")
    unit["arm"]["library_identity_unchanged"] = True


def audit_served(unit, served, measured, out, cache):
    """Run the proof's byte audit and SASS/PTX witnesses on the mapped file."""
    from experiments.numerics_rows.serving_build import compare_binaries

    records = unit["arm"]["library_identity_comparisons"] = []
    for library in served:
        key = (measured["library_so_sha256"], library["library_so_sha256"])
        if key not in cache:
            directory = out / library["library_so_sha256"]
            try:
                comparison = compare_binaries(measured, library, measured["library"]["sass"]["path"], directory)
            except Exception as error:
                unit["arm"]["library_identity_verified"] = False
                raise Invalid(f"served library byte audit: {error}") from error
            path = directory / "comparison.json"
            write_json(path, comparison)
            cache[key] = {"path": str(path), "sha256": sha256(path), "identity_equal": comparison["identity_equal"],
                          "unexplained_bytes": comparison["raw_file_audit"]["unexplained_bytes"] + sum(r["unexplained_bytes"] for r in comparison["cubin_audits"]),
                          "sass_token_masked_equal": comparison["sass_token_masked_equal"], "ptx": comparison["ptx"]}
        records.append(cache[key])
        if not cache[key]["identity_equal"]:
            unit["arm"]["library_identity_verified"] = False
            raise Invalid("served library byte audit or SASS/PTX witness differs from H1/P")


def compare_output(unit, case, got, ref, mask=None, operands=None, *, bound=None):
    import torch

    if got.shape != ref.shape or got.dtype != ref.dtype:
        unit["findings"].append(f"{case}: wrong shape or dtype")
        return {"equal": False, "mismatches": max(got.numel(), ref.numel())}
    finite_reference(ref)
    if bound is not None:
        finite_outputs(unit, case, got, bound)
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


def dispatch(unit, expected, emitted, loaded, lane_reason=None):
    unit["arm"].update(emitted_pairs=emitted, loaded_modules=loaded, lane_reason=lane_reason)
    if not emitted:
        if lane_reason:
            unit["unattested"].append(f"NOT_EXERCISED: {lane_reason}")
        else:
            unit["findings"].append("cell pair did not dispatch and has no named predicate refusal")
    elif any(tuple(pair) != tuple(expected) for pair in emitted):
        unit["findings"].append("telemetry pair is not the unit's exact pair")
    if not loaded and unit["decoder"] != "native_window_gemm":
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


def dot_reference(a, weight, a_scale, scale, *, exact=True, with_bounds=False, splits=1, top_k=0, router_weight=None):
    import torch

    # Sparse exact cases use only the columns that the quantizer actually kept nonzero.
    outputs, masks, bounds = [], [], []
    for row in range(a.shape[0]):
        columns = (a[row] != 0).nonzero().flatten()
        if columns.numel():
            products = weight[:, columns.to(weight.device)].to(a.device).double() * a[row, columns].double()[None, :]
            mask = x_products(products) if exact else torch.ones(weight.shape[0], dtype=torch.bool, device=a.device)
            acc = products.sum(-1)
        else:
            mask = torch.ones(weight.shape[0], dtype=torch.bool, device=a.device)
            acc = torch.zeros(weight.shape[0], dtype=torch.float64, device=a.device)
            products = torch.zeros(weight.shape[0], 1, dtype=torch.float64, device=a.device)
        bound = FiniteBound.dot(products, weight.shape[1], splits, top_k)
        value = acc.float()
        if a_scale is not None:
            value = value * a_scale[row].float()
            bound = bound.multiply(a_scale[row])
        value = value * scale.float()
        bound = bound.multiply(scale)
        if router_weight is not None:
            value = value * router_weight[row].float()
            bound = bound.multiply(router_weight[row])
        bounds.append(bound.rounded(bf16=True))
        outputs.append(value.to(torch.bfloat16))
        masks.append(mask)
    result = torch.stack(outputs), torch.stack(masks)
    return (*result, FiniteBound.join(bounds, stack=True)) if with_bounds else result


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
        from experiments.numerics_rows.population import store_for

        self.entry, self.cut, self.store = entry, cut, store_for(entry)
        self.device = "cuda"
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

    def reference(self, a, a_scale, exact=True, splits=1):
        import torch

        parts, masks, bounds = [], [], []
        for _, weight, scale, _ in self.roles:
            if self.family == "e2m1":
                scale = torch.tensor([float(scale)], device=weight.device, dtype=torch.float32) / torch.tensor(self.gs, device=weight.device, dtype=torch.float32)
            y, mask, bound = dot_reference(a, weight, a_scale, scale, exact=exact, with_bounds=True, splits=splits)
            parts.append(y)
            masks.append(mask)
            bounds.append(bound)
        return torch.cat(parts, 1), torch.cat(masks, 1), FiniteBound.join(bounds, 1)

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
    reference, mask, bound = adapter.reference(a, a_scale, exact=exact, splits=unit["m_config"]["k_split"])
    finite_reference(reference)
    if exact:
        require_x(mask, sweep=all(len(columns) <= 1 for _, columns, _ in recipe))
    try:
        got = recorder.call("dense:apply", "apply", lambda: opaque_apply(adapter.apply, recorder, x))
    except BoundaryError as error:
        if error.attributed_to is None:
            adapter.staged(q, sf, recorder)
        raise
    composition = adapter.staged(q, sf, recorder)
    finite_outputs(unit, "dense:apply", got, bound)
    finite_outputs(unit, "dense:staged", composition, bound)
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
        unit["exact"]["separation"]["records"].append({k: v for k, v in record.items() if k not in ("recipes", "uncovered")})
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


def diagnostic_cells(units):
    result = []
    for cell in sorted({unit["cell"] for unit in units}):
        owned = [u for u in units if u["cell"] == cell and u["role"] == "unit"]
        failed = sum(u["outcome"] == "FAIL" for u in owned)
        passed = sum(u["outcome"] == "PASS" for u in owned)
        result.append({"id": cell, "verdict": "FAIL" if failed else "PASS" if owned and passed == len(owned) else "UNATTESTED",
                       "units_pass": passed, "units_failed": failed, "units_unattested": len(owned) - failed - passed})
    return result


def planned_units(contract, image_digest, population, manifest):
    """Enumerate every expected unit before a loader or native call can fail."""
    from experiments.numerics_rows.cases import admitted_units, configurations
    from experiments.numerics_rows.population import module_key

    if manifest.get("sample"):
        raise Invalid("a C1 sample cannot supply a row manifest")
    if manifest.get("generation_finished") is not True:
        raise Invalid("C1 generation is not finished")
    modules = {module_key(m): m for m in manifest["modules"]}
    names = [module_key(e) for e in population["modules"]]
    if len(modules) != len(manifest["modules"]) or set(names) != set(modules) or len(set(names)) != len(names):
        raise Invalid("case manifest modules differ from the population")
    groups = []
    for entry in population["modules"]:
        whole = modules[module_key(entry)]
        for cut in ({"rank": 0, "size": 1}, {"rank": 0, "size": 2}, {"rank": 1, "size": 2}):
            matching = [c for c in whole["cuts"] if c["tp"] == cut]
            if len(matching) != 1 or not matching[0]["m_configs"]:
                raise Invalid("case manifest omitted or repeated a required cut or M configuration")
            module, units = {**whole, **matching[0]}, []
            expected_roles = {(0, name) for name, _ in entry["roles"]} if module["structure"] == "dense" else {
                (expert, name) for expert in range(entry["experts"]) for name in ("gate_proj", "up_proj", "down_proj")}
            roles = [(r["expert"], r["name"]) for r in module["roles"]]
            if len(roles) != len(expected_roles) or set(roles) != expected_roles:
                raise Invalid("case manifest omitted or repeated a declared role")
            expected = configurations(**module["launch_geometry"], library=entry["library"], sms=population["sms"], structure=module["structure"])
            if module["m_configs"] != expected:
                raise Invalid("case manifest omitted or changed an expected M configuration")
            for config in module["m_configs"]:
                scopes = admitted_units(contract, image_digest, entry["contract_family"], module["structure"], module["q256"], entry["pair"], config["M"])
                if not scopes:
                    raise Invalid("population pair has no matching contract unit")
                for scope in scopes:
                    unit = new_unit({**scope, "module": entry["name"], "symbol": entry["pair"][0], "decoder": entry["pair"][1], "rung": module["q256"], "structure": module["structure"],
                                     "m_config": config, "tp": dict(cut),
                                     "arm": {"selector_env": entry.get("selector_env", {}), "profiler_kernels": []}})
                    if cut["size"] > 1 and entry["contract_family"] != "TESSERA_E4M3_K1":
                        unit["role"], unit["admitted_by_v68"] = "observation", False
                    finish_unit(unit)
                    units.append(unit)
            groups.append((entry, module, cut, units))
    return groups


def check_manifest_source(manifest, commit, contract_path):
    if manifest.get("source_commit") != commit or manifest.get("contract_sha256") != sha256(contract_path):
        raise Invalid("case manifest source commit or contract differs from the row")


def rule_receipt(rule, freeze, h1, h1_path):
    if set(freeze.get("decisions", {})) != {f"D{i}" for i in range(1, 9)}:
        raise Invalid("freeze record must include the reviewed decisions D1 through D8")
    job = h1.get("flux_job_id")
    if not job:
        job = (h1_path.parent.parent / "jobid").read_text().strip()
    if not job or h1.get("preconditions_confirmed") is not True:
        raise Invalid("rule receipt requires a confirmed H1 job")
    preconditions = {name: {"result": "PASS", "h1_job": job} for name in ("P1", "P2", "P3", "P4", "P5")}
    if "p3_receipt" in h1:
        preconditions["P3"]["cpu_receipt"] = h1["p3_receipt"]
    if h1["P2"].get("result") != "PASS":
        preconditions["P2"]["result"] = h1["P2"]["result"]
    return {"path": str(rule), **freeze, "preconditions": preconditions,
            "h1_path": str(h1_path), "h1_sha256": sha256(h1_path)}


def check_rule_freeze(rule, freeze, *, canary=False):
    if freeze.get("sha256") != sha256(rule) or not freeze.get("reviewer"):
        raise Invalid("rule is not the reviewed rule")
    if canary:
        if freeze.get("canary") is not True or freeze.get("frozen_utc") is not None:
            raise Invalid("canary record must say canary=true and frozen_utc=null")
    elif freeze.get("canary") or not freeze.get("frozen_utc"):
        raise Invalid("rule is not the reviewed, frozen rule")


def selected_groups(groups, size=None, rank=None):
    if size is None and rank is None:
        return groups
    if size not in (1, 2) or rank is None or not 0 <= rank < size:
        raise Invalid("select both a valid TP size and TP rank")
    selected = [group for group in groups if group[2] == {"size": size, "rank": rank}]
    if not selected:
        raise Invalid("selected TP cut is absent from the validated full manifest")
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "out", "manifest", "population", "build", "q1", "h1", "rule", "freeze"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--image", choices=("G", "N"), required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--build-provenance", type=Path, required=True)
    p.add_argument("--module", action="append", default=[], help="Explicit module name in the signed population; repeat for a subset")
    p.add_argument("--canary", action="store_true", help="Uncounted pre-freeze run; its receipt cannot be joined")
    p.add_argument("--tp-size", type=int, choices=(1, 2))
    p.add_argument("--tp-rank", type=int, choices=(0, 1))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "tessera.numerics_row.v2", "canary": args.canary,
               "row": {"name": args.name, "flux_job_id": os.environ.get("FLUX_JOB_ID"), "host": socket.gethostname(),
                       "utc_start": datetime.now(timezone.utc).isoformat(), "utc_end": None, "exit_status": None, "stop_cause": None,
                       "finalization_finished": False},
               "tool": {}, "rule": {}, "runtime": {}, "tessera": {}, "probe": {}, "inputs": {},
               "modules": [], "units": [], "negative_controls": [], "diagnostic_cells": [], "findings": []}
    from experiments.numerics_rows.gpu_time import RowMeter

    meter = RowMeter()
    try:
        run(args, receipt)
    except Exception as error:
        receipt["row"].update(utc_end=datetime.now(timezone.utc).isoformat(), exit_status=3, stop_cause=str(error), finalization_finished=False)
        if isinstance(error, Invalid):
            receipt["row"]["outcome"] = "INVALID"
            for unit in receipt["units"]:
                unit["invalid"].append(str(error))
                unit["completed"] = True
        receipt["findings"].append(traceback.format_exc())
        for unit in receipt["units"]:
            finish_unit(unit)
        receipt["diagnostic_cells"] = diagnostic_cells(receipt["units"])
        write_json(args.out / "numerics_row.json", receipt)
        raise SystemExit(3) from error
    finally:
        receipt["row"]["gpu_minutes"] = meter.finish()
        write_json(args.out / "numerics_row.json", receipt)


def run(args, receipt):
    from experiments.numerics_rows.population import load_signed_population, select_modules

    receipt["row"]["finalization_finished"] = False
    population, population_digest = load_signed_population(args.population)
    population = select_modules(population, args.module)
    runtime = check_runtime(args.image)
    sys.path.insert(0, str(args.source / "src"))
    from experiments.numerics_rows.serving_build import check_compile_paths, check_retained_source

    retained_source = check_retained_source(args.source, args.build, args.build_provenance)
    commit = retained_source["commit"]
    receipt["tessera"]["retained_source"] = retained_source
    compile_paths = check_compile_paths(args.build, args.build_provenance)
    receipt["tessera"]["serving_compile"] = compile_paths
    import tessera
    import torch
    from experiments import dense_fused_oracle as dense
    from experiments.numerics_rows.h1 import negative_controls
    from experiments.numerics_rows.cases import DTable
    from experiments.numerics_rows.routed import Routed
    from experiments.numerics_rows import run_routed
    from experiments.numerics_rows.scale_launch import check as scale_index_checks
    from experiments.numerics_rows.sass_slice import match_library
    from experiments.numerics_rows.common import register_native_ops, source_digest

    tool_root = Path(__file__).resolve().parents[2]
    receipt["tool"] = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tool_root, text=True).strip(),
                       "tree_sha256": source_digest(tool_root, ".")}
    torch.set_num_threads(1)
    runtime["native_ops"] = register_native_ops(args.image)
    receipt["runtime"] = runtime
    write_json(args.out / "native_ops.json", runtime["native_ops"])
    if torch.cuda.get_device_capability() != (12, 1):
        raise Invalid("device capability differs from the sm_121 cell")
    freeze = json.loads(args.freeze.read_text())
    check_rule_freeze(args.rule, freeze, canary=getattr(args, "canary", False))
    receipt["canary"] = getattr(args, "canary", False)
    manifest, built, h1 = (json.loads(path.read_text()) for path in (args.manifest, args.build, args.h1))
    check_manifest_source(manifest, commit, args.source / "src/tessera/serving/runtime_contract.json")
    if manifest["population_sha256"] != population_digest or manifest["q1_table_sha256"] != sha256(args.q1):
        raise Invalid("case manifest input identity differs")
    if len(manifest["modules"]) != len(population["modules"]):
        raise Invalid("case manifest omitted a population module")
    keys = {e["library"] if e["library"] != "triton" else "e4m3mma" for e in population["modules"]}
    if len(keys) != 1:
        raise Invalid("a row must contain one library arm")
    key, = keys
    if h1.get("preconditions_confirmed") is not True or h1["build_sha256"] != sha256(args.build) or h1["P1"]["mismatches"] or any(not v.get("P4") or not v.get("P5", True) for v in h1["libraries"].values()):
        raise Invalid("H1 did not confirm this binary set before the freeze")
    q1 = json.loads(args.q1.read_text())
    contract = json.loads((args.source / "src/tessera/serving/runtime_contract.json").read_text())
    groups = planned_units(contract, runtime["image_digest"], population, manifest)
    groups = selected_groups(groups, getattr(args, "tp_size", None), getattr(args, "tp_rank", None))
    receipt["units"] = [unit for _, _, _, units in groups for unit in units]
    write_json(args.out / "numerics_row.json", receipt)
    if built["commit"] != commit or built["runtime"]["image_digest"] != runtime["image_digest"]:
        raise Invalid("library build does not match the cell")
    slices = {r["key"]: match_library(r, args.out / "binary-identity" / r["key"]) for r in built["libraries"]}
    for library in built["libraries"]:
        bind_library_identity(library, slices[library["key"]], h1["P3"][library["key"]])
        d, = [t for t in q1["tables"] if t["key"] == library["key"]]
        if d["library_so_sha256"] != library["library"]["sha256"] or d["probe_sha256"] != library["probe"]["sha256"]:
            raise Invalid("Q1 D table belongs to a different library or probe")
    library, = [r for r in built["libraries"] if r["key"] == key]
    d, = [t for t in q1["tables"] if t["key"] == key]
    receipt.update({"rule": rule_receipt(args.rule, freeze, h1, args.h1),
               "runtime": {**runtime, "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
                           "driver": __import__("subprocess").check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()},
               "tessera": {"commit": commit, "contract_sha256": sha256(args.source / "src/tessera/serving/runtime_contract.json"), "src_tree_sha256": source_digest(args.source), "kernel_source_sha256": KERNEL_SHA256,
                           "serving_compile": compile_paths, "retained_source": retained_source, "build_provenance_sha256": sha256(args.build_provenance),
                           "build_provenance": str(args.build_provenance),
                           "cell_kernel_build": f"legacy-toolchain/{runtime['vllm']}/{runtime['torch']}", "matches_cell": True},
               "probe": {"source_sha256": library["probe_source_sha256"], "cflags": library["cflags"],
                         "so_sha256": library["probe"]["sha256"], "L_bits": d["L_bits"], "dom_size": d["dom_size"],
                         "d_table_sha256": d["d_table_sha256"], "d_nondecreasing_0_L": d["d_nondecreasing_0_L"],
                         "sass_slice": slices[key], "finite_certificate": DTable(d).finite_certificate},
               "inputs": {"population_sha256": population_digest, "build_sha256": sha256(args.build), "case_manifest_sha256": sha256(args.manifest),
                          "selected_modules": [entry["name"] for entry in population["modules"]],
                          "selected_cuts": [dict(cut) for _, _, cut, _ in groups],
                          "q1_table_sha256": sha256(args.q1), "reference_selftest": selftest(device="cuda")},
               "negative_controls": negative_controls()})
    exit_status = 0
    served_cache = {}
    try:
        with dense.init_vllm_world1():
            for entry, module, cut, units in groups:
                with dense.e4m3_instruction("f16" if entry["library"] == "e4m3" else "e4m3"), dense.dense_lane(entry["library"] != "triton"):
                    routed = module["structure"] == "routed_moe"
                    key = entry["library"] if entry["library"] != "triton" else "e4m3mma"
                    table = DTable(next(t for t in q1["tables"] if t["key"] == key)) if routed else None
                    adapter = Routed(entry, cut, module, table) if routed else Dense(entry, cut, module)
                    receipt["modules"].append(module_receipt(module, adapter, routed))
                    for unit in units:
                        adapter.splits = unit["m_config"]["k_split"]
                        unit["tp"].update(tp_facts(adapter, module, routed))
                        recorder = BoundaryRecorder(torch.cuda.synchronize, unit["boundaries"])
                        try:
                            native_source = Path(next(r for r in compile_paths if r["key"] == key)["native_source_path"])
                            actual, served = identify_served(adapter, routed, native_source, args.out / "served-binaries", served_cache)
                            check_served_identity(unit, served, library)
                            audit_served(unit, served, library, args.out / "served-comparisons", served_cache)
                            adapter.served_identity = served[0] if served else None
                            if receipt["inputs"]["reference_selftest"]["mismatches"] or not all(c["flagged_exactly"] for c in receipt["negative_controls"]):
                                raise Invalid("reference self-test or mutated reference control failed")
                            if any(not role["matches_attested_wire"] for role in module["roles"]):
                                raise Invalid("wire stamp differs from attested_wire")
                            if module["launch_geometry"] != actual_geometry(adapter, routed):
                                raise Invalid("C1 launch geometry differs from the loaded operands")
                            unit["scales"] = scale_receipt(adapter, routed)
                            ready = check_preconditions(unit, scale_equal=unit["scales"]["equal"], routed=routed,
                                slice_matched=slices[key]["matched"], monotonic=next(t for t in q1["tables"] if t["key"] == key)["d_nondecreasing_0_L"])
                            if not ready:
                                unit["completed"] = True
                            else:
                                verify_cases(adapter, unit, module, q1, table)
                                cases_completed = False
                                try:
                                    with native_boundaries({key: actual} if actual is not None else {}, recorder):
                                        if routed:
                                            run_routed.run(adapter, unit, module, recorder)
                                        else:
                                            dense_cases(adapter, unit, module, recorder)
                                            dense_diagnostics(adapter, unit, recorder)
                                        scale_index_checks(adapter, unit, module, recorder)
                                        profile_unit(adapter, unit, recorder, routed=routed)
                                    cases_completed = True
                                except BoundaryError as error:
                                    record_cuda_error(unit, error)
                                finally:
                                    _, after = identify_served(adapter, routed, native_source, args.out / "served-binaries", served_cache)
                                    check_served_after(unit, served, after)
                                if cases_completed:
                                    dispatch(unit, entry["pair"], adapter.pairs(),
                                             after, adapter.info.get("lane_reason"))
                                    unit["completed"] = True
                        except BoundaryError as error:
                            record_cuda_error(unit, error)
                        except Invalid as error:
                            unit["invalid"].append(str(error))
                            unit["completed"] = True
                        finish_unit(unit)
                        write_json(args.out / "numerics_row.json", receipt)
    except Exception as error:
        receipt["row"]["stop_cause"] = str(error)
        receipt["findings"].append(traceback.format_exc())
        exit_status = 3
    if not receipt["units"]:
        receipt["row"]["stop_cause"] = receipt["row"]["stop_cause"] or "no population units ran"
        exit_status = 3
    for unit in receipt["units"]:
        finish_unit(unit)
    for _, _, _, units in groups:
        attach_separation(units)
    for unit in receipt["units"]:
        finish_unit(unit)
    if any(unit["outcome"] in ("FAIL", "STOPPED", "INVALID") for unit in receipt["units"]):
        exit_status = 3
    receipt["units"].extend(historical_units(contract, runtime, receipt["units"]))
    receipt["diagnostic_cells"] = diagnostic_cells(receipt["units"])
    receipt["row"].update(utc_end=datetime.now(timezone.utc).isoformat(), exit_status=exit_status, finalization_finished=True)
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
            ordered, _, certificate = adapter.reference(a, a_scale, exact=False, splits=unit["m_config"]["k_split"])
            finite_reference(ordered)
            got = recorder.call("diagnostic:apply", "apply", lambda: opaque_apply(adapter.apply, recorder, x))
            staged = adapter.staged(q, sf, recorder)
            if not compare(got, staged)["equal"]:
                unit["findings"].append("realistic apply differs from its staged composition")
            reference, totals = [], []
            for _, weight, scale, _ in adapter.roles:
                c = (scale.float() / torch.tensor(adapter.gs, dtype=torch.float32, device=scale.device)).double() if adapter.family == "e2m1" else scale.double()
                aa = a.double() * (a_scale[:, None].double() if a_scale is not None else 1)
                reference.append((aa @ weight.double().T) * c)
                totals.append((aa.abs() @ weight.double().abs().T) * c.abs())
            bound = torch.cat(totals, 1)
            finite_outputs(unit, f"diagnostic:{distribution}:{seed}:apply", got, certificate)
            finite_outputs(unit, f"diagnostic:{distribution}:{seed}:staged", staged, certificate)
            unit["exact"]["composition"]["cases"] += 1
            unit["exact"]["composition"]["equal"] &= compare(got, staged)["equal"]
            unit["diagnostic"].append(diagnostic(got, torch.cat(reference, 1), bound, adapter.cols,
                                                 unit["m_config"]["k_split"], distribution, seed))
            if seed == 0 and distribution == "gauss":
                buffer = torch.empty_like(got)
                graph_record(unit, "dense-staged", lambda: buffer.copy_(adapter.staged(q, sf, recorder)), buffer, recorder)


def resolve_symbols(functions, names, demangled):
    """Resolve complete profiler names to exact symbols from this binary's SASS."""
    normalize = lambda name: re.sub(r"\s+", "", name)
    aliases = {}
    for symbol, name in zip(functions, demangled, strict=True):
        for alias in (symbol, name):
            aliases.setdefault(normalize(alias), set()).add(symbol)
    found, unresolved = [], []
    for name in names:
        matches = aliases.get(normalize(name), set())
        if len(matches) == 1:
            found.append({"profiler_name": name, "symbol": next(iter(matches))})
        elif any(part in name for part in ("routed", "dense", "window", "token_sum")):
            unresolved.append(name)
    return sorted(found, key=lambda r: (r["symbol"], r["profiler_name"])), sorted(unresolved)


def actual_geometry(adapter, routed):
    from tessera.routed_fused import dense_module_launch_enabled

    if routed:
        roles = [adapter.native.gate]
    elif adapter.family == "e2m1":
        roles = adapter.layer.tessera_a4_roles
    else:
        roles = adapter.layer.tessera_native.role_bundles
    merged = not routed and adapter.family == "e4m3" and adapter.library != "triton" and dense_module_launch_enabled()
    rows = sum(role.rows for role in roles) if merged else roles[0].rows
    words = roles[0].tile_words
    words = int(words.reshape(-1)[0]) if hasattr(words, "reshape") else int(words)
    return {"rows": rows, "cols": roles[0].cols, "tile_words": words,
            "blocks": sum(-(-role.rows // 128) for role in roles) if merged else -(-rows // 128), "merged": merged}


def attach_separation(units):
    def identity(unit):
        if unit.get("arm", {}).get("unresolved_kernel_names"):
            return ()
        return tuple(sorted({(r["symbol"], r["library_so_sha256"]) for r in unit["m_config"].get("compiled_kernel_symbols", [])}))
    for unit in units:
        unit["m_config"]["separation_source"] = None
        if not unit.get("needs_separation_transfer"):
            continue
        symbols = identity(unit)
        source = next((other for other in units if other.get("separation_census_complete")
            and other["outcome"] == "PASS" and symbols and identity(other) == symbols
            and all(unit[key] == other[key] for key in ("symbol", "decoder", "rung", "tp"))), None)
        if source is None:
            unit["unattested"].append("incomplete separation: no census with the same compiled kernel symbols and binary")
        else:
            unit["exact"]["separation"] = copy.deepcopy(source["exact"]["separation"])
            unit["m_config"]["separation_source"] = {"unit_id": unit_identity(source), "symbol": symbols[0][0], "library_so_sha256": symbols[0][1]}


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
    unit["arm"]["profiler_kernels"] = kernel_table(profile, limit=100000)
    unit["m_config"]["compiled_kernel_symbols"] = []
    if not routed and adapter.library == "triton":
        return
    library = adapter.served_identity
    if sha256(library["sass"]["path"]) != library["sass"]["sha256"]:
        raise Invalid("SASS symbol list changed")
    from experiments.t4_code.sass_dump import parse_sass

    functions = sorted(parse_sass(Path(library["sass"]["path"]).read_text()))
    demangled = subprocess.check_output(["c++filt"], input="\n".join(functions) + "\n", text=True).splitlines()
    symbols, unresolved = resolve_symbols(functions, [r["name"] for r in unit["arm"]["profiler_kernels"]], demangled)
    unit["m_config"]["compiled_kernel_symbols"] = [{**record, **{k: library[k] for k in IDENTITY_FIELDS if k != "library_code_sections"}} for record in symbols]
    unit["arm"]["unresolved_kernel_names"] = unresolved


def opaque_apply(function, recorder, *args):
    previous = recorder.attribute
    recorder.attribute = False
    try:
        return function(*args)
    finally:
        recorder.attribute = previous


def served_library(adapter, routed):
    from tessera import routed_fused as rf, routed_fused_e2m1 as fe
    if adapter.family == "e2m1":
        return fe._ext()
    if not routed:
        native = adapter.layer.tessera_native
        if native.lane == "triton":
            return None
        # The pinned owner exposes these served roles only through its private tuple.
        keys = {role.library for role in native._PreparedDenseNativeModule__fused}
        if len(keys) != 1:
            raise Invalid("served dense roles do not select one library")
        key, = keys
    else:
        key = adapter.native.library
    return (adapter.native.uniform.module if routed and key != "e2m1" else
            fe._ext() if key == "e2m1" else rf._ext(key))


def identify_served(adapter, routed, source, out, cache):
    try:
        actual = served_library(adapter, routed)
        return actual, unit_libraries(actual, source, out, cache)
    except Exception as error:
        raise Invalid(f"served library identity: {error}") from error


def unit_libraries(actual, source, out, cache):
    if actual is None:
        return []
    from tessera._dev.native_identity import loaded_native_identity

    evidence = loaded_native_identity(actual, source)
    key = evidence["sha256"]
    if key not in cache:
        directory = out / evidence["sha256"]
        data = Path(evidence["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != evidence["sha256"]:
            raise Invalid("served mapped library changed during hashing")
        identity = library_identity(evidence["path"], directory / "cubins", data=data)
        sass = directory / "library.sass"
        with sass.open("w") as stream:
            subprocess.run(["cuobjdump", "-sass", identity["snapshot_path"]], stdout=stream, check=True)
        cache[key] = {**identity, "sass": {"path": str(sass), "sha256": sha256(sass)}}
    return [{"name": actual.__name__, **evidence, **cache[key], "native_source_path": str(source), "native_source_sha256": sha256(source)}]


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
    from experiments.numerics_rows.cases import QuantizerTable, column_coverage, routing, separation_route
    import torch
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
        for record in role["separation"].get("by_m", [role["separation"]]):
            router_weight = 1.0
            if "route" in record:
                route = record["route"]
                if route != separation_route(route["M"], adapter.experts, role["expert"]):
                    raise Invalid("C1 separation route differs from the tested routing census")
                ids, weights = routing(route["M"], adapter.experts, route["count"], route["selected"])
                routes = adapter.route(torch.tensor(ids, device="cuda"), torch.tensor(weights, device="cuda", dtype=torch.float32))
                position = (routes.flat_sorted == route["flat_index"]).nonzero().flatten()
                if position.numel() != 1:
                    raise Invalid("C1 separation route is missing from the native routing")
                router_weight = float(routes.rw_sorted[position[0]])
                if router_weight != route["weight"]:
                    raise Invalid("C1 separation router weight differs from rw_sorted")
            checked = column_coverage(weight.numpy(), scale.numpy(), adapter.family, QuantizerTable(q1, adapter.family, role["gs"]),
                peer=None if peer is None else peer[0].numpy(), peer_scales=None if peer is None else peer_scale.numpy(),
                table=table, stage=record["stage"], lut=role["lut"], router_weight=router_weight)
            coverage(unit, "separation", checked["complete"], record["complete"])
            for field in ("elements", "lower_excluded", "upper_excluded", "both", "recipes"):
                if checked[field] != record[field]:
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
                unit["m_config"].update(compiled_kernel_symbols=[], separation_source=None)
                unit["unattested"].append("NOT_DISPATCHABLE (D1)")
                unit.update(outcome="UNATTESTED", unattested_reason=unit["unattested"], reasons=unit["unattested"])
                unit["unit_id"] = unit_identity(unit)
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
