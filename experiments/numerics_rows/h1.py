"""Check the numerics preconditions on synthetic wires before the freeze."""
from __future__ import annotations

from experiments.numerics_rows.common import error_text, error_trace

import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from experiments.numerics_rows.common import BoundaryError, BoundaryRecorder, Invalid, check_runtime, check_source, load_build, register_native_ops, sha256, write_json
from experiments.numerics_rows.reference import compare, selftest


def install_libraries(records):
    from tessera import routed_fused as rf, routed_fused_e2m1 as fe
    from tessera._dev.native_identity import loaded_native_identity

    loaded = {}
    for rec in records:
        path = rec["library"]["path"]
        spec = importlib.util.spec_from_file_location(rec["name"], path)
        lib = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lib)
        loaded_native_identity(lib, rf.__file__, expected_sha256=rec["library"]["sha256"])
        loaded[rec["key"]] = lib
    rf._ext = lambda key: loaded[key]
    if "e2m1" in loaded:
        fe._LIB, fe._LIB_PREFETCH = loaded["e2m1"], 0
    return loaded


def p2_ties(recorder):
    import numpy as np
    import torch
    from tessera import window_gemm as wg
    from window_gemm_grouped_support import Expert

    expert = Expert(256, 256, (4,) * 256, 31, family="e4m3")
    expert.unit.native.fill_(0x38)  # Every decoded E4M3 value is 1.
    bundle = wg.prepare_window_gemm(expert.unit, block_m=64, block_n=64, block_k=64, quantizer="native")
    # These outputs are exact BF16 midpoints with alternating even and odd lower values.
    scales = torch.tensor([1 + (2 * i + 1) * 2**-8 for i in range(128)] * 2, device="cuda")
    bundle = dataclasses.replace(bundle, scale=scales)
    x = torch.zeros(4, 256, dtype=torch.float8_e4m3fn, device="cuda")
    x[:, 0] = 1
    a = torch.ones(4, device="cuda")
    got = recorder.call("P2:ties", "tessera", lambda: bundle(x, a))
    want = scales.to(torch.bfloat16).expand_as(got)
    ties_equal = torch.equal(got.view(torch.int16), want.view(torch.int16))
    # Search represented scale triples whose two multiplication orders differ after BF16 RN.
    rng = np.random.default_rng(0)
    s = rng.uniform(0.5, 2, 500_000).astype(np.float32)
    t = rng.uniform(0.5, 2, 500_000).astype(np.float32)
    acc = np.float32(3)
    ordered = ((acc * s).astype(np.float32) * t).astype(np.float32)
    changed = (acc * (s * t).astype(np.float32)).astype(np.float32)
    def bbits(v):
        bits = v.view(np.uint32)
        return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    indices = np.flatnonzero(bbits(ordered) != bbits(changed))[:256]
    order_equal = bool(len(indices))
    for index in indices:
        scale = torch.full((256,), float(t[index]), device="cuda")
        launch = dataclasses.replace(bundle, scale=scale)
        x[:, 0] = 3
        a.fill_(float(s[index]))
        actual = recorder.call(f"P2:order:{index}", "tessera", lambda: launch(x, a))
        expected = int(bbits(ordered[index:index + 1])[0])
        order_equal &= bool((actual.view(torch.int16).to(torch.int32).bitwise_and(65535) == expected).all())
    return {"ties": 128, "ties_equal": ties_equal, "order_cases": len(indices), "order_equal": order_equal,
            "result": "PASS" if ties_equal and order_equal else "FAIL"}


def setup_cases():
    from experiments.numerics_rows.dependency_code import canonical

    target = {"backend": "cuda", "arch": 121, "warp_size": 32}
    repack = "_window_repack_kernel"
    cases = [(repack, 4096, 1024, 0, 4096, {(14,): 256}),
             (repack, 2048, 4096, 1024, 1024, {(14,): 256})]
    cases.extend(("_unpack_body_kernel", rows, cols, 0, rows, {(3,): rows, (4,): cols, (5,): 1, (6,): 256})
                 for rows, cols in ((4096, 12288), (12288, 4096), (2048, 4096), (4096, 2048)))
    for entry, rows, cols, row0, rows_local, constants in cases:
        yield f"P2:bytes:{entry}:{rows}x{cols}", rows, cols, row0, rows_local, {
            "entry": entry, "target": target, "constants": canonical(constants)}


def matches_setup_key(identity, key):
    return all(identity.get(field) == value for field, value in key.items())


def setup_keys(path):
    manifest = json.loads(path.read_text())
    if manifest["controls_passed"] is not True:
        raise Invalid("CPU compile receipt has failed controls")
    keys = {}
    matched = set()
    for case, *_, key in setup_cases():
        matches = [row for row in manifest["cubins"] if matches_setup_key(row, key)]
        if len(matches) != 1:
            raise Invalid(f"CPU compile receipt has no unique cubin for {case}")
        row, = matches
        digest = row["cubin_sha256"]
        if (not re.fullmatch(r"[0-9a-f]{64}", digest) or row["file"] != digest + ".cubin"
                or row["expected_cubin_sha256"] not in (None, digest)
                or sha256(path.with_name(row["file"])) != digest):
            raise Invalid(f"CPU compile cubin bytes differ for {case}")
        keys[case] = {**key, "cubin_sha256": digest}
        matched.add(row["file"])
    if len(matched) != len(manifest["cubins"]):
        raise Invalid("CPU compile receipt has the wrong number of cubins")
    return keys


def setup_reference(plane, rows, cols, *, repack, row0=0, rows_local=None):
    import numpy as np

    if repack:
        # Put tiles first and reverse the bytes in each 32-bit word.
        stop = rows if rows_local is None else row0 + rows_local
        cut = plane.reshape(cols, rows // 2)[:, row0 // 2:stop // 2]
        return cut.reshape(cols, -1, 64, 4).transpose(1, 0, 2, 3)[..., ::-1].ravel()
    packed = plane.reshape(cols, rows // 2).T
    result = np.empty((rows, cols), dtype=np.uint8)
    result[::2], result[1::2] = packed >> 4, packed & 15
    return result


def check_setup_bytes(actual, expected, changed, expected_changed, byte_index):
    import numpy as np

    if any(out.shape != expected.shape or out.dtype != expected.dtype
           for out in (actual, changed, expected_changed)):
        raise Invalid("Setup output shape or dtype differs from the CPU reference")
    def delta(output):
        diff = output != expected
        count = int(np.count_nonzero(diff))
        return {"count": count, "indices": [int(diff.argmax())] if count == 1 else []}

    clean = int(np.count_nonzero(actual != expected))
    predicted, observed = delta(expected_changed), delta(changed)
    detected = predicted["count"] == 1 and observed == predicted and np.array_equal(changed, expected_changed)
    return {"bytes": expected.nbytes, "mismatches": clean,
            "fault": {"byte_index": byte_index, "input_bit": 0, "predicted": predicted,
                      "observed": observed, "detected": detected},
            "result": "PASS" if clean == 0 and detected else "FAIL"}


def record_setup_p2(p2, case, key, records, check):
    p2.setdefault("byte_checks", {})[case] = check
    observed = [r for r in records if r["case"] == case]
    if len(observed) != 1 or not matches_setup_key(observed[0]["identity"], key):
        raise Invalid(f"Loaded Triton cubin identity differs for {case}")
    if check["result"] == "PASS":
        p2["triton_specializations"].append(observed[0])


def run_setup_p2(p2, path, recorder, records, save):
    import numpy as np
    import torch
    from tessera import kernel_wire

    p2["setup_cubins_sha256"] = sha256(path)
    keys = setup_keys(path)
    device = torch.device("cuda")
    rng = np.random.default_rng(0)
    for case, rows, cols, row0, rows_local, key in setup_cases():
        repack = key["entry"] == "_window_repack_kernel"
        plane = rng.integers(0, 256, size=rows * cols // 2, dtype=np.uint8)
        plane[:256] = np.arange(256, dtype=np.uint8)
        if repack:
            columns = torch.arange(cols, dtype=torch.int32, device=device)
            starts = columns.to(torch.int64) * (rows * 4)
            def launch():
                return kernel_wire.window_repack_stream_cuda(
                    torch.from_numpy(plane).to(device), col_starts=starts, perm=columns, row0=row0,
                    rows_local=rows_local, rate=4, group_col0=0, group_byte0=0, n_cols=cols,
                    n_tiles=rows_local // 512, chunk_bytes=256, tile_bytes=cols * 256, device=device)
        else:
            def launch():
                return kernel_wire.unpack_body_cuda(plane.tobytes(), (4,) * cols, rows, device, span=1)
        ref_args = {"repack": repack, "row0": row0, "rows_local": rows_local}
        expected = setup_reference(plane, rows, cols, **ref_args)
        actual = recorder.call(case, "tessera", launch).cpu().numpy()
        byte_index = rows // 2 + row0 // 2 + 1
        plane[byte_index] ^= 1
        expected_changed = setup_reference(plane, rows, cols, **ref_args)
        changed = recorder.call(case, "tessera", launch).cpu().numpy()
        record_setup_p2(p2, case, keys[case], records,
                        check_setup_bytes(actual, expected, changed, expected_changed, byte_index))
        save()


def setup_p2_passed(p2):
    checks = p2.get("byte_checks", {})
    records = [r["case"] for r in p2.get("triton_specializations", []) if r["case"].startswith("P2:bytes:")]
    return (set(checks) == {case for case, *_ in setup_cases()} == set(records)
            and len(records) == len(checks)
            and all(c["result"] == "PASS" and c["mismatches"] == 0
                    and c["fault"]["detected"] is True and c["fault"].get("predicted", {}).get("count") == 1
                    and len(c["fault"]["predicted"]["indices"]) == 1
                    and c["fault"]["predicted"] == c["fault"]["observed"] for c in checks.values()))


def confirm_setup_p2(p2, path):
    if p2.get("setup_cubins_sha256") != sha256(path) or not setup_p2_passed(p2):
        raise Invalid("H1 setup byte checks are incomplete or use another CPU compile receipt")
    keys = setup_keys(path)
    if any(not matches_setup_key(record["identity"], keys[record["case"]])
           for record in p2["triton_specializations"] if record["case"] in keys):
        raise Invalid("H1 setup identity differs from the CPU compile receipt")


def dense_preconditions(key, recorder):
    import torch
    from experiments import dense_fused_oracle as dense
    from experiments.numerics_rows.synthetic import encode_module
    from tessera import routed_fused as rf

    family = "value" if key == "value" else "e4m3"
    blob, scheme, _ = encode_module(family, [("gate_proj", 512), ("up_proj", 512)], 256, q256=1024)
    class SyntheticStore:
        schemes = {"synthetic.gate_up_proj": scheme}
        def wire(self, name):
            return torch.frombuffer(bytearray(blob), dtype=torch.uint8)
    results = []
    for rank in ((0, 1) if key in ("e4m3", "e4m3mma") else (0,)):
        layer, method, info = dense.build_served(SyntheticStore(), "synthetic.gate_up_proj", mode="resident",
                                                tp_rank=rank, tp_size=2 if key != "value" else 1,
                                                fused=True, instruction="f16" if key == "e4m3" else "e4m3")
        from tessera.fused import parse_fused
        from tessera.unit_artifact import parse_unit_artifact
        tensors = dict(layer.tessera_native.named_tensors())
        equal = True
        for index, member in enumerate(parse_fused(blob)):
            unit = parse_unit_artifact(member.blob, device="cuda").unit
            expected = unit.scale_rows.float() * float(unit.scale_global)
            if key != "value":
                expected = expected[rank * 256:(rank + 1) * 256]
            equal &= torch.equal(tensors[f"roles.{index}.scale"].reshape(-1).view(torch.int32), expected.view(torch.int32))
        x = torch.zeros(2, 256, device="cuda", dtype=torch.bfloat16)
        x[:, 0] = 1
        recorder.call(f"P5:dense:{key}:{rank}", "apply", lambda: method.apply(layer, x))
        results.append({"rank": rank, "build": info, "P4_equal": equal})
    return {"P4": all(r["P4_equal"] for r in results), "P5": len(results) == (1 if key == "value" else 2), "ranks": results,
            "routed": routed_preconditions(key, recorder)}


def routed_preconditions(key, recorder):
    import torch
    from experiments import routed_pair_oracle as routed
    from experiments.numerics_rows.synthetic import encode_module
    from experiments.numerics_rows.routed import build_served

    routed.HIDDEN, routed.INTER, routed.TOP_K = 256, 512, 2
    blobs = {p: [encode_module("value" if key == "value" else "e4m3", [(p, rows)], cols, q256=1024, seed=e)[0] for e in range(2)]
             for p, rows, cols in (("gate_proj", 512, 256), ("up_proj", 512, 256), ("down_proj", 256, 512))}
    family = {"family": "TESSERA_BF16" if key == "value" else "TESSERA_FP8"}
    scheme, _ = routed.scheme_for(family, blobs, 2)
    builds = []
    for rank in ((0,) if key == "value" else (0, 1)):
        cfg = routed.moe_config(2, 10.0)
        parallel = cfg.moe_parallel_config
        parallel.tp_rank, parallel.tp_size = rank, 1 if key == "value" else 2
        layer, method, info = build_served(family, scheme, blobs, {}, 2, cfg, 10.0, "synthetic.experts")
        native = method._native
        from tessera.fused import parse_fused
        from tessera.unit_artifact import parse_unit_artifact
        scale_equal = True
        for projection, field in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
            expected = []
            for blob in blobs[projection]:
                unit = parse_unit_artifact(parse_fused(blob)[0].blob, device="cuda").unit
                value = unit.scale_rows.float() * float(unit.scale_global)
                expected.append(value if projection == "down_proj" or key == "value" else value[rank * 256:(rank + 1) * 256])
            scale_equal &= torch.equal(getattr(native, field).scale_all.view(torch.int32), torch.stack(expected).view(torch.int32))
        builds.append({"rank": rank, "build": info, "P4_equal": scale_equal, "has_init": bool(native.gate.has_init.any()),
                       "start_state_sha256": __import__("hashlib").sha256(native.gate.init_all.cpu().numpy().tobytes()).hexdigest()})
    return {"P4": all(b["P4_equal"] for b in builds), "P5": len(builds) == (1 if key == "value" else 2), "ranks": builds}


def wrong_scale_index(key, recorder):
    import torch
    from tessera import routed_fused as rf, window_gemm as wg
    from window_gemm_grouped_support import Expert

    expert = Expert(256, 256, (4,) * 256, 51, family="e4m3" if key != "value" else "value")
    if key == "value":
        expert.unit.table.fill_(1)
    else:
        expert.unit.native.fill_(0x38)
    expert.unit.scale.copy_(torch.arange(1, 257, device="cuda", dtype=torch.float32))
    with __import__("experiments.dense_fused_oracle", fromlist=["e4m3_instruction"]).e4m3_instruction("f16" if key == "e4m3" else "e4m3"):
        role = rf.prepare_dense_role(wg.prepare_window_gemm(expert.unit))
    x = torch.zeros(1, 256, dtype=torch.bfloat16 if key == "value" else torch.float8_e4m3fn, device="cuda")
    x[:, 0] = 1
    a = None if key == "value" else torch.ones(1, device="cuda")
    got, wrong = (torch.empty(1, 256, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    recorder.call("scale-index:correct", "tessera", lambda: rf.dense_forward(role, x, a, got, counter))
    changed = dataclasses.replace(role, wscale=role.wscale.roll(1, 1))
    recorder.call("scale-index:wrong", "tessera", lambda: rf.dense_forward(changed, x, a, wrong, counter))
    reference = torch.arange(1, 257, device="cuda", dtype=torch.float32).bfloat16().reshape(1, -1)
    return {"mutation": "row-scale-index", "flagged_exactly": compare(got, reference)["equal"]
            and compare(wrong, reference.roll(1, 1))["equal"] and compare(wrong, reference)["mismatches"] == reference.numel()}


def negative_controls(*, device="cuda"):
    import torch

    wanted = torch.tensor([1.0, -2.0, 3.0], dtype=torch.bfloat16, device=device)
    controls = []
    for index in range(wanted.numel()):
        mutation = wanted.clone()
        mutation.view(torch.int16)[index] ^= 1
        result = compare(wanted, mutation)
        controls.append({"mutation": f"reference-bit:{index}", "flagged_exactly": result["mismatches"] == 1
                         and result["first_mismatch"]["index"] == [index]})
    return controls


def subnormal_case(key, recorder):
    if key != "value":
        return {"result": "NOT_APPLICABLE", "reason": "not representable; native E4M3 codes start at 2^-9 and E2M1 codes at 1/2; their supported block scales cannot make a native FP32 operand subnormal"}
    import torch
    from tessera import routed_fused as rf, window_gemm as wg
    from window_gemm_grouped_support import Expert

    expert = Expert(256, 256, (4,) * 256, 7)
    expert.unit.table.fill_(2**-133)
    expert.unit.scale.fill_(1)
    bundle = wg.prepare_window_gemm(expert.unit)
    role = rf.prepare_dense_role(bundle)
    x = torch.zeros(1, 256, dtype=torch.bfloat16, device="cuda")
    x[0, 0] = 2.0**64
    got = torch.empty(1, 256, dtype=torch.bfloat16, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    recorder.call("subnormal-operand", "tessera", lambda: rf.dense_forward(role, x, None, got, counter))
    expected = torch.full_like(got, 2**-69)
    return {"result": "PASS" if torch.equal(got, expected) else "FAIL", "expected": 2**-69,
            "mismatches": int((got != expected).sum())}


def e2m1_preconditions(lib, recorder):
    import torch
    from experiments.t4_code import fused_e2m1_check as check
    from tessera.compact_prep import parse_compact_wire, prepare_window_lut_compact
    from tessera.routed_fused_e2m1 import prepare_dense_role, dense_forward_quantized
    from experiments.numerics_rows.quantizers import quantize

    blob = check.encode(256, 256, 896, 19, "cuda")
    unit = prepare_window_lut_compact(parse_compact_wire(blob, device="cuda"))
    role = prepare_dense_role(unit, torch.tensor([1.0], device="cuda"))
    expected = torch.tensor([unit.global_scale], dtype=torch.float32, device="cuda") / role.gs
    x = check.onehot_x(2, 256, 0, "cuda")
    q, sf = recorder.call("H1:e2m1:quantizer", "quantizer", lambda: quantize("e2m1", x))
    got = recorder.call("H1:e2m1:dense", "tessera", lambda: dense_forward_quantized(role, q, sf))
    # A wrong ratio index must be visible in the exact check.
    wrong = dataclasses.replace(role, ratio=role.ratio * 2)
    mutated = recorder.call("H1:e2m1:wrong-scale", "tessera", lambda: dense_forward_quantized(wrong, q, sf))
    from experiments.numerics_rows.row import dot_reference
    a, weight = check.a_deq(q, sf), check.ref_weight(blob, None, "cuda")
    reference, mask = dot_reference(a, weight, None, role.ratio)
    wrong_reference, _ = dot_reference(a, weight, None, wrong.ratio)
    control = compare(mutated, reference)
    return {"P4": torch.equal(role.ratio.view(torch.int32), expected.view(torch.int32)),
            "wrong_scale_detected": bool(mask.all()) and compare(got, reference)["equal"]
                and compare(mutated, wrong_reference)["equal"] and control["mismatches"] > 0,
            "wrong_scale_mismatches": control["mismatches"], "rung": 896,
            "routed": e2m1_routed_preconditions(recorder)}



def e2m1_routed_preconditions(recorder):
    import torch
    from experiments import routed_pair_oracle as routed
    from experiments.t4_code.fused_e2m1_check import encode, ref_weight
    from tessera.fused import pack_fused, parse_fused
    from tessera.unit_artifact import parse_unit_artifact
    from experiments.numerics_rows.quantizers import quantize

    routed.HIDDEN, routed.INTER, routed.TOP_K = 256, 512, 2
    blobs = {p: [pack_fused([(p, rows, encode(rows, cols, 896, 40 + e, "cuda"))]) for e in range(2)]
             for p, rows, cols in (("gate_proj", 512, 256), ("up_proj", 512, 256), ("down_proj", 256, 512))}
    family = {"family": "TESSERA_NVFP4"}
    scheme, _ = routed.scheme_for(family, blobs, 2)
    scales = {p: [1.0, 2.0] for p in blobs}
    layer, method, info = routed.build_after(family, scheme, blobs, scales, 2, routed.moe_config(2, 10.0), 10.0, "synthetic.experts")
    native = layer.tessera_routed_fused
    equal = True
    for projection, field in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
        gs = native.gs2 if projection == "down_proj" else native.gs13
        expected = torch.tensor([parse_unit_artifact(parse_fused(blob)[0].blob, device="cuda").unit.scale_global for blob in blobs[projection]], dtype=torch.float32, device="cuda") / gs
        equal &= torch.equal(expected.view(torch.int32), getattr(native, "ratio_" + field).view(torch.int32))
    # Index mutation: exchange two distinct expert ratios in the loaded down copy.
    ratios = native.ratio_down.clone()
    ratios[0], ratios[1] = 1.0, 2.0
    correct = dataclasses.replace(native, ratio_down=ratios)
    wrong = dataclasses.replace(native, ratio_down=ratios.flip(0))
    ids = torch.tensor([[0, 1]], device="cuda", dtype=torch.int64)
    weights = torch.full((1, 2), 0.5, device="cuda")
    routes = native._routing(ids, weights)
    x = torch.zeros(2, 512, device="cuda", dtype=torch.bfloat16)
    x[:, 0] = 1
    q, sf = recorder.call("H1:e2m1:routed-quantizer", "quantizer", lambda: quantize("e2m1", x, float(native.gs2)))
    q, sf = q.contiguous(), sf.view(torch.uint8).contiguous()
    got, changed = (torch.empty(2, 256, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    for owner, out, name in ((correct, got, "correct"), (wrong, changed, "wrong-index")):
        recorder.call("H1:e2m1:" + name, "tessera", lambda: owner._launch(2, q, sf, routes, a_row_mode=1, mul_weight=False, limit=float("inf"), out=out, counter=1))
    from experiments.numerics_rows.row import dot_reference
    from experiments.t4_code.fused_e2m1_check import a_deq
    reference, wrong_reference = torch.empty_like(got), torch.empty_like(got)
    all_exact = True
    for row, flat in enumerate(routes.flat_sorted.tolist()):
        expert = int(ids.flatten()[flat])
        weight = ref_weight(parse_fused(blobs["down_proj"][expert])[0].blob, None, "cuda")
        for owner, ref in ((correct, reference), (wrong, wrong_reference)):
            value, mask = dot_reference(a_deq(q, sf)[row:row + 1], weight, None, owner.ratio_down[expert:expert + 1])
            ref[flat] = value[0]
            all_exact &= bool(mask.all())
    expected_mismatches = compare(wrong_reference, reference)["mismatches"]
    control = {"mutation": "expert-ratio-index", "flagged_exactly": all_exact and compare(got, reference)["equal"]
               and compare(changed, wrong_reference)["equal"] and expected_mismatches > 0
               and compare(changed, reference)["mismatches"] == expected_mismatches}
    return {"P4": equal, "P5": True, "build": info, "negative_control": control}

def cuda_control(where, image):
    """Use a separate process because a device assertion poisons its CUDA context."""
    import torch

    registration = register_native_ops(image)
    records = []
    recorder = BoundaryRecorder(torch.cuda.synchronize, records)
    value = torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    def fault():
        torch._assert_async(value == 0, "numerics injected CUDA error")
    if where == "outside":
        fault()
    try:
        recorder.call("injected-launch", "tessera", fault if where == "inside" else lambda: value.add_(1))
    except BoundaryError as error:
        print(json.dumps({"native_ops": registration, "where": where, "attributed_to": error.attributed_to, "boundaries": records}))
        return error.attributed_to == ("injected-launch" if where == "inside" else None)
    return False


def cuda_control_record(where, run):
    child = {}
    for line in run.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("where") == where:
            child = value
    return {"where": where, "exit_status": run.returncode, "stdout": run.stdout, "stderr": run.stderr,
            "attributed_to": child.get("attributed_to"), "boundaries": child.get("boundaries", []),
            "attribution_confirmed": "attributed_to" in child
                and child["attributed_to"] == ("injected-launch" if where == "inside" else None),
            "scope": "harness check of CUDA error attribution, not of the cell kernel",
            "injection": "async device assert in a helper kernel; cell binary unchanged"}


def gpu_preconditions(report):
    libraries = report.get("libraries", {})
    controls = report.get("cuda_controls", [])
    for key, value in libraries.items():
        routed = value.get("routed") or {}
        subnormal = value.get("subnormal_operand", {})
        if key != "value" and subnormal == {"result": "UNATTESTED", "reason": "not representable; this library's FP32 products cannot be subnormal"}:
            subnormal.update(result="NOT_APPLICABLE", original_result="UNATTESTED")
        if key not in ("e4m3", "e4m3mma"):
            value["P5"] = {"result": "NOT_APPLICABLE", "reason": "pass-rule section 0 P5 requires E4M3 TP2 ranks"}
            if routed:
                routed["P5"] = dict(value["P5"])
        if key != "e2m1":
            value["wrong_scale_detected"] = {"result": "NOT_APPLICABLE", "reason": "E2M1 dense-ratio control; this library uses the row-scale control in negative_controls"}
            if routed:
                routed["negative_control"] = {"result": "NOT_APPLICABLE", "reason": "expert-ratio-index control applies to E2M1 only"}
    return bool(
        not report.get("error") and libraries
        and set(libraries) == set(report.get("runtime", {}).get("libraries", []))
        and report.get("P1", {}).get("samples") == 1_000_000 and report["P1"].get("mismatches") == 0
        and (set(libraries) == {"value"} or report.get("P2", {}).get("result") == "PASS")
        and ("setup_cubins_sha256" not in report.get("P2", {}) or setup_p2_passed(report["P2"]))
        and all(not v.get("error") and v.get("P4") is True
                and (v.get("routed") or {}).get("P4") is True
                and (key not in ("e4m3", "e4m3mma") or v.get("P5") is True and v["routed"].get("P5") is True)
                and (key != "e2m1" or v.get("wrong_scale_detected") is True
                     and (v["routed"].get("negative_control") or {}).get("flagged_exactly") is True)
                and v.get("subnormal_operand", {}).get("result") == ("PASS" if key == "value" else "NOT_APPLICABLE")
                and (key == "value" or "not representable" in v["subnormal_operand"].get("reason", ""))
                for key, v in libraries.items())
        and len(controls) == 2 and {c["where"] for c in controls} == {"inside", "outside"}
        and all(c["exit_status"] == 0 and c.get("attribution_confirmed") is True
                and c.get("attributed_to") == ("injected-launch" if c["where"] == "inside" else None) for c in controls)
        and report.get("negative_controls") and all(c["flagged_exactly"] for c in report["negative_controls"]))


def confirm_p3(report, records):
    """Attach a CPU proof only to the binary and probe checked by this H1 run."""
    report["preconditions_confirmed"] = False
    report["gpu_preconditions_confirmed"] = gpu_preconditions(report)
    if not report["gpu_preconditions_confirmed"]:
        raise ValueError("H1 GPU preconditions are not confirmed")
    expected = set(report["runtime"]["libraries"])
    for key in expected:
        if key not in records or key not in report.get("P3", {}):
            raise ValueError(f"P3 receipt is missing {key}")
        for field in ("library_so_sha256", "probe_sha256"):
            digest = report["P3"][key].get(field)
            if not digest or records[key].get(field) != digest:
                raise ValueError(f"P3 {key} {field} differs from the H1 binary")
    report.setdefault("P3_at_gpu_run", report["P3"])
    report["P3"] = {key: records[key] for key in sorted(expected)}
    report["preconditions_confirmed"] = all(
        r.get("matched") is True and r.get("status") == "MATCHED" and r.get("unresolved") == []
        and r.get("differences") == [] and r.get("mode0_instantiations", 0) > 0
        and r.get("quotients", 0) == r.get("quotients_examined") and r.get("quotients", 0) > 0
        for r in report["P3"].values())


def run_checks(args, report):
    def save():
        report["environment_diagnostics"] = args.launch_state.diagnostics()
        write_json(args.out / "h1.json", report)

    report["runtime"] = check_runtime(args.image)
    built = load_build(args.build, report["runtime"])
    check_source(args.source, built["commit"])
    sys.path.insert(0, str(args.source / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
    import torch
    args.launch_state.bind_math(torch)
    from experiments.numerics_rows.dependency_code import CodeAudit
    args.code_audit = CodeAudit(report["runtime"]["image_digest"], args.out / "compiled-code").install()
    report["triton_specializations"] = args.code_audit.records
    report.update(args.launch_state.fields())
    import tessera
    from experiments.dense_fused_oracle import e4m3_instruction, init_vllm_world1
    from experiments.numerics_rows.sass_slice import match_library

    torch.set_num_threads(1)
    report["runtime"]["sms"] = torch.cuda.get_device_properties(0).multi_processor_count
    report["native_ops"] = register_native_ops(args.image)
    write_json(args.out / "native_ops.json", report["native_ops"])
    report["build_sha256"] = sha256(args.build)
    report["P1"] = selftest(device="cuda")
    save()
    report["negative_controls"] = negative_controls()
    save()
    recorder = BoundaryRecorder(torch.cuda.synchronize, report["boundaries"])
    recorder.before_call = args.launch_state.check
    recorder.context = args.code_audit.context
    loaded = install_libraries(built["libraries"])
    report["P3"] = {}
    for record in built["libraries"]:
        report["P3"][record["key"]] = match_library(record, args.out / "binary-identity" / record["key"])
        save()
    with init_vllm_world1():
        report["P2"] = p2_ties(recorder) if set(loaded) != {"value"} else {"result": "not applicable to value-only build"}
        report["P2"]["triton_specializations"] = [r for r in args.code_audit.records if r["case"].startswith("P2:")]
        save()
        if args.setup_cubins:
            run_setup_p2(report["P2"], args.setup_cubins, recorder, args.code_audit.records, save)
        for key, lib in loaded.items():
            entry = report["libraries"].setdefault(key, {})
            try:
                with e4m3_instruction("f16" if key == "e4m3" else "e4m3"):
                    entry.update(e2m1_preconditions(lib, recorder) if key == "e2m1" else dense_preconditions(key, recorder))
                    save()
                    entry["subnormal_operand"] = subnormal_case(key, recorder)
                    save()
                    if key != "e2m1":
                        report["negative_controls"].append(wrong_scale_index(key, recorder))
            except Exception as error:
                entry.update(error=error_text(error), traceback=error_trace(error))
            finally:
                save()
    for where in ("inside", "outside"):
        env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(args.source.resolve() / "src"), str(Path(__file__).resolve().parents[2])))}
        run = subprocess.run([sys.executable, "-m", "experiments.numerics_rows.h1", "--cuda-control", where, "--image", args.image], env=env, capture_output=True, text=True)
        report["cuda_controls"].append(cuda_control_record(where, run))
        save()
    report["gpu_preconditions_confirmed"] = gpu_preconditions(report)
    report["preconditions_confirmed"] = report["gpu_preconditions_confirmed"] and all(p["matched"] for p in report["P3"].values())
    save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--image", choices=("G", "N"))
    p.add_argument("--build", type=Path)
    p.add_argument("--setup-cubins", type=Path)
    p.add_argument("--cuda-control", choices=("inside", "outside"))
    p.add_argument("--gpu-receipt", type=Path)
    p.add_argument("--p3-receipt", type=Path)
    args = p.parse_args()
    if args.cuda_control:
        raise SystemExit(0 if cuda_control(args.cuda_control, args.image) else 3)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.gpu_receipt:
        report = json.loads(args.gpu_receipt.read_text())
        report["preconditions_confirmed"] = False
        try:
            if args.setup_cubins:
                confirm_setup_p2(report.get("P2", {}), args.setup_cubins)
            confirm_p3(report, json.loads(args.p3_receipt.read_text()))
            jobid = args.p3_receipt.parent.parent / "jobid"
            report["p3_receipt"] = {"path": str(args.p3_receipt), "sha256": sha256(args.p3_receipt),
                                    "flux_job_id": jobid.read_text().strip() if jobid.is_file() else None}
            report["gpu_receipt"] = {"path": str(args.gpu_receipt), "sha256": sha256(args.gpu_receipt)}
        except Exception as error:
            report["error"] = {"type": type(error).__name__, "message": error_text(error), "traceback": error_trace(error)}
        finally:
            write_json(args.out / "h1.json", report)
        raise SystemExit(0 if report["preconditions_confirmed"] else 3)
    report = {"boundaries": [], "libraries": {}, "cuda_controls": [], "negative_controls": [],
              "preconditions_confirmed": False}
    write_json(args.out / "h1.json", report)
    try:
        from experiments.numerics_rows.launch_state import LEVERS, LaunchState
        args.launch_state = LaunchState(LEVERS, fresh=True)
        run_checks(args, report)
    except Exception as error:
        report["preconditions_confirmed"] = False
        report["error"] = {"type": type(error).__name__, "message": error_text(error), "traceback": error_trace(error)}
    finally:
        if getattr(args, "code_audit", None) is not None:
            args.code_audit.close()
        write_json(args.out / "h1.json", report)
    raise SystemExit(0 if report["preconditions_confirmed"] else 3)


if __name__ == "__main__":
    main()
