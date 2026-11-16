"""Check the numerics preconditions on synthetic wires before the freeze."""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import traceback

from experiments.numerics_rows.common import BoundaryError, BoundaryRecorder, check_runtime, check_source, sha256, write_json
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


def dense_preconditions(key, recorder):
    import torch
    from experiments import dense_fused_oracle as dense
    from test_dense_fused_window import _encode_module
    from tessera import routed_fused as rf

    family = "value" if key == "value" else "e4m3"
    blob, scheme, _ = _encode_module(family, [("gate_proj", 512), ("up_proj", 512)], 256, q256=1024)
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
    from test_dense_fused_window import _encode_module

    routed.HIDDEN, routed.INTER, routed.TOP_K = 256, 512, 2
    blobs = {p: [_encode_module("value" if key == "value" else "e4m3", [(p, rows)], cols, q256=1024, seed=e)[0] for e in range(2)]
             for p, rows, cols in (("gate_proj", 512, 256), ("up_proj", 512, 256), ("down_proj", 256, 512))}
    family = {"family": "TESSERA_BF16" if key == "value" else "TESSERA_FP8"}
    scheme, _ = routed.scheme_for(family, blobs, 2)
    builds = []
    for rank in ((0,) if key == "value" else (0, 1)):
        cfg = routed.moe_config(2, 10.0)
        parallel = cfg.moe_parallel_config
        parallel.tp_rank, parallel.tp_size = rank, 1 if key == "value" else 2
        layer, method, info = routed.build_after(family, scheme, blobs, {}, 2, cfg, 10.0, "synthetic.experts")
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
    return {"mutation": "row-scale-index", "flagged_exactly": bool((got != wrong).all())}


def negative_controls():
    import torch

    wanted = torch.tensor([1.0, -2.0, 3.0], dtype=torch.bfloat16, device="cuda")
    controls = []
    for index in range(wanted.numel()):
        mutation = wanted.clone()
        mutation.view(torch.int16)[index] ^= 1
        result = compare(wanted, mutation)
        controls.append({"mutation": f"reference-bit:{index}", "flagged_exactly": result["mismatches"] == 1
                         and result["first_mismatch"]["index"] == [index]})
    return controls


def subnormal_case(key, recorder):
    import torch
    from tessera import routed_fused as rf, window_gemm as wg
    from window_gemm_grouped_support import Expert

    if key != "value":
        return {"result": "UNATTESTED", "reason": "BF16 weight 2^-133 is not representable by this library's weight alphabet"}
    expert = Expert(256, 256, (4,) * 256, 7)
    expert.unit.table.fill_(2**-133)
    expert.unit.scale.fill_(1)
    bundle = wg.prepare_window_gemm(expert.unit)
    role = rf.prepare_dense_role(bundle)
    x = torch.zeros(1, 256, dtype=torch.bfloat16, device="cuda")
    x[0, 0] = 2**64
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
    return {"P4": torch.equal(role.ratio.view(torch.int32), expected.view(torch.int32)),
            "wrong_scale_detected": bool((mutated != got).any()), "rung": 896,
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
    got, changed = (torch.empty(2, 256, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    for owner, out, name in ((correct, got, "correct"), (wrong, changed, "wrong-index")):
        recorder.call("H1:e2m1:" + name, "tessera", lambda: owner._launch(2, q, sf, routes, a_row_mode=1, mul_weight=False, limit=float("inf"), out=out, counter=1))
    control = {"mutation": "expert-ratio-index", "flagged_exactly": bool((got != changed).any())}
    return {"P4": equal, "P5": True, "build": info, "negative_control": control}

def cuda_control(where):
    """Use a separate process because a device assertion poisons its CUDA context."""
    import torch

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
        print(json.dumps({"where": where, "attributed_to": error.attributed_to, "boundaries": records}))
        return error.attributed_to == ("injected-launch" if where == "inside" else None)
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--image", choices=("G", "N"))
    p.add_argument("--build", type=Path)
    p.add_argument("--cuda-control", choices=("inside", "outside"))
    args = p.parse_args()
    if args.cuda_control:
        raise SystemExit(0 if cuda_control(args.cuda_control) else 3)
    check_source(args.source)
    check_runtime(args.image)
    sys.path.insert(0, str(args.source / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
    import tessera
    import torch
    from experiments.dense_fused_oracle import e4m3_instruction, init_vllm_world1
    from experiments.numerics_rows.sass_slice import match_library

    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"build_sha256": sha256(args.build), "P1": selftest(), "boundaries": [], "libraries": {}, "cuda_controls": [], "negative_controls": negative_controls()}
    recorder = BoundaryRecorder(torch.cuda.synchronize, report["boundaries"])
    built = json.loads(args.build.read_text())
    loaded = install_libraries(built["libraries"])
    report["P3"] = {record["key"]: match_library(record) for record in built["libraries"]}
    with init_vllm_world1():
        report["P2"] = p2_ties(recorder) if args.image == "G" else {"result": "not applicable to value-only image"}
        for key, lib in loaded.items():
            try:
                with e4m3_instruction("f16" if key == "e4m3" else "e4m3"):
                    report["libraries"][key] = e2m1_preconditions(lib, recorder) if key == "e2m1" else dense_preconditions(key, recorder)
                    report["libraries"][key]["subnormal_operand"] = subnormal_case(key, recorder)
                    if key != "e2m1":
                        report["negative_controls"].append(wrong_scale_index(key, recorder))
            except Exception as error:
                report["libraries"][key] = {"error": str(error), "traceback": traceback.format_exc()}
            write_json(args.out / "h1.json", report)
    for where in ("inside", "outside"):
        run = subprocess.run([sys.executable, "-m", "experiments.numerics_rows.h1", "--cuda-control", where], capture_output=True, text=True)
        report["cuda_controls"].append({"where": where, "exit_status": run.returncode, "stdout": run.stdout, "stderr": run.stderr})
    report["preconditions_confirmed"] = (
        not report["P1"]["mismatches"] and (args.image == "N" or report["P2"].get("result") == "PASS")
        and all(p["matched"] for p in report["P3"].values())
        and all(v.get("P4") and v.get("P5", True)
                and (v.get("routed") is None or v["routed"]["P4"] and v["routed"]["P5"])
                and v.get("wrong_scale_detected", True)
                and (v.get("routed", {}).get("negative_control", {}).get("flagged_exactly", True)) for v in report["libraries"].values())
        and all(c["exit_status"] == 0 for c in report["cuda_controls"])
        and all(v.get("subnormal_operand", {}).get("result") != "FAIL" for v in report["libraries"].values())
        and all(c["flagged_exactly"] for c in report["negative_controls"]))
    write_json(args.out / "h1.json", report)
    raise SystemExit(0 if report["preconditions_confirmed"] else 3)


if __name__ == "__main__":
    main()
