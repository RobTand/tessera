#!/usr/bin/env python3
"""Oracle and before/after profiles for the fused window kernel's DENSE
identity (contract v43, the dense follow-up to tessera#640).

WHAT IT CHECKS.  Real production wires -- the q256 1024 dense modules of the
eight-layer GLM-5.3-Flash stub B (``STUB``), the same checkpoint the v43
census cells were re-earned on -- are loaded through the SAME builder and
callbacks a serve uses (``lane.build_tessera_method`` -> ``create_weights`` ->
``process_weights_after_loading`` -> ``apply``) at TP 1 and both ranks of TP 2,
in both residencies, and the served output is held to an independent
reference.  Two builds of every module over the same bytes: the fused lane
(the default) and the Triton window GEMM under ``TESSERA_DENSE_FUSED=0``, so
the lane the cells already named is measured beside the one v43 adds.  The
launch pair is not asserted from the family: it is read off the route's own
``emit_route`` record after ``apply`` (``telemetry.read_route``).

THE REFERENCE.  The materialising reader (``scheme.parse_tessera_blob_for_
scheme``), cut to the rank by the same ``shard_parsed_roles`` the retained
reference preparation uses, then ``fp8_route.prepare_tessera_fp8_module``
(E4M3 bytes and the fp32 row scale) or ``bf16_route.prepare_tessera_bf16_
module`` (table values and the row scale, folded once to bf16 exactly as
``materialize_bf16_folded`` and the served kernel fold it).  The activation is
the runtime's own per-token dynamic E4M3 quantiser (``native_ops.native_fp8_
quant``) for the E4M3 family and the bf16 input for the value family; every
product is summed in fp64 on exactly representable operands and rounded once
to bf16, so the reference is exact to ~1e-15 before that rounding.  Because the
quantiser runs OUTSIDE the kernel and is fed the same input, the quantised
operands are bit-identical on both sides and no rounding-boundary flip can
occur: the bound is the GEMM's alone.

THE BOUND (dtype-derived, per output element; nothing here is fitted).  The
kernel accumulates exact fp32 products (e4m3 x e4m3 and bf16 x bf16 both are)
over K columns, in the split-K regime as S fp32 partials summed in a fixed
order (K + S accumulation steps), then applies at most two fp32 epilogue
multiplies (``(acc * a_scale) * w_scale`` for E4M3; none for the folded value
family) and rounds once to bf16:
  |n - r| <= gamma(K + S, u_acc) Sigma + gamma(2, u32)(|r| + gamma(K + S, u_acc) Sigma)
             + u16 (|r| + that)
with Sigma = sum_k |a_k w_k| |scales| in fp64, u_acc = 2^-23 (one fp32 ulp per
accumulation step, covering a truncating tensor-core adder), u32 = 2^-24,
u16 = 2^-8 (``routed_pair_oracle.gemm_bound``, ``epilogue_mults=2``).  The
Triton lane is held to the same bound.  Fused-vs-Triton is reported
descriptively in bf16 ulps of the row max (two accumulation orders of one
fp32 sum), never as the pass criterion.

PROFILES (``--mode profile``).  Each module at TP 1, resident, eager: the
fused lane and the Triton lane over the same resident bundles at every M in
``--m``.  Each leg records wall time per forward (CUDA events, after warmup), a
``torch.profiler`` kernel table sorted by self device time, and a steady
unprofiled replay window whose UTC bounds are written out so the Netdata GPU
power series can be read for exactly that window; an in-process NVML power
sampler records mean and peak W beside it, and forwards per joule is derived
from the two.  At M = 1 the wire bytes per forward over the time is the
achieved read bandwidth against the 239.4 GB/s the box measures.

Run inside the serving image through ``experiments/dense_fused_oracle.sh``.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import socket
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import routed_pair_oracle as rpo  # noqa: E402  (bound arithmetic, profiler and power helpers)

STUB = "/mnt/shared/tessera-runs/moe/u1-stubs-20260926/stub-B"
#: The q256 1024 dense modules of stub B (rate 4 in every column, rows a
#: multiple of 128): the ones the fused identity serves.  Everything else in
#: the stub is q256 832/880/960/1088 and keeps the Triton lane.
MODULES = (
    "model.language_model.layers.5.mlp.shared_experts.down_proj",      # TESSERA_FP8, row-parallel
    "model.language_model.layers.5.mlp.shared_experts.gate_up_proj",   # TESSERA_BF16, column-parallel
    "model.language_model.layers.7.mlp.shared_experts.gate_up_proj",   # TESSERA_FP8, column-parallel
)
ENVELOPE_W = 140.0
M1_READ_GBPS = 239.4      # the box's measured M = 1 read ceiling (GB10, LPDDR5x)
ISSUE = "RobTand/tessera#640 (dense follow-up, contract v43)"

log = rpo.log


class Store:
    """safetensors reader over one export (index + lazily opened shards)."""

    def __init__(self, root):
        from safetensors import safe_open

        self.root = root
        self._safe_open = safe_open
        self.index = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self.config = json.load(open(os.path.join(root, "config.json")))
        self._open = {}
        self.schemes = {}
        for group in self.config["quantization_config"]["config_groups"].values():
            for target in group["targets"]:
                self.schemes[target] = group["scheme"]

    def get(self, name):
        shard = self.index[name]
        if shard not in self._open:
            self._open[shard] = self._safe_open(os.path.join(self.root, shard), "pt", device="cpu")
        return self._open[shard].get_tensor(name)

    def wire(self, module):
        return self.get(module + ".wire_bytes")


def init_vllm_world1():
    """vLLM's LinearMethodBase reads the TP group at construction.  One process,
    one rank: a world-1 gloo group; the TP2 cut is the LAYER's tp_rank/tp_size,
    which is what the Tessera shard planner reads (tessera#303)."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment

    stack = contextlib.ExitStack()
    tmp = tempfile.mkdtemp(prefix="dense-oracle-tp1-")
    stack.enter_context(set_current_vllm_config(VllmConfig()))
    init_distributed_environment(world_size=1, rank=0,
                                 distributed_init_method="file://" + os.path.join(tmp, "rdv"),
                                 local_rank=0, backend="gloo")
    ensure_model_parallel_initialized(1, 1)
    return stack


def module_kind(module):
    return "row" if module.endswith("down_proj") else "col"


@contextlib.contextmanager
def dense_lane(fused: bool):
    """The lane decision reads ``TESSERA_DENSE_FUSED`` at weight load."""
    from tessera.routed_fused import ENV_TOGGLE_DENSE

    saved = os.environ.get(ENV_TOGGLE_DENSE)
    if fused:
        os.environ.pop(ENV_TOGGLE_DENSE, None)
    else:
        os.environ[ENV_TOGGLE_DENSE] = "0"
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(ENV_TOGGLE_DENSE, None)
        else:
            os.environ[ENV_TOGGLE_DENSE] = saved


def build_served(store, module, *, mode, tp_rank, tp_size, fused):
    """The served path, exactly as a serve builds it, on one lane."""
    from tessera.serving.lane import build_tessera_method

    scheme = store.schemes[module]
    method = build_tessera_method(scheme, module, mode)
    layer = torch.nn.Module()
    layer.tp_rank, layer.tp_size = tp_rank, tp_size
    layer.prefix = module
    rows, cols = int(scheme["rows"]), int(scheme["columns"])
    role_rows = [int(r) for _n, r in scheme["roles"]]
    if module_kind(module) == "col":
        ops, inpp = [r // tp_size for r in role_rows], cols
    else:
        ops, inpp = role_rows, cols // tp_size
    method.create_weights(layer, input_size_per_partition=inpp, output_partition_sizes=ops,
                          input_size=cols, output_size=rows, params_dtype=torch.bfloat16,
                          weight_loader=None)
    layer.wire_bytes.data = store.wire(module).clone()
    layer.to("cuda")
    t0 = time.time()
    with dense_lane(fused), torch.no_grad():
        method.process_weights_after_loading(layer)
    torch.cuda.synchronize()
    resident = list(method.resident_tensors(layer))
    info = {"module": module, "family": scheme["family"], "q256": int(scheme["q256"]),
            "rows": rows, "columns": cols, "roles": [[n, int(r)] for n, r in scheme["roles"]],
            "wire_bytes": int(scheme["wire_bytes"]), "mode": mode, "tp": [tp_rank, tp_size],
            "local_rows": int(layer.tessera_rows), "local_columns": int(layer.tessera_columns),
            "lane": layer.tessera_lane, "lane_reason": layer.tessera_lane_reason,
            "launch_pair": [layer.tessera_symbol, layer.tessera_decoder],
            "layout": [dict(rates_hist=_hist(f.rates), rows_p=int(f.rows_p), cols=int(f.cols),
                            row_offset=int(f.row_offset), has_history=bool(f.has_history))
                       for f in layer.tessera_native.layout_facts()],
            "resident_bytes": sum(t.numel() * t.element_size() for _, t in resident),
            "resident_fused_tables": sum(1 for n, _ in resident if n.endswith("fused_table16")),
            "prepare_s": time.time() - t0}
    return layer, method, info


def _hist(rates):
    out = {}
    for r in rates:
        out[str(int(r))] = out.get(str(int(r)), 0) + 1
    return out


def reference_weight(store, module, plan):
    """This rank's exact weight as fp64 ``[local_rows, local_cols]`` off the
    materialising reader, and the cut's facts."""
    from tessera.serving import bf16_route, fp8_route
    from tessera.serving.scheme import parse_tessera_blob_for_scheme
    from tessera.serving.sharding import shard_parsed_roles

    scheme = store.schemes[module]
    blob = store.wire(module).contiguous().numpy().tobytes()
    parsed = parse_tessera_blob_for_scheme(blob, scheme, module, device="cuda")
    roles = shard_parsed_roles(parsed, plan)
    if scheme["family"] == "TESSERA_FP8":
        prepared = fp8_route.prepare_tessera_fp8_module(roles, device="cuda")
        codes = prepared.decode().view(torch.float8_e4m3fn).double()
        scale = prepared.row_scale().double()
        return codes * scale[:, None], {"kind": "e4m3_bytes_times_row_scale",
                                        "row_scale_max": float(scale.max()),
                                        "epilogue_mults": 2}
    prepared = bf16_route.prepare_tessera_bf16_module(roles, device="cuda")
    values = prepared.decode()
    scale = prepared.row_scale()
    folded = (values.float() * scale[:, None]).to(torch.bfloat16)
    return folded.double(), {"kind": "bf16_folded_once", "row_scale_max": float(scale.max()),
                             "epilogue_mults": 0}


def make_x(m, cols, seed, sigma):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(m, cols, device="cuda", generator=g) * sigma).to(torch.bfloat16)


def reference_and_bound(family, x, w64, epilogue_mults, k_steps):
    """``(ref bf16, r fp64, bound fp64)`` for the served function of ``x``."""
    from tessera.serving.native_ops import native_fp8_quant, require_native_fp8_quant

    if family == "TESSERA_FP8":
        require_native_fp8_quant("dense fused oracle")
        a_q, a_s = native_fp8_quant(x.contiguous())
        a = a_q.double() * a_s.reshape(-1, 1).double()
        a_abs = a.abs()
    else:
        a = x.double()
        a_abs = a.abs()
    r = a @ w64.t()
    sigma = a_abs @ w64.abs().t()
    bound = rpo.gemm_bound(sigma, r, k_steps, epilogue_mults=epilogue_mults)
    bound = bound + rpo.U16 * (r.abs() + bound)
    return r.to(torch.bfloat16), r, bound


def route_record(layer):
    from tessera.serving.telemetry import read_route

    return read_route(layer)


def oracle_case(store, module, m, seed, sigma, mode, tp_rank, tp_size, sms):
    from tessera import routed_fused as rf
    from tessera.serving.native_window import LANE_FUSED, LANE_TRITON

    case = {"module": module, "M": m, "mode": mode, "tp": [tp_rank, tp_size], "legs": {}}
    fused_layer, fused_method, fused_info = build_served(
        store, module, mode=mode, tp_rank=tp_rank, tp_size=tp_size, fused=True)
    triton_layer, triton_method, triton_info = build_served(
        store, module, mode=mode, tp_rank=tp_rank, tp_size=tp_size, fused=False)
    case["fused_build"], case["triton_build"] = fused_info, triton_info
    case["lane_ok"] = fused_info["lane"] == LANE_FUSED and triton_info["lane"] == LANE_TRITON
    plan = fused_layer.tessera_shard_plan
    w64, ref_facts = reference_weight(store, module, plan)
    case["reference"] = ref_facts
    cols = int(fused_layer.tessera_columns)
    x = make_x(m, cols, seed, sigma)
    # Accumulation steps: K columns plus the split-K partial sum (the largest
    # split any role of this module takes at this M; a column-parallel module
    # cuts each role's rows across ranks, a row-parallel one its columns).
    role_rows = [int(r) // (tp_size if module_kind(module) == "col" else 1)
                 for _n, r in fused_info["roles"]]
    split = max(rf.dense_k_split(m, rows, cols, sms) for rows in role_rows)
    case["k_split"] = int(split)
    ref, r, bound = reference_and_bound(fused_info["family"], x, w64, ref_facts["epilogue_mults"],
                                        cols + int(split))
    outputs = {}
    for leg, layer, method in (("fused", fused_layer, fused_method), ("triton", triton_layer, triton_method)):
        with torch.no_grad():
            y = method.apply(layer, x)
            y2 = method.apply(layer, x)
        torch.cuda.synchronize()
        rec = route_record(layer)
        outputs[leg] = y
        diff = (y.double() - r).abs()
        case["legs"][leg] = {
            "route_record": rec,
            "pair_is_the_layers": rec is not None and (rec["symbol"], rec["decoder"]) == tuple(
                fused_info["launch_pair"] if leg == "fused" else triton_info["launch_pair"]),
            "policy": rec["policy"] if rec else None,
            "shape": rec["shape"] if rec else None,
            "vs_reference": rpo.summarize(diff, bound, r),
            "vs_reference_bf16_ulps": rpo.bf16_ulp_stats(y, ref),
            "deterministic": bool(torch.equal(y, y2)),
        }
    case["fused_vs_triton_bf16_ulps"] = rpo.bf16_ulp_stats(outputs["fused"], outputs["triton"])
    case["fused_vs_triton_max_abs"] = float((outputs["fused"].float() - outputs["triton"].float()).abs().max())
    case["residency"] = {
        "fused_bytes": fused_info["resident_bytes"], "triton_bytes": triton_info["resident_bytes"],
        "delta_bytes": fused_info["resident_bytes"] - triton_info["resident_bytes"],
        "fused_tables": fused_info["resident_fused_tables"],
        "expected_delta_bytes": fused_info["resident_fused_tables"] * (rf.TABLE_ENTRIES * 2 + 4)}
    case["pass"] = bool(
        case["lane_ok"]
        and all(leg["vs_reference"]["pass"] and leg["deterministic"] and leg["pair_is_the_layers"]
                and leg["policy"] == f"{fused_info['family']}:{mode}"
                for leg in case["legs"].values())
        and case["residency"]["delta_bytes"] == case["residency"]["expected_delta_bytes"])
    del fused_layer, triton_layer, w64
    torch.cuda.empty_cache()
    return case, outputs["fused"]


def run_oracle(args):
    from tessera import routed_fused as rf

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = Store(args.stub)
    sms = rf._sm_count(torch.cuda.current_device())
    report = {"schema": "tessera.dense_fused_oracle.v1", "issue": ISSUE,
              "provenance": provenance(args), "stub": args.stub,
              "stub_config_sha256": hashlib.sha256(Path(args.stub, "config.json").read_bytes()).hexdigest(),
              "sms": sms, "bound_constants": {"u16": rpo.U16, "u32": rpo.U32, "u_acc": rpo.U_ACC,
                                              "fp64_slack_relative_to_sigma": rpo.F64_SLACK},
              "modules": {}, "cases": []}
    tp_legs = [(0, 1), (0, 2), (1, 2)]
    ms = [int(v) for v in args.m.split(",")]
    for module in args.modules.split(","):
        entry = {"scheme": store.schemes[module], "wire_sha256": hashlib.sha256(
            store.wire(module).contiguous().numpy().tobytes()).hexdigest()}
        report["modules"][module] = entry
        for tp_rank, tp_size in tp_legs:
            for m in ms:
                name = f"{module.split('.')[-3]}.{module.split('.')[-1]}[tp{tp_size}r{tp_rank}] M={m}"
                try:
                    case, y_res = oracle_case(store, module, m, args.seed + m, args.sigma,
                                              "resident", tp_rank, tp_size, sms)
                    if tp_size == 1 and m in (1, ms[-1]):
                        # The streamed residency prepares the same packed
                        # repack; the identity must be bitwise the same forward.
                        streamed, y_str = oracle_case(store, module, m, args.seed + m, args.sigma,
                                                      "streamed", tp_rank, tp_size, sms)
                        streamed["bitwise_equal_to_resident"] = bool(torch.equal(y_res, y_str))
                        streamed["pass"] = streamed["pass"] and streamed["bitwise_equal_to_resident"]
                        report["cases"].append(streamed)
                        log(name, "streamed", "PASS" if streamed["pass"] else "FAIL",
                            "bitwise==resident", streamed["bitwise_equal_to_resident"])
                    report["cases"].append(case)
                    f, t = case["legs"]["fused"]["vs_reference"], case["legs"]["triton"]["vs_reference"]
                    log(name, "PASS" if case["pass"] else "FAIL",
                        f"S={case['k_split']}",
                        f"fused diff/bound {f['max_diff_over_bound']:.3f} viol {f['violations']}",
                        f"triton diff/bound {t['max_diff_over_bound']:.3f} viol {t['violations']}",
                        f"fused-vs-triton ulps {case['fused_vs_triton_bf16_ulps']['max_diff_in_bf16_ulps_of_row_max']:.2f}")
                except Exception as exc:  # noqa: BLE001
                    report["cases"].append({"module": module, "M": m, "tp": [tp_rank, tp_size],
                                            "error": f"{type(exc).__name__}: {exc}",
                                            "traceback": traceback.format_exc()[-6000:], "pass": False})
                    log(name, "ERROR", exc)
                    traceback.print_exc()
                (out_dir / "oracle.json").write_text(json.dumps(report, indent=1, default=str))
    report["pass"] = bool(report["cases"]) and all(c.get("pass") for c in report["cases"])
    report["summary"] = {
        "cases": len(report["cases"]),
        "violations_fused": sum(c["legs"]["fused"]["vs_reference"]["violations"]
                                for c in report["cases"] if "legs" in c),
        "violations_triton": sum(c["legs"]["triton"]["vs_reference"]["violations"]
                                 for c in report["cases"] if "legs" in c),
        "max_diff_over_bound_fused": max((c["legs"]["fused"]["vs_reference"]["max_diff_over_bound"]
                                          for c in report["cases"] if "legs" in c), default=None),
        "max_diff_over_bound_triton": max((c["legs"]["triton"]["vs_reference"]["max_diff_over_bound"]
                                           for c in report["cases"] if "legs" in c), default=None),
        "max_fused_vs_triton_row_ulps": max((c["fused_vs_triton_bf16_ulps"]["max_diff_in_bf16_ulps_of_row_max"]
                                             for c in report["cases"] if "legs" in c), default=None),
        "all_deterministic": all(leg["deterministic"] for c in report["cases"] if "legs" in c
                                 for leg in c["legs"].values()),
    }
    (out_dir / "oracle.json").write_text(json.dumps(report, indent=1, default=str))
    log("oracle", "PASS" if report["pass"] else "FAIL", json.dumps(report["summary"]))
    return 0 if report["pass"] else 3


def run_profile(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = Store(args.stub)
    sampler = rpo.PowerSampler(hz=10.0)
    sampler.start()
    report = {"schema": "tessera.dense_fused_profiles.v1", "issue": ISSUE,
              "provenance": provenance(args), "stub": args.stub,
              "regime": "eager (no CUDA graph), resident, TP1, one dense module per case",
              "power_sampler": sampler.source, "envelope_w": ENVELOPE_W,
              "m1_read_ceiling_gbps": M1_READ_GBPS, "idle_windows": [], "modules": {}}
    time.sleep(args.idle_s)
    t0 = time.time()
    time.sleep(max(args.idle_s, 5.0))
    report["idle_windows"].append({"utc_start": t0, "utc_end": time.time(),
                                   "in_process_power": sampler.window(t0, time.time())})
    ms = [int(v) for v in args.m.split(",")]
    for module in args.modules.split(","):
        entry = {"legs": {}, "legs_built": []}
        report["modules"][module] = entry
        try:
            layers = {}
            for leg, fused in (("fused", True), ("triton", False)):
                layer, method, info = build_served(store, module, mode="resident", tp_rank=0,
                                                   tp_size=1, fused=fused)
                layers[leg] = (layer, method)
                entry[f"{leg}_build"] = info
                entry["legs_built"].append(leg)
            wire_bytes = entry["fused_build"]["resident_bytes"]
            for m in ms:
                x = make_x(m, int(layers["fused"][0].tessera_columns), args.seed + m, args.sigma)
                a = layers["fused"][1].apply(layers["fused"][0], x)
                b = layers["triton"][1].apply(layers["triton"][0], x)
                torch.cuda.synchronize()
                entry.setdefault("fused_vs_triton", {})[str(m)] = rpo.bf16_ulp_stats(a, b)
                for leg, (layer, method) in layers.items():
                    name = f"{module.split('.')[-3]}.{module.split('.')[-1]}_{leg}_M{m}"
                    log("profiling", name)
                    try:
                        iters_wall = 200 if m <= 8 else 30
                        iters_prof = 20 if m <= 8 else 5
                        rec = rpo.profile_leg(name, lambda _l=layer, _m=method, _x=x: _m.apply(_l, _x),
                                              args, sampler, iters_wall, iters_prof, out_dir)
                        power = rec["power_window"]["in_process_power"]
                        rec["forwards_per_joule"] = (rec["power_window"]["forwards_per_s"] / power["mean_w"]
                                                     if power.get("mean_w") else None)
                        rec["ms_x_mean_w_mJ_per_forward"] = (rec["ms_per_forward_cuda_events"] * power["mean_w"]
                                                             if power.get("mean_w") else None)
                        rec["resident_bytes"] = wire_bytes
                        rec["achieved_read_gbps_if_wire_bound"] = wire_bytes / (rec["ms_per_forward_cuda_events"] * 1e6)
                        rec["fraction_of_m1_read_ceiling"] = rec["achieved_read_gbps_if_wire_bound"] / M1_READ_GBPS
                        entry["legs"][name] = rec
                        log(name, round(rec["ms_per_forward_cuda_events"], 4), "ms", power)
                    except Exception as exc:  # noqa: BLE001
                        entry["legs"][name] = {"error": f"{type(exc).__name__}: {exc}",
                                               "traceback": traceback.format_exc()[-4000:]}
                        log(name, "FAILED", exc)
                    (out_dir / "profiles.json").write_text(json.dumps(report, indent=1, default=str))
            del layers
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc()[-6000:]
            traceback.print_exc()
        (out_dir / "profiles.json").write_text(json.dumps(report, indent=1, default=str))
    t0 = time.time()
    time.sleep(max(args.idle_s, 5.0))
    report["idle_windows"].append({"utc_start": t0, "utc_end": time.time(),
                                   "in_process_power": sampler.window(t0, time.time())})
    sampler.stop_flag = True
    report["pass"] = bool(report["modules"])
    for module, entry in report["modules"].items():
        short = f"{module.split('.')[-3]}.{module.split('.')[-1]}"
        expected = {f"{short}_{leg}_M{m}" for leg in ("fused", "triton") for m in ms}
        complete = (not entry.get("error") and set(entry["legs"]) == expected
                    and all(not leg.get("error") for leg in entry["legs"].values()))
        entry["pass"] = complete
        report["pass"] = report["pass"] and complete
    (out_dir / "profiles.json").write_text(json.dumps(report, indent=1, default=str))
    return 0 if report["pass"] else 3


def provenance(args):
    here = Path(__file__).resolve().parents[1]
    return {"host": os.environ.get("HOST_NAME") or socket.gethostname(),
            "device": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "vllm": __import__("vllm").__version__,
            "image": args.image,
            "tessera_head": args.tessera_head,
            "tessera_worktree_state": args.tessera_state,
            "src_tree_sha256": rpo.src_tree_digest(here),
            "pb_action": os.environ.get("PRISMABUILD_ACTION_KEY") or os.environ.get("PB_ACTION_KEY"),
            "argv": sys.argv, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--mode", choices=("oracle", "profile"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stub", default=STUB)
    ap.add_argument("--modules", default=",".join(MODULES))
    ap.add_argument("--m", default="1,3,64,512,2048")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=643)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--power-s", type=float, default=20.0)
    ap.add_argument("--idle-s", type=float, default=5.0)
    ap.add_argument("--image", default=os.environ.get("ORACLE_IMAGE", ""))
    ap.add_argument("--tessera-head", default=os.environ.get("TESSERA_HEAD", ""))
    ap.add_argument("--tessera-state", default=os.environ.get("TESSERA_STATE", ""))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    log("host", os.environ.get("HOST_NAME"), "device", torch.cuda.get_device_name(0))
    with init_vllm_world1():
        if args.mode == "oracle":
            return run_oracle(args)
        return run_profile(args)


if __name__ == "__main__":
    sys.exit(main())
