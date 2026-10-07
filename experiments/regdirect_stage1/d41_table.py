"""The D41 v2 table of the register-direct kernel (dec-1007-074543-94b8): a speed scenario.

Reads the stage-1 class runs (``bench_stage1.py``: one ``stage1.json`` per run), the x86
ptxas resources (``compile.py``), an Nsight Compute LaunchStats pass on the same kernel
source (shared memory per launch) and the published E4M3 table whose CPU quality rows this
table cites by lineage.  It writes one ``fleet.rung_allowability.v2`` table with
``serving_qualified`` false: ``admit_rung`` waits on it, and PACT reads it only as a
labelled speed scenario.  Rows whose cells are not all measured stay pending.

  python d41_table.py --runs A/stage1.json B/stage1.json --compile C/compile.json \
      --ncu N/ncu.csv --quality V0009.json --source-commit SHA --out T.json
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from datetime import datetime, timezone
from fractions import Fraction

from tessera import regdirect_routed as rr
from tessera.export import E4M3_RECIPE
from tessera.rung_allowability import admit_rung, validate_table

FORMAT = "TESSERA_E4M3_K1"
#: CEO, 2026-10-07: until PrismaBuild 1598 is fixed, timings use ordinary GPU-exclusive
#: admission; one class is re-measured under the measurement class once 1598 is live.
ADMISSION = "GPU-exclusive, not quiet-host certified"
SHAPES = {0: ("gate_up", 1024, 4096), 2: ("down", 4096, 1024)}   # GLM-5.3 Flash TP2 rank, per projection
MS = (1, 16, 2048, 4096)
RUNG_MIN, RUNG_MAX = 256 * min(rr.SERVED_RATES), 256 * max(rr.SERVED_RATES)   # the code rates the kernel serves
STEP = 2
SHAPE_STEPS = {"down": 16}                                        # one 64-column unit k-step of K=1024


def recipe():
    r = E4M3_RECIPE
    return {"body": r.body.name.lower(), "span": r.span, "plane": r.scale_plane.name.lower(),
            "window_bits": r.window_bits, "seed": r.window_seed, "sigma": r.window_sigma,
            "channel_sigma": r.channel_sigma}


def template(name):
    """``rd_prefill<0, 8, false, false>`` -> the ptxas key (the launched kernel's own name)."""
    m = re.search(r"(rd_(?:decode|prefill)<[^>]*>)", name)
    if m is None:
        raise ValueError(f"not a register-direct kernel: {name}")
    # ptxas and torch print bool template arguments as true/false; Nsight Compute prints 1/0.
    return m.group(1).replace("true", "1").replace("false", "0")


def ncu_launch(path):
    """``{template: {metric: (value, unit)}}`` from the LaunchStats rows of an NCU CSV."""
    out = {}
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh)]
    head = next(i for i, r in enumerate(rows) if "Kernel Name" in r)
    cols = rows[head]
    k, sec, met, unit, val = (cols.index(c) for c in ("Kernel Name", "Section Name", "Metric Name", "Metric Unit", "Metric Value"))
    for r in rows[head + 1:]:
        if len(r) <= val or "rd_" not in r[k] or r[sec] != "Launch Statistics":
            continue
        out.setdefault(template(r[k]), {})[r[met]] = (r[val].replace(",", ""), r[unit])
    return out


def compile_receipt_matches(comp, source_sha) -> bool:
    """The compile receipt's kernel source against the checkout's: a recorded identity, so a D32 seal
    (``tessera.dev_mode.seal_check``): certified mode refuses, dev mode warns and continues."""
    from tessera.dev_mode import seal_check
    return seal_check("kernel source", comp["source_sha256"], source_sha, where="d41_table compile receipt",
                      refusal=ValueError("the compile receipt is not of the measured kernel source"))


def require_checked_cells(run, path):
    """Every timed register-direct cell needs a passing correctness check at its own (mode, profile, M)."""
    if run.get("stopped"):
        raise ValueError(f"{path}: the run stopped ({run['stopped']})")
    checks = {(int(c["mode"]), str(c["profile"]), int(c["M"])): bool(c["pass"]) for c in run.get("check", [])}
    for key in run["cells"]:
        arm, mode, profile, m = key.split(".")
        if arm != "regdirect":
            continue
        cell = (int(mode[4:]), profile, int(m[1:]))
        if not checks.get(cell, False):
            state = "failed" if cell in checks else "has no correctness check"
            raise ValueError(f"{path}: timed cell {mode}.{profile}.{m} {state}")


def to_bytes(value, unit):
    unit = unit.split("/")[0]                      # NCU states shared memory per block: "byte/block"
    scale = {"byte": 1, "Kbyte": 1000, "KB": 1000, "Kibyte": 1024, "KiB": 1024, "Mbyte": 10**6}[unit]
    return int(round(float(value) * scale))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--compile", required=True)
    ap.add_argument("--ncu", required=True)
    ap.add_argument("--quality", required=True, help="a published E4M3 D41 table; its CPU quality rows are cited")
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    source_sha = rr.kernel_source_sha256()
    comp = json.load(open(a.compile))
    compile_receipt_matches(comp, source_sha)
    regs = {template(k["kernel"]): k for k in comp["kernels"]}
    launch = ncu_launch(a.ncu)
    build_id = f"regdirect-e4m3-sm_121-{source_sha[:16]}"
    rec = recipe()

    cells, runs = {}, []
    for path in a.runs:
        run = json.load(open(path))
        meta = run["meta"]
        require_checked_cells(run, path)
        runs.append({"path": path, "action_key": meta["pb_action"], "host": meta["host"], "snapshot": meta["head"],
                     "checks_passed": len(run.get("check", []))})
        for key, cell in run["cells"].items():
            arm, mode, profile, m = key.split(".")
            if arm != "regdirect":
                continue
            md, M = int(mode[4:]), int(m[1:])
            shape, rows, cols = SHAPES[md]
            name = max(cell["torch_profiler"], key=cell["torch_profiler"].get)
            tpl = template(name)
            reg, lst = regs[tpl], launch.get(tpl)
            if lst is None:
                raise ValueError(f"no NCU launch statistics for {tpl}")
            dyn = to_bytes(*lst["Dynamic Shared Memory Per Block"])
            sta = to_bytes(*lst["Static Shared Memory Per Block"])
            avail = to_bytes(*lst["Shared Memory Configuration Size"])
            base = run["cells"][f"baseline.{mode}.{profile}.{m}"]
            weights = cell["meta"]["touched"] * rows * cols * (2 if md == 0 else 1)
            bits = Fraction(cell["meta"]["wire_bytes"] * 8 * 256, weights)
            ks = cols // (rr.GROUPS[md] * rr.KSTEP)
            low, rem = divmod(cell["q256"], 256)          # a k-step rung mixes low and low + 1
            upper = rem * ks // 256
            widths = [low, low + 1] if rem else [low]
            a_t, b_t = cell["pass_medians_cold_us"]
            cells.setdefault(cell["q256"], {})[(shape, M)] = {
                "cell_id": f"routed:{shape}:M{M}", "kernel_kind": "routed", "shape_id": shape, "M": M,
                "measurement_status": "measured", "kernel_time_us": 0.5 * a_t + 0.5 * b_t, "pass_times_us": [a_t, b_t],
                "kernel_path": name, "measurement_build_id": build_id,
                "geometry": {
                    "bits_per_256_weight_tile": {"numerator": bits.numerator, "denominator": bits.denominator},
                    "alignment": {"kind": "fragment_order", "owner": "tessera.fragment_wire", "lanes": 32,
                                  "history_lanes": rr.HIST_LANES, "unit_words": [32 * r for r in widths], "slot_words": None},
                    "shared_memory": {"kind": "used", "requested_bytes": dyn + sta, "available_bytes": avail, "fits": dyn + sta <= avail,
                                      "dynamic_bytes": dyn, "static_bytes": sta, "source": "Nsight Compute LaunchStats"},
                    "register_pressure": {"compiler": "cuda_ptxas", "REG": reg["registers"], "STACK": reg["stack_bytes"],
                                          "LOCAL": reg["spill_store_bytes"], "SHARED": sta,
                                          "spill_load_bytes": reg["spill_load_bytes"]},
                    "decode_width": {"window_bits": rec["window_bits"], "value_bits": 8, "run_widths": widths,
                                     "word_stages": None, "kstep_columns": rr.KSTEP, "prefetch_depth": rr.DECODE_DEPTH,
                                     "superblock_routes": cell["meta"]["superblock"], "k_parts": cell["meta"]["k_parts"],
                                     "route_tiles": cell["meta"]["route_tiles"], "grid": cell["meta"]["grid"]},
                    "body_kind": "window", "decoder_kind": "register_direct", "decoder_owner": "tessera.regdirect_routed",
                    "execution_scope": "register_direct_fragment",
                    "word_ring": {"kind": "register", "owner": "tessera.regdirect_routed"}, "recipe": rec},
                "evidence": {"action_key": meta["pb_action"], "host": meta["host"], "profile": profile,
                             "upper_k_steps": upper, "unit_k_steps": ks, "wire": "synthetic fragment_synth stack, 288 experts",
                             "routing": "balanced", "rows": rows, "columns": cols,
                             "comparison_id": f"{meta['pb_action'][:12]}:{profile}:{m}", "paired_seed_contract": "fixed crc32 seeds",
                             "timing_statistic": meta["statistic"], "timer": "CUDA graph replay, cold L2", "admission": ADMISSION,
                             "todays_kernel_same_run_us": base["cold_us"]}}

    quality_src = {r["rung"]: r for r in json.load(open(a.quality))["rungs"]}
    keys = [(s, M) for s, _, _ in SHAPES.values() for M in MS]
    rows_out = []
    for q in range(RUNG_MIN, RUNG_MAX + 1, STEP):
        need = [(s, M) for s, M in keys if (q - RUNG_MIN) % SHAPE_STEPS.get(s, STEP) == 0]
        have = cells.get(q, {})
        src = quality_src.get(q, {}).get("quality", {})
        measured = all(k in have for k in need) and src.get("measurement_status") == "measured"
        quality = {}
        if measured:
            quality = copy.deepcopy(src)
            quality["scope"] = {"format": FORMAT, "grid": "E4M3", "arity": 1, "rung": q, "recipe": rec,
                                "kernel_kinds": ["routed"], "owner": "tessera.export.encode_linear"}
            quality["lineage"] = {"table": a.quality, "note": "CPU weight-space quality of the Bresenham placement at "
                                  "this rung; k-step placement quality is the separate weight-space screen"}
            quality.setdefault("anomaly_flags", [])
        rows_out.append({"rung": q, "measurement_status": "measured" if measured else "pending",
                         "supported": True if measured else None, "anomaly_flags": list(quality.get("anomaly_flags", [])),
                         "observations": [], "excluded": False, "dominating_rung": None,
                         "measurements": [have[k] for k in need if k in have],
                         "quality": quality, "dominance_evidence": [], "lineage": {}})
    table = {"schema": "fleet.rung_allowability.v2", "table_version": a.version, "table_status": "partial",
             "format": FORMAT, "generated_at": datetime.now(timezone.utc).isoformat(),
             "kernel_build": {"id": build_id, "source_commit": a.source_commit, "library_variant": "regdirect",
                              "architecture": "sm_121",
                              "activation_contract": "float8_e4m3fn x = 0.5 N(0,1); per-row fp32 scale U(0.01, 0.11); BF16 output",
                              "metadata": {"serving_qualified": False, "kernel_source_sha256": source_sha,
                                           "compile_action": comp.get("action"), "runs": runs,
                                           "scenario": "register_direct speed scenario; never served speed", "admission": ADMISSION,
                                           "pending": "re-measure one class under the measurement class after PrismaBuild 1598",
                                           "decision": "dec-1007-074543-94b8"}},
             "scope": {"rung_min": RUNG_MIN, "rung_max": RUNG_MAX, "grid_step_q256": STEP, "grid_steps_q256": SHAPE_STEPS,
                       "grid_owner": "tessera.grammar.K_STEP_COLUMNS: one rate per 32-column k-step, quota per unit "
                                     "(gate/up K=4096: 128 k-steps; down K=1024 per TP2 rank: 16 unit k-steps of 64 columns)",
                       "required_cells": [{"cell_id": f"routed:{s}:M{M}", "kernel_kind": "routed", "shape_id": s, "M": M} for s, M in keys],
                       "shapes": [{"shape_id": s, "kernel_kind": "routed", "rows": r, "columns": c} for s, r, c in SHAPES.values()],
                       "timing_statistic": "mean of forward and reverse pass medians; cold L2"},
             "rungs": rows_out}
    validate_table(table)
    hold = admit_rung(table, format=FORMAT, kernel_build_id=build_id, rung=RUNG_MIN)
    if hold["reason"] != "kernel_not_serving_qualified":
        raise AssertionError(f"the table must not admit a rung before serving qualification: {hold}")
    json.dump(table, open(a.out, "w"), indent=1)
    done = sorted(r["rung"] for r in rows_out if r["measurement_status"] == "measured")
    print(json.dumps({"out": a.out, "build": build_id, "measured_rungs": done,
                      "cells_measured": sum(len(v) for v in cells.values())}))


if __name__ == "__main__":
    main()
