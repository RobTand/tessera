"""Recompute the fixed screen from its retained raw samples. No GPU work runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw_path = args.root / "t16_decode_once.json"
    raw = json.loads(raw_path.read_text())
    if raw["thresholds"] != {"go_prefill_ratio": 1.15, "kill_prefill_ratio": 1.5}:
        raise ValueError("The receipt does not use the fixed thresholds.")
    rows, differences = {}, []
    for name, cell in raw["cells"].items():
        forward = statistics.median(cell["passes"]["F"]["samples_ms"])
        reverse = statistics.median(cell["passes"]["R"]["samples_ms"])
        mean = (forward + reverse) / 2
        spread = abs(forward - reverse) / mean
        for field, value in (("forward_median_ms", forward), ("reverse_median_ms", reverse),
                             ("mean_ms", mean), ("spread_fraction", spread)):
            differences.append(abs(cell[field] - value))
        rows[name] = {"forward_ms": forward, "reverse_ms": reverse, "mean_ms": mean,
                      "spread_fraction": spread, "samples_per_order": len(cell["passes"]["F"]["samples_ms"]),
                      "profile_us": cell["profile"]["kernel_us_per_call"],
                      "launches": cell["profile"]["launches_per_call"],
                      "mean_w": cell["power"]["mean_w"],
                      "calls_per_j_screen": cell["power"]["calls_per_j"]}
    ratios = {}
    for m in (16, 2048, 4096):
        control = rows[f"cuBLAS_same_wire_BF16:M{m}"]["mean_ms"]
        ratios[str(m)] = {}
        for arm in ("T16_R2048_window", "T16_decode_once_BF16"):
            ratio = rows[f"{arm}:M{m}"]["mean_ms"] / control
            ratios[str(m)][arm] = ratio
            differences.append(abs(raw["screen"]["ratios"][str(m)][arm] - ratio))
    ratio = ratios["2048"]["T16_decode_once_BF16"]
    verdict = "GO" if ratio <= 1.15 else "KILL" if ratio > 1.5 else "INCONCLUSIVE"
    if verdict != raw["screen"]["threshold_band"] or max(differences) != 0:
        raise ValueError("The published reduction differs from its raw samples.")
    guard = json.loads((args.root / "memory_guard.json").read_text())
    netdata = json.loads((args.root / "netdata.json").read_text())
    power = {host: {"stats": box["nvidia_smi.gpu_power_draw"]["stats"],
                    "update_every_s": box["nvidia_smi.gpu_power_draw"]["update_every_s"],
                    "bucket_s": box["nvidia_smi.gpu_power_draw"]["returned_bucket_s"]}
             for host, box in netdata["phases"]["action"]["boxes"].items()}
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [raw_path, args.root / "memory_guard.json", args.root / "netdata.json",
                           args.root / "memory_receipt.json", *sorted(args.root.glob("*.trace.json"))]}
    result = {"schema": "tessera.t16_decode_once.verification.v1", "verdict": verdict,
              "m2048_ratio": ratio, "maximum_reduction_difference": max(differences),
              "cells": rows, "ratios": ratios, "memory": raw["memory"], "guard": guard,
              "correctness": raw["correctness"], "netdata_power": power, "hashes": hashes,
              "limits": "This command verifies reductions and file digests. It does not rerun measurements or verify worker attestation."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"verdict": verdict, "m2048_ratio": ratio,
                      "maximum_reduction_difference": max(differences), "out": str(args.out)}))


if __name__ == "__main__":
    main()
