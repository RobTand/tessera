"""Matched master-production and PR-production D41 timing at uniform cells."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ("master", "pr")


def command(argv, **kwargs):
    return subprocess.run(argv, check=True, **kwargs)


def source_tree(args, out):
    from tessera.dev_mode import seal_check
    actual = command(["git", "rev-parse", args.master_ref + "^{commit}"], capture_output=True, text=True).stdout.strip()
    seal_check("recorded master source", args.master_head, actual, where="uniform production panel")
    source = out / "source-master"
    source.mkdir()
    archive = out / "master-source.tar"
    with archive.open("wb") as stream:
        command(["git", "archive", "--format=tar", actual, "src", "pyproject.toml"], stdout=stream)
    with tarfile.open(archive) as bundle:
        bundle.extractall(source, filter="data")
    return source, actual, hashlib.sha256(archive.read_bytes()).hexdigest()


def arm(args, out, master_source, heads, name, q, M, timer, order):
    destination = out / f"{order}-R{q}-M{M}-{timer}-{name}"
    source = master_source if name == "master" else ROOT
    extension = out / ("ext-" + name)
    extension.mkdir(exist_ok=True)
    environment = dict(os.environ, BENCH_PY="bench_uniform_production_arm.py",
                       BENCH_SRC=str(source / "src"), BENCH_PROJECT_FILE=str(source / "pyproject.toml"),
                       NATIVE_CONTAINER_SRC="/comparison-source/src", BENCH_EXT_DIR=str(extension))
    argv = ["bash", str(HERE / "bench_t8r.sh"), str(ROOT), str(destination),
            "--arm", name, "--source-head", heads[name], "--q256", str(q), "--M", str(M),
            "--timer", timer, "--order", order, "--seed", str(args.seed),
            "--iters", str(args.iters), "--warmup", str(args.warmup),
            "--admission-label", args.admission_label]
    if args.cpu_preflight:
        argv.append("--cpu-preflight")
    with (out / (destination.name + ".log")).open("w") as stream:
        command(argv, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    return json.loads((destination / "uniform_production_arm.json").read_text())


def compare(records, q, M, timer, label):
    rows = {(r["order"], r["arm"]): r for r in records}
    if set(rows) != {(o, a) for o in ("F", "R") for a in ARMS}:
        raise ValueError("the production comparison lacks a complete F/R arm population")
    reference = rows["F", "master"]
    for value in rows.values():
        if value["status"] != "complete" or value["plane_hashes"] != reference["plane_hashes"] or value["input_hashes"] != reference["input_hashes"]:
            raise ValueError("the production arms did not read identical complete weight and input bytes")
        if [r["generation"] for r in value["rows"]] != [0, 1]:
            raise ValueError("the production arm lacks both routing generations")
        for generation, r in enumerate(value["rows"]):
            if not r["eager_graph_bitwise"] or r["output_sha256"] != reference["rows"][generation]["output_sha256"]:
                raise ValueError("the master and PR production outputs differ in BF16 bits")
    values = {}
    generation_ratios = []
    for name in ARMS:
        medians = {}
        for order in ("F", "R"):
            raw = [s for r in rows[order, name]["rows"] for s in r["wall"]["raw_samples_ms"]]
            medians[order] = statistics.median(raw)
        mean = (medians["F"] + medians["R"]) / 2
        values[name] = {"F_ms": medians["F"], "R_ms": medians["R"], "mean_ms": mean,
                        "FR_spread": abs(medians["F"] - medians["R"]) / mean}
    for generation in (0, 1):
        for order in ("F", "R"):
            base = rows[order, "master"]["rows"][generation]["wall"]["median_ms"]
            candidate = rows[order, "pr"]["rows"][generation]["wall"]["median_ms"]
            generation_ratios.append({"generation": generation, "order": order,
                                      "master_ms": base, "pr_ms": candidate, "ratio": candidate / base})
    ratio = values["pr"]["mean_ms"] / values["master"]["mean_ms"]
    spread = max(values[name]["FR_spread"] for name in ARMS)
    return {"q256": q, "M": M, "timer": timer, "admission": label,
            "master": values["master"], "pr": values["pr"], "ratio": ratio,
            "relative_excess": ratio - 1, "spread_bound": spread,
            "spread_rule": "larger measured relative F/R spread of the two production arms",
            "slower_beyond_spread": ratio - 1 > spread, "bitwise": True,
            "generation_order_ratios": generation_ratios,
            "source_heads": {name: rows["F", name]["source_head"] for name in ARMS},
            "entries": {name: rows["F", name]["entry"] for name in ARMS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--master-ref", required=True)
    parser.add_argument("--master-head", required=True)
    parser.add_argument("--pr-head", required=True)
    parser.add_argument("--cpu-preflight", action="store_true")
    parser.add_argument("--rungs", default="512,768,1024")
    parser.add_argument("--ms", default="1,2048")
    parser.add_argument("--timers", default="eager,graph")
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=6141)
    parser.add_argument("--admission-label", default="GPU-exclusive, not quiet-host certified")
    args = parser.parse_args()
    rungs = [int(q) for q in args.rungs.split(",")]
    ms = [int(m) for m in args.ms.split(",")]
    timers = args.timers.split(",")
    if sorted(rungs) != [512, 768, 1024] or sorted(ms) != [1, 2048] or sorted(timers) != ["eager", "graph"]:
        parser.error("the panel requires R512/R768/R1024, M1/M2048 and eager/graph")
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    report_path = out / "uniform_production_panel.json"
    result = {"status": "started", "start_unix": time.time(), "admission": args.admission_label,
              "arguments": vars(args), "cells": [], "claim_scope": "matched production adapters; no serving or quality qualification",
              "comparator": "master production entry, never raw old-pure calls"}
    report_path.write_text(json.dumps(result, indent=1) + "\n")
    master_source, actual, archive_hash = source_tree(args, out)
    heads = {"master": actual, "pr": args.pr_head}
    result.update(source_heads=heads, master_archive_sha256=archive_hash,
                  raw_checkout_head=command(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    if args.cpu_preflight:
        for q in rungs:
            checked = [arm(args, out, master_source, heads, name, q, 1, "eager", "F") for name in ARMS]
            if any(r["status"] != "cpu_preflight_passed" or r["gpu_results"] != 0 for r in checked):
                raise ValueError("the same-entry production preflight is incomplete")
            if checked[0]["plane_hashes"] != checked[1]["plane_hashes"] or checked[0]["input_hashes"] != checked[1]["input_hashes"]:
                raise ValueError("the preflight sources do not produce identical tiny input bytes")
            result["cells"].append({"q256": q, "arms": checked, "input_bytes_match": True})
        result.update(status="cpu_preflight_passed", gpu_results=0, end_unix=time.time())
    else:
        for q in rungs:
            for M in ms:
                for timer in timers:
                    measured = []
                    for order in ("F", "R"):
                        for name in (ARMS if order == "F" else tuple(reversed(ARMS))):
                            measured.append(arm(args, out, master_source, heads, name, q, M, timer, order))
                    result["cells"].append(compare(measured, q, M, timer, args.admission_label))
                    report_path.write_text(json.dumps(result, indent=1) + "\n")
        result.update(status="complete", end_unix=time.time(),
                      needs_uniform_fast_path=any(c["slower_beyond_spread"] for c in result["cells"]))
    report_path.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"status": result["status"], "cells": len(result["cells"]), "source_heads": heads}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
