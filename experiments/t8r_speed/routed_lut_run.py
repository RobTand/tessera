"""Finite PB-owned build prerequisite or same-host paired experiment842."""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from pb_staged_store import StagedInputs
from routed_lut.owner import source_root
from routed_lut_arm import publish

IMAGE = "localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a"
FILTER = r"routed_fused_kernel<\(bool\)1,[ ]*\(int\)0,[ ]*\(bool\)0,[ ]*\(bool\)0,[ ]*\(int\)4,[ ]*\(bool\)0,[ ]*\(int\)128>"
CASES = {"real", "prefix-1", "prefix-127", "prefix-129", "prefix-2047", "single-expert-tail-129"}


def require_equal_witnesses(a, b):
    wa, wb = a["output_witnesses"], b["output_witnesses"]
    if set(wa) != CASES or wa != wb or a["input_hashes"] != b["input_hashes"]:
        raise ValueError("paired output/input bits differ; no timing admitted")


def archive(root, rc):
    # Same diagnosis archive format; no source packages/models/cache are copied.
    files = [p for p in root.rglob("*") if p.is_file() and
             "roots" not in p.relative_to(root).parts and
             p.suffix in {".json", ".log", ".txt", ".csv", ".stderr", ".so", ".ncu-rep"}]
    proof = {"returncode": rc, "files": {str(p.relative_to(root)):
             {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}}
    publish(root / "artifacts.json", proof)
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as stream:
        for path in files + [root / "artifacts.json"]:
            stream.add(path, arcname=str(path.relative_to(root)))
    print("ROUTED_GATE_ARTIFACTS_BEGIN")
    print(base64.b64encode(data.getvalue()).decode())
    print("ROUTED_GATE_ARTIFACTS_END")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "pair"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--descriptor")
    args = ap.parse_args()
    if not os.environ.get("PRISMABUILD_ACTION_KEY") or os.environ.get("ORACLE_IMAGE") != IMAGE:
        raise ValueError("exact PB/image admission required")
    if args.mode == "pair" and not args.descriptor:
        raise ValueError("sealed paired descriptor required")
    checkout = Path.cwd(); root = checkout / ".routed-lut"
    root.mkdir(exist_ok=False)
    start = time.time(); rc = 1; units = 0
    def progress(phase):
        nonlocal units
        units += 1
        helper = os.environ.get("PRISMABUILD_ACTION_PROGRESS_HELPER")
        if not helper: raise ValueError("declared progress helper missing")
        subprocess.run([sys.executable, helper, "--phase", phase, "--units", str(units)], check=True)
    try:
        reader = StagedInputs(args.manifest)
        try:
            roots = {arm: source_root(reader,
                     "/mnt/shared/astra-routed-lut-20261002/inputs/baseline-source.tar.gz",
                     "/mnt/shared/astra-routed-lut-20261002/inputs/readonly.patch", root / "roots" / arm, arm)
                     for arm in ["A", "B"]}
            publish(root / "source-roots.json", {"roots": {a: str(p) for a, p in roots.items()}, "staged_reads": reader.reads})
        finally: reader.close()
        progress("source")
        env = {**os.environ, "BENCH_STRICT_STAGED": "1", "BENCH_PY": "routed_lut_arm.py",
               "TESSERA_FUSED_E4M3_MMA": "e4m3",
               "BENCH_RO_MOUNTS": "/mnt/shared/astra-routed-lut-20261002/inputs"}
        def arm(name, which, phase):
            output = root / name
            child = {**env, "BENCH_SRC": str(roots[which])}
            child.pop("BENCH_EXT_DIR", None)
            if phase != "build":
                extensions = root / "extensions" / which
                extensions.mkdir(parents=True, exist_ok=True)
                child["BENCH_EXT_DIR"] = str(extensions)
            if phase == "profile": child.update(BENCH_NCU="1", BENCH_NCU_KERNELS=FILTER)
            argv = ["bash", "experiments/t8r_speed/bench_t8r.sh", str(checkout), str(output),
                    "--arm", which, "--phase", phase, "--artifact",
                    "/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported"]
            if args.mode == "pair": argv += ["--input-manifest", args.manifest, "--descriptor", args.descriptor]
            with (root / (name + ".log")).open("xb") as log:
                subprocess.run(argv, env=child, check=True, stdout=log, stderr=subprocess.STDOUT)
            result = json.loads((output / ("build.json" if phase == "build" else "arm.json")).read_text())
            if result["arm"] != which or result["phase"] != phase:
                raise ValueError("foreign arm result")
            if phase == "profile":
                for page, extra in [("raw", []), ("source", ["--print-source", "sass"])]:
                    with (output / (page + ".csv")).open("xb") as stream:
                        subprocess.run(["/opt/nvidia/nsight-compute/2025.3.1/ncu", "--import", str(output / "t8r.ncu-rep"), "--page", page, "--csv", *extra], check=True, stdout=stream)
            progress(name)
            return result
        if args.mode == "build":
            arm("build", "B", "build")
        else:
            a = arm("checkA", "A", "check"); b = arm("checkB", "B", "check")
            require_equal_witnesses(a, b)
            timed = {name: arm(name, which, "time") for name, which in [("A1", "A"), ("B1", "B"), ("B2", "B"), ("A2", "A")]}
            arm("profileA", "A", "profile"); arm("profileB", "B", "profile")
            costs = {k: 1000 * v["steady"]["seconds"] / v["steady"]["calls"] for k, v in timed.items()}
            paired = {"first": costs["B1"] / costs["A1"], "reverse": costs["B2"] / costs["A2"]}
            publish(root / "paired-result.json", {"steady_wall_ms_per_forward": costs, "paired_ratio": paired,
                    "wall_threshold_passes": all(x <= .98 for x in paired.values()),
                    "counter_acceptance": "requires actual MIO/FULL/memory comparison; not inferred from timing",
                    "energy_status": "HOLD", "scope": "historical proxy only"})
        (root / "action-window.txt").write_text(f"{start} {time.time()}\n")
        if args.mode == "pair":
            subprocess.run([sys.executable, "experiments/t8r_speed/routed_gate_netdata.py", str(root)], check=True)
        rc = 0
        archive(root, rc)
        progress("publish")
        return 0
    finally:
        if rc:
            archive(root, rc)


if __name__ == "__main__":
    raise SystemExit(main())
