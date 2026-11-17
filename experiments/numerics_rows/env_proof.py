"""Compare container start inputs without writing environment values.

Workers send temporary per-name comparison digests through a Unix socket.
They never send values. The coordinator keeps those digests only in memory.
Results contain names and the single launch-state digest for each snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from experiments.numerics_rows.common import IMAGES, sha256, write_json
from experiments.numerics_rows.launch_state import LaunchState
from experiments.numerics_rows.start_env import container_options, worker_command, specification_sha256

IMPORT_PATHS = ("torch", "triton", "vllm", "tessera", "vllm.config", "vllm.distributed", "cv2")


def import_dependencies(state, progress):
    """Exercise H1's lazy imports and attribute environment writes without values."""
    imports, writes = [], []
    environment = type(os.environ)
    original = environment.__setitem__
    def record(mapping, name, value):
        if mapping is os.environ and mapping.get(name) != value:
            caller = sys._getframe(1)
            writes.append({"name": name, "file": caller.f_code.co_filename, "line": caller.f_lineno})
        return original(mapping, name, value)
    environment.__setitem__ = record
    try:
        for name in IMPORT_PATHS:
            progress("import-" + name)
            try:
                module = importlib.import_module(name)
                if name == "torch":
                    state.bind_math(module)
                imports.append({"module": name, "imported": True})
            except Exception as error:
                imports.append({"module": name, "imported": False, "exception_type": type(error).__name__})
    finally:
        environment.__setitem__ = original
    return imports, writes


def comparison_names(left, right):
    return sorted(k for k in left.keys() | right.keys() if left.get(k) != right.get(k))


def worker(imports):
    # Connect before imports so the coordinator can distinguish startup from import work.
    with socket.socket(socket.AF_UNIX) as channel:
        channel.connect("/ext/environment.sock")
        def send(data):
            channel.sendall((json.dumps(data, ensure_ascii=True) + "\n").encode("ascii"))
        send({"stage": "worker-start"})
        state = LaunchState(fresh=True)
        def snapshot():
            return {"fields": {k: v for k, v in state.fields().items() if k != "launch_selectors"}, "names": sorted(state.current()),
                    "comparison": {k: hashlib.sha256(v.encode("utf-8", "surrogateescape")).hexdigest()
                                   for k, v in state.current().items()}}
        data = {"start": snapshot(), "imports": []}
        if imports:
            data["imports"], data["environment_writes"] = import_dependencies(state, lambda stage: send({"stage": stage}))
            data["after_imports"] = snapshot()
        data["diagnostics"] = state.diagnostics()
        send({"stage": "snapshot", "data": data})


def receive(listener, process, progress, *, clock=time.monotonic, timeout=240):
    """Wait for worker events, while checking the container process on every poll.

    Do not wait for a socket timeout after an already failed OCI start. Import
    stages are sent before the import, and private comparison data stays in memory.
    """
    start = clock()
    listener.settimeout(0.25)
    while True:
        code = process.poll()
        if code is not None and code != 0:
            progress("container-exit", container_exit=code)
            raise RuntimeError("environment proof container exited before worker connection")
        if clock() - start >= timeout:
            raise TimeoutError("worker connection timeout")
        try:
            channel, _ = listener.accept()
            break
        except socket.timeout:
            if process.poll() is not None:
                progress("container-exit", container_exit=process.poll())
                raise RuntimeError("environment proof container exited before worker connection")
    with channel, channel.makefile("rb") as stream:
        channel.settimeout(timeout)
        while True:
            line = stream.readline()
            if not line:
                raise RuntimeError("environment proof worker closed before snapshot")
            event = json.loads(line)
            stage = event["stage"]
            if stage not in ("worker-start", "snapshot", *("import-" + name for name in IMPORT_PATHS)):
                raise RuntimeError("unknown environment proof stage")
            progress(stage)
            if stage == "snapshot":
                return event["data"]


def run(root, scratch):
    started = time.monotonic()
    report = {"verdict": "INVALID", "finished": False, "arm": None, "stage": "image-inspect", "stages": []}
    path = root / "results/environment-proof.json"
    arm_started, stage_started, current_arm = started, started, None
    def progress(stage, **fields):
        nonlocal arm_started, stage_started, current_arm
        if report["arm"] != current_arm:
            current_arm, arm_started = report["arm"], time.monotonic()
        now = time.monotonic()
        if report["stages"]:
            report["stages"][-1]["elapsed_in_stage_seconds"] = now - stage_started
        stage_started = now
        report.update(stage=stage, elapsed_seconds=now - started,
                      arm_elapsed_seconds=now - arm_started, stage_elapsed_seconds=0, **fields)
        report["stages"].append({k: report[k] for k in ("arm", "stage", "elapsed_seconds", "arm_elapsed_seconds")})
        write_json(path, report)
    progress("image-inspect")
    try:
        return run_arms(root, scratch, report, progress)
    except Exception as error:
        report.update(exception_type=type(error).__name__, elapsed_seconds=time.monotonic() - started,
                      arm_elapsed_seconds=time.monotonic() - arm_started, stage_elapsed_seconds=time.monotonic() - stage_started)
        write_json(path, report)
        return 3


def run_arms(root, scratch, report, progress):
    inspect, = json.loads((scratch / "image-inspect.json").read_text())
    if not inspect["Id"].removeprefix("sha256:").startswith(IMAGES["N"]["local_id"]):
        raise RuntimeError("environment proof requires IMG-N")
    digests = {inspect.get("Digest"), *(v.rsplit("@", 1)[-1] for v in inspect.get("RepoDigests", []))}
    if IMAGES["N"]["image_digest"] not in digests:
        raise RuntimeError("environment proof image digest differs")
    out = root / "results"
    write_json(out / "image-environment-names.json", sorted(item.split("=", 1)[0] for item in inspect["Config"].get("Env", [])))
    source = root / "cell"
    lane = source / "experiments/tessera_plugin_served.sh"
    identity = source / "experiments/build_identity.sh"
    if not all(text in lane.read_text() for text in ("-e TORCH_EXTENSIONS_DIR=/ext -e TMPDIR=/ext", "TESSERA_SERVE_MODE=", "TESSERA_GPU_MEM_UTIL:-0.85", "$(build_identity_docker_env)")):
        raise RuntimeError("serve environment specification changed; review required")
    report.update({"schema": "tessera.environment_proof.v1", "verdict": "INVALID", "device_launches": 0,
              "start_environment_spec_sha256": specification_sha256(),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(), "serve_source_sha256": sha256(lane),
              "build_identity_source_sha256": sha256(identity), "snapshots": {}, "differences": {},
              "oci_scope": "CPU Podman only; Docker --gpus and Podman CDI differences require the GPU canary"})
    snapshots = {}
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(scratch / "environment.sock"))
        listener.listen(1)
        for kind in ("row", "c1", "serve-spec"):
            report["arm"] = kind
            progress("container-start")
            arguments = ["--worker"] + (["--imports"] if kind == "row" else [])
            command = worker_command(source, "experiments.numerics_rows.env_proof", arguments)
            script = root / f"{kind}.sh"
            script.write_text("#!/bin/bash\nset -euo pipefail\nexec " + command + "\n")
            options = container_options(scratch)
            name = f"tessera-env-proof-{root.name}-{kind}"
            call = ["podman", "run", "--rm", "--name", name, "--pull=never", "--cgroups=disabled", "--network=none",
                    "-v", "/mnt/shared:/mnt/shared:ro", "-v", f"{root}/code:/numerics:ro", *options,
                    "--entrypoint", "/bin/bash", IMAGES["N"]["local_id"], "-euo", "pipefail", str(script)]
            process = subprocess.Popen(call, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                data = receive(listener, process, progress)
                progress("container-wait")
                code = process.wait(timeout=60)
                report["container_exit"] = code
                if code != 0:
                    raise RuntimeError("environment proof container failed: " + kind)
            finally:
                subprocess.run(["podman", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if process.poll() is None:
                    process.kill()
                    process.wait()
            snapshots[kind] = data["start"]
            report["snapshots"][kind] = data["start"]["fields"]
            if kind == "row":
                after = data["after_imports"]
                before = data["start"]["comparison"]
                report["imports"] = data["imports"]
                report["environment_writes"] = data["environment_writes"]
                report["snapshots"]["row-after-imports"] = {**after["fields"], "current_names": after["names"]}
                report["changed_or_removed_after_imports"] = sorted(k for k, v in before.items() if after["comparison"].get(k) != v)
                report["environment_diagnostics"] = data["diagnostics"]
    for kind in ("c1", "serve-spec"):
        report["differences"][kind] = comparison_names(snapshots["row"]["comparison"], snapshots[kind]["comparison"])
    same = not any(report["differences"].values()) and not report["changed_or_removed_after_imports"]
    report["verdict"] = "SAME" if same and all(r["imported"] for r in report["imports"]) else "DIFFERENT"
    report.update(finished=True, stage="complete")
    write_json(out / "environment-proof.json", report)
    print(json.dumps({"verdict": report["verdict"], "differing_names": report["differences"],
                      "changed_or_removed_names": report["changed_or_removed_after_imports"]}, sort_keys=True))
    return 0 if report["verdict"] == "SAME" else 3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--imports", action="store_true")
    p.add_argument("--root", type=Path)
    p.add_argument("--scratch", type=Path)
    args = p.parse_args()
    if args.worker:
        worker(args.imports)
    else:
        result = run(args.root, args.scratch)
        raise SystemExit(result)


if __name__ == "__main__":
    main()
