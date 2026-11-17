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
from pathlib import Path
import socket
import subprocess
import sys

from experiments.numerics_rows.common import IMAGES, sha256, write_json
from experiments.numerics_rows.launch_state import LaunchState
from experiments.numerics_rows.start_env import container_options, worker_command, specification_sha256


def comparison_names(left, right):
    return sorted(k for k in left.keys() | right.keys() if left.get(k) != right.get(k))


def worker(imports):
    state = LaunchState(fresh=True)
    def snapshot():
        return {"fields": {k: v for k, v in state.fields().items() if k != "launch_selectors"}, "names": sorted(state.current()),
                "comparison": {k: hashlib.sha256(v.encode("utf-8", "surrogateescape")).hexdigest()
                               for k, v in state.current().items()}}
    data = {"start": snapshot(), "imports": []}
    if imports:
        for name in ("torch", "triton", "vllm", "tessera"):
            try:
                importlib.import_module(name)
                data["imports"].append({"module": name, "imported": True})
            except Exception as error:
                data["imports"].append({"module": name, "imported": False, "exception_type": type(error).__name__})
        data["after_imports"] = snapshot()
    data["diagnostics"] = state.diagnostics()
    with socket.socket(socket.AF_UNIX) as channel:
        channel.connect("/ext/environment.sock")
        channel.sendall(json.dumps(data, ensure_ascii=True).encode("ascii"))


def serve_options(scratch):
    """Repeat the lane's -e set independently; add the declared OCI constants."""
    return ["--ipc=host", "--hostname=tessera-worker", "--workdir=/work",
            "-v", f"{scratch}:/ext:rw", "-v", f"{scratch}/home:/root:rw",
            "-v", f"{scratch}/image-inspect.json:/run/numerics/image-inspect.json:ro",
            "--env", "TORCH_EXTENSIONS_DIR=/ext", "--env", "TMPDIR=/ext",
            "--env", "TESSERA_SERVE_MODE=resident", "--env", "TESSERA_GPU_MEM_UTIL=0.85",
            "--env", "HOME=/root", "--env", "HOSTNAME=tessera-worker", "--env", "container=oci",
            "--env", "OMP_NUM_THREADS=1", "--env", "MKL_NUM_THREADS=1",
            "--env", "OPENBLAS_NUM_THREADS=1", "--env", "MAX_JOBS=1"]


def run(root, scratch):
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
    report = {"schema": "tessera.environment_proof.v1", "verdict": "INVALID", "device_launches": 0,
              "start_environment_spec_sha256": specification_sha256(), "serve_source_sha256": sha256(lane),
              "build_identity_source_sha256": sha256(identity), "snapshots": {}, "differences": {},
              "oci_scope": "CPU Podman only; Docker --gpus and Podman CDI differences require the GPU canary"}
    snapshots = {}
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(scratch / "environment.sock"))
        listener.listen(1)
        listener.settimeout(240)
        for kind in ("row", "c1", "serve-spec"):
            arguments = ["--worker"] + (["--imports"] if kind == "row" else [])
            command = worker_command(source, "experiments.numerics_rows.env_proof", arguments)
            script = root / f"{kind}.sh"
            script.write_text("#!/bin/bash\nset -euo pipefail\nexec " + command + "\n")
            options = serve_options(scratch) if kind == "serve-spec" else container_options(scratch)
            name = f"tessera-env-proof-{root.name}-{kind}"
            call = ["podman", "run", "--rm", "--name", name, "--pull=never", "--cgroups=disabled", "--network=none",
                    "-v", "/mnt/shared:/mnt/shared:ro", "-v", f"{root}/code:/numerics:ro", *options,
                    "--entrypoint", "/bin/bash", IMAGES["N"]["local_id"], "-euo", "pipefail", str(script)]
            process = subprocess.Popen(call, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                connection, _ = listener.accept()
                with connection, connection.makefile("rb") as stream:
                    connection.settimeout(240)
                    data = json.load(stream)
                if process.wait(timeout=60) != 0:
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
                report["snapshots"]["row-after-imports"] = {**after["fields"], "current_names": after["names"]}
                report["changed_or_removed_after_imports"] = sorted(k for k, v in before.items() if after["comparison"].get(k) != v)
                report["environment_diagnostics"] = data["diagnostics"]
    for kind in ("c1", "serve-spec"):
        report["differences"][kind] = comparison_names(snapshots["row"]["comparison"], snapshots[kind]["comparison"])
    same = not any(report["differences"].values()) and not report["changed_or_removed_after_imports"]
    report["verdict"] = "SAME" if same and all(r["imported"] for r in report["imports"]) else "DIFFERENT"
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
        try:
            result = run(args.root, args.scratch)
        except Exception as error:
            write_json(args.root / "results/environment-proof.json", {"verdict": "INVALID", "finished": False,
                       "exception_type": type(error).__name__})
            result = 3
        raise SystemExit(result)


if __name__ == "__main__":
    main()
