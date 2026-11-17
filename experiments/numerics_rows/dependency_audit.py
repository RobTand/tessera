"""Collect pinned-package source reads and observed serve inputs for review.

Run image inspection in an admitted CPU job. This collector does not certify
that a source audit is complete. A reviewer must trace the counted paths.
The process mode reads existing processes and never starts a serve.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess

from experiments.numerics_rows.common import check_runtime, sha256, write_json
from experiments.numerics_rows.launch_state import LaunchState, PREFIXES, digest


def serve_inputs(artifact):
    """Record launch-policy values and all other names, never unrelated values."""
    processes = {}
    for path in Path("/proc").glob("[0-9]*"):
        try:
            command = (path / "cmdline").read_bytes()
            parent = int(next(s.split()[1] for s in (path / "status").read_text().splitlines() if s.startswith("PPid:")))
            processes[int(path.name)] = (path, command, parent)
        except (OSError, StopIteration, ValueError):
            continue
    selected = {pid for pid, (_, cmd, _) in processes.items() if artifact.encode() in cmd and b"vllm" in cmd and pid != os.getpid()}
    while True:
        children = {pid for pid, (_, _, parent) in processes.items() if parent in selected}
        if children <= selected:
            break
        selected |= children
    observed = []
    for pid in sorted(selected):
        path, _, parent = processes[pid]
        try:
            before = (path / "stat").read_text().rsplit(")", 1)[1].split()[19]
            raw = (path / "environ").read_bytes()
            env = dict(os.fsdecode(item).split("=", 1) for item in raw.split(b"\0") if b"=" in item)
            after = (path / "stat").read_text().rsplit(")", 1)[1].split()[19]
            if before != after:
                raise RuntimeError("process identity changed while reading inputs")
            state = LaunchState()
            included = {k: v for k, v in env.items() if state.policy.get(k) != "exclude" and
                        (k.startswith(PREFIXES) or state.policy.get(k) == "include")}
            if any(re.search(r"API_KEY|SECRET|PASSWORD|AUTH_TOKEN|ACCESS_TOKEN", k) for k in included):
                raise RuntimeError("included input has a secret-like name; no values recorded")
            observed.append({"pid": pid, "ppid": parent, "start_ticks": before, "environment_names": sorted(env),
                             "launch_state": included, "launch_state_sha256": digest(included), "launch_levers_sha256": state.levers_sha256,
                             "unclassified_names": sorted(set(env) - set(included) - set(state.policy)),
                             "excluded_diagnostics": {k: env[k] for k in env if state.policy.get(k) == "exclude"}})
        except (OSError, ValueError, RuntimeError) as error:
            observed.append({"pid": pid, "error": str(error)})
    return {"host": socket.gethostname(), "utc": datetime.now(timezone.utc).isoformat(),
            "artifact": artifact, "observed": bool(observed), "processes": observed,
            "limitation": "/proc environ is the process-start environment; latched or later writes need the worker-start validator"}


def package_reads(root):
    reads, errors = [], []
    for path in sorted(root.rglob("*.py")):
        try:
            text = path.read_text()
            if not any(word in text for word in ("environ", "getenv")):
                continue
            tree = ast.parse(text)
            file_sha256 = sha256(path)
            for node in ast.walk(tree):
                expression = ast.unparse(node.func) if isinstance(node, ast.Call) else ast.unparse(node.value) if isinstance(node, ast.Subscript) else ""
                if "environ" not in expression and "getenv" not in expression:
                    continue
                argument = node.args[0] if isinstance(node, ast.Call) and node.args else node.slice if isinstance(node, ast.Subscript) else None
                name = argument.value if isinstance(argument, ast.Constant) and isinstance(argument.value, str) else None
                reads.append({"file": str(path), "line": node.lineno, "name": name,
                              "expression": ast.unparse(node), "file_sha256": file_sha256})
        except (OSError, SyntaxError, UnicodeError) as error:
            errors.append({"file": str(path), "error": str(error)})
    return reads, errors


def image_inputs(out):
    state = LaunchState(fresh=True)
    runtime = check_runtime("N")
    packages, reads, errors, libraries = {}, [], [], []
    for name in ("torch", "triton", "vllm", "nvidia-cublas-cu13", "nvidia-cudnn-cu13"):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = {"absent": True}
            continue
        root = Path(dist.locate_file(name.replace("-", "_")))
        packages[name] = {"version": dist.version, "root": str(root)}
        if root.is_dir():
            found, failed = package_reads(root)
            reads.extend(found)
            errors.extend(failed)
        for item in dist.files or ():
            if str(item).endswith(".so") or ".so." in str(item):
                path = Path(dist.locate_file(item)).resolve()
                with path.open("rb") as stream:
                    file_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
                libraries.append({"package": name, "path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256})
    import torch
    state.bind_math(torch)
    sources = out / "hook-sources"
    sources.mkdir()
    for package, relative in (("triton", "compiler/compiler.py"), ("triton", "runtime/jit.py"),
                              ("triton", "backends/nvidia/driver.py"), ("triton", "backends/nvidia/compiler.py")):
        if "root" not in packages[package]:
            continue
        path = Path(packages[package]["root"]) / relative
        if path.is_file():
            (sources / relative.replace("/", "_")).write_bytes(path.read_bytes())
    return {"runtime": runtime, "complete": False, "review_required": "Trace counted preparation, quantization, composition and launch paths; classify dynamic reads and names outside conservative prefixes.",
            "packages": packages, "reads": reads, "parse_errors": errors, "image_libraries": libraries,
            **state.fields(), "environment_diagnostics": state.diagnostics(),
            "cuobjdump": subprocess.run(["cuobjdump", "--version"], capture_output=True, text=True).stdout}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--serve-artifact")
    p.add_argument("--code-rosters", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = serve_inputs(args.serve_artifact) if args.serve_artifact else image_inputs(args.out)
    if args.code_rosters:
        import ctypes
        from experiments.numerics_rows.dependency_code import CodeAudit, IMAGE_CODE

        inventory = json.loads(IMAGE_CODE.read_text())
        handles = [ctypes.CDLL(r["path"]) for r in inventory["libraries"]]
        audit = CodeAudit(report["runtime"]["image_digest"], args.out / "compiled-code").install()
        try:
            libraries = audit.image_libraries()
            if len(libraries) != len(handles) or any(not r["symbols"] for r in libraries):
                raise RuntimeError("pinned image CUDA function roster is missing or empty")
            write_json(args.out / "image-code-rosters.json", {"runtime": report["runtime"], "libraries": libraries,
                       "triton_load_run_observer_installed": True, "device_launches": 0})
        finally:
            audit.close()
    write_json(args.out / "dependency-audit.json", report)


if __name__ == "__main__":
    main()
