#!/usr/bin/env python3
"""Launch one epilogue-fold bench container (tessera#522).

Adapted from PrismaQuant's ``fp4-load-sweep-qwen3-0.6b-20260913/stage/control/sweep_launch.py``
(PQ #573), which measured the same cells in the same image.  Kept: the image
declaration built by Tessera's own resolver from docker's RepoDigests, the
selection stamp naming the serving configuration, and the plugin installer that
checks the vLLM core manifest before and after installing this tree.  Changed:
the Tessera tree is the checkout this action runs in, the driver runs
``bench_epilogue_fold.py``, and no Triton cache is seeded (the bench's compile
check writes its own; nothing here binds to a full-engine run's cache).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import uuid


def digest(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="fresh evidence directory")
    parser.add_argument("--tessera-tree", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--prep", type=Path, required=True, help="the prepared cells directory")
    parser.add_argument("--serving-config", type=Path, required=True)
    parser.add_argument("--core-manifest", type=Path, required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--arms", default="inplace,v2_host,v2_device,cutlass")
    parser.add_argument("--baseline-route", default="", help="path inside the tree, or empty")
    parser.add_argument("--power-seconds", default="5.0")
    parser.add_argument("--timeout-s", type=int, default=1700)
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True)
    tree = args.tessera_tree.resolve()
    control = tree / "experiments" / "nvfp4_epilogue"
    configuration_sha256 = digest(args.serving_config)
    base = json.loads(args.serving_config.read_text())["runtime_image"]
    inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", base]))[0]
    assert base in inspected.get("RepoDigests", []), "pinned base must appear in inspected RepoDigests"
    image_id = inspected["Id"]

    spec = importlib.util.spec_from_file_location("tessera_runtime_image_launcher",
                                                  tree / "src/tessera/serving/runtime_image.py")
    runtime_image = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime_image)
    contract = json.loads((tree / "src/tessera/serving/runtime_contract.json").read_text())
    declaration = runtime_image.require_pinned(base, contract=contract, inspector=lambda reference: {
        "present": True, "local_id": image_id,
        "repo_digests": sorted(inspected["RepoDigests"]), "error": None})
    declaration["selection"] = {
        "scope": "explicit_research_configuration_not_packaged_default_or_cell_promotion",
        "configuration_sha256": configuration_sha256,
        "packaged_default_reference": runtime_image.pinned_reference(contract)}
    image_environment = runtime_image.container_env(declaration)
    (out / "launcher-image-inspect.json").write_text(json.dumps(inspected, indent=2))
    (out / "runtime-image-declaration.json").write_text(json.dumps(declaration, indent=2))
    for name in ("triton", "flashinfer", "xdg", "extensions", "tmp", "inductor"):
        (out / "cache" / name).mkdir(parents=True, exist_ok=True)

    name = "tessera-epilogue-" + uuid.uuid4().hex[:12]
    command = ["docker", "run", "--rm", "--gpus", "all", "--ipc", "host", "--network", "none", "--name", name,
        "--volume", f"{tree}:/tessera:ro", "--volume", f"{args.prep}:/prep:ro",
        "--volume", "/mnt/shared:/mnt/shared:ro", "--volume", f"{out}:/out",
        "--workdir", "/tessera",
        "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "TESSERA_SERVE_MODE=resident",
        "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1", "--env", "VLLM_NO_USAGE_STATS=1",
        "--env", "OMP_NUM_THREADS=1", "--env", "MKL_NUM_THREADS=1",
        "--env", "TRITON_CACHE_DIR=/out/cache/triton", "--env", "TORCHINDUCTOR_CACHE_DIR=/out/cache/inductor",
        "--env", "FLASHINFER_WORKSPACE_BASE=/out/cache/flashinfer",
        "--env", "XDG_CACHE_HOME=/out/cache/xdg", "--env", "TORCH_EXTENSIONS_DIR=/out/cache/extensions",
        "--env", "TMPDIR=/out/cache/tmp", "--env", "HOME=/out/cache",
        *[item for key, value in image_environment.items() for item in ("--env", key + "=" + value)],
        "--entrypoint", "python3", image_id, "/tessera/experiments/nvfp4_epilogue/full_engine_plugin_install.py",
        "--evidence-dir", "/out", "--base-reference", base, "--launcher-image-id", image_id,
        "--launcher-image-inspect", "/out/launcher-image-inspect.json",
        "--source-tree", "/tessera", "--source-commit", args.source_commit,
        "--core-manifest", str(args.core_manifest), "--",
        "bash", "/tessera/experiments/nvfp4_epilogue/driver.sh", args.rows, args.arms,
        args.baseline_route, args.power_seconds]
    (out / "launch-command.json").write_text(json.dumps(command, indent=2))
    result = subprocess.run(["timeout", "--signal=INT", "--kill-after=60", str(args.timeout_s), *command])
    print(json.dumps({"returncode": result.returncode, "out": str(out), "control": str(control)}), flush=True)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
