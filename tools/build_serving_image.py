"""Print or submit the explicit PrismaBuild image job. No GPU work is requested."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PB_RUN = "/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py"


def build_job(root, manifest, tag, *, cpus=4, memory_gb=8):
    root = Path(root)
    if manifest.get("schema") != "tessera.serving-image.v1":
        raise ValueError("The image manifest schema is not supported.")
    base = manifest["base_image"]
    if "@sha256:" not in base or len(base.rsplit("@sha256:", 1)[1]) != 64:
        raise ValueError("The base image must name its manifest digest.")
    patches = manifest["patches"]
    for name in patches:
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("Patch files must use local Python file names.")
        if not (root / "images/serving/patches" / name).is_file():
            raise ValueError(f"The patch file is absent: {name}")
    if manifest["architecture"] not in {"x86", "aarch64"}:
        raise ValueError("The image architecture is not supported.")
    # Model support patches do not change the native kernel source identity.
    digest = hashlib.sha256()
    for source in sorted((root / "src").rglob("*")):
        if source.is_file() and source.suffix in {".py", ".cu", ".cuh", ".cpp", ".h"}:
            digest.update(source.relative_to(root).as_posix().encode() + b"\0")
            digest.update(source.read_bytes())
    kernel_build = "source:" + digest.hexdigest()
    command = [sys.executable, PB_RUN, "--cwd", str(root), "--tag", manifest["architecture"],
        "--container-image", base, "--cpus", str(cpus), "--demand", f"mem_gb={memory_gb}",
        "--timeout-s", "1800", "--env", "OMP_NUM_THREADS=1", "--env", "MKL_NUM_THREADS=1",
        "--env", "OPENBLAS_NUM_THREADS=1", "--env", f"MAX_JOBS={cpus}", "--", "docker", "build",
        "--network=none", "--pull=false", "--build-arg", f"BASE_IMAGE={base}",
        "--build-arg", "PATCH_FILES=" + ",".join(patches), "--build-arg",
        "KERNEL_BUILD=" + kernel_build, "-f", "images/serving/Dockerfile", "-t", tag, "."]
    return {"kernel_build": kernel_build, "command": command, "gpu": False, "submitted": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--memory-gb", type=int, default=8)
    parser.add_argument("--submit", action="store_true", help="Submit the image job through PrismaBuild.")
    parser.add_argument("--dry-run", action="store_true", help="Print the job without a build.")
    args = parser.parse_args(argv)
    if args.submit and args.dry_run:
        parser.error("Select either submit or dry run.")
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(Path(args.manifest).read_text())
    job = build_job(root, manifest, args.image_tag, cpus=args.cpus, memory_gb=args.memory_gb)
    print(json.dumps(job, sort_keys=True))
    if args.submit:
        return subprocess.call(job["command"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
