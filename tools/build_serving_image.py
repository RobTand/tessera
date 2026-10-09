"""Print or submit an owned PrismaBuild image job. No graphics processor is requested."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PB_RUN = "/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py"


def _kernel_build():
    digest = hashlib.sha256()
    for source in sorted((ROOT / "src").rglob("*")):
        if source.is_file() and source.suffix in {".py", ".cu", ".cuh", ".cpp", ".h"}:
            digest.update(source.relative_to(ROOT).as_posix().encode() + b"\0")
            digest.update(source.read_bytes())
    return "source:" + digest.hexdigest()


def builder_command(manifest, tag, output_directory, kernel_build):
    """Run BuildKit inside the container that PrismaBuild owns."""
    return ["docker", "run", "--rm", "--security-opt", "seccomp=unconfined",
        "--security-opt", "apparmor=unconfined", "--env",
        "BUILDKITD_FLAGS=--oci-worker-no-process-sandbox --oci-worker-snapshotter=native",
        "--env", "OMP_NUM_THREADS=1", "--env", "MKL_NUM_THREADS=1",
        "--env", "OPENBLAS_NUM_THREADS=1", "--volume", f"{ROOT}:/src:ro",
        "--volume", f"{output_directory}:/out", "--entrypoint", "buildctl-daemonless.sh",
        manifest["builder_image"], "build", "--frontend", "dockerfile.v0",
        "--local", "context=/src", "--local", "dockerfile=/src", "--opt",
        "filename=images/serving/Dockerfile", "--opt", "build-arg:BASE_IMAGE=" + manifest["base_image"],
        "--opt", "build-arg:PATCH_FILES=" + ",".join(manifest["patches"]),
        "--opt", "build-arg:KERNEL_BUILD=" + kernel_build, "--output",
        f"type=docker,name={tag},dest=/out/serving-image.tar"]


def build_job(manifest, tag, output_directory, *, cpus=4, memory_gb=8):
    if manifest.get("schema") != "tessera.serving-image.v1":
        raise ValueError("The image manifest schema is not supported.")
    for field in ("base_image", "builder_image"):
        reference = manifest[field]
        if "@sha256:" not in reference or len(reference.rsplit("@sha256:", 1)[1]) != 64:
            raise ValueError(f"{field} must name its manifest digest.")
    for name in manifest["patches"]:
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("Patch files must use local Python file names.")
        if not (ROOT / "images/serving/patches" / name).is_file():
            raise ValueError(f"The patch file is absent: {name}")
    if manifest["architecture"] not in {"x86", "aarch64"}:
        raise ValueError("The image architecture is not supported.")
    output_directory = str(Path(output_directory).resolve())
    kernel_build = _kernel_build()
    payload = ["python3", "tools/build_serving_image.py", "--inside-action", "--manifest-json",
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), "--image-tag", tag,
        "--output-directory", output_directory, "--cpus", str(cpus), "--memory-gb", str(memory_gb)]
    command = [sys.executable, PB_RUN, "--cwd", str(ROOT), "--tag", manifest["architecture"],
        "--container-image", manifest["builder_image"], "--cpus", str(cpus), "--demand",
        f"mem_gb={memory_gb}", "--timeout-s", "1800", "--env", "OMP_NUM_THREADS=1",
        "--env", "MKL_NUM_THREADS=1", "--env", "OPENBLAS_NUM_THREADS=1", "--env",
        f"MAX_JOBS={cpus}", "--", *payload]
    return {"kernel_build": kernel_build, "command": command, "gpu": False, "submitted": False,
        "builder_command": builder_command(manifest, tag, output_directory, kernel_build),
        "output_path": str(Path(output_directory) / "serving-image.tar")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--manifest-json")
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--output-directory", required=True, help="Use an output directory on the shared mount.")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--memory-gb", type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--submit", action="store_true", help="Submit the image job through PrismaBuild.")
    mode.add_argument("--inside-action", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--dry-run", action="store_true", help="Print the job without a build.")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest_json if args.manifest_json is not None else Path(args.manifest).read_text())
    job = build_job(manifest, args.image_tag, args.output_directory, cpus=args.cpus, memory_gb=args.memory_gb)
    print(json.dumps(job, sort_keys=True))
    if args.inside_action:
        Path(args.output_directory).mkdir(parents=True, exist_ok=True)
        return subprocess.call(job["builder_command"])
    if args.submit:
        return subprocess.call(job["command"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
