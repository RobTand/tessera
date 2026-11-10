"""Preview a serving-image build. No graphics processor is requested.

``--submit`` and ``--run`` refuse because no reviewed build path exists yet.
The lead must choose a rootless Podman build in Flux plus one reviewed load of
a pinned-digest tarball, or one reviewed root-Docker entry. Saved historical
launches remain collectable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import _suite_flux  # noqa: E402
#: Image manifest architecture to the Flux resource architecture.
ARCHITECTURES = {"x86": "x86_64", "aarch64": "aarch64"}
#: The suite transport's ceiling (``_suite_flux.validate``); a full build that needs more is a held decision.
TIMEOUT_S = 1790
BUILD_REFUSAL = ("The serving image targets the root Docker store; no reviewed serving wrapper exists yet. "
                  "The lead must choose a rootless Podman build in a Flux job plus one reviewed docker load "
                  "of a pinned-digest tarball, or one reviewed root-Docker entry. --run and --submit are refused.")


def _kernel_build():
    digest = hashlib.sha256()
    for source in sorted((ROOT / "src").rglob("*")):
        if source.is_file() and source.suffix in {".py", ".cu", ".cuh", ".cpp", ".h"}:
            digest.update(source.relative_to(ROOT).as_posix().encode() + b"\0")
            digest.update(source.read_bytes())
    return "source:" + digest.hexdigest()


def builder_command(manifest, tag, output_directory, kernel_build, *, owner=None):
    """Run BuildKit in a container this job owns; ``owner`` names it and records its id."""
    owned = ["--name", owner["name"], "--cidfile", owner["cidfile"]] if owner is not None else []
    return ["docker", "run", "--rm", *owned, "--security-opt", "seccomp=unconfined",
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
    if manifest["architecture"] not in ARCHITECTURES:
        raise ValueError("The image architecture is not supported.")
    _suite_flux.validate(cpus, memory_gb, TIMEOUT_S, {})
    output_directory = str(Path(output_directory).resolve())
    kernel_build = _kernel_build()
    payload = ["tools/build_serving_image.py", "--run", "--manifest-json",
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), "--image-tag", tag,
        "--output-directory", output_directory, "--cpus", str(cpus), "--memory-gb", str(memory_gb)]
    return {"kernel_build": kernel_build, "gpu": False, "submitted": False,
        "request": {"architecture": ARCHITECTURES[manifest["architecture"]], "cores": cpus,
                    "memory_gib": memory_gb, "timeout_s": TIMEOUT_S},
        "payload": payload,
        "builder_command": builder_command(manifest, tag, output_directory, kernel_build),
        "output_path": str(Path(output_directory) / "serving-image.tar")}


def container_name(identity):
    return "tessera-image-build-" + re.sub(r"[^A-Za-z0-9]", "", identity)


def main(argv=None):
    refusal = subprocess.run(['/bin/bash', str(ROOT / 'experiments/flux_docker_guard.sh')],
                             capture_output=True, text=True)
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest")
    source.add_argument("--manifest-json")
    parser.add_argument("--image-tag")
    parser.add_argument("--output-directory", help="Use an output directory on the shared mount.")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--memory-gb", type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--submit", action="store_true", help="Refused pending the lead's root-Docker-store decision.")
    mode.add_argument("--run", action="store_true", help="Refused until a reviewed build path exists.")
    mode.add_argument("--dry-run", action="store_true", help="Print the job without a build.")
    mode.add_argument("--collect", metavar="DIR", help="Collect a saved Flux launch; only verified terminal success exits zero.")
    args = parser.parse_args(argv)
    if args.submit or args.run:
        print(BUILD_REFUSAL, file=sys.stderr)
        return 2
    if args.collect:
        record = _suite_flux.collect(Path(args.collect))
        print(json.dumps(record, sort_keys=True))
        return 0 if (record.get("state") == "terminal" and record.get("returncode") == 0
                     and not record.get("evidence_errors")) else 1
    if refusal.returncode:
        print(refusal.stderr, file=sys.stderr, end='')
        return refusal.returncode
    if not (args.manifest or args.manifest_json) or not args.image_tag or not args.output_directory:
        parser.error("a build requires a manifest, --image-tag and --output-directory")
    manifest = json.loads(args.manifest_json if args.manifest_json is not None else Path(args.manifest).read_text())
    job = build_job(manifest, args.image_tag, args.output_directory, cpus=args.cpus, memory_gb=args.memory_gb)
    print(json.dumps(job, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
