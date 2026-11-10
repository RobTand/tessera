"""Print, submit or run an owned serving-image build job. No graphics processor is requested.

``--submit`` hands the job to the native Flux suite transport (``_suite_flux.submit``);
the Flux job's payload is this tool's ``--run`` mode, which executes the BuildKit
container and removes the container it owns.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
#: Image manifest architecture to the Flux resource architecture.
ARCHITECTURES = {"x86": "x86_64", "aarch64": "aarch64"}
#: The suite transport's ceiling (``_suite_flux.validate``); a full build that needs more is a held decision.
TIMEOUT_S = 1790


def _terminate(signum, frame):
    raise SystemExit(128 + signum)


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
    sys.path.insert(0, str(ROOT / "tools"))
    import _suite_flux
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


def submit(job, launch_directory, checkout, python):
    """Hand the job to the native Flux suite transport and return its saved record."""
    sys.path.insert(0, str(ROOT / "tools"))
    import _suite_flux
    request = job["request"]
    return _suite_flux.submit(launch_directory, checkout, [str(python), "-s", *job["payload"]],
                              request["cores"], request["memory_gib"], request["timeout_s"], {},
                              architecture=request["architecture"])


def remove_owned_container(cidfile, runner=subprocess.run):
    """Remove the container this job started. True when none it owns remains."""
    path = Path(cidfile)
    if not path.is_file():
        return True
    container = path.read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{12,64}", container):
        return False
    result = runner(["docker", "rm", "-f", container], check=False, capture_output=True, text=True)
    return result.returncode == 0 or "No such container" in (result.stderr or "")


def run_build(manifest, tag, output_directory, kernel_build, *, runner=subprocess.run, environ=os.environ):
    """Flux job payload: run BuildKit, then remove the container whatever the outcome."""
    scratch = Path(environ["TMPDIR"])
    owner = {"name": "tessera-image-build-" + re.sub(r"[^A-Za-z0-9]", "", environ.get("FLUX_JOB_ID", str(os.getpid()))),
             "cidfile": str(scratch / "builder.cid")}
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    previous = signal.signal(signal.SIGTERM, _terminate)
    try:
        code = runner(builder_command(manifest, tag, output_directory, kernel_build, owner=owner),
                      check=False).returncode
    finally:
        try:
            removed = remove_owned_container(owner["cidfile"], runner)
        finally:
            signal.signal(signal.SIGTERM, previous)
    return code or (0 if removed else 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest")
    source.add_argument("--manifest-json")
    parser.add_argument("--image-tag")
    parser.add_argument("--output-directory", help="Use an output directory on the shared mount.")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--memory-gb", type=int, default=8)
    parser.add_argument("--launch-directory", help="New directory for the Flux launch record (--submit).")
    parser.add_argument("--checkout", help="Detached clean shared checkout to run in (--submit).")
    parser.add_argument("--python", default=sys.executable, help="Interpreter the Flux job runs (--submit).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--submit", action="store_true", help="Submit the image job through the native Flux transport.")
    mode.add_argument("--run", action="store_true", help="Run the build in this process (the Flux job payload).")
    mode.add_argument("--dry-run", action="store_true", help="Print the job without a build.")
    mode.add_argument("--collect", metavar="DIR", help="Collect a saved Flux launch; only verified terminal success exits zero.")
    args = parser.parse_args(argv)
    if args.collect:
        import _suite_flux
        record = _suite_flux.collect(Path(args.collect))
        print(json.dumps(record, sort_keys=True))
        return 0 if (record.get("state") == "terminal" and record.get("returncode") == 0
                     and not record.get("evidence_errors")) else 1
    if not (args.manifest or args.manifest_json) or not args.image_tag or not args.output_directory:
        parser.error("a build requires a manifest, --image-tag and --output-directory")
    manifest = json.loads(args.manifest_json if args.manifest_json is not None else Path(args.manifest).read_text())
    job = build_job(manifest, args.image_tag, args.output_directory, cpus=args.cpus, memory_gb=args.memory_gb)
    print(json.dumps(job, sort_keys=True))
    if args.run:
        return run_build(manifest, args.image_tag, str(Path(args.output_directory).resolve()), job["kernel_build"])
    if args.submit:
        if args.launch_directory is None or args.checkout is None:
            parser.error("--submit requires --launch-directory and --checkout")
        record = submit(job, args.launch_directory, args.checkout, args.python)
        print(json.dumps(record, sort_keys=True))
        if record.get("state") == "terminal":
            return 0 if record.get("returncode") == 0 and not record.get("evidence_errors") else 1
        return 0 if record.get("state") == "running" else 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
