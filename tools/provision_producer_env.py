#!/usr/bin/env python3
"""Provision the scoped producer environment overlay for a Tessera commit.

The precedent (fleet record prismaquant-pr-2131, PB action e329d098...) is
followed exactly, because it is the one recipe already qualified:

* the wheel is built from the CLEAN COMMITTED tree (``git archive HEAD``) --
  never from a dirty snapshot tree, and never by copying repository tooling
  into the environment: pyproject's ``tessera._dev*`` exclusion stands, so the
  installed package genuinely is what a consumer installs;
* the venv is created ``--copies --without-pip`` from a CLEAN base interpreter,
  which is only ever READ: the base's scientific stack rides along through a
  ``scientific-dependencies/`` hardlink overlay (``cp -al``) plus one
  ``.pth``, so torch/numpy/safetensors resolve without mutating the base or
  duplicating gigabytes;
* ``tessera-quant`` is installed ``--no-deps`` from that wheel with the base's
  pip pointed at the new interpreter; the extracted source is REMOVED before
  any probe, so the probe sees only the installed distribution;
* the record is append-only: an existing venv or record is a refusal, never
  an overwrite, and no pre-existing environment of any name is touched.

``--phase acquire`` (the default) provisions and records; ``--phase qualify``
additionally binds the checkout to its exact clean HEAD and runs the core
sibling's ``authenticate_producer_python()`` inside the new interpreter
against ``TESSERA_PRODUCER_SOURCE``, writing ``qualification.json`` -- that
phase refuses unless ``--final`` is passed, because final installed-Tessera
qualification waits for the frozen coherent core commit.

usage (as a PB CPU action payload, run from the sealed checkout root):
  python tools/provision_producer_env.py \
      --base /home/rob/venvs/pq-cu130 \
      --name pq-b770d-producer-<commit8>-<date> [--phase qualify --final]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CHECKOUT = Path(__file__).resolve().parents[1]
SCHEMA = "tessera.producer-env-acquisition.v1"

#: The installed-package digest rule of the existing qualification descriptor
#: (prismaquant.test_projection_producer.v1): every shipped ``tessera/**``
#: file with a Python, JSON or CUDA suffix, hashed as ``name\\0bytes`` in
#: sorted order.  This is the descriptor's convention, restated nowhere else.
PROBE_SUFFIXES = {".py", ".json", ".cu"}


def git_in(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


#: The campaign's source ancestor every qualified producer tree must descend
#: from (the b770 descriptor's source_commit).
B770_ANCESTOR = "b770727c50eef822132518bdc4fd6efe84359c9e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def interpreter_facts(python: Path) -> dict:
    """Version, executable digest and torch facts of one interpreter, read-only."""
    out = subprocess.run(
        [str(python), "-I", "-c",
         "import hashlib, json, sys\n"
         "from pathlib import Path\n"
         "rec = {'python': sys.version.split()[0],\n"
         "       'executable': str(Path(sys.executable).resolve()),\n"
         "       'executable_sha256': hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()}\n"
         "try:\n"
         "    import torch\n"
         "    rec['torch'] = torch.__version__\n"
         "    rec['torch_cuda_build'] = torch.version.cuda\n"
         "    rec['torch_cuda_available'] = bool(torch.cuda.is_available())\n"
         "except ImportError:\n"
         "    rec['torch'] = None\n"
         "try:\n"
         "    import tessera\n"
         "    rec['tessera_file'] = str(Path(tessera.__file__).resolve())\n"
         "    rec['tessera_version'] = getattr(tessera, '__version__', None)\n"
         "    from tessera.serving.source_identity import serving_source_sha256\n"
         "    rec['serving_source_sha256_installed'] = serving_source_sha256()\n"
         "except ImportError:\n"
         "    rec['tessera_file'] = None\n"
         "print(json.dumps(rec))\n"],
        check=True, capture_output=True, text=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def installed_package_probe(python: Path) -> dict:
    """The descriptor's triple, probed in the selected interpreter, isolated."""
    probe = r'''
from pathlib import Path
import hashlib, importlib.metadata as md, json, sys
import tessera.producer_plan as producer
prefix = Path(sys.prefix).resolve()
assert Path(producer.__file__).resolve().is_relative_to(prefix), "tessera import escaped the venv"
dist = md.distribution('tessera-quant')
digest = hashlib.sha256()
for file in sorted(dist.files, key=str):
    name = str(file)
    if name.startswith('tessera/') and Path(name).suffix in {'.py', '.json', '.cu'}:
        path = Path(dist.locate_file(file)).resolve()
        assert path.is_relative_to(prefix), str(path)
        digest.update(name.encode() + b'\0' + path.read_bytes())
print(json.dumps({'executable_sha256': hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
                  'module': 'tessera.producer_plan',
                  'module_path': str(Path(producer.__file__).resolve()),
                  'module_sha256': hashlib.sha256(Path(producer.__file__).read_bytes()).hexdigest(),
                  'package_payload_sha256': digest.hexdigest()}))
'''
    out = subprocess.run([str(python), "-I", "-c", probe],
                         check=True, capture_output=True, text=True, cwd="/")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True,
                    help="clean base interpreter the overlay reads (never mutated)")
    ap.add_argument("--name", required=True,
                    help="the new venv's directory name under the base's parent; the "
                         "same name on every eligible host is what makes the path portable")
    ap.add_argument("--phase", choices=("acquire", "qualify"), default="acquire")
    ap.add_argument("--final", action="store_true",
                    help="qualify only: assert the frozen coherent core commit (final "
                         "qualification waits for it otherwise)")
    ap.add_argument("--expected-commit", default=None,
                    help="qualify only: refuse unless HEAD is exactly this commit")
    ap.add_argument("--source-ref", type=Path, default=None,
                    help="qualify only: the EXTERNALLY QUALIFIED immutable checkout whose "
                         "src/tessera the authentication binds; a PB snapshot path is not "
                         "one, so qualify refuses without it")
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
        ap.error("--name must be one directory component under the base environment parent")

    if not (CHECKOUT / "src" / "tessera").is_dir() or not (CHECKOUT / "pyproject.toml").is_file():
        raise SystemExit(f"{CHECKOUT} is not a Tessera checkout")
    base = args.base.resolve()
    base_python = base / "bin" / "python"
    if not base_python.is_file():
        raise SystemExit(f"--base has no interpreter: {base_python}")
    venv_root = base.parent / args.name
    if venv_root.exists():
        raise SystemExit(f"refusing: {venv_root} already exists (records are append-only; "
                         "provision a new name instead of mutating a provisioned environment)")

    # The authoritative BUILD git root.  Qualify never builds from this script's
    # PB-snapshot root (a parentless tree whose HEAD is not the branch commit it
    # may be expected to match): it builds from the validated external
    # --source-ref checkout -- real HEAD, genuine ancestry, clean committed
    # payload -- and authenticates against that same ref.  Acquire (prototype)
    # archives the sealed snapshot's HEAD and says so in its record.
    if args.phase == "qualify":
        if not args.source_ref:
            raise SystemExit("--phase qualify requires --source-ref: the externally "
                             "qualified immutable checkout to build and authenticate; a "
                             "PB snapshot root is not one")
        build_root = Path(args.source_ref).resolve()
        if not (build_root / "src" / "tessera").is_dir() or not (build_root / "pyproject.toml").is_file():
            raise SystemExit(f"--source-ref {build_root} is not a Tessera checkout")
        if not (build_root / ".git").exists():
            raise SystemExit(f"--source-ref {build_root} is not a git checkout; "
                             "qualification binds shipped paths to an exact HEAD")
        dirty = git_in(build_root, "status", "--porcelain", "--untracked-files=no")
        if dirty:
            raise SystemExit(f"qualification binds shipped paths to exact HEAD; {build_root} "
                             f"is dirty:\n{dirty}")
        commit = git_in(build_root, "rev-parse", "HEAD")
        if args.expected_commit and commit != args.expected_commit:
            raise SystemExit(f"--source-ref HEAD {commit} is not the expected frozen "
                             f"commit {args.expected_commit}")
        ancestry = subprocess.run(["git", "-C", str(build_root), "merge-base", "--is-ancestor",
                                   B770_ANCESTOR, "HEAD"], capture_output=True, text=True)
        if ancestry.returncode != 0:
            raise SystemExit(f"--source-ref HEAD {commit} does not descend from the campaign "
                             f"source ancestor {B770_ANCESTOR}")
        if not args.final:
            raise SystemExit("qualification waits for the frozen coherent core commit; "
                             "pass --final only when the lead announces it")
        dirty_files = []
    else:
        build_root = CHECKOUT
        commit = git_in(build_root, "rev-parse", "HEAD")
        dirty_files = git_in(build_root, "status", "--porcelain",
                             "--untracked-files=no").splitlines()

    # Exclusively reserve only fresh paths; never remove a pre-existing path.
    venv_root.mkdir()
    work = Path(tempfile.mkdtemp(prefix=f"{args.name}-build-", dir="/home/rob/tmp"))

    # 1. The clean committed tree of the BUILD root, as an archive, built into a
    #    genuine wheel.
    archive = work / "source.tar"
    with archive.open("wb") as handle:
        subprocess.run(["git", "-C", str(build_root), "archive", "HEAD"], check=True,
                       stdout=handle)
    source = work / "source"
    source.mkdir()
    subprocess.run(["tar", "-xf", str(archive), "-C", str(source)], check=True)
    archive_sha = sha256_file(archive)
    subprocess.run([str(base_python), "-m", "pip", "wheel", "--no-deps", "-w",
                    str(work / "wheels"), str(source)], check=True,
                   capture_output=True, text=True)
    wheels = sorted((work / "wheels").glob("tessera_quant-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one tessera_quant wheel, built {wheels}")
    wheel = wheels[0]
    wheel_sha = sha256_file(wheel)
    # 2. The extracted source goes away before anything probes the install.
    shutil.rmtree(source)
    archive.unlink()

    # 3. The venv, from the clean base, by the precedent recipe.
    #    (--without-scm-ignore-files is a 3.14 spelling; older interpreters
    #    that lack it take the same recipe without it.)
    venv_argv = [str(base_python), "-m", "venv", "--copies", "--without-pip"]
    out = subprocess.run([*venv_argv, "--help"], capture_output=True, text=True)
    if "--without-scm-ignore-files" in out.stdout:
        venv_argv.append("--without-scm-ignore-files")
    subprocess.run([*venv_argv, str(venv_root)], check=True)
    python = venv_root / "bin" / "python"
    overlay = venv_root / "scientific-dependencies"
    base_site = sorted((base / "lib").glob("python*/site-packages"))
    if len(base_site) != 1:
        raise SystemExit(f"--base site-packages not found uniquely: {base_site}")
    subprocess.run(["cp", "-al", str(base_site[0]) + "/.", str(overlay)], check=True)
    pth = next((venv_root / "lib").glob("python*/site-packages")) / "scientific-dependencies.pth"
    pth.write_text(f"{overlay}\n")

    # 4. The wheel, installed --no-deps through the base's pip.
    subprocess.run([str(base_python), "-m", "pip", "--python", str(python), "install",
                    "--no-deps", str(wheel)], check=True, capture_output=True, text=True)

    # 5. Preflight and probe the INSTALLED distribution only (source is gone).
    help_run = subprocess.run([str(python), "-I", "-m", "tessera.producer_plan", "--help"],
                              capture_output=True, text=True, cwd="/")
    if help_run.returncode != 0:
        raise SystemExit(f"producer_plan --help preflight failed:\n{help_run.stderr}")
    probe = installed_package_probe(python)
    facts = interpreter_facts(python)
    base_facts = interpreter_facts(base_python)

    rec = {
        "schema": SCHEMA,
        "phase": args.phase,
        "created_utc": utc(),
        "host": os.uname().nodename,
        "machine": os.uname().machine,
        "source": {"build_git_root": str(build_root),
                   "build_git_root_kind": ("externally qualified --source-ref"
                                           if args.phase == "qualify" else
                                           "PB sealed snapshot (acquire prototype)"),
                   "commit": commit,
                   "commit_title": git_in(build_root, "log", "-1", "--format=%s"),
                   "ancestor": B770_ANCESTOR if args.phase == "qualify" else None,
                   "archive_sha256": archive_sha, "wheel_sha256": wheel_sha,
                   "wheel_bytes": wheel.stat().st_size,
                   "snapshot_dirty_files": dirty_files,
                   "snapshot_note": ("qualify built from the --source-ref checkout's clean "
                                     "committed HEAD; expected-commit and ancestry checked "
                                     "THERE" if args.phase == "qualify" else
                                     "the wheel is built from this sealed snapshot's git "
                                     "archive HEAD; dirty overlay listed for provenance "
                                     "only; final qualification re-provisions from the "
                                     "frozen --source-ref commit under a new venv name")},
        "base": {"path": str(base), "read_only": True,
                 "pyvenv_command": (base / "pyvenv.cfg").read_text().split("command = ")[-1].strip()
                 if (base / "pyvenv.cfg").is_file() else None,
                 **{k: v for k, v in base_facts.items()}},
        "venv": {"path": str(venv_root), "interpreter": str(python),
                 "overlay": {"method": "cp -al hardlinks", "directory": str(overlay),
                             "pth": str(pth)},
                 **{k: v for k, v in facts.items()},
                 "preflight": {"producer_plan_help_rc": help_run.returncode,
                               "first_line": (help_run.stdout.splitlines() or [""])[0]}},
        "installed": probe,
        "creator": os.environ.get("PB_WORKER", "ProducerBatchProbeEnvironment"),
        "provision": "PB CPU action; no GPU; no pre-existing environment touched",
    }

    record = venv_root / ("qualification.json" if args.phase == "qualify" else "acquisition.json")
    if record.exists():
        raise SystemExit(f"refusing to overwrite {record}")

    if args.phase == "qualify":
        # The core sibling owns the authentication; the qualify phase runs it
        # inside THIS interpreter against the SAME validated --source-ref
        # checkout the wheel was built from.
        env = dict(os.environ, TESSERA_PRODUCER_PYTHON=str(python),
                   TESSERA_PRODUCER_SOURCE=str(build_root / "src" / "tessera"))
        auth = subprocess.run([str(python), "-I", "-c",
                               "from tessera.export_serving import authenticate_producer_python as a;"
                               " import json; print(json.dumps(a()))"],
                              env=env, capture_output=True, text=True, cwd="/")
        if auth.returncode != 0:
            raise SystemExit(f"authenticate_producer_python refused the new environment:\n"
                             f"{auth.stdout}\n{auth.stderr}")
        receipt = json.loads(auth.stdout.strip().splitlines()[-1])
        if not isinstance(receipt, dict) or receipt.get("schema") != "tessera.producer_python.v1":
            raise SystemExit("producer authentication returned no selected-producer receipt")
        if (receipt.get("git_head") != commit
                or receipt.get("requested_interpreter") != str(python)
                or receipt.get("package_sha256") != receipt.get("expected_package_sha256")):
            raise SystemExit("producer authentication does not bind the requested final source and interpreter")
        rec["authentication"] = receipt
        (venv_root / "authentication.json").write_text(json.dumps(receipt, indent=1) + "\n")
    # A qualification record is a completed authentication, never a intent
    # document published before a failed or absent selection check.
    record.write_text(json.dumps(rec, indent=1) + "\n")

    shutil.rmtree(work, ignore_errors=True)
    print("QUALIFIED_PRODUCER " + json.dumps(rec["installed"], sort_keys=True))
    print(json.dumps({"record": str(record), "interpreter": str(python),
                      "commit": commit, "wheel_sha256": wheel_sha}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
