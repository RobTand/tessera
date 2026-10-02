"""Closed two-arm source admission for the routed LUT experiment (#842).

Uses existing StagedInputs reads, BENCH_SRC roots and NativeCallback. It is
experiment tooling, not an installed runtime or a native binary loader.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

BASELINE_SOURCE = "3a32d040668cc1fe5678c2088deb5afa8cd6f227a7920dfbd899c90863c03254"
CANDIDATE_SOURCE = "915aa1b88a4f247b12c910b77047e670e87ad09fdbcefe7c52f2adc9b0a975ef"
PATCH_SHA = "35720e938967c8539cecc8d869aae420d7ddb0cd5d72977121462e5a8371bd37"
ARCHIVE_SHA = "0f75b911835183a2fe996731450214f2a07f5a4325fcff5cc1a14e900735449a"
BASELINE_BINARY = "0f953b69f4bc10df29f3bd82b9811cbf4c307eaf48d66331c8c2ea1f1a1334af"
OWNERS = {
    "tessera/__init__.py": "9e681d73db412b0c437af4fc5e247170a068ab05709a88a51da2e5d292104a69",
    "tessera/routed_fused.py": "e4aab881f8ac975ffb57d502ef98e8d18184eb531e31bf23745690825e1b7bbd",
    "tessera/serving/ext.py": "9bbca0a6d1c99ccc8eb6a6c2f669139ccab9376374ca5859d763e15918ce4ccc",
}
KERNEL = "tessera/serving/csrc/routed_fused_window.cu"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def source_root(reader, archive_path, patch_path, destination, arm):
    """Materialize only the two reviewed roots from public pinned owned bytes."""
    if arm not in {"A", "B"}:
        raise ValueError("unknown routed LUT arm")
    archive = reader.read(archive_path)
    patch = reader.read(patch_path)
    if digest(archive) != ARCHIVE_SHA or digest(patch) != PATCH_SHA:
        raise ValueError("routed LUT source archive/patch differs")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as stream:
        members = stream.getmembers()
        if sum(x.size for x in members) > 64 << 20:
            raise ValueError("routed LUT source archive too large")
        for member in members:
            path = Path(member.name)
            if (path.is_absolute() or ".." in path.parts
                    or (path.parts[:2] != ("src", "tessera")
                        and not (member.isdir() and path.parts == ("src",)))
                    or not (member.isdir() or member.isfile())):
                raise ValueError("foreign routed LUT source archive member")
        stream.extractall(destination, members=members, filter="data")
    if arm == "B":
        # Fixed reviewed patch; no caller-supplied diff or origin fallback.
        subprocess.run(["git", "apply", "-"],
                       cwd=destination, input=patch, check=True,
                       env={**os.environ, "GIT_CEILING_DIRECTORIES": str(destination.parent.resolve())},
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    root = destination / "src"
    verify_root(root, arm)
    return root


def verify_root(root, arm):
    root = Path(root).resolve(strict=True)
    expected = {**OWNERS, KERNEL: BASELINE_SOURCE if arm == "A" else CANDIDATE_SOURCE}
    if arm not in {"A", "B"}:
        raise ValueError("unknown routed LUT arm")
    for name, sha in expected.items():
        path = root / name
        if path.is_symlink() or digest(path.read_bytes()) != sha:
            raise ValueError("routed LUT actual source owner differs: " + name)
    return root


def verify_imports(root, arm, tessera, rf, ext):
    root = verify_root(root, arm)
    for module, relative in [(tessera, "tessera/__init__.py"),
                             (rf, "tessera/routed_fused.py"),
                             (ext, "tessera/serving/ext.py")]:
        if Path(module.__file__).resolve(strict=True) != root / relative:
            raise ValueError("foreign routed LUT Python owner origin")
    if Path(ext.native_source_path("tessera_routed_fused_mma_e4m3")).resolve() != root / KERNEL:
        raise ValueError("foreign routed LUT kernel owner origin")


def arm_descriptor(raw, arm, expected_flags):
    """Closed descriptor is a declared input, never a trustless runtime knob."""
    doc = json.loads(raw)
    if set(doc) != {"schema", "arms"} or doc["schema"] != "tessera.routed_lut_pair.v1":
        raise ValueError("unsupported routed LUT descriptor")
    if set(doc["arms"]) != {"A", "B"} or arm not in {"A", "B"}:
        raise ValueError("unknown routed LUT descriptor arm")
    entry = doc["arms"][arm]
    if set(entry) != {"binary", "binary_sha256", "kernel_sha256", "compile_flags", "owners"}:
        raise ValueError("routed LUT descriptor fields differ")
    sha = entry["binary_sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("invalid routed LUT binary digest")
    if arm == "A" and sha != BASELINE_BINARY:
        raise ValueError("routed LUT baseline binary differs")
    if entry["kernel_sha256"] != (BASELINE_SOURCE if arm == "A" else CANDIDATE_SOURCE):
        raise ValueError("routed LUT descriptor source differs")
    if entry["owners"] != OWNERS or entry["compile_flags"] != expected_flags:
        raise ValueError("routed LUT descriptor owner/flags differ")
    if not isinstance(entry["binary"], str) or not entry["binary"].startswith("/mnt/shared/") or ".." in Path(entry["binary"]).parts:
        raise ValueError("invalid routed LUT binary path")
    return entry
