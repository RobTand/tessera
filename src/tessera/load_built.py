"""Load exact retained native bytes without a build or a build-tree write."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

ENV = "TESSERA_LOAD_BUILT"
RECORD_ENV = "NATIVE_LIBRARY_RECORD_DIR"


def refuse_build(module: str) -> None:
    if ENV in os.environ:
        raise RuntimeError(f"{module}: TESSERA_LOAD_BUILT has no retained loader for this library")


def load_built(module: str, *, torch):
    if ENV not in os.environ:
        return None
    from ._dev.native_identity import loaded_native_identity
    from .serving.backend import platform_token, probed_platform_token

    root = Path(os.environ[ENV]).resolve(strict=True)
    record_root = Path(os.environ[RECORD_ENV]).resolve(strict=True) if RECORD_ENV in os.environ else None
    if record_root is None or not record_root.is_dir() or record_root.is_relative_to(root):
        raise RuntimeError("NATIVE_LIBRARY_RECORD_DIR must name an existing directory outside the built tree")
    manifest = (root / "build.json").read_bytes()
    rows = [row for row in json.loads(manifest)["libraries"] if row["name"] == module]
    if len(rows) != 1:
        raise RuntimeError(f"{module}: build.json must name exactly one retained library")
    row = rows[0]
    token = probed_platform_token(torch=torch)
    if platform_token(torch=torch) != token or f"code={token}" not in " ".join(row["cflags"]):
        raise RuntimeError(f"{module}: retained library platform does not match the device")
    expected = row["library_so_sha256"]
    canonical = row["library_canonical_code_sha256"]
    cubins = row["cubin_canonical_sha256s"]
    if (not isinstance(cubins, list) or not cubins or
            any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in [expected, canonical, *cubins])):
        raise RuntimeError(f"{module}: build.json has incomplete code identity")
    path = (root / module / f"{module}.so").resolve(strict=True)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != expected:
            raise RuntimeError(f"{module}: retained library sha256 does not match build.json")
        spec = importlib.util.spec_from_file_location(module, path)
        lib = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lib)
        identity = loaded_native_identity(lib, __file__, expected_sha256=expected)
        if identity["path"] != str(path):
            raise RuntimeError(f"{module}: loaded library path does not match build.json")
    record = dict(key=row["key"], name=module, sha256=digest, identity_check="PASS",
                  build_json_sha256=hashlib.sha256(manifest).hexdigest(),
                  library_canonical_code_sha256=canonical, cubin_canonical_sha256s=cubins)
    with (record_root / f"pid-{os.getpid()}.jsonl").open("a") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
    return lib
