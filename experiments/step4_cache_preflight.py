"""Check the pinned image's cache writers after the installer drops to UID 1000."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

SCHEMA = "tessera.step4_worker_cache_preflight.v1"
CACHE_ENV = {
    "HOME": "/jit/home", "XDG_CACHE_HOME": "/jit/xdg", "TMPDIR": "/jit/tmp",
    "TRITON_CACHE_DIR": "/jit/triton", "TORCH_EXTENSIONS_DIR": "/jit/torch-extensions",
    "TORCHINDUCTOR_CACHE_DIR": "/jit/inductor", "CUDA_CACHE_PATH": "/jit/cuda-cache",
}


def _write_read(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="pact-uid-preflight-", dir=directory, delete=True) as handle:
        handle.write(b"worker-cache-write-read")
        handle.flush()
        handle.seek(0)
        if handle.read() != b"worker-cache-write-read":
            raise RuntimeError("worker cache did not retain written bytes")


def check_worker_caches(output: Path) -> dict:
    if os.getuid() != 1000 or os.getgid() != 1000:
        raise RuntimeError("cache preflight did not run as pinned engine worker UID/GID 1000")
    if {key: os.environ.get(key) for key in CACHE_ENV} != CACHE_ENV:
        raise RuntimeError("worker cache environment differs from explicit pinned-image roots")
    import torch
    import torch._inductor.codecache as inductor
    import torch.utils.cpp_extension as extension
    import triton.runtime.cache as triton_cache

    actual = {"torch_inductor": inductor.cache_dir(),
              "torch_extensions": extension._get_build_directory("pact_cache_preflight", False),
              "tempfile": tempfile.gettempdir()}
    expected = {"torch_inductor": CACHE_ENV["TORCHINDUCTOR_CACHE_DIR"],
                "torch_extensions": CACHE_ENV["TORCH_EXTENSIONS_DIR"] + "/pact_cache_preflight",
                "tempfile": CACHE_ENV["TMPDIR"]}
    if actual != expected:
        raise RuntimeError(f"framework cache paths differ from explicit roots: {actual}")
    for path in CACHE_ENV.values():
        _write_read(Path(path))
    for path in actual.values():
        _write_read(Path(path))
    # Exercise Triton's cache manager, rather than only touching its parent.
    manager = triton_cache.get_cache_manager(hashlib.sha256(b"pact-cache-preflight").hexdigest())
    cached = Path(manager.put(b"worker-triton-write-read", "pact-preflight.bin"))
    if cached.read_bytes() != b"worker-triton-write-read":
        raise RuntimeError("worker Triton cache did not retain written bytes")
    record = {"schema": SCHEMA, "uid": os.getuid(), "gid": os.getgid(),
              "torch_version": torch.__version__, "environment": CACHE_ENV,
              "framework_paths": actual, "triton_cache_file": str(cached),
              "write_read": True}
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record
