"""Measure the native interpreter, standard library and installed package bytes.

The container's scoped-site seal cannot identify a host interpreter and its
standard library. This native seal includes both, without importing the test
dependencies. Run with the named interpreter's isolated, no-bytecode mode.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import sysconfig


def identity():
    sites = sorted({Path(sysconfig.get_path(key)).resolve() for key in ("purelib", "platlib")})
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    files = {Path(sys.executable).resolve()}
    library = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var("LDLIBRARY")
    if library.is_file():
        files.add(library.resolve())
    config = Path(sys.prefix) / "pyvenv.cfg"
    if config.is_file():
        files.add(config)
    for root in [stdlib, *sites]:
        if any(root.is_relative_to(path) for path in
               (Path("/home/rob/.codex"), Path("/home/rob/.claude"))):
            raise ValueError("runtime root resolves outside allowed dependency roots")
        for directory, dirs, names in os.walk(root, followlinks=False):
            parent = Path(directory)
            # Installed packages are traversed once, through their own roots.
            dirs[:] = sorted(name for name in dirs if not (
                root == stdlib and (parent / name).resolve() in sites))
            if any((parent / name).is_symlink() for name in dirs):
                raise ValueError("runtime contains an unsupported directory symlink")
            files.update(parent / name for name in names)
    digest = hashlib.sha256()
    for path in sorted(files):
        resolved = path.resolve(strict=True)
        for restricted in (Path("/home/rob/.codex"), Path("/home/rob/.claude")):
            if resolved.is_relative_to(restricted):
                raise ValueError("runtime file resolves outside allowed dependency roots")
        if not resolved.is_file():
            raise ValueError("runtime dependency is not a regular file")
        metadata = resolved.stat()
        content = hashlib.sha256()
        with resolved.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                content.update(chunk)
        entry = [str(path), str(resolved), metadata.st_mode & 0o777, metadata.st_size,
                 content.hexdigest()]
        digest.update(json.dumps(entry, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")
    versions = sorted((d.metadata["Name"], d.version)
                      for root in sites for d in importlib.metadata.distributions(path=[str(root)]))
    if not {"pytest", "pytest-xdist"} <= {name.lower() for name, version in versions}:
        raise ValueError("native suite runtime requires pytest and pytest-xdist")
    return {"schema": "tessera.suite_runtime.v1", "executable": str(Path(sys.executable).resolve()),
            "version": sys.version, "files": len(files), "sha256": digest.hexdigest(),
            "versions": versions}


if __name__ == "__main__":
    print(json.dumps(identity(), sort_keys=True))
