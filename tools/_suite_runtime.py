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


def identity(extra_sites=(), require_suite=True):
    sites = sorted({*[Path(sysconfig.get_path(key)).resolve() for key in ("purelib", "platlib")],
                    *[Path(site).resolve(strict=True) for site in extra_sites]})
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    files = {Path(sys.executable).resolve()}
    library = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var("LDLIBRARY")
    if library.is_file():
        files.add(library.resolve())
    config = Path(sys.prefix) / "pyvenv.cfg"
    if config.is_file():
        files.add(config)
    # Traverse resolved directories once, recording every alias before deduping.
    # Installed environments legitimately share package/library directories.
    directories = []
    visited = set()
    pending = [stdlib, *sites]
    restricted = (Path("/home/rob/.codex"), Path("/home/rob/.claude"))
    while pending:
        directory = pending.pop()
        resolved = directory.resolve(strict=True)
        if any(resolved.is_relative_to(path) for path in restricted):
            raise ValueError("runtime root resolves outside allowed dependency roots")
        directories.append([str(directory), str(resolved),
                            os.readlink(directory) if directory.is_symlink() else None])
        if resolved in visited:
            continue
        visited.add(resolved)
        for path in sorted(directory.iterdir()):
            target = path.resolve(strict=True)
            if any(target.is_relative_to(root) for root in restricted):
                raise ValueError("runtime file resolves outside allowed dependency roots")
            if target.is_dir():
                pending.append(path)
            else:
                files.add(path)
    digest = hashlib.sha256()
    for entry in sorted(directories):
        digest.update(json.dumps(entry, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")
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
        after = resolved.stat()
        stable = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (any(getattr(after, key) != getattr(metadata, key) for key in stable)
                or path.resolve(strict=True) != resolved):
            raise ValueError("runtime dependency changed while measuring")
        entry = [str(path), str(resolved), metadata.st_mode & 0o777, metadata.st_size,
                 content.hexdigest()]
        digest.update(json.dumps(entry, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")
    versions = sorted((d.metadata["Name"], d.version)
                      for root in sites for d in importlib.metadata.distributions(path=[str(root)]))
    if require_suite and not {"pytest", "pytest-xdist"} <= {name.lower() for name, version in versions}:
        raise ValueError("native suite runtime requires pytest and pytest-xdist")
    return {"schema": "tessera.suite_runtime.v1", "executable": str(Path(sys.executable).resolve()),
            "version": sys.version, "files": len(files), "sha256": digest.hexdigest(), "require_suite": require_suite,
            "versions": versions}


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["--wrapper"]):
        raise SystemExit("only the internal --wrapper mode is supported")
    print(json.dumps(identity(require_suite=sys.argv[1:] != ["--wrapper"]), sort_keys=True))
