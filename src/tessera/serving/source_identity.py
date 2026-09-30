"""Which Tessera code a cell was measured on, as a digest of its source.

A lane cell in ``runtime_contract.json`` is a claim about what the serving code
executes. Its ``runtime`` block names the image and the toolchain, and it can
also name the Tessera code: ``tessera_commit`` for a person to find the tree,
and ``serving_source_sha256`` for a program to compare. The serve is often an
editable install (``pip install -e``), which records no commit, so the value a
reader compares is a digest of the files, not a commit.

The digest covers **every source file in the package**: each ``.py`` file and
each native source (``.cu``, ``.cpp`` and the like) under ``tessera/``. It
does not try to cover only the modules the serving package can reach. That
closure was measured at 82 of the package's 93 modules, and computing it means
following string-named route builders and ``import_module`` calls, which a
static walk can miss. A digest that can be too short is the defect this field
exists to rule out. The whole package cannot be short, so it moves on some
edits that could not change a serve. Each of those costs one re-attestation,
which is the safe error.

The module uses only the standard library and reads files as bytes. It never
parses or runs them, so a consumer can call it with no GPU, no torch and no
vLLM.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..source_profiles import PACKAGE_SOURCE_V2, source_profiles

__all__ = [
    "SOURCE_IDENTITY_ALGORITHM",
    "SOURCE_SUFFIXES",
    "serving_source_files",
    "serving_source_sha256",
    "serving_source_profiles",
    "PACKAGE_SOURCE_V2",
]

#: Names the digest's recipe. A change to what the digest covers or how it
#: hashes gets a new name, so two digests are comparable only under one name.
SOURCE_IDENTITY_ALGORITHM = "tessera.package_source.v1"

#: The suffixes that are code: Python, and the native sources an extension is
#: built from. Data files, including ``runtime_contract.json``, are not code.
SOURCE_SUFFIXES = (".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp")


def _default_root() -> Path:
    """The directory holding the ``tessera`` package this module came from."""
    return Path(__file__).resolve().parents[2]


def serving_source_files(src: Path | None = None) -> tuple[Path, ...]:
    """Every source file the digest covers, sorted, under ``src/tessera``.

    ``src`` is the directory that CONTAINS the ``tessera`` package: a
    checkout's ``src/``, or ``site-packages``. By default it is the one this
    module was imported from.
    """
    src = (src if src is not None else _default_root()).resolve()
    package = src / "tessera"
    if not (package / "serving").is_dir():
        raise FileNotFoundError(f"{package} is not a Tessera package with a serving plugin")
    return tuple(sorted(
        path for path in package.rglob("*")
        if path.suffix in SOURCE_SUFFIXES and path.is_file()
        and "__pycache__" not in path.parts))


def _profiles(src: Path) -> dict[str, str]:
    return source_profiles(
        ((path.relative_to(src).as_posix(), path.read_bytes())
         for path in serving_source_files(src)),
        legacy_profile=SOURCE_IDENTITY_ALGORITHM,
        legacy_prefix=SOURCE_IDENTITY_ALGORITHM.encode() + b"\0")


def _digest(src: Path) -> str:
    return _profiles(src)[SOURCE_IDENTITY_ALGORITHM]


@lru_cache(maxsize=None)
def _cached_digest(src: str) -> tuple[tuple[str, str], ...]:
    # Both profiles describe the same startup snapshot. The memo is private
    # and immutable; callers receive fresh dictionaries rather than its state.
    return tuple(_profiles(Path(src)).items())


def serving_source_sha256(src: Path | None = None) -> str:
    """The sha256 of the package's source paths and bytes under ``src``.

    Paths are hashed relative to ``src``, so a checkout's ``src/`` and an
    installed ``site-packages`` holding the same files give the same digest.
    The value is cached per ``src`` for the life of the process: a serve
    reports the code it started on, not an edit made to the checkout while it
    ran.
    """
    root = (src if src is not None else _default_root()).resolve()
    return dict(_cached_digest(str(root)))[SOURCE_IDENTITY_ALGORITHM]


def serving_source_profiles(src: Path | None = None) -> dict[str, str]:
    """The labelled legacy and length-framed profiles of one startup snapshot.

    This does not rewrite a cell's existing ``serving_source_sha256`` stamp or
    reinterpret it as v2. Qualification records keep their declared profile.
    """
    root = (src if src is not None else _default_root()).resolve()
    return dict(_cached_digest(str(root)))
