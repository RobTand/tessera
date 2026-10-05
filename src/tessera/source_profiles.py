"""Standard-library source framing; no serving, encoder or wire imports."""
from __future__ import annotations

import fnmatch
import hashlib
from collections.abc import Iterable
from pathlib import PurePosixPath

PACKAGE_SOURCE_V2 = "tessera.package_source.v2"
ENCODER_SOURCE_V1 = "tessera.encoder_source.v1"


def source_profiles(records: Iterable[tuple[str, bytes]], *, legacy_profile: str,
                    legacy_prefix: bytes = b"") -> dict[str, str]:
    """Label the unchanged caller-ordered v1 and byte-name-ordered v2.

    v2 begins with ``tessera.package_source.v2\0``. Each record contains the
    unsigned eight-byte big-endian UTF-8 name length, name bytes, unsigned
    eight-byte big-endian content length and raw content. NUL is ordinary
    content, not a record separator. Selection and relative-name roots belong
    to the caller; neither is silently broadened here.
    """
    entries = list(records)
    if len({name for name, _ in entries}) != len(entries):
        raise ValueError("source profiles require unique file names")
    legacy = hashlib.sha256(legacy_prefix)
    encoded = []
    for name, raw in entries:
        if not isinstance(name, str) or not isinstance(raw, bytes):
            raise TypeError("source profiles require UTF-8 names and byte contents")
        name_bytes = name.encode("utf-8")
        legacy.update(name_bytes + b"\0")
        legacy.update(raw)
        legacy.update(b"\0")
        encoded.append((name_bytes, raw))
    framed = hashlib.sha256(PACKAGE_SOURCE_V2.encode("ascii") + b"\0")
    for name_bytes, raw in sorted(encoded, key=lambda entry: entry[0]):
        framed.update(len(name_bytes).to_bytes(8, "big"))
        framed.update(name_bytes)
        framed.update(len(raw).to_bytes(8, "big"))
        framed.update(raw)
    return {legacy_profile: legacy.hexdigest(), PACKAGE_SOURCE_V2: framed.hexdigest()}


def packaged_projection_config(data: bytes, *, name: str = "pyproject.toml") -> dict:
    """The parsed packaging config of the exact committed bytes, or a refusal.

    The caller reads the bytes from its Git object store: the projection is
    derived from the COMMITTED packaging owners, never a mutable worktree
    file -- assume-unchanged and skip-worktree can conceal tracked edits,
    and a dirty config would otherwise project a payload the commit never
    shipped.  The same framing the wheel build reads decides, from the same
    bytes it would have read.
    """
    import tomllib

    try:
        return tomllib.loads(data.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} is not UTF-8: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{name} is not valid TOML: {exc}") from exc


def _find_table(config: dict) -> dict:
    tool = config.get("tool") or {}
    setuptools = tool.get("setuptools") or {}
    packages = setuptools.get("packages") or {}
    find = packages.get("find")
    return find if isinstance(find, dict) else {}


def _package_of(relative: str) -> str:
    """The dotted package a where-root-relative path belongs to."""
    return ".".join(PurePosixPath(relative).parts[:-1])


def shipped_payload_paths(config: dict, tracked: Iterable[str]) -> list[str]:
    """What a wheel built from ``config`` ships, of these tracked paths.

    ``tracked`` are checkout-root-relative POSIX paths (``git ls-tree HEAD``'s
    names) under the ``where`` root; the answer is where-root-relative and
    sorted.  The rule is the packaging owners', restated from the config they
    own, never a second roster: a ``.py`` file ships unless its package is
    excluded by ``tool.setuptools.packages.find`` (fnmatch patterns over
    dotted package names -- ``tessera._dev*`` here, the repository's own
    tooling), an ``include`` list admits only matching packages when present,
    and any other file ships only where ``tool.setuptools.package-data``
    declares a glob that matches it, package-relative.  Everything else --
    an undeclared ``.c``, ``py.typed``, a stray README -- is not a wheel
    payload, and projecting it as one would make a real install refuse.
    """
    find = _find_table(config)
    where = find.get("where", ["."])
    if isinstance(where, str):
        where = [where]
    include = find.get("include") or None
    exclude = find.get("exclude") or []
    package_data = ((config.get("tool") or {}).get("setuptools") or {}
                    ).get("package-data") or {}
    shipped = []
    for path in tracked:
        for root in where:
            prefix = root.rstrip("/") + "/"
            if not path.startswith(prefix):
                continue
            relative = path[len(prefix):]
            is_module = relative.endswith(".py")
            owner = _package_of(relative)
            if include is not None and not any(fnmatch.fnmatch(owner, pattern)
                                               for pattern in include):
                continue
            if any(fnmatch.fnmatch(owner, pattern) for pattern in exclude):
                continue
            if is_module:
                shipped.append(relative)
                break
            # A data file ships where a ``tool.setuptools.package-data`` KEY's
            # directory prefixes it and a glob from that directory matches --
            # setuptools' own attribution, which is why a ``csrc`` subdirectory
            # is not a package of its own here.
            for key, patterns in package_data.items():
                prefix_dir = "" if key == "*" else key.replace(".", "/")
                if prefix_dir and not relative.startswith(prefix_dir + "/"):
                    continue
                within = relative[len(prefix_dir) + 1:] if prefix_dir else relative
                if any(PurePosixPath(within).match(pattern) for pattern in (patterns or [])):
                    shipped.append(relative)
                    break
            break
    return sorted(shipped)


def shipped_payload_profiles(files: "dict[str, bytes]") -> "dict[str, str]":
    """The labelled source profiles of one projected install payload.

    ``files`` maps package-relative POSIX paths to their verified bytes (the
    caller proves they are the checkout's committed bytes).  Both sides of an
    installed-vs-expected comparison frame their records through here, so the
    digests are comparable by construction; the v2 profile is the one a
    receipt binds.
    """
    return source_profiles(sorted(files.items()), legacy_profile=PACKAGE_SOURCE_V2)
