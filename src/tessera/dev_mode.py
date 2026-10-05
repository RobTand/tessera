"""``PRISMAQUANT_DEV_MODE`` in Tessera: sealing is off unless a run asks for it.

CEO decision D32 (restating Rob, 2026-09-24): "All sealing should be disabled
until further notice." A run-identity seal -- a recorded source, commit,
digest, producer, pin or provenance identity compared with the running one --
stamps and continues in dev mode instead of refusing. This module is the one
home for that switch in this package, mirroring ``prismaquant.dev_mode``
(PQ #1147) so both sides of the repository boundary share one default: dev
mode is ON unless ``PRISMAQUANT_DEV_MODE`` is exactly ``0``.

The contract:

* Dev mode is ON unless ``PRISMAQUANT_DEV_MODE`` is exactly ``0``. Unset,
  empty, ``1`` and any other value are dev mode. ``0`` is certified mode, and
  certified mode behaves exactly as the gates did before dev mode: every seal
  refuses with the same exception type and message the site raised.
* Every run-identity comparison goes through :func:`seal_check`. On a mismatch
  in dev mode it prints one ``[DEV-MODE]`` line naming both values and returns
  ``False``; the caller continues with the stored data. It never archives and
  never recomputes.
* Dev mode computes no digest over existing data only to satisfy an identity
  comparison. A site that would compute one passes :data:`NOT_COMPUTED` as the
  ``actual`` side instead; the stamp line names it. A digest written beside new
  bytes, and the check that reads those bytes back against it, is integrity,
  not sealing, and refuses in both modes: a missing file, a corrupt file and
  bytes that do not match their own digest still refuse.
* Certified mode is only for an artifact Rob explicitly returns the seal for.
  No caller forces ``0``.

Only stdlib is imported here, so any Tessera entry point can take the switch
without torch or the serving stack.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from typing import Any

__all__ = [
    "DEV_MODE_ENV",
    "NOT_COMPUTED",
    "dev_mode_enabled",
    "dev_warning",
    "seal_check",
]

#: The one environment variable. Read at the gates, never cached, so a
#: subprocess or a container inherits it through the environment alone.
DEV_MODE_ENV = "PRISMAQUANT_DEV_MODE"

#: The ``actual`` side a dev-mode site passes when satisfying the comparison
#: would require computing a digest over existing data (PQ #1147). The
#: ``[DEV-MODE]`` line names it; certified mode never passes it, and a
#: certified check that meets it refuses, because it differs from every
#: recorded identity.
NOT_COMPUTED = "not computed"


def dev_mode_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether dev mode is ON: ``PRISMAQUANT_DEV_MODE`` is not exactly ``0``."""
    environ = os.environ if environ is None else environ
    return environ.get(DEV_MODE_ENV, "") != "0"


def dev_warning(message: str) -> None:
    """The loud line every suspended gate prints instead of refusing.

    One stable ``[DEV-MODE]`` prefix so a dev run's suspended gates are as
    grep-able as its stamps: ``grep DEV-MODE`` finds every place a certified
    run would have stopped.
    """
    print(f"[DEV-MODE] {message}", flush=True)


#: Longest rendering of one side of a mismatch in a ``[DEV-MODE]`` line.
_SHOWN_CHARS = 160


def _shown(value: Any) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= _SHOWN_CHARS:
        return text
    try:
        encoded = json.dumps(value, sort_keys=True, default=repr).encode()
    except (TypeError, ValueError):
        encoded = text.encode()
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return f"{text[:_SHOWN_CHARS]}... ({len(text)} chars, sha256 {digest}...)"


def _first_difference(expected: Any, actual: Any, path: str = "") -> tuple[str, Any, Any]:
    """The first field two identities disagree on, for the stamp line."""
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key in sorted(set(expected) | set(actual), key=str):
            left, right = expected.get(key), actual.get(key)
            if key not in expected or key not in actual or left != right:
                return _first_difference(left, right, f"{path}.{key}" if path else str(key))
    return path, expected, actual


def _refusal(refusal, kind: str, expected: Any, actual: Any, where: str) -> BaseException:
    message = f"{where}: {kind} differs: expected {expected!r}, actual {actual!r}"
    if refusal is None:
        return RuntimeError(message)
    if isinstance(refusal, BaseException):
        return refusal
    if isinstance(refusal, type) and issubclass(refusal, BaseException):
        return refusal(message)
    return refusal()


def seal_check(kind: str, expected: Any, actual: Any, *, where: str,
               refusal: BaseException | type | Callable[[], BaseException] | None = None,
               same: bool | None = None, environ: Mapping[str, str] | None = None) -> bool:
    """Compare a recorded identity with the running one: the one seal helper.

    Returns ``True`` when the two agree. On a mismatch, certified mode
    (``PRISMAQUANT_DEV_MODE=0``) raises ``refusal`` -- the exception the site
    raised before dev mode, so its type and message are unchanged -- and dev
    mode prints one ``[DEV-MODE]`` line naming both values and returns
    ``False``. The caller then continues with the stored data.

    ``refusal`` is an exception instance, an exception class (called with a
    generated message) or a zero-argument callable that returns one. ``same``
    replaces ``expected == actual`` for a site whose comparison is not plain
    equality (canonical bytes, a set, a combined condition).

    Only seals go through here. A check of bytes against the digest written
    with them is integrity and raises in both modes.
    """
    agree = (expected == actual) if same is None else bool(same)
    if agree:
        return True
    if not dev_mode_enabled(environ):
        raise _refusal(refusal, kind, expected, actual, where)
    field, left, right = _first_difference(expected, actual)
    named = f" at {field}" if field else ""
    if isinstance(right, str) and right == NOT_COMPUTED:
        dev_warning(
            f"seal {kind}{named} not computed ({where}): recorded {_shown(left)}; "
            "sealing is off (D32), continuing with the stored data")
        return False
    dev_warning(
        f"seal {kind} differs{named} ({where}): expected {_shown(left)}, "
        f"actual {_shown(right)}; sealing is off (D32), continuing with "
        "the stored data")
    return False
