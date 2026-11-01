"""Verbatim upstream dependency excerpts; see README.md for provenance."""


from __future__ import annotations



from collections.abc import Callable, Mapping, Sequence
import re
from typing import NoReturn


class PrismaBuildError(RuntimeError):
    """Base class for PrismaBuild core failures."""


class ActionContractError(PrismaBuildError, ValueError):
    """An action, closure, receipt, or worker identity is not exact."""


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _fail(message: str) -> NoReturn:
    raise ActionContractError(message)


def _exact_mapping(
    value: object, *, keys: frozenset[str], where: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(f"{where} must be an object")
    if any(type(key) is not str for key in value):
        _fail(f"{where} keys must be strings")
    actual = set(value)
    if actual != set(keys):
        _fail(
            f"{where} fields differ: missing={sorted(set(keys) - actual)}, "
            f"extra={sorted(actual - set(keys))}"
        )
    return value


def _text(
    value: object,
    *,
    where: str,
    pattern: re.Pattern[str] | None = None,
    allow_empty: bool = False,
    allow_control: bool = False,
    fail: Callable[[str], NoReturn] = _fail,
) -> str:
    """Validate one text field.

    Identifiers, keys, digests, paths and tokens refuse every control
    character: nothing that names a thing may carry bytes a log line or a
    filename cannot show.  Payload text -- an argv element, an environment
    value, a params string value -- is what ``execve`` and JSON carry
    verbatim, and its only illegal byte is NUL.  A ``bash -lc`` script with a
    newline in it is an ordinary argument, not a contract violation (issue
    #21), so payload callers pass ``allow_control=True``.
    """

    if type(value) is not str or (not value and not allow_empty):
        fail(f"{where} must be a {'string' if allow_empty else 'non-empty string'}")
    if "\x00" in value:
        fail(f"{where} contains a NUL character")
    if not allow_control and any(ord(char) < 32 for char in value):
        fail(f"{where} contains a NUL or control character")
    if pattern is not None and pattern.fullmatch(value) is None:
        fail(f"{where} has an invalid value")
    return value


def _nonnegative_integer(
    value: object, *, where: str, fail: Callable[[str], NoReturn] = _fail
) -> int:
    if type(value) is not int or value < 0:
        fail(f"{where} must be a non-negative integer")
    return value


def _sha256(
    value: object, *, where: str, fail: Callable[[str], NoReturn] = _fail
) -> str:
    return _text(value, where=where, pattern=_SHA256_RE, fail=fail)
