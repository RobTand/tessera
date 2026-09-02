"""Strictly parsed, process-stable environment flags for the serving plugin.

One rule, said once: a serving flag is parsed strictly (a typo raises and
names the accepted spellings) and *latched* to the first value the process
observed (a later change raises rather than mixing two dispatch behaviours
inside one run).  Both halves are load-bearing.  A flag that silently ignores
``true`` or reads ``off`` as ON turns an intended A/B into an unlabelled run
of the other arm, which is worse than a crash because the numbers look fine;
and a residency mode that moved between two forwards would make the run's
numbers describe neither setting.

This is Gridbook's ``lane_select`` contract, reduced to what the Tessera
plugin needs and owned here so the plugin depends on no other runtime.
"""
from __future__ import annotations

import os

__all__ = ["latched_bool", "latched_mode", "reset_for_tests"]

#: flag name -> the raw string this process latched.  One dict rather than a
#: module-level variable per flag, so ``reset_for_tests`` cannot miss one.
_LATCHED: dict[str, str] = {}

_TRUE = ("1",)
_FALSE = ("", "0")


def reset_for_tests(flag: str | None = None) -> None:
    """Clear one latch, or all of them (tests only)."""
    if flag is None:
        _LATCHED.clear()
    else:
        _LATCHED.pop(flag, None)


def _latch(flag: str, current: str) -> str:
    """Pin ``flag`` to its first observed value; raise if it changes."""
    if flag not in _LATCHED:
        _LATCHED[flag] = current
    elif current != _LATCHED[flag]:
        raise RuntimeError(
            f"{flag} changed after Tessera dispatch was fixed (was "
            f"{_LATCHED[flag]!r}, now {current!r}); restart the process instead of "
            "changing serving behaviour within one run")
    return _LATCHED[flag]


def latched_bool(flag: str, *, default: bool = False, meaning: str = "this behaviour") -> bool:
    """A strictly parsed, process-stable on/off flag.

    Accepts only ``''`` (unset), ``'0'`` and ``'1'``, surrounding whitespace
    stripped.  ``default`` is what an UNSET flag means; ``'0'`` and ``'1'``
    always mean exactly themselves, so an opt-out flag is spelled
    ``default=True`` rather than by inverting the comparison at the call site.
    """
    current = os.environ.get(flag, "").strip()
    if current not in _TRUE + _FALSE:
        raise ValueError(
            f"invalid {flag}={current!r}; expected '1' to enable {meaning}, '0' to "
            f"disable it, or leave it unset for the default "
            f"({'enabled' if default else 'disabled'})")
    value = _latch(flag, current)
    if value == "":
        return default
    return value in _TRUE


def latched_mode(flag: str, *, modes: tuple[str, ...], meaning: str, unset_help: str) -> str:
    """A strictly parsed, process-stable multi-way selector with NO default.

    An unset flag is an error carrying ``unset_help``: this selector exists
    for choices the plugin will not make on the operator's behalf.
    """
    current = os.environ.get(flag, "").strip().lower()
    if current not in modes:
        raise ValueError(
            f"{flag} must be one of {modes} to select {meaning}, got {current!r}. "
            f"{unset_help}")
    return _latch(flag, current)
