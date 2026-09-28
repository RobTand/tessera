"""The ``--producer-authority`` option every export driver shares (tessera#599).

A rooted cached-unit bundle (``tessera.cached_units.v2``) and a Hessian
reference document bind records a PRODUCER writes, so the producer ships their
reader: a self-contained Python file defining ``PRODUCER_AUTHORITY``, a
``tessera.cached_unit.ReuseAuthority``, whose optional
``canonical_hessian_capture`` attribute is the producer's ``(schema, source)``
pair for the calibration cache a reference binds.  Tessera names none of those
records.

This module is the one place a driver learns that file: :func:`add_argument`
declares the option with one help text, and :func:`load` reads it with one set
of refusals, so every driver that takes the option takes it with the same
semantics and the same words.  The drivers that do are published, as data, in
the packaged runtime contract's ``producer_interface.reuse_authority.drivers``;
``tests/test_producer_authority_drivers.py`` derives that list from the tree
and refuses a contract that disagrees.

The file is loaded by path, not imported by name: the producer's package need
not be importable in the process that runs the driver, and a producer's own
checkout is where its reader lives.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

from tessera.serving.contract import REUSE_AUTHORITY_ATTRIBUTE, REUSE_AUTHORITY_OPTION

#: The option's spelling, as the packaged contract publishes it.  Owned by
#: :mod:`tessera.serving.contract`, which must not import this module: this
#: one executes a file by path, and the contract is on every test's import
#: path, so the dependency points this way.
OPTION = REUSE_AUTHORITY_OPTION

#: The module attribute a producer authority file must define.
ATTRIBUTE = REUSE_AUTHORITY_ATTRIBUTE

HELP = ("the producer's authority file (a module defining PRODUCER_AUTHORITY, "
        "a tessera.cached_unit.ReuseAuthority). Rooted --cached-units bundles "
        "and --hessian reference documents bind producer records; without it "
        "they refuse by name")


def add_argument(parser, *, help: str = HELP) -> None:
    """Declare ``--producer-authority`` on a driver's parser."""
    parser.add_argument(OPTION, type=Path, default=None, help=help)


def load(path):
    """``(authority, canonical_capture)`` from a producer's authority file.

    ``path`` must name an absolute regular file.  The file is executed once
    per content digest (a second driver in one process reuses the module) and
    must define ``PRODUCER_AUTHORITY`` as a ``ReuseAuthority``;
    ``canonical_capture`` is its ``canonical_hessian_capture`` normalized, or
    ``None`` when it defines none.  The path and digest are printed, so a run's
    log names the reader that judged its producer records.
    """
    from tessera.cached_unit import ReuseAuthority
    from tessera.hessian_capture import normalize_canonical_capture
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise SystemExit(f"{OPTION} must name an absolute regular file: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    name = "tessera_producer_authority_" + digest
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    authority = getattr(module, ATTRIBUTE, None)
    if not isinstance(authority, ReuseAuthority):
        raise SystemExit(f"{OPTION} {path} defines no ReuseAuthority {ATTRIBUTE}")
    canonical = normalize_canonical_capture(getattr(authority, "canonical_hessian_capture", None))
    print(f"producer authority: {path} sha256={digest}", flush=True)
    return authority, canonical


def canonical_capture(path):
    """The canonical capture a driver passes to ``from_capture``: ``None`` without a file."""
    return None if path is None else load(path)[1]
