"""Source and signature facts for inspected stock overrides; no installation policy.

Each lever supplies its own immutable pins and expected parameters. Matching
preserves complete inspected digest tuples instead of mixing module pins.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InspectedInterface:
    name: str
    digests: tuple[str, ...]


INSPECTION_ERRORS=(OSError,AttributeError,TypeError,ValueError)


def stock_attribute(owner, name, default=None):
    """Unavailable inspected attributes decline; installation policy stays local."""
    try:
        return getattr(owner,name,default)
    except INSPECTION_ERRORS:
        return default


def source_digest(module: Any) -> str | None:
    """An unreadable source is a non-match, never an install exception."""
    try:
        path=getattr(module,'__file__',None)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
    except INSPECTION_ERRORS:
        return None


# Prefill's existing name is an alias to the same observation used by all levers.
module_digest=source_digest


def match_modules(modules, names, interfaces):
    actual=tuple(module_digest(module) for module in modules)
    for interface in interfaces:
        if actual==interface.digests:
            return interface,''
    detail=', '.join(f'{name}={digest}' for name,digest in zip(names,actual))
    return None,f'no inspected interface matches ({detail})'


def import_modules(names):
    modules=[]
    for name in names:
        try:
            modules.append(importlib.import_module(name))
        except Exception as exc:
            return None,f'{name} not importable ({type(exc).__name__}: {exc})'
    return tuple(modules),''


def signature_parameters(owner, name):
    try:
        return tuple(inspect.signature(getattr(owner,name)).parameters)
    except INSPECTION_ERRORS:
        return None
