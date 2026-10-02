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


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None


def source_digest(module: Any) -> str | None:
    """The existing raw source observation; the caller owns its refusal policy."""
    return _file_digest(getattr(module, '__file__', None))


def module_digest(module: Any) -> str | None:
    """The existing prefill source observer's OSError-to-nonmatch behavior."""
    path=getattr(module,'__file__',None)
    try:
        return _file_digest(path)
    except OSError:
        return None


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
    return tuple(inspect.signature(getattr(owner,name)).parameters)
