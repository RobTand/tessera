"""Rebuild one method from source text, with one block of its body replaced.

A serve-time override that changes a few lines inside a stock vLLM method
(``glm53_prefill``'s KDA conv split) recompiles that method from the stock
module's own source.  The caller reads the source, after its digest check has
matched the bytes to an inspected interface.  This module turns that text
into a function and reads no file itself.

The read and rebuild remain separate responsibilities. The #808 draft
selector follows recognized calls across this boundary: splitting the work
alone no longer proves that an unknown read cannot execute Python. Production
callers digest-check inspected vLLM source, but the generic module parameter
also accepts runtime-created modules in tests. The selector cannot yet prove
external origin at every caller, so this path conservatively forces the shared
conftest's population until that separate selection contract is resolved.
"""
from __future__ import annotations

import ast
import textwrap
from typing import Any, Callable


def rebuild_method(src: str, namespace: dict[str, Any], cls_name: str, method_name: str,
                   decorator: str, old_block: str, new_block: str, filename: str, *,
                   block_name: str = "the replaced block") -> tuple[Callable | None, str]:
    """``cls_name.method_name`` from ``src`` with ``old_block`` replaced by ``new_block``.

    The method must carry exactly the one decorator ``decorator`` (by name), and
    ``old_block`` must occur in it exactly once.  The edited method is dedented,
    compiled as ``filename`` and executed in ``namespace``.  Returns
    ``(function, "")``, the function undecorated, or ``(None, why)``.
    """
    tree = ast.parse(src)
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name), None)
    fn = cls and next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name), None)
    if fn is None:
        return None, f"{cls_name}.{method_name} not found"
    decorators = [d.id if isinstance(d, ast.Name) else ast.dump(d) for d in fn.decorator_list]
    if decorators != [decorator]:
        return None, f"{method_name} decorators {decorators} are not [{decorator}]"
    lines = src.splitlines(keepends=True)
    method = "".join(lines[fn.lineno - 1:fn.end_lineno])  # the def line through the body
    n = method.count(old_block)
    if n != 1:
        return None, f"{block_name} occurs {n} times in {method_name}"
    method = textwrap.dedent(method.replace(old_block, new_block))
    try:
        exec(compile(method, filename, "exec"), namespace)
    except Exception as exc:  # noqa: BLE001 - any failure is a decline
        return None, f"recompiling {method_name} failed ({type(exc).__name__}: {exc})"
    return namespace[method_name], ""
