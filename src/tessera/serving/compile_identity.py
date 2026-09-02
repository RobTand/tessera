"""Fold the plugin's runtime identity into vLLM's compile-cache key.

WHY THIS EXISTS.  vLLM keys every ``torch.compile`` artifact it caches -- the
per-model backbone directory and, under ``VLLM_USE_AOT_COMPILE`` (the 0.28
default), the AOT-compiled forward -- by ``VllmConfig.compute_hash()`` plus the
contents of the source files Dynamo traced.  Neither sees a residency mode:
``TESSERA_SERVE_MODE=resident`` and ``=streamed`` trace the same files into
different forwards (one reads the tile the loader decoded, the other decodes
the wire every forward), so both modes would land on one AOT key.  The second
process to start then loads the first's function with guards disabled (vLLM
verifies file contents, not the trace) and its bytecode dereferences attributes
the running mode never set.  Measured 2026-09-02 on vLLM 0.28.0 with the same
routes under Gridbook's flag pair: a resident route census after a streamed one
in the same cache died at the first forward on ``'NoneType' object has no
attribute '_PreparedTesseraModule__roles'``.  Loud, but a supported
configuration must not crash on the operator's default ``~/.cache/vllm``.

``VllmConfig.additional_config`` is the one input to that hash a plugin can
reach: vLLM hashes it by JSON content.  The Tessera config declares its mode
there from ``get_quant_method``, which runs during model construction -- after
the config is current, before anything computes a hash for a cache key (the
backbone key at the first compile, the AOT key at the first forward, the
worker's startup-plan fingerprint after load).  Two modes then key two caches,
and a plugin version bump re-keys both at once.  vLLM's own traced-file
checksum still covers code changes within one mode.
"""
from __future__ import annotations

from typing import Any

__all__ = ["TESSERA_KEY", "declare_compile_identity", "declare_compile_identity_in"]

TESSERA_KEY = "tessera"


def _plugin_version() -> str:
    from . import __version__

    return __version__


def _forward_is_compiled(config: Any) -> bool:
    """True unless vLLM's compilation mode is NONE (``--enforce-eager``)."""
    compilation = getattr(config, "compilation_config", None)
    mode = getattr(compilation, "mode", None)
    name = getattr(mode, "name", None)
    return name is not None and name != "NONE"


def declare_compile_identity_in(config: Any, **facts: str) -> dict | None:
    """Record ``facts`` under ``config.additional_config["tessera"]``.

    Returns the record.  Returns None, declaring nothing, when the config's
    ``additional_config`` is not a dict and the forward is not compiled (there
    is no cache key to protect).  Refuses a non-dict ``additional_config``
    under a compiled forward, and a fact that contradicts one declared earlier
    in this process: one process serves one residency mode.
    """
    extra = getattr(config, "additional_config", None)
    if not isinstance(extra, dict):
        if _forward_is_compiled(config):
            raise RuntimeError(
                f"cannot fold the Tessera serving identity {facts!r} into vLLM's "
                f"compile-cache key: additional_config is a {type(extra).__name__}, not "
                "a dict, and the forward is compiled, so two residency modes would share "
                "one cached forward; serve with --enforce-eager or pass a dict "
                "--additional-config")
        return None
    record = extra.setdefault(TESSERA_KEY, {})
    if not isinstance(record, dict):
        raise RuntimeError(
            f"additional_config[{TESSERA_KEY!r}] is a {type(record).__name__}, not a "
            "dict; the Tessera plugin owns that key for its compile-cache identity")
    record.setdefault("version", _plugin_version())
    for key, value in facts.items():
        prior = record.get(key)
        if prior is not None and prior != value:
            raise RuntimeError(
                f"Tessera serving identity {key}={value!r} contradicts {key}={prior!r} "
                "declared earlier in this process; one process serves one residency mode")
        record[key] = value
    return record


def declare_compile_identity(**facts: str) -> dict | None:
    """``declare_compile_identity_in`` on vLLM's current config.

    None when vLLM is absent or no config is current (methods built bare in
    tests): there is no compile cache to key.
    """
    try:
        from vllm import config as vllm_config_module
    except ImportError:
        return None
    getter = getattr(vllm_config_module, "get_current_vllm_config_or_none", None)
    if getter is not None:
        config = getter()
    else:  # older vLLM: the getter asserts when nothing is current
        try:
            config = vllm_config_module.get_current_vllm_config()
        except AssertionError:
            config = None
    if config is None:
        return None
    return declare_compile_identity_in(config, **facts)
