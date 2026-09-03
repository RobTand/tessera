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

ONE LEVEL DOWN: THE LANE (issue #91).  ``serve_mode`` separates the two
residencies, and for a while that was read as separating the graphs.  It does
not.  Inside ``streamed`` the FP8 and BF16 routes branch on
``getattr(layer, "tessera_gemv", None)`` -- a PYTHON ATTRIBUTE Dynamo resolves
at trace time -- so a module whose window-GEMV holder prepared traces one
opaque ``tessera::fp8_streamed_apply`` node and a module whose holder did not
traces a window decode plus ``torch._scaled_mm``.  Different nodes, different
kernels, different numerics.  The refusals that produce the second state are
per unit and reachable on one box: a rate-3 wire, ``L != 14``, a shard start
state, or a cold toolchain (``prepare_fp8_gemv``).  Both states are
``serve_mode == "streamed"``, and -- this is what makes it silent -- the
traced SOURCE FILES are byte-identical in both, because the branch is data,
not code.  vLLM 0.28 hashes the AOT key from the env compile factors,
``VllmConfig.compute_hash()`` and the forward's qualname BEFORE Dynamo runs
(``compilation/decorators.py:537-552``, whose own comment says it "contains
all of the factors except for the source files being traced through");
the sources are checked only on LOAD, and against the file list the SAVED
artifact carries (``_verify_source_unchanged``), not the list the loading
process would have traced.  So the second run finds the first run's artifact
under its own key, checks the files the first run traced, finds them
unchanged, and replays the wrong graph.

``note_traced_dispatch`` closes that.  Every Tessera module reports, at
``process_weights_after_loading``, the OP its forward will dispatch through --
which is exactly the question ``fp8_gemv.census_expected`` answers for the
census, asked one stage earlier.  The declared fact is a digest over the
sorted ``(module, op)`` pairs, PER MODULE and not a per-checkpoint boolean,
because the refusals are per unit: two runs of one checkpoint can put the same
NUMBER of modules on the GEMV lane and a different SET of them, and a boolean
(or a count) calls those two runs the same graph.  A digest calls them apart.

Weight loading is early enough.  ``gpu_worker.py`` runs ``load_model`` (450),
then ``determine_available_memory`` (475), then ``maybe_apply_startup_plan``
(487, the first ``compute_hash`` in the worker), then
``compile_or_warm_up_model`` (694, where the backbone dir and the AOT key are
computed); ``process_weights_after_loading`` is inside the first of those, and
``VllmConfig.compute_hash`` memoises nothing.  What is NOT guaranteed is that
a vLLM config is CURRENT that late -- ``set_current_vllm_config`` wraps model
construction -- so the record declared from ``get_quant_method`` is REMEMBERED
here and written into directly, rather than looked up again and silently
dropped.
"""
from __future__ import annotations

import collections
import hashlib
from typing import Any

__all__ = ["TESSERA_KEY", "DISPATCH_FACT", "declare_compile_identity",
           "declare_compile_identity_in", "note_traced_dispatch",
           "traced_dispatch", "reset_for_tests"]

TESSERA_KEY = "tessera"

#: The fact ``note_traced_dispatch`` maintains: a histogram of the ops the
#: traced modules dispatch through plus a digest of the per-module pairs.
#: Named rather than spelt inline because the tests read it.
DISPATCH_FACT = "traced_dispatch"

#: The record ``declare_compile_identity_in`` last wrote, and the per-module
#: dispatch accumulated into it.  Process-global for the same reason the
#: contradiction check below is: one process serves one model.  Keyed off the
#: RECORD object so that declaring into a second config (a test, never a
#: serve) starts a fresh accumulation instead of inheriting the first's.
_STATE: dict[str, Any] = {"record": None, "dispatch": {}}


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
    # Remember the record the late per-module facts are written into, and start
    # a fresh accumulation when it is a different record: one process serves one
    # model, so a second record is a test, not a second lane of one serve.
    if _STATE["record"] is not record:
        _STATE["record"] = record
        _STATE["dispatch"] = {}
    elif _STATE["dispatch"]:
        record[DISPATCH_FACT] = _render_dispatch(_STATE["dispatch"])
    return record


def _render_dispatch(dispatch: dict) -> str:
    """``{module: op}`` -> the fact, ``<op>=<n>[+<op>=<n>...]:<digest>``.

    The histogram is for the reader (``additional_config`` is echoed in vLLM's
    engine-init line); the digest is what makes the fact exact.  It is a
    ``hashlib`` digest of the SORTED pairs, never ``hash()``: the value has to
    be the same string in the next process, and PYTHONHASHSEED is not.  Sorted
    also means the order modules happen to be loaded in cannot change the key.
    """
    pairs = "\n".join(f"{name}={op}" for name, op in sorted(dispatch.items()))
    digest = hashlib.sha256(pairs.encode()).hexdigest()[:16]
    counts = collections.Counter(dispatch.values())
    histogram = "+".join(f"{op}={n}" for op, n in sorted(counts.items()))
    # ``#`` separates the two halves because an op name contains ``::``.
    return f"{histogram}#{digest}"


def note_traced_dispatch(prefix: str, op: str) -> str | None:
    """Record that module ``prefix``'s forward will trace a call to ``op``.

    Called once per module from ``process_weights_after_loading``, where the
    lane is decided -- the earliest point at which the answer is known, and
    still inside ``load_model``, which is before every ``compute_hash`` the
    worker computes (see the module docstring).  Returns the fact's new value,
    or None when nothing was declared (vLLM absent, or an uncompiled forward
    over an ``additional_config`` this plugin may not extend): there is no
    cache key to protect and no record to write into.

    ``op`` is the operator the branch dispatches through, not a description of
    it -- ``tessera::fp8_streamed_apply`` for the window-GEMV lane, the route's
    own ``gemm_symbol`` for the materialised one -- because the object the key
    must separate is the node the graph contains.
    """
    record = _STATE["record"]
    if record is None:
        return None
    _STATE["dispatch"][str(prefix)] = str(op)
    value = _render_dispatch(_STATE["dispatch"])
    record[DISPATCH_FACT] = value
    return value


def traced_dispatch() -> dict:
    """The per-module dispatch accumulated so far (a copy), for tests."""
    return dict(_STATE["dispatch"])


def reset_for_tests() -> None:
    """Forget the remembered record and its dispatch.

    Only tests need this: a serve builds one model in one process.  Named the
    way ``ext.reset_for_tests`` is, for the same reason.
    """
    _STATE["record"] = None
    _STATE["dispatch"] = {}


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
