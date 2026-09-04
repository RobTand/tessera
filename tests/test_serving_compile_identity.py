"""The serving identity vLLM's compile cache is keyed by.

``tessera/serving/compile_identity.py`` carries the why and the measurement.
Here: the record the config writes, its idempotence per process, the two
refusals, and -- with vLLM importable -- that two residency modes make two
``VllmConfig`` hashes, which is the property every vLLM compile-cache key
inherits.

Ported from Gridbook's ``test_compile_identity.py``.  The key is ``"tessera"``
and the declared fact is ``serve_mode``; Gridbook declared three lane modes
under one key, this plugin declares one, so the multi-lane tests are gone
(see the report).  WHO declares also moved: Gridbook's lane BUILDERS declared,
here ``config.TesseraConfig.get_quant_method`` does, and that is pinned in
``test_serving_dispatch.py`` where the vLLM stubs live.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

import tessera.serving
from tessera.serving.compile_identity import (
    DISPATCH_FACT, TESSERA_KEY, declare_compile_identity, declare_compile_identity_in,
    note_traced_dispatch, reset_for_tests, traced_dispatch)


@pytest.fixture(autouse=True)
def _forget_the_remembered_record():
    """One process serves one model; a test file is many.

    ``declare_compile_identity_in`` remembers the record it wrote so the
    per-module facts that arrive later (at weight load, outside
    ``set_current_vllm_config``) have somewhere to go.  Tests declare into a
    dozen configs, so each one starts from nothing.
    """
    reset_for_tests()
    yield
    reset_for_tests()


def _config(mode="VLLM_COMPILE", extra=None):
    return SimpleNamespace(
        additional_config={} if extra is None else extra,
        compilation_config=SimpleNamespace(mode=SimpleNamespace(name=mode)))


def test_declares_mode_and_release_under_the_tessera_key():
    cfg = _config()
    rec = declare_compile_identity_in(cfg, serve_mode="streamed")
    assert cfg.additional_config[TESSERA_KEY] is rec
    assert rec == {"version": tessera.serving.__version__, "serve_mode": "streamed"}


def test_two_modes_differ_by_content_and_one_process_serves_one_mode():
    a, b = _config(), _config()
    declare_compile_identity_in(a, serve_mode="resident")
    declare_compile_identity_in(b, serve_mode="streamed")
    # vLLM hashes additional_config as json.dumps(..., sort_keys=True)
    assert (json.dumps(a.additional_config, sort_keys=True)
            != json.dumps(b.additional_config, sort_keys=True))
    # a config that declares twice is a no-op the second time
    declare_compile_identity_in(a, serve_mode="resident")
    with pytest.raises(RuntimeError, match="contradicts"):
        declare_compile_identity_in(a, serve_mode="streamed")


def test_operator_additional_config_is_extended_not_replaced():
    cfg = _config(extra={"theirs": 1})
    declare_compile_identity_in(cfg, serve_mode="resident")
    assert cfg.additional_config["theirs"] == 1
    assert cfg.additional_config[TESSERA_KEY]["serve_mode"] == "resident"


def test_unextendable_additional_config_refuses_only_under_a_compiled_forward():
    class Opaque:
        def compute_hash(self):
            return "x"

    with pytest.raises(RuntimeError, match="not a dict"):
        declare_compile_identity_in(_config(extra=Opaque()), serve_mode="resident")
    assert declare_compile_identity_in(
        _config(mode="NONE", extra=Opaque()), serve_mode="resident") is None


def test_a_foreign_tessera_key_is_refused():
    cfg = _config(extra={TESSERA_KEY: "theirs"})
    with pytest.raises(RuntimeError, match="owns that key"):
        declare_compile_identity_in(cfg, serve_mode="resident")


def test_no_current_config_declares_nothing():
    # vLLM absent: ImportError; vLLM present but no set_current_vllm_config: None
    assert declare_compile_identity(serve_mode="resident") is None


def test_vllm_hashes_the_two_modes_apart():
    pytest.importorskip("vllm")
    from vllm.config import VllmConfig, set_current_vllm_config

    hashes = {}
    for mode in ("resident", "streamed"):
        cfg = VllmConfig()
        with set_current_vllm_config(cfg):
            rec = declare_compile_identity(serve_mode=mode)
        assert rec is cfg.additional_config[TESSERA_KEY]
        assert rec["serve_mode"] == mode
        hashes[mode] = cfg.compute_hash()
    assert hashes["resident"] != hashes["streamed"]
    again = VllmConfig()
    with set_current_vllm_config(again):
        declare_compile_identity(serve_mode="resident")
    assert again.compute_hash() == hashes["resident"]


# ---------------------------------------------------------------------------
# The lane, one level below the mode (issue #91).  Two streamed serves of one
# checkpoint take different traced graphs -- ``tessera::fp8_streamed_apply`` on
# the window-GEMV lane, a window decode plus ``torch._scaled_mm`` without it --
# over byte-identical sources.  Everything below asserts the fact that
# separates them is in the record vLLM hashes.

GEMV_OP = "tessera::fp8_streamed_apply"
GEMM_OP = "torch._scaled_mm"
MODULES = ("model.layers.0.mlp.down_proj", "model.layers.0.self_attn.qkv_proj",
           "model.layers.1.mlp.down_proj")


def _declared(mode="streamed"):
    cfg = _config()
    declare_compile_identity_in(cfg, serve_mode=mode)
    return cfg


def _identity(cfg):
    """What vLLM hashes: ``additional_config`` as JSON, sort_keys (config/vllm.py:520)."""
    return json.dumps(cfg.additional_config, sort_keys=True)


def test_the_two_lane_states_are_two_identities():
    gemv, torch_window = _declared(), None
    for name in MODULES:
        note_traced_dispatch(name, GEMV_OP)
    a = _identity(gemv)
    reset_for_tests()
    torch_window = _declared()
    for name in MODULES:
        note_traced_dispatch(name, GEMM_OP)
    b = _identity(torch_window)
    assert gemv.additional_config[TESSERA_KEY]["serve_mode"] == "streamed"
    assert torch_window.additional_config[TESSERA_KEY]["serve_mode"] == "streamed"
    assert a != b, "two streamed graphs, one identity: the compile cache would share a slot"


def test_one_lane_state_is_one_identity_however_the_modules_are_ordered():
    first = _declared()
    for name in MODULES:
        note_traced_dispatch(name, GEMV_OP)
    a = _identity(first)
    reset_for_tests()
    second = _declared()
    for name in reversed(MODULES):   # load order is not the key
        note_traced_dispatch(name, GEMV_OP)
    assert _identity(second) == a
    # and a module that reports twice (a re-processed layer) is one entry
    note_traced_dispatch(MODULES[0], GEMV_OP)
    assert _identity(second) == a


def test_a_mixed_checkpoint_needs_the_SET_not_a_count():
    """Why the fact is per module: the refusals (rate-3, ``L != 14``, a shard
    start state) are per unit, so two runs can put the same NUMBER of modules
    on the GEMV lane and a different SET of them.  A boolean or a count calls
    those two runs one graph; they are not one graph."""
    one = _declared()
    note_traced_dispatch(MODULES[0], GEMV_OP)
    note_traced_dispatch(MODULES[1], GEMM_OP)
    a = _identity(one)
    reset_for_tests()
    other = _declared()
    note_traced_dispatch(MODULES[0], GEMM_OP)
    note_traced_dispatch(MODULES[1], GEMV_OP)
    b = _identity(other)
    counts = [rec[DISPATCH_FACT].rpartition("#")[0]
              for rec in (one.additional_config[TESSERA_KEY],
                          other.additional_config[TESSERA_KEY])]
    assert counts[0] == counts[1], "the histogram is equal: only the digest can separate these"
    assert a != b


def test_the_fact_is_a_stable_digest_not_a_python_hash():
    cfg = _declared()
    note_traced_dispatch(MODULES[0], GEMV_OP)
    value = cfg.additional_config[TESSERA_KEY][DISPATCH_FACT]
    op, _, digest = value.rpartition("#")
    assert op == f"{GEMV_OP}=1"
    assert len(digest) == 16 and all(c in "0123456789abcdef" for c in digest), value
    expected = hashlib.sha256(f"{MODULES[0]}={GEMV_OP}".encode()).hexdigest()[:16]
    assert digest == expected, "the digest must be reproducible in the next process"


def test_nothing_is_declared_when_nothing_was_declared_into():
    """No vLLM, or an ``additional_config`` this plugin may not extend under an
    uncompiled forward: there is no record and no cache key to protect."""
    assert note_traced_dispatch(MODULES[0], GEMV_OP) is None
    assert traced_dispatch() == {}


def test_a_second_config_starts_a_fresh_accumulation():
    first = _declared()
    note_traced_dispatch(MODULES[0], GEMV_OP)
    second = _declared()          # a second model in one process: a test, not a serve
    assert DISPATCH_FACT not in second.additional_config[TESSERA_KEY]
    note_traced_dispatch(MODULES[1], GEMM_OP)
    assert traced_dispatch() == {MODULES[1]: GEMM_OP}
    assert first.additional_config[TESSERA_KEY][DISPATCH_FACT] != \
        second.additional_config[TESSERA_KEY][DISPATCH_FACT]


def test_vllm_hashes_the_two_lane_states_apart():
    pytest.importorskip("vllm")
    from vllm.config import VllmConfig, set_current_vllm_config

    hashes = {}
    for lane in (GEMV_OP, GEMM_OP):
        reset_for_tests()
        cfg = VllmConfig()
        with set_current_vllm_config(cfg):
            declare_compile_identity(serve_mode="streamed")
        for name in MODULES:                       # at weight load: no current config
            note_traced_dispatch(name, lane)
        hashes[lane] = cfg.compute_hash()
    assert hashes[GEMV_OP] != hashes[GEMM_OP]
    reset_for_tests()
    again = VllmConfig()
    with set_current_vllm_config(again):
        declare_compile_identity(serve_mode="streamed")
    for name in MODULES:
        note_traced_dispatch(name, GEMV_OP)
    assert again.compute_hash() == hashes[GEMV_OP]
