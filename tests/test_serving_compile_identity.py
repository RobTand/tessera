"""The compile cache keeps the actual serving mode in its identity.

These tests cover the declared record, its mode, and its refusal conditions.
The runtime test compares the hashes for resident and streamed execution.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import tessera.serving
from tessera.serving.compile_identity import (
    TESSERA_KEY, declare_compile_identity, declare_compile_identity_in,
    declared_forward_is_compiled, reset_for_tests)


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


@pytest.mark.parametrize("mode,expected", [("VLLM_COMPILE", True), ("NONE", False)])
def test_declared_compile_mode_is_a_snapshot_and_reset_forgets_it(mode, expected):
    assert declared_forward_is_compiled() is False
    config = _config(mode=mode)
    declare_compile_identity_in(config, serve_mode="resident")
    config.compilation_config.mode.name = "NONE" if expected else "VLLM_COMPILE"
    assert declared_forward_is_compiled() is expected
    reset_for_tests()
    assert declared_forward_is_compiled() is False


def test_an_eager_unextendable_config_replaces_the_saved_compile_mode():
    declare_compile_identity_in(_config(), serve_mode="resident")
    assert declared_forward_is_compiled() is True
    assert declare_compile_identity_in(_config(mode="NONE", extra=object()),
                                       serve_mode="resident") is None
    assert declared_forward_is_compiled() is False


def test_no_current_config_declares_nothing():
    # vLLM absent: ImportError; vLLM present but no set_current_vllm_config: None
    assert declare_compile_identity(serve_mode="resident") is None




def _real_vllm_hash_check(code):
    # A fresh interpreter cannot read another test's fake vLLM package.
    import subprocess
    import sys

    imports = """
import importlib.util, sys
if importlib.util.find_spec("vllm") is None:
    sys.exit(77)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform
current_platform.import_ir_kernels()
from tessera.serving.compile_identity import (
    TESSERA_KEY, declare_compile_identity, note_traced_dispatch, reset_for_tests)
"""
    constants = (f"FIRST_OP, SECOND_OP, MODULES = "
                 f"{(WINDOW_GEMM_SYMBOL, FUSED_WINDOW_DENSE_SYMBOL, MODULES)!r}\n")
    result = subprocess.run([sys.executable, "-c", imports + constants + code],
                            capture_output=True, text=True)
    if result.returncode == 77:
        pytest.skip("real vLLM is absent from the test interpreter")
    assert result.returncode == 0, result.stdout + result.stderr


def test_vllm_hashes_the_two_modes_apart():
    _real_vllm_hash_check(r"""
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
""")



from tessera.serving.compile_identity import note_traced_dispatch, traced_dispatch
from tessera.serving.scheme import WINDOW_GEMM_SYMBOL, FUSED_WINDOW_DENSE_SYMBOL

MODULES = ("model.layers.0.mlp.down_proj", "model.layers.0.self_attn.qkv_proj",
           "model.layers.1.mlp.down_proj")


def _declared(mode="streamed"):
    cfg = _config()
    declare_compile_identity_in(cfg, serve_mode=mode)
    return cfg


def _identity(cfg):
    return json.dumps(cfg.additional_config, sort_keys=True)


def test_the_two_lane_states_are_two_identities():
    first = _declared()
    for name in MODULES:
        note_traced_dispatch(name, WINDOW_GEMM_SYMBOL)
    a = _identity(first)
    reset_for_tests()
    second = _declared()
    for name in MODULES:
        note_traced_dispatch(name, FUSED_WINDOW_DENSE_SYMBOL)
    assert _identity(second) != a


def test_one_lane_state_is_one_identity_however_the_modules_are_ordered():
    first = _declared()
    for name in MODULES:
        note_traced_dispatch(name, WINDOW_GEMM_SYMBOL)
    a = _identity(first)
    reset_for_tests()
    second = _declared()
    for name in reversed(MODULES):
        note_traced_dispatch(name, WINDOW_GEMM_SYMBOL)
    assert _identity(second) == a
    note_traced_dispatch(MODULES[0], WINDOW_GEMM_SYMBOL)
    assert _identity(second) == a


def test_a_mixed_checkpoint_needs_the_set_not_a_count():
    first = _declared()
    note_traced_dispatch(MODULES[0], WINDOW_GEMM_SYMBOL)
    note_traced_dispatch(MODULES[1], FUSED_WINDOW_DENSE_SYMBOL)
    a = _identity(first)
    reset_for_tests()
    second = _declared()
    note_traced_dispatch(MODULES[0], FUSED_WINDOW_DENSE_SYMBOL)
    note_traced_dispatch(MODULES[1], WINDOW_GEMM_SYMBOL)
    assert _identity(second) != a


def test_a_second_config_starts_a_fresh_accumulation():
    first = _declared()
    note_traced_dispatch(MODULES[0], WINDOW_GEMM_SYMBOL)
    a = _identity(first)
    second = _declared()
    note_traced_dispatch(MODULES[1], FUSED_WINDOW_DENSE_SYMBOL)
    assert traced_dispatch() == {MODULES[1]: FUSED_WINDOW_DENSE_SYMBOL}
    assert _identity(first) == a
    assert _identity(second) != a


def test_vllm_hashes_the_two_lane_states_apart():
    _real_vllm_hash_check(r"""
hashes = {}
for operation in (FIRST_OP, SECOND_OP):
    reset_for_tests()
    cfg = VllmConfig()
    with set_current_vllm_config(cfg):
        declare_compile_identity(serve_mode="streamed")
    for name in MODULES:
        note_traced_dispatch(name, operation)
    hashes[operation] = cfg.compute_hash()
assert hashes[FIRST_OP] != hashes[SECOND_OP]
reset_for_tests()
again = VllmConfig()
with set_current_vllm_config(again):
    declare_compile_identity(serve_mode="streamed")
for name in MODULES:
    note_traced_dispatch(name, FIRST_OP)
assert again.compute_hash() == hashes[FIRST_OP]
""")
