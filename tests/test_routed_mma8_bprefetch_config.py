"""Build-scoped MMA8 B-fragment controls; CPU checks are not CUDA proof."""
import importlib.util
import sys
from types import SimpleNamespace

import pytest

from tessera import routed_fused as rf
from tessera.errors import GrammarError

ENV = "TESSERA_ROUTED_FUSED_MMA8_GATE_UP_B_PREFETCH"
DEFINE = "-D" + ENV + "="


def fresh(monkeypatch, value=None):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    name = "tessera._mma8_b_prefetch_control"
    spec = importlib.util.spec_from_file_location(name, rf.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value,want", [(None, 0), ("0", 0), ("1", 1)])
def test_build_selection_is_explicit_and_frozen(monkeypatch, value, want):
    module = fresh(monkeypatch, value)
    assert module.MMA8_GATE_UP_B_PREFETCH == want
    monkeypatch.setenv(ENV, str(1 - want))
    assert DEFINE + str(want) in module._cflags("sm_121", True, True)


@pytest.mark.parametrize("value", ["", "2", "-1", "true", "01"])
def test_unknown_build_choice_refused_before_compile(monkeypatch, value):
    with pytest.raises(GrammarError, match=ENV):
        fresh(monkeypatch, value)


@pytest.mark.parametrize("fp8,mma8,fp4", [(False, False, False), (True, False, False),
                                       (True, False, True)])
def test_other_family_compile_flags_unchanged(monkeypatch, fp8, mma8, fp4):
    module = fresh(monkeypatch, "1")
    assert not any(flag.startswith(DEFINE) for flag in
                   module._cflags("sm_121", fp8, mma8, fp4))


def native_metadata(module, b_prefetch):
    # These stand for compiled attributes only. No fake CUDA numerical execution.
    return SimpleNamespace(
        BM=module.BM, BN=module.BN, HALF=module.HALF, BK=module.BK,
        DENSE_ROW_QUANTUM=module.DENSE_ROW_QUANTUM, RATE_MIN=module.RATE_MIN,
        ROUTED_RATE_MAX=module.RATE_MAX, RATE_MAX=module.DENSE_RATE_MAX["e4m3"],
        SLOT_WORDS_MAX=module.slot_words_for_rate(module.DENSE_RATE_MAX["e4m3"]),
        BDESC_INTS=module.BDESC_INTS, WINDOW_BITS=module.WINDOW_BITS,
        FAMILY_FP8=True, FAMILY_MMA8=True,
        PAIRED_K32_BUILD=module._paired_k32_build_enabled(True),
        STAGES=module.STAGES, MAX_ROLES=module.MAX_ROLES,
        WORD_STAGES=module.WORD_STAGES,
        WORD_STAGES_MIN=module.WORD_STAGES_MIN,
        SMEM_FIXED_GATE_UP=module.SMEM_FIXED_MMA8[0],
        SMEM_FIXED_DOWN=module.SMEM_FIXED_MMA8[2], BM_WIDE=module.BM_WIDE,
        A_REGION_BYTES_WIDE=module.a_region_bytes(module.BM_WIDE, mma8=True),
        HAS_WIDE_GATE_UP=module.has_width("e4m3mma", 0, module.BM_WIDE),
        HAS_WIDE_DOWN=module.has_width("e4m3mma", 2, module.BM_WIDE),
        GATE_UP_RATE_MAX=max(module.routed_lane_rates("e4m3mma")),
        MMA8_GATE_UP_B_PREFETCH=b_prefetch)


@pytest.mark.parametrize("want,have", [(0, 1), (1, 0)])
def test_wrong_compiled_schedule_refused(monkeypatch, want, have):
    module = fresh(monkeypatch, str(want))
    lib = native_metadata(module, have)
    monkeypatch.setattr(module, "build_library", lambda *args: lib)
    with pytest.raises(GrammarError, match="MMA8_GATE_UP_B_PREFETCH"):
        module._ext("e4m3mma")


@pytest.mark.parametrize("value", ["0", "1"])
def test_matching_compiled_schedule_admitted(monkeypatch, value):
    module = fresh(monkeypatch, value)
    lib = native_metadata(module, int(value))
    monkeypatch.setattr(module, "build_library", lambda *args: lib)
    assert module._ext("e4m3mma") is lib
