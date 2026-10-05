"""Build-scoped MMA8 activation ring (tessera#739); CPU checks are not CUDA proof."""
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tessera import routed_fused as rf
from tessera.errors import GrammarError

ENV = "TESSERA_ROUTED_FUSED_MMA8_A_RING"
DEFINE = "-D" + ENV + "="
KERNEL = Path(rf.__file__).parent / "serving" / rf.SOURCE


def fresh(monkeypatch, value=None):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    name = "tessera._mma8_a_ring_control"
    spec = importlib.util.spec_from_file_location(name, rf.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def kernel_choices():
    """The ring choices the kernel's own static_assert admits."""
    text = KERNEL.read_text()
    lo, hi = re.search(r"TESSERA_ROUTED_FUSED_MMA8_A_RING >= (\d+) && TESSERA_ROUTED_FUSED_MMA8_A_RING <= (\d+)",
                       text).groups()
    return [str(v) for v in range(int(lo), int(hi) + 1)]


def test_default_is_off_and_layout_is_the_published_one(monkeypatch):
    module = fresh(monkeypatch)
    assert module.MMA8_A_RING == 0
    assert module.SMEM_FIXED_MMA8 == {0: 47_312, 1: 47_312, 2: 30_736}
    assert DEFINE + "0" in module._cflags("sm_121", True, True)


@pytest.mark.parametrize("value", kernel_choices())
def test_build_selection_is_explicit_and_frozen(monkeypatch, value):
    module = fresh(monkeypatch, value)
    assert module.MMA8_A_RING == int(value)
    monkeypatch.setenv(ENV, "0" if value != "0" else "1")
    assert DEFINE + value in module._cflags("sm_121", True, True)


@pytest.mark.parametrize("value", ["", "2", "3", "-1", "true", "01"])
def test_unknown_build_choice_refused_before_compile(monkeypatch, value):
    with pytest.raises(GrammarError, match=ENV):
        fresh(monkeypatch, value)


@pytest.mark.parametrize("fp8,mma8,fp4", [(False, False, False), (True, False, False),
                                       (True, False, True)])
def test_other_family_compile_flags_and_layout_unchanged(monkeypatch, fp8, mma8, fp4):
    off, on = fresh(monkeypatch, "0"), fresh(monkeypatch, "1")
    assert not any(flag.startswith(DEFINE) for flag in on._cflags("sm_121", fp8, mma8, fp4))
    assert on.SMEM_FIXED == off.SMEM_FIXED
    for bm in (on.BM, on.BM_WIDE):
        assert on.a_region_bytes(bm) == off.a_region_bytes(bm)


@pytest.mark.parametrize("value", [v for v in kernel_choices() if v != "0"])
def test_ring_moves_the_mma8_layout_by_its_stages(monkeypatch, value):
    """The ring is WORD_STAGES raw tiles of ``bm`` one-byte rows, after the A tiles."""
    off, on = fresh(monkeypatch, "0"), fresh(monkeypatch, value)
    for bm in (on.BM, on.BM_WIDE):
        ring = on.WORD_STAGES * bm * on.BK
        assert on.a_region_bytes(bm, mma8=True) - off.a_region_bytes(bm, mma8=True) == ring
        for mode in (0, 2):
            for sw in (4, 8, 12, 16):
                assert on.launch_smem_bytes(mode, sw, mma8=True, bm=bm) \
                    == off.launch_smem_bytes(mode, sw, mma8=True, bm=bm) + ring
                # three word stages still fit, so the ring has its stages
                assert on.word_stages(mode, sw, mma8=True) == on.WORD_STAGES
                assert on.launch_smem_bytes(mode, sw, mma8=True, bm=bm) <= on.SM121_MAX_DYNAMIC_SMEM


def native_metadata(module, ring):
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
        MMA8_GATE_UP_B_PREFETCH=bool(module.MMA8_GATE_UP_B_PREFETCH),
        MMA8_A_RING=ring)


@pytest.mark.parametrize("want,have", [(0, 1), (1, 0), (1, None), (0, None)])
def test_wrong_compiled_ring_refused(monkeypatch, want, have):
    module = fresh(monkeypatch, str(want))
    lib = native_metadata(module, have)
    monkeypatch.setattr(module, "build_library", lambda *args: lib)
    with pytest.raises(GrammarError, match="MMA8_A_RING"):
        module._ext("e4m3mma")


@pytest.mark.parametrize("value", kernel_choices())
def test_matching_compiled_ring_admitted(monkeypatch, value):
    module = fresh(monkeypatch, value)
    lib = native_metadata(module, int(value))
    monkeypatch.setattr(module, "build_library", lambda *args: lib)
    assert module._ext("e4m3mma") is lib
