"""The window GEMV's launch plan against a device's shared-memory budget.

Torch-CPU only: every test here reads arithmetic and a plan, and none of them
builds or launches the kernel, so the file runs on a box with no GPU and no
toolchain -- which is where the AMD budget it prices has to be decided
(RobTand/tessera#454).

Two devices, two budgets, and the difference is the point.  A CUDA device
publishes a small static per-block limit and an opt-in ceiling above it that
``launch_typed`` climbs with ``cudaFuncSetAttribute`` before each launch, so no
ceiling binds the plan and the plan is the one the 2026-09-02 sweep measured --
pinned here byte for byte.  A 64 KiB AMD workgroup has no ladder to climb
(``shared_memory_per_block_optin`` is not even an attribute on torch
2.11+rocm7.2; ``hipFuncSetAttribute`` cannot raise the limit), so 65,536 B is
the budget, it binds, and the plan is solved inside it.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.errors import GrammarError                          # noqa: E402
from tessera import kernel_window_gemv as kg                     # noqa: E402
from tessera.kernel_roster import WINDOW_GEMV_SOURCE             # noqa: E402

BF16 = torch.bfloat16
FP32 = torch.float32

#: A 64 KiB workgroup: gfx1150, gfx1151 and gfx1201 alike.  65,536 B compiles
#: and 65,537 B is refused on all three, and the RX 9070 XT's ``rocminfo``
#: GROUP segment reads 65,536 B
#: (``/home/rob/tmp/agents/rdna35-port-spike/receipts/14_lds_ceiling_probe.txt``,
#: ``20_rocminfo.txt``).
RDNA_LDS = 65536

#: sm_121 as the device reports it -- ``experiments/results/kernel-window/
#: trace_kernel_window.json`` (48 SMs, 49,152 B per block, 101,376 B opt-in,
#: 102,400 B per SM).  The opt-in ceiling holds every plan in the menu, which
#: is why no budget binds on this device.
SM_121 = dict(multi_processor_count=48, shared_memory_per_block=49152,
              shared_memory_per_block_optin=101376,
              shared_memory_per_multiprocessor=102400)

#: gfx1201 as ROCm torch reports it -- receipt
#: ``/home/rob/tmp/agents/t454-lds-plan/receipts/01_gfx1201_device_properties.txt``
#: (torch 2.11.0+rocm7.2.4, HIP 7.2.53211, AMD Radeon RX 9070 XT).  There is no
#: ``shared_memory_per_block_optin`` key at all, and ``multi_processor_count``
#: is 32 where ``rocminfo`` reports 64 compute units: torch counts WORKGROUP
#: PROCESSORS on RDNA, and a WGP is two CUs.  ``blocks = sm_count * per_sm`` is
#: therefore 32 grid slots per resident wave on this part, not 64.
GFX1201 = dict(multi_processor_count=32, shared_memory_per_block=65536,
               shared_memory_per_multiprocessor=65536)


class _Props:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def _as_device(monkeypatch, fields):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_a, **_k: _Props(**fields))


def _no_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


# --------------------------------------------------------------------------
# the mirror
# --------------------------------------------------------------------------

def test_the_plan_prices_the_shared_memory_the_kernel_asks_for():
    """``plan_smem_bytes`` is a mirror of one line of ``window_gemv.cu``, so the
    two spellings are read back off the source here.  A mirror nothing checks
    is the drift issue #145 filed for the rate roster; the roster cannot cover
    these two because they are an expression and a constant, not a declaration."""
    text = Path(WINDOW_GEMV_SOURCE).read_text()
    assert f"constexpr int RED_STRIDE = {kg.RED_STRIDE};" in text
    assert ("const size_t smem = tbl_bytes + (size_t)mt * 16 * RED_STRIDE * 4 + "
            "2 * (size_t)max_item_cols(mt) * mt * 4;") in text
    assert "constexpr int max_item_cols(int mt) { return mt <= 2 ? 1024 : 256; }" in text
    for mt in (1, 2, 4, 8):
        assert kg.max_item_cols(mt) == (1024 if mt <= 2 else 256)


@pytest.mark.parametrize("mt,bf16,fp32", [(1, 43072, 75840), (2, 53376, 86144),
                                          (4, 49408, 82176), (8, 66048, 98816)])
def test_the_launch_asks_for_these_bytes(mt, bf16, fp32):
    assert kg.plan_smem_bytes(mt, table_dtype=BF16) == bf16
    assert kg.plan_smem_bytes(mt, table_dtype=FP32) == fp32


def test_the_m_8_tile_is_priced_twice_and_only_one_of_them_is_launchable():
    """The 128-column M=8 plan is 57,856 B of *designed* budget.  It is not yet
    what a launch asks for: ``window_gemv.cu`` lays its x tile out from
    ``constexpr MAX_COLS = max_item_cols(MT)`` and puts the second buffer at
    ``xs0 + MAX_COLS * MT``, so an M=8 launch asks for 66,048 B whatever the
    plan's item width, and 66,048 B does not fit a 64 KiB workgroup.  Closing
    that gap is a change to the ``.cu`` (RobTand/tessera#453 owns the file);
    until it lands the M=8 plan is arithmetic awaiting a launch receipt."""
    assert kg.plan_smem_bytes(8, table_dtype=BF16, item_cols=128) == 57856
    assert kg.plan_smem_bytes(8, table_dtype=BF16) == 66048
    assert kg.plan_smem_bytes(8, table_dtype=BF16) > RDNA_LDS >= 57856


# --------------------------------------------------------------------------
# the budget
# --------------------------------------------------------------------------

def test_a_cuda_opt_in_ladder_is_not_a_budget(monkeypatch):
    _as_device(monkeypatch, SM_121)
    assert kg.device_shared_mem_per_block() is None


def test_a_device_without_a_ladder_reports_its_hard_ceiling(monkeypatch):
    _as_device(monkeypatch, GFX1201)
    assert kg.device_shared_mem_per_block() == RDNA_LDS


def test_no_device_no_budget(monkeypatch):
    _no_device(monkeypatch)
    assert kg.device_shared_mem_per_block() is None


@pytest.mark.parametrize("mt,cols", [(1, 1024), (2, 1024), (4, 256), (8, 128)])
def test_the_widest_tile_that_fits_64_kib(mt, cols):
    assert kg.item_cols_for_budget(mt, RDNA_LDS, table_dtype=BF16) == cols


@pytest.mark.parametrize("mt", [1, 2, 4, 8])
def test_the_fp32_table_is_refused_and_the_message_names_the_budget(mt):
    with pytest.raises(GrammarError) as raised:
        kg.item_cols_for_budget(mt, RDNA_LDS, table_dtype=FP32)
    said = str(raised.value)
    assert str(RDNA_LDS) in said and "no opt-in ladder" in said
    assert str(kg.plan_smem_bytes(mt, table_dtype=BF16)) in said   # what does fit


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------

#: What ``default_plan`` returns today on sm_121, per M tile: (rpl, blocks,
#: cols_per_item) for the bf16 table, and the fp32 table's blocks.  Pinned so
#: that the AMD budget cannot move a CUDA plan by a byte.
SM_121_PLAN = {1: (16, 96, 256), 2: (16, 96, 256), 4: (8, 96, 256), 8: (8, 96, 256)}


@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_the_sm_121_plan_is_unchanged(monkeypatch, m):
    _as_device(monkeypatch, SM_121)
    rpl, blocks, cols = SM_121_PLAN[m]
    assert kg.default_plan(9728, 9728, m) == kg.Plan(
        rpl=rpl, warps=16, blocks=blocks, cols_per_item=cols, table_dtype=BF16,
        item_cost=24, balanced=True)
    assert kg.default_plan(9728, 9728, m, table_dtype=FP32) == kg.Plan(
        rpl=rpl, warps=16, blocks=48, cols_per_item=cols, table_dtype=FP32,
        item_cost=24, balanced=True)


@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_a_host_with_no_device_plans_exactly_as_sm_121_does(monkeypatch, m):
    """The CPU fallback is 48 SMs and no budget -- an unknown ceiling is not a
    measured 64 KiB one, so it changes nothing."""
    _no_device(monkeypatch)
    _as = kg.default_plan(9728, 9728, m)
    rpl, blocks, cols = SM_121_PLAN[m]
    assert (_as.rpl, _as.blocks, _as.cols_per_item) == (rpl, blocks, cols)


@pytest.mark.parametrize("m,cols,tile,smem", [(1, 256, 1024, 43072), (2, 256, 1024, 53376),
                                             (4, 256, 256, 49408), (8, 128, 128, 57856)])
def test_a_64_kib_device_keeps_the_small_plans_and_caps_m_8(m, cols, tile, smem):
    """``cols_per_item`` is how wide an ITEM is; the x tile the launch allocates
    is the cap the budget allows, which is why M<=2 still prices 1024 columns
    while cutting 256-column items (the sweep's shape, unchanged)."""
    mt = kg._m_tile(m)
    plan = kg.default_plan(9728, 9728, m, sm_count=32, shared_mem_per_block=RDNA_LDS)
    assert plan.cols_per_item == cols
    assert kg.item_cols_for_budget(mt, RDNA_LDS, table_dtype=BF16) == tile
    assert kg.plan_smem_bytes(mt, table_dtype=BF16, item_cols=tile) == smem
    assert smem <= RDNA_LDS


@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_per_sm_comes_from_the_budget_not_the_table_dtype(m):
    """43-58 KiB per block leaves room for one workgroup in a 64 KiB CU, not the
    two the bf16 table used to assume (and that ``__launch_bounds__(512, 2)``
    asks for -- on HIP its second argument is min waves per EU and cannot be
    honoured at this LDS size either; recorded, not changed)."""
    plan = kg.default_plan(9728, 9728, m, sm_count=32, shared_mem_per_block=RDNA_LDS)
    assert plan.blocks == 32          # 32 WGPs * one resident block
    assert kg.default_plan(9728, 9728, m, sm_count=32).blocks == 64   # no budget: the old 2


def test_the_plan_refuses_the_fp32_table_on_a_64_kib_device():
    with pytest.raises(GrammarError, match=str(RDNA_LDS)):
        kg.default_plan(9728, 9728, 1, sm_count=32, shared_mem_per_block=RDNA_LDS,
                        table_dtype=FP32)


def test_a_gfx1201_device_needs_no_argument(monkeypatch):
    """The AMD plan is what the device says, not what a caller remembers."""
    _as_device(monkeypatch, GFX1201)
    assert kg.default_plan(9728, 9728, 8).cols_per_item == 128
    assert kg.default_plan(9728, 9728, 1).blocks == 32
    with pytest.raises(GrammarError):
        kg.default_plan(9728, 9728, 1, table_dtype=FP32)


# --------------------------------------------------------------------------
# the item tables the plan's cap produces
# --------------------------------------------------------------------------

def _repacked(cols: int, rate: int = 4, n_tiles: int = 2) -> kg.Repacked:
    """A run table and nothing else: ``plan_items`` reads ``runs``, ``n_tiles``
    and the words' device, which is all an item table is cut from."""
    runs = torch.tensor([[rate, 0, cols, 0]], dtype=torch.int32)
    return kg.Repacked(words=torch.zeros(1, dtype=torch.int32), tile_words=0, n_tiles=n_tiles,
                       rows=n_tiles * kg.TILE_ROWS, cols=cols, rows_p=n_tiles * kg.TILE_ROWS,
                       perm=torch.arange(cols, dtype=torch.int32), runs=runs, rates=(rate,))


def test_items_for_honours_the_plan_cap():
    """``items_for`` cuts at ``min(plan.cols_per_item, max_item_cols(mt))`` -- the
    path an M=8 launch takes to the 128-column plan."""
    rep = _repacked(2048)
    plan = kg.default_plan(1024, 2048, 8, sm_count=32, shared_mem_per_block=RDNA_LDS)
    assert plan.cols_per_item == 128
    assert int(kg.items_for(rep, plan, 8)[:, 3].max()) <= 128
    wide = kg.default_plan(1024, 2048, 8, sm_count=32)
    assert int(kg.items_for(rep, wide, 8)[:, 3].max()) <= 256
