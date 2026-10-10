"""Gated L1 activation prefetch on the 16-bit E4M3 library (tessera#739).

The fused routed chunk loop waits on the activation global one chunk
ahead. The E4M3 instruction already prefetches it on routed one-run
launches. This file pins the same gate for the 16-bit E4M3 library.
Distance comes from ``A_PREFETCH`` there. No shared memory moves.
Dense, two-run, value and FP4 launches do not move. CPU source checks
only; they check source shape, not CUDA numerics.
"""
from pathlib import Path

KERNEL = Path(__file__).resolve().parents[1] / "src" / "tessera" / "serving" / "csrc" / "routed_fused_window.cu"


def _text():
    return KERNEL.read_text()


def _line_with(text, needle, after=0):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if needle in line:
            if after == 0:
                return line
            after -= 1
    raise AssertionError(f"missing block: {needle}")


def test_each_library_reads_its_own_distance():
    """The defect: the 16-bit E4M3 path read distance 0. Pin the full arm map:
    the E4M3 instruction and the 16-bit E4M3 library read A_PREFETCH, the value
    family reads VALUE_A_PREFETCH, the FP4 lane reads 0."""
    text = _text()
    chunk = text.split("constexpr int PREFETCH_DISTANCE")[1].split(";")[0]
    norm = " ".join(chunk.split())
    assert norm == "= FAMILY_MMA8 ? A_PREFETCH : (FAMILY_FP8 ? A_PREFETCH : (!FAMILY_FP4 ? VALUE_A_PREFETCH : 0))", norm


def test_prefetch_stays_gated_to_routed_one_run():
    """Dense and two-run launches keep no prefetch."""
    text = _text()
    line = _line_with(text, "constexpr bool PREFETCH_A")
    assert "!DENSE" in line and "!TWO" in line


def test_value_and_fp4_paths_do_not_move():
    """The value arm and the FP4 lane keep their distances (pinned above)."""
    text = _text()
    chunk = text.split("constexpr int PREFETCH_DISTANCE")[1].split(";")[0]
    norm = " ".join(chunk.split())
    assert "(!FAMILY_FP4 ? VALUE_A_PREFETCH : 0)" in norm, norm


def test_prefetch_adds_no_shared_memory():
    """The hint changes no layout owner."""
    text = _text()
    owners = (
        "__host__ __device__ constexpr int a_region_bytes",
        "__host__ __device__ constexpr int smem_bytes_ws",
        "struct Layout {",
    )
    for owner in owners:
        assert owner in text
        tail = text.split(owner, 1)[1]
        end = tail.index("};") if owner.startswith("struct") else tail.index("}")
        assert "PREFETCH" not in tail[:end]
