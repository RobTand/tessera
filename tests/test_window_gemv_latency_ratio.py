"""Issue #109: the served latency ratio, and the arithmetic it refuses on.

``experiments/window_gemv_latency_ratio.py`` turns two per-arm receipts into
one number -- fallback over engaged -- and the whole value of that number is
that it is orientable, sourced, and withheld when the receipts cannot carry
it.  Three properties are pinned here because each of them was got wrong once:

* **The metric may not exist under the name the receipt promised.**  The
  2026-09-03 pair was driven by a script reading
  ``vllm:time_per_output_token_seconds``, a stem vLLM 0.28 does not publish
  (the release calls it ``vllm:request_time_per_output_token_seconds``; fixed
  in ``b8ef715``, committed 23:24:09Z -- *after* both arms had started, armA
  at 23:18:28Z and armB at 23:22:53Z).  Both receipts therefore carry
  ``tpot: null`` while the histogram behind it moved perfectly well, and every
  per-output-token number quoted for that pair is recoverable only from the
  ``series_moved`` dump.  The vectors below are those receipts' own bytes.

* **The ratio has a direction, and a reader must not have to guess it.**
  Greater than one means the arm WITHOUT the built kernel is slower, i.e. the
  lane is faster.

* **A ratio from a contended pair is not evidence.**  Contention is read off
  each arm's own receipt, and a receipt too old to carry the swap-activity
  field says "cannot say" rather than "nothing moved".
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "window_gemv_latency_ratio", ROOT / "experiments" / "window_gemv_latency_ratio.py")
RATIO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RATIO)


#: Verbatim from ``latency-armA-streamed-eager.json`` (lane built) and
#: ``latency-armB-streamed-eager.json`` (read-only extensions root, so the
#: route takes its published fallback), 2026-09-03, ``windows.decode`` and
#: ``windows.prefill``.  Held as literals rather than read from
#: ``/home/rob/tessera-runs`` so the test pins the arithmetic on a machine
#: that never held those runs.
ARM_A = {
    "decode": {"tpot": None, "itl": None,
               "ttft": {"count": 12.0, "mean_s": 0.07033886512120564,
                        "sum_s": 0.8440663814544678},
               "series_moved": {
                   "vllm:request_time_per_output_token_seconds_sum": 0.5224967625750208,
                   "vllm:request_time_per_output_token_seconds_count": 12.0,
                   "vllm:inter_token_latency_seconds_sum": 66.35708884702763,
                   "vllm:inter_token_latency_seconds_count": 1524.0,
                   "vllm:time_to_first_token_seconds_sum": 0.8440663814544678,
                   "vllm:time_to_first_token_seconds_count": 12.0}},
    "prefill": {"tpot": None, "itl": None,
                "ttft": {"count": 12.0, "mean_s": 0.0697175661722819,
                         "sum_s": 0.8366107940673828},
                "series_moved": {
                    "vllm:time_to_first_token_seconds_sum": 0.8366107940673828,
                    "vllm:time_to_first_token_seconds_count": 12.0,
                    "vllm:request_time_per_output_token_seconds_count": 12.0}},
}
ARM_B = {
    "decode": {"tpot": None, "itl": None,
               "ttft": {"count": 12.0, "mean_s": 0.5416906277338663,
                        "sum_s": 6.5002875328063965},
               "series_moved": {
                   "vllm:request_time_per_output_token_seconds_sum": 4.192399552503764,
                   "vllm:request_time_per_output_token_seconds_count": 12.0,
                   "vllm:inter_token_latency_seconds_sum": 532.434743167978,
                   "vllm:inter_token_latency_seconds_count": 1524.0,
                   "vllm:time_to_first_token_seconds_sum": 6.5002875328063965,
                   "vllm:time_to_first_token_seconds_count": 12.0}},
    "prefill": {"tpot": None, "itl": None,
                "ttft": {"count": 12.0, "mean_s": 0.29905442396799725,
                         "sum_s": 3.588653087615967},
                "series_moved": {
                    "vllm:time_to_first_token_seconds_sum": 3.588653087615967,
                    "vllm:time_to_first_token_seconds_count": 12.0,
                    "vllm:request_time_per_output_token_seconds_count": 12.0}},
}


def test_named_field_wins_and_says_so():
    got = RATIO._metric(ARM_A["decode"], "ttft")
    assert got["source"] == "histogram"
    assert got["mean_s"] == pytest.approx(0.07033886512120564)


def test_tpot_is_recovered_from_series_moved_when_the_field_is_null():
    """The stem the receipts were driven by did not exist; the delta did."""
    for arm, expected in ((ARM_A, 0.5224967625750208 / 12.0),
                          (ARM_B, 4.192399552503764 / 12.0)):
        got = RATIO._metric(arm["decode"], "tpot")
        assert got is not None, "a null field with a moved histogram must not read as absent"
        assert got["source"] == "series_moved"
        assert got["mean_s"] == pytest.approx(expected)


def test_absent_series_reads_as_absent_not_as_zero():
    """prefill publishes no per-output-token SUM: one token, no output gap."""
    assert RATIO._metric(ARM_A["prefill"], "tpot") is None


def test_ratio_is_fallback_over_engaged():
    r = RATIO._ratio(RATIO._metric(ARM_B["decode"], "tpot"),
                     RATIO._metric(ARM_A["decode"], "tpot"))
    assert r["ratio_fallback_over_engaged"] == pytest.approx(
        4.192399552503764 / 0.5224967625750208, rel=1e-4)
    assert r["ratio_fallback_over_engaged"] > 1  # the fallback arm is the slow one
    assert r["source"] == "series_moved/series_moved"


def test_the_prefill_control_moves_too():
    """The window the lane cannot serve moves 4.29x on the same pair.

    ``GEMV_MAX_M = 8`` refuses a 512-row prefill by name, so both arms run the
    same materialised path there.  A null arm that moves 4.29x is the reason
    the 8.02x decode figure from that pair is not a lane measurement.
    """
    ctrl = RATIO._ratio(RATIO._metric(ARM_B["prefill"], "ttft"),
                        RATIO._metric(ARM_A["prefill"], "ttft"))
    decode = RATIO._ratio(RATIO._metric(ARM_B["decode"], "tpot"),
                          RATIO._metric(ARM_A["decode"], "tpot"))
    assert ctrl["ratio_fallback_over_engaged"] == pytest.approx(4.2896, abs=5e-4)
    assert decode["ratio_fallback_over_engaged"] == pytest.approx(8.0238, abs=5e-4)


def test_contention_distinguishes_cannot_say_from_nothing_moved():
    old = RATIO._contention({"contended": True, "max_load1_per_cpu": 3.413})
    assert old["swap_moved_during_windows"] is None      # schema /1: cannot say
    quiet = RATIO._contention({"contended": False, "swap_io": {
        "decode": {"moved": False}, "prefill": {"moved": False}}})
    assert quiet["swap_moved_during_windows"] is False   # schema /2: said, and no
    paged = RATIO._contention({"contended": False, "swap_io": {
        "decode": {"moved": True}, "prefill": {"moved": False}}})
    assert paged["swap_moved_during_windows"] is True


def test_trace_clock_is_base_plus_ts(tmp_path):
    """A chrome trace's absolute clock is ``baseTimeNanoseconds + ts``.

    Cutting both arms by the driver's ``marks_unix`` only works if this
    conversion is right; an off-by-a-base cut silently returns zero kernels
    and a lane share of nothing.
    """
    import json
    base_ns = 1_788_479_556_000_000_000
    base_s = base_ns / 1e9
    doc = {"baseTimeNanoseconds": base_ns, "traceEvents": [
        {"ph": "X", "cat": "kernel", "name": "window_gemv_kernel<14>", "ts": 0.0, "dur": 100.0},
        {"ph": "X", "cat": "kernel", "name": "window_gemv_kernel<14>", "ts": 5e6, "dur": 200.0},
        {"ph": "X", "cat": "kernel", "name": "internal::gemvx::kernel", "ts": 9e6, "dur": 300.0},
        {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 1.0, "dur": 9.0},
    ]}
    p = tmp_path / "rank0.pt.trace.json"
    p.write_text(json.dumps(doc))
    w = RATIO.trace_window(str(p), base_s + 4.0, base_s + 10.0)
    assert w["kernels_in_trace"] == 3          # the cpu_op is not a kernel
    assert w["kernels_in_window"] == 2         # ts=0 falls outside the cut
    assert w["device_ms_in_window"] == pytest.approx(0.5)
    assert w["trace_span_utc"][0] == pytest.approx(base_s)
