"""Fixed-tolerance failures for the packed-stock gate (tessera#1182).

A fixed tolerance ignores the reduction length and the cancellation in the
measured error. The derived bound accepts a correct cancellation result that a
fixed tolerance would refuse, and refuses a wrong tiny output that a fixed
tolerance would accept. The comparison needs torch and skips without it.
"""
import functools
import importlib.util
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the packed-stock comparison needs torch")

from tessera.fp4_arithmetic import (
    FP4QualificationError as Refusal,
    check_packed_stock_arithmetic,
    derive_packed_stock_bound,
    stock_magnitude_upper,
)
from test_fp4_arithmetic_attestation import packed_shapes_fixture_report


def test_fixed_tolerance_would_refuse_a_correct_cancellation_result():
    report = packed_shapes_fixture_report((2, 8, 4096))
    kwargs = dict(k=4096, global_scale=1, report=report, device="CPU test fixture",
                  physical_device="fixture-device", shape=(2, 8, 4096))
    allowance, _ = derive_packed_stock_bound(64.0, **kwargs)
    fixed_tolerance = 1e-4
    measured_error = 0.01
    assert allowance["atol"] > measured_error > fixed_tolerance
    check_kwargs = {key: value for key, value in kwargs.items() if key not in ("device", "shape")}
    actual = torch.full((2, 8), measured_error, dtype=torch.float32)
    expected = torch.zeros((2, 8), dtype=torch.float32)
    receipt = check_packed_stock_arithmetic(actual, expected, 64.0, **check_kwargs)
    assert receipt["max_abs_error"] > fixed_tolerance


def test_fixed_tolerance_would_accept_a_wrong_tiny_output():
    report = packed_shapes_fixture_report((1, 4, 256))
    kwargs = dict(k=256, global_scale=1, report=report, device="CPU test fixture",
                  physical_device="fixture-device", shape=(1, 4, 256))
    allowance, _ = derive_packed_stock_bound(1e-7, **kwargs)
    assert allowance["atol"] < 1e-6 < 1e-3
    check_kwargs = {key: value for key, value in kwargs.items() if key not in ("device", "shape")}
    actual = torch.full((1, 4), 1e-6, dtype=torch.float32)
    expected = torch.zeros((1, 4), dtype=torch.float32)
    with pytest.raises(Refusal, match="exceeds"):
        check_packed_stock_arithmetic(actual, expected, 1e-7, **check_kwargs)


# CPU regression controls for the exact-zero magnitude and the per-segment
# grouped comparison (tessera#1465). The fixture report is a CPU test fixture:
# nothing here qualifies a device, and every receipt must stay unqualified.
DEVICE = "CPU test fixture"
PHYSICAL = "fixture-device"
GROUPED_K = 4096
GROUPED_COLS = 8
# The grouped run_correctness case: segment 0 is rows [:3], segment 1 is rows [3:].
GROUPED_ROWS = ((0, 3), (3, 5))
GROUPED_SCALES = (1, 4)
FIXED_TOLERANCE = 1e-4
LAYOUTS = {"large_first": (64.0, 1e-7), "tiny_first": (1e-7, 64.0)}


@functools.lru_cache(maxsize=None)
def _load_geometry_adapter():
    """Import the measurement adapter file by path, as test_d41_e2m1_geometry does.

    Cached so the file's import-time ``sys.path`` edit runs once per test module.
    """
    path = Path(__file__).parents[1] / "experiments" / "t4_code" / "bench_geometry_e2m1.py"
    spec = importlib.util.spec_from_file_location("bench_geometry_e2m1_grouped_controls", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _grouped_report():
    return packed_shapes_fixture_report(
        *((stop - start, GROUPED_COLS, GROUPED_K) for start, stop in GROUPED_ROWS))


def _segments(magnitudes):
    return [(start, stop, magnitude, scale)
            for (start, stop), magnitude, scale in zip(GROUPED_ROWS, magnitudes, GROUPED_SCALES)]


def _own_atols(report, magnitudes):
    atols = []
    for (start, stop), magnitude, scale in zip(GROUPED_ROWS, magnitudes, GROUPED_SCALES):
        allowance, _ = derive_packed_stock_bound(magnitude, k=GROUPED_K, global_scale=scale,
            report=report, device=DEVICE, physical_device=PHYSICAL,
            shape=(stop - start, GROUPED_COLS, GROUPED_K))
        atols.append(allowance["atol"])
    return atols


def _run_grouped(actual, expected, segments, report):
    return _load_geometry_adapter()._check_grouped_segments(
        actual, expected, segments, k=GROUPED_K, report=report, physical_device=PHYSICAL)


def _grouped_outputs():
    return (torch.zeros((GROUPED_ROWS[-1][1], GROUPED_COLS), dtype=torch.float32),
            torch.zeros((GROUPED_ROWS[-1][1], GROUPED_COLS), dtype=torch.float32))


def _float32_beyond(bound):
    value = torch.tensor(bound, dtype=torch.float32)
    while float(value) <= bound:
        value = torch.nextafter(value, torch.tensor(float("inf"), dtype=torch.float32))
    return value


def _float32_within(bound):
    value = torch.tensor(bound, dtype=torch.float32)
    while float(value) >= bound:
        value = torch.nextafter(value, torch.tensor(float("-inf"), dtype=torch.float32))
    return value


def _assert_unqualified(receipts):
    for receipt in receipts:
        assert receipt["arithmetic_qualified"] is False
        assert receipt["native_arithmetic_qualified"] is False
        assert receipt["status"] == "attested_contract_pending_reviews"


def _refusal_for(atol):
    return pytest.raises(Refusal, match=re.escape(f"exceeds {atol}"))


def test_exact_zero_magnitude_through_the_stock_api_and_grouped_helper():
    report = packed_shapes_fixture_report(
        (1, 4, 256), *((stop - start, GROUPED_COLS, GROUPED_K) for start, stop in GROUPED_ROWS))
    assert stock_magnitude_upper(0.0, k=256, report=report, device=DEVICE, physical_device=PHYSICAL,
                                 shape=(1, 4, 256)) == 0.0
    # The stock API itself: zero magnitude derives atol == 0.
    zeros = torch.zeros((1, 4), dtype=torch.float32)
    receipt = check_packed_stock_arithmetic(zeros, zeros.clone(), 0.0, k=256, global_scale=1,
                                            report=report, physical_device=PHYSICAL)
    assert receipt["atol"] == 0.0 and receipt["max_abs_error"] == 0.0
    _assert_unqualified([receipt])
    smallest = torch.zeros((1, 4), dtype=torch.float32)
    smallest[0, 0] = 2.0**-149
    assert float(smallest[0, 0]) == 2.0**-149
    with pytest.raises(Refusal, match="exceeds"):
        check_packed_stock_arithmetic(smallest, zeros, 0.0, k=256, global_scale=1,
                                      report=report, physical_device=PHYSICAL)
    # The same zero magnitude per grouped segment, through the shared helper.
    actual, expected = _grouped_outputs()
    receipts = _run_grouped(actual, expected, _segments((0.0, 0.0)), report)
    assert [r["atol"] for r in receipts] == [0.0, 0.0]
    assert [r["max_abs_error"] for r in receipts] == [0.0, 0.0]
    _assert_unqualified(receipts)
    for start, stop in GROUPED_ROWS:
        corrupted = actual.clone()
        corrupted[start, 0] = 2.0**-149
        with _refusal_for(0.0):
            _run_grouped(corrupted, expected, _segments((0.0, 0.0)), report)
    # A zero segment beside a nonzero one keeps its own zero allowance.
    corrupted = actual.clone()
    corrupted[GROUPED_ROWS[1][0], 0] = 2.0**-149
    with _refusal_for(0.0):
        _run_grouped(corrupted, expected, _segments((64.0, 0.0)), report)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_each_grouped_segment_accepts_its_cancellation_result_under_its_own_bound(layout):
    magnitudes = LAYOUTS[layout]
    report = _grouped_report()
    atols = _own_atols(report, magnitudes)
    assert atols[0] != atols[1]
    for index, (start, stop) in enumerate(GROUPED_ROWS):
        actual, expected = _grouped_outputs()
        if magnitudes[index] == 64.0:
            error = 0.01
            assert atols[index] > error > FIXED_TOLERANCE
        else:
            error = float(_float32_within(atols[index]))
            assert 0 < error < atols[index]
        actual[start:stop] = error
        receipts = _run_grouped(actual, expected, _segments(magnitudes), report)
        assert receipts[index]["atol"] == atols[index]
        assert receipts[index]["max_abs_error"] > 0
        assert receipts[index]["operand_magnitude_upper"] == magnitudes[index]
        assert receipts[index]["weight_global_exponent"] == GROUPED_SCALES[index].bit_length() - 1
        _assert_unqualified(receipts)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_each_grouped_segment_refuses_a_wrong_tiny_output(layout):
    magnitudes = LAYOUTS[layout]
    report = _grouped_report()
    atols = _own_atols(report, magnitudes)
    index = magnitudes.index(1e-7)
    start, stop = GROUPED_ROWS[index]
    assert atols[index] < 1e-6 < FIXED_TOLERANCE
    actual, expected = _grouped_outputs()
    actual[start:stop] = 1e-6
    with _refusal_for(atols[index]):
        _run_grouped(actual, expected, _segments(magnitudes), report)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_each_grouped_segment_refuses_an_error_just_beyond_its_derived_bound(layout):
    magnitudes = LAYOUTS[layout]
    report = _grouped_report()
    atols = _own_atols(report, magnitudes)
    for index, (start, stop) in enumerate(GROUPED_ROWS):
        beyond = _float32_beyond(atols[index])
        assert atols[index] < float(beyond) < atols[index] * (1 + 2.0**-20)
        actual, expected = _grouped_outputs()
        actual[start, 0] = beyond
        with _refusal_for(atols[index]):
            _run_grouped(actual, expected, _segments(magnitudes), report)
        # Just inside the same bound is accepted, so the refusal is the bound itself.
        actual[start, 0] = _float32_within(atols[index])
        receipts = _run_grouped(actual, expected, _segments(magnitudes), report)
        assert receipts[index]["max_abs_error"] <= atols[index]
        _assert_unqualified(receipts)


def test_grouped_helper_refuses_corruption_in_the_first_segment_only():
    magnitudes = LAYOUTS["large_first"]
    report = _grouped_report()
    atols = _own_atols(report, magnitudes)
    actual, expected = _grouped_outputs()
    actual[GROUPED_ROWS[0][0], 0] = _float32_beyond(atols[0])
    # The last segment is valid and compares clean on its own.
    check_packed_stock_arithmetic(actual[3:], expected[3:], magnitudes[1], k=GROUPED_K,
        global_scale=GROUPED_SCALES[1], report=report, physical_device=PHYSICAL)
    with _refusal_for(atols[0]):
        _run_grouped(actual, expected, _segments(magnitudes), report)


def test_grouped_helper_refuses_corruption_in_the_last_segment_only():
    magnitudes = LAYOUTS["large_first"]
    report = _grouped_report()
    atols = _own_atols(report, magnitudes)
    actual, expected = _grouped_outputs()
    actual[GROUPED_ROWS[1][0], 0] = _float32_beyond(atols[1])
    # The first segment is valid and compares clean on its own.
    check_packed_stock_arithmetic(actual[:3], expected[:3], magnitudes[0], k=GROUPED_K,
        global_scale=GROUPED_SCALES[0], report=report, physical_device=PHYSICAL)
    with _refusal_for(atols[1]):
        _run_grouped(actual, expected, _segments(magnitudes), report)


def test_omitted_or_borrowed_segment_allowance_would_admit_the_corruption():
    magnitudes = LAYOUTS["large_first"]
    report = _grouped_report()
    atols = _own_atols(report, magnitudes)
    wrong_tiny = 1e-6
    assert atols[1] < wrong_tiny < atols[0]
    actual, expected = _grouped_outputs()
    actual[GROUPED_ROWS[1][0], 0] = wrong_tiny
    with _refusal_for(atols[1]):
        _run_grouped(actual, expected, _segments(magnitudes), report)
    # Mutant: the last segment is left out of the segment list, so its rows are never compared.
    receipts = _run_grouped(actual, expected, _segments(magnitudes)[:1], report)
    assert len(receipts) == 1
    # Mutant: the first segment's allowance (also the maximum) is used for every segment.
    first_for_all = [(start, stop, magnitudes[0], scale)
                     for (start, stop), scale in zip(GROUPED_ROWS, GROUPED_SCALES)]
    assert max(magnitudes) == magnitudes[0]
    receipts = _run_grouped(actual, expected, first_for_all, report)
    assert receipts[1]["operand_magnitude_upper"] == magnitudes[0]
    # The borrowed allowance admits the error that the segment's own allowance refuses.
    assert atols[1] < receipts[1]["max_abs_error"] <= receipts[1]["atol"]
    # The opposite borrowing refuses a correct result: the last allowance on the first segment.
    actual, expected = _grouped_outputs()
    actual[:3] = 0.01
    assert atols[0] > 0.01 > atols[1]
    _run_grouped(actual, expected, _segments(magnitudes), report)
    last_for_all = [(start, stop, magnitudes[1], scale)
                    for (start, stop), scale in zip(GROUPED_ROWS, GROUPED_SCALES)]
    with _refusal_for(atols[1]):
        _run_grouped(actual, expected, last_for_all, report)
    # Mutant: the first segment's check is omitted while only it is corrupted.
    actual, expected = _grouped_outputs()
    actual[0, 0] = _float32_beyond(atols[0])
    with _refusal_for(atols[0]):
        _run_grouped(actual, expected, _segments(magnitudes), report)
    receipts = _run_grouped(actual, expected, _segments(magnitudes)[1:], report)
    assert len(receipts) == 1


def test_open_ended_last_segment_keeps_the_run_correctness_slice_semantics():
    """run_correctness hands the helper its last segment as rows [3:], so stop is None."""
    magnitudes = LAYOUTS["large_first"]
    report = _grouped_report()
    open_ended = [(0, 3, magnitudes[0], GROUPED_SCALES[0]), (3, None, magnitudes[1], GROUPED_SCALES[1])]
    actual, expected = _grouped_outputs()
    receipts = _run_grouped(actual, expected, open_ended, report)
    assert receipts == _run_grouped(actual, expected, _segments(magnitudes), report)
    _assert_unqualified(receipts)
    # Like actual[3:] against expected[3:], a row-count mismatch still reaches the shape refusal.
    longer = torch.zeros((GROUPED_ROWS[-1][1] + 1, GROUPED_COLS), dtype=torch.float32)
    for mismatched in ((longer, expected), (actual, longer)):
        with pytest.raises(ValueError, match="equal nonempty output shapes"):
            _run_grouped(*mismatched, open_ended, report)
    # Mutant: a closed range ending at the shorter output clips both outputs to the same
    # rows, so the extra row is never compared and the mismatch goes unnoticed.
    assert len(_run_grouped(longer, expected, _segments(magnitudes), report)) == 2


def test_run_correctness_resolves_the_helper_that_the_controls_call():
    """run_correctness needs CUDA, HOST_NAME and an attestation file, so it cannot run on CPU.

    This guards the wiring instead: its code resolves the module-level helper by name, so
    the grouped comparison is not inlined again beside the helper the controls exercise.
    """
    adapter = _load_geometry_adapter()
    assert callable(adapter._check_grouped_segments)
    assert "_check_grouped_segments" in adapter.run_correctness.__code__.co_names
