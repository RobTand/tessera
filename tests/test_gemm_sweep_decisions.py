"""The dense-GEMM sweep's decision rules, exercised behaviorally (#850, #806).

The rules live in ``experiments/t8r_speed/gemm_sweep_decisions.py`` -- torch-free,
imported directly by path -- and these tests call them.  Nothing here reads the
sweep's source text: a rule is pinned by what it returns, not by how it is
spelled.
"""
import importlib.util
from pathlib import Path

import pytest

DECISIONS = Path(__file__).resolve().parents[1] / "experiments/t8r_speed/gemm_sweep_decisions.py"


def decisions():
    spec = importlib.util.spec_from_file_location("gemm_sweep_decisions", DECISIONS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the A/B decision reads the conventional median --------------------------


def test_leads_causal_example_emits_no_saving():
    """reference [1,1,1,100,100,100] vs pick [49,50,51,52,53,54]: slower, no saving."""
    ref_med, pick_med, saving_ms = decisions().abba_summary(
        [1.0, 1.0, 1.0, 100.0, 100.0, 100.0], [49.0, 50.0, 51.0, 52.0, 53.0, 54.0])
    assert (ref_med, pick_med) == (50.5, 51.5)
    assert saving_ms is None, "the true medians say the pick is slower: no saving exists"


def test_a_faster_pick_reports_the_true_median_saving():
    ref_med, pick_med, saving_ms = decisions().abba_summary(
        [10.0, 10.0, 10.0, 90.0, 90.0, 90.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert (ref_med, pick_med) == (50.0, 3.5)
    assert saving_ms == 46.5


@pytest.mark.parametrize("samples,expected",
                         [([3, 1, 2], 2), ([4, 1, 3, 2], 2.5), ([2, 1], 1.5), ([7], 7)])
def test_the_median_is_the_conventional_one_for_odd_and_even_groups(samples, expected):
    ref_med, pick_med, saving_ms = decisions().abba_summary(samples, samples)
    assert ref_med == expected
    assert pick_med == expected
    assert saving_ms is None


def test_non_bitwise_summary_uses_the_conventional_median_and_keeps_its_raw_population():
    summary = decisions().non_bitwise_fraction_summary([0.5, 0.1, 0.7, 0.3])
    assert summary["count"] == 4
    assert summary["min"] == 0.1
    assert summary["median"] == 0.4
    assert summary["raw"] == [0.1, 0.3, 0.5, 0.7]


def test_non_bitwise_summary_of_an_empty_population():
    assert decisions().non_bitwise_fraction_summary([]) == {
        "count": 0, "min": None, "median": None, "raw": []}


# --- reference qualification fails closed -------------------------------------


def test_a_reference_qualifies_only_on_match_and_determinism():
    assert decisions().reference_qualified(True, True) is True
    assert decisions().reference_qualified(True, False) is False
    assert decisions().reference_qualified(False, True) is False
    assert decisions().reference_qualified(False, False) is False


def test_an_unqualified_reference_refuses_a_faster_pick():
    assert decisions().admit_pick(False, True, saving_ms=5.0) == \
        (False, "reference kernel does not match the served kernel")
    assert decisions().admit_pick(True, False, saving_ms=5.0) == \
        (False, "reference does not repeat bit-deterministically")


def test_a_qualified_faster_pick_is_admitted():
    assert decisions().admit_pick(True, True, saving_ms=5.0) == (True, None)


def test_without_a_saving_nothing_is_admitted_but_an_unqualified_reference_still_says_why():
    assert decisions().admit_pick(True, True, saving_ms=None) == (False, None)
    assert decisions().admit_pick(False, False, saving_ms=None) == \
        (False, "reference kernel does not match the served kernel")


# --- the synthetic screen touches no capture path ------------------------------


class _Probe:
    """Injected filesystem: every stat/open is recorded; none reach a disk."""

    def __init__(self, tensor):
        self._tensor = tensor
        self.touched = []

    def exists(self, path):
        self.touched.append(("stat", path))
        return True

    def load(self, path):
        self.touched.append(("open", path))
        return self._tensor


class _Grid:
    """The shape/slice surface ``resolve_real_input`` uses of a capture tensor."""

    def __init__(self, rows, cols):
        self.shape = (rows, cols)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            _, cols = key
            return _Grid(self.shape[0], cols.stop if isinstance(cols, slice) else cols)
        if isinstance(key, slice):
            return _Grid(key.stop if key.stop is not None else self.shape[0], self.shape[1])
        raise AssertionError(f"unexpected capture index {key!r}")


CAPTURE = ("kda_hidden.pt", None, "/mutable/capture/kda_hidden.pt")


def test_synthetic_only_never_stats_or_opens_a_capture_path():
    probe = _Probe(_Grid(8192, 4096))
    resolved = decisions().resolve_real_input(False, CAPTURE, 2048, 4096,
                                              probe.exists, probe.load)
    assert resolved == {"real": None, "real_source": "not_measured"}
    assert probe.touched == [], "a synthetic screen must not touch the capture root"


def test_the_default_posture_still_opens_the_recorded_capture():
    probe = _Probe(_Grid(8192, 4096))
    resolved = decisions().resolve_real_input(True, CAPTURE, 2048, 4096,
                                              probe.exists, probe.load)
    assert resolved["real"] is not None
    assert resolved["real_source"] == "kda_hidden.pt"
    assert probe.touched == [("stat", CAPTURE[2]), ("open", CAPTURE[2])]


def test_a_missing_capture_is_named_not_silently_skipped():
    probe = _Probe(_Grid(8192, 4096))
    probe.exists = lambda path: False
    resolved = decisions().resolve_real_input(True, CAPTURE, 2048, 4096,
                                              probe.exists, probe.load)
    assert resolved == {"real": None, "real_source": "missing /mutable/capture/kda_hidden.pt"}


def test_a_capture_that_does_not_cover_the_shape_is_named():
    probe = _Probe(_Grid(512, 4096))
    resolved = decisions().resolve_real_input(True, CAPTURE, 2048, 4096,
                                              probe.exists, probe.load)
    assert resolved == {"real": None,
                        "real_source": "kda_hidden.pt: shape (512, 4096) does not cover [2048, 4096]"}


def test_a_column_slice_is_named_in_the_source():
    capture = ("shared_down_input.pt", 1024, "/mutable/capture/shared_down_input.pt")
    probe = _Probe(_Grid(8192, 4096))
    resolved = decisions().resolve_real_input(True, capture, 2048, 1024,
                                              probe.exists, probe.load)
    assert resolved["real"] is not None
    assert resolved["real_source"] == "shared_down_input.pt (columns 0:1024)"
