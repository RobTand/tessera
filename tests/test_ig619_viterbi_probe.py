"""The tessera#619 probe states a matched pair, and refuses a pair that is not one.

``experiments/ig619_viterbi_power.py`` times the window Viterbi's two schedules
(``TESSERA_WINDOW_BEST_FORM`` unset and ``1``) in ONE process on ONE tensor, so
a box difference cannot sit inside a ratio.  A ratio between two schedules is
only a result if both wrote the same bytes, which the probe has to decide, not
the person reading its receipt: the earlier probe recorded ``byte_identical:
false`` and still exited 0, so a faster wrong answer was quotable.

What is pinned here is the bookkeeping that carries the claim, on CPU and
without a device: the arm order, the byte comparison and its refusal, the exact
split of units between threads, the ratios, and the two instrument readings
(box Netdata window, in-process sampler).  The GPU run is what measures.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "experiments" / "ig619_viterbi_power.py"
sys.path.insert(0, str(ROOT / "experiments"))

import ig619_viterbi_power as probe  # noqa: E402


@pytest.mark.parametrize("forms", [("front", "best"), ("a", "b", "c"), ("only",)])
def test_abba_runs_every_form_twice_and_ends_where_it_began(forms):
    """Drift in a box's clocks or temperature must fall on both forms alike."""
    order = probe.arm_schedule(forms)
    assert sorted(order) == sorted(list(forms) * 2)
    assert order[0] == order[-1] == forms[0]
    assert order == list(forms) + list(reversed(forms))


def _arm(name, blobs):
    return {"name": name, "sha256": list(blobs)}


def test_identical_arms_pass():
    blobs = ["a" * 64, "b" * 64, "c" * 64]
    probe.check_identity([_arm("front", blobs), _arm("best", blobs), _arm("best/2t", blobs)])


def test_a_differing_blob_is_refused_by_arm_and_unit():
    good = ["a" * 64, "b" * 64, "c" * 64]
    bad = ["a" * 64, "b" * 64, "d" * 64]
    with pytest.raises(probe.IdentityError) as caught:
        probe.check_identity([_arm("front", good), _arm("best", good), _arm("best/2t", bad)])
    text = str(caught.value)
    assert "best/2t" in text and "unit 2" in text


def test_a_missing_unit_is_a_difference_not_a_shorter_pass():
    blobs = ["a" * 64, "b" * 64]
    with pytest.raises(probe.IdentityError) as caught:
        probe.check_identity([_arm("front", blobs), _arm("best/2t", blobs[:1])])
    assert "best/2t" in str(caught.value)


def test_an_empty_comparison_is_refused():
    """No arms, or an arm that encoded nothing, proves nothing."""
    with pytest.raises(probe.IdentityError):
        probe.check_identity([])
    with pytest.raises(probe.IdentityError):
        probe.check_identity([_arm("front", [])])


@pytest.mark.parametrize("units,threads", [(8, 2), (6, 3), (4, 1), (12, 4)])
def test_split_units_covers_every_unit_once_in_order(units, threads):
    parts = probe.split_units(units, threads)
    assert len(parts) == threads
    assert sorted(i for part in parts for i in part) == list(range(units))
    assert {len(part) for part in parts} == {units // threads}
    assert all(part == sorted(part) for part in parts)


@pytest.mark.parametrize("units,threads", [(7, 2), (3, 4), (0, 2), (4, 0)])
def test_split_units_refuses_a_remainder_by_name(units, threads):
    """A leftover unit would make the arms encode different work."""
    with pytest.raises(ValueError) as caught:
        probe.split_units(units, threads)
    assert str(units) in str(caught.value) or str(threads) in str(caught.value)


def test_compare_reads_candidate_against_base_in_that_direction():
    # Attempt 2's own row for BF16 at q256 1024: 0.2116 -> 0.5962 units/s,
    # 325.9 -> 144.2 J/unit.  Faster is above 1, cheaper is below 1.
    out = probe.compare({"units_per_s": 0.2116, "j_per_unit": 325.9},
                        {"units_per_s": 0.5962, "j_per_unit": 144.2})
    assert out["speedup"] == pytest.approx(0.5962 / 0.2116)
    assert out["joule_ratio"] == pytest.approx(144.2 / 325.9)
    assert out["speedup"] > 1 > out["joule_ratio"]


def test_compare_does_not_invent_a_joule_ratio_without_energy():
    out = probe.compare({"units_per_s": 1.0, "j_per_unit": None},
                        {"units_per_s": 2.0, "j_per_unit": 3.0})
    assert out["speedup"] == pytest.approx(2.0)
    assert out["joule_ratio"] is None


def _power_series(mean, groups, accepted_s):
    return {
        "nvidia_smi.gpu_power_draw": {
            "stats": {"power_draw": {"points": groups, "min": mean - 1, "max": mean + 1,
                                     "mean": mean, "median": mean}},
            "coverage": {"accepted_groups": groups, "accepted_coverage_s": accepted_s,
                         "unobserved_requested_s": 0.0},
        }
    }


def test_netdata_summary_divides_by_the_envelope():
    out = probe.netdata_power_summary(_power_series(70.0, 3, 30.0))
    assert out["mean_w"] == 70.0
    assert out["mean_fraction_of_envelope"] == pytest.approx(70.0 / probe.ENVELOPE_W)
    assert out["accepted_groups"] == 3 and out["accepted_coverage_s"] == 30.0


def test_netdata_summary_says_unobserved_not_zero_watts():
    """An empty window is no evidence; it is never a reading of 0 W."""
    series = {"nvidia_smi.gpu_power_draw": {
        "stats": {"power_draw": {"points": 0, "note": "no samples in this window"}},
        "coverage": {"accepted_groups": 0, "accepted_coverage_s": 0, "unobserved_requested_s": 12.0}}}
    out = probe.netdata_power_summary(series)
    assert out["mean_w"] is None and out["accepted_groups"] == 0
    assert "mean_fraction_of_envelope" not in out or out["mean_fraction_of_envelope"] is None


def test_netdata_summary_carries_a_failed_read_as_an_error():
    out = probe.netdata_power_summary({"nvidia_smi.gpu_power_draw": {"error": "URLError: down"}})
    assert out["error"] == "URLError: down" and out.get("mean_w") is None


def test_sampler_window_reports_points_and_never_a_zero_mean():
    sampler = probe.Power()
    sampler._samples = [(10.0, 60.0), (10.5, 70.0), (11.0, 80.0), (30.0, 5.0)]
    inside = sampler.window(10.0, 11.0)
    assert inside["points"] == 3 and inside["mean_w"] == 70.0
    assert inside["mean_fraction_of_envelope"] == pytest.approx(70.0 / probe.ENVELOPE_W)
    assert "mean_sm_mhz" not in inside, "no clock samples, no clock reading"
    empty = sampler.window(20.0, 21.0)
    assert empty["points"] == 0 and "mean_w" not in empty


def test_sampler_window_reports_the_sm_clock_beside_the_power():
    """A power cap shows as a clock before it shows as a lower wattage."""
    sampler = probe.Power()
    sampler._samples = [(10.0, 60.0), (10.5, 70.0), (11.0, 80.0)]
    sampler._clocks = [(10.0, 2400.0), (10.5, 2200.0), (11.0, 2000.0), (30.0, 100.0)]
    out = sampler.window(10.0, 11.0)
    assert out["mean_sm_mhz"] == 2200 and out["min_sm_mhz"] == 2000


def test_units_must_split_evenly_before_anything_touches_a_device():
    """The refusal comes from the argument parser, so no CUDA is needed to see it."""
    with pytest.raises(SystemExit) as caught:
        probe.parse_args(["--units", "7", "--threads", "2"])
    assert caught.value.code != 0
    ok = probe.parse_args(["--units", "8", "--threads", "2"])
    assert ok.units == 8 and ok.threads == 2


def _timed_arm(name, form, units, wall, watts, threads=1, rate_streams=True):
    return {"name": name, "form": form, "threads": threads, "rate_streams": rate_streams,
            "units": units, "wall_s": wall, "units_per_s": units / wall,
            "power": {"points": 10, "mean_w": watts},
            "j_per_unit": watts * wall / units}


def test_pool_is_total_over_total_not_a_mean_of_ratios():
    short = _timed_arm("x#1", "front", units=4, wall=20.0, watts=60.0)
    long = _timed_arm("x#2", "front", units=8, wall=40.0, watts=80.0)
    out = probe.pool([short, long])
    assert out["units"] == 12 and out["wall_s"] == 60.0
    assert out["units_per_s"] == pytest.approx(12 / 60.0)
    # (60 W * 20 s + 80 W * 40 s) / 12 units; the mean of the two arms' own
    # J/unit (300 and 400) would say 350 and weight the short arm like the long.
    assert out["j_per_unit"] == pytest.approx((60 * 20 + 80 * 40) / 12)
    assert out["mean_w"] == pytest.approx((60 * 20 + 80 * 40) / 60)


def test_pool_has_no_joules_when_any_arm_had_no_power_reading():
    blind = _timed_arm("x#1", "front", units=4, wall=20.0, watts=60.0)
    blind["power"] = {"points": 0}
    out = probe.pool([blind, _timed_arm("x#2", "front", units=4, wall=20.0, watts=60.0)])
    assert out["j_per_unit"] is None and out["mean_w"] is None
    assert out["units_per_s"] == pytest.approx(8 / 40.0)


def test_comparisons_read_each_axis_in_its_own_direction():
    arms = [_timed_arm("front#1", "front", 12, 60.0, 70.0),
            _timed_arm("best#1", "best", 12, 20.0, 85.0),
            _timed_arm("best#2", "best", 12, 20.0, 85.0),
            _timed_arm("front#2", "front", 12, 60.0, 70.0),
            _timed_arm("front/rs0", "front", 12, 61.0, 69.0, rate_streams=False),
            _timed_arm("best/rs0", "best", 12, 21.0, 84.0, rate_streams=False),
            _timed_arm("front/2t", "front", 12, 40.0, 75.0, threads=2),
            _timed_arm("best/2t", "best", 12, 20.0, 85.0, threads=2)]
    out = probe._comparisons(arms, threads=2)
    assert out["best_over_front"]["speedup"] == pytest.approx(3.0)
    assert out["best_over_front"]["joule_ratio"] < 1
    # More threads: faster is above 1 whatever the form.
    assert out["threads_over_one"]["front"]["speedup"] == pytest.approx(1.5)
    assert out["threads_over_one"]["best"]["speedup"] == pytest.approx(1.0)
    # Rate streams ON against OFF: the pooled ON arms are the candidate.
    assert out["rate_streams_on_over_off"]["front"]["speedup"] == pytest.approx(61.0 / 60.0)
    assert out["repeat_spread"]["front"]["speedup"] == pytest.approx(1.0)


def test_comparisons_leave_out_rate_streams_for_a_single_rate_rung():
    arms = [_timed_arm(n, f, 12, w, 70.0) for n, f, w in
            (("front#1", "front", 60.0), ("best#1", "best", 20.0), ("best#2", "best", 20.0),
             ("front#2", "front", 60.0), ("front/2t", "front", 40.0), ("best/2t", "best", 20.0))]
    assert "rate_streams_on_over_off" not in probe._comparisons(arms, threads=2)


def test_a_window_shorter_than_one_netdata_second_is_not_sent():
    """No network is touched: the answer is the refusal itself."""
    out = probe.netdata_power("127.0.0.1", 100.2, 100.9)
    assert "error" in out and out.get("mean_w") is None


@pytest.mark.parametrize("line,want", [
    ("70.31, 2411", (70.31, 2411.0)),
    ("5.29, 208\n", (5.29, 208.0)),
    ("5.29", (5.29, None)),                 # the power-only query
    ("66.0, [N/A]", (66.0, None)),           # a clock the driver will not say
    ("[N/A], 2411", (None, None)),           # no power, no sample
    ("", (None, None)),
    ("garbage, 12", (None, None)),
])
def test_one_sampler_line_is_a_power_and_maybe_a_clock(line, want):
    assert probe.parse_sample(line) == want


def _receipt():
    arms = [_timed_arm("front#1", "front", 12, 60.0, 70.0),
            _timed_arm("best#1", "best", 12, 20.0, 85.0),
            _timed_arm("best#2", "best", 12, 20.0, 85.0),
            _timed_arm("front#2", "front", 12, 60.0, 70.0),
            _timed_arm("front/2t", "front", 12, 40.0, 75.0, threads=2),
            _timed_arm("best/2t", "best", 12, 20.0, 85.0, threads=2)]
    for arm in arms:
        arm["power"].update(mean_fraction_of_envelope=arm["power"]["mean_w"] / 140.0,
                            mean_sm_mhz=2400)
    arms[0]["netdata"] = {"mean_w": 69.5, "accepted_groups": 5}
    rung = {"grid": "bf16", "q256": 1024, "rows": 2048, "cols": 4096, "units": 12,
            "arms": arms, "identity": {"ok": True, "arms": 6, "units": 12},
            "comparisons": probe._comparisons(arms, threads=2),
            "plans": [{"best_form": True, "rate": 4, "steps": 2048, "width": 512,
                       "batches": 8, "grid": [16, 256]}],
            "profile": {"front": {"wall_s": 4.7, "device_s": 4.62, "device_over_wall": 0.98,
                                  "top": [{"name": "_step", "count": 1048576,
                                           "us_per_launch": 4.37}]},
                        "best": None}}
    return {"host": "sparky", "device": "NVIDIA GB10", "torch": "2.11.0+cu130",
            "head": "abcdef0123456789", "idle_power": {"mean_w": 5.1}, "rungs": [rung]}


def test_the_rendered_tables_carry_the_receipt_numbers():
    text = probe.render_tables(_receipt())
    assert "bf16 q256 1024, 12 units of 2048x4096" in text
    for arm in ("front#1", "best#2", "front/2t"):
        assert f"| {arm} |" in text
    assert "speedup 3.000x" in text           # best over front, pooled ABBA
    assert "| _step |" in text                  # the profile row is there
    assert "1048576" in text and "[16, 256]" in text
    assert "69.50" in text, "the Netdata reading sits beside the in-process one"


def test_rendering_survives_a_receipt_with_missing_readings():
    rec = _receipt()
    rec["rungs"][0]["comparisons"] = None
    rec["rungs"][0]["profile"] = None
    rec["rungs"][0]["arms"][1]["j_per_unit"] = None
    rec["rungs"][0]["arms"][1]["power"] = {"points": 0}
    text = probe.render_tables(rec)
    assert "| best#1 |" in text and "-" in text


def test_a_stale_tessera_earlier_on_the_path_cannot_stand_in_for_the_snapshot(tmp_path):
    """Its import root comes from its own location.  The first three arms of
    tessera#619 failed because a stale venv copy of ``tessera`` was found
    before the snapshot's; the probe must find its own tree first whatever the
    interpreter's path holds."""
    stale = tmp_path / "stale" / "tessera"
    stale.mkdir(parents=True)
    (stale / "__init__.py").write_text("")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(tmp_path / "stale"), str(ROOT / "experiments")])}
    code = ("import importlib.util, runpy; "
            f"runpy.run_path({str(PROBE)!r}, run_name='probe'); "
            "print(importlib.util.find_spec('tessera').origin)")
    found = subprocess.run([sys.executable, "-c", code], env=env, check=True,
                           capture_output=True, text=True).stdout.strip()
    assert Path(found) == ROOT / "src" / "tessera" / "__init__.py"
