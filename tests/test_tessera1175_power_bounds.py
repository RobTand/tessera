"""Rerun check for the tessera#1175 paired power receipt.

The receipt JSON holds raw paired series plus a derived block. This
test recomputes every derived number from the raw block and checks
the file is self-consistent. It also checks the acceptance shape:
one shared UTC bound set, both series, and a clock delta.
"""
import json
from pathlib import Path

RECEIPT = (Path(__file__).resolve().parents[1] / "docs" / "measurements"
           / "tessera1175-paired-power-20261010.json")

JOULE_BOUND_PCT = 1.5


def load():
    return json.loads(RECEIPT.read_text())


def trapz(samples):
    total = 0.0
    for (a, wa), (b, wb) in zip(sorted(samples), sorted(samples)[1:]):
        total += (wa + wb) / 2.0 * (b - a)
    return total


def window_energy(fast, a, b):
    pts = [(t, w) for t, w in sorted(fast) if a - 0.2 <= t <= b + 0.2]
    total = 0.0
    for (x, wx), (y, wy) in zip(pts, pts[1:]):
        xx, yy = max(x, a), min(y, b)
        if yy > xx:
            total += (wx + wy) / 2.0 * (yy - xx)
    return total


def median(xs):
    ordered = sorted(xs)
    return ordered[len(ordered) // 2]


def primary():
    return load()["primary"]


def test_shared_bound_set_holds_both_series():
    raw = primary()["raw"]
    t0, t1 = raw["interval_unix"]
    assert t1 - t0 > 60.0
    fast = raw["fast_power_samples"]
    assert len(fast) > 100
    assert all(t0 <= t <= t1 for t, _ in fast)
    assert all(w > 0 for _, w in fast)
    for box in ("sparky", "sparklina"):
        entry = raw["netdata"][box]
        assert entry["ok"] is True
        query = entry["query"]
        assert "after=" in query and "before=" in query
        response = entry["raw_response"]
        assert (entry["returned_view"] == response["view"])
        data = response["result"]["data"]
        assert len(data) > 10


def test_clock_delta_is_known_on_both_hosts():
    raw = primary()["raw"]
    for box in ("sparky", "sparklina"):
        for tag in ("clock_delta_pre", "clock_delta_post"):
            probes = [p for p in raw[tag][box] if p.get("ok")]
            assert len(probes) >= 3
            assert all(p["rtt_s"] < 0.1 for p in probes)
    derived = primary()["derived"]["clock_delta"]
    assert abs(derived["sparklina_clock_delta_pre_median_s"]
               - derived["sparky_clock_delta_pre_median_s"]) < 0.05
    assert abs(derived["sparklina_clock_delta_post_median_s"]
               - derived["sparky_clock_delta_post_median_s"]) < 0.05


def test_derived_block_recomputes_from_raw():
    prim = primary()
    raw = prim["raw"]
    derived = prim["derived"]
    t0, t1 = raw["interval_unix"]
    assert derived["interval_unix"] == [t0, t1]
    assert derived["duration_s"] == t1 - t0
    fast = raw["fast_power_samples"]
    assert derived["fast_n"] == len(fast)
    assert derived["fast_energy_trapz_j"] == trapz(fast)
    nd = sorted(raw["netdata"]["sparklina"]["raw_response"]
                ["result"]["data"])
    last = nd[-1][0]
    assert derived["netdata_span"] == [nd[0][0], last]
    ef = window_energy(fast, t0, last)
    en = float(sum(r[1][0] for r in nd if t0 <= r[0] <= last))
    assert abs(derived["intersect_energy_fast_j"] - ef) < 1e-6
    assert abs(derived["intersect_energy_netdata_j"] - en) < 1e-9
    want = 100.0 * (en - ef) / ef
    assert abs(derived["intersect_bias_pct"] - want) < 1e-9
    a2 = t0 + 10
    ef2 = window_energy(fast, a2, last)
    en2 = float(sum(r[1][0] for r in nd if a2 <= r[0] <= last))
    assert abs(derived["steady_bias_pct"]
               - 100.0 * (en2 - ef2) / ef2) < 1e-9
    for box in ("sparky", "sparklina"):
        for tag in ("clock_delta_pre", "clock_delta_post"):
            probes = [p["server_minus_client_s"]
                      for p in raw[tag][box] if p.get("ok")]
            key = "%s_%s_median_s" % (box, tag)
            assert derived["clock_delta"][key] == median(probes)


def test_steady_bias_fits_declared_joule_bound():
    derived = primary()["derived"]
    assert abs(derived["steady_bias_pct"]) <= JOULE_BOUND_PCT
    for shift, value in derived["shift_sensitivity_pct"].items():
        assert abs(value) <= JOULE_BOUND_PCT, shift
