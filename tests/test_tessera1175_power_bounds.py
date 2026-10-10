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


def fast_window_energy(fast, a, b):
    """Trapezoid fast energy over exactly [a, b]."""
    pts = [(t, w) for t, w in sorted(fast) if a - 0.2 <= t <= b + 0.2]
    total = 0.0
    for (x, wx), (y, wy) in zip(pts, pts[1:]):
        xx, yy = max(x, a), min(y, b)
        if yy > xx:
            total += (wx + wy) / 2.0 * (yy - xx)
    return total


def aligned_bias_pct(fast, netdata_rows, a, b, shift_s):
    """Bias over one true window at clock shift s.

    Netdata stamp t ends its 1 s group (bounded_groups), so whole
    groups within (a, b] span (a, b]. A shift of plus s puts that
    span at true time (a-s, b-s]; fast energy covers [a-s, b-s].
    Both series span the same true seconds at every shift.
    """
    rows = sorted(netdata_rows)
    en = float(sum(v[0] for t, v in rows if a < t <= b))
    ef = fast_window_energy(fast, a - shift_s, b - shift_s)
    return en, ef, 100.0 * (en - ef) / ef


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
    import math
    full_a = math.ceil(t0)
    steady_a = math.ceil(t0 + 10)
    assert derived["intersect_window"] == [full_a, last]
    assert derived["steady_window"] == [steady_a, last]
    en_full, ef_full, want_full = aligned_bias_pct(fast, nd, full_a,
                                                   last, 0.0)
    assert abs(derived["intersect_energy_fast_j"] - ef_full) < 1e-6
    assert abs(derived["intersect_energy_netdata_j"] - en_full) < 1e-9
    assert abs(derived["intersect_bias_pct"] - want_full) < 1e-9
    en, ef, want = aligned_bias_pct(fast, nd, steady_a, last, 0.0)
    assert abs(derived["steady_bias_pct"] - want) < 1e-9
    mean = en_full / (last - full_a)
    assert abs(derived["netdata_in_window_mean_w"] - mean) < 1e-9
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

STEADINESS_GATE_PCT = 3.0


def head_tail_drift_pct(fast, t0, t1):
    """Head mean and tail mean drift over the fast series."""
    ordered = sorted(fast)
    head = [w for t, w in ordered if t0 + 10 <= t <= t0 + 25]
    tail = [w for t, w in ordered if t >= t1 - 15]
    hm = sum(head) / len(head)
    tm = sum(tail) / len(tail)
    return hm, tm, 100.0 * (tm - hm) / hm


def sparklina_rows(raw):
    return raw["netdata"]["sparklina"]["raw_response"]["result"]["data"]


def test_shift_sensitivity_recomputes_from_raw():
    import math
    for name in ("primary", "supporting_shared_run"):
        block = load()[name]
        raw, derived = block["raw"], block["derived"]
        t0, _ = raw["interval_unix"]
        rows = sparklina_rows(raw)
        last = sorted(rows)[-1][0]
        a = math.ceil(t0 + 10)
        assert derived["steady_window"] == [a, last]
        for shift, value in derived["shift_sensitivity_pct"].items():
            _, _, want = aligned_bias_pct(raw["fast_power_samples"],
                                          rows, a, last, float(shift))
            assert abs(value - want) < 1e-9, (name, shift)


def test_shared_utc_bound_set_is_exact():
    from datetime import datetime
    raw = primary()["raw"]
    derived = primary()["derived"]
    t0, t1 = raw["interval_unix"]
    assert derived["interval_unix"] == [t0, t1]
    lo, hi = (datetime.fromisoformat(x) for x in derived["utc_bounds"])
    assert abs(lo.timestamp() - t0) < 1e-3
    assert abs(hi.timestamp() - t1) < 1e-3
    for box in ("sparky", "sparklina"):
        query = raw["netdata"][box]["query"]
        assert ("after=%d" % int(t0)) in query
        assert ("before=%d" % (int(t1) + 1)) in query


def test_steadiness_metric_in_derived_block():
    for name in ("primary", "supporting_shared_run"):
        block = load()[name]
        raw, derived = block["raw"], block["derived"]
        t0, t1 = raw["interval_unix"]
        hm, tm, drift = head_tail_drift_pct(raw["fast_power_samples"],
                                            t0, t1)
        assert abs(derived["fast_head15_mean_w"] - hm) < 1e-9, name
        assert abs(derived["fast_tail15_mean_w"] - tm) < 1e-9, name
        assert abs(derived["steady_drift_pct"] - drift) < 1e-9, name


def test_primary_passes_steadiness_gate_and_shared_fails():
    prim = primary()["derived"]["steady_drift_pct"]
    shared = load()["supporting_shared_run"]["derived"]["steady_drift_pct"]
    assert abs(prim) <= STEADINESS_GATE_PCT
    assert abs(shared) > STEADINESS_GATE_PCT


def test_bound_derivation_covers_primary_shifts():
    derived = primary()["derived"]
    seen = [abs(derived["steady_bias_pct"])]
    seen += [abs(v) for v in derived["shift_sensitivity_pct"].values()]
    assert max(seen) < JOULE_BOUND_PCT
    assert abs(max(seen) - 0.850070778042077) < 1e-9


def test_shared_steady_bias_outside_bound():
    """Unsteady shared run lands just outside the bound.

    The drift gate (-5.30 % vs 3 % scope rule) and the joule bound
    agree at the boundary: steady bias -1.63 % exceeds 1.5 % by
    only 0.13 pp. The margin is thin, so the receipt requires a
    second exclusive capture before rank use.
    """
    shared = load()["supporting_shared_run"]["derived"]
    assert abs(shared["steady_bias_pct"]) > JOULE_BOUND_PCT
    assert abs(shared["steady_bias_pct"] - 1.6269294732745554) < 1e-9

