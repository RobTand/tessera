"""Pure CPU exact-encoding tests; no GPU or complete-memory qualification."""
import copy
import hashlib
import json
import math

import pytest

from experiments.full_engine_snapshot_codec import (
    CanonicalHistoryPrefix, SnapshotFramePool, canonical_snapshot_digest,
    compact_capture, expand_capture,
)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def test_history_row_reuse_preserves_exact_json_hashes_and_actual_revisions():
    encoder = CanonicalHistoryPrefix()
    first = {"action": "alloc", "time_us": 1, "frames": [{"name": "µ\\\"", "line": 2}]}
    versions = [[], [first], [copy.deepcopy(first)], [copy.deepcopy(first), {"action": "snapshot"}]]
    versions += [[{**first, "time_us": value}, {"action": "snapshot"}]
                 for value in (2, 2.0, True)]
    versions += [[{"action": "alloc", "value": -0.0, "frames": [1]}],
                 [{"action": "alloc", "value": 0.0, "frames": [1.0]}], []]
    observations = []
    for version in versions:
        observation = encoder.observe(version)
        canonical = encoded(version)
        assert observation["history_prefix_sha256"] == hashlib.sha256(canonical).hexdigest()
        assert observation["serialized_history_prefix_bytes"] == len(canonical)
        observations.append(observation)
    assert observations[2]["encoded_rows"] == 0
    assert observations[2]["reused_rows"] == 1
    assert observations[3]["encoded_rows"] == 1
    assert observations[3]["reused_rows"] == 1
    assert all(observations[index]["encoded_rows"] == 1 for index in (4, 5, 6, 8))
    assert observations[-1]["retained_encoded_row_bytes"] == 0
    assert observations[-1]["peak_retained_encoded_row_bytes"] > 0


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_reuse_never_hides_invalid_numbers(value):
    encoder = CanonicalHistoryPrefix()
    encoder.observe([{"time_us": 1}])
    with pytest.raises(ValueError):
        encoder.observe([{"time_us": value}])


def test_row_chunks_share_interned_frame_bytes_across_returned_snapshots():
    pool = SnapshotFramePool()
    encoder = CanonicalHistoryPrefix(pool)
    trace = [{"time_us": index, "frames": [{"name": "same long stack " * 100, "line": 2}]}
             for index in range(20)]
    first = encoder.observe(trace)
    assert len(pool.entries) == 1
    assert len({id(row["frames"]) for row in trace}) == 1
    next_trace = copy.deepcopy(trace)
    next_trace[0]["time_us"] = 100
    second = encoder.observe(next_trace)
    assert second["encoded_rows"] == 1 and second["reused_rows"] == 19
    assert len(pool.entries) == 1
    assert next_trace[0]["frames"] is trace[0]["frames"]
    assert first["retained_encoded_row_bytes"] > 10 * first["unique_frame_encoded_bytes"]


def test_compact_capture_round_trip_and_ledger_refusals_match_exactly():
    from pathlib import Path
    from experiments.full_engine_resources import analyze_engine_resource_ledger
    original = json.loads((Path(__file__).parent / "fixtures/full_engine_resource_ledger.json").read_text())
    wire = compact_capture(original)
    restored = expand_capture(json.loads(encoded(wire)))
    assert encoded(restored) == encoded(original)
    assert canonical_snapshot_digest(restored) == hashlib.sha256(encoded(original)).hexdigest()
    assert analyze_engine_resource_ledger(wire) == analyze_engine_resource_ledger(original)


def test_compaction_writes_repeated_stack_values_once():
    frames = [{"filename": "source.py", "name": "long stack " * 100, "line": 1}]
    raw = {"schema": "tessera.full_engine_resource_capture.v1",
           "torch_snapshot": {"device_traces": [[{"frames": copy.deepcopy(frames)} for _ in range(20)]]},
           "checkpoints": [{"segments": [{"blocks": [{"frames": copy.deepcopy(frames)}]}]} for _ in range(20)]}
    wire = compact_capture(raw)
    assert len(wire["frame_dictionary"]) == 1
    assert len(encoded(wire)) < len(encoded(raw)) / 10
    assert encoded(expand_capture(wire)) == encoded(raw)


@pytest.mark.parametrize("defect", ["hash", "index", "boolean_index", "collision", "missing_table", "malformed_entry"])
def test_compact_frame_damage_is_rejected(defect):
    raw = {"schema": "tessera.full_engine_resource_capture.v1", "value": {"frames": [{"line": 1}]}}
    wire = compact_capture(raw)
    if defect == "hash":
        wire["frame_dictionary"][0]["value"][0]["line"] = 2
    elif defect == "index":
        wire["value"]["frames_ref"] = 10
    elif defect == "boolean_index":
        wire["value"]["frames_ref"] = False
    elif defect == "collision":
        wire["value"]["frames"] = []
    elif defect == "missing_table":
        wire.pop("frame_dictionary")
    else:
        wire["frame_dictionary"][0]["extra"] = 1
    with pytest.raises(ValueError):
        expand_capture(wire)


@pytest.mark.parametrize("value", [None, True, 1, 1.0, -0.0, "µ\n", (1, 2),
                                  {"frames": [{"name": "µ", "line": 1}], "nested": [False, {"a": 0.5}]}])
def test_streamed_canonical_digest_matches_standard_json(value):
    assert canonical_snapshot_digest(value) == hashlib.sha256(encoded(value)).hexdigest()
