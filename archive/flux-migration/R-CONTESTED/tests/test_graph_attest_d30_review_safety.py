"""Behavior regressions for the PR961 safety-review findings; CPU/PB only."""
import json
from pathlib import Path
import pytest
from test_graph_attest_d30_memory_guard import (
    CID, FLOOR, FakeClock, DockerStub, d30_adapter, identity, new_adapter,
    owned_labels, server_adapter, write_claim,
)
import managed_window as window
import rank_window


def test_low_memory_precedes_remote_guard_and_journal(tmp_path, monkeypatch):
    clock = FakeClock()
    adapter = d30_adapter(tmp_path, monkeypatch, read=lambda: 1.99, clock=clock)
    publications = []
    adapter.abort = lambda **fields: publications.append(fields)
    def remote_guard():
        raise OSError("shared queue unavailable")
    adapter.guard = remote_guard
    original = Path.open
    def blocked(path, *args, **kwargs):
        if path.is_relative_to(adapter.rdv):
            raise OSError("shared journal unavailable")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", blocked)
    with pytest.raises(window.Refused, match="below 2 GiB"):
        adapter.tick()
    assert publications and FLOOR in publications[0]["error"]
    assert adapter.memory_summary["minimum_gib"] == 1.99
    assert any("shared journal" in item["error"] for item in adapter.abort_errors)


def test_term_error_stays_secondary_to_original_floor(tmp_path, monkeypatch):
    clock = FakeClock()
    adapter = d30_adapter(tmp_path, monkeypatch, read=lambda: 1.99, clock=clock)
    adapter.active = dict(cid=CID)
    publications = []
    adapter.abort = lambda **fields: publications.append(fields)
    def failed_term(cid):
        raise window.Refused("exact-owned TERM failed")
    adapter._term_server = failed_term
    with pytest.raises(window.Refused, match="below 2 GiB"):
        adapter.tick()
    assert FLOOR in publications[0]["error"]
    assert any("exact-owned TERM failed" in item["error"] for item in adapter.abort_errors)
    assert json.loads((adapter.rdv / "memory-summary-rank0.json").read_text())["minimum_gib"] == 1.99


def test_first_failed_marker_preserves_original_reason_and_timestamp(tmp_path, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(window, "time", clock)
    owned = identity(0)
    owned.update(claimed_unix=clock.time()-5, window_end_unix=clock.time()+600)
    write_claim(tmp_path, owned)
    rdv = tmp_path / "rdv"
    rdv.mkdir()
    meeting = window.Rendezvous(rdv, owned, tmp_path, window.Envelope(clock.time()+600, cleanup_seconds=60))
    meeting.publish("failed", error=FLOOR)
    first = (rdv / "failed-rank0.json").read_bytes()
    clock.advance(1)
    meeting.publish("failed", error="secondary termination fault")
    assert (rdv / "failed-rank0.json").read_bytes() == first


def test_expired_container_grace_still_dispatches_exact_owned_kill(tmp_path, monkeypatch):
    clock = FakeClock()
    stub = DockerStub(owned_labels(identity(0)), dies_on_term=False)
    adapter = server_adapter(tmp_path, monkeypatch, stub, clock=clock)
    adapter.envelope.end = clock.monotonic()+.5
    def normal(argv, **kwargs):
        if clock.monotonic() >= adapter.envelope.end:
            raise TimeoutError("original cleanup envelope expired")
        return stub(argv, **kwargs)
    adapter.command = normal
    def rescue(cid, operation, **kwargs):
        assert cid == CID
        argv = ["docker", "inspect", cid] if operation == "inspect" else ["docker", "kill", "--signal", operation, cid]
        return stub(argv, **kwargs)
    adapter.envelope.run_container_control = rescue
    record = adapter._stop_server(CID)
    assert [event["signal"] for event in record["signals"]] == ["SIGTERM", "SIGKILL"]
    assert record["deadline_shortened_grace"] and not stub.running
    assert all(command[-1] == CID for command in stub.commands)


def test_failed_cleanup_stops_waiting_when_peer_claim_ends(tmp_path, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(window, "time", clock)
    monkeypatch.setattr(rank_window, "time", clock)
    queue, rdv = tmp_path / "queue", tmp_path / "rdv"
    rdv.mkdir()
    ranks = [identity(rank) for rank in (0,1)]
    for owned in ranks:
        owned.update(claimed_unix=clock.time()-5, window_end_unix=clock.time()+600)
        write_claim(queue, owned)
    window.atomic_json(rdv / "rank1.json", ranks[1])
    adapter = new_adapter(tmp_path, ranks[0], 0)
    adapter.config = dict(source_commit="unit", producer_commit="unit", window_mode="graph-control")
    def preflight():
        (queue / "claimed" / (ranks[1]["action_key"]+".json")).unlink()
        raise window.Refused("controlled preflight failure")
    adapter.preflight = preflight
    adapter.cleanup = lambda arm: dict(containers_empty=True, gpu_descendants_empty=True)
    start = clock.monotonic()
    code = rank_window.run_rank(adapter.config, ranks[0], queue, rdv, [dict(arm="unit")], adapter,
                               window.Envelope(clock.time()+600, cleanup_seconds=60), poll_seconds=.2)
    assert code == 1 and clock.monotonic()-start < 1
    outcome = window.read_json(rdv / "outcome-rank0.json")
    assert "controlled preflight failure" in outcome["error"] and outcome["completed_arms"] == []
    assert outcome["peer_cleanup_acknowledgement"]["available"] is False
    assert outcome["window_end_unix"]-outcome["ended_unix"] >= 5
