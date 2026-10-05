"""CPU-only protocol tests; simulated claims are not live two-host qualification."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/graph_attest_702"
sys.path.insert(0, str(HERE))
import managed_window as window


def identity(rank=0, **fields):
    key = str(rank + 1) * 64
    nonce = str(rank + 3) * 32
    return dict(rank=rank, host=window.HOSTS[rank], action_key=key, nonce=nonce,
                scope_id=window.scope_name(key, nonce), container_owner=f"owner-{rank}",
                claimed_unix=time.time() - 1, run_id="fresh-run", input_sha256="a" * 64,
                window_end_unix=time.time() + 10, **fields)


def claim(queue, owned):
    path = queue / "claimed" / (owned["action_key"] + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(action_key=owned["action_key"], claimed_host=owned["host"],
                 claimed_unix=owned["claimed_unix"], container_owner=owned["container_owner"],
                 resource_scope={k: owned[k] for k in ("action_key", "nonce", "scope_id")})
    window.atomic_json(path, value)
    return path


def test_terminal_done_never_authorizes_a_rank(tmp_path):
    owned = identity()
    path = claim(tmp_path, owned)
    (tmp_path / "done").mkdir()
    path.rename(tmp_path / "done" / path.name)
    with pytest.raises(FileNotFoundError):
        window.require_claim(owned, tmp_path)


@pytest.mark.parametrize("field,value", [("nonce", "f" * 32), ("scope_id", "old.slice"),
                                         ("action_key", "f" * 64)])
def test_a_superseded_attempt_is_not_live(tmp_path, field, value):
    owned = identity()
    path = claim(tmp_path, owned)
    changed = json.loads(path.read_text())
    changed["resource_scope"][field] = value
    window.atomic_json(path, changed)
    with pytest.raises(window.Refused, match="live PB"):
        window.require_claim(owned, tmp_path)


def test_rendezvous_is_bound_to_fresh_attempt_and_timestamp_not_mtime(tmp_path):
    queue, rdv = tmp_path / "queue", tmp_path / "rdv"
    rdv.mkdir()
    owned, peer = identity(), identity(1)
    for value in (owned, peer):
        claim(queue, value)
    envelope = window.Envelope(time.time() + 10, cleanup_seconds=1)
    meeting = window.Rendezvous(rdv, owned, queue, envelope)
    window.atomic_json(rdv / "rank1.json", peer)
    meeting.bind_peer()
    path = rdv / "ready-rank1.json"
    window.atomic_json(path, dict(peer, stage="ready", written_unix=peer["claimed_unix"] - 1))
    os.utime(path, None)
    with pytest.raises(window.Refused, match="timestamp"):
        meeting.wait("ready")
    window.atomic_json(path, dict(peer, stage="ready", written_unix=time.time()))
    assert meeting.wait("ready")["nonce"] == peer["nonce"]
    with pytest.raises(window.Refused, match="already used"):
        window.Rendezvous(rdv, owned, queue, envelope)


def test_missing_peer_consumes_rendezvous_budget(tmp_path):
    rdv = tmp_path / "rdv"
    rdv.mkdir()
    owned = identity()
    claim(tmp_path / "queue", owned)
    meeting = window.Rendezvous(rdv, owned, tmp_path / "queue",
                                window.Envelope(time.time() + .15, cleanup_seconds=.02))
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        meeting.bind_peer()
    assert time.monotonic() - started < 1


def test_one_envelope_cannot_reset_for_later_arms():
    envelope = window.Envelope(time.time() + .2, cleanup_seconds=.03)
    end = envelope.end
    envelope.tighten(time.time() + 5400)
    assert envelope.end == end
    time.sleep(.18)
    with pytest.raises(TimeoutError):
        envelope.remaining()
    assert envelope.remaining(cleanup=True) > 0


def test_deadline_kills_real_cpu_subprocess_group_and_retains_output(tmp_path):
    output = tmp_path / "partial.log"
    envelope = window.Envelope(time.time() + 2, cleanup_seconds=.2)
    with output.open("w") as stream, pytest.raises(TimeoutError):
        envelope.run([sys.executable, "-c", "import subprocess,sys,time; "
                      "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
                      "print(p.pid,flush=True);time.sleep(30)"], stdout=stream, limit=.2)
    pid = int(output.read_text().strip())
    stat = Path(f"/proc/{pid}/stat")
    # A reparented zombie is physically dead; it holds no resources or GPU.
    assert not stat.exists() or stat.read_text().split()[2] == "Z"


def terminal(owned):
    return dict(action_key=owned["action_key"], claimed_host=owned["host"],
                claimed_unix=owned["claimed_unix"],
                resource_scope=dict(nonce=owned["nonce"], scope_id=owned["scope_id"]),
                resource_scope_cleanup=dict(complete=True, nonce=owned["nonce"],
                  export=dict(scope_id=owned["scope_id"], empty=True, tickets_pending=False,
                              stopped_unix=time.time())))


@pytest.mark.parametrize("mutation", ["no_export", "tickets", "populated", "nonce", "late", "settle"])
def test_exit_zero_alone_is_not_a_physical_handoff(mutation):
    owned = identity()
    value = terminal(owned)
    cleanup = value["resource_scope_cleanup"]
    if mutation == "no_export": cleanup.pop("export")
    if mutation == "tickets": cleanup["export"]["tickets_pending"] = True
    if mutation == "populated": cleanup["export"]["empty"] = False
    if mutation == "nonce": cleanup["nonce"] = "f" * 32
    if mutation == "late": cleanup["export"]["stopped_unix"] = owned["window_end_unix"] + 1
    if mutation == "settle": cleanup["settle_error"] = "container removal failed"
    with pytest.raises(window.Refused):
        window.terminal_cleanup(owned, value)
    assert window.terminal_cleanup(owned, terminal(owned))["complete"]
