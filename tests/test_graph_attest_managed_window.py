"""CPU-only protocol tests; simulated claims are not live two-host qualification."""
from __future__ import annotations

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
                              stopped_unix=time.time(), released=True, retired=False, settled=True)))


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


@pytest.mark.parametrize("fault", ["none", "wrong_scope", "logs", "remove", "copy"])
def test_production_local_cleanup_surfaces_errors_without_foreign_deletion(tmp_path, fault):
    import rank_window
    owned = identity()
    adapter = rank_window.LocalArm.__new__(rank_window.LocalArm)
    adapter.identity, adapter.rank = owned, 0
    adapter.envelope = window.Envelope(time.time() + 8, cleanup_seconds=1)
    adapter.work, adapter.rdv = tmp_path / "work", tmp_path / "rdv"
    adapter.work.mkdir(); adapter.rdv.mkdir()
    out = adapter.work / "aE1"
    out.mkdir()
    (out / "partial.txt").write_text("partial evidence")
    (adapter.work / "memwatch.txt").write_text("simulated floor samples")
    foreign = tmp_path / "foreign-artifact"
    foreign.write_text("untouched")
    cid = "a" * 64
    (adapter.work / "aE1.cid").write_text(cid)
    adapter.active = dict(cid=cid, cidfile=str(adapter.work / "aE1.cid"), out=str(out), name="owned-only")
    state = dict(present=True, removal_called=False)
    labels = {"prismabuild.scope": owned["scope_id"], "prismabuild.action": owned["container_owner"],
              "org.prismaquant.graph-window": owned["run_id"], "org.prismaquant.attempt": owned["nonce"]}
    if fault == "wrong_scope": labels["prismabuild.scope"] = "foreign.slice"
    def command(argv, **kwargs):
        # Actual bounded CPU command exits exercise the production cleanup control flow.
        if argv[0] in ("cp", "sha256sum"):
            if fault == "copy" and argv[0] == "cp": argv = ["cp", str(tmp_path / "absent-source"), argv[-1]]
            return adapter.envelope.run(argv, **kwargs)
        if argv[:2] == ["docker", "inspect"]:
            payload = json.dumps([dict(Id=cid, Config=dict(Labels=labels),
                      HostConfig=dict(CgroupParent=owned["scope_id"]), State=dict(Pid=999, Running=True))])
        elif argv[:2] == ["docker", "logs"]:
            if fault == "logs":
                return adapter.envelope.run([sys.executable, "-c", "raise SystemExit(7)"], **kwargs)
            payload = "simulated engine log"
        elif argv[:3] == ["docker", "rm", "-f"]:
            assert argv[-1] == cid
            state["removal_called"] = True
            if fault == "remove":
                return adapter.envelope.run([sys.executable, "-c", "raise SystemExit(7)"], **kwargs)
            state["present"] = False
            payload = cid
        elif argv[:2] == ["docker", "ps"]:
            payload = cid if state["present"] else ""
        elif argv[0] == "nvidia-smi":
            payload = ""  # simulated absence, never a physical GPU observation
        else:
            raise AssertionError(argv)
        return adapter.envelope.run([sys.executable, "-c", "import sys;sys.stdout.write(sys.argv[1])", payload], **kwargs)
    adapter.command = command
    if fault == "none":
        assert adapter.cleanup(dict(arm="aE1"))["containers_empty"]
    else:
        with pytest.raises(window.Refused):
            adapter.cleanup(dict(arm="aE1"))
    assert foreign.read_text() == "untouched"
    assert (out / "partial.txt").read_text() == "partial evidence"
    assert state["removal_called"] == (fault != "wrong_scope")
    if fault not in ("remove", "wrong_scope"):
        assert state["present"] is False


def test_source_digest_keeps_pr930_relative_sha256sum_spelling():
    import tp2_recipe
    done = subprocess.run(["bash", "-c", "find src -type f -name '*.py' | sort | xargs sha256sum | sha256sum"],
                          cwd=ROOT, capture_output=True, text=True, check=True, timeout=10)
    assert tp2_recipe.src_sha(ROOT) == done.stdout.split()[0]


def test_simulated_outcomes_never_release_live_window_ownership(tmp_path):
    import window_driver
    window.atomic_json(tmp_path / "outcome-rank0.json", dict(simulation=True))
    result = window_driver.collect(tmp_path, tmp_path / "queue")
    assert result["ownership_released"] is False
    assert "simulated CPU" in result["error"]


@pytest.mark.parametrize("field,value", [("released", 1), ("retired", None), ("settled", "yes")])
def test_broker_release_flags_are_exact_typed_proof(field, value):
    owned = identity()
    value_record = terminal(owned)
    value_record["resource_scope_cleanup"]["export"][field] = value
    with pytest.raises(window.Refused, match="broker scope cleanup"):
        window.terminal_cleanup(owned, value_record)

@pytest.mark.parametrize("stage", ["bind", "ready"] )
def test_health_floor_is_sampled_while_waiting_for_a_peer(tmp_path, stage):
    rdv, queue = tmp_path / "rdv", tmp_path / "queue"
    rdv.mkdir()
    owned = identity()
    claim(queue, owned)
    meeting = window.Rendezvous(rdv, owned, queue, window.Envelope(time.time() + 8, cleanup_seconds=1))
    if stage == "ready":
        peer = identity(1)
        claim(queue, peer)
        window.atomic_json(rdv / "rank1.json", peer)
        meeting.bind_peer()
    def floor_breach():
        raise window.Refused("simulated 16 GiB floor breach during barrier")
    with pytest.raises(window.Refused, match="floor breach during barrier"):
        if stage == "bind":
            meeting.bind_peer(tick=floor_breach)
        else:
            meeting.wait("ready", tick=floor_breach)

@pytest.mark.parametrize("eager", ["1", "0"])
def test_local_container_and_entrypoint_shells_compile_without_launching(tmp_path, eager):
    import tp2_recipe as recipe
    arm = recipe.arm_settings("aE1" if eager == "1" else "aGR",
                              dict(EAGER=eager, SPEC_JSON=json.dumps(recipe.MTP),
                                   COMPILATION_JSON="" if eager == "1" else json.dumps(recipe.GRAPH)))
    argv = recipe.container(dict(ts=str(ROOT), artifact=recipe.CONTROL, image=recipe.IMAGE, fabric="socket"),
                            arm, identity(), tmp_path / "out", tmp_path / "ext", tmp_path / "cid", {})
    subprocess.run(["bash", "-n", "-c", argv[-1]], check=True, timeout=10)
    for name in ("arm_tp2.sh", "drive_tp2.sh"):
        subprocess.run(["bash", "-n", str(HERE / name)], check=True, timeout=10)

def test_peer_wait_is_3600_seconds_and_does_not_inherit_the_live_window(tmp_path):
    owned = identity()
    owned["claimed_unix"] = time.time() - 3601
    queue, rdv = tmp_path / "queue", tmp_path / "rdv"
    rdv.mkdir(); claim(queue, owned)
    meeting = window.Rendezvous(rdv, owned, queue, window.Envelope(time.time() + 8, cleanup_seconds=1))
    with pytest.raises(TimeoutError, match="3600-second peer"):
        meeting.bind_peer()
