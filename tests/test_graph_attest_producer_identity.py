"""Real Git objects, edited producer bytes and a self-supplied reviewed label.

Only unrelated runtime/image/queue inputs are mocked; no GPU work occurs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/graph_attest_702"
sys.path.insert(0, str(HERE))
import managed_window
import tp2_recipe as recipe
import watch_window_queue
import window_driver as driver

FILES = ("managed_window.py", "tp2_recipe.py", "rank_window.py", "window_driver.py", "submit.py",
         "watch_window_queue.py", "arm_tp2.sh", "drive_tp2.sh", "plan-artifact.txt",
         "eager_benchmark.py", "plan-eager-window4.txt")


class ReachedAdmission(RuntimeError):
    pass


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT).strip()


def disk_digest(root):
    return hashlib.sha256("".join(f"{hashlib.sha256((root / 'experiments/graph_attest_702' / name).read_bytes()).hexdigest()}  {name}\n"
                                 for name in FILES).encode()).hexdigest()


def make_producer(tmp_path, *, message="reviewed producer"):
    root = tmp_path / "producer"
    (root / "experiments/graph_attest_702").mkdir(parents=True)
    for name in FILES:
        (root / "experiments/graph_attest_702" / name).write_bytes((HERE / name).read_bytes())
    git(root, "init", "-q")
    git(root, "config", "user.name", "CPU fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "add", ".")
    git(root, "commit", "-qm", message)
    return root, git(root, "rev-parse", "HEAD")


def fixture_inputs(monkeypatch, tmp_path, producer, reviewed):
    config = dict(ts=str(tmp_path / "runtime"), artifact=str(tmp_path / "control"),
                  receipts=str(tmp_path / "fixture-root" / "arms"),
                  fabric="socket", source_commit="c" * 40, src_sha256="d" * 64,
                  producer_commit=reviewed, producer_sha256=disk_digest(producer))
    env = dict(TS=config["ts"], ARTIFACT=config["artifact"], RECEIPTS=config["receipts"], FABRIC="socket",
               SOURCE_COMMIT=config["source_commit"], SOURCE_SHA256=config["src_sha256"],
               PRODUCER_COMMIT=reviewed, PRODUCER_SHA256=config["producer_sha256"])
    monkeypatch.setattr(driver, "__file__", str(producer / "experiments/graph_attest_702/window_driver.py"))
    monkeypatch.setattr(recipe, "inputs", lambda *a, **kw: config)
    monkeypatch.setattr(recipe, "plan", lambda p, **kw: [dict(arm="aE1", fabric="socket")])
    monkeypatch.setattr(driver, "rows", lambda *a, **kw: [])
    def reached(*args, **kwargs):
        raise ReachedAdmission("unreviewed producer reached admission/queue inspection")
    monkeypatch.setattr(driver, "diskcheck", reached)
    monkeypatch.setattr(watch_window_queue, "inspect", reached)
    root = tmp_path / "submitted"
    root.mkdir()
    (root / "inputs.json").write_text(json.dumps(dict(config=config, predecessors=[], census_action_keys=["1" * 64, "2" * 64])))
    (root / "manifest.json").write_text("[]")
    reviews = tmp_path / "reviews.json"
    reviews.write_text(json.dumps({who: dict(verdict="APPROVE", head_sha=reviewed) for who in ("parent", "D5")}))
    return config, env, root, reviews


@pytest.mark.parametrize("entry", ["prepare", "submit"])
@pytest.mark.parametrize("attack", ["dirty-edited-code", "different-unreviewed-head"])
def test_reviewed_label_and_forged_disk_digest_do_not_authorize_changed_code(tmp_path, monkeypatch, entry, attack):
    producer, reviewed = make_producer(tmp_path)
    changed = producer / "experiments/graph_attest_702/rank_window.py"
    changed.write_text(changed.read_text() + "\n# unreviewed execution change\n")
    if attack == "different-unreviewed-head":
        git(producer, "add", ".")
        git(producer, "commit", "-qm", "unreviewed change")
    config, env, root, reviews = fixture_inputs(monkeypatch, tmp_path, producer, reviewed)
    # This digest is deliberately self-supplied from the edited/unreviewed files.
    assert env["PRODUCER_SHA256"] == disk_digest(producer)
    with pytest.raises(managed_window.Refused, match="producer"):
        if entry == "prepare":
            driver.prepare(Path(env["RECEIPTS"]).parent, tmp_path / "unused", env, None, ["1" * 64, "2" * 64])
        else:
            driver.submit(root, reviews)


@pytest.mark.parametrize("entry", ["prepare", "submit"])
def test_clean_exact_reviewed_producer_reaches_the_next_real_gate(tmp_path, monkeypatch, entry):
    producer, reviewed = make_producer(tmp_path)
    config, env, root, reviews = fixture_inputs(monkeypatch, tmp_path, producer, reviewed)
    if entry == "prepare":
        # A portable local fixture reaches the production shared-storage boundary
        # after its real reviewed-producer checks; it must not impersonate a box.
        with pytest.raises(managed_window.Refused, match="rendezvous must be shared"):
            driver.prepare(Path(env["RECEIPTS"]).parent, tmp_path / "unused", env, None, ["1" * 64, "2" * 64])
    else:
        with pytest.raises(ReachedAdmission):
            driver.submit(root, reviews)

def test_pb_synthetic_head_keeps_reviewed_object_and_identical_clean_producer(tmp_path):
    producer, reviewed = make_producer(tmp_path)
    (producer / ".pb-closure.json").write_text("{}")
    git(producer, "add", ".pb-closure.json")
    git(producer, "commit", "-qm", "PB synthetic snapshot")
    assert git(producer, "rev-parse", "HEAD") != reviewed
    assert recipe.require_producer(producer, reviewed, disk_digest(producer), exact_head=False) == disk_digest(producer)
    with pytest.raises(managed_window.Refused, match="HEAD"):
        recipe.require_producer(producer, reviewed, disk_digest(producer), exact_head=True)


def test_parentless_materialization_without_reviewed_git_object_refuses_by_name(tmp_path):
    original, reviewed = make_producer(tmp_path / "original")
    # Never create the reviewed object here: amending would retain it in the object database.
    producer, other = make_producer(tmp_path / "parentless", message="parentless materialization")
    with pytest.raises(managed_window.Refused, match="reviewed Git object unavailable"):
        recipe.require_producer(producer, reviewed, disk_digest(producer), exact_head=False)
