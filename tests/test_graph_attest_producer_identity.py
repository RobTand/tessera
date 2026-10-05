"""Real Git objects, edited producer bytes and a self-supplied reviewed label.

Only unrelated runtime/image/queue inputs are mocked; no GPU work occurs.

D32: dev mode is ON unless ``PRISMAQUANT_DEV_MODE`` is exactly ``0``. The
run-identity producer comparisons stamp one ``[DEV-MODE]`` line and continue
with the stored data instead of refusing: ``require_producer`` keeps its
signature and returns the stored expected digest without Git or digest
computes, and ``producer_sha`` reports the stored ``PRODUCER_SHA256`` (or
``NOT_COMPUTED``) without hashing. Certified ``0`` keeps every legacy
refusal, so the producer/Git-object/self-label regressions pin ``0``
explicitly. Unchanged in both modes: the exact executing-code parent/D5
review, the inputs.json/memory-policy own-byte seals and every
admission/ownership/safety gate. The exact-HEAD checkout comparison itself
is a D32 seal: certified keeps its refusal, dev stamps and the
executing-code review still refuses.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tessera.dev_mode import DEV_MODE_ENV, NOT_COMPUTED

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/graph_attest_702"
sys.path.insert(0, str(HERE))
import managed_window
import tp2_recipe as recipe
import watch_window_queue
import window_driver as driver


class ReachedAdmission(RuntimeError):
    pass


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT).strip()


def disk_digest(root):
    return hashlib.sha256("".join(f"{hashlib.sha256((root / 'experiments/graph_attest_702' / name).read_bytes()).hexdigest()}  {name}\n"
                                 for name in recipe.PRODUCER_FILES).encode()).hexdigest()


def make_producer(tmp_path, *, message="reviewed producer"):
    root = tmp_path / "producer"
    (root / "experiments/graph_attest_702").mkdir(parents=True)
    for name in recipe.PRODUCER_FILES:
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
    managed_window.atomic_json(root / "memory-policy.json", getattr(managed_window, "MEMORY_POLICY", {}))
    (root / "inputs.json").write_text(json.dumps(dict(config=config, predecessors=[],
        census_action_keys=["1" * 64, "2" * 64],
        memory_policy_sha256=recipe.sha(root / "memory-policy.json"))))
    (root / "manifest.json").write_text("[]")
    reviews = tmp_path / "reviews.json"
    reviews.write_text(json.dumps({who: dict(verdict="APPROVE", head_sha=reviewed) for who in ("parent", "D5")}))
    return config, env, root, reviews


@pytest.mark.parametrize("entry", ["prepare", "submit"])
@pytest.mark.parametrize("attack", ["dirty-edited-code", "different-unreviewed-head"])
def test_reviewed_label_and_forged_disk_digest_do_not_authorize_changed_code(tmp_path, monkeypatch, entry, attack):
    # Certified mode: the forged-label attack must still refuse, never stamp.
    monkeypatch.setenv(DEV_MODE_ENV, "0")
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
    # Default dev mode: an exact clean reviewed producer still flows past the
    # identity stamp to the next real gate, certified or dev alike.
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

def test_pb_synthetic_head_keeps_reviewed_object_and_identical_clean_producer(tmp_path, monkeypatch):
    # Certified mode: the full reviewed-Git-object binding (byte equality
    # against git show objects) is pinned here; the dev-mode stamping of the
    # checkout/digest identities and the executing-code review gate have
    # their own tests.
    monkeypatch.setenv(DEV_MODE_ENV, "0")
    producer, reviewed = make_producer(tmp_path)
    (producer / ".pb-closure.json").write_text("{}")
    git(producer, "add", ".pb-closure.json")
    git(producer, "commit", "-qm", "PB synthetic snapshot")
    assert git(producer, "rev-parse", "HEAD") != reviewed
    assert recipe.require_producer(producer, reviewed, disk_digest(producer), exact_head=False) == disk_digest(producer)
    with pytest.raises(managed_window.Refused, match="HEAD"):
        recipe.require_producer(producer, reviewed, disk_digest(producer), exact_head=True)


def test_parentless_materialization_without_reviewed_git_object_refuses_by_name(tmp_path, monkeypatch):
    # Certified mode: in dev mode this boundary stamps and continues (below).
    monkeypatch.setenv(DEV_MODE_ENV, "0")
    original, reviewed = make_producer(tmp_path / "original")
    # Never create the reviewed object here: amending would retain it in the object database.
    producer, other = make_producer(tmp_path / "parentless", message="parentless materialization")
    with pytest.raises(managed_window.Refused, match="reviewed Git object unavailable"):
        recipe.require_producer(producer, reviewed, disk_digest(producer), exact_head=False)


@pytest.mark.parametrize("fault", ["content", "seal"])
def test_memory_policy_own_byte_integrity_refuses_in_both_modes(tmp_path, monkeypatch, fault):
    """A memory-policy byte error against its own recorded SHA refuses in dev
    and certified alike: own-byte integrity is never a run-identity stamp."""
    producer, reviewed = make_producer(tmp_path)
    config, env, root, reviews = fixture_inputs(monkeypatch, tmp_path, producer, reviewed)
    path = root / "memory-policy.json"
    policy = json.loads(path.read_text())
    setup = json.loads((root / "inputs.json").read_text())
    if fault == "content":
        policy["start_gib"] = 112
        managed_window.atomic_json(path, policy)
    else:
        setup["memory_policy_sha256"] = "0" * 64
    managed_window.atomic_json(root / "inputs.json", setup)
    with pytest.raises(managed_window.Refused, match="memory policy changed"):
        driver.submit(root, reviews)
    monkeypatch.setenv(DEV_MODE_ENV, "0")
    with pytest.raises(managed_window.Refused, match="memory policy changed"):
        driver.submit(root, reviews)


def test_restamped_policy_value_stamps_in_dev_and_refuses_certified(tmp_path, monkeypatch, capsys):
    """The restamped policy-versus-running VALUE comparison is a seal (D32):
    dev mode stamps ``[DEV-MODE]`` and continues with the stored policy;
    certified mode keeps the verbatim refusal."""
    producer, reviewed = make_producer(tmp_path)
    config, env, root, reviews = fixture_inputs(monkeypatch, tmp_path, producer, reviewed)
    path = root / "memory-policy.json"
    policy = json.loads(path.read_text())
    policy["start_gib"] = 112
    managed_window.atomic_json(path, policy)
    setup = json.loads((root / "inputs.json").read_text())
    setup["memory_policy_sha256"] = recipe.sha(path)
    managed_window.atomic_json(root / "inputs.json", setup)
    monkeypatch.setenv(DEV_MODE_ENV, "1")
    with pytest.raises(ReachedAdmission):
        driver.submit(root, reviews)
    assert "[DEV-MODE]" in capsys.readouterr().out
    monkeypatch.setenv(DEV_MODE_ENV, "0")
    with pytest.raises(managed_window.Refused, match="memory policy changed"):
        driver.submit(root, reviews)


def test_dev_mode_missing_reviewed_object_stamps_and_continues_without_git(tmp_path, monkeypatch, capsys):
    """D32: a parentless materialization without the reviewed object stamps in
    dev mode and continues with the stored data; no Git call and no digest
    compute happen, and the stored expected digest is returned verbatim."""
    original, reviewed = make_producer(tmp_path / "original")
    # Never create the reviewed object here: amending would retain it in the object database.
    producer, other = make_producer(tmp_path / "parentless", message="parentless materialization")

    def no_git(argv, **kwargs):
        raise AssertionError(f"dev mode must not call Git for run identity: {argv}")

    def no_digest(*args, **kwargs):
        raise AssertionError("dev mode must not compute the producer digest")

    monkeypatch.setattr(recipe, "producer_sha", no_digest)
    monkeypatch.setenv(DEV_MODE_ENV, "1")
    assert recipe.require_producer(producer, reviewed, disk_digest(producer),
                                   exact_head=False, runner=no_git) == disk_digest(producer)
    assert "[DEV-MODE]" in capsys.readouterr().out


def test_dev_mode_digest_mismatch_stamps_and_returns_the_stored_expected(tmp_path, monkeypatch, capsys):
    """D32: a self-supplied ``PRODUCER_SHA256`` binds nothing in dev mode: the
    mismatch stamps and the stored expected is returned without Git or a fresh
    digest; certified mode keeps the producer refusal."""
    producer, reviewed = make_producer(tmp_path)
    changed = producer / "experiments/graph_attest_702/rank_window.py"
    changed.write_text(changed.read_text() + "\n# unreviewed dev edit\n")

    def no_git(argv, **kwargs):
        raise AssertionError(f"dev mode must not call Git for run identity: {argv}")

    def no_digest(*args, **kwargs):
        raise AssertionError("dev mode must not compute the producer digest")

    monkeypatch.setattr(recipe, "producer_sha", no_digest)
    monkeypatch.setenv(DEV_MODE_ENV, "1")
    stored = "f" * 64
    assert recipe.require_producer(producer, reviewed, stored,
                                   exact_head=False, runner=no_git) == stored
    assert "[DEV-MODE]" in capsys.readouterr().out
    monkeypatch.setenv(DEV_MODE_ENV, "0")
    monkeypatch.delattr(recipe, "producer_sha")
    with pytest.raises(managed_window.Refused, match="producer"):
        recipe.require_producer(producer, reviewed, stored, exact_head=False)


def test_dev_mode_keeps_the_exact_executing_code_review_gate(tmp_path, monkeypatch):
    producer, reviewed = make_producer(tmp_path)
    config, env, root, reviews = fixture_inputs(monkeypatch, tmp_path, producer, reviewed)
    monkeypatch.setenv(DEV_MODE_ENV, "1")
    (producer / "changed-code.py").write_text("# controlled new code revision\n")
    git(producer, "add", "changed-code.py")
    git(producer, "commit", "-qm", "new code not covered by recorded review")
    with pytest.raises(managed_window.Refused, match="executing-code"):
        driver.submit(root, reviews)


def test_dev_mode_producer_sha_reports_the_stored_env_without_hashing(monkeypatch):
    """``producer_sha`` is a disk inspection, never an authority: in dev mode
    it reports the stored ``PRODUCER_SHA256`` or ``NOT_COMPUTED`` instead of
    computing a fresh digest; certified mode keeps the real disk digest."""
    monkeypatch.setenv(DEV_MODE_ENV, "1")
    monkeypatch.setenv("PRODUCER_SHA256", "stored-producer-sha")
    assert recipe.producer_sha() == "stored-producer-sha"
    monkeypatch.delenv("PRODUCER_SHA256")
    assert recipe.producer_sha() == NOT_COMPUTED
    monkeypatch.setenv(DEV_MODE_ENV, "0")
    assert recipe.producer_sha() == disk_digest(ROOT)
