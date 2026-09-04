"""Effective source identity must not mistake PB scaffolding for source."""
import hashlib
import importlib
import json
import subprocess

import pytest


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), "-c", "user.name=Test",
                                    "-c", "user.email=test@example.invalid", *args]).decode().strip()


def _snapshot(tmp_path, name, content="same source\n", *, extra=None, bad=None):
    root = tmp_path / name / "pending" / "checkout"
    root.mkdir(parents=True)
    requests = tmp_path / name / "requests"
    _git(root, "init", "-q")
    (root / "source.py").write_text(content)
    if extra:
        for path, value in extra.items():
            (root / path).write_text(value)
    stamp = {"cwd": ".", "head": "a" * 40, "dirty_sha256": "b" * 64}
    variables = {"PRISMABUILD_CONTAINER_OWNER": "e" * 64}
    command = ["python", "-m", "pytest", name]
    demand = {"cpu": 1, "mem_gb": 4}
    placement = {"required_tags": [name]}
    fingerprint = hashlib.sha256(json.dumps(
        [command, ".", demand, variables, {k: stamp[k] for k in ("head", "dirty_sha256")}, placement],
        sort_keys=True).encode()).hexdigest()[:16]
    filename = f".pbrun-closure.{fingerprint}.json"
    if bad == "filename":
        filename = ".pbrun-closure.0123456789abcdef.json"
    if bad == "extra-key":
        stamp["not_generated"] = True
    if bad == "cwd":
        stamp["cwd"] = "elsewhere"
    raw = json.dumps(stamp, indent=1, sort_keys=True).encode()
    (root / filename).write_bytes(raw)
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "PrismaBuild pbrun checkout snapshot v1")
    head = _git(root, "rev-parse", "HEAD")
    entry = {"path": filename, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if bad == "closure-size":
        entry["bytes"] += 1
    if bad == "closure-hash":
        entry["sha256"] = "0" * 64
    closure = {"schema": "prismaquant.prismabuild.code_closure.v1", "files": [entry]}
    closure["closure_sha256"] = _hash(closure)
    if bad == "closure-null":
        closure = None
    snapshot_input = {"id": "pbrun.checkout-snapshot", "bytes": 123, "sha256": "c" * 64}
    action = {
        "schema": "prismaquant.prismabuild.action.v2",
        "task": {"definition_id": "fleet/pbrun", "definition_version": "v1",
                 "result_path": f"pbrun_result.{fingerprint}.txt"},
        "params": {"command": command, "cwd": ".", "demand": demand,
                   "placement": placement, "checkout_snapshot": {
                       "schema": "prismaquant.prismabuild.pbrun_checkout_snapshot.v1",
                       "commit": "0" * 40 if bad == "snapshot" else head,
                       "subdirectory": ".", "input": snapshot_input}},
        "inputs": [snapshot_input], "environment": {
            "variables": None if bad == "variables-null" else variables},
        "code_closure": closure,
    }
    action["action_key"] = _hash(action)
    key = action["action_key"]
    moved = root.parent.with_name(f"{key[:12]}.fixture")
    root.parent.rename(moved)
    root = moved / "checkout"
    request = requests / key[:2] / f"{key}.json"
    request.parent.mkdir(parents=True)
    request.write_text(json.dumps(action))
    return root, requests, request, filename


def _measure(fixture, **kwargs):
    module = importlib.import_module("tessera.suite_source")
    root, requests, _, _ = fixture
    return module.measured_source(root, request_root=requests, owner="e" * 64, **kwargs)


def test_arm_specific_snapshots_retain_ids_but_have_one_effective_source(tmp_path):
    left = _measure(_snapshot(tmp_path, "gpu"))
    right = _measure(_snapshot(tmp_path, "x86"))
    assert left["snapshot_commit"] != right["snapshot_commit"]
    assert left["verification"] == right["verification"] == "verified"
    assert left["sha256"] == right["sha256"]
    assert len(left["excluded_metadata"]) == len(right["excluded_metadata"]) == 1


def test_same_original_head_does_not_hide_changed_source_bytes(tmp_path):
    left = _measure(_snapshot(tmp_path, "gpu"))
    right = _measure(_snapshot(tmp_path, "x86", "different dirty source at same head\n"))
    assert left["sha256"] != right["sha256"]


def test_extra_exact_grammar_metadata_is_source_not_scaffolding(tmp_path):
    name = ".pbrun-closure.ffffffffffffffff.json"
    first = json.dumps({"cwd": ".", "head": "a" * 40, "dirty_sha256": "c" * 64})
    second = json.dumps({"cwd": ".", "head": "b" * 40, "dirty_sha256": "c" * 64})
    left = _measure(_snapshot(tmp_path, "gpu", extra={name: first}))
    right = _measure(_snapshot(tmp_path, "x86", extra={name: second}))
    assert left["sha256"] != right["sha256"]
    assert name not in [row["path"] for row in left["excluded_metadata"]]


@pytest.mark.parametrize("bad", ["filename", "extra-key", "cwd", "closure-size", "closure-hash", "snapshot"])
def test_unverifiable_closure_metadata_never_establishes_source_equivalence(tmp_path, bad):
    record = _measure(_snapshot(tmp_path, "gpu", bad=bad))
    assert record["verification"] == "unknown"
    assert record["sha256"] is None and record["excluded_metadata"] == []
    assert record["reason"]


@pytest.mark.parametrize("change", ["bytes", "mode", "delete", "untracked", "closure"])
def test_dirty_materialized_snapshot_is_not_reported_as_its_old_source(tmp_path, change):
    fixture = _snapshot(tmp_path, "gpu")
    root, _, _, stamp = fixture
    if change == "bytes":
        (root / "source.py").write_text("changed\n")
    elif change == "mode":
        (root / "source.py").chmod(0o755)
    elif change == "delete":
        (root / "source.py").unlink()
    elif change == "untracked":
        (root / "new-source.py").write_text("new\n")
    else:
        (root / stamp).write_text("{}")
    record = _measure(fixture)
    assert record["verification"] == "unknown" and record["sha256"] is None


def test_missing_or_ambiguous_action_lookup_is_unknown(tmp_path):
    fixture = _snapshot(tmp_path, "gpu")
    root, requests, request, _ = fixture
    saved = request.read_bytes()
    request.unlink()
    assert _measure(fixture)["verification"] == "unknown"
    request.write_bytes(saved)
    request.with_name(request.stem[:12] + "f" * 52 + ".json").write_bytes(saved)
    assert _measure(fixture)["verification"] == "unknown"


def test_malformed_action_mappings_report_unknown_instead_of_aborting(tmp_path):
    for bad in ("variables-null", "closure-null"):
        record = _measure(_snapshot(tmp_path, bad, bad=bad))
        assert record["verification"] == "unknown" and record["sha256"] is None


def test_mode_symlink_and_nul_safe_path_identity_are_preserved(tmp_path):
    fixtures = [_snapshot(tmp_path, name) for name in ("base", "mode", "link", "name")]
    # Ordinary non-PB Git commits still measure their whole source roster.
    for index, (root, _, _, stamp) in enumerate(fixtures):
        _git(root, "rm", "-q", stamp)
        if index == 1:
            (root / "source.py").chmod(0o755)
        elif index == 2:
            (root / "source.py").unlink()
            (root / "source.py").symlink_to("target")
        elif index == 3:
            (root / "source.py").rename(root / "source\nwith\ttabs.py")
        _git(root, "add", "-A")
        _git(root, "commit", "--allow-empty", "-qm", "ordinary source")
    records = [_measure(item) for item in fixtures]
    assert all(row["verification"] == "verified" for row in records)
    assert len({row["sha256"] for row in records}) == len(records)
