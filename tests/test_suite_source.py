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


def _snapshot(tmp_path, name, content="same source\n", *, extra=None, bad=None,
              version=1, schema_version=None):
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
    _git(root, "commit", "-qm", f"PrismaBuild pbrun checkout snapshot v{version}")
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
                       "schema": f"prismaquant.prismabuild.pbrun_checkout_snapshot.v{schema_version or version}",
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
    module = importlib.import_module("tessera._dev.suite_source")
    root, requests, _, _ = fixture
    return module.measured_source(root, request_root=requests, owner="e" * 64, **kwargs)


@pytest.mark.parametrize("versions", [(1, 1), (2, 2), (1, 2)])
def test_arm_specific_snapshots_retain_ids_but_have_one_effective_source(tmp_path, versions):
    left = _measure(_snapshot(tmp_path, "gpu", version=versions[0]))
    right = _measure(_snapshot(tmp_path, "x86", version=versions[1]))
    assert left["snapshot_commit"] != right["snapshot_commit"]
    assert left["verification"] == right["verification"] == "verified"
    assert left["sha256"] == right["sha256"]
    assert len(left["excluded_metadata"]) == len(right["excluded_metadata"]) == 1
    changed = _measure(_snapshot(tmp_path, "changed", "different source\n",
                                 version=versions[1]))
    assert changed["sha256"] != right["sha256"]


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
@pytest.mark.parametrize("version", [1, 2])
def test_unverifiable_closure_metadata_never_establishes_source_equivalence(tmp_path, bad, version):
    record = _measure(_snapshot(tmp_path, "gpu", bad=bad, version=version))
    assert record["verification"] == "unknown"
    assert record["sha256"] is None and record["excluded_metadata"] == []
    assert record["reason"]


@pytest.mark.parametrize("version,schema_version", [(3, 3), (2, 1)])
def test_unknown_or_mismatched_snapshot_versions_cannot_establish_identity(
        tmp_path, version, schema_version):
    record = _measure(_snapshot(tmp_path, "gpu", version=version,
                                schema_version=schema_version))
    assert record["verification"] == "unknown"
    assert record["sha256"] is None and record["excluded_metadata"] == []
    assert "snapshot" in record["reason"]


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


def _plain(tmp_path, name, body="A\n"):
    """An ordinary (non-PB) checkout, which is what a local suite runs in."""

    root = tmp_path / name
    root.mkdir()
    _git(root, "init", "-q")
    (root / "source.py").write_text(body)
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "ordinary source")
    return root


def _module():
    return importlib.import_module("tessera._dev.suite_source")


def test_a_clean_source_switch_during_the_run_is_not_attested(tmp_path):
    """#219: identity was sampled after execution, and named the wrong tree.

    A suite imports A, the shared checkout is fast-forwarded to B while it
    runs, and the terminal summary hashes B: clean tree, HEAD stable across
    the hashing interval, ``verification: verified``.  Python is still holding
    the modules it imported from A, so the receipt attests a tree that was
    never tested.
    """

    module = _module()
    root = _plain(tmp_path, "checkout")
    entry = module.measured_source(root)
    assert entry["verification"] == "verified", entry

    (root / "source.py").write_text("B\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "the checkout moved, cleanly, mid-run")

    published = module.measured_source(root, entry=entry)
    assert published["verification"] == "unknown", published
    assert published["sha256"] is None, published
    assert published["measurement_span"]["agrees"] is False, published
    assert published["measurement_span"]["entry_sha256"] == entry["sha256"]
    assert "entry" in published["reason"], published


def test_an_unchanged_source_is_attested_and_says_it_was_bound(tmp_path):
    """The valid case, and it has to say what makes it valid."""

    module = _module()
    root = _plain(tmp_path, "checkout")
    entry = module.measured_source(root)
    published = module.measured_source(root, entry=entry)

    assert published["verification"] == "verified", published
    assert published["sha256"] == entry["sha256"]
    assert published["measurement_span"]["agrees"] is True, published


def test_an_unverifiable_entry_cannot_bind_a_verified_publication(tmp_path):
    """Unknown at entry is not agreement; it is the absence of it."""

    module = _module()
    root = _plain(tmp_path, "checkout")
    unknown_entry = {"schema": module.SCHEMA, "snapshot_commit": None,
                     "sha256": None, "verification": "unknown",
                     "excluded_metadata": [], "reason": "no git here"}

    published = module.measured_source(root, entry=unknown_entry)
    assert published["verification"] == "unknown", published
    assert published["measurement_span"]["agrees"] is False, published


def test_the_immutable_snapshot_case_still_binds(tmp_path):
    """A PB snapshot cannot move under a run, and must stay attestable."""

    fixture = _snapshot(tmp_path, "gpu")
    entry = _measure(fixture)
    published = _measure(fixture, entry=entry)
    assert published["verification"] == "verified", published
    assert published["sha256"] == entry["sha256"]
    assert len(published["excluded_metadata"]) == 1


def test_a_population_measured_by_several_processes_must_agree(tmp_path):
    """Under -n the controller hashes its filesystem and the workers ran.

    The canonical population is written by a process that executed none of the
    tests it reports.  Its own hash is therefore a claim about the controller's
    filesystem, and it becomes a claim about the measured source only when the
    processes that did the executing say the same thing.
    """

    module = _module()
    root = _plain(tmp_path, "checkout")
    entry = module.measured_source(root)
    identity = module.measured_source(root, entry=entry)
    other = dict(identity, sha256="f" * 64)

    assert module.agreed_source(identity, {}) == identity
    agreed = module.agreed_source(identity, {"gw0": identity, "gw1": identity})
    assert agreed["verification"] == "verified", agreed
    assert agreed["sha256"] == identity["sha256"]
    assert agreed["workers"] == {"gw0": "agrees", "gw1": "agrees"}, agreed

    split = module.agreed_source(identity, {"gw0": identity, "gw1": other})
    assert split["verification"] == "unknown", split
    assert split["sha256"] is None, split
    assert "gw1" in split["reason"], split

    silent = module.agreed_source(identity, {"gw0": identity, "gw1": None})
    assert silent["verification"] == "unknown", silent
    assert "gw1" in silent["reason"], silent


def test_a_worker_that_reports_only_its_entry_identity_establishes_nothing(tmp_path):
    """An entry identity is a seed; it was taken before the worker ran (#291).

    It is a verified hash of the same clean tree, so every check the aggregate
    used to make passed on it -- which is how a worker whose FINAL measurement
    refused could be published as agreeing.  What separates the two is the
    span: only ``measured_source(..., entry=...)`` measures across the tests
    the worker actually ran, and only that record may establish agreement.
    """

    module = _module()
    root = _plain(tmp_path, "checkout")
    entry = module.measured_source(root)
    identity = module.measured_source(root, entry=entry)

    assert module.is_entry_bound(identity) is True, identity
    assert module.is_entry_bound(entry) is False, entry
    assert entry["verification"] == "verified" and entry["sha256"] == identity["sha256"]

    seeded = module.agreed_source(identity, {"gw0": entry})
    assert seeded["verification"] == "unknown", seeded
    assert seeded["sha256"] is None, seeded
    assert "entry" in seeded["workers"]["gw0"], seeded
    assert "gw0" in seeded["reason"], seeded

    # A worker whose own binding refused is named for that, not for the seed.
    refused = module.agreed_source(
        identity, {"gw0": dict(identity, verification="unknown", sha256=None)})
    assert "verified source identity" in refused["workers"]["gw0"], refused


def test_the_suite_publishes_an_entry_bound_identity_its_workers_agree_with():
    """The rule, wired into the file that publishes the population.

    ``tests/conftest.py`` captures the entry identity above its first import
    of the code under test, and folds each worker's reported identity into
    what it publishes.  This drives those two seams directly, because xdist is
    absent from this interpreter and the seam is the thing under test.
    """

    import types

    import conftest

    assert conftest.SOURCE_AT_ENTRY["schema"] == _module().SCHEMA
    saved = dict(conftest._WORKER_SOURCES)
    try:
        conftest._WORKER_SOURCES.clear()
        alone = conftest.published_source_identity()
        assert "measurement_span" in alone, alone
        assert "workers" not in alone, alone

        # Entry-BOUND, and disagreeing on the hash: the branch that names what
        # the other process measured.  An unbound record would be refused one
        # step earlier, which is the subject of the xdist tests in
        # ``tests/test_cuda_surface.py``.
        node = types.SimpleNamespace(
            gateway=types.SimpleNamespace(id="gw3"),
            workeroutput={"tessera_source_identity":
                          dict(conftest.SOURCE_AT_ENTRY, sha256="f" * 64,
                               verification="verified",
                               measurement_span={"agrees": True})})
        conftest.pytest_testnodedown(node, None)
        disagreed = conftest.published_source_identity()
        assert disagreed["verification"] == "unknown", disagreed
        assert disagreed["sha256"] is None, disagreed
        assert "gw3" in disagreed["reason"], disagreed
        assert "ffffffffffff" in disagreed["workers"]["gw3"], disagreed
    finally:
        conftest._WORKER_SOURCES.clear()
        conftest._WORKER_SOURCES.update(saved)
