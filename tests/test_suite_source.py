"""Effective source identity must not mistake executor scaffolding for source.

A materialized checkout may carry a file the executor generated.  It is left
out of the hash only when a declared source verifier vouches for it, and
Tessera still checks every file the verifier names.  These tests drive that
seam with small verifier commands written under ``tmp_path``; they name no
executor.
"""
import hashlib
import importlib
import json
import shlex
import subprocess
import sys

import pytest

GENERATED = "executor-stamp.json"


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), "-c", "user.name=Test",
                                    "-c", "user.email=test@example.invalid", *args]).decode().strip()


def _module():
    return importlib.import_module("tessera._dev.suite_source")


def _snapshot(tmp_path, name, content="same source\n", *, stamp=None, extra=None):
    """A checkout whose commit carries one generated file beside the source."""

    root = tmp_path / name / "checkout"
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    (root / "source.py").write_text(content)
    for path, value in (extra or {}).items():
        (root / path).write_text(value)
    raw = json.dumps(stamp or {"arm": name}).encode()
    (root / GENERATED).write_bytes(raw)
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "materialized snapshot")
    return root, {"path": GENERATED, "bytes": len(raw),
                  "sha256": hashlib.sha256(raw).hexdigest(),
                  "action_key": "a" * 64, "request_sha256": "b" * 64}


def _verifier(tmp_path, *, generated=None, exit_code=0, stdout=None):
    """A verifier command that prints ``generated`` or exits ``exit_code``.

    It also records the root and commit it was asked about.
    """

    script = tmp_path / f"verifier-{len(list(tmp_path.glob('verifier-*')))}.py"
    asked = script.with_suffix(".asked")
    text = stdout if stdout is not None else json.dumps(
        {"schema": "example.generated.v1", "generated": generated or []})
    script.write_text(
        "import sys, pathlib\n"
        f"pathlib.Path({str(asked)!r}).write_text(' '.join(sys.argv[1:]))\n"
        f"sys.stdout.write({text!r})\n"
        f"sys.stderr.write('verifier says no')\n"
        f"sys.exit({exit_code})\n")
    return [sys.executable, str(script)], asked


def _measure(root, verifier, **kwargs):
    return _module().measured_source(root, verifier=verifier, **kwargs)


def test_arm_specific_generated_files_leave_one_effective_source(tmp_path):
    left, left_entry = _snapshot(tmp_path, "gpu")
    right, right_entry = _snapshot(tmp_path, "x86")
    left_record = _measure(left, _verifier(tmp_path, generated=[left_entry])[0])
    right_record = _measure(right, _verifier(tmp_path, generated=[right_entry])[0])
    assert left_record["snapshot_commit"] != right_record["snapshot_commit"]
    assert left_record["verification"] == right_record["verification"] == "verified"
    assert left_record["sha256"] == right_record["sha256"]
    assert left_record["excluded_metadata"] == [left_entry]
    assert right_record["excluded_metadata"] == [right_entry]
    changed, changed_entry = _snapshot(tmp_path, "changed", "different source\n",
                                       stamp={"arm": "x86"})
    changed_record = _measure(changed, _verifier(tmp_path, generated=[changed_entry])[0])
    assert changed_record["sha256"] != right_record["sha256"]


def test_the_verifier_is_asked_about_this_checkout_and_commit(tmp_path):
    root, entry = _snapshot(tmp_path, "gpu")
    verifier, asked = _verifier(tmp_path, generated=[entry])
    record = _measure(root, verifier)
    assert asked.read_text() == f"{root} {record['snapshot_commit']}"


def test_the_environment_declares_the_verifier_by_default(tmp_path, monkeypatch):
    root, entry = _snapshot(tmp_path, "gpu")
    verifier, _ = _verifier(tmp_path, generated=[entry])
    monkeypatch.setenv(_module().VERIFIER_ENV, shlex.join(verifier))
    assert _module().measured_source(root)["excluded_metadata"] == [entry]
    monkeypatch.setenv(_module().VERIFIER_ENV, "  ")
    record = _module().measured_source(root)
    assert record["verification"] == "unknown" and "names no command" in record["reason"]


def test_without_a_verifier_every_tracked_file_is_source(tmp_path, monkeypatch):
    monkeypatch.delenv(_module().VERIFIER_ENV, raising=False)
    left, _ = _snapshot(tmp_path, "gpu")
    right, _ = _snapshot(tmp_path, "x86")
    left_record, right_record = _module().measured_source(left), _measure(right, None)
    assert left_record["verification"] == right_record["verification"] == "verified"
    assert left_record["excluded_metadata"] == right_record["excluded_metadata"] == []
    # The generated files differ, so without a verifier the arms never agree.
    assert left_record["sha256"] != right_record["sha256"]


def test_a_file_the_verifier_did_not_name_is_source(tmp_path):
    name = "lookalike-stamp.json"
    left, left_entry = _snapshot(tmp_path, "gpu", extra={name: "one"})
    right, right_entry = _snapshot(tmp_path, "x86", extra={name: "two"})
    left_record = _measure(left, _verifier(tmp_path, generated=[left_entry])[0])
    right_record = _measure(right, _verifier(tmp_path, generated=[right_entry])[0])
    assert left_record["sha256"] != right_record["sha256"]
    assert name not in [row["path"] for row in left_record["excluded_metadata"]]


def _broken(entry, how):
    if how == "size":
        return [dict(entry, bytes=entry["bytes"] + 1)]
    if how == "digest":
        return [dict(entry, sha256="0" * 64)]
    if how == "absent":
        return [dict(entry, path="not-there.json")]
    if how == "escape":
        return [dict(entry, path="../outside.json")]
    if how == "absolute":
        return [dict(entry, path="/etc/hostname")]
    if how == "unnormalized":
        return [dict(entry, path="./" + entry["path"])]
    if how == "repeated":
        return [entry, entry]
    if how == "no-digest":
        return [{"path": entry["path"]}]
    return ["not an object"]


@pytest.mark.parametrize("how", ["size", "digest", "absent", "escape", "absolute",
                                 "unnormalized", "repeated", "no-digest", "shape"])
def test_a_generated_file_tessera_cannot_check_is_never_left_out(tmp_path, how):
    root, entry = _snapshot(tmp_path, "gpu")
    record = _measure(root, _verifier(tmp_path, generated=_broken(entry, how))[0])
    assert record["verification"] == "unknown", record
    assert record["sha256"] is None and record["excluded_metadata"] == []
    assert record["reason"]


def test_a_generated_file_absent_from_the_commit_is_never_left_out(tmp_path):
    """A file the verifier vouches for must be the commit's blob, not an extra."""

    root, entry = _snapshot(tmp_path, "gpu")
    _git(root, "rm", "-q", "--cached", GENERATED)
    _git(root, "commit", "-qm", "stamp untracked")
    (root / ".git" / "info" / "exclude").write_text(GENERATED + "\n")
    record = _measure(root, _verifier(tmp_path, generated=[entry])[0])
    assert record["verification"] == "unknown", record
    assert record["excluded_metadata"] == []


@pytest.mark.parametrize("verifier", ["refuses", "no-json", "no-list", "missing"])
def test_a_verifier_that_does_not_vouch_makes_the_identity_unknown(tmp_path, verifier):
    root, entry = _snapshot(tmp_path, "gpu")
    if verifier == "refuses":
        command = _verifier(tmp_path, generated=[entry], exit_code=1)[0]
    elif verifier == "no-json":
        command = _verifier(tmp_path, stdout="snapshot verified")[0]
    elif verifier == "no-list":
        command = _verifier(tmp_path, stdout=json.dumps({"generated": "all"}))[0]
    else:
        command = [str(tmp_path / "no-such-verifier")]
    record = _measure(root, command)
    assert record["verification"] == "unknown", record
    assert record["sha256"] is None and record["excluded_metadata"] == []
    assert "verifier" in record["reason"]
    if verifier == "refuses":
        assert "verifier says no" in record["reason"]


@pytest.mark.parametrize("change", ["bytes", "mode", "delete", "untracked", "generated"])
def test_dirty_materialized_snapshot_is_not_reported_as_its_old_source(tmp_path, change):
    root, entry = _snapshot(tmp_path, "gpu")
    if change == "bytes":
        (root / "source.py").write_text("changed\n")
    elif change == "mode":
        (root / "source.py").chmod(0o755)
    elif change == "delete":
        (root / "source.py").unlink()
    elif change == "untracked":
        (root / "new-source.py").write_text("new\n")
    else:
        (root / GENERATED).write_text("{}")
    record = _measure(root, _verifier(tmp_path, generated=[entry])[0])
    assert record["verification"] == "unknown" and record["sha256"] is None


def test_mode_symlink_and_nul_safe_path_identity_are_preserved(tmp_path):
    fixtures = [_snapshot(tmp_path, name)[0] for name in ("base", "mode", "link", "name")]
    # Ordinary Git commits still measure their whole source roster.
    for index, root in enumerate(fixtures):
        _git(root, "rm", "-q", GENERATED)
        if index == 1:
            (root / "source.py").chmod(0o755)
        elif index == 2:
            (root / "source.py").unlink()
            (root / "source.py").symlink_to("target")
        elif index == 3:
            (root / "source.py").rename(root / "source\nwith\ttabs.py")
        _git(root, "add", "-A")
        _git(root, "commit", "--allow-empty", "-qm", "ordinary source")
    records = [_measure(item, None) for item in fixtures]
    assert all(row["verification"] == "verified" for row in records)
    assert len({row["sha256"] for row in records}) == len(records)


def _plain(tmp_path, name, body="A\n"):
    """An ordinary checkout, which is what a local suite runs in."""

    root = tmp_path / name
    root.mkdir()
    _git(root, "init", "-q")
    (root / "source.py").write_text(body)
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "ordinary source")
    return root


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
    """A materialized snapshot cannot move under a run, and must stay attestable."""

    root, generated = _snapshot(tmp_path, "gpu")
    verifier = _verifier(tmp_path, generated=[generated])[0]
    entry = _measure(root, verifier)
    published = _measure(root, verifier, entry=entry)
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
