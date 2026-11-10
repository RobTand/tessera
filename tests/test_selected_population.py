"""A population of shards is green only if the files, the tree and the exits all agree.

``tools/selected_population.py`` submits the files a change selects as shards
through ``tools/merge_suite.py`` and judges them from merge_suite's receipt.
These tests give it saved native records, a controlled CLI and a real git checkout, then change one fact at a time.  A
population that skipped a file, measured another tree, or hid a failed shard
is the defect tessera#1069 records, so each of those is a case.
"""

import hashlib
import importlib.util
import json
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tessera._dev.suite_source import measured_source

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sp = _load("_selected_population", ROOT / "tools" / "selected_population.py")
helpers = _load("_merge_test_helpers_for_population", TESTS / "test_merge_suite.py")

SHARDS = {"shard-00": ["tests/test_a.py", "tests/test_b.py"], "shard-01": ["tests/test_c.py"]}




def _git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test",
                    "-c", "user.email=test@example.invalid", *args],
                   check=True, capture_output=True)


def _checkout(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "source.py").write_text("VALUE = 1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "source")
    return root








def _receipt(world, **kwargs):
    return sp.assemble(world.selection, world.receipt_dir, world.checkout, **kwargs)


def test_a_population_is_green_when_the_files_the_tree_and_the_exits_agree(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    receipt = _receipt(world)
    assert receipt["population_problems"] == []
    assert receipt["verdict"].startswith("green on 2 population(s)"), receipt["verdict"]
    assert all(record["exit_status_observed"] for record in receipt["arms"])


@pytest.mark.parametrize("ran, phrase", [
    pytest.param({"shard-00": ["tests/test_a.py"]}, "selected but never run: tests/test_b.py",
                 id="a-selected-file-nobody-ran"),
    pytest.param({"shard-01": ["tests/test_c.py", "tests/test_x.py"]},
                 "run but not selected: tests/test_x.py", id="a-file-nobody-selected"),
    pytest.param({"shard-01": ["tests/test_c.py", "tests/test_a.py"]},
                 "run more than once: tests/test_a.py", id="a-file-run-twice"),
])
def test_the_sealed_commands_must_run_the_selected_files_exactly(tmp_path, monkeypatch, ran, phrase):
    world = _world(tmp_path, monkeypatch, ran=ran)
    receipt = _receipt(world)
    assert receipt["verdict"].startswith("incomplete:"), receipt["verdict"]
    assert any(phrase in problem for problem in receipt["population_problems"]), \
        receipt["population_problems"]


def test_shards_of_another_tree_than_the_checkouts_are_not_green(tmp_path, monkeypatch):
    _dev_mode(monkeypatch, certified=True)
    world = _world(tmp_path, monkeypatch, source="e" * 64)
    receipt = _receipt(world)
    assert receipt["verdict"].startswith("incomplete:"), receipt["verdict"]
    assert any("not the checkout's" in problem for problem in receipt["population_problems"])


def test_a_dirty_checkout_cannot_vouch_for_the_shards(tmp_path, monkeypatch):
    _dev_mode(monkeypatch, certified=True)
    world = _world(tmp_path, monkeypatch)
    (world.checkout / "stray.py").write_text("x = 1\n")
    receipt = _receipt(world)
    assert receipt["verdict"].startswith("incomplete:"), receipt["verdict"]
    assert any("checkout's own source identity is not verified" in problem
               for problem in receipt["population_problems"])


def test_two_shards_that_measured_different_source_are_not_green(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch, sources={"shard-01": "f" * 64})
    assert not _receipt(world)["verdict"].startswith("green on")


def test_a_nonzero_client_code_is_not_hidden_by_a_clean_pool_record(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    sp.write_client_result(world.receipt_dir, "shard-00", {"returncode": 0})
    sp.write_client_result(world.receipt_dir, "shard-01", {"returncode": 1})
    receipt = _receipt(world)
    assert receipt["verdict"].startswith("incomplete:"), receipt["verdict"]
    assert "shard-01: Flux client returned 1" in receipt["population_problems"]


def test_a_shard_native_execution_failed_is_red(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch, state={"shard-01": ("failed", 1)})
    verdict = _receipt(world)["verdict"]
    assert verdict.startswith("red"), verdict


def test_a_shard_with_no_native_record_is_not_green(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch, state={"shard-01": None})
    receipt = _receipt(world)
    assert not receipt["verdict"].startswith("green on"), receipt["verdict"]
    assert any("no sealed command is bound" in problem for problem in receipt["population_problems"])


def test_a_module_a_shard_did_not_collect_is_a_problem_a_pass_count_hides(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch, not_collected={"shard-00": ["tests/test_native.py"]})
    receipt = _receipt(world)
    assert receipt["verdict"].startswith("incomplete:"), receipt["verdict"]
    assert any("not collected: tests/test_native.py" in problem
               for problem in receipt["population_problems"])


def test_the_report_keeps_one_block_per_shard_and_every_reason_verbatim(tmp_path, monkeypatch):
    """Two reasons that share a long opening are two rows, and neither is cut."""

    shared = "box artifact absent: checkpoints and serve logs this box produced -- /runs/stock/serve_qwen_"
    reasons = {shared + "k2.log is not on this box": 1, shared + "k2-graph.log is not on this box": 1,
               "needs a CUDA device": 14}
    world = _world(tmp_path, monkeypatch, skip_reasons={"shard-00": reasons, "shard-01": reasons})
    lines = sp.report_lines(_receipt(world))
    assert [line.split()[0] for line in lines if " rc=" in line] == ["shard-00", "shard-01"]
    for reason in reasons:
        assert sum(line.endswith("  " + reason) for line in lines) == 2, reason
    assert sum(line.strip().split()[0].isdigit() for line in lines if line.startswith("     ")) == 6


@pytest.mark.parametrize("count", [1, 2, 3, 7])
def test_every_file_lands_in_exactly_one_shard_and_the_split_is_stable(count):
    sizes = {f"tests/test_{i}.py": size for i, size in enumerate([90, 5, 5, 40, 40, 1, 33])}
    shards = sp.balanced_shards(sizes, count)
    assert sorted(path for shard in shards for path in shard) == sorted(sizes)
    assert shards == sp.balanced_shards(dict(reversed(list(sizes.items()))), count)
    assert all(shards)
    loads = [sum(sizes[path] for path in shard) for shard in shards]
    assert max(loads) - min(loads) <= max(sizes.values())


def test_a_selection_smaller_than_the_shard_count_gets_fewer_shards():
    assert len(sp.balanced_shards({"a.py": 1, "b.py": 2}, 12)) == 2
    with pytest.raises(ValueError, match="at least one shard"):
        sp.balanced_shards({"a.py": 1}, 0)










@pytest.mark.parametrize("old_receipt", [False, True])
def test_submit_time_master_is_unknown_for_a_legacy_resume(tmp_path, monkeypatch, old_receipt):
    """Legacy selection/receipt files cannot prove submission-time master.

    Pre-fix (both cases): AssertionError: legacy runs have no submit-time
    evidence; do not backfill from current master or an old receipt.
    """

    _dev_mode(monkeypatch)
    world = _world(tmp_path, monkeypatch)
    _, _, submitted, _ = helpers._master_history(world.checkout, "master")
    expected = _receipt(world)
    if old_receipt:
        prior = {**expected, "population": {**expected["population"],
                 "master_head_at_submit": submitted, "master_ref_used": "master", "is_master_head": True}}
        (world.receipt_dir / sp.RECEIPT).write_text(json.dumps(prior))
    monkeypatch.setattr(sp, "submit_all", lambda *args: pytest.fail("resume submitted shards"))
    assert sp.main(["--checkout", str(world.checkout), "--resume", str(world.receipt_dir)]) == 0
    receipt = json.loads((world.receipt_dir / sp.RECEIPT).read_text())
    assert receipt["verdict"] == expected["verdict"]
    assert receipt["commits_measured"] == expected["commits_measured"]
    assert receipt["population"]["master_head_at_submit"] is None, (
        "legacy runs have no submit-time evidence; do not backfill from current master or an old receipt")
    assert receipt["population"]["is_master_head"] is None


def test_resume_rebuilds_the_receipt_and_submits_nothing(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)

    def refuse(*args, **kwargs):
        raise AssertionError("--resume submitted a shard")

    monkeypatch.setattr(sp.merge_suite, "_submit", refuse)
    status = sp.main(["--resume", str(world.receipt_dir), "--checkout", str(world.checkout)])
    out = capsys.readouterr().out
    receipt = json.loads((world.receipt_dir / sp.RECEIPT).read_text())
    assert status == 0, out
    assert receipt["schema"] == "tessera.selected_population.flux.v2"
    assert receipt["assembled_by"] == "resume"
    assert "selected_population: green on 2 population(s)" in out


def test_resume_exits_nonzero_for_a_population_that_is_not_green(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch, ran={"shard-00": ["tests/test_a.py"]})
    status = sp.main(["--resume", str(world.receipt_dir), "--checkout", str(world.checkout)])
    assert status == 1
    assert "PROBLEM: " in capsys.readouterr().out


def test_a_dry_run_prints_each_shards_command_and_creates_nothing(tmp_path, monkeypatch, capsys):
    selection = {"source_identity": {"sha256": "a" * 64}, "base": "origin/master", "verdict": "narrowed", "forces_full": [],
                 "excluded_tests": [], "tests": sorted(sum(SHARDS.values(), [])), "shards": SHARDS}
    monkeypatch.setattr(sp, "choose", lambda checkout, base, count: selection)
    out_dir = tmp_path / "never-created"
    status = sp.main(["--dry-run", "--checkout", str(_checkout(tmp_path)), "--out", str(out_dir)])
    lines = [line for line in capsys.readouterr().out.splitlines() if "-m pytest" in line]
    assert status == 0
    assert not out_dir.exists()
    assert len(lines) == 2
    for line, files in zip(lines, SHARDS.values()):
        assert all(path in line for path in files)
        import shlex
        argv, why = sp.merge_suite._pytest_argv(shlex.split(line))
        assert why is None
        assert sp.merge_suite._targets_of(argv) == files
        destination, why = sp.merge_suite._effective_surface_json(argv)
        assert why is None and Path(destination).parent == out_dir
        assert "-n 2 --dist worksteal" in line
        assert "--cpus" not in line


def test_every_shard_runs_under_the_interpreter_the_run_names(tmp_path, monkeypatch, capsys):
    """The arm's default interpreter may lack what the suite imports.

    merge_suite's x86 arm names ``pb-cpu``, which has no ``jsonschema``, so a
    test that needs it fails there for want of the package and not for any
    defect.  The run names its interpreter, and the selection records it so
    ``--resume`` reads the same one.
    """

    arm = sp.shard_arm("shard-00", ["tests/test_a.py"], python="/opt/venv/bin/python")
    assert arm["python"] == "/opt/venv/bin/python"
    assert sp.shard_arm("shard-00", ["tests/test_a.py"])["python"] == sp.merge_suite.ARMS["x86"]["python"]
    selection = {"source_identity": {"sha256": "a" * 64}, "base": "origin/master", "verdict": "narrowed", "forces_full": [],
                 "excluded_tests": [], "tests": sorted(sum(SHARDS.values(), [])), "shards": SHARDS}
    monkeypatch.setattr(sp, "choose", lambda checkout, base, count: dict(selection))
    status = sp.main(["--dry-run", "--checkout", str(_checkout(tmp_path)),
                      "--out", str(tmp_path / "never"), "--python", "/opt/venv/bin/python"])
    lines = [line for line in capsys.readouterr().out.splitlines() if "-m pytest" in line]
    assert status == 0 and len(lines) == 2
    assert all("/opt/venv/bin/python -s -m pytest" in line for line in lines)


def test_no_more_than_two_native_clients_run_at_once(tmp_path, monkeypatch):
    """Twelve shards queue behind the two reserved Flux job slots."""

    shards = {f"shard-{i:02d}": [f"tests/test_{i}.py"] for i in range(12)}
    selection = {"shards": shards, "source_identity": {"sha256": "a" * 64}}
    lock, release = threading.Lock(), threading.Event()
    state = {"now": 0, "peak": 0}

    def client(name, arm, args, receipt_dir):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            if state["now"] >= 2:
                release.set()
        assert release.wait(20), "second client never arrived"
        with lock:
            state["now"] -= 1
        return {"arm": name, "returncode": 0}

    monkeypatch.setattr(sp.merge_suite, "_submit", client)
    args = SimpleNamespace(cpus=2, mem_gb=4, pytest_arg=[], timeout_s=1.0, wait_s=1.0,
                           checkout=tmp_path, dry_run=True, artifact_root=[])
    results = sp.submit_all(selection, args, tmp_path)
    assert sorted(results) == sorted(shards)
    assert sp.MAX_CLIENTS == 2
    assert state["peak"] == 2


def test_a_nonzero_client_result_survives_a_resume(tmp_path, monkeypatch, capsys):
    """A population rejected for a pbrun code must stay rejected after the submitter dies.

    The pool's record of the action can be clean while Flux client returned 1, and a
    resume that read only the pool would turn that rejected run green.
    """

    world = _world(tmp_path, monkeypatch)
    sp.write_client_result(world.receipt_dir, "shard-01", {"returncode": 1, "elapsed_s": 3.0})
    status = sp.main(["--resume", str(world.receipt_dir), "--checkout", str(world.checkout)])
    receipt = json.loads((world.receipt_dir / sp.RECEIPT).read_text())
    assert status == 1, capsys.readouterr().out
    assert "shard-01: Flux client returned 1" in receipt["population_problems"]
    assert receipt["verdict"].startswith("incomplete:"), receipt["verdict"]


def test_a_resume_names_the_shards_whose_client_result_was_never_recorded(tmp_path, monkeypatch, capsys):
    """An unknown client result is stated, and the pool's own records still decide."""

    world = _world(tmp_path, monkeypatch)
    sp.write_client_result(world.receipt_dir, "shard-00", {"returncode": 0})
    status = sp.main(["--resume", str(world.receipt_dir), "--checkout", str(world.checkout)])
    out = capsys.readouterr().out
    receipt = json.loads((world.receipt_dir / sp.RECEIPT).read_text())
    assert status == 0, out
    assert receipt["clients_not_recorded"] == ["shard-01"]
    states = {record["arm"]: record["client_result"] for record in receipt["arms"]}
    assert states == {"shard-00": "recorded", "shard-01": "not recorded"}
    assert "client result not recorded for: shard-01" in out


def test_each_clients_result_is_written_when_that_client_exits(tmp_path, monkeypatch):
    selection = {"shards": SHARDS, "source_identity": {"sha256": "a" * 64}}
    answers = {"shard-00": {"returncode": 0, "elapsed_s": 1.5, "stderr_tail": ["ok"], "flux": "native a"},
               "shard-01": {"returncode": 1, "elapsed_s": 2.5, "stderr_tail": ["boom"], "flux": "native b"}}
    monkeypatch.setattr(sp.merge_suite, "_submit", lambda name, arm, args, receipt_dir: answers[name])
    args = SimpleNamespace(cpus=2, mem_gb=4, pytest_arg=[], timeout_s=1.0, wait_s=1.0,
                           checkout=tmp_path, dry_run=False, artifact_root=[])
    sp.submit_all(selection, args, tmp_path)
    for name, answer in answers.items():
        assert sp.read_client_result(tmp_path, name) == {key: answer.get(key) for key in sp._CLIENT_FIELDS}
    assert sp.read_client_result(tmp_path, "shard-02") is None


def _dev_mode(monkeypatch, certified=False):
    """Dev mode is ON unless PRISMAQUANT_DEV_MODE is exactly 0 (D32)."""

    if certified:
        monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    else:
        monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)


@pytest.mark.parametrize("fault", ["another-tree", "dirty-checkout"])
def test_in_dev_mode_the_checkout_source_comparison_is_stamped_and_not_computed(
        tmp_path, monkeypatch, capsys, fault):
    """D32: a source identity comparison never refuses in dev mode and computes no digest for itself."""

    _dev_mode(monkeypatch)
    world = _world(tmp_path, monkeypatch, source="e" * 64 if fault == "another-tree" else None)
    if fault == "dirty-checkout":
        (world.checkout / "stray.py").write_text("x = 1\n")

    def measured(*args, **kwargs):
        raise AssertionError("dev mode computed the checkout's digest for a seal")

    monkeypatch.setattr(sp, "measured_source", measured)
    receipt = _receipt(world)
    assert receipt["verdict"].startswith("green on 2 population(s)"), receipt["verdict"]
    assert receipt["population_problems"] == []
    assert receipt["checkout_source"]["verification"] == "not computed"
    assert capsys.readouterr().out.count("[DEV-MODE]") == 1


@pytest.mark.parametrize("fault, phrase", [
    pytest.param("another-tree", "not the checkout's", id="another-tree"),
    pytest.param("dirty-checkout", "checkout's own source identity is not verified", id="dirty-checkout"),
])
def test_in_certified_mode_the_checkout_source_comparison_still_refuses(
        tmp_path, monkeypatch, fault, phrase):
    _dev_mode(monkeypatch, certified=True)
    world = _world(tmp_path, monkeypatch, source="e" * 64 if fault == "another-tree" else None)
    if fault == "dirty-checkout":
        (world.checkout / "stray.py").write_text("x = 1\n")
    receipt = _receipt(world)
    assert receipt["verdict"].startswith("incomplete:"), receipt["verdict"]
    assert any(phrase in problem for problem in receipt["population_problems"])


def test_in_certified_mode_a_matching_source_is_green(tmp_path, monkeypatch):
    _dev_mode(monkeypatch, certified=True)
    receipt = _receipt(_world(tmp_path, monkeypatch))
    assert receipt["verdict"].startswith("green on 2 population(s)"), receipt["verdict"]
    assert receipt["checkout_source"]["verification"] == "verified"


@pytest.mark.parametrize("certified", [False, True])
def test_file_coverage_stays_strict_in_both_modes(tmp_path, monkeypatch, certified):
    """D32 converts seals only.  Which files ran is correctness, and it refuses in both modes."""

    _dev_mode(monkeypatch, certified=certified)
    world = _world(tmp_path, monkeypatch, ran={"shard-00": ["tests/test_a.py"]})
    receipt = _receipt(world)
    assert receipt["verdict"].startswith("incomplete:"), receipt["verdict"]
    assert any("never run" in problem for problem in receipt["population_problems"])


def _world(tmp_path, monkeypatch, *, shards=None, ran=None, state=None, source=None,
           skip_reasons=None, not_collected=None, selected=None, sources=None):
    shards = SHARDS if shards is None else shards
    checkout = _checkout(tmp_path)
    receipt_dir = tmp_path / "receipt"; receipt_dir.mkdir()
    identity = measured_source(checkout, verifier=None)
    selection = {"base": "HEAD", "base_commit": identity["snapshot_commit"],
        "head_commit": identity["snapshot_commit"], "source_identity": identity,
        "verdict": "narrowed", "forces_full": [], "excluded_tests": [], "shards": shards,
        "tests": sorted(selected or {path for files in shards.values() for path in files})}
    jobs = helpers._native_jobs(sp.merge_suite, monkeypatch)
    for name, files in shards.items():
        surface = helpers._population("x86", source=(sources or {}).get(name, source or identity["sha256"]),
                                     commit=identity["snapshot_commit"])["surface"]
        surface["skip_reasons"] = (skip_reasons or {}).get(name, {})
        surface["not_collected"] = (not_collected or {}).get(name, [])
        arm = sp.shard_arm(name, ran.get(name, files) if ran else files)
        command = sp.merge_suite._timed_command(sp.merge_suite._command(
            arm, receipt_dir / f"surface.{name}.json", sp.PYTEST_ARGS, 2), 60)
        status = (state or {}).get(name, ("done", 0))
        world = jobs.add(receipt_dir, name, surface, command=command,
            code=status[1] if status else 0, application=sp.selection_binding(selection, receipt_dir))
        if status is None:
            (world.directory / "submission.json").unlink()
    (receipt_dir / sp.SELECTION).write_bytes(sp.merge_suite.suite_flux.encoded(selection))
    return SimpleNamespace(checkout=checkout, receipt_dir=receipt_dir, selection=selection, jobs=jobs)


@pytest.mark.parametrize("answer,code", [({"verdict": "narrowed", "tests": ["tests/test_b.py", "tests/test_a.py"]}, 0),
    ({"verdict": "narrowed", "tests": ["tests/gone.py"]}, 0), ({}, 2)])
def test_selection_binds_resolved_refs_and_actual_source(tmp_path, monkeypatch, answer, code):
    checkout = _checkout(tmp_path)
    (checkout / "tests").mkdir()
    for name in ("test_a.py", "test_b.py"):
        (checkout / "tests" / name).write_text("def test_one(): pass\n")
    _git(checkout, "add", "."); _git(checkout, "commit", "-qm", "tests")
    selector = tmp_path / "selector.py"
    selector.write_text(f"import json,sys; print({json.dumps(json.dumps(answer))}); sys.exit({code})\n")
    monkeypatch.setattr(sp, "SELECTOR", selector)
    monkeypatch.setattr(sp.merge_suite.suite_flux, "source", lambda path: measured_source(path, verifier=None))
    if code or "tests/gone.py" in answer.get("tests", []):
        with pytest.raises(ValueError, match="no selection|does not hold"):
            sp.choose(checkout, "HEAD~1", 2)
    else:
        result = sp.choose(checkout, "HEAD~1", 2)
        assert result["tests"] == ["tests/test_a.py", "tests/test_b.py"]
        assert result["head_commit"] == measured_source(checkout, verifier=None)["snapshot_commit"]
        assert result["base_commit"] == subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD~1"], text=True).strip()
        assert result["source_identity"] == measured_source(checkout, verifier=None)
        assert sorted(path for files in result["shards"].values() for path in files) == result["tests"]


def test_all_known_native_ids_are_collected_after_a_sibling_submit_error(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    visited = []
    def submit(name, arm, args, directory):
        if name == "shard-00":
            raise ValueError("submission response lost, never retry")
        return sp.merge_suite._resume(name, arm, directory)
    original = sp.merge_suite.suite_flux.collect
    def collect(directory):
        visited.append(directory.name)
        return original(directory)
    monkeypatch.setattr(sp.merge_suite, "_submit", submit)
    monkeypatch.setattr(sp.merge_suite.suite_flux, "collect", collect)
    args = SimpleNamespace(cpus=2, mem_gb=4, pytest_arg=[], timeout_s=60, wait_s=60,
        checkout=world.checkout, dry_run=False, artifact_root=[])
    results = sp.submit_all(world.selection, args, world.receipt_dir)
    assert set(visited) == {"flux.shard-00", "flux.shard-01"}
    assert "submission response lost" in results["shard-00"]["client_error"]
    assert all(record["exit_status_observed"] for record in results.values())
    assert not any("submit" in call for call in world.jobs.calls)


def test_selector_subprocess_disables_user_site(tmp_path, monkeypatch):
    import shlex

    checkout = _checkout(tmp_path)
    (checkout / "tests").mkdir()
    (checkout / "tests/test_a.py").write_text("def test_one(): pass\n")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "selected tests")
    selector = tmp_path / "selector.py"
    selector.write_text('import json,sys\nassert sys.flags.no_user_site == 1\n'
        'print(json.dumps({"verdict":"narrowed","tests":["tests/test_a.py"]}))\n')
    interpreter = tmp_path / "python"
    interpreter.write_text('#!/bin/sh\ncase "$1" in -s|-I) ;; *) '
        'echo "explicit isolated Python flag required" >&2; exit 91;; esac\n'
        'exec ' + shlex.quote(sp.sys.executable) + ' "$@"\n')
    interpreter.chmod(0o755)
    monkeypatch.setattr(sp.sys, "executable", str(interpreter))
    monkeypatch.setattr(sp, "SELECTOR", selector)
    monkeypatch.setattr(sp.merge_suite.suite_flux, "source",
        lambda path: measured_source(path, verifier=None))
    result = sp.choose(checkout, "HEAD~1", 1)
    assert result["tests"] == ["tests/test_a.py"]
    assert result["shards"] == {"shard-00": ["tests/test_a.py"]}
    assert result["source_identity"]["verification"] == "verified"


def test_replaced_selection_cannot_borrow_saved_native_jobs(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    assert _receipt(world)["verdict"].startswith("green")
    world.selection["base_commit"] = "e" * 40
    receipt = _receipt(world)
    assert not receipt["verdict"].startswith("green")
    assert any("immutable selection" in problem for problem in receipt["population_problems"])


def test_explicit_scope_preserves_full_selector_verdict_and_source(tmp_path, monkeypatch):
    checkout = _checkout(tmp_path)
    (checkout / "tests").mkdir()
    for name in ("test_a.py", "test_b.py"):
        (checkout / "tests" / name).write_text("def test_one(): pass\n")
    _git(checkout, "add", "."); _git(checkout, "commit", "-qm", "tests")
    selector = tmp_path / "selector.py"
    selector.write_text('print(\'{"verdict":"full","tests":["tests/test_a.py","tests/test_b.py"]}\')\n')
    monkeypatch.setattr(sp, "SELECTOR", selector)
    monkeypatch.setattr(sp.merge_suite.suite_flux, "source", lambda path: measured_source(path, verifier=None))
    result = sp.choose(checkout, "HEAD~1", 2, targets=["tests/test_a.py"])
    assert result["verdict"] == "full"
    assert result["selector_tests"] == ["tests/test_a.py", "tests/test_b.py"]
    assert result["tests"] == ["tests/test_a.py"]
    assert result["scope"] == "explicit subset of selector population"
    assert result["shards"] == {"shard-00": ["tests/test_a.py"]}
    with pytest.raises(ValueError, match="selected"):
        sp.choose(checkout, "HEAD~1", 2, targets=["tests/not_selected.py"])


def test_concurrency_above_tessera_cap_is_refused(tmp_path):
    args = SimpleNamespace(max_jobs=3)
    with pytest.raises(ValueError, match='max-jobs.*2'):
        sp.submit_all({'shards': {}}, args, tmp_path)
