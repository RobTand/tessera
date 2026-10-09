"""A population of shards is green only if the files, the tree and the exits all agree.

``tools/selected_population.py`` submits the files a change selects as shards
through ``tools/merge_suite.py`` and judges them from merge_suite's receipt.
These tests give it a pool of its own (the fleet's layout, built in
``tmp_path``) and a real git checkout, then change one fact at a time.  A
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


@pytest.fixture(autouse=True)
def _restore_pool_paths(monkeypatch):
    """``main`` points merge_suite at a pool; every test puts it back."""

    monkeypatch.setattr(sp.merge_suite, "POOL_QUEUE", sp.merge_suite.POOL_QUEUE)
    monkeypatch.setattr(sp.merge_suite, "POOL_CAS_REQUESTS", sp.merge_suite.POOL_CAS_REQUESTS)


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


def _key(index):
    return f"{index:02x}" + "0" * 62


def _publish(path, producer, source, skip_reasons, not_collected):
    """A shard's population with the producer stamp a real run carries."""

    surface = helpers._population("x86", source=source)["surface"]
    surface["skip_reasons"] = skip_reasons
    surface["not_collected"] = not_collected
    surface["source_identity"]["excluded_metadata"] = [{
        "path": f".pbrun-closure.{producer.stem[:16]}.json", "bytes": 153,
        "sha256": "d" * 64, "action_key": producer.stem,
        "request_sha256": hashlib.sha256(producer.read_bytes()).hexdigest()}]
    path.write_text(json.dumps(surface))


def _world(tmp_path, monkeypatch, *, shards=None, ran=None, state=None, source=None,
           skip_reasons=None, not_collected=None, selected=None, sources=None):
    """A checkout, a pool and a receipt directory holding exactly this population.

    ``ran`` is what each sealed command ran (default: the files it was given),
    ``state`` each action's queue ending (default: done, exit 0), and ``source``
    the effective source every shard published (default: the checkout's own).
    A shard named in ``shards`` and left out of ``state`` has no pool record.
    """

    shards = SHARDS if shards is None else shards
    checkout = _checkout(tmp_path)
    receipt_dir = tmp_path / "receipt"
    receipt_dir.mkdir()
    own = measured_source(checkout, verifier=None)["sha256"]
    for index, (name, files) in enumerate(shards.items()):
        surface = receipt_dir / f"surface.{name}.json"
        arm = sp.shard_arm(name, ran.get(name, files) if ran else files)
        command = sp.merge_suite._timed_command(
            sp.merge_suite._command(arm, surface, sp.PYTEST_ARGS, 2), 2400.0)
        queue = (state or {}).get(name, ("done", 0))
        actions = [(_key(index), queue[0], queue[1], "dl380g10")] if queue else []
        pool_queue, pool_requests = helpers._fake_pool(
            tmp_path / "pool", surface, actions, command=command)
        request = helpers._request_of(pool_requests, _key(index))
        if not request.exists():
            helpers._fake_pool(tmp_path / "pool", surface, [(_key(index), "done", 0, "x")],
                               command=command)
        _publish(surface, request, (sources or {}).get(name, source or own),
                 (skip_reasons or {}).get(name, {}), (not_collected or {}).get(name, []))
        if not queue:
            for folder in ("done", "failed"):
                (pool_queue / folder / f"{_key(index)}.json").unlink(missing_ok=True)
    selection = {"base": "origin/master", "verdict": "narrowed", "forces_full": [],
                 "excluded_tests": [], "shards": shards,
                 "tests": sorted(selected or {path for files in shards.values() for path in files})}
    (receipt_dir / sp.SELECTION).write_text(json.dumps(selection))
    sp.merge_suite.POOL_QUEUE, sp.merge_suite.POOL_CAS_REQUESTS = pool_queue, pool_requests
    return SimpleNamespace(checkout=checkout, receipt_dir=receipt_dir, selection=selection,
                           pool=tmp_path / "pool")


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
    world = _world(tmp_path, monkeypatch, source="e" * 64)
    receipt = _receipt(world)
    assert receipt["verdict"].startswith("incomplete:"), receipt["verdict"]
    assert any("not the checkout's" in problem for problem in receipt["population_problems"])


def test_a_dirty_checkout_cannot_vouch_for_the_shards(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    (world.checkout / "stray.py").write_text("x = 1\n")
    receipt = _receipt(world)
    assert receipt["verdict"].startswith("incomplete:"), receipt["verdict"]
    assert any("checkout's own source identity is not verified" in problem
               for problem in receipt["population_problems"])


def test_two_shards_that_measured_different_source_are_not_green(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch, sources={"shard-01": "f" * 64})
    assert not _receipt(world)["verdict"].startswith("green on")


def test_a_nonzero_pbrun_code_is_not_hidden_by_a_clean_pool_record(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    sp.write_client_result(world.receipt_dir, "shard-00", {"returncode": 0})
    sp.write_client_result(world.receipt_dir, "shard-01", {"returncode": 1})
    receipt = _receipt(world)
    assert receipt["verdict"].startswith("incomplete:"), receipt["verdict"]
    assert "shard-01: pbrun returned 1" in receipt["population_problems"]


def test_a_shard_the_pool_failed_is_red(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch, state={"shard-01": ("failed", 1)})
    verdict = _receipt(world)["verdict"]
    assert verdict.startswith("red"), verdict


def test_a_shard_with_no_pool_record_is_not_green(tmp_path, monkeypatch):
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


def _selector(monkeypatch, stdout="", returncode=0, stderr=""):
    asked = []

    def fake(command, **kwargs):
        asked.append((command, kwargs))
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(sp.subprocess, "run", fake)
    return asked


def test_the_selection_is_the_selectors_answer_split_into_shards(tmp_path, monkeypatch):
    for name, size in (("test_a.py", 30), ("test_b.py", 20), ("test_c.py", 10)):
        (tmp_path / name).write_text("x" * size)
    answer = {"verdict": "narrowed", "forces_full": [], "excluded_tests": [],
              "tests": ["test_c.py", "test_a.py", "test_b.py"]}
    asked = _selector(monkeypatch, json.dumps(answer))
    selection = sp.choose(tmp_path, "origin/master", 2)
    assert selection["tests"] == ["test_a.py", "test_b.py", "test_c.py"]
    assert sorted(path for files in selection["shards"].values() for path in files) == selection["tests"]
    assert list(selection["shards"]) == ["shard-00", "shard-01"]
    assert "origin/master...HEAD" in asked[0][0]
    assert asked[0][1]["cwd"] == tmp_path


@pytest.mark.parametrize("stdout, returncode, phrase", [
    pytest.param("", 0, "no selection", id="an-empty-change-prints-nothing"),
    pytest.param("", 2, "no selection", id="a-selector-that-failed"),
    pytest.param(json.dumps({"verdict": "narrowed", "tests": ["gone.py"]}), 0,
                 "does not hold: ['gone.py']", id="a-file-the-tree-lacks"),
])
def test_a_selection_that_cannot_be_run_is_refused_by_name(tmp_path, monkeypatch, stdout, returncode, phrase):
    _selector(monkeypatch, stdout, returncode)
    with pytest.raises(ValueError, match="no selection|does not hold") as error:
        sp.choose(tmp_path, "origin/master", 2)
    assert phrase in str(error.value)


def test_resume_rebuilds_the_receipt_and_submits_nothing(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)

    def refuse(*args, **kwargs):
        raise AssertionError("--resume submitted a shard")

    monkeypatch.setattr(sp.merge_suite, "_submit", refuse)
    status = sp.main(["--resume", str(world.receipt_dir), "--checkout", str(world.checkout),
                      "--pool-root", str(world.pool)])
    out = capsys.readouterr().out
    receipt = json.loads((world.receipt_dir / sp.RECEIPT).read_text())
    assert status == 0, out
    assert receipt["schema"] == "tessera.selected_population.v1"
    assert receipt["assembled_by"] == "resume"
    assert "selected_population: green on 2 population(s)" in out


def test_resume_exits_nonzero_for_a_population_that_is_not_green(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch, ran={"shard-00": ["tests/test_a.py"]})
    status = sp.main(["--resume", str(world.receipt_dir), "--checkout", str(world.checkout),
                      "--pool-root", str(world.pool)])
    assert status == 1
    assert "PROBLEM: " in capsys.readouterr().out


def test_a_dry_run_prints_each_shards_command_and_creates_nothing(tmp_path, monkeypatch, capsys):
    selection = {"base": "origin/master", "verdict": "narrowed", "forces_full": [],
                 "excluded_tests": [], "tests": sorted(sum(SHARDS.values(), [])), "shards": SHARDS}
    monkeypatch.setattr(sp, "choose", lambda checkout, base, count: selection)
    out_dir = tmp_path / "never-created"
    status = sp.main(["--dry-run", "--checkout", str(_checkout(tmp_path)), "--out", str(out_dir)])
    lines = [line for line in capsys.readouterr().out.splitlines() if "pbrun.py" in line]
    assert status == 0
    assert not out_dir.exists()
    assert len(lines) == 2
    for line, files in zip(lines, SHARDS.values()):
        assert all(path in line for path in files)
        assert f"--surface-json {out_dir}/surface.shard-" in line
        assert "-n 2 --dist worksteal" in line
        assert "--cpus 2" in line


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
    selection = {"base": "origin/master", "verdict": "narrowed", "forces_full": [],
                 "excluded_tests": [], "tests": sorted(sum(SHARDS.values(), [])), "shards": SHARDS}
    monkeypatch.setattr(sp, "choose", lambda checkout, base, count: dict(selection))
    status = sp.main(["--dry-run", "--checkout", str(_checkout(tmp_path)),
                      "--out", str(tmp_path / "never"), "--python", "/opt/venv/bin/python"])
    lines = [line for line in capsys.readouterr().out.splitlines() if "pbrun.py" in line]
    assert status == 0 and len(lines) == 2
    assert all("/opt/venv/bin/python -m pytest" in line for line in lines)


def test_no_more_than_eight_pbrun_clients_run_at_once(tmp_path, monkeypatch):
    """D21 caps one agent at eight PrismaBuild clients; the shard count is a separate number.

    Twelve shards queue behind eight clients.  The fake client holds until
    eight are in flight, so a runner that starts all twelve at once shows a
    peak of twelve and one that starts eight shows exactly eight.
    """

    shards = {f"shard-{i:02d}": [f"tests/test_{i}.py"] for i in range(12)}
    selection = {"shards": shards}
    lock, release = threading.Lock(), threading.Event()
    state = {"now": 0, "peak": 0}

    def client(name, arm, args, receipt_dir):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            if state["now"] >= 8:
                release.set()
        release.wait(5)
        time.sleep(0.02)
        with lock:
            state["now"] -= 1
        return {"arm": name, "returncode": 0}

    monkeypatch.setattr(sp.merge_suite, "_submit", client)
    args = SimpleNamespace(cpus=2, mem_gb=4, pytest_arg=[], timeout_s=1.0, wait_s=1.0,
                           checkout=tmp_path, dry_run=True, artifact_root=[])
    results = sp.submit_all(selection, args, tmp_path)
    assert sorted(results) == sorted(shards)
    assert sp.MAX_CLIENTS == 8
    assert state["peak"] == 8


def test_a_nonzero_client_result_survives_a_resume(tmp_path, monkeypatch, capsys):
    """A population rejected for a pbrun code must stay rejected after the submitter dies.

    The pool's record of the action can be clean while pbrun returned 1, and a
    resume that read only the pool would turn that rejected run green.
    """

    world = _world(tmp_path, monkeypatch)
    sp.write_client_result(world.receipt_dir, "shard-01", {"returncode": 1, "elapsed_s": 3.0})
    status = sp.main(["--resume", str(world.receipt_dir), "--checkout", str(world.checkout),
                      "--pool-root", str(world.pool)])
    receipt = json.loads((world.receipt_dir / sp.RECEIPT).read_text())
    assert status == 1, capsys.readouterr().out
    assert "shard-01: pbrun returned 1" in receipt["population_problems"]
    assert receipt["verdict"].startswith("incomplete:"), receipt["verdict"]


def test_a_resume_names_the_shards_whose_client_result_was_never_recorded(tmp_path, monkeypatch, capsys):
    """An unknown client result is stated, and the pool's own records still decide."""

    world = _world(tmp_path, monkeypatch)
    sp.write_client_result(world.receipt_dir, "shard-00", {"returncode": 0})
    status = sp.main(["--resume", str(world.receipt_dir), "--checkout", str(world.checkout),
                      "--pool-root", str(world.pool)])
    out = capsys.readouterr().out
    receipt = json.loads((world.receipt_dir / sp.RECEIPT).read_text())
    assert status == 0, out
    assert receipt["clients_not_recorded"] == ["shard-01"]
    states = {record["arm"]: record["client_result"] for record in receipt["arms"]}
    assert states == {"shard-00": "recorded", "shard-01": "not recorded"}
    assert "client result not recorded for: shard-01" in out


def test_each_clients_result_is_written_when_that_client_exits(tmp_path, monkeypatch):
    selection = {"shards": SHARDS}
    answers = {"shard-00": {"returncode": 0, "elapsed_s": 1.5, "stderr_tail": ["ok"], "pbrun": "pbrun a"},
               "shard-01": {"returncode": 1, "elapsed_s": 2.5, "stderr_tail": ["boom"], "pbrun": "pbrun b"}}
    monkeypatch.setattr(sp.merge_suite, "_submit", lambda name, arm, args, receipt_dir: answers[name])
    args = SimpleNamespace(cpus=2, mem_gb=4, pytest_arg=[], timeout_s=1.0, wait_s=1.0,
                           checkout=tmp_path, dry_run=False, artifact_root=[])
    sp.submit_all(selection, args, tmp_path)
    for name, answer in answers.items():
        assert sp.read_client_result(tmp_path, name) == answer
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
