"""Named Window4 selection and refusal controls, not live fit/speed evidence."""
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1] / "experiments/graph_attest_702"
sys.path.insert(0, str(HERE))
import tp2_recipe as recipe
from managed_window import Refused

MODE = "window4-eager-2048-4096"


def eager_plan(tmp_path):
    path = tmp_path / "plan.txt"
    path.write_text('\n'.join(f'{name} EAGER=1 SPEC_JSON={json.dumps(recipe.MTP, separators=(",", ":"))} MAX_BATCHED={chunk} FABRIC=socket'
                              for name, chunk in (("eager2048", 2048), ("eager4096", 4096))) + '\n')
    return path


def test_named_eager_plan_selects_only_2048_then_4096(tmp_path):
    arms = recipe.plan(eager_plan(tmp_path), mode=MODE)
    assert [(a["arm"], a["max_batched"], a["eager"]) for a in arms] == [
        ("eager2048", 2048, "1"), ("eager4096", 4096, "1")]


def test_named_eager_serve_changes_only_chunk_within_the_pair(tmp_path):
    arms = recipe.plan(eager_plan(tmp_path), mode=MODE)
    config = dict(artifact=recipe.CONTROL, window_mode=MODE, profile_dir="/profile")
    commands = [recipe.serve(config, a, 1) for a in arms]
    for command in commands:
        assert command[command.index("--max-num-seqs") + 1] == "1"
        assert command[command.index("--max-model-len") + 1] == "8448"
        assert command[command.index("--kv-cache-memory-bytes") + 1] == "2147483648"
        assert "--enforce-eager" in command and "--headless" in command
    slot = commands[0].index("--max-num-batched-tokens") + 1
    assert commands[0][slot] == "2048" and commands[1][slot] == "4096"
    commands[1][slot] = commands[0][slot]
    assert commands[0] == commands[1]


def test_graph_control_does_not_accept_the_eager_plan(tmp_path):
    with pytest.raises(Refused, match="finite control"):
        recipe.plan(eager_plan(tmp_path))


@pytest.mark.parametrize("replacement", ["8192", "1024"])
def test_named_eager_plan_refuses_undeclared_chunks(tmp_path, replacement):
    path = eager_plan(tmp_path)
    path.write_text(path.read_text().replace("MAX_BATCHED=4096", "MAX_BATCHED=" + replacement))
    with pytest.raises(Refused, match="2048.*4096"):
        recipe.plan(path, mode=MODE)


@pytest.mark.parametrize("old,new", [("FABRIC=socket", "FABRIC=roce"), ("EAGER=1", "EAGER=0"),
                                   ("MAX_BATCHED=4096", "MAX_BATCHED=2048")])
def test_eager_plan_refuses_fabric_graph_or_scope_substitution(tmp_path, old, new):
    path = eager_plan(tmp_path)
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(Refused):
        recipe.plan(path, mode=MODE)


def test_graph_input_refusal_remains_c4_and_2048(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    env = dict(TS=str(HERE.parents[1]), ARTIFACT=str(tmp_path), RECEIPTS=str(tmp_path / "arms"), FABRIC="socket")
    for override in (dict(MAX_NUM_SEQS="1"), dict(MAX_BATCHED="4096")):
        with pytest.raises(Refused, match="no scope substitution"):
            recipe.inputs(dict(env, **override), live=False)
    selected = recipe.inputs(dict(env, WINDOW_MODE=MODE, MAX_NUM_SEQS="1", MAX_BATCHED="4096"), live=False)
    assert selected["window_mode"] == MODE


@pytest.mark.parametrize("name", ["success", "probe_failure", "timeout", "preflight", "floor", "copy_failure"])
def test_eager_lifecycle_uses_rank1_client_and_stops_later_arm_on_failure(tmp_path, name):
    from test_graph_attest_window_scenarios import scenario
    outcomes, living, seconds = scenario(tmp_path, name, mode=MODE)
    assert seconds < 10 and not living
    assert all(outcome["simulation"] and not outcome["ownership_released"] for outcome in outcomes)
    rdv = tmp_path / "rdv"
    assert not (rdv / "eager2048-probes-rank0.json").exists()
    if name == "success":
        assert all(outcome["returncode"] == 0 and outcome["completed_arms"] == ["eager2048", "eager4096"]
                   for outcome in outcomes)
        assert (rdv / "eager2048-probes-rank1.json").exists()
    else:
        assert any(outcome["returncode"] for outcome in outcomes)
        assert not (rdv / "eager4096.rank1.pid").exists()


def test_runtime_candidate_review_is_exact_and_independent_of_producer(tmp_path, monkeypatch):
    from test_graph_attest_producer_identity import make_producer, fixture_inputs, ReachedAdmission
    import window_driver as driver
    producer, reviewed = make_producer(tmp_path)
    config, env, root, reviews = fixture_inputs(monkeypatch, tmp_path, producer, reviewed)
    config.update(window_mode=MODE, pq_pin_commit="e" * 40, artifact_manifest=str(tmp_path / "fixture.json"))
    setup = json.loads((root / "inputs.json").read_bytes())
    setup["config"] = config
    (root / "inputs.json").write_text(json.dumps(setup))
    proof = json.loads(reviews.read_bytes())
    with pytest.raises(Refused, match="runtime candidate parent"):
        driver.submit(root, reviews)
    proof["runtime"] = {who: dict(verdict="APPROVE", head_sha="e" * 40) for who in ("parent", "D5")}
    proof["runtime"]["D5"]["head_sha"] = reviewed
    reviews.write_text(json.dumps(proof))
    with pytest.raises(Refused, match="runtime candidate D5"):
        driver.submit(root, reviews)
    proof["runtime"]["D5"]["head_sha"] = "e" * 40
    reviews.write_text(json.dumps(proof))
    with pytest.raises(ReachedAdmission):
        driver.submit(root, reviews)


@pytest.mark.parametrize("fault", ["skipped", "usage", "stream"])
def test_exact_timing_population_refuses_partial_or_wrong_counts(fault):
    import eager_benchmark as benchmark
    result = dict(cells={f"host-L{length}-c1": dict(complete=True, trials=[
        dict(trial=trial, requests=[dict(error=None, usage=dict(prompt_tokens=length, completion_tokens=128),
                                       generation=dict(done=True), completion_tokens=128)])
        for trial in range(1, 11)]) for length in (512, 2048, 8192)})
    benchmark.require_timing(result)
    cell = result["cells"]["host-L8192-c1"]
    if fault == "skipped": cell["skipped"] = True
    if fault == "usage": cell["trials"][0]["requests"][0]["usage"]["completion_tokens"] = 127
    if fault == "stream": cell["trials"][0]["requests"][0]["generation"]["done"] = False
    with pytest.raises(Refused):
        benchmark.require_timing(result)
