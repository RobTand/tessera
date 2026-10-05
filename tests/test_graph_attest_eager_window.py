"""Named Window4 selection and refusal controls, not live fit/speed evidence."""
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1] / "experiments/graph_attest_702"
sys.path.insert(0, str(HERE))
import tp2_recipe as recipe
from managed_window import Refused
from tessera.dev_mode import DEV_MODE_ENV

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


@pytest.fixture
def bound_files(tmp_path, monkeypatch):
    """Synthetic real files; only fixed identity constants/shared root are substituted."""
    import hashlib
    import eager_benchmark as benchmark
    source, artifact, client, panel = [tmp_path / name for name in ("source", "artifact", "client", "panel")]
    for path in (source / "src/tessera/serving", artifact, client, panel): path.mkdir(parents=True)
    contract = source / "src/tessera/serving/runtime_contract.json"
    contract.write_text("fixture contract")
    monkeypatch.setattr(benchmark, "CONTRACT_SHA", benchmark.sha(contract))
    entries = []
    for i in range(128):
        name = f"{i:03}.safetensors" if i < 120 else f"{i:03}.json"
        path = artifact / name
        path.write_text("fixture bytes")
        entries.append(dict(name=name, bytes=path.stat().st_size, sha256=benchmark.sha(path)))
    manifest = tmp_path / "inventory.json"
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
    manifest.write_bytes(canonical)
    monkeypatch.setattr(benchmark, "ARTIFACT_SHA", hashlib.sha256(canonical).hexdigest())
    monkeypatch.setattr(benchmark, "SHARED_ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "CLIENT", client)
    monkeypatch.setattr(benchmark, "PANEL", panel)
    for name, constant in (("u4_speed_client.py", "TIMING_SHA"), ("comparison_inputs.py", "PROFILE_SHA")):
        path = client / name
        path.write_text("fixture program")
        monkeypatch.setattr(benchmark, constant, benchmark.sha(path))
    for name, constant in (("prompts.json", "PROMPTS_SHA"), ("manifest-decode.json", "MANIFEST_SHA")):
        path = panel / name
        path.write_text("fixture input")
        monkeypatch.setattr(benchmark, constant, benchmark.sha(path))
    identity = client / "source_identity.json"
    identity.write_text(json.dumps(dict(files={name: benchmark.sha(client / name) for name in
                                             ("u4_speed_client.py", "comparison_inputs.py")})))
    monkeypatch.setattr(benchmark, "SOURCE_IDENTITY_SHA", benchmark.sha(identity))
    env = dict(SOURCE_COMMIT=benchmark.RUNTIME_COMMIT, TS=str(source), PQ_PIN_COMMIT=benchmark.PQ_PIN_COMMIT,
               ARTIFACT_MANIFEST=str(manifest), RECEIPTS=str(tmp_path / "arms"))
    return benchmark, env, artifact, entries


IDENTITY_FAULTS = ("client", "prompts", "contract", "pin", "power")


@pytest.mark.parametrize("fault", ["none", "inventory", "roster", "metadata", "client", "prompts", "contract", "pin", "power"])
def test_actual_binding_hashes_and_roster_refuse_drift(bound_files, fault, monkeypatch):
    benchmark, env, artifact, entries = bound_files
    monkeypatch.delenv(DEV_MODE_ENV, raising=False)  # data damage refuses with dev ON; identity faults force 0 below
    if fault == "none":
        result = benchmark.bindings(env, artifact)
        assert result["artifact_files"] == 128 and result["pq_pin_commit"] == benchmark.PQ_PIN_COMMIT
        return
    if fault == "inventory": Path(env["ARTIFACT_MANIFEST"]).write_text("[]")
    if fault == "roster": (artifact / entries[0]["name"]).unlink()
    if fault == "metadata": (artifact / entries[-1]["name"]).write_text("different len")
    if fault == "client": (benchmark.CLIENT / "u4_speed_client.py").write_text("changed")
    if fault == "prompts": (benchmark.PANEL / "prompts.json").write_text("changed")
    if fault == "contract": (Path(env["TS"]) / "src/tessera/serving/runtime_contract.json").write_text("changed")
    if fault == "pin": env["PQ_PIN_COMMIT"] = "e" * 40
    if fault == "power": monkeypatch.setattr(benchmark, "POWER_SHA", "f" * 64)
    if fault in IDENTITY_FAULTS:
        monkeypatch.setenv(DEV_MODE_ENV, "0")  # identity seals refuse only in certified mode
    with pytest.raises(Refused): benchmark.bindings(env, artifact)


@pytest.mark.parametrize("fault", ["source", "pin", "contract", "prompts"])
def test_dev_mode_stamps_identity_drift_and_continues_with_stored(bound_files, fault, monkeypatch, capsys):
    """D32: identity drift stamps [DEV-MODE] lines and returns the stored bindings."""
    benchmark, env, artifact, entries = bound_files
    monkeypatch.delenv(DEV_MODE_ENV, raising=False)
    if fault == "source": env["SOURCE_COMMIT"] = "0" * 40
    if fault == "pin": env["PQ_PIN_COMMIT"] = "e" * 40
    if fault == "contract": (Path(env["TS"]) / "src/tessera/serving/runtime_contract.json").write_text("changed")
    if fault == "prompts": (benchmark.PANEL / "prompts.json").write_text("changed")
    if fault == "power": monkeypatch.setattr(benchmark, "POWER_SHA", "f" * 64)
    result = benchmark.bindings(env, artifact)
    out = capsys.readouterr().out
    # Dev stamps every suspended digest seal (8 not-computed) plus a drifted
    # stored-value seal for source/pin.
    assert out.count("[DEV-MODE]") == (9 if fault in ("source", "pin") else 8)
    assert out.count("not computed") == 8 and "continuing with the stored data" in out
    assert result["artifact_files"] == 128 and result["artifact_bytes"] == sum(e["bytes"] for e in entries)
    if fault == "source": assert "seal source commit differs" in out
    if fault == "pin":
        assert "seal PQ pin commit differs" in out
        assert result["pq_pin_commit"] == benchmark.PQ_PIN_COMMIT
    if fault == "contract": assert result["runtime_contract_sha256"] == benchmark.CONTRACT_SHA
    if fault == "prompts": assert result["prompts_sha256"] == benchmark.PROMPTS_SHA


def test_dev_mode_stamps_drifted_instrument_whose_members_still_match(bound_files, monkeypatch, capsys):
    """An evolved instrument stamps its fixed historical seal, then passes member integrity."""
    benchmark, env, artifact, entries = bound_files
    monkeypatch.delenv(DEV_MODE_ENV, raising=False)
    program = benchmark.CLIENT / "u4_speed_client.py"
    program.write_text("evolved program")
    identity = benchmark.CLIENT / "source_identity.json"
    members = json.loads(identity.read_text())
    members["files"]["u4_speed_client.py"] = benchmark.sha(program)
    identity.write_text(json.dumps(members))
    monkeypatch.setattr(benchmark, "SOURCE_IDENTITY_SHA", benchmark.sha(identity))
    result = benchmark.bindings(env, artifact)
    out = capsys.readouterr().out
    assert out.count("[DEV-MODE]") == 8 and out.count("not computed") == 8
    assert result["timing_program_sha256"] == benchmark.TIMING_SHA  # the stored October 5 program


def test_dev_mode_still_refuses_source_identity_member_drift(bound_files, monkeypatch):
    """Corrupting a member's bytes refuses even in dev: recorded-hash integrity, not sealing."""
    benchmark, env, artifact, entries = bound_files
    monkeypatch.delenv(DEV_MODE_ENV, raising=False)
    program = benchmark.CLIENT / "u4_speed_client.py"
    program.write_text("corrupted")
    monkeypatch.setattr(benchmark, "TIMING_SHA", benchmark.sha(program))  # its historical identity agrees
    with pytest.raises(Refused, match="source identity member changed"):
        benchmark.bindings(env, artifact)


def test_certified_mode_computes_identity_digests_and_agrees(bound_files, monkeypatch, capsys):
    """PRISMAQUANT_DEV_MODE=0 computes every identity digest and passes the matching fixture."""
    benchmark, env, artifact, entries = bound_files
    monkeypatch.setenv(DEV_MODE_ENV, "0")
    result = benchmark.bindings(env, artifact)
    assert capsys.readouterr().out == ""
    assert result["artifact_content_sha256"] == benchmark.ARTIFACT_SHA
    assert result["runtime_contract_sha256"] == benchmark.CONTRACT_SHA


def test_complete_census_allows_prices_but_refuses_another_sealed_pair(tmp_path, monkeypatch):
    pb = pytest.importorskip("prismabuild.core", reason="requires the published PB SDK")
    import watch_window_queue as watch
    monkeypatch.setattr(watch, "REQUESTS", tmp_path / "requests")
    (tmp_path / "fixture.py").write_text("# CPU fixture closure")
    toolchain = {**pb.executable_toolchain_contract(sys.executable), **pb.live_platform_toolchain_contract()}
    actions = []
    for kind in ("generation", "price", "pair"):
        task_class = "generation" if kind == "generation" else "measurement"
        command = [sys.executable, "fixture.py"] if kind != "pair" else [sys.executable, "rank_window.py", "--rank", "1", "--run", str(tmp_path / "inputs.json")]
        action = pb.seal_action(dict(schema=pb.ACTION_SCHEMA_V2,
            task=dict(definition_id="tests/window4-census", definition_version="1", task_class=task_class,
                      determinism="deterministic", artifact_family="generic", artifact_kind="generic",
                      argv=command, working_directory=".", result_path="result.json"),
            inputs=[], code_closure=pb.build_code_closure(tmp_path, ["fixture.py"]), params=dict(command=command),
            environment=dict(variables={"GRAPH_WINDOW_INPUT_SHA256": "a" * 64} if kind == "pair" else {}, toolchain=toolchain),
            execution_scope=dict(portability="host_class_keyed", platform_key=None, host_class="gb10")))
        path = watch.REQUESTS / action["action_key"][:2] / (action["action_key"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(action))
        actions.append(action)
    observed = dict(complete=True, queue_snapshot=dict(jobs=[dict(state="READY", action_key=a["action_key"]) for a in actions]))
    assert [row["action_key"] for row in watch.other_windows(observed)] == [actions[2]["action_key"]]
    observed["queue_snapshot"]["jobs"].pop()
    assert watch.other_windows(observed) == []  # price measurement is legitimate; PB elects/fences it
    observed["queue_snapshot"]["jobs"].append(dict(state="CLAIMED", action_key=actions[2]["action_key"]))
    observed["complete"] = False
    with pytest.raises(RuntimeError, match="complete queue snapshot"): watch.other_windows(observed)
    observed["complete"] = True
    path.write_text("{}")
    with pytest.raises(pb.ActionContractError): watch.other_windows(observed)


def test_sampler_reads_the_actual_admitted_cpu_cgroup_not_a_gpu_claim():
    import os
    import rank_window
    scope = os.environ.get("PRISMABUILD_ACTION_SCOPE")
    if not scope: pytest.skip("real cgroup sampler needs an admitted PB CPU scope")
    sample = rank_window.scope_memory(dict(rank=0, scope_id=scope))
    assert sample["scope_id"] == scope and sample["scope_memory_peak_bytes"] >= sample["scope_memory_current_bytes"]
    assert "CUDA coverage is unproven" in sample["note"]
    with pytest.raises(Refused, match="outside its owned"): rank_window.scope_memory(dict(rank=0, scope_id="foreign.slice"))
