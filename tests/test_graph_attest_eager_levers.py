"""D30 eager selector/output controls; no GPU numerical or served qualification."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parents[1] / "experiments/graph_attest_702"
sys.path.insert(0, str(HERE))
import eager_benchmark as benchmark
import tp2_recipe as recipe
from managed_window import Refused

MODE = "ship-eager-levers-4096"
FLAGS = {"TESSERA_E4M3_DECODE_ONCE": ("0", "1"),
         "TESSERA_GLM53_KDA_CONV_SPLIT": ("off", "on"),
         "TESSERA_ROUTED_PIECE_MAJOR": ("0", "1")}


def lever_plan(tmp_path, enabled=tuple(FLAGS)):
    path = tmp_path / "plan.txt"
    path.write_text("\n".join(
        f'eager4096_{arm} FABRIC=socket EAGER=1 MAX_BATCHED=4096 '
        f'SPEC_JSON={json.dumps(recipe.MTP, separators=(",", ":"))} '
        + " ".join(f"{key}={values[int(arm == 'on' and key in enabled)]}" for key, values in FLAGS.items())
        for arm in ("off", "on")) + "\n")
    return path


@pytest.mark.parametrize("enabled", [tuple(FLAGS), *[(key,) for key in FLAGS]])
def test_eager_lever_pair_and_both_container_ranks(tmp_path, enabled):
    arms = recipe.plan(lever_plan(tmp_path, enabled), mode=MODE)
    assert [(a["arm"], a["max_batched"], a["eager"]) for a in arms] == [
        ("eager4096_off", 4096, "1"), ("eager4096_on", 4096, "1")]
    config = dict(ts="/runtime", artifact="/artifact", fabric="socket", image="image", window_mode=MODE, profile_dir="/profiles")
    for rank in (0, 1):
        for arm in arms:
            argv = recipe.container(config, arm, dict(rank=rank, run_id="run", nonce="nonce"),
                                    tmp_path, tmp_path, tmp_path / "cid", {key: "incorrect" for key in FLAGS})
            env = dict(argv[i + 1].split("=", 1) for i, value in enumerate(argv[:-1]) if value == "-e")
            assert {key: env[key] for key in FLAGS} == arm["lever_env"]
            assert all(env[key] == choices[int(arm["arm"].endswith("_on") and key in enabled)] for key, choices in FLAGS.items())
            assert "--enforce-eager" in argv[-1] and "--max-num-seqs 1" in argv[-1]


@pytest.mark.parametrize("old,new", [("MAX_BATCHED=4096", "MAX_BATCHED=8192"),
    ("EAGER=1", "EAGER=0"), ("FABRIC=socket", "FABRIC=roce"),
    ("TESSERA_E4M3_DECODE_ONCE=1", "TESSERA_E4M3_DECODE_ONCE=true"),
    ("TESSERA_ROUTED_PIECE_MAJOR=0", "TESSERA_ROUTED_PIECE_MAJOR=1"),
    ("TESSERA_GLM53_KDA_CONV_SPLIT=off", "TESSERA_GLM53_KDA_CONV_SPLIT=auto"),
    ("TESSERA_E4M3_DECODE_ONCE=0", ""), ("EAGER=1", "EAGER=1 EXTRA=1")])
def test_eager_lever_scope_refusals(tmp_path, old, new):
    path = lever_plan(tmp_path)
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(Refused):
        recipe.plan(path, mode=MODE)


def test_eager_on_arm_cannot_be_noop(tmp_path):
    with pytest.raises(Refused):
        recipe.plan(lever_plan(tmp_path, ()), mode=MODE)


def test_eager_lever_real_driver_cpu_dry_run(tmp_path):
    import os
    (tmp_path / "config.json").write_text("{}")
    env = dict(os.environ, TS=str(HERE.parents[1]), ARTIFACT=str(tmp_path), RECEIPTS=str(tmp_path / "arms"),
               FABRIC="socket", WINDOW_MODE=MODE, MAX_NUM_SEQS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    done = subprocess.run(["bash", str(HERE / "drive_tp2.sh"), str(lever_plan(tmp_path)), "--dry-run"],
                          env=env, text=True, capture_output=True, timeout=30)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "eager4096_off" in done.stdout and "eager4096_on" in done.stdout
    for key, choices in FLAGS.items():
        assert all(f"{key}={value}" in done.stdout for value in choices)
    assert "107 GiB" in done.stdout and "1 Hz strict <2 GiB dual-rank abort" in done.stdout


def population():
    # Real protocol geometry, inexpensive synthetic CPU tokens.
    prompts = dict(warmup=1, prompts={str(length): {"1": [[[trial] * length] for trial in range(11)]}
                                    for length in (512, 2048, 8192)})
    prompt_sha = hashlib.sha256((json.dumps(prompts) + "\n").encode()).hexdigest()
    doc = dict(prompts_sha256=prompt_sha, model="glm53-artifact", labels=dict(mode="eager", fabric="socket"),
               config=dict(lens=[512, 2048, 8192], conc=[1], trials=10, warmup=1, output=128, temperature=0.0, ignore_eos=True), cells={})
    for length in (512, 2048, 8192):
        entries = [dict(trial=trial, requests=[dict(status=200, error=None, prompt_tokens_sent=length,
            prompt_sha256=hashlib.sha256(json.dumps([trial] * length).encode()).hexdigest(), completion_tokens=128,
            usage=dict(prompt_tokens=length, completion_tokens=128), generation=dict(done=True,
            choices=[dict(index=0, text="same output é", finish_reason="length")]))]) for trial in range(11)]
        doc["cells"][f"host-L{length}-c1"] = dict(L=length, c=1, skipped=False, complete=True, warmup=entries[0], trials=entries[1:])
    return doc, prompts, prompt_sha


def test_complete_outputs_hash_and_chunking_independence():
    doc, prompts, digest = population()
    before = benchmark.output_hashes(doc, prompts, digest)
    assert before["requests_per_arm"] == 33
    after_doc = copy.deepcopy(doc)
    choices = after_doc["cells"]["host-L8192-c1"]["warmup"]["requests"][0]["generation"]["choices"]
    choices[:] = [dict(index=0, text="same output ", finish_reason=None), dict(index=0, text="é", finish_reason="length")]
    after = benchmark.output_hashes(after_doc, prompts, digest)
    assert benchmark.compare_output_hashes(before, after)["passed"] is True
    choices[1]["text"] = "different"
    changed = benchmark.output_hashes(after_doc, prompts, digest)
    assert benchmark.compare_output_hashes(before, changed)["passed"] is False


@pytest.mark.parametrize("fault", ["missing_warmup", "missing_trial", "extra_cell", "wrong_prompt", "wrong_usage", "incomplete", "bad_finish", "wrong_config"])
def test_output_hashes_refuse_partial_or_unbound_population(fault):
    doc, prompts, digest = population()
    cell = doc["cells"]["host-L512-c1"]
    request = cell["warmup"]["requests"][0]
    if fault == "missing_warmup": cell.pop("warmup")
    if fault == "missing_trial": cell["trials"].pop()
    if fault == "extra_cell": doc["cells"]["undeclared"] = cell
    if fault == "wrong_prompt": request["prompt_sha256"] = "0" * 64
    if fault == "wrong_usage": request["usage"]["completion_tokens"] = 127
    if fault == "incomplete": request["generation"]["done"] = False
    if fault == "bad_finish": request["generation"]["choices"][0]["finish_reason"] = "stop"
    if fault == "wrong_config": doc["config"]["temperature"] = 1
    with pytest.raises(Refused):
        benchmark.output_hashes(doc, prompts, digest)


@pytest.fixture(autouse=True)
def local_generation_reader(tmp_path, monkeypatch):
    # Exercise the real maintained validator without requiring a box artifact.
    client = tmp_path / "client-source"
    client.mkdir()
    source = HERE.parents[1] / "tools/served_generation_client.py"
    (client / "u4_speed_client.py").write_bytes(source.read_bytes())
    monkeypatch.setattr(benchmark, "CLIENT", client)


def _frozen_flag_paths():
    import os
    import torch
    from tessera.serving import e4m3_prefill, flags, glm53_prefill, moe_route, native_window
    from test_piece_major_reader_boundaries import prepared
    from types import SimpleNamespace
    outcomes = []
    for enabled in (False, True):
        flags.reset_for_tests(e4m3_prefill.FLAG)
        os.environ[e4m3_prefill.FLAG] = str(int(enabled))
        os.environ["TESSERA_GLM53_KDA_CONV_SPLIT"] = "on" if enabled else "off"
        os.environ["TESSERA_ROUTED_PIECE_MAJOR"] = str(int(enabled))
        os.environ["TESSERA_ROUTED_FUSED"] = "1"
        os.environ["TESSERA_FUSED_E4M3_MMA"] = "e4m3"
        assert e4m3_prefill.enabled() is enabled
        assert glm53_prefill.kda_conv_split_mode() == ("on" if enabled else "off")
        assert moe_route._piece_major_admissible("e4m3") is enabled
        bundle = prepared()
        role = SimpleNamespace(name="projection", rows=bundle.rows, bundle=bundle)
        module = native_window.PreparedDenseNativeModule([role], rows=bundle.rows, columns=bundle.cols,
                                                         device=bundle.device, family=bundle.family)
        if e4m3_prefill.enabled():
            module.attach_decoded(e4m3_prefill.DecodedE4M3(
                torch.zeros(module.rows, module.columns, dtype=torch.float8_e4m3fn), torch.ones(module.rows)))
        outcomes.append(module.launch_pair_for(e4m3_prefill.MIN_M))
        assert module.launch_pair_for(e4m3_prefill.MIN_M - 1) == module.launch_pair
    assert outcomes[0] != outcomes[1]
    print("frozen runtime: all three flags read; decode-once launch changes above MIN_M")


def test_frozen_2db_flags_reach_different_runtime_selectors(tmp_path):
    pytest.importorskip("torch")
    import inspect
    import io
    import os
    import tarfile
    # Import only the runtime's archived source in a fresh process. Future
    # changes to the producer's src do not change or stale this pin check.
    archive = subprocess.check_output(["git", "archive", benchmark.RUNTIME_COMMIT, "src", "pyproject.toml",
                                       "tests/test_piece_major_reader_boundaries.py"], cwd=HERE.parents[1])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(tmp_path, filter="data")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(tmp_path / part) for part in ("src", "tests")))
    program = inspect.getsource(_frozen_flag_paths) + "\n_frozen_flag_paths()\n"
    done = subprocess.run([sys.executable, "-c", program], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "all three flags read" in done.stdout



@pytest.mark.parametrize("mismatch", [False, True])
def test_live_probe_path_writes_hashes_and_stops_on_mismatch(tmp_path, monkeypatch, mismatch):
    from types import SimpleNamespace
    doc, prompts, _ = population()
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps(prompts) + "\n")
    profile_manifest = tmp_path / "manifest.json"
    profile_manifest.write_text("{}")
    (benchmark.CLIENT / "comparison_inputs.py").write_text(
        "def load_manifest(path): return {}, None, None, None\n"
        "def declared_cells(manifest): return [dict(kind='prefill', L=512), dict(kind='decode', L=512)]\n")
    commands = []
    def command(argv, **kwargs):
        commands.append(argv)
        if "--trials" in argv:
            value = copy.deepcopy(doc)
            if mismatch and "_on" in argv[argv.index("--out") + 1]:
                value["cells"]["host-L8192-c1"]["trials"][-1]["requests"][0]["generation"]["choices"][0]["text"] = "different"
            Path(argv[argv.index("--out") + 1]).write_text(json.dumps(value))
            Path(argv[argv.index("--events") + 1]).write_text("fixture events\n")
    adapter = SimpleNamespace(config=dict(window_mode=MODE, profile_dir=str(tmp_path / "profiles"),
        prompts=str(prompt_path), profile_manifest=str(profile_manifest)), rdv=tmp_path / "rdv",
        identity=dict(rank=1), command=command, tick=lambda: None, envelope=SimpleNamespace(remaining=lambda: 30))
    arms = recipe.plan(lever_plan(tmp_path), mode=MODE)
    benchmark.probes(adapter, arms[0], dict(rank=0))
    assert (adapter.rdv / "arms/eager4096_off/output-hashes.json").exists()
    commands.clear()
    if mismatch:
        with pytest.raises(Refused, match="hashes differ"):
            benchmark.probes(adapter, arms[1], dict(rank=0))
        assert not any("profile" in argv for argv in commands)
    else:
        benchmark.probes(adapter, arms[1], dict(rank=0))
        assert len([argv for argv in commands if "profile" in argv]) == 2
        binding = json.loads((adapter.rdv / "arms/eager4096_on/invocation.json").read_bytes())
        assert binding["lever_env"] == arms[1]["lever_env"] and binding["output_hashes_sha256"]
        assert binding["schema"] == "tessera.eager_lever_invocation.v1"
    comparison = json.loads((adapter.rdv / "arms/eager4096_on/output-comparison.json").read_bytes())
    assert comparison["passed"] is not mismatch


@pytest.mark.parametrize("fault", ["reverse", "missing", "extra"])
def test_eager_lever_exact_order_population(tmp_path, fault):
    path = lever_plan(tmp_path)
    rows = path.read_text().splitlines()
    if fault == "reverse": rows.reverse()
    if fault == "missing": rows.pop()
    if fault == "extra": rows.append(rows[0])
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(Refused, match="exactly"):
        recipe.plan(path, mode=MODE)


@pytest.mark.parametrize("override", [dict(MAX_BATCHED="2048"), dict(MAX_NUM_SEQS="4"),
    dict(FABRIC="roce"), dict(FLOOR_GIB="1"), dict(EXPECT_PEAK_GIB="97"), dict(SERVE_MODE="streamed")])
def test_eager_lever_inputs_do_not_relax_guards(tmp_path, override):
    (tmp_path / "config.json").write_text("{}")
    env = dict(TS=str(HERE.parents[1]), ARTIFACT=str(tmp_path), RECEIPTS=str(tmp_path / "arms"), FABRIC="socket", WINDOW_MODE=MODE)
    assert recipe.inputs(env, live=False)["window_mode"] == MODE
    with pytest.raises(Refused):
        recipe.inputs(dict(env, **override), live=False)


def test_eager_lever_plan_is_in_exact_producer_roster_and_old_pairs_stay_unchanged(tmp_path):
    assert recipe.plan(HERE / "plan-eager-levers-4096.txt", mode=MODE) == recipe.plan(lever_plan(tmp_path), mode=MODE)
    assert "plan-eager-levers-4096.txt" in recipe.PRODUCER_FILES
    for mode, plan in ((benchmark.MODE, "plan-eager-window4.txt"), (benchmark.SHIP_MODE, "plan-eager-ship-8192.txt")):
        arms = recipe.plan(HERE / plan, mode=mode)
        assert [(a["arm"], str(a["max_batched"])) for a in arms] == benchmark.PAIRS[mode]
        assert all("lever_env" not in arm for arm in arms)
        config = dict(ts="/runtime", artifact="/artifact", fabric="socket", image="image", window_mode=mode, profile_dir="/profiles")
        argv = recipe.container(config, arms[0], dict(rank=0, run_id="run", nonce="nonce"), tmp_path, tmp_path, tmp_path / "cid", {})
        assert not any(value.startswith(key + "=") for value in argv for key in FLAGS)

