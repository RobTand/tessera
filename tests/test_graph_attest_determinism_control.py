"""Bounded seeded controls: complete outputs, preserved safety and isolated levers."""
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1] / "experiments/graph_attest_702"
sys.path.insert(0, str(HERE))
import tp2_recipe as recipe
from managed_window import CLEANUP_SECONDS, Refused

import eager_determinism as control
from eager_determinism import (FLAGS, MODE, MODES, NAMES, PHASE_MODES, PHASE_PEER_WAIT_SECONDS,
                               PHASE_SECONDS, PIECE_MAJOR_MODE, PROTOCOL, VALUES)

CONTROL_ARMS, PM_ARMS = NAMES[:2], NAMES[2:]


def local_generation_reader(directory):
    """Point the control at the real maintained validator without a box artifact.

    ``eager_benchmark.CLIENT`` is a shared-storage directory the hosted runner
    does not have; the same source is tools/served_generation_client.py.
    """
    import eager_benchmark
    client = Path(directory) / "client-source"
    client.mkdir(parents=True, exist_ok=True)
    source = HERE.parents[1] / "tools/served_generation_client.py"
    (client / "u4_speed_client.py").write_bytes(source.read_bytes())
    return eager_benchmark, client


@pytest.fixture(autouse=True)
def local_client(tmp_path, monkeypatch):
    benchmark, client = local_generation_reader(tmp_path)
    monkeypatch.setattr(benchmark, "CLIENT", client)


def phase_plan(tmp_path, mode):
    """Per-phase arm rows: the control phase is exactly its two all-OFF restart
    blocks; the piece-major phase is exactly its matched OFF block plus the
    piece-major arm."""
    path = tmp_path / f"plan-{mode}.txt"
    lines = []
    for name in MODES[mode]:
        lines.append(f'{name} FABRIC=socket EAGER=1 MAX_BATCHED=4096 '
                     f'SPEC_JSON={json.dumps(recipe.MTP, separators=(",", ":"))} ' +
                     " ".join(f"{key}={value}" for key, value in zip(FLAGS, VALUES[NAMES.index(name)])))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_old_multi_lever_scope_is_not_admitted(tmp_path):
    env = dict(TS=str(HERE.parents[1]), ARTIFACT=str(tmp_path), RECEIPTS=str(tmp_path / "arms"),
               FABRIC="socket", WINDOW_MODE="investigate-eager-determinism-2048")
    with pytest.raises(Refused, match="unknown WINDOW_MODE"):
        recipe.inputs(env, live=False)

def test_control_admits_two_fresh_off_blocks_and_pm_pair_admits_matched_off_then_piece_major(tmp_path):
    arms = recipe.plan(phase_plan(tmp_path, MODE), mode=MODE)
    assert [arm["arm"] for arm in arms] == list(CONTROL_ARMS)
    assert all(arm["max_batched"] == 4096 and arm["eager"] == "1" and arm["fabric"] == "socket"
               for arm in arms)
    assert [arm["lever_env"] for arm in arms] == [dict(zip(FLAGS, VALUES[0]))] * 2
    arms = recipe.plan(phase_plan(tmp_path, PIECE_MAJOR_MODE), mode=PIECE_MAJOR_MODE)
    assert [arm["arm"] for arm in arms] == list(PM_ARMS)
    assert [arm["lever_env"] for arm in arms] == [dict(zip(FLAGS, VALUES[2])), dict(zip(FLAGS, VALUES[3]))]


@pytest.mark.parametrize("mode,old,new", [
    (MODE, "control_off_restart", "control_pm_off"),                       # other phase's arm
    (MODE, "control_off_restart", "control_off_first"),                    # reordered/repeated pair
    (MODE, "MAX_BATCHED=4096", "MAX_BATCHED=2048"),
    (MODE, "FABRIC=socket", "FABRIC=roce"),
    (MODE, "TESSERA_E4M3_DECODE_ONCE=0", "TESSERA_E4M3_DECODE_ONCE=1"),    # decode-once stays out
    (PIECE_MAJOR_MODE, "control_piece_major", "control_kda_split"),        # removed arm
    (PIECE_MAJOR_MODE, "MAX_BATCHED=4096", "MAX_BATCHED=2048"),
    (PIECE_MAJOR_MODE, "TESSERA_ROUTED_PIECE_MAJOR=1", "TESSERA_ROUTED_PIECE_MAJOR=0"),
    (PIECE_MAJOR_MODE, "TESSERA_GLM53_KDA_CONV_SPLIT=off", "TESSERA_GLM53_KDA_CONV_SPLIT=on"),
])
def test_phase_refuses_reordered_blocks_or_any_other_lever(tmp_path, mode, old, new):
    path = phase_plan(tmp_path, mode)
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(Refused):
        recipe.plan(path, mode=mode)


def test_existing_ship_pair_still_refuses_off_off(tmp_path):
    path = phase_plan(tmp_path, MODE)
    rows = path.read_text().splitlines()[:2]
    path.write_text(rows[0].replace(NAMES[0], "eager4096_off") + "\n" +
                    rows[1].replace(NAMES[1], "eager4096_on") + "\n")
    with pytest.raises(Refused, match="at least one on"):
        recipe.plan(path, mode=recipe.EAGER_LEVER_MODE)


def population_fixture():
    import hashlib
    from eager_benchmark import MODEL
    from seeded_control_client import SCHEMA
    prompts = dict(warmup=1, prompts={"2048": {"1": [[[trial] * 2048] for trial in range(11)]}})
    digest = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    records = []
    for trial, prompt in enumerate(control.prompt_population(prompts)):
        records.append(dict(trial=trial, slot=0, warmup=trial == 0, request_id=f"L2048-c1/trial{trial}/slot0",
                            request_seed=trial, request_payload=dict(
                                model=MODEL, prompt=prompt, max_tokens=128, temperature=0.0, seed=trial,
                                ignore_eos=True, stream=True, stream_options=PROTOCOL["stream_options"]),
                            status=200, error=None, prompt_tokens_sent=2048,
                            prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                            started_unix=1000.0 + trial,
                            ended_unix=1000.0 + trial + 1.0 + 0.01 * trial,
                            completion_tokens=128, usage=dict(prompt_tokens=2048, completion_tokens=128, total_tokens=2176),
                            generation=dict(done=True, choices=[dict(index=0, text="full completion", finish_reason="length")])))
    return prompts, digest, dict(schema=SCHEMA, model=MODEL, protocol=PROTOCOL,
                                 prompts_sha256=digest, requests=records)


def timed_fixture(name, *, change=False):
    """The seeded population as the arm-wide record set; actual per-request
    durations differ by arm so the selfvariation observation is real."""
    _, _, result = population_fixture()
    offset = 0.05 * (NAMES.index(name) + 1)
    for record in result["requests"]:
        record["started_unix"] = 1000.0 + record["trial"]
        record["ended_unix"] = record["started_unix"] + 1.0 + offset + 0.01 * record["trial"]
    if change:
        result["requests"][10]["generation"]["choices"][0]["text"] += " different"
    return result


def test_control_full_text_comparison_includes_warmup_and_suffix():
    import copy
    prompts, digest, result = population_fixture()
    before = control.output_population(result, prompts, digest)
    changed = copy.deepcopy(result)
    changed["requests"][0]["generation"]["choices"][0]["text"] += " changed suffix"
    after = control.output_population(changed, prompts, digest)
    comparison = control.compare(before, after)
    assert comparison["passed"] is False
    assert comparison["differing_requests"] == ["L2048-c1/trial0/slot0"]
    assert before["requests_per_arm"] == 11
    assert before["outputs"]["L2048-c1/trial10/slot0"]["text"] == "full completion"


@pytest.mark.parametrize("mutation", ["missing-warmup", "extra-trial", "seed", "payload-seed", "usage",
                                      "finish", "done", "prompt", "scope", "server-seed"])
def test_control_refuses_incomplete_or_unpaired_request_inputs(mutation):
    import copy
    prompts, digest, original = population_fixture()
    result = copy.deepcopy(original)
    record = result["requests"][0]
    if mutation == "missing-warmup": result["requests"].pop(0)
    elif mutation == "extra-trial": result["requests"].append(copy.deepcopy(record))
    elif mutation == "seed": record["request_seed"] = 99
    elif mutation == "payload-seed": record["request_payload"]["seed"] = 99
    elif mutation == "usage": record["usage"]["completion_tokens"] = 127
    elif mutation == "finish": record["generation"]["choices"][0]["finish_reason"] = "stop"
    elif mutation == "done": record["generation"]["done"] = False
    elif mutation == "prompt": record["prompt_sha256"] = "wrong"
    elif mutation == "scope": result["protocol"]["lens"] = [512, 2048, 8192]
    elif mutation == "server-seed": result["protocol"]["server_seed"] = 99
    with pytest.raises(Refused):
        control.output_population(result, prompts, digest)


def store_population(root, name, *, change=False):
    import hashlib
    prompts, _, _ = population_fixture()
    prompt_path = root.parent / "prompts.json"
    prompt_path.write_text(json.dumps(prompts))
    result = timed_fixture(name, change=change)
    result["prompts_sha256"] = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    directory = root / "arms" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "invocation.json").write_text(json.dumps(dict(source_bindings=dict(prompts=str(prompt_path)))))
    (directory / "control.json").write_text(json.dumps(result))
    (directory / "output-hashes.json").write_text(json.dumps(
        control.output_population(result, prompts, result["prompts_sha256"])))


def test_control_baseline_retains_mismatch_and_binds_only_raw_records(tmp_path):
    root = tmp_path / "run"
    store_population(root, NAMES[0])
    store_population(root, NAMES[1], change=True)
    reference, comparison = control.control_baseline(root)
    assert comparison["passed"] is False                      # retained, not a completion failure
    assert comparison["differing_requests"] == ["L2048-c1/trial10/slot0"]
    assert comparison["requests_per_arm"] == 11
    assert reference["requests_per_arm"] == 11 and reference == control.stored_outputs(root, NAMES[0])
    (root / "arms" / NAMES[1] / "output-comparison.json").write_text('{"passed": true}')
    comparison = control.control_baseline(root)[1]
    assert comparison["passed"] is False                       # a label cannot authorize anything
    store_population(root, NAMES[1])
    reference, comparison = control.control_baseline(root)
    assert comparison["passed"] is True and comparison["requests_per_arm"] == 11
    path = root / "arms" / NAMES[1] / "output-hashes.json"
    data = json.loads(path.read_bytes())
    data["outputs"]["L2048-c1/trial10/slot0"]["text"] = "tampered"
    path.write_text(json.dumps(data))
    with pytest.raises(Refused, match="own raw records"):
        control.stored_outputs(root, NAMES[1])


def test_control_preregistration_records_actual_selfvariation_never_a_tolerance(tmp_path):
    root = tmp_path / "run"
    store_population(root, NAMES[0])
    store_population(root, NAMES[1])
    record = control.preregister(root, dict(window_mode=MODE))
    assert record["schema"] == "tessera.eager_control_preregistration.v1"
    assert record["control_root"] == str(root)
    assert record["off_deterministic"] is control.control_baseline(root)[1]["passed"]
    assert record["off_timing_seconds"] == [control.timing_rows(root, name) for name in CONTROL_ARMS]
    off = record["off_timing_seconds"]
    assert len(off) == 2 and all(len(row) == 10 and all(value > 0 for value in row) for row in off)
    assert record["per_request_observed_range_seconds"] == [max(values) - min(values) for values in zip(*off)]
    assert record["numerical_quality_metric"] is None and record["numerical_quality_tolerance"] is None
    assert record["timing_judgment"] and record["quality_limitation"]
    assert record["adoption"] is False and record["performance_claim"] is False
    persisted = json.loads((root / "control-preregistration.json").read_text())
    assert persisted == record


def test_piece_major_preregistration_binds_control_root_and_actual_records(tmp_path):
    baseline = tmp_path / "control-run"
    store_population(baseline, NAMES[0])
    store_population(baseline, NAMES[1], change=True)          # mismatched restart retained
    root = tmp_path / "pm-run"
    store_population(root, NAMES[2])
    config = dict(window_mode=PIECE_MAJOR_MODE, control_root=str(baseline))
    record = control.preregister(root, config, matched=True)
    assert record["control_root"] == str(baseline)
    assert record["off_deterministic"] is False                # recorded, still no prohibition
    assert len(record["off_timing_seconds"]) == 3
    assert record["off_timing_seconds"][2] == control.timing_rows(root, NAMES[2])
    off = record["off_timing_seconds"]
    assert record["per_request_observed_range_seconds"] == [max(values) - min(values) for values in zip(*off)]
    assert record["numerical_quality_metric"] is None and record["numerical_quality_tolerance"] is None
    control.require_piece_major_control(root, config)          # no bit equality gate on the baseline mismatch
    tampered = json.loads((root / "control-preregistration.json").read_text())
    tampered["off_timing_seconds"][2] = [value + 5 for value in tampered["off_timing_seconds"][2]]
    (root / "control-preregistration.json").write_text(json.dumps(tampered))
    with pytest.raises(Refused, match="preregistration differs from its actual matched OFF observations"):
        control.require_piece_major_control(root, config)
    (root / "control-preregistration.json").unlink()           # an absent preregistration is not synthesized
    with pytest.raises((Refused, OSError)):
        control.require_piece_major_control(root, config)
    control.preregister(root, config, matched=True)
    control.require_piece_major_control(root, config)
    restored = json.loads((baseline / "arms" / NAMES[1] / "output-hashes.json").read_text())
    restored["outputs"]["L2048-c1/trial10/slot0"]["text"] = "tampered"
    (baseline / "arms" / NAMES[1] / "output-hashes.json").write_text(json.dumps(restored))
    with pytest.raises(Refused, match="own raw records"):      # baseline tampering is never accepted
        control.require_piece_major_control(root, config)


@pytest.mark.parametrize("mode", PHASE_MODES)
def test_phase_requires_a_synthetic_data_manifest_for_pb_staging(tmp_path, mode):
    (tmp_path / "config.json").write_text("{}")
    env = dict(TS=str(HERE.parents[1]), ARTIFACT=str(tmp_path), RECEIPTS=str(tmp_path / "arms"),
               FABRIC="socket", WINDOW_MODE=mode, SOURCE_COMMIT="a" * 40, SOURCE_SHA256="b" * 64,
               PRODUCER_COMMIT="c" * 40, PRODUCER_SHA256="d" * 64)
    if mode == PIECE_MAJOR_MODE:
        env["CONTROL_ROOT"] = str(tmp_path / "control-run")
        env["PROFILE_MANIFEST"] = str(tmp_path / "profile-manifest.json")
    with pytest.raises(Refused, match="bounded resident phases require the actual DATA_MANIFEST"):
        recipe.inputs(env, live=False)
    env["DATA_MANIFEST"] = str(tmp_path / "data-manifest.json")  # synthetic; no live shared path
    (tmp_path / "data-manifest.json").write_text(json.dumps(dict(files=[])))
    config = recipe.inputs(env, live=False)
    assert config["data_manifest"] == env["DATA_MANIFEST"]


def test_piece_major_phase_requires_explicit_control_root_and_profile_manifest(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    env = dict(TS=str(HERE.parents[1]), ARTIFACT=str(tmp_path), RECEIPTS=str(tmp_path / "arms"),
               FABRIC="socket", WINDOW_MODE=PIECE_MAJOR_MODE, SOURCE_COMMIT="a" * 40, SOURCE_SHA256="b" * 64,
               PRODUCER_COMMIT="c" * 40, PRODUCER_SHA256="d" * 64, DATA_MANIFEST=str(tmp_path / "data-manifest.json"))
    (tmp_path / "data-manifest.json").write_text(json.dumps(dict(files=[])))
    with pytest.raises(Refused, match="requires the actual completed OFF/OFF CONTROL_ROOT"):
        recipe.inputs(env, live=False)
    env["CONTROL_ROOT"] = str(tmp_path / "control-run")
    with pytest.raises(Refused, match="same-instrument L2048 PROFILE_MANIFEST"):
        recipe.inputs(env, live=False)
    env["PROFILE_MANIFEST"] = str(tmp_path / "profile-manifest.json")
    config = recipe.inputs(env, live=False)
    assert config["control_root"] == env["CONTROL_ROOT"] and config["profile_manifest"] == env["PROFILE_MANIFEST"]
    env["WINDOW_MODE"] = MODE
    config = recipe.inputs(env, live=False)
    assert "control_root" not in config and "profile_manifest" not in config


@pytest.mark.parametrize("mode", PHASE_MODES)
def test_phase_rows_own_bounded_caps_seeds_and_separate_profiles(tmp_path, mode):
    import window_driver as driver
    (tmp_path / "config.json").write_text("{}")
    env = dict(TS=str(HERE.parents[1]), ARTIFACT=str(tmp_path), RECEIPTS=str(tmp_path / "arms"),
               FABRIC="socket", WINDOW_MODE=mode, SOURCE_COMMIT="a" * 40, SOURCE_SHA256="b" * 64,
               PRODUCER_COMMIT="c" * 40, PRODUCER_SHA256="d" * 64)
    if mode == PIECE_MAJOR_MODE:
        env["CONTROL_ROOT"] = str(tmp_path / "control-run")
        env["PROFILE_MANIFEST"] = str(tmp_path / "profile-manifest.json")
    env["DATA_MANIFEST"] = str(tmp_path / "data-manifest.json")  # synthetic; no live shared path
    (tmp_path / "data-manifest.json").write_text(json.dumps(dict(files=[])))
    config = recipe.inputs(env, live=False)
    (tmp_path / "inputs.json").write_text(json.dumps(config))
    assert config["window_seconds"] == PHASE_SECONDS and config["peer_wait_seconds"] == PHASE_PEER_WAIT_SECONDS
    assert config["control_protocol"] == PROTOCOL
    assert config["data_manifest"] == env["DATA_MANIFEST"]
    rows = driver.rows(tmp_path, config, env)
    assert [row["demand"] for row in rows] == [dict(cpu=8, mem_gb=104, gpu=1), dict(cpu=6, mem_gb=104, gpu=1)]
    assert all(row["priority"] == -10 and row["gpu_memory_gb"] == 102 and row["exclusive"] and
               row["measurement"] and row["host_class"] == "gb10" and row["timeout_s"] == PHASE_SECONDS
               for row in rows)
    assert all(row["data_manifest"] == env["DATA_MANIFEST"] and row["residency"] == "stage" and
               row["residency_ram"] == "auto" and row["residency_share"] == "auto" for row in rows)
    assert all(row["env"]["GRAPH_PEER_WAIT_SECONDS"] == str(PHASE_PEER_WAIT_SECONDS) for row in rows)
    assert all(key in rows[0]["env"] for key in ("WINDOW_MODE", "SOURCE_COMMIT", "SOURCE_SHA256", "DATA_MANIFEST"))
    assert all(("CONTROL_ROOT" in row["env"] and "PROFILE_MANIFEST" in row["env"])
               if mode == PIECE_MAJOR_MODE else
               ("CONTROL_ROOT" not in row["env"] and "PROFILE_MANIFEST" not in row["env"])
               for row in rows)
    for arm in recipe.plan(phase_plan(tmp_path, mode), mode=mode):
        commands = [recipe.serve(config, arm, rank) for rank in (0, 1)]
        for command in commands:
            assert command[command.index("--seed") + 1] == "0"
            assert command[command.index("--max-num-batched-tokens") + 1] == "4096"
            assert "--enforce-eager" in command
            if mode == MODE:
                assert "--profiler-config" not in command
            else:
                profiler = json.loads(command[command.index("--profiler-config") + 1])
                assert profiler["torch_profiler_dir"] == config["profile_dir"]
        container = recipe.container(config, arm, dict(rank=0, run_id="test", nonce="a" * 32),
                                     tmp_path / "out", tmp_path / "ext", tmp_path / "cid", {})
        assert all(f"{key}={value}" in container for key, value in arm["lever_env"].items())
        assert "PYTHONPATH=/ga702-observer:/digest" in container
        assert f"TESSERA_ROUTE_TRACE=/out/{arm['arm']}.rank0.route-trace.json" in container
        profile_dir = Path(config["profile_dir"]) / arm["arm"]
        assert (f"{profile_dir}:{profile_dir}" in container) == (mode == PIECE_MAJOR_MODE)


def recording_command_store(commands, *, changed_on=None, changed=False):
    """Fake adapter command: the seeded all11 client writes its own population;
    profile instrument calls write their events stream."""
    def command(argv, **kwargs):
        commands.append(argv)
        if "--server-seed" in argv:
            name = Path(argv[argv.index("--out") + 1]).parent.name
            Path(argv[argv.index("--out") + 1]).write_text(
                json.dumps(timed_fixture(name, change=changed and name == changed_on)))
        elif "--kind" in argv:
            Path(argv[argv.index("--events") + 1]).write_text("fixture profile events\n")
    return command


@pytest.mark.parametrize("changed", [False, True], ids=["matched-piece-major", "differing-piece-major"])
def test_piece_major_probe_reads_matched_comparator_and_keeps_profiles_after_all11(tmp_path, monkeypatch, changed):
    import eager_benchmark as benchmark
    from types import SimpleNamespace
    baseline = tmp_path / "control-run"
    root = tmp_path / "pm-run"
    store_population(baseline, NAMES[0])
    store_population(baseline, NAMES[1])
    store_population(root, NAMES[2])
    prompts_path = tmp_path / "prompts.json"  # the single store_population prompt file
    prompts, _, _ = population_fixture()
    prompts_path.write_text(json.dumps(prompts))
    client = benchmark.CLIENT
    (client / "comparison_inputs.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "def load_manifest(path): return json.loads(Path(path).read_text()), None, None, None\n"
        "def declared_cells(manifest): return manifest['cells']\n")
    monkeypatch.setattr(benchmark, "CLIENT", client)
    manifest = tmp_path / "profile-manifest.json"
    cells = [dict(kind="prefill", L=2048), dict(kind="decode", L=2048)]
    manifest.write_text(json.dumps(dict(cells=cells)))
    config = dict(window_mode=PIECE_MAJOR_MODE, control_root=str(baseline),
                  profile_dir=str(tmp_path / "profiles"), prompts=str(prompts_path),
                  profile_manifest=str(manifest))
    control.preregister(root, config, matched=True)  # the pm_off arm's own probes wrote this in production
    commands = []
    adapter = SimpleNamespace(config=config, rdv=root, identity=dict(rank=1),
                              command=recording_command_store(commands, changed=changed, changed_on=NAMES[-1]),
                              tick=lambda: None, envelope=SimpleNamespace(remaining=lambda: 30))
    arm = {row["arm"]: row for row in recipe.plan(phase_plan(tmp_path, PIECE_MAJOR_MODE), mode=PIECE_MAJOR_MODE)}[NAMES[-1]]
    finished = control.probes(adapter, arm, dict(rank=0))
    seeded = [argv for argv in commands if "--server-seed" in argv]
    profiles = [argv for argv in commands if "--kind" in argv]
    powers = [argv for argv in commands if "--window" in argv]
    assert len(seeded) == 1 and seeded[0] == commands[0]      # the eleven seeded requests run first
    assert seeded[0][seeded[0].index("--server-seed") + 1] == "0"
    assert seeded[0][seeded[0].index("--request-seed-base") + 1] == "0"
    assert profiles and commands.index(profiles[0]) > commands.index(seeded[0])  # separate, after all11
    assert [(argv[argv.index("--kind") + 1], int(argv[argv.index("--length") + 1])) for argv in profiles] == \
        [(cell["kind"], cell["L"]) for cell in cells]
    assert all(argv[argv.index("--directory") + 1] == str(Path(config["profile_dir"]) / NAMES[-1])
               for argv in profiles)
    assert len(powers) == 2 and [argv[argv.index("--host") + 1] for argv in powers] == ["sparklina", "sparky"]
    assert finished["profile_dir"].endswith("/" + NAMES[-1]) and finished["events"].endswith("events.jsonl")
    binding = json.loads((root / "arms" / NAMES[-1] / "invocation.json").read_text())
    assert binding["timing_sha256"] == benchmark.sha(root / "arms" / NAMES[-1] / "control.json")
    recorded = json.loads((root / "arms" / NAMES[-1] / "output-comparison.json").read_text())
    assert recorded["passed"] is (not changed) and recorded["requests_per_arm"] == 11
    if changed:
        assert recorded["differing_requests"] == ["L2048-c1/trial10/slot0"]


@pytest.mark.parametrize("changed", [False, True], ids=["matching-off", "mismatched-off"])
def test_control_probe_dispatch_records_off_restart_comparison_without_failing(tmp_path, changed):
    from types import SimpleNamespace
    root = tmp_path / "run"
    prompts_path = tmp_path / "prompts.json"
    prompts, _, _ = population_fixture()
    prompts_path.write_text(json.dumps(prompts))
    config = dict(window_mode=MODE, prompts=str(prompts_path))
    commands = []
    adapter = SimpleNamespace(config=config, rdv=root, identity=dict(rank=1),
                              command=recording_command_store(commands, changed=changed, changed_on=NAMES[1]),
                              tick=lambda: None, envelope=SimpleNamespace(remaining=lambda: 30))
    finished = None
    for arm in recipe.plan(phase_plan(tmp_path, MODE), mode=MODE):
        finished = control.probes(adapter, arm, dict(rank=0))
    assert finished["correctness_only"] is True and "profile_dir" not in finished and "events" not in finished
    assert len(commands) == 2
    for command in commands:
        assert command[command.index("--server-seed") + 1] == "0"
        assert command[command.index("--request-seed-base") + 1] == "0"
    recorded = json.loads((root / "arms" / NAMES[1] / "output-comparison.json").read_text())
    assert recorded["passed"] is (not changed) and recorded["requests_per_arm"] == 11
    prereg = json.loads((root / "control-preregistration.json").read_text())
    assert prereg == control.preregistration_value(root, config)
    assert prereg["off_deterministic"] is (not changed)        # mismatch retained, never gated


def test_seeded_client_sends_seed_and_retains_entire_http_stream():
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from seeded_control_client import request
    captured = []

    class Server(BaseHTTPRequestHandler):
        def do_POST(self):
            captured.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for text, finish, count in (("first", None, 1), (" suffix", "length", 128)):
                chunk = dict(choices=[dict(index=0, text=text, finish_reason=finish)],
                             usage=dict(prompt_tokens=2048, completion_tokens=count, total_tokens=2048 + count))
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Server)
    worker = threading.Thread(target=server.handle_request)
    worker.start()
    try:
        response = request(f"http://127.0.0.1:{server.server_port}", "glm53-artifact", [1] * 2048, 7, 5)
        assert captured[0]["seed"] == response["request_seed"] == 7
        assert captured[0] == response["request_payload"]
        assert response["started_unix"] <= response["ended_unix"]
        assert response["generation"]["done"] is True and response["error"] is None
        assert "".join(choice["text"] for choice in response["generation"]["choices"]) == "first suffix"
        assert response["completion_tokens"] == 128
    finally:
        server.server_close()
        worker.join(timeout=5)


@pytest.mark.parametrize("phase", ["control", "pm"])
@pytest.mark.parametrize("changed", [False, True], ids=["matching-off", "mismatched-off"])
def test_real_rank_lifecycle_restarts_both_ranks_and_retains_mismatch_without_failing(tmp_path, phase, changed):
    import os
    import signal
    import subprocess
    import time
    import managed_window as window
    from test_graph_attest_window_scenarios import alive, make_identity, write_claim
    root, queue = tmp_path / "run", tmp_path / "queue"
    root.mkdir()
    mode = MODE if phase == "control" else PIECE_MAJOR_MODE
    if phase == "pm":
        baseline = tmp_path / "control-run"
        store_population(baseline, NAMES[0])
        store_population(baseline, NAMES[1])
    arms = recipe.plan(phase_plan(tmp_path, mode), mode=mode)
    start = time.time()
    identities = [make_identity(rank, start, start + 20) for rank in (0, 1)]
    for identity in identities:
        identity["peer_wait_seconds"] = PHASE_PEER_WAIT_SECONDS

    children = []
    try:
        for identity in identities:
            write_claim(queue, identity)
            window.atomic_json(tmp_path / f"identity{identity['rank']}.json", identity)
            log = (tmp_path / f"controller{identity['rank']}.log").open("w")
            process = subprocess.Popen([sys.executable, str(Path(__file__)), "--control-rank",
                                        str(tmp_path), str(identity["rank"]), str(int(changed)), phase],
                                       stdout=log, stderr=subprocess.STDOUT)
            children.append((process, log))
        for process, log in children:
            process.wait(timeout=25)
            log.close()
        outcomes = [window.read_json(root / f"outcome-rank{rank}.json") for rank in (0, 1)]
        pids = [int(path.read_text()) for path in root.glob("*.pid")]
        assert not [pid for pid in pids if alive(pid)]
        assert all(outcome["simulation"] and not outcome["ownership_released"] for outcome in outcomes)
        completed = list(MODES[mode])
        assert all(outcome["returncode"] == 0 and outcome["completed_arms"] == completed for outcome in outcomes)
        first, second = (NAMES[0], NAMES[1]) if phase == "control" else (NAMES[2], NAMES[3])
        for rank in (0, 1):
            assert (root / f"{first}.rank{rank}.pid").read_text() != (root / f"{second}.rank{rank}.pid").read_text()
    finally:
        for process, log in children:
            if process.poll() is None:
                process.kill()
                process.wait()
            log.close()
        for path in root.glob("*.pid"):
            pid = int(path.read_text())
            if alive(pid):
                os.killpg(pid, signal.SIGKILL)


def test_native_observer_counts_the_actual_forwarded_layout_boolean_without_torch():
    import ast
    import threading
    from types import SimpleNamespace
    source = HERE / "observer/usercustomize.py"
    tree = ast.parse(source.read_text())
    node = next(row for row in tree.body if isinstance(row, ast.FunctionDef) and row.name == "_ga_patch_routed")
    calls = []
    native = SimpleNamespace(routed_fused_forward=lambda *args: calls.append(args) or "native-result")
    module = SimpleNamespace(_ext=lambda library: native)
    counts = dict(serving=SimpleNamespace(on=True), lock=threading.Lock(), counts={}, dirty=False)
    namespace = {"_GA": counts}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    namespace["_ga_patch_routed"](module)
    for piece_major in (False, True):
        args = [None] * 33
        args[0], args[2], args[20], args[32] = 0, SimpleNamespace(shape=(2048, 32)), piece_major, 64
        assert module._ext("e4m3mma").routed_fused_forward(*args) == "native-result"
    assert [args[20] for args in calls] == [False, True]
    assert len(counts["counts"]) == 2 and all(value == 1 for value in counts["counts"].values())
    assert any("piece_major=0" in key for key in counts["counts"])
    assert any("piece_major=1" in key for key in counts["counts"])
    counts["serving"].on = False
    module._ext("e4m3mma").routed_fused_forward(*calls[0])
    assert sum(counts["counts"].values()) == 2  # startup is not a served launch



if __name__ == "__main__":
    import managed_window as window
    import rank_window
    from test_graph_attest_window_scenarios import CpuArm
    parent, rank, changed, phase = Path(sys.argv[2]), int(sys.argv[3]), bool(int(sys.argv[4])), sys.argv[5]
    mode = MODE if phase == "control" else PIECE_MAJOR_MODE
    benchmark, client = local_generation_reader(parent / f"client-rank{rank}")
    benchmark.CLIENT = client
    root, queue = parent / "run", parent / "queue"
    owned = window.read_json(parent / f"identity{rank}.json")
    envelope = window.Envelope(owned["window_end_unix"], cleanup_seconds=1)
    config = dict(window_mode=mode, fabric="socket")
    if phase == "pm":
        config["control_root"] = str(parent / "control-run")

    class ControlArm(CpuArm):
        def probes(self, arm, peer):
            name = arm["arm"]
            store_population(root, name,
                             change=changed and name == (NAMES[1] if phase == "control" else NAMES[-1]))
            if name == NAMES[2]:
                control.preregister(root, config, matched=True)
            return dict(correctness_only=True)

    adapter = ControlArm(root, owned, envelope, "success")
    arms = [dict(arm=name) for name in MODES[mode]]
    raise SystemExit(rank_window.run_rank(config, owned, queue, root, arms, adapter, envelope, poll_seconds=.01))
