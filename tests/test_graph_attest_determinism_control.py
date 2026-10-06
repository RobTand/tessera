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


def staged_packet(tmp_path, monkeypatch):
    """Real temporary files and FDs; only the external lease service is substituted."""
    import hashlib
    import os
    import types
    import eager_benchmark as benchmark
    artifact, panel, runtime, stage, rdv = [tmp_path / name for name in ("artifact", "panel", "runtime", "stage", "rdv")]
    for directory in (artifact, panel, runtime / "src/tessera/serving", stage, rdv):
        directory.mkdir(parents=True)
    (artifact / "config.json").write_text('{"model_type":"fixture"}')
    for index in range(127):
        (artifact / f"weights-{index:03}.bin").write_bytes(bytes([index]) * 64)
    entries = []
    for path in sorted(artifact.iterdir()):
        entries.append(dict(name=path.name, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        (stage / path.name).write_bytes(path.read_bytes())
    content = tmp_path / "content.json"
    content.write_text(json.dumps(entries))
    data = tmp_path / "data.json"
    data.write_text(json.dumps(dict(entries=[dict(path=str(artifact / row["name"]), offset=0,
        bytes=row["bytes"], sha256=row["sha256"]) for row in entries], entry_count=128,
        total_bytes=sum(row["bytes"] for row in entries))))
    prompts, _, _ = population_fixture()
    (panel / "prompts.json").write_text(json.dumps(prompts))
    (runtime / "src/tessera/serving/runtime_contract.json").write_text('{"schema":"fixture"}')
    monkeypatch.setattr(benchmark, "PANEL", panel)
    read = lambda path: (json.loads(Path(path).read_bytes()), "identity")
    key = lambda path, offset: f"{path}:{offset}"
    sdk = types.SimpleNamespace(read_data_manifest=read, injected_context=lambda: dict(ok=True,
        ctx=dict(action_key="a" * 64, queue_root=str(tmp_path / "queue"), map_path=str(tmp_path / "map"))),
        PoolQueue=lambda root: types.SimpleNamespace(root=root), RESIDENCY="residency", residency_map_key=key,
        read_residency_map=lambda path: dict(manifest_sha256=hashlib.sha256(data.read_bytes()).hexdigest(), tier_id="fixture"),
        covers_for_keys=lambda *args, **kwargs: dict(ok=True, covers=[], expected={key(row["path"], row["offset"]):
            dict(bytes=row["bytes"], sha256=row["sha256"]) for row in read(data)[0]["entries"]}),
        acquire_for=lambda *args, **kwargs: dict(ok=True, pin_id="pin", ref_id="ref", pin=dict(stage_root=str(stage))),
        open_pinned=lambda queue, pin, ref, name: (os.open(stage / Path(name.rsplit(":", 1)[0]).name, os.O_RDONLY),
                                                    dict(tier="fixture")),
        release=lambda *args, **kwargs: True)
    package = types.ModuleType("prismabuild")
    package.client = sdk
    monkeypatch.setitem(sys.modules, "prismabuild", package)
    return dict(window_mode=MODE, artifact=str(artifact), artifact_manifest=str(content),
                data_manifest=str(data), ts=str(runtime)), rdv


def check_staged_packet(config, rdv, gate):
    if gate == "preflight":
        return control.input_preflight(config)
    import os
    import rank_window
    adapter = rank_window.LocalArm.__new__(rank_window.LocalArm)
    adapter.config, adapter.rdv, adapter.rank = config, rdv, 0
    adapter.staged_inputs, adapter.staged_fds, adapter.staged_file_mounts = None, [], []
    try:
        adapter.pin_inputs()
        assert len(adapter.staged_file_mounts) == 128
    finally:
        for fd in adapter.staged_fds:
            os.close(fd)
        if adapter.staged_inputs is not None:
            adapter.staged_inputs.close()


@pytest.mark.parametrize("gate", ["rank", "preflight"])
@pytest.mark.parametrize("dev", ["1", "0"])
def test_d32_staged_digest_provenance_stamps_dev_but_refuses_certified(tmp_path, monkeypatch, capsys, gate, dev):
    config, rdv = staged_packet(tmp_path, monkeypatch)
    content = Path(config["artifact_manifest"])
    rows = json.loads(content.read_bytes())
    rows[0]["sha256"] = "f" * 64  # recorded provenance; current declared bytes keep their own digest
    content.write_text(json.dumps(rows))
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev)
    if dev == "0":
        with pytest.raises(Refused):
            check_staged_packet(config, rdv, gate)
    else:
        check_staged_packet(config, rdv, gate)
        assert "[DEV-MODE]" in capsys.readouterr().out


@pytest.mark.parametrize("gate", ["rank", "preflight"])
@pytest.mark.parametrize("dev", ["1", "0"])
@pytest.mark.parametrize("fault", ["path", "offset", "bytes", "population", "file-length"])
def test_d32_current_staged_ranges_and_file_bytes_always_refuse(tmp_path, monkeypatch, gate, dev, fault):
    config, rdv = staged_packet(tmp_path, monkeypatch)
    data = Path(config["data_manifest"])
    declared = json.loads(data.read_bytes())
    if fault == "path": declared["entries"][0]["path"] += ".wrong"
    elif fault == "offset": declared["entries"][0]["offset"] = 1
    elif fault == "bytes": declared["entries"][0]["bytes"] += 1
    elif fault == "population": declared["entries"].pop()
    else:
        declared["entries"][0]["bytes"] += 1
        content = Path(config["artifact_manifest"])
        rows = json.loads(content.read_bytes())
        rows[0]["bytes"] += 1
        content.write_text(json.dumps(rows))
    data.write_text(json.dumps(declared))
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev)
    with pytest.raises((Refused, ValueError), match="ranges|byte length"):
        check_staged_packet(config, rdv, gate)


@pytest.mark.parametrize("dev", ["1", "0"])
@pytest.mark.parametrize("label", ["control_root", "runtime_comparisons", "runtime_comparable"])
def test_d32_preregistration_label_drift_preserves_actual_facts(tmp_path, monkeypatch, capsys, dev, label):
    baseline, root = tmp_path / "baseline", tmp_path / "pm"
    for name in CONTROL_ARMS: store_population(baseline, name)
    store_population(root, NAMES[2])
    config = dict(control_root=str(baseline))
    record = control.preregister(root, config, matched=True)
    if label == "control_root": record[label] = str(tmp_path / "recorded-alias")
    elif label == "runtime_comparable": record[label] = not record[label]
    else: record[label][0]["source_commit"] = "recorded-older-source-label"
    (root / "control-preregistration.json").write_text(json.dumps(record))
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev)
    if dev == "0":
        with pytest.raises(Refused):
            control.require_piece_major_control(root, config)
    else:
        control.require_piece_major_control(root, config)
        assert "[DEV-MODE]" in capsys.readouterr().out


@pytest.mark.parametrize("dev", ["1", "0"])
@pytest.mark.parametrize("fault", ["timing", "protocol", "raw-seed"])
def test_d32_preregistration_actual_facts_and_raw_population_remain_refusals(tmp_path, monkeypatch, dev, fault):
    baseline, root = tmp_path / "baseline", tmp_path / "pm"
    for name in CONTROL_ARMS: store_population(baseline, name)
    store_population(root, NAMES[2])
    config = dict(control_root=str(baseline))
    record = control.preregister(root, config, matched=True)
    if fault == "timing": record["off_timing_seconds"][0][0] += 5
    elif fault == "protocol": record["runtime_comparisons"][0]["control_protocol"] = dict(PROTOCOL, server_seed=99)
    else:
        path = baseline / "arms" / NAMES[0] / "control.json"
        raw = json.loads(path.read_bytes())
        raw["requests"][0]["request_seed"] = 99
        path.write_text(json.dumps(raw))
    (root / "control-preregistration.json").write_text(json.dumps(record))
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev)
    with pytest.raises(Refused):
        control.require_piece_major_control(root, config)



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
