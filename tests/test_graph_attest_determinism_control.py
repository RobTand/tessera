"""Reduced seeded investigation controls; no GPU or served parity qualification."""
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1] / "experiments/graph_attest_702"
sys.path.insert(0, str(HERE))
import tp2_recipe as recipe
from managed_window import Refused

MODE = "investigate-eager-determinism-2048"
NAMES = ("control_off_first", "control_off_restart", "control_decode_once", "control_kda_split", "control_piece_major")
FLAGS = ("TESSERA_E4M3_DECODE_ONCE", "TESSERA_GLM53_KDA_CONV_SPLIT", "TESSERA_ROUTED_PIECE_MAJOR")


def control_plan(tmp_path):
    path = tmp_path / "control.txt"
    lines = []
    for index, name in enumerate(NAMES):
        values = ["0", "off", "0"]
        if index >= 2:
            values[index - 2] = ("1", "on", "1")[index - 2]
        lines.append(f'{name} FABRIC=socket EAGER=1 MAX_BATCHED=4096 '
                     f'SPEC_JSON={json.dumps(recipe.MTP, separators=(",", ":"))} ' +
                     " ".join(f"{key}={value}" for key, value in zip(FLAGS, values)))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_dedicated_control_admits_two_off_restarts_then_only_single_levers(tmp_path):
    arms = recipe.plan(control_plan(tmp_path), mode=MODE)
    assert [arm["arm"] for arm in arms] == list(NAMES)
    assert all(arm["max_batched"] == 4096 and arm["eager"] == "1" for arm in arms)
    assert [arm["lever_env"] for arm in arms[:2]] == [dict(zip(FLAGS, ("0", "off", "0")))] * 2
    assert [sum(arm["lever_env"][key] in ("1", "on") for key in FLAGS) for arm in arms[2:]] == [1, 1, 1]


@pytest.mark.parametrize("old,new", [
    ("control_off_restart", "control_off_first"),
    ("MAX_BATCHED=4096", "MAX_BATCHED=2048"),
    ("FABRIC=socket", "FABRIC=roce"),
    ("TESSERA_E4M3_DECODE_ONCE=0", "TESSERA_E4M3_DECODE_ONCE=1"),
    ("TESSERA_ROUTED_PIECE_MAJOR=1", "TESSERA_ROUTED_PIECE_MAJOR=0"),
])
def test_control_refuses_reordered_population_or_extra_levers(tmp_path, old, new):
    path = control_plan(tmp_path)
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(Refused):
        recipe.plan(path, mode=MODE)


def test_existing_ship_pair_still_refuses_off_off(tmp_path):
    path = control_plan(tmp_path)
    rows = path.read_text().splitlines()[:2]
    path.write_text(rows[0].replace(NAMES[0], "eager4096_off") + "\n" +
                    rows[1].replace(NAMES[1], "eager4096_on") + "\n")
    with pytest.raises(Refused, match="at least one on"):
        recipe.plan(path, mode=recipe.EAGER_LEVER_MODE)


def population_fixture():
    import hashlib
    import eager_determinism as control
    from eager_benchmark import MODEL
    from seeded_control_client import SCHEMA
    prompts = dict(warmup=1, prompts={"2048": {"1": [[[trial] * 2048] for trial in range(11)]}})
    digest = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    records = []
    for trial, prompt in enumerate(control.prompt_population(prompts)):
        records.append(dict(trial=trial, slot=0, warmup=trial == 0, request_id=f"L2048-c1/trial{trial}/slot0",
                            request_seed=trial, request_payload=dict(
                                model=MODEL, prompt=prompt, max_tokens=128, temperature=0.0, seed=trial,
                                ignore_eos=True, stream=True, stream_options=control.PROTOCOL["stream_options"]),
                            status=200, error=None, prompt_tokens_sent=2048,
                            prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                            completion_tokens=128, usage=dict(prompt_tokens=2048, completion_tokens=128, total_tokens=2176),
                            generation=dict(done=True, choices=[dict(index=0, text="full completion", finish_reason="length")])))
    return prompts, digest, dict(schema=SCHEMA, model=MODEL, protocol=control.PROTOCOL,
                                prompts_sha256=digest, requests=records)


def test_control_full_text_comparison_includes_warmup_and_suffix():
    import copy
    import eager_determinism as control
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
    import eager_determinism as control
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
    import eager_determinism as control
    import hashlib
    prompts, _, result = population_fixture()
    prompt_path = root.parent / "prompts.json"
    prompt_path.write_text(json.dumps(prompts))
    result["prompts_sha256"] = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    if change: result["requests"][10]["generation"]["choices"][0]["text"] += " different"
    directory = root / "arms" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "invocation.json").write_text(json.dumps(dict(source_bindings=dict(prompts=str(prompt_path)))))
    (directory / "control.json").write_text(json.dumps(result))
    (directory / "output-hashes.json").write_text(json.dumps(
        control.output_population(result, prompts, result["prompts_sha256"])))


def test_bisection_authorization_uses_raw_full_off_outputs_not_success_label(tmp_path):
    import eager_determinism as control
    root = tmp_path / "run"
    store_population(root, NAMES[0])
    store_population(root, NAMES[1], change=True)
    (root / "arms" / NAMES[1] / "output-comparison.json").write_text("{\"passed\":true}")
    with pytest.raises(Refused, match="SERVING NONDETERMINISM"):
        control.require_deterministic_off(root)
    store_population(root, NAMES[1])
    assert control.require_deterministic_off(root)["requests_per_arm"] == 11
    path = root / "arms" / NAMES[1] / "output-hashes.json"
    data = json.loads(path.read_bytes())
    data["outputs"]["L2048-c1/trial10/slot0"]["text"] = "tampered"
    path.write_text(json.dumps(data))
    with pytest.raises(Refused, match="own raw records"):
        control.require_deterministic_off(root)


def test_control_manifest_and_server_seed_are_normal_priority_and_unchanged_caps(tmp_path):
    import window_driver as driver
    (tmp_path / "config.json").write_text("{}")
    env = dict(TS=str(HERE.parents[1]), ARTIFACT=str(tmp_path), RECEIPTS=str(tmp_path / "arms"),
               FABRIC="socket", WINDOW_MODE=MODE, SOURCE_COMMIT="a" * 40, SOURCE_SHA256="b" * 64,
               PRODUCER_COMMIT="c" * 40, PRODUCER_SHA256="d" * 64)
    config = recipe.inputs(env, live=False)
    (tmp_path / "inputs.json").write_text(json.dumps(config))
    rows = driver.rows(tmp_path, config, env)
    assert [row["demand"] for row in rows] == [dict(cpu=8, mem_gb=104, gpu=1), dict(cpu=6, mem_gb=104, gpu=1)]
    assert all(row["priority"] == 0 and row["gpu_memory_gb"] == 102 and row["exclusive"] and
               row["measurement"] and row["host_class"] == "gb10" and row["timeout_s"] == 5400 for row in rows)
    for arm in recipe.plan(control_plan(tmp_path), mode=MODE):
        commands = [recipe.serve(config, arm, rank) for rank in (0, 1)]
        for command in commands:
            assert command[command.index("--seed") + 1] == "0"
            assert command[command.index("--max-num-batched-tokens") + 1] == "4096"
            assert "--profiler-config" not in command
            assert "--enforce-eager" in command
        container = recipe.container(config, arm, dict(rank=0, run_id="test", nonce="a" * 32),
                                     tmp_path / "out", tmp_path / "ext", tmp_path / "cid", {})
        assert all(f"{key}={value}" in container for key, value in arm["lever_env"].items())


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
        assert response["generation"]["done"] is True and response["error"] is None
        assert "".join(choice["text"] for choice in response["generation"]["choices"]) == "first suffix"
        assert response["completion_tokens"] == 128
    finally:
        server.server_close()
        worker.join(timeout=5)


@pytest.mark.parametrize("changed", [False, True], ids=["matching-off", "mismatched-off"])
def test_real_rank_lifecycle_restarts_off_and_gates_both_lever_starts(tmp_path, changed):
    import os
    import signal
    import subprocess
    import time
    import managed_window as window
    from test_graph_attest_window_scenarios import alive, make_identity, write_claim
    root, queue = tmp_path / "run", tmp_path / "queue"
    root.mkdir()
    arms = recipe.plan(control_plan(tmp_path), mode=MODE)
    start = time.time()
    identities = [make_identity(rank, start, start + 20) for rank in (0, 1)]


    children = []
    try:
        for identity in identities:
            write_claim(queue, identity)
            window.atomic_json(tmp_path / f"identity{identity['rank']}.json", identity)
            log = (tmp_path / f"controller{identity['rank']}.log").open("w")
            process = subprocess.Popen([sys.executable, str(Path(__file__)), "--control-rank",
                                        str(tmp_path), str(identity["rank"]), str(int(changed))],
                                       stdout=log, stderr=subprocess.STDOUT)
            children.append((process, log))
        for process, log in children:
            process.wait(timeout=25)
            log.close()
        outcomes = [window.read_json(root / f"outcome-rank{rank}.json") for rank in (0, 1)]
        pids = [int(path.read_text()) for path in root.glob("*.pid")]
        assert not [pid for pid in pids if alive(pid)]
        assert all(outcome["simulation"] and not outcome["ownership_released"] for outcome in outcomes)
        if changed:
            assert all(outcome["returncode"] == 1 and outcome["completed_arms"] == list(NAMES[:2]) for outcome in outcomes)
            assert any("SERVING NONDETERMINISM" in outcome["error"] for outcome in outcomes)
            assert not list(root.glob("control_decode_once*.pid"))
        else:
            assert all(outcome["returncode"] == 0 and outcome["completed_arms"] == list(NAMES) for outcome in outcomes)
        for rank in (0, 1):
            assert (root / f"{NAMES[0]}.rank{rank}.pid").read_text() != (root / f"{NAMES[1]}.rank{rank}.pid").read_text()
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


if __name__ == "__main__":
    import managed_window as window
    import rank_window
    from test_graph_attest_window_scenarios import CpuArm
    parent, rank, changed = Path(sys.argv[2]), int(sys.argv[3]), bool(int(sys.argv[4]))
    root, queue = parent / "run", parent / "queue"
    owned = window.read_json(parent / f"identity{rank}.json")
    envelope = window.Envelope(owned["window_end_unix"], cleanup_seconds=1)

    class ControlArm(CpuArm):
        def probes(self, arm, peer):
            store_population(root, arm["arm"], change=changed and arm["arm"] == NAMES[1])
            return dict(correctness_only=True)

    adapter = ControlArm(root, owned, envelope, "success")
    arms = [dict(arm=name) for name in NAMES]
    raise SystemExit(rank_window.run_rank(dict(window_mode=MODE, fabric="socket"), owned, queue, root, arms,
                                         adapter, envelope, poll_seconds=.01))
