"""CPU checks for the artifact-scope (TP 2) graph-equals-eager tooling (tessera#702).

``experiments/graph_attest_702/arm_tp2.sh --dry-run`` starts nothing, so it is
driven here: the two ranks must serve the same engine (one argv, apart from the
rank flags), and an arm that cannot be one measurement is refused before any
box is touched.  ``receipt.py``'s dispatch reader keys each rank's graph
managers apart, so one rank that never replayed a captured class cannot hide
behind the other's counts.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import shlex
import subprocess

import pytest

from tessera import graph_receipt

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARM_TP2 = ROOT / "experiments" / "graph_attest_702" / "arm_tp2.sh"
RELEASE_CC = '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY"}'
MTP = '{"method":"mtp","num_speculative_tokens":1,"draft_tensor_parallel_size":2,"moe_backend":"triton"}'
RANK_FLAGS = {"1": ["--node-rank", "1", "--headless"],
              "0": ["--node-rank", "0", "--host", "0.0.0.0", "--port", "8142"]}


def _dry_run(tmp_path, arm="aGR", **env):
    artifact = tmp_path / "artifact"
    artifact.mkdir(exist_ok=True)
    (artifact / "config.json").write_text("{}")
    base = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "TS": str(ROOT),
            "ARTIFACT": str(artifact), "RECEIPTS": str(tmp_path / "receipts"),
            "EAGER": "0", "COMPILATION_JSON": RELEASE_CC, "SPEC_JSON": MTP}
    base.update(env)
    return subprocess.run(["bash", str(ARM_TP2), "--dry-run", arm], env=base,
                          capture_output=True, text=True, timeout=120)


def _serve_argv(stdout, rank):
    """The ``vllm serve`` argv one rank's container execs, as the dry run prints it."""
    line = next(l for l in stdout.splitlines() if l.lstrip().startswith(f"serve rank{rank}:"))
    return shlex.split(line.split(":", 1)[1])


def test_both_ranks_serve_one_engine(tmp_path):
    done = _dry_run(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    argv = {}
    for rank, flags in RANK_FLAGS.items():
        words = _serve_argv(done.stdout, rank)
        start = words.index(flags[0])
        assert words[start:start + len(flags)] == flags
        argv[rank] = words[:start] + words[start + len(flags):]
    assert argv["0"] == argv["1"]
    engine = argv["0"]
    for flag, value in (("--tensor-parallel-size", "2"), ("--nnodes", "2"), ("--max-num-seqs", "4"),
                        ("--max-model-len", "8448"), ("--compilation-config", RELEASE_CC),
                        ("--speculative-config", MTP), ("--kv-cache-dtype", "fp8_ds_mla")):
        assert engine[engine.index(flag) + 1] == value, flag
    assert "--enforce-eager" not in engine


def test_an_eager_arm_serves_the_same_engine_but_eager(tmp_path):
    done = _dry_run(tmp_path, arm="aE1", EAGER="1", COMPILATION_JSON="")
    assert done.returncode == 0, done.stdout + done.stderr
    engine = _serve_argv(done.stdout, "0")
    assert "--enforce-eager" in engine and "--compilation-config" not in engine


@pytest.mark.parametrize("env, needle", [
    ({"EAGER": "1"}, "EAGER=1 takes no COMPILATION_JSON"),
    ({"FABRIC": "infiniband"}, "FABRIC must be socket or roce"),
])
def test_an_arm_that_is_not_one_measurement_is_refused(tmp_path, env, needle):
    done = _dry_run(tmp_path, **env)
    assert done.returncode != 0 and needle in done.stdout


def test_receipts_are_never_merged_into_an_existing_arm(tmp_path):
    (tmp_path / "receipts" / "aGR").mkdir(parents=True)
    done = _dry_run(tmp_path)
    assert done.returncode == 3 and "never merged" in done.stdout


def _receipt_tool():
    spec = importlib.util.spec_from_file_location(
        "ga702_receipt", ROOT / "experiments" / "graph_attest_702" / "receipt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dispatch(path, *, replays_long):
    path.write_text(json.dumps({
        "captured": {"ModelCudaGraphManager@1": [2, 4, 6, 8]},
        "counts": {f"ModelCudaGraphManager|FULL|tokens={n}|reqs={n // 2}": 3 for n in (2, 4, 6, 8)},
        "tessera_classes": {"captured": {"ModelCudaGraphManager": {"2048": 4, "8448": 4}},
                            "replays": {"ModelCudaGraphManager|2048": 40,
                                        "ModelCudaGraphManager|8448": replays_long}}}))


def test_each_rank_must_replay_what_it_captured(tmp_path):
    tool = _receipt_tool()
    _dispatch(tmp_path / "aGR.rank0.dispatch.11.json", replays_long=5)
    _dispatch(tmp_path / "aGR.rank1.dispatch.12.json", replays_long=0)
    _dispatch(tmp_path / "aGR10.rank0.dispatch.13.json", replays_long=0)   # another arm's file
    graph = tool.graph_record(tmp_path, "aGR")
    assert sorted(graph["managers"]) == ["rank0:ModelCudaGraphManager", "rank1:ModelCudaGraphManager"]
    assert graph["classes"]["replays"]["rank1:ModelCudaGraphManager|8448"] == 0
    assert graph_receipt.replayed_everything(graph) is False
    _dispatch(tmp_path / "aGR.rank1.dispatch.12.json", replays_long=5)
    assert graph_receipt.replayed_everything(tool.graph_record(tmp_path, "aGR")) is True


def test_a_one_box_arm_reads_as_before(tmp_path):
    tool = _receipt_tool()
    _dispatch(tmp_path / "rG3.dispatch.7.json", replays_long=2)
    graph = tool.graph_record(tmp_path, "rG3")
    assert list(graph["managers"]) == ["ModelCudaGraphManager"]
    assert graph_receipt.replayed_everything(graph) is True


def test_the_committed_artifact_plan_dry_runs_end_to_end(tmp_path):
    """The plan a window will run: three arms, one engine, eager twice around the graph arm."""
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "TS": str(ROOT), "ARTIFACT": str(artifact),
           "RECEIPTS": str(tmp_path / "receipts")}
    here = ROOT / "experiments" / "graph_attest_702"
    done = subprocess.run(["bash", str(here / "drive_tp2.sh"), str(here / "plan-artifact.txt"), "--dry-run"],
                          env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    engines = [_serve_argv(block, "0") for block in done.stdout.split("== ")[1:] if "serve rank0" in block]
    assert len(engines) == 3
    strip = lambda argv: [w for w in argv if w not in ("--enforce-eager",)]
    eager, graph = engines[0], engines[1]
    assert engines[0] == engines[2]
    cc = graph.index("--compilation-config")
    assert strip(eager) == graph[:cc] + graph[cc + 2:]
