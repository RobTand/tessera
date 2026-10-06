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
import sys

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
    base = {"PATH": "/usr/bin:/bin", "RUNTIME_IMAGE_PY": sys.executable, "HOME": str(tmp_path), "TS": str(ROOT),
            "ARTIFACT": str(artifact), "RECEIPTS": str(tmp_path / "receipts"),
            "EAGER": "0", "COMPILATION_JSON": RELEASE_CC, "SPEC_JSON": MTP, "FABRIC": "socket"}
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
    env = {"PATH": "/usr/bin:/bin", "RUNTIME_IMAGE_PY": sys.executable, "HOME": str(tmp_path), "TS": str(ROOT), "ARTIFACT": str(artifact),
           "RECEIPTS": str(tmp_path / "receipts"), "FABRIC": "socket"}
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


REF = "localhost/prismaquant/spark-vllm-nccl230@sha256:"


def _arm_dir(root, name, *, image_digest="a", max_model_len=8448, model, script="c" * 64,
             kernel='{"enable_flashinfer_autotune":false}', extra=(), drop=()):
    """An arm directory as arm.sh writes it after review 2: the runtime gate's bare digest
    (``image_digest_resolved=sha256:<hex>``, runtime_image.py's real shape) beside the whole
    reference it resolved (``image_resolved_reference=<repo>@sha256:<hex>``)."""
    d = root / name
    d.mkdir()
    lines = [f"arm={name}", "eager=1", f"image={REF}{'0' * 64}",
             f"image_digest_resolved=sha256:{image_digest * 64}",
             f"image_resolved_reference={REF}{image_digest * 64}",
             f"serve_args=--host 0.0.0.0 --tensor-parallel-size 1 --max-model-len {max_model_len} "
             f"--max-num-seqs 8 --kernel-config '{kernel}'",
             "spec_json=", f"src_sha256={'s' * 64}", f"hooks_sha256={'h' * 64}",
             f"equal_script_sha256={script}", f"model={model}", *extra]
    lines = [l for l in lines if l.split("=", 1)[0] not in drop]
    (d / f"engine-args-{name}.txt").write_text("\n".join(lines) + "\n")
    return d


@pytest.fixture
def model(tmp_path):
    m = tmp_path / "model"
    m.mkdir()
    (m / "config.json").write_text("{}")
    return m


def _scopes(tool, root, names, legacy=None):
    return {n: tool.arm_scope(root / n, n, legacy) for n in names}


def test_the_image_is_the_whole_resolved_reference_never_the_bare_digest(tmp_path, model):
    tool = _receipt_tool()
    _arm_dir(tmp_path, "e1", model=model)
    assert tool.arm_scope(tmp_path / "e1", "e1")["image"] == f"{REF}{'a' * 64}"


def test_a_bare_digest_alone_names_no_image(tmp_path, model):
    """The post-measurement arm.sh recorded only the bare digest; with no reference beside it
    and no digest-pinned declaration, the arm has no image scope."""
    tool = _receipt_tool()
    _arm_dir(tmp_path, "e1", model=model, drop=("image_resolved_reference", "image"),
             extra=("image=localhost/prismaquant/spark-vllm:latest",))
    with pytest.raises(SystemExit, match="did not record.*image"):
        tool.arm_scope(tmp_path / "e1", "e1")


def test_arms_that_measured_different_serves_make_no_receipt(tmp_path, model):
    """Review of #930: a scope field is read from what each arm recorded, and arms that
    disagree on any of them (here the image and max_model_len) are not one measurement."""
    tool = _receipt_tool()
    _arm_dir(tmp_path, "e1", model=model)
    _arm_dir(tmp_path, "g1", image_digest="b", max_model_len=4096, model=model)
    scopes = _scopes(tool, tmp_path, ("e1", "g1"))
    assert scopes["e1"]["tensor_parallel_size"] == 1 and scopes["g1"]["max_model_len"] == 4096
    with pytest.raises(SystemExit, match=r"differ in \['image', 'max_model_len'") as refused:
        tool.one_measurement(scopes)
    assert "serve_flags" in str(refused.value)   # the argv carries max_model_len too
    _arm_dir(tmp_path, "g2", model=model)
    assert tool.one_measurement(_scopes(tool, tmp_path, ("e1", "g2")))["image"].endswith("a" * 64)


@pytest.mark.parametrize("change, field", [
    (dict(script="d" * 64), "equal_script_sha256"),
    (dict(kernel='{"enable_flashinfer_autotune":true}'), "kernel_config"),

    (dict(extra=("tessera_env=TESSERA_FUSED_E4M3_MMA=e4m3",)), "tessera_env"),
])
def test_arms_that_ran_another_script_or_setup_make_no_receipt(tmp_path, model, change, field):
    tool = _receipt_tool()
    _arm_dir(tmp_path, "e1", model=model)
    _arm_dir(tmp_path, "g1", model=model, **change)
    with pytest.raises(SystemExit, match=field):
        tool.one_measurement(_scopes(tool, tmp_path, ("e1", "g1")))


def test_the_operator_priority_a_graph_arm_pins_is_not_a_setup_difference(tmp_path, model):
    """rG4 pins eager's IR priority in its kernel config: the execution mode under test."""
    tool = _receipt_tool()
    _arm_dir(tmp_path, "e1", model=model)
    _arm_dir(tmp_path, "g4", model=model, kernel='{"enable_flashinfer_autotune":false,'
             '"ir_op_priority":{"rms_norm":["vllm_c","native"]}}')
    tool.one_measurement(_scopes(tool, tmp_path, ("e1", "g4")))


def test_a_legacy_arm_takes_its_script_from_its_own_snapshot_or_has_no_scope(tmp_path, model):
    tool = _receipt_tool()
    _arm_dir(tmp_path, "e1", model=model, drop=("equal_script_sha256",))
    with pytest.raises(SystemExit, match="equal_script_sha256"):
        tool.arm_scope(tmp_path / "e1", "e1")
    legacy = {"e1": {"sha256": "c" * 64, "source": "snapshot"}}
    assert tool.arm_scope(tmp_path / "e1", "e1", legacy)["equal_script_sha256"] == "c" * 64


@pytest.mark.parametrize("files, tp, ok", [
    (["aGR.rank0.dispatch.1.json", "aGR.rank1.dispatch.2.json"], 2, True),
    (["aGR.rank1.dispatch.2.json"], 2, False),                 # the head's copy never arrived
    (["aGR.dispatch.1.json"], 1, True),
    (["aGR.rank0.dispatch.1.json"], 1, False),
])
def test_a_tensor_parallel_arm_needs_every_ranks_dispatch_log(tmp_path, files, tp, ok):
    tool = _receipt_tool()
    for name in files:
        (tmp_path / name).write_text("{}")
    if ok:
        tool.require_every_rank(tmp_path, "aGR", tp)
    else:
        with pytest.raises(SystemExit, match="cannot show it replayed"):
            tool.require_every_rank(tmp_path, "aGR", tp)


# ------------------------------------------------------------------ v2: the fabric


def _tp2_arm_dir(root, name, model, *, requested="roce", observed="IB", drop=()):
    banner = f"Using network {observed}"
    return _arm_dir(root, name, model=model, drop=drop, extra=(
        f"fabric_requested={requested}", f"fabric_observed=rank0:{banner};rank1:{banner}")) \
        if "serve_args" not in drop else None


def _make_tp2(d):
    f = d / f"engine-args-{d.name}.txt"
    f.write_text(f.read_text().replace("--tensor-parallel-size 1", "--tensor-parallel-size 2"))


def test_a_tp2_arms_fabric_is_what_both_ranks_banners_say(tmp_path, model):
    tool = _receipt_tool()
    _make_tp2(_tp2_arm_dir(tmp_path, "e1", model, requested="roce", observed="IB"))
    _make_tp2(_tp2_arm_dir(tmp_path, "s1", model, requested="socket", observed="Socket"))
    assert tool.arm_scope(tmp_path / "e1", "e1")["fabric"] == "roce"
    assert tool.arm_scope(tmp_path / "s1", "s1")["fabric"] == "socket"
    with pytest.raises(SystemExit, match="fabric"):
        tool.one_measurement(_scopes(tool, tmp_path, ("e1", "s1")))


def test_a_one_rank_arm_has_no_fabric(tmp_path, model):
    tool = _receipt_tool()
    _arm_dir(tmp_path, "e1", model=model)
    assert tool.arm_scope(tmp_path / "e1", "e1")["fabric"] == "none"


@pytest.mark.parametrize("requested, observed", [("roce", "Socket"), ("socket", "IB")])
def test_a_tp2_arm_whose_banners_contradict_its_request_has_no_scope(tmp_path, model, requested,
                                                                     observed):
    tool = _receipt_tool()
    _make_tp2(_tp2_arm_dir(tmp_path, "e1", model, requested=requested, observed=observed))
    with pytest.raises(SystemExit, match="fabric"):
        tool.arm_scope(tmp_path / "e1", "e1")


def test_a_tp2_arm_without_banners_has_no_scope(tmp_path, model):
    tool = _receipt_tool()
    d = _arm_dir(tmp_path, "e1", model=model, extra=("fabric_requested=roce",))
    _make_tp2(d)
    with pytest.raises(SystemExit, match="fabric"):
        tool.arm_scope(d, "e1")


def test_the_window_plan_runs_on_sockets():
    """The window-3 control records FABRIC=socket: the Spark pair serves on sockets (RoCE
    ibv_reg_mr fails ENOMEM there), and a card attests the serve its slots measured."""
    plan = (ROOT / "experiments" / "graph_attest_702" / "plan-artifact.txt").read_text()
    arms = [l for l in plan.splitlines() if l.strip() and not l.startswith("#")]
    assert arms and all("FABRIC=socket" in l.split() for l in arms)


def test_an_arm_names_its_fabric_or_does_not_run(tmp_path):
    done = _dry_run(tmp_path, FABRIC="")
    assert done.returncode != 0 and "FABRIC" in (done.stdout + done.stderr)
def test_dry_recipe_never_launches_a_remote_rank_by_ssh(tmp_path):
    done = _dry_run(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "ssh" not in done.stdout.lower()
    assert "local PB rank action" in done.stdout


def test_dry_recipe_declares_one_window_including_owned_cleanup(tmp_path):
    done = _dry_run(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "5400-second whole-window" in done.stdout
    assert "cleanup reserve" in done.stdout


@pytest.mark.parametrize("floor,allowed", [("2", True), ("16", False)])
def test_optional_floor_input_uses_the_current_shared_policy(tmp_path, floor, allowed):
    done = _dry_run(tmp_path, FLOOR_GIB=floor)
    if allowed:
        assert done.returncode == 0, done.stdout + done.stderr
    else:
        assert done.returncode == 3 and "FLOOR_GIB=2" in done.stdout
