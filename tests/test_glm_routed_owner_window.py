"""The two-box window wrapper's own control flow, on the CPU.

No daemon, no GPU, no container: ``docker`` is a fake on ``PATH`` that answers
the gate's ``image inspect``, reports no running container to the serve lock,
and RECORDS the ``run`` argv.  What is under test is the wrapper -- the
world/rank/rendezvous guards, the request's own declaration against the process
about to run it, and that BOTH ranks are started in every mode including
``--prepare``.  The harness's prepare joins the declared world and its warmup
calls the runner's own late all-reduce, so a prepare started on one rank alone
is not a run of this operator.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import box_artifacts

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "src/tessera/serving/runtime_contract.json").read_text())
PINNED = CONTRACT["versions"]["default_serve_image"]


def _fake_docker(tmp_path, record):
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    (fake / "docker").write_text(
        "#!/usr/bin/env bash\n"
        'case "$1" in\n'
        "  image) if [ \"$3\" = \"$FAKE_PINNED\" ]; then\n"
        "           printf 'sha256:%s\\t[\"%s\"]\\n' \"$FAKE_LOCAL_ID\" \"$FAKE_PINNED\";\n"
        "         else\n"
        "           printf 'sha256:%s\\t[\"vllm/vllm-openai@sha256:%064d\"]\\n' \"$FAKE_LOCAL_ID\" 0;\n"
        "         fi; exit 0 ;;\n"
        "  ps) exit 0 ;;\n"
        '  run) printf "%s\\n" "$@" >> "$FAKE_RUN_RECORD"; exit 0 ;;\n'
        "esac\n"
        "exit 0\n")
    (fake / "docker").chmod(0o755)
    return fake


def _tree(tmp_path):
    """A throwaway checkout: the wrapper refuses inputs outside its own root."""
    root = tmp_path / "tree"
    (root / "experiments").mkdir(parents=True)
    for name in ("glm_routed_owner_window.sh", "runtime_image.sh", "serve_lock.sh"):
        shutil.copy(ROOT / "experiments" / name, root / "experiments" / name)
    (root / "src").symlink_to(ROOT / "src")
    for name in ("requests", "panels", "receipts"):
        (root / name).mkdir()
    return root


def _run(tmp_path, root, *, request=None, panel=None, out=None, mode="prepare",
         world=2, rank=0, rendezvous="tcp://box-a:29500", image=PINNED, extra_env=None):
    record = tmp_path / f"docker-run-{len(list(tmp_path.glob('docker-run-*.txt')))}.txt"
    fake = _fake_docker(tmp_path, record)
    env = dict(os.environ,
               PATH=f"{fake}:{os.environ['PATH']}",
               FAKE_PINNED=PINNED, FAKE_LOCAL_ID="sha256:" + "aa" * 32,
               FAKE_RUN_RECORD=str(record),
               SERVE_LOCK=str(tmp_path / "serve.lock"),
               TMPDIR=box_artifacts.scratch_tmpdir())
    env.update(RANK=str(rank), WORLD=str(world), RATE="a16", MODE=mode,
               REQUEST=str(request or root / "requests" / "a16-tp2.json"),
               OUT=str(out or root / "receipts" / ("a16-tp2-rank%d.json" % rank)))
    if panel is not None:
        env["PANEL"] = str(panel)
    if rendezvous is not None:
        env["RENDEZVOUS"] = rendezvous
    env.update(extra_env or {})
    proc = subprocess.run(["bash", str(root / "experiments" / "glm_routed_owner_window.sh"), image],
                          env=env, capture_output=True, text=True)
    record = Path(env["FAKE_RUN_RECORD"])
    return proc, (record.read_text().splitlines() if record.exists() else [])


def _write_request(root, *, world=2, rank=0, rendezvous="tcp://box-a:29500", name="a16-tp2.json"):
    path = root / "requests" / name
    path.write_text(json.dumps({"schema": "tessera.native_moe_request.v1", "unit": "u",
                                "distributed": {"world_size": world, "rank": rank,
                                                "init_method": rendezvous}}))
    return path


def _write_panel(root, name="a16-tp2.json"):
    path = root / "panels" / name
    path.write_text(json.dumps({"schema": "tessera.native_moe_panel.v1"}))
    return path


@pytest.mark.parametrize("kwargs, message", [
    ({"world": 1, "rank": 1, "rendezvous": None}, "world 1 is rank 0"),
    ({"world": 1, "rank": 0, "rendezvous": "tcp://box-a:29500"}, "world 1 takes no rendezvous"),
    ({"world": 2, "rank": 0, "rendezvous": None}, "needs an explicit tcp:// rendezvous"),
    ({"world": 2, "rank": 7, "rendezvous": "tcp://box-a:29500"}, "world 2 is rank 0 or 1"),
])
def test_the_world_and_rank_guards_refuse_before_any_container(tmp_path, kwargs, message):
    root = _tree(tmp_path)
    _write_request(root, name="a16-tp2.json")
    proc, runs = _run(tmp_path, root, **kwargs)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert message in proc.stderr, proc.stderr
    assert runs == [], "a refused invocation started a container"


def test_a_mode_that_prices_must_name_its_panel(tmp_path):
    root = _tree(tmp_path)
    _write_request(root)
    proc, runs = _run(tmp_path, root, mode="panel")
    assert proc.returncode == 2 and "independent frozen panel" in proc.stderr
    assert runs == []


def test_prepare_takes_no_panel(tmp_path):
    root = _tree(tmp_path)
    _write_request(root)
    proc, runs = _run(tmp_path, root, mode="prepare", panel=_write_panel(root))
    assert proc.returncode == 2 and "prepare takes no panel" in proc.stderr
    assert runs == []


def test_an_input_outside_the_tree_is_refused(tmp_path):
    root = _tree(tmp_path)
    outside = tmp_path / "elsewhere-request.json"
    outside.write_text("{}")
    proc, runs = _run(tmp_path, root, request=outside)
    assert proc.returncode == 2 and "keep REQUEST inside" in proc.stderr
    assert runs == []


def test_a_non_resident_mode_is_refused(tmp_path):
    root = _tree(tmp_path)
    _write_request(root)
    proc, runs = _run(tmp_path, root, extra_env={"TESSERA_SERVE_MODE": "streamed"})
    assert proc.returncode == 2 and "measured resident" in proc.stderr
    assert runs == []


def test_the_request_world_must_be_the_process_that_runs_it(tmp_path):
    root = _tree(tmp_path)
    _write_request(root, world=2, rank=0)
    proc, runs = _run(tmp_path, root, rank=1)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "the request declares world/rank 2/0" in proc.stderr
    assert runs == []


def test_an_unpinned_image_never_reaches_the_container(tmp_path):
    root = _tree(tmp_path)
    _write_request(root)
    repository = PINNED.partition("@")[0]
    assert ":" not in repository, "the pin is a digest reference on a bare repository"
    proc, runs = _run(tmp_path, root, image=f"{repository}:latest")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert runs == []


@pytest.mark.parametrize("mode", ["prepare", "panel", "profile"])
@pytest.mark.parametrize("rank", [0, 1])
def test_both_ranks_run_every_mode_through_the_gated_wrapper(tmp_path, mode, rank):
    root = _tree(tmp_path)
    request = _write_request(root, rank=rank)
    panel = _write_panel(root) if mode != "prepare" else None
    proc, runs = _run(tmp_path, root, mode=mode, rank=rank, panel=panel)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert runs, "the wrapper did not start the container"
    argv = " ".join(runs)
    assert "--request /workspace/tessera/requests/a16-tp2.json" in argv
    assert "--out /receipts/a16-tp2-rank%d.json" % rank in argv
    if mode == "prepare":
        # The harness CLI requires exactly one of --prepare/--panel, so prepare
        # has to actually pass it rather than rely on the panel's absence.
        assert "--prepare" in argv
        assert "--panel" not in argv and "--profile" not in argv
    if mode == "panel":
        assert "--panel /workspace/tessera/panels/a16-tp2.json" in argv
    if mode == "profile":
        assert "--profile" in argv
    # The aggregate cap is the container's own limit, and the tree is read-only.
    assert "--memory=32g" in argv and "--memory-swap=32g" in argv
    assert any(part.endswith(":/workspace/tessera:ro") for part in runs)
    # The request's wire and tensor paths live on the shared mount.
    assert "/mnt/shared:/mnt/shared:ro" in runs


def test_a_fresh_output_directory_is_created_before_it_is_resolved(tmp_path):
    root = _tree(tmp_path)
    _write_request(root)
    out = root / "receipts" / "a16" / "tp2" / "rank0.json"
    proc, runs = _run(tmp_path, root, out=out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "--out /receipts/rank0.json" in " ".join(runs)


def test_a_resource_library_is_forwarded_and_must_be_on_the_shared_mount(tmp_path):
    root = _tree(tmp_path)
    _write_request(root)
    library = Path("/mnt/shared/tessera-measurements/libcollector.so")
    proc, runs = _run(tmp_path, root, extra_env={"RESOURCE_LIBRARY": str(library)})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"--resource-library {library}" in " ".join(runs)
    proc, runs = _run(tmp_path, root, extra_env={"RESOURCE_LIBRARY": str(tmp_path / "lib.so")})
    assert proc.returncode == 2 and "must be on /mnt/shared" in proc.stderr
    assert runs == []


def test_world_one_requires_the_four_field_block_the_harness_accepts(tmp_path):
    root = _tree(tmp_path)
    path = root / "requests" / "a16-tp1.json"
    path.write_text(json.dumps({"schema": "tessera.native_moe_request.v1",
                                "distributed": {"world_size": 1, "rank": 0,
                                                "init_method": None}}))
    proc, runs = _run(tmp_path, root, request=path, world=1, rank=0, rendezvous=None)
    assert proc.returncode == 2 and "four-field single-rank block" in proc.stderr
    assert runs == []
