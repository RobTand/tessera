"""Portable protocol and population refusals for the A8S MNBT matrix."""
import copy
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1] / "experiments/graph_attest_702"
sys.path.insert(0, str(HERE))
import tp2_recipe as recipe
from managed_window import Refused
import eager_benchmark as benchmark


def test_matrix_refuses_mtp_on():
    name, env = recipe.parse_plan(HERE / "plan-graph-mnbt-matrix.txt")[0]
    env["SPEC_JSON"] = json.dumps(recipe.MTP, separators=(",", ":"))
    with pytest.raises(Refused, match="requires MTP off"):
        recipe.pair_arm(name, env, benchmark.MATRIX_MODE, exact_keys=True)


@pytest.mark.parametrize("flag,value", [
    ("TESSERA_E4M3_DECODE_ONCE", "0"),
    ("TESSERA_GLM53_KDA_CONV_SPLIT", "off"),
    ("TESSERA_ROUTED_PIECE_MAJOR", "0"),
])
def test_matrix_refuses_each_disabled_lever(flag, value):
    name, env = recipe.parse_plan(HERE / "plan-graph-mnbt-matrix.txt")[0]
    env[flag] = value
    with pytest.raises(Refused, match="requires all three levers explicitly ON"):
        recipe.pair_arm(name, env, benchmark.MATRIX_MODE, exact_keys=True)


def test_matrix_timing_requires_all_nine_cells():
    good = {"cells": {}}
    for length in (512, 2048, 8192):
        for conc in (1, 4, 8):
            good["cells"][f"host-L{length}-c{conc}"] = {
                "skipped": False, "complete": True,
                "trials": [{"trial": t, "requests": [
                    {"error": None, "usage": {"prompt_tokens": length, "completion_tokens": 128},
                     "generation": {"done": True}, "completion_tokens": 128}
                    for _ in range(conc)]} for t in range(1, 11)]}
    benchmark.require_matrix_timing(good)
    bad = copy.deepcopy(good)
    del bad["cells"]["host-L8192-c8"]
    with pytest.raises(Refused):
        benchmark.require_matrix_timing(bad)


SELECTED = [[4096, 512, 8], [4096, 2048, 1], [4096, 2048, 4],
            [4096, 2048, 8], [4096, 8192, 1], [4096, 8192, 4], [4096, 8192, 8]]


def test_selected_plan_keeps_only_required_arm():
    arms = recipe.plan(HERE / "plan-graph-mnbt-matrix.txt", mode=benchmark.MATRIX_MODE,
                       matrix_cells=SELECTED)
    assert [arm["arm"] for arm in arms] == ["graph_mnbt4096_on"]
    assert arms[0]["max_batched"] == 4096


@pytest.mark.parametrize("cells", [[], [[4096, 512, 8], [4096, 512, 8]],
                                   [[2048, 512, 2]], [[8192, 512, 1]],
                                   [[4096, 1024, 1]], [[4096, 512]],
                                   [[4096, "512", 1]], {"cells": SELECTED}])
def test_selected_matrix_refuses_invalid_rows(cells):
    with pytest.raises(Refused, match="matrix selection"):
        benchmark.parse_matrix_cells(json.dumps(cells))


def test_selected_matrix_has_no_fixed_missing_roster():
    cells = [[4096, 8192, 4], [2048, 512, 1]]
    assert benchmark.parse_matrix_cells(json.dumps(cells)) == cells
    arms = recipe.plan(HERE / "plan-graph-mnbt-matrix.txt", mode=benchmark.MATRIX_MODE,
                       matrix_cells=cells)
    assert [arm["max_batched"] for arm in arms] == [2048, 4096]


def test_selected_entry_and_unchanged_default(tmp_path):
    import os
    import subprocess
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")
    env = dict(os.environ, WINDOW_MODE=benchmark.MATRIX_MODE, TS=str(HERE.parents[1]),
               ARTIFACT=str(artifact), RECEIPTS=str(tmp_path / "arms"), FABRIC="socket")
    env.pop("MATRIX_CELLS", None)
    command = ["bash", str(HERE / "drive_tp2.sh"), str(HERE / "plan-graph-mnbt-matrix.txt"), "--dry-run"]
    default = subprocess.run(command, env=env, capture_output=True, text=True)
    assert default.returncode == 0, default.stderr
    assert "== arm graph_mnbt2048_on" in default.stdout
    assert "== arm graph_mnbt4096_on" in default.stdout
    selected = subprocess.run(command + ["--matrix-cell", "4096:512:8"], env=env,
                              capture_output=True, text=True)
    assert selected.returncode == 0, selected.stderr
    assert "== arm graph_mnbt2048_on" not in selected.stdout
    assert "== arm graph_mnbt4096_on" in selected.stdout
    assert "L512 c8" in selected.stdout
    assert "18-cell" not in selected.stdout


def test_selected_scope_survives_inputs_and_rank_manifest(tmp_path):
    import window_driver as driver
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")
    env = dict(WINDOW_MODE=benchmark.MATRIX_MODE, TS=str(HERE.parents[1]),
               ARTIFACT=str(artifact), RECEIPTS=str(tmp_path / "arms"), FABRIC="socket",
               MATRIX_CELLS=json.dumps(SELECTED), SOURCE_COMMIT="source", SOURCE_SHA256="digest",
               PRODUCER_COMMIT="producer", PRODUCER_SHA256="digest")
    config = recipe.inputs(env, live=False)
    assert config["matrix_cells"] == SELECTED
    (tmp_path / "inputs.json").write_text(json.dumps(config))
    ranks = driver.rows(tmp_path, config, env)
    assert all(json.loads(rank["env"]["MATRIX_CELLS"]) == SELECTED for rank in ranks)
    assert all(rank["priority"] == 0 and rank["timeout_s"] == 5400 for rank in ranks)
    assert all(rank["demand"]["mem_gb"] == 104 and rank["gpu_memory_gb"] == 102 for rank in ranks)
    current = copy.deepcopy(config)
    current["matrix_cells"] = [[4096, 512, 1]]
    with pytest.raises(Refused, match="comparability"):
        recipe.check_control_record(config, current, where="test", refusal=Refused("identity"))


def test_selected_scope_refuses_other_window_mode(tmp_path):
    with pytest.raises(Refused, match="matrix selection"):
        recipe.inputs({"WINDOW_MODE": "ship-graph-2048", "MATRIX_CELLS": json.dumps(SELECTED)}, live=False)


def test_selected_timing_requires_exact_population():
    def result(cells):
        return {"cells": {f"host-L{length}-c{conc}": {
            "skipped": False, "complete": True,
            "trials": [{"trial": trial, "requests": [
                {"error": None, "usage": {"prompt_tokens": length, "completion_tokens": 128},
                 "generation": {"done": True}, "completion_tokens": 128}
                for _ in range(conc)]} for trial in range(1, 11)]}
            for length, conc in cells}}
    pairs = [(length, conc) for _, length, conc in SELECTED]
    benchmark.require_matrix_timing(result(pairs), cells=pairs)
    for changed in (pairs[:-1], pairs + [(512, 1)]):
        with pytest.raises(Refused, match="population"):
            benchmark.require_matrix_timing(result(changed), cells=pairs)
    bad = result(pairs)
    bad["cells"]["host-L512-c8"]["trials"][0]["requests"][0]["completion_tokens"] = 127
    with pytest.raises(Refused, match="incomplete"):
        benchmark.require_matrix_timing(bad, cells=pairs)


def test_selected_probes_use_frozen_client_without_cartesian_replay(tmp_path, monkeypatch):
    from types import SimpleNamespace
    calls = []
    rdv = tmp_path / "window"
    rdv.mkdir()
    prompts = rdv / "matrix-prompts.json"
    prompts.write_text("{}")
    monkeypatch.setattr(benchmark, "matrix_derived_prompts", lambda path: {"path": str(prompts)})
    monkeypatch.setattr(benchmark, "profile_and_power", lambda *args: {"profile_arm": args[1]["arm"]})

    def command(argv, **kwargs):
        calls.append(argv)
        length = int(argv[argv.index("--lens") + 1])
        conc = int(argv[argv.index("--conc") + 1])
        cell = {"skipped": False, "complete": True, "trials": [
            {"trial": trial, "requests": [{"error": None,
             "usage": {"prompt_tokens": length, "completion_tokens": 128},
             "generation": {"done": True}, "completion_tokens": 128} for _ in range(conc)]}
            for trial in range(1, 11)]}
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps({
            "cells": {f"host-L{length}-c{conc}": cell}, "config": {"lens": [length], "conc": [conc]},
            "started_unix": 1, "ended_unix": 2}))

    config = {"window_mode": benchmark.MATRIX_MODE, "matrix_cells": SELECTED,
              "profile_dir": str(tmp_path / "profiles"), "prompts": str(prompts)}
    adapter = SimpleNamespace(config=config, rdv=rdv, identity={}, command=command,
                              tick=lambda: None, envelope=SimpleNamespace(remaining=lambda: 5400))
    arm = {"arm": "graph_mnbt4096_on", "max_batched": 4096, "lever_env": benchmark.MATRIX_LEVERS_ON}
    assert benchmark.probes(adapter, arm, {}) == {"profile_arm": arm["arm"]}
    assert len(calls) == len(SELECTED)
    assert all(argv[1] == str(benchmark.CLIENT / "u4_speed_client.py") for argv in calls)
    assert [(int(argv[argv.index("--lens") + 1]), int(argv[argv.index("--conc") + 1]))
            for argv in calls] == [(length, conc) for _, length, conc in SELECTED]
    aggregate = json.loads((rdv / "arms" / arm["arm"] / "timing.json").read_text())
    assert set(aggregate["cells"]) == {f"host-L{length}-c{conc}" for _, length, conc in SELECTED}
    assert aggregate["config"]["selected_cells"] == [[length, conc] for _, length, conc in SELECTED]
    invocation = json.loads((rdv / "arms" / arm["arm"] / "invocation.json").read_text())
    assert invocation["matrix"]["cells"] == aggregate["config"]["selected_cells"]
    assert len(invocation["timing_runs"]) == len(SELECTED)

