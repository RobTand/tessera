"""Explicit 18-cell A8S graph MNBT matrix scope, not live fit/speed evidence."""
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1] / "experiments/graph_attest_702"
sys.path.insert(0, str(HERE))
import tp2_recipe as recipe
from managed_window import Refused
import eager_benchmark as benchmark


def matrix_plan(tmp_path):
    src = HERE / "plan-graph-mnbt-matrix.txt"
    path = tmp_path / "matrix-plan.txt"
    path.write_bytes(src.read_bytes())
    return path


def test_matrix_pair_selects_mnbt2048_then_4096_with_mtp_off(tmp_path):
    arms = recipe.plan(matrix_plan(tmp_path), mode=benchmark.MATRIX_MODE)
    assert [(a["arm"], a["max_batched"], a["eager"]) for a in arms] == [
        ("graph_mnbt2048_on", 2048, "0"), ("graph_mnbt4096_on", 4096, "0")]
    for arm in arms:
        assert arm["spec"] == "null"
        assert arm["lever_env"] == benchmark.MATRIX_LEVERS_ON

def test_matrix_serve_uses_c8_and_omits_speculative_config():
    for max_batched in (2048, 4096):
        arm = {"arm": "probe", "eager": "0", "compilation": json.dumps(recipe.GRAPH, separators=(",", ":")),
               "spec": "null", "max_batched": max_batched, "fabric": "socket",
               "lever_env": dict(benchmark.MATRIX_LEVERS_ON)}
        for rank in (0, 1):
            argv = recipe.serve(dict(artifact="/artifact", window_mode=benchmark.MATRIX_MODE,
                                     profile_dir="/profiles"), arm, rank)
            assert "--speculative-config" not in argv
            assert argv[argv.index("--max-num-seqs") + 1] == "8"
            assert argv[argv.index("--max-num-batched-tokens") + 1] == str(max_batched)
            assert "--compilation-config" in argv


def test_matrix_refuses_mtp_on_and_lever_off(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text(
        'graph_mnbt2048_on FABRIC=socket EAGER=0 MAX_BATCHED=2048 '
        f'COMPILATION_JSON={json.dumps(recipe.GRAPH, separators=(",", ":"))} '
        f'SPEC_JSON={json.dumps(recipe.MTP, separators=(",", ":"))} '
        'TESSERA_E4M3_DECODE_ONCE=1 TESSERA_GLM53_KDA_CONV_SPLIT=on TESSERA_ROUTED_PIECE_MAJOR=1\n'
        'graph_mnbt4096_on FABRIC=socket EAGER=0 MAX_BATCHED=4096 '
        f'COMPILATION_JSON={json.dumps(recipe.GRAPH, separators=(",", ":"))} '
        'SPEC_JSON=null TESSERA_E4M3_DECODE_ONCE=0 TESSERA_GLM53_KDA_CONV_SPLIT=off TESSERA_ROUTED_PIECE_MAJOR=0\n')
    with pytest.raises(Refused):
        recipe.plan(path, mode=benchmark.MATRIX_MODE)


def test_matrix_timing_requires_all_nine_cells():
    import copy
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


def test_matrix_derived_prompts_replicate_frozen_c1(tmp_path):
    derived = benchmark.matrix_derived_prompts(tmp_path)
    blob = json.loads(Path(derived["path"]).read_bytes())
    assert blob["lens"] == [512, 2048, 8192]
    assert blob["concurrency"] == [1, 4, 8]
    frozen = json.loads((benchmark.PANEL / "prompts.json").read_bytes())
    for length in ("512", "2048", "8192"):
        for trial in range(11):
            base = frozen["prompts"][length]["1"][trial][0]
            assert blob["prompts"][length]["1"][trial][0] == base
            assert blob["prompts"][length]["4"][trial] == [base] * 4
            assert blob["prompts"][length]["8"][trial] == [base] * 8


def test_old_graph_ship_pair_still_requires_mtp1(tmp_path):
    path = tmp_path / "old.txt"
    path.write_text(
        'graph2048_off FABRIC=socket EAGER=0 MAX_BATCHED=2048 '
        f'COMPILATION_JSON={json.dumps(recipe.GRAPH, separators=(",", ":"))} '
        'SPEC_JSON=null TESSERA_E4M3_DECODE_ONCE=0 TESSERA_GLM53_KDA_CONV_SPLIT=off TESSERA_ROUTED_PIECE_MAJOR=0\n'
        'graph2048_on FABRIC=socket EAGER=0 MAX_BATCHED=2048 '
        f'COMPILATION_JSON={json.dumps(recipe.GRAPH, separators=(",", ":"))} '
        'SPEC_JSON=null TESSERA_E4M3_DECODE_ONCE=1 TESSERA_GLM53_KDA_CONV_SPLIT=on TESSERA_ROUTED_PIECE_MAJOR=1\n')
    with pytest.raises(Refused):
        recipe.plan(path, mode=benchmark.GRAPH_SHIP_MODE)
