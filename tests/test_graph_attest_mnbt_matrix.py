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
