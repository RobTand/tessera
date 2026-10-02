"""The bitwise gate must refuse before paired timing can be admitted."""
import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/t8r_speed"))
from routed_lut_run import CASES, require_equal_witnesses


def records():
    return {"output_witnesses": {k: {"shape": [2048, 4096], "sha256": "a" * 64} for k in CASES},
            "input_hashes": {"x": "a" * 64, "ids": "b" * 64, "weights": "c" * 64}}


def test_same_actual_witnesses_admit():
    require_equal_witnesses(records(), records())


@pytest.mark.parametrize("fault", ["wrong_bit", "missing_tail", "input_change"] )
def test_wrong_or_incomplete_witness_refuses_timing(fault):
    a = records(); b = copy.deepcopy(a)
    if fault == "wrong_bit": b["output_witnesses"]["real"]["sha256"] = "f" * 64
    elif fault == "missing_tail": b["output_witnesses"].pop("single-expert-tail-129")
    else: b["input_hashes"]["ids"] = "f" * 64
    with pytest.raises(ValueError, match="no timing admitted"):
        require_equal_witnesses(a, b)
