"""Measurement acceptance for native dense operator receipts (tessera#1182).

A timing-admissible receipt closes only with every phase measured, every
tensor digest present, and every transfer byte count tracked. Each gap
refuses by name.
"""
import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "experiments" / "bench_native_operator.py"


def load_bench():
    spec = importlib.util.spec_from_file_location("bench_native_operator_accept", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load_bench()

DIGESTS = ("input", "reference_qdq", "qdq", "reference_output", "output")


def tensor(name):
    return {"shape": [2, 4], "dtype": "bfloat16", "logical_bytes": 16,
            "content_sha256": "a" * 64}


def good_receipt():
    phases, resources = {}, {}
    for phase in bench.PHASES:
        phases[phase] = {"m": 1, "measurement": {"samples_ms": [0.1, 0.2, 0.3]},
                         "route": {"state": "served"},
                         **{name: tensor(name) for name in DIGESTS}}
        resources[phase] = {"input_bytes": 64, "output_bytes": 128,
                            "torch_peak_increment_bytes": 0}
    return {"schema": bench.RECEIPT_SCHEMA, "status": "timing_admissible",
            "panel_sha256": "b" * 64, "runtime_sha256": "c" * 64,
            "phases": phases,
            "resources": {"status": "complete_operator_bound", "phases": resources}}


def test_complete_timing_admissible_receipt_is_accepted():
    assert bench.accept_measurement(good_receipt()) is None


def test_missing_phase_refuses_by_name():
    receipt = good_receipt()
    del receipt["phases"]["decode"]
    with pytest.raises(ValueError, match="decode"):
        bench.accept_measurement(receipt)


def test_unmeasured_phase_refuses_by_name():
    receipt = good_receipt()
    receipt["phases"]["prefill"]["measurement"] = None
    with pytest.raises(ValueError, match="prefill.*measurement"):
        bench.accept_measurement(receipt)


def test_missing_tensor_digest_refuses_by_name():
    receipt = good_receipt()
    del receipt["phases"]["decode"]["output"]["content_sha256"]
    with pytest.raises(ValueError, match="decode.*output.*digest"):
        bench.accept_measurement(receipt)


def test_missing_panel_digest_refuses_by_name():
    receipt = good_receipt()
    receipt["panel_sha256"] = "short"
    with pytest.raises(ValueError, match="panel_sha256"):
        bench.accept_measurement(receipt)


def test_untracked_transfer_bytes_refuse_by_name():
    receipt = good_receipt()
    del receipt["resources"]["phases"]["prefill"]["output_bytes"]
    with pytest.raises(ValueError, match="prefill.*transfer"):
        bench.accept_measurement(receipt)


def test_non_admissible_status_refuses():
    receipt = good_receipt()
    receipt["status"] = "numerical_refused"
    with pytest.raises(ValueError, match="timing_admissible"):
        bench.accept_measurement(receipt)
