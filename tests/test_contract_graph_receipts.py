"""A compiled cell names the graph receipt it rests on (contract v48, tessera#702).

``execution_modes: ["compiled"]`` is ``enforce_eager=False``, which names no
configuration: on the vLLM nightly stack vLLM's default graph serve of GLM-5.3
selected other op implementations than eager and moved every completion of the
equality suite (tessera#702), so "compiled" alone is not a serve a consumer can
reproduce.  These tests pin the validator's half: a compiled cell must name a
``lane_eligibility.graph_receipts`` entry measured on its own image and
toolchain, an eager-only cell must not, and every receipt must be named.
They mutate the PACKAGED table, so they also prove the shipped document passes.
"""
from __future__ import annotations

import copy

import pytest

from tessera.serving.contract import (GRAPH_CAPTURE_EVERY_COUNT, cell_graph_receipt,
                                      load_serving_contract, validate_graph_receipts,
                                      validate_serving_contract)


def _receipt(cell, rid="test_graph_receipt"):
    runtime = cell["runtime"]
    return {
        "id": rid, "image": runtime["image"], "vllm": runtime["vllm"],
        "torch": runtime["torch"], "model_type": "glm5_next_text",
        "runner_sha256": {"v1/worker/gpu/model_runner.py": "a" * 64},
        "serve": {"compilation_config": {"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY"},
                  "kernel_config": {"enable_flashinfer_autotune": False},
                  "env": {"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"},
                  "capture_sizes": GRAPH_CAPTURE_EVERY_COUNT},
        "equivalence": {"verdict": "eager_equivalent", "criterion": "tessera#508 membership",
                        "receipt": "docs/measurements/x.md",
                        "data": "experiments/results/x.json"},
        "speculative": [],
    }


def _eager_cell(doc):
    return next(c for c in doc["lane_eligibility"]["cells"]
                if c["runtime"]["execution_modes"] == ["eager"])


def _compiled(doc, rid="test_graph_receipt"):
    """The packaged table with one eager cell made compiled on a new receipt."""
    doc = copy.deepcopy(doc)
    cell = _eager_cell(doc)
    doc["lane_eligibility"].setdefault("graph_receipts", []).append(_receipt(cell, rid))
    cell["runtime"]["execution_modes"] = ["eager", "compiled"]
    cell["runtime"]["graph_receipt"] = rid
    return doc, cell


def test_the_packaged_table_names_every_receipt_it_publishes():
    doc = load_serving_contract()
    receipts = validate_graph_receipts(doc["lane_eligibility"].get("graph_receipts", []), "x")
    named = set()
    for cell in doc["lane_eligibility"]["cells"]:
        receipt = cell_graph_receipt(cell, doc)
        compiled = "compiled" in cell["runtime"]["execution_modes"]
        assert (receipt is not None) == compiled, cell["id"]
        if receipt is not None:
            named.add(receipt["id"])
    assert named == set(receipts)


def test_a_compiled_cell_on_its_own_receipt_validates():
    doc, cell = _compiled(load_serving_contract())
    validate_serving_contract(doc)
    assert cell_graph_receipt(cell, doc)["id"] == "test_graph_receipt"


def test_a_compiled_cell_without_a_receipt_is_refused():
    doc, cell = _compiled(load_serving_contract())
    del cell["runtime"]["graph_receipt"]
    doc["lane_eligibility"]["graph_receipts"] = [
        r for r in doc["lane_eligibility"]["graph_receipts"] if r["id"] != "test_graph_receipt"]
    with pytest.raises(ValueError, match="claims execution mode 'compiled' without"):
        validate_serving_contract(doc)


def test_an_eager_cell_naming_a_receipt_is_refused():
    doc, cell = _compiled(load_serving_contract())
    cell["runtime"]["execution_modes"] = ["eager"]
    with pytest.raises(ValueError, match="a graph receipt scopes a compiled cell only"):
        validate_serving_contract(doc)


@pytest.mark.parametrize("field", ["image", "vllm", "torch"])
def test_a_receipt_from_another_runtime_is_refused(field):
    doc, cell = _compiled(load_serving_contract())
    receipt = doc["lane_eligibility"]["graph_receipts"][-1]
    receipt[field] = ("other/image@sha256:" + "f" * 64) if field == "image" else "other"
    with pytest.raises(ValueError, match="a graph receipt attests the image it was measured on"):
        validate_serving_contract(doc)


def test_an_unnamed_receipt_is_refused():
    doc = copy.deepcopy(load_serving_contract())
    doc["lane_eligibility"].setdefault("graph_receipts", []).append(
        _receipt(_eager_cell(doc), "orphan"))
    with pytest.raises(ValueError, match=r"publishes \['orphan'\], which no compiled cell names"):
        validate_serving_contract(doc)


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r["equivalence"].update(verdict="different"), "must be one of"),
    (lambda r: r["serve"].update(capture_sizes="default"), "capture_sizes must be"),
    (lambda r: r["serve"]["compilation_config"].update(cudagraph_capture_sizes=[1, 2]),
     "must not carry cudagraph_capture_sizes"),
    (lambda r: r["serve"]["compilation_config"].pop("cudagraph_mode"), "naming its cudagraph_mode"),
    (lambda r: r["equivalence"].update(receipt="/abs/x.md"), "repository path under"),
    (lambda r: r["runner_sha256"].update({"../x.py": "a" * 64}), "vllm-package-relative"),
    (lambda r: r["speculative"].append({"method": "mtp", "num_speculative_tokens": 0,
                                        **{k: r["equivalence"][k] for k in r["equivalence"]}}),
     "num_speculative_tokens >= 1"),
    (lambda r: r.update(extra=1), "extra"),
])
def test_a_malformed_receipt_is_refused(mutate, message):
    doc, _cell = _compiled(load_serving_contract())
    mutate(doc["lane_eligibility"]["graph_receipts"][-1])
    with pytest.raises(ValueError, match=message):
        validate_serving_contract(doc)
