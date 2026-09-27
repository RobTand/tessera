"""Every model read the census makes reaches every rank.

THE DEFECT THIS PINS.  ``LLM.apply_model(fn)`` sends ``fn`` as an RPC argument,
and vLLM's multiprocess and ray executors write RPC arguments with the stdlib
pickler.  The first TP2 census on image X (stub D, 2026-09-27 01:08Z) died at
the name map with ``AttributeError: Can't pickle local object
'main.<locals>.<lambda>'``; the other ``apply_model`` calls would have been
written as references to ``__main__``, which a ray or vLLM worker cannot
resolve.  One rank never showed it, because the in-process executor calls
``fn`` without pickling.  The census now sends each read as the RPC METHOD,
which vLLM writes with cloudpickle, by value.

THE FAIL-BEFORE.  On the pre-change tree the tool has five
``llm.apply_model(`` call sites and no ``on_every_rank_model``.
"""
from __future__ import annotations

import re
from pathlib import Path

from tools.tessera_route_census import on_every_rank_model

TOOL = Path(__file__).resolve().parents[1] / "tools" / "tessera_route_census.py"


def test_the_census_sends_no_model_read_as_an_rpc_argument():
    source = TOOL.read_text()
    assert not re.search(r"\bllm\.apply_model\(", source)


def test_each_model_read_goes_through_the_by_value_wrapper():
    source = TOOL.read_text()
    for fn in ("declared_in_module_space", "census", "lane_refusals", "rank_identity"):
        assert re.search(
            rf"llm\.collective_rpc\(\s*on_every_rank_model\({fn}\b", source), fn


def test_the_wrapper_calls_fn_on_the_workers_model_with_its_arguments():
    class Worker:
        def __init__(self, model):
            self.model = model

        def get_model(self):
            return self.model

    def fn(model, targets, extra):
        return (model, tuple(targets), extra)

    call = on_every_rank_model(fn, ["a", "b"], 3)
    assert call(Worker("m0")) == ("m0", ("a", "b"), 3)
    assert on_every_rank_model(lambda m: m)(Worker("m1")) == "m1"
