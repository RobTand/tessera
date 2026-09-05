"""Runtime-scoped census agreement must not borrow another image's receipt."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from tessera.serving.census import cell_launch_agreement
from tessera.serving.runtime_image import container_env, resolve


IMAGE = "example/runtime@sha256:" + "1" * 64
OTHER_IMAGE = "example/runtime@sha256:" + "2" * 64


def _case(*, structure="dense", execution_modes=("eager",)):
    symbol = ("torch._scaled_mm" if structure == "dense"
              else "vllm.fused_moe.modular_kernel")
    records = {"decode": {"model.proj": {
        "kind": "dense" if structure == "dense" else "moe",
        "policy": "TESSERA_FP8:resident", "symbol": symbol,
        # The one-row forward the decode cell covers: a record counted as
        # decode evidence has to carry the shape that ran (#207).
        "shape": "M1:N4096:K4096",
        "decoder": "torch_materialize_stock"}}}
    cell = {
        "id": "synthetic", "platform": "sm_121", "structure": structure,
        "family": "E4M3", "regime": "decode", "rungs_q256": [1024],
        "requires_serve_flags": ["TESSERA_SERVE_MODE=resident"],
        "runtime": {"image": IMAGE, "execution_modes": list(execution_modes)},
        "executes": [{"symbol": symbol, "decoder": "torch_materialize_stock"}],
    }
    kwargs = {
        "cells": [cell], "phase_regimes": {"decode": "decode"},
        "platform": "sm_121", "structure": structure,
        "rungs_by_module": {"model.proj": 1024},
        "families_by_route": {"TESSERA_FP8": "E4M3"},
    }
    return records, kwargs


def _tool():
    path = Path(__file__).resolve().parents[1] / "tools" / "tessera_route_census.py"
    spec = importlib.util.spec_from_file_location("runtime_scoped_census", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_runtime_context_cannot_borrow_a_cell():
    records, kwargs = _case()
    block, problems = cell_launch_agreement(records, **kwargs)
    assert problems == []
    assert block["agrees"] is None
    assert block["phases"]["decode"]["unattested"] == 1


@pytest.mark.parametrize("image,mode", [
    (OTHER_IMAGE, "eager"), (IMAGE, "compiled"),
    (None, "eager"), (IMAGE, None), (IMAGE, "unknown"),
])
def test_uncovered_runtime_image_or_mode_stays_unattested(image, mode):
    records, kwargs = _case()
    block, problems = cell_launch_agreement(
        records, **kwargs, runtime_image=image, execution_mode=mode)
    assert problems == []
    assert block["agrees"] is None
    assert block["phases"]["decode"]["covered_by_cell"] == 0
    assert block["runtime"] == {"image": image, "execution_mode": mode}


def test_matching_runtime_preserves_launch_refusal_and_context():
    records, kwargs = _case()
    block, problems = cell_launch_agreement(
        records, **kwargs, runtime_image=IMAGE, execution_mode="eager")
    assert block["agrees"] is True and problems == []
    assert block["runtime"] == {"image": IMAGE, "execution_mode": "eager"}
    records["decode"]["model.proj"]["symbol"] = "wrong.kernel"
    block, problems = cell_launch_agreement(
        records, **kwargs, runtime_image=IMAGE, execution_mode="eager")
    assert block["agrees"] is False
    assert len(problems) == 1 and "wrong.kernel" in problems[0]


def test_cell_without_runtime_scope_does_not_inherit_a_global_pin():
    records, kwargs = _case()
    del kwargs["cells"][0]["runtime"]
    block, problems = cell_launch_agreement(
        records, **kwargs, runtime_image=IMAGE, execution_mode="eager")
    assert block["agrees"] is None and problems == []


def test_compiled_moe_single_launch_can_agree_but_dense_stays_unattested():
    for structure in ("routed_moe", "dense"):
        records, kwargs = _case(structure=structure, execution_modes=("compiled",))
        block, problems = cell_launch_agreement(
            records, **kwargs, runtime_image=IMAGE, execution_mode="compiled")
        assert problems == []
        assert block["agrees"] is (True if structure == "routed_moe" else None)
        if structure == "dense":
            assert "compiled" in block["unsupported_reason"]
            assert block["phases"]["decode"]["unattested"] == 1


def test_compiled_combined_moe_trace_does_not_claim_a_single_launch():
    records, kwargs = _case(structure="routed_moe", execution_modes=("compiled",))
    records["decode"]["model.proj"]["symbol"] += "+another.kernel"
    block, problems = cell_launch_agreement(
        records, **kwargs, runtime_image=IMAGE, execution_mode="compiled")
    assert problems == [] and block["agrees"] is None
    assert block["phases"]["decode"]["unsupported_records"] == 1


def test_all_structure_agreement_threads_runtime_without_borrowing_dense_image():
    records, kwargs = _case(structure="routed_moe")
    dense_records, dense_kwargs = _case()
    dense = copy.deepcopy(dense_kwargs["cells"][0])
    dense["id"] = "dense_other_image"
    dense["runtime"]["image"] = OTHER_IMAGE
    records["decode"]["model.dense"] = dense_records["decode"]["model.proj"]
    tool_kwargs = {key: kwargs[key] for key in ("cells", "phase_regimes", "platform",
                                               "families_by_route")}
    tool_kwargs["cells"] = kwargs["cells"] + [dense]
    tool_kwargs.update(
        declared_rungs={"model.proj": 1024, "model.dense": 1024},
        record_owners={"decode": {name: name for name in records["decode"]}},
        runtime_image=IMAGE, execution_mode="eager")
    block, problems = _tool().all_structure_agreement(records, **tool_kwargs)
    assert problems == [] and block["agrees"] is True
    assert block["runtime"] == {"image": IMAGE, "execution_mode": "eager"}
    assert block["structures"]["dense"]["agrees"] is None
    assert block["structures"]["routed_moe"]["agrees"] is True


@pytest.mark.parametrize("image", [None, "example/runtime:latest", OTHER_IMAGE.upper()])
def test_cli_requires_an_exact_runtime_image_before_loading_vllm(image, capsys):
    argv = ["checkpoint", "receipt.json"]
    if image is not None:
        argv += ["--runtime-image", image]
    with pytest.raises(SystemExit) as raised:
        _tool().parse_args(argv)
    assert raised.value.code == 2
    assert "--runtime-image" in capsys.readouterr().err


@pytest.mark.parametrize("compiled", [False, True])
def test_cli_records_mode_from_the_flag_that_controls_llm(compiled):
    argv = ["checkpoint", "receipt.json", "--runtime-image", IMAGE]
    if compiled:
        argv.append("--compiled")
    # The image must be attested from inside the container since #132; what is
    # under test here is the mode, so the launcher's environment is supplied
    # by the code the launcher itself calls.
    args = _tool().parse_args(argv, env=container_env(resolve(IMAGE, inspector=lambda _r: {
        "present": True, "local_id": "sha256:" + "ab" * 32, "repo_digests": [IMAGE]})))
    assert args.runtime_image == IMAGE
    assert args.execution_mode == ("compiled" if compiled else "eager")
